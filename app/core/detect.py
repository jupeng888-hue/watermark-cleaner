# -*- coding: utf-8 -*-
"""水印检测：定位「产品之外」的文字水印。
- 文字检测：局部背景差分法（平滑背景上任意颜色文字均有效），装有 Tesseract 时 OCR 优先
- 产品主体：局部差分最大连通体（产品+手持部分是一整块），水印蒙版自动排除产品
"""
import cv2
import numpy as np


def _local_diff(img_bgr, ksize=51):
    """每个像素与局部背景（大核中值）的最大通道差。背景渐变不影响。"""
    bg = cv2.medianBlur(img_bgr, ksize)
    return np.abs(img_bgr.astype(np.int16) - bg.astype(np.int16)).max(axis=2).astype(np.uint8)


def _largest_component(fg, h, w, min_ratio=0.03, max_ratio=0.90):
    """前景二值图 → 最大连通体（填洞）。面积不合理返回 None。"""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if num <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = stats[biggest, cv2.CC_STAT_AREA]
    if not (min_ratio * h * w <= area <= max_ratio * h * w):
        return None
    sub = np.where(labels == biggest, 255, 0).astype(np.uint8)
    sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    # 填充内部小空洞（大面积纯色产品中心与局部背景同色会漏成洞）；
    # 只填小洞，避免吞掉被产品包围的大片背景（如双臂/杯间空隙、上方标题区）
    inv = cv2.bitwise_not(sub)
    ff = inv.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, ff_mask, (0, 0), 128)
    holes = (ff == 255).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(holes, 8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] <= 0.05 * h * w:
            sub[lab == i] = 255
    return sub


def product_mask(img_bgr):
    """产品主体蒙版（255=产品）。
    从四边泛洪「低差分海」= 真背景（背景渐变/纹理都可达），
    不可达区域取大连通体 = 产品；产品内部空洞全部填实。
    注意：不能用「最大连通体=产品」，实测渐变背景上背景+手会连成
    最大体、产品反而变成洞（本工具真实杯子图踩过此坑，见防错清单 #17）。"""
    h, w = img_bgr.shape[:2]
    # 墙 = Sobel 梯度（不能用局部差分：又直又锐的长边上中值会精确翻转，
    # 差分恒为 0，墙会漏——合成测试图踩过此坑，见防错清单 #19）
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    mag = cv2.add(cv2.absdiff(gx, gx * 0), cv2.absdiff(gy, gy * 0))  # |gx|+|gy|
    wall = (mag > 30).astype(np.uint8)
    wall = cv2.dilate(wall, np.ones((3, 3), np.uint8))
    sea = (1 - wall).astype(np.uint8)  # 1=可从背景走进来
    ff = sea.copy()
    ffm = np.zeros((h + 2, w + 2), np.uint8)
    for x in range(0, w, 2):
        for y in (0, h - 1):
            if ff[y, x] == 1:
                cv2.floodFill(ff, ffm, (x, y), 2)
    for y in range(0, h, 2):
        for x in (0, w - 1):
            if ff[y, x] == 1:
                cv2.floodFill(ff, ffm, (x, y), 2)
    fg = np.where(ff == 1, 255, 0).astype(np.uint8)  # 不可达 = 产品候选
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    pm = np.zeros((h, w), np.uint8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= 0.03 * h * w:
            pm[labels == i] = 255
    if cv2.countNonZero(pm) == 0:
        return pm
    pm = cv2.morphologyEx(pm, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    # 产品内部空洞（高光/印刷低差分区）全部填实
    inv = cv2.bitwise_not(pm)
    ff2 = inv.copy()
    m3 = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff2, m3, (0, 0), 128)
    pm[ff2 == 255] = 255
    return pm


def _text_boxes_tesseract(img_bgr):
    try:
        import pytesseract
        data = pytesseract.image_to_data(
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
            config="--psm 11", output_type=pytesseract.Output.DICT)
        boxes = []
        for i, txt in enumerate(data["text"]):
            if txt.strip() and float(data["conf"][i]) > 30:
                boxes.append((data["left"][i], data["top"][i],
                              data["width"][i], data["height"][i]))
        return boxes
    except Exception:
        return None  # 未安装 tesseract


def _text_boxes_diff(img_bgr):
    """局部差分文字行检测：真实电商图实测有效（红字/灰字/半透明字均可）。
    横排、竖排文字都检测（两个方向分别做闭运算合并）。"""
    H, W = img_bgr.shape[:2]
    diff = _local_diff(img_bgr)
    _, bw = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    boxes = []
    # 长边放宽：一行水印可以很长（真实图 4 字标题宽达 0.52*W，曾被
    # 0.35 上限误杀，见防错清单 #18）；真正该限制的是短边=行的粗细
    cap_long = 0.9 * max(H, W)
    cap_thick = 0.15 * max(H, W)
    for kernel, horizontal in [(np.ones((9, 25), np.uint8), True),
                               (np.ones((25, 9), np.uint8), False)]:
        merged = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            long_ok = (w >= 1.5 * h) if horizontal else (h >= 1.5 * w)
            if (10 <= max(h, w) <= cap_long and 8 <= min(h, w) <= cap_thick
                    and long_ok and w * h >= 200):
                dup = any(abs(x - bx) < 15 and abs(y - by) < 15 for bx, by, _, _ in boxes)
                if not dup:
                    boxes.append((x, y, w, h))
    return boxes


def logo_boxes_auto(img_bgr, pm):
    """无模板标志检测：产品内部的高差分「块状」成分按邻近聚类。
    - 只取产品腐蚀 15px 后的内部区域：贴边的杯身高光/轮廓天然被排除
    - 成分形状过滤：min(w,h)>=5 且长宽比<=6（字母/图形是块；高光是细长条）
    - 框外扩 12px 重叠即同组（竖排字母间隙 <12px 会聚成一列，杯盖阴影 >12px 不会串组）
    返回 [(x, y, w, h)]。"""
    H, W = img_bgr.shape[:2]
    diff = _local_diff(img_bgr)
    _, bw = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    pm_er = cv2.erode(pm, np.ones((15, 15), np.uint8))
    bw_in = cv2.bitwise_and(bw, pm_er)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw_in, 8)
    comps = []
    for i in range(1, num):
        x, y, cw, ch, area = stats[i]
        if area >= 30 and min(cw, ch) >= 5 and max(cw, ch) / max(min(cw, ch), 1) <= 5.5:
            comps.append([int(x), int(y), int(x + cw), int(y + ch)])
    groups = []
    for cb in comps:
        ex = [cb[0] - 12, cb[1] - 12, cb[2] + 12, cb[3] + 12]
        hit = None
        for g in groups:
            if not (ex[2] < g[0] or ex[0] > g[2] or ex[3] < g[1] or ex[1] > g[3]):
                hit = g
                break
        if hit is None:
            groups.append(list(cb))
        else:
            hit[0] = min(hit[0], cb[0]); hit[1] = min(hit[1], cb[1])
            hit[2] = max(hit[2], cb[2]); hit[3] = max(hit[3], cb[3])
    changed = True
    while changed:  # 链式邻近需要传递合并
        changed = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = groups[i], groups[j]
                ea = [a[0] - 12, a[1] - 12, a[2] + 12, a[3] + 12]
                if not (ea[2] < b[0] or ea[0] > b[2] or ea[3] < b[1] or ea[1] > b[3]):
                    a[0] = min(a[0], b[0]); a[1] = min(a[1], b[1])
                    a[2] = max(a[2], b[2]); a[3] = max(a[3], b[3])
                    groups.pop(j)
                    changed = True
                    break
            if changed:
                break
    return [(x0, y0, x1 - x0, y1 - y0) for x0, y0, x1, y1 in groups
            if (x1 - x0) * (y1 - y0) >= 600 and max(x1 - x0, y1 - y0) <= 0.5 * max(H, W)]


def detect_watermark_mask(img_bgr, extra_boxes=None, protect_product=True):
    """返回 (mask, boxes)。mask 255=建议擦除的水印区域（已排除产品）。"""
    h, w = img_bgr.shape[:2]
    # 差分法（主线，实测最稳）+ Tesseract（若安装）取并集，重叠框去重
    boxes = _text_boxes_diff(img_bgr)
    for tb in (_text_boxes_tesseract(img_bgr) or []):
        tx, ty, tw, th = tb
        dup = any(abs(tx - x) < 15 and abs(ty - y) < 15 for x, y, _, _ in boxes)
        if not dup:
            boxes.append(tb)

    text_mask = np.zeros((h, w), np.uint8)
    pm = product_mask(img_bgr) if protect_product else np.zeros((h, w), np.uint8)
    # 25px 安全带：浅色/低对比产品的边缘分割可能漏几像素，宁可留白不可伤图
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    pm_d = cv2.dilate(pm, k)
    for (x, y, bw, bh) in boxes:
        box_area = bw * bh
        overlap = cv2.countNonZero(pm_d[y:y + bh, x:x + bw])
        if protect_product and box_area > 0 and overlap / box_area > 0.5:
            continue  # 框大部分在产品上 = 产品自身的印刷/标志，保护不擦
        cv2.rectangle(text_mask, (x, y), (x + bw, y + bh), 255, -1)
    text_mask = cv2.dilate(text_mask, np.ones((5, 5), np.uint8))
    if protect_product:
        # 像素级再扣一次：跨背景的检测框边缘可能扫到产品轮廓
        text_mask = cv2.bitwise_and(text_mask, cv2.bitwise_not(pm_d))

    if extra_boxes:  # 用户手动框选区域
        for (x0, y0, x1, y1) in extra_boxes:
            cv2.rectangle(text_mask, (x0, y0), (x1, y1), 255, -1)
    return text_mask, boxes
