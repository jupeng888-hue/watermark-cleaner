# -*- coding: utf-8 -*-
"""水印检测：定位「产品之外」的文字水印。
- 文字检测：局部背景差分法（平滑背景上任意颜色文字均有效），装有 Tesseract 时 OCR 补充
- 产品主体：双路分割（局部差分 + 四角色差），框级+像素级双重保护产品不被误擦
"""
import cv2
import numpy as np


def _local_diff(img_bgr, ksize=51):
    """每个像素与局部背景（大核中值）的最大通道差。背景渐变不影响。"""
    bg = cv2.medianBlur(img_bgr, ksize)
    return np.abs(img_bgr.astype(np.int16) - bg.astype(np.int16)).max(axis=2).astype(np.uint8)


def _largest_component(fg, h, w, min_ratio=0.03, max_ratio=0.90):
    """前景二值图 → 最大连通体（填小洞）。面积不合理返回 None。"""
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
    双路互补：局部差分适合有纹理/渐变背景；四角背景色差适合纯色平涂图。"""
    h, w = img_bgr.shape[:2]

    # 路径 A：局部背景差分（真实照片优先）
    diff = _local_diff(img_bgr)
    fg_a = cv2.morphologyEx((diff > 40).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    sub = _largest_component(fg_a, h, w)
    if sub is not None:
        return sub

    # 路径 B：四角中位色距离（纯色平涂图兜底）
    c = 20
    corners = np.concatenate([
        img_bgr[:c, :c].reshape(-1, 3), img_bgr[:c, -c:].reshape(-1, 3),
        img_bgr[-c:, :c].reshape(-1, 3), img_bgr[-c:, -c:].reshape(-1, 3)])
    bg = np.median(corners, axis=0)
    dist = np.linalg.norm(img_bgr.astype(np.float32) - bg, axis=2)
    fg_b = cv2.morphologyEx((dist > 30).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    sub = _largest_component(fg_b, h, w)
    if sub is not None:
        return sub
    return np.zeros((h, w), np.uint8)


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
    """局部差分文字行检测：真实电商图实测有效（红字/灰字/半透明字均可）。"""
    H, W = img_bgr.shape[:2]
    diff = _local_diff(img_bgr)
    _, bw = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((9, 25), np.uint8))  # 横向合并成文字行
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if 10 <= h <= 0.25 * H and w >= 1.5 * h and w * h >= 200:
            boxes.append((x, y, w, h))
    return boxes


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
    pm_d = cv2.dilate(pm, np.ones((7, 7), np.uint8))
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
