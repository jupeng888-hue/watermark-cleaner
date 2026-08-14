# -*- coding: utf-8 -*-
"""白底条覆盖：在产品上的敏感标志位置盖条 + "your logo here"。
- 条底色自动取标志周围的产品颜色（不再一律白色）
- 条按 logo 点集主方向旋转对齐（杯子斜放/标志斜印都不歪）
- 尺寸 = minAreaRect + 小内边距：完全包住 logo，但不超出其高度/宽度
敏感标志定位：模板匹配（多尺度 + NMS）、手动框选、或自动聚类（pipeline）。
注意：warpAffine 的旋转方向与 atan2 量出来的角符号相反，必须用 -rot（防错清单 #21）。"""
import cv2
import numpy as np


def find_logo_boxes(img_bgr, template_bgr, threshold=0.72, scales=None):
    """多尺度模板匹配，返回 [(x,y,w,h,score)]。模板建议裁剪成只有 logo 的小图。"""
    if scales is None:
        scales = np.linspace(0.4, 1.6, 13)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    th, tw = template_bgr.shape[:2]
    tgray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    rects = []
    for s in scales:
        rs = cv2.resize(tgray, (max(4, int(tw * s)), max(4, int(th * s))))
        if rs.shape[0] >= gray.shape[0] or rs.shape[1] >= gray.shape[1]:
            continue
        res = cv2.matchTemplate(gray, rs, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= threshold)
        h2, w2 = rs.shape
        for x, y in zip(xs, ys):
            rects.append([int(x), int(y), w2, h2, float(res[y, x])])
    if not rects:
        return []
    boxes = np.array([[x, y, x + w, y + h] for x, y, w, h, _ in rects], np.float32)
    scores = np.array([s for *_, s in rects], np.float32)
    keep = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), threshold, 0.3)
    keep = np.array(keep).flatten() if len(keep) else []
    return [rects[i] for i in keep]


def rect_from_points(pts, pad=8):
    """点集 → (中心, (w,h), 旋转角°)。长边沿点集主方向：
    横条 rot=偏离水平角，竖条 rot=偏离竖直角，均在 (-45,45]。"""
    (cx, cy), _, _ = cv2.minAreaRect(pts)
    boxp = cv2.boxPoints(cv2.minAreaRect(pts))
    e0 = boxp[1] - boxp[0]
    e1 = boxp[2] - boxp[1]
    long_e = e0 if np.hypot(*e0) >= np.hypot(*e1) else e1
    length = float(max(np.hypot(*e0), np.hypot(*e1)))
    width = float(min(np.hypot(*e0), np.hypot(*e1)))
    alpha = float(np.degrees(np.arctan2(long_e[1], long_e[0])))  # 与水平夹角
    if alpha > 90:
        alpha -= 180
    if alpha <= -90:
        alpha += 180
    if abs(alpha) <= 45:   # 横条
        return (float(cx), float(cy)), (length + 2 * pad, width + 2 * pad), alpha
    rot = alpha - 90 if alpha > 0 else alpha + 90  # 竖条
    return (float(cx), float(cy)), (width + 2 * pad, length + 2 * pad), rot


def _cup_axis(comp):
    """杯身轴线方向：按行求产品像素中心再拟合 x = a*y + b（带离群剔除）。
    行中心就是圆台形杯身的中轴，比 fitLine 全点拟合稳——
    后者会被杯带/杯盖/手的像素分布带偏，导致条的横轴与杯底不平行
    （用户明确要求：条的横轴要与杯子底平行，见防错清单 #27）。
    返回单位向量 v（统一朝上）。行数不足返回 None，调用方退回 fitLine。"""
    widths = comp.sum(axis=1) // 255
    wmax = widths.max()
    if wmax < 10:
        return None
    rows = [(y, np.nonzero(comp[y])[0].mean())
            for y in range(comp.shape[0]) if widths[y] >= 0.6 * wmax]
    if len(rows) < 10:
        return None
    ys = np.array([r[0] for r in rows], np.float64)
    xs = np.array([r[1] for r in rows], np.float64)
    a, b = np.polyfit(ys, xs, 1)
    res = xs - (a * ys + b)  # 剔除杯盖/杯带造成的离群行再拟合一次
    keep = np.abs(res) <= max(3.0, 1.5 * res.std())
    if keep.sum() >= 10:
        a, b = np.polyfit(ys[keep], xs[keep], 1)
    if abs(a) > 1:  # 倾斜超 45°（横躺产品），行中心法不适用，交回 fitLine
        return None
    v = np.array([a, 1.0])
    v = v / np.hypot(*v)
    if v[1] > 0:
        v = -v  # 统一朝上
    return v


def strip_frame_from_product(pts, pm, pad=8):
    """logo 点集 + 产品蒙版 → (中心, (w,h), rot)。
    方向取整只杯子连通体的行中心轴线（条的横轴与杯底平行，防错清单 #27）——
    必须取 logo 所在的整只产品连通体：只截标志附近的小区域会量出"竖直"，
    而杯子的真实刚体倾斜（实测 ±3° 左右）就丢了，用户一眼看出条歪；
    尺寸取 logo 点集在主轴/副轴上的投影范围 + pad：紧贴印刷区域，不伸到杯盖。"""
    H, W = pm.shape[:2]
    # logo 中心点所在的产品连通体 = 这只杯子（整只，含真实倾斜）
    cx_i = int(np.clip(pts[:, 0, 0].mean(), 0, W - 1))
    cy_i = int(np.clip(pts[:, 0, 1].mean(), 0, H - 1))
    comp = None
    if pm[cy_i, cx_i] > 0:
        num, lab, stats, _ = cv2.connectedComponentsWithStats(pm, 8)
        cid = lab[cy_i, cx_i]
        if cid > 0 and stats[cid, cv2.CC_STAT_AREA] >= 2000:
            comp = np.where(lab == cid, 255, 0).astype(np.uint8)
    v = _cup_axis(comp) if comp is not None else None
    if v is None:  # 行中心拟合失败，退回 fitLine 全点拟合（防错清单 #24）
        if comp is None:
            return None
        ppts = cv2.findNonZero(comp)
        if ppts is None or len(ppts) < 20:
            return None
        vx, vy = cv2.fitLine(ppts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()[:2]
        vx, vy = float(vx), float(vy)
        if abs(vx) > abs(vy):
            vx, vy = vy, vx  # 以竖直为基准
        n = np.hypot(vx, vy)
        v = np.array([vx / n, vy / n])
        if v[1] > 0:
            v = -v  # 统一朝上
    u = np.array([-v[1], v[0]])
    p = pts[:, 0, :].astype(np.float32)
    pv = p @ v
    pu = p @ u
    len_v = float(pv.max() - pv.min() + 2 * pad)
    len_u = float(pu.max() - pu.min() + 2 * pad)
    mid_v = (pv.max() + pv.min()) / 2
    mid_u = (pu.max() + pu.min()) / 2
    center = mid_v * v + mid_u * u
    rot = float(np.degrees(np.arctan2(v[0], -v[1])))  # 主轴偏离竖直的角度
    return (float(center[0]), float(center[1])), (len_u, len_v), rot


def draw_strip(img_bgr, pts, text="your logo here", pm=None, pad=10):
    """按点集贴合盖条。pts: Nx1x2 点集（logo 像素）。pm: 产品蒙版（取方向+取色）。"""
    out = img_bgr.copy()
    H, W = out.shape[:2]
    frame = strip_frame_from_product(pts, pm, pad=pad) if pm is not None else None
    if frame is None:
        frame = rect_from_points(pts, pad=pad)
    (cx, cy), (sw, sh), rot = frame
    rot = -rot  # warpAffine 旋转方向与 atan2 符号相反
    # 轴对齐圆角条蒙版（之后再旋转）
    base_mask = np.zeros((H, W), np.uint8)
    r = max(6, int(min(sw, sh) // 5))
    x0, y0 = int(round(cx - sw / 2)), int(round(cy - sh / 2))
    x1, y1 = int(round(cx + sw / 2)), int(round(cy + sh / 2))
    cv2.rectangle(base_mask, (x0 + r, y0), (x1 - r, y1), 255, -1)
    cv2.rectangle(base_mask, (x0, y0 + r), (x1, y1 - r), 255, -1)
    for ccx, ccy in [(x0 + r, y0 + r), (x1 - r, y0 + r),
                     (x0 + r, y1 - r), (x1 - r, y1 - r)]:
        cv2.circle(base_mask, (ccx, ccy), r, 255, -1)
    # 条底色：环形带内产品像素的中位色；取样不足退回白色
    ring = cv2.subtract(cv2.dilate(base_mask, np.ones((37, 37), np.uint8)), base_mask)
    if pm is not None:
        ring = cv2.bitwise_and(ring, pm)
    ys, xs = np.where(ring > 0)
    prod = tuple(int(v) for v in np.median(out[ys, xs], axis=0)) if len(xs) > 50 else (255, 255, 255)
    # 彩色产品一律白色条；白色产品白盖白看不见，换橙色条（防错清单 #25）
    lum0 = 0.114 * prod[0] + 0.587 * prod[1] + 0.299 * prod[2]
    fill = (0, 165, 255) if lum0 > 200 else (255, 255, 255)  # BGR: 橙 / 白
    M = cv2.getRotationMatrix2D((cx, cy), rot, 1.0)
    rot_mask = cv2.warpAffine(base_mask, M, (W, H))
    out = np.where((rot_mask > 0)[:, :, None], np.full_like(out, fill), out)
    # 文字颜色：白条配深灰字，橙条配白字
    tc = (245, 245, 245) if fill == (0, 165, 255) else (110, 110, 110)
    tw_max, th_max = (sw, sh) if sw >= sh else (sh, sw)
    # 4x 大字渲染再 INTER_AREA 缩小：直接小字号渲染笔画糊成团（防错清单 #22）
    big = 4.0
    thick = max(3, int(round(big * 3.2)))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, big, thick)
    ratio = min(tw_max * 0.88 / tw, th_max * 0.66 / th, 1.0)
    p2 = 16
    tmask = np.zeros((th + 2 * p2, tw + 2 * p2), np.uint8)
    cv2.putText(tmask, text, (p2, th + p2 // 2), cv2.FONT_HERSHEY_SIMPLEX,
                big, 255, thick, cv2.LINE_AA)
    pad_s = max(2, int(p2 * ratio))
    nw = max(8, int(round(tw * ratio))) + 2 * pad_s
    nh = max(6, int(round(th * ratio))) + 2 * pad_s
    tmask = cv2.resize(tmask, (nw, nh), interpolation=cv2.INTER_AREA)
    if sh > sw:  # 竖条：文字转 90°
        tmask = cv2.rotate(tmask, cv2.ROTATE_90_CLOCKWISE)
    th3, tw3 = tmask.shape[:2]
    side = int(max(th3, tw3, sw, sh) * 2) + 4
    pmask = np.zeros((side, side), np.uint8)
    py, px = (side - th3) // 2, (side - tw3) // 2
    pmask[py:py + th3, px:px + tw3] = tmask
    M2 = cv2.getRotationMatrix2D((side / 2, side / 2), rot, 1.0)
    M2[0, 2] += cx - side / 2
    M2[1, 2] += cy - side / 2
    wm = cv2.warpAffine(pmask, M2, (W, H))  # 灰度蒙版，自带抗锯齿
    # 透明度混合而非硬切：旋转后边缘平滑不毛糙
    a = (wm.astype(np.float32) / 255.0)[:, :, None]
    out = (out.astype(np.float32) * (1 - a) + np.full_like(out, tc, dtype=np.float32) * a).astype(np.uint8)
    return out


def draw_white_strip(img_bgr, box, text="your logo here", pm=None, pad_ratio=0.18):
    """轴对齐盒盖条（模板匹配/手动框选用）。内部走 draw_strip，底色同样自动取产品色。"""
    x, y, w, h = [int(v) for v in box[:4]]
    pts = np.array([[[x, y]], [[x + w, y]], [[x, y + h]], [[x + w, y + h]]], np.int32)
    pad = max(6, int(min(w, h) * pad_ratio))
    return draw_strip(img_bgr, pts, text=text, pm=pm, pad=pad)
