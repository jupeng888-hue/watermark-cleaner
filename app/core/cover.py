# -*- coding: utf-8 -*-
"""白底条覆盖：在产品上的敏感标志位置盖白底条 + "your logo here"。
敏感标志定位：用户提供 Logo 模板图做模板匹配（多尺度 + NMS），或手动框选。
竖版标志（如竖排文字 logo）自动旋转文字。"""
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


def draw_white_strip(img_bgr, box, text="your logo here", pad_ratio=0.18):
    """在 box 处绘制紧贴的白底圆角条 + 灰色文字；竖版区域自动旋转文字。"""
    out = img_bgr.copy()
    x, y, w, h = [int(v) for v in box[:4]]
    pw = max(6, int(w * pad_ratio))
    ph = max(6, int(h * pad_ratio))
    x0, y0 = max(0, x - pw), max(0, y - ph)
    x1, y1 = min(out.shape[1], x + w + pw), min(out.shape[0], y + h + ph)
    bw, bh = x1 - x0, y1 - y0
    # 圆角白底条
    radius = max(6, min(bw, bh) // 5)
    mask = np.zeros(out.shape[:2], np.uint8)
    cv2.rectangle(mask, (x0 + radius, y0), (x1 - radius, y1), 255, -1)
    cv2.rectangle(mask, (x0, y0 + radius), (x1, y1 - radius), 255, -1)
    for cx, cy in [(x0 + radius, y0 + radius), (x1 - radius, y0 + radius),
                   (x0 + radius, y1 - radius), (x1 - radius, y1 - radius)]:
        cv2.circle(mask, (cx, cy), radius, 255, -1)
    region = out[y0:y1, x0:x1]
    m3 = (mask[y0:y1, x0:x1] > 0)[:, :, None]
    out[y0:y1, x0:x1] = np.where(m3, np.full_like(region, 255), region)

    vertical = bh > bw * 1.3
    tw_max, th_max = (bh, bw) if vertical else (bw, bh)
    scale = 1.0
    for _ in range(30):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        if tw <= tw_max * 0.9 and th <= th_max * 0.6:
            break
        scale *= 0.9
    pad = 10
    txt_img = np.full((th + 2 * pad, tw + 2 * pad, 3), 255, np.uint8)
    cv2.putText(txt_img, text, (pad, th + pad // 2), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (150, 150, 150), 2, cv2.LINE_AA)
    if vertical:
        txt_img = cv2.rotate(txt_img, cv2.ROTATE_90_CLOCKWISE)
    th2, tw2 = txt_img.shape[:2]
    ty = y0 + max(0, (bh - th2) // 2)
    tx = x0 + max(0, (bw - tw2) // 2)
    ey, ex = min(ty + th2, y1), min(tx + tw2, x1)
    out[ty:ey, tx:ex] = txt_img[:ey - ty, :ex - tx]
    return out
