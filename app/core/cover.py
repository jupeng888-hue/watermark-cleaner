# -*- coding: utf-8 -*-
"""白底条覆盖：在产品上的敏感标志位置盖白底圆角条 + "your logo here"。
敏感标志定位：用户提供 Logo 模板图做模板匹配（多尺度），或手动框选。"""
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
    # NMS 去重
    if not rects:
        return []
    boxes = np.array([[x, y, x + w, y + h] for x, y, w, h, _ in rects], np.float32)
    scores = np.array([s for *_, s in rects], np.float32)
    keep = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), threshold, 0.3)
    keep = np.array(keep).flatten() if len(keep) else []
    return [rects[i] for i in keep]


def draw_white_strip(img_bgr, box, text="your logo here", pad_ratio=0.35):
    """在 box 处绘制白底圆角条 + 灰色文字。"""
    out = img_bgr.copy()
    x, y, w, h = [int(v) for v in box[:4]]
    pw, ph = int(w * pad_ratio), int(h * pad_ratio)
    x0, y0 = max(0, x - pw), max(0, y - ph)
    x1, y1 = min(out.shape[1], x + w + pw), min(out.shape[0], y + h + ph)
    bw, bh = x1 - x0, y1 - y0
    radius = max(4, bh // 6)
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), -1)
    # 圆角化四角
    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    corners = np.zeros_like(out)
    cv2.rectangle(corners, (x0, y0), (x1, y1), (255, 255, 255), -1)
    mask = np.zeros(out.shape[:2], np.uint8)
    cv2.rectangle(mask, (x0 + radius, y0), (x1 - radius, y1), 255, -1)
    cv2.rectangle(mask, (x0, y0 + radius), (x1, y1 - radius), 255, -1)
    for cx, cy in [(x0 + radius, y0 + radius), (x1 - radius, y0 + radius),
                   (x0 + radius, y1 - radius), (x1 - radius, y1 - radius)]:
        cv2.circle(mask, (cx, cy), radius, 255, -1)
    region = out[y0:y1, x0:x1]
    m3 = (mask[y0:y1, x0:x1] > 0)[:, :, None]
    white = np.full_like(region, 255)
    out[y0:y1, x0:x1] = np.where(m3, white, region)
    # 文字自适应大小
    scale = 0.5
    for _ in range(20):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        if tw <= bw * 0.85 and th <= bh * 0.5:
            break
        scale *= 0.9
    tx = x0 + (bw - tw) // 2
    ty = y0 + (bh + th) // 2
    cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (150, 150, 150), 1, cv2.LINE_AA)
    return out
