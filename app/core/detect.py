# -*- coding: utf-8 -*-
"""水印检测：定位「产品之外」的文字水印。
- 产品主体：OpenCV 显著性检测 + GrabCut 细化（白底电商图效果好，零模型）
- 文字水印：优先 Tesseract OCR（装了就用），否则 MSER 笔画启发式
- 最终水印蒙版 = 文字区域 - 产品区域（膨胀后），保护产品不被误擦
"""
import shutil
import cv2
import numpy as np


def product_mask(img_bgr):
    """返回产品主体蒙版（255=产品）。白底/纯色底产品图适用。"""
    h, w = img_bgr.shape[:2]
    # 背景色估计：取四角中位色，距离背景色远的像素视为前景（白底/纯色底产品图适用）
    c = 20
    corners = np.concatenate([
        img_bgr[:c, :c].reshape(-1, 3), img_bgr[:c, -c:].reshape(-1, 3),
        img_bgr[-c:, :c].reshape(-1, 3), img_bgr[-c:, -c:].reshape(-1, 3)])
    bg = np.median(corners, axis=0)
    dist = np.linalg.norm(img_bgr.astype(np.float32) - bg, axis=2)
    seed = (dist > 30).astype(np.uint8)
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    # 取最大连通域作为 GrabCut 初始前景
    num, labels, stats, _ = cv2.connectedComponentsWithStats(seed, 8)
    if num <= 1:
        return np.zeros((h, w), np.uint8)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[biggest, cv2.CC_STAT_AREA] < 0.01 * h * w:
        return np.zeros((h, w), np.uint8)  # 前景太小，认为没有明显产品
    seed = (labels == biggest).astype(np.uint8)
    x, y, bw, bh = stats[biggest, 0], stats[biggest, 1], stats[biggest, 2], stats[biggest, 3]
    rect = (max(0, x - 10), max(0, y - 10), min(w, bw + 20), min(h, bh + 20))
    gc_mask = np.where(seed > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img_bgr, gc_mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
        out = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        out = seed * 255
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


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


def _text_boxes_mser(img_bgr):
    """无 tesseract 时的笔画启发式文字区域检测。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create(_min_area=30, _max_area=8000)
    regions, _ = mser.detectRegions(gray)
    mask = np.zeros_like(gray)
    for r in regions:
        cv2.fillConvexPoly(mask, r, 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w >= 12 and 4 <= h <= 200 and w / max(h, 1) > 0.8:
            boxes.append((x, y, w, h))
    return boxes


def detect_watermark_mask(img_bgr, extra_boxes=None, protect_product=True):
    """返回 (mask, boxes)。mask 255=建议擦除的水印区域（已排除产品）。"""
    h, w = img_bgr.shape[:2]
    boxes = _text_boxes_tesseract(img_bgr) if shutil.which("tesseract") or True else None
    if not boxes:
        boxes = _text_boxes_mser(img_bgr)

    text_mask = np.zeros((h, w), np.uint8)
    for (x, y, bw, bh) in boxes:
        cv2.rectangle(text_mask, (x, y), (x + bw, y + bh), 255, -1)
    text_mask = cv2.dilate(text_mask, np.ones((5, 5), np.uint8))

    if protect_product:
        pm = cv2.dilate(product_mask(img_bgr), np.ones((15, 15), np.uint8), iterations=2)
        text_mask = cv2.bitwise_and(text_mask, cv2.bitwise_not(pm))

    if extra_boxes:  # 用户手动框选区域（强制纳入，不受产品保护限制时由 GUI 传入开关）
        for (x0, y0, x1, y1) in extra_boxes:
            cv2.rectangle(text_mask, (x0, y0), (x1, y1), 255, -1)
    return text_mask, boxes
