# -*- coding: utf-8 -*-
"""批量处理流水线：读图 -> 检测水印 -> 去水印 -> 盖白底条 -> 导出。"""
import os
import cv2

from . import detect, inpaint, cover

SUPPORTED = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def imread_unicode(path):
    import numpy as np
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if ok:
        buf.tofile(path)
    return ok


def process_image(img_bgr, opts):
    """opts: dict(remove_watermark, extra_boxes, protect_product,
                 logo_template, logo_boxes, strip_text, strip_enabled)"""
    out = img_bgr
    info = {"watermark_boxes": []}

    if opts.get("remove_watermark", True):
        mask, boxes = detect.detect_watermark_mask(
            out, extra_boxes=opts.get("extra_boxes"),
            protect_product=opts.get("protect_product", True))
        info["watermark_boxes"] = boxes
        out = inpaint.inpaint(out, mask)

    if opts.get("strip_enabled"):
        logo_boxes = list(opts.get("logo_boxes") or [])
        if opts.get("logo_template") is not None:
            logo_boxes += cover.find_logo_boxes(out, opts["logo_template"])
        if not logo_boxes:
            # 无模板时：自动识别产品内部的高对比印刷标志（字母/图形聚类）
            pm = detect.product_mask(out)
            logo_boxes += detect.logo_boxes_auto(out, pm)
        seen = []
        for b in logo_boxes:
            if not any(abs(b[0] - s[0]) < 10 and abs(b[1] - s[1]) < 10 for s in seen):
                seen.append(b)
                out = cover.draw_white_strip(out, b, text=opts.get("strip_text", "your logo here"))
        info["logo_count"] = len(seen)
    return out, info


def process_batch(paths, out_dir, opts, progress_cb=None):
    os.makedirs(out_dir, exist_ok=True)
    ext = ".png" if opts.get("png") else ".jpg"  # PNG=无损导出
    results = []
    for i, p in enumerate(paths):
        try:
            img = imread_unicode(p)
            if img is None:
                raise ValueError("图片解码失败")
            out, info = process_image(img, opts)
            dst = os.path.join(out_dir, os.path.splitext(os.path.basename(p))[0] + "_clean" + ext)
            imwrite_unicode(dst, out)
            results.append((p, dst, None, info))
        except Exception as e:
            results.append((p, None, str(e), None))
        if progress_cb:
            progress_cb(i + 1, len(paths), p)
    return results
