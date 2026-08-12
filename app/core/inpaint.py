# -*- coding: utf-8 -*-
"""去水印：优先 LaMa ONNX（onnxruntime，CPU 可跑，有 GPU 自动加速），
模型未下载时降级 OpenCV inpaint 兜底。"""
import cv2
import numpy as np

from . import model_manager

_LAMA_SIZE = 512  # lama_fp32.onnx 固定输入尺寸
_session = None


def _get_session():
    global _session
    if _session is not None:
        return _session
    if not model_manager.lama_available():
        return None
    import onnxruntime as ort
    providers = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
    providers = [p for p in providers if p in ort.get_available_providers()] or ["CPUExecutionProvider"]
    _session = ort.InferenceSession(model_manager.lama_path(), providers=providers)
    return _session


def _lama_block(img_bgr, mask):
    """对单个 512 块跑 LaMa。img_bgr: HxWx3 uint8，mask: HxW uint8(255=待修复)。"""
    sess = _get_session()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img_rgb, (_LAMA_SIZE, _LAMA_SIZE)).astype(np.float32) / 255.0
    msk = cv2.resize(mask, (_LAMA_SIZE, _LAMA_SIZE), interpolation=cv2.INTER_NEAREST)
    msk = (msk > 0).astype(np.float32)[None, None, :, :]
    img = img.transpose(2, 0, 1)[None, :, :, :]
    out = sess.run(None, {"image": img, "mask": msk})[0]
    out = out[0].transpose(1, 2, 0)
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return cv2.resize(out, (img_bgr.shape[1], img_bgr.shape[0]))


def _iter_mask_blocks(mask, pad=64):
    """按连通区域把 mask 切成若干带 padding 的块，避免整图缩放损失细节。"""
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    h, w = mask.shape
    for i in range(1, num):
        x, y, bw, bh, _ = stats[i]
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        block_mask = np.zeros_like(mask)
        block_mask[y0:y1, x0:x1] = np.where(labels[y0:y1, x0:x1] == i, 255, 0)
        yield (x0, y0, x1, y1), block_mask


def inpaint(img_bgr, mask, dilate=3):
    """img_bgr + mask(255=水印区域) -> 修复后的图。自动选择 LaMa 或 OpenCV。"""
    if mask is None or cv2.countNonZero(mask) == 0:
        return img_bgr
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=dilate)

    if _get_session() is not None:
        result = img_bgr.copy()
        for (x0, y0, x1, y1), block_mask in _iter_mask_blocks(mask):
            block_img = result[y0:y1, x0:x1]
            bm = block_mask[y0:y1, x0:x1]
            fixed = _lama_block(block_img, bm)
            m3 = (bm > 0)[:, :, None]
            result[y0:y1, x0:x1] = np.where(m3, fixed, block_img)
        return result
    # 兜底：OpenCV（效果一般，仅应急）
    return cv2.inpaint(img_bgr, mask, 5, cv2.INPAINT_TELEA)
