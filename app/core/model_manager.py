# -*- coding: utf-8 -*-
"""模型管理：首次使用时下载 LaMa ONNX 模型（免费开源，Apache-2.0）。
不打包进安装包，保持安装包小。支持 HF 官方源与 hf-mirror 镜像。"""
import os
import urllib.request

MODEL_DIR = os.path.join(os.path.expanduser("~"), ".watermark_cleaner", "models")

LAMA_ONNX_NAME = "lama_fp32.onnx"
LAMA_URLS = [
    "https://hf-mirror.com/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",  # 国内镜像优先
    "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
]


def lama_path():
    return os.path.join(MODEL_DIR, LAMA_ONNX_NAME)


def lama_available():
    p = lama_path()
    return os.path.isfile(p) and os.path.getsize(p) > 10 * 1024 * 1024


def download_lama(progress_cb=None):
    """下载 LaMa ONNX，progress_cb(downloaded_bytes) 回调。返回模型路径。"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    dst = lama_path()
    tmp = dst + ".part"
    last_err = None
    for url in LAMA_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    if progress_cb:
                        progress_cb(os.path.getsize(tmp))
            os.replace(tmp, dst)
            return dst
        except Exception as e:  # 换下一个源
            last_err = e
            if os.path.exists(tmp):
                os.remove(tmp)
    raise RuntimeError(f"模型下载失败，请检查网络后重试：{last_err}")
