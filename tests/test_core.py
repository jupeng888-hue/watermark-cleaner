# -*- coding: utf-8 -*-
"""核心流水线自动化测试：用合成图验证 检测/去水印/盖条/批量 全流程。"""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from core import detect, inpaint, cover, pipeline  # noqa: E402


@pytest.fixture(scope="module")
def sample():
    img = np.full((800, 1000, 3), 245, np.uint8)
    rng = np.random.default_rng(7)
    for _ in range(30):
        cv2.circle(img, (int(rng.integers(0, 1000)), int(rng.integers(0, 800))),
                   int(rng.integers(10, 60)), (220, 220, 220), -1)
    cv2.rectangle(img, (400, 250), (600, 600), (60, 120, 200), -1)
    cv2.rectangle(img, (460, 180), (540, 250), (60, 120, 200), -1)
    cv2.putText(img, "BrandX", (430, 440), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    cv2.putText(img, "SAMPLE WATERMARK", (80, 720), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (120, 120, 120), 2)
    template = img[410:450, 425:580].copy()
    return img, template


def test_product_mask_covers_product(sample):
    img, _ = sample
    pm = detect.product_mask(img)
    assert pm is not None and pm.shape == img.shape[:2]
    assert pm[420, 500] == 255
    assert pm[50, 50] == 0
    ratio = cv2.countNonZero(pm) / pm.size
    assert 0.02 < ratio < 0.6


def test_watermark_mask_excludes_product(sample):
    img, _ = sample
    mask, boxes = detect.detect_watermark_mask(
        img, extra_boxes=[(70, 685, 470, 730)])
    assert mask.shape == img.shape[:2]
    assert cv2.countNonZero(mask) > 0
    pm = detect.product_mask(img)
    overlap = cv2.countNonZero(cv2.bitwise_and(mask, pm))
    assert overlap == 0


def test_inpaint_removes_watermark(sample):
    img, _ = sample
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.rectangle(mask, (70, 685), (470, 730), 255, -1)
    out = inpaint.inpaint(img, mask)
    assert out.shape == img.shape
    before = img[690:725, 80:465].std()
    after = out[690:725, 80:465].std()
    assert after < before


def test_inpaint_empty_mask_passthrough(sample):
    img, _ = sample
    out = inpaint.inpaint(img, np.zeros(img.shape[:2], np.uint8))
    assert np.array_equal(out, img)


def test_logo_match_and_strip(sample):
    img, template = sample
    boxes = cover.find_logo_boxes(img, template, threshold=0.6)
    assert len(boxes) >= 1
    x, y, w, h, score = boxes[0]
    assert abs(x - 425) < 20 and abs(y - 410) < 20
    out = cover.draw_white_strip(img, boxes[0], text="your logo here")
    strip_region = out[y:y + h, x:x + w]
    assert strip_region.mean() > 200


def test_batch_pipeline(sample, tmp_path):
    img, template = sample
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    paths = []
    for i in range(2):
        p = in_dir / f"tu{i}.jpg"
        assert pipeline.imwrite_unicode(str(p), img)
        paths.append(str(p))
    opts = {"remove_watermark": True, "protect_product": True,
            "strip_enabled": True, "strip_text": "your logo here",
            "logo_template": template,
            "extra_boxes": [(70, 685, 470, 730)]}
    results = pipeline.process_batch(paths, str(tmp_path / "out"), opts)
    assert len(results) == 2
    for _, dst, err, _ in results:
        assert err is None and os.path.isfile(dst)


def test_unicode_io(sample, tmp_path):
    img, _ = sample
    p = str(tmp_path / "中文路径测试.jpg")
    assert pipeline.imwrite_unicode(p, img)
    back = pipeline.imread_unicode(p)
    assert back is not None and back.shape == img.shape
