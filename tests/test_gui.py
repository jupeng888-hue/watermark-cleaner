# -*- coding: utf-8 -*-
"""GUI 自动化测试：离屏模式下真实驱动「添加图片 → 点开始 → 批量处理 → 导出」全流程。"""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox, QFileDialog  # noqa: E402


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    yield a


@pytest.fixture()
def win(app, monkeypatch):
    import main
    monkeypatch.setattr(main.MainWindow, "_check_model", lambda self: None)  # 跳过下载弹窗
    w = main.MainWindow()
    yield w
    w.close()


def _sample_image():
    img = np.full((600, 800, 3), 245, np.uint8)
    cv2.rectangle(img, (300, 200), (500, 500), (60, 120, 200), -1)
    cv2.putText(img, "SAMPLE", (60, 560), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (120, 120, 120), 2)
    return img


def test_mainwindow_constructs(win):
    assert win.ck_remove.isChecked()
    assert win.ck_protect.isChecked()
    assert win.strip_text.text() == "your logo here"
    assert win.preview is not None
    assert win.btn_run.isEnabled()


def test_run_batch_empty_shows_warning(win, monkeypatch):
    called = {}
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: called.setdefault("warned", True))
    win.run_batch()
    assert called.get("warned")  # 空列表必须有提示，不能静默


def test_run_batch_cancel_dialog_shows_status(win, monkeypatch):
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: "")
    win.paths.append("fake.jpg")  # 直接注入路径，绕过文件对话框
    win.run_batch()
    assert "已取消" in win.statusBar().currentMessage()


def test_full_batch_flow(win, monkeypatch, tmp_path):
    """模拟真实使用：加图 → 选输出目录 → 后台线程处理 → 产出文件。"""
    from core import pipeline
    img_path = str(tmp_path / "产品图.jpg")
    assert pipeline.imwrite_unicode(img_path, _sample_image())
    out_dir = str(tmp_path / "out")

    monkeypatch.setattr(QFileDialog, "getOpenFileNames",
                        lambda *a, **k: ([img_path], ""))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: out_dir)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)

    win.add_images()
    assert len(win.paths) == 1
    assert win.preview._img is not None  # 预览已加载

    win.run_batch()
    assert not win.btn_run.isEnabled()   # 处理中按钮应禁用

    # 等后台线程真正跑完（offscreen 下事件循环需手动驱动）
    win.worker.wait(30000)
    QApplication.processEvents()
    assert win.btn_run.isEnabled()       # 完成后恢复

    outputs = [f for f in os.listdir(out_dir) if f.endswith("_clean.jpg")]
    assert len(outputs) == 1             # 结果文件真实产出
    result = pipeline.imread_unicode(os.path.join(out_dir, outputs[0]))
    assert result is not None and result.shape == (600, 800, 3)
