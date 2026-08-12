# -*- coding: utf-8 -*-
"""GUI 冒烟测试：离屏模式（offscreen）下主窗口能创建、控件齐全，不需要显示器。"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    yield a


def test_mainwindow_constructs(app, monkeypatch):
    monkeypatch.setattr("core.model_manager.lama_available", lambda: True)  # 跳过下载弹窗
    import main
    win = main.MainWindow()
    assert win.ck_remove.isChecked()
    assert win.ck_protect.isChecked()
    assert win.strip_text.text() == "your logo here"
    assert win.preview is not None
    win.close()
