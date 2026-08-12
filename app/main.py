# -*- coding: utf-8 -*-
"""水印清理工具 - Windows 桌面版（纯本地、免费）
功能：识别产品外水印并批量去除；产品上敏感标志盖 "your logo here" 白底条。
依赖：pip install PySide6 opencv-python numpy onnxruntime (可选 pytesseract)
"""
import os
import sys
import cv2
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal, QRect, QPoint
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QListWidget, QListWidgetItem,
    QLabel, QPushButton, QCheckBox, QLineEdit, QFileDialog, QVBoxLayout,
    QHBoxLayout, QProgressBar, QMessageBox, QGroupBox, QSplitter,
)

from core import pipeline, model_manager


def cv2_to_qpixmap(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QPixmap.fromImage(QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy())


class PreviewLabel(QLabel):
    """预览图 + 鼠标拖拽框选（额外水印区域 / Logo 区域）。"""
    box_drawn = Signal(tuple)

    def __init__(self):
        super().__init__("拖入或添加图片后在此预览；按住左键拖拽可框选区域")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background:#2b2b2b;color:#aaa;")
        self._img = None
        self._start = None
        self._rect = None
        self.boxes = []  # 已确认框（原图坐标）

    def set_image(self, img_bgr):
        self._img = img_bgr
        self.boxes = []
        self._refresh()

    def _to_view(self, x, y):
        if self._img is None:
            return 0, 0
        pm_w, pm_h = self.width(), self.height()
        h, w = self._img.shape[:2]
        s = min(pm_w / w, pm_h / h)
        return int(x * s + (pm_w - w * s) / 2), int(y * s + (pm_h - h * s) / 2)

    def _to_img(self, vx, vy):
        h, w = self._img.shape[:2]
        s = min(self.width() / w, self.height() / h)
        ox, oy = (self.width() - w * s) / 2, (self.height() - h * s) / 2
        return int((vx - ox) / s), int((vy - oy) / s)

    def _refresh(self):
        if self._img is None:
            return
        vis = self._img.copy()
        for (x0, y0, x1, y1) in self.boxes:
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 200, 255), 2)
        pm = cv2_to_qpixmap(vis).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pm)

    def resizeEvent(self, e):
        self._refresh()
        super().resizeEvent(e)

    def mousePressEvent(self, e):
        if self._img is not None and e.button() == Qt.LeftButton:
            self._start = e.position().toPoint()

    def mouseReleaseEvent(self, e):
        if self._img is None or self._start is None:
            return
        p0, p1 = self._start, e.position().toPoint()
        x0, y0 = self._to_img(min(p0.x(), p1.x()), min(p0.y(), p1.y()))
        x1, y1 = self._to_img(max(p0.x(), p1.x()), max(p0.y(), p1.y()))
        h, w = self._img.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 - x0 > 5 and y1 - y0 > 5:
            self.boxes.append((x0, y0, x1, y1))
            self.box_drawn.emit((x0, y0, x1, y1))
            self._refresh()
        self._start = None


class Worker(QThread):
    progress = Signal(int, int, str)
    done = Signal(list)

    def __init__(self, paths, out_dir, opts):
        super().__init__()
        self.paths, self.out_dir, self.opts = paths, out_dir, opts

    def run(self):
        results = pipeline.process_batch(
            self.paths, self.out_dir, self.opts,
            progress_cb=lambda i, n, p: self.progress.emit(i, n, p))
        self.done.emit(results)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("水印清理工具（免费版）")
        self.resize(1100, 700)
        self.paths = []
        self.template = None
        self._build_ui()
        self._check_model()

    def _build_ui(self):
        root = QSplitter()
        # 左侧：文件列表 + 按钮
        left = QWidget(); lv = QVBoxLayout(left)
        self.file_list = QListWidget()
        btn_add = QPushButton("添加图片")
        btn_add.clicked.connect(self.add_images)
        btn_clear = QPushButton("清空列表")
        btn_clear.clicked.connect(lambda: (self.paths.clear(), self.file_list.clear()))
        lv.addWidget(QLabel("待处理图片（批量）："))
        lv.addWidget(self.file_list)
        lv.addWidget(btn_add); lv.addWidget(btn_clear)

        opt = QGroupBox("功能选项"); ov = QVBoxLayout(opt)
        self.ck_remove = QCheckBox("去除产品外的水印（自动检测）"); self.ck_remove.setChecked(True)
        self.ck_protect = QCheckBox("保护产品主体不被误擦"); self.ck_protect.setChecked(True)
        self.ck_strip = QCheckBox("产品上敏感标志盖白底条")
        self.strip_text = QLineEdit("your logo here")
        btn_tpl = QPushButton("选择敏感标志模板图（logo 小图）")
        btn_tpl.clicked.connect(self.pick_template)
        ov.addWidget(self.ck_remove); ov.addWidget(self.ck_protect)
        ov.addWidget(self.ck_strip); ov.addWidget(self.strip_text); ov.addWidget(btn_tpl)
        lv.addWidget(opt)

        self.progress = QProgressBar()
        btn_run = QPushButton("开始批量处理")
        btn_run.setStyleSheet("font-weight:bold;padding:8px;")
        btn_run.clicked.connect(self.run_batch)
        lv.addWidget(self.progress); lv.addWidget(btn_run)

        # 右侧：预览
        right = QWidget(); rv = QVBoxLayout(right)
        self.preview = PreviewLabel()
        self.preview.box_drawn.connect(self.on_box)
        btn_undo = QPushButton("撤销上一个框选")
        btn_undo.clicked.connect(lambda: (self.preview.boxes and self.preview.boxes.pop(), self.preview._refresh()))
        rv.addWidget(QLabel("预览（拖拽框选：勾去水印=补充水印区域；勾白底条=指定标志位置）："))
        rv.addWidget(self.preview); rv.addWidget(btn_undo)

        root.addWidget(left); root.addWidget(right)
        root.setSizes([420, 680])
        self.setCentralWidget(root)

    def _check_model(self):
        if model_manager.lama_available():
            return
        ret = QMessageBox.question(
            self, "下载去水印模型",
            "首次使用需要下载免费开源的 LaMa 去水印模型（约 200MB，仅下载一次）。\n"
            "现在下载？（选否则先用简易修复兜底，效果略差）")
        if ret == QMessageBox.Yes:
            try:
                model_manager.download_lama()
                QMessageBox.information(self, "完成", "模型下载完成。")
            except Exception as e:
                QMessageBox.warning(self, "下载失败", str(e))

    def add_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "", "图片 (*.jpg *.jpeg *.png *.bmp *.webp)")
        for f in files:
            if f not in self.paths:
                self.paths.append(f)
                QListWidgetItem(os.path.basename(f), self.file_list)
        if files:
            self.show_image(files[0])

    def show_image(self, path):
        img = pipeline.imread_unicode(path)
        if img is not None:
            self.preview.set_image(img)

    def on_box(self, box):
        pass  # 框已记录在 preview.boxes，批量处理时按选项使用

    def pick_template(self):
        f, _ = QFileDialog.getOpenFileName(self, "选择 logo 模板图", "", "图片 (*.png *.jpg)")
        if f:
            self.template = pipeline.imread_unicode(f)
            QMessageBox.information(self, "模板已加载", os.path.basename(f))

    def run_batch(self):
        if not self.paths:
            QMessageBox.warning(self, "提示", "请先添加图片。")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if not out_dir:
            return
        manual_boxes = list(self.preview.boxes)
        opts = {
            "remove_watermark": self.ck_remove.isChecked(),
            "protect_product": self.ck_protect.isChecked(),
            "strip_enabled": self.ck_strip.isChecked(),
            "strip_text": self.strip_text.text() or "your logo here",
            "logo_template": self.template,
            "extra_boxes": manual_boxes if self.ck_remove.isChecked() else None,
            "logo_boxes": manual_boxes if (self.ck_strip.isChecked() and not self.ck_remove.isChecked()) else None,
        }
        self.worker = Worker(self.paths, out_dir, opts)
        self.worker.progress.connect(lambda i, n, p: (
            self.progress.setMaximum(n), self.progress.setValue(i),
            self.progress.setFormat(f"%v/%m  {os.path.basename(p)}")))
        self.worker.done.connect(self.on_done)
        self.worker.start()

    def on_done(self, results):
        ok = sum(1 for _, dst, err, _ in results if dst)
        fails = [(p, err) for p, dst, err, _ in results if err]
        msg = f"完成 {ok}/{len(results)} 张。"
        if fails:
            msg += "\n失败：\n" + "\n".join(f"{os.path.basename(p)}: {e}" for p, e in fails[:5])
        QMessageBox.information(self, "批量处理结束", msg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
