# -*- coding: utf-8 -*-
"""水印清理工具 - Windows 桌面版（纯本地、免费）
功能：识别产品外水印并批量去除；产品上敏感标志盖 "your logo here" 白底条。
"""
import os
import sys
import logging
import traceback
import cv2

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QListWidget, QListWidgetItem,
    QLabel, QPushButton, QCheckBox, QLineEdit, QFileDialog, QVBoxLayout,
    QProgressBar, QMessageBox, QGroupBox, QSplitter,
)

from core import pipeline, model_manager

# ---- 日志：写在程序旁边的 log 文件，出错有据可查 ----
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "watermark_cleaner.log")
logging.basicConfig(
    filename=LOG_FILE, level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8")
log = logging.getLogger("app")


def install_excepthook():
    """任何未捕获异常都弹窗 + 写日志，不再静默无反应。"""
    def hook(exc_type, exc, tb):
        msg = "".join(traceback.format_exception(exc_type, exc, tb))
        log.error("未捕获异常:\n%s", msg)
        QMessageBox.critical(None, "出错了",
                             f"程序遇到错误，详细信息已写入日志：\n{LOG_FILE}\n\n{exc}")
    sys.excepthook = hook


def cv2_to_qpixmap(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QPixmap.fromImage(QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy())


class PreviewLabel(QLabel):
    """预览图 + 鼠标拖拽框选（额外水印区域 / Logo 区域）。"""
    box_drawn = Signal(tuple)

    def __init__(self):
        super().__init__("添加图片后在此预览；按住左键拖拽可框选区域")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background:#2b2b2b;color:#aaa;")
        self._img = None
        self._start = None
        self.boxes = []

    def set_image(self, img_bgr):
        self._img = img_bgr
        self.boxes = []
        self._refresh()

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
        try:
            results = pipeline.process_batch(
                self.paths, self.out_dir, self.opts,
                progress_cb=lambda i, n, p: self.progress.emit(i, n, p))
            self.done.emit(results)
        except Exception:
            log.error("批量处理线程异常:\n%s", traceback.format_exc())
            self.done.emit([(p, None, traceback.format_exc(limit=3), None) for p in self.paths])


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("水印清理工具（免费版）")
        self.resize(1100, 700)
        self.paths = []
        self.template = None
        self.last_out_dir = os.path.join(BASE_DIR, "输出结果")
        self.worker = None
        self._build_ui()
        self.statusBar().showMessage(f"就绪。日志文件：{LOG_FILE}")
        self._check_model()

    def _build_ui(self):
        root = QSplitter()
        left = QWidget(); lv = QVBoxLayout(left)
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self._on_item_clicked)
        btn_add = QPushButton("添加图片")
        btn_add.clicked.connect(self.add_images)
        btn_clear = QPushButton("清空列表")
        btn_clear.clicked.connect(lambda: (self.paths.clear(), self.file_list.clear()))
        lv.addWidget(QLabel("待处理图片（批量，点击列表项可预览）："))
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
        self.btn_run = QPushButton("开始批量处理")
        self.btn_run.setStyleSheet("font-weight:bold;padding:8px;")
        self.btn_run.clicked.connect(self.run_batch)
        lv.addWidget(self.progress); lv.addWidget(self.btn_run)

        right = QWidget(); rv = QVBoxLayout(right)
        self.preview = PreviewLabel()
        self.preview.box_drawn.connect(lambda b: self.statusBar().showMessage(f"已框选区域 {b}"))
        btn_undo = QPushButton("撤销上一个框选")
        btn_undo.clicked.connect(lambda: (self.preview.boxes and self.preview.boxes.pop(), self.preview._refresh()))
        rv.addWidget(QLabel("预览（拖拽框选：勾去水印=补充水印区域；勾白底条=指定标志位置）："))
        rv.addWidget(self.preview); rv.addWidget(btn_undo)

        root.addWidget(left); root.addWidget(right)
        root.setSizes([420, 680])
        self.setCentralWidget(root)

    def _on_item_clicked(self, item):
        row = self.file_list.row(item)
        if 0 <= row < len(self.paths):
            self.show_image(self.paths[row])

    def _check_model(self):
        if model_manager.lama_available():
            self.statusBar().showMessage("LaMa AI 模型已就绪。")
            return
        ret = QMessageBox.question(
            self, "下载去水印模型",
            "首次使用需要下载免费开源的 LaMa 去水印模型（约 200MB，仅下载一次）。\n"
            "现在下载？（选否则先用简易修复兜底，效果略差）")
        if ret == QMessageBox.Yes:
            self.statusBar().showMessage("模型下载中，请稍候……")
            QApplication.processEvents()
            try:
                model_manager.download_lama()
                QMessageBox.information(self, "完成", "模型下载完成。")
            except Exception as e:
                log.error("模型下载失败: %s", e)
                QMessageBox.warning(self, "下载失败",
                                    f"{e}\n\n不影响使用，将先用简易修复兜底。")

    def add_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "", "图片 (*.jpg *.jpeg *.png *.bmp *.webp)")
        for f in files:
            if f not in self.paths:
                self.paths.append(f)
                QListWidgetItem(os.path.basename(f), self.file_list)
        if files:
            self.show_image(files[-1])
            self.statusBar().showMessage(f"已添加 {len(self.paths)} 张图片。")

    def show_image(self, path):
        img = pipeline.imread_unicode(path)
        if img is None:
            QMessageBox.warning(self, "提示", f"无法读取图片：{path}")
            return
        self.preview.set_image(img)

    def pick_template(self):
        f, _ = QFileDialog.getOpenFileName(self, "选择 logo 模板图", "", "图片 (*.png *.jpg)")
        if f:
            self.template = pipeline.imread_unicode(f)
            if self.template is None:
                QMessageBox.warning(self, "提示", "模板图读取失败。")
            else:
                self.statusBar().showMessage(f"模板已加载：{os.path.basename(f)}")

    def run_batch(self):
        if not self.paths:
            QMessageBox.warning(self, "提示", "请先点「添加图片」选择要处理的图片。")
            return
        out_dir = QFileDialog.getExistingDirectory(
            self, "选择输出文件夹（处理结果会保存到这里）", self.last_out_dir)
        if not out_dir:
            self.statusBar().showMessage("已取消：未选择输出文件夹。")
            return
        self.last_out_dir = out_dir
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
        log.info("开始批量处理 %d 张，输出到 %s", len(self.paths), out_dir)
        self.btn_run.setEnabled(False)
        self.btn_run.setText("处理中，请稍候……")
        self.progress.setValue(0)
        self.worker = Worker(self.paths, out_dir, opts)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self.on_done)
        self.worker.start()
        self.statusBar().showMessage("批量处理进行中……")

    def _on_progress(self, i, n, p):
        self.progress.setMaximum(n)
        self.progress.setValue(i)
        self.progress.setFormat(f"%v/%m  {os.path.basename(p)}")

    def on_done(self, results):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("开始批量处理")
        ok = sum(1 for _, dst, err, _ in results if dst)
        fails = [(p, err) for p, dst, err, _ in results if err]
        log.info("批量处理结束：成功 %d 失败 %d", ok, len(fails))
        msg = f"处理完成：成功 {ok}/{len(results)} 张。\n输出位置：{self.last_out_dir}"
        if fails:
            msg += "\n\n失败明细：\n" + "\n".join(
                f"{os.path.basename(p)}: {str(e)[:200]}" for p, e in fails[:5])
            msg += f"\n\n完整日志：{LOG_FILE}"
        QMessageBox.information(self, "批量处理结束", msg)
        self.statusBar().showMessage(f"完成 {ok}/{len(results)}，输出：{self.last_out_dir}")
        if ok and sys.platform.startswith("win"):
            os.startfile(self.last_out_dir)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    install_excepthook()
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
