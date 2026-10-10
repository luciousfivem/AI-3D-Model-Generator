"""Left panel: describe an object or pick a picture, then turn it into 3D."""

from __future__ import annotations

import random

from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets

from ..engines import QUALITY_PRESETS, STYLE_PRESETS, EngineInfo


def pil_to_pixmap(image: Image.Image, size: int | None = None) -> QtGui.QPixmap:
    """PIL image -> QPixmap (alpha shown over a checkerboard-neutral dark background)."""
    rgba = image.convert("RGBA")
    if size:
        rgba = rgba.copy()
        rgba.thumbnail((size, size))
    data = rgba.tobytes("raw", "RGBA")
    qimage = QtGui.QImage(data, rgba.width, rgba.height, 4 * rgba.width, QtGui.QImage.Format_RGBA8888).copy()
    return QtGui.QPixmap.fromImage(qimage)


class CreatePanel(QtWidgets.QWidget):
    draw_requested = QtCore.Signal(str, str, int)  # prompt, style key, seed
    open_picture_requested = QtCore.Signal()
    make3d_requested = QtCore.Signal(str, str, int, bool)  # engine key, quality key, seed, remove background
    cancel_requested = QtCore.Signal()
    engine_changed = QtCore.Signal(str)

    def __init__(self, engines: list[EngineInfo], settings, parent=None):
        super().__init__(parent)
        self._engines = {e.key: e for e in engines}
        self._candidates: list[Image.Image] = []
        self._reference: Image.Image | None = None
        self._busy = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self.banner = QtWidgets.QLabel()
        self.banner.setObjectName("banner")
        self.banner.setWordWrap(True)
        self.banner.hide()
        layout.addWidget(self.banner)

        # ---- 1. picture
        box = QtWidgets.QGroupBox("1  Get a picture")
        form = QtWidgets.QVBoxLayout(box)
        self.prompt = QtWidgets.QPlainTextEdit()
        self.prompt.setPlaceholderText("Describe what you want to print, e.g.\n“a small dragon sitting on a rock”")
        self.prompt.setFixedHeight(66)
        form.addWidget(self.prompt)

        row = QtWidgets.QHBoxLayout()
        self.style = QtWidgets.QComboBox()
        for key, (label, _) in STYLE_PRESETS.items():
            self.style.addItem(label, key)
        self.style.setCurrentIndex(1)  # figurine: the most print-friendly default
        row.addWidget(self.style, 1)
        self.draw_button = QtWidgets.QPushButton("Draw 4 pictures")
        self.draw_button.setObjectName("primary")
        self.draw_button.setToolTip("Creates four reference pictures from your description.\nYou pick the best one.")
        row.addWidget(self.draw_button)
        form.addLayout(row)

        self.open_button = QtWidgets.QPushButton("…or open your own picture")
        form.addWidget(self.open_button)

        self.gallery = QtWidgets.QListWidget()
        self.gallery.setViewMode(QtWidgets.QListView.IconMode)
        self.gallery.setIconSize(QtCore.QSize(92, 92))
        self.gallery.setResizeMode(QtWidgets.QListView.Adjust)
        self.gallery.setMovement(QtWidgets.QListView.Static)
        self.gallery.setSpacing(4)
        self.gallery.setFixedHeight(108)
        self.gallery.hide()
        form.addWidget(self.gallery)
        layout.addWidget(box)

        # ---- reference preview
        self.preview = QtWidgets.QLabel("No picture yet")
        self.preview.setObjectName("hint")
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setFixedHeight(150)
        self.preview.setStyleSheet("background:#161a1f; border:1px solid #343b44; border-radius:6px;")
        layout.addWidget(self.preview)

        # ---- 2. make 3D
        box2 = QtWidgets.QGroupBox("2  Make it 3D")
        grid = QtWidgets.QGridLayout(box2)
        grid.addWidget(QtWidgets.QLabel("Model"), 0, 0)
        self.engine = QtWidgets.QComboBox()
        for info in engines:
            self.engine.addItem(info.label, info.key)
        index = self.engine.findData(settings.engine)
        self.engine.setCurrentIndex(max(index, 0))
        grid.addWidget(self.engine, 0, 1, 1, 2)

        grid.addWidget(QtWidgets.QLabel("Detail"), 1, 0)
        self.quality = QtWidgets.QComboBox()
        for key, preset in QUALITY_PRESETS.items():
            self.quality.addItem(preset["label"], key)
        self.quality.setCurrentIndex(max(self.quality.findData(settings.quality), 0))
        grid.addWidget(self.quality, 1, 1, 1, 2)

        grid.addWidget(QtWidgets.QLabel("Variation"), 2, 0)
        self.seed = QtWidgets.QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(random.randint(0, 999_999))
        self.seed.setToolTip("Same picture + same variation number = same model.\nChange it to get a different result.")
        grid.addWidget(self.seed, 2, 1)
        self.dice = QtWidgets.QToolButton()
        self.dice.setText("🎲")
        self.dice.setToolTip("New random variation")
        grid.addWidget(self.dice, 2, 2)

        self.remove_bg = QtWidgets.QCheckBox("Remove the picture's background")
        self.remove_bg.setChecked(settings.remove_background)
        grid.addWidget(self.remove_bg, 3, 0, 1, 3)

        self.engine_note = QtWidgets.QLabel()
        self.engine_note.setObjectName("hint")
        self.engine_note.setWordWrap(True)
        grid.addWidget(self.engine_note, 4, 0, 1, 3)

        self.make_button = QtWidgets.QPushButton("Make 3D model")
        self.make_button.setObjectName("primary")
        self.make_button.setEnabled(False)
        grid.addWidget(self.make_button, 5, 0, 1, 3)
        layout.addWidget(box2)

        # ---- progress
        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        progress_row = QtWidgets.QHBoxLayout()
        progress_row.addWidget(self.progress, 1)
        progress_row.addWidget(self.cancel_button)
        self.status.hide()
        self.progress.hide()
        self.cancel_button.hide()
        layout.addWidget(self.status)
        layout.addLayout(progress_row)
        layout.addStretch(1)

        # ---- wiring
        self.draw_button.clicked.connect(self._emit_draw)
        self.open_button.clicked.connect(self.open_picture_requested)
        self.make_button.clicked.connect(self._emit_make)
        self.cancel_button.clicked.connect(self.cancel_requested)
        self.dice.clicked.connect(lambda: self.seed.setValue(random.randint(0, 999_999)))
        self.gallery.currentRowChanged.connect(self._candidate_chosen)
        self.engine.currentIndexChanged.connect(self._engine_changed)
        self._engine_changed()

    # ------------------------------------------------------------------ state

    def selected_engine(self) -> str:
        return self.engine.currentData()

    def reference_image(self) -> Image.Image | None:
        return self._reference

    def set_banner(self, text: str) -> None:
        self.banner.setText(text)
        self.banner.setVisible(bool(text))

    def select_engine(self, key: str) -> None:
        index = self.engine.findData(key)
        if index >= 0:
            self.engine.setCurrentIndex(index)

    def set_reference(self, image: Image.Image | None) -> None:
        self._reference = image
        if image is None:
            self.preview.setPixmap(QtGui.QPixmap())
            self.preview.setText("No picture yet")
        else:
            self.preview.setPixmap(pil_to_pixmap(image, 140))
        self._refresh_enabled()

    def set_candidates(self, images: list[Image.Image]) -> None:
        self._candidates = images
        self.gallery.blockSignals(True)
        self.gallery.clear()
        for index, image in enumerate(images):
            item = QtWidgets.QListWidgetItem(QtGui.QIcon(pil_to_pixmap(image, 92)), "")
            item.setToolTip(f"Picture {index + 1}")
            self.gallery.addItem(item)
        self.gallery.blockSignals(False)
        self.gallery.setVisible(bool(images))
        if images:
            self.gallery.setCurrentRow(0)  # emits -> sets the reference

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        for widget in (self.draw_button, self.open_button, self.engine, self.quality, self.seed, self.dice, self.remove_bg):
            widget.setEnabled(not busy)
        self.cancel_button.setVisible(busy)
        self.progress.setVisible(busy)
        self.status.setVisible(busy or bool(self.status.text()))
        self._refresh_enabled()

    def set_progress(self, fraction: float, message: str) -> None:
        if fraction < 0:
            self.progress.setRange(0, 0)  # busy indicator
        else:
            self.progress.setRange(0, 1000)
            self.progress.setValue(int(fraction * 1000))
        self.status.setText(message)
        self.status.show()

    def set_message(self, message: str) -> None:
        self.status.setText(message)
        self.status.setVisible(bool(message))

    # ------------------------------------------------------------------ internals

    def _refresh_enabled(self) -> None:
        self.make_button.setEnabled(self._reference is not None and not self._busy)

    def _candidate_chosen(self, row: int) -> None:
        if 0 <= row < len(self._candidates):
            self.set_reference(self._candidates[row])

    def _engine_changed(self) -> None:
        info = self._engines.get(self.selected_engine())
        if info is not None:
            licence = "personal use only" if not info.commercial_use else "free for any use"
            vram = f"about {info.approx_vram_gb:.0f} GB of GPU memory" if info.approx_vram_gb else "no GPU needed"
            download = f", {info.download_gb:.1f} GB one-time download" if info.download_gb else ""
            self.engine_note.setText(f"{info.description}\nUses {vram}{download}. License: {licence}.")
            self.engine_note.setToolTip(info.license)
            self.engine_changed.emit(info.key)

    def _emit_draw(self) -> None:
        self.draw_requested.emit(self.prompt.toPlainText(), self.style.currentData(), self.seed.value())

    def _emit_make(self) -> None:
        self.make3d_requested.emit(
            self.selected_engine(), self.quality.currentData(), self.seed.value(), self.remove_bg.isChecked()
        )
