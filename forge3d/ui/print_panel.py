"""Right panel: size, orientation, repair, the printability report, and export."""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from .. import meshops
from ..meshops import FORMATS, PrepOptions, PrintReport
from . import theme

_UP_LABELS = {
    "y": "Y is up (AI models)",
    "z": "Z is up (slicer files)",
    "-y": "-Y is up",
    "-z": "-Z is up (upside down)",
    "x": "X is up",
    "-x": "-X is up",
}


class PrintPanel(QtWidgets.QWidget):
    changed = QtCore.Signal()  # any repair/size option changed (re-run the pipeline)
    rotate_requested = QtCore.Signal(str, float)
    auto_orient_requested = QtCore.Signal()
    thickness_requested = QtCore.Signal()
    export_requested = QtCore.Signal(str)
    view_changed = QtCore.Signal()

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(6)

        # ---- size
        size_box = QtWidgets.QGroupBox("Size")
        size = QtWidgets.QGridLayout(size_box)
        self.keep_scale = QtWidgets.QCheckBox("Keep the model's own size")
        size.addWidget(self.keep_scale, 0, 0, 1, 2)
        size.addWidget(QtWidgets.QLabel("Height"), 1, 0)
        self.height = QtWidgets.QDoubleSpinBox()
        self.height.setRange(1.0, 1000.0)
        self.height.setDecimals(1)
        self.height.setSuffix(" mm")
        self.height.setValue(settings.target_height_mm)
        size.addWidget(self.height, 1, 1)
        self.dimensions = QtWidgets.QLabel("-")
        self.dimensions.setObjectName("hint")
        size.addWidget(self.dimensions, 2, 0, 1, 2)
        layout.addWidget(size_box)

        # ---- orientation
        orient_box = QtWidgets.QGroupBox("Orientation")
        orient = QtWidgets.QVBoxLayout(orient_box)
        self.up_axis = QtWidgets.QComboBox()
        for key in meshops.UP_AXES:
            self.up_axis.addItem(_UP_LABELS[key], key)
        self.up_axis.setToolTip("If the model lies on its side or upside down, choose which axis is really 'up'.")
        orient.addWidget(self.up_axis)
        rotate_row = QtWidgets.QHBoxLayout()
        for axis in "xyz":
            for sign, arrow in ((-90, "−"), (90, "+")):
                button = QtWidgets.QToolButton()
                button.setText(f"{axis.upper()}{arrow}90°")
                button.setToolTip(f"Rotate {sign:+d}° around the {axis.upper()} axis")
                button.clicked.connect(lambda _=False, a=axis, s=sign: self.rotate_requested.emit(a, float(s)))
                rotate_row.addWidget(button)
        orient.addLayout(rotate_row)
        self.auto_button = QtWidgets.QPushButton("Auto-orient for fewest supports")
        self.auto_button.clicked.connect(self.auto_orient_requested)
        orient.addWidget(self.auto_button)
        layout.addWidget(orient_box)

        # ---- repair
        repair_box = QtWidgets.QGroupBox("Clean-up")
        repair = QtWidgets.QGridLayout(repair_box)
        self.floaters = QtWidgets.QCheckBox("Remove loose fragments")
        self.floaters.setChecked(True)
        self.watertight = QtWidgets.QCheckBox("Fix holes (make watertight)")
        self.watertight.setChecked(True)
        repair.addWidget(self.floaters, 0, 0, 1, 2)
        repair.addWidget(self.watertight, 1, 0, 1, 2)
        repair.addWidget(QtWidgets.QLabel("Smooth"), 2, 0)
        self.smooth = QtWidgets.QSpinBox()
        self.smooth.setRange(0, 20)
        self.smooth.setSuffix(" passes")
        self.smooth.setToolTip("Removes the faint ridges AI meshes have, without shrinking the model.")
        repair.addWidget(self.smooth, 2, 1)
        self.decimate_on = QtWidgets.QCheckBox("Fewer triangles")
        self.decimate = QtWidgets.QSpinBox()
        self.decimate.setRange(2_000, 2_000_000)
        self.decimate.setSingleStep(10_000)
        self.decimate.setValue(150_000)
        self.decimate.setEnabled(False)
        self.decimate.setToolTip("Smaller files, faster slicing. 100,000+ is indistinguishable when printed.")
        repair.addWidget(self.decimate_on, 3, 0)
        repair.addWidget(self.decimate, 3, 1)
        repair.addWidget(QtWidgets.QLabel("Flat base"), 4, 0)
        self.flat_base = QtWidgets.QDoubleSpinBox()
        self.flat_base.setRange(0.0, 20.0)
        self.flat_base.setSingleStep(0.5)
        self.flat_base.setSuffix(" mm")
        self.flat_base.setToolTip("Slices this much off the bottom so the model sits flat and sticks to the bed.")
        repair.addWidget(self.flat_base, 4, 1)
        layout.addWidget(repair_box)

        # ---- report
        report_box = QtWidgets.QGroupBox("Print check")
        report = QtWidgets.QVBoxLayout(report_box)
        self.summary = QtWidgets.QLabel("Make or open a model to check it.")
        self.summary.setWordWrap(True)
        report.addWidget(self.summary)
        self.issues = QtWidgets.QListWidget()
        self.issues.setWordWrap(True)
        self.issues.setFixedHeight(84)
        self.issues.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        report.addWidget(self.issues)
        self.thickness_button = QtWidgets.QPushButton("Check wall thickness")
        self.thickness_button.clicked.connect(self.thickness_requested)
        report.addWidget(self.thickness_button)
        self.show_overhang = QtWidgets.QCheckBox("Show where supports are needed (red)")
        self.wireframe = QtWidgets.QCheckBox("Show triangle edges")
        report.addWidget(self.show_overhang)
        report.addWidget(self.wireframe)
        layout.addWidget(report_box)

        # ---- export
        export_box = QtWidgets.QGroupBox("Export")
        export = QtWidgets.QHBoxLayout(export_box)
        self.format = QtWidgets.QComboBox()
        for extension, (label, note) in FORMATS.items():
            self.format.addItem(label, extension)
            self.format.setItemData(self.format.count() - 1, note, QtCore.Qt.ToolTipRole)
        self.format.setCurrentIndex(self.format.findData("3mf"))
        export.addWidget(self.format, 1)
        self.export_button = QtWidgets.QPushButton("Export…")
        self.export_button.setObjectName("primary")
        self.export_button.clicked.connect(lambda: self.export_requested.emit(self.format.currentData()))
        export.addWidget(self.export_button)
        layout.addWidget(export_box)
        layout.addStretch(1)

        self.set_enabled(False)

        for signal in (
            self.keep_scale.toggled,
            self.height.editingFinished,
            self.up_axis.currentIndexChanged,
            self.floaters.toggled,
            self.watertight.toggled,
            self.smooth.editingFinished,
            self.decimate_on.toggled,
            self.decimate.editingFinished,
            self.flat_base.editingFinished,
        ):
            signal.connect(self.changed)
        self.keep_scale.toggled.connect(lambda on: self.height.setEnabled(not on))
        self.decimate_on.toggled.connect(self.decimate.setEnabled)
        self.show_overhang.toggled.connect(self.view_changed)
        self.wireframe.toggled.connect(self.view_changed)

    # ------------------------------------------------------------------ options

    def read_into(self, options: PrepOptions) -> PrepOptions:
        options.source_up = self.up_axis.currentData()
        options.height_mm = None if self.keep_scale.isChecked() else self.height.value()
        options.remove_floaters = self.floaters.isChecked()
        options.make_watertight = self.watertight.isChecked()
        options.smooth_iterations = self.smooth.value()
        options.decimate_faces = self.decimate.value() if self.decimate_on.isChecked() else None
        options.flat_base_mm = self.flat_base.value()
        return options

    def set_source_up(self, up: str) -> None:
        blocked = self.up_axis.blockSignals(True)
        self.up_axis.setCurrentIndex(max(self.up_axis.findData(up), 0))
        self.up_axis.blockSignals(blocked)

    def set_enabled(self, enabled: bool) -> None:
        for widget in (
            self.keep_scale, self.height, self.up_axis, self.auto_button, self.floaters, self.watertight,
            self.smooth, self.decimate_on, self.flat_base, self.thickness_button, self.show_overhang,
            self.wireframe, self.format, self.export_button,
        ):
            widget.setEnabled(enabled)
        self.decimate.setEnabled(enabled and self.decimate_on.isChecked())
        self.height.setEnabled(enabled and not self.keep_scale.isChecked())

    # ------------------------------------------------------------------ report

    def show_report(self, report: PrintReport, notes: list[str] | None = None) -> None:
        sx, sy, sz = report.size_mm
        self.dimensions.setText(f"{sx:.1f} × {sy:.1f} × {sz:.1f} mm   (bed {self.settings.bed_x:.0f} × {self.settings.bed_y:.0f})")
        material = f"{report.volume_mm3 / 1000:.1f} cm³ ≈ {report.weight_g:.0f} g solid" if report.volume_mm3 else "volume unknown (not watertight)"
        self.summary.setText(
            f"<b>{report.n_faces:,}</b> triangles · {material}<br>"
            f"{report.overhang_pct:.0f}% of the surface overhangs"
            + (f" · median wall {report.median_wall_mm:.1f} mm" if report.median_wall_mm else "")
        )

        self.issues.clear()
        icons = {"error": ("✖", theme.ERROR), "warn": ("▲", theme.WARN), "info": ("✔", theme.OK)}
        for issue in report.issues:
            symbol, colour = icons[issue.severity]
            item = QtWidgets.QListWidgetItem(f"{symbol}  {issue.message}")
            item.setForeground(QtGui.QBrush(QtGui.QColor(colour)))
            self.issues.addItem(item)
        for note in notes or []:
            item = QtWidgets.QListWidgetItem(f"•  {note}")
            item.setForeground(QtGui.QBrush(QtGui.QColor(theme.MUTED)))
            self.issues.addItem(item)

    def clear_report(self) -> None:
        self.summary.setText("Make or open a model to check it.")
        self.dimensions.setText("-")
        self.issues.clear()
