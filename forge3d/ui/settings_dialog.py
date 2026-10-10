"""Printer, material and performance settings."""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ..config import Settings

# name -> density in g/cm^3
MATERIALS = {
    "PLA": 1.24,
    "PETG": 1.27,
    "ABS": 1.04,
    "ASA": 1.07,
    "TPU": 1.21,
    "Resin": 1.10,
}


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.settings = settings
        root = QtWidgets.QVBoxLayout(self)

        printer = QtWidgets.QGroupBox("Your printer")
        form = QtWidgets.QFormLayout(printer)
        self.bed = []
        for label, value in (("Bed width (X)", settings.bed_x), ("Bed depth (Y)", settings.bed_y), ("Max height (Z)", settings.bed_z)):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(20, 2000)
            spin.setDecimals(0)
            spin.setSuffix(" mm")
            spin.setValue(value)
            form.addRow(label, spin)
            self.bed.append(spin)
        self.material = QtWidgets.QComboBox()
        for name, density in MATERIALS.items():
            self.material.addItem(f"{name}  ({density:.2f} g/cm³)", density)
        closest = min(range(self.material.count()), key=lambda i: abs(self.material.itemData(i) - settings.density_g_cm3))
        self.material.setCurrentIndex(closest)
        form.addRow("Material", self.material)
        root.addWidget(printer)

        checks = QtWidgets.QGroupBox("Print checks")
        form2 = QtWidgets.QFormLayout(checks)
        self.min_wall = QtWidgets.QDoubleSpinBox()
        self.min_wall.setRange(0.2, 5.0)
        self.min_wall.setSingleStep(0.1)
        self.min_wall.setSuffix(" mm")
        self.min_wall.setValue(settings.min_wall_mm)
        self.min_wall.setToolTip("Thinnest wall you trust your printer with (about 2 × the nozzle width).")
        form2.addRow("Thinnest printable wall", self.min_wall)
        self.overhang = QtWidgets.QSpinBox()
        self.overhang.setRange(20, 80)
        self.overhang.setSuffix("°")
        self.overhang.setValue(int(settings.max_overhang_deg))
        self.overhang.setToolTip("Steepest overhang (from vertical) that prints cleanly without supports. 45° is typical.")
        form2.addRow("Overhang limit", self.overhang)
        root.addWidget(checks)

        perf = QtWidgets.QGroupBox("Performance")
        form3 = QtWidgets.QFormLayout(perf)
        self.low_vram = QtWidgets.QComboBox()
        self.low_vram.addItem("Automatic", None)
        self.low_vram.addItem("Always on (slower, uses less GPU memory)", True)
        self.low_vram.addItem("Always off", False)
        self.low_vram.setCurrentIndex({None: 0, True: 1, False: 2}[settings.low_vram])
        form3.addRow("Low-VRAM mode", self.low_vram)
        root.addWidget(perf)

        folders = QtWidgets.QGroupBox("Folders")
        grid = QtWidgets.QGridLayout(folders)
        for row, (label, path) in enumerate(
            (("AI models", settings.resolved_models_dir()), ("Exports", settings.resolved_output_dir()))
        ):
            grid.addWidget(QtWidgets.QLabel(label), row, 0)
            where = QtWidgets.QLabel(str(path))
            where.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            where.setWordWrap(True)
            grid.addWidget(where, row, 1)
            button = QtWidgets.QPushButton("Open")
            button.clicked.connect(lambda _=False, p=path: QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(p))))
            grid.addWidget(button, row, 2)
        grid.setColumnStretch(1, 1)
        root.addWidget(folders)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def apply(self) -> None:
        s = self.settings
        s.bed_x, s.bed_y, s.bed_z = (spin.value() for spin in self.bed)
        s.density_g_cm3 = float(self.material.currentData())
        s.min_wall_mm = self.min_wall.value()
        s.max_overhang_deg = float(self.overhang.value())
        s.low_vram = self.low_vram.currentData()
        s.save()
