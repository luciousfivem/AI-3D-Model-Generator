"""A dark Fusion theme (consistent across Windows versions, and easy on the eyes next to a 3D view)."""

from __future__ import annotations

from PySide6 import QtGui, QtWidgets

ACCENT = "#4c9be8"
WARN = "#e8a33d"
ERROR = "#e5534b"
OK = "#57ab5a"
MUTED = "#8b949e"


def apply_dark_theme(app: QtWidgets.QApplication) -> None:
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    colors = {
        QtGui.QPalette.Window: "#1f2329",
        QtGui.QPalette.WindowText: "#d7dce2",
        QtGui.QPalette.Base: "#161a1f",
        QtGui.QPalette.AlternateBase: "#1f2329",
        QtGui.QPalette.ToolTipBase: "#2b3138",
        QtGui.QPalette.ToolTipText: "#d7dce2",
        QtGui.QPalette.Text: "#d7dce2",
        QtGui.QPalette.Button: "#2b3138",
        QtGui.QPalette.ButtonText: "#d7dce2",
        QtGui.QPalette.BrightText: "#ffffff",
        QtGui.QPalette.Link: ACCENT,
        QtGui.QPalette.Highlight: ACCENT,
        QtGui.QPalette.HighlightedText: "#ffffff",
        QtGui.QPalette.PlaceholderText: "#6b7480",
    }
    for role, value in colors.items():
        palette.setColor(role, QtGui.QColor(value))
    for role in (QtGui.QPalette.WindowText, QtGui.QPalette.Text, QtGui.QPalette.ButtonText):
        palette.setColor(QtGui.QPalette.Disabled, role, QtGui.QColor("#6b7480"))
    app.setPalette(palette)
    app.setStyleSheet(
        f"""
        QToolTip {{ border: 1px solid #3a424b; padding: 4px; }}
        QGroupBox {{ border: 1px solid #343b44; border-radius: 6px; margin-top: 14px; padding: 10px 8px 8px 8px; }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {MUTED}; }}
        QPushButton {{ padding: 6px 12px; border-radius: 5px; border: 1px solid #3a424b; }}
        QPushButton:hover {{ border-color: {ACCENT}; }}
        QPushButton:disabled {{ color: #5b636d; }}
        QPushButton#primary {{ background: {ACCENT}; color: white; border: none; font-weight: 600; padding: 9px 14px; }}
        QPushButton#primary:hover {{ background: #5aa8f2; }}
        QPushButton#primary:disabled {{ background: #2d3a4a; color: #6b7480; }}
        QProgressBar {{ border: 1px solid #343b44; border-radius: 4px; text-align: center; height: 16px; }}
        QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
        QLabel#hint {{ color: {MUTED}; }}
        QLabel#banner {{ background: #3a3020; color: {WARN}; border: 1px solid #5a4a2a; border-radius: 5px; padding: 6px 8px; }}
        QListWidget {{ border: 1px solid #343b44; border-radius: 4px; }}
        QListWidget::item:selected {{ border: 2px solid {ACCENT}; }}
        QDockWidget::title {{ padding: 6px; background: #1b1f24; }}
        """
    )
