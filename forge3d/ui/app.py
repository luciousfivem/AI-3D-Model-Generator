"""Application entry point (``forge3d-gui`` / ``python -m forge3d``)."""

from __future__ import annotations

import sys

from PySide6 import QtCore, QtWidgets

from .. import APP_NAME
from ..config import Settings
from . import theme


def main(argv: list[str] | None = None) -> int:
    QtCore.QCoreApplication.setApplicationName(APP_NAME)
    app = QtWidgets.QApplication(argv if argv is not None else sys.argv)
    theme.apply_dark_theme(app)

    from .main_window import MainWindow  # after QApplication exists

    window = MainWindow(Settings.load())
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
