"""Run long operations off the UI thread, with progress and Cancel."""

from __future__ import annotations

import logging
import traceback
from typing import Any, Callable

from PySide6 import QtCore

from ..config import data_dir
from ..engines import Cancelled, CancelToken, EngineUnavailable, OutOfVram

log = logging.getLogger("forge3d")


def configure_logging() -> None:
    if log.handlers:
        return
    log.setLevel(logging.INFO)
    try:
        handler = logging.FileHandler(data_dir() / "forge3d.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(handler)
    except OSError:
        log.addHandler(logging.NullHandler())


class JobThread(QtCore.QThread):
    """Runs ``fn(progress, cancel_token)``. Exactly one of succeeded / failed / cancelled is emitted."""

    progress = QtCore.Signal(float, str)
    succeeded = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    cancelled = QtCore.Signal()

    def __init__(self, fn: Callable[[Callable[[float, str], None], CancelToken], Any], parent=None):
        super().__init__(parent)
        self._fn = fn
        self.token = CancelToken()

    def cancel(self) -> None:
        self.token.cancel()

    def run(self) -> None:
        try:
            result = self._fn(lambda fraction, message: self.progress.emit(fraction, message), self.token)
        except Cancelled:
            self.cancelled.emit()
        except (EngineUnavailable, OutOfVram, ValueError, FileNotFoundError) as error:
            self.failed.emit(str(error))
        except MemoryError:
            self.failed.emit("The computer ran out of memory. Try a lower quality setting or close other programs.")
        except Exception as error:  # last resort: never let a worker crash the app silently
            log.error("Unhandled error in job:\n%s", traceback.format_exc())
            self.failed.emit(f"Something unexpected went wrong: {error}\n(Details were written to forge3d.log)")
        else:
            self.succeeded.emit(result)
