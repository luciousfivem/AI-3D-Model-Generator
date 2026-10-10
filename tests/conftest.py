import os
import time

import pytest


@pytest.fixture(autouse=True)
def isolated_profile(tmp_path, monkeypatch):
    """Keep tests away from the real settings/model folders and from leaking environment changes."""
    monkeypatch.setattr("forge3d.config.user_config_dir", lambda *a, **k: str(tmp_path / "config"))
    monkeypatch.setattr("forge3d.config.user_data_dir", lambda *a, **k: str(tmp_path / "data"))
    for key in ("HY3DGEN_MODELS", "HF_HOME", "U2NET_HOME"):
        monkeypatch.delenv(key, raising=False)  # restored on teardown even if the code under test sets them


@pytest.fixture(scope="session")
def qapp():
    if not (os.environ.get("DISPLAY") or os.environ.get("QT_QPA_PLATFORM")):
        pytest.skip("GUI tests need a display (on headless Linux run them under xvfb-run)")
    from PySide6 import QtWidgets

    from forge3d.ui import theme

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["forge3d-tests"])
    theme.apply_dark_theme(app)
    return app


@pytest.fixture()
def wait(qapp):
    def _wait(predicate, timeout=60.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            qapp.processEvents()
            if predicate():
                return True
            time.sleep(0.01)
        qapp.processEvents()
        return predicate()

    return _wait
