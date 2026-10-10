import time

import numpy as np
import pytest
import trimesh
from PIL import Image, ImageDraw

from forge3d import imaging, meshops
from forge3d.config import Settings
from forge3d.engines import EngineUnavailable

pytestmark = pytest.mark.gui


@pytest.fixture()
def window(qapp, tmp_path, monkeypatch):
    from PySide6 import QtWidgets

    from forge3d.ui.main_window import MainWindow

    monkeypatch.setattr(imaging, "remove_background", imaging._fallback_cutout)  # no model download in tests
    # any modal dialog would hang the suite: record it instead
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda parent, title, text, *a, **k: shown.append(text))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda parent, title, text, *a, **k: shown.append(text))

    settings = Settings(models_dir=str(tmp_path / "models"), output_dir=str(tmp_path / "out"), engine="inflate")
    win = MainWindow(settings)
    win.messages = shown
    win.resize(1360, 860)
    win.show()
    yield win
    win.close()
    qapp.processEvents()


def save_picture(path):
    image = Image.new("RGB", (300, 300), "white")
    ImageDraw.Draw(image).ellipse([60, 50, 240, 250], fill=(30, 120, 200))
    image.save(path)
    return path


def make_model(window, wait, tmp_path):
    window.load_picture(save_picture(tmp_path / "ball.png"))
    window.create.select_engine("inflate")
    window.create.make_button.click()
    assert wait(lambda: window.final is not None and window._job is None and window._prep_job is None)


# --------------------------------------------------------------------------- tests


def test_starts_empty_with_actions_disabled(window):
    assert window.final is None
    assert not window.create.make_button.isEnabled()
    assert not window.print.export_button.isEnabled()
    assert not window.export_action.isEnabled()


def test_picture_to_printable_model_to_export(window, wait, tmp_path):
    make_model(window, wait, tmp_path)
    assert window.report.printable and window.report.fits_bed
    assert window.final.extents[2] == pytest.approx(window.print.height.value(), rel=1e-3)
    assert window.final.bounds[0][2] == pytest.approx(0.0, abs=1e-6)  # resting on the bed
    assert window.print.export_button.isEnabled()
    assert "triangles" in window.print.summary.text()

    # export through the real code path (file dialog replaced by a fixed answer)
    from PySide6 import QtWidgets

    for extension in ("stl", "3mf"):
        target = tmp_path / f"out.{extension}"
        QtWidgets.QFileDialog.getSaveFileName = staticmethod(lambda *a, t=target, **k: (str(t), ""))
        window.export(extension)
        reloaded = meshops.load_mesh(target)
        assert reloaded.is_watertight
        assert reloaded.extents[2] == pytest.approx(window.final.extents[2], rel=1e-3)
    assert "Saved" in window.statusBar().currentMessage()


def test_viewport_actually_draws_the_model(window, wait, tmp_path):
    make_model(window, wait, tmp_path)
    wait(lambda: False, 0.5)
    image = window.viewport.snapshot()
    pixels = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.width(), -1)[..., :3]
    background = pixels[2, 2]
    differing = (np.abs(pixels.astype(int) - background.astype(int)).sum(axis=2) > 30).mean()
    assert differing > 0.05, "the viewport looks empty"


def test_overhang_overlay_paints_red(window, wait, tmp_path):
    from PySide6 import QtGui

    # a wide cone resting on its tip needs supports everywhere
    cone = trimesh.creation.cone(radius=30, height=10)
    cone.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    window._mesh_ready(cone, "z")
    window.print.keep_scale.setChecked(True)
    assert wait(lambda: window.final is not None and window._prep_job is None)

    def red_share():
        wait(lambda: False, 0.4)
        image = window.viewport.snapshot().convertToFormat(QtGui.QImage.Format_RGBA8888)  # Qt hands back BGRA
        px = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.width(), -1)[..., :3].astype(int)
        r, g, b = px[..., 0], px[..., 1], px[..., 2]
        return int(((r > 70) & (r > 2 * g) & (r > 2 * b)).sum())  # "reddish", whatever the shading does to it

    window.viewport.opts["elevation"] = -35  # overhangs face downward: look from underneath, as a user would
    before = red_share()
    window.print.show_overhang.setChecked(True)
    after = red_share()
    assert after > before + 150, f"expected a red overlay, saw {before} -> {after} reddish pixels"


def test_rotate_buttons_change_the_model_and_stay_on_the_bed(window, wait, tmp_path):
    box = trimesh.creation.box([20, 20, 60])
    window._mesh_ready(box, "z")
    window.print.keep_scale.setChecked(True)
    assert wait(lambda: window.final is not None and window._prep_job is None)
    assert window.final.extents[2] == pytest.approx(60, rel=1e-3)

    window._on_rotate("x", 90)
    assert wait(lambda: window._prep_job is None and window.final.extents[2] < 30)
    assert window.final.extents[2] == pytest.approx(20, rel=1e-3)
    assert window.final.bounds[0][2] == pytest.approx(0, abs=1e-6)


def test_auto_orient_lays_a_wide_overhanging_part_flat(window, wait):
    cone = trimesh.creation.cone(radius=30, height=10)  # apex up; flip it so supports would be needed
    cone.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    window._mesh_ready(cone, "z")
    window.print.keep_scale.setChecked(True)
    assert wait(lambda: window.final is not None and window._prep_job is None)
    assert window.report.overhang_pct > 50

    window.print.auto_orient_requested.emit()
    assert wait(lambda: window.report.overhang_pct < 5 and window._prep_job is None, timeout=30)


def test_opening_a_broken_mesh_repairs_it(window, wait, tmp_path):
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=25)
    sphere.update_faces(np.arange(len(sphere.faces)) % 25 != 0)  # punch holes
    path = tmp_path / "broken.stl"
    sphere.export(path)

    window.load_mesh(path)
    assert wait(lambda: window.final is not None and window._prep_job is None)
    assert window.report.watertight and window.report.printable

    window.print.watertight.setChecked(False)  # turning repair off must be honoured and reported
    assert wait(lambda: window._prep_job is None and not window.report.watertight, timeout=30)
    assert any(i.code == "not_watertight" for i in window.report.issues)


def test_slider_changes_are_coalesced(window, wait, tmp_path):
    make_model(window, wait, tmp_path)
    runs = []
    original = window._run_prep
    window._run_prep = lambda request: (runs.append(1), original(request))
    for height in (30, 40, 50, 70):
        window.print.height.setValue(height)
        window._schedule_recompute()
    assert wait(lambda: window.final.extents[2] == pytest.approx(70, rel=1e-3) and window._prep_job is None, 20)
    assert len(runs) == 1


def test_text_to_picture_success_and_failure(window, wait, monkeypatch):
    images = [Image.new("RGB", (64, 64), c) for c in ("red", "green", "blue", "gray")]
    monkeypatch.setattr(window.forge, "text_to_images", lambda *a, **k: images)
    window.create.prompt.setPlainText("a small dragon")
    window.create.draw_button.click()
    assert wait(lambda: window.create.gallery.count() == 4 and window._job is None)
    assert window.create.reference_image() is images[0]  # the first picture is pre-selected
    window.create.gallery.setCurrentRow(2)
    assert window.create.reference_image() is images[2]
    assert window.create.make_button.isEnabled()

    def boom(*a, **k):
        raise EngineUnavailable("The AI packages are not installed (torch).")

    monkeypatch.setattr(window.forge, "text_to_images", boom)
    window.create.draw_button.click()
    assert wait(lambda: window._job is None and window.messages)
    assert "not installed" in window.messages[-1]
    assert window.create.draw_button.isEnabled()  # UI recovered


def test_empty_prompt_is_refused_politely(window):
    window.create.draw_button.click()
    assert window.messages and "description" in window.messages[-1].lower()
    assert window._job is None


def test_cancel_stops_a_running_generation(window, wait, monkeypatch, tmp_path):
    started = []

    def slow(image, *, progress, cancel, **kwargs):
        started.append(1)
        for _ in range(500):
            cancel.check()
            progress(0.2, "working…")
            time.sleep(0.01)
        raise AssertionError("cancel was ignored")

    monkeypatch.setattr(window.forge, "image_to_mesh", slow)
    window.load_picture(save_picture(tmp_path / "x.png"))
    window.create.make_button.click()
    assert wait(lambda: started)
    assert window.create.cancel_button.isVisible()
    window.create.cancel_button.click()
    assert wait(lambda: window._job is None, 10)
    assert window.create.status.text() == "Cancelled."
    assert window.final is None
    assert window.create.draw_button.isEnabled() and not window.create.cancel_button.isVisible()


def test_unexpected_errors_are_reported_not_fatal(window, wait, monkeypatch, tmp_path):
    def crash(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(window.forge, "image_to_mesh", crash)
    window.load_picture(save_picture(tmp_path / "x.png"))
    window.create.make_button.click()
    assert wait(lambda: window.messages and window._job is None)
    assert "kaboom" in window.messages[-1] and "forge3d.log" in window.messages[-1]


def test_settings_dialog_updates_bed_and_report(window, wait, tmp_path, monkeypatch):
    from forge3d.ui.settings_dialog import SettingsDialog

    make_model(window, wait, tmp_path)
    dialog = SettingsDialog(window.settings, window)
    for spin, value in zip(dialog.bed, (50, 50, 50)):  # a tiny printer: the 60 mm model cannot fit
        spin.setValue(value)
    dialog.apply()
    window.viewport.set_bed(window.settings.bed_x, window.settings.bed_y)
    window._recompute()
    assert wait(lambda: window._prep_job is None and not window.report.fits_bed, 20)
    assert any(i.code == "too_big" for i in window.report.issues)
    assert Settings.load().bed_x == 50  # persisted


def test_dropping_files_loads_pictures_and_models(window, wait, tmp_path):
    from PySide6 import QtCore, QtGui

    def drop(path):
        data = QtCore.QMimeData()
        data.setUrls([QtCore.QUrl.fromLocalFile(str(path))])
        event = QtGui.QDropEvent(
            QtCore.QPointF(10, 10), QtCore.Qt.CopyAction, data, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier
        )
        window.dropEvent(event)

    drop(save_picture(tmp_path / "dropped.png"))
    assert window.create.reference_image() is not None

    stl = tmp_path / "cube.stl"
    trimesh.creation.box([10, 10, 10]).export(stl)
    drop(stl)
    assert wait(lambda: window.final is not None and window._prep_job is None)
    assert window.report.printable
