"""The Forge3D main window."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import trimesh
from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets

from .. import __version__, hardware, meshops
from ..config import Settings
from ..engines import list_image_to_3d
from ..pipeline import Forge
from .create_panel import CreatePanel
from .jobs import JobThread, configure_logging
from .print_panel import PrintPanel
from .settings_dialog import SettingsDialog
from .viewport import Viewport

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MESH_EXTENSIONS = {".stl", ".obj", ".glb", ".gltf", ".ply", ".3mf", ".off"}
_MESH_FILTER = "3D models (*.stl *.obj *.glb *.gltf *.ply *.3mf *.off);;All files (*)"
_IMAGE_FILTER = "Pictures (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*)"


@dataclass
class _PrepResult:
    repaired: trimesh.Trimesh
    key: tuple
    mesh: trimesh.Trimesh
    report: meshops.PrintReport
    notes: list[str]
    frame: bool


def _slug(text: str, fallback: str = "model") -> str:
    words = re.findall(r"[A-Za-z0-9]+", text.lower())[:5]
    return "_".join(words) or fallback


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, settings: Settings | None = None, forge: Forge | None = None):
        super().__init__()
        configure_logging()
        self.settings = settings or Settings.load()
        self.forge = forge or Forge(self.settings)

        self.raw: trimesh.Trimesh | None = None  # straight from the model / file
        self.options = meshops.PrepOptions()
        self._cache: tuple[tuple, trimesh.Trimesh] | None = None  # (repair key, stage-A mesh)
        self.final: trimesh.Trimesh | None = None
        self.report: meshops.PrintReport | None = None
        self._prompt_for_name = ""
        self._last_notes: list[str] = []

        self._job: JobThread | None = None  # generation (exclusive)
        self._prep_job: JobThread | None = None  # print prep (cheap, coalesced)
        self._prep_pending: dict | None = None
        self._generation = 0

        self.setWindowTitle("Forge3D — AI models for 3D printing")
        self.resize(1360, 860)
        self.setAcceptDrops(True)
        self._build_ui()
        self._build_menu()

        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(350)
        self._debounce.timeout.connect(lambda: self._recompute())

        QtCore.QTimer.singleShot(0, self._probe_hardware)

    # ================================================================== construction

    def _build_ui(self) -> None:
        self.viewport = Viewport()
        self.viewport.set_bed(self.settings.bed_x, self.settings.bed_y)

        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setContentsMargins(8, 6, 8, 6)
        reset = QtWidgets.QPushButton("Reset view")
        reset.clicked.connect(self.viewport.reset_camera)
        toolbar.addWidget(reset)
        hint = QtWidgets.QLabel("Left-drag: rotate · Right-drag: move · Wheel: zoom")
        hint.setObjectName("hint")
        toolbar.addWidget(hint)
        toolbar.addStretch(1)
        center = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(center)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addLayout(toolbar)
        column.addWidget(self.viewport, 1)
        self.setCentralWidget(center)

        self.create = CreatePanel(list_image_to_3d(), self.settings)
        self.create.draw_requested.connect(self._on_draw)
        self.create.open_picture_requested.connect(self.open_picture_dialog)
        self.create.make3d_requested.connect(self._on_make3d)
        self.create.cancel_requested.connect(self._on_cancel)
        left = QtWidgets.QDockWidget("Create", self)
        left.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setWidget(self.create)
        scroll.setMinimumWidth(360)
        left.setWidget(scroll)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, left)

        self.print = PrintPanel(self.settings)
        self.print.changed.connect(self._schedule_recompute)
        self.print.rotate_requested.connect(self._on_rotate)
        self.print.auto_orient_requested.connect(self._on_auto_orient)
        self.print.thickness_requested.connect(lambda: self._recompute(thickness=True))
        self.print.export_requested.connect(self.export)
        self.print.view_changed.connect(self._refresh_view)
        right = QtWidgets.QDockWidget("Prepare for printing", self)
        right.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        scroll2 = QtWidgets.QScrollArea()
        scroll2.setWidgetResizable(True)
        scroll2.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll2.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll2.setWidget(self.print)
        scroll2.setMinimumWidth(372)
        right.setWidget(scroll2)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, right)

        self.gpu_label = QtWidgets.QLabel("Checking hardware…")
        self.statusBar().addPermanentWidget(self.gpu_label)
        self.statusBar().showMessage("Ready")

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction("Open picture…", self.open_picture_dialog, QtGui.QKeySequence("Ctrl+I"))
        file_menu.addAction("Open 3D model…", self.open_mesh_dialog, QtGui.QKeySequence.Open)
        file_menu.addSeparator()
        self.export_action = file_menu.addAction("Export…", lambda: self.export(self.print.format.currentData()), QtGui.QKeySequence("Ctrl+E"))
        self.export_action.setEnabled(False)
        file_menu.addSeparator()
        file_menu.addAction("Quit", self.close, QtGui.QKeySequence.Quit)

        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.addAction("Settings…", self.open_settings, QtGui.QKeySequence("Ctrl+,"))

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction("Hardware check", self.show_hardware_report)
        help_menu.addAction("Open exports folder", lambda: self._open_folder(self.settings.resolved_output_dir()))
        help_menu.addAction("Open models folder", lambda: self._open_folder(self.settings.resolved_models_dir()))
        help_menu.addSeparator()
        help_menu.addAction("About", self.show_about)

    # ================================================================== hardware banner

    def _probe_hardware(self) -> None:
        def probe(progress, cancel):
            return hardware.missing_ai_packages(), hardware.gpu_info()

        job = JobThread(probe, self)
        job.succeeded.connect(self._hardware_ready)
        job.finished.connect(job.deleteLater)
        self._probe_job = job
        job.start()

    def _hardware_ready(self, result) -> None:
        missing, gpu = result
        self.gpu_label.setText(f"GPU: {gpu.summary()}")
        if missing:
            self.create.set_banner(
                "The AI packages are not installed yet, so only the instant “Inflate silhouette” model and the "
                "print tools work. Run <b>scripts\\install.ps1</b> to enable the AI models."
            )
            if self.create.selected_engine() != "inflate":
                self.create.select_engine("inflate")
        elif not gpu.available:
            self.create.set_banner(
                f"{gpu.summary()}. AI models will run on the CPU, which can take 10+ minutes per model. "
                "An NVIDIA GPU is strongly recommended."
            )
        else:
            self.create.set_banner("")

    # ================================================================== jobs

    def _start_job(self, fn, on_success, *, busy_text: str) -> None:
        if self._job is not None:
            return
        job = JobThread(fn, self)
        self._job = job
        self.create.set_busy(True)
        self.create.set_progress(-1.0, busy_text)
        job.progress.connect(self.create.set_progress)
        job.succeeded.connect(on_success)
        job.failed.connect(self._job_failed)
        job.cancelled.connect(lambda: self.create.set_message("Cancelled."))
        job.finished.connect(self._job_finished)
        job.start()

    def _job_finished(self) -> None:
        job, self._job = self._job, None
        if job is not None:
            job.deleteLater()
        self.create.set_busy(False)

    def _job_failed(self, message: str) -> None:
        self.create.set_message("")
        QtWidgets.QMessageBox.warning(self, "Forge3D", message)

    def _on_cancel(self) -> None:
        if self._job is not None:
            self._job.cancel()
            self.create.set_progress(-1.0, "Cancelling…")

    # ---------------------------------------------------------------- picture

    def _on_draw(self, prompt: str, style: str, seed: int) -> None:
        if not prompt.strip():
            QtWidgets.QMessageBox.information(self, "Forge3D", "Type a description of what you want first.")
            return
        self._prompt_for_name = prompt
        self._start_job(
            lambda progress, cancel: self.forge.text_to_images(
                prompt, style=style, count=4, seed=seed, progress=progress, cancel=cancel
            ),
            self._pictures_ready,
            busy_text="Starting the picture model…",
        )

    def _pictures_ready(self, images: list[Image.Image]) -> None:
        self.create.set_candidates(images)
        self.create.set_message("Pick the picture closest to what you want, then press “Make 3D model”.")

    def open_picture_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open a picture", "", _IMAGE_FILTER)
        if path:
            self.load_picture(Path(path))

    def load_picture(self, path: Path) -> None:
        try:
            with Image.open(path) as opened:
                image = opened.copy()
        except Exception as error:
            QtWidgets.QMessageBox.warning(self, "Forge3D", f"Could not open that picture: {error}")
            return
        self._prompt_for_name = path.stem
        self.create.set_candidates([])
        self.create.set_reference(image)
        self.create.set_message("Press “Make 3D model” to turn this picture into a model.")

    # ---------------------------------------------------------------- 3D

    def _on_make3d(self, engine: str, quality: str, seed: int, remove_bg: bool) -> None:
        image = self.create.reference_image()
        if image is None:
            return
        self.settings.engine, self.settings.quality, self.settings.remove_background = engine, quality, remove_bg
        self.settings.save()
        self._start_job(
            lambda progress, cancel: self.forge.image_to_mesh(
                image, engine_key=engine, quality=quality, seed=seed, remove_background=remove_bg,
                progress=progress, cancel=cancel,
            ),
            lambda mesh: self._mesh_ready(mesh, self.forge.native_up(engine)),
            busy_text="Starting…",
        )

    def open_mesh_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open a 3D model", "", _MESH_FILTER)
        if path:
            self.load_mesh(Path(path))

    def load_mesh(self, path: Path) -> None:
        try:
            mesh = meshops.load_mesh(path)
        except Exception as error:
            QtWidgets.QMessageBox.warning(self, "Forge3D", f"Could not open that model: {error}")
            return
        self._prompt_for_name = path.stem
        # glTF is Y-up by specification; STL/OBJ/PLY/3MF from slicers and CAD are Z-up.
        self._mesh_ready(mesh, "y" if path.suffix.lower() in (".glb", ".gltf") else "z")
        self.statusBar().showMessage(f"Opened {path.name}", 6000)

    def _mesh_ready(self, mesh: trimesh.Trimesh, source_up: str) -> None:
        self.raw = mesh
        self._cache = None
        self.options = meshops.PrepOptions()
        self.print.set_source_up(source_up)
        self.print.set_enabled(True)
        self.export_action.setEnabled(True)
        self.create.set_message("Done. Check the model on the right, then export it.")
        self._recompute(frame=True)

    # ================================================================== print prep

    def _schedule_recompute(self) -> None:
        self._debounce.start()

    def _recompute(self, *, frame: bool = False, thickness: bool = False) -> None:
        if self.raw is None:
            return
        self.print.read_into(self.options)
        request = {
            "raw": self.raw,
            "options": meshops.PrepOptions(**{**self.options.__dict__, "rotation": self.options.rotation.copy()}),
            "cache": self._cache,
            "frame": frame,
            "thickness": thickness,
            "settings": Settings(**self.settings.__dict__),
        }
        if self._prep_job is not None:
            self._prep_pending = request  # run once the current pass finishes (latest wins)
            return
        self._run_prep(request)

    def _run_prep(self, request: dict) -> None:
        def work(progress, cancel):
            options, settings = request["options"], request["settings"]
            notes: list[str] = []
            cached = request["cache"]
            if cached is not None and cached[0] == options.repair_key():
                repaired = cached[1]
            else:
                progress(-1.0, "Repairing mesh…")
                repaired = meshops.repair_stage(request["raw"], options, notes.append)
            mesh = meshops.placement_stage(repaired, options)
            report = meshops.analyze(
                mesh,
                bed_mm=(settings.bed_x, settings.bed_y, settings.bed_z),
                max_overhang_deg=settings.max_overhang_deg,
                min_wall_mm=settings.min_wall_mm,
                density_g_cm3=settings.density_g_cm3,
                measure_thickness=request["thickness"],
            )
            return _PrepResult(repaired, options.repair_key(), mesh, report, notes, request["frame"])

        job = JobThread(work, self)
        self._prep_job = job
        self.statusBar().showMessage("Preparing model…")
        job.succeeded.connect(self._prep_done)
        job.failed.connect(self._job_failed)
        job.finished.connect(self._prep_finished)
        job.start()

    def _prep_finished(self) -> None:
        job, self._prep_job = self._prep_job, None
        if job is not None:
            job.deleteLater()
        pending, self._prep_pending = self._prep_pending, None
        if pending is not None:
            pending["cache"] = self._cache
            self._run_prep(pending)
        else:
            self.statusBar().clearMessage()

    def _prep_done(self, result: _PrepResult) -> None:
        self._cache = (result.key, result.repaired)
        self.final, self.report = result.mesh, result.report
        self._last_notes = result.notes
        self.print.show_report(result.report, result.notes)
        self._refresh_view(frame=result.frame)

    def _refresh_view(self, frame: bool = False) -> None:
        if self.final is None:
            return
        self.viewport.show_mesh(
            self.final,
            overhang_deg=self.settings.max_overhang_deg if self.print.show_overhang.isChecked() else None,
            wireframe=self.print.wireframe.isChecked(),
            frame=frame is True,
        )

    def _on_rotate(self, axis: str, degrees: float) -> None:
        if self.raw is None:
            return
        self.options.rotation = meshops.axis_rotation(axis, degrees) @ self.options.rotation
        self._recompute()

    def _on_auto_orient(self) -> None:
        if self.raw is None:
            return
        self.print.read_into(self.options)
        raw, options, cache, settings = self.raw, self.options, self._cache, self.settings

        def search(progress, cancel):
            progress(-1.0, "Finding the best orientation…")
            if cache is not None and cache[0] == options.repair_key():
                repaired = cache[1]
            else:
                repaired = meshops.repair_stage(raw, options)
            return meshops.choose_orientation(repaired, options, settings.max_overhang_deg)

        job = JobThread(search, self)
        job.succeeded.connect(self._orientation_found)
        job.failed.connect(self._job_failed)
        job.finished.connect(job.deleteLater)
        self._orient_job = job
        self.statusBar().showMessage("Finding the best orientation…")
        job.start()

    def _orientation_found(self, result) -> None:
        rotation, info = result
        self.options.rotation = rotation
        self._recompute()
        self.statusBar().showMessage(
            f"Auto-orient: {info['overhang_fraction'] * 100:.0f}% of the surface still overhangs in the best position", 8000
        )

    # ================================================================== export

    def export(self, extension: str) -> None:
        if self.final is None:
            return
        suggested = self.settings.resolved_output_dir() / f"{_slug(self._prompt_for_name)}.{extension}"
        label = meshops.FORMATS[extension][0]
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export model", str(suggested), f"{label} (*.{extension})"
        )
        if not path:
            return
        path = Path(path)
        if path.suffix.lower() != f".{extension}":
            path = path.with_suffix(f".{extension}")
        try:
            meshops.export_mesh(self.final, path)
        except Exception as error:
            QtWidgets.QMessageBox.warning(self, "Forge3D", f"Could not save the file: {error}")
            return
        warning = ""
        if self.report is not None and not self.report.printable:
            warning = "  (it still has problems; see Print check)"
        self.statusBar().showMessage(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB){warning}", 12000)

    # ================================================================== dialogs & misc

    def open_settings(self) -> None:
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec():
            dialog.apply()
            self.viewport.set_bed(self.settings.bed_x, self.settings.bed_y)
            self._recompute()

    def show_hardware_report(self) -> None:
        gpu = hardware.gpu_info()
        missing = hardware.missing_ai_packages()
        lines = [
            f"GPU: {gpu.summary()}",
            "AI packages: " + ("all installed" if not missing else "missing " + ", ".join(missing)),
            f"Models folder: {self.settings.resolved_models_dir()}",
        ]
        QtWidgets.QMessageBox.information(self, "Hardware check", "\n".join(lines))

    def show_about(self) -> None:
        QtWidgets.QMessageBox.about(
            self,
            "About Forge3D",
            f"<b>Forge3D {__version__}</b><br>A local, subscription-free AI 3D model generator for 3D printing.<br><br>"
            "Everything runs on your computer. Models are downloaded once from Hugging Face.<br><br>"
            "<b>Licenses of the AI models:</b> Hunyuan3D-2 (Tencent) and SDXL-Turbo (Stability AI) are free for "
            "personal and research use but not for commercial use. Hunyuan3D-2 is not licensed in the EU, UK or "
            "South Korea. The &quot;Inflate silhouette&quot; engine and the print tools are MIT-licensed.",
        )

    @staticmethod
    def _open_folder(path: Path) -> None:
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

    # ---------------------------------------------------------------- drag & drop

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                self.load_picture(path)
                return
            if path.suffix.lower() in MESH_EXTENSIONS:
                self.load_mesh(path)
                return

    # ---------------------------------------------------------------- shutdown

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        for job in (self._job, self._prep_job):
            if job is not None:
                job.cancel()
                job.wait(3000)
        self.settings.engine = self.create.selected_engine()
        self.settings.target_height_mm = self.print.height.value()
        self.settings.save()
        self.forge.engines.release_all()
        super().closeEvent(event)
