"""High-level generation API used by both the GUI and the command line (no Qt in here)."""

from __future__ import annotations

import threading

import trimesh
from PIL import Image

from . import imaging
from .config import Settings
from .engines import (
    CancelToken,
    EngineManager,
    ProgressFn,
    create_image_to_3d,
    no_progress,
)


def _scaled(progress: ProgressFn, low: float, high: float) -> ProgressFn:
    """Map a stage's 0..1 progress into [low, high] of the whole job (-1 'busy' passes through)."""

    def report(fraction: float, message: str) -> None:
        progress(fraction if fraction < 0 else low + (high - low) * min(max(fraction, 0.0), 1.0), message)

    return report


class Forge:
    """Generates pictures and raw meshes. Print preparation lives in :mod:`forge3d.meshops`."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.load()
        self.settings.apply_environment()
        self.engines = EngineManager(self.settings)
        self._lock = threading.Lock()  # one GPU job at a time

    def native_up(self, engine_key: str) -> str:
        """Up-axis of meshes produced by ``engine_key`` (feed to ``PrepOptions.source_up``)."""
        return create_image_to_3d(engine_key, self.settings).native_up

    def text_to_images(
        self,
        prompt: str,
        *,
        style: str = "plain",
        count: int = 4,
        seed: int = 0,
        progress: ProgressFn = no_progress,
        cancel: CancelToken | None = None,
    ) -> list[Image.Image]:
        if not prompt.strip():
            raise ValueError("Type a description of what you want first.")
        cancel = cancel or CancelToken()
        with self._lock:
            engine = self.engines.text_to_image(style, _scaled(progress, 0.0, 0.5), cancel)
            return engine.generate(prompt, count=count, seed=seed, progress=_scaled(progress, 0.5, 1.0), cancel=cancel)

    def image_to_mesh(
        self,
        image: Image.Image,
        *,
        engine_key: str | None = None,
        quality: str | None = None,
        seed: int = 0,
        remove_background: bool | None = None,
        progress: ProgressFn = no_progress,
        cancel: CancelToken | None = None,
    ) -> trimesh.Trimesh:
        """Lift ``image`` to a raw mesh in the engine's native frame."""
        cancel = cancel or CancelToken()
        engine_key = engine_key or self.settings.engine
        quality = quality or self.settings.quality
        remove_bg = self.settings.remove_background if remove_background is None else remove_background

        progress(0.0, "Preparing picture (first run downloads a small background-removal model)...")
        prepared = imaging.prepare_input(image, remove_bg=remove_bg)
        cancel.check()
        with self._lock:
            engine = self.engines.image_to_3d(engine_key, _scaled(progress, 0.02, 0.35), cancel)
            return engine.generate(
                prepared, quality=quality, seed=seed, progress=_scaled(progress, 0.35, 1.0), cancel=cancel
            )
