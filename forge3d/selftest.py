"""One-command end-to-end check for a new machine: generate a small model and report time and GPU memory."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import hardware, meshops
from .config import Settings
from .engines import ProgressFn, no_progress
from .pipeline import Forge


def shaded_ball(size: int = 512) -> Image.Image:
    """A lit sphere on white. It has the depth cues image-to-3D models expect (flat drawings do not)."""
    y, x = np.mgrid[-1 : 1 : size * 1j, -1 : 1 : size * 1j]
    r2 = x**2 + y**2
    z = np.sqrt(np.clip(1 - r2 / 0.64, 0, None))
    light = np.clip(0.35 + 0.65 * (0.32 * x - 0.4 * y + 0.76 * z), 0, 1)
    pixels = np.where(r2[..., None] < 0.64, np.array([200, 90, 60]) * light[..., None], 255)
    return Image.fromarray(pixels.astype(np.uint8))


@dataclass
class SelfTestResult:
    engine: str
    device: str
    seconds: float
    faces: int
    watertight: bool
    printable: bool
    size_mm: tuple[float, float, float]
    peak_vram_gb: float | None  # None on CPU
    total_vram_gb: float | None

    @property
    def ok(self) -> bool:
        return self.faces > 1000 and self.watertight and self.printable


def run_selftest(
    settings: Settings, engine_key: str | None = None, quality: str = "standard", progress: ProgressFn = no_progress
) -> SelfTestResult:
    forge = Forge(settings)
    engine_key = engine_key or settings.engine
    gpu = hardware.gpu_info()
    torch = None
    if gpu.available:
        import torch

        torch.cuda.reset_peak_memory_stats()

    started = time.time()
    mesh = forge.image_to_mesh(shaded_ball(), engine_key=engine_key, quality=quality, seed=1, progress=progress)
    elapsed = time.time() - started

    options = meshops.PrepOptions(source_up=forge.native_up(engine_key), height_mm=40)
    prepared = meshops.prepare(mesh, options)
    report = meshops.analyze(prepared)
    peak = None
    if torch is not None:
        peak = max(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()) / 1024**3
    forge.engines.release_all()
    return SelfTestResult(
        engine=engine_key,
        device="cuda" if gpu.available else "cpu",
        seconds=elapsed,
        faces=len(mesh.faces),
        watertight=report.watertight,
        printable=report.printable,
        size_mm=report.size_mm,
        peak_vram_gb=peak,
        total_vram_gb=gpu.total_gb if gpu.available else None,
    )
