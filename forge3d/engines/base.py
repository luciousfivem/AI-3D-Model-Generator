"""Engine interfaces shared by the GUI, the CLI and the tests."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import trimesh
from PIL import Image

ProgressFn = Callable[[float, str], None]  # fraction 0..1 (or -1 for "busy"), message


def no_progress(fraction: float, message: str) -> None:
    pass


class Cancelled(Exception):
    """Raised inside a job when the user pressed Cancel."""


class CancelToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self._event.is_set():
            raise Cancelled()


class EngineUnavailable(RuntimeError):
    """The engine cannot run here (missing packages, no GPU...). The message is shown to the user."""


class OutOfVram(RuntimeError):
    """The GPU ran out of memory mid-job. The message tells the user how to fit in less."""


@dataclass(frozen=True)
class EngineInfo:
    key: str
    label: str
    description: str
    license: str
    commercial_use: bool
    approx_vram_gb: float
    download_gb: float


# Generation quality presets: octree resolution = how finely the surface is sampled.
QUALITY_PRESETS = {
    "draft": {"label": "Draft (fast)", "octree_resolution": 192, "steps_scale": 0.6},
    "standard": {"label": "Standard", "octree_resolution": 256, "steps_scale": 1.0},
    "high": {"label": "High detail (slower, more VRAM)", "octree_resolution": 320, "steps_scale": 1.0},
    "ultra": {"label": "Ultra (needs 12 GB+)", "octree_resolution": 384, "steps_scale": 1.3},
}


class ImageTo3DEngine(ABC):
    """Turns one picture into a mesh. Meshes come back in the model's own frame (see ``native_up``)."""

    info: EngineInfo
    native_up: str = "y"

    @abstractmethod
    def check_available(self) -> None:
        """Raise :class:`EngineUnavailable` with an actionable message if the engine cannot run."""

    @abstractmethod
    def load(self, progress: ProgressFn, cancel: CancelToken) -> None:
        """Download (first run) and load the weights onto the best device."""

    @abstractmethod
    def unload(self) -> None:
        """Release GPU memory."""

    @property
    @abstractmethod
    def loaded(self) -> bool: ...

    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        *,
        quality: str,
        seed: int,
        progress: ProgressFn,
        cancel: CancelToken,
    ) -> trimesh.Trimesh: ...


class TextToImageEngine(ABC):
    """Turns a prompt into candidate pictures to feed an :class:`ImageTo3DEngine`."""

    @abstractmethod
    def check_available(self) -> None: ...

    @abstractmethod
    def load(self, progress: ProgressFn, cancel: CancelToken) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @property
    @abstractmethod
    def loaded(self) -> bool: ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        count: int,
        seed: int,
        progress: ProgressFn,
        cancel: CancelToken,
    ) -> list[Image.Image]: ...
