"""Engine registry and the VRAM-aware engine manager."""

from __future__ import annotations

from ..config import Settings
from .base import (
    QUALITY_PRESETS,
    Cancelled,
    CancelToken,
    EngineInfo,
    EngineUnavailable,
    ImageTo3DEngine,
    OutOfVram,
    ProgressFn,
    TextToImageEngine,
    no_progress,
)
from .hunyuan3d import VARIANTS, Hunyuan3DEngine, engine_info
from .inflate import InflateEngine
from .text2image import STYLE_PRESETS, SDXLTurboEngine, build_prompt

DEFAULT_ENGINE = "hunyuan3d-mini-turbo"


def list_image_to_3d() -> list[EngineInfo]:
    """All selectable image-to-3D engines, best first."""
    return [engine_info(v) for v in VARIANTS.values()] + [InflateEngine.info]


def create_image_to_3d(key: str, settings: Settings) -> ImageTo3DEngine:
    if key == InflateEngine.info.key:
        return InflateEngine()
    if key in VARIANTS:
        return Hunyuan3DEngine(VARIANTS[key], settings)
    raise KeyError(f"Unknown engine '{key}'. Available: {', '.join(e.key for e in list_image_to_3d())}")


class EngineManager:
    """Keeps at most one big model in GPU memory at a time.

    An 8 GB card cannot hold the text-to-image model and the 3D model together, so switching stages
    unloads the previous one first.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._image_to_3d: ImageTo3DEngine | None = None
        self._text_to_image: TextToImageEngine | None = None

    def image_to_3d(self, key: str, progress: ProgressFn, cancel: CancelToken) -> ImageTo3DEngine:
        if self._text_to_image is not None:
            self._text_to_image.unload()
            self._text_to_image = None
        if self._image_to_3d is not None and self._image_to_3d.info.key != key:
            self._image_to_3d.unload()
            self._image_to_3d = None
        if self._image_to_3d is None:
            self._image_to_3d = create_image_to_3d(key, self.settings)
        self._image_to_3d.load(progress, cancel)
        return self._image_to_3d

    def text_to_image(self, style: str, progress: ProgressFn, cancel: CancelToken) -> TextToImageEngine:
        if self._image_to_3d is not None:
            self._image_to_3d.unload()
            self._image_to_3d = None
        if self._text_to_image is None:
            self._text_to_image = SDXLTurboEngine(self.settings, style)
        self._text_to_image.style = style
        self._text_to_image.load(progress, cancel)
        return self._text_to_image

    def release_all(self) -> None:
        for engine in (self._image_to_3d, self._text_to_image):
            if engine is not None:
                engine.unload()
        self._image_to_3d = None
        self._text_to_image = None


__all__ = [
    "DEFAULT_ENGINE",
    "QUALITY_PRESETS",
    "STYLE_PRESETS",
    "Cancelled",
    "CancelToken",
    "EngineInfo",
    "EngineManager",
    "EngineUnavailable",
    "Hunyuan3DEngine",
    "ImageTo3DEngine",
    "InflateEngine",
    "OutOfVram",
    "ProgressFn",
    "SDXLTurboEngine",
    "TextToImageEngine",
    "build_prompt",
    "create_image_to_3d",
    "list_image_to_3d",
    "no_progress",
]
