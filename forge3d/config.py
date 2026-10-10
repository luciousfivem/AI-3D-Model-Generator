"""User settings and on-disk locations."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from platformdirs import user_config_dir, user_data_dir

from . import APP_NAME


def config_dir() -> Path:
    path = Path(user_config_dir(APP_NAME, appauthor=False))
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    path = Path(user_data_dir(APP_NAME, appauthor=False))
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class Settings:
    # Printer / material
    bed_x: float = 220.0  # mm (Ender-3 class default)
    bed_y: float = 220.0
    bed_z: float = 250.0
    density_g_cm3: float = 1.24  # PLA
    min_wall_mm: float = 0.8  # ~2 perimeters with a 0.4 mm nozzle
    max_overhang_deg: float = 45.0  # steepest unsupported overhang, measured from vertical

    # Generation
    engine: str = "hunyuan3d-mini-turbo"
    quality: str = "standard"
    text_to_image_model: str = "stabilityai/sdxl-turbo"
    low_vram: bool | None = None  # None = decide from the GPU's memory
    remove_background: bool = True

    # Print prep defaults
    target_height_mm: float = 60.0

    # Locations ("" = default)
    models_dir: str = ""
    output_dir: str = ""

    @staticmethod
    def path() -> Path:
        return config_dir() / "settings.json"

    @classmethod
    def load(cls) -> "Settings":
        try:
            raw = json.loads(cls.path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self) -> None:
        try:
            self.path().write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        except OSError:
            pass  # a read-only profile must not stop the app

    def resolved_models_dir(self) -> Path:
        path = Path(self.models_dir) if self.models_dir else data_dir() / "models"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def resolved_output_dir(self) -> Path:
        path = Path(self.output_dir) if self.output_dir else data_dir() / "output"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def apply_environment(self) -> None:
        """Point the model libraries at our models folder (call before importing them)."""
        models = str(self.resolved_models_dir())
        os.environ["HY3DGEN_MODELS"] = models  # hy3dgen looks here before downloading
        os.environ.setdefault("HF_HOME", str(Path(models) / "_hf_cache"))
        os.environ.setdefault("U2NET_HOME", str(Path(models) / "_rembg"))  # background-removal model
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
