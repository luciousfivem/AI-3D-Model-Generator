"""What can this machine run? (Safe to call with or without PyTorch installed.)"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass
class GpuInfo:
    available: bool
    name: str = ""
    total_gb: float = 0.0
    free_gb: float = 0.0
    note: str = ""  # why the GPU is unusable, if it is

    def summary(self) -> str:
        if self.available:
            return f"{self.name} - {self.total_gb:.1f} GB ({self.free_gb:.1f} GB free)"
        return self.note or "No GPU"


def has_module(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


# import name -> pip name, for a helpful message
_AI_PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "diffusers": "diffusers",
    "transformers": "transformers",
    "hy3dgen": "hy3dgen",
    "omegaconf": "omegaconf",
    "einops": "einops",
    "skimage": "scikit-image",
    "pymeshlab": "pymeshlab",
    "cv2": "opencv-python-headless",
    "yaml": "pyyaml",
}


def missing_ai_packages() -> list[str]:
    """pip names of AI-stack packages that are not installed."""
    return [pip for module, pip in _AI_PACKAGES.items() if not has_module(module)]


def gpu_info() -> GpuInfo:
    if not has_module("torch"):
        return GpuInfo(False, note="PyTorch is not installed")
    import torch

    if not torch.cuda.is_available():
        build = getattr(torch.version, "cuda", None)
        why = "PyTorch is a CPU-only build" if build is None else "No CUDA GPU detected (update the NVIDIA driver)"
        return GpuInfo(False, note=why)
    try:
        index = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(index)
        return GpuInfo(True, torch.cuda.get_device_name(index), total / 1024**3, free / 1024**3)
    except Exception as error:  # a driver hiccup must not crash the UI
        return GpuInfo(False, note=f"CUDA error: {error}")


def should_use_low_vram(setting: bool | None, gpu: GpuInfo | None = None) -> bool:
    """Explicit setting wins; otherwise enable offloading on cards with under 7 GB total."""
    if setting is not None:
        return setting
    gpu = gpu or gpu_info()
    return gpu.available and gpu.total_gb < 7.0
