"""Image -> 3D shape with Tencent's Hunyuan3D-2 family (shape generation only).

Texture generation is deliberately skipped: it needs ~16 GB of VRAM plus CUDA extensions that must be
compiled on Windows, and a colour texture is useless for a single-material 3D print anyway.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import trimesh
from PIL import Image

from .. import hardware
from ..config import Settings
from .base import (
    QUALITY_PRESETS,
    CancelToken,
    EngineInfo,
    EngineUnavailable,
    ImageTo3DEngine,
    OutOfVram,
    ProgressFn,
)
from .download import ensure_files

LICENSE = "Tencent Hunyuan Non-Commercial (personal use; not licensed in the EU, UK or South Korea)"


@dataclass(frozen=True)
class HunyuanVariant:
    key: str
    label: str
    description: str
    repo: str
    subfolder: str
    turbo_vae: str | None  # subfolder of the "turbo" VAE for FlashVDM variants
    steps: int
    approx_vram_gb: float
    download_gb: float

    @property
    def turbo(self) -> bool:
        return self.turbo_vae is not None


VARIANTS: dict[str, HunyuanVariant] = {
    v.key: v
    for v in [
        HunyuanVariant(
            "hunyuan3d-mini-turbo",
            "Hunyuan3D-2 Mini Turbo - fast (recommended for 8 GB)",
            "Smallest and fastest. Good detail for figurines and props.",
            "tencent/Hunyuan3D-2mini",
            "hunyuan3d-dit-v2-mini-turbo",
            "hunyuan3d-vae-v2-mini-turbo",
            5,
            5.0,
            4.2,
        ),
        HunyuanVariant(
            "hunyuan3d-mini",
            "Hunyuan3D-2 Mini - balanced",
            "Same size as Mini Turbo but uses more sampling steps for cleaner shapes.",
            "tencent/Hunyuan3D-2mini",
            "hunyuan3d-dit-v2-mini",
            None,
            30,
            5.0,
            3.8,
        ),
        HunyuanVariant(
            "hunyuan3d-2-turbo",
            "Hunyuan3D-2 Turbo - full size, fast",
            "Full-size model with fast sampling. Best detail that still fits an 8 GB card.",
            "tencent/Hunyuan3D-2",
            "hunyuan3d-dit-v2-0-turbo",
            "hunyuan3d-vae-v2-0-turbo",
            5,
            6.5,
            5.3,
        ),
        HunyuanVariant(
            "hunyuan3d-2",
            "Hunyuan3D-2 - full size, best quality (slow)",
            "Highest quality of the family. Tight on 8 GB: use Standard quality or Low-VRAM mode.",
            "tencent/Hunyuan3D-2",
            "hunyuan3d-dit-v2-0",
            None,
            30,
            6.5,
            4.9,
        ),
    ]
}


def engine_info(variant: HunyuanVariant) -> EngineInfo:
    return EngineInfo(
        key=variant.key,
        label=variant.label,
        description=variant.description,
        license=LICENSE,
        commercial_use=False,
        approx_vram_gb=variant.approx_vram_gb,
        download_gb=variant.download_gb,
    )


class Hunyuan3DEngine(ImageTo3DEngine):
    native_up = "y"  # glTF-style: meshes come out Y-up

    def __init__(self, variant: HunyuanVariant, settings: Settings):
        self.variant = variant
        self.settings = settings
        self.info = engine_info(variant)
        self._pipe = None
        self._device = "cpu"
        self._low_vram = False

    # ------------------------------------------------------------------ lifecycle

    def check_available(self) -> None:
        missing = hardware.missing_ai_packages()
        if missing:
            raise EngineUnavailable(
                "The AI packages are not installed (" + ", ".join(missing) + "). "
                "Run scripts\\install.ps1 again, or: pip install -r requirements-ai.txt"
            )

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    def model_files(self) -> list[tuple[str, list[str]]]:
        v = self.variant
        groups = [(v.repo, [f"{v.subfolder}/config.yaml", f"{v.subfolder}/model.fp16.safetensors"])]
        if v.turbo_vae:
            groups.append((v.repo, [f"{v.turbo_vae}/config.yaml", f"{v.turbo_vae}/model.fp16.safetensors"]))
        return groups

    def load(self, progress: ProgressFn, cancel: CancelToken) -> None:
        if self.loaded:
            return
        self.check_available()
        self.settings.apply_environment()
        models_dir = self.settings.resolved_models_dir()

        for repo, files in self.model_files():
            ensure_files(models_dir, repo, files, progress, cancel, self.variant.label.split(" - ")[0])
        cancel.check()

        progress(-1.0, "Loading model into memory...")
        import torch
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        gpu = hardware.gpu_info()
        self._device = "cuda" if gpu.available else "cpu"
        dtype = torch.float16 if gpu.available else torch.float32  # fp16 is slow/unsupported on most CPUs
        self._low_vram = gpu.available and hardware.should_use_low_vram(self.settings.low_vram, gpu)

        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            self.variant.repo,
            subfolder=self.variant.subfolder,
            use_safetensors=True,
            device=self._device,
            dtype=dtype,
        )
        if self.variant.turbo:
            # 'mc' = scikit-image marching cubes; the faster 'dmc' needs the 'diso' CUDA extension,
            # which has no Windows wheels.
            pipe.enable_flashvdm(mc_algo="mc")
            # enable_flashvdm swaps in a turbo VAE that is always created in fp16. On a GPU that matches the
            # pipeline; on the CPU fallback (fp32) it would crash with a dtype mismatch, so match it up.
            pipe.vae.to(dtype=dtype)
        if self._low_vram:
            pipe.enable_model_cpu_offload()
        self._pipe = pipe

    def unload(self) -> None:
        self._pipe = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ------------------------------------------------------------------ generation

    def _steps(self, quality: str) -> int:
        if self.variant.turbo:
            return self.variant.steps  # distilled for exactly this many steps
        return max(12, round(self.variant.steps * QUALITY_PRESETS[quality]["steps_scale"]))

    def generate(
        self,
        image: Image.Image,
        *,
        quality: str,
        seed: int,
        progress: ProgressFn,
        cancel: CancelToken,
    ) -> trimesh.Trimesh:
        if not self.loaded:
            self.load(progress, cancel)
        import torch

        steps = self._steps(quality)
        resolution = QUALITY_PRESETS[quality]["octree_resolution"]
        if self._device == "cpu":
            progress(0.02, "No usable GPU: running on the CPU (this will be very slow)")

        def on_step(index: int, _timestep, _outputs) -> None:
            cancel.check()
            done = index + 1
            if done < steps:
                progress(0.05 + 0.80 * done / steps, f"Sculpting shape... step {done}/{steps}")
            else:
                progress(0.88, "Extracting surface mesh...")

        generator = torch.Generator(device="cpu").manual_seed(int(seed) & 0x7FFFFFFF)
        try:
            outputs = self._pipe(
                image=image,
                num_inference_steps=steps,
                guidance_scale=5.0,
                generator=generator,
                octree_resolution=resolution,
                num_chunks=4000 if self._low_vram else 8000,
                output_type="trimesh",
                callback=on_step,
                callback_steps=1,
                enable_pbar=False,
            )
        except torch.cuda.OutOfMemoryError as error:
            self.unload()
            raise OutOfVram(
                "The GPU ran out of memory. Try: Draft quality, the Mini Turbo model, closing other programs that "
                "use the GPU (browsers, games), or enabling Low-VRAM mode in Settings."
            ) from error

        mesh = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        if isinstance(mesh, (list, tuple)):
            mesh = mesh[0]
        if mesh is None or len(getattr(mesh, "faces", [])) == 0:
            raise RuntimeError(
                "The model produced no surface. Try a different picture (one clear object, plain background) "
                "or a different seed."
            )
        progress(0.95, "Shape generated")
        return mesh
