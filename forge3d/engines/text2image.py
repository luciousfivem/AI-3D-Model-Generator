"""Prompt -> candidate pictures (step one of text-to-3D).

Meshy-style "text to 3D" is really two stages: draw a clean reference picture of the object, then lift
it to 3D. Doing it in two visible steps lets you pick the best picture before spending time on 3D.
"""

from __future__ import annotations

import gc

from PIL import Image

from .. import hardware
from ..config import Settings
from .base import CancelToken, EngineUnavailable, OutOfVram, ProgressFn, TextToImageEngine
from .download import ensure_files

STYLE_PRESETS = {
    "plain": ("Plain (as typed)", ""),
    "figurine": ("Figurine / miniature", "tabletop miniature figurine, sculpted details, matte grey clay"),
    "character": ("Character", "stylized 3d character, full body, neutral standing pose, matte clay render"),
    "prop": ("Prop / object", "detailed prop, matte material"),
    "mechanical": ("Mechanical part", "clean industrial design, mechanical part, matte grey plastic"),
    "lowpoly": ("Low-poly", "low poly stylized 3d model, simple faceted shapes"),
}

# Keeps the subject isolated and fully visible, which is what image-to-3D models need.
_SUBJECT_SUFFIX = "single object, centered, full object in frame, plain white background, soft studio lighting, 3d render"

DEFAULT_REPO = "stabilityai/sdxl-turbo"

# The files diffusers needs for fp16 inference. Naming them skips the 10 GB fp32 weights and ONNX copies.
SDXL_TURBO_FILES = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model.fp16.safetensors",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer_2/merges.txt",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/vocab.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
]


def build_prompt(prompt: str, style: str = "plain") -> str:
    parts = [prompt.strip().rstrip(",."), STYLE_PRESETS.get(style, STYLE_PRESETS["plain"])[1], _SUBJECT_SUFFIX]
    return ", ".join(p for p in parts if p)


class SDXLTurboEngine(TextToImageEngine):
    SIZE = 512  # SDXL-Turbo is trained for 512x512; larger sizes degrade

    def __init__(self, settings: Settings, style: str = "plain"):
        self.settings = settings
        self.style = style
        self._pipe = None
        self._device = "cpu"

    def check_available(self) -> None:
        missing = [p for p in hardware.missing_ai_packages() if p in ("torch", "diffusers", "transformers")]
        if not hardware.has_module("accelerate"):
            missing.append("accelerate")
        if missing:
            raise EngineUnavailable(
                "The AI packages are not installed (" + ", ".join(missing) + "). Run scripts\\install.ps1 again."
            )

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    def load(self, progress: ProgressFn, cancel: CancelToken) -> None:
        if self.loaded:
            return
        self.check_available()
        self.settings.apply_environment()
        import torch
        from diffusers import AutoPipelineForText2Image

        repo = self.settings.text_to_image_model
        models_dir = self.settings.resolved_models_dir()
        if repo == DEFAULT_REPO:
            ensure_files(models_dir, repo, SDXL_TURBO_FILES, progress, cancel, "the text-to-image model")
            path = str(models_dir / repo)
        else:
            path = repo  # a custom model: let diffusers fetch it (no progress bar)
        cancel.check()

        progress(-1.0, "Loading text-to-image model...")
        gpu = hardware.gpu_info()
        self._device = "cuda" if gpu.available else "cpu"
        pipe = AutoPipelineForText2Image.from_pretrained(
            path,
            torch_dtype=torch.float16 if gpu.available else torch.float32,
            variant="fp16",
            use_safetensors=True,
        )
        if gpu.available:
            if gpu.total_gb < 12.0:
                pipe.enable_model_cpu_offload()  # SDXL does not fit alongside a desktop on an 8 GB card
            else:
                pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
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

    def generate(
        self,
        prompt: str,
        *,
        count: int,
        seed: int,
        progress: ProgressFn,
        cancel: CancelToken,
    ) -> list[Image.Image]:
        if not self.loaded:
            self.load(progress, cancel)
        import torch

        full_prompt = build_prompt(prompt, self.style)
        images: list[Image.Image] = []
        for index in range(count):
            cancel.check()

            def on_step(_pipe, step, _timestep, kwargs, index=index):
                cancel.check()
                progress((index + (step + 1) / 4) / count, f"Drawing picture {index + 1} of {count}...")
                return kwargs

            generator = torch.Generator(device="cpu").manual_seed((int(seed) + index) & 0x7FFFFFFF)
            try:
                result = self._pipe(
                    prompt=full_prompt,
                    num_inference_steps=4,
                    guidance_scale=0.0,  # Turbo is distilled without classifier-free guidance
                    width=self.SIZE,
                    height=self.SIZE,
                    generator=generator,
                    callback_on_step_end=on_step,
                )
            except torch.cuda.OutOfMemoryError as error:
                self.unload()
                raise OutOfVram("The GPU ran out of memory while drawing pictures. Close other GPU programs.") from error
            images.append(result.images[0])
        return images
