"""Runs the real Hunyuan3D-2 Mini Turbo model. Skipped unless you opt in:

    set FORGE3D_TEST_MODELS=C:\\path\\to\\models      (a folder that already holds the downloaded model)
    pytest tests/test_real_inference.py -m slow -s

On a GPU this takes seconds; on a CPU a few minutes. It is the quickest way to confirm a new machine works.
"""

import os
from pathlib import Path

import pytest

from forge3d import hardware, meshops
from forge3d.config import Settings
from forge3d.engines.download import missing_files
from forge3d.engines.hunyuan3d import VARIANTS
from forge3d.pipeline import Forge
from forge3d.selftest import shaded_ball

pytestmark = pytest.mark.slow

MODELS = os.environ.get("FORGE3D_TEST_MODELS", "")


@pytest.mark.skipif(not MODELS, reason="set FORGE3D_TEST_MODELS to a folder containing the downloaded model")
def test_hunyuan3d_mini_turbo_makes_a_printable_mesh():
    variant = VARIANTS["hunyuan3d-mini-turbo"]
    wanted = [f"{variant.subfolder}/model.fp16.safetensors", f"{variant.turbo_vae}/model.fp16.safetensors"]
    if missing_files(Path(MODELS), variant.repo, wanted):
        pytest.skip(f"model files not found under {MODELS}")
    if not hardware.has_module("hy3dgen"):
        pytest.skip("AI packages not installed")

    forge = Forge(Settings(models_dir=MODELS))
    mesh = forge.image_to_mesh(shaded_ball(), engine_key=variant.key, quality="draft", seed=1)
    assert len(mesh.faces) > 1000

    prepared = meshops.prepare(mesh, meshops.PrepOptions(source_up=forge.native_up(variant.key), height_mm=40))
    report = meshops.analyze(prepared)
    assert report.watertight and report.printable
    assert prepared.extents[2] == pytest.approx(40, rel=1e-3)
    forge.engines.release_all()
