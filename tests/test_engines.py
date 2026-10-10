import numpy as np
import pytest
from PIL import Image, ImageDraw

from forge3d import hardware, imaging, meshops
from forge3d.config import Settings
from forge3d.engines import (
    Cancelled,
    CancelToken,
    EngineManager,
    InflateEngine,
    build_prompt,
    create_image_to_3d,
    list_image_to_3d,
)
from forge3d.engines.download import missing_files
from forge3d.pipeline import Forge


def disc_image(size=200, background=(255, 255, 255, 255)):
    image = Image.new("RGBA", (size, size), background)
    ImageDraw.Draw(image).ellipse([40, 30, size - 40, size - 30], fill=(200, 40, 40, 255))
    return image


# --------------------------------------------------------------------------- imaging


def test_fallback_cutout_removes_plain_background_but_keeps_subject():
    cut = imaging._fallback_cutout(disc_image())
    alpha = np.asarray(cut.getchannel("A"))
    assert alpha[2, 2] == 0 and alpha[-3, -3] == 0  # corners are background
    assert alpha[100, 100] == 255  # centre is subject


def test_fallback_cutout_keeps_white_details_inside_the_subject():
    image = disc_image()
    ImageDraw.Draw(image).ellipse([90, 80, 110, 100], fill=(255, 255, 255, 255))  # white "eye"
    alpha = np.asarray(imaging._fallback_cutout(image).getchannel("A"))
    assert alpha[90, 100] == 255, "enclosed white regions must not be cut out"


def test_has_cutout():
    assert not imaging.has_cutout(disc_image())
    transparent = disc_image(background=(0, 0, 0, 0))
    assert imaging.has_cutout(transparent)
    assert not imaging.has_cutout(disc_image().convert("RGB"))


def test_crop_to_subject_is_square_and_padded():
    cut = imaging._fallback_cutout(disc_image(300))
    cropped = imaging.crop_to_subject(cut)
    assert cropped.width == cropped.height
    alpha = np.asarray(cropped.getchannel("A"))
    assert alpha[0].max() == 0 and alpha[:, 0].max() == 0  # breathing room on every side


def test_crop_to_subject_rejects_empty_images():
    with pytest.raises(ValueError):
        imaging.crop_to_subject(Image.new("RGBA", (50, 50), (0, 0, 0, 0)))


def test_prepare_input_returns_rgba_square(monkeypatch):
    monkeypatch.setattr(imaging, "remove_background", imaging._fallback_cutout)  # no model download in tests
    out = imaging.prepare_input(disc_image(300).convert("RGB"), remove_bg=True)
    assert out.mode == "RGBA" and out.width == out.height


# --------------------------------------------------------------------------- engines


def test_inflate_makes_a_printable_pillow():
    engine = create_image_to_3d("inflate", Settings())
    mesh = engine.generate(
        imaging.crop_to_subject(imaging._fallback_cutout(disc_image())),
        quality="draft",
        seed=0,
        progress=lambda f, m: None,
        cancel=CancelToken(),
    )
    assert mesh.is_watertight and mesh.volume > 0
    sx, sy, sz = mesh.extents
    assert sx == pytest.approx(sy, rel=0.15)  # a disc
    assert sz > 0.2 * sx  # actually has thickness

    prepared = meshops.prepare(mesh, meshops.PrepOptions(source_up=engine.native_up, height_mm=50))
    report = meshops.analyze(prepared)
    assert report.printable and report.fits_bed
    assert prepared.extents[2] == pytest.approx(50.0, rel=1e-3)


def test_inflate_is_deterministic_regardless_of_seed():
    engine = InflateEngine()
    image = imaging.crop_to_subject(imaging._fallback_cutout(disc_image()))
    a = engine.generate(image, quality="draft", seed=1, progress=lambda *_: None, cancel=CancelToken())
    b = engine.generate(image, quality="draft", seed=999, progress=lambda *_: None, cancel=CancelToken())
    assert np.allclose(a.vertices, b.vertices)


def test_inflate_stands_a_logo_upright():
    """Picture 'up' must end up as +Z after print prep (so a tall logo prints standing up)."""
    image = Image.new("RGBA", (120, 240), (255, 255, 255, 255))
    ImageDraw.Draw(image).rectangle([30, 20, 90, 220], fill=(0, 0, 0, 255))  # tall bar
    engine = InflateEngine()
    mesh = engine.generate(imaging.crop_to_subject(imaging._fallback_cutout(image)), quality="draft", seed=0,
                           progress=lambda *_: None, cancel=CancelToken())
    out = meshops.prepare(mesh, meshops.PrepOptions(source_up=engine.native_up, height_mm=None))
    assert out.extents[2] > out.extents[0] * 2  # taller than wide


def test_forge_pipeline_reports_monotonic_progress_and_can_cancel(monkeypatch):
    monkeypatch.setattr(imaging, "remove_background", imaging._fallback_cutout)
    forge = Forge(Settings())
    seen = []
    mesh = forge.image_to_mesh(
        disc_image(), engine_key="inflate", quality="draft", progress=lambda f, m: seen.append((f, m))
    )
    assert mesh.is_watertight
    fractions = [f for f, _ in seen if f >= 0]
    assert fractions == sorted(fractions) and fractions[0] <= 0.05 and fractions[-1] >= 0.9

    token = CancelToken()
    token.cancel()
    with pytest.raises(Cancelled):
        forge.image_to_mesh(disc_image(), engine_key="inflate", quality="draft", cancel=token)


def test_registry_lists_engines_and_rejects_unknown_keys():
    keys = [e.key for e in list_image_to_3d()]
    assert "hunyuan3d-mini-turbo" in keys and keys[-1] == "inflate"
    assert all(not e.commercial_use for e in list_image_to_3d() if e.key.startswith("hunyuan"))
    with pytest.raises(KeyError):
        create_image_to_3d("nope", Settings())


def test_manager_keeps_one_engine_and_releases():
    manager = EngineManager(Settings())
    first = manager.image_to_3d("inflate", lambda *_: None, CancelToken())
    assert manager.image_to_3d("inflate", lambda *_: None, CancelToken()) is first
    manager.release_all()
    assert manager._image_to_3d is None


def test_unavailable_engines_explain_themselves(monkeypatch):
    from forge3d.engines import EngineUnavailable

    monkeypatch.setattr(hardware, "missing_ai_packages", lambda: ["torch", "hy3dgen"])
    engine = create_image_to_3d("hunyuan3d-mini-turbo", Settings())
    with pytest.raises(EngineUnavailable, match="torch, hy3dgen"):
        engine.check_available()


def test_text_prompt_template_keeps_subject_isolated():
    prompt = build_prompt("a small dragon", "figurine")
    assert prompt.startswith("a small dragon") and "figurine" in prompt and "white background" in prompt
    assert build_prompt("a cup,", "plain").startswith("a cup, single object")


# --------------------------------------------------------------------------- misc


def test_missing_files_checks_the_layout_the_hunyuan_loader_expects(tmp_path):
    repo, sub = "tencent/Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini-turbo"
    files = [f"{sub}/config.yaml", f"{sub}/model.fp16.safetensors"]
    assert missing_files(tmp_path, repo, files) == files
    target = tmp_path / repo / sub
    target.mkdir(parents=True)
    (target / "config.yaml").write_text("x")
    assert missing_files(tmp_path, repo, files) == [files[1]]


def test_hardware_reports_cleanly_without_a_gpu():
    info = hardware.gpu_info()
    assert isinstance(info.summary(), str) and info.summary()
    assert isinstance(hardware.missing_ai_packages(), list)
    assert hardware.should_use_low_vram(True) is True
    assert hardware.should_use_low_vram(False) is False
    assert hardware.should_use_low_vram(None, hardware.GpuInfo(True, "x", 6.0, 5.0)) is True
    assert hardware.should_use_low_vram(None, hardware.GpuInfo(True, "x", 8.0, 7.0)) is False


def test_settings_round_trip_and_ignore_unknown_keys(tmp_path, monkeypatch):
    monkeypatch.setattr("forge3d.config.user_config_dir", lambda *a, **k: str(tmp_path))
    settings = Settings(bed_x=256.0, engine="inflate")
    settings.save()
    (tmp_path / "settings.json").write_text(
        (tmp_path / "settings.json").read_text().replace("{", '{"from_the_future": 1,', 1)
    )
    loaded = Settings.load()
    assert loaded.bed_x == 256.0 and loaded.engine == "inflate"


def test_settings_survive_a_corrupt_file(tmp_path, monkeypatch):
    monkeypatch.setattr("forge3d.config.user_config_dir", lambda *a, **k: str(tmp_path))
    (tmp_path / "settings.json").write_text("{not json")
    assert Settings.load() == Settings()


def test_selftest_runs_end_to_end_with_the_no_ai_engine():
    from forge3d.selftest import run_selftest

    result = run_selftest(Settings(), "inflate", "draft")
    assert result.ok and result.engine == "inflate"
    assert result.size_mm[2] == pytest.approx(40, rel=1e-3)
    assert result.peak_vram_gb is None or result.peak_vram_gb >= 0
