"""Command-line interface: ``forge3d gen | prep | engines | doctor | gui``."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from . import __version__, hardware, meshops
from .config import Settings
from .engines import (
    QUALITY_PRESETS,
    STYLE_PRESETS,
    CancelToken,
    EngineUnavailable,
    OutOfVram,
    list_image_to_3d,
)


def _progress_printer():
    last = {"message": ""}

    def report(fraction: float, message: str) -> None:
        if message == last["message"] and fraction >= 0:
            return
        last["message"] = message
        prefix = "   ..." if fraction < 0 else f"{fraction * 100:5.1f}%"
        print(f"\r{prefix} {message:<90}", end="", file=sys.stderr, flush=True)

    return report


def _add_prep_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("print preparation")
    group.add_argument("--height", type=float, default=None, metavar="MM",
                       help="scale to this height in mm (default: the saved setting, 60)")
    group.add_argument("--keep-scale", action="store_true", help="do not rescale; keep the model's own size")
    group.add_argument("--up-axis", default=None, choices=meshops.UP_AXES,
                       help="which axis points up in the source mesh (default: the engine's, or z for files)")
    group.add_argument("--auto-orient", action="store_true", help="pick the orientation needing the least support")
    group.add_argument("--smooth", type=int, default=0, metavar="N", help="N volume-preserving smoothing passes")
    group.add_argument("--decimate", type=int, default=None, metavar="FACES", help="reduce to about this many triangles")
    group.add_argument("--flat-base", type=float, default=0.0, metavar="MM", help="cut MM off the bottom for a flat base")
    group.add_argument("--no-repair", action="store_true", help="skip floater removal and watertight repair")
    group.add_argument("--thickness", action="store_true", help="also estimate wall thickness (slower)")


def _prepare_and_report(raw, args, settings: Settings, source_up: str, out_path: Path) -> int:
    options = meshops.PrepOptions(
        source_up=args.up_axis or source_up,
        height_mm=None if args.keep_scale else (args.height or settings.target_height_mm),
        remove_floaters=not args.no_repair,
        make_watertight=not args.no_repair,
        smooth_iterations=args.smooth,
        decimate_faces=args.decimate,
        flat_base_mm=args.flat_base,
    )
    log: list[str] = []
    repaired = meshops.repair_stage(raw, options, log.append)
    if args.auto_orient:
        options.rotation, info = meshops.choose_orientation(repaired, options, settings.max_overhang_deg)
        log.append(f"Auto-orient: {info['overhang_fraction'] * 100:.0f}% of the surface overhangs after rotating")
    mesh = meshops.placement_stage(repaired, options)

    report = meshops.analyze(
        mesh,
        bed_mm=(settings.bed_x, settings.bed_y, settings.bed_z),
        max_overhang_deg=settings.max_overhang_deg,
        min_wall_mm=settings.min_wall_mm,
        density_g_cm3=settings.density_g_cm3,
        measure_thickness=args.thickness,
    )
    meshops.export_mesh(mesh, out_path)

    print(file=sys.stderr)
    for line in log:
        print(f"  - {line}")
    sx, sy, sz = report.size_mm
    print(f"Saved {out_path}")
    print(f"  size      {sx:.1f} x {sy:.1f} x {sz:.1f} mm   ({report.n_faces:,} triangles)")
    if report.volume_mm3:
        print(f"  material  {report.volume_mm3 / 1000:.1f} cm3, about {report.weight_g:.0f} g if printed solid")
    print(f"  supports  {report.overhang_pct:.0f}% of the surface overhangs")
    for issue in report.issues:
        tag = {"error": "ERROR", "warn": "warn ", "info": "ok   "}[issue.severity]
        print(f"  [{tag}] {issue.message}")
    return 0 if report.printable else 2


def _cmd_gen(args) -> int:
    from PIL import Image

    from .pipeline import Forge

    settings = Settings.load()
    forge = Forge(settings)
    cancel = CancelToken()
    progress = _progress_printer()
    out_path = Path(args.output)

    if args.image:
        image = Image.open(args.image)
    else:
        images = forge.text_to_images(
            args.prompt, style=args.style, count=args.candidates, seed=args.seed, progress=progress, cancel=cancel
        )
        for index, candidate in enumerate(images):
            candidate.save(out_path.with_name(f"{out_path.stem}_picture{index + 1}.png"))
        pick = min(max(args.pick, 1), len(images)) - 1
        image = images[pick]
        print(f"\nDrew {len(images)} picture(s); using #{pick + 1}. Others saved next to the output.", file=sys.stderr)

    engine_key = args.engine or settings.engine
    raw = forge.image_to_mesh(
        image,
        engine_key=engine_key,
        quality=args.quality or settings.quality,
        seed=args.seed,
        remove_background=not args.keep_background,
        progress=progress,
        cancel=cancel,
    )
    return _prepare_and_report(raw, args, settings, forge.native_up(engine_key), out_path)


def _cmd_prep(args) -> int:
    settings = Settings.load()
    raw = meshops.load_mesh(args.input)
    return _prepare_and_report(raw, args, settings, "z", Path(args.output))


def _cmd_engines(_args) -> int:
    print("Image -> 3D engines:\n")
    for info in list_image_to_3d():
        commercial = "commercial use OK" if info.commercial_use else "personal use only"
        print(f"  {info.key}")
        print(f"      {info.label}")
        print(f"      ~{info.approx_vram_gb:.1f} GB VRAM, {info.download_gb:.1f} GB download, {commercial}")
    print("\nQuality presets: " + ", ".join(QUALITY_PRESETS))
    print("Text styles:     " + ", ".join(STYLE_PRESETS))
    return 0


def _cmd_doctor(_args) -> int:
    settings = Settings.load()
    gpu = hardware.gpu_info()
    missing = hardware.missing_ai_packages()
    models = settings.resolved_models_dir()
    free_gb = shutil.disk_usage(models).free / 1024**3

    print(f"Forge3D {__version__}  (Python {sys.version.split()[0]}, {sys.platform})")
    print(f"GPU:        {gpu.summary()}")
    print(f"AI stack:   {'complete' if not missing else 'missing: ' + ', '.join(missing)}")
    print(f"Models dir: {models}  ({free_gb:.0f} GB free)")
    from .engines.download import missing_files
    from .engines.hunyuan3d import VARIANTS

    for variant in VARIANTS.values():
        files = [f"{variant.subfolder}/model.fp16.safetensors"]
        state = "downloaded" if not missing_files(models, variant.repo, files) else "not downloaded yet"
        print(f"  {variant.key:<24} {state}")
    ok = gpu.available and not missing
    print("\n" + ("Ready for AI generation." if ok else "AI generation is not ready; the 'inflate' engine and "
                                                         "print-prep tools still work."))
    return 0 if ok else 1


def _cmd_download(args) -> int:
    from .engines.base import CancelToken
    from .engines.download import ensure_files
    from .engines.hunyuan3d import VARIANTS, Hunyuan3DEngine
    from .engines.text2image import DEFAULT_REPO, SDXL_TURBO_FILES

    settings = Settings.load()
    settings.apply_environment()
    models = settings.resolved_models_dir()
    progress, cancel = _progress_printer(), CancelToken()

    keys = list(args.engines) or ([] if args.text_to_image else [settings.engine])
    for key in keys:
        if key == "inflate":
            continue
        if key not in VARIANTS:
            raise ValueError(f"Unknown engine '{key}'. Choose from: {', '.join(VARIANTS)}")
        engine = Hunyuan3DEngine(VARIANTS[key], settings)
        for repo, files in engine.model_files():
            ensure_files(models, repo, files, progress, cancel, engine.variant.label.split(" - ")[0])
        print(f"\r  {key}: ready{' ' * 80}")
    if args.text_to_image:
        ensure_files(models, DEFAULT_REPO, SDXL_TURBO_FILES, progress, cancel, "the text-to-image model")
        print(f"\r  text-to-image model: ready{' ' * 80}")
    print(f"Models are stored in {models}")
    return 0


def _cmd_selftest(args) -> int:
    from .selftest import run_selftest

    settings = Settings.load()
    engine = args.engine or settings.engine
    print(f"Generating a test model with '{engine}' ({args.quality} detail). The first run downloads the model.")
    result = run_selftest(settings, engine, args.quality, _progress_printer())
    print()
    print(f"  device       {result.device}")
    print(f"  time         {result.seconds:.1f} s")
    print(f"  mesh         {result.faces:,} triangles, watertight: {result.watertight}")
    if result.peak_vram_gb is not None:
        print(f"  GPU memory   peak {result.peak_vram_gb:.1f} GB of {result.total_vram_gb:.1f} GB")
    print("\n" + ("PASSED: this machine can generate printable models." if result.ok else "FAILED: the result was not printable."))
    return 0 if result.ok else 1


def _cmd_gui(_args) -> int:
    from .ui.app import main as gui_main

    return gui_main()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forge3d", description="Local AI 3D model generator for 3D printing.")
    parser.add_argument("--version", action="version", version=f"forge3d {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("gen", help="generate a printable model from text or a picture")
    source = gen.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="describe the object")
    source.add_argument("--image", help="path to a picture of the object")
    gen.add_argument("-o", "--output", required=True, help="output file: .stl .3mf .obj .ply .glb")
    gen.add_argument("--style", default="plain", choices=list(STYLE_PRESETS), help="text style preset")
    gen.add_argument("--candidates", type=int, default=1, help="how many pictures to draw from the prompt")
    gen.add_argument("--pick", type=int, default=1, help="which candidate picture to turn into 3D")
    gen.add_argument("--engine", choices=[e.key for e in list_image_to_3d()], help="image-to-3D engine")
    gen.add_argument("--quality", choices=list(QUALITY_PRESETS), help="surface detail")
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--keep-background", action="store_true", help="do not remove the picture's background")
    _add_prep_args(gen)
    gen.set_defaults(func=_cmd_gen)

    prep = sub.add_parser("prep", help="repair and size an existing mesh for printing")
    prep.add_argument("input", help="mesh file (.stl .obj .glb .ply .3mf ...)")
    prep.add_argument("-o", "--output", required=True)
    _add_prep_args(prep)
    prep.set_defaults(func=_cmd_prep)

    download = sub.add_parser("download", help="download AI models now (otherwise they download on first use)")
    download.add_argument("engines", nargs="*", help="engine keys (default: your selected engine)")
    download.add_argument("--text-to-image", action="store_true", help="also download the text-to-image model (~7 GB)")
    download.set_defaults(func=_cmd_download)

    sub.add_parser("engines", help="list engines").set_defaults(func=_cmd_engines)
    sub.add_parser("doctor", help="check GPU, packages and downloaded models").set_defaults(func=_cmd_doctor)
    selftest = sub.add_parser("selftest", help="generate one model end to end and report time and GPU memory use")
    selftest.add_argument("--engine", choices=[e.key for e in list_image_to_3d()])
    selftest.add_argument("--quality", choices=list(QUALITY_PRESETS), default="standard")
    selftest.set_defaults(func=_cmd_selftest)

    sub.add_parser("gui", help="open the desktop app").set_defaults(func=_cmd_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    except (EngineUnavailable, OutOfVram, ValueError, FileNotFoundError) as error:
        print(f"\nError: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
