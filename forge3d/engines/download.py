"""Model downloads with real byte progress, resume and cancel, fetching only the files we need.

Why not ``huggingface_hub``? Two reasons. Its per-file progress hooks differ between versions (older ones
do not pass ``tqdm_class`` to individual files, so a 4 GB download would show no progress), and the
Hunyuan3D repos ship every checkpoint twice (``.ckpt`` and ``.safetensors``), which its snapshot download
would fetch in full. A plain HTTPS download of exactly the files we name has neither problem.

Files land in ``<models_dir>/<repo_id>/<filename>``, which is where Hunyuan3D's own loader looks first
(``HY3DGEN_MODELS``), so it never tries to download anything itself.
"""

from __future__ import annotations

import http.client
import os
import urllib.error
import urllib.request
from pathlib import Path

from .base import CancelToken, EngineUnavailable, ProgressFn

_CHUNK = 1 << 20
_TIMEOUT = 60


def endpoint() -> str:
    """Hugging Face base URL (``HF_ENDPOINT`` lets users on restricted networks point at a mirror)."""
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def file_url(repo_id: str, filename: str, revision: str = "main") -> str:
    return f"{endpoint()}/{repo_id}/resolve/{revision}/{filename}"


def local_path(models_dir: Path, repo_id: str, filename: str) -> Path:
    return Path(models_dir) / repo_id / filename


def missing_files(models_dir: Path, repo_id: str, filenames: list[str]) -> list[str]:
    return [f for f in filenames if not local_path(models_dir, repo_id, f).is_file()]


def _open(url: str, headers: dict[str, str] | None = None, method: str = "GET"):
    request = urllib.request.Request(url, headers={"User-Agent": "forge3d", **(headers or {})}, method=method)
    return urllib.request.urlopen(request, timeout=_TIMEOUT)


def remote_size(url: str) -> int:
    with _open(url, method="HEAD") as response:
        return int(response.headers.get("Content-Length", 0))


def _fetch(url: str, destination: Path, expected: int, on_bytes, cancel: CancelToken) -> None:
    """Download ``url`` to ``destination`` via a ``.part`` file; an interrupted download resumes next time."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if expected and have > expected:
        part.unlink()
        have = 0

    headers = {"Range": f"bytes={have}-"} if have else {}
    try:
        response = _open(url, headers)
    except urllib.error.HTTPError as error:
        if error.code == 416 and part.exists():  # range not satisfiable: the .part file is already complete
            response = None
        else:
            raise
    if response is not None:
        with response:
            resumed = response.status == 206
            if have and not resumed:
                have = 0  # the server ignored the Range header and is resending everything
            on_bytes(have, reset=True)
            with open(part, "ab" if resumed else "wb") as out:
                while True:
                    cancel.check()
                    chunk = response.read(_CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
                    have += len(chunk)
                    on_bytes(have)

    size = part.stat().st_size
    if expected and size != expected:
        if size > expected:
            part.unlink(missing_ok=True)  # corrupt: start over next time
            raise EngineUnavailable(f"{destination.name} came back larger than expected. Please try again.")
        raise EngineUnavailable(  # short: the connection dropped. Keep the .part file so the next run resumes.
            f"{destination.name}: the download was interrupted at {size / 1e9:.2f} of {expected / 1e9:.2f} GB. "
            "Run it again to resume."
        )
    part.replace(destination)


def ensure_files(
    models_dir: Path,
    repo_id: str,
    filenames: list[str],
    progress: ProgressFn,
    cancel: CancelToken,
    label: str,
) -> None:
    """Download whichever of ``filenames`` are not already under ``models_dir/repo_id``."""
    todo = missing_files(models_dir, repo_id, filenames)
    if not todo:
        return
    cancel.check()
    progress(0.0, f"Contacting the model server for {label}...")
    try:
        urls = {name: file_url(repo_id, name) for name in todo}
        sizes = {name: remote_size(url) for name, url in urls.items()}
        grand_total = sum(sizes.values()) or 1
        finished = 0
        for index, name in enumerate(todo):
            cancel.check()

            def on_bytes(done: int, reset: bool = False, name=name, finished=finished, index=index) -> None:
                overall = finished + done
                progress(
                    min(overall / grand_total, 1.0),
                    f"Downloading {label}: {overall / 1e9:.2f} / {grand_total / 1e9:.2f} GB "
                    f"(file {index + 1} of {len(todo)}; first run only)",
                )

            _fetch(urls[name], local_path(models_dir, repo_id, name), sizes[name], on_bytes, cancel)
            finished += sizes[name]
    except EngineUnavailable:
        raise
    except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError) as error:
        if isinstance(error, OSError) and getattr(error, "errno", None) == 28:
            raise EngineUnavailable(f"Not enough free disk space in {models_dir} to download {label}.") from error
        raise EngineUnavailable(
            f"Could not download {label}. Check your internet connection (models are saved in {models_dir}). "
            f"Downloads resume where they stopped. Details: {error}"
        ) from error
