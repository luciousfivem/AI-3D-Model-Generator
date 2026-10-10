import http.server
import threading

import pytest

from forge3d.engines import Cancelled, CancelToken, EngineUnavailable
from forge3d.engines.download import ensure_files, file_url, local_path, missing_files

REPO = "acme/model"
PAYLOAD = bytes(range(256)) * 20_000  # ~5 MB, so several 1 MB chunks


class _Handler(http.server.BaseHTTPRequestHandler):
    """Serves PAYLOAD for any path, with HTTP Range support (like the Hugging Face CDN)."""

    truncate = False  # simulate a server that sends fewer bytes than it promises
    ignore_range = False

    def log_message(self, *args):
        pass

    def _send(self, body_allowed: bool):
        start = 0
        rng = self.headers.get("Range")
        if rng and not self.ignore_range:
            start = int(rng.split("=")[1].split("-")[0])
        if start >= len(PAYLOAD):
            self.send_response(416)
            self.end_headers()
            return
        body = PAYLOAD[start:]
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body_allowed:
            self.wfile.write(body[: len(body) // 2] if self.truncate else body)

    def do_HEAD(self):
        self._send(False)

    def do_GET(self):
        self._send(True)


@pytest.fixture()
def server(monkeypatch):
    handler = type("H", (_Handler,), {})
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{httpd.server_address[1]}")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    yield handler
    httpd.shutdown()


def test_url_layout_follows_hugging_face_conventions(monkeypatch):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    assert file_url("tencent/Hunyuan3D-2mini", "sub/model.fp16.safetensors") == (
        "https://huggingface.co/tencent/Hunyuan3D-2mini/resolve/main/sub/model.fp16.safetensors"
    )


def test_downloads_with_monotonic_progress(server, tmp_path):
    seen = []
    ensure_files(tmp_path, REPO, ["a/one.bin", "b/two.bin"], lambda f, m: seen.append((f, m)), CancelToken(), "test model")
    for name in ("a/one.bin", "b/two.bin"):
        assert local_path(tmp_path, REPO, name).read_bytes() == PAYLOAD
    fractions = [f for f, _ in seen]
    assert fractions == sorted(fractions)
    assert fractions[-1] == pytest.approx(1.0)
    assert any("file 2 of 2" in m for _, m in seen)
    assert not list(tmp_path.rglob("*.part"))


def test_already_present_files_are_not_fetched_again(server, tmp_path):
    target = local_path(tmp_path, REPO, "x.bin")
    target.parent.mkdir(parents=True)
    target.write_bytes(b"cached")
    ensure_files(tmp_path, REPO, ["x.bin"], lambda f, m: pytest.fail("should not download"), CancelToken(), "m")
    assert target.read_bytes() == b"cached"
    assert missing_files(tmp_path, REPO, ["x.bin", "y.bin"]) == ["y.bin"]


def test_cancel_mid_download_keeps_a_partial_file_and_resumes(server, tmp_path):
    token = CancelToken()

    def cancel_after_first_chunks(fraction, message):
        if fraction > 0.3:
            token.cancel()

    with pytest.raises(Cancelled):
        ensure_files(tmp_path, REPO, ["big.bin"], cancel_after_first_chunks, token, "m")
    part = local_path(tmp_path, REPO, "big.bin.part")
    assert part.exists() and 0 < part.stat().st_size < len(PAYLOAD)
    assert not local_path(tmp_path, REPO, "big.bin").exists()

    ensure_files(tmp_path, REPO, ["big.bin"], lambda f, m: None, CancelToken(), "m")  # resumes
    assert local_path(tmp_path, REPO, "big.bin").read_bytes() == PAYLOAD
    assert not part.exists()


def test_resume_survives_a_server_that_ignores_range(server, tmp_path):
    server.ignore_range = True
    part = local_path(tmp_path, REPO, "f.bin.part")
    part.parent.mkdir(parents=True)
    part.write_bytes(PAYLOAD[:1000])
    ensure_files(tmp_path, REPO, ["f.bin"], lambda f, m: None, CancelToken(), "m")
    assert local_path(tmp_path, REPO, "f.bin").read_bytes() == PAYLOAD


def test_dropped_connection_is_reported_and_resumable(server, tmp_path):
    server.truncate = True  # the server hangs up halfway through
    with pytest.raises(EngineUnavailable, match="resume"):
        ensure_files(tmp_path, REPO, ["t.bin"], lambda f, m: None, CancelToken(), "m")
    assert not local_path(tmp_path, REPO, "t.bin").exists(), "a partial file must never be mistaken for a model"
    assert local_path(tmp_path, REPO, "t.bin.part").exists()

    server.truncate = False  # connection is back: picks up where it stopped
    ensure_files(tmp_path, REPO, ["t.bin"], lambda f, m: None, CancelToken(), "m")
    assert local_path(tmp_path, REPO, "t.bin").read_bytes() == PAYLOAD


def test_unreachable_server_gives_an_actionable_message(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:1")  # nothing listens here
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    with pytest.raises(EngineUnavailable, match="internet connection"):
        ensure_files(tmp_path, REPO, ["x.bin"], lambda f, m: None, CancelToken(), "the model")
