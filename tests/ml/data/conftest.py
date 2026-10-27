"""A local HTTP file server with Range support and fault injection, for streaming tests."""

import shutil
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from ml.data.download import FASTTEXT_MAGIC


@dataclass
class FileServer:
    """Serves ``root``; ``drops[path] = n`` sends n bytes of the next response, then hangs up."""

    root: Path
    base_url: str = ""
    drops: dict[str, int] = field(default_factory=dict)
    ignore_range: bool = False
    requests: list[tuple[str, str | None]] = field(default_factory=list)  # (path, Range header)

    def url(self, name: str) -> str:
        return f"{self.base_url}/{name}"


def _handler(state: FileServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            name = self.path.lstrip("/").split("?")[0]
            state.requests.append((name, self.headers.get("Range")))
            path = state.root / name
            if not path.is_file():
                self.send_error(404)
                return
            data = path.read_bytes()
            start = 0
            range_header = self.headers.get("Range")
            if range_header and not state.ignore_range:
                start = int(range_header.split("=")[1].split("-")[0])
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
            else:
                self.send_response(200)
            body = data[start:]
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Last-Modified", "Mon, 12 Oct 2026 00:00:00 GMT")
            self.end_headers()
            drop = state.drops.pop(name, None)
            self.wfile.write(body if drop is None else body[:drop])

        def log_message(self, *args: object) -> None:
            pass

    return Handler


def start_file_server(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[FileServer, ThreadingHTTPServer]:
    """A running server over ``root``; proxies are disabled for localhost. The caller shuts the
    server down (``file_server`` below, or a session-scoped user such as tests/service)."""
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    root.mkdir(exist_ok=True)
    state = FileServer(root)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(state))
    state.base_url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return state, server


@pytest.fixture
def file_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FileServer]:
    """A running server over ``tmp_path / "served"``; proxies are disabled for localhost."""
    state, server = start_file_server(tmp_path / "served", monkeypatch)
    yield state
    server.shutdown()
    server.server_close()


@pytest.fixture
def tiny_cfg(
    tmp_config: dict[str, Any], file_server: FileServer, tiny_dataset: Path
) -> dict[str, Any]:
    """A config whose sources are the tiny raw fixture served over HTTP, with small checkpoints."""
    return point_at_tiny(tmp_config, file_server, tiny_dataset)


def point_at_tiny(
    tmp_config: dict[str, Any], file_server: FileServer, tiny_dataset: Path
) -> dict[str, Any]:
    """Serve the tiny raw fixture from ``file_server`` and point the config's sources at it."""
    for path in (tiny_dataset / "raw").iterdir():
        shutil.copy(path, file_server.root / path.name)
    (file_server.root / "lid.176.ftz").write_bytes(FASTTEXT_MAGIC + b"fake model")
    tmp_config["sources"].update(
        fasttext_lid=file_server.url("lid.176.ftz"),
        ratings_5core=file_server.url("Books.csv.gz"),
        meta_books=file_server.url("meta_Books.jsonl.gz"),
        reviews_books=file_server.url("Books.jsonl.gz"),
        goodreads_base=file_server.base_url,
    )
    tmp_config["data"]["stream"] = {
        "chunk_bytes": 4096,
        "retries": 2,
        "timeout_s": 5,
        "checkpoint_rows": 10,
    }
    return tmp_config
