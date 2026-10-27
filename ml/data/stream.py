"""Stream lines from a URL or a local file, gunzipping on the fly; nothing is written to disk.

Large sources (meta_Books 4.94 GB, reviews 6.22 GB, the Goodreads graph) are never stored
(docs/decisions.md D18). Compressed bytes arrive over HTTP, are decompressed in memory and split
into lines. If the connection drops, the reader reconnects with an HTTP Range request starting at
the exact compressed byte it reached and keeps feeding the same decompressor, so a network
glitch costs nothing. A crashed process cannot resume a gzip stream (the decompressor state is
gone), so stages checkpoint their *output* with ``PartWriter`` and skip already-processed lines
when they restart.
"""

import json
import os
import time
import zlib
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import urllib3

from ml.common.errors import StreamError
from ml.common.manifest import write_json_atomic

_GZIP_WBITS = 16 + zlib.MAX_WBITS
_TRANSIENT = (requests.RequestException, urllib3.exceptions.HTTPError, OSError)


@dataclass(frozen=True)
class StreamSettings:
    """Network behaviour; values come from ``data.stream`` in ml/config.yaml."""

    chunk_bytes: int
    retries: int
    timeout_s: float
    backoff_s: float = 1.0  # first retry delay; doubles per consecutive failure

    @classmethod
    def from_config(cls, stream: Mapping[str, Any]) -> "StreamSettings":
        """Build from the ``data.stream`` mapping (``checkpoint_rows`` is for ``PartWriter``)."""
        return cls(int(stream["chunk_bytes"]), int(stream["retries"]), float(stream["timeout_s"]))


@dataclass
class StreamStats:
    """What a stream read; stages copy it into their manifest notes."""

    source: str
    lines: int = 0
    bytes_read: int = 0  # compressed bytes received
    reconnects: int = 0
    size: int | None = None  # Content-Length or file size
    last_modified: str | None = None
    complete: bool = False  # read to the end, not stopped by a line limit

    def as_note(self) -> dict[str, Any]:
        """JSON-ready copy for ``ctx.notes``."""
        return asdict(self)


def is_url(source: str) -> bool:
    """True for http(s) sources."""
    return source.startswith(("http://", "https://"))


def iter_lines(
    source: str,
    settings: StreamSettings,
    *,
    limit: int | None = None,
    stats: StreamStats | None = None,
    tee: BinaryIO | None = None,
) -> Iterator[str]:
    """Yield the lines of ``source`` (URL or path) without line endings (LF or CRLF).

    ``.gz`` sources are decompressed (multi-member gzip included) and must end with a complete
    gzip trailer; a URL must deliver exactly its Content-Length. ``limit`` stops after that many
    lines and leaves ``stats.complete`` False. ``tee`` receives the raw (compressed) bytes, for
    the few sources that are kept.
    """
    stats = stats if stats is not None else StreamStats(source)
    gunzip = _Gunzip() if source.split("?")[0].endswith(".gz") else None
    chunks = iter_bytes(source, settings, stats)
    buffer = b""
    try:
        for chunk in chunks:
            if tee is not None:
                tee.write(chunk)
            buffer += gunzip.feed(chunk) if gunzip else chunk
            *lines, buffer = buffer.split(b"\n")
            for line in lines:
                if limit is not None and stats.lines >= limit:
                    return
                stats.lines += 1
                yield line.removesuffix(b"\r").decode("utf-8", "replace")
        if gunzip and not gunzip.finished:
            raise StreamError(f"{source}: gzip stream ended early (truncated file?)")
        if buffer and (limit is None or stats.lines < limit):
            stats.lines += 1
            yield buffer.removesuffix(b"\r").decode("utf-8", "replace")
        stats.complete = True
    finally:
        chunks.close()


def iter_bytes(source: str, settings: StreamSettings, stats: StreamStats) -> Iterator[bytes]:
    """Raw (still compressed) bytes of ``source``; ``stats.bytes_read`` tracks the offset."""
    if not is_url(source):
        stats.size = os.path.getsize(source)
        with open(source, "rb") as f:
            while chunk := f.read(settings.chunk_bytes):
                stats.bytes_read += len(chunk)
                yield chunk
        return
    yield from _http_bytes(source, settings, stats)


def _http_bytes(url: str, settings: StreamSettings, stats: StreamStats) -> Iterator[bytes]:
    session = requests.Session()
    failures = 0
    while True:
        headers = {"Range": f"bytes={stats.bytes_read}-"} if stats.bytes_read else {}
        try:
            with session.get(url, stream=True, timeout=settings.timeout_s, headers=headers) as r:
                _check_response(r, url, stats)
                # raw bytes: never let requests undo a Content-Encoding, offsets must stay exact
                for chunk in r.raw.stream(settings.chunk_bytes, decode_content=False):
                    if chunk:
                        stats.bytes_read += len(chunk)
                        failures = 0
                        yield chunk
            if stats.size is None or stats.bytes_read == stats.size:
                return
            if stats.bytes_read > stats.size:
                raise StreamError(f"{url}: received more bytes than its Content-Length")
            raise requests.ConnectionError(f"closed at byte {stats.bytes_read} of {stats.size}")
        except _TRANSIENT as e:
            failures += 1
            if failures > settings.retries:
                raise StreamError(f"{url}: gave up after {settings.retries} retries: {e}") from e
            stats.reconnects += 1
            time.sleep(min(settings.backoff_s * 2 ** (failures - 1), 60.0))


def _check_response(r: requests.Response, url: str, stats: StreamStats) -> None:
    if r.status_code >= 500:
        raise requests.ConnectionError(f"server error {r.status_code}")  # transient: retried
    if stats.bytes_read == 0:
        if r.status_code != 200:
            raise StreamError(f"{url}: HTTP {r.status_code}")
        length = r.headers.get("Content-Length")
        stats.size = int(length) if length else None
        stats.last_modified = r.headers.get("Last-Modified")
        return
    content_range = r.headers.get("Content-Range", "")
    if r.status_code != 206 or not content_range.startswith(f"bytes {stats.bytes_read}-"):
        raise StreamError(f"{url}: cannot resume at byte {stats.bytes_read} (HTTP {r.status_code})")


class _Gunzip:
    """Incremental gunzip that also handles concatenated gzip members."""

    def __init__(self) -> None:
        self._d = zlib.decompressobj(_GZIP_WBITS)

    def feed(self, chunk: bytes) -> bytes:
        out = [self._d.decompress(chunk)]
        while self._d.eof and self._d.unused_data:
            rest = self._d.unused_data
            self._d = zlib.decompressobj(_GZIP_WBITS)
            out.append(self._d.decompress(rest))
        return b"".join(out)

    @property
    def finished(self) -> bool:
        return self._d.eof


@dataclass
class PartWriter:
    """Checkpointed output for a long stream, kept in a stage's work dir.

    Records are buffered and written as numbered parquet parts; after each part a progress file
    stores the number of input lines fully processed and the running counters, for that part and
    every earlier one. A restarted stage builds a new ``PartWriter`` on the same work dir, skips
    the first ``resume_lines`` input lines and continues; counters resume from the checkpoint, so
    totals match an uninterrupted run.

    On Google Drive a crash can lose a part file while ``progress.json`` survives (the FUSE mount
    does not keep write order). Every recorded part is therefore checked on resume, and the writer
    rolls back to the last part that is present and readable (``rolled_back`` says from where).
    ``finished`` is True once ``close`` recorded a complete stream, so a stage can skip
    re-streaming when only its post-processing failed.
    """

    work_dir: Path
    every: int
    counters: Counter[str] = field(default_factory=Counter, init=False)
    resume_lines: int = field(default=0, init=False)
    finished: bool = field(default=False, init=False)
    rolled_back: int | None = field(default=None, init=False)  # parts recorded before a rollback
    _parts: int = field(default=0, init=False)
    _history: list[dict[str, Any]] = field(default_factory=list, init=False)
    _buffer: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        progress = self.work_dir / "progress.json"
        if progress.exists():
            state = json.loads(progress.read_text(encoding="utf-8"))
            self.resume_lines, self._parts = state["lines"], state["parts"]
            self.counters = Counter(state["counters"])
            self._history = list(state.get("history", []))
            self.finished = bool(state.get("finished", False))
            good = next((k for k in range(self._parts) if not self._readable(k)), self._parts)
            if good < self._parts:
                self._roll_back(good)
        for stray in self.work_dir.glob("part-*.parquet"):  # written after the last checkpoint
            if int(stray.stem.split("-")[1]) >= self._parts:
                stray.unlink()

    def _readable(self, k: int) -> bool:
        try:
            pq.read_metadata(self._part(k))
        except (OSError, pa.ArrowInvalid):
            return False
        return True

    def _roll_back(self, good: int) -> None:
        """Keep parts ``0..good-1`` only. Without per-part history (checkpoints written before
        it existed) the stream restarts from the beginning."""
        self.rolled_back = self._parts
        if len(self._history) < good or good == 0:
            good = 0
        self._parts = good
        self.resume_lines = self._history[good - 1]["lines"] if good else 0
        self.counters = Counter(self._history[good - 1]["counters"] if good else {})
        self._history = self._history[:good]
        self.finished = False
        self._save()

    def add(self, record: dict[str, Any]) -> None:
        """Buffer one output record."""
        self._buffer.append(record)

    def checkpoint(self, lines_done: int) -> None:
        """Call after finishing input line ``lines_done``; writes a part when the buffer is full."""
        if len(self._buffer) >= self.every:
            self._flush(lines_done)

    def close(self, lines_done: int) -> list[Path]:
        """Flush the remainder, mark the stream finished, and return the part files in input
        order. Read them one at a time: together they can exceed Colab's memory (``books_raw`` is
        about 1.7 GB on disk)."""
        if self._buffer:
            self._flush(lines_done)
        self.resume_lines, self.finished = max(self.resume_lines, lines_done), True
        self._save()
        return [self._part(i) for i in range(self._parts)]

    def _flush(self, lines_done: int) -> None:
        path = self._part(self._parts)
        tmp = path.with_name(path.name + ".tmp")
        pd.DataFrame(self._buffer).to_parquet(tmp, index=False, compression="zstd")
        os.replace(tmp, path)
        self._parts += 1
        self._buffer.clear()
        self.resume_lines = lines_done
        self._history.append({"lines": lines_done, "counters": dict(self.counters)})
        self._save()

    def _save(self) -> None:
        state = {
            "lines": self.resume_lines,
            "parts": self._parts,
            "counters": dict(self.counters),
            "finished": self.finished,
            "history": self._history,
        }
        write_json_atomic(self.work_dir / "progress.json", state)

    def _part(self, i: int) -> Path:
        return self.work_dir / f"part-{i:05d}.parquet"
