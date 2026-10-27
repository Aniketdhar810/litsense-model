"""Streaming reader: gunzip on the fly, resume after a dropped connection, detect truncation."""

import gzip
import io
import json
from pathlib import Path

import pandas as pd
import pytest
import requests

from ml.common.errors import StreamError
from ml.data.stream import PartWriter, StreamSettings, StreamStats, iter_lines
from tests.ml.data.conftest import FileServer

SETTINGS = StreamSettings(chunk_bytes=64, retries=3, timeout_s=5, backoff_s=0.0)
LINES = [json.dumps({"parent_asin": f"B{i:09d}", "title": f"Book {i}"}) for i in range(400)]


def write_gz(path: Path, lines: list[str], *, members: int = 1) -> bytes:
    """Gzip ``lines`` into ``members`` concatenated gzip members; returns the bytes."""
    blob = b""
    size = -(-len(lines) // members)
    for start in range(0, len(lines), size):
        out = io.BytesIO()
        with gzip.GzipFile(fileobj=out, mode="wb", mtime=0) as gz:
            gz.write(("\n".join(lines[start : start + size]) + "\n").encode())
        blob += out.getvalue()
    path.write_bytes(blob)
    return blob


def test_url_and_local_streams_match(file_server: FileServer) -> None:
    blob = write_gz(file_server.root / "meta.jsonl.gz", LINES)
    local = StreamStats("local")
    assert list(iter_lines(str(file_server.root / "meta.jsonl.gz"), SETTINGS, stats=local)) == LINES
    remote = StreamStats("remote")
    assert list(iter_lines(file_server.url("meta.jsonl.gz"), SETTINGS, stats=remote)) == LINES
    for stats in (local, remote):
        assert stats.complete and stats.lines == len(LINES)
        assert stats.size == stats.bytes_read == len(blob)
    assert remote.last_modified == "Mon, 12 Oct 2026 00:00:00 GMT" and remote.reconnects == 0


def test_plain_text_and_missing_final_newline(tmp_path: Path) -> None:
    (tmp_path / "r.csv").write_text("a,b\n1,2\n3,4")
    assert list(iter_lines(str(tmp_path / "r.csv"), SETTINGS)) == ["a,b", "1,2", "3,4"]


def test_concatenated_gzip_members(tmp_path: Path) -> None:
    write_gz(tmp_path / "multi.gz", LINES, members=3)
    assert list(iter_lines(str(tmp_path / "multi.gz"), SETTINGS)) == LINES


def test_limit_stops_early(file_server: FileServer) -> None:
    write_gz(file_server.root / "meta.jsonl.gz", LINES)
    stats = StreamStats("remote")
    assert (
        list(iter_lines(file_server.url("meta.jsonl.gz"), SETTINGS, limit=10, stats=stats))
        == (LINES[:10])
    )
    assert stats.lines == 10 and not stats.complete
    assert len(file_server.requests) == 1


def test_dropped_connection_resumes_with_range(file_server: FileServer) -> None:
    write_gz(file_server.root / "meta.jsonl.gz", LINES)
    file_server.drops["meta.jsonl.gz"] = 300
    stats = StreamStats("remote")
    assert list(iter_lines(file_server.url("meta.jsonl.gz"), SETTINGS, stats=stats)) == LINES
    assert stats.reconnects == 1 and stats.complete
    assert file_server.requests == [("meta.jsonl.gz", None), ("meta.jsonl.gz", "bytes=300-")]


def test_server_that_ignores_range_cannot_resume(file_server: FileServer) -> None:
    write_gz(file_server.root / "meta.jsonl.gz", LINES)
    file_server.drops["meta.jsonl.gz"] = 300
    file_server.ignore_range = True
    with pytest.raises(StreamError, match="cannot resume at byte 300"):
        list(iter_lines(file_server.url("meta.jsonl.gz"), SETTINGS))


def test_truncated_gzip_is_detected(file_server: FileServer) -> None:
    blob = write_gz(file_server.root / "meta.jsonl.gz", LINES)
    (file_server.root / "cut.jsonl.gz").write_bytes(blob[:-20])
    with pytest.raises(StreamError, match="ended early"):
        list(iter_lines(file_server.url("cut.jsonl.gz"), SETTINGS))
    with pytest.raises(StreamError, match="ended early"):
        list(iter_lines(str(file_server.root / "cut.jsonl.gz"), SETTINGS))


def test_http_errors_and_unreachable_hosts(
    file_server: FileServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(StreamError, match="HTTP 404"):
        list(iter_lines(file_server.url("missing.gz"), SETTINGS))
    attempts: list[str] = []

    def refuse(self: requests.Session, url: str, **kwargs: object) -> None:
        attempts.append(url)
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests.Session, "get", refuse)
    with pytest.raises(StreamError, match="gave up after 3 retries"):
        list(iter_lines("http://unreachable.invalid/x.gz", SETTINGS))
    assert len(attempts) == 4  # first try + 3 retries


def read_parts(parts: list[Path], columns: list[str]) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)[columns]


def process(writer: PartWriter, lines: list[str], crash_at: int | None = None) -> None:
    """A stage-like loop: keep even ids, count odd ones, checkpoint after every line."""
    for n, line in enumerate(lines, start=1):
        if n <= writer.resume_lines:
            continue
        if n == crash_at:
            raise RuntimeError("colab disconnected")
        record = json.loads(line)
        if int(record["parent_asin"][1:]) % 2:
            writer.counters["odd"] += 1
        else:
            writer.add(record)
        writer.checkpoint(n)


def test_part_writer_resumes_where_it_stopped(tmp_path: Path) -> None:
    columns = ["parent_asin", "title"]
    clean = PartWriter(tmp_path / "clean", every=10)
    (tmp_path / "clean").mkdir()
    process(clean, LINES)
    expected = read_parts(clean.close(len(LINES)), columns)

    work = tmp_path / "crashy"
    work.mkdir()
    first = PartWriter(work, every=10)
    with pytest.raises(RuntimeError):
        process(first, LINES, crash_at=95)
    resumed = PartWriter(work, every=10)
    assert resumed.resume_lines == 79  # the 4th part (ids 60-78) was the last one written
    assert resumed.counters["odd"] == 39
    process(resumed, LINES)
    result = read_parts(resumed.close(len(LINES)), columns)
    assert result.equals(expected) and resumed.counters == clean.counters


def test_part_writer_drops_parts_written_after_the_last_checkpoint(tmp_path: Path) -> None:
    writer = PartWriter(tmp_path, every=1)
    writer.add({"parent_asin": "a", "title": "x"})
    writer.checkpoint(1)
    (tmp_path / "part-00001.parquet").write_bytes(b"half-written")
    assert PartWriter(tmp_path, every=1).close(1) == [tmp_path / "part-00000.parquet"]
    assert not (tmp_path / "part-00001.parquet").exists()
    empty = PartWriter(tmp_path / "none", every=5)
    assert empty.close(0) == []


def test_part_writer_rolls_back_to_the_last_part_on_disk(tmp_path: Path) -> None:
    """Drive can lose a part while progress.json survives (Colab run, 2026-10-14)."""
    columns = ["parent_asin", "title"]
    clean = PartWriter(tmp_path / "clean", every=10)
    (tmp_path / "clean").mkdir()
    process(clean, LINES)
    expected = read_parts(clean.close(len(LINES)), columns)

    work = tmp_path / "lossy"
    work.mkdir()
    first = PartWriter(work, every=10)
    process(first, LINES)
    written = len(first.close(len(LINES)))
    (work / "part-00002.parquet").unlink()  # lost by the Drive mount
    resumed = PartWriter(work, every=10)
    assert resumed.rolled_back == written and not resumed.finished
    assert resumed.resume_lines == 39  # the end of part 1 (ids 20-38)
    assert resumed.counters["odd"] == 19
    assert not (work / "part-00003.parquet").exists()
    process(resumed, LINES)
    result = read_parts(resumed.close(len(LINES)), columns)
    assert result.equals(expected) and resumed.counters == clean.counters


def test_part_writer_without_history_restarts(tmp_path: Path) -> None:
    writer = PartWriter(tmp_path, every=1)
    writer.add({"parent_asin": "a", "title": "x"})
    writer.checkpoint(1)
    state = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    state["parts"] = 2  # an old-format checkpoint that names a part Drive never wrote
    del state["history"]
    (tmp_path / "progress.json").write_text(json.dumps(state), encoding="utf-8")
    restarted = PartWriter(tmp_path, every=1)
    assert restarted.rolled_back == 2 and restarted.resume_lines == 0
    assert restarted.close(0) == []


def test_finished_stream_is_not_streamed_again(tmp_path: Path) -> None:
    writer = PartWriter(tmp_path, every=10)
    process(writer, LINES)
    parts = writer.close(len(LINES))
    again = PartWriter(tmp_path, every=10)
    assert again.finished and again.resume_lines == len(LINES)
    assert again.close(again.resume_lines) == parts
