"""Stages data.download and data.lid: the two raw files the pipeline keeps (CLAUDE.md s7.1).

data.download fetches and validates the 5-core ratings CSV (D19); data.lid keeps fastText's
language-ID model ``lid.176.ftz`` (0.9 MB) for data.clean (D30).

The only raw file the pipeline keeps: 0.14 GB, read by every filter, stored under ``data/`` so it
survives Colab restarts. It is always fetched whole, even in ``--sample`` runs, because it is the
item list everything else is filtered against. The large sources are streamed by the stages that
read them (``url:`` inputs) and never stored.
"""

import struct
from collections import Counter
from datetime import UTC, datetime

from ml.common.errors import DataError
from ml.common.stage import StageContext, stage
from ml.data.stream import StreamSettings, StreamStats, iter_bytes, iter_lines

RATINGS_HEADER = ["user_id", "parent_asin", "rating", "timestamp"]
TS_MS_RANGE = (8e11, 1.8e12)  # CLAUDE.md s6.1: timestamps are ms (1995-2027); seconds fail this
VALID_STARS = {"1.0", "2.0", "3.0", "4.0", "5.0"}
FASTTEXT_MAGIC = struct.pack("<i", 793712314)  # first 4 bytes of every fastText model file


@stage(
    "data.download",
    inputs=["url:ratings_5core"],
    outputs=["data:ratings_5core.csv.gz"],
    config_keys=["data.stream"],
)
def download_ratings(ctx: StageContext) -> None:
    """Copy the 5-core CSV byte for byte while checking its header, stars and ms timestamps."""
    url = ctx.source("url:ratings_5core", full=True)
    stats = StreamStats(url)
    users: set[str] = set()
    items: set[str] = set()
    stars: Counter[str] = Counter()
    ts_min, ts_max = float("inf"), float("-inf")
    with open(ctx.out("data:ratings_5core.csv.gz"), "wb") as out:
        lines = iter_lines(
            url, StreamSettings.from_config(ctx.param("data.stream")), stats=stats, tee=out
        )
        header = next(lines, "").split(",")
        if header != RATINGS_HEADER:
            raise DataError(f"5-core header is {header}, expected {RATINGS_HEADER}")
        for n, line in enumerate(lines, start=2):
            fields = line.split(",")
            if len(fields) != 4 or fields[2] not in VALID_STARS:
                raise DataError(f"5-core line {n} is malformed: {line[:120]!r}")
            ts = int(fields[3])
            if not TS_MS_RANGE[0] < ts < TS_MS_RANGE[1]:
                raise DataError(f"5-core line {n}: timestamp {ts} is not in milliseconds")
            users.add(fields[0])
            items.add(fields[1])
            stars[fields[2]] += 1
            ts_min, ts_max = min(ts_min, ts), max(ts_max, ts)
    ctx.count("ratings", stats.lines - 1)
    ctx.count("users", len(users))
    ctx.count("items", len(items))
    ctx.notes["source"] = stats.as_note()
    ctx.notes["stars"] = dict(sorted(stars.items()))
    ctx.notes["time_range"] = [_iso(ts_min), _iso(ts_max)] if items else []


@stage(
    "data.lid",
    inputs=["url:fasttext_lid"],
    outputs=["data:lid.176.ftz"],
    config_keys=["data.stream"],
)
def download_lid(ctx: StageContext) -> None:
    """Copy fastText's language-ID model and check that it is a fastText file."""
    url = ctx.source("url:fasttext_lid", full=True)
    stats = StreamStats(url)
    path = ctx.out("data:lid.176.ftz")
    with open(path, "wb") as out:
        for chunk in iter_bytes(url, StreamSettings.from_config(ctx.param("data.stream")), stats):
            out.write(chunk)
    with open(path, "rb") as f:
        if f.read(4) != FASTTEXT_MAGIC:
            raise DataError(f"{url} is not a fastText model (bad magic number)")
    ctx.count("bytes", stats.bytes_read)
    ctx.notes["source"] = stats.as_note()


def _iso(ms: float) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat(timespec="seconds")
