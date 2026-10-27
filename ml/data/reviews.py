"""Stage data.reviews (CLAUDE.md s7.5): review text for the 5-core ratings of kept works.

The reviews file (6.22 GB, 29.5M lines) is streamed from its URL and never stored (D18). A line is
JSON-parsed only when its ``(user_id, parent_asin)`` is a 5-core rating of an edition whose work
``data.canonical`` kept; both ids are read from the raw line first. Matches are checkpointed with
``PartWriter``, so a restarted run resumes after the last checkpoint.

Duplicates are then removed in two passes over the parts (the text never sits in memory at once):

1. one review per ``(user_id, canonical_id)``, the latest: the same review posted on the hardcover
   and the paperback, or on two editions that ``data.canonical`` merged;
2. one review per ``(user_id, text_hash)``: the same text pasted on two works that stayed apart.

5-core ratings without a review keep their rating; ``data.kcore_split`` builds interactions from the
5-core file and marks ``has_text`` from this table.
"""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ml.common.errors import DataError
from ml.common.log import Progress
from ml.common.stage import StageContext, stage
from ml.data.download import TS_MS_RANGE
from ml.data.meta import PARQUET_COMPRESSION
from ml.data.parsing import extract_parent_asin, json_str_field
from ml.data.stream import PartWriter, StreamSettings, StreamStats, iter_lines

REVIEW_SCHEMA = pa.schema(
    [
        ("user_id", pa.string()),
        ("parent_asin", pa.string()),
        ("canonical_id", pa.string()),
        ("rating", pa.float32()),
        ("ts", pa.int64()),  # ms
        ("title", pa.string()),
        ("text", pa.string()),
        ("helpful_vote", pa.int32()),
        ("verified", pa.bool_()),
        ("text_hash", pa.string()),  # sha1 of the normalized text; None when it has no words
    ]
)
KEY_COLUMNS = ["user_id", "canonical_id", "ts", "text_hash"]
_TAG_RE = re.compile(r"<[^>]+>")


def text_hash(text: str | None) -> str | None:
    """Short sha1 of a review text with tags, case and punctuation removed (catches copy-paste
    across editions); None for a text without words."""
    norm = " ".join(re.sub(r"[\W_]+", " ", _TAG_RE.sub(" ", text or "").lower()).split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16] if norm else None


def pair_key(user_id: str, parent_asin: str) -> str:
    """Set key of a (user, edition) rating."""
    return f"{user_id}\t{parent_asin}"


def load_pairs(ratings: Path, kept: Mapping[str, str]) -> set[str]:
    """5-core ``(user_id, parent_asin)`` pairs whose edition belongs to a kept work."""
    frame = pd.read_csv(
        ratings, usecols=["user_id", "parent_asin"], dtype="string[pyarrow]", engine="pyarrow"
    )
    frame = frame[frame["parent_asin"].isin(list(kept))]
    return {pair_key(u, a) for u, a in zip(frame["user_id"], frame["parent_asin"], strict=True)}


def review_row(record: Mapping[str, Any], canonical_id: str) -> dict[str, Any]:
    """One ``reviews_raw`` row from a review record (joined on ``parent_asin``, never ``asin``)."""
    ts = int(record["timestamp"])
    if not TS_MS_RANGE[0] < ts < TS_MS_RANGE[1]:
        raise DataError(f"review timestamp {ts} is not in milliseconds")
    text = record.get("text") or ""
    return {
        "user_id": record["user_id"],
        "parent_asin": record["parent_asin"],
        "canonical_id": canonical_id,
        "rating": float(record["rating"]),
        "ts": ts,
        "title": record.get("title") or None,
        "text": text,
        "helpful_vote": int(record.get("helpful_vote") or 0),
        "verified": bool(record.get("verified_purchase")),
        "text_hash": text_hash(text),
    }


def dedupe(keys: pd.DataFrame) -> tuple[pd.Index, dict[str, int]]:
    """Rows to keep from ``keys`` (``KEY_COLUMNS``, indexed by global row number): the latest
    review per (user, work), then one per (user, text). Returns the kept index and drop counts."""
    latest = keys.sort_values("ts", ascending=False, kind="stable")
    one_per_work = latest[~latest.duplicated(["user_id", "canonical_id"])]
    hashed = one_per_work["text_hash"].notna()
    copies = hashed & one_per_work.duplicated(["user_id", "text_hash"])
    kept = one_per_work[~copies]
    counts = {
        "dropped_same_work": len(keys) - len(one_per_work),
        "dropped_copied_text": int(copies.sum()),
    }
    return kept.index.sort_values(), counts


def write_reviews(parts: Sequence[Path], path: Path) -> dict[str, int]:
    """Deduplicate the checkpoint parts into one parquet file, one part in memory at a time."""
    columns = [*KEY_COLUMNS, "parent_asin"]
    frames = [pd.read_parquet(p, columns=columns) for p in parts]
    keys = pd.concat(frames or [pd.DataFrame(columns=columns)], ignore_index=True)
    kept, counts = dedupe(keys)
    counts["pairs_with_review"] = len(keys.drop_duplicates(["user_id", "parent_asin"]))
    keep = pd.Series(False, index=keys.index)
    keep[kept] = True
    offset = 0
    with pq.ParquetWriter(path, REVIEW_SCHEMA, compression=PARQUET_COMPRESSION) as out:
        for part in parts:
            frame = pd.read_parquet(part)
            mask = keep.iloc[offset : offset + len(frame)].to_numpy()
            offset += len(frame)
            frame = frame[mask].reindex(columns=REVIEW_SCHEMA.names)
            out.write_table(pa.Table.from_pandas(frame, schema=REVIEW_SCHEMA, preserve_index=False))
    counts["reviews"] = len(kept)
    return counts


@stage(
    "data.reviews",
    inputs=["data:ratings_5core.csv.gz", "data:asin_to_canonical.parquet", "url:reviews_books"],
    outputs=["data:reviews_raw.parquet"],
    config_keys=["data.stream"],
)
def build_reviews(ctx: StageContext) -> None:
    """Stream the reviews and keep those of 5-core ratings of kept works, deduplicated."""
    mapping = pd.read_parquet(ctx.path("data:asin_to_canonical.parquet"))
    mapping = mapping[mapping["status"] == "kept"]
    kept = dict(zip(mapping["parent_asin"], mapping["canonical_id"], strict=True))
    pairs = load_pairs(ctx.path("data:ratings_5core.csv.gz"), kept)
    ctx.count("5core_pairs_of_kept_works", len(pairs))

    stream = ctx.param("data.stream")
    source = ctx.source("url:reviews_books")
    stats = StreamStats(source)
    writer = PartWriter(ctx.work_dir, every=int(stream["checkpoint_rows"]))
    if writer.rolled_back is not None:
        ctx.log.warning(
            "checkpoint parts were missing (Drive lost them); rolled back from %d parts to the "
            "last good checkpoint at line %d",
            writer.rolled_back,
            writer.resume_lines,
        )
    if writer.finished:
        ctx.log.info("stream already finished (%d lines); deduplicating only", writer.resume_lines)
    elif writer.resume_lines:
        ctx.log.info("resuming after line %d from checkpoint", writer.resume_lines)
    streamed = not writer.finished
    if streamed:
        settings = StreamSettings.from_config(stream)
        lines = iter_lines(source, settings, limit=ctx.sample, stats=stats)
        progress = Progress(ctx.log, "reviews")
        for n, line in enumerate(lines, start=1):
            progress.update(
                stats.bytes_read,
                stats.size,
                f"; {n:,} lines, {writer.counters['matched']:,} matched",
            )
            if n <= writer.resume_lines:
                continue
            asin = extract_parent_asin(line)
            user = json_str_field(line, "user_id", last=True)
            if asin is not None and user is not None and pair_key(user, asin) in pairs:
                writer.counters["matched"] += 1
                writer.add(review_row(json.loads(line), kept[asin]))
            writer.checkpoint(n)
    n_pairs = len(pairs)
    del pairs  # about 0.6 GB on the full run; the dedup below needs the memory
    parts = writer.close(stats.lines if not writer.finished else writer.resume_lines)
    counts = write_reviews(parts, ctx.out("data:reviews_raw.parquet"))
    ctx.count("review_lines_read", writer.resume_lines)
    ctx.count("reviews_matched", writer.counters["matched"])
    for key, value in counts.items():
        ctx.count(key, value)
    ctx.count("pairs_without_review", n_pairs - counts["pairs_with_review"])
    # a rerun that reused a finished stream has no stream statistics of its own
    ctx.notes["source"] = stats.as_note() if streamed else {"reused_finished_stream": True}
