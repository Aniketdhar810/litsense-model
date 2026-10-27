"""Stage data.kcore_split (CLAUDE.md s7.7): the frozen dataset.

1. Interactions = 5-core ratings of editions in kept works, collapsed to one per (user, work): the
   latest rating when a reader rated two editions of one work. ``has_text`` = the reader's review of
   that work survived ``data.clean`` with ``text_ok``.
2. k-core (``data.kcore``), then seeded user sampling to ``data.target_interactions[version]``
   (``ml/data/kcore.py``; the fraction is recorded in the manifest and in docs/decisions.md).
3. Global temporal split at the ``data.split_quantiles`` of the timestamps.
4. ``id_map.json``: sorted user ids and canonical ids. It is created here once; every later stage
   loads it and nothing else builds ids (s2.2 bug 2).
"""

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ml.common.errors import DataError
from ml.common.stage import StageContext, stage
from ml.data.clean import CLEAN_SCHEMA
from ml.data.kcore import check_kcore, kcore, size_to_target
from ml.data.meta import PARQUET_COMPRESSION

INTERACTION_SCHEMA = pa.schema(
    [
        ("u", pa.int32()),
        ("i", pa.int32()),
        ("user_id", pa.string()),
        ("canonical_id", pa.string()),
        ("rating", pa.float32()),
        ("ts", pa.int64()),  # ms
        ("split", pa.string()),  # train | val | test
        ("has_text", pa.bool_()),
    ]
)
REVIEW_OUT_SCHEMA = pa.schema(
    [
        ("review_row", pa.int64()),
        ("u", pa.int32()),
        ("i", pa.int32()),
        ("rating", pa.float32()),
        ("ts", pa.int64()),
        ("title", pa.string()),
        ("text", pa.string()),
        ("helpful_vote", pa.int32()),
        ("verified", pa.bool_()),
        ("text_ok", pa.bool_()),
        ("text_drop_reason", pa.string()),
    ]
)
BOOK_COLUMNS = [
    "i", "canonical_id", "title", "author", "author_key", "description", "features", "genres",
    "primary_genre", "categories_raw", "slices", "primary_slice", "year", "pages", "image",
    "isbn13", "member_asins", "edition_kinds", "series_id", "series_name", "series_pos",
    "gr_matched", "n_members",
]  # fmt: skip
SPLITS = ("train", "val", "test")


def build_interactions(
    ratings: pd.DataFrame, editions: Mapping[str, str], text_pairs: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    """One interaction per (user, kept work) with ``has_text``; returns the frame and counts."""
    counts = {"ratings_5core": len(ratings)}
    ratings = ratings[ratings["parent_asin"].isin(list(editions))].copy()
    counts["ratings_of_kept_works"] = len(ratings)
    ratings["canonical_id"] = ratings["parent_asin"].map(editions)
    ratings = ratings.sort_values("timestamp", ascending=False, kind="stable")
    ratings = ratings.drop_duplicates(["user_id", "canonical_id"])
    counts["interactions_one_per_work"] = len(ratings)
    frame = pd.DataFrame(
        {
            "user_id": ratings["user_id"].astype(str),
            "canonical_id": ratings["canonical_id"].astype(str),
            "rating": ratings["rating"].astype("float32"),
            "ts": ratings["timestamp"].astype("int64"),
        }
    )
    ok = text_pairs[text_pairs["text_ok"]].drop_duplicates(["user_id", "canonical_id"])
    keys = pd.MultiIndex.from_frame(frame[["user_id", "canonical_id"]])
    frame["has_text"] = keys.isin(pd.MultiIndex.from_frame(ok[["user_id", "canonical_id"]]))
    return frame.sort_values(["user_id", "ts"], kind="stable").reset_index(drop=True), counts


def temporal_split(df: pd.DataFrame, quantiles: Sequence[float]) -> tuple[pd.Series, int, int]:
    """``train`` (ts < t1), ``val`` (t1 <= ts < t2), ``test`` (ts >= t2) at the given quantiles."""
    t1, t2 = (int(v) for v in np.quantile(df["ts"].to_numpy(), list(quantiles)))
    split = np.where(df["ts"] < t1, "train", np.where(df["ts"] < t2, "val", "test"))
    return pd.Series(split, index=df.index), t1, t2


def split_report(df: pd.DataFrame, t1: int, t2: int) -> dict[str, Any]:
    """``splits.json``: boundaries, sizes, and how many val/test users and items are cold."""

    def iso(ms: int) -> str:
        return datetime.fromtimestamp(ms / 1000, UTC).isoformat(timespec="seconds")

    train = df[df["split"] == "train"]
    report: dict[str, Any] = {
        "t1": {"ms": t1, "iso": iso(t1)},
        "t2": {"ms": t2, "iso": iso(t2)},
        "splits": {},
    }
    for name in SPLITS:
        part = df[df["split"] == name]
        entry = {
            "interactions": len(part),
            "users": int(part["u"].nunique()),
            "items": int(part["i"].nunique()),
            "share_relevant": round(float((part["rating"] >= 4).mean()), 4) if len(part) else None,
        }
        if name != "train":
            users, items = part["u"].unique(), part["i"].unique()
            entry["users_without_train"] = round(float(np.mean(~np.isin(users, train["u"]))), 4)
            entry["items_without_train"] = round(float(np.mean(~np.isin(items, train["i"]))), 4)
        report["splits"][name] = entry
    return report


def slice_mix(df: pd.DataFrame, books: pd.DataFrame) -> dict[str, dict[str, dict[str, int]]]:
    """Interactions and works per primary genre slice and per primary genre (the EDA's "how much
    SF & Fantasy survives" view at each step)."""
    meta = books.set_index("canonical_id")[["primary_slice", "primary_genre"]]
    joined = df[["canonical_id"]].join(meta, on="canonical_id")
    mix: dict[str, dict[str, dict[str, int]]] = {}
    for key in ("primary_slice", "primary_genre"):
        grouped = joined.groupby(key)["canonical_id"].agg(["size", "nunique"])
        mix[key] = {
            str(k): {"interactions": int(r["size"]), "works": int(r["nunique"])}
            for k, r in grouped.sort_values("size", ascending=False).iterrows()
        }
    return mix


def check_chronology(df: pd.DataFrame, t1: int, t2: int) -> None:
    """max(train.ts) < t1 <= min(val.ts) and max(val.ts) < t2 <= min(test.ts) (s16.2)."""
    by = df.groupby("split")["ts"].agg(["min", "max"])
    ok = (
        ("train" not in by.index or by.loc["train", "max"] < t1)
        and ("val" not in by.index or (t1 <= by.loc["val", "min"] and by.loc["val", "max"] < t2))
        and ("test" not in by.index or t2 <= by.loc["test", "min"])
    )
    if not ok:
        raise DataError("temporal split chronology violated")


def write_reviews(
    source: pq.ParquetFile, ids: pd.DataFrame, path: Any, batch_rows: int = 100_000
) -> int:
    """Reviews of the final interactions with integer ids, in record batches."""
    written = 0
    with pq.ParquetWriter(path, REVIEW_OUT_SCHEMA, compression=PARQUET_COMPRESSION) as out:
        for batch in source.iter_batches(batch_size=batch_rows, columns=CLEAN_SCHEMA.names):
            frame = batch.to_pandas().merge(ids, on=["user_id", "canonical_id"], how="inner")
            frame["review_row"] = np.arange(written, written + len(frame), dtype=np.int64)
            written += len(frame)
            frame = frame.reindex(columns=REVIEW_OUT_SCHEMA.names)
            out.write_table(
                pa.Table.from_pandas(frame, schema=REVIEW_OUT_SCHEMA, preserve_index=False)
            )
    return written


def write_parquet(frame: pd.DataFrame, schema: pa.Schema | None, path: Any) -> None:
    """zstd parquet with an optional explicit schema."""
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
    pq.write_table(table, path, compression=PARQUET_COMPRESSION)


@stage(
    "data.kcore_split",
    inputs=[
        "data:ratings_5core.csv.gz",
        "data:asin_to_canonical.parquet",
        "data:books_canonical.parquet",
        "data:reviews_clean.parquet",
    ],
    outputs=[
        "data:interactions.parquet",
        "data:books.parquet",
        "data:reviews.parquet",
        "data:id_map.json",
        "data:splits.json",
    ],
    config_keys=[
        "project.data_version",
        "data.kcore",
        "data.target_interactions",
        "data.size_tolerance",
        "data.split_quantiles",
    ],
)
def build_kcore_split(ctx: StageContext) -> None:
    """Interactions, k-core, sizing, temporal split, id map, and the dataset tables."""
    mapping = pd.read_parquet(ctx.path("data:asin_to_canonical.parquet"))
    mapping = mapping[mapping["status"] == "kept"]
    editions = dict(zip(mapping["parent_asin"], mapping["canonical_id"], strict=True))
    ratings = pd.read_csv(
        ctx.path("data:ratings_5core.csv.gz"),
        dtype={"user_id": "string[pyarrow]", "parent_asin": "string[pyarrow]"},
        engine="pyarrow",
    )
    reviews_path = ctx.path("data:reviews_clean.parquet")
    text_pairs = pd.read_parquet(reviews_path, columns=["user_id", "canonical_id", "text_ok"])
    df, counts = build_interactions(ratings, editions, text_pairs)
    del ratings

    kc = ctx.param("data.kcore")
    ku, ki = int(kc["user"]), int(kc["item"])
    all_books = pd.read_parquet(ctx.path("data:books_canonical.parquet"))
    df = kcore(df, ku, ki)
    counts["after_kcore"] = len(df)
    kcore_mix = slice_mix(df, all_books)
    target = int(ctx.param("data.target_interactions")[ctx.param("project.data_version")])
    df, sizing = size_to_target(
        df, ku, ki, target, float(ctx.param("data.size_tolerance")), ctx.rng
    )
    check_kcore(df, ku, ki)
    counts["after_sizing"] = len(df)
    ctx.notes["sizing"] = {
        "target": target,
        "user_fraction": sizing.fraction,
        "steps": sizing.steps,
    }

    users = sorted(df["user_id"].unique())
    items = sorted(df["canonical_id"].unique())
    df["u"] = df["user_id"].map({k: n for n, k in enumerate(users)}).astype("int32")
    df["i"] = df["canonical_id"].map({k: n for n, k in enumerate(items)}).astype("int32")
    split, t1, t2 = temporal_split(df, ctx.param("data.split_quantiles"))
    df["split"] = split
    check_chronology(df, t1, t2)

    df = df.sort_values(["u", "ts"], kind="stable").reset_index(drop=True)
    write_parquet(
        df[INTERACTION_SCHEMA.names], INTERACTION_SCHEMA, ctx.out("data:interactions.parquet")
    )
    books = all_books[all_books["canonical_id"].isin(items)].copy()
    books["i"] = books["canonical_id"].map({k: n for n, k in enumerate(items)}).astype("int32")
    write_parquet(
        books.sort_values("i").reindex(columns=BOOK_COLUMNS), None, ctx.out("data:books.parquet")
    )
    ids = df[["user_id", "canonical_id", "u", "i"]]
    n_reviews = write_reviews(pq.ParquetFile(reviews_path), ids, ctx.out("data:reviews.parquet"))

    id_map = {"users": users, "items": items}
    ctx.out("data:id_map.json").write_text(json.dumps(id_map) + "\n", encoding="utf-8")
    report = split_report(df, t1, t2) | {"sizing": ctx.notes["sizing"], "attrition": counts}
    report["mix"] = {"after_kcore": kcore_mix, "after_sizing": slice_mix(df, all_books)}
    path = ctx.out("data:splits.json")
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for key, value in counts.items():
        ctx.count(key, value)
    ctx.count("users", len(users))
    ctx.count("items", len(items))
    ctx.count("reviews", n_reviews)
    ctx.count("interactions_with_text", int(df["has_text"].sum()))
    ctx.notes["splits"] = {s: report["splits"][s]["interactions"] for s in SPLITS}
