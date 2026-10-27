"""data.kcore_split: k-core invariants, sizing, temporal chronology, one stable id map
(CLAUDE.md s7.7, s16.2)."""

import json
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

import ml.data.canonical  # noqa: F401  (registers the stage)
import ml.data.clean as clean
import ml.data.download  # noqa: F401  (registers the stage)
import ml.data.eda  # noqa: F401  (registers the stage)
import ml.data.goodreads  # noqa: F401  (registers the stage)
import ml.data.meta  # noqa: F401  (registers the stage)
import ml.data.reviews  # noqa: F401  (registers the stage)
import ml.data.split as split
from ml.common.errors import DataError
from ml.common.manifest import read_manifest
from ml.common.stage import RunPaths, run_all, run_stage
from ml.data.kcore import check_kcore, kcore, size_to_target

Config = dict[str, Any]


def random_interactions(n_users: int, n_items: int, n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pop = 1 / np.arange(1, n_items + 1) ** 0.6
    frame = pd.DataFrame(
        {
            "user_id": [f"u{x}" for x in rng.integers(0, n_users, n)],
            "canonical_id": [f"w{x}" for x in rng.choice(n_items, n, p=pop / pop.sum())],
            "ts": rng.integers(1_500_000_000_000, 1_600_000_000_000, n),
        }
    )
    return frame.drop_duplicates(["user_id", "canonical_id"]).reset_index(drop=True)


def test_kcore_invariants_hold() -> None:  # s16.2
    df = kcore(random_interactions(400, 300, 6000), 5, 10)
    assert len(df) > 0
    check_kcore(df, 5, 10)
    assert df.groupby("user_id").size().min() >= 5
    assert df.groupby("canonical_id").size().min() >= 10
    with pytest.raises(DataError, match="k-core"):
        check_kcore(random_interactions(50, 300, 200), 5, 10)


def test_sizing_reaches_the_target_reproducibly() -> None:
    df = kcore(random_interactions(3000, 500, 60000), 5, 10)
    sized, info = size_to_target(df, 5, 10, 10000, 0.15, np.random.default_rng(42))
    assert abs(len(sized) - 10000) <= 1500 and 0 < info.fraction < 1
    check_kcore(sized, 5, 10)
    again, info2 = size_to_target(df, 5, 10, 10000, 0.15, np.random.default_rng(42))
    assert info2 == info and again.equals(sized)
    same, info3 = size_to_target(df, 5, 10, len(df), 0.15, np.random.default_rng(0))
    assert info3.fraction == 1.0 and len(same) == len(df)


def test_temporal_split_chronology() -> None:  # s16.2
    df = random_interactions(200, 100, 3000)
    df["split"], t1, t2 = split.temporal_split(df, [0.85, 0.925])
    split.check_chronology(df, t1, t2)
    assert (
        df.loc[df["split"] == "train", "ts"].max() < t1 <= df.loc[df["split"] == "val", "ts"].min()
    )
    assert (
        df.loc[df["split"] == "val", "ts"].max() < t2 <= df.loc[df["split"] == "test", "ts"].min()
    )
    df.loc[df.index[0], "split"] = "test"
    df.loc[df.index[0], "ts"] = t1 - 1
    with pytest.raises(DataError, match="chronology"):
        split.check_chronology(df, t1, t2)


def test_one_interaction_per_work_with_text_flag() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2"],
            "parent_asin": ["hard", "paper", "other", "hard"],
            "rating": [3.0, 5.0, 4.0, 4.0],
            "timestamp": [100, 200, 150, 120],
        }
    )
    editions = {"hard": "w1", "paper": "w1"}  # "other" belongs to no kept work
    texts = pd.DataFrame(
        {"user_id": ["u1", "u2"], "canonical_id": ["w1", "w1"], "text_ok": [True, False]}
    )
    frame, counts = split.build_interactions(ratings, editions, texts)
    assert counts == {
        "ratings_5core": 4,
        "ratings_of_kept_works": 3,
        "interactions_one_per_work": 2,
    }
    u1 = frame[frame["user_id"] == "u1"].iloc[0]
    assert (u1["rating"], u1["ts"], u1["has_text"]) == (5.0, 200, True)  # the latest edition rating
    assert not frame.loc[frame["user_id"] == "u2", "has_text"].iloc[0]


@pytest.fixture
def small_core(tiny_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> Config:
    """The tiny fixture is too small for a 5/10 core; use 2/2 and no sizing."""
    monkeypatch.setattr(clean, "load_language_id", lambda path: lambda text: ("en", 0.9))
    tiny_cfg["data"]["kcore"] = {"user": 2, "item": 2}
    tiny_cfg["data"]["target_interactions"] = {"v0": 100000, "v1": 250000}
    return tiny_cfg


def test_stage_on_the_tiny_fixture(small_core: Config) -> None:
    run_all(small_core, until="data.kcore_split")
    paths = RunPaths.from_config(small_core, None)
    inter = pq.read_table(paths.resolve("data:interactions.parquet"))
    assert inter.schema.equals(split.INTERACTION_SCHEMA)
    df = inter.to_pandas()
    id_map = json.loads(paths.resolve("data:id_map.json").read_text(encoding="utf-8"))
    assert id_map["users"] == sorted(id_map["users"]) and id_map["items"] == sorted(id_map["items"])
    assert [id_map["users"][u] for u in df["u"]] == list(df["user_id"])
    assert [id_map["items"][i] for i in df["i"]] == list(df["canonical_id"])
    assert not df.duplicated(["u", "i"]).any()
    check_kcore(df, 2, 2)
    report = json.loads(paths.resolve("data:splits.json").read_text(encoding="utf-8"))
    split.check_chronology(df, report["t1"]["ms"], report["t2"]["ms"])
    assert sum(s["interactions"] for s in report["splits"].values()) == len(df)

    books = pd.read_parquet(paths.resolve("data:books.parquet"))
    assert list(books["i"]) == list(range(len(id_map["items"])))
    assert list(books["canonical_id"]) == id_map["items"]
    reviews = pq.read_table(paths.resolve("data:reviews.parquet"))
    assert reviews.schema.equals(split.REVIEW_OUT_SCHEMA)
    rv = reviews.to_pandas()
    assert set(zip(rv["u"], rv["i"], strict=True)) <= set(zip(df["u"], df["i"], strict=True))
    assert list(rv["review_row"]) == list(range(len(rv)))
    with_text = set(zip(rv.loc[rv["text_ok"], "u"], rv.loc[rv["text_ok"], "i"], strict=True))
    assert with_text == set(
        zip(df.loc[df["has_text"], "u"], df.loc[df["has_text"], "i"], strict=True)
    )

    first = paths.resolve("data:id_map.json").read_bytes()
    run_stage("data.kcore_split", small_core, force=True)
    assert paths.resolve("data:id_map.json").read_bytes() == first  # stable across reruns
    notes = read_manifest(paths.manifest("data.kcore_split"))["notes"]
    assert notes["sizing"]["user_fraction"] == 1.0 and notes["counts"]["users"] == len(
        id_map["users"]
    )


def test_eda_on_the_tiny_fixture(small_core: Config) -> None:
    run_all(small_core, until="data.eda")
    paths = RunPaths.from_config(small_core, None)
    out = paths.resolve("results:eda/")
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    inter = pd.read_parquet(paths.resolve("data:interactions.parquet"))
    assert summary["dataset"]["interactions"] == len(inter)
    steps = [row["step"] for row in summary["attrition"]]
    assert steps[0] == "5-core" and steps[-1] == "sized to target"
    sizes = [r["interactions"] for r in summary["attrition"] if r["interactions"] is not None]
    assert sizes == sorted(sizes, reverse=True)  # attrition never grows
    assert set(summary["slice_mix"]) == {"works_kept", "after_kcore", "after_sizing"}
    for name in ("degree_loglog", "longtail_curve", "ratings_by_split", "review_length",
                 "timeline", "genres", "members_per_work", "slice_mix"):  # fmt: skip
        assert (out / f"{name}.png").stat().st_size > 1000
    assert (out / "attrition.csv").exists()
