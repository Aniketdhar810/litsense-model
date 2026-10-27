"""Baselines (CLAUDE.md s9.1-9.3), the content text builder (s10.1, s16.2 leakage) and the
eval.val stage on the tiny fixture."""

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

import ml.data.canonical  # noqa: F401  (registers the stage)
import ml.data.clean as clean
import ml.data.download  # noqa: F401  (registers the stage)
import ml.data.eda  # noqa: F401  (registers the stage)
import ml.data.goodreads  # noqa: F401  (registers the stage)
import ml.data.meta  # noqa: F401  (registers the stage)
import ml.data.reviews  # noqa: F401  (registers the stage)
import ml.data.split  # noqa: F401  (registers the stage)
import ml.eval.validation  # noqa: F401  (registers the stage)
import ml.models.lightgcn  # noqa: F401  (registers the stage)
import ml.models.projection  # noqa: F401  (registers the stage)
import ml.nlp.content  # noqa: F401  (registers the stage)
from ml.common.manifest import read_manifest
from ml.common.stage import RunPaths, run_all
from ml.eval.data import FitData
from ml.models.baselines import (
    ALSScorer,
    ContentScorer,
    ItemKNNScorer,
    PopularityScorer,
    RandomScorer,
)
from ml.nlp import content, embed

Config = dict[str, Any]


class StubEncoder:
    """Deterministic 16-d embeddings from a text hash; whitespace tokens."""

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for t in texts:
            seed = int(hashlib.sha1(t.encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).normal(size=16)
            rows.append(v / np.linalg.norm(v))
        return np.array(rows, dtype=np.float32)

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def truncate(self, text: str, max_tokens: int) -> str:
        return " ".join(text.split()[:max_tokens])


def fit_data(seed: int = 0, n_users: int = 60, n_items: int = 30) -> FitData:
    rng = np.random.default_rng(seed)
    pop = 1 / np.arange(1, n_items) ** 1.2  # item n_items-1 is never rated
    rows = {
        (int(u), int(rng.choice(n_items - 1, p=pop / pop.sum())))
        for u in rng.integers(0, n_users, 600)
    }
    frame = pd.DataFrame(sorted(rows), columns=["u", "i"])
    frame["rating"] = rng.choice([3.0, 4.0, 5.0], len(frame))
    frame["ts"] = rng.integers(1, 1000, len(frame))
    return FitData(frame, n_users, n_items, cutoff=1000)


def test_item_text_puts_metadata_first_and_never_reviews() -> None:  # s16.2 leakage
    book = {"title": "Circe", "author": "Madeline Miller", "genres": ["Fantasy"],
            "features": "word " * 500, "description": "Review: a reviewer loved it"}  # fmt: skip
    text = content.item_text(book, StubEncoder(), 256)
    assert text.startswith("Circe. By Madeline Miller. Genres: Fantasy.")
    assert len(text.split()) <= 256 - content.SPECIAL_TOKENS
    assert "reviewer" not in text  # the blurb wins over the editorial description
    no_blurb = content.item_text({"title": "X", "genres": []}, StubEncoder(), 256)
    from_parquet = {"title": "X", "genres": np.array(["Fantasy", "Mystery"])}  # Colab crash
    assert "Genres: Fantasy, Mystery." in content.item_text(from_parquet, StubEncoder(), 256)
    assert no_blurb == "X. By Unknown. Genres: ."


def test_neighbours_and_anisotropy() -> None:
    emb = StubEncoder().encode([f"book {i}" for i in range(40)])
    nn = content.nearest_neighbours(emb, 5, batch=7)
    assert nn.shape == (40, 5) and not (nn == np.arange(40)[:, None]).any()
    sims = emb @ emb.T
    assert sims[0, nn[0, 0]] == pytest.approx(np.max(np.delete(sims[0], 0)))
    assert abs(content.mean_pairwise_cosine(emb, 500, np.random.default_rng(0))) < 0.5


def test_baselines_score_shapes_and_cold_items() -> None:
    data = fit_data()
    users = np.arange(10)
    emb = StubEncoder().encode([f"b{i}" for i in range(data.n_items)])
    for scorer in (
        RandomScorer(0),
        PopularityScorer({"bayes_m": 10, "trend_half_life_days": 180}),
        ItemKNNScorer({"k": 10, "shrink": 10}),
        ALSScorer({"factors": 8, "regularization": 0.01, "iterations": 5, "alpha": 20}, 0),
        ContentScorer(emb, 4.0),
    ):
        scorer.fit(data)
        scores = scorer.score(users)
        assert scores.shape == (10, data.n_items) and scores.dtype == np.float32
        assert scorer.cutoff == data.cutoff
        if scorer.name in ("popularity", "itemknn", "als"):  # CF scorers: unseen item is -inf
            assert np.isneginf(scores[:, -1]).all()
            assert np.isfinite(scores[:, :-1]).any()
    content_scorer = ContentScorer(emb, 4.0)
    content_scorer.fit(data)
    assert np.allclose(
        np.linalg.norm(content_scorer.profiles[np.unique(data.interactions["u"])], axis=1), 1
    )


def test_popularity_extras() -> None:
    pop = PopularityScorer({"bayes_m": 10, "trend_half_life_days": 180})
    pop.fit(fit_data())
    assert pop.counts[0] >= pop.counts[5]
    assert np.all(np.isfinite(pop.bayes_rating)) and pop.trending[-1] == 0


def test_itemknn_keeps_k_neighbours() -> None:
    knn = ItemKNNScorer({"k": 3, "shrink": 0})
    knn.fit(fit_data())
    per_row = np.diff(knn.sim.indptr)
    assert per_row.max() <= 3 and knn.sim.diagonal().sum() == 0


@pytest.fixture
def small_core(tiny_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> Config:
    monkeypatch.setattr(clean, "load_language_id", lambda path: lambda text: ("en", 0.9))
    monkeypatch.setattr(embed, "load_encoder", lambda *args: StubEncoder())
    tiny_cfg["data"]["kcore"] = {"user": 2, "item": 2}
    tiny_cfg["eval"]["ks"] = [5, 10]
    tiny_cfg["eval"]["bootstrap_resamples"] = 50
    tiny_cfg["models"]["lightgcn"] |= {
        "dim": 8,
        "batch_size": 64,
        "max_epochs": 4,
        "eval_every": 2,
        "patience": 1,
        "inner_val_min_user": 3,
    }
    tiny_cfg["models"]["projection"] |= {"epochs": 1, "steps_per_epoch": 5, "batch_size": 32}
    return tiny_cfg


def test_stage_on_the_tiny_fixture(small_core: Config) -> None:
    run_all(small_core, until="eval.val")
    paths = RunPaths.from_config(small_core, None)
    out = paths.resolve("results:val/")
    table = pd.read_csv(out / "table_main.csv")
    assert set(table["model"]) == {
        "random",
        "popularity",
        "itemknn",
        "als",
        "content",
        "lightgcn_raw",
        "lightgcn_norm",
        "projection",
        "projection_shuffled",
        "static_fusion_0.5",
        "static_fusion_0.7",
        "static_fusion_0.9",
    }
    assert (table["ndcg@10_low"] <= table["ndcg@10"]).all()
    assert (table["ndcg@10"] <= table["ndcg@10_high"]).all()
    assert table["catalogue_items"].nunique() == 1
    pop = json.loads((out / "popularity.json").read_text(encoding="utf-8"))
    splits = json.loads(paths.resolve("data:splits.json").read_text(encoding="utf-8"))
    assert pop["cutoff"] == splits["t1"]["ms"] and pop["mode"] == "val"
    manifest = read_manifest(paths.manifest("eval.val"))
    assert manifest["cutoff"] == splits["t1"]["ms"]
    for name in ("table_strata.csv", "table_longtail.csv", "bootstrap.csv",
                 "bars_ndcg_recall.png", "ndcg_by_bucket.png"):  # fmt: skip
        assert (out / name).exists()
    paired = pd.read_csv(out / "bootstrap.csv")
    assert set(paired["vs"]) == {"popularity", "lightgcn_raw", "lightgcn_norm"}
    assert not (paired["vs"] == paired["model"]).any()
    lgcn = json.loads((out / "lightgcn_norm.json").read_text(encoding="utf-8"))
    assert lgcn["artifact"]["cutoff"] == splits["t1"]["ms"]  # the manifest-side record (s2.2 #5)
    assert lgcn["artifact"]["model"] == "lightgcn" and lgcn["artifact"]["mode"] == "val"
    user = np.load(paths.resolve("artifacts:eval_val/lightgcn/lgcn_user.npy"))
    norm = np.load(paths.resolve("artifacts:eval_val/lightgcn/lgcn_item_norm.npy"))
    assert user.shape[1] == 8 and np.allclose(np.linalg.norm(norm, axis=1), 1, atol=1e-5)
    trained = read_manifest(paths.manifest("models.lightgcn_val"))
    assert trained["cutoff"] == splits["t1"]["ms"] and trained["status"] == "complete"
    assert paths.resolve("results:training/lightgcn_val.csv").exists()
    proj = json.loads((out / "projection_shuffled.json").read_text(encoding="utf-8"))
    assert proj["artifact"]["shuffled_content"] and proj["artifact"]["cutoff"] == splits["t1"]["ms"]
    emb = np.load(paths.resolve("artifacts:content/content_item.npy"))
    books = pd.read_parquet(paths.resolve("data:books.parquet"))
    assert emb.shape == (len(books), 16)
