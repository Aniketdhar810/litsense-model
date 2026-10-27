"""LightGCN (CLAUDE.md s9.4) and the stored-embedding scorers (s16.2 Models)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from ml.eval.data import FitData  # noqa: E402
from ml.eval.errors import LeakageError  # noqa: E402
from ml.eval.metrics import per_user_metrics  # noqa: E402
from ml.models.embedding_scorers import EmbeddingArtifact, EmbeddingScorer  # noqa: E402
from ml.models.lightgcn import (  # noqa: E402
    embedding_health,
    inner_split,
    normalized_adjacency,
    sample_negatives,
    train_lightgcn,
)
from ml.models.lightgcn_sweep import grid  # noqa: E402

PARAMS = {"dim": 16, "layers": 3, "lr": 0.05, "batch_size": 64, "reg": 1e-4, "n_negs": 5,
          "eval_every": 2, "patience": 2}  # fmt: skip


def two_taste_groups(seed: int = 0) -> FitData:
    """Users 0-19 read items 0-14, users 20-39 read items 15-29; item 30 is never read."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(40):
        lo = 0 if u < 20 else 15
        rows += [(u, int(i)) for i in rng.choice(np.arange(lo, lo + 15), 8, replace=False)]
    frame = pd.DataFrame(rows, columns=["u", "i"])
    frame["rating"] = 5.0
    frame["ts"] = np.arange(len(frame))
    return FitData(frame, 40, 31, cutoff=len(frame))


def test_negative_sampler_never_returns_a_positive() -> None:
    n_items = 20
    users = np.repeat(np.arange(3), 50)
    positives = {0: range(19), 1: range(0, 20, 2), 2: [5]}  # user 0 has one allowed item
    pos_keys = np.sort(np.array([u * n_items + i for u, items in positives.items() for i in items]))
    neg = sample_negatives(users, n_items, pos_keys, 7, np.random.default_rng(0), rounds=2)
    assert neg.shape == (150, 7)
    for u, row in zip(users, neg, strict=True):
        assert not set(row.tolist()) & set(positives[int(u)])
    assert (neg[users == 0] == 19).all()


def test_normalized_adjacency() -> None:
    users, items = np.array([0, 0, 1]), np.array([0, 1, 1])
    adj = normalized_adjacency(users, items, n_users=3, n_items=3).toarray()
    assert adj.shape == (6, 6) and np.allclose(adj, adj.T)
    # degrees: u0=2, u1=1, u2=0, i0=1, i1=2, i2=0
    assert adj[0, 3] == pytest.approx(1 / np.sqrt(2 * 1))
    assert adj[0, 4] == pytest.approx(1 / np.sqrt(2 * 2))
    assert not adj[2].any() and not adj[5].any()  # isolated nodes: zero rows, no division by 0


def test_inner_split_holds_out_the_latest_rows() -> None:
    rows = pd.DataFrame({"u": [0] * 10 + [1] * 3, "i": list(range(10)) + [0, 1, 2],
                         "ts": list(range(10, 0, -1)) + [1, 2, 3]})  # fmt: skip
    held = inner_split(rows, fraction=0.2, min_user=6)
    assert held.index.equals(rows.index)
    assert sorted(rows.loc[held, "i"]) == [0, 1]  # user 0's two newest rows by ts
    assert not held[rows["u"] == 1].any()  # below min_user: nothing held out


def test_temporal_inner_split_is_a_global_time_cutoff() -> None:
    rows = pd.DataFrame({"u": [0] * 10 + [1] * 10, "i": list(range(20)),
                         "ts": list(range(1, 11)) + list(range(11, 21))})  # fmt: skip
    held = inner_split(rows, fraction=0.2, min_user=6, mode="temporal")
    assert held.index.equals(rows.index)
    assert sorted(rows.loc[held, "ts"]) == [17, 18, 19, 20]  # the newest 20% overall
    assert not held[rows["u"] == 0].any()  # user 0 is all old: nothing held, min_user ignored
    assert not inner_split(rows.iloc[:0], 0.2, 6, mode="temporal").any()
    with pytest.raises(ValueError, match="inner_split"):
        inner_split(rows, 0.2, 6, mode="random")


def training_ndcg(user: np.ndarray, item: np.ndarray, data: FitData) -> float:
    scores = user @ item.T
    values = []
    for u, group in data.interactions.groupby("u"):
        top = np.argsort(-scores[u], kind="stable")[:10]
        values.append(per_user_metrics(top, np.sort(group["i"].to_numpy()), [10])["ndcg@10"])
    return float(np.mean(values))


def test_lightgcn_beats_random_on_training_ndcg() -> None:
    data = two_taste_groups()
    result = train_lightgcn(data.interactions, 40, 31, PARAMS, seed=0, epochs=30)
    assert result.user.shape == (40, 16) and result.item.shape == (31, 16)
    rng = np.random.default_rng(1)
    random = training_ndcg(rng.normal(size=(40, 16)), rng.normal(size=(31, 16)), data)
    assert training_ndcg(result.user, result.item, data) > random + 0.2


def test_training_is_seeded_and_early_stopping_is_recorded() -> None:
    data = two_taste_groups()
    a = train_lightgcn(data.interactions, 40, 31, PARAMS, seed=3, epochs=4)
    b = train_lightgcn(data.interactions, 40, 31, PARAMS, seed=3, epochs=4)
    assert np.array_equal(a.user, b.user) and np.array_equal(a.item, b.item)
    metrics = iter([0.1, 0.3, 0.2, 0.2, 0.2])
    run = train_lightgcn(
        data.interactions, 40, 31, PARAMS, 0, 20, evaluate_fn=lambda u, i: next(metrics)
    )
    assert run.best_epoch == 4 and run.best_metric == 0.3
    assert run.epochs == 8  # two evaluations without improvement (patience 2)
    assert [h["epoch"] for h in run.history] == [2, 4, 6, 8]


def test_embedding_health_flags_anisotropy_and_skips_cold_items() -> None:
    rng = np.random.default_rng(0)
    spread = rng.normal(size=(300, 16)).astype(np.float32)
    warm = np.ones(300, bool)
    warm[-1] = False
    spread[-1] = 1e6  # a cold item with a huge norm must not affect the warm statistics
    health = embedding_health(spread, warm)
    assert health["items"] == 299 and abs(health["mean_pairwise_cosine"]) < 0.05
    assert health["norm_min"] <= health["norm_median"] <= health["norm_max"] < 1e5
    same_way = np.abs(spread) + 2  # every vector in the positive orthant: anisotropic
    assert embedding_health(same_way, warm)["mean_pairwise_cosine"] > 0.5
    # exact mean over ordered pairs, checked against the direct O(n^2) formula on a tiny sample
    tiny = rng.normal(size=(5, 3))
    unit = tiny / np.linalg.norm(tiny, axis=1, keepdims=True)
    direct = (unit @ unit.T)[~np.eye(5, dtype=bool)].mean()
    assert embedding_health(tiny, np.ones(5, bool))["mean_pairwise_cosine"] == pytest.approx(direct)
    assert np.isnan(embedding_health(tiny, np.zeros(5, bool))["mean_pairwise_cosine"])


def test_sweep_grid_is_the_product_in_config_order() -> None:
    points = grid({"dims": [64, 128], "regs": [1e-4, 1e-3]})
    assert points == [{"dim": 64, "reg": 1e-4}, {"dim": 64, "reg": 1e-3},
                      {"dim": 128, "reg": 1e-4}, {"dim": 128, "reg": 1e-3}]  # fmt: skip


def test_embedding_scorers_cold_items_and_leakage_checks() -> None:
    data = two_taste_groups()
    rng = np.random.default_rng(0)
    degree = np.bincount(data.interactions["i"], minlength=31)
    artifact = EmbeddingArtifact(rng.normal(size=(40, 4)).astype(np.float32),
                                 rng.normal(size=(31, 4)).astype(np.float32), degree,
                                 {"cutoff": data.cutoff})  # fmt: skip
    raw = EmbeddingScorer("raw", artifact, normalize_items=False)
    norm = EmbeddingScorer("norm", artifact, normalize_items=True)
    for scorer in (raw, norm):
        scorer.fit(data)
        scores = scorer.score(np.arange(5))
        assert scores.dtype == np.float32 and np.isneginf(scores[:, 30]).all()
        assert np.isfinite(scores[:, :30]).all()
    expected = (
        artifact.user[:5] @ (artifact.item / np.linalg.norm(artifact.item, axis=1)[:, None]).T
    )
    assert np.allclose(norm.score(np.arange(5))[:, :30], expected[:, :30], atol=1e-5)
    other = FitData(data.interactions.iloc[1:], 40, 31, data.cutoff)
    with pytest.raises(LeakageError, match="degrees"):
        raw.fit(other)
    stale = EmbeddingScorer("old", EmbeddingArtifact(artifact.user, artifact.item, degree,
                                                     {"cutoff": 1}), False)  # fmt: skip
    with pytest.raises(LeakageError, match="cutoff"):
        stale.fit(data)
