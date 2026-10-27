"""Projection hybrid and static fusion (CLAUDE.md s9.5, s16.2 Models)."""

import numpy as np
import pytest

pytest.importorskip("torch")

from ml.eval.errors import LeakageError  # noqa: E402
from ml.models.embedding_scorers import (  # noqa: E402
    EmbeddingArtifact,
    EmbeddingScorer,
    StaticFusionScorer,
    l2_normalize_rows,
    pca,
)
from ml.models.projection import train_projection  # noqa: E402
from tests.ml.models.test_lightgcn import two_taste_groups  # noqa: E402

PARAMS = {"lr": 0.01, "epochs": 3, "steps_per_epoch": 30, "batch_size": 64, "n_negs": 4,
          "normalize_items": True}  # fmt: skip


def frozen_inputs() -> tuple:
    data = two_taste_groups()
    rng = np.random.default_rng(0)
    user = rng.normal(size=(40, 8)).astype(np.float32)
    item = rng.normal(size=(31, 8)).astype(np.float32)
    content = l2_normalize_rows(rng.normal(size=(31, 12)).astype(np.float32))
    degree = np.bincount(data.interactions["i"], minlength=31)
    return data, user, item, degree, content


def test_projection_normalizes_like_it_trains_and_cold_items_use_content_only() -> None:
    data, user, item, degree, content = frozen_inputs()
    assert degree[30] == 0
    w, proj, losses = train_projection(data.interactions, user, item, degree, content, PARAMS, 0)
    assert w.shape == (8, 12) and proj.shape == (31, 8) and len(losses) == 3
    assert np.allclose(np.linalg.norm(proj, axis=1), 1, atol=1e-5)  # the training scorer
    expected = l2_normalize_rows(item + content @ w.T)
    expected[30] = l2_normalize_rows((content[30] @ w.T)[None])[0]  # e_gcn = 0 for cold items
    assert np.allclose(proj, expected, atol=1e-5)
    raw_params = PARAMS | {"normalize_items": False}
    w_raw, proj_raw, _ = train_projection(
        data.interactions, user, item, degree, content, raw_params, 0
    )
    assert np.allclose(proj_raw[:30], item[:30] + content[:30] @ w_raw.T, atol=1e-5)
    assert not np.allclose(np.linalg.norm(proj_raw, axis=1), 1)


def test_projection_is_seeded_and_learns() -> None:
    data, user, item, degree, content = frozen_inputs()
    a = train_projection(data.interactions, user, item, degree, content, PARAMS, 1)
    b = train_projection(data.interactions, user, item, degree, content, PARAMS, 1)
    assert np.array_equal(a[0], b[0])
    long = PARAMS | {"epochs": 6}
    losses = train_projection(data.interactions, user, item, degree, content, long, 1)[2]
    assert losses[-1] < losses[0]


def test_projection_zero_init_starts_at_lightgcn_and_w_reg_shrinks_w() -> None:
    data, user, item, degree, content = frozen_inputs()
    zero = PARAMS | {"normalize_items": False, "w_init": "zero", "epochs": 0}
    w0, proj0, losses = train_projection(data.interactions, user, item, degree, content, zero, 0)
    assert losses == [] and not w0.any()
    assert np.allclose(proj0[:30], item[:30]) and not proj0[30].any()  # exactly raw LightGCN
    free = PARAMS | {"normalize_items": False, "w_init": "zero", "epochs": 4}
    tight = free | {"w_reg": 10.0}
    w_free = train_projection(data.interactions, user, item, degree, content, free, 0)[0]
    w_tight = train_projection(data.interactions, user, item, degree, content, tight, 0)[0]
    assert w_free.any() and np.linalg.norm(w_tight) < np.linalg.norm(w_free)
    with pytest.raises(ValueError, match="w_init"):
        train_projection(data.interactions, user, item, degree, content, free | {"w_init": "x"}, 0)


def test_gcn_dropout_trains_w_harder_and_scores_with_full_gcn() -> None:
    data, user, item, degree, content = frozen_inputs()
    base = PARAMS | {"normalize_items": False, "w_init": "zero", "epochs": 4}
    drop = base | {"gcn_dropout": 0.5}
    frozen = (data.interactions, user, item, degree, content)
    w_plain = train_projection(*frozen, base, 0)[0]
    w_drop, proj_drop, _ = train_projection(*frozen, drop, 0)
    assert not np.array_equal(w_plain, w_drop)  # dropout changes the gradient path
    assert np.allclose(proj_drop[:30], item[:30] + content[:30] @ w_drop.T, atol=1e-5)  # no mask
    a = train_projection(*frozen, drop, 2)
    b = train_projection(*frozen, drop, 2)
    assert np.array_equal(a[0], b[0])  # the mask is seeded
    with pytest.raises(ValueError, match="gcn_dropout"):
        train_projection(*frozen, base | {"gcn_dropout": 1}, 0)


def test_projection_scorer_ranks_cold_items() -> None:
    data, user, item, degree, content = frozen_inputs()
    _, proj, _ = train_projection(data.interactions, user, item, degree, content, PARAMS, 0)
    artifact = EmbeddingArtifact(user, proj, degree, {"cutoff": data.cutoff})
    scorer = EmbeddingScorer("projection", artifact, normalize_items=True, score_cold=True)
    scorer.fit(data)
    assert np.isfinite(scorer.score(np.arange(5))).all()


def test_static_fusion() -> None:
    data, user, item, degree, content = frozen_inputs()
    artifact = EmbeddingArtifact(user, item, degree, {"cutoff": data.cutoff})
    reduced = pca(content, 8)
    assert reduced.shape == (31, 8) and np.allclose(reduced.mean(axis=0), 0, atol=1e-5)
    pure_cf = StaticFusionScorer(artifact, content, 1.0, 4.0)
    pure_cf.fit(data)
    scores = pure_cf.score(np.arange(5))
    norm = EmbeddingScorer("n", artifact, normalize_items=True)
    norm.fit(data)
    ranks = np.argsort(-norm.score(np.arange(5))[:, :30], axis=1)
    assert np.array_equal(np.argsort(-scores[:, :30], axis=1), ranks)
    mixed = StaticFusionScorer(artifact, content, 0.5, 4.0)
    mixed.fit(data)
    assert mixed.name == "static_fusion_0.5" and np.isfinite(mixed.score(np.arange(5))).all()
    assert np.all(mixed.score(np.arange(5))[:, 30] != 0)  # the cold item is scored by content
    stale = StaticFusionScorer(
        EmbeddingArtifact(user, item, degree, {"cutoff": 1}), content, 0.5, 4
    )
    with pytest.raises(LeakageError, match="cutoff"):
        stale.fit(data)
