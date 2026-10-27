"""Evaluation harness (CLAUDE.md s8, s16.2): metrics, masking, degenerate scores, leakage guards,
bootstrap, strata, test lock."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.common.errors import StageError
from ml.eval import bootstrap, metrics
from ml.eval.data import FitData, build_eval_data, longtail_items, sparsity_bucket
from ml.eval.errors import DegenerateScoresError, LeakageError
from ml.eval.evaluator import evaluate, top_k
from ml.eval.lock import acquire_test_lock
from ml.eval.strata import by_sparsity, cold_item_rows, longtail_rows

KS = (1, 3)
SPLITS = {"t1": {"ms": 1000}, "t2": {"ms": 2000}}


def test_hand_computed_metrics() -> None:  # s16.2
    m = metrics.per_user_metrics(np.array([3, 5, 1, 7]), np.array([1, 3]), [3])
    assert m["precision@3"] == pytest.approx(2 / 3)
    assert m["recall@3"] == 1.0 and m["hit@3"] == 1.0 and m["mrr@3"] == 1.0
    expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert m["ndcg@3"] == pytest.approx(expected) and m["ndcg@3"] == pytest.approx(0.9197, abs=1e-4)
    zero = metrics.per_user_metrics(np.array([1, 2, 3]), np.array([9]), [3])
    assert all(v == 0 for v in zero.values())


def test_beyond_accuracy() -> None:
    tops = np.array([[0, 1], [0, 2]])
    assert metrics.coverage(tops, 2, 10) == pytest.approx(0.3)
    assert metrics.gini(np.array([[0], [1]]), 1, 2) == pytest.approx(0.0)
    assert metrics.gini(np.array([[0], [0]]), 1, 2) > 0
    tail = np.array([False, True, True])
    assert metrics.tail_share(tops, 2, tail) == pytest.approx(0.5)
    emb = np.eye(3, dtype=np.float32)
    assert metrics.ild(tops, 2, emb) == pytest.approx(1.0)
    assert metrics.novelty(np.array([[0]]), 1, np.array([2, 2])) == pytest.approx(1.0)


def toy_interactions() -> pd.DataFrame:
    rows = [  # (u, i, rating, ts, split)
        (0, 0, 5, 10, "train"), (0, 1, 4, 20, "train"), (0, 2, 5, 1500, "val"),
        (0, 3, 2, 1600, "val"), (1, 1, 5, 30, "train"), (1, 3, 4, 1100, "val"),
        (2, 0, 4, 40, "train"), (2, 4, 5, 2500, "test"), (3, 2, 5, 1200, "val"),
    ]  # fmt: skip
    return pd.DataFrame(rows, columns=["u", "i", "rating", "ts", "split"])


def toy_data(mode: str = "val"):  # type: ignore[no-untyped-def]
    return build_eval_data(toy_interactions(), SPLITS, mode, 4, 6, 4, 0.5)  # type: ignore[arg-type]


def test_eval_data_follows_the_protocol() -> None:  # s8.1
    val = toy_data("val")
    assert list(val.users) == [0, 1]  # user 3 has no train interactions; user 2 no relevant val
    assert list(val.truth[0]) == [2] and list(val.mask[0]) == [0, 1]  # rating 2 is not relevant
    assert val.cutoff == 1000 and list(val.fit_counts) == [2, 1, 1, 0]
    test = toy_data("test")
    assert list(test.users) == [2] and list(test.mask[2]) == [0] and test.cutoff == 2000
    with pytest.raises(LeakageError, match="cutoff"):
        FitData(toy_interactions(), 4, 6, cutoff=1000)


class Fixed:
    """A scorer returning a fixed score row for every user."""

    def __init__(self, row: np.ndarray, cutoff: int = 1000, name: str = "fixed") -> None:
        self.row, self.cutoff, self.name = np.asarray(row, np.float32), cutoff, name

    def fit(self, data: FitData) -> None:
        pass

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.tile(self.row, (len(users), 1))


def test_masks_are_applied_and_truth_is_kept() -> None:  # s16.2
    data = toy_data()
    favour_seen = Fixed(np.array([9, 8, 1, 2, 0.5, 0.1]))  # fit items 0, 1 score highest
    result = evaluate(favour_seen, data, KS)
    for r, u in enumerate(result.rows["u"]):
        assert not set(result.tops[r]) & set(data.mask[u].tolist())
    assert result.rows.set_index("u").loc[0, "hit@3"] == 1.0  # item 2 (truth) is never masked
    assert result.beyond["catalogue_items"] == 6


def test_degenerate_scores_raise() -> None:  # s16.2, the all-zero user vector bug
    with pytest.raises(DegenerateScoresError):
        evaluate(Fixed(np.zeros(6)), toy_data(), KS)
    candidates = np.full(6, -np.inf)
    candidates[[2, 3]] = [1.0, 0.5]  # a candidate ranker: -inf outside its candidates is fine
    evaluate(Fixed(candidates), toy_data(), KS)


def test_cutoff_mismatch_is_refused() -> None:  # s16.2 leakage
    with pytest.raises(LeakageError, match="cutoff"):
        evaluate(Fixed(np.arange(6.0), cutoff=2000), toy_data(), KS)


def test_bootstrap_of_identical_models_contains_zero() -> None:  # s16.2
    rng = np.random.default_rng(0)
    values = rng.random(200)
    diff = bootstrap.paired(values, values, 500, np.random.default_rng(1))
    assert diff.low <= 0 <= diff.high and diff.mean == 0
    better = bootstrap.paired(values + 0.1, values, 500, np.random.default_rng(1))
    assert better.low > 0 and better.share_le_zero == 0
    ci = bootstrap.mean_ci(values, 500, np.random.default_rng(2))
    assert ci.low < ci.mean < ci.high


def test_strata() -> None:
    data = toy_data()
    result = evaluate(Fixed(np.array([0, 0, 3, 2, 1, 0.5])), data, KS)
    table = by_sparsity(result, data, [[1, 2], [3, 1000000]])
    assert list(table["bucket"]) == ["1-2"] and int(table["users"].iloc[0]) == 2
    assert sparsity_bucket(25, [[1, 2], [21, 1000000]]) == "21+"
    assert list(longtail_items(np.array([5, 0, 3, 0]), 0.5)) == [False, True, False, True]
    tail = longtail_rows(result, data, KS)
    assert set(tail["u"]) <= set(data.users.tolist())
    assert set(cold_item_rows(result, data, KS)["u"]) == {0, 1}  # items 2 and 3 are never in train


def test_top_k_orders_best_first() -> None:
    assert top_k(np.array([[0.1, 0.9, 0.5, 0.7]]), 3).tolist() == [[1, 3, 2]]


def test_test_lock_refuses_a_second_run(tmp_path: Path) -> None:  # s8.8, s16.2
    decisions = tmp_path / "decisions.md"
    decisions.write_text("# decisions\n", encoding="utf-8")
    with pytest.raises(StageError, match="--confirm"):
        acquire_test_lock(tmp_path / "test", "abc", decisions, confirm=False)
    acquire_test_lock(tmp_path / "test", "abc", decisions, confirm=True)
    with pytest.raises(StageError, match="already used"):
        acquire_test_lock(tmp_path / "test", "def", decisions, confirm=True)
    acquire_test_lock(tmp_path / "test", "def", decisions, confirm=True, reuse=True)
    assert "WARNING" in decisions.read_text(encoding="utf-8")
