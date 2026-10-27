"""Aspect validation arithmetic (CLAUDE.md s11.5 steps 3 to 5)."""

import numpy as np
import pandas as pd
import pytest

from ml.nlp import validate_aspects as va
from ml.nlp.aspect_seeds import ASPECTS

N = len(ASPECTS)


def _frame(labels: list[dict[str, str]]) -> pd.DataFrame:
    rows = []
    for k, lab in enumerate(labels):
        row = {"sent_id": k, "text": f"sentence {k}"}
        row.update({a: lab.get(a, "") for a in ASPECTS})
        rows.append(row)
    return pd.DataFrame(rows)


def test_parse_labels_accepts_plus_minus_zero_and_unicode_minus() -> None:
    frame = _frame(
        [{"plot": "+"}, {"plot": "-", "humor": "0"}, {"plot": " − "}, {}, {"plot": '"+"'},
         {"plot": "'-", "humor": "x"}, {"plot": "=+", "humor": "+'"}, {"plot": "=-"}]
    )  # fmt: skip
    truth = va.parse_labels(frame)
    plot = truth[:, ASPECTS.index("plot")].tolist()
    assert plot == [True, True, True, False, True, True, True, True]
    humor = truth[:, ASPECTS.index("humor")].tolist()
    assert humor == [False, True, False, False, False, False, True, False]
    assert truth.sum() == 9
    with pytest.raises(ValueError, match="lacks the column"):
        va.parse_labels(frame.drop(columns=["ending"]))


def test_prf_and_kappa() -> None:
    pred = np.array([True, True, False, False])
    truth = np.array([True, False, True, False])
    assert va.prf(pred, truth) == (0.5, 0.5, 0.5)
    assert va.prf(np.zeros(4, bool), np.zeros(4, bool)) == (0.0, 0.0, 0.0)
    assert va.cohen_kappa(truth, truth) == 1.0
    assert va.cohen_kappa(truth, ~truth) == pytest.approx(-1.0)
    assert np.isnan(va.cohen_kappa(np.ones(4, bool), np.ones(4, bool)))  # chance agreement total


def test_tune_tau_picks_the_best_f1_and_the_larger_tie() -> None:
    sims = np.array([0.9, 0.7, 0.5, 0.3, 0.2])
    lexicon = np.zeros(5, bool)
    truth = np.array([True, True, False, False, False])
    tau, f1 = va.tune_tau(sims, lexicon, truth, [0.3, 0.4, 0.5, 0.6, 0.7], 0.45)
    assert tau == 0.7 and f1 == 1.0  # 0.6 and 0.7 both give F1 1.0; ties go to the larger
    tau2, f1_2 = va.tune_tau(sims, lexicon, truth, [0.3, 0.4, 0.5], 0.45)
    assert tau2 == 0.5 and f1_2 == pytest.approx(0.8)  # 0.5 keeps one false positive
    tau3, f1_3 = va.tune_tau(sims, lexicon, np.zeros(5, bool), [0.3], 0.45)
    assert tau3 == 0.45 and np.isnan(f1_3)


def test_validate_splits_tunes_and_flags() -> None:
    rng = np.random.default_rng(0)
    n = 60
    truth = np.zeros((n, N), bool)
    truth[:, 0] = rng.random(n) < 0.5  # plot: separable by similarity
    sims = np.where(truth, rng.uniform(0.7, 0.9, (n, N)), rng.uniform(0.1, 0.5, (n, N))).astype(
        np.float32
    )
    lexicon = np.zeros((n, N), bool)
    current = np.full(N, 0.45, np.float32)
    out = va.validate(sims, lexicon, truth, current, [0.4, 0.5, 0.6, 0.7], 0.5, 0.4, seed=1)
    plot = out.table[out.table["aspect"] == "plot"].iloc[0]
    assert plot["f1"] == 1.0 and 0.5 <= plot["tau_tuned"] <= 0.7 and plot["flag"] == ""
    others = out.table[out.table["aspect"] != "plot"]
    assert (others["flag"] == "no_positive_in_tune_half").all()
    assert np.allclose(others["tau_tuned"], 0.45)  # unchanged without positives (float32 tau)
    assert out.summary["aspects_scored"] == 1 and out.summary["macro_f1"] == 1.0
    assert out.summary["tune_rows"] == 30 and out.summary["test_rows"] == 30
    assert out.summary["proposed_tau"]["plot"] == plot["tau_tuned"]


def test_agreement_on_shared_rows() -> None:
    first = _frame([{"plot": "+"}, {"humor": "-"}, {}, {"plot": "0"}])
    second = _frame([{"plot": "+"}, {"humor": "+"}, {"plot": "-"}])  # rows 0..2 only
    out = va.agreement(first, second)
    assert out["shared_rows"] == 3
    assert out["kappa"]["humor"] == 1.0  # identical presence on the shared rows
    assert out["kappa"]["plot"] is not None and out["kappa"]["plot"] < 1.0
    assert out["any_aspect_agreement"] == pytest.approx(2 / 3)
