"""Aspect validation against the labelled sample (CLAUDE.md s11.5 steps 3 to 5), stage
``nlp.validate_aspects``.

Input: the labelled ``aspects_300.csv`` (labels ``+``, ``-``, ``0`` = the aspect is mentioned,
blank = absent), copied into the raw directory as ``raw:aspects_300.csv`` so its size and mtime
enter the fingerprint (the Colab cell copies it from ``docs/annotation/``). A second labeller's
file ``aspects_300_second.csv`` in the same directory, when present, gives Cohen's kappa per
aspect on the rows both labelled (it is read but not fingerprinted: rerun with ``--force`` after
changing it).

Detection is recomputed for the labelled sentences from the stored sentence embeddings and the
stored prototypes (no encoder needed): ``hit = max prototype cosine >= tau[a] or lexicon match``,
exactly as ``nlp.aspects``. The sentences are split in a seeded half: on the tuning half the stage
picks, per aspect, the ``tau`` from ``aspects.validation.tau_grid`` with the best F1 (ties go to
the larger, more conservative threshold); on the test half it reports precision, recall and F1 per
aspect and macro, both at the tuned ``tau`` and at the current config ``tau``, plus the lexicon
alone. Outputs ``results/<v>/nlp/aspect_validation.json`` (everything, including the proposed
``aspects.tau`` block the user confirms before editing ``ml/config.yaml``) and
``aspect_validation.csv`` (one row per aspect). Aspects with no labelled positive on a half keep
the current ``tau`` and are flagged; aspects below ``aspects.validation.min_f1`` are flagged for a
lexicon change, a merge or a drop (the user decides).
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.common.stage import StageContext, stage
from ml.nlp.aspect_seeds import ASPECTS, compile_lexicon
from ml.nlp.aspects import N_PROTOTYPES, lexicon_hits, tau_vector

LABELLED = "raw:aspects_300.csv"
SECOND_FILE = "aspects_300_second.csv"
JSON_OUT = "results:nlp/aspect_validation.json"
CSV_OUT = "results:nlp/aspect_validation.csv"
PRESENT = {"+", "-", "0", "−"}  # the unicode minus some editors insert


def parse_labels(frame: pd.DataFrame) -> np.ndarray:
    """(n, 12) bool: the aspect is mentioned (``+``, ``-`` or ``0``); anything else is absent."""
    out = np.zeros((len(frame), len(ASPECTS)), dtype=bool)
    for a, name in enumerate(ASPECTS):
        if name not in frame:
            raise ValueError(f"labelled file lacks the column {name!r}")
        # Labellers who typed "+", '+, =+ or +' to stop the spreadsheet treating + as a formula
        values = frame[name].astype(str).str.strip().str.strip("\"'‘’“”=")
        out[:, a] = values.str.strip().isin(PRESENT).to_numpy()
    return out


def read_labelled(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False, dtype=str)
    if "sent_id" not in frame or "text" not in frame:
        raise ValueError(f"{path.name}: expected sent_id and text columns")
    frame["sent_id"] = frame["sent_id"].astype(np.int64)
    if not frame["sent_id"].is_unique:
        raise ValueError(f"{path.name}: duplicate sent_id rows")
    return frame


def prf(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    """Precision, recall, F1 of one aspect (bool arrays); 0 when undefined."""
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa for two bool label vectors; NaN when agreement by chance is total."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    n = len(a)
    if n == 0:
        return float("nan")
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return float("nan") if pe == 1.0 else (po - pe) / (1 - pe)


def tune_tau(
    sims: np.ndarray, lexicon: np.ndarray, truth: np.ndarray, grid: Sequence[float], current: float
) -> tuple[float, float]:
    """``(tau, f1)`` with the best F1 on one aspect's tuning rows (largest tau among ties); the
    current tau with F1 NaN when the rows hold no positive."""
    if not truth.any():
        return float(current), float("nan")
    best_tau, best_f1 = float(current), -1.0
    for tau in grid:
        _, _, f1 = prf((sims >= tau) | lexicon, truth)
        if f1 >= best_f1:  # >= so ties move to the larger (later) threshold
            best_tau, best_f1 = float(tau), f1
    return best_tau, best_f1


def _mean(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else float("nan")


@dataclass
class Validation:
    table: pd.DataFrame
    summary: dict[str, Any]


def validate(
    sims: np.ndarray,
    lexicon: np.ndarray,
    truth: np.ndarray,
    current_tau: np.ndarray,
    grid: Sequence[float],
    tune_fraction: float,
    min_f1: float,
    seed: int,
) -> Validation:
    """Split, tune per aspect, evaluate on the held-out half (s11.5 step 4)."""
    n = len(truth)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_tune = int(round(tune_fraction * n))
    tune, test = order[:n_tune], order[n_tune:]
    rows = []
    for a, name in enumerate(ASPECTS):
        tau, tune_f1 = tune_tau(
            sims[tune, a], lexicon[tune, a], truth[tune, a], grid, float(current_tau[a])
        )
        pred_tuned = (sims[test, a] >= tau) | lexicon[test, a]
        pred_current = (sims[test, a] >= current_tau[a]) | lexicon[test, a]
        p, r, f = prf(pred_tuned, truth[test, a])
        pc, rc, fc = prf(pred_current, truth[test, a])
        pl, rl, fl = prf(lexicon[test, a], truth[test, a])
        rows.append(
            {
                "aspect": name,
                "labelled_positives": int(truth[:, a].sum()),
                "tune_positives": int(truth[tune, a].sum()),
                "test_positives": int(truth[test, a].sum()),
                "tau_current": float(current_tau[a]),
                "tau_tuned": tau,
                "tune_f1": tune_f1,
                "precision": p,
                "recall": r,
                "f1": f,
                "precision_current_tau": pc,
                "recall_current_tau": rc,
                "f1_current_tau": fc,
                "precision_lexicon_only": pl,
                "recall_lexicon_only": rl,
                "f1_lexicon_only": fl,
                "flag": (
                    "no_positive_in_tune_half"
                    if np.isnan(tune_f1)
                    else (
                        "no_positive_in_test_half"
                        if not truth[test, a].any()
                        else ("below_min_f1" if f < min_f1 else "")
                    )
                ),  # fmt: skip
            }
        )
    table = pd.DataFrame(rows)
    scored = table[table["test_positives"] > 0]
    summary = {
        "sentences": int(n),
        "tune_rows": int(len(tune)),
        "test_rows": int(len(test)),
        "macro_f1": _mean(scored["f1"]),
        "macro_f1_current_tau": _mean(scored["f1_current_tau"]),
        "macro_f1_lexicon_only": _mean(scored["f1_lexicon_only"]),
        "aspects_scored": int(len(scored)),
        "proposed_tau": {r["aspect"]: round(float(r["tau_tuned"]), 3) for r in rows},
        "flags": {r["aspect"]: r["flag"] for r in rows if r["flag"]},
    }  # fmt: skip
    return Validation(table, summary)


def agreement(first: pd.DataFrame, second: pd.DataFrame) -> dict[str, Any]:
    """Cohen's kappa per aspect on the rows both labellers covered."""
    shared = first.merge(second, on="sent_id", suffixes=("", "_second"))
    if shared.empty:
        return {"shared_rows": 0, "kappa": {}}
    a = parse_labels(shared[list(ASPECTS)])
    b = parse_labels(
        shared[[f"{n}_second" for n in ASPECTS]].rename(columns=lambda c: c.removesuffix("_second"))
    )
    kappa = {name: cohen_kappa(a[:, k], b[:, k]) for k, name in enumerate(ASPECTS)}
    finite = [v for v in kappa.values() if not np.isnan(v)]
    return {
        "shared_rows": int(len(shared)),
        "kappa": {k: (None if np.isnan(v) else round(v, 3)) for k, v in kappa.items()},
        "mean_kappa": float(np.mean(finite)) if finite else None,
        "any_aspect_agreement": float((a.any(axis=1) == b.any(axis=1)).mean()),
    }


def similarities(emb: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """(n, 12) best cosine over each aspect's prototypes (same arithmetic as nlp.aspects)."""
    flat = prototypes.reshape(-1, prototypes.shape[-1]).astype(np.float32)
    sims = np.asarray(emb, dtype=np.float32) @ flat.T
    return sims.reshape(len(emb), len(ASPECTS), N_PROTOTYPES).max(axis=2)


@stage(
    "nlp.validate_aspects",
    inputs=[
        LABELLED,
        "data:sent_emb.f16.npy",
        "data:sentences.parquet",
        "artifacts:nlp/aspect_prototypes.npy",
    ],
    outputs=[JSON_OUT, CSV_OUT],
    config_keys=["aspects.names", "aspects.tau_default", "aspects.tau", "aspects.validation"],
)
def validate_aspects(ctx: StageContext) -> None:
    """Tune tau per aspect on half the labelled sentences, report P/R/F1 on the other half."""
    if tuple(ctx.param("aspects.names")) != ASPECTS:
        raise ValueError("aspects.names must equal aspect_seeds.ASPECTS, in order")
    params: Mapping[str, Any] = ctx.param("aspects.validation")
    lo, hi, step = (float(x) for x in params["tau_grid"])
    grid = np.round(np.arange(lo, hi + step / 2, step), 4).tolist()
    labelled_path = ctx.path(LABELLED)
    first = read_labelled(labelled_path)
    sentences = pd.read_parquet(ctx.path("data:sentences.parquet"), columns=["sent_id", "text"])
    known = set(sentences["sent_id"].tolist())
    missing = [int(s) for s in first["sent_id"] if int(s) not in known]
    if missing:
        raise ValueError(f"{len(missing)} labelled sent_ids are not in sentences.parquet")
    truth = parse_labels(first)
    filled = (first[list(ASPECTS)].astype(str).apply(lambda c: c.str.strip() != "")).any(axis=1)
    labelled_rows = filled.to_numpy()
    emb = np.load(ctx.path("data:sent_emb.f16.npy"), mmap_mode="r")
    sent_ids = first["sent_id"].to_numpy(np.int64)
    sims = similarities(emb[sent_ids], np.load(ctx.path("artifacts:nlp/aspect_prototypes.npy")))
    text_by_id = dict(zip(sentences["sent_id"].tolist(), sentences["text"].tolist(), strict=True))
    lexicon = lexicon_hits([text_by_id[int(s)] for s in sent_ids], compile_lexicon())
    current = tau_vector(
        {"tau_default": ctx.param("aspects.tau_default"), "tau": ctx.param("aspects.tau")}
    )
    result = validate(
        sims, lexicon, truth, current, grid, float(params["tune_fraction"]),
        float(params["min_f1"]), ctx.seed,
    )  # fmt: skip
    second_path = labelled_path.parent / SECOND_FILE
    kappa = agreement(first, read_labelled(second_path)) if second_path.exists() else None

    summary = {
        **result.summary,
        "rows_with_any_label_filled": int(labelled_rows.sum()),
        "rows_all_blank": int((~labelled_rows).sum()),
        "target_macro_f1": float(params["target_macro_f1"]),
        "min_f1": float(params["min_f1"]),
        "tau_grid": [lo, hi, step],
        "agreement": kappa,
        "config_block": {"tau": result.summary["proposed_tau"]},
    }
    ctx.out(JSON_OUT).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    result.table.round(4).to_csv(ctx.out(CSV_OUT), index=False)
    ctx.count("labelled_sentences", len(first))
    ctx.count("aspects_scored", summary["aspects_scored"])
    ctx.notes["macro_f1"] = summary["macro_f1"]
    ctx.notes["proposed_tau"] = summary["proposed_tau"]
    ctx.log.info("macro F1 %.3f (current tau %.3f, lexicon only %.3f); proposed aspects.tau: %s",
                 summary["macro_f1"], summary["macro_f1_current_tau"],
                 summary["macro_f1_lexicon_only"], json.dumps(summary["proposed_tau"]))  # fmt: skip
    if summary["flags"]:
        ctx.log.warning("flagged aspects: %s", json.dumps(summary["flags"]))
