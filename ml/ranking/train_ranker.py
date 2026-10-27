"""LambdaRank training (CLAUDE.md s13.5) and the val ablation over feature groups (s13.7), stages
``ranking.train`` and ``ranking.ablation_val``.

``ranking.train`` fits ``LGBMRanker`` on the train-split users of ``training.parquet`` with early
stopping on the held-out users, saves ``ranker.txt`` + ``feature_list.json`` (what the bundle
ships), gain importances (CSV + bar chart) and compares, on the held-out users' full candidate
lists, the ranker against the blend (s12.3) and against the CF score alone. Metrics come from
``ml.eval.metrics.per_user_metrics`` with each user's real number of relevant val items, so a
positive the candidates missed still counts as a miss (candidate recall is not hidden).

``ranking.ablation_val`` retrains the same ranker with the feature-group masks M4..M8b of s13.7
on the same rows and reports each on the held-out users with 95% CIs and a paired bootstrap
against M4 and against the blend. Post-processing (MMR, genre cap, exploration slot) is not part
of this offline table; it reorders a top list for diversity and is measured through the
evaluator's beyond-accuracy metrics, not here.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from litsense_core import features
from ml.common.stage import StageContext, stage
from ml.eval import report
from ml.eval.metrics import per_user_metrics
from ml.ranking.build_training import RANKER, TRAINING, TRAINING_META

RANKER_TXT = f"artifacts:{RANKER}/ranker.txt"
FEATURE_LIST = f"artifacts:{RANKER}/feature_list.json"
RANKER_META = f"artifacts:{RANKER}/meta.json"
IMPORTANCE_CSV = "results:ranker/importance.csv"
IMPORTANCE_PNG = "results:ranker/importance.png"
VS_BLEND = "results:ranker/ranker_vs_blend.csv"
VS_BLEND_PAIRED = "results:ranker/ranker_vs_blend_bootstrap.csv"
ABLATION = "results:ranker/ablation_val.csv"
ABLATION_PAIRED = "results:ranker/ablation_val_bootstrap.csv"
ABLATION_PNG = "results:ranker/ablation_val.png"
PAIRED_METRICS = ("ndcg@10", "recall@20")
BASE = ("CF", "META", "USER", "SOURCE")
# s13.7 rows M4..M8b as unions of the s13.3 feature groups.
ABLATION_MASKS: dict[str, tuple[str, ...]] = {
    "M4": BASE,
    "M5": (*BASE, "CONTENT", "REVIEW_EMB"),
    "M6": (*BASE, "ASPECT"),
    "M7": (*BASE, "CONTENT", "REVIEW_EMB", "ASPECT"),
    "M8": (*BASE, "CONTENT", "REVIEW_EMB", "ASPECT", "AUTHOR", "PROJ"),
    "M8a": (*BASE, "CONTENT", "REVIEW_EMB", "ASPECT", "PROJ"),
    "M8b": (*BASE, "CONTENT", "REVIEW_EMB", "AUTHOR", "PROJ"),
}


def group_sizes(users: np.ndarray) -> np.ndarray:
    """Sizes of the contiguous runs of ``users`` (rows are sorted per user)."""
    if len(users) == 0:
        return np.zeros(0, dtype=np.int64)
    change = np.flatnonzero(np.diff(users) != 0) + 1
    bounds = np.concatenate([[0], change, [len(users)]])
    sizes = np.diff(bounds)
    assert int(sizes.sum()) == len(users)
    return sizes


def feature_names(groups: Sequence[str] | None) -> list[str]:
    """Feature columns of the given s13.3 groups, in ``FEATURES`` order (all when None)."""
    if groups is None:
        return list(features.FEATURES)
    wanted = {f for g in groups for f in features.GROUPS[g]}
    return [f for f in features.FEATURES if f in wanted]


def fit_ranker(
    frame: pd.DataFrame, names: Sequence[str], params: Mapping[str, Any], seed: int
) -> tuple[Any, int]:
    """``(booster, best_iteration)`` from the s13.5 recipe; early stopping on the held-out users
    when they hold at least one positive, else the full ``n_estimators``."""
    import lightgbm as lgb

    train = frame[frame["split"] == "train"]
    valid = frame[frame["split"] == "valid"]
    if train.empty or not train["label"].any():
        raise ValueError("no training rows with a positive label")
    x_tr = train[list(names)].to_numpy(np.float32)
    y_tr = train["label"].to_numpy(np.int32)
    g_tr = group_sizes(train["u"].to_numpy())
    model = lgb.LGBMRanker(
        objective=str(params["objective"]),
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["num_leaves"]),
        min_child_samples=int(params["min_child_samples"]),
        colsample_bytree=float(params["colsample_bytree"]),
        subsample=float(params["subsample"]),
        subsample_freq=int(params["subsample_freq"]),
        random_state=seed,
        verbose=-1,
    )
    if len(valid) and valid["label"].any():
        model.fit(
            x_tr, y_tr, group=g_tr,
            eval_set=[(valid[list(names)].to_numpy(np.float32), valid["label"].to_numpy(np.int32))],
            eval_group=[group_sizes(valid["u"].to_numpy())],
            eval_at=[int(k) for k in params["eval_at"]],
            callbacks=[lgb.early_stopping(int(params["early_stopping_rounds"]), verbose=False)],
        )  # fmt: skip
        best = int(model.best_iteration_ or model.n_estimators)
    else:
        model.fit(x_tr, y_tr, group=g_tr)
        best = int(model.n_estimators)
    return model.booster_, best


def held_out_rows(
    frame: pd.DataFrame, scores: np.ndarray, ks: Sequence[int], name: str
) -> pd.DataFrame:
    """Per-user metrics on the held-out users' full candidate lists ranked by ``scores``. The
    user's relevant items the candidates missed are represented by placeholder ids, so recall
    and NDCG keep their true denominators."""
    k_max = max(ks)
    rows = []
    frame = frame.assign(_score=np.asarray(scores, dtype=np.float32))
    for u, g in frame.groupby("u", sort=True):
        order = np.argsort(-g["_score"].to_numpy(), kind="stable")
        ranked = g["i"].to_numpy(np.int64)[order]
        top = np.concatenate([ranked, np.full(max(0, k_max - len(ranked)), -1)])[:k_max]
        found = g.loc[g["label"] == 1, "i"].to_numpy(np.int64)
        missing = int(g["n_truth"].iloc[0]) - len(found)
        truth = np.concatenate([found, -2 - np.arange(max(missing, 0))])
        rows.append({"u": int(u), **per_user_metrics(top, truth, ks)})
    out = pd.DataFrame(rows)
    out.attrs["model"] = name
    return out


def importance_table(booster: Any, names: Sequence[str]) -> pd.DataFrame:
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    table = pd.DataFrame(
        {
            "feature": list(names),
            "group": [features.group_of(n) for n in names],
            "gain": gain,
            "split": split,
        }  # fmt: skip
    )
    total = float(table["gain"].sum()) or 1.0
    table["gain_share"] = table["gain"] / total
    return table.sort_values("gain", ascending=False).reset_index(drop=True)


def plot_importance(table: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = table.head(20)[::-1]
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(top) + 1))
    ax.barh(top["feature"], top["gain_share"])
    ax.set_xlabel("gain share")
    fig.tight_layout()
    fig.savefig(path, dpi=report.DPI)
    plt.close(fig)


def _tables(
    rows: Mapping[str, pd.DataFrame], references: Sequence[str], n_boot: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    main = report.main_table(rows, {n: {} for n in rows}, n_boot, seed)
    paired = pd.concat(
        [report.paired_table(rows, ref, PAIRED_METRICS, n_boot, seed) for ref in references],
        ignore_index=True,
    )
    return main, paired


@stage(
    "ranking.train",
    inputs=[TRAINING, TRAINING_META],
    outputs=[
        RANKER_TXT,
        FEATURE_LIST,
        RANKER_META,
        IMPORTANCE_CSV,
        IMPORTANCE_PNG,
        VS_BLEND,
        VS_BLEND_PAIRED,
    ],
    config_keys=["ranking.lgbm", "eval.ks", "eval.bootstrap_resamples"],
)
def train(ctx: StageContext) -> None:
    """Fit the LambdaRank ranker, save it with its feature list, report importances and the
    held-out comparison against the blend."""
    frame = pd.read_parquet(ctx.path(TRAINING))
    meta_in = json.loads(ctx.path(TRAINING_META).read_text(encoding="utf-8"))
    if list(meta_in["features"]) != list(features.FEATURES):
        raise ValueError("training.parquet was built with a different feature list; rebuild it")
    ctx.cutoff = int(meta_in["cutoff"])
    params = ctx.param("ranking.lgbm")
    ks = [int(k) for k in ctx.param("eval.ks")]
    n_boot = int(ctx.param("eval.bootstrap_resamples"))
    names = list(features.FEATURES)
    booster, best = fit_ranker(frame, names, params, ctx.seed)
    booster.save_model(str(ctx.out(RANKER_TXT)), num_iteration=best)
    ctx.out(FEATURE_LIST).write_text(json.dumps(names), encoding="utf-8")
    table = importance_table(booster, names)
    table.to_csv(ctx.out(IMPORTANCE_CSV), index=False)
    plot_importance(table, ctx.out(IMPORTANCE_PNG))

    valid = frame[frame["split"] == "valid"].reset_index(drop=True)
    rows: dict[str, pd.DataFrame] = {}
    if len(valid):
        pred = booster.predict(valid[names].to_numpy(np.float32), num_iteration=best)
        rows["ranker"] = held_out_rows(valid, pred, ks, "ranker")
        rows["blend"] = held_out_rows(valid, valid["blend_score"].to_numpy(), ks, "blend")
        cf = np.nan_to_num(valid["cf_score"].to_numpy(np.float32), nan=-1e9)
        rows["cf_score_only"] = held_out_rows(valid, cf, ks, "cf_score_only")
        main, paired = _tables(rows, ["blend"], n_boot, ctx.seed)
    else:
        main = pd.DataFrame(columns=["model", "users"])
        paired = pd.DataFrame(columns=["stratum", "model", "vs", "metric", "diff", "low", "high"])
    main.to_csv(ctx.out(VS_BLEND), index=False)
    paired.to_csv(ctx.out(VS_BLEND_PAIRED), index=False)

    top8 = table.head(8)
    meta = {
        "cutoff": ctx.cutoff,
        "params": dict(params),
        "best_iteration": best,
        "features": names,
        "rows_train": int((frame["split"] == "train").sum()),
        "rows_valid": int(len(valid)),
        "users_train": int(frame.loc[frame["split"] == "train", "u"].nunique()),
        "users_valid": int(valid["u"].nunique()) if len(valid) else 0,
        "top8_by_gain": top8["feature"].tolist(),
        "author_in_top8": bool((top8["feature"] == "author_affinity").any()),
        "aspect_group_in_top8": bool((top8["group"] == "ASPECT").any()),
        "held_out": main.to_dict(orient="records"),
        "paired_vs_blend": paired.to_dict(orient="records"),
    }
    ctx.out(RANKER_META).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    ctx.count("best_iteration", best)
    ctx.count("users_valid", meta["users_valid"])
    if len(main):
        ctx.notes["ndcg@10"] = {r["model"]: r.get("ndcg@10") for r in meta["held_out"]}
        ctx.log.info("held-out ndcg@10: %s", ctx.notes["ndcg@10"])


@stage(
    "ranking.ablation_val",
    inputs=[TRAINING, TRAINING_META],
    outputs=[ABLATION, ABLATION_PAIRED, ABLATION_PNG],
    config_keys=["ranking.lgbm", "eval.ks", "eval.bootstrap_resamples"],
)
def ablation_val(ctx: StageContext) -> None:
    """M4..M8b on the held-out users (s13.7, val only), with CIs and paired bootstraps."""
    frame = pd.read_parquet(ctx.path(TRAINING))
    meta_in = json.loads(ctx.path(TRAINING_META).read_text(encoding="utf-8"))
    ctx.cutoff = int(meta_in["cutoff"])
    params = ctx.param("ranking.lgbm")
    ks = [int(k) for k in ctx.param("eval.ks")]
    n_boot = int(ctx.param("eval.bootstrap_resamples"))
    valid = frame[frame["split"] == "valid"].reset_index(drop=True)
    if valid.empty:
        raise ValueError("no held-out users in training.parquet; the ablation needs them")
    rows: dict[str, pd.DataFrame] = {
        "blend": held_out_rows(valid, valid["blend_score"].to_numpy(), ks, "blend")
    }
    best_by_model: dict[str, int] = {}
    for model, groups in ABLATION_MASKS.items():
        names = feature_names(groups)
        booster, best = fit_ranker(frame, names, params, ctx.seed)
        best_by_model[model] = best
        pred = booster.predict(valid[names].to_numpy(np.float32), num_iteration=best)
        rows[model] = held_out_rows(valid, pred, ks, model)
        ctx.log.info("%s (%d features, %d trees): ndcg@10 %.4f", model, len(names), best,
                     rows[model]["ndcg@10"].mean())  # fmt: skip
    main, paired = _tables(rows, ["M4", "blend"], n_boot, ctx.seed)
    main["features"] = [
        len(feature_names(ABLATION_MASKS[m])) if m in ABLATION_MASKS else np.nan
        for m in main["model"]
    ]
    main["best_iteration"] = [best_by_model.get(m, np.nan) for m in main["model"]]
    main.to_csv(ctx.out(ABLATION), index=False)
    paired.to_csv(ctx.out(ABLATION_PAIRED), index=False)
    report.plot_bars(main, ctx.out(ABLATION_PNG))
    ctx.count("users_valid", int(valid["u"].nunique()))
    ctx.notes["ndcg@10"] = {
        m: float(v) for m, v in zip(main["model"], main["ndcg@10"], strict=True)
    }
    m6_vs_m4 = paired[(paired["model"] == "M6") & (paired["vs"] == "M4")
                      & (paired["metric"] == "ndcg@10")]  # fmt: skip
    if len(m6_vs_m4):
        d = m6_vs_m4.iloc[0]
        ctx.notes["M6_vs_M4_ndcg@10"] = {"diff": float(d["diff"]), "low": float(d["low"]),
                                         "high": float(d["high"])}  # fmt: skip
