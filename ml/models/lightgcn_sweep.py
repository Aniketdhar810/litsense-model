"""LightGCN hyperparameter sweep on the inner validation split (D47).

Stage ``models.lightgcn_sweep`` trains one LightGCN per (dim, reg) in ``models.lightgcn_sweep`` with
every other setting from ``models.lightgcn``, scores each on the inner split exactly like
``models.lightgcn_val`` does, and writes one row per configuration to
``results/<v>/training/lightgcn_sweep.csv``. Nothing is saved to ``artifacts/`` and ``eval.val``
never reads it: the user reads the table, edits ``models.lightgcn`` and reruns ``run all``.

It registers last, so ``run all --until eval.val`` does not run it; run it on its own:
``python -m ml.stages run models.lightgcn_sweep``.
"""

import itertools
import json
import time
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd
import torch

from ml.common.io import load_id_map
from ml.common.stage import StageContext, stage
from ml.eval.data import build_eval_data
from ml.models.lightgcn import (
    embedding_health,
    inner_eval_data,
    inner_metric,
    train_lightgcn,
)

SWEEP_CSV = "results:training/lightgcn_sweep.csv"
COLUMNS = ("dim", "reg", "batch_size", "best_epoch", "search_epochs", "best_inner_metric",
           "final_bpr", "mean_pairwise_cosine", "norm_median", "seconds")  # fmt: skip


def grid(sweep: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every (dim, reg) pair of ``models.lightgcn_sweep``, in config order."""
    dims: Iterable[int] = sweep["dims"]
    regs: Iterable[float] = sweep["regs"]
    return [{"dim": int(d), "reg": float(r)} for d, r in itertools.product(dims, regs)]


@stage(
    "models.lightgcn_sweep",
    inputs=["data:interactions.parquet", "data:splits.json", "data:id_map.json"],
    outputs=[SWEEP_CSV],
    config_keys=[
        "models.lightgcn_sweep",
        "models.lightgcn",
        "data.relevance_min_rating",
        "data.longtail_fraction",
    ],
)
def build_lightgcn_sweep(ctx: StageContext) -> None:
    """Inner-split early-stopping search over the (dim, reg) grid; one CSV row per point."""
    ids = load_id_map(ctx.path("data:id_map.json"))
    splits = json.loads(ctx.path("data:splits.json").read_text(encoding="utf-8"))
    relevance = float(ctx.param("data.relevance_min_rating"))
    longtail = float(ctx.param("data.longtail_fraction"))
    data = build_eval_data(
        pd.read_parquet(ctx.path("data:interactions.parquet")),
        splits, "val", ids.n_users, ids.n_items, relevance, longtail,
    )  # fmt: skip
    ctx.cutoff = data.cutoff
    base = dict(ctx.param("models.lightgcn"))
    sweep = ctx.param("models.lightgcn_sweep")
    max_epochs = int(sweep.get("max_epochs", base["max_epochs"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inner = inner_eval_data(
        data, float(base["inner_val_fraction"]), int(base["inner_val_min_user"]),
        relevance, longtail, str(base.get("inner_split", "user")),
    )  # fmt: skip
    if not len(inner.users):
        raise ValueError("no inner validation users; the sweep has nothing to score")
    metric = inner_metric(inner, str(base["metric"]))

    rows = []
    points = grid(sweep)
    for n, point in enumerate(points, 1):
        params = base | point
        ctx.log.info("[%d/%d] dim %d, reg %g on %s", n, len(points), point["dim"], point["reg"],
                     device)  # fmt: skip
        started = time.monotonic()
        result = train_lightgcn(
            inner.fit.interactions, ids.n_users, ids.n_items, params, ctx.seed, max_epochs,
            device, metric,
        )  # fmt: skip
        health = embedding_health(result.item, inner.item_counts > 0)
        rows.append({
            "dim": point["dim"],
            "reg": point["reg"],
            "batch_size": int(params["batch_size"]),
            "best_epoch": result.best_epoch,
            "search_epochs": result.epochs,
            "best_inner_metric": result.best_metric,
            "final_bpr": result.history[-1]["bpr"] if result.history else float("nan"),
            "mean_pairwise_cosine": health["mean_pairwise_cosine"],
            "norm_median": health["norm_median"],
            "seconds": round(time.monotonic() - started, 1),
        })  # fmt: skip
        ctx.log.info("[%d/%d] best %s %.4f at epoch %d (%.0f s)", n, len(points), base["metric"],
                     result.best_metric, result.best_epoch, rows[-1]["seconds"])  # fmt: skip
    table = pd.DataFrame(rows, columns=list(COLUMNS))
    table.to_csv(ctx.out(SWEEP_CSV), index=False)
    best = table.sort_values("best_inner_metric", ascending=False).iloc[0]
    ctx.notes["best"] = {"dim": int(best["dim"]), "reg": float(best["reg"]),
                         "best_epoch": int(best["best_epoch"]),
                         "best_inner_metric": float(best["best_inner_metric"])}  # fmt: skip
    ctx.count("sweep_points", len(points))
