"""Stage eval.val (CLAUDE.md s8-9): every model on validation, with strata, bootstrap intervals,
paired comparisons, tables and figures.

Baselines are cheap to fit, so they are refit here instead of keeping model artifacts (D40).
LightGCN and the projection hybrid (with its shuffled control) are trained by their own stages and
loaded from ``artifacts/<v>/eval_val/``; their artifact metadata is copied next to their results
(s2.2 #5). Static fusion (s9.5 control) is built here, one row per ``models.static_fusion_alphas``
value. Every model is compared, paired, with popularity and with both LightGCN variants (s2.2 #1).
"""

import json

import numpy as np
import pandas as pd

from ml.common.io import load_id_map
from ml.common.stage import StageContext, stage
from ml.eval import report
from ml.eval.data import build_eval_data
from ml.eval.evaluator import EvalResult, evaluate
from ml.eval.strata import by_sparsity, cold_item_rows, longtail_rows
from ml.models.baselines import (
    ALSScorer,
    ContentScorer,
    ItemKNNScorer,
    PopularityScorer,
    RandomScorer,
)
from ml.models.embedding_scorers import EmbeddingArtifact, EmbeddingScorer, StaticFusionScorer

REFERENCES = ("popularity", "lightgcn_raw", "lightgcn_norm")
PAIRED_METRICS = ("ndcg@10", "recall@20")
LIGHTGCN_INPUTS = [
    f"artifacts:eval_val/lightgcn/{name}"
    for name in ("lgcn_user.npy", "lgcn_item.npy", "item_degree.npy", "meta.json")
]
PROJECTIONS = ("projection", "projection_shuffled")
PROJECTION_INPUTS = [
    f"artifacts:eval_val/{model}/{name}"
    for model in PROJECTIONS
    for name in ("user.npy", "proj_item.npy", "item_degree.npy", "meta.json")
]


@stage(
    "eval.val",
    inputs=[
        "data:interactions.parquet",
        "data:splits.json",
        "data:id_map.json",
        "artifacts:content/content_item.npy",
        *LIGHTGCN_INPUTS,
        *PROJECTION_INPUTS,
    ],
    outputs=["results:val/"],
    config_keys=[
        "eval",
        "data.relevance_min_rating",
        "data.longtail_fraction",
        "data.sparsity_buckets",
        "models.popularity",
        "models.itemknn",
        "models.als",
        "models.static_fusion_alphas",
    ],
)
def build_val(ctx: StageContext) -> None:
    """Validation results for the baselines, LightGCN, the projection hybrid and controls."""
    out = ctx.out("results:val/")
    ids = load_id_map(ctx.path("data:id_map.json"))
    splits = json.loads(ctx.path("data:splits.json").read_text(encoding="utf-8"))
    relevance = float(ctx.param("data.relevance_min_rating"))
    data = build_eval_data(
        pd.read_parquet(ctx.path("data:interactions.parquet")),
        splits,
        "val",
        ids.n_users,
        ids.n_items,
        relevance,
        float(ctx.param("data.longtail_fraction")),
    )
    ctx.cutoff = data.cutoff
    emb = np.load(ctx.path("artifacts:content/content_item.npy"))
    ev = ctx.param("eval")
    ks, n_boot = tuple(int(k) for k in ev["ks"]), int(ev["bootstrap_resamples"])
    buckets = ctx.param("data.sparsity_buckets")
    lightgcn = EmbeddingArtifact.load(
        ctx.path(LIGHTGCN_INPUTS[0]).parent, "lgcn_user.npy", "lgcn_item.npy"
    )
    scorers = [
        RandomScorer(ctx.seed),
        PopularityScorer(ctx.param("models.popularity")),
        ItemKNNScorer(ctx.param("models.itemknn")),
        ALSScorer(ctx.param("models.als"), ctx.seed),
        ContentScorer(emb, relevance),
        EmbeddingScorer("lightgcn_raw", lightgcn, normalize_items=False),
        EmbeddingScorer("lightgcn_norm", lightgcn, normalize_items=True),
    ]
    for model in PROJECTIONS:
        folder = ctx.path(f"artifacts:eval_val/{model}/meta.json").parent
        artifact = EmbeddingArtifact.load(folder, "user.npy", "proj_item.npy")
        scorers.append(
            EmbeddingScorer(
                model, artifact, bool(artifact.meta["normalize_items"]), score_cold=True
            )
        )
    scorers += [
        StaticFusionScorer(lightgcn, emb, float(alpha), relevance)
        for alpha in ctx.param("models.static_fusion_alphas")
    ]

    results: dict[str, EvalResult] = {}
    tail_rows, cold_rows, strata = {}, {}, []
    for scorer in scorers:
        scorer.fit(data.fit)
        result = evaluate(scorer, data, ks, embeddings=emb)
        results[scorer.name] = result
        tail_rows[scorer.name] = longtail_rows(result, data, ks)
        cold_rows[scorer.name] = cold_item_rows(result, data, ks)
        strata.append(by_sparsity(result, data, buckets).assign(model=scorer.name))
        result.rows.to_parquet(out / f"{scorer.name}.users.parquet", index=False)
        artifact = getattr(scorer, "artifact", None)
        summary = {
            "model": scorer.name,
            "mode": "val",
            "cutoff": data.cutoff,
            "users": len(result.rows),
            "means": result.means,
            "beyond": result.beyond,
            "longtail_means": tail_rows[scorer.name].drop(columns="u").mean().to_dict(),
            "cold_item_means": cold_rows[scorer.name].drop(columns="u").mean().to_dict()
            if len(cold_rows[scorer.name])
            else {},
            "artifact": artifact.meta if artifact is not None else None,
        }
        (out / f"{scorer.name}.json").write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
        ctx.log.info("%s: ndcg@10 %.4f", scorer.name, result.means.get("ndcg@10", float("nan")))

    rows = {n: r.rows for n, r in results.items()}
    beyond = {n: r.beyond for n, r in results.items()}
    main = report.main_table(rows, beyond, n_boot, ctx.seed)
    main.to_csv(out / "table_main.csv", index=False)
    strata_table = pd.concat(strata, ignore_index=True)
    strata_table.to_csv(out / "table_strata.csv", index=False)
    longtail = report.main_table(tail_rows, {n: {} for n in tail_rows}, n_boot, ctx.seed).assign(
        stratum="longtail"
    )
    cold = report.main_table(
        {n: r for n, r in cold_rows.items() if len(r)}, {n: {} for n in cold_rows}, n_boot, ctx.seed
    ).assign(stratum="cold_items")
    pd.concat([longtail, cold], ignore_index=True).to_csv(out / "table_longtail.csv", index=False)
    paired = pd.concat(
        [
            table
            for ref in REFERENCES
            for table in (
                report.paired_table(rows, ref, PAIRED_METRICS, n_boot, ctx.seed),
                report.paired_table(tail_rows, ref, PAIRED_METRICS, n_boot, ctx.seed, "longtail"),
            )
        ],
        ignore_index=True,
    )
    paired.to_csv(out / "bootstrap.csv", index=False)
    report.plot_bars(main, out / "bars_ndcg_recall.png")
    report.plot_by_bucket(strata_table, out / "ndcg_by_bucket.png")

    ctx.count("evaluated_users", len(data.users))
    ctx.count("catalogue_items", ids.n_items)
    ctx.notes["ndcg@10"] = {n: r.means.get("ndcg@10") for n, r in results.items()}
