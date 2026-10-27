"""The ranker under the s8 protocol on every val user, stage ``ranking.eval_val`` (D66, D67).

``ranking.train`` reports held-out users whose candidates hold a positive, so its numbers are not
comparable with ``results/<v>/val/table_main.csv``. This stage scores all evaluated val users the
way serving would (profile -> seven sources -> filters -> features -> ranker, over the val bundle),
puts every non-candidate below every candidate (ordered by popularity, so no user's row is
degenerate) and runs the same ``evaluate`` as ``eval.val``: full-catalogue top-K, sparsity strata,
long-tail and cold-item truth, beyond-accuracy metrics, paired bootstraps against popularity and
both LightGCN variants from the stored per-user rows.

D67: the ranker is trained on val labels, so a user must never be scored by a ranker that saw
their own val positives. Users are split into ``ranking.eval_folds`` seeded folds and each fold is
scored by a ranker fit on the training rows of the other folds (cross-fitting). Books dated after
the cutoff year are dropped from the candidates (``candidate_lists(max_year=...)``) because the
static catalogue would otherwise hand the ranker val-window releases.

Four rows: the ranker with exact corpus vectors for every reader (the information LightGCN has),
the ranker with the pseudo-CF vector a new reader would get, the blend over the same candidates
(what served before the ranker), and ``ranker_home``: the home screen as the reader sees it, the
``continue_series`` rail (the exact next volumes the s13.2 feed rule sets aside) above the ranked
feed. The filter diagnostic found those next volumes to be 7% of the positives the sources reach.
"""

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from litsense_core import candidates as cand
from litsense_core import features
from litsense_core.profile import Profile, build_profile
from litsense_core.rank import blend
from litsense_core.store import ArtifactStore, read_store
from ml.common.io import load_id_map
from ml.common.stage import StageContext, stage
from ml.eval import report
from ml.eval.data import EvalData, FitData, build_eval_data
from ml.eval.evaluator import EvalResult, evaluate
from ml.eval.strata import by_sparsity, cold_item_rows, longtail_rows
from ml.ranking.build_training import (
    REVIEW_ASPECTS,
    TRAINING,
    TRAINING_META,
    candidate_lists,
    cutoff_year,
    user_state,
)
from ml.ranking.train_ranker import fit_ranker
from ml.serve.export import VAL_BUNDLE_DIR

REFERENCES = ("popularity", "lightgcn_raw", "lightgcn_norm")
PAIRED_METRICS = ("ndcg@10", "recall@20")
OUT = "results:ranker"
OUTPUTS = {
    "main": f"{OUT}/eval_val_main.csv",
    "strata": f"{OUT}/eval_val_strata.csv",
    "longtail": f"{OUT}/eval_val_longtail.csv",
    "bootstrap": f"{OUT}/eval_val_bootstrap.csv",
    "bars": f"{OUT}/eval_val_bars.png",
    "summary": f"{OUT}/eval_val.json",
}
NON_CANDIDATE = -1e6  # below any ranker or blend score; popularity breaks ties among them
RAIL = 1e6  # above any ranker or blend score: the continue_series rail (home-screen row)

# (user, profile, candidate ids, provenance, sources) -> one score per candidate
ScoreFn = Callable[
    [int, Profile, list[int], dict[int, list[str]], dict[str, list[int]]], np.ndarray
]


def assign_folds(users: np.ndarray, n_folds: int, seed: int) -> dict[int, int]:
    """Seeded user -> fold in ``range(n_folds)``."""
    rng = np.random.default_rng(seed)
    return {
        int(u): int(f) for u, f in zip(users, rng.integers(0, n_folds, len(users)), strict=True)
    }


def cross_fit(
    frame: pd.DataFrame, fold_of: dict[int, int], n_folds: int, params: Any, seed: int
) -> list[tuple[Any, int]]:
    """One ``(booster, best_iteration)`` per fold, fit on the training rows of the other folds."""
    folds = frame["u"].map(fold_of).to_numpy()
    out = []
    for k in range(n_folds):
        rows = frame[folds != k]
        if rows.empty or not rows["label"].any():
            raise ValueError(f"fold {k}: no training rows with a positive outside the fold")
        out.append(fit_ranker(rows, list(features.FEATURES), params, seed))
    return out


class CandidateScorer:
    """``ml.eval.evaluator.Scorer`` over the serving path: candidates scored by ``score_fn``,
    everything else ranked last by popularity."""

    def __init__(
        self,
        name: str,
        store: ArtifactStore,
        train_by_user: dict[int, pd.DataFrame],
        reviews_by_user: dict[int, pd.DataFrame],
        user_ids: list[str],
        aspect_names: list[str],
        exact_cf: bool,
        score_fn: ScoreFn,
        min_reviews: int,
        include_rail: bool = False,
    ) -> None:
        self.name = name
        self.include_rail = include_rail  # home screen: continue_series rail above the feed
        self.store = store
        self.train_by_user = train_by_user
        self.reviews_by_user = reviews_by_user
        self.user_ids = user_ids
        self.aspect_names = aspect_names
        self.exact_cf = exact_cf
        self.score_fn = score_fn
        self.min_reviews = min_reviews
        self.cutoff: int | None = int(store.cutoff)
        self.max_year = cutoff_year(int(store.cutoff))
        self.n_candidates: list[int] = []

    def fit(self, data: FitData) -> None:
        if data.cutoff != self.store.cutoff:
            raise ValueError(f"val bundle cutoff {self.store.cutoff} != fit cutoff {data.cutoff}")
        self.floor = (NON_CANDIDATE + self.store.catalog.popularity).astype(np.float32)

    def score(self, users: np.ndarray) -> np.ndarray:
        out = np.tile(self.floor, (len(users), 1))
        for r, u in enumerate(users.tolist()):
            state = user_state(
                u,
                self.user_ids[u] if self.exact_cf else None,
                self.train_by_user[u],
                self.reviews_by_user.get(u),
                self.aspect_names,
            )
            profile = build_profile(state, self.store)
            sources, _ids, provenance, kept = candidate_lists(
                profile, self.store, self.min_reviews, self.max_year
            )
            self.n_candidates.append(len(kept))
            if kept:
                out[r, np.asarray(kept, dtype=np.int64)] = self.score_fn(
                    u, profile, kept, provenance, sources
                )
            if self.include_rail:
                # The exact next volumes the feed excludes (s13.2) sit in the continue_series rail
                # at the top of the home screen; count them above the feed, in rail order.
                rail = cand.next_in_series(self.store, profile)
                for pos, i in enumerate(rail):
                    out[r, i] = RAIL + len(rail) - pos
        return out


def ranker_fn(
    boosters: list[tuple[Any, int]], fold_of: dict[int, int], store: ArtifactStore
) -> ScoreFn:
    """Each user is scored by the fold ranker that never saw their val labels (D67)."""

    def score(
        u: int, profile: Profile, ids: list[int], prov: dict[int, list[str]], _s: Any
    ) -> np.ndarray:
        booster, best = boosters[fold_of[u]]
        x = features.assemble(profile, ids, prov, store, expected_cutoff=store.cutoff)
        return np.asarray(booster.predict(x, num_iteration=best), dtype=np.float32)

    return score


def blend_fn(store: ArtifactStore) -> ScoreFn:
    def score(
        _u: int, profile: Profile, ids: list[int], _p: Any, sources: dict[str, list[int]]
    ) -> np.ndarray:
        return blend(ids, sources, profile, store, store.config["blend"])[0]

    return score


def _summary(
    results: dict[str, EvalResult],
    data: EvalData,
    scorers: list[CandidateScorer],
    n_folds: int,
    best_iterations: list[int],
) -> dict[str, Any]:
    return {
        "cutoff": data.cutoff,
        "users": int(len(data.users)),
        "folds": n_folds,
        "best_iteration_per_fold": best_iterations,
        "means": {n: r.means for n, r in results.items()},
        "beyond": {n: r.beyond for n, r in results.items()},
        "candidates_per_user": {
            s.name: {
                "mean": float(np.mean(s.n_candidates)) if s.n_candidates else 0.0,
                "min": int(min(s.n_candidates)) if s.n_candidates else 0,
            }
            for s in scorers
        },
    }


@stage(
    "ranking.eval_val",
    inputs=[
        VAL_BUNDLE_DIR,
        TRAINING,
        TRAINING_META,
        REVIEW_ASPECTS,
        "data:interactions.parquet",
        "data:splits.json",
        "data:id_map.json",
        "artifacts:content/content_item.npy",
        "results:val/",
    ],
    outputs=list(OUTPUTS.values()),
    config_keys=[
        "eval",
        "data.relevance_min_rating",
        "data.longtail_fraction",
        "data.sparsity_buckets",
        "ranking.min_reviews_main_feed",
        "ranking.eval_folds",
        "ranking.lgbm",
        "aspects.names",
    ],
)
def eval_val(ctx: StageContext) -> None:
    """Cross-fitted ranker and blend on every val user under the s8 protocol, next to the stored
    baselines."""
    ids = load_id_map(ctx.path("data:id_map.json"))
    splits = json.loads(ctx.path("data:splits.json").read_text(encoding="utf-8"))
    inter = pd.read_parquet(ctx.path("data:interactions.parquet"))
    data = build_eval_data(
        inter, splits, "val", ids.n_users, ids.n_items,
        float(ctx.param("data.relevance_min_rating")), float(ctx.param("data.longtail_fraction")),
    )  # fmt: skip
    ctx.cutoff = data.cutoff
    store = read_store(ctx.path(VAL_BUNDLE_DIR))
    frame = pd.read_parquet(ctx.path(TRAINING))
    meta_in = json.loads(ctx.path(TRAINING_META).read_text(encoding="utf-8"))
    if list(meta_in["features"]) != list(features.FEATURES):
        raise ValueError("training.parquet was built with a different feature list; rebuild it")
    if int(meta_in["cutoff"]) != data.cutoff:
        raise ValueError("training.parquet cutoff differs from t1")
    n_folds = int(ctx.param("ranking.eval_folds"))
    fold_of = assign_folds(data.users, n_folds, ctx.seed)
    if not set(frame["u"].unique()) <= set(fold_of):
        raise ValueError("training.parquet holds users that are not evaluated val users")
    boosters = cross_fit(frame, fold_of, n_folds, ctx.param("ranking.lgbm"), ctx.seed)
    ctx.log.info("cross-fit: %d folds, best iterations %s", n_folds, [b for _, b in boosters])

    train = inter.loc[inter["split"] == "train", ["u", "i", "rating", "ts"]]
    train_by_user = {int(u): g for u, g in train.groupby("u", sort=False)}
    reviews = pd.read_parquet(ctx.path(REVIEW_ASPECTS))
    reviews_by_user = {int(u): g for u, g in reviews.groupby("u", sort=False)}
    common = (train_by_user, reviews_by_user, ids.users, list(ctx.param("aspects.names")))
    min_reviews = int(ctx.param("ranking.min_reviews_main_feed"))
    ranker = ranker_fn(boosters, fold_of, store)
    scorers = [
        CandidateScorer("ranker", store, *common, True, ranker, min_reviews),
        CandidateScorer("ranker_pseudo_cf", store, *common, False, ranker, min_reviews),
        CandidateScorer("blend", store, *common, True, blend_fn(store), min_reviews),
        CandidateScorer(
            "ranker_home", store, *common, True, ranker, min_reviews, include_rail=True
        ),
    ]
    emb = np.load(ctx.path("artifacts:content/content_item.npy"))
    ev = ctx.param("eval")
    ks, n_boot = tuple(int(k) for k in ev["ks"]), int(ev["bootstrap_resamples"])
    buckets = ctx.param("data.sparsity_buckets")

    results: dict[str, EvalResult] = {}
    tail_rows, cold_rows, strata = {}, {}, []
    for scorer in scorers:
        scorer.fit(data.fit)
        result = evaluate(scorer, data, ks, embeddings=emb)
        results[scorer.name] = result
        tail_rows[scorer.name] = longtail_rows(result, data, ks)
        cold_rows[scorer.name] = cold_item_rows(result, data, ks)
        strata.append(by_sparsity(result, data, buckets).assign(model=scorer.name))
        ctx.log.info(
            "%s: ndcg@10 %.4f (%.0f candidates per user)",
            scorer.name,
            result.means.get("ndcg@10", float("nan")),
            np.mean(scorer.n_candidates),
        )

    # Stored baseline rows (eval.val) for the paired comparisons and the shared table.
    val_dir = ctx.path("results:val/")
    rows = {n: r.rows for n, r in results.items()}
    beyond = {n: r.beyond for n, r in results.items()}
    for ref in REFERENCES:
        path = val_dir / f"{ref}.users.parquet"
        if path.exists():
            rows[ref] = pd.read_parquet(path)
            beyond[ref] = {}
    main = report.main_table(rows, beyond, n_boot, ctx.seed)
    main.to_csv(ctx.out(OUTPUTS["main"]), index=False)
    pd.concat(strata, ignore_index=True).to_csv(ctx.out(OUTPUTS["strata"]), index=False)
    longtail = report.main_table(tail_rows, {n: {} for n in tail_rows}, n_boot, ctx.seed).assign(
        stratum="longtail"
    )
    cold = report.main_table(
        {n: r for n, r in cold_rows.items() if len(r)}, {n: {} for n in cold_rows}, n_boot, ctx.seed
    ).assign(stratum="cold_items")
    pd.concat([longtail, cold], ignore_index=True).to_csv(ctx.out(OUTPUTS["longtail"]), index=False)
    paired = pd.concat(
        [report.paired_table(rows, ref, PAIRED_METRICS, n_boot, ctx.seed)
         for ref in REFERENCES if ref in rows]
        + [report.paired_table(rows, "blend", PAIRED_METRICS, n_boot, ctx.seed)],
        ignore_index=True,
    )  # fmt: skip
    paired = paired[paired["model"].isin(results)].reset_index(drop=True)
    paired.to_csv(ctx.out(OUTPUTS["bootstrap"]), index=False)
    report.plot_bars(main, ctx.out(OUTPUTS["bars"]))
    summary = _summary(results, data, scorers, n_folds, [b for _, b in boosters])
    ctx.out(OUTPUTS["summary"]).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    ctx.count("evaluated_users", len(data.users))
    ctx.notes["ndcg@10"] = {n: r.means.get("ndcg@10") for n, r in results.items()}
