"""Ranker training data (CLAUDE.md s13.4) and the candidate-recall diagnostic (s13.1), stage
``ranking.build_training``.

For every val-evaluated user (>= 1 relevant val interaction, >= 1 train interaction) the stage
builds a ``UserState`` from the user's train interactions plus their train-window review aspects
(``user_review_aspects.parquet`` as ``ReviewAnalysis`` rows), then runs the real ``litsense_core``
path over the val bundle (``serve.export_val``): profile -> seven candidate sources -> union ->
filters -> ``features.assemble``. Label 1 = the candidate is a val interaction rated >= the
relevance threshold (val items rated lower stay hard negatives). Users with no positive among the
candidates are dropped and counted. Train-split users keep every positive and up to
``negatives_per_positive`` sampled negatives; held-out users keep all candidates, so the ranker
and the blend can be compared on the same lists. D66: a seeded ``exact_cf_share`` of the users are
passed as corpus readers (exact LightGCN vector, ``cf_available`` 1), the rest get the pseudo-CF
vector a new reader would (0.5), so the ranker sees both regimes it serves.
"""

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from litsense_core import candidates as cand
from litsense_core import features
from litsense_core.profile import Profile, build_profile
from litsense_core.rank import blend
from litsense_core.schemas import RatingEvent, ReviewAnalysis, UserState
from litsense_core.store import ArtifactStore, read_store
from ml.common.io import load_id_map
from ml.common.stage import StageContext, stage
from ml.eval.data import EvalData, build_eval_data
from ml.serve.export import VAL_BUNDLE_DIR

RANKER = "eval_val/ranker"
TRAINING = f"artifacts:{RANKER}/training.parquet"
TRAINING_META = f"artifacts:{RANKER}/training_meta.json"
RECALL = "results:ranker/candidate_recall.csv"
REVIEW_ASPECTS = "artifacts:eval_val/aspects/user_review_aspects.parquet"
META_COLUMNS = ("u", "i", "label", "split", "cf_mode", "n_truth", "blend_score")


@dataclass
class UserExample:
    """One user's candidate list with labels, blend scores and features."""

    u: int
    ids: np.ndarray
    labels: np.ndarray
    blend: np.ndarray
    x: np.ndarray
    n_truth: int


@dataclass
class RecallStats:
    """Per-source candidate recall over the users that reached candidate generation."""

    users: int = 0
    found: dict[str, float] = field(default_factory=dict)  # source -> summed recall share
    unique: dict[str, float] = field(default_factory=dict)  # source -> summed unique share
    with_source: dict[str, int] = field(default_factory=dict)
    union: float = 0.0
    union_filtered: float = 0.0

    def add(
        self, sources: Mapping[str, list[int]], ids: list[int], filtered: list[int], truth: set[int]
    ) -> None:
        self.users += 1
        n = max(len(truth), 1)
        hits = {s: set(lst) & truth for s, lst in sources.items()}
        for s, h in hits.items():
            self.found[s] = self.found.get(s, 0.0) + len(h) / n
            self.with_source[s] = self.with_source.get(s, 0) + 1
            others = (
                set().union(*(o for t, o in hits.items() if t != s)) if len(hits) > 1 else set()
            )
            self.unique[s] = self.unique.get(s, 0.0) + len(h - others) / n
        self.union += len(set(ids) & truth) / n
        self.union_filtered += len(set(filtered) & truth) / n

    def table(self) -> pd.DataFrame:
        rows = [
            {
                "source": s,
                "recall": self.found[s] / max(self.users, 1),
                "unique_contribution": self.unique[s] / max(self.users, 1),
                "users_with_source": self.with_source[s],
            }
            for s in cand.SOURCE_ORDER
            if s in self.found
        ]
        rows.append({"source": "union", "recall": self.union / max(self.users, 1),
                     "unique_contribution": np.nan, "users_with_source": self.users})  # fmt: skip
        rows.append({"source": "union_after_filters",
                     "recall": self.union_filtered / max(self.users, 1),
                     "unique_contribution": np.nan, "users_with_source": self.users})  # fmt: skip
        return pd.DataFrame(rows)


def user_state(
    u: int,
    user_id: str | None,
    train_rows: pd.DataFrame,
    reviews: pd.DataFrame | None,
    aspect_names: list[str],
) -> UserState:
    """``UserState`` from a user's train interactions (ratings with timestamps) and their
    train-window review aspects; ``user_id`` set makes them a corpus reader (exact CF vector)."""
    ratings = [
        RatingEvent(book_id=int(i), stars=float(r), ts=int(t))
        for i, r, t in zip(train_rows["i"], train_rows["rating"], train_rows["ts"], strict=True)
    ]
    analyses: list[ReviewAnalysis] = []
    if reviews is not None and len(reviews):
        for (i, _row), group in reviews.groupby(["i", "review_row"], sort=False):
            analyses.append(
                ReviewAnalysis(
                    book_id=int(i),
                    ts=int(group["ts"].iloc[0]),
                    aspects={
                        aspect_names[int(a)]: float(p)
                        for a, p in zip(group["aspect"], group["polarity"], strict=True)
                    },
                )
            )
    return UserState(user_id=u, corpus_user_id=user_id, ratings=ratings, review_analyses=analyses)


def cutoff_year(cutoff_ms: int) -> int:
    return datetime.fromtimestamp(cutoff_ms / 1000, tz=UTC).year


def candidate_lists(
    profile: Profile, store: ArtifactStore, min_reviews: int, max_year: int | None = None
) -> tuple[dict[str, list[int]], list[int], dict[int, list[str]], list[int]]:
    """Sources, union ids with provenance, and the ids left after the s13.2 filters. With
    ``max_year`` (offline, the cutoff year), books dated after it are dropped too: the catalogue
    is static, so without this a val-window release reaches the candidates before it exists and
    its age gives the ranker a signal serving never has (D67)."""
    cfg = store.config
    sources = cand.generate(profile, store, cfg["quotas"][profile.bucket])
    ids, provenance = cand.union(sources, int(cfg["candidates_total"]))
    kept = cand.apply_filters(ids, provenance, profile, store, None, None, min_reviews)
    if max_year is not None:
        year = store.catalog.year
        kept = [i for i in kept if int(year[i]) <= max_year]
    return sources, ids, provenance, kept


def build_example(
    u: int,
    state: UserState,
    truth: np.ndarray,
    store: ArtifactStore,
    min_reviews: int,
    cutoff: int,
    stats: RecallStats,
) -> UserExample | None:
    """Candidates, labels, blend scores and features for one user; None without a positive."""
    profile = build_profile(state, store)
    sources, ids, provenance, kept = candidate_lists(
        profile, store, min_reviews, cutoff_year(cutoff)
    )
    truth_set = {int(t) for t in truth.tolist()}
    stats.add(sources, ids, kept, truth_set)
    if not kept:
        return None
    arr = np.asarray(kept, dtype=np.int64)
    labels = np.isin(arr, truth).astype(np.int8)
    if not labels.any():
        return None
    scores, _ = blend(kept, sources, profile, store, store.config["blend"])
    x = features.assemble(profile, kept, provenance, store, expected_cutoff=cutoff)
    return UserExample(u, arr, labels, scores, x, len(truth_set))


def sample_negatives(ex: UserExample, per_positive: int, rng: np.random.Generator) -> np.ndarray:
    """Row mask keeping every positive and up to ``per_positive`` negatives per positive."""
    pos = np.flatnonzero(ex.labels == 1)
    neg = np.flatnonzero(ex.labels == 0)
    budget = per_positive * len(pos)
    if len(neg) > budget:
        neg = rng.choice(neg, size=budget, replace=False)
    keep = np.zeros(len(ex.labels), dtype=bool)
    keep[pos] = True
    keep[neg] = True
    return keep


def to_frame(ex: UserExample, keep: np.ndarray, split: str, cf_mode: str) -> pd.DataFrame:
    frame = pd.DataFrame(ex.x[keep], columns=list(features.FEATURES))
    frame.insert(0, "blend_score", ex.blend[keep].astype(np.float32))
    frame.insert(0, "n_truth", np.int32(ex.n_truth))
    frame.insert(0, "cf_mode", cf_mode)
    frame.insert(0, "split", split)
    frame.insert(0, "label", ex.labels[keep])
    frame.insert(0, "i", ex.ids[keep].astype(np.int32))
    frame.insert(0, "u", np.int32(ex.u))
    return frame


def assign_users(
    users: np.ndarray, train_share: float, exact_share: float, seed: int
) -> tuple[dict[int, str], dict[int, str]]:
    """Seeded user -> split ("train"/"valid") and user -> cf mode ("exact"/"pseudo")."""
    rng = np.random.default_rng(seed)
    draw_split = rng.random(len(users))
    draw_cf = rng.random(len(users))
    split = {
        int(u): "train" if d < train_share else "valid"
        for u, d in zip(users, draw_split, strict=True)
    }
    cf = {
        int(u): "exact" if d < exact_share else "pseudo"
        for u, d in zip(users, draw_cf, strict=True)
    }
    return split, cf


def build_training_frame(
    store: ArtifactStore,
    data: EvalData,
    train_by_user: dict[int, pd.DataFrame],
    reviews_by_user: dict[int, pd.DataFrame],
    user_ids: list[str],
    aspect_names: list[str],
    cfg: Mapping[str, Any],
    seed: int,
    log: Any = None,
) -> tuple[pd.DataFrame, RecallStats, dict[str, Any]]:
    """The training table over ``data.users`` (sorted, so groups are contiguous)."""
    split_of, cf_of = assign_users(
        data.users, float(cfg["ranker_user_split"]), float(cfg["exact_cf_share"]), seed
    )
    rng = np.random.default_rng(seed + 1)
    stats = RecallStats()
    frames: list[pd.DataFrame] = []
    dropped = 0
    started = time.monotonic()
    for k, u in enumerate(data.users.tolist()):
        state = user_state(
            u,
            user_ids[u] if cf_of[u] == "exact" else None,
            train_by_user[u],
            reviews_by_user.get(u),
            aspect_names,
        )
        ex = build_example(
            u, state, data.truth[u], store, int(cfg["min_reviews_main_feed"]), data.cutoff, stats
        )
        if ex is None:
            dropped += 1
            continue
        keep = (
            sample_negatives(ex, int(cfg["negatives_per_positive"]), rng)
            if split_of[u] == "train"
            else np.ones(len(ex.labels), dtype=bool)
        )
        frames.append(to_frame(ex, keep, split_of[u], cf_of[u]))
        if log is not None and (k + 1) % 500 == 0:
            log.info("%d / %d users, %.0f s", k + 1, len(data.users), time.monotonic() - started)
    columns = [*META_COLUMNS, *features.FEATURES]
    frame = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame({c: pd.Series(dtype=float) for c in columns})
    )

    def users_where(column: str, value: str) -> int:
        return int(frame.loc[frame[column] == value, "u"].nunique()) if len(frame) else 0

    counts = {
        "users_evaluated": int(len(data.users)),
        "users_kept": int(len(frames)),
        "users_dropped_no_positive": int(dropped),
        "rows": int(len(frame)),
        "positives": int(frame["label"].sum()) if len(frame) else 0,
        "users_train": users_where("split", "train"),
        "users_valid": users_where("split", "valid"),
        "users_exact_cf": users_where("cf_mode", "exact"),
    }
    return frame[columns], stats, counts


@stage(
    "ranking.build_training",
    inputs=[
        VAL_BUNDLE_DIR,
        "data:interactions.parquet",
        "data:splits.json",
        "data:id_map.json",
        REVIEW_ASPECTS,
    ],
    outputs=[TRAINING, TRAINING_META, RECALL],
    config_keys=[
        "ranking.candidates_total",
        "ranking.quotas",
        "ranking.buckets",
        "ranking.min_reviews_main_feed",
        "ranking.negatives_per_positive",
        "ranking.ranker_user_split",
        "ranking.exact_cf_share",
        "ranking.blend",
        "data.relevance_min_rating",
        "data.longtail_fraction",
        "aspects.names",
    ],
)
def build_training(ctx: StageContext) -> None:
    """Ranker training table over the val bundle, plus the candidate-recall diagnostic."""
    ids = load_id_map(ctx.path("data:id_map.json"))
    splits = json.loads(ctx.path("data:splits.json").read_text(encoding="utf-8"))
    inter = pd.read_parquet(ctx.path("data:interactions.parquet"))
    data = build_eval_data(
        inter, splits, "val", ids.n_users, ids.n_items,
        float(ctx.param("data.relevance_min_rating")), float(ctx.param("data.longtail_fraction")),
    )  # fmt: skip
    store = read_store(ctx.path(VAL_BUNDLE_DIR))
    if store.cutoff != data.cutoff:
        raise ValueError(f"val bundle cutoff {store.cutoff} != t1 {data.cutoff}")
    ctx.cutoff = data.cutoff
    train = inter.loc[inter["split"] == "train", ["u", "i", "rating", "ts"]]
    train_by_user = {int(u): g for u, g in train.groupby("u", sort=False)}
    reviews = pd.read_parquet(ctx.path(REVIEW_ASPECTS))
    reviews_by_user = {int(u): g for u, g in reviews.groupby("u", sort=False)}
    cfg = {
        key: ctx.param(f"ranking.{key}")
        for key in (
            "min_reviews_main_feed",
            "negatives_per_positive",
            "ranker_user_split",
            "exact_cf_share",
        )
    }
    frame, stats, counts = build_training_frame(
        store, data, train_by_user, reviews_by_user, ids.users,
        list(ctx.param("aspects.names")), cfg, ctx.seed, ctx.log,
    )  # fmt: skip
    frame.to_parquet(ctx.out(TRAINING), index=False, compression="zstd")
    recall = stats.table()
    recall.to_csv(ctx.out(RECALL), index=False)
    meta = {
        "cutoff": data.cutoff,
        "bundle_version": store.version,
        "features": list(features.FEATURES),
        "candidate_recall": recall.to_dict(orient="records"),
        **counts,
    }
    ctx.out(TRAINING_META).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    for key, value in counts.items():
        ctx.count(key, value)
    union = recall.loc[recall["source"] == "union", "recall"]
    ctx.notes["candidate_recall_union"] = float(union.iloc[0]) if len(union) else None
