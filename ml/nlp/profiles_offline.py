"""Item and corpus-user aspect profiles per cutoff (CLAUDE.md s11.8), stages nlp.profiles_val and
nlp.profiles_serve.

Per item ``i`` and aspect ``a``, over sentences of reviews with ``ts < cutoff``:

* weight ``w = 1 + log1p(helpful_vote)`` per sentence mention; ``raw = sum(w * polarity) / sum(w)``;
* ``n`` = distinct reviews mentioning ``a``; ``shrunk = (n * raw + k * mean_a) / (n + k)`` with
  ``mean_a`` the weighted global mean polarity of ``a`` and ``k = aspects.shrinkage_k``;
* ``mention_rate = n / n_reviews_i`` (reviews of ``i`` with at least one kept sentence);
  ``pos_rate`` / ``neg_rate`` = share of mentioning reviews whose mean polarity is above
  ``nlp.pos_threshold`` / below ``nlp.neg_threshold``;
* descriptors (darkness, pace, complexity_level) = per-item mean of the sentence descriptor
  scores shrunk toward 0 with the same ``k``, clipped to [-1, 1].

Per corpus user: the explicit aspect vector and its confidence (``litsense_core.aspects``), with
time decay measured from the cutoff, plus ``user_review_aspects.parquet`` (one row per review and
aspect) so ranker training can attach a reader's own review opinions as ``ReviewAnalysis``.
Each set also writes ``aspect_topitems.json``: per aspect the top ``aspects.topitems.n`` items by
shrunk value among items with at least ``aspects.topitems.min_reviews`` mentioning reviews (the
aspect candidate source, s13.1; the val set's lists feed ranker training, D66). Every builder
raises ``LeakageError`` on rows at or after its cutoff.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from litsense_core import aspects as shared
from ml.common.io import load_id_map
from ml.common.stage import StageContext, stage
from ml.nlp.aspect_seeds import ASPECTS
from ml.nlp.aspects import DESCRIPTORS
from ml.nlp.cutoffs import ArtifactSet, check_cutoff, cutoff_for, rows_before

N_ASPECTS = len(ASPECTS)
ITEM_FILES = (
    "item_aspects.npy",
    "item_mention_rate.npy",
    "item_pos_rate.npy",
    "item_neg_rate.npy",
    "item_descriptors.npy",
    "aspect_means.npy",
)
USER_FILES = ("user_aspects.npy", "user_aspect_conf.npy", "user_review_aspects.parquet")
TOPITEMS_FILE = "aspect_topitems.json"
SENTENCE_COLUMNS = ["sent_id", "review_row", "u", "i", "ts", "rating", "helpful_vote"]


@dataclass
class ItemProfiles:
    aspects: np.ndarray  # (n_items, 12) shrunk polarity
    mention_rate: np.ndarray
    pos_rate: np.ndarray
    neg_rate: np.ndarray
    descriptors: np.ndarray  # (n_items, 3)
    means: np.ndarray  # (12,) global weighted mean polarity per aspect
    n_mentioning: np.ndarray  # (n_items, 12) distinct reviews mentioning the aspect
    n_reviews: np.ndarray  # (n_items,) reviews with a kept sentence


def mention_weights(helpful_vote: np.ndarray) -> np.ndarray:
    """``1 + log1p(helpful_vote)`` (s11.8)."""
    return (1.0 + np.log1p(np.clip(np.asarray(helpful_vote, dtype=np.float64), 0, None))).astype(
        np.float32
    )


def review_aspect_means(mentions: pd.DataFrame) -> pd.DataFrame:
    """Per (review_row, aspect): the review's mean polarity for that aspect, with its ``u``, ``i``,
    ``ts`` and ``helpful_vote``. One review may hold several sentences on the same aspect."""
    keys = ["review_row", "aspect"]
    agg = mentions.groupby(keys, sort=False).agg(
        polarity=("polarity", "mean"),
        u=("u", "first"),
        i=("i", "first"),
        ts=("ts", "first"),
        helpful_vote=("helpful_vote", "first"),
    )
    return agg.reset_index()


def build_item_profiles(
    sentences: pd.DataFrame,
    polarity: pd.DataFrame,
    descriptor_scores: pd.DataFrame,
    n_items: int,
    cutoff: int,
    k: float,
    pos_threshold: float,
    neg_threshold: float,
) -> ItemProfiles:
    """Item aspect profiles from the sentence table (rows before ``cutoff``), the (sentence,
    aspect) polarity table and the per-sentence descriptor scores. Raises ``LeakageError`` when a
    sentence is at or after the cutoff."""
    check_cutoff(sentences["ts"].to_numpy(), cutoff, "sentences")
    if sentences["i"].to_numpy().max(initial=-1) >= n_items:
        raise ValueError("sentence item ids exceed the id map")
    mentions = polarity.merge(sentences, on="sent_id", how="inner")
    w = mention_weights(mentions["helpful_vote"].to_numpy())
    cell = mentions["i"].to_numpy(np.int64) * N_ASPECTS + mentions["aspect"].to_numpy(np.int64)
    size = n_items * N_ASPECTS
    w_sum = np.bincount(cell, weights=w, minlength=size)
    wp_sum = np.bincount(
        cell, weights=w * mentions["polarity"].to_numpy(np.float64), minlength=size
    )
    raw = np.divide(wp_sum, w_sum, out=np.zeros(size), where=w_sum > 0).reshape(n_items, N_ASPECTS)
    a_w = np.bincount(mentions["aspect"], weights=w, minlength=N_ASPECTS)
    a_wp = np.bincount(mentions["aspect"], weights=w * mentions["polarity"], minlength=N_ASPECTS)
    means = np.divide(a_wp, a_w, out=np.zeros(N_ASPECTS), where=a_w > 0).astype(np.float32)

    per_review = review_aspect_means(mentions)
    r_cell = per_review["i"].to_numpy(np.int64) * N_ASPECTS + per_review["aspect"].to_numpy(
        np.int64
    )
    n = np.bincount(r_cell, minlength=size).reshape(n_items, N_ASPECTS).astype(np.float32)
    pos = np.bincount(r_cell, weights=per_review["polarity"] > pos_threshold, minlength=size)
    neg = np.bincount(r_cell, weights=per_review["polarity"] < neg_threshold, minlength=size)
    pos_rate = np.divide(pos.reshape(n_items, N_ASPECTS), n, out=np.zeros_like(n), where=n > 0)
    neg_rate = np.divide(neg.reshape(n_items, N_ASPECTS), n, out=np.zeros_like(n), where=n > 0)
    reviews_per_item = sentences.drop_duplicates(["review_row"])["i"].to_numpy(np.int64)
    n_reviews = np.bincount(reviews_per_item, minlength=n_items).astype(np.float32)
    mention_rate = np.divide(
        n, n_reviews[:, None], out=np.zeros_like(n), where=n_reviews[:, None] > 0
    )

    scored = descriptor_scores.merge(sentences[["sent_id", "i"]], on="sent_id", how="inner")
    desc = np.zeros((n_items, len(DESCRIPTORS)), dtype=np.float32)
    for d, name in enumerate(DESCRIPTORS):
        values = scored[name].to_numpy(np.float64)
        ok = ~np.isnan(values)
        items = scored["i"].to_numpy(np.int64)[ok]
        total = np.bincount(items, weights=values[ok], minlength=n_items)
        count = np.bincount(items, minlength=n_items)
        desc[:, d] = np.clip(total / (count + k), -1.0, 1.0)  # shrunk toward 0
    return ItemProfiles(
        aspects=shared.shrink(raw, n, means, k),
        mention_rate=mention_rate.astype(np.float32),
        pos_rate=pos_rate.astype(np.float32),
        neg_rate=neg_rate.astype(np.float32),
        descriptors=desc,
        means=means,
        n_mentioning=n,
        n_reviews=n_reviews,
    )


def build_user_profiles(
    per_review: pd.DataFrame,
    n_users: int,
    cutoff: int,
    half_life_days: float,
    conf_k: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Explicit aspect vectors ``(n_users, 12)`` and confidences from per-(review, aspect) rows
    (``u``, ``ts``, ``aspect``, ``polarity``), decay measured from ``cutoff``. Vectorized twin of
    ``litsense_core.aspects.explicit_aspect_vec`` (tested to agree)."""
    check_cutoff(per_review["ts"].to_numpy(), cutoff, "reviews")
    if per_review["u"].to_numpy().max(initial=-1) >= n_users:
        raise ValueError("review user ids exceed the id map")
    d = shared.decay(cutoff - per_review["ts"].to_numpy(np.int64), half_life_days)
    cell = per_review["u"].to_numpy(np.int64) * N_ASPECTS + per_review["aspect"].to_numpy(np.int64)
    size = n_users * N_ASPECTS
    d_sum = np.bincount(cell, weights=d, minlength=size)
    dp_sum = np.bincount(
        cell, weights=d * per_review["polarity"].to_numpy(np.float64), minlength=size
    )
    vec = np.divide(dp_sum, d_sum, out=np.zeros(size), where=d_sum > 0).reshape(n_users, N_ASPECTS)
    n = np.bincount(cell, minlength=size).reshape(n_users, N_ASPECTS).astype(np.float32)
    return vec.astype(np.float32), (n / (n + conf_k)).astype(np.float32)


def aspect_topitems(profiles: ItemProfiles, n: int, min_reviews: int) -> dict[str, list[int]]:
    """Per aspect, the top ``n`` items by shrunk value among items with >= ``min_reviews``
    mentioning reviews (s13.1)."""
    out: dict[str, list[int]] = {}
    for a, name in enumerate(ASPECTS):
        eligible = np.flatnonzero(profiles.n_mentioning[:, a] >= min_reviews)
        order = np.argsort(-profiles.aspects[eligible, a], kind="stable")
        out[name] = eligible[order][:n].astype(int).tolist()
    return out


def _register(artifact_set: ArtifactSet, stage_name: str) -> None:
    folder = f"{artifact_set}/aspects"
    # D66: every artifact set writes the top-items lists, so the val bundle (ranker training)
    # has the aspect candidate source at the val cutoff, not the serve one.
    files = [*ITEM_FILES, *USER_FILES, "meta.json", TOPITEMS_FILE]

    @stage(
        stage_name,
        inputs=[
            "data:sentences.parquet",
            "data:sentence_aspects.parquet",
            "data:aspect_polarity.parquet",
            "data:splits.json",
            "data:id_map.json",
        ],
        outputs=[f"artifacts:{folder}/{name}" for name in files],
        config_keys=[
            "aspects.names",
            "aspects.descriptors",
            "aspects.shrinkage_k",
            "aspects.user_half_life_days",
            "aspects.explicit_conf_k",
            "aspects.topitems",
            "nlp.pos_threshold",
            "nlp.neg_threshold",
        ],
    )
    def build(ctx: StageContext) -> None:
        """Item aspect profiles, descriptors and corpus-user explicit vectors for one cutoff."""
        if tuple(ctx.param("aspects.names")) != ASPECTS:
            raise ValueError("aspects.names must equal aspect_seeds.ASPECTS, in order")
        if tuple(ctx.param("aspects.descriptors")) != DESCRIPTORS:
            raise ValueError("aspects.descriptors must equal aspect_seeds.DESCRIPTOR_POLES order")
        ids = load_id_map(ctx.path("data:id_map.json"))
        splits: Mapping[str, Any] = json.loads(
            ctx.path("data:splits.json").read_text(encoding="utf-8")
        )
        k = float(ctx.param("aspects.shrinkage_k"))
        sentences = pd.read_parquet(ctx.path("data:sentences.parquet"), columns=SENTENCE_COLUMNS)
        polarity = pd.read_parquet(ctx.path("data:aspect_polarity.parquet"))
        descriptors = pd.read_parquet(
            ctx.path("data:sentence_aspects.parquet"), columns=["sent_id", *DESCRIPTORS]
        )
        max_ts = int(sentences["ts"].max()) if len(sentences) else 0
        cutoff = cutoff_for(artifact_set, splits, max_ts)
        ctx.cutoff = cutoff
        before = rows_before(sentences, cutoff)
        items = build_item_profiles(
            before, polarity, descriptors, ids.n_items, cutoff, k,
            float(ctx.param("nlp.pos_threshold")), float(ctx.param("nlp.neg_threshold")),
        )  # fmt: skip
        per_review = review_aspect_means(polarity.merge(before, on="sent_id", how="inner"))
        half_life = float(ctx.param("aspects.user_half_life_days"))
        conf_k = float(ctx.param("aspects.explicit_conf_k"))
        users, conf = build_user_profiles(per_review, ids.n_users, cutoff, half_life, conf_k)

        out = {name: ctx.out(f"artifacts:{folder}/{name}") for name in files}
        np.save(out["item_aspects.npy"], items.aspects)
        np.save(out["item_mention_rate.npy"], items.mention_rate)
        np.save(out["item_pos_rate.npy"], items.pos_rate)
        np.save(out["item_neg_rate.npy"], items.neg_rate)
        np.save(out["item_descriptors.npy"], items.descriptors)
        np.save(out["aspect_means.npy"], items.means)
        np.save(out["user_aspects.npy"], users)
        np.save(out["user_aspect_conf.npy"], conf)
        per_review.sort_values(["u", "ts", "aspect"]).reindex(
            columns=["u", "i", "review_row", "ts", "aspect", "polarity", "helpful_vote"]
        ).astype({"aspect": "int8", "polarity": "float32"}).to_parquet(
            out["user_review_aspects.parquet"], index=False, compression="zstd"
        )
        top = ctx.param("aspects.topitems")
        topitems = aspect_topitems(items, int(top["n"]), int(top["min_reviews"]))
        out[TOPITEMS_FILE].write_text(json.dumps(topitems), encoding="utf-8")
        ctx.count("topitems_min", min(len(v) for v in topitems.values()))
        counts = {
            "sentences_before_cutoff": int(len(before)),
            "aspect_mentions": int(len(polarity.merge(before[["sent_id"]], on="sent_id"))),
            "review_aspect_rows": int(len(per_review)),
            "items_with_any_mention": int((items.n_mentioning.sum(axis=1) > 0).sum()),
            "users_with_review_aspects": int((conf.sum(axis=1) > 0).sum()),
        }
        meta = {
            "artifact_set": artifact_set,
            "cutoff": cutoff,
            "shrinkage_k": k,
            "aspect_means": {
                a: round(float(m), 4) for a, m in zip(ASPECTS, items.means, strict=True)
            },
            "n_items": ids.n_items,
            "n_users": ids.n_users,
            **counts,
        }
        out["meta.json"].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        for key, value in counts.items():
            ctx.count(key, value)

    build.__name__ = f"build_profiles_{artifact_set}"


_register("eval_val", "nlp.profiles_val")
_register("serve", "nlp.profiles_serve")
