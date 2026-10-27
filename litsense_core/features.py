"""The ranker feature table (CLAUDE.md s13.3): 27 features plus one flag per candidate source.

``assemble`` builds the ``(n_candidates, len(FEATURES))`` float32 matrix with vectorized numpy
over the candidate array (target < 15 ms for 200 candidates). Missing values are ``np.nan``, which
LightGBM handles; the column order is ``FEATURES`` and is saved as ``feature_list.json`` next to the
ranker, so a bundle whose list differs is refused instead of silently misaligned. The same code runs
offline (``ml/ranking/build_training.py``, over the val bundle) and online (``recommend``), which is
what keeps the ranker's training and serving features identical (principle 4).
"""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

import numpy as np

from litsense_core.aspects import implicit_aspect_vec
from litsense_core.candidates import SOURCE_ORDER
from litsense_core.profile import Profile
from litsense_core.rank import aspect_match_scores
from litsense_core.store import ArtifactStore

SOURCE_FLAGS = tuple(f"src_{s}" for s in SOURCE_ORDER)
FEATURES: tuple[str, ...] = (
    "cf_score",
    "cf_norm_score",
    "cf_rank_pct",
    "cf_available",
    "proj_hybrid_score",
    "semantic_sim",
    "semantic_recent_sim",
    "onboarding_sim",
    "max_item_sim",
    "top5_item_sim",
    "aspect_match",
    "aspect_match_recent",
    "aspect_conflict",
    "descriptor_match",
    "user_aspect_confidence",
    "review_centroid_sim",
    "genre_overlap",
    "disliked_genre_hit",
    "decade_match",
    "author_affinity",
    "item_bayes_rating",
    "item_count_log",
    "item_rating_std",
    "item_age_years",
    "user_activity_log",
    "user_mean_rating",
    "n_sources",
    *SOURCE_FLAGS,
)
# Feature groups of s13.3 (the ablation masks of s13.7 are unions of these).
GROUPS: dict[str, tuple[str, ...]] = {
    "CF": ("cf_score", "cf_norm_score", "cf_rank_pct", "cf_available"),
    "PROJ": ("proj_hybrid_score",),
    "CONTENT": (
        "semantic_sim",
        "semantic_recent_sim",
        "onboarding_sim",
        "max_item_sim",
        "top5_item_sim",
    ),
    "ASPECT": (
        "aspect_match",
        "aspect_match_recent",
        "aspect_conflict",
        "descriptor_match",
        "user_aspect_confidence",
    ),
    "REVIEW_EMB": ("review_centroid_sim",),
    "META": (
        "genre_overlap",
        "disliked_genre_hit",
        "decade_match",
        "item_bayes_rating",
        "item_count_log",
        "item_rating_std",
        "item_age_years",
    ),
    "AUTHOR": ("author_affinity",),
    "USER": ("user_activity_log", "user_mean_rating"),
    "SOURCE": ("n_sources", *SOURCE_FLAGS),
}
assert sorted(f for g in GROUPS.values() for f in g) == sorted(FEATURES)
LIKED_FOR_SIMS = 50  # liked items considered for max/top5 item similarity (as in reasons.py)
MS_PER_YEAR = 365.25 * 86_400_000


def group_of(feature: str) -> str:
    return next(g for g, names in GROUPS.items() if feature in names)


def columns(names: Sequence[str]) -> np.ndarray:
    """Indices of ``names`` in ``FEATURES`` (for group masks)."""
    index = {f: k for k, f in enumerate(FEATURES)}
    return np.array([index[n] for n in names], dtype=np.int64)


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def _cos(vec: np.ndarray | None, rows: np.ndarray) -> np.ndarray:
    """Cosine of ``vec`` with every row; NaN column when ``vec`` is None or zero."""
    n = rows.shape[0]
    if vec is None:
        return np.full(n, np.nan, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return np.full(n, np.nan, dtype=np.float32)
    return (_unit_rows(rows) @ (vec / norm)).astype(np.float32)


def _rank_pct(values: np.ndarray) -> np.ndarray:
    """Percentile rank in [0, 1] (1 = best) among finite values; NaN stays NaN."""
    out = np.full(len(values), np.nan, dtype=np.float32)
    finite = np.isfinite(values)
    m = int(finite.sum())
    if m == 0:
        return out
    order = np.argsort(values[finite], kind="stable")
    ranks = np.empty(m, dtype=np.float32)
    ranks[order] = np.arange(m, dtype=np.float32)
    out[finite] = ranks / max(m - 1, 1)
    return out


def user_review_centroid(profile: Profile, store: ArtifactStore) -> np.ndarray | None:
    """The reader's review centroid: the offline one for a corpus demo reader, else the
    rating-weighted mean of the rated items' review centroids (a proxy until the reader has
    reviews of their own); None without centroid artifacts or when the result is zero."""
    if store.review_centroid_item is None:
        return None
    if profile.corpus_row is not None and store.corpus_centroids is not None:
        vec = np.asarray(store.corpus_centroids[profile.corpus_row], dtype=np.float32)
        if float(np.linalg.norm(vec)) > 0:
            return vec
    liked = profile.liked[:LIKED_FOR_SIMS]
    if not liked:
        return None
    rows = np.asarray(store.review_centroid_item[np.asarray(liked, dtype=np.int64)], np.float32)
    w = np.array([max(profile.stars.get(b, 4.0) - 3.0, 0.5) for b in liked], dtype=np.float32)
    vec = (w[:, None] * rows).sum(axis=0) / float(w.sum())
    return vec if float(np.linalg.norm(vec)) > 0 else None


def assemble(
    profile: Profile,
    ids: Sequence[int],
    provenance: Mapping[int, Sequence[str]],
    store: ArtifactStore,
    expected_cutoff: int | None = None,
) -> np.ndarray:
    """Feature matrix for ``ids`` in ``FEATURES`` order. ``expected_cutoff`` is the leakage guard
    for offline use: it must equal ``store.cutoff`` (the bundle the features come from)."""
    if expected_cutoff is not None and int(expected_cutoff) != int(store.cutoff):
        raise ValueError(f"store cutoff {store.cutoff} != expected {expected_cutoff}")
    ids_arr = np.asarray(ids, dtype=np.int64)
    n = len(ids_arr)
    out = np.full((n, len(FEATURES)), np.nan, dtype=np.float32)
    if n == 0:
        return out
    col = {f: k for k, f in enumerate(FEATURES)}
    cat = store.catalog

    # CF group
    if profile.cf_vec is not None:
        cf = store.lgcn_rows(ids_arr) @ profile.cf_vec
        out[:, col["cf_score"]] = cf
        out[:, col["cf_norm_score"]] = (
            np.asarray(store.lgcn_item_norm[ids_arr], dtype=np.float32) @ profile.cf_vec
        )
        out[:, col["cf_rank_pct"]] = _rank_pct(cf)
        if store.proj_item is not None:
            out[:, col["proj_hybrid_score"]] = (
                np.asarray(store.proj_item[ids_arr], dtype=np.float32) @ profile.cf_vec
            )
    out[:, col["cf_available"]] = profile.cf_available

    # CONTENT group
    content = store.content_rows(ids_arr)
    out[:, col["semantic_sim"]] = _cos(profile.semantic_vec, content)
    recent = None
    if profile.anchors:
        recent = store.content_rows(np.asarray(profile.anchors, dtype=np.int64)).mean(axis=0)
    out[:, col["semantic_recent_sim"]] = _cos(recent, content)
    out[:, col["onboarding_sim"]] = _cos(profile.onboarding_vec, content)
    liked = profile.liked[:LIKED_FOR_SIMS]
    if liked:
        sims = (
            _unit_rows(content)
            @ _unit_rows(store.content_rows(np.asarray(liked, dtype=np.int64))).T
        )  # (n, liked)
        out[:, col["max_item_sim"]] = sims.max(axis=1)
        top = np.sort(sims, axis=1)[:, -5:]
        out[:, col["top5_item_sim"]] = top.mean(axis=1)

    # ASPECT group
    if profile.aspect_vec is not None and store.item_aspects is not None:
        means = store.aspect_means
        item_centered = store.item_aspects[ids_arr] - (0.0 if means is None else means)
        out[:, col["aspect_match"]] = aspect_match_scores(ids_arr, profile, store)
        conf = (
            profile.aspect_conf
            if profile.aspect_conf is not None
            else np.zeros_like(profile.aspect_vec)
        )
        out[:, col["aspect_conflict"]] = (
            (0.5 + 0.5 * conf) * np.maximum(0.0, -profile.aspect_vec * item_centered)
        ).sum(axis=1)
        out[:, col["user_aspect_confidence"]] = float(conf.mean())
        if profile.anchors and means is not None:
            recent_vec = implicit_aspect_vec(
                store.item_aspects,
                means,
                np.asarray(profile.anchors, dtype=np.int64),
                np.ones(len(profile.anchors), dtype=np.float32),
            )
            if recent_vec is not None and float(np.linalg.norm(recent_vec)) > 0:
                num = item_centered @ recent_vec
                den = np.linalg.norm(item_centered, axis=1) * float(np.linalg.norm(recent_vec))
                out[:, col["aspect_match_recent"]] = np.divide(
                    num, den, out=np.zeros(n, dtype=np.float32), where=den > 0
                )
    if profile.descriptor_pref is not None and store.item_descriptors is not None:
        diff = store.item_descriptors[ids_arr] - profile.descriptor_pref
        out[:, col["descriptor_match"]] = -np.linalg.norm(diff, axis=1)

    # REVIEW_EMB
    centroid = user_review_centroid(profile, store)
    if centroid is not None and store.review_centroid_item is not None:
        rows = np.asarray(store.review_centroid_item[ids_arr], dtype=np.float32)
        sim = _cos(centroid, rows)
        sim[np.linalg.norm(rows, axis=1) == 0] = np.nan  # items without reviews before the cutoff
        out[:, col["review_centroid_sim"]] = sim

    # META, AUTHOR, USER
    positive = {g: v for g, v in profile.genre_affinity.items() if v > 0}
    scale = max(positive.values(), default=0.0)
    disliked = profile.disliked_genres | profile.muted_genres
    overlap = np.zeros(n, dtype=np.float32)
    hit = np.zeros(n, dtype=np.float32)
    for r, i in enumerate(ids_arr.tolist()):
        genres = cat.genres_of(i)
        if scale > 0:
            overlap[r] = sum(positive.get(g, 0.0) for g in genres) / scale
        hit[r] = float(any(g in disliked for g in genres))
    out[:, col["genre_overlap"]] = overlap
    out[:, col["disliked_genre_hit"]] = hit
    years = cat.year[ids_arr].astype(np.float32)
    known = years >= 0
    liked_years = cat.year[np.asarray(profile.liked, dtype=np.int64)] if profile.liked else None
    if liked_years is not None and (liked_years >= 0).any():
        liked_decades = (liked_years[liked_years >= 0] // 10).astype(np.int64)
        decade = np.where(known, years // 10, -1).astype(np.int64)
        share = np.array(
            [float((liked_decades == d).mean()) if d >= 0 else np.nan for d in decade.tolist()],
            dtype=np.float32,
        )
        out[:, col["decade_match"]] = share
    out[:, col["author_affinity"]] = [
        profile.favorite_authors.get(int(a), 0.0) for a in cat.author_id[ids_arr].tolist()
    ]
    out[:, col["item_bayes_rating"]] = cat.bayes_rating[ids_arr]
    out[:, col["item_count_log"]] = np.log1p(cat.rating_count[ids_arr].astype(np.float32))
    out[:, col["item_rating_std"]] = cat.rating_std[ids_arr]
    cutoff_year = datetime.fromtimestamp(store.cutoff / 1000, tz=UTC).year
    # Clipped at 0: a book dated after the cutoff is "new", not "from the future" (D67). Offline,
    # such books are also dropped from the candidates before this is computed.
    out[:, col["item_age_years"]] = np.where(known, np.maximum(cutoff_year - years, 0.0), np.nan)
    out[:, col["user_activity_log"]] = float(np.log1p(profile.n))
    if profile.stars:
        out[:, col["user_mean_rating"]] = float(np.mean(list(profile.stars.values())))

    # SOURCE flags
    flags = np.zeros((n, len(SOURCE_ORDER)), dtype=np.float32)
    for r, i in enumerate(ids_arr.tolist()):
        for s in provenance.get(int(i), ()):
            if s in SOURCE_ORDER:
                flags[r, SOURCE_ORDER.index(s)] = 1.0
    out[:, col["n_sources"]] = flags.sum(axis=1)
    out[:, col[SOURCE_FLAGS[0]] : col[SOURCE_FLAGS[0]] + len(SOURCE_ORDER)] = flags
    return out
