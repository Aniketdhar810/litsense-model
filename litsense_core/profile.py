"""Online reader profile from a ``UserState`` (CLAUDE.md s12.2, s11.8), recomputed every call.

D56: interaction count and bucket, the collaborative vector (exact LightGCN vector for corpus demo
users, else the rating-weighted mean of liked items' LightGCN vectors), the semantic vector
(rating-weighted mean of liked items' content vectors), genre and author affinities, series
progress and the seen set. D63 adds the s11.8 aspect side through ``litsense_core.aspects``
(implicit from rated items' shrunk profiles, explicit from the reader's own review analyses,
onboarding and pinned aspects), the descriptor preference, and the onboarding text vector through
the bundled ONNX encoder (0.6 picks + 0.4 text, s12.2). Each part is skipped when its bundle file
is absent, so older bundles still load.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from litsense_core.aspects import combine, decay, explicit_aspect_vec, implicit_aspect_vec
from litsense_core.schemas import Bucket, UserState
from litsense_core.store import ArtifactStore

# s11.8: rating weight r_w, 5 -> 1.0 ... 1 -> -1.0 (mirrors aspects.rating_weights in config).
RATING_WEIGHTS = {5: 1.0, 4: 0.5, 3: 0.0, 2: -0.5, 1: -1.0}
LIST_WEIGHT = 0.5  # reading-list adds count half an interaction (s12.2)
DAY_MS = 86_400_000
RECENT = 3  # item_item anchors: the last liked items (s13.1)
TOP_AUTHORS = 20


def rating_weight(stars: float) -> float:
    return RATING_WEIGHTS[int(min(5, max(1, round(stars))))]


@dataclass
class Profile:
    n: float
    bucket: Bucket
    seen: set[int]
    stars: dict[int, float]
    liked: list[int]  # rated >= 4, newest first
    rated: list[int]  # every rated item, newest first
    anchors: list[int]  # last RECENT liked items (or onboarding picks)
    cf_vec: np.ndarray | None
    cf_available: float  # 1 exact corpus vector, 0.5 pseudo, 0 none
    semantic_vec: np.ndarray | None
    onboarding_vec: np.ndarray | None
    genre_affinity: dict[str, float]
    preferred_genres: list[str]
    disliked_genres: set[str]
    favorite_authors: dict[int, float]  # author_id -> affinity in [0, 1]
    favorite_author_liked: dict[int, int]  # author_id -> the liked book that made them a favorite
    series_progress: dict[int, float]  # series_id -> max series_pos read
    muted_genres: set[str]
    onboarding_genres: list[str]
    aspect_vec: np.ndarray | None = None  # (12,) in [-1, 1]; None without aspect artifacts
    aspect_conf: np.ndarray | None = None  # (12,) confidence of the explicit side, 0 when none
    descriptor_pref: np.ndarray | None = None  # (3,) darkness, pace, complexity_level
    pinned_aspects: list[int] = field(default_factory=list)
    corpus_row: int | None = None  # row in the bundle's corpus_users when the reader is one
    extras: dict[str, Any] = field(default_factory=dict)

    def top_aspects(self, sign: int = 1, threshold: float = 0.3) -> list[tuple[int, float]]:
        """Aspect indices whose value crosses ``threshold`` in the direction ``sign``, strongest
        first."""
        if self.aspect_vec is None:
            return []
        vals = self.aspect_vec * sign
        order = np.argsort(-vals, kind="stable")
        return [(int(a), float(self.aspect_vec[a])) for a in order if vals[a] > threshold]


def bucket_for(n: float, buckets: dict[str, Any]) -> Bucket:
    """The bucket whose [lower, upper] range contains n (fractional n is allowed)."""
    chosen: str = "cold"
    for name, (lower, _upper) in buckets.items():
        if n >= float(lower):
            chosen = name
    return chosen  # type: ignore[return-value]


def _unit(vec: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    return None if norm == 0.0 else (vec / norm).astype(np.float32)


def weighted_mean(matrix_rows: np.ndarray, weights: np.ndarray) -> np.ndarray | None:
    """Σ w·row / Σ |w|; None when there is nothing to average."""
    if len(weights) == 0 or float(np.abs(weights).sum()) == 0.0:
        return None
    return (weights[:, None] * matrix_rows).sum(axis=0) / float(np.abs(weights).sum())


def build_profile(state: UserState, store: ArtifactStore) -> Profile:
    """Rebuild the reader's profile from scratch (no incremental state, s1.5)."""
    cat = store.catalog
    for book_id in (
        [e.book_id for e in state.ratings]
        + [e.book_id for e in state.reading_list]
        + list(state.dismissed)
        + list(state.viewed)
        + list(state.onboarding_picks)
        + [r.book_id for r in state.review_analyses if r.book_id is not None]
    ):
        cat.check_id(book_id)

    ratings = sorted(state.ratings, key=lambda e: -e.ts)
    latest: dict[int, float] = {}
    latest_ts: dict[int, int] = {}
    for e in ratings:  # newest rating of a book wins
        if e.book_id not in latest:
            latest[e.book_id], latest_ts[e.book_id] = float(e.stars), int(e.ts)
    rated = list(latest)
    liked = [b for b in rated if latest[b] >= 4.0]
    n = len(rated) + LIST_WEIGHT * len(state.reading_list)
    corpus_row = (
        store.corpus_users.get(state.corpus_user_id) if state.corpus_user_id is not None else None
    )
    if corpus_row is not None:
        # A demo reader's history lives in the corpus, not in the state: count it for the bucket
        # so the exact LightGCN vector gets the cf quota it was built for (D56 fix).
        n = max(n, float(store.corpus_counts[corpus_row]))
    bucket = bucket_for(n, store.config["buckets"])

    # Time decay d = 0.5^(age_days / half_life); half life 730 days (aspects.user_half_life_days).
    now = max(latest_ts.values(), default=store.cutoff)
    weights_by_item = {
        b: rating_weight(s) * 0.5 ** ((now - latest_ts[b]) / DAY_MS / 730.0)
        for b, s in latest.items()
    }
    liked_ids = np.array(liked, dtype=np.int64)
    liked_w = np.array([weights_by_item[b] for b in liked], dtype=np.float32)
    pick_ids = np.array([b for b in state.onboarding_picks if b not in latest], dtype=np.int64)
    pick_w = np.full(len(pick_ids), rating_weight(5.0), dtype=np.float32)
    list_ids = np.array(
        [e.book_id for e in state.reading_list if e.book_id not in latest], dtype=np.int64
    )
    list_w = np.full(len(list_ids), LIST_WEIGHT, dtype=np.float32)

    # cf_vec (s12.2): exact corpus vector, else pseudo from liked + picks + list items.
    cf_vec: np.ndarray | None = None
    cf_available = 0.0
    if corpus_row is not None:
        cf_vec = store.corpus_vectors[corpus_row].astype(np.float32)
        cf_available = 1.0
    else:
        ids = np.concatenate([liked_ids, pick_ids, list_ids])
        w = np.concatenate([liked_w, pick_w, list_w])
        if len(ids):
            mean = weighted_mean(store.lgcn_rows(ids), w)
            if mean is not None and float(np.linalg.norm(mean)) > 0:
                cf_vec, cf_available = mean.astype(np.float32), 0.5

    # semantic_vec: weighted mean of liked + picked items' content vectors, L2-normalized; with
    # onboarding text and an encoder, 0.6 picks + 0.4 text (s12.2). onboarding_vec is kept apart.
    sem_ids = np.concatenate([liked_ids, pick_ids])
    sem_w = np.concatenate([liked_w, pick_w])
    semantic_vec = None
    if len(sem_ids):
        mean = weighted_mean(store.content_rows(sem_ids), sem_w)
        semantic_vec = None if mean is None else _unit(mean)
    onboarding_vec = None
    if state.onboarding_text and state.onboarding_text.strip() and store.encoder is not None:
        onboarding_vec = _unit(store.encoder.encode([state.onboarding_text.strip()])[0])
        if onboarding_vec is not None:
            semantic_vec = (
                _unit(0.6 * semantic_vec + 0.4 * onboarding_vec)
                if semantic_vec is not None
                else onboarding_vec
            )

    aspect_vec, aspect_conf, descriptor_pref, pinned = aspect_side(
        state, store, weights_by_item, latest_ts, now
    )

    # Genre and author affinities (s11.8), onboarding and muted genres folded in.
    genre_aff: dict[str, float] = {}
    author_aff: dict[int, float] = {}
    fav_liked: dict[int, int] = {}
    for b, w in weights_by_item.items():
        for g in cat.genres_of(b):
            genre_aff[g] = genre_aff.get(g, 0.0) + w
        a = int(cat.author_id[b])
        if a >= 0:
            author_aff[a] = author_aff.get(a, 0.0) + w
            if latest[b] >= 4.0 and a not in fav_liked:
                fav_liked[a] = b
    for b in pick_ids.tolist():
        for g in cat.genres_of(b):
            genre_aff[g] = genre_aff.get(g, 0.0) + 1.0
    onboarding = state.onboarding
    onboarding_genres = list(onboarding.liked_genres) if onboarding else []
    for g in onboarding_genres:
        genre_aff[g] = genre_aff.get(g, 0.0) + 1.0
    for g in onboarding.disliked_genres if onboarding else []:
        genre_aff[g] = genre_aff.get(g, 0.0) - 1.0
    muted = set(state.overrides.muted_genres)
    preferred = [g for g, v in sorted(genre_aff.items(), key=lambda kv: -kv[1]) if v > 0]
    disliked = {g for g, v in genre_aff.items() if v < -0.3} | muted
    scale = max((v for a, v in author_aff.items() if a in fav_liked), default=0.0)
    favorites = {
        a: (v / scale if scale > 0 else 0.0)
        for a, v in sorted(author_aff.items(), key=lambda kv: -kv[1])
        if a in fav_liked and v > 0
    }
    favorites = dict(list(favorites.items())[:TOP_AUTHORS])

    finished = set(rated) | {e.book_id for e in state.reading_list if e.status == "read"}
    progress: dict[int, float] = {}
    for b in finished:
        s = int(cat.series_id[b])
        pos = float(cat.series_pos[b])
        if s >= 0 and not np.isnan(pos):
            progress[s] = max(progress.get(s, 0.0), pos)
    seen = (
        set(rated)
        | {e.book_id for e in state.reading_list if e.status in ("read", "dropped")}
        | set(state.dismissed)
        | set(state.onboarding_picks)
    )
    anchors = liked[:RECENT] or list(state.onboarding_picks[:RECENT])
    return Profile(
        n=n,
        bucket=bucket,
        seen=seen,
        stars=latest,
        liked=liked,
        rated=rated,
        anchors=anchors,
        cf_vec=cf_vec,
        cf_available=cf_available,
        semantic_vec=semantic_vec,
        onboarding_vec=onboarding_vec,
        genre_affinity=genre_aff,
        preferred_genres=preferred,
        disliked_genres=disliked,
        favorite_authors=favorites,
        favorite_author_liked=fav_liked,
        series_progress=progress,
        muted_genres=muted,
        onboarding_genres=onboarding_genres,
        aspect_vec=aspect_vec,
        aspect_conf=aspect_conf,
        descriptor_pref=descriptor_pref,
        pinned_aspects=pinned,
        corpus_row=corpus_row,
    )


def aspect_side(
    state: UserState,
    store: ArtifactStore,
    weights_by_item: dict[int, float],
    latest_ts: dict[int, int],
    now: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, list[int]]:
    """s11.8 user aspect vector, its confidence, the descriptor preference and the pinned aspect
    indices; all None/empty when the bundle has no aspect profiles."""
    if store.item_aspects is None or store.aspect_means is None or not store.aspect_names:
        return None, None, None, []
    n_aspects = store.n_aspects
    cfg = store.config.get("aspects") or {}
    half_life = float(cfg.get("user_half_life_days", 730))
    conf_k = float(cfg.get("explicit_conf_k", 3))
    explicit_weight = float(cfg.get("user_explicit_weight", 0.6))

    rated_ids = np.array(list(weights_by_item), dtype=np.int64)
    rated_w = np.array(list(weights_by_item.values()), dtype=np.float32)
    implicit = implicit_aspect_vec(store.item_aspects, store.aspect_means, rated_ids, rated_w)

    explicit = conf = None
    reviews = [r for r in state.review_analyses if r.aspects]
    corpus_row = (
        store.corpus_users.get(state.corpus_user_id) if state.corpus_user_id is not None else None
    )
    if (
        not reviews
        and corpus_row is not None
        and store.corpus_aspects is not None
        and store.corpus_aspect_conf is not None
    ):
        # A demo reader's own reviews live in the corpus: use the offline explicit vector (D64).
        explicit = store.corpus_aspects[corpus_row].astype(np.float32)
        conf = store.corpus_aspect_conf[corpus_row].astype(np.float32)
    elif reviews:
        matrix = np.full((len(reviews), n_aspects), np.nan, dtype=np.float32)
        ages = np.zeros(len(reviews), dtype=np.int64)
        for row, r in enumerate(reviews):
            for name, polarity in r.aspects.items():
                a = store.aspect_index(name)
                if a is not None:
                    matrix[row, a] = float(np.clip(polarity, -1.0, 1.0))
            ages[row] = max(0, now - int(r.ts)) if r.ts is not None else 0
        explicit, conf = explicit_aspect_vec(matrix, decay(ages, half_life), n_aspects, conf_k)

    onboarding = state.onboarding
    positive = [a for n in (onboarding.positive_aspects if onboarding else [])
                if (a := store.aspect_index(n)) is not None]  # fmt: skip
    negative = [a for n in (onboarding.negative_aspects if onboarding else [])
                if (a := store.aspect_index(n)) is not None]  # fmt: skip
    pinned = [a for n in state.overrides.pinned_aspects if (a := store.aspect_index(n)) is not None]
    if explicit is not None and conf is not None:
        # Review polarity is mostly positive (aspect means 0.4 to 0.56 on v0), so a raw explicit
        # vector says "likes everything". Center it on the aspect means like the implicit side,
        # only where the reader actually mentioned the aspect (D65).
        explicit = np.where(conf > 0, explicit - store.aspect_means, 0.0).astype(np.float32)
    aspect_vec = combine(implicit, explicit, conf, explicit_weight, positive, negative, pinned,
                         n_aspects)  # fmt: skip
    aspect_conf = conf if conf is not None else np.zeros(n_aspects, dtype=np.float32)

    descriptor_pref = None
    if store.item_descriptors is not None:
        names = list(cfg.get("descriptors") or [])
        pref = weighted_mean(store.item_descriptors[rated_ids], rated_w) if len(rated_ids) else None
        pref = np.zeros(store.item_descriptors.shape[1], dtype=np.float32) if pref is None else pref
        for k, name in enumerate(names[: len(pref)]):
            if onboarding and name in onboarding.descriptors:
                pref[k] += float(onboarding.descriptors[name])
        if len(rated_ids) or (onboarding and onboarding.descriptors):
            descriptor_pref = np.clip(pref, -1.0, 1.0).astype(np.float32)
    return aspect_vec, aspect_conf, descriptor_pref, pinned
