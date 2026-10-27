"""Blend ranker and post-processing (CLAUDE.md s12.3, s13.6).

The blend is the MVP ranker and the permanent fallback: per-source rank scores r = 1 - rank/len
(0 if absent), alpha(n) = n / (n + alpha_k), and

    score = alpha * r_cf + (1 - alpha) * r_semantic + w_pop * r_popularity + w_aspect * aspect_match

where ``aspect_match`` (D63) is the cosine between the reader's aspect vector and the item's
centered shrunk aspect profile, each dimension weighted by 0.5 + 0.5 * explicit confidence, and
0 without aspect artifacts. Post-processing: overrides (muted genre -> -inf, pinned aspect ->
+ ``pinned_aspect_boost`` * item aspect), MMR over the top ``mmr_pool`` by content similarity, a
genre cap per window of 10 and an exploration slot for the best longtail-source item.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from litsense_core.profile import Profile
from litsense_core.store import ArtifactStore


def rank_scores(ids: Sequence[int], source_list: Sequence[int]) -> np.ndarray:
    """r = 1 - rank / len(source list) for ids in the list, else 0."""
    position = {int(b): r for r, b in enumerate(source_list)}
    n = max(1, len(source_list))
    return np.array([1.0 - position[i] / n if i in position else 0.0 for i in ids], np.float32)


def blend(
    ids: Sequence[int],
    sources: Mapping[str, list[int]],
    profile: Profile,
    store: ArtifactStore,
    params: Mapping[str, float],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Blend scores for ``ids`` and the per-source rank scores used (for evidence signals)."""
    alpha_k = float(params["alpha_k"])
    alpha = profile.n / (profile.n + alpha_k) if profile.cf_vec is not None else 0.0
    if profile.cf_available == 0.5:
        alpha *= 0.5  # pseudo-CF enters with low weight (s12.4, "warming")
    r_cf = rank_scores(ids, sources.get("cf", []))
    r_sem = rank_scores(ids, sources.get("semantic", []))
    r_pop = rank_scores(ids, sources.get("popularity", []))
    if profile.semantic_vec is None and profile.cf_vec is None:
        r_pop = np.maximum(r_pop, rank_scores(ids, list(store.trending_top)))
    aspect_match = aspect_match_scores(ids, profile, store)
    score = (
        alpha * r_cf
        + (1.0 - alpha) * r_sem
        + float(params["w_pop"]) * r_pop
        + float(params["w_aspect"]) * aspect_match
    )
    # Small, stable tie-breakers so a pure-popularity reader still gets an ordered list.
    score += 1e-3 * store.catalog.popularity[np.asarray(ids, dtype=np.int64)]
    return score.astype(np.float32), {
        "cf": r_cf,
        "semantic": r_sem,
        "popularity": r_pop,
        "aspect_match": aspect_match,
    }


def aspect_match_scores(ids: Sequence[int], profile: Profile, store: ArtifactStore) -> np.ndarray:
    """Confidence-weighted cosine between the reader's aspect vector and each item's centered
    shrunk profile (s13.3 feature 11); zeros when either side is missing."""
    n = len(ids)
    if (
        n == 0
        or profile.aspect_vec is None
        or store.item_aspects is None
        or store.aspect_means is None
    ):
        return np.zeros(n, dtype=np.float32)
    conf = (
        profile.aspect_conf
        if profile.aspect_conf is not None
        else np.zeros_like(profile.aspect_vec)
    )
    w = 0.5 + 0.5 * conf
    u = w * profile.aspect_vec
    items = (store.item_aspects[np.asarray(ids, dtype=np.int64)] - store.aspect_means) * w
    num = items @ u
    den = np.linalg.norm(items, axis=1) * float(np.linalg.norm(u))
    return np.divide(num, den, out=np.zeros(n, dtype=np.float32), where=den > 0).astype(np.float32)


def postprocess(
    ids: Sequence[int],
    scores: np.ndarray,
    provenance: Mapping[int, list[str]],
    profile: Profile,
    store: ArtifactStore,
    params: Mapping[str, float],
    k: int,
) -> list[tuple[int, float]]:
    """s13.6 steps 2 to 6 over already-filtered candidates; returns ``(book_id, score)`` top-k."""
    if not len(ids):
        return []
    ids_arr = np.asarray(ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32).copy()
    cat = store.catalog
    for n, i in enumerate(ids_arr):
        if set(cat.genres_of(int(i))) & profile.muted_genres:
            scores[n] = -np.inf
    if profile.pinned_aspects and store.item_aspects is not None:
        boost = float((store.config.get("overrides") or {}).get("pinned_aspect_boost", 0.0))
        for a in profile.pinned_aspects:
            scores += boost * store.item_aspects[ids_arr, a]
    keep = np.isfinite(scores)
    ids_arr, scores = ids_arr[keep], scores[keep]
    if not len(ids_arr):
        return []

    order = np.argsort(-scores, kind="stable")
    pool_n = min(len(order), int(params["mmr_pool"]))
    pool = order[:pool_n]
    lam = float(params["mmr_lambda"])
    lo, hi = float(scores[pool].min()), float(scores[pool].max())
    norm = (scores[pool] - lo) / (hi - lo) if hi > lo else np.ones(pool_n, dtype=np.float32)
    content = store.content_rows(ids_arr[pool])
    sims = content @ content.T
    chosen: list[int] = []
    remaining = list(range(pool_n))
    while remaining and len(chosen) < pool_n:
        if chosen:
            max_sim = sims[np.ix_(remaining, chosen)].max(axis=1)
        else:
            max_sim = np.zeros(len(remaining), dtype=np.float32)
        mmr = lam * norm[remaining] - (1.0 - lam) * max_sim
        best = remaining[int(np.argmax(mmr))]
        chosen.append(best)
        remaining.remove(best)
    ranked = [int(ids_arr[pool[c]]) for c in chosen] + [int(ids_arr[o]) for o in order[pool_n:]]
    score_of = {int(i): float(s) for i, s in zip(ids_arr, scores, strict=True)}

    cap = int(params["genre_cap"])
    final: list[int] = []
    deferred: list[int] = []
    for i in ranked:
        window = final[-9:]
        g = cat.primary_genre_of(i)
        if sum(cat.primary_genre_of(j) == g for j in window) >= cap:
            deferred.append(i)
        else:
            final.append(i)
    final += deferred

    slot = int(params["explore_slot"]) - 1
    if 0 <= slot < len(final):
        tail = next(
            (i for i in ranked if "longtail" in provenance.get(i, []) and i not in final[:slot]),
            None,
        )
        if tail is not None and tail != final[slot]:
            final = [i for i in final if i != tail]
            final.insert(slot, tail)
    return [(i, score_of[i]) for i in final[:k]]
