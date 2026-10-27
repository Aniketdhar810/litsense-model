"""The public API of CLAUDE.md s15.1 over a loaded ``ArtifactStore`` (s12.6; D56, D63).

Blend ranker over the seven candidate sources, template reasons with aspect evidence and safe
snippets, series guard, MMR and genre cap; free-text search, onboarding text and review analysis
through the bundled ONNX encoder. Functions that need later phases (LightGBM ranker, LLM) return
valid but degraded results (s15.3) and say so; a bundle without the Phase 3 files degrades the
same way.
"""

import importlib
import re
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np

from litsense_core import __api_version__, features
from litsense_core import candidates as cand
from litsense_core import store as store_mod
from litsense_core.profile import Profile, build_profile
from litsense_core.rank import blend, postprocess
from litsense_core.reasons import build_evidence, template_reason
from litsense_core.schemas import (
    ASPECT_LABELS,
    AspectBar,
    BecauseYouLiked,
    BookInsights,
    ChatReply,
    EntityMatch,
    Explanation,
    Filters,
    HomeRails,
    OnboardingResult,
    RecItem,
    RecResponse,
    ReviewAnalysis,
    SearchHit,
    SessionState,
    Snippet,
    UserState,
)
from litsense_core.store import ArtifactStore, download_bundle, load_store  # noqa: F401 (re-export)

GRID_MIN_RATINGS = 50
CHAT_K = 5


def _store() -> ArtifactStore:
    return store_mod.current_store()


def health() -> dict[str, Any]:
    return _store().health()


def _norm(text: str) -> str:
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return re.sub(r"^(?:the|a|an) ", "", " ".join(text.split()))


def _rec_item(
    book_id: int,
    rank: int,
    score: float,
    sources: list[str],
    profile: Profile,
    store: ArtifactStore,
    reason: str | None = None,
    signals: dict[str, Any] | None = None,
) -> RecItem:
    evidence = build_evidence(book_id, profile, store, sources, signals)
    return RecItem(
        book_id=book_id,
        rank=rank,
        score=round(float(score), 6),
        sources=sources,
        reason=reason or template_reason(book_id, profile, store, evidence),
        evidence=evidence,
    )


def ranker_scores(
    profile: Profile,
    ids: list[int],
    provenance: dict[int, list[str]],
    store: ArtifactStore,
) -> np.ndarray | None:
    """LightGBM scores over ``features.assemble`` (s13.6 step 1), or None when the bundle has no
    ranker or scoring fails, in which case the caller falls back to the blend (s15.3)."""
    if store.ranker is None:
        return None
    try:
        x = features.assemble(profile, ids, provenance, store)
        pred = np.asarray(store.ranker.predict(x), dtype=np.float32)
    except Exception:  # noqa: BLE001 (model trouble degrades instead of raising, s15.3)
        return None
    if pred.shape != (len(ids),) or not np.all(np.isfinite(pred)):
        return None
    return pred


def _ranked(
    profile: Profile,
    store: ArtifactStore,
    filters: Filters | None,
    exclude: list[int] | None,
    k: int,
    quota_scale: float = 1.0,
    use_ranker: bool = True,
) -> tuple[list[tuple[int, float]], dict[int, list[str]], dict[str, list[int]], str]:
    """Candidates -> filters -> scores -> post-processing; the last element says which ranker
    scored (``"ranker"`` or ``"blend"``)."""
    cfg = store.config
    quotas = {s: int(round(q * quota_scale)) for s, q in cfg["quotas"][profile.bucket].items()}
    sources = cand.generate(profile, store, quotas)
    ids, provenance = cand.union(sources, int(cfg["candidates_total"]))
    ids = cand.apply_filters(ids, provenance, profile, store, filters, exclude,
                             int(cfg["min_reviews_main_feed"]))  # fmt: skip
    if not ids:
        return [], provenance, sources, "blend"
    scores = ranker_scores(profile, ids, provenance, store) if use_ranker else None
    ranked_by = "ranker"
    if scores is None:
        scores, _ = blend(ids, sources, profile, store, cfg["blend"])
        ranked_by = "blend"
    top = postprocess(ids, scores, provenance, profile, store, cfg["postprocess"], k)
    return top, provenance, sources, ranked_by


def recommend(
    state: UserState,
    k: int = 10,
    filters: Filters | None = None,
    exclude: list[int] | None = None,
    use_ranker: bool = True,
) -> RecResponse:
    """Top-k main feed. With ``use_ranker`` the bundled LightGBM ranker scores the candidates
    (D66); the blend ranks when the bundle has none or scoring fails, and ``degraded`` says so
    (s15.3). The feed never calls the LLM (s14, 80 ms budget): reasons are templates, LLM text
    comes from ``explain`` per item on demand (D69)."""
    started = time.perf_counter()
    if k < 1:
        raise ValueError("k must be >= 1")
    store = _store()
    for i in exclude or []:
        store.catalog.check_id(i)
    profile = build_profile(state, store)
    top, provenance, sources, ranked_by = _ranked(
        profile, store, filters, exclude, k, use_ranker=use_ranker
    )
    position = {s: {b: r for r, b in enumerate(lst)} for s, lst in sources.items()}
    items = [
        _rec_item(
            i,
            rank,
            score,
            provenance.get(i, []),
            profile,
            store,
            signals={
                "cf_rank": position.get("cf", {}).get(i),
                "semantic_rank": position.get("semantic", {}).get(i),
                "ranked_by": ranked_by,
            },
        )  # fmt: skip
        for rank, (i, score) in enumerate(top, 1)
    ]
    return RecResponse(
        items=items,
        continue_series=_continue_series(profile, store),
        model_version=store.version,
        profile_version=state.profile_version,
        bucket=profile.bucket,
        degraded=bool(use_ranker and ranked_by != "ranker"),
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
    )


def _continue_series(profile: Profile, store: ArtifactStore) -> list[RecItem]:
    nxt = cand.next_in_series(store, profile)
    return [
        _rec_item(i, r, 1.0, ["item_item"], profile, store, "Next in a series you are reading")
        for r, i in enumerate(nxt, 1)
    ]


def _neighbours(book_id: int, profile: Profile, store: ArtifactStore, k: int,
                also_exclude: set[int]) -> list[tuple[int, float]]:  # fmt: skip
    """Content neighbours of ``book_id`` that pass the feed rules, re-ranked by the blend when
    the reader has a profile (s12.6 ``similar_books``)."""
    cat = store.catalog
    blocked = profile.seen | also_exclude | {book_id}
    pool = [int(j) for j in store.item_nn[book_id] if int(j) not in blocked]
    pool = [j for j in pool if not set(cat.genres_of(j)) & profile.muted_genres
            and cand.series_allows(store, j, profile)]  # fmt: skip
    if not pool:
        return []
    anchor = store.content_rows(np.array([book_id]))[0]
    sims = store.content_rows(np.asarray(pool, dtype=np.int64)) @ anchor
    if profile.cf_vec is not None or profile.semantic_vec is not None:
        sources = {"semantic": pool}
        if profile.cf_vec is not None:
            cf_scores = store.lgcn_rows(np.asarray(pool, dtype=np.int64)) @ profile.cf_vec
            sources["cf"] = [pool[i] for i in np.argsort(-cf_scores, kind="stable")]
        blended, _ = blend(pool, sources, profile, store, store.config["blend"])
        sims = 0.5 * sims + 0.5 * blended
    order = np.argsort(-sims, kind="stable")[:k]
    return [(pool[o], float(sims[o])) for o in order]


def similar_books(book_id: int, k: int = 10, state: UserState | None = None) -> list[RecItem]:
    store = _store()
    book_id = store.catalog.check_id(book_id)
    profile = build_profile(state or UserState(), store)
    stars = profile.stars.get(book_id)
    title = store.catalog.title[book_id]
    reason = f"Similar to {title}" + (f", which you rated {stars:g}★" if stars else "")
    return [
        _rec_item(i, r, s, ["item_item"], profile, store, reason)
        for r, (i, s) in enumerate(_neighbours(book_id, profile, store, k, set()), 1)
    ]


def more_by_author(book_id: int, k: int = 10, state: UserState | None = None) -> list[RecItem]:
    store = _store()
    cat = store.catalog
    book_id = cat.check_id(book_id)
    profile = build_profile(state or UserState(), store)
    author_id = int(cat.author_id[book_id])
    if author_id < 0:
        return []
    same = [i for i in cat.author_items[author_id] if i != book_id and i not in profile.seen
            and not set(cat.genres_of(i)) & profile.muted_genres]  # fmt: skip
    same.sort(key=lambda i: (
        float(cat.series_pos[i]) if not np.isnan(cat.series_pos[i]) else 99.0,
        -float(cat.popularity[i]), i,
    ))  # fmt: skip
    reason = f"More by {cat.author[book_id]}"
    return [
        _rec_item(i, r, float(cat.popularity[i]), ["author"], profile, store, reason)
        for r, i in enumerate(same[:k], 1)
    ]


def semantic_search(query: str, k: int = 20, filters: Filters | None = None) -> list[SearchHit]:
    """Title and author matches first, then MiniLM query embedding against the content index
    through the bundled ONNX encoder (``matched == "semantic"``, D63); title/author only when the
    bundle has no encoder."""
    store = _store()
    cat = store.catalog
    words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 1}
    if not words:
        return []
    allowed = [i for i in range(cat.n_items) if cand.passes_filters(store, i, filters)]
    hits = []
    for i in allowed:
        t = len(words & set(re.findall(r"[a-z0-9]+", cat.title[i].lower())))
        a = len(words & set(re.findall(r"[a-z0-9]+", (cat.author[i] or "").lower())))
        if t or a:
            score = min(1.0, (t + a) / len(words))
            hits.append((score, float(cat.popularity[i]), i, "title" if t >= a else "author"))
    hits.sort(key=lambda h: (-h[0], -h[1], h[2]))
    out = [SearchHit(book_id=i, score=round(s, 4), matched=m) for s, _, i, m in hits[:k]]
    if store.encoder is not None and len(out) < k:
        q = store.encoder.encode([query])[0]
        taken = {h.book_id for h in out}
        ids, scores = store.content_index.search(
            q, k, exclude=taken, allow=allowed if filters is not None else None
        )
        out += [
            SearchHit(book_id=int(i), score=round(float(s), 4), matched="semantic")
            for i, s in zip(ids.tolist(), scores.tolist(), strict=True)
        ][: k - len(out)]
    return out


def resolve_entities(query: str) -> EntityMatch:
    """Author-mode search (unchanged behaviour; ``litsense_core.entities``)."""
    return _module("entities").resolve_entities(query, _store())


def onboarding_grid(n: int = 60) -> list[int]:
    """s12.6: top items per primary genre by Bayesian rating with >= 50 ratings, round-robin."""
    store = _store()
    cat = store.catalog
    min_ratings = GRID_MIN_RATINGS if (cat.rating_count >= GRID_MIN_RATINGS).sum() >= n else 1
    per_genre = []
    for items in cat.genre_items:
        pool = [i for i in items if cat.rating_count[i] >= min_ratings]
        pool.sort(key=lambda i: (-float(cat.bayes_rating[i]), i))
        if pool:
            per_genre.append(pool)
    per_genre.sort(key=len, reverse=True)
    out: list[int] = []
    depth = 0
    while len(out) < n and any(depth < len(p) for p in per_genre):
        for p in per_genre:
            if depth < len(p) and len(out) < n:
                out.append(p[depth])
        depth += 1
    return out


def parse_onboarding(text: str, picks: list[int]) -> OnboardingResult:
    """LLM parse into the vocabularies when the client is available (``method="llm"``), else
    the keyword parse (``embed_only``); see ``litsense_core.onboarding`` (s14.4)."""
    return _module("onboarding").parse_onboarding(text, picks, _store())


def explain(state: UserState, book_id: int) -> Explanation:
    """LLM text when the client is available and the text passes the validators, else the
    template reason (s14.6; ``litsense_core.explain``)."""
    return _module("explain").explain(state, book_id, _store())


def _module(name: str) -> Any:
    """A Phase 6 submodule whose name a public function shadows on the package."""
    return importlib.import_module(f"litsense_core.{name}")


def chat_turn(state: UserState, session: SessionState | None, message: str) -> ChatReply:
    """One chat turn (s14.7): LLM intent parse with the rule parser as fallback, anchors
    resolved through ``entities``, the shared candidate and ranking path, LLM reply text with
    the template as fallback; see ``litsense_core.chat``."""
    return _module("chat").chat_turn(state, session, message, _store())


def book_insights(book_id: int) -> BookInsights:
    """Aspect bars from the bundled item profiles and up to five safe snippets (D63); empty bars
    and snippets on a bundle without them. ``review_summary`` is the offline ``genai.synthesis``
    text when the bundle carries one (D70), else None."""
    store = _store()
    cat = store.catalog
    book_id = cat.check_id(book_id)
    bars: list[AspectBar] = []
    if (
        store.item_pos_rate is not None
        and store.item_neg_rate is not None
        and store.item_mention_rate is not None
    ):
        for a, name in enumerate(store.aspect_names):
            bars.append(
                AspectBar(
                    aspect=name,
                    label=ASPECT_LABELS.get(name, name),
                    pos_rate=round(float(store.item_pos_rate[book_id, a]), 4),
                    neg_rate=round(float(store.item_neg_rate[book_id, a]), 4),
                    mention_rate=round(float(store.item_mention_rate[book_id, a]), 4),
                )
            )
    snippets: list[Snippet] = []
    per_aspect = store.snippets.for_item(book_id) if store.snippets is not None else {}
    for name, slots in per_aspect.items():
        for entry in [*slots.get("pos", []), *slots.get("neg", [])]:
            snippets.append(
                Snippet(
                    text=entry["text"],
                    review_ref=int(entry["review_row"]),
                    helpful_votes=int(entry["helpful_votes"]),
                    aspect=name,
                )
            )
    snippets.sort(key=lambda s: -s.helpful_votes)
    return BookInsights(
        book_id=book_id,
        aspect_bars=bars,
        review_summary=cat.summary_of(book_id),
        snippets=snippets[:5],
        n_reviews=int(cat.n_reviews[book_id]),
    )


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def analyze_user_review(text: str) -> ReviewAnalysis:
    """s12.6: regex sentence split, ONNX embedding, prototype match with tau per aspect, VADER
    compound as polarity, spoiler count from the bundled scorer. Empty on a bundle without the
    encoder or prototypes."""
    store = _store()
    sentences = [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]
    if (
        not sentences
        or store.encoder is None
        or store.aspect_prototypes is None
        or store.aspect_tau is None
    ):
        return ReviewAnalysis(aspects={}, spoiler_sentences=0)
    emb = store.encoder.encode(sentences)
    sims = np.einsum("nd,apd->nap", emb, store.aspect_prototypes).max(axis=2)  # (n, 12)
    hits = sims >= store.aspect_tau
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    vader = SentimentIntensityAnalyzer()
    polarity = np.array([vader.polarity_scores(s)["compound"] for s in sentences], np.float32)
    aspects: dict[str, float] = {}
    for a, name in enumerate(store.aspect_names):
        rows = np.flatnonzero(hits[:, a])
        if rows.size:
            aspects[name] = round(float(np.clip(polarity[rows].mean(), -1.0, 1.0)), 3)
    spoilers = 0
    if store.spoiler is not None:
        pos = np.arange(len(sentences))
        n = np.full(len(sentences), len(sentences))
        spoilers = int(store.spoiler.flags(emb, sentences, pos, n).sum())
    return ReviewAnalysis(aspects=aspects, spoiler_sentences=spoilers)


def metrics_report() -> dict[str, Any]:
    store = _store()
    return {
        "is_mock": False,
        "api_version": __api_version__,
        "bundle_version": store.version,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **store.metrics,
    }


def home_rails(state: UserState) -> HomeRails:
    store = _store()
    cat = store.catalog
    profile = build_profile(state, store)
    top = recommend(state, k=10)
    anchors = profile.liked[:2] or list(state.onboarding_picks[:2])
    trending = [i for i in store.trending_top.tolist() if i not in profile.seen
                and not set(cat.genres_of(i)) & profile.muted_genres][:20]  # fmt: skip
    fav = [a for a, _ in profile.favorite_authors.items() if len(cat.author_items[a]) >= 2]
    by_fav = []
    for a in fav:
        by_fav += [i for i in cat.author_items[a] if i not in profile.seen]
    viewed = list(dict.fromkeys(state.viewed[:3]))
    exploring = []
    for v in viewed:
        exploring += [i for i, _ in _neighbours(v, profile, store, 4, set(viewed))]
    wanted = [e.book_id for e in state.reading_list if e.status == "want_to_read"][:3]
    from_list = []
    for w in wanted:
        from_list += [i for i, _ in _neighbours(w, profile, store, 4, set(wanted))]
    gems: list[tuple[int, float]] = []
    if profile.n >= 5:
        tail_top, _, _, _ = _ranked(profile, store, None, None, 10, quota_scale=1.0)
        gems = [(i, s) for i, s in tail_top if cat.is_longtail[i]]
    return HomeRails(
        top_picks=top.items,
        because_you_liked=[
            BecauseYouLiked(anchor_book_id=a, items=similar_books(a, 10, state)) for a in anchors
        ],
        trending=[
            _rec_item(
                i,
                r,
                float(cat.trending[i]),
                ["popularity"],
                profile,
                store,
                "Popular with readers right now",
            )  # fmt: skip
            for r, i in enumerate(trending, 1)
        ],
        more_by_authors=[
            _rec_item(i, r, float(cat.popularity[i]), ["author"], profile, store)
            for r, i in enumerate(list(dict.fromkeys(by_fav))[:10], 1)
        ],
        continue_exploring=[
            _rec_item(i, r, 0.0, ["item_item"], profile, store)
            for r, i in enumerate(list(dict.fromkeys(exploring))[:10], 1)
        ],
        hidden_gems=[
            _rec_item(
                i, r, s, ["longtail"], profile, store, "A lesser-known pick that fits your taste"
            )
            for r, (i, s) in enumerate(gems[:10], 1)
        ],
        from_reading_list=[
            _rec_item(i, r, 0.0, ["item_item"], profile, store)
            for r, i in enumerate(list(dict.fromkeys(from_list))[:10], 1)
        ],
        continue_series=_continue_series(profile, store),
    )
