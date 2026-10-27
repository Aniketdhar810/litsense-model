"""A hand-built 12-book ``ArtifactStore`` for the GenAI unit tests (no pipeline, no torch, no
ONNX). Shapes follow the bundle (CLAUDE.md s4.6) at toy sizes; values are seeded randoms except
the titles, authors, genres and snippets, which the entity, validator and chat tests rely on."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from litsense_core.spoiler import SpoilerScorer
from litsense_core.store import ArtifactStore, Catalog, SnippetFile
from litsense_core.vector_index import VectorIndex

ASPECTS = [
    "plot", "writing_style", "characters", "world_building", "pacing", "editing_quality",
    "dialogue", "humor", "emotional_impact", "ending", "complexity", "tone_dark",
]  # fmt: skip
BOOKS: list[tuple[str, str | None, list[str], int, int, str | None, float]] = [
    # title, author, genres, year, pages, series, series_pos
    ("Project Hail Mary", "Andy Weir", ["Science Fiction"], 2021, 476, None, np.nan),
    ("The Martian", "Andy Weir", ["Science Fiction"], 2011, 369, None, np.nan),
    ("Artemis", "Andy Weir", ["Science Fiction", "Thriller"], 2017, 305, None, np.nan),
    ("Circe", "Madeline Miller", ["Fantasy", "Historical Fiction"], 2018, 393, None, np.nan),
    ("The Song of Achilles", "Madeline Miller", ["Fantasy"], 2011, 352, None, np.nan),
    ("Mistborn: The Final Empire", "Brandon Sanderson", ["Fantasy"], 2006, 541, "mistborn", 1.0),
    ("Mistborn: The Well of Ascension", "Brandon Sanderson", ["Fantasy"], 2007, 590, "mistborn",
     2.0),
    ("Mistborn: The Hero of Ages", "Brandon Sanderson", ["Fantasy"], 2008, 572, "mistborn", 3.0),
    ("Gone Girl", "Gillian Flynn", ["Thriller", "Mystery"], 2012, 415, None, np.nan),
    ("The Thursday Murder Club", "Richard Osman", ["Mystery"], 2020, 382, None, np.nan),
    ("Dune", "Frank Herbert", ["Science Fiction"], 1965, 412, None, np.nan),
    ("Untitled Debut", None, ["Literary Fiction"], 2023, 210, None, np.nan),
]  # fmt: skip
SNIPPETS = {
    0: {"plot": {"pos": [{"text": "The problem-solving science kept me hooked from the first page.",
                          "review_row": 10, "helpful_votes": 40, "polarity": 0.9}], "neg": []},
        "humor": {"pos": [{"text": "Funny in exactly the right places.", "review_row": 11,
                           "helpful_votes": 5, "polarity": 0.8}], "neg": []}},
    3: {"writing_style": {"pos": [{"text": "The prose is gorgeous and lyrical.", "review_row": 20,
                                   "helpful_votes": 12, "polarity": 0.9}],
                          "neg": [{"text": "The middle dragged on forever.", "review_row": 21,
                                   "helpful_votes": 3, "polarity": -0.7}]}},
}  # fmt: skip
SPOILER_PATTERNS = [r"\bdies\b", r"\bdied\b", r"\bkilled\b", "turns out", "the ending",
                    "twist (is|was)", "revealed to be", "the killer", "ends up"]  # fmt: skip
CONTENT_DIM, LGCN_DIM, NN = 8, 4, 5


def build_store(folder: Path, config_llm: dict[str, Any] | None = None) -> ArtifactStore:
    rng = np.random.default_rng(7)
    n = len(BOOKS)
    titles = [b[0] for b in BOOKS]
    authors = [b[1] for b in BOOKS]
    genres = [list(b[2]) for b in BOOKS]
    genre_labels = sorted({g for gs in genres for g in gs})
    author_labels = sorted({a for a in authors if a})
    series_labels = sorted({b[5] for b in BOOKS if b[5]})
    gi = {g: k for k, g in enumerate(genre_labels)}
    indptr = np.zeros(n + 1, dtype=np.int64)
    indices: list[int] = []
    for i, gs in enumerate(genres):
        indices.extend(gi[g] for g in gs)
        indptr[i + 1] = len(indices)
    author_id = np.array([author_labels.index(a) if a else -1 for a in authors], np.int32)
    series_id = np.array([series_labels.index(b[5]) if b[5] else -1 for b in BOOKS], np.int32)
    counts = rng.integers(20, 400, size=n).astype(np.int32)
    content = rng.normal(size=(n, CONTENT_DIM)).astype(np.float32)
    content[1] = content[0] + 0.05 * rng.normal(size=CONTENT_DIM)  # The Martian ~ Hail Mary
    content /= np.linalg.norm(content, axis=1, keepdims=True)
    sims = content @ content.T
    np.fill_diagonal(sims, -np.inf)
    item_nn = np.argsort(-sims, axis=1)[:, :NN].astype(np.int32)
    catalog = Catalog(
        year=np.array([b[3] for b in BOOKS], np.int32),
        pages=np.array([b[4] for b in BOOKS], np.int32),
        series_id=series_id,
        series_pos=np.array([b[6] for b in BOOKS], np.float32),
        genre_indptr=indptr,
        genre_indices=np.asarray(indices, np.int32),
        primary_genre=np.array([gi[gs[0]] for gs in genres], np.int32),
        bayes_rating=rng.uniform(3.5, 4.8, size=n).astype(np.float32),
        rating_count=counts,
        rating_std=rng.uniform(0.5, 1.2, size=n).astype(np.float32),
        popularity=(np.log1p(counts) / np.log1p(counts.max())).astype(np.float32),
        trending=counts.astype(np.float32),
        is_longtail=counts < 100,
        n_reviews=(counts // 2).astype(np.int32),
        author_id=author_id,
        title=titles,
        author=authors,
        genres=genres,
        image=[None] * n,
        description=[f"Description of {t}." for t in titles],
        canonical_id=[f"am_{i}" for i in range(n)],
        series_name=[b[5] for b in BOOKS],
        genre_labels=genre_labels,
        author_labels=author_labels,
        series_labels=series_labels,
        author_items=[
            [i for i in range(n) if author_id[i] == k] for k in range(len(author_labels))
        ],
        series_items=[
            [i for i in range(n) if series_id[i] == k] for k in range(len(series_labels))
        ],
        genre_items=[
            [i for i in range(n) if gi[genres[i][0]] == k] for k in range(len(genre_labels))
        ],
    )
    folder.mkdir(parents=True, exist_ok=True)
    offsets = np.zeros(n + 1, dtype=np.int64)
    with (folder / "evidence_snippets.jsonl").open("wb") as f:
        for i in range(n):
            line = json.dumps(SNIPPETS.get(i, {})).encode("utf-8")
            f.write(line + b"\n")
            offsets[i + 1] = offsets[i] + len(line) + 1
    lgcn = rng.normal(size=(n, LGCN_DIM)).astype(np.float32)
    aspects = rng.uniform(0.2, 0.8, size=(n, 12)).astype(np.float32)
    aspects[0, 0], aspects[0, 7] = 0.9, 0.85  # Hail Mary: plot and humor praised
    aspects[3, 1] = 0.9  # Circe: prose
    mention = rng.uniform(0.05, 0.5, size=(n, 12)).astype(np.float32)
    pos_rate = rng.uniform(0.3, 0.9, size=(n, 12)).astype(np.float32)
    pos_rate[0, 0], mention[0, 0] = 0.74, 0.9  # plot is Hail Mary's top aspect (has a snippet)
    config = {
        "buckets": {"cold": [0, 0], "warming": [1, 4], "blending": [5, 19], "warm": [20, 10**6]},
        "quotas": {
            b: {"cf": 6, "semantic": 6, "item_item": 6, "popularity": 6, "author": 4, "aspect": 4,
                "longtail": 4}
            for b in ("cold", "warming", "blending", "warm")
        },
        "candidates_total": 12,
        "blend": {"alpha_k": 8, "w_pop": 0.1, "w_aspect": 0.15},
        "postprocess": {"mmr_lambda": 0.75, "mmr_pool": 12, "genre_cap": 4, "explore_slot": 8},
        "overrides": {"pinned_aspect_boost": 0.1},
        "min_reviews_main_feed": 0,
        "aspects": {"names": ASPECTS, "tau_default": 0.45, "tau": {}, "shrinkage_k": 5,
                    "user_half_life_days": 730, "user_explicit_weight": 0.6, "explicit_conf_k": 3,
                    "descriptors": ["darkness", "pace", "complexity_level"],
                    "rating_weights": {5: 1.0, 4: 0.5, 3: 0.0, 2: -0.5, 1: -1.0}},
        "serving": {"onnx_threads": 1, "verify_hashes_on_load": False},
    }  # fmt: skip
    if config_llm is not None:
        config["llm"] = config_llm
    return ArtifactStore(
        bundle_dir=folder,
        version="fake-1",
        data_version="tiny",
        cutoff=1_700_000_000_000,
        manifest={"n_items": n},
        config=config,
        catalog=catalog,
        lgcn_item=lgcn.astype(np.float16),
        lgcn_item_norm=(lgcn / np.linalg.norm(lgcn, axis=1, keepdims=True)).astype(np.float16),
        content_item=content.astype(np.float16),
        item_nn=item_nn,
        trending_top=np.argsort(-counts, kind="stable").astype(np.int64),
        content_index=VectorIndex(content, use_faiss=False),
        corpus_users={},
        corpus_vectors=np.zeros((0, LGCN_DIM), np.float32),
        corpus_counts=np.zeros(0, np.int64),
        metrics={"models": []},
        absent=("ranker.txt", "minilm.onnx"),
        aspect_names=list(ASPECTS),
        aspect_tau=np.full(12, 0.45, np.float32),
        item_aspects=aspects,
        item_mention_rate=mention,
        item_pos_rate=pos_rate,
        item_neg_rate=(1 - pos_rate) * 0.5,
        item_descriptors=rng.uniform(-0.8, 0.8, size=(n, 3)).astype(np.float32),
        aspect_means=aspects.mean(axis=0),
        aspect_topitems={
            a: [int(i) for i in np.argsort(-aspects[:, k])[:6]] for k, a in enumerate(ASPECTS)
        },  # fmt: skip
        snippets=SnippetFile(folder / "evidence_snippets.jsonl", offsets),
        spoiler=SpoilerScorer.regex_only(SPOILER_PATTERNS, 0.15, 0.05, 0.3),
    )
