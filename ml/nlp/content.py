"""Stage nlp.content (CLAUDE.md s10.1): item content embeddings and content neighbours.

Text per book: "{title}. By {author}. Genres: {labels}. {blurb}". The header goes first and the
blurb fills the remaining token budget, so title, author and genres are never cut (s2.2 #12). The
blurb is ``features`` (usually the publisher's premise), else ``description`` (D24, D40). No review
text and no LLM summary ever goes in (s1.5): review semantics live in separate centroids. Metadata
does not depend on time, so one matrix serves every cutoff.
"""

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ml.common.stage import StageContext, stage
from ml.nlp import embed
from ml.nlp.embed import TextEncoder

SPECIAL_TOKENS = 2  # [CLS] and [SEP]
NEIGHBOURS = 50
ANISOTROPY_PAIRS = 2000
NN_SAMPLES = 20
NN_SHOWN = 5


def blurb(book: Mapping[str, Any]) -> str:
    """The book's premise text: ``features`` if present, else ``description``."""
    for key in ("features", "description"):
        text = book.get(key)
        if isinstance(text, str) and text.strip():
            return text.strip()
    return ""


def item_text(book: Mapping[str, Any], encoder: TextEncoder, max_seq_length: int) -> str:
    """Header first, blurb truncated by tokens to fill the rest of ``max_seq_length``."""
    genres = book.get("genres")
    genres = ", ".join([] if genres is None else list(genres))  # a numpy array from parquet
    header = f"{book['title']}. By {book.get('author') or 'Unknown'}. Genres: {genres}."
    budget = max_seq_length - SPECIAL_TOKENS - encoder.count_tokens(header)
    text = blurb(book)
    if not text or budget <= 0:
        return header
    return f"{header} {encoder.truncate(text, budget)}"


def mean_pairwise_cosine(emb: np.ndarray, pairs: int, rng: np.random.Generator) -> float:
    """Mean cosine over random distinct item pairs (anisotropy check, s2.2 #9)."""
    if len(emb) < 2:
        return float("nan")
    a = rng.integers(0, len(emb), pairs)
    b = (a + rng.integers(1, len(emb), pairs)) % len(emb)
    return float(np.mean(np.sum(emb[a] * emb[b], axis=1)))


def nearest_neighbours(emb: np.ndarray, k: int, batch: int = 2048) -> np.ndarray:
    """(n, k) int32: top-k cosine neighbours of every item, excluding itself, best first."""
    k = min(k, len(emb) - 1)
    out = np.empty((len(emb), k), dtype=np.int32)
    for start in range(0, len(emb), batch):
        sims = emb[start : start + batch] @ emb.T
        rows = np.arange(len(sims))
        sims[rows, start + rows] = -np.inf
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sims, part, 1), axis=1, kind="stable")
        out[start : start + len(sims)] = np.take_along_axis(part, order, 1)
    return out


@stage(
    "nlp.content",
    inputs=["data:books.parquet"],
    outputs=[
        "artifacts:content/content_item.npy",
        "artifacts:content/item_nn.npy",
        "results:content/content_checks.json",
        "results:content/content_nn_samples.txt",
    ],
    config_keys=["content"],
)
def build_content(ctx: StageContext) -> None:
    """Embed every book's metadata text; write neighbours and sanity checks."""
    params = ctx.param("content")
    max_len = int(params["max_seq_length"])
    encoder = embed.load_encoder(params["model_name"], int(params["batch_size"]), max_len)
    books = pd.read_parquet(ctx.path("data:books.parquet")).sort_values("i")
    if list(books["i"]) != list(range(len(books))):
        raise ValueError("books.parquet rows must be items 0..n-1 in order")
    texts = [item_text(b, encoder, max_len) for b in books.to_dict("records")]
    emb = encoder.encode(texts).astype(np.float32)
    if emb.shape[0] != len(books):
        raise ValueError(f"encoder returned {emb.shape[0]} rows for {len(books)} books")
    neighbours = nearest_neighbours(emb, NEIGHBOURS)
    np.save(ctx.out("artifacts:content/content_item.npy"), emb)
    np.save(ctx.out("artifacts:content/item_nn.npy"), neighbours)

    anisotropy = mean_pairwise_cosine(emb, ANISOTROPY_PAIRS, ctx.rng)
    if anisotropy > float(params["anisotropy_warn"]):
        ctx.log.warning("mean pairwise cosine %.3f is above anisotropy_warn", anisotropy)
    checks = {
        "items": len(books),
        "dim": int(emb.shape[1]),
        "mean_pairwise_cosine": round(anisotropy, 4),
        "anisotropy_warn": float(params["anisotropy_warn"]),
        "items_without_blurb": int(sum(not blurb(b) for b in books.to_dict("records"))),
    }
    path = ctx.out("results:content/content_checks.json")
    path.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    titles = books["title"].tolist()
    sample = ctx.rng.choice(len(books), size=min(NN_SAMPLES, len(books)), replace=False)
    lines = []
    for i in sorted(sample.tolist()):
        lines.append(titles[i])
        lines += [f"    {titles[j]}" for j in neighbours[i, :NN_SHOWN]]
    ctx.out("results:content/content_nn_samples.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    for key, value in checks.items():
        if isinstance(value, int):
            ctx.count(key, value)
    ctx.notes["mean_pairwise_cosine"] = checks["mean_pairwise_cosine"]
