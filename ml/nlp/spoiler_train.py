"""Spoiler classifier (CLAUDE.md s11.7): stages nlp.spoiler_train and nlp.spoiler_flags.

``nlp.spoiler_train`` follows the D20 pattern of ``data.goodreads``: with
``spoiler.train_on_goodreads: false`` (the default) it reads nothing from the network and writes a
regex-only model (empty coefficients), so every downstream stage has one code path. When the user
enables it, the stage streams ``goodreads_reviews_spoiler.json.gz`` (never stored), keeps a
seeded, class-balanced reservoir of ``spoiler.sample_size`` sentences, holds out
``spoiler.holdout_fraction`` of the *books* (group split), embeds with MiniLM, fits
``LogisticRegression(class_weight="balanced")`` over ``spoiler.C_grid`` choosing C by held-out
AUC, and sets the probability threshold to the largest value that still gives spoiler recall >=
``spoiler.target_recall`` on the held-out set. Metrics go to
``results/<v>/nlp/spoiler_metrics.json``, the model to
``artifacts/<v>/serve/spoiler/spoiler_lr.npz``.

``nlp.spoiler_flags`` applies the scorer (``litsense_core/spoiler.py``: probability + regex boost
+ late-position boost) to every Amazon sentence and writes ``sentence_spoiler.parquet``
(``sent_id``, ``spoiler_p``, ``spoiler_score``, ``spoiler_flag``) plus the flagged share.
"""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from litsense_core.spoiler import SpoilerScorer
from ml.common.log import Progress
from ml.common.stage import StageContext, stage
from ml.data.stream import StreamSettings, StreamStats, iter_lines
from ml.nlp import embed
from ml.nlp.aspects import iter_chunks
from ml.nlp.sentences import keep_sentence

SOURCE = "url:goodreads_reviews_spoiler.json.gz"
MODEL = "artifacts:serve/spoiler/spoiler_lr.npz"
METRICS = "results:nlp/spoiler_metrics.json"
FLAGS = "data:sentence_spoiler.parquet"
FLAG_REPORT = "results:nlp/spoiler_flags.json"
FLAG_SCHEMA = pa.schema(
    [
        ("sent_id", pa.int64()),
        ("spoiler_p", pa.float32()),
        ("spoiler_score", pa.float32()),
        ("spoiler_flag", pa.bool_()),
    ]
)
PARQUET_COMPRESSION = "zstd"


@dataclass
class Reservoir:
    """Seeded reservoir of up to ``capacity`` (book_id, sentence) pairs for one label."""

    capacity: int
    rng: np.random.Generator
    seen: int = 0

    def __post_init__(self) -> None:
        self.books: list[str] = []
        self.texts: list[str] = []

    def add(self, book: str, text: str) -> None:
        self.seen += 1
        if len(self.texts) < self.capacity:
            self.books.append(book)
            self.texts.append(text)
            return
        j = int(self.rng.integers(0, self.seen))
        if j < self.capacity:
            self.books[j], self.texts[j] = book, text


def review_sentences(record: Mapping[str, Any]) -> list[tuple[int, str]]:
    """``[(label, sentence), ...]`` of a Goodreads spoiler record; raises on an unexpected shape
    (s0.2 rule 5: inspect, do not assume)."""
    rows = record.get("review_sentences")
    if not isinstance(rows, list):
        raise ValueError("review_sentences is not a list")
    out = []
    for row in rows:
        if not (isinstance(row, list) and len(row) == 2 and row[0] in (0, 1, "0", "1")):
            raise ValueError(f"unexpected review_sentences entry: {row!r}")
        out.append((int(row[0]), str(row[1])))
    return out


def sample_sentences(
    lines: Iterable[str],
    sample_size: int,
    rng: np.random.Generator,
    min_tokens: int,
    max_tokens: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Class-balanced reservoir sample: ``sample_size / 2`` spoiler and non-spoiler sentences of
    ``min_tokens``-``max_tokens`` words, with their ``book_id``."""
    half = sample_size // 2
    pools = {0: Reservoir(half, rng), 1: Reservoir(half, rng)}
    reviews = 0
    for line in lines:
        record = json.loads(line)
        reviews += 1
        book = str(record.get("book_id"))
        for label, text in review_sentences(record):
            if keep_sentence(text, min_tokens, max_tokens):
                pools[label].add(book, text)
    frame = pd.concat(
        [
            pd.DataFrame({"book_id": pool.books, "text": pool.texts, "label": label})
            for label, pool in pools.items()
        ],
        ignore_index=True,
    )
    counts = {
        "reviews_streamed": reviews,
        "sentences_seen_spoiler": pools[1].seen,
        "sentences_seen_non_spoiler": pools[0].seen,
        "sampled_spoiler": len(pools[1].texts),
        "sampled_non_spoiler": len(pools[0].texts),
    }
    return frame, counts


def group_holdout(book_ids: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask: True for rows whose book is in the held-out ``fraction`` of books."""
    books = np.unique(book_ids)
    held = set(rng.permutation(books)[: int(round(fraction * len(books)))].tolist())
    return np.array([b in held for b in book_ids], dtype=bool)


def recall_threshold(probabilities: np.ndarray, labels: np.ndarray, target_recall: float) -> float:
    """The largest threshold whose spoiler recall on (probabilities, labels) is >= target."""
    positives = np.sort(probabilities[labels == 1])
    if len(positives) == 0:
        return 0.5
    # flagging p >= t catches every positive with p >= t; need share >= target
    k = int(np.ceil(target_recall * len(positives)))  # at least k positives must be >= t
    return float(positives[max(len(positives) - k, 0)])


def fit_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_hold: np.ndarray,
    y_hold: np.ndarray,
    c_grid: list[float],
    seed: int,
) -> tuple[np.ndarray, float, float, dict[str, float]]:
    """``(coef, intercept, best_C, auc_by_C)`` by held-out AUC over ``c_grid``."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    best: tuple[float, Any, float] | None = None
    aucs: dict[str, float] = {}
    for c in c_grid:
        model = LogisticRegression(
            C=float(c), class_weight="balanced", max_iter=2000, random_state=seed
        )
        model.fit(x_train, y_train)
        auc = float(roc_auc_score(y_hold, model.predict_proba(x_hold)[:, 1]))
        aucs[str(c)] = round(auc, 4)
        if best is None or auc > best[0]:
            best = (auc, model, float(c))
    assert best is not None
    _, model, c = best
    return model.coef_[0].astype(np.float32), float(model.intercept_[0]), c, aucs


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney, ties at half credit), identical to sklearn's
    ``roc_auc_score`` without importing scikit-learn, which the CI subset does not install."""
    from scipy.stats import rankdata

    positive = np.asarray(labels) == 1
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(np.asarray(scores, dtype=np.float64))
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def holdout_metrics(
    probabilities: np.ndarray, labels: np.ndarray, threshold: float
) -> dict[str, float]:
    flagged = probabilities >= threshold
    tp = int((flagged & (labels == 1)).sum())
    fp = int((flagged & (labels == 0)).sum())
    fn = int((~flagged & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    auc = roc_auc(labels, probabilities)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4),
        "threshold": round(float(threshold), 4),
        "flagged_share": round(float(flagged.mean()), 4) if len(flagged) else 0.0,
    }


@stage(
    "nlp.spoiler_train",
    inputs=[SOURCE],
    outputs=[MODEL, METRICS],
    config_keys=[
        "spoiler",
        "data.stream",
        "content",
        "nlp.embedding_dim",
        "nlp.sentence_embed_batch",
        "nlp.min_sentence_tokens",
        "nlp.max_sentence_tokens",
    ],
)
def train_spoiler(ctx: StageContext) -> None:
    """Fit the spoiler classifier on Goodreads sentences, or write the regex-only model."""
    params: Mapping[str, Any] = ctx.param("spoiler")
    patterns = list(params["regex_patterns"])
    boosts = (
        float(params["regex_boost"]),
        float(params["late_position_boost"]),
        float(params["late_fraction"]),
    )
    enabled = bool(params["train_on_goodreads"])
    ctx.notes["train_on_goodreads"] = enabled
    if not enabled:
        SpoilerScorer.regex_only(patterns, *boosts).to_npz(ctx.out(MODEL))
        metrics = {"model": "regex_only", "note": "spoiler.train_on_goodreads is false"}
        ctx.out(METRICS).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        ctx.log.info("spoiler.train_on_goodreads is false: wrote the regex-only spoiler model")
        return

    settings = StreamSettings.from_config(ctx.param("data.stream"))
    stats = StreamStats(ctx.source(SOURCE))
    progress = Progress(ctx.log, SOURCE.removeprefix("url:"))

    def lines() -> Iterable[str]:
        for line in iter_lines(stats.source, settings, limit=ctx.sample, stats=stats):
            progress.update(stats.bytes_read, stats.size, f"; {stats.lines:,} lines")
            yield line

    sample, counts = sample_sentences(
        lines(), int(params["sample_size"]), ctx.rng,
        int(ctx.param("nlp.min_sentence_tokens")), int(ctx.param("nlp.max_sentence_tokens")),
    )  # fmt: skip
    ctx.notes["source"] = stats.as_note()
    if sample["label"].nunique() < 2:
        raise ValueError("the Goodreads sample has only one class; check the source")
    held = group_holdout(sample["book_id"].to_numpy(), float(params["holdout_fraction"]), ctx.rng)
    content = ctx.param("content")
    dim = int(ctx.param("nlp.embedding_dim"))
    encoder = embed.load_encoder(
        content["model_name"],
        int(ctx.param("nlp.sentence_embed_batch")),
        int(content["max_seq_length"]),
    )
    x = encoder.encode(sample["text"].tolist()).astype(np.float32)
    if x.shape != (len(sample), dim):
        raise ValueError(f"encoder returned {x.shape}, expected {(len(sample), dim)}")
    y = sample["label"].to_numpy(np.int64)
    coef, intercept, best_c, aucs = fit_classifier(
        x[~held], y[~held], x[held], y[held], [float(c) for c in params["C_grid"]], ctx.seed
    )
    scorer = SpoilerScorer(coef, intercept, 0.0, patterns, *boosts)
    p_hold = scorer.probability(x[held])
    threshold = recall_threshold(p_hold, y[held], float(params["target_recall"]))
    scorer.threshold = threshold
    scorer.to_npz(ctx.out(MODEL))
    metrics = {
        "model": "logistic_regression",
        "C": best_c,
        "auc_by_C": aucs,
        "target_recall": float(params["target_recall"]),
        "holdout": holdout_metrics(p_hold, y[held], threshold),
        "train_rows": int((~held).sum()),
        "holdout_rows": int(held.sum()),
        "holdout_books": int(len(np.unique(sample["book_id"].to_numpy()[held]))),
        **counts,
    }
    ctx.out(METRICS).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    for key, value in counts.items():
        ctx.count(key, value)
    ctx.count("train_rows", metrics["train_rows"])
    ctx.count("holdout_rows", metrics["holdout_rows"])
    ctx.notes["holdout"] = metrics["holdout"]


@stage(
    "nlp.spoiler_flags",
    inputs=["data:sentences.parquet", "data:sent_emb.f16.npy", MODEL],
    outputs=[FLAGS, FLAG_REPORT],
    config_keys=["nlp.embedding_dim", "nlp.chunk_rows"],
)
def flag_sentences(ctx: StageContext) -> None:
    """Score every Amazon sentence with the spoiler scorer (probability + boosts)."""
    scorer = SpoilerScorer.from_npz(ctx.path(MODEL))
    dim, chunk_rows = int(ctx.param("nlp.embedding_dim")), int(ctx.param("nlp.chunk_rows"))
    sentences = pd.read_parquet(
        ctx.path("data:sentences.parquet"), columns=["sent_id", "pos", "n_in_review", "text"]
    )
    emb = np.load(ctx.path("data:sent_emb.f16.npy"), mmap_mode="r")
    if emb.shape != (len(sentences), dim):
        raise ValueError(f"sent_emb shape {emb.shape} != ({len(sentences)}, {dim})")
    if scorer.has_model and scorer.coef.shape[0] != dim:
        raise ValueError("spoiler model dim differs from nlp.embedding_dim")
    n = len(sentences)
    flagged = regex_hits = late_hits = 0
    texts = sentences["text"].tolist()
    pos = sentences["pos"].to_numpy()
    n_in = sentences["n_in_review"].to_numpy()
    with pq.ParquetWriter(ctx.out(FLAGS), FLAG_SCHEMA, compression=PARQUET_COMPRESSION) as out:
        for start, stop in iter_chunks(n, chunk_rows):
            chunk = np.asarray(emb[start:stop], dtype=np.float32)
            p = scorer.probability(chunk)
            score = scorer.score(chunk, texts[start:stop], pos[start:stop], n_in[start:stop])
            flag = score >= scorer.threshold - 1e-6
            flagged += int(flag.sum())
            regex_hits += int(scorer.regex_hits(texts[start:stop]).sum())
            late_hits += int(scorer.late(pos[start:stop], n_in[start:stop]).sum())
            frame = pd.DataFrame(
                {
                    "sent_id": sentences["sent_id"].to_numpy()[start:stop],
                    "spoiler_p": p,
                    "spoiler_score": score,
                    "spoiler_flag": flag,
                }
            )
            out.write_table(pa.Table.from_pandas(frame, schema=FLAG_SCHEMA, preserve_index=False))
    report = {
        "sentences": n,
        "flagged": flagged,
        "flagged_share": round(flagged / n, 4) if n else None,
        "regex_hits": regex_hits,
        "late_position": late_hits,
        "model": "logistic_regression" if scorer.has_model else "regex_only",
        "threshold": scorer.threshold,
    }
    ctx.out(FLAG_REPORT).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    ctx.count("sentences", n)
    ctx.count("flagged", flagged)
    ctx.count("regex_hits", regex_hits)
