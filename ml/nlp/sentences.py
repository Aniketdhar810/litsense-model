"""Stage nlp.sentences (CLAUDE.md s11.1): the sentence table and its MiniLM embeddings.

``reviews.parquet`` rows with ``text_ok`` are split into sentences (spaCy ``blank("en")`` with the
sentencizer); sentences of ``nlp.min_sentence_tokens`` to ``nlp.max_sentence_tokens`` whitespace
tokens are kept. ``pos`` is the sentence's index among every sentence of its review and
``n_in_review`` that review's sentence count, both before the length filter, so "last 30% of the
review" (s11.7) means the same thing for every sentence. Every kept sentence is embedded with the
same MiniLM model as the item content (mean-pooled, L2-normalized, s2.2 #9) into a float16
``sent_emb.f16.npy`` whose row index is ``sent_id``. The embedding pass is checkpointed in parts of
``nlp.chunk_rows`` sentences (``ml/nlp/parts.py``).

The table is built once over all reviews; cutoffs are applied by filtering on ``ts`` when
aggregating (``ml/nlp/cutoffs.py``).
"""

import json
import os
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ml.common.stage import StageContext, stage
from ml.nlp import embed
from ml.nlp.parts import PartStore

SENTENCE_SCHEMA = pa.schema(
    [
        ("sent_id", pa.int64()),
        ("review_row", pa.int64()),
        ("u", pa.int32()),
        ("i", pa.int32()),
        ("ts", pa.int64()),
        ("rating", pa.float32()),
        ("helpful_vote", pa.int32()),
        ("pos", pa.int32()),
        ("n_in_review", pa.int32()),
        ("text", pa.string()),
    ]
)
REVIEW_COLUMNS = ["review_row", "u", "i", "ts", "rating", "helpful_vote", "text", "text_ok"]
PARQUET_COMPRESSION = "zstd"
REVIEW_BATCH_ROWS = 20_000
SPACY_BATCH = 1000

Splitter = Callable[[list[str]], list[list[str]]]  # texts -> sentences per text


def load_splitter(n_process: int, batch_size: int = SPACY_BATCH) -> Splitter:
    """spaCy ``blank("en")`` + sentencizer over ``nlp.pipe`` (tests replace this with a stub)."""
    import spacy

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

    def split(texts: list[str]) -> list[list[str]]:
        docs = nlp.pipe(texts, batch_size=batch_size, n_process=n_process)
        return [[s.text.strip() for s in doc.sents if s.text.strip()] for doc in docs]

    return split


def keep_sentence(text: str, min_tokens: int, max_tokens: int) -> bool:
    """Length rule: ``min_tokens`` to ``max_tokens`` whitespace tokens."""
    return min_tokens <= len(text.split()) <= max_tokens


def sentence_rows(
    reviews: pd.DataFrame,
    splitter: Splitter,
    min_tokens: int,
    max_tokens: int,
    first_id: int,
) -> tuple[pd.DataFrame, int]:
    """Sentences of one batch of ``text_ok`` reviews, ids from ``first_id``; also returns how many
    sentences the length rule dropped."""
    per_review = splitter(reviews["text"].tolist())
    columns: dict[str, list[Any]] = {name: [] for name in SENTENCE_SCHEMA.names}
    dropped = 0
    sent_id = first_id
    for review, sentences in zip(reviews.itertuples(index=False), per_review, strict=True):
        for pos, text in enumerate(sentences):
            if not keep_sentence(text, min_tokens, max_tokens):
                dropped += 1
                continue
            columns["sent_id"].append(sent_id)
            columns["review_row"].append(int(review.review_row))
            columns["u"].append(int(review.u))
            columns["i"].append(int(review.i))
            columns["ts"].append(int(review.ts))
            columns["rating"].append(float(review.rating))
            columns["helpful_vote"].append(int(review.helpful_vote))
            columns["pos"].append(pos)
            columns["n_in_review"].append(len(sentences))
            columns["text"].append(text)
            sent_id += 1
    return pd.DataFrame(columns), dropped


def write_sentence_table(
    reviews_path: Path,
    out_path: Path,
    splitter: Splitter,
    min_tokens: int,
    max_tokens: int,
    batch_rows: int = REVIEW_BATCH_ROWS,
) -> dict[str, int]:
    """Stream ``reviews.parquet`` into the sentence table at ``out_path`` (written atomically)."""
    counts = {"reviews": 0, "reviews_text_ok": 0, "sentences_split": 0, "sentences": 0}
    tmp = out_path.with_name(out_path.name + ".tmp")
    source = pq.ParquetFile(reviews_path)
    with pq.ParquetWriter(tmp, SENTENCE_SCHEMA, compression=PARQUET_COMPRESSION) as out:
        for batch in source.iter_batches(batch_size=batch_rows, columns=REVIEW_COLUMNS):
            frame = batch.to_pandas()
            counts["reviews"] += len(frame)
            frame = frame[frame["text_ok"]]
            counts["reviews_text_ok"] += len(frame)
            if frame.empty:
                continue
            rows, dropped = sentence_rows(
                frame, splitter, min_tokens, max_tokens, counts["sentences"]
            )
            counts["sentences_split"] += len(rows) + dropped
            counts["sentences"] += len(rows)
            out.write_table(
                pa.Table.from_pandas(rows, schema=SENTENCE_SCHEMA, preserve_index=False)
            )
    os.replace(tmp, out_path)
    return counts


def embed_sentences(
    texts: list[str],
    encoder: embed.TextEncoder,
    parts: PartStore,
    dim: int,
    log: Any = None,
) -> None:
    """Embed ``texts`` part by part into ``parts`` (float16), resuming after the last good part."""
    for k in range(parts.first_pending(), parts.n_parts):
        start, stop = parts.bounds(k)
        vectors = encoder.encode(texts[start:stop])
        if vectors.shape != (stop - start, dim):
            raise ValueError(
                f"encoder returned shape {vectors.shape}, expected {(stop - start, dim)}; "
                "check nlp.embedding_dim"
            )
        parts.save(k, vectors.astype(np.float16))
        if log is not None:
            log.info("embedded %d / %d sentences", stop, parts.n_rows)


def _readable_parquet(path: Path) -> bool:
    try:
        pq.read_metadata(path)
    except (OSError, pa.ArrowException):
        return False
    return True


@stage(
    "nlp.sentences",
    inputs=["data:reviews.parquet"],
    outputs=["data:sentences.parquet", "data:sent_emb.f16.npy"],
    config_keys=[
        "nlp.spacy_n_process",
        "nlp.min_sentence_tokens",
        "nlp.max_sentence_tokens",
        "nlp.sentence_embed_batch",
        "nlp.embedding_dim",
        "nlp.chunk_rows",
        "content",
    ],
)
def build_sentences(ctx: StageContext) -> None:
    """Split every usable review into sentences and embed them (checkpointed)."""
    params: Mapping[str, Any] = {
        key.removeprefix("nlp."): ctx.param(key)
        for key in ctx.stage.config_keys
        if key.startswith("nlp.")
    }
    content: Mapping[str, Any] = ctx.param("content")
    min_tokens, max_tokens = int(params["min_sentence_tokens"]), int(params["max_sentence_tokens"])
    dim, chunk_rows = int(params["embedding_dim"]), int(params["chunk_rows"])

    table = ctx.work_dir / "sentences.parquet"
    counts_file = ctx.work_dir / "counts.json"
    if _readable_parquet(table) and counts_file.exists():
        ctx.log.info("reusing the sentence table checkpoint")
        counts = json.loads(counts_file.read_text(encoding="utf-8"))
    else:
        splitter = load_splitter(int(params["spacy_n_process"]))
        counts = write_sentence_table(
            ctx.path("data:reviews.parquet"), table, splitter, min_tokens, max_tokens
        )
        counts_file.write_text(json.dumps(counts), encoding="utf-8")
    sentences = pd.read_parquet(table, columns=["sent_id", "text"])
    n = len(sentences)
    if n and not np.array_equal(sentences["sent_id"].to_numpy(), np.arange(n)):
        raise ValueError("sent_id must be the row index 0..n-1")

    emb_path = ctx.out("data:sent_emb.f16.npy")
    if n == 0:
        ctx.log.warning("no usable sentences; writing an empty embedding matrix")
        np.save(emb_path, np.zeros((0, dim), dtype=np.float16))
    else:
        encoder = embed.load_encoder(
            content["model_name"],
            int(params["sentence_embed_batch"]),
            int(content["max_seq_length"]),
        )
        parts = PartStore(ctx.work_dir, "emb", n, chunk_rows)
        embed_sentences(sentences["text"].tolist(), encoder, parts, dim, ctx.log)
        out = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float16, shape=(n, dim))
        parts.concat(out)
        out.flush()
        del out
    shutil.copyfile(table, ctx.out("data:sentences.parquet"))
    for key, value in counts.items():
        ctx.count(key, value)
    ctx.notes["embedding_dim"] = dim
