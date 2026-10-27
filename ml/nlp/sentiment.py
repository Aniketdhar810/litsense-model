"""Sentence polarity (CLAUDE.md s11.4), stage nlp.sentiment. Offline only.

DistilBERT-SST2 scores only sentences with at least one aspect hit; polarity = p_pos - p_neg in
[-1, 1]. A sentence that hits two or more aspects is split into clauses on
``nlp.clause_split_markers``; each clause is embedded, run through the same aspect detector and
scored on its own, and every aspect of the sentence takes the mean polarity of the clauses that
mention it (the whole-sentence polarity when no clause does). So "Wonderful characters, but the
middle third really dragged." gives characters > 0 and pacing < 0.

Output: ``aspect_polarity.parquet``, one row per (sentence, aspect) hit with the polarity that
aspect gets and whether it came from a clause. Profiles (s11.8) and snippets (s11.9) read it.
"""

import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ml.common.stage import StageContext, stage
from ml.nlp import embed
from ml.nlp.aspect_seeds import ASPECTS
from ml.nlp.aspects import AspectDetector, check_taxonomy, iter_chunks, tau_vector, unpack_mask
from ml.nlp.parts import PartStore

PARQUET_COMPRESSION = "zstd"
POLARITY_SCHEMA = pa.schema(
    [
        ("sent_id", pa.int64()),
        ("aspect", pa.int8()),
        ("polarity", pa.float32()),
        ("from_clause", pa.bool_()),
    ]
)
PART_DTYPE = np.dtype(
    [("sent_id", np.int64), ("aspect", np.int8), ("polarity", np.float32), ("from_clause", bool)]
)
Polarity = Callable[[list[str]], np.ndarray]  # texts -> (n,) float32 in [-1, 1]


def load_polarity_model(model_name: str, batch_size: int, max_length: int) -> Polarity:
    """DistilBERT-SST2 as a function of texts (fp16 on GPU; tests replace this with a stub)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    if device == "cuda":
        model.half()
    labels = {v.lower(): k for k, v in model.config.id2label.items()}
    pos, neg = labels["positive"], labels["negative"]

    def score(texts: list[str]) -> np.ndarray:
        out = np.empty(len(texts), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = tokenizer(
                    texts[start : start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(device)
                probs = torch.softmax(model(**batch).logits.float(), dim=-1)
                out[start : start + len(probs)] = (probs[:, pos] - probs[:, neg]).cpu().numpy()
        return out

    return score


def compile_markers(markers: Sequence[str]) -> re.Pattern[str]:
    """One case-insensitive pattern splitting on any ``nlp.clause_split_markers`` entry."""
    return re.compile("|".join(re.escape(m) for m in markers), re.IGNORECASE)


def split_clauses(text: str, markers: re.Pattern[str]) -> list[str]:
    """Clauses of ``text`` around the markers, trimmed of surrounding punctuation; the whole text
    when no marker is present."""
    clauses = [c.strip(" ,;:") for c in markers.split(text)]
    clauses = [c for c in clauses if c]
    return clauses if len(clauses) >= 2 else [text]


def aspect_polarities(
    sent_ids: np.ndarray,
    texts: Sequence[str],
    hits: np.ndarray,
    polarity: Polarity,
    detector: AspectDetector,
    encoder: embed.TextEncoder,
    markers: re.Pattern[str],
) -> np.ndarray:
    """``PART_DTYPE`` rows for a chunk of sentences that each hit at least one aspect."""
    texts = list(texts)
    sentence_polarity = polarity(texts) if texts else np.zeros(0, dtype=np.float32)
    multi = np.flatnonzero(hits.sum(axis=1) >= 2)
    clause_texts: list[str] = []
    clause_owner: list[int] = []  # row index of the parent sentence
    for row in multi:
        clauses = split_clauses(texts[row], markers)
        if len(clauses) >= 2:
            clause_texts += clauses
            clause_owner += [int(row)] * len(clauses)
    clause_polarity: dict[int, list[tuple[np.ndarray, float]]] = {}
    if clause_texts:
        found = detector.detect(encoder.encode(clause_texts), clause_texts)
        scores = polarity(clause_texts)
        for owner, hit_row, score in zip(clause_owner, found.hits, scores, strict=True):
            clause_polarity.setdefault(owner, []).append((hit_row, float(score)))

    rows: list[tuple[int, int, float, bool]] = []
    for row, (sid, hit_row) in enumerate(zip(sent_ids, hits, strict=True)):
        for a in np.flatnonzero(hit_row):
            values = [s for h, s in clause_polarity.get(row, []) if h[a]]
            if values:
                rows.append((int(sid), int(a), float(np.mean(values)), True))
            else:
                rows.append((int(sid), int(a), float(sentence_polarity[row]), False))
    return np.array(rows, dtype=PART_DTYPE)


def _rows_to_frame(rows: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sent_id": rows["sent_id"].astype(np.int64),
            "aspect": rows["aspect"].astype(np.int8),
            "polarity": rows["polarity"].astype(np.float32),
            "from_clause": rows["from_clause"].astype(bool),
        }
    )


@stage(
    "nlp.sentiment",
    inputs=[
        "data:sentences.parquet",
        "data:sentence_aspects.parquet",
        "artifacts:nlp/aspect_prototypes.npy",
        "artifacts:nlp/descriptor_poles.npy",
    ],
    outputs=["data:aspect_polarity.parquet", "results:nlp/sentiment_checks.json"],
    config_keys=[
        "nlp.sentiment_model",
        "nlp.sentiment_batch",
        "nlp.sentiment_max_length",
        "nlp.clause_split_markers",
        "nlp.pos_threshold",
        "nlp.neg_threshold",
        "nlp.sentence_embed_batch",
        "nlp.embedding_dim",
        "nlp.chunk_rows",
        "aspects.names",
        "aspects.tau_default",
        "aspects.tau",
        "aspects.descriptors",
        "content",
    ],
)
def build_sentiment(ctx: StageContext) -> None:
    """Score every aspect-bearing sentence (clauses when it hits several aspects), checkpointed."""
    params: Mapping[str, Any] = {
        key.removeprefix("nlp."): ctx.param(key)
        for key in ctx.stage.config_keys
        if key.startswith("nlp.")
    }
    aspects_cfg = {
        k: ctx.param(f"aspects.{k}") for k in ("names", "tau_default", "tau", "descriptors")
    }
    check_taxonomy(aspects_cfg)
    content = ctx.param("content")
    dim, chunk_rows = int(params["embedding_dim"]), int(params["chunk_rows"])
    detector = AspectDetector(
        np.load(ctx.path("artifacts:nlp/aspect_prototypes.npy")),
        np.load(ctx.path("artifacts:nlp/descriptor_poles.npy")),
        tau_vector(aspects_cfg),
        dim,
    )
    markers = compile_markers(params["clause_split_markers"])

    aspects = pd.read_parquet(
        ctx.path("data:sentence_aspects.parquet"), columns=["sent_id", "aspect_mask"]
    )
    aspects = aspects[aspects["aspect_mask"] > 0].reset_index(drop=True)
    sentences = pd.read_parquet(ctx.path("data:sentences.parquet"), columns=["sent_id", "text"])
    scored = aspects.merge(sentences, on="sent_id", how="left", validate="one_to_one")
    if scored["text"].isna().any():
        raise ValueError("sentence_aspects refers to sent_ids missing from sentences.parquet")
    n = len(scored)
    ctx.count("sentences_with_hit", n)

    rows = np.zeros(0, dtype=PART_DTYPE)
    if n:
        polarity = load_polarity_model(
            str(params["sentiment_model"]),
            int(params["sentiment_batch"]),
            int(params["sentiment_max_length"]),
        )
        encoder = embed.load_encoder(
            content["model_name"],
            int(params["sentence_embed_batch"]),
            int(content["max_seq_length"]),
        )
        parts = PartStore(ctx.work_dir, "polarity", n, chunk_rows, fixed_rows=False)
        ids, texts = scored["sent_id"].to_numpy(), scored["text"].tolist()
        hits = unpack_mask(scored["aspect_mask"].to_numpy())
        for k, (start, stop) in enumerate(iter_chunks(n, chunk_rows)):
            if parts.is_done(k):
                continue
            part = aspect_polarities(
                ids[start:stop],
                texts[start:stop],
                hits[start:stop],
                polarity,
                detector,
                encoder,
                markers,
            )
            parts.save(k, part)
            ctx.log.info("scored %d / %d sentences", stop, n)
        rows = np.concatenate([parts.load(k) for k in range(parts.n_parts)])

    frame = _rows_to_frame(rows)
    pq.write_table(
        pa.Table.from_pandas(frame, schema=POLARITY_SCHEMA, preserve_index=False),
        ctx.out("data:aspect_polarity.parquet"),
        compression=PARQUET_COMPRESSION,
    )
    pos, neg = float(params["pos_threshold"]), float(params["neg_threshold"])
    per_aspect = {}
    for a, name in enumerate(ASPECTS):
        values = frame.loc[frame["aspect"] == a, "polarity"].to_numpy()
        per_aspect[name] = {
            "mentions": int(len(values)),
            "mean_polarity": round(float(values.mean()), 4) if len(values) else None,
            "share_positive": round(float((values > pos).mean()), 4) if len(values) else None,
            "share_negative": round(float((values < neg).mean()), 4) if len(values) else None,
        }
    checks = {
        "sentences_scored": n,
        "aspect_mentions": int(len(frame)),
        "clause_scored_mentions": int(frame["from_clause"].sum()),
        "pos_threshold": pos,
        "neg_threshold": neg,
        "per_aspect": per_aspect,
    }
    ctx.out("results:nlp/sentiment_checks.json").write_text(
        json.dumps(checks, indent=2) + "\n", encoding="utf-8"
    )
    ctx.count("aspect_mentions", len(frame))
    ctx.count("clause_scored_mentions", checks["clause_scored_mentions"])
