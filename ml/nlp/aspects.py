"""Aspect detection (CLAUDE.md s11.3), stage nlp.aspects.

A sentence mentions aspect ``a`` when the cosine between its embedding and the closest of the
aspect's eight prototype sentences reaches ``tau[a]`` (``aspects.tau``, default
``aspects.tau_default``), or when its text matches the aspect's lexicon with word boundaries
(``ml/nlp/aspect_seeds.py``). Per sentence the stage stores a uint16 bitmask of hits (bit = aspect
index), the lexicon-only mask, the best prototype similarity, and the three descriptor scores
(cos to the mean plus pole minus cos to the mean minus pole, NaN unless the linked aspect hit).

Prototype and pole embeddings come from the same MiniLM model as everything else and are saved
as artifacts (``aspect_prototypes.npy`` (12, 8, d), ``descriptor_poles.npy`` (3, 2, d)) so the
sentiment stage and, later, the serving bundle reuse exactly these vectors.
"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ml.common.stage import StageContext, stage
from ml.nlp import embed
from ml.nlp.aspect_seeds import ASPECTS, DESCRIPTOR_POLES, PROTOTYPES, compile_lexicon

N_PROTOTYPES = 8
DESCRIPTORS: tuple[str, ...] = tuple(DESCRIPTOR_POLES)
PARQUET_COMPRESSION = "zstd"
ASPECT_SCHEMA = pa.schema(
    [
        ("sent_id", pa.int64()),
        ("aspect_mask", pa.uint16()),
        ("lexicon_mask", pa.uint16()),
        ("top_sim", pa.float32()),
        *((name, pa.float32()) for name in DESCRIPTORS),
    ]
)


def check_taxonomy(aspects_cfg: Mapping[str, Any]) -> None:
    """``aspects.names`` and ``aspects.descriptors`` must be the seed module's order (the index)."""
    if tuple(aspects_cfg["names"]) != ASPECTS:
        raise ValueError("aspects.names must equal aspect_seeds.ASPECTS, in order")
    if tuple(aspects_cfg["descriptors"]) != DESCRIPTORS:
        raise ValueError("aspects.descriptors must equal aspect_seeds.DESCRIPTOR_POLES, in order")


def tau_vector(aspects_cfg: Mapping[str, Any]) -> np.ndarray:
    """(12,) float32 threshold per aspect: ``aspects.tau[name]`` else ``aspects.tau_default``."""
    default = float(aspects_cfg["tau_default"])
    overrides = aspects_cfg.get("tau") or {}
    return np.array([float(overrides.get(a, default)) for a in ASPECTS], dtype=np.float32)


def pack_mask(hits: np.ndarray) -> np.ndarray:
    """(n, 12) bool -> (n,) uint16 with bit ``a`` set for aspect ``a``."""
    weights = (1 << np.arange(hits.shape[1])).astype(np.uint16)
    return (hits.astype(np.uint16) * weights).sum(axis=1).astype(np.uint16)


def unpack_mask(mask: np.ndarray, n_aspects: int = len(ASPECTS)) -> np.ndarray:
    """(n,) uint16 -> (n, n_aspects) bool."""
    bits = np.asarray(mask, dtype=np.uint16)[:, None] >> np.arange(n_aspects, dtype=np.uint16)
    return (bits & 1).astype(bool)


def lexicon_hits(texts: Sequence[str], patterns: Mapping[str, re.Pattern[str]]) -> np.ndarray:
    """(n, 12) bool: word-bounded, case-insensitive lexicon matches per aspect."""
    out = np.zeros((len(texts), len(ASPECTS)), dtype=bool)
    for a, name in enumerate(ASPECTS):
        pattern = patterns[name]
        out[:, a] = [pattern.search(t) is not None for t in texts]
    return out


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return (matrix / np.clip(norms, 1e-12, None)).astype(np.float32)


def embed_prototypes(encoder: embed.TextEncoder) -> np.ndarray:
    """(12, 8, d) float32, L2-normalized, in ``ASPECTS`` order."""
    for name in ASPECTS:
        if len(PROTOTYPES[name]) != N_PROTOTYPES:
            raise ValueError(f"aspect {name!r} needs {N_PROTOTYPES} prototypes")
    flat = encoder.encode([text for name in ASPECTS for text in PROTOTYPES[name]])
    return _normalize(flat.reshape(len(ASPECTS), N_PROTOTYPES, -1))


def embed_poles(encoder: embed.TextEncoder) -> np.ndarray:
    """(3, 2, d) float32: the normalized mean of each descriptor's plus (0) and minus (1) poles."""
    rows = []
    for name in DESCRIPTORS:
        spec = DESCRIPTOR_POLES[name]
        plus = encoder.encode(list(spec["plus"])).mean(axis=0)  # type: ignore[call-overload]
        minus = encoder.encode(list(spec["minus"])).mean(axis=0)  # type: ignore[call-overload]
        rows.append(np.stack([plus, minus]))
    return _normalize(np.stack(rows))


@dataclass(frozen=True)
class Detection:
    """Per-sentence detection results: bool (n, 12) hits and (n, 12) prototype similarities."""

    hits: np.ndarray
    sims: np.ndarray
    lexicon: np.ndarray


class AspectDetector:
    """Prototype-or-lexicon detection over normalized sentence embeddings (s11.3)."""

    def __init__(
        self, prototypes: np.ndarray, poles: np.ndarray, tau: np.ndarray, dim: int
    ) -> None:
        if prototypes.shape != (len(ASPECTS), N_PROTOTYPES, dim):
            raise ValueError(f"aspect_prototypes shape {prototypes.shape} != (12, 8, {dim})")
        if poles.shape != (len(DESCRIPTORS), 2, dim):
            raise ValueError(f"descriptor_poles shape {poles.shape} != (3, 2, {dim})")
        if tau.shape != (len(ASPECTS),):
            raise ValueError("tau must have one value per aspect")
        self.prototypes = prototypes.astype(np.float32)
        self.flat_prototypes = self.prototypes.reshape(-1, dim)
        self.poles = poles.astype(np.float32)
        self.tau = tau.astype(np.float32)
        self.dim = dim
        self.patterns = compile_lexicon()
        self.pole_aspect = np.array(
            [ASPECTS.index(DESCRIPTOR_POLES[d]["aspect"]) for d in DESCRIPTORS]  # type: ignore[arg-type]
        )

    def similarities(self, emb: np.ndarray) -> np.ndarray:
        """(n, 12): best cosine over each aspect's prototypes (embeddings cast to float32)."""
        emb = np.asarray(emb, dtype=np.float32)
        sims = emb @ self.flat_prototypes.T
        return sims.reshape(len(emb), len(ASPECTS), N_PROTOTYPES).max(axis=2)

    def detect(self, emb: np.ndarray, texts: Sequence[str]) -> Detection:
        """Hits = prototype similarity >= tau or a lexicon match."""
        if len(texts) != len(emb):
            raise ValueError("texts and embeddings differ in length")
        sims = self.similarities(emb)
        lexicon = lexicon_hits(texts, self.patterns)
        return Detection(hits=(sims >= self.tau) | lexicon, sims=sims, lexicon=lexicon)

    def descriptor_scores(self, emb: np.ndarray, hits: np.ndarray) -> np.ndarray:
        """(n, 3) float32: cos(plus pole) - cos(minus pole), NaN where the linked aspect missed."""
        emb = np.asarray(emb, dtype=np.float32)
        plus = emb @ self.poles[:, 0, :].T
        minus = emb @ self.poles[:, 1, :].T
        scores = (plus - minus).astype(np.float32)
        scores[~hits[:, self.pole_aspect]] = np.nan
        return scores


def iter_chunks(n: int, chunk_rows: int) -> list[tuple[int, int]]:
    """``[start, stop)`` row ranges of size ``chunk_rows`` covering ``n`` rows."""
    return [(s, min(s + chunk_rows, n)) for s in range(0, n, chunk_rows)]


def detection_frame(
    detector: AspectDetector, sent_ids: np.ndarray, emb: np.ndarray, texts: Sequence[str]
) -> pd.DataFrame:
    """One ``ASPECT_SCHEMA`` row per sentence of a chunk."""
    found = detector.detect(emb, texts)
    descriptors = detector.descriptor_scores(emb, found.hits)
    frame = pd.DataFrame(
        {
            "sent_id": np.asarray(sent_ids, dtype=np.int64),
            "aspect_mask": pack_mask(found.hits),
            "lexicon_mask": pack_mask(found.lexicon),
            "top_sim": found.sims.max(axis=1).astype(np.float32) if len(texts) else [],
        }
    )
    for d, name in enumerate(DESCRIPTORS):
        frame[name] = descriptors[:, d]
    return frame


@stage(
    "nlp.aspects",
    inputs=["data:sentences.parquet", "data:sent_emb.f16.npy"],
    outputs=[
        "artifacts:nlp/aspect_prototypes.npy",
        "artifacts:nlp/descriptor_poles.npy",
        "data:sentence_aspects.parquet",
        "results:nlp/aspect_checks.json",
    ],
    config_keys=[
        "aspects.names",
        "aspects.tau_default",
        "aspects.tau",
        "aspects.descriptors",
        "content",
        "nlp.embedding_dim",
        "nlp.chunk_rows",
        "nlp.sentence_embed_batch",
    ],
)
def build_aspects(ctx: StageContext) -> None:
    """Embed the prototypes, then detect aspects for every sentence, chunk by chunk."""
    aspects_cfg = {
        k: ctx.param(f"aspects.{k}") for k in ("names", "tau_default", "tau", "descriptors")
    }
    check_taxonomy(aspects_cfg)
    content = ctx.param("content")
    dim, chunk_rows = int(ctx.param("nlp.embedding_dim")), int(ctx.param("nlp.chunk_rows"))
    encoder = embed.load_encoder(
        content["model_name"],
        int(ctx.param("nlp.sentence_embed_batch")),
        int(content["max_seq_length"]),
    )
    prototypes, poles = embed_prototypes(encoder), embed_poles(encoder)
    detector = AspectDetector(prototypes, poles, tau_vector(aspects_cfg), dim)
    np.save(ctx.out("artifacts:nlp/aspect_prototypes.npy"), prototypes)
    np.save(ctx.out("artifacts:nlp/descriptor_poles.npy"), poles)

    sentences = pd.read_parquet(ctx.path("data:sentences.parquet"), columns=["sent_id", "text"])
    emb = np.load(ctx.path("data:sent_emb.f16.npy"), mmap_mode="r")
    if emb.shape != (len(sentences), dim):
        raise ValueError(f"sent_emb shape {emb.shape} != ({len(sentences)}, {dim})")
    hits_per_aspect = np.zeros(len(ASPECTS), dtype=np.int64)
    lexicon_per_aspect = np.zeros(len(ASPECTS), dtype=np.int64)
    prototype_per_aspect = np.zeros(len(ASPECTS), dtype=np.int64)
    with_hit = 0
    texts = sentences["text"].tolist()
    ids = sentences["sent_id"].to_numpy()
    with pq.ParquetWriter(
        ctx.out("data:sentence_aspects.parquet"), ASPECT_SCHEMA, compression=PARQUET_COMPRESSION
    ) as out:
        for start, stop in iter_chunks(len(sentences), chunk_rows):
            chunk_emb = np.asarray(emb[start:stop], dtype=np.float32)
            frame = detection_frame(detector, ids[start:stop], chunk_emb, texts[start:stop])
            hits = unpack_mask(frame["aspect_mask"].to_numpy())
            lexicon = unpack_mask(frame["lexicon_mask"].to_numpy())
            hits_per_aspect += hits.sum(axis=0)
            lexicon_per_aspect += lexicon.sum(axis=0)
            prototype_per_aspect += (hits & ~lexicon).sum(axis=0)
            with_hit += int(hits.any(axis=1).sum())
            out.write_table(pa.Table.from_pandas(frame, schema=ASPECT_SCHEMA, preserve_index=False))
            ctx.log.info("detected aspects for %d / %d sentences", stop, len(sentences))

    n = len(sentences)
    checks = {
        "sentences": n,
        "sentences_with_hit": with_hit,
        "share_with_hit": round(with_hit / n, 4) if n else None,
        "tau": {a: float(t) for a, t in zip(ASPECTS, detector.tau, strict=True)},
        "per_aspect": {
            a: {
                "hits": int(hits_per_aspect[k]),
                "lexicon_hits": int(lexicon_per_aspect[k]),
                "prototype_only_hits": int(prototype_per_aspect[k]),
                "share": round(int(hits_per_aspect[k]) / n, 4) if n else None,
            }
            for k, a in enumerate(ASPECTS)
        },
    }
    ctx.out("results:nlp/aspect_checks.json").write_text(
        json.dumps(checks, indent=2) + "\n", encoding="utf-8"
    )
    ctx.count("sentences", n)
    ctx.count("sentences_with_hit", with_hit)
    ctx.notes["hits_per_aspect"] = dict(zip(ASPECTS, hits_per_aspect.tolist(), strict=True))
