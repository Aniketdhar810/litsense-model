"""Review centroids (CLAUDE.md s11.6), stages nlp.centroids_val and nlp.centroids_serve.

``review_centroid_item[i]`` is the L2-normalized mean of the sentence embeddings from reviews of
item ``i`` with rating >= ``data.relevance_min_rating`` and ``ts < cutoff`` (a zero row when none);
``review_centroid_user[u]`` is the same over the user's own review sentences. One artifact set per
cutoff (``ml/nlp/cutoffs.py``): ``eval_val`` at t1 now, ``serve`` over every review; ``eval_test``
(t2) is built in Week 13 with the other test-mode artifacts through the same ``build_centroids``.
The builder raises ``LeakageError`` when handed a sentence at or after its cutoff.
"""

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ml.common.io import load_id_map
from ml.common.stage import StageContext, stage
from ml.nlp.aspects import iter_chunks
from ml.nlp.cutoffs import ArtifactSet, check_cutoff, cutoff_for, rows_before

CENTROID_FILES = ("review_centroid_item.npy", "review_centroid_user.npy", "meta.json")


def _normalize_rows(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(sums, axis=1, keepdims=True)
    out = np.where(norms > 0, sums / np.clip(norms, 1e-12, None), 0.0).astype(np.float32)
    out[counts == 0] = 0.0
    return out


def build_centroids(
    sentences: pd.DataFrame,
    emb: Any,
    n_items: int,
    n_users: int,
    cutoff: int,
    min_rating: float,
    chunk_rows: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Item and user review centroids from ``sentences`` (``sent_id``, ``u``, ``i``, ``ts``,
    ``rating``) and the embedding matrix ``emb`` indexed by ``sent_id``. Raises ``LeakageError``
    if any sentence is at or after ``cutoff``."""
    check_cutoff(sentences["ts"].to_numpy(), cutoff, "sentences")
    liked = sentences[sentences["rating"].to_numpy() >= min_rating]
    dim = emb.shape[1]
    item_sum = np.zeros((n_items, dim), dtype=np.float64)
    user_sum = np.zeros((n_users, dim), dtype=np.float64)
    sent_ids = liked["sent_id"].to_numpy(np.int64)
    items = liked["i"].to_numpy(np.int64)
    users = liked["u"].to_numpy(np.int64)
    if len(sent_ids) and (items.max() >= n_items or users.max() >= n_users):
        raise ValueError("sentence item or user ids exceed the id map")
    order = np.argsort(sent_ids, kind="stable")  # read the memmap in row order
    sent_ids, items, users = sent_ids[order], items[order], users[order]
    for start, stop in iter_chunks(len(sent_ids), chunk_rows):
        rows = sent_ids[start:stop]
        vectors = np.asarray(emb[rows], dtype=np.float32)  # one float32 copy of the chunk
        ones = np.ones(len(rows), dtype=np.float32)
        cols = np.arange(len(rows))
        by_item = sp.csr_matrix((ones, (items[start:stop], cols)), shape=(n_items, len(rows)))
        by_user = sp.csr_matrix((ones, (users[start:stop], cols)), shape=(n_users, len(rows)))
        item_sum += by_item @ vectors
        user_sum += by_user @ vectors
    item_counts = np.bincount(items, minlength=n_items)
    user_counts = np.bincount(users, minlength=n_users)
    counts = {
        "sentences_before_cutoff": int(len(sentences)),
        "sentences_used": int(len(sent_ids)),
        "items_with_centroid": int((item_counts > 0).sum()),
        "users_with_centroid": int((user_counts > 0).sum()),
    }
    return _normalize_rows(item_sum, item_counts), _normalize_rows(user_sum, user_counts), counts


def _register(artifact_set: ArtifactSet, stage_name: str) -> None:
    folder = f"{artifact_set}/review_centroids"

    @stage(
        stage_name,
        inputs=[
            "data:sentences.parquet",
            "data:sent_emb.f16.npy",
            "data:splits.json",
            "data:id_map.json",
        ],
        outputs=[f"artifacts:{folder}/{name}" for name in CENTROID_FILES],
        config_keys=["data.relevance_min_rating", "nlp.embedding_dim", "nlp.chunk_rows"],
    )
    def build(ctx: StageContext) -> None:
        """Item and user review centroids for one artifact set."""
        ids = load_id_map(ctx.path("data:id_map.json"))
        splits: Mapping[str, Any] = json.loads(
            ctx.path("data:splits.json").read_text(encoding="utf-8")
        )
        dim, chunk_rows = int(ctx.param("nlp.embedding_dim")), int(ctx.param("nlp.chunk_rows"))
        min_rating = float(ctx.param("data.relevance_min_rating"))
        sentences = pd.read_parquet(
            ctx.path("data:sentences.parquet"), columns=["sent_id", "u", "i", "ts", "rating"]
        )
        emb = np.load(ctx.path("data:sent_emb.f16.npy"), mmap_mode="r")
        if emb.shape != (len(sentences), dim):
            raise ValueError(f"sent_emb shape {emb.shape} != ({len(sentences)}, {dim})")
        max_ts = int(sentences["ts"].max()) if len(sentences) else 0
        cutoff = cutoff_for(artifact_set, splits, max_ts)
        ctx.cutoff = cutoff
        item, user, counts = build_centroids(
            rows_before(sentences, cutoff),
            emb,
            ids.n_items,
            ids.n_users,
            cutoff,
            min_rating,
            chunk_rows,
        )
        np.save(ctx.out(f"artifacts:{folder}/review_centroid_item.npy"), item)
        np.save(ctx.out(f"artifacts:{folder}/review_centroid_user.npy"), user)
        meta = {
            "artifact_set": artifact_set,
            "cutoff": cutoff,
            "min_rating": min_rating,
            "dim": dim,
            "n_items": ids.n_items,
            "n_users": ids.n_users,
            **counts,
        }
        ctx.out(f"artifacts:{folder}/meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        for key, value in counts.items():
            ctx.count(key, value)

    build.__name__ = f"build_centroids_{artifact_set}"


_register("eval_val", "nlp.centroids_val")
_register("serve", "nlp.centroids_serve")
