"""Serve artifact set and bundle export (CLAUDE.md s3, s4.6, s12.7; D56, D63, D66).

Three stages:

* ``models.lightgcn_serve``: LightGCN fit on every interaction (train, val and test) for exactly the
  epoch count that early stopping chose on the val artifact set (D43 pattern: no search on the full
  data, nothing to validate against). Cutoff = newest timestamp + 1 ms. Writes
  ``artifacts/<v>/serve/lightgcn/``.
* ``serve.export_val`` (D66): the same bundle layout built from the ``eval_val`` artifact set
  (LightGCN on train, aspect profiles, centroids and catalogue statistics at t1). It exists so
  ranker training runs the real ``litsense_core`` profile -> candidates -> features path on
  leakage-safe artifacts; it ships no snippets, spoiler model, encoder or ranker. Every user with a
  train interaction is a corpus user there, so training can use exact LightGCN vectors.
* ``serve.export``: collects the serve artifacts and the time-independent ones (content embeddings,
  neighbours, catalogue), casts embeddings to float16, builds the indices of s4.6, copies the
  Phase 3 serve artifacts (D63) and the ranker trained on the val bundle (D66), writes
  ``manifest.json`` (shapes, sha256, cutoff, config slice) and the metrics report, and tars the
  folder. Files that need later phases (projection, faiss index) are listed in the manifest as
  absent, so ``litsense_core`` degrades instead of guessing.
"""

import gzip
import hashlib
import json
import shutil
import tarfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ml.common.io import IdMap, load_id_map
from ml.common.stage import StageContext, stage
from ml.eval.data import longtail_items
from ml.models.embedding_scorers import l2_normalize_rows
from ml.models.lightgcn import LIGHTGCN_FILES, embedding_health, train_lightgcn, user_norm_summary

DAY_MS = 86_400_000
SERVE_LGCN = "serve/lightgcn"
VAL_LGCN = "eval_val/lightgcn"
BUNDLE_DIR = "artifacts:serve_bundle/"
BUNDLE_TAR = "artifacts:serve_bundle.tar.gz"
VAL_BUNDLE_DIR = "artifacts:eval_val/bundle/"
RANKER_DIR = "eval_val/ranker"
RANKER_FILES = ("ranker.txt", "feature_list.json")
TRENDING_TOP = 200
CORPUS_USERS = 300
# Phase 3 artifacts copied into a bundle as they are (D63); both artifact sets write them.
ASPECT_FILES = (
    "item_aspects.npy",
    "item_mention_rate.npy",
    "item_pos_rate.npy",
    "item_neg_rate.npy",
    "item_descriptors.npy",
    "aspect_means.npy",
    "aspect_topitems.json",
)
USER_ASPECT_FILES = ("user_aspects.npy", "user_aspect_conf.npy")
CENTROID_FILES = ("review_centroid_item.npy", "review_centroid_user.npy")
SNIPPETS = "artifacts:serve/snippets/evidence_snippets.json.gz"
SPOILER = "artifacts:serve/spoiler/spoiler_lr.npz"
ONNX_FP32, ONNX_INT8, TOKENIZER = (
    "artifacts:serve/onnx/minilm.onnx",
    "artifacts:serve/onnx/minilm.int8.onnx",
    "artifacts:serve/onnx/tokenizer.json",
)
PROTOTYPES, POLES = "artifacts:nlp/aspect_prototypes.npy", "artifacts:nlp/descriptor_poles.npy"
# Bundle files no export can fill on v0 yet; litsense_core treats each as optional.
ABSENT_FILES = ("proj_item.f16.npy", "content.faiss")
# Serve-only files, listed as absent in the val bundle.
SERVE_ONLY_FILES = (
    "evidence_snippets.jsonl",
    "snippets_offsets.npy",
    "spoiler_lr.npz",
    "minilm.onnx",
    "tokenizer.json",
    *RANKER_FILES,
)
CONFIG_KEYS = (
    "project.data_version",
    "models.popularity",
    "data.longtail_fraction",
    "ranking.buckets",
    "ranking.quotas",
    "ranking.candidates_total",
    "ranking.blend",
    "ranking.postprocess",
    "ranking.min_reviews_main_feed",
    "ranking.overrides",
    "aspects",
    "content.max_seq_length",
    "serving",
    "llm",
)
SUMMARIES = "artifacts:serve/genai/review_summaries.jsonl"


# ------------------------------------------------------------------ models.lightgcn_serve


@stage(
    "models.lightgcn_serve",
    inputs=[
        "data:interactions.parquet",
        "data:id_map.json",
        f"artifacts:{VAL_LGCN}/meta.json",
    ],
    outputs=[f"artifacts:{SERVE_LGCN}/{name}" for name in LIGHTGCN_FILES],
    config_keys=["models.lightgcn"],
)
def build_lightgcn_serve(ctx: StageContext) -> None:
    """LightGCN on all interactions for the val-chosen epoch count; the serve artifact set."""
    ids = load_id_map(ctx.path("data:id_map.json"))
    val_meta = json.loads(ctx.path(f"artifacts:{VAL_LGCN}/meta.json").read_text("utf-8"))
    params = ctx.param("models.lightgcn")
    rows = pd.read_parquet(ctx.path("data:interactions.parquet"), columns=["u", "i", "ts"])
    cutoff = int(rows["ts"].max()) + 1
    ctx.cutoff = cutoff
    epochs = int(val_meta["best_epoch"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx.log.info("training on %s for %d epochs (val best_epoch), %d interactions", device, epochs,
                 len(rows))  # fmt: skip
    started = time.monotonic()
    result = train_lightgcn(rows, ids.n_users, ids.n_items, params, ctx.seed, epochs, device)
    dim = int(params["dim"])
    assert result.user.shape == (ids.n_users, dim), result.user.shape  # s2.2 #8
    assert result.item.shape == (ids.n_items, dim), result.item.shape
    counts = np.bincount(rows["i"], minlength=ids.n_items).astype(np.int64)

    out = {name: ctx.out(f"artifacts:{SERVE_LGCN}/{name}") for name in LIGHTGCN_FILES}
    np.save(out["lgcn_user.npy"], result.user)
    np.save(out["lgcn_item.npy"], result.item)
    np.save(out["lgcn_item_norm.npy"], l2_normalize_rows(result.item))
    np.save(out["item_degree.npy"], counts)
    meta = {
        "model": "lightgcn",
        "mode": "serve",
        "cutoff": cutoff,
        "params": dict(params),
        "epochs": epochs,
        "epochs_from": "eval_val best_epoch",
        "val_best_inner_metric": val_meta.get("best_inner_metric"),
        "zero_degree_items": int((counts == 0).sum()),
        "embedding_health": embedding_health(result.item, counts > 0),
        "user_norm": user_norm_summary(result.user),
        "device": device,
        "torch": torch.__version__,
        "seconds": round(time.monotonic() - started, 1),
    }
    out["meta.json"].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    ctx.count("fit_interactions", len(rows))
    ctx.count("epochs", epochs)
    ctx.count("zero_degree_items", meta["zero_degree_items"])


# ------------------------------------------------------------------ helpers


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _str_or_none(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def _genre_lists(books: pd.DataFrame) -> list[list[str]]:
    out = []
    for raw in books["genres"].tolist():
        labels = [] if raw is None else [str(g) for g in list(raw) if str(g).strip()]
        out.append(labels)
    return out


def catalog_arrays(
    books: pd.DataFrame,
    inter: pd.DataFrame,
    n_items: int,
    cutoff: int,
    bayes_m: float,
    half_life_days: float,
    longtail_fraction: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """``catalog.npz`` arrays (s4.6) and the index maps (``author``, ``series``, ``genre``)."""
    genres = _genre_lists(books)
    genre_labels = sorted({g for labels in genres for g in labels})
    genre_id = {g: k for k, g in enumerate(genre_labels)}
    indptr = np.zeros(n_items + 1, dtype=np.int32)
    indices = []
    for i, labels in enumerate(genres):
        indices.extend(genre_id[g] for g in labels)
        indptr[i + 1] = len(indices)
    primary = [_str_or_none(p) or (labels[0] if labels else "") for p, labels in
               zip(books["primary_genre"].tolist(), genres, strict=True)]  # fmt: skip
    for p in primary:
        genre_id.setdefault(p, len(genre_id))
    genre_labels = [g for g, _ in sorted(genre_id.items(), key=lambda kv: kv[1])]
    primary_genre = np.array([genre_id[p] for p in primary], dtype=np.int32)

    authors = [_str_or_none(a) for a in books["author"].tolist()]
    author_labels = sorted({a for a in authors if a})
    author_id_of = {a: k for k, a in enumerate(author_labels)}
    author_id = np.array([author_id_of[a] if a else -1 for a in authors], dtype=np.int32)

    series = [_str_or_none(s) for s in books["series_id"].tolist()]
    series_labels = sorted({s for s in series if s})
    series_id_of = {s: k for k, s in enumerate(series_labels)}
    series_id = np.array([series_id_of[s] if s else -1 for s in series], dtype=np.int32)
    series_pos = pd.to_numeric(books["series_pos"], errors="coerce").to_numpy(np.float32)

    counts = np.bincount(inter["i"], minlength=n_items).astype(np.float32)
    sums = np.bincount(inter["i"], weights=inter["rating"], minlength=n_items)
    sq = np.bincount(inter["i"], weights=inter["rating"] ** 2, minlength=n_items)
    global_mean = float(inter["rating"].mean()) if len(inter) else 0.0
    bayes = (sums + bayes_m * global_mean) / (counts + bayes_m)
    mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    var = np.divide(sq, counts, out=np.zeros_like(sq), where=counts > 0) - mean**2
    rating_std = np.sqrt(np.clip(var, 0, None))
    age_days = (cutoff - inter["ts"].to_numpy()) / DAY_MS
    trending = np.bincount(
        inter["i"], weights=np.exp(-age_days / half_life_days), minlength=n_items
    )
    popularity = np.log1p(counts) / max(float(np.log1p(counts.max())), 1e-9)
    tail = np.zeros(n_items, dtype=bool)
    tail[longtail_items(counts.astype(np.int64), longtail_fraction)] = True
    n_reviews = np.bincount(inter.loc[inter["has_text"], "i"], minlength=n_items)

    arrays = {
        "year": pd.to_numeric(books["year"], errors="coerce").fillna(-1).to_numpy(np.int32),
        "pages": pd.to_numeric(books["pages"], errors="coerce").fillna(-1).to_numpy(np.int32),
        "series_id": series_id,
        "series_pos": series_pos,
        "genre_indptr": indptr,
        "genre_indices": np.asarray(indices, dtype=np.int32),
        "primary_genre": primary_genre,
        "bayes_rating": bayes.astype(np.float32),
        "rating_count": counts.astype(np.int32),
        "rating_std": rating_std.astype(np.float32),
        "popularity": popularity.astype(np.float32),
        "trending": trending.astype(np.float32),
        "is_longtail": tail,
        "n_reviews": n_reviews.astype(np.int32),
        "author_id": author_id,
    }
    indices_json = {
        "author_index": {
            "labels": author_labels,
            "items": [np.flatnonzero(author_id == k).tolist() for k in range(len(author_labels))],
        },  # fmt: skip
        "series_index": {
            "labels": series_labels,
            "names": [
                _str_or_none(books.loc[books["series_id"].astype(str) == s, "series_name"].iloc[0])
                if (books["series_id"].astype(str) == s).any()
                else None
                for s in series_labels
            ],
            "items": [np.flatnonzero(series_id == k).tolist() for k in range(len(series_labels))],
        },  # fmt: skip
        "genre_index": {
            "labels": genre_labels,
            "items": [
                np.flatnonzero(primary_genre == k).tolist() for k in range(len(genre_labels))
            ],
        },  # fmt: skip
    }
    return arrays, indices_json


def catalog_text(
    books: pd.DataFrame, summaries: Mapping[int, str] | None = None
) -> dict[str, list[Any]]:
    """``catalog_text.json.gz``: the strings the product shows (s4.6). ``summaries`` are the
    ``genai.synthesis`` review summaries by item (D70); items without one keep ``None``."""
    genres = _genre_lists(books)
    summaries = summaries or {}
    return {
        "canonical_id": [str(c) for c in books["canonical_id"].tolist()],
        "title": [str(t) for t in books["title"].tolist()],
        "author": [_str_or_none(a) for a in books["author"].tolist()],
        "genres": genres,
        "image": [_str_or_none(u) for u in books["image"].tolist()],
        "description": [
            (_str_or_none(d) or _str_or_none(f) or "")[:600]
            for d, f in zip(books["description"].tolist(), books["features"].tolist(), strict=True)
        ],  # fmt: skip
        "isbn13": [_str_or_none(x) for x in books["isbn13"].tolist()],
        "series_name": [_str_or_none(x) for x in books["series_name"].tolist()],
        "review_summary": [summaries.get(int(i)) for i in books["i"].tolist()],
    }


def read_summaries(path: Path | None) -> dict[int, str]:
    """``{item: summary}`` from ``genai.synthesis``'s JSONL (absent or empty -> no summaries)."""
    out: dict[int, str] = {}
    if path is None or not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row.get("summary"):
                    out[int(row["i"])] = str(row["summary"])
    return out


def corpus_users(
    inter: pd.DataFrame,
    user_ids: list[str],
    lgcn_user: np.ndarray,
    limit: int,
    user_aspects: np.ndarray | None = None,
    user_aspect_conf: np.ndarray | None = None,
    user_centroid: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """The ``limit`` most active readers with their exact LightGCN vectors (demo accounts) and,
    when given, their offline explicit aspect vectors and confidences (D64) and review centroids
    (D66). ``limit >= n_users`` keeps every reader with an interaction (the val bundle)."""
    counts = np.bincount(inter["u"], minlength=len(user_ids))
    top = np.argsort(-counts, kind="stable")[: min(limit, int((counts > 0).sum()))]
    out = {
        "user_index": top.astype(np.int32),
        "user_id": np.array([user_ids[u] for u in top], dtype=object).astype(str),
        "n_interactions": counts[top].astype(np.int32),
        "lgcn_user": lgcn_user[top].astype(np.float16),
    }
    if user_aspects is not None and user_aspect_conf is not None:
        out["aspects"] = user_aspects[top].astype(np.float32)
        out["aspect_conf"] = user_aspect_conf[top].astype(np.float32)
    if user_centroid is not None:
        out["centroid"] = user_centroid[top].astype(np.float16)
    return out


def write_snippets_jsonl(source: Path, folder: Path, n_items: int) -> dict[str, int]:
    """``evidence_snippets.jsonl`` (line k = item k's ``{aspect: {"pos": [...], "neg": [...]}}``,
    ``{}`` when none) plus ``snippets_offsets.npy`` (int64, n_items + 1 byte offsets), so the
    serving package reads one item's snippets with a seek instead of holding 200K dicts in
    memory (D64)."""
    with gzip.open(source, "rt", encoding="utf-8") as f:
        snips = json.load(f)
    offsets = np.zeros(n_items + 1, dtype=np.int64)
    items_with = 0
    with (folder / "evidence_snippets.jsonl").open("wb") as out:
        for i in range(n_items):
            entry = snips.get(str(i), {})
            items_with += bool(entry)
            line = json.dumps(entry, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            out.write(line + b"\n")
            offsets[i + 1] = offsets[i] + len(line) + 1
    np.save(folder / "snippets_offsets.npy", offsets)
    return {"items_with_snippets": items_with, "bytes": int(offsets[-1])}


def metrics_report(val_dir: Path, data_version: str) -> dict[str, Any]:
    """s15.6 admin JSON from ``results/<v>/val`` (every number comes from those files)."""
    main = val_dir / "table_main.csv"
    if not main.exists():
        return {"data_version": data_version, "models": [], "note": "no val results in this run"}
    table = pd.read_csv(main)
    metric_names = [c for c in table.columns if "@" in c and not c.endswith(("_low", "_high"))]
    models = []
    for _, row in table.iterrows():
        metrics = {m: float(row[m]) for m in metric_names}
        ci = {m: [float(row[f"{m}_low"]), float(row[f"{m}_high"])]
              for m in metric_names if f"{m}_low" in table.columns}  # fmt: skip
        models.append({"id": row["model"], "name": row["model"], "mode": "val",
                       "metrics": metrics, "ci95": ci, "users": int(row["users"])})  # fmt: skip
    out: dict[str, Any] = {"data_version": data_version, "models": models}
    for name in ("table_longtail.csv", "table_strata.csv", "bootstrap.csv"):
        path = val_dir / name
        if path.exists():
            out[name.removesuffix(".csv")] = pd.read_csv(path).to_dict(orient="records")
    return out


# ------------------------------------------------------------------ bundle writer


@dataclass
class BundleSpec:
    """Everything ``write_bundle`` needs, resolved by the calling stage."""

    label: str  # bundle_version prefix ("v0" serve, "v0-val" val)
    ids: IdMap
    books: pd.DataFrame
    inter: pd.DataFrame  # rows before the cutoff: u, i, rating, ts, has_text
    cutoff: int
    lgcn_dir: Path  # lgcn_user / lgcn_item / lgcn_item_norm .npy + meta.json
    content: np.ndarray
    item_nn: np.ndarray
    aspects_dir: Path  # ASPECT_FILES + USER_ASPECT_FILES
    centroids_dir: Path  # CENTROID_FILES
    prototypes: Path
    poles: Path
    corpus_limit: int
    params: dict[str, Any]  # the CONFIG_KEYS slice
    metrics: dict[str, Any] = field(default_factory=dict)
    snippets: Path | None = None
    spoiler: Path | None = None
    onnx: Path | None = None
    tokenizer: Path | None = None
    ranker_dir: Path | None = None
    summaries: Path | None = None  # genai.synthesis JSONL (D70)
    tar: Path | None = None


def write_bundle(folder: Path, spec: BundleSpec, log: Any) -> dict[str, Any]:
    """Write one bundle folder (and optionally its tar) and return its manifest."""
    started = time.monotonic()
    ids, books, inter, cutoff = spec.ids, spec.books, spec.inter, spec.cutoff
    if list(books["i"]) != list(range(ids.n_items)):
        raise ValueError("books.parquet rows must be items 0..n-1 in id_map order")
    if len(inter) and int(inter["ts"].max()) >= cutoff:
        raise ValueError("bundle interactions reach the cutoff (leakage)")
    lgcn_meta = json.loads((spec.lgcn_dir / "meta.json").read_text("utf-8"))
    if int(lgcn_meta["cutoff"]) != cutoff:
        raise ValueError(f"LightGCN cutoff {lgcn_meta['cutoff']} != bundle cutoff {cutoff}")
    lgcn_item = np.load(spec.lgcn_dir / "lgcn_item.npy").astype(np.float32)
    lgcn_item_norm = np.load(spec.lgcn_dir / "lgcn_item_norm.npy")
    lgcn_user = np.load(spec.lgcn_dir / "lgcn_user.npy").astype(np.float32)
    content, item_nn = spec.content.astype(np.float32), spec.item_nn.astype(np.int32)
    for name, arr in (("lgcn_item", lgcn_item), ("content", content), ("item_nn", item_nn)):
        if arr.shape[0] != ids.n_items:
            raise ValueError(f"{name} has {arr.shape[0]} rows for {ids.n_items} items")

    pop = spec.params["models.popularity"]
    arrays, indices = catalog_arrays(
        books, inter, ids.n_items, cutoff, float(pop["bayes_m"]),
        float(pop["trend_half_life_days"]), float(spec.params["data.longtail_fraction"]),
    )  # fmt: skip
    trending_top = np.argsort(-arrays["trending"], kind="stable")[:TRENDING_TOP].astype(np.int32)

    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / "catalog.npz", **arrays)
    summaries = read_summaries(spec.summaries)
    with gzip.open(folder / "catalog_text.json.gz", "wt", encoding="utf-8") as f:
        json.dump(catalog_text(books, summaries), f, ensure_ascii=False)
    np.save(folder / "lgcn_item.f16.npy", lgcn_item.astype(np.float16))
    np.save(folder / "lgcn_item_norm.f16.npy", lgcn_item_norm.astype(np.float16))
    np.save(folder / "content_item.f16.npy", content.astype(np.float16))
    np.save(folder / "item_nn.npy", item_nn)
    np.save(folder / "trending_top.npy", trending_top)
    for name, payload in indices.items():
        (folder / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")

    centroid_item = np.load(spec.centroids_dir / "review_centroid_item.npy").astype(np.float32)
    centroid_user = np.load(spec.centroids_dir / "review_centroid_user.npy").astype(np.float32)
    if centroid_item.shape[0] != ids.n_items or centroid_user.shape[0] != ids.n_users:
        raise ValueError("review centroids do not match the id map")
    np.savez(
        folder / "corpus_users.npz",
        **corpus_users(
            inter,
            ids.users,
            lgcn_user,
            spec.corpus_limit,
            np.load(spec.aspects_dir / "user_aspects.npy"),
            np.load(spec.aspects_dir / "user_aspect_conf.npy"),
            centroid_user,
        ),  # fmt: skip
    )
    (folder / "metrics.json").write_text(json.dumps(spec.metrics, indent=1), encoding="utf-8")

    # Phase 3 artifacts (D63): aspect profiles, centroids, prototypes; serve-only extras.
    n_aspects = len(spec.params["aspects"]["names"])
    for name in ASPECT_FILES:
        src = spec.aspects_dir / name
        if name.endswith(".npy"):
            arr = np.load(src)
            expect = n_aspects if name == "aspect_means.npy" else ids.n_items
            if arr.shape[0] != expect:
                raise ValueError(f"{name} has shape {arr.shape}, expected {expect} rows")
            np.save(folder / name, arr.astype(np.float32))
        else:
            shutil.copyfile(src, folder / name)
    np.save(folder / "review_centroid_item.f16.npy", centroid_item.astype(np.float16))
    np.save(folder / "aspect_prototypes.npy", np.load(spec.prototypes).astype(np.float32))
    np.save(folder / "descriptor_poles.npy", np.load(spec.poles).astype(np.float32))
    if spec.snippets is not None:
        write_snippets_jsonl(spec.snippets, folder, ids.n_items)
    if spec.spoiler is not None:
        shutil.copyfile(spec.spoiler, folder / "spoiler_lr.npz")
    onnx_meta = None
    if spec.onnx is not None and spec.tokenizer is not None:
        shutil.copyfile(spec.onnx, folder / "minilm.onnx")
        shutil.copyfile(spec.tokenizer, folder / "tokenizer.json")
        onnx_meta = {
            "quantized": bool(spec.params["serving"]["onnx_quantized"]),
            "source": spec.onnx.name,
            "max_seq_length": int(spec.params["content.max_seq_length"]),
        }
    if spec.ranker_dir is not None:
        for name in RANKER_FILES:
            shutil.copyfile(spec.ranker_dir / name, folder / name)

    files = sorted(p for p in folder.iterdir() if p.is_file() and p.name != "manifest.json")
    present = {p.name for p in files}
    absent = [*ABSENT_FILES, *[n for n in SERVE_ONLY_FILES if n not in present]]
    if onnx_meta is not None:
        absent.append("minilm.int8.onnx" if not onnx_meta["quantized"] else "minilm.fp32.onnx")
    manifest = {
        "bundle_version": f"{spec.label}-{time.strftime('%Y%m%d')}",
        "api_version": "0.1.0",
        "data_version": str(spec.params["project.data_version"]),
        "cutoff": cutoff,
        "n_items": ids.n_items,
        "n_users": ids.n_users,
        "dims": {
            "lgcn": int(lgcn_item.shape[1]),
            "content": int(content.shape[1]),
            "item_nn": int(item_nn.shape[1]),
        },  # fmt: skip
        "lightgcn": {k: lgcn_meta.get(k) for k in ("epochs", "params", "embedding_health")},
        "onnx": onnx_meta,
        "review_summaries": len(summaries),
        "files": {p.name: {"sha256": sha256_of(p), "bytes": p.stat().st_size} for p in files},
        "absent": absent,
        "config": {
            "buckets": spec.params["ranking.buckets"],
            "quotas": spec.params["ranking.quotas"],
            "candidates_total": spec.params["ranking.candidates_total"],
            "blend": spec.params["ranking.blend"],
            "postprocess": spec.params["ranking.postprocess"],
            "overrides": spec.params["ranking.overrides"],
            "min_reviews_main_feed": spec.params["ranking.min_reviews_main_feed"],
            "aspects": spec.params["aspects"],
            "serving": spec.params["serving"],
            "llm": spec.params["llm"],  # the client's config slice (D68); never the key
        },
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    size_mb = None
    if spec.tar is not None:
        with tarfile.open(spec.tar, "w:gz") as tar:
            for p in [*files, folder / "manifest.json"]:
                tar.add(p, arcname=f"serve_bundle/{p.name}")
        size_mb = spec.tar.stat().st_size / 2**20
    log.info("bundle %s: %d files%s, built in %.0f s", manifest["bundle_version"], len(files) + 1,
             f", {size_mb:.1f} MB tar" if size_mb is not None else "",
             time.monotonic() - started)  # fmt: skip
    manifest["_tar_mb"] = None if size_mb is None else round(size_mb, 2)
    return manifest


def _params(ctx: StageContext) -> dict[str, Any]:
    return {key: ctx.param(key) for key in CONFIG_KEYS}


def _books(ctx: StageContext) -> pd.DataFrame:
    return pd.read_parquet(ctx.path("data:books.parquet")).sort_values("i").reset_index(drop=True)


INTER_COLUMNS = ["u", "i", "rating", "ts", "has_text", "split"]


# ------------------------------------------------------------------ serve.export_val


@stage(
    "serve.export_val",
    inputs=[
        "data:books.parquet",
        "data:interactions.parquet",
        "data:id_map.json",
        "data:splits.json",
        "artifacts:content/content_item.npy",
        "artifacts:content/item_nn.npy",
        *[f"artifacts:{VAL_LGCN}/{name}" for name in LIGHTGCN_FILES],
        *[f"artifacts:eval_val/aspects/{name}" for name in (*ASPECT_FILES, *USER_ASPECT_FILES)],
        *[f"artifacts:eval_val/review_centroids/{name}" for name in CENTROID_FILES],
        PROTOTYPES,
        POLES,
    ],
    outputs=[VAL_BUNDLE_DIR],
    config_keys=list(CONFIG_KEYS),
)
def build_val_bundle(ctx: StageContext) -> None:
    """The bundle layout over the eval_val artifact set (cutoff t1), for ranker training."""
    ids = load_id_map(ctx.path("data:id_map.json"))
    splits = json.loads(ctx.path("data:splits.json").read_text(encoding="utf-8"))
    cutoff = int(splits["t1"]["ms"])
    inter = pd.read_parquet(ctx.path("data:interactions.parquet"), columns=INTER_COLUMNS)
    inter = inter[inter["split"] == "train"].drop(columns="split").reset_index(drop=True)
    spec = BundleSpec(
        label=f"{ctx.param('project.data_version')}-val",
        ids=ids,
        books=_books(ctx),
        inter=inter,
        cutoff=cutoff,
        lgcn_dir=ctx.path(f"artifacts:{VAL_LGCN}/meta.json").parent,
        content=np.load(ctx.path("artifacts:content/content_item.npy")),
        item_nn=np.load(ctx.path("artifacts:content/item_nn.npy")),
        aspects_dir=ctx.path("artifacts:eval_val/aspects/item_aspects.npy").parent,
        centroids_dir=ctx.path(
            "artifacts:eval_val/review_centroids/review_centroid_item.npy"
        ).parent,
        prototypes=ctx.path(PROTOTYPES),
        poles=ctx.path(POLES),
        corpus_limit=ids.n_users,  # every reader with a train interaction (D66)
        params=_params(ctx),
    )
    manifest = write_bundle(ctx.out(VAL_BUNDLE_DIR), spec, ctx.log)
    ctx.cutoff = cutoff
    ctx.count("fit_interactions", len(inter))
    ctx.count("bundle_files", len(manifest["files"]) + 1)
    ctx.notes["bundle_version"] = manifest["bundle_version"]


# ------------------------------------------------------------------ serve.export


@stage(
    "serve.export",
    inputs=[
        "data:books.parquet",
        "data:interactions.parquet",
        "data:id_map.json",
        "artifacts:content/content_item.npy",
        "artifacts:content/item_nn.npy",
        *[f"artifacts:{SERVE_LGCN}/{name}" for name in LIGHTGCN_FILES],
        *[f"artifacts:serve/aspects/{name}" for name in (*ASPECT_FILES, *USER_ASPECT_FILES)],
        *[f"artifacts:serve/review_centroids/{name}" for name in CENTROID_FILES],
        *[f"artifacts:{RANKER_DIR}/{name}" for name in RANKER_FILES],
        SNIPPETS,
        SPOILER,
        ONNX_FP32,
        ONNX_INT8,
        TOKENIZER,
        PROTOTYPES,
        POLES,
        SUMMARIES,
        "results:val/",
    ],
    outputs=[BUNDLE_DIR, BUNDLE_TAR],
    config_keys=list(CONFIG_KEYS),
)
def build_bundle(ctx: StageContext) -> None:
    """Bundle v0: catalogue, LightGCN and content embeddings, indices, Phase 3 files, the ranker,
    manifest and metrics."""
    ids = load_id_map(ctx.path("data:id_map.json"))
    inter = pd.read_parquet(ctx.path("data:interactions.parquet"), columns=INTER_COLUMNS[:-1])
    lgcn_dir = ctx.path(f"artifacts:{SERVE_LGCN}/meta.json").parent
    cutoff = int(json.loads((lgcn_dir / "meta.json").read_text("utf-8"))["cutoff"])
    if int(inter["ts"].max()) >= cutoff:
        raise ValueError("serve LightGCN cutoff is older than the interactions")
    quantized = bool(ctx.param("serving")["onnx_quantized"])
    version = str(ctx.param("project.data_version"))
    spec = BundleSpec(
        label=version,
        ids=ids,
        books=_books(ctx),
        inter=inter,
        cutoff=cutoff,
        lgcn_dir=lgcn_dir,
        content=np.load(ctx.path("artifacts:content/content_item.npy")),
        item_nn=np.load(ctx.path("artifacts:content/item_nn.npy")),
        aspects_dir=ctx.path("artifacts:serve/aspects/item_aspects.npy").parent,
        centroids_dir=ctx.path("artifacts:serve/review_centroids/review_centroid_item.npy").parent,
        prototypes=ctx.path(PROTOTYPES),
        poles=ctx.path(POLES),
        corpus_limit=CORPUS_USERS,
        params=_params(ctx),
        metrics=metrics_report(ctx.path("results:val/"), version),
        snippets=ctx.path(SNIPPETS),
        spoiler=ctx.path(SPOILER),
        onnx=ctx.path(ONNX_INT8 if quantized else ONNX_FP32),
        tokenizer=ctx.path(TOKENIZER),
        ranker_dir=ctx.path(f"artifacts:{RANKER_DIR}/ranker.txt").parent,
        summaries=ctx.path(SUMMARIES),
        tar=ctx.out(BUNDLE_TAR),
    )
    folder = ctx.out(BUNDLE_DIR)
    manifest = write_bundle(folder, spec, ctx.log)
    ctx.count("bundle_files", len(manifest["files"]) + 1)
    ctx.count("corpus_users", int(np.load(folder / "corpus_users.npz")["user_index"].shape[0]))
    ctx.notes["bundle_version"] = manifest["bundle_version"]
    ctx.notes["tar_mb"] = manifest["_tar_mb"]
    ctx.notes["cutoff"] = cutoff
    ctx.count("review_summaries", int(manifest["review_summaries"]))
    ctx.cutoff = cutoff
