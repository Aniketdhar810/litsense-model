"""ArtifactStore: the serving bundle in memory (CLAUDE.md s12.1).

``load_store(bundle_dir)`` reads ``manifest.json``, memory-maps the large matrices, casts the
small arrays once, builds the indices and keeps the result as a process-wide singleton that the
public functions in ``litsense_core.recommend`` use. Files the manifest lists as ``absent`` (ranker,
ONNX encoder, aspects, snippets, spoiler model) are simply ``None`` here, and the functions that
need them degrade (s15.3) instead of raising. Only numpy, httpx and the standard library are used.
"""

import gzip
import hashlib
import json
import logging
import tarfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from litsense_core import __api_version__
from litsense_core.vector_index import VectorIndex

_LOCK = threading.Lock()
_STORE: "ArtifactStore | None" = None


class BundleError(RuntimeError):
    """The bundle on disk does not match its manifest."""


class SnippetFile:
    """Evidence snippets read per item from ``evidence_snippets.jsonl`` with a byte-offset table,
    so the process never holds every snippet as Python objects (D64). Thread-safe: each read
    opens its own handle."""

    def __init__(self, path: Path, offsets: np.ndarray) -> None:
        self.path = path
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.n_items = int(len(self.offsets) - 1)

    def for_item(self, book_id: int) -> dict[str, Any]:
        if not 0 <= book_id < self.n_items:
            return {}
        start, stop = int(self.offsets[book_id]), int(self.offsets[book_id + 1])
        if stop - start <= 3:  # "{}\n"
            return {}
        with self.path.open("rb") as f:
            f.seek(start)
            return json.loads(f.read(stop - start).decode("utf-8"))


@dataclass
class Catalog:
    """Per-item arrays and strings (``catalog.npz`` + ``catalog_text.json.gz``)."""

    year: np.ndarray
    pages: np.ndarray
    series_id: np.ndarray  # -1 when none; index into series_labels
    series_pos: np.ndarray  # float32, nan when none
    genre_indptr: np.ndarray
    genre_indices: np.ndarray
    primary_genre: np.ndarray  # index into genre_labels
    bayes_rating: np.ndarray
    rating_count: np.ndarray
    rating_std: np.ndarray
    popularity: np.ndarray
    trending: np.ndarray
    is_longtail: np.ndarray
    n_reviews: np.ndarray
    author_id: np.ndarray  # -1 when none; index into author_labels
    title: list[str]
    author: list[str | None]
    genres: list[list[str]]
    image: list[str | None]
    description: list[str]
    canonical_id: list[str]
    series_name: list[str | None]
    genre_labels: list[str]
    author_labels: list[str]
    series_labels: list[str]
    author_items: list[list[int]]
    series_items: list[list[int]]
    genre_items: list[list[int]]
    review_summary: list[str | None] = field(default_factory=list)  # genai.synthesis (D70)

    @property
    def n_items(self) -> int:
        return len(self.title)

    def summary_of(self, i: int) -> str | None:
        return self.review_summary[i] if i < len(self.review_summary) else None

    def genres_of(self, i: int) -> list[str]:
        return self.genres[i]

    def primary_genre_of(self, i: int) -> str:
        return self.genre_labels[int(self.primary_genre[i])]

    def author_of(self, i: int) -> str | None:
        return self.author[i]

    def check_id(self, book_id: int) -> int:
        if not isinstance(book_id, int | np.integer) or not 0 <= int(book_id) < self.n_items:
            raise ValueError(f"unknown book_id {book_id!r}")
        return int(book_id)


@dataclass
class ArtifactStore:
    """Everything ``recommend`` needs, read-only after load."""

    bundle_dir: Path
    version: str
    data_version: str
    cutoff: int
    manifest: dict[str, Any]
    config: dict[str, Any]
    catalog: Catalog
    lgcn_item: np.ndarray  # (n_items, d) float16 memmap
    lgcn_item_norm: np.ndarray
    content_item: np.ndarray  # (n_items, 384) float16 memmap
    item_nn: np.ndarray  # (n_items, k) int32
    trending_top: np.ndarray
    content_index: VectorIndex
    corpus_users: dict[str, int]  # corpus user id -> row in corpus_vectors
    corpus_vectors: np.ndarray  # (n_corpus, d) float32
    corpus_counts: np.ndarray  # (n_corpus,) interactions each demo reader has in the corpus
    metrics: dict[str, Any] = field(default_factory=dict)
    absent: tuple[str, ...] = ()
    # Phase 3 files (D63); None means the bundle predates them and the feature degrades (s15.3).
    aspect_names: list[str] = field(default_factory=list)
    aspect_tau: np.ndarray | None = None  # (12,) per-aspect prototype threshold
    item_aspects: np.ndarray | None = None  # (n_items, 12) shrunk polarity
    item_mention_rate: np.ndarray | None = None
    item_pos_rate: np.ndarray | None = None
    item_neg_rate: np.ndarray | None = None
    item_descriptors: np.ndarray | None = None  # (n_items, 3)
    aspect_means: np.ndarray | None = None  # (12,)
    aspect_topitems: dict[str, list[int]] | None = None
    snippets: "SnippetFile | None" = None  # per-item {aspect: {"pos": [...], "neg": [...]}}
    corpus_aspects: np.ndarray | None = None  # (n_corpus, 12) offline explicit vectors (D64)
    corpus_aspect_conf: np.ndarray | None = None
    corpus_centroids: np.ndarray | None = None  # (n_corpus, 384) offline review centroids (D66)
    review_centroid_item: np.ndarray | None = None  # (n_items, 384) float16 memmap
    aspect_prototypes: np.ndarray | None = None  # (12, 8, 384)
    descriptor_poles: np.ndarray | None = None  # (3, 2, 384)
    spoiler: Any = None  # litsense_core.spoiler.SpoilerScorer
    encoder: Any = None  # litsense_core.encoder.MiniLMEncoder
    proj_item: np.ndarray | None = None  # (n_items, d) projection hybrid items (absent on v0)
    ranker: Any = None  # lightgbm.Booster over features.FEATURES (D66); None -> blend ranks
    ranker_error: str | None = None  # why the booster did not load (shown in health(), D74)

    @property
    def n_items(self) -> int:
        return self.catalog.n_items

    @property
    def n_aspects(self) -> int:
        return len(self.aspect_names)

    def aspect_index(self, name: str) -> int | None:
        try:
            return self.aspect_names.index(name)
        except ValueError:
            return None

    def lgcn_rows(self, ids: np.ndarray) -> np.ndarray:
        return np.asarray(self.lgcn_item[ids], dtype=np.float32)

    def content_rows(self, ids: np.ndarray) -> np.ndarray:
        return np.asarray(self.content_item[ids], dtype=np.float32)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "mock": False,
            "api_version": __api_version__,
            "bundle_version": self.version,
            "data_version": self.data_version,
            "cutoff": self.cutoff,
            "rss_mb": rss_mb(),
            "ranker": self.ranker is not None,
            "ranker_error": self.ranker_error,
            "encoder": self.encoder is not None,
            "aspects": self.item_aspects is not None,
            "snippets": self.snippets is not None,
            "spoiler": self.spoiler is not None,
            "llm": llm_available(),
            "n_books": self.n_items,
            "index_backend": self.content_index.backend,
            "absent": list(self.absent),
        }


def llm_available() -> bool:
    """True only when the LLM client is configured, enabled, has a model id and a key (s14.2)."""
    from litsense_core.llm import get_client

    return get_client().available


def rss_mb() -> float | None:
    """Resident memory from /proc (Linux, where the backend runs); None elsewhere."""
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            kb = next(int(line.split()[1]) for line in f if line.startswith("VmRSS:"))
    except (OSError, StopIteration, ValueError):
        return None
    return round(kb / 1024, 1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_catalog(folder: Path, n_items: int) -> Catalog:
    arrays = np.load(folder / "catalog.npz")
    with gzip.open(folder / "catalog_text.json.gz", "rt", encoding="utf-8") as f:
        text = json.load(f)
    indices = {
        name: json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))
        for name in ("author_index", "series_index", "genre_index")
    }
    for key in ("year", "primary_genre", "popularity", "author_id"):
        if arrays[key].shape[0] != n_items:
            raise BundleError(
                f"catalog.npz/{key} has {arrays[key].shape[0]} rows, manifest {n_items}"
            )
    if len(text["title"]) != n_items:
        raise BundleError(f"catalog_text has {len(text['title'])} titles, manifest {n_items}")
    return Catalog(
        year=arrays["year"].astype(np.int32),
        pages=arrays["pages"].astype(np.int32),
        series_id=arrays["series_id"].astype(np.int32),
        series_pos=arrays["series_pos"].astype(np.float32),
        genre_indptr=arrays["genre_indptr"].astype(np.int64),
        genre_indices=arrays["genre_indices"].astype(np.int32),
        primary_genre=arrays["primary_genre"].astype(np.int32),
        bayes_rating=arrays["bayes_rating"].astype(np.float32),
        rating_count=arrays["rating_count"].astype(np.int32),
        rating_std=arrays["rating_std"].astype(np.float32),
        popularity=arrays["popularity"].astype(np.float32),
        trending=arrays["trending"].astype(np.float32),
        is_longtail=arrays["is_longtail"].astype(bool),
        n_reviews=arrays["n_reviews"].astype(np.int32),
        author_id=arrays["author_id"].astype(np.int32),
        title=list(text["title"]),
        author=list(text["author"]),
        genres=[list(g) for g in text["genres"]],
        image=list(text["image"]),
        description=list(text["description"]),
        canonical_id=list(text["canonical_id"]),
        series_name=list(text["series_name"]),
        genre_labels=list(indices["genre_index"]["labels"]),
        author_labels=list(indices["author_index"]["labels"]),
        series_labels=list(indices["series_index"]["labels"]),
        author_items=[list(map(int, x)) for x in indices["author_index"]["items"]],
        series_items=[list(map(int, x)) for x in indices["series_index"]["items"]],
        genre_items=[list(map(int, x)) for x in indices["genre_index"]["items"]],
        review_summary=[s or None for s in (text.get("review_summary") or [None] * n_items)],
    )


def read_store(bundle_dir: str | Path, verify_hashes: bool | None = None) -> ArtifactStore:
    """Load a bundle folder into a new ``ArtifactStore`` (no singleton; tests use this)."""
    folder = Path(bundle_dir)
    if (folder / "serve_bundle" / "manifest.json").exists():  # an extracted tar
        folder = folder / "serve_bundle"
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        raise BundleError(f"no manifest.json in {folder}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = dict(manifest.get("config", {}))
    if verify_hashes is None:
        verify_hashes = bool(config.get("serving", {}).get("verify_hashes_on_load", False))
    for name, entry in manifest["files"].items():
        path = folder / name
        if not path.exists():
            raise BundleError(f"bundle file {name} listed in the manifest is missing")
        if verify_hashes and _sha256(path) != entry["sha256"]:
            raise BundleError(f"bundle file {name} does not match its sha256")
    n_items = int(manifest["n_items"])
    catalog = _load_catalog(folder, n_items)
    lgcn_item = np.load(folder / "lgcn_item.f16.npy", mmap_mode="r")
    lgcn_norm = np.load(folder / "lgcn_item_norm.f16.npy", mmap_mode="r")
    content = np.load(folder / "content_item.f16.npy", mmap_mode="r")
    item_nn = np.load(folder / "item_nn.npy").astype(np.int32)
    for name, arr in (("lgcn_item", lgcn_item), ("content_item", content), ("item_nn", item_nn)):
        if arr.shape[0] != n_items:
            raise BundleError(f"{name} has {arr.shape[0]} rows, manifest says {n_items}")
    dims = manifest.get("dims", {})
    if dims and (lgcn_item.shape[1] != dims["lgcn"] or content.shape[1] != dims["content"]):
        raise BundleError("embedding dims disagree with the manifest")
    users = np.load(folder / "corpus_users.npz", allow_pickle=False)
    corpus = {str(u): k for k, u in enumerate(users["user_id"].tolist())}
    corpus_aspects = (
        np.asarray(users["aspects"], dtype=np.float32) if "aspects" in users.files else None
    )
    corpus_conf = (
        np.asarray(users["aspect_conf"], dtype=np.float32) if "aspect_conf" in users.files else None
    )
    corpus_centroids = (
        np.asarray(users["centroid"], dtype=np.float32) if "centroid" in users.files else None
    )
    metrics_path = folder / "metrics.json"
    metrics = json.loads(metrics_path.read_text("utf-8")) if metrics_path.exists() else {}
    content_f32 = np.asarray(content, dtype=np.float32)  # one full copy for exact search
    extras = _load_phase3(folder, manifest, config, n_items)
    return ArtifactStore(
        bundle_dir=folder,
        version=str(manifest["bundle_version"]),
        data_version=str(manifest["data_version"]),
        cutoff=int(manifest["cutoff"]),
        manifest=manifest,
        config=config,
        catalog=catalog,
        lgcn_item=lgcn_item,
        lgcn_item_norm=lgcn_norm,
        content_item=content,
        item_nn=item_nn,
        trending_top=np.load(folder / "trending_top.npy").astype(np.int64),
        content_index=VectorIndex(content_f32),
        corpus_users=corpus,
        corpus_vectors=np.asarray(users["lgcn_user"], dtype=np.float32),
        corpus_counts=np.asarray(users["n_interactions"], dtype=np.int64),
        corpus_aspects=corpus_aspects,
        corpus_aspect_conf=corpus_conf,
        corpus_centroids=corpus_centroids,
        metrics=metrics,
        absent=tuple(manifest.get("absent", ())),
        **extras,
    )


def _optional(folder: Path, name: str) -> Path | None:
    path = folder / name
    return path if path.exists() else None


def _load_phase3(
    folder: Path, manifest: dict[str, Any], config: dict[str, Any], n_items: int
) -> dict[str, Any]:
    """Aspect profiles, snippets, centroids, prototypes, spoiler scorer and ONNX encoder when the
    bundle has them (D63). Each is independent: a missing file leaves its feature degraded."""
    out: dict[str, Any] = {}
    aspects_cfg = config.get("aspects") or {}
    names = list(aspects_cfg.get("names") or [])
    out["aspect_names"] = names
    if names:
        default = float(aspects_cfg.get("tau_default", 0.45))
        overrides = aspects_cfg.get("tau") or {}
        out["aspect_tau"] = np.array([float(overrides.get(a, default)) for a in names], np.float32)
    for key, name in (
        ("item_aspects", "item_aspects.npy"),
        ("item_mention_rate", "item_mention_rate.npy"),
        ("item_pos_rate", "item_pos_rate.npy"),
        ("item_neg_rate", "item_neg_rate.npy"),
        ("item_descriptors", "item_descriptors.npy"),
    ):
        path = _optional(folder, name)
        if path is not None:
            arr = np.load(path).astype(np.float32)
            if arr.shape[0] != n_items:
                raise BundleError(f"{name} has {arr.shape[0]} rows, manifest says {n_items}")
            out[key] = arr
    if (path := _optional(folder, "aspect_means.npy")) is not None:
        out["aspect_means"] = np.load(path).astype(np.float32)
    if (path := _optional(folder, "aspect_topitems.json")) is not None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        out["aspect_topitems"] = {str(k): [int(i) for i in v] for k, v in raw.items()}
    jsonl, offsets = (
        _optional(folder, "evidence_snippets.jsonl"),
        _optional(folder, "snippets_offsets.npy"),
    )
    if jsonl is not None and offsets is not None:
        table = np.load(offsets)
        if len(table) != n_items + 1:
            raise BundleError("snippets_offsets.npy does not match the catalogue size")
        out["snippets"] = SnippetFile(jsonl, table)
    if (path := _optional(folder, "review_centroid_item.f16.npy")) is not None:
        out["review_centroid_item"] = np.load(path, mmap_mode="r")
    if (path := _optional(folder, "aspect_prototypes.npy")) is not None:
        out["aspect_prototypes"] = np.load(path).astype(np.float32)
    if (path := _optional(folder, "descriptor_poles.npy")) is not None:
        out["descriptor_poles"] = np.load(path).astype(np.float32)
    if (path := _optional(folder, "spoiler_lr.npz")) is not None:
        from litsense_core.spoiler import SpoilerScorer

        out["spoiler"] = SpoilerScorer.from_npz(path)
    onnx, tok = _optional(folder, "minilm.onnx"), _optional(folder, "tokenizer.json")
    if onnx is not None and tok is not None:
        from litsense_core.encoder import MiniLMEncoder

        onnx_meta = manifest.get("onnx") or {}
        threads = int((config.get("serving") or {}).get("onnx_threads", 1))
        try:
            out["encoder"] = MiniLMEncoder(
                onnx, tok, int(onnx_meta.get("max_seq_length", 256)), threads
            )
        except Exception:  # noqa: BLE001 (a broken runtime degrades, s15.3)
            out["encoder"] = None
    if (path := _optional(folder, "proj_item.f16.npy")) is not None:
        out["proj_item"] = np.load(path, mmap_mode="r")
    out["ranker"], out["ranker_error"] = _load_ranker(folder)
    return out


def _load_ranker(folder: Path) -> tuple[Any, str | None]:
    """``(booster, error)``: the LightGBM LambdaRank booster (``ranker.txt``) when the bundle
    ships one whose ``feature_list.json`` equals ``features.FEATURES`` in order; otherwise
    ``(None, why)``, so the blend ranks, ``recommend`` reports ``degraded`` (s15.3) and
    ``health()["ranker_error"]`` says what to fix. A column-order mismatch would score silently
    wrong, which is why it is refused rather than tolerated. The usual runtime cause is a
    minimal image without ``libgomp.so.1``, which lightgbm needs (D74)."""
    model, names = _optional(folder, "ranker.txt"), _optional(folder, "feature_list.json")
    if model is None or names is None:
        return None, "ranker.txt or feature_list.json not in the bundle"
    from litsense_core.features import FEATURES

    if list(json.loads(names.read_text(encoding="utf-8"))) != list(FEATURES):
        return None, "feature_list.json does not match features.FEATURES"
    try:
        import lightgbm as lgb

        return lgb.Booster(model_file=str(model)), None
    except Exception as exc:  # noqa: BLE001 (a broken runtime degrades, s15.3)
        message = f"{type(exc).__name__}: {str(exc)[:200]}"
        logging.getLogger("litsense_core").warning(
            "ranker not loaded, blend will rank: %s", message
        )
        return None, message


def load_store(bundle_dir: str) -> ArtifactStore:
    """Process-wide singleton (s12.1): the first call loads, later calls return the same store
    unless ``bundle_dir`` changes."""
    global _STORE
    with _LOCK:
        if _STORE is None or Path(bundle_dir).resolve() not in (
            _STORE.bundle_dir.resolve(),
            _STORE.bundle_dir.parent.resolve(),
        ):
            _STORE = read_store(bundle_dir)
            _configure_llm_from(_STORE)
        return _STORE


def _configure_llm_from(store: ArtifactStore) -> None:
    """Install the LLM client from the bundle's ``llm`` config slice unless the backend already
    configured one (``litsense_core.configure_llm``); bundles without the slice fall back to
    ``ml/config.yaml`` by path (D68)."""
    from litsense_core.llm.client import configure_default

    configure_default(store.config.get("llm"))


def current_store() -> ArtifactStore:
    """The loaded store, or a clear error telling the caller to ``load_store`` first."""
    if _STORE is None:
        raise RuntimeError("litsense_core.load_store(bundle_dir) must be called first")
    return _STORE


def set_store(store: ArtifactStore | None) -> None:
    """Install a store built elsewhere (tests)."""
    global _STORE
    with _LOCK:
        _STORE = store


def download_bundle(url: str, dest: str) -> str:
    """Fetch ``serve_bundle.tar.gz`` from ``url`` into ``dest`` and extract it; returns the
    folder to pass to ``load_store``. Skips the download when a manifest is already there."""
    import httpx

    target = Path(dest)
    target.mkdir(parents=True, exist_ok=True)
    if (target / "serve_bundle" / "manifest.json").exists():
        return str(target / "serve_bundle")
    tar_path = target / "serve_bundle.tar.gz"
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
        response.raise_for_status()
        with tar_path.open("wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
    with tarfile.open(tar_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.startswith("serve_bundle/")]
        tar.extractall(target, members=members, filter="data")
    return str(target / "serve_bundle")
