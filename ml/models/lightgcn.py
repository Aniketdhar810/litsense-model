"""LightGCN (CLAUDE.md s9.4), ported from the team's earlier notebook with the s2.2 fixes.

* Graph: fit-window interactions as implicit edges, symmetric-normalized D^-1/2 A D^-1/2 over the
  bipartite graph; zero-degree nodes get zero rows.
* Model: Xavier-uniform user and item embeddings, K propagation layers, final = mean of layers 0..K.
* Loss: BPR with ``n_negs`` negatives per positive (mean over negatives) plus L2 on the batch's
  layer-0 embeddings divided by the batch size.
* Early stopping on an inner validation set (``inner_split``: ``temporal`` holds out the last
  ``inner_val_fraction`` of the fit window by time, D48; ``user`` holds out each user's last
  ``inner_val_fraction``), scored by the shared evaluator; then a refit on the whole fit window
  for the best epoch count. Only that refit is saved, and only after it finishes (s2.2 #6).

Stage ``models.lightgcn_val`` trains on train (cutoff t1) and writes ``artifacts/<v>/eval_val/
lightgcn/``. The test-mode artifact set (train + val, t2) is built in Week 13 by the same function.
"""

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import nn

from ml.common.io import load_id_map
from ml.common.stage import StageContext, stage
from ml.eval.data import EvalData, Mode, build_eval_data
from ml.eval.evaluator import evaluate
from ml.models.embedding_scorers import EmbeddingArtifact, EmbeddingScorer, l2_normalize_rows

EvalFn = Callable[[np.ndarray, np.ndarray], float]


def normalized_adjacency(
    users: np.ndarray, items: np.ndarray, n_users: int, n_items: int
) -> sp.coo_matrix:
    """D^-1/2 A D^-1/2 of the (users + items) bipartite graph; isolated nodes stay zero."""
    n = n_users + n_items
    rows = np.concatenate([users, items + n_users])
    cols = np.concatenate([items + n_users, users])
    adj = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n))
    adj.data[:] = 1.0  # a repeated (user, item) pair is still one edge
    degree = np.asarray(adj.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degree)
    inv_sqrt[degree > 0] = degree[degree > 0] ** -0.5
    scale = sp.diags(inv_sqrt)
    return (scale @ adj @ scale).tocoo().astype(np.float32)


def to_torch_sparse(matrix: sp.coo_matrix, device: str) -> torch.Tensor:
    """A coalesced ``torch.sparse_coo_tensor`` on ``device``."""
    indices = torch.from_numpy(np.vstack([matrix.row, matrix.col]).astype(np.int64))
    values = torch.from_numpy(matrix.data.astype(np.float32))
    tensor = torch.sparse_coo_tensor(indices, values, matrix.shape, check_invariants=True)
    return tensor.coalesce().to(device)


def sample_negatives(
    users: np.ndarray,
    n_items: int,
    pos_keys: np.ndarray,
    n_negs: int,
    rng: np.random.Generator,
    rounds: int = 10,
) -> np.ndarray:
    """(len(users), n_negs) items the users never interacted with in the fit window.

    ``pos_keys`` is the sorted int64 array of ``u * n_items + i`` over every fit positive. Clashes
    are redrawn ``rounds`` times (vectorized); the rare survivors are drawn from the user's
    complement, so no positive is ever returned."""
    users = np.asarray(users, dtype=np.int64)
    neg = rng.integers(0, n_items, size=(len(users), n_negs))

    def clashes() -> np.ndarray:
        keys = users[:, None] * n_items + neg
        idx = np.clip(np.searchsorted(pos_keys, keys), 0, len(pos_keys) - 1)
        return pos_keys[idx] == keys

    for _ in range(rounds):
        bad = clashes()
        if not bad.any():
            return neg
        neg[bad] = rng.integers(0, n_items, size=int(bad.sum()))
    for r, c in zip(*np.nonzero(clashes()), strict=True):
        u = users[r]
        lo, hi = np.searchsorted(pos_keys, [u * n_items, (u + 1) * n_items])
        allowed = np.setdiff1d(np.arange(n_items), pos_keys[lo:hi] - u * n_items)
        neg[r, c] = rng.choice(allowed)
    return neg


class LightGCN(nn.Module):
    """User and item ego embeddings propagated over the normalized graph."""

    def __init__(self, n_users: int, n_items: int, dim: int, layers: int) -> None:
        super().__init__()
        self.n_users, self.layers = n_users, layers
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        nn.init.xavier_uniform_(self.user.weight)
        nn.init.xavier_uniform_(self.item.weight)

    def forward(self, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Final user and item embeddings: the mean of layers 0..K."""
        x = torch.cat([self.user.weight, self.item.weight])
        layers = [x]
        for _ in range(self.layers):
            x = torch.sparse.mm(adj, x)
            layers.append(x)
        final = torch.stack(layers).mean(dim=0)
        return final[: self.n_users], final[self.n_users :]


def bpr_loss(
    model: LightGCN,
    adj: torch.Tensor,
    users: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    reg: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(total loss, BPR part). ``neg`` is (batch, n_negs); BPR is averaged over negatives and the
    L2 term covers the batch's layer-0 embeddings, divided by the batch size."""
    user_final, item_final = model(adj)
    u = user_final[users]
    pos_scores = (u * item_final[pos]).sum(-1)
    neg_scores = (u[:, None, :] * item_final[neg]).sum(-1)
    bpr = nn.functional.softplus(neg_scores - pos_scores[:, None]).mean()
    ego = (
        model.user.weight[users].pow(2).sum()
        + model.item.weight[pos].pow(2).sum()
        + model.item.weight[neg].pow(2).sum() / neg.shape[1]
    )
    return bpr + reg * 0.5 * ego / len(users), bpr


@dataclass
class TrainResult:
    """Final embeddings plus the early-stopping record."""

    user: np.ndarray
    item: np.ndarray
    epochs: int  # epochs actually trained
    best_epoch: int  # best inner-validation epoch (= epochs when nothing was evaluated)
    best_metric: float = float("nan")
    history: list[dict[str, float]] = field(default_factory=list)


def _embeddings(model: LightGCN, adj: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        user, item = model(adj)
    model.train()
    return user.cpu().numpy().astype(np.float32), item.cpu().numpy().astype(np.float32)


def train_lightgcn(
    rows: pd.DataFrame,
    n_users: int,
    n_items: int,
    params: Mapping[str, Any],
    seed: int,
    epochs: int,
    device: str = "cpu",
    evaluate_fn: EvalFn | None = None,
    log: Callable[[str], None] | None = None,
) -> TrainResult:
    """Train for up to ``epochs`` epochs (one epoch = every fit positive once, in shuffled
    batches). With ``evaluate_fn`` (final user and item embeddings -> metric), evaluate every
    ``eval_every`` epochs and stop after ``patience`` evaluations without improvement."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    users = rows["u"].to_numpy(np.int64)
    items = rows["i"].to_numpy(np.int64)
    adj = to_torch_sparse(normalized_adjacency(users, items, n_users, n_items), device)
    pos_keys = np.unique(users * n_items + items)
    model = LightGCN(n_users, n_items, int(params["dim"]), int(params["layers"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params["lr"]))
    batch, n_negs, reg = int(params["batch_size"]), int(params["n_negs"]), float(params["reg"])
    eval_every, patience = int(params["eval_every"]), int(params["patience"])

    result = TrainResult(np.empty(0), np.empty(0), 0, epochs)
    best, stale, started = -np.inf, 0, time.monotonic()
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(users))
        loss_sum = 0.0
        for start in range(0, len(order), batch):
            b = order[start : start + batch]
            neg = sample_negatives(users[b], n_items, pos_keys, n_negs, rng)
            loss, bpr = bpr_loss(
                model,
                adj,
                torch.from_numpy(users[b]).to(device),
                torch.from_numpy(items[b]).to(device),
                torch.from_numpy(neg).to(device),
                reg,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(bpr.detach()) * len(b)
        result.epochs = epoch
        if evaluate_fn is None or epoch % eval_every:
            continue
        metric = evaluate_fn(*_embeddings(model, adj))
        seconds = time.monotonic() - started
        result.history.append({"epoch": epoch, "bpr": loss_sum / len(order),
                               "metric": metric, "seconds": seconds})  # fmt: skip
        if log:
            log(f"epoch {epoch}: bpr {loss_sum / len(order):.4f}, metric {metric:.4f}, "
                f"{seconds:.0f} s")  # fmt: skip
        if metric > best:
            best, stale = metric, 0
            result.best_epoch, result.best_metric = epoch, metric
        else:
            stale += 1
            if stale >= patience:
                break
    result.user, result.item = _embeddings(model, adj)
    return result


def embedding_health(
    item: np.ndarray, warm: np.ndarray, sample: int = 2000, seed: int = 0
) -> dict[str, float]:
    """Anisotropy and scale of item embeddings (s2.2 #9, D47): mean pairwise cosine over up to
    ``sample`` warm items, and the min/median/max row norm over warm items. A mean cosine above
    0.5 means the vectors point the same way and ranking degenerates toward item order."""
    rows = item[np.asarray(warm, bool)]
    if len(rows) < 2:
        nan = float("nan")
        return {"mean_pairwise_cosine": nan, "norm_min": nan, "norm_median": nan, "norm_max": nan,
                "items": int(len(rows))}  # fmt: skip
    rng = np.random.default_rng(seed)
    picked = rows[rng.choice(len(rows), min(sample, len(rows)), replace=False)]
    unit = l2_normalize_rows(picked.astype(np.float64))
    n = len(unit)
    total = float((unit.sum(axis=0) ** 2).sum())  # sum over all ordered pairs incl. self
    mean_cos = (total - n) / (n * (n - 1))
    norms = np.linalg.norm(rows, axis=1)
    return {
        "mean_pairwise_cosine": float(mean_cos),
        "norm_min": float(norms.min()),
        "norm_median": float(np.median(norms)),
        "norm_max": float(norms.max()),
        "items": int(len(rows)),
    }


def user_norm_summary(user: np.ndarray) -> dict[str, float]:
    """Min, median and max row norm of the user embeddings."""
    norms = np.linalg.norm(user, axis=1)
    return {"min": float(norms.min()), "median": float(np.median(norms)), "max": float(norms.max())}


def inner_split(
    rows: pd.DataFrame, fraction: float, min_user: int, mode: str = "user"
) -> pd.Series:
    """Boolean mask of held-out rows.

    ``"user"``: the last ``fraction`` (at least one) of each user's fit interactions by time, for
    users with at least ``min_user`` of them. ``"temporal"`` (D48): every row at or after the
    ``1 - fraction`` quantile of the fit timestamps, a global time cutoff like the outer val, so
    held-out users have only their older rows to learn from and some held-out items are cold,
    exactly as in the outer evaluation. ``min_user`` is ignored in temporal mode."""
    if mode == "temporal":
        if not len(rows):
            return pd.Series(False, index=rows.index)
        cutoff = np.quantile(rows["ts"].to_numpy(), 1 - fraction)
        return pd.Series(rows["ts"].to_numpy() >= cutoff, index=rows.index)
    if mode != "user":
        raise ValueError(f"inner_split must be 'user' or 'temporal', got {mode!r}")
    ordered = rows.sort_values(["u", "ts", "i"], kind="stable")
    position = ordered.groupby("u").cumcount().to_numpy()
    size = ordered.groupby("u")["u"].transform("size").to_numpy()
    held_n = np.where(size >= min_user, np.maximum(1, np.floor(size * fraction)), 0)
    return pd.Series(position >= size - held_n, index=ordered.index).reindex(rows.index)


def inner_eval_data(
    data: EvalData,
    fraction: float,
    min_user: int,
    relevance: float,
    longtail: float,
    mode: str = "user",
) -> EvalData:
    """Inner validation carved from ``data``'s fit window, built by the shared ``build_eval_data``
    (held rows play the role of the val split; same relevance rule and masks)."""
    rows = data.fit.interactions.copy()
    rows["split"] = np.where(inner_split(rows, fraction, min_user, mode), "val", "train")
    splits = {"t1": {"ms": data.cutoff}}
    return build_eval_data(
        rows, splits, "val", data.fit.n_users, data.fit.n_items, relevance, longtail
    )


def inner_metric(data: EvalData, metric: str) -> EvalFn:
    """``evaluate_fn`` for ``train_lightgcn``: ``metric`` (e.g. ``ndcg@20``) of the raw LightGCN
    scorer on the inner validation users, through the shared evaluator."""
    k = int(metric.split("@")[1])

    def run(user: np.ndarray, item: np.ndarray) -> float:
        artifact = EmbeddingArtifact(user, item, data.item_counts, {"cutoff": data.cutoff})
        scorer = EmbeddingScorer("lightgcn_inner", artifact, normalize_items=False)
        scorer.fit(data.fit)
        return float(evaluate(scorer, data, ks=(k,)).means[metric])

    return run


def build_lightgcn(ctx: StageContext, mode: Mode, folder: str, history: str) -> None:
    """Early-stopping search on the inner split, refit on the whole fit window, save."""
    ids = load_id_map(ctx.path("data:id_map.json"))
    splits = json.loads(ctx.path("data:splits.json").read_text(encoding="utf-8"))
    relevance = float(ctx.param("data.relevance_min_rating"))
    longtail = float(ctx.param("data.longtail_fraction"))
    data = build_eval_data(
        pd.read_parquet(ctx.path("data:interactions.parquet")),
        splits, mode, ids.n_users, ids.n_items, relevance, longtail,
    )  # fmt: skip
    ctx.cutoff = data.cutoff
    params = ctx.param("models.lightgcn")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx.log.info("training on %s (torch %s)", device, torch.__version__)
    started = time.monotonic()

    inner = inner_eval_data(
        data, float(params["inner_val_fraction"]), int(params["inner_val_min_user"]),
        relevance, longtail, str(params.get("inner_split", "user")),
    )  # fmt: skip
    ctx.count("fit_interactions", len(data.fit.interactions))
    ctx.count("inner_train_interactions", len(inner.fit.interactions))
    ctx.count("inner_val_users", len(inner.users))
    max_epochs = int(params["max_epochs"])
    if len(inner.users):
        search = train_lightgcn(
            inner.fit.interactions, ids.n_users, ids.n_items, params, ctx.seed, max_epochs,
            device, inner_metric(inner, str(params["metric"])), ctx.log.info,
        )  # fmt: skip
    else:
        ctx.log.warning("no inner validation users; training for max_epochs without stopping")
        search = TrainResult(np.empty(0), np.empty(0), 0, max_epochs)
    final = train_lightgcn(
        data.fit.interactions, ids.n_users, ids.n_items, params, ctx.seed, search.best_epoch,
        device,
    )  # fmt: skip
    dim = int(params["dim"])
    assert final.user.shape == (ids.n_users, dim), final.user.shape  # s2.2 #8: dims from config
    assert final.item.shape == (ids.n_items, dim), final.item.shape

    out = {name: ctx.out(f"artifacts:{folder}/{name}") for name in LIGHTGCN_FILES}
    np.save(out["lgcn_user.npy"], final.user)
    np.save(out["lgcn_item.npy"], final.item)
    np.save(out["lgcn_item_norm.npy"], l2_normalize_rows(final.item))
    np.save(out["item_degree.npy"], data.item_counts.astype(np.int64))
    meta = {
        "model": "lightgcn",
        "mode": mode,
        "cutoff": data.cutoff,
        "params": dict(params),
        "best_epoch": search.best_epoch,
        "search_epochs": search.epochs,
        "best_inner_metric": search.best_metric,
        "zero_degree_items": int((data.item_counts == 0).sum()),
        "embedding_health": embedding_health(final.item, data.item_counts > 0),
        "user_norm": user_norm_summary(final.user),
        "device": device,
        "torch": torch.__version__,
        "seconds": round(time.monotonic() - started, 1),
    }
    health = meta["embedding_health"]
    cosine, lo, hi = health["mean_pairwise_cosine"], health["norm_min"], health["norm_max"]
    ctx.log.info("item embeddings: mean pairwise cosine %.3f, norms %.2f..%.2f", cosine, lo, hi)
    if cosine > 0.5:
        ctx.log.warning("item embeddings are anisotropic (s2.2 #9): mean cosine > 0.5")
    out["meta.json"].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(search.history, columns=["epoch", "bpr", "metric", "seconds"]).to_csv(
        ctx.out(history), index=False
    )
    ctx.count("best_epoch", search.best_epoch)
    ctx.count("zero_degree_items", meta["zero_degree_items"])
    ctx.notes["best_inner_metric"] = search.best_metric


LIGHTGCN_FILES = ("lgcn_user.npy", "lgcn_item.npy", "lgcn_item_norm.npy", "item_degree.npy",
                  "meta.json")  # fmt: skip
VAL_FOLDER = "eval_val/lightgcn"
VAL_HISTORY = "results:training/lightgcn_val.csv"


@stage(
    "models.lightgcn_val",
    inputs=["data:interactions.parquet", "data:splits.json", "data:id_map.json"],
    outputs=[f"artifacts:{VAL_FOLDER}/{name}" for name in LIGHTGCN_FILES] + [VAL_HISTORY],
    config_keys=["models.lightgcn", "data.relevance_min_rating", "data.longtail_fraction"],
)
def build_lightgcn_val(ctx: StageContext) -> None:
    """LightGCN fit on train (cutoff t1) for validation."""
    build_lightgcn(ctx, "val", VAL_FOLDER, VAL_HISTORY)
