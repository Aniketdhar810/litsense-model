"""Projection hybrid and its shuffled-text control (CLAUDE.md s9.5).

Item vector e'_i = e_gcn_i + W c_i, with LightGCN user and item embeddings frozen and only
W (dim x 384, no bias) learned by BPR. e_gcn is zero for items without fit interactions, so cold
items reach CF space only through W c_i. With ``normalize_items`` the same normalized score
u . e'_i / |e'_i| is used in training and in evaluation (s2.2 #1); the default (D47) is the paper's
additive fusion without normalization. ``w_init`` ``"zero"`` starts W at zero, so the first step
scores exactly like raw LightGCN and training learns only the text correction; ``w_reg`` adds
0.5 * w_reg * |W|^2 to every step's loss. ``gcn_dropout`` (D48) zeroes e_gcn for a random
fraction of items at each step (training only): the frozen LightGCN already ranks almost every
fit pair correctly (BPR about 0.006), so without it W gets no gradient; with it, W must score the
masked items from text alone, which is the cold-item case it exists for.

The shuffled control trains identically with the content rows permuted across items (seeded): if it
matches the real projection, the gain is not semantic.

Stage ``models.projection_val`` writes ``artifacts/<v>/eval_val/projection/`` and
``.../projection_shuffled/``: ``W.npy``, ``proj_item.npy`` (normalized when ``normalize_items``),
the LightGCN user vectors it scores with, ``item_degree.npy`` and ``meta.json``.
"""

import json
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from ml.common.io import load_id_map
from ml.common.stage import StageContext, stage
from ml.eval.data import build_eval_data
from ml.models.embedding_scorers import EmbeddingArtifact, check_artifact
from ml.models.lightgcn import embedding_health, sample_negatives

LIGHTGCN = "eval_val/lightgcn"
FILES = ("W.npy", "proj_item.npy", "user.npy", "item_degree.npy", "meta.json")
VARIANTS = {"projection": False, "projection_shuffled": True}  # name -> shuffled content


def project_items(
    gcn: torch.Tensor, content: torch.Tensor, w: torch.Tensor, normalize: bool
) -> torch.Tensor:
    """e' = e_gcn + c W^T, optionally L2-normalized by row."""
    items = gcn + content @ w.T
    return nn.functional.normalize(items, dim=-1) if normalize else items


def train_projection(
    rows: pd.DataFrame,
    user: np.ndarray,
    item: np.ndarray,
    degree: np.ndarray,
    content: np.ndarray,
    params: Mapping[str, Any],
    seed: int,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Learn W by BPR over the fit positives; returns (W, item matrix e', mean BPR per epoch)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_items = len(item)
    users = rows["u"].to_numpy(np.int64)
    items = rows["i"].to_numpy(np.int64)
    pos_keys = np.unique(users * n_items + items)
    gcn_np = item * (degree > 0)[:, None]  # zero for items without fit interactions
    u_t = torch.from_numpy(user.astype(np.float32)).to(device)
    gcn = torch.from_numpy(gcn_np.astype(np.float32)).to(device)
    c = torch.from_numpy(content.astype(np.float32)).to(device)
    w = nn.Parameter(torch.empty(item.shape[1], content.shape[1], device=device))
    w_init = str(params.get("w_init", "xavier"))
    if w_init == "zero":
        nn.init.zeros_(w)
    elif w_init == "xavier":
        nn.init.xavier_uniform_(w)
    else:
        raise ValueError(f"models.projection.w_init must be 'zero' or 'xavier', got {w_init!r}")
    w_reg = float(params.get("w_reg", 0.0))
    dropout = float(params.get("gcn_dropout", 0.0))
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"models.projection.gcn_dropout must be in [0, 1), got {dropout}")
    optimizer = torch.optim.Adam([w], lr=float(params["lr"]))
    normalize = bool(params["normalize_items"])
    batch, n_negs = int(params["batch_size"]), int(params["n_negs"])
    generator = torch.Generator(device=device).manual_seed(seed)

    losses = []
    for _ in range(int(params["epochs"])):
        total = 0.0
        for _ in range(int(params["steps_per_epoch"])):
            idx = rng.integers(0, len(users), batch)
            neg = torch.from_numpy(sample_negatives(users[idx], n_items, pos_keys, n_negs, rng))
            b_users = torch.from_numpy(users[idx]).to(device)
            b_pos = torch.from_numpy(items[idx]).to(device)
            neg = neg.to(device)
            u = u_t[b_users]
            step_gcn = gcn
            if dropout:
                # D48: hide the LightGCN half of a random item subset this step, so the pairs that
                # the frozen LightGCN already separates still give W a gradient (cold-item case).
                keep = torch.rand(n_items, generator=generator, device=device) >= dropout
                step_gcn = gcn * keep[:, None]
            pos_vec = project_items(step_gcn[b_pos], c[b_pos], w, normalize)
            neg_vec = project_items(step_gcn[neg], c[neg], w, normalize)
            pos_s = (u * pos_vec).sum(-1)
            neg_s = (u[:, None, :] * neg_vec).sum(-1)
            bpr = nn.functional.softplus(neg_s - pos_s[:, None]).mean()
            loss = bpr + 0.5 * w_reg * w.pow(2).sum() if w_reg else bpr
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(bpr.detach())
        losses.append(total / int(params["steps_per_epoch"]))
    with torch.no_grad():
        final = project_items(gcn, c, w, normalize).cpu().numpy().astype(np.float32)
    return w.detach().cpu().numpy().astype(np.float32), final, losses


@stage(
    "models.projection_val",
    inputs=[
        "data:interactions.parquet",
        "data:splits.json",
        "data:id_map.json",
        "artifacts:content/content_item.npy",
        *[
            f"artifacts:{LIGHTGCN}/{n}"
            for n in ("lgcn_user.npy", "lgcn_item.npy", "item_degree.npy", "meta.json")
        ],
    ],  # fmt: skip
    outputs=[f"artifacts:eval_val/{v}/{n}" for v in VARIANTS for n in FILES],
    config_keys=["models.projection", "data.relevance_min_rating", "data.longtail_fraction"],
)
def build_projection_val(ctx: StageContext) -> None:
    """Projection hybrid and its shuffled control, trained on train (cutoff t1)."""
    ids = load_id_map(ctx.path("data:id_map.json"))
    splits = json.loads(ctx.path("data:splits.json").read_text(encoding="utf-8"))
    data = build_eval_data(
        pd.read_parquet(ctx.path("data:interactions.parquet")),
        splits, "val", ids.n_users, ids.n_items,
        float(ctx.param("data.relevance_min_rating")), float(ctx.param("data.longtail_fraction")),
    )  # fmt: skip
    ctx.cutoff = data.cutoff
    lightgcn = EmbeddingArtifact.load(
        ctx.path(f"artifacts:{LIGHTGCN}/meta.json").parent, "lgcn_user.npy", "lgcn_item.npy"
    )
    check_artifact("projection input", lightgcn, data.fit)
    content = np.load(ctx.path("artifacts:content/content_item.npy")).astype(np.float32)
    if len(content) != ids.n_items:
        raise ValueError(f"content has {len(content)} rows for {ids.n_items} items")
    params = ctx.param("models.projection")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not params["shuffled_control"]:
        raise ValueError("models.projection.shuffled_control must stay on (s9.5 control)")

    for name, shuffled in VARIANTS.items():
        started = time.monotonic()
        rows = content[ctx.rng.permutation(len(content))] if shuffled else content
        w, proj, losses = train_projection(
            data.fit.interactions, lightgcn.user, lightgcn.item, lightgcn.degree, rows,
            params, ctx.seed, device,
        )  # fmt: skip
        assert w.shape == (lightgcn.item.shape[1], content.shape[1]), w.shape  # s2.2 #8
        folder = f"artifacts:eval_val/{name}"
        np.save(ctx.out(f"{folder}/W.npy"), w)
        np.save(ctx.out(f"{folder}/proj_item.npy"), proj)
        np.save(ctx.out(f"{folder}/user.npy"), lightgcn.user)
        np.save(ctx.out(f"{folder}/item_degree.npy"), lightgcn.degree)
        meta = {
            "model": name,
            "mode": "val",
            "cutoff": data.cutoff,
            "params": dict(params),
            "shuffled_content": shuffled,
            "normalize_items": bool(params["normalize_items"]),
            "lightgcn": {k: lightgcn.meta[k] for k in ("model", "cutoff", "best_epoch")},
            "epoch_bpr": losses,
            "w_frobenius": float(np.linalg.norm(w)),  # 0 means W never moved from zero init
            "embedding_health": embedding_health(proj, lightgcn.degree > 0),
            "device": device,
            "torch": torch.__version__,
            "seconds": round(time.monotonic() - started, 1),
        }
        ctx.out(f"{folder}/meta.json").write_text(json.dumps(meta, indent=2) + "\n", "utf-8")
        ctx.log.info("%s: final bpr %.4f in %.0f s", name, losses[-1], meta["seconds"])
        ctx.notes[f"{name}_final_bpr"] = losses[-1]
