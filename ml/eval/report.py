"""Tables and figures from evaluation results (CLAUDE.md s8.7). Every number comes from the
per-user rows the evaluator wrote; nothing is typed by hand."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ml.eval import bootstrap  # noqa: E402

DPI = 150
HEADLINE = ("ndcg@10", "recall@20", "precision@10", "mrr@10", "hit@10")


def main_table(
    rows: Mapping[str, pd.DataFrame], beyond: Mapping[str, Mapping[str, float]], n: int, seed: int
) -> pd.DataFrame:
    """One line per model: headline means with 95% CIs, users, and beyond-accuracy metrics."""
    lines = []
    for name, frame in rows.items():
        line: dict[str, object] = {"model": name, "users": len(frame)}
        for metric in (m for m in HEADLINE if m in frame):
            ci = bootstrap.mean_ci(frame[metric].to_numpy(), n, np.random.default_rng(seed))
            line |= {metric: ci.mean, f"{metric}_low": ci.low, f"{metric}_high": ci.high}
        line |= dict(beyond[name])
        lines.append(line)
    return pd.DataFrame(lines)


def paired_table(
    rows: Mapping[str, pd.DataFrame],
    reference: str,
    metrics: Sequence[str],
    n: int,
    seed: int,
    label: str = "all",
) -> pd.DataFrame:
    """Every model vs ``reference`` on the users both evaluated (paired bootstrap)."""
    lines = []
    ref = rows[reference].set_index("u")
    for name, frame in rows.items():
        if name == reference:
            continue
        both = frame.set_index("u").join(ref, rsuffix="_ref", how="inner")
        for metric in (m for m in metrics if m in both):
            d = bootstrap.paired(
                both[metric].to_numpy(), both[f"{metric}_ref"].to_numpy(), n,
                np.random.default_rng(seed),
            )  # fmt: skip
            lines.append(
                {"stratum": label, "model": name, "vs": reference, "metric": metric,
                 "diff": d.mean, "low": d.low, "high": d.high,
                 "share_le_zero": d.share_le_zero, "users": d.users}
            )  # fmt: skip
    return pd.DataFrame(lines)


def plot_bars(
    table: pd.DataFrame, path: Path, metrics: Sequence[str] = ("ndcg@10", "recall@20")
) -> None:
    """Grouped bars with 95% CIs for the headline metrics."""
    metrics = [m for m in metrics if m in table]
    x = np.arange(len(table))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(max(5, len(table) * 1.2), 3.5))
    for j, metric in enumerate(metrics):
        mean = table[metric].to_numpy()
        err = [mean - table[f"{metric}_low"].to_numpy(), table[f"{metric}_high"].to_numpy() - mean]
        ax.bar(x + j * width, mean, width, yerr=err, capsize=3, label=metric)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2, table["model"])
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_by_bucket(strata: pd.DataFrame, path: Path, metric: str = "ndcg@10") -> None:
    """One line per model across user-sparsity buckets."""
    if metric not in strata:
        return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for name, group in strata.groupby("model", sort=False):
        ax.plot(group["bucket"], group[metric], marker="o", label=name)
    ax.set_xlabel("fit interactions per user")
    ax.set_ylabel(metric)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
