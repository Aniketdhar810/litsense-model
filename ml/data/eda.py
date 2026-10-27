"""Stage data.eda (CLAUDE.md s7.8): dataset statistics and figures for the report.

Writes ``results/<v>/eda/summary.json``, ``attrition.csv`` and PNG figures (150 dpi). Every number
comes from the pipeline's tables and manifests; nothing is typed by hand. The genre-slice mix is
reported at each step (works kept, k-core, sizing) to show how much SF & Fantasy survives (user
request, Week 2).
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ml.common.stage import StageContext, stage  # noqa: E402

DPI = 150


def attrition(
    notes: Mapping[str, Mapping[str, Any]],
    canonical: Mapping[str, Any],
    splits: Mapping[str, Any],
    final: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Users, items and interactions after each pipeline step (None where a step does not track
    that quantity)."""
    dl, meta = notes["data.download"]["counts"], notes["data.meta"]["counts"]
    status, steps = canonical["status"], splits["attrition"]
    all_works = sum(v["interactions"] for v in status.values())
    boxed = status.get("boxset", {"works": 0, "interactions": 0})
    rows = [
        ("5-core", dl["users"], dl["items"], dl["ratings"]),
        ("metadata found", None, meta["meta_matched"], None),
        ("non-narrative removed", None, meta["books_raw"],
         sum(notes["data.meta"]["edition_kind_interactions"].values())),
        ("box sets removed", None, canonical["editions"] - canonical["boxsets"],
         all_works - boxed["interactions"]),
        ("editions merged into works", None, canonical["works"] - boxed["works"],
         all_works - boxed["interactions"]),
        ("genre slices (any edition)", None, canonical["kept"]["works"],
         canonical["kept"]["interactions"]),
        ("one rating per (user, work)", None, None, steps["interactions_one_per_work"]),
        ("k-core", None, None, steps["after_kcore"]),
        ("sized to target", final["users"], final["items"], steps["after_sizing"]),
    ]  # fmt: skip
    return [{"step": s, "users": u, "items": i, "interactions": n} for s, u, i, n in rows]


def save(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_degrees(inter: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, key, label in ((axes[0], "u", "user"), (axes[1], "i", "item")):
        counts = inter.groupby(key).size().value_counts().sort_index()
        ax.loglog(counts.index, counts.to_numpy(), "o", markersize=3)
        ax.set_xlabel(f"interactions per {label}")
        ax.set_ylabel(f"{label}s")
    save(fig, path)


def longtail_curve(inter: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Share of items (most popular first) vs cumulative share of interactions."""
    counts = np.sort(inter.groupby("i").size().to_numpy())[::-1]
    return np.arange(1, len(counts) + 1) / len(counts), np.cumsum(counts) / counts.sum()


def plot_longtail(inter: pd.DataFrame, path: Path) -> None:
    x, y = longtail_curve(inter)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(x, y)
    ax.set_xlabel("share of items (most popular first)")
    ax.set_ylabel("share of interactions")
    save(fig, path)


def plot_ratings(inter: pd.DataFrame, path: Path) -> None:
    shares = pd.crosstab(inter["rating"], inter["split"], normalize="columns")
    shares["all"] = inter["rating"].value_counts(normalize=True).sort_index()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    shares.reindex(columns=[c for c in ("all", "train", "val", "test") if c in shares]).plot.bar(
        ax=ax
    )
    ax.set_xlabel("stars")
    ax.set_ylabel("share of ratings")
    save(fig, path)


def plot_review_lengths(reviews: pd.DataFrame, path: Path) -> None:
    lengths = reviews["text"].str.len()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(lengths[lengths > 0], bins=np.logspace(0, np.log10(max(lengths.max(), 10)), 50))
    ax.set_xscale("log")
    ax.set_xlabel("review length (characters)")
    ax.set_ylabel("reviews")
    save(fig, path)


def plot_timeline(inter: pd.DataFrame, splits: Mapping[str, Any], path: Path) -> None:
    months = pd.to_datetime(inter["ts"], unit="ms").dt.to_period("M").value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(months.index.to_timestamp(), months.to_numpy())
    for key in ("t1", "t2"):
        ax.axvline(pd.to_datetime(splits[key]["ms"], unit="ms"), color="k", linestyle="--")
    ax.set_ylabel("interactions per month")
    save(fig, path)


def plot_barh(values: Mapping[str, int], xlabel: str, path: Path, top: int = 20) -> None:
    items = sorted(values.items(), key=lambda kv: kv[1])[-top:]
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(items) + 1))
    ax.barh([k for k, _ in items], [v for _, v in items])
    ax.set_xlabel(xlabel)
    save(fig, path)


def plot_slice_mix(mix: Mapping[str, Mapping[str, int]], path: Path) -> None:
    frame = pd.DataFrame(mix).fillna(0)
    shares = frame / frame.sum()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    shares.T.plot.bar(stacked=True, ax=ax)
    ax.set_ylabel("share of interactions")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    save(fig, path)


@stage(
    "data.eda",
    inputs=[
        "data:ratings_5core.csv.gz",
        "data:books_raw.parquet",
        "results:canonical/canonical_report.json",
        "data:reviews_clean.parquet",
        "data:interactions.parquet",
        "data:books.parquet",
        "data:reviews.parquet",
        "data:splits.json",
    ],
    outputs=["results:eda/"],
    config_keys=["data.relevance_min_rating"],
)
def build_eda(ctx: StageContext) -> None:
    """Dataset statistics, the attrition table and the s7.8 figures."""
    out = ctx.out("results:eda/")
    canonical = json.loads(ctx.path("results:canonical/canonical_report.json").read_text("utf-8"))
    splits = json.loads(ctx.path("data:splits.json").read_text("utf-8"))
    notes = {s: ctx.upstream_notes(s) for s in ("data.download", "data.meta", "data.clean")}
    inter = pd.read_parquet(ctx.path("data:interactions.parquet"))
    books = pd.read_parquet(ctx.path("data:books.parquet"))
    reviews = pd.read_parquet(
        ctx.path("data:reviews.parquet"), columns=["text", "text_ok", "text_drop_reason"]
    )
    raw = pd.read_parquet(ctx.path("data:books_raw.parquet"), columns=["parent_asin", "isbn10"])
    final = {"users": int(inter["u"].nunique()), "items": int(inter["i"].nunique()),
             "interactions": len(inter)}  # fmt: skip

    rows = attrition(notes, canonical, splits, final)
    pd.DataFrame(rows).to_csv(out / "attrition.csv", index=False)
    relevant = float(ctx.param("data.relevance_min_rating"))
    authors = books["author_key"].value_counts()
    x, y = longtail_curve(inter)
    summary = {
        "dataset": final
        | {
            "density": round(len(inter) / (final["users"] * final["items"]), 6),
            "interactions_with_text": int(inter["has_text"].sum()),
        },
        "attrition": rows,
        "ratings": {
            "share_relevant_all": round(float((inter["rating"] >= relevant).mean()), 4),
            "share_relevant_by_split": {
                s: round(float((g["rating"] >= relevant).mean()), 4)
                for s, g in inter.groupby("split")
            },
        },
        "longtail": {"interaction_share_of_top_20pct_items": round(float(np.interp(0.2, x, y)), 4)},
        "reviews": {
            "rows": len(reviews),
            "text_ok_share": round(float(reviews["text_ok"].mean()), 4) if len(reviews) else None,
            "drop_reasons": reviews["text_drop_reason"]
            .value_counts(normalize=True)
            .round(4)
            .to_dict(),
            "clean_stage_counts": notes["data.clean"]["counts"],
        },
        "catalogue": {
            "primary_genre_items": {
                k: int(v) for k, v in books["primary_genre"].value_counts().items()
            },
            "members_per_work": {
                str(int(k)): int(v)
                for k, v in books["n_members"].value_counts().sort_index().items()
            },
            "goodreads_match_rate": canonical["goodreads"]["edition_match_rate"],
            "parent_asin_is_isbn10_share": round(
                float((raw["parent_asin"] == raw["isbn10"]).mean()), 4
            ),
            "authors_with_2plus_books": int((authors >= 2).sum()),
            "items_by_authors_with_2plus_books": round(
                float(authors[authors >= 2].sum() / max(len(books), 1)), 4
            ),
        },
        "splits": splits["splits"],
        "slice_mix": {
            "works_kept": {
                k: v["interactions"] for k, v in canonical["kept"]["by_primary_slice"].items()
            },
            "after_kcore": {
                k: v["interactions"]
                for k, v in splits["mix"]["after_kcore"]["primary_slice"].items()
            },
            "after_sizing": {
                k: v["interactions"]
                for k, v in splits["mix"]["after_sizing"]["primary_slice"].items()
            },
        },
        "genre_mix_after_sizing": {
            k: v["interactions"] for k, v in splits["mix"]["after_sizing"]["primary_genre"].items()
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    plot_degrees(inter, out / "degree_loglog.png")
    plot_longtail(inter, out / "longtail_curve.png")
    plot_ratings(inter, out / "ratings_by_split.png")
    plot_review_lengths(reviews, out / "review_length.png")
    plot_timeline(inter, splits, out / "timeline.png")
    plot_barh(summary["genre_mix_after_sizing"], "interactions", out / "genres.png")
    plot_barh({str(k): int(v) for k, v in summary["catalogue"]["members_per_work"].items()},
              "works", out / "members_per_work.png")  # fmt: skip
    plot_slice_mix(summary["slice_mix"], out / "slice_mix.png")
    for key, value in final.items():
        ctx.count(key, value)
