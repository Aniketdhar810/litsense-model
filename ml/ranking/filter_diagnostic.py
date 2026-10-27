"""Which s13.2 filter removes the positives the candidate sources found, stage
``ranking.filter_diagnostic`` (D67 follow-up).

On v0 the seven sources reach 28.8% of the val positives but only 15.9% survive the filters. For
every val user this stage takes the positives inside the candidate union and assigns each lost
one its first reason in filter order (``next_in_series``, ``thin`` = fewer than
``min_reviews_main_feed`` text reviews before the cutoff and not from the longtail source,
``series_guard_skipped`` / ``series_guard_unknown`` = the s13.2 guard with the reader in the series
or with no volume of it in their history, ``future`` = published after the cutoff year,
``muted``), then recomputes the
union-after-filters recall for ``min_reviews_main_feed`` in 0..3. It also tabulates how many text
reviews the val positives had at t1, which is what the filter looks at. Output:
``results/<v>/ranker/filter_diagnostic.csv`` (recall per min_reviews value) and
``filter_diagnostic.json`` (loss reasons and the review-count histogram). The serving filter is a
product rule (items need reviews for explanations), so the decision stays with the user.
"""

import json
from typing import Any

import numpy as np
import pandas as pd

from litsense_core import candidates as cand
from litsense_core.profile import build_profile
from litsense_core.store import read_store
from ml.common.io import load_id_map
from ml.common.stage import StageContext, stage
from ml.eval.data import build_eval_data
from ml.ranking.build_training import REVIEW_ASPECTS, cutoff_year, user_state
from ml.serve.export import VAL_BUNDLE_DIR

CSV = "results:ranker/filter_diagnostic.csv"
JSON = "results:ranker/filter_diagnostic.json"
MIN_REVIEWS_GRID = (0, 1, 2, 3, 5)
REASONS = (
    "next_in_series",
    "thin",
    "series_guard_skipped",  # the reader is in the series but has not reached volume p - 1
    "series_guard_unknown",  # no volume of that series in the reader's history at all
    "future",
    "muted",
    "kept",
)


def loss_reason(
    i: int, provenance: dict[int, list[str]], profile: Any, store: Any, min_reviews: int,
    max_year: int, nxt: set[int],
) -> str:  # fmt: skip
    """First filter (in ``apply_filters`` order, then the D67 year rule) that drops item ``i``."""
    cat = store.catalog
    if i in nxt:
        return "next_in_series"
    if set(cat.genres_of(i)) & profile.muted_genres:
        return "muted"
    if int(cat.n_reviews[i]) < min_reviews and "longtail" not in provenance.get(i, []):
        return "thin"
    if not cand.series_allows(store, i, profile):
        in_series = int(cat.series_id[i]) in profile.series_progress
        return "series_guard_skipped" if in_series else "series_guard_unknown"
    if int(cat.year[i]) > max_year:
        return "future"
    return "kept"


@stage(
    "ranking.filter_diagnostic",
    inputs=[
        VAL_BUNDLE_DIR,
        REVIEW_ASPECTS,
        "data:interactions.parquet",
        "data:splits.json",
        "data:id_map.json",
    ],
    outputs=[CSV, JSON],
    config_keys=[
        "data.relevance_min_rating",
        "data.longtail_fraction",
        "ranking.min_reviews_main_feed",
        "aspects.names",
    ],
)
def filter_diagnostic(ctx: StageContext) -> None:
    """Loss reasons for the positives the sources found, and recall per min_reviews value."""
    ids = load_id_map(ctx.path("data:id_map.json"))
    splits = json.loads(ctx.path("data:splits.json").read_text(encoding="utf-8"))
    inter = pd.read_parquet(ctx.path("data:interactions.parquet"))
    data = build_eval_data(
        inter, splits, "val", ids.n_users, ids.n_items,
        float(ctx.param("data.relevance_min_rating")), float(ctx.param("data.longtail_fraction")),
    )  # fmt: skip
    ctx.cutoff = data.cutoff
    store = read_store(ctx.path(VAL_BUNDLE_DIR))
    cat = store.catalog
    max_year = cutoff_year(data.cutoff)
    current = int(ctx.param("ranking.min_reviews_main_feed"))
    train = inter.loc[inter["split"] == "train", ["u", "i", "rating", "ts"]]
    train_by_user = {int(u): g for u, g in train.groupby("u", sort=False)}
    reviews = pd.read_parquet(ctx.path(REVIEW_ASPECTS))
    reviews_by_user = {int(u): g for u, g in reviews.groupby("u", sort=False)}
    names = list(ctx.param("aspects.names"))

    reasons = dict.fromkeys(REASONS, 0)
    recall_sum = dict.fromkeys(MIN_REVIEWS_GRID, 0.0)
    users_with_positive = dict.fromkeys(MIN_REVIEWS_GRID, 0)
    union_sum = 0.0
    n_truth_total = 0
    truth_reviews: list[int] = []
    for u in data.users.tolist():
        truth = data.truth[u]
        truth_reviews.extend(int(cat.n_reviews[i]) for i in truth.tolist())
        n_truth_total += len(truth)
        state = user_state(u, ids.users[u], train_by_user[u], reviews_by_user.get(u), names)
        profile = build_profile(state, store)
        sources = cand.generate(profile, store, store.config["quotas"][profile.bucket])
        union, provenance = cand.union(sources, int(store.config["candidates_total"]))
        found = [i for i in union if i in set(truth.tolist())]
        union_sum += len(found) / len(truth)
        nxt = set(cand.next_in_series(store, profile))
        for i in found:
            reasons[loss_reason(i, provenance, profile, store, current, max_year, nxt)] += 1
        for m in MIN_REVIEWS_GRID:
            kept = cand.apply_filters(union, provenance, profile, store, None, None, m)
            kept = [i for i in kept if int(cat.year[i]) <= max_year]
            hits = len(set(kept) & set(truth.tolist()))
            recall_sum[m] += hits / len(truth)
            users_with_positive[m] += int(hits > 0)

    n_users = len(data.users)
    table = pd.DataFrame(
        {
            "min_reviews_main_feed": list(MIN_REVIEWS_GRID),
            "recall_after_filters": [recall_sum[m] / n_users for m in MIN_REVIEWS_GRID],
            "users_with_positive": [users_with_positive[m] for m in MIN_REVIEWS_GRID],
            "current": [m == current for m in MIN_REVIEWS_GRID],
        }
    )
    table.to_csv(ctx.out(CSV), index=False)
    counts = np.bincount(np.clip(np.asarray(truth_reviews, dtype=np.int64), 0, 10), minlength=11)
    summary = {
        "cutoff": data.cutoff,
        "users": n_users,
        "val_positives": n_truth_total,
        "union_recall": union_sum / n_users,
        "positives_in_union": int(sum(reasons.values())),
        "loss_reasons_at_current_filter": reasons,
        "val_positives_by_text_reviews_at_t1": {
            (str(k) if k < 10 else "10+"): int(c) for k, c in enumerate(counts.tolist())
        },
        "catalogue_items_with_fewer_than_current": int((cat.n_reviews < current).sum()),
        "catalogue_items": int(cat.n_items),
    }
    ctx.out(JSON).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    ctx.count("positives_in_union", summary["positives_in_union"])
    for key, value in reasons.items():
        ctx.count(f"lost_{key}" if key != "kept" else "kept", value)
    ctx.notes["recall_after_filters"] = {
        str(m): round(recall_sum[m] / n_users, 4) for m in MIN_REVIEWS_GRID
    }
