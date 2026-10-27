"""Evidence snippets (CLAUDE.md s11.9), stage nlp.snippets, serve cutoff only.

Per item and aspect: up to ``aspects.snippets.positive`` sentences with polarity above
``nlp.pos_threshold`` and ``aspects.snippets.negative`` below ``nlp.neg_threshold``, between
``min_chars`` and ``max_chars`` characters, ranked by helpful votes then |polarity|, one copy of a
text per item. Sentences with ``spoiler_flag`` (``nlp.spoiler_flags``: the classifier when it is
trained, else the regex rule) are dropped. These are the only review sentences the product ever
shows.
"""

import gzip
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ml.common.io import load_id_map
from ml.common.stage import StageContext, stage
from ml.nlp.aspect_seeds import ASPECTS
from ml.nlp.spoiler_train import FLAGS

FOLDER = "serve/snippets"
FILES = ("evidence_snippets.json.gz", "meta.json")


def select_snippets(
    mentions: pd.DataFrame,
    params: Mapping[str, Any],
    pos_threshold: float,
    neg_threshold: float,
) -> tuple[dict[str, dict[str, dict[str, list[dict[str, Any]]]]], dict[str, int]]:
    """``{item: {aspect: {"pos": [...], "neg": [...]}}}`` from (sentence, aspect) mentions with
    ``i``, ``aspect``, ``polarity``, ``text``, ``review_row``, ``helpful_vote`` and
    ``spoiler_flag`` columns."""
    lo, hi = int(params["min_chars"]), int(params["max_chars"])
    n_pos, n_neg = int(params["positive"]), int(params["negative"])
    counts = {"mentions": int(len(mentions))}
    length = mentions["text"].str.len()
    kept = mentions[(length >= lo) & (length <= hi)]
    counts["dropped_length"] = counts["mentions"] - len(kept)
    is_spoiler = kept["spoiler_flag"].to_numpy(bool)
    kept = kept[~is_spoiler]
    counts["dropped_spoiler"] = int(is_spoiler.sum())
    sign = np.where(
        kept["polarity"] > pos_threshold, 1, np.where(kept["polarity"] < neg_threshold, -1, 0)
    )
    kept = kept.assign(sign=sign)
    kept = kept[kept["sign"] != 0]
    counts["dropped_neutral"] = (
        counts["mentions"] - counts["dropped_length"] - counts["dropped_spoiler"] - len(kept)
    )
    kept = kept.assign(abs_pol=kept["polarity"].abs())
    kept = kept.sort_values(
        ["i", "aspect", "sign", "helpful_vote", "abs_pol"],
        ascending=[True, True, True, False, False],
        kind="stable",
    ).drop_duplicates(["i", "aspect", "sign", "text"])
    kept = kept.assign(rank=kept.groupby(["i", "aspect", "sign"], sort=False).cumcount())
    limit = np.where(kept["sign"] > 0, n_pos, n_neg)
    kept = kept[kept["rank"].to_numpy() < limit]

    out: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    for row in kept.itertuples(index=False):
        aspect = ASPECTS[int(row.aspect)]
        slot = out.setdefault(str(int(row.i)), {}).setdefault(aspect, {"pos": [], "neg": []})
        slot["pos" if row.sign > 0 else "neg"].append(
            {
                "text": str(row.text),
                "review_row": int(row.review_row),
                "helpful_votes": int(row.helpful_vote),
                "polarity": round(float(row.polarity), 4),
            }
        )
    counts["snippets"] = int(len(kept))
    counts["items_with_snippets"] = len(out)
    return out, counts


@stage(
    "nlp.snippets",
    inputs=["data:sentences.parquet", "data:aspect_polarity.parquet", FLAGS, "data:id_map.json"],
    outputs=[f"artifacts:{FOLDER}/{name}" for name in FILES],
    config_keys=["aspects.names", "aspects.snippets", "nlp.pos_threshold", "nlp.neg_threshold"],
)
def build_snippets(ctx: StageContext) -> None:
    """Spoiler-safe evidence sentences per item and aspect (serve set)."""
    if tuple(ctx.param("aspects.names")) != ASPECTS:
        raise ValueError("aspects.names must equal aspect_seeds.ASPECTS, in order")
    ids = load_id_map(ctx.path("data:id_map.json"))
    sentences = pd.read_parquet(
        ctx.path("data:sentences.parquet"),
        columns=["sent_id", "review_row", "i", "ts", "helpful_vote", "text"],
    )
    if len(sentences) and int(sentences["i"].max()) >= ids.n_items:
        raise ValueError("sentence item ids exceed the id map")
    flags = pd.read_parquet(ctx.path(FLAGS), columns=["sent_id", "spoiler_flag"])
    polarity = pd.read_parquet(ctx.path("data:aspect_polarity.parquet"))
    mentions = polarity.merge(sentences, on="sent_id", how="inner").merge(
        flags, on="sent_id", how="left", validate="many_to_one"
    )
    if mentions["spoiler_flag"].isna().any():
        raise ValueError("sentence_spoiler.parquet does not cover every sentence")
    cutoff = int(sentences["ts"].max()) + 1 if len(sentences) else 0
    ctx.cutoff = cutoff
    snippets, counts = select_snippets(
        mentions,
        ctx.param("aspects.snippets"),
        float(ctx.param("nlp.pos_threshold")),
        float(ctx.param("nlp.neg_threshold")),
    )
    with gzip.open(
        ctx.out(f"artifacts:{FOLDER}/evidence_snippets.json.gz"), "wt", encoding="utf-8"
    ) as f:
        json.dump(snippets, f, ensure_ascii=False)
    meta = {
        "artifact_set": "serve",
        "cutoff": cutoff,
        "params": dict(ctx.param("aspects.snippets")),
        **counts,
    }
    ctx.out(f"artifacts:{FOLDER}/meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    for key, value in counts.items():
        ctx.count(key, value)
