"""Aspect annotation export (CLAUDE.md s11.5 step 1), stage nlp.annotation_export.

A seeded sample of sentences for the user and the teammate to label: ``aspects.annotation_sample``
gives the total and how many of them have at least one predicted aspect hit (the rest have none).
The CSV has ``sent_id``, ``text`` and one blank column per aspect (labels: ``+``, ``-``, ``0`` for
mentioned-neutral, blank for absent). Predictions are deliberately not in the file: labellers must
not see them. ``nlp.validate_aspects`` (s11.5 steps 4-5) later joins the labelled file back to
``sentence_aspects.parquet`` by ``sent_id``.

The repo copy lives at ``docs/annotation/aspects_300.csv``: the user copies the stage output
there and commits it.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ml.common.stage import StageContext, stage
from ml.nlp.aspect_seeds import ASPECTS

LABEL_COLUMNS: tuple[str, ...] = ("sent_id", "text", *ASPECTS)


def sample_for_annotation(
    sentences: pd.DataFrame,
    aspect_masks: pd.DataFrame,
    total: int,
    with_hit: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """``total`` sentences, ``with_hit`` of them with a predicted hit, shuffled, blank label
    columns. Raises ``ValueError`` when either pool is too small."""
    if not 0 <= with_hit <= total:
        raise ValueError("annotation_sample.with_hit must be between 0 and total")
    frame = sentences[["sent_id", "text"]].merge(
        aspect_masks[["sent_id", "aspect_mask"]], on="sent_id", validate="one_to_one"
    )
    hit = frame[frame["aspect_mask"] > 0]
    miss = frame[frame["aspect_mask"] == 0]
    if len(hit) < with_hit or len(miss) < total - with_hit:
        raise ValueError(
            f"need {with_hit} sentences with a hit and {total - with_hit} without; "
            f"have {len(hit)} and {len(miss)}"
        )
    picked = pd.concat(
        [
            hit.iloc[rng.choice(len(hit), size=with_hit, replace=False)],
            miss.iloc[rng.choice(len(miss), size=total - with_hit, replace=False)],
        ]
    )
    picked = picked.iloc[rng.permutation(len(picked))].reset_index(drop=True)
    out = picked[["sent_id", "text"]].copy()
    for name in ASPECTS:
        out[name] = ""
    return out[list(LABEL_COLUMNS)]


@stage(
    "nlp.annotation_export",
    inputs=["data:sentences.parquet", "data:sentence_aspects.parquet"],
    outputs=["results:nlp/annotation/aspects_300.csv"],
    config_keys=["aspects.annotation_sample"],
)
def export_annotation(ctx: StageContext) -> None:
    """Write the blank annotation CSV (seeded sample)."""
    params: Mapping[str, Any] = ctx.param("aspects.annotation_sample")
    sentences = pd.read_parquet(ctx.path("data:sentences.parquet"), columns=["sent_id", "text"])
    masks = pd.read_parquet(
        ctx.path("data:sentence_aspects.parquet"), columns=["sent_id", "aspect_mask"]
    )
    sample = sample_for_annotation(
        sentences, masks, int(params["total"]), int(params["with_hit"]), ctx.rng
    )
    sample.to_csv(
        ctx.out("results:nlp/annotation/aspects_300.csv"), index=False, lineterminator="\n"
    )
    ctx.count("sentences_exported", len(sample))
    ctx.count("sentences_with_hit", int(params["with_hit"]))
