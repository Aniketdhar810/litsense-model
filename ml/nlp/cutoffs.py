"""Cutoffs for review-derived artifacts (CLAUDE.md s3 rule 2, s11).

Everything built from review text exists once per artifact set: ``eval_val`` (rows before t1),
``eval_test`` (before t2) and ``serve`` (every row; its cutoff is one millisecond after the newest
row so the manifest still records a number). Builders take the cutoff and raise ``LeakageError``
when handed a newer row, so a wrong filter upstream fails instead of leaking.
"""

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd

from ml.eval.errors import LeakageError

ArtifactSet = Literal["eval_val", "eval_test", "serve"]
ARTIFACT_SETS: tuple[ArtifactSet, ...] = ("eval_val", "eval_test", "serve")
SPLIT_KEY: dict[str, str] = {"eval_val": "t1", "eval_test": "t2"}


def cutoff_for(artifact_set: ArtifactSet, splits: Mapping[str, Any], max_ts: int) -> int:
    """The cutoff (ms) of an artifact set: t1 or t2 from ``splits.json``, or ``max_ts + 1`` for
    ``serve`` so that every row is older than it."""
    if artifact_set == "serve":
        return int(max_ts) + 1
    return int(splits[SPLIT_KEY[artifact_set]]["ms"])


def rows_before(frame: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """Rows whose ``ts`` is strictly before ``cutoff``."""
    return frame[frame["ts"].to_numpy() < cutoff]


def check_cutoff(ts: Any, cutoff: int, what: str = "rows") -> None:
    """Raise ``LeakageError`` if any timestamp in ``ts`` is at or after ``cutoff``."""
    values = np.asarray(ts, dtype=np.int64)
    if values.size and int(values.max()) >= cutoff:
        raise LeakageError(f"{what} include timestamps at or after the cutoff {cutoff}")
