"""k-core filtering and seeded user sampling to a target size (CLAUDE.md s7.7 steps 1-2)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.common.errors import DataError

SIZING_STEPS = 40  # bisection steps over the user fraction


def kcore(df: pd.DataFrame, ku: int, ki: int) -> pd.DataFrame:
    """Largest subset where every user has >= ``ku`` and every item >= ``ki`` interactions."""
    while True:
        n = len(df)
        df = df[df.groupby("user_id")["canonical_id"].transform("size") >= ku]
        df = df[df.groupby("canonical_id")["user_id"].transform("size") >= ki]
        if len(df) == n:
            return df.reset_index(drop=True)


def check_kcore(df: pd.DataFrame, ku: int, ki: int) -> None:
    """Raise DataError unless the k-core invariants hold."""
    if len(df) and (
        df.groupby("user_id").size().min() < ku or df.groupby("canonical_id").size().min() < ki
    ):
        raise DataError(f"k-core invariants violated (users >= {ku}, items >= {ki})")


@dataclass(frozen=True)
class Sizing:
    """Outcome of ``size_to_target``: the user fraction used and the resulting size."""

    fraction: float
    interactions: int
    steps: int


def size_to_target(
    df: pd.DataFrame,
    ku: int,
    ki: int,
    target: int,
    tolerance: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, Sizing]:
    """Sample users (seeded) and re-run the k-core until the size is within ``target`` +/-
    ``tolerance``. Every user draws one uniform number once; a fraction f keeps the users whose
    draw is below f, so a larger f keeps a superset of users. f is found by bisection."""
    if len(df) <= target * (1 + tolerance):
        return df, Sizing(1.0, len(df), 0)
    users = np.sort(df["user_id"].unique())
    draw = pd.Series(rng.random(len(users)), index=users)
    user_draw = draw.reindex(df["user_id"]).to_numpy()
    low, high = 0.0, 1.0
    best: tuple[pd.DataFrame, float] | None = None
    for step in range(1, SIZING_STEPS + 1):
        f = (low + high) / 2
        sized = kcore(df[user_draw < f], ku, ki)
        if abs(len(sized) - target) <= target * tolerance:
            return sized, Sizing(f, len(sized), step)
        if best is None or abs(len(sized) - target) < abs(len(best[0]) - target):
            best = (sized, f)
        low, high = (f, high) if len(sized) < target else (low, f)
    sized, f = best
    raise DataError(
        f"could not size to {target} +/- {tolerance:.0%}: closest {len(sized)} at f={f:.5f}"
    )
