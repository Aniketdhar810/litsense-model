"""Checkpointed array parts for long GPU passes (the D37 rule: checkpoints validate their parts).

A pass over ``n_rows`` inputs is cut into chunks of ``chunk_rows``; each finished chunk is saved as
its own ``.npy`` in the stage's work directory (written to a temporary name, then renamed). On
resume, a part counts as done only if it loads and has the expected number of rows, so a file
truncated by a lost Colab session is redone instead of trusted. ``concat`` writes every part into
the final array in order.
"""

import os
from pathlib import Path

import numpy as np


class PartStore:
    """Parts of one pass: ``part_0000.npy`` ... under ``directory`` with the given prefix."""

    def __init__(
        self,
        directory: Path,
        prefix: str,
        n_rows: int,
        chunk_rows: int,
        *,
        fixed_rows: bool = True,
    ) -> None:
        """``fixed_rows``: every part holds exactly one output row per input row (embeddings), so
        its length is checked; False for parts whose length depends on the data (long tables)."""
        if chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")
        self.directory = Path(directory)
        self.prefix = prefix
        self.n_rows = int(n_rows)
        self.chunk_rows = int(chunk_rows)
        self.fixed_rows = fixed_rows
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def n_parts(self) -> int:
        return -(-self.n_rows // self.chunk_rows)

    def bounds(self, k: int) -> tuple[int, int]:
        """Input rows ``[start, stop)`` covered by part ``k``."""
        start = k * self.chunk_rows
        return start, min(start + self.chunk_rows, self.n_rows)

    def path(self, k: int) -> Path:
        return self.directory / f"{self.prefix}_{k:04d}.npy"

    def is_done(self, k: int) -> bool:
        """True if part ``k`` exists and holds the rows it should; a bad file is removed."""
        path = self.path(k)
        if not path.exists():
            return False
        start, stop = self.bounds(k)
        try:
            rows = np.load(path, mmap_mode="r").shape[0]
        except (OSError, ValueError):
            rows = -1
        if rows < 0 or (self.fixed_rows and rows != stop - start):
            path.unlink()
            return False
        return True

    def first_pending(self) -> int:
        """Index of the first part not yet done (``n_parts`` when everything is)."""
        return next((k for k in range(self.n_parts) if not self.is_done(k)), self.n_parts)

    def save(self, k: int, array: np.ndarray) -> None:
        """Write part ``k`` atomically; its row count must match ``bounds(k)``."""
        start, stop = self.bounds(k)
        if self.fixed_rows and array.shape[0] != stop - start:
            raise ValueError(f"part {k} has {array.shape[0]} rows, expected {stop - start}")
        tmp = self.path(k).with_suffix(".tmp.npy")
        np.save(tmp, array)
        os.replace(tmp, self.path(k))

    def load(self, k: int) -> np.ndarray:
        return np.load(self.path(k))

    def concat(self, out: np.ndarray) -> None:
        """Copy every part into ``out`` (shape ``(n_rows, ...)``, for example an open memmap)."""
        if not self.fixed_rows:
            raise ValueError("concat needs fixed-size parts")
        if out.shape[0] != self.n_rows:
            raise ValueError(f"target has {out.shape[0]} rows, expected {self.n_rows}")
        for k in range(self.n_parts):
            if not self.is_done(k):
                raise ValueError(f"part {k} is missing or incomplete")
            start, stop = self.bounds(k)
            out[start:stop] = self.load(k)
