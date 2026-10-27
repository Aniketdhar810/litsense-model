"""Logging setup shared by stages and scripts."""

import logging
import time

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
PROGRESS_EVERY_S = 60.0  # long streams print a status line about once a minute


def get_logger(name: str) -> logging.Logger:
    """Return a child of the ``litsense`` logger, attaching one stderr handler on first use."""
    root = logging.getLogger("litsense")
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    return root.getChild(name)


class Progress:
    """Logs a one-line status at most every ``every_s`` seconds while a long stream runs, so a
    multi-minute Colab cell visibly advances."""

    def __init__(self, log: logging.Logger, label: str, every_s: float = PROGRESS_EVERY_S) -> None:
        self.log = log
        self.label = label
        self.every_s = every_s
        self._start = self._last = time.monotonic()

    def update(self, done_bytes: int, total_bytes: int | None, detail: str = "") -> None:
        """Log progress if ``every_s`` has passed since the last line."""
        now = time.monotonic()
        if now - self._last < self.every_s:
            return
        self._last = now
        rate = done_bytes / max(now - self._start, 1e-9)
        if total_bytes:
            eta_min = (total_bytes - done_bytes) / max(rate, 1.0) / 60
            share = done_bytes / total_bytes
            where = f"{done_bytes / 1e9:.2f}/{total_bytes / 1e9:.2f} GB ({share:.1%})"
            tail = f", about {eta_min:.0f} min left"
        else:
            where, tail = f"{done_bytes / 1e9:.2f} GB", ""
        self.log.info("%s: %s at %.1f MB/s%s%s", self.label, where, rate / 1e6, tail, detail)
