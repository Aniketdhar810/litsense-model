"""Request budget (CLAUDE.md s14.2): a daily counter and a requests-per-minute token bucket.

The daily counter lives in the cache's sqlite file (or in a callback the backend provides, so a
multi-process deployment can count centrally). The limiter never sleeps on the online path: when
no token is free the client reports ``fallback_reason="rate_limited"`` and the template answers.
Offline batch work (review synthesis) asks for a blocking acquire instead.
"""

import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime

from litsense_core.llm.cache import SqliteCache


class DailyBudget:
    """``take()`` consumes one request of today's budget; False once it is spent."""

    def __init__(
        self,
        limit: int,
        cache: SqliteCache | None = None,
        take_callback: Callable[[], bool] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.limit = int(limit)
        self.cache = cache
        self.take_callback = take_callback
        self.clock = clock
        self._lock = threading.Lock()
        self._memory: dict[str, int] = {}

    def day(self) -> str:
        return datetime.fromtimestamp(self.clock(), tz=UTC).strftime("%Y-%m-%d")

    def used(self) -> int:
        day = self.day()
        if self.cache is not None:
            return self.cache.budget_get(day)
        return self._memory.get(day, 0)

    def take(self) -> bool:
        if self.take_callback is not None:
            return bool(self.take_callback())
        with self._lock:
            day = self.day()
            if self.cache is not None:
                if self.cache.budget_get(day) >= self.limit:
                    return False
                self.cache.budget_increment(day)
                return True
            used = self._memory.get(day, 0)
            if used >= self.limit:
                return False
            self._memory[day] = used + 1
            return True


class RateLimiter:
    """Token bucket refilled at ``rpm`` tokens per minute, capacity ``rpm``."""

    def __init__(self, rpm: int, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:  # fmt: skip
        self.rpm = max(1, int(rpm))
        self.clock = clock
        self.sleep = sleep
        self._tokens = float(self.rpm)
        self._last = clock()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = self.clock()
        self._tokens = min(float(self.rpm), self._tokens + (now - self._last) * self.rpm / 60.0)
        self._last = now

    def acquire(self, block: bool = False) -> bool:
        """Take one token. Non-blocking by default; ``block=True`` waits for the refill."""
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
                wait = (1.0 - self._tokens) * 60.0 / self.rpm
            if not block:
                return False
            self.sleep(wait)
