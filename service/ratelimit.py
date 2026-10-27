"""Per-client token bucket, in process (D79).

It holds state in this process only, which is right for the single uvicorn process this service
runs. With several instances or workers, keep the ``take(key)`` call and move the buckets to Redis
(a Lua token bucket, or INCR plus EXPIRE per second) so every instance sees the same counts.
"""

import threading
import time
from collections import OrderedDict
from collections.abc import Callable


class TokenBucketLimiter:
    """``rate`` tokens per second up to ``burst`` per key; ``rate`` 0 turns limiting off. The
    least recently seen keys are dropped beyond ``max_keys`` (a dropped key starts full)."""

    def __init__(
        self,
        rate: float,
        burst: int,
        max_keys: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.rate = float(rate)
        self.burst = float(burst)
        self.max_keys = max_keys
        self._clock = clock
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.rate > 0

    def take(self, key: str) -> float:
        """0.0 when a token was taken, else the seconds until the next token."""
        if not self.enabled:
            return 0.0
        now = self._clock()
        with self._lock:
            tokens, last = self._buckets.get(key, (self.burst, now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens >= 1.0:
                self._buckets[key] = (tokens - 1.0, now)
                wait = 0.0
            else:
                self._buckets[key] = (tokens, now)
                wait = (1.0 - tokens) / self.rate
            self._buckets.move_to_end(key)
            while len(self._buckets) > self.max_keys:
                self._buckets.popitem(last=False)
        return wait
