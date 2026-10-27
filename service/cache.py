"""A small thread-safe LRU cache with a time-to-live (explain and book-insights, D79)."""

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Generic, TypeVar

V = TypeVar("V")


class TTLCache(Generic[V]):
    """At most ``max_entries`` values, each valid for ``ttl_s`` seconds. ``max_entries`` or
    ``ttl_s`` of 0 turns the cache off."""

    def __init__(
        self, max_entries: int, ttl_s: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self._clock = clock
        self._data: OrderedDict[str, tuple[float, V]] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0 and self.ttl_s > 0

    def get(self, key: str) -> V | None:
        if not self.enabled:
            return None
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            expires, value = entry
            if self._clock() >= expires:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: V) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._data[key] = (self._clock() + self.ttl_s, value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)
