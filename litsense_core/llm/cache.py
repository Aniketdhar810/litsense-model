"""Sqlite response cache (CLAUDE.md s14.2), thread-safe, with a TTL.

Keys are sha256 over ``prompt_id | prompt version | model | role params | canonical JSON of the
variables`` (built by the client); values are the raw text the provider returned. The same file
also holds the daily request counter (``litsense_core.llm.budget``), so one small file under
``llm.cache_path`` is all the LLM layer writes to disk. ``":memory:"`` keeps everything in RAM.
"""

import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = (
    "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, prompt_id TEXT, value TEXT,"
    " created REAL)",
    "CREATE TABLE IF NOT EXISTS budget (day TEXT PRIMARY KEY, count INTEGER)",
)


class SqliteCache:
    def __init__(self, path: str | Path, ttl_hours: float) -> None:
        self.path = str(path)
        self.ttl_s = float(ttl_hours) * 3600.0
        self._lock = threading.Lock()
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        with self._lock:
            for statement in SCHEMA:
                self._conn.execute(statement)
            self._conn.commit()

    def get(self, key: str, now: float | None = None) -> str | None:
        now = time.time() if now is None else now
        with self._lock:
            row = self._conn.execute(
                "SELECT value, created FROM responses WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        value, created = row
        if self.ttl_s > 0 and now - float(created) > self.ttl_s:
            self.delete(key)
            return None
        return str(value)

    def set(self, key: str, prompt_id: str, value: str, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses (key, prompt_id, value, created)"
                " VALUES (?, ?, ?, ?)",
                (key, prompt_id, value, now),
            )
            self._conn.commit()

    def delete(self, key: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM responses WHERE key = ?", (key,))
            self._conn.commit()

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0])

    # Daily budget rows (used by ``budget.DailyBudget``).
    def budget_get(self, day: str) -> int:
        with self._lock:
            row = self._conn.execute("SELECT count FROM budget WHERE day = ?", (day,)).fetchone()
        return int(row[0]) if row else 0

    def budget_increment(self, day: str) -> int:
        with self._lock:
            self._conn.execute(
                "INSERT INTO budget (day, count) VALUES (?, 1)"
                " ON CONFLICT(day) DO UPDATE SET count = count + 1",
                (day,),
            )
            self._conn.commit()
            row = self._conn.execute("SELECT count FROM budget WHERE day = ?", (day,)).fetchone()
            return int(row[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()
