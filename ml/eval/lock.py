"""The test-split lock (CLAUDE.md s8.8): test mode runs once, in Week 13.

``eval.test --confirm`` (built in Week 13) calls ``acquire_test_lock`` before touching the test
split. A second run is refused unless ``reuse`` is explicitly passed (``--i-know-this-reuses-test``,
only after asking the user), and then a warning is appended to docs/decisions.md.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from ml.common.errors import StageError


def acquire_test_lock(
    test_dir: Path, git_sha: str, decisions: Path, *, confirm: bool, reuse: bool = False
) -> Path:
    """Write ``test_dir/LOCKED`` (timestamp, git SHA) or refuse; returns the lock path."""
    if not confirm:
        raise StageError("test mode needs --confirm (CLAUDE.md s8.8)")
    lock = Path(test_dir) / "LOCKED"
    now = datetime.now(UTC).isoformat(timespec="seconds")
    if lock.exists():
        if not reuse:
            raise StageError(f"the test split was already used ({lock}); refusing to rerun")
        with open(decisions, "a", encoding="utf-8") as f:
            f.write(
                f"\n- **WARNING {now}:** the test split was evaluated again at {git_sha} "
                f"(--i-know-this-reuses-test); earlier lock: {lock.read_text(encoding='utf-8')}\n"
            )
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"locked_at": now, "git_sha": git_sha}) + "\n", encoding="utf-8")
    return lock
