"""Shared readers for the frozen dataset (CLAUDE.md s7.7). Ids come only from ``id_map.json``."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IdMap:
    """Position = integer id: ``users[u]`` and ``items[i]`` are the original string ids."""

    users: list[str]
    items: list[str]

    @property
    def n_users(self) -> int:
        return len(self.users)

    @property
    def n_items(self) -> int:
        return len(self.items)


def load_id_map(path: Path) -> IdMap:
    """The one id map written by data.kcore_split; never rebuild ids anywhere else (s2.2 bug 2)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return IdMap(list(raw["users"]), list(raw["items"]))
