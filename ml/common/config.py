"""Load and validate ml/config.yaml, the only home of hyperparameters, paths and thresholds."""

import re
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any

import yaml

from ml.common.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "ml" / "config.yaml"

REQUIRED_SECTIONS = (
    "project",
    "sources",
    "data",
    "canonical",
    "models",
    "content",
    "nlp",
    "aspects",
    "spoiler",
    "ranking",
    "serving",
    "llm",
    "eval",
)
STREAM_KEYS = ("chunk_bytes", "retries", "timeout_s", "checkpoint_rows")
REGEX_LISTS = (
    "data.title_blocklist",
    "data.shipping_patterns",
    "canonical.boxset_patterns",
    "canonical.edition_noise",
    "canonical.subtitle_noise_patterns",
    "spoiler.regex_patterns",
)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML config (default ``ml/config.yaml``), validate it and return it as a dict."""
    with open(path or DEFAULT_CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)
    return cfg


def config_get(cfg: Mapping[str, Any], key: str) -> Any:
    """Return the value at a dotted key such as ``"data.kcore.user"``."""
    node: Any = cfg
    for part in key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise ConfigError(f"unknown config key: {key}")
        node = node[part]
    return node


def validate_config(cfg: Any) -> None:
    """Raise ConfigError listing every problem found; checks structure, not tuning choices."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("config must be a mapping")
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ConfigError(f"missing config sections: {missing}")
    not_mappings = [s for s in REQUIRED_SECTIONS if not isinstance(cfg[s], Mapping)]
    if not_mappings:  # e.g. a section left empty in the YAML loads as None
        raise ConfigError(f"config sections must be mappings: {not_mappings}")
    checks: list[tuple[str, Callable[[], list[str]]]] = [
        *((key, partial(_check_regex_list, cfg, key)) for key in REGEX_LISTS),
        ("aspects", partial(_check_aspects, cfg["aspects"])),
        ("ranking", partial(_check_buckets, cfg["ranking"])),
        ("data", partial(_check_data, cfg["data"])),
    ]
    problems: list[str] = []
    for name, check in checks:
        try:
            problems += check()
        except (ConfigError, KeyError, TypeError, IndexError) as e:  # missing or mistyped keys
            problems.append(f"{name}: malformed ({type(e).__name__}: {e})")
    if not isinstance(cfg["project"].get("seed"), int):
        problems.append("project.seed must be an int")
    if problems:
        raise ConfigError("invalid config:\n  " + "\n  ".join(problems))


def _check_regex_list(cfg: Mapping[str, Any], key: str) -> list[str]:
    problems = []
    for pattern in config_get(cfg, key):
        if not isinstance(pattern, str):  # YAML turns bare yes/no/numbers into non-strings
            problems.append(f"{key}: {pattern!r} is not a string")
            continue
        try:
            re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            problems.append(f"{key}: {pattern!r} does not compile ({e})")
    return problems


def _check_aspects(aspects: Mapping[str, Any]) -> list[str]:
    problems = []
    names = aspects["names"]
    if len(names) != 12 or len(set(names)) != 12:
        problems.append("aspects.names must hold 12 unique aspects")
    unknown = set(aspects.get("tau") or {}) - set(names)
    if unknown:
        problems.append(f"aspects.tau has unknown aspects: {sorted(unknown)}")
    if len(set(aspects["descriptors"])) != 3:
        problems.append("aspects.descriptors must hold 3 unique descriptors")
    if set(aspects["rating_weights"]) != {1, 2, 3, 4, 5}:
        problems.append("aspects.rating_weights must have integer keys 1..5")
    return problems


def _check_buckets(ranking: Mapping[str, Any]) -> list[str]:
    problems = []
    buckets = ranking["buckets"]
    bounds = sorted(buckets.values())
    if bounds[0][0] != 0:
        problems.append("ranking.buckets must start at 0 interactions")
    for (lo, hi), (next_lo, _) in zip(bounds, bounds[1:], strict=False):
        if hi < lo or next_lo != hi + 1:
            problems.append(f"ranking.buckets are not contiguous at [{lo}, {hi}]")
    quotas = ranking["quotas"]
    if set(quotas) != set(buckets):
        problems.append("ranking.quotas and ranking.buckets must name the same buckets")
    sources = {frozenset(q) for q in quotas.values()}
    if len(sources) != 1:
        problems.append("ranking.quotas must list the same candidate sources for every bucket")
    if any(not isinstance(n, int) or n < 0 for q in quotas.values() for n in q.values()):
        problems.append("ranking.quotas must be non-negative ints")
    return problems


def _check_data(data: Mapping[str, Any]) -> list[str]:
    problems = []
    q = data["split_quantiles"]
    if len(q) != 2 or not 0 < q[0] < q[1] < 1:
        problems.append("data.split_quantiles must be two increasing values in (0, 1)")
    strata = data["sparsity_buckets"]
    if strata[0][0] != 1 or any(b[0] != a[1] + 1 for a, b in zip(strata, strata[1:], strict=False)):
        problems.append("data.sparsity_buckets must be contiguous and start at 1")
    if not isinstance(data["use_goodreads_graph"], bool):
        problems.append("data.use_goodreads_graph must be true or false")
    stream = data["stream"]
    if any(not isinstance(stream[k], int | float) or stream[k] <= 0 for k in STREAM_KEYS):
        problems.append(f"data.stream needs positive {', '.join(STREAM_KEYS)}")
    return problems
