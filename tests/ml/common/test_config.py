"""ml/config.yaml loads, validates, and rejects structural mistakes (CLAUDE.md s5)."""

import copy
import re
from typing import Any

import pytest

from ml.common.config import REGEX_LISTS, config_get, validate_config
from ml.common.errors import ConfigError


def test_repo_config_is_valid(repo_config: dict[str, Any]) -> None:
    validate_config(repo_config)
    assert repo_config["project"]["seed"] == 42
    assert config_get(repo_config, "data.kcore.user") == 5
    assert config_get(repo_config, "aspects.rating_weights") == {
        5: 1.0,
        4: 0.5,
        3: 0.0,
        2: -0.5,
        1: -1.0,
    }


def test_regex_patterns_compile_with_intended_text(repo_config: dict[str, Any]) -> None:
    for key in REGEX_LISTS:
        for pattern in config_get(repo_config, key):
            re.compile(pattern, re.IGNORECASE)
    ship = [re.compile(p, re.I) for p in config_get(repo_config, "data.shipping_patterns")]
    assert any(p.search("Pages were falling out when it arrived") for p in ship)
    box = [re.compile(p, re.I) for p in config_get(repo_config, "canonical.boxset_patterns")]
    assert any(p.search("The Ember Cycle Box Set: Books 1-3") for p in box)
    assert not any(p.search("Mistborn: The Final Empire") for p in box)


def test_unknown_key_raises(repo_config: dict[str, Any]) -> None:
    with pytest.raises(ConfigError, match="unknown config key"):
        config_get(repo_config, "data.no_such_key")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c["spoiler"]["regex_patterns"].append("twist (is"), "does not compile"),
        (lambda c: c["data"]["title_blocklist"].append(True), "is not a string"),
        (lambda c: c["aspects"]["names"].pop(), "12 unique aspects"),
        (lambda c: c["aspects"]["tau"].update(prose=0.5), "unknown aspects"),
        (lambda c: c["ranking"]["buckets"].update(warming=[1, 3]), "not contiguous"),
        (lambda c: c["ranking"]["quotas"]["warm"].pop("longtail"), "same candidate sources"),
        (lambda c: c["data"].update(split_quantiles=[0.9, 0.85]), "split_quantiles"),
        (lambda c: c.pop("llm"), "missing config sections"),
        (lambda c: c.update(ranking=None), r"must be mappings: \['ranking'\]"),
        (lambda c: c["aspects"].pop("names"), r"aspects: malformed \(KeyError"),
        (lambda c: c["spoiler"].update(regex_patterns=None), "regex_patterns: malformed"),
    ],
)
def test_structural_mistakes_are_rejected(
    repo_config: dict[str, Any], mutate: Any, message: str
) -> None:
    cfg = copy.deepcopy(repo_config)
    mutate(cfg)
    with pytest.raises(ConfigError, match=message):
        validate_config(cfg)
