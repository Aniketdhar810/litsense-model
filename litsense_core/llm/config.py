"""The ``llm:`` block of ``ml/config.yaml`` as a typed object (CLAUDE.md s5, s14.2).

Defaults mirror the brief's block with the LLM switched off, so a client built without any
config is a safe no-op. ``litsense_core`` never imports ``ml``; when a bundle manifest carries no
``llm`` slice, ``find_config`` reads the YAML file by path (the repo checkout or the installed
package data), and the backend may pass its own mapping to ``litsense_core.configure_llm``.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

UNSET_MODEL = "SET-ME"
ROLES = ("explain", "chat", "parse", "synthesis")
PROVIDERS = ("gemini", "openai_compat", "fake")


@dataclass(frozen=True)
class LLMConfig:
    enabled: bool = False
    provider: str = "gemini"
    model: str = UNSET_MODEL
    model_quality: str = UNSET_MODEL
    api_key_env: str = "GEMINI_API_KEY"
    temperature: dict[str, float] = field(
        default_factory=lambda: {"explain": 0.3, "chat": 0.5, "parse": 0.0, "synthesis": 0.3}
    )
    max_output_tokens: dict[str, int] = field(
        default_factory=lambda: {"explain": 120, "chat": 400, "parse": 400, "synthesis": 200}
    )
    timeout_s: float = 8.0
    max_retries: int = 2
    rpm_limit: int = 12
    daily_request_budget: int = 1200
    cache_path: str = ".cache/llm.sqlite"
    cache_ttl_hours: float = 168.0
    explain_max_words: int = 60
    chat_max_words: int = 120
    base_url: str | None = None  # openai_compat endpoints only

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "LLMConfig":
        """Build from the YAML block; unknown keys are ignored, missing keys keep the defaults."""
        if not raw:
            return cls()
        base = cls()
        kwargs: dict[str, Any] = {}
        for name in cls.__dataclass_fields__:
            if name in raw and raw[name] is not None:
                kwargs[name] = raw[name]
        for name in ("temperature", "max_output_tokens"):
            if name in kwargs:
                merged = dict(getattr(base, name))
                merged.update({str(k): v for k, v in dict(kwargs[name]).items()})
                kwargs[name] = merged
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def model_is_set(self) -> bool:
        return bool(self.model) and self.model != UNSET_MODEL

    @property
    def quality_model_is_set(self) -> bool:
        return bool(self.model_quality) and self.model_quality != UNSET_MODEL

    def api_key(self) -> str | None:
        """The key from the environment variable the config names; never stored or logged."""
        if self.provider == "fake":
            return "fake"
        value = os.environ.get(self.api_key_env, "")
        return value.strip() or None

    def role_params(self, role: str) -> tuple[float, int]:
        if role not in ROLES:
            raise ValueError(f"unknown LLM role {role!r}; expected one of {ROLES}")
        return float(self.temperature[role]), int(self.max_output_tokens[role])


def find_config(explicit: Mapping[str, Any] | None = None) -> LLMConfig:
    """``explicit`` (a bundle's config slice or the backend's mapping) wins; otherwise the
    ``llm:`` block of ``ml/config.yaml`` next to this package, if any; otherwise the defaults."""
    if explicit:
        return LLMConfig.from_mapping(explicit)
    path = Path(__file__).resolve().parents[2] / "ml" / "config.yaml"
    if path.exists():
        import yaml

        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return LLMConfig.from_mapping(raw.get("llm"))
    return LLMConfig()
