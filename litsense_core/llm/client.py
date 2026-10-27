"""``LLMClient`` (CLAUDE.md s14.2): prompts, cache, budget, limiter, retries, privacy guard.

Public surface:

* ``LLMClient(config, on_event=None, adapter=None, budget_callback=None)``.
* ``client.text(prompt_id, variables, role)`` and ``client.json(prompt_id, variables, schema,
  role)`` return an ``LLMResult``; ``result.text`` / ``result.data`` are ``None`` with a
  ``fallback_reason`` whenever the LLM cannot or should not answer. Nothing raises for model
  trouble (s15.3). ``PrivacyError`` is the one exception: a caller passed a user identifier.
* ``configure(config, on_event)`` / ``get_client()``: the process-wide client the public API uses,
  installed by ``load_store`` from the bundle's config slice (or the YAML by path, D68).

Why a result object instead of ``BaseModel | None`` (the brief's signature): the callers and the
backend's logs need the reason for a fallback (``disabled``, ``not_configured``, ``no_api_key``,
``budget``, ``rate_limited``, ``timeout``, ``provider_error``, ``invalid_json``), which a bare
``None`` cannot carry; the fake provider and the audit scripts count them.
"""

import contextlib
import hashlib
import json
import random
import re
import string
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from litsense_core.llm.adapters import (
    Adapter,
    ProviderError,
    QuotaExhausted,
    RateLimited,
    TimedOut,
    Unavailable,
    make_adapter,
)
from litsense_core.llm.budget import DailyBudget, RateLimiter
from litsense_core.llm.cache import SqliteCache
from litsense_core.llm.config import LLMConfig, find_config

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
PROMPT_VERSION = 1
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PII_KEYS = frozenset({"user_id", "corpus_user_id", "email", "username", "user_name", "history"})
MAX_BACKOFF_S = 8.0


class PrivacyError(ValueError):
    """A prompt variable looks like a user identifier (s3.10)."""


@dataclass(frozen=True)
class Prompt:
    id: str
    version: int
    role: str
    body: string.Template

    def render(self, variables: Mapping[str, Any]) -> str:
        return self.body.substitute({k: _as_text(v) for k, v in variables.items()})


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool | int | float) or value is None:
        return json.dumps(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def load_prompts(folder: Path = PROMPT_DIR, version: int = PROMPT_VERSION) -> dict[str, Prompt]:
    """Every ``<id>.v<version>.md`` in ``folder`` parsed once (front matter + body)."""
    prompts: dict[str, Prompt] = {}
    for path in sorted(folder.glob(f"*.v{version}.md")):
        text = path.read_text(encoding="utf-8")
        m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
        if m is None:
            raise ValueError(f"prompt file {path.name} has no front matter")
        meta = dict(line.split(":", 1) for line in m.group(1).splitlines() if ":" in line)
        meta = {k.strip(): v.strip() for k, v in meta.items()}
        prompt = Prompt(meta["id"], int(meta["version"]), meta["role"], string.Template(m.group(2)))
        if prompt.id != path.name.split(".")[0]:
            raise ValueError(f"prompt id {prompt.id!r} does not match file {path.name}")
        prompts[prompt.id] = prompt
    return prompts


def strip_fences(raw: str) -> str:
    """Remove a surrounding code fence and any prose before the first ``{``."""
    text = _FENCE.sub("", raw.strip())
    start = text.find("{")
    end = text.rfind("}")
    if start > 0 or (start == 0 and end < len(text) - 1):
        text = text[start : end + 1] if start >= 0 and end > start else text
    return text.strip()


@dataclass
class LLMResult:
    """What one call returned. ``ok`` means text (and ``data`` for JSON calls) is usable."""

    prompt_id: str
    text: str | None = None
    data: BaseModel | None = None
    cached: bool = False
    fallback_reason: str | None = None
    latency_ms: float = 0.0
    model: str = ""
    attempts: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.text is not None and self.fallback_reason is None


class LLMClient:
    """Thread-safe (locks around the cache, the budget and the adapter construction)."""

    def __init__(
        self,
        config: LLMConfig | Mapping[str, Any] | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        adapter: Adapter | None = None,
        budget_callback: Callable[[], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        model: str | None = None,
    ) -> None:
        self.config = config if isinstance(config, LLMConfig) else LLMConfig.from_mapping(config)
        self.model = model or self.config.model
        self.on_event = on_event
        self.prompts = load_prompts()
        self.cache = SqliteCache(self.config.cache_path, self.config.cache_ttl_hours)
        self.budget = DailyBudget(self.config.daily_request_budget, self.cache, budget_callback)
        self.limiter = RateLimiter(self.config.rpm_limit, clock, sleep)
        self.blocking_limiter = False  # offline batch jobs wait for a token instead of failing
        self._sleep = sleep
        self._adapter: Adapter | None = adapter
        self._adapter_error: str | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ availability
    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def unavailable_reason(self) -> str | None:
        """None when a call may go out; otherwise why not (never touches the network)."""
        if not self.config.enabled:
            return "disabled"
        if self.config.provider not in ("fake",) and (not self.model or self.model == "SET-ME"):
            return "not_configured"
        if self.config.api_key() is None:
            return "no_api_key"
        return None

    @property
    def available(self) -> bool:
        return self.unavailable_reason() is None

    def status(self) -> dict[str, Any]:
        """For ``health()``: never includes the key."""
        return {
            "available": self.available,
            "enabled": self.config.enabled,
            "provider": self.config.provider,
            "model": self.model,
            "reason": self.unavailable_reason(),
            "budget_used_today": self.budget.used(),
            "budget": self.config.daily_request_budget,
            "cache_entries": self.cache.count(),
        }

    def adapter(self) -> Adapter:
        with self._lock:
            if self._adapter is None:
                self._adapter = make_adapter(
                    self.config.provider, self.model, self.config.api_key(), self.config.base_url
                )
            return self._adapter

    # ------------------------------------------------------------------ calls
    def text(self, prompt_id: str, variables: Mapping[str, Any], role: str) -> LLMResult:
        """Free-text completion; ``result.text`` is the stripped output."""
        return self._call(prompt_id, variables, role, None)

    def json(
        self, prompt_id: str, variables: Mapping[str, Any], schema: type[BaseModel], role: str
    ) -> LLMResult:
        """JSON completion validated with ``schema``; ``result.data`` is the model instance."""
        return self._call(prompt_id, variables, role, schema)

    def _call(
        self,
        prompt_id: str,
        variables: Mapping[str, Any],
        role: str,
        schema: type[BaseModel] | None,
    ) -> LLMResult:
        started = time.perf_counter()
        result = LLMResult(prompt_id=prompt_id, model=self.model)
        self._assert_no_pii(variables)
        reason = self.unavailable_reason()
        if reason is not None:
            return self._finish(result, started, reason)
        prompt = self.prompts.get(prompt_id)
        if prompt is None:
            return self._finish(result, started, "unknown_prompt")
        temperature, max_tokens = self.config.role_params(role)
        key = self.cache_key(prompt, variables, temperature, max_tokens)
        hit = self.cache.get(key)
        if hit is not None:
            parsed = self._parse(hit, schema)
            if parsed is not None or schema is None:
                result.text, result.data, result.cached = hit, parsed, True
                return self._finish(result, started, None)
            self.cache.delete(key)  # a stale entry that no longer validates
        rendered = prompt.render(variables)
        json_schema = schema.model_json_schema() if schema is not None else None
        text = rendered
        reason = "provider_error"
        for attempt in range(self.config.max_retries + 1):
            result.attempts = attempt + 1
            if not self.limiter.acquire(block=self.blocking_limiter):
                reason = "rate_limited"
                break
            if not self.budget.take():
                reason = "budget"
                break
            try:
                adapter = self.adapter()
                setattr(adapter, "prompt_id", prompt_id)  # noqa: B010 (fake adapter only)
                raw = adapter.generate(
                    text,
                    json_schema=json_schema,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.config.timeout_s,
                )
            except QuotaExhausted as exc:
                reason = "quota"  # a per-day quota: retrying is pointless until the reset
                self._adapter_error = str(exc)
                break
            except RateLimited as exc:
                reason = "unavailable" if isinstance(exc, Unavailable) else "rate_limited"
                self._adapter_error = str(exc)
                self._sleep(min(MAX_BACKOFF_S, 2.0**attempt) + random.random())
                continue
            except TimedOut:
                reason = "timeout"
                break
            except NotImplementedError as exc:
                reason = "provider_error"
                self._adapter_error = str(exc)
                break
            except (ProviderError, ValueError) as exc:
                reason = "provider_error"
                self._adapter_error = str(exc)
                break
            cleaned = strip_fences(raw) if schema is not None else raw.strip()
            parsed = self._parse(cleaned, schema)
            if schema is not None and parsed is None:
                reason = "invalid_json"
                # One repair retry that sends the validation error back (s14.2).
                text = (
                    rendered + "\n\nYour previous answer was not valid JSON for the schema"
                    f" ({self._last_error}). Return only the corrected JSON object."
                )
                continue
            self.cache.set(key, prompt_id, cleaned)
            result.text, result.data = cleaned, parsed
            return self._finish(result, started, None)
        return self._finish(result, started, reason)

    _last_error = ""

    def _parse(self, text: str, schema: type[BaseModel] | None) -> BaseModel | None:
        if schema is None:
            return None
        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            self._last_error = str(exc).splitlines()[0][:200]
            return None
        except ValueError as exc:
            self._last_error = str(exc)[:200]
            return None

    def _finish(self, result: LLMResult, started: float, reason: str | None) -> LLMResult:
        result.fallback_reason = reason
        result.latency_ms = round((time.perf_counter() - started) * 1000, 3)
        event = {
            "prompt_id": result.prompt_id,
            "model": result.model,
            "latency_ms": result.latency_ms,
            "cached": result.cached,
            "ok": result.ok,
            "fallback_reason": reason,
            "attempts": result.attempts,
            "tokens": None,
        }
        result.events.append(event)
        if self.on_event is not None:
            with contextlib.suppress(Exception):  # a logging callback must never break a reply
                self.on_event(event)
        return result

    # ------------------------------------------------------------------ helpers
    def cache_key(
        self, prompt: Prompt, variables: Mapping[str, Any], temperature: float, max_tokens: int
    ) -> str:
        material = "|".join(
            [
                prompt.id,
                str(prompt.version),
                self.model,
                f"t={temperature:g};m={max_tokens}",
                json.dumps(variables, sort_keys=True, ensure_ascii=False, default=str),
            ]
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    @staticmethod
    def _assert_no_pii(variables: Mapping[str, Any]) -> None:
        """s3.10: raise if a key or string value looks like a user identifier."""

        def walk(node: Any, path: str) -> None:
            if isinstance(node, Mapping):
                for k, v in node.items():
                    if str(k) in _PII_KEYS:
                        raise PrivacyError(f"prompt variable {path}{k} would send a user id")
                    walk(v, f"{path}{k}.")
            elif isinstance(node, list | tuple):
                for i, v in enumerate(node):
                    walk(v, f"{path}{i}.")
            elif isinstance(node, str) and _EMAIL.search(node):
                raise PrivacyError(f"prompt variable {path.rstrip('.')} contains an email address")

        walk(variables, "")


# ---------------------------------------------------------------------- process-wide client
_CLIENT: LLMClient | None = None
_EXPLICIT = False  # True once configure()/set_client() installed one (load_store then keeps it)
_CLIENT_LOCK = threading.Lock()


def configure(
    config: Mapping[str, Any] | LLMConfig | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    **kwargs: Any,
) -> LLMClient:
    """Install the client the public API uses. ``config`` is the ``llm:`` mapping (a bundle's
    slice or the backend's own); ``None`` reads ``ml/config.yaml`` by path when it exists."""
    global _CLIENT, _EXPLICIT
    cfg = config if isinstance(config, LLMConfig) else find_config(config)
    with _CLIENT_LOCK:
        _CLIENT = LLMClient(cfg, on_event, **kwargs)
        _EXPLICIT = True
        return _CLIENT


def configure_default(config: Mapping[str, Any] | None) -> LLMClient:
    """``load_store``'s hook: install a client from the bundle's slice unless the backend
    already configured one explicitly."""
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None or not _EXPLICIT:
            _CLIENT = LLMClient(find_config(config))
        return _CLIENT


def get_client() -> LLMClient:
    """The configured client, or a default-config one (LLM off unless the YAML enables it)."""
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = LLMClient(find_config(None))
        return _CLIENT


def set_client(client: LLMClient | None) -> None:
    """Install a client built elsewhere (tests); ``None`` clears it."""
    global _CLIENT, _EXPLICIT
    with _CLIENT_LOCK:
        _CLIENT = client
        _EXPLICIT = client is not None
