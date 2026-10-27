"""Adapter protocol and the three error classes the client retries or gives up on."""

from typing import Any, Protocol


class ProviderError(RuntimeError):
    """The provider answered with an error the client should not retry (4xx other than 429,
    malformed response, missing SDK)."""


class RateLimited(ProviderError):
    """HTTP 429 or a quota message: retried with backoff."""


class QuotaExhausted(RateLimited):
    """A 429 for a per-day quota (``...PerDay...`` in Google's message): nothing until the reset,
    so callers stop instead of retrying; reported as ``quota``."""


class Unavailable(RateLimited):
    """HTTP 5xx (503 "model overloaded"): retried like a 429, reported as ``unavailable``."""


class TimedOut(ProviderError):
    """The call exceeded ``llm.timeout_s``."""


class Adapter(Protocol):
    name: str

    def generate(
        self,
        prompt: str,
        *,
        json_schema: dict[str, Any] | None,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> str:
        """Return the raw text of one completion (JSON text when ``json_schema`` is given)."""
        ...
