"""Provider adapters (CLAUDE.md s14.2): one ``generate`` method each, same errors."""

from litsense_core.llm.adapters.base import (
    Adapter,
    ProviderError,
    QuotaExhausted,
    RateLimited,
    TimedOut,
    Unavailable,
)


def make_adapter(provider: str, model: str, api_key: str | None, base_url: str | None) -> Adapter:
    """Instantiate the adapter named in ``llm.provider``; SDKs are imported lazily."""
    if provider == "fake":
        from litsense_core.llm.adapters.fake import FakeAdapter

        return FakeAdapter()
    if provider == "gemini":
        from litsense_core.llm.adapters.gemini import GeminiAdapter

        return GeminiAdapter(model, api_key or "")
    if provider == "openai_compat":
        from litsense_core.llm.adapters.openai_compat import OpenAICompatAdapter

        return OpenAICompatAdapter(model, api_key or "", base_url)
    raise ValueError(f"unknown llm.provider {provider!r}")


__all__ = [
    "Adapter",
    "ProviderError",
    "QuotaExhausted",
    "RateLimited",
    "TimedOut",
    "Unavailable",
    "make_adapter",
]
