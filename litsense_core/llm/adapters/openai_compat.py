"""OpenAI-compatible endpoints (a local Ollama server, for example). Not wired yet: the project
runs on Gemini, and adding this path needs the user's decision on the endpoint and a dependency
check (httpx is in core, so no new package would be needed)."""

from typing import Any

from litsense_core.llm.adapters.base import ProviderError


class OpenAICompatAdapter:
    name = "openai_compat"

    def __init__(self, model: str, api_key: str, base_url: str | None) -> None:
        self.model, self.base_url = model, base_url

    def generate(
        self,
        prompt: str,
        *,
        json_schema: dict[str, Any] | None,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> str:
        raise NotImplementedError(
            "llm.provider 'openai_compat' is not implemented yet; use 'gemini' or 'fake'"
            " (docs/decisions.md D68)"
        ) from ProviderError("openai_compat stub")
