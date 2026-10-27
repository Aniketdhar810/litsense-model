"""Gemini through the ``google-genai`` SDK (requirements-core). Imported only when the provider
is ``gemini`` and the client is actually about to call the network."""

from typing import Any

from litsense_core.llm.adapters.base import (
    ProviderError,
    QuotaExhausted,
    RateLimited,
    TimedOut,
    Unavailable,
)


class GeminiAdapter:
    name = "gemini"

    def __init__(self, model: str, api_key: str) -> None:
        if not api_key:
            raise ProviderError("no API key in the environment")
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover (google-genai is pinned in core)
            raise ProviderError("google-genai is not installed") from exc
        self.model = model
        self._genai = genai
        self._client = genai.Client(api_key=api_key)

    def generate(
        self,
        prompt: str,
        *,
        json_schema: dict[str, Any] | None,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> str:
        types = self._genai.types
        config: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "http_options": types.HttpOptions(timeout=int(timeout * 1000)),
            # No tools are ever passed; this only silences the SDK's function-calling notice.
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        if json_schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_json_schema"] = json_schema
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=prompt, config=types.GenerateContentConfig(**config)
            )
        except Exception as exc:  # noqa: BLE001 (every SDK error maps to one of three classes)
            raise _classify(exc) from exc
        text = getattr(response, "text", None)
        if not text:
            raise ProviderError("empty response")
        return str(text)


def _classify(exc: Exception) -> ProviderError:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    message = str(exc)
    if code == 429 or "429" in message or "RESOURCE_EXHAUSTED" in message:
        if "PerDay" in message or "per day" in message.lower():
            return QuotaExhausted(message[:300])
        return RateLimited(message[:200])
    lower = message.lower()
    # 504 = the server-side deadline (our timeout) expired: no backoff, the caller falls back.
    if code == 504 or "504" in message or "timeout" in lower or "timed out" in lower:
        return TimedOut(message[:200])
    if (isinstance(code, int) and code >= 500) or "503" in message or "UNAVAILABLE" in message:
        return Unavailable(message[:200])  # 5xx: retried like a 429 (s14.2), reported apart
    return ProviderError(message[:200])
