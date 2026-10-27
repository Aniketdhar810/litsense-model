"""LLM client for the GenAI layer (CLAUDE.md s14.2): provider adapters, prompt files, cache,
budget and the module-wide client that ``explain``, ``chat_turn`` and ``parse_onboarding`` use.

The LLM never selects items (s3.1). Every call returns an ``LLMResult`` whose ``fallback_reason``
says why no text came back; nothing in this package raises for model trouble (s15.3). The only
exception is the privacy guard, which raises ``PrivacyError`` when a caller tries to send a user
identifier, because that is a programming error and not a model failure.
"""

from litsense_core.llm.client import (
    LLMClient,
    LLMResult,
    PrivacyError,
    configure,
    get_client,
)
from litsense_core.llm.config import LLMConfig

__all__ = ["LLMClient", "LLMConfig", "LLMResult", "PrivacyError", "configure", "get_client"]
