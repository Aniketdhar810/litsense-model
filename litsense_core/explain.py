"""LLM explanations with the template as the permanent fallback (CLAUDE.md s14.6).

``explain(state, book_id, store)`` builds the evidence exactly as before, asks the LLM when the
client is available and the reader has not disabled AI, validates the text and falls back to the
template reason otherwise. ``llm_reason`` is that step over a prebuilt ``Evidence`` (the audit
scripts use it). The main feed never calls it: ``RecItem.reason`` stays the template (s14, the
80 ms budget). With the LLM off, ``explain`` returns byte-for-byte what it returned before Phase 6.
"""

from dataclasses import dataclass

from litsense_core import validators
from litsense_core.llm import LLMResult, PrivacyError, get_client
from litsense_core.profile import Profile, build_profile
from litsense_core.reasons import build_evidence, template_reason
from litsense_core.schemas import Evidence, Explanation, UserState
from litsense_core.store import ArtifactStore

# Fallback reasons that mean "the LLM was never going to answer" rather than a failure
# (``ReasonResult.failed`` separates the two for the audit's rejection counts, D68).
OFF_REASONS = frozenset({"disabled", "not_configured", "no_api_key", "disabled_ai"})


@dataclass
class ReasonResult:
    text: str | None  # validated LLM text, or None
    rejections: list[str]
    fallback_reason: str | None  # why there is no LLM text (None when text is set)

    @property
    def failed(self) -> bool:
        """The LLM was available and still produced nothing usable."""
        return self.text is None and self.fallback_reason not in OFF_REASONS


def llm_reason(evidence: Evidence, store: ArtifactStore, disabled_ai: bool = False) -> ReasonResult:
    """One validated LLM explanation from the evidence alone (no facts outside it)."""
    client = get_client()
    if disabled_ai:
        return ReasonResult(None, [], "disabled_ai")
    reason = client.unavailable_reason()
    if reason is not None:
        return ReasonResult(None, [], reason)
    try:
        result: LLMResult = client.text(
            "explain", {"evidence_json": validators.evidence_json(evidence)}, "explain"
        )
    except PrivacyError:
        return ReasonResult(None, ["privacy"], "privacy")
    if not result.ok:
        return ReasonResult(None, [], result.fallback_reason or "provider_error")
    text = " ".join(str(result.text).split())
    problems = validators.check(
        text, evidence.allowed_titles(), evidence, client.config.explain_max_words, store
    )
    if problems:
        return ReasonResult(None, problems, "rejected")
    return ReasonResult(text, [], None)


def explain(state: UserState, book_id: int, store: ArtifactStore) -> Explanation:
    book_id = store.catalog.check_id(book_id)
    profile: Profile = build_profile(state, store)
    evidence = build_evidence(book_id, profile, store, ["explain"])
    out = llm_reason(evidence, store, state.overrides.disabled_ai)
    if out.text is not None:
        return Explanation(book_id=book_id, text=out.text, method="llm", evidence=evidence)
    return Explanation(
        book_id=book_id,
        text=template_reason(book_id, profile, store, evidence),
        method="template",
        evidence=evidence,
        rejections=out.rejections,
    )
