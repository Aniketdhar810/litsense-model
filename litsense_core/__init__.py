"""LitSense serving package: recommendations, explanations and chat over a prebuilt bundle.

Stateless: the backend builds a ``UserState`` for every call and persists whatever comes back.
It never imports torch or the ``ml`` package. ``load_store(bundle_dir)`` once at boot, then the
public functions of CLAUDE.md s15.1 (re-exported here) serve from the bundle; ``litsense_core.mock``
keeps the same signatures over a synthetic catalogue for development without a bundle.
"""

__api_version__ = "0.1.0"

# Phase 6 modules share names with public functions (``explain``, ``chat_turn`` lives in ``chat``,
# ``parse_onboarding`` in ``onboarding``, ``resolve_entities`` in ``entities``). Import the modules
# first so the function names bound below win on the package; code that needs the modules uses
# ``importlib.import_module`` (see ``recommend.py``).
import litsense_core.chat  # noqa: E402, F401
import litsense_core.entities  # noqa: E402, F401
import litsense_core.explain  # noqa: E402, F401
import litsense_core.onboarding  # noqa: E402, F401
from litsense_core.llm.client import configure as configure_llm  # noqa: E402
from litsense_core.recommend import (  # noqa: E402 (needs __api_version__ above)
    analyze_user_review,
    book_insights,
    chat_turn,
    download_bundle,
    explain,
    health,
    home_rails,
    load_store,
    metrics_report,
    more_by_author,
    onboarding_grid,
    parse_onboarding,
    recommend,
    resolve_entities,
    semantic_search,
    similar_books,
)

__all__ = [
    "__api_version__",
    "analyze_user_review",
    "book_insights",
    "chat_turn",
    "configure_llm",
    "download_bundle",
    "explain",
    "health",
    "home_rails",
    "load_store",
    "metrics_report",
    "more_by_author",
    "onboarding_grid",
    "parse_onboarding",
    "recommend",
    "resolve_entities",
    "semantic_search",
    "similar_books",
]
