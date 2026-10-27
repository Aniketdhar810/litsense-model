"""API schemas (CLAUDE.md s15.2): validation, defaults, JSON round trips."""

from typing import Any, get_args

import pytest
from pydantic import ValidationError

from litsense_core.schemas import (
    ASPECT_LABELS,
    Bucket,
    ChatReply,
    Evidence,
    RatingEvent,
    SessionState,
    UserState,
)

TS = 1_700_000_000_000


def evidence() -> Evidence:
    return Evidence(
        book={"book_id": 3, "title": "The Salt Orchard", "author": "A", "genres": ["Fantasy"]},
        user_evidence={"similar_liked_items": [{"book_id": 1, "title": "Ashfall", "stars": 5.0}]},
        item_evidence={},
        signals={},
    )


def test_rating_rejects_seconds_and_out_of_range_stars() -> None:
    RatingEvent(book_id=0, stars=4.5, ts=TS)
    with pytest.raises(ValidationError):
        RatingEvent(book_id=0, stars=4, ts=1_700_000_000)  # epoch seconds
    with pytest.raises(ValidationError):
        RatingEvent(book_id=0, stars=6, ts=TS)
    with pytest.raises(ValidationError):
        RatingEvent(book_id=-1, stars=4, ts=TS)


def test_defaults_are_not_shared_between_instances() -> None:
    a, b = UserState(), UserState()
    a.dismissed.append(1)
    a.overrides.muted_genres.append("Horror")
    assert b.dismissed == [] and b.overrides.muted_genres == []


def test_allowed_titles() -> None:
    assert evidence().allowed_titles() == {"The Salt Orchard", "Ashfall"}


def test_chat_reply_round_trips_through_json() -> None:
    session = SessionState(
        session_id="s1", last_shown=[3], history=[{"role": "user", "text": "hi"}]
    )
    reply = ChatReply(
        intro="Here.",
        items=[{"book_id": 3, "reason": "Similar to Ashfall", "evidence": evidence()}],
        clarification={"question": "Which?", "options": [{"label": "Ashfall", "value": 1}]},
        session=session,
        method="template",
        intent="similar_to",
    )
    assert ChatReply.model_validate_json(reply.model_dump_json()) == reply
    with pytest.raises(ValidationError):
        ChatReply(intro="x", session=session, method="template", intent="chit_chat")


def test_vocabularies_mirror_config(repo_config: dict[str, Any]) -> None:
    assert list(ASPECT_LABELS) == repo_config["aspects"]["names"]
    assert set(get_args(Bucket)) == set(repo_config["ranking"]["buckets"])
