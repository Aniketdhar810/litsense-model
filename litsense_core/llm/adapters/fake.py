"""Scripted adapter for tests and the LLM-off demo (CLAUDE.md s16.1 ``FakeAdapter``).

``script`` maps a prompt id (the first line of the rendered prompt is not available here, so the
client passes the id through ``prompt_id``) to a list of responses consumed in order; the last
one repeats. A response may be a string, an exception instance (raised), or a callable taking
the rendered prompt. With no script for a prompt id, ``default`` answers: a deterministic JSON
object for JSON requests and a short sentence otherwise.
"""

import json
import re
from collections.abc import Callable
from typing import Any

Response = str | BaseException | Callable[[str], str]


class FakeAdapter:
    name = "fake"

    def __init__(self, script: dict[str, list[Response]] | None = None) -> None:
        self.script: dict[str, list[Response]] = {k: list(v) for k, v in (script or {}).items()}
        self.calls: list[dict[str, Any]] = []
        self.prompt_id = ""  # set by the client before each call

    def generate(
        self,
        prompt: str,
        *,
        json_schema: dict[str, Any] | None,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> str:
        self.calls.append(
            {
                "prompt_id": self.prompt_id,
                "prompt": prompt,
                "json": json_schema is not None,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout,
            }
        )
        queue = self.script.get(self.prompt_id)
        if queue:
            response = queue.pop(0) if len(queue) > 1 else queue[0]
            if isinstance(response, BaseException):
                raise response
            if callable(response):
                return response(prompt)
            return response
        return self.default(prompt, json_schema)

    @staticmethod
    def default(prompt: str, json_schema: dict[str, Any] | None) -> str:
        """A canned answer: the schema's required fields with empty values, or one sentence."""
        if json_schema is None:
            return "A fitting pick for this reader."
        props = json_schema.get("properties", {})
        out: dict[str, Any] = {}
        for name, spec in props.items():
            kind = spec.get("type")
            if kind is None and "anyOf" in spec:
                kind = next((a.get("type") for a in spec["anyOf"] if a.get("type") != "null"),
                            None)  # fmt: skip
            out[name] = {"string": "A canned fake response.", "integer": 0, "number": 0.0,
                         "boolean": False, "array": [], "object": {}}.get(kind)  # fmt: skip
        return json.dumps(out)


def prompt_field(prompt: str, label: str) -> str:
    """Helper for scripted callables: the rest of the line after ``LABEL:`` (rendered JSON
    variables are single-line)."""
    m = re.search(rf"^{re.escape(label)}:[ \t]*(.*)$", prompt, re.M)
    return m.group(1).strip() if m else ""
