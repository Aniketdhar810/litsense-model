"""Structured JSON logs, one object per line, carrying the request id (D77).

Nothing here logs request bodies. Callers pass explicit fields; the access log carries the route
template, never a query string, a ``UserState``, a chat message, a review text or a prompt.
"""

import contextvars
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, TextIO

REQUEST_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
QUIET_LOGGERS = ("httpx", "httpcore", "google_genai", "urllib3")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        request_id = REQUEST_ID.get()
        if request_id:
            out["request_id"] = request_id
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            out.update(fields)
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO", stream: TextIO | None = None) -> None:
    """Root logger to JSON on stdout; uvicorn's loggers propagate to it. httpx and the Gemini SDK
    stay at WARNING so request URLs never reach the log."""
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True


def log(logger: logging.Logger, level: int, msg: str, **fields: Any) -> None:
    logger.log(level, msg, extra={"fields": fields})
