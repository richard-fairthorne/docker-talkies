"""JSON structured logging — level via LOG_LEVEL or TALKIES_LOG_LEVEL.

Each record carries timestamp, level, logger name, message, source file,
line number, function name, plus any `extra={...}` fields. FATAL is mapped
to CRITICAL. Default level INFO.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone

_RESERVED = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "taskName",
    "message",
    "asctime",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        payload: dict[str, object] = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "file": record.filename,
            "line": record.lineno,
            "func": record.funcName,
        }
        for key, val in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            try:
                json.dumps(val)
                payload[key] = val
            except (TypeError, ValueError):
                payload[key] = repr(val)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False)


# Accepted level names → stdlib numeric level. The user-facing set is
# debug / info / warn / error / fatal; `warning` and `critical` are also
# accepted so the stdlib canonical names work too.
_LEVEL_ALIASES = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "FATAL": logging.CRITICAL,
    "CRITICAL": logging.CRITICAL,
}
_DEFAULT_LEVEL_NAME = "INFO"
_USER_FACING_LEVELS = ("debug", "info", "warn", "error", "fatal")


def _resolve_level() -> int:
    raw = (
        (
            os.environ.get("TALKIES_LOG_LEVEL")
            or os.environ.get("LOG_LEVEL")
            or _DEFAULT_LEVEL_NAME
        )
        .strip()
        .upper()
    )
    if raw not in _LEVEL_ALIASES:
        raise ValueError(
            f"TALKIES_LOG_LEVEL={raw!r} is not a valid level; "
            f"choose one of {_USER_FACING_LEVELS}"
        )
    return _LEVEL_ALIASES[raw]


def configure() -> None:
    level = _resolve_level()
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)
    # Quiet noisy libs unless we're at DEBUG
    # python_multipart floods at DEBUG with one record per parse callback —
    # not useful unless debugging multipart itself.
    logging.getLogger("python_multipart").setLevel(logging.INFO)
    if level > logging.DEBUG:
        for name in ("urllib3", "huggingface_hub", "filelock", "asyncio"):
            logging.getLogger(name).setLevel(logging.WARNING)
    else:
        # DEBUG logs FULL request + response content at the HTTP boundary —
        # TTS input text / instructions, cloned-voice reference transcripts,
        # ASR transcripts. That's PII. Warn loudly so nobody ships DEBUG to
        # production and discovers user speech in their log aggregator.
        logging.getLogger("talkies.logging").warning(
            "DEBUG level active — request/response BODIES (incl. transcripts "
            "+ TTS input text) are logged in full. This exposes PII. Do not "
            "run DEBUG in production.",
        )
