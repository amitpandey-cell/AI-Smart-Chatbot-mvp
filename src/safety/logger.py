"""Redacting logging formatter and logging utilities.

Scrubs sensitive authentication tokens, API keys, and authorization headers
from log outputs to prevent credential leaks.
"""

import logging
import re
from typing import Pattern, List

# Patterns matching sensitive tokens, keys, and authorization strings
REDACTION_PATTERNS: List[Pattern] = [
    re.compile(r"(AIzaSy[A-Za-z0-9_-]{33})", re.IGNORECASE),  # Google API keys
    re.compile(r"(sk-[A-Za-z0-9_-]{20,})", re.IGNORECASE),   # OpenAI secret keys
    re.compile(r"(Bearer\s+)[A-Za-z0-9_\-\.]{15,}", re.IGNORECASE),  # Bearer tokens
    re.compile(r"(['\"]?password['\"]?\s*[:=]\s*['\"])([^'\"]+)(['\"])", re.IGNORECASE),
    re.compile(r"(['\"]?api[_-]?key['\"]?\s*[:=]\s*['\"])([^'\"]+)(['\"])", re.IGNORECASE),
    re.compile(r"(['\"]?authorization['\"]?\s*[:=]\s*['\"])([^'\"]+)(['\"])", re.IGNORECASE),
]


class RedactingFormatter(logging.Formatter):
    """Logging formatter that redacts sensitive strings from log messages."""

    def format(self, record: logging.LogRecord) -> str:
        original_message = super().format(record)
        redacted = original_message
        for pattern in REDACTION_PATTERNS:
            if pattern.groups == 1:
                redacted = pattern.sub("[REDACTED_API_KEY]", redacted)
            elif pattern.groups == 3:
                redacted = pattern.sub(r"\g<1>[REDACTED]\g<3>", redacted)
            else:
                redacted = pattern.sub(r"\g<1>[REDACTED_TOKEN]", redacted)
        return redacted


def get_logger(name: str) -> logging.Logger:
    """Configure and retrieve a logger equipped with redacting formatter."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = RedactingFormatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
