"""Named constants for PrivacyGuard Pipeline.

Centralises magic numbers/literals used inside the detection, masking,
LLM proxy, and logging logic. Pydantic ``Settings`` defaults in
``config.py`` are intentionally left untouched — they are already named
via ``Field(default=...)`` and are part of the public configuration
surface, not internal algorithm literals.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Detection confidence
# ---------------------------------------------------------------------------
PATTERN_CONFIDENCE = 0.95
NATASHA_NER_CONFIDENCE = 0.85
NATASHA_ADDR_CONFIDENCE = 0.75
NATASHA_ADDR_HOUSE_HEURISTIC_CONFIDENCE = 0.6
CONTEXT_RESOLVED_CONFIDENCE = 0.7
CONTEXT_LOOKBACK_WORDS = 3

# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------
INN_VALID_LENGTHS = (10, 12)
PASSPORT_DIGIT_COUNT = 10
IP_OCTET_MIN = 0
IP_OCTET_MAX = 255
LUHN_MODULO = 10
LUHN_DOUBLE_SUBTRACT = 9

# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------
TOKEN_HEX_LENGTH = 8

# ---------------------------------------------------------------------------
# LLM proxy
# ---------------------------------------------------------------------------
HTTP_TIMEOUT_SECONDS = 120.0
HTTP_CONNECT_TIMEOUT_SECONDS = 30.0
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 4096
CLAUDE_API_VERSION = '2023-06-01'
HTTP_STATUS_UNAUTHORIZED = 401

# ---------------------------------------------------------------------------
# Text truncation
# ---------------------------------------------------------------------------
PANEL_PREVIEW_CHARS = 500
LOG_PREVIEW_CHARS = 100
AUDIT_PREVIEW_CHARS = 200

# ---------------------------------------------------------------------------
# PDF anonymization
# ---------------------------------------------------------------------------
PDF_OCR_LANGUAGE = 'rus'
PDF_OCR_ZOOM = 3.0
PDF_REDACTION_PADDING = 2.0
