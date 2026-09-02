"""Validators for PII detection patterns.

Each validator function takes a match result and returns True/False.
"""

from __future__ import annotations

import re
from typing import Callable

from privacyguard_pipeline.constants import (
    INN_VALID_LENGTHS,
    IP_OCTET_MAX,
    IP_OCTET_MIN,
    LUHN_DOUBLE_SUBTRACT,
    LUHN_MODULO,
    PASSPORT_DIGIT_COUNT,
)
from privacyguard_pipeline.detection.patterns import IP_RE

# ---------------------------------------------------------------------------
# Luhn algorithm
# ---------------------------------------------------------------------------


def luhn_check(digits: str) -> bool:
    """Validate a number using the Luhn algorithm."""
    total = 0
    reverse_digits = digits[::-1]
    for i, ch in enumerate(reverse_digits):
        if not ch.isdigit():
            return False
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= LUHN_DOUBLE_SUBTRACT
        total += n
    return total % LUHN_MODULO == 0


# ---------------------------------------------------------------------------
# Type-specific validators
# ---------------------------------------------------------------------------


def validate_ip(octets: tuple[str, ...]) -> bool:
    """Validate IPv4 address octets."""
    if len(octets) != 4:
        return False
    for octet in octets:
        val = int(octet)
        if val < IP_OCTET_MIN or val > IP_OCTET_MAX:
            return False
    return True


def validate_card(raw: str) -> bool:
    """Validate a bank card number using the Luhn algorithm."""
    clean = raw.replace('-', '').replace(' ', '')
    return bool(clean) and luhn_check(clean)


def validate_inn(raw: str) -> bool:
    """Validate INN: must be exactly 10 or 12 digits."""
    clean = raw.strip()
    return len(clean) in INN_VALID_LENGTHS


def validate_passport(raw: str) -> bool:
    """Validate passport: series (4) + number (6) = 10 digits."""
    clean = raw.replace(' ', '').replace('-', '')
    return len(clean) == PASSPORT_DIGIT_COUNT


def validate_phone(text: str, start: int, end: int) -> bool:
    """Validate a phone number using the phonenumbers library."""
    try:
        from phonenumbers import PhoneNumberMatcher

        matches = list(
            PhoneNumberMatcher(text[start:end], 'RU'),
        )
        if not matches:
            matches = list(
                PhoneNumberMatcher(text[start:end], None),
            )
        return bool(matches)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Validator registry
# ---------------------------------------------------------------------------

ValidatorFunc = Callable[[str, str, int, int], bool]

VALIDATOR_REGISTRY: dict[str, ValidatorFunc] = {
    'CARD': lambda raw, text, start, end: validate_card(raw),
    'IP': lambda raw, text, start, end: validate_ip(
        re.match(IP_RE, raw).groups() if re.match(IP_RE, raw) else (),  # type: ignore[union-attr]
    ),
    'INN': lambda raw, text, start, end: validate_inn(raw),
    'PASSPORT': lambda raw, text, start, end: validate_passport(raw),
    'PHONE': lambda raw, text, start, end: validate_phone(text, start, end),
}
