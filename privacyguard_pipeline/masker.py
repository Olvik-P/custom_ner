"""Masking and demasking module for PrivacyGuard Pipeline.

Generates unique tokens for PII spans and manages the mapping dictionary.
All replacements are performed from end to start to preserve index correctness.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from privacyguard_pipeline._common import PIISpan

logger = logging.getLogger(__name__)


@dataclass
class MappingEntry:
    """A single token-to-original mapping entry.

    Attributes:
        token: The masking token (e.g. ``<PHONE_A7B3C902>``).
        original: The original PII value.
        entity_type: Type of the PII entity.
    """

    token: str
    original: str
    entity_type: str


class Masker:
    """Masks PII spans with unique tokens and restores original values.

    The mapping dict exists only in memory and is cleared after demasking.

    Attributes:
        mapping: Dictionary mapping tokens to MappingEntry objects.
    """

    def __init__(self) -> None:
        self._mapping: dict[str, MappingEntry] = {}

    # ------------------------------------------------------------------
    # Token generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_token(entity_type: str) -> str:
        """Generate a unique masking token.

        Args:
            entity_type: Type of PII entity (PHONE, PER, LOC, etc.).

        Returns:
            Token string in format ``<TYPE_UUID8>``.
        """
        short_uuid = uuid.uuid4().hex[:8].upper()
        return f"<{entity_type}_{short_uuid}>"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mask(self, text: str, spans: list[PIISpan]) -> str:
        """Replace all PII spans with masking tokens.

        Performs replacements from end to start to preserve indices.

        Args:
            text: Original text.
            spans: Detected PII spans, sorted by position.

        Returns:
            Text with PII replaced by tokens.
        """
        # Sort spans from end to start to preserve index correctness
        sorted_spans = sorted(spans, key=lambda s: s.end, reverse=True)

        result = text
        for span in sorted_spans:
            token = self._generate_token(span.entity_type)
            self._mapping[token] = MappingEntry(
                token=token,
                original=span.text,
                entity_type=span.entity_type,
            )
            result = result[:span.start] + token + result[span.end:]
            logger.debug("Masked '%s' -> %s", span.text[:20], token)

        return result

    def demask(self, text: str) -> str:
        """Restore original values from masking tokens.

        Args:
            text: Text possibly containing masking tokens.

        Returns:
            Text with tokens replaced by original values.
        """
        result = text
        for entry in self._mapping.values():
            if entry.token in result:
                result = result.replace(entry.token, entry.original)
                logger.debug("Demasked %s -> '%s'",
                             entry.token, entry.original[:20])

        return result

    def clear(self) -> None:
        """Clear the mapping dictionary after demasking is complete."""
        count = len(self._mapping)
        self._mapping.clear()
        logger.debug("Cleared %d mapping entries", count)

    @property
    def mapping(self) -> dict[str, MappingEntry]:
        """Get the current mapping dictionary (read-only access)."""
        return dict(self._mapping)

    @property
    def mapping_size(self) -> int:
        """Return the number of entries in the mapping."""
        return len(self._mapping)
