"""Pipeline orchestrator for PrivacyGuard.

Coordinates: detection -> masking -> LLM -> demasking.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from privacyguard_pipeline.audit import AuditLogger, audit_logger
from privacyguard_pipeline.config import settings
from privacyguard_pipeline.exceptions import (
    LLMConnectionError,
    TextTooLongError
)
from privacyguard_pipeline.llm_proxy import LLMProxy
from privacyguard_pipeline.masker import Masker
from privacyguard_pipeline.pii_detector import DetectionResult, PIIDetector

logger = logging.getLogger(__name__)


class PrivacyGuardPipeline:
    """Orchestrates PII detection, masking, LLM proxy, and demasking.

    Usage:
        pipeline = PrivacyGuardPipeline()
        result = await pipeline.process("Some text with PII")
    """

    def __init__(
        self,
        detector: PIIDetector | None = None,
        masker: Masker | None = None,
        llm_proxy: LLMProxy | None = None,
        audit: AuditLogger | None = None,
    ) -> None:
        self.detector = detector or PIIDetector()
        self.masker = masker or Masker()
        self.llm_proxy = llm_proxy or LLMProxy()
        self.audit = audit or audit_logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process(
        self,
        text: str,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Run the full anonymization -> LLM -> deanonymization pipeline.

        Args:
            text: Input text that may contain PII.
            system_prompt: Optional system prompt for the LLM.

        Returns:
            Dictionary with keys:
                - anonymized_text: Text with PII replaced by tokens.
                - llm_response: LLM response with PII restored.
                - stats: Session statistics.

        Raises:
            TextTooLongError: If text exceeds maximum allowed length.
        """
        self._validate_text_length(text)
        start_time = time.time()

        # Step 1: Detect PII
        detection_result, entity_types = self._detect_pii(text)

        # Step 2: Mask
        anonymized_text = self._mask_text(text, detection_result)

        # Step 3: Send to LLM
        llm_response, llm_success, llm_error = await self._call_llm(
            anonymized_text,
            system_prompt,
        )

        # Log LLM request
        self.audit.log_llm_request(
            anonymized_text,
            success=llm_success,
            error=llm_error,
        )

        # Step 4: Demask LLM response
        llm_response = self._demask_response(llm_response)

        # Step 5: Clear mapping (security: mapping exists only in memory)
        self.masker.clear()

        elapsed = time.time() - start_time
        logger.info("Pipeline completed in %.2f ms", elapsed * 1000)

        return {
            "anonymized_text": anonymized_text,
            "llm_response": llm_response,
            "stats": {
                **self.audit.get_stats(),
                "processing_time_ms": round(elapsed * 1000, 2),
                "entities_detected": len(entity_types),
                "entity_types": dict(detection_result.layer_stats),
            },
        }

    async def close(self) -> None:
        """Clean up resources."""
        await self.llm_proxy.close()
        logger.info("Pipeline resources cleaned up")

    # ------------------------------------------------------------------
    # Private steps
    # ------------------------------------------------------------------

    def _validate_text_length(self, text: str) -> None:
        """Validate that text length does not exceed the maximum.

        Args:
            text: Input text to validate.

        Raises:
            TextTooLongError: If text exceeds maximum allowed length.
        """
        if len(text) > settings.max_text_length:
            raise TextTooLongError(
                f"Text length {len(text)} exceeds maximum "
                f"{settings.max_text_length}",
            )

    def _detect_pii(
        self,
        text: str,
    ) -> tuple[DetectionResult, list[str]]:
        """Run PII detection on the input text.

        Args:
            text: Input text to scan.

        Returns:
            Tuple of (detection_result, list_of_entity_types).
        """
        logger.info("Starting PII detection")
        detection_result = self.detector.detect(text)
        entity_types = [s.entity_type for s in detection_result.spans]

        if entity_types:
            self.audit.log_detection(entity_types, len(text))
            logger.info(
                "Detected %d PII entities: %s",
                len(entity_types),
                dict(detection_result.layer_stats),
            )
        else:
            logger.info("No PII detected")

        return detection_result, entity_types

    def _mask_text(
        self,
        text: str,
        detection_result: DetectionResult,
    ) -> str:
        """Replace detected PII spans with masking tokens.

        Args:
            text: Original text.
            detection_result: Detected PII spans.

        Returns:
            Text with PII replaced by tokens.
        """
        logger.info("Masking PII spans")
        anonymized_text = self.masker.mask(text, detection_result.spans)
        logger.debug(
            "Masked text (%d chars): %s",
            len(anonymized_text),
            anonymized_text[:100],
        )
        return anonymized_text

    async def _call_llm(
        self,
        anonymized_text: str,
        system_prompt: str | None = None,
    ) -> tuple[str, bool, str | None]:
        """Send anonymized text to the LLM API.

        Implements graceful degradation: if the LLM call fails,
        returns an empty response with error details.

        Args:
            anonymized_text: Text with PII replaced by tokens.
            system_prompt: Optional system prompt for the LLM.

        Returns:
            Tuple of (llm_response, success_flag, error_message_or_None).
        """
        if not settings.has_any_llm_key:
            logger.warning(
                "No LLM API key configured. Skipping LLM request.",
            )
            return "", False, None

        try:
            logger.info("Sending to LLM API")
            response = await self.llm_proxy.send(
                anonymized_text,
                system_prompt,
            )
            logger.info("LLM response received (%d chars)", len(response))
            return response, True, None
        except LLMConnectionError as exc:
            logger.error("LLM API error: %s", exc)
            return "", False, str(exc)
        except Exception as exc:
            logger.error("Unexpected LLM error: %s", exc)
            return "", False, str(exc)

    def _demask_response(self, llm_response: str) -> str:
        """Restore original PII values in the LLM response.

        Args:
            llm_response: LLM response possibly containing masking tokens.

        Returns:
            LLM response with tokens replaced by original values.
        """
        if not llm_response:
            return llm_response

        logger.info("Demasking LLM response")
        return self.masker.demask(llm_response)
