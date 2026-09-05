"""Оркестратор пайплайна для PrivacyGuard.

Координирует: детекция -> маскирование -> LLM -> демаскирование.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from privacyguard_pipeline.audit import AuditLogger, audit_logger
from privacyguard_pipeline.config import settings
from privacyguard_pipeline.constants import LOG_PREVIEW_CHARS
from privacyguard_pipeline.detection import DetectionResult, PIIDetector
from privacyguard_pipeline.exceptions import (
    LLMConnectionError,
    TextTooLongError,
)
from privacyguard_pipeline.llm_proxy import LLMProxy
from privacyguard_pipeline.masker import Masker

logger = logging.getLogger(__name__)


class PrivacyGuardPipeline:
    """Оркеструет детекцию PII, маскирование, LLM-прокси и демаскирование.

    Использование:
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
    # Публичный API
    # ------------------------------------------------------------------

    async def process(
        self,
        text: str,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Прогоняет полный пайплайн анонимизация -> LLM -> деанонимизация.

        Args:
            text: Входной текст, который может содержать PII.
            system_prompt: Опциональный системный промпт для LLM.

        Returns:
            Словарь с ключами:
                - anonymized_text: Текст с PII, заменённым на токены.
                - llm_response: Ответ LLM с восстановленным PII.
                - stats: Статистика сессии.

        Raises:
            TextTooLongError: Если текст превышает максимально
                допустимую длину.
        """
        self._validate_text_length(text)
        start_time = time.time()

        # Шаг 1: Детекция PII
        detection_result, entity_types = self._detect_pii(text)

        # Шаг 2: Маскирование
        anonymized_text = self._mask_text(text, detection_result)

        try:
            # Шаг 3: Отправка в LLM
            llm_response, llm_success, llm_error = await self._call_llm(
                anonymized_text,
                system_prompt,
            )

            # Логируем запрос к LLM
            self.audit.log_llm_request(
                anonymized_text,
                success=llm_success,
                error=llm_error,
            )

            # Шаг 4: Демаскирование ответа LLM
            llm_response = self._demask_response(llm_response)
        finally:
            # Шаг 5: Очистка соответствий (безопасность: соответствия
            # существуют только в памяти). Выполняется, даже если вызов
            # LLM отменён или выбросил исключение, поэтому соответствие
            # посреди запроса никогда не переживает до следующего
            # вызова.
            self.masker.clear()

        elapsed = time.time() - start_time
        logger.info('Pipeline completed in %.2f ms', elapsed * 1000)

        return {
            'anonymized_text': anonymized_text,
            'llm_response': llm_response,
            'stats': {
                **self.audit.get_stats(),
                'processing_time_ms': round(elapsed * 1000, 2),
                'entities_detected': len(entity_types),
                'entity_types': dict(detection_result.layer_stats),
            },
        }

    async def close(self) -> None:
        """Освобождает ресурсы."""
        await self.llm_proxy.close()
        logger.info('Pipeline resources cleaned up')

    # ------------------------------------------------------------------
    # Приватные шаги
    # ------------------------------------------------------------------

    def _validate_text_length(self, text: str) -> None:
        """Проверяет, что длина текста не превышает максимум.

        Args:
            text: Входной текст для проверки.

        Raises:
            TextTooLongError: Если текст превышает максимально
                допустимую длину.
        """
        if len(text) > settings.max_text_length:
            raise TextTooLongError(
                f'Text length {len(text)} exceeds maximum '
                f'{settings.max_text_length}',
            )

    def _detect_pii(
        self,
        text: str,
    ) -> tuple[DetectionResult, list[str]]:
        """Прогоняет детекцию PII по входному тексту.

        Args:
            text: Входной текст для сканирования.

        Returns:
            Кортеж (detection_result, список_типов_сущностей).
        """
        logger.info('Starting PII detection')
        detection_result = self.detector.detect(text)
        entity_types = [s.entity_type for s in detection_result.spans]

        if entity_types:
            self.audit.log_detection(entity_types, len(text))
            logger.info(
                'Detected %d PII entities: %s',
                len(entity_types),
                dict(detection_result.layer_stats),
            )
        else:
            logger.info('No PII detected')

        return detection_result, entity_types

    def _mask_text(
        self,
        text: str,
        detection_result: DetectionResult,
    ) -> str:
        """Заменяет обнаруженные PII-спаны маскирующими токенами.

        Args:
            text: Исходный текст.
            detection_result: Обнаруженные PII-спаны.

        Returns:
            Текст с PII, заменённым на токены.
        """
        logger.info('Masking PII spans')
        anonymized_text = self.masker.mask(text, detection_result.spans)
        logger.debug(
            'Masked text (%d chars): %s',
            len(anonymized_text),
            anonymized_text[:LOG_PREVIEW_CHARS],
        )
        return anonymized_text

    async def _call_llm(
        self,
        anonymized_text: str,
        system_prompt: str | None = None,
    ) -> tuple[str, bool, str | None]:
        """Отправляет анонимизированный текст в API LLM.

        Реализует корректную деградацию: если вызов LLM не удался,
        возвращает пустой ответ с деталями ошибки.

        Args:
            anonymized_text: Текст с PII, заменённым на токены.
            system_prompt: Опциональный системный промпт для LLM.

        Returns:
            Кортеж (llm_response, флаг_успеха, сообщение_об_ошибке_или_None).
        """
        if not settings.has_any_llm_key:
            logger.warning(
                'No LLM API key configured. Skipping LLM request.',
            )
            return '', False, None

        try:
            logger.info('Sending to LLM API')
            response = await self.llm_proxy.send(
                anonymized_text,
                system_prompt,
            )
            logger.info('LLM response received (%d chars)', len(response))
            return response, True, None
        except LLMConnectionError as exc:
            logger.error('LLM API error: %s', exc)
            return '', False, str(exc)
        except Exception as exc:
            logger.error('Unexpected LLM error: %s', exc)
            return '', False, str(exc)

    def _demask_response(self, llm_response: str) -> str:
        """Восстанавливает исходные значения PII в ответе LLM.

        Args:
            llm_response: Ответ LLM, возможно содержащий маскирующие
                токены.

        Returns:
            Ответ LLM с токенами, заменёнными на исходные значения.
        """
        if not llm_response:
            return llm_response

        logger.info('Demasking LLM response')
        return self.masker.demask(llm_response)
