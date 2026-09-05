"""Модуль маскирования и демаскирования для PrivacyGuard Pipeline.

Генерирует уникальные токены для PII-спанов и управляет словарём
соответствий. Все замены выполняются от конца к началу, чтобы сохранить
корректность индексов.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from privacyguard_pipeline.constants import TOKEN_HEX_LENGTH
from privacyguard_pipeline.detection import PIISpan

logger = logging.getLogger(__name__)


@dataclass
class MappingEntry:
    """Одна запись соответствия токен-оригинал.

    Attributes:
        token: Маскирующий токен (например, ``<PHONE_A7B3C902>``).
        original: Исходное значение PII.
        entity_type: Тип PII-сущности.
    """

    token: str
    original: str
    entity_type: str


class Masker:
    """Маскирует PII-спаны уникальными токенами и восстанавливает значения.

    Словарь соответствий существует только в памяти и очищается после
    демаскирования.

    Attributes:
        mapping: Словарь, сопоставляющий токены объектам MappingEntry.
    """

    def __init__(self) -> None:
        self._mapping: dict[str, MappingEntry] = {}

    # ------------------------------------------------------------------
    # Генерация токенов
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_token(entity_type: str) -> str:
        """Генерирует уникальный маскирующий токен.

        Args:
            entity_type: Тип PII-сущности (PHONE, PER, LOC и т.д.).

        Returns:
            Строка токена в формате ``<TYPE_UUID8>``.
        """
        short_uuid = uuid.uuid4().hex[:TOKEN_HEX_LENGTH].upper()
        return f'<{entity_type}_{short_uuid}>'

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def mask(self, text: str, spans: list[PIISpan]) -> str:
        """Заменяет все PII-спаны маскирующими токенами.

        Выполняет замены от конца к началу, чтобы сохранить индексы.

        Args:
            text: Исходный текст.
            spans: Обнаруженные PII-спаны, отсортированные по позиции.

        Returns:
            Текст с PII, заменённым на токены.
        """
        # Сортируем спаны от конца к началу, чтобы сохранить
        # корректность индексов
        sorted_spans = sorted(spans, key=lambda s: s.end, reverse=True)

        result = text
        for span in sorted_spans:
            token = self._generate_token(span.entity_type)
            self._mapping[token] = MappingEntry(
                token=token,
                original=span.text,
                entity_type=span.entity_type,
            )
            result = result[: span.start] + token + result[span.end :]
            logger.debug("Masked '%s' -> %s", span.text[:20], token)

        return result

    def demask(self, text: str) -> str:
        """Восстанавливает исходные значения из маскирующих токенов.

        Args:
            text: Текст, возможно содержащий маскирующие токены.

        Returns:
            Текст с токенами, заменёнными на исходные значения.
        """
        result = text
        for entry in self._mapping.values():
            if entry.token in result:
                result = result.replace(entry.token, entry.original)
                logger.debug(
                    "Demasked %s -> '%s'", entry.token, entry.original[:20]
                )

        return result

    def clear(self) -> None:
        """Очищает словарь соответствий после завершения демаскирования."""
        count = len(self._mapping)
        self._mapping.clear()
        logger.debug('Cleared %d mapping entries', count)

    @property
    def mapping(self) -> dict[str, MappingEntry]:
        """Возвращает текущий словарь соответствий (только для чтения)."""
        return dict(self._mapping)

    @property
    def mapping_size(self) -> int:
        """Возвращает число записей в словаре соответствий."""
        return len(self._mapping)
