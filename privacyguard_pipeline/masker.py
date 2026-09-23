"""Модуль маскирования и демаскирования для PrivacyGuard Pipeline.

Генерирует токены для PII-спанов, стабильные для одной и той же сущности
в пределах одного вызова :meth:`Masker.mask`, и управляет словарём
соответствий. Все замены выполняются от конца к началу, чтобы сохранить
корректность индексов.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass

from privacyguard_pipeline.constants import TOKEN_HEX_LENGTH
from privacyguard_pipeline.detection import PIISpan

logger = logging.getLogger(__name__)

# Организационно-правовые формы, отсекаемые с края нормализованного
# названия организации (ORG), чтобы "ООО Ромашка" и "Ромашка"
# нормализовались в одну и ту же сущность. Список отсортирован от более
# длинных форм к более коротким, чтобы избежать двусмысленности при
# сопоставлении регулярным выражением.
_ORG_LEGAL_FORMS = (
    'ооо',
    'зао',
    'оао',
    'пао',
    'нко',
    'чоу',
    'гуп',
    'муп',
    'ип',
    'ао',
)
_ORG_FORMS_PATTERN = '|'.join(_ORG_LEGAL_FORMS)
_ORG_LEGAL_FORM_PREFIX_RE = re.compile(rf'^(?:{_ORG_FORMS_PATTERN})\b\.?\s*')
_ORG_LEGAL_FORM_SUFFIX_RE = re.compile(rf'\s*\b(?:{_ORG_FORMS_PATTERN})\.?$')

# Небуквенные/нецифровые символы (пробелы, кавычки, пунктуация) с краёв
# нормализуемого текста.
_EDGE_NON_WORD_RE = re.compile(r'^[^\w]+|[^\w]+$')
_INNER_WHITESPACE_RE = re.compile(r'\s+')


def _strip_edges_and_collapse(text: str) -> str:
    """Убирает краевую пунктуацию/кавычки и схлопывает внутренние пробелы."""
    stripped = _EDGE_NON_WORD_RE.sub('', text)
    return _INNER_WHITESPACE_RE.sub(' ', stripped).strip()


def _normalize_entity_text(entity_type: str, text: str) -> str:
    """Нормализует текст сущности для сопоставления повторных упоминаний.

    Приводит к нижнему регистру и обрезает краевые пробелы/пунктуацию для
    всех типов сущностей; для ``entity_type == 'ORG'`` дополнительно
    отсекает организационно-правовую форму с края названия.

    Args:
        entity_type: Тип PII-сущности (PHONE, PER, ORG, LOC и т.д.).
        text: Исходный текст спана.

    Returns:
        Нормализованная строка, используемая как часть ключа
        переиспользования токена.
    """
    normalized = _strip_edges_and_collapse(text.lower())
    if entity_type == 'ORG':
        normalized = _ORG_LEGAL_FORM_PREFIX_RE.sub('', normalized)
        normalized = _ORG_LEGAL_FORM_SUFFIX_RE.sub('', normalized)
        normalized = _strip_edges_and_collapse(normalized)
    return normalized


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
    """Маскирует PII-спаны токенами и восстанавливает значения.

    Повторные упоминания одной и той же нормализованной сущности в
    пределах одного вызова :meth:`mask` получают один и тот же токен.
    Словари соответствий существуют только в памяти и очищаются после
    демаскирования.

    Attributes:
        mapping: Словарь, сопоставляющий токены объектам MappingEntry.
    """

    def __init__(self) -> None:
        self._mapping: dict[str, MappingEntry] = {}
        # Ключ - entity_key (entity_type + нормализованный текст),
        # значение - уже выданный этой сущности токен. Позволяет
        # повторным упоминаниям одной сущности в пределах одного
        # документа (одного вызова mask()) получать один и тот же токен.
        # Живёт и чистится синхронно с _mapping.
        self._entity_tokens: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Генерация токенов
    # ------------------------------------------------------------------

    @staticmethod
    def _entity_key(entity_type: str, normalized_text: str) -> str:
        """Строит ключ сущности для индекса переиспользования токенов."""
        return f'{entity_type}:{normalized_text}'

    @staticmethod
    def _generate_token(entity_type: str, entity_key: str) -> str:
        """Генерирует детерминированный маскирующий токен.

        Одинаковый ``entity_key`` всегда даёт один и тот же токен -
        используется ``hashlib`` (а не встроенный ``hash()``, который не
        детерминирован между запусками процесса).

        Args:
            entity_type: Тип PII-сущности (PHONE, PER, LOC и т.д.).
            entity_key: Ключ сущности (см. ``_entity_key``), из которого
                вычисляется хеш.

        Returns:
            Строка токена в формате ``<TYPE_HASH8>``.
        """
        digest = hashlib.sha256(entity_key.encode('utf-8')).hexdigest()
        short_hash = digest[:TOKEN_HEX_LENGTH].upper()
        return f'<{entity_type}_{short_hash}>'

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def mask(self, text: str, spans: list[PIISpan]) -> str:
        """Заменяет все PII-спаны маскирующими токенами.

        Токены для новых сущностей выдаются в порядке чтения (по
        возрастанию позиции), чтобы канонический ``original`` в
        ``mapping`` всегда соответствовал первому по тексту упоминанию
        сущности, а не тому спану, что случайно обработан первым при
        замене. Сама замена в тексте выполняется отдельным проходом от
        конца к началу, чтобы сохранить индексы.

        Args:
            text: Исходный текст.
            spans: Обнаруженные PII-спаны, отсортированные по позиции.

        Returns:
            Текст с PII, заменённым на токены.
        """
        for span in sorted(spans, key=lambda s: s.start):
            normalized = _normalize_entity_text(span.entity_type, span.text)
            entity_key = self._entity_key(span.entity_type, normalized)
            if entity_key not in self._entity_tokens:
                token = self._generate_token(span.entity_type, entity_key)
                self._entity_tokens[entity_key] = token
                self._mapping[token] = MappingEntry(
                    token=token,
                    original=span.text,
                    entity_type=span.entity_type,
                )

        # Сортируем спаны от конца к началу, чтобы сохранить
        # корректность индексов
        sorted_spans = sorted(spans, key=lambda s: s.end, reverse=True)

        result = text
        for span in sorted_spans:
            normalized = _normalize_entity_text(span.entity_type, span.text)
            entity_key = self._entity_key(span.entity_type, normalized)
            token = self._entity_tokens[entity_key]
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
        """Очищает словари соответствий после завершения демаскирования."""
        count = len(self._mapping)
        self._mapping.clear()
        self._entity_tokens.clear()
        logger.debug('Cleared %d mapping entries', count)

    @property
    def mapping(self) -> dict[str, MappingEntry]:
        """Возвращает текущий словарь соответствий (только для чтения)."""
        return dict(self._mapping)

    @property
    def mapping_size(self) -> int:
        """Возвращает число записей в словаре соответствий."""
        return len(self._mapping)
