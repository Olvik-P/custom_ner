"""Модуль логирования и аудита для PrivacyGuard Pipeline.

Логирует события детекции PII (только типы, не значения), статистику
сессии и журнал запросов к LLM.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from privacyguard_pipeline.constants import AUDIT_PREVIEW_CHARS

logger = logging.getLogger(__name__)


def configure_structlog(log_dir: str | Path = 'logs') -> None:
    """Настраивает процессоры и фабрики structlog.

    Args:
        log_dir: Директория для файлов логов.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt='iso'),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


class AuditLogger:
    """Аудит-логирование событий детекции PII и запросов к LLM.

    Логирует только типы сущностей, никогда сами значения PII.
    Ведёт статистику на уровне сессии.
    """

    def __init__(self, log_dir: str | Path = 'logs') -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Статистика сессии
        self._entity_counts: Counter[str] = Counter()
        self._total_requests: int = 0
        self._successful_requests: int = 0
        self._failed_requests: int = 0
        self._session_start: float = time.time()

        # Настройка structlog
        configure_structlog(log_dir)

        self._audit_logger = structlog.get_logger('privacyguard.audit')
        self._llm_logger = structlog.get_logger('privacyguard.llm')

    # ------------------------------------------------------------------
    # Методы логирования
    # ------------------------------------------------------------------

    def log_detection(
        self,
        entity_types: list[str],
        text_length: int,
    ) -> None:
        """Логирует событие детекции PII.

        Логирует только типы сущностей, никогда сами значения PII.

        Args:
            entity_types: Список обнаруженных типов PII-сущностей.
            text_length: Длина обработанного текста.
        """
        type_counts = Counter(entity_types)
        self._entity_counts.update(type_counts)

        self._audit_logger.info(
            'pii_detected',
            entity_types=list(type_counts.keys()),
            entity_counts=dict(type_counts),
            text_length=text_length,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def log_llm_request(
        self,
        anonymized_text: str,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Логирует запрос к API LLM.

        Args:
            anonymized_text: Анонимизированный промпт, отправленный LLM.
            success: Успешен ли запрос.
            error: Сообщение об ошибке, если запрос провалился.
        """
        self._total_requests += 1
        if success:
            self._successful_requests += 1
        else:
            self._failed_requests += 1

        log_data: dict[str, Any] = {
            'anonymized_prompt_preview': anonymized_text[:AUDIT_PREVIEW_CHARS],
            'success': success,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
        if error:
            log_data['error'] = error

        if success:
            self._llm_logger.info('llm_request', **log_data)
        else:
            self._llm_logger.error('llm_request_failed', **log_data)

    # ------------------------------------------------------------------
    # Статистика
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, int | float | dict[str, int]]:
        """Возвращает статистику сессии.

        Returns:
            Словарь со статистикой сессии.
        """
        session_duration = time.time() - self._session_start
        return {
            'session_duration_seconds': round(session_duration, 2),
            'total_requests': self._total_requests,
            'successful_requests': self._successful_requests,
            'failed_requests': self._failed_requests,
            'entity_counts': dict(self._entity_counts),
            'total_entities_detected': sum(self._entity_counts.values()),
        }

    def reset_stats(self) -> None:
        """Сбрасывает статистику сессии."""
        self._entity_counts.clear()
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._session_start = time.time()
        logger.debug('Session statistics reset')


# Модульный синглтон
audit_logger = AuditLogger()
