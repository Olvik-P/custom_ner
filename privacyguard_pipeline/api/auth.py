"""Зависимость аутентификации по API-ключу для HTTP API.

Сравнивает заголовок X-API-Key с Settings.api_key через сравнение за
константное время, чтобы избежать тайминг-атаки на проверку ключа.
Отключение требования (api_key_required=False) предназначено только
для локальной разработки, поэтому всегда пишет заметное предупреждение
при старте, а не остаётся об этом умолчанием.
"""

from __future__ import annotations

import hmac
import logging

from fastapi import Header, HTTPException, status

from privacyguard_pipeline.config import settings

logger = logging.getLogger(__name__)


def warn_if_auth_disabled() -> None:
    """Пишет заметное предупреждение, если аутентификация по ключу отключена.

    Вызывается один раз при старте сервера, чтобы dev-only конфигурация
    никогда молча не попала на настоящий деплой.
    """
    if not settings.api_key_required:
        logger.warning(
            'API_KEY_REQUIRED=false — the HTTP API is accepting '
            'requests WITHOUT an API key. This is intended for local '
            'development only; do not run this configuration where '
            'PII could be exposed to untrusted callers.',
        )


async def require_api_key(
    x_api_key: str | None = Header(default=None),
) -> None:
    """Зависимость FastAPI, требующая заголовок X-API-Key.

    Args:
        x_api_key: Значение заголовка запроса X-API-Key, если оно
            присутствует.

    Raises:
        HTTPException: 401, если ключ отсутствует или неверен, а
            api_key_required равен True.
    """
    if not settings.api_key_required:
        return

    if not x_api_key or not hmac.compare_digest(
        x_api_key,
        settings.api_key,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Missing or invalid API key',
        )
