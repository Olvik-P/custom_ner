"""Фабрика FastAPI-приложения для HTTP API PrivacyGuard.

PIIDetector и LLMProxy создаются здесь один раз, при старте, и
переиспользуются во всех запросах (см. design.md - Decision 1): ни
один из них не хранит производное от PII состояние между вызовами, а
создание их на каждый запрос заново перезагружало бы модели Natasha
(PIIDetector) или теряло бы пул HTTP-соединений (LLMProxy) при каждом
вызове. Только per-request Masker (создаётся в routes.py) должен быть
привязан к запросу, так как именно он хранит чувствительное
соответствие токен -> значение.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from privacyguard_pipeline.api.auth import warn_if_auth_disabled
from privacyguard_pipeline.api.routes import router
from privacyguard_pipeline.detection import PIIDetector
from privacyguard_pipeline.llm_proxy import LLMProxy

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    warn_if_auth_disabled()
    app.state.detector = PIIDetector()
    app.state.llm_proxy = LLMProxy()
    logger.info('PrivacyGuard API started')
    try:
        yield
    finally:
        await app.state.llm_proxy.close()
        logger.info('PrivacyGuard API shut down')


def create_app() -> FastAPI:
    """Собирает приложение FastAPI."""
    app = FastAPI(
        title='PrivacyGuard Pipeline API',
        version='1.2.0',
        lifespan=_lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
