"""FastAPI app factory for the PrivacyGuard HTTP API.

PIIDetector and LLMProxy are constructed once here, at startup, and
shared across all requests (see design.md - Decision 1): neither holds
PII-derived state between calls, and constructing them per-request
would reload Natasha's models (PIIDetector) or drop HTTP connection
pooling (LLMProxy) on every call. Only the per-request Masker (built
in routes.py) needs to be request-scoped, since it's the one that
holds the sensitive token -> value mapping.
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
    """Build the FastAPI application."""
    app = FastAPI(
        title='PrivacyGuard Pipeline API',
        version='1.2.0',
        lifespan=_lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
