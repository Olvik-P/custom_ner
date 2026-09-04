"""API-key authentication dependency for the HTTP API.

Compares the X-API-Key header against Settings.api_key using a
constant-time comparison, to avoid a timing side-channel on the key
check. Disabling the requirement (api_key_required=False) is meant for
local/dev use only, so it always logs an always-visible startup
warning rather than staying silent about it.
"""

from __future__ import annotations

import hmac
import logging

from fastapi import Header, HTTPException, status

from privacyguard_pipeline.config import settings

logger = logging.getLogger(__name__)


def warn_if_auth_disabled() -> None:
    """Log a prominent warning if API key auth is disabled.

    Called once at server startup so a dev-only configuration never
    silently reaches a real deployment.
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
    """FastAPI dependency enforcing the X-API-Key header.

    Args:
        x_api_key: Value of the X-API-Key request header, if present.

    Raises:
        HTTPException: 401 if the key is missing or invalid and
            api_key_required is True.
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
