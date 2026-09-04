"""Run the PrivacyGuard HTTP API server.

Usage:
    python -m privacyguard_pipeline.api
"""

from __future__ import annotations

import uvicorn

from privacyguard_pipeline.config import settings


def main() -> None:
    uvicorn.run(
        'privacyguard_pipeline.api.app:app',
        host='0.0.0.0',
        port=settings.api_port,
    )


if __name__ == '__main__':
    main()
