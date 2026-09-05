"""Запускает MCP-сервер PrivacyGuard поверх stdio.

Использование:
    python -m privacyguard_pipeline.mcp_server

Предназначен для запуска локальным MCP-хостом (например, Claude
Desktop/Code) как дочерний процесс — не открывает сетевой порт и не
требует API-ключа: доверие обеспечивается тем, что процесс порождён
самим хостом (см. design.md - Decision 6).
"""

from __future__ import annotations

from privacyguard_pipeline.mcp_server.server import server


def main() -> None:
    server.run(transport='stdio')


if __name__ == '__main__':
    main()
