"""Session-scoped реестр handle -> Masker для MCP-сервера.

В отличие от req-scoped Masker в http-api (создаётся и clear()-ится в
рамках одного HTTP-запроса — см. api/routes.py), здесь соответствие
токен -> значение обязано пережить хотя бы один вызов инструмента между
``mask`` и ``demask`` в рамках одного MCP-диалога. Поэтому Masker живёт
здесь, в реестре, привязанном к процессу MCP-сервера, а не к отдельному
вызову инструмента (design.md - Decision 1).

TTL защищает от случая, когда ``demask`` так и не был вызван (диалог
прерван, агент не вернул токены и т.п.) — без него сырой PII мог бы жить
в памяти процесса неограниченно долго (design.md - Decision 2).
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from privacyguard_pipeline.masker import Masker

logger = logging.getLogger(__name__)


class HandleNotFoundError(Exception):
    """handle неизвестен реестру.

    Возникает, если handle никогда не выдавался, уже был использован
    для успешного demask, либо истёк по TTL до вызова demask.
    """


@dataclass
class _RegistryEntry:
    masker: Masker
    created_at: float
    masked_file_path: Path | None = None


class MaskRegistry:
    """Хранит Masker'ы по непрозрачному handle, пока не будет вызван demask.

    Использование:
        registry = MaskRegistry(ttl_seconds=300)
        handle = registry.register(masker)
        ...
        masker = registry.pop(handle)  # снимает запись из реестра
    """

    def __init__(self, ttl_seconds: float) -> None:
        self._ttl_seconds = ttl_seconds
        self._entries: dict[str, _RegistryEntry] = {}

    def register(
        self,
        masker: Masker,
        masked_file_path: Path | None = None,
    ) -> str:
        """Регистрирует Masker под новым непрозрачным handle.

        Args:
            masker: Masker с уже выполненным mask() для этого вызова.
            masked_file_path: Путь к маскированному файлу на диске,
                связанному с этим handle (только для handle от
                mask_file) — удаляется автоматически при pop()/sweep().
                None для handle от обычного mask() над голым текстом.

        Returns:
            Непрозрачный handle, идентифицирующий эту запись для
            последующего pop().
        """
        self.sweep()
        handle = uuid.uuid4().hex
        self._entries[handle] = _RegistryEntry(
            masker=masker,
            created_at=time.monotonic(),
            masked_file_path=masked_file_path,
        )
        return handle

    def pop(self, handle: str) -> Masker:
        """Снимает и возвращает Masker по handle, делая его непригодным снова.

        Если с этим handle был связан маскированный файл на диске
        (см. register()), он удаляется здесь же — маскированный файл
        никогда не должен пережить свой handle (design.md - Decision 5).

        Args:
            handle: handle, ранее возвращённый register().

        Returns:
            Masker, зарегистрированный под этим handle.

        Raises:
            HandleNotFoundError: Если handle неизвестен, уже был снят
                предыдущим pop(), либо истёк по TTL.
        """
        self.sweep()
        entry = self._entries.pop(handle, None)
        if entry is None:
            raise HandleNotFoundError(handle)
        if entry.masked_file_path is not None:
            entry.masked_file_path.unlink(missing_ok=True)
        return entry.masker

    def sweep(self) -> int:
        """Удаляет все записи, просроченные по TTL.

        Вызывается как при каждом register()/pop() (немедленная
        гигиена), так и периодически фоновой задачей сервера (см.
        mcp_server/server.py) — на случай, если ни одного нового вызова
        register()/pop() долго не происходит. Если у просроченной записи
        был связанный маскированный файл на диске, он тоже удаляется —
        та же гарантия, что и у pop() (design.md - Decision 5), на
        случай, если handle от mask_file так и не был закрыт явно.

        Returns:
            Число удалённых просроченных записей.
        """
        now = time.monotonic()
        expired = [
            handle
            for handle, entry in self._entries.items()
            if now - entry.created_at >= self._ttl_seconds
        ]
        for handle in expired:
            entry = self._entries.pop(handle, None)
            if entry is not None and entry.masked_file_path is not None:
                entry.masked_file_path.unlink(missing_ok=True)
        if expired:
            logger.info(
                'Expired %d unused mask handle(s) after TTL',
                len(expired),
            )
        return len(expired)

    def __len__(self) -> int:
        return len(self._entries)
