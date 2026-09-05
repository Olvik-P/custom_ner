"""Тесты жизненного цикла PrivacyGuardPipeline: соответствие PII никогда
не должно пережить единственный запрос, который его создал, независимо
от того, как этот запрос завершается.
"""

from __future__ import annotations

import asyncio

import pytest

from privacyguard_pipeline.detection import PIIDetector
from privacyguard_pipeline.llm_proxy import LLMProxy
from privacyguard_pipeline.masker import Masker
from privacyguard_pipeline.pipeline import PrivacyGuardPipeline


class _HangingLLMProxy(LLMProxy):
    """Фейковый LLMProxy, чей send() блокируется, пока его не отменят."""

    def __init__(self) -> None:
        pass  # пропускаем LLMProxy.__init__: реальные settings/client не нужны

    async def send(
        self,
        anonymized_text: str,
        system_prompt: str | None = None,
    ) -> str:
        await asyncio.Event().wait()
        return ''  # pragma: no cover - никогда не достигается

    async def close(self) -> None:
        pass


class TestMaskerClearedOnCancellation:
    async def test_mapping_cleared_when_llm_call_is_cancelled(
        self,
        detector: PIIDetector,
    ) -> None:
        masker = Masker()
        pipeline = PrivacyGuardPipeline(
            detector=detector,
            masker=masker,
            llm_proxy=_HangingLLMProxy(),
        )

        task = asyncio.ensure_future(
            pipeline.process('Меня зовут Иван Петров'),
        )
        await asyncio.sleep(0)  # даём дойти до зависающего вызова send()
        assert masker.mapping  # проверка: маскирование уже произошло

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert masker.mapping == {}


class TestMappingDoesNotCarryOverBetweenRequests:
    async def test_second_request_only_maps_its_own_pii(
        self,
        detector: PIIDetector,
    ) -> None:
        masker = Masker()
        pipeline = PrivacyGuardPipeline(
            detector=detector,
            masker=masker,
            llm_proxy=_HangingLLMProxy(),
        )

        first_task = asyncio.ensure_future(
            pipeline.process('Первый: Иван Петров'),
        )
        await asyncio.sleep(0)
        first_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_task
        assert masker.mapping == {}

        second_task = asyncio.ensure_future(
            pipeline.process('Второй: Мария Смирнова'),
        )
        await asyncio.sleep(0)  # даём дойти до зависающего вызова send()
        assert masker.mapping
        assert all(
            entry.original not in ('Иван', 'Петров')
            for entry in masker.mapping.values()
        )

        second_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second_task
        assert masker.mapping == {}
