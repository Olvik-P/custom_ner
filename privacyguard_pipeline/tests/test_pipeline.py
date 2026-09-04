"""PrivacyGuardPipeline lifecycle tests: the PII mapping must never
survive past the single request that created it, regardless of how
that request terminates.
"""

from __future__ import annotations

import asyncio

import pytest
from privacyguard_pipeline.detection import PIIDetector
from privacyguard_pipeline.llm_proxy import LLMProxy
from privacyguard_pipeline.masker import Masker
from privacyguard_pipeline.pipeline import PrivacyGuardPipeline


class _HangingLLMProxy(LLMProxy):
    """Fake LLMProxy whose send() blocks until the caller cancels it."""

    def __init__(self) -> None:
        pass  # skip LLMProxy.__init__: no real settings/client needed

    async def send(
        self,
        anonymized_text: str,
        system_prompt: str | None = None,
    ) -> str:
        await asyncio.Event().wait()
        return ''  # pragma: no cover - never reached

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
        await asyncio.sleep(0)  # let it reach the hanging send() call
        assert masker.mapping  # sanity: masking already happened

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
        await asyncio.sleep(0)  # let it reach the hanging send() call
        assert masker.mapping
        assert all(
            entry.original not in ('Иван', 'Петров')
            for entry in masker.mapping.values()
        )

        second_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second_task
        assert masker.mapping == {}
