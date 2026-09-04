"""LLMProxy must classify a malformed 200 OK response body the same
way as a transport failure (LLMConnectionError per its documented
contract), not leak a raw KeyError/IndexError.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from privacyguard_pipeline.config import settings
from privacyguard_pipeline.exceptions import LLMConnectionError
from privacyguard_pipeline.llm_proxy import LLMProxy


def _client_returning(json_body: dict[str, Any]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=json_body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestMalformedResponseClassification:
    async def test_openai_empty_choices_raises_llm_connection_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings, 'llm_provider', 'openai')
        monkeypatch.setattr(settings, 'openai_api_key', 'test-key')

        proxy = LLMProxy()
        proxy._http_client = _client_returning({'choices': []})

        with pytest.raises(LLMConnectionError):
            await proxy.send('anonymized text')

    async def test_claude_empty_content_raises_llm_connection_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings, 'llm_provider', 'claude')
        monkeypatch.setattr(settings, 'claude_api_key', 'test-key')

        proxy = LLMProxy()
        proxy._http_client = _client_returning({'content': []})

        with pytest.raises(LLMConnectionError):
            await proxy.send('anonymized text')

    async def test_openai_missing_message_key_raises_llm_connection_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings, 'llm_provider', 'openai')
        monkeypatch.setattr(settings, 'openai_api_key', 'test-key')

        proxy = LLMProxy()
        proxy._http_client = _client_returning(
            {'choices': [{'unexpected': 'shape'}]},
        )

        with pytest.raises(LLMConnectionError):
            await proxy.send('anonymized text')
