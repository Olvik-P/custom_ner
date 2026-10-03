"""Переопределение порога уверенности через HTTP API на один запрос."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from privacyguard_pipeline.api.app import create_app
from privacyguard_pipeline.config import settings
from privacyguard_pipeline.pipeline import PrivacyGuardPipeline

_TEXT = 'номер 4510123456'


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(settings, 'api_key_required', False)

    async def _no_llm(
        self: PrivacyGuardPipeline,
        anonymized_text: str,
        system_prompt: str | None = None,
    ) -> tuple[str, bool, str | None]:
        return anonymized_text, True, None

    monkeypatch.setattr(PrivacyGuardPipeline, '_call_llm', _no_llm)
    with TestClient(create_app()) as test_client:
        yield test_client


def _anonymize(client: TestClient, **extra: Any) -> dict[str, Any]:
    response = client.post('/v1/anonymize', json={'text': _TEXT, **extra})
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def test_default_threshold_leaves_bare_digits_unmasked(
    client: TestClient,
) -> None:
    assert _anonymize(client)['anonymized_text'] == _TEXT


def test_min_confidence_zero_masks_the_candidate(client: TestClient) -> None:
    body = _anonymize(client, min_confidence=0)

    assert '4510123456' not in body['anonymized_text']


def test_override_does_not_leak_into_next_request(
    client: TestClient,
) -> None:
    _anonymize(client, min_confidence=0)

    assert _anonymize(client)['anonymized_text'] == _TEXT


@pytest.mark.parametrize('value', [-0.1, 1.5])
def test_out_of_range_value_is_rejected(
    client: TestClient,
    value: float,
) -> None:
    response = client.post(
        '/v1/anonymize',
        json={'text': _TEXT, 'min_confidence': value},
    )

    assert response.status_code == 422


@pytest.mark.parametrize('value', ['-0.1', '1.5'])
def test_pdf_endpoint_rejects_out_of_range_value(
    client: TestClient,
    value: str,
) -> None:
    response = client.post(
        '/v1/anonymize/pdf',
        files={'file': ('x.pdf', b'%PDF-1.4', 'application/pdf')},
        data={'min_confidence': value},
    )

    assert response.status_code == 422
