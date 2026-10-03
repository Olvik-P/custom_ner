"""Списки allow/deny в теле запроса HTTP API."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from privacyguard_pipeline.api.app import create_app
from privacyguard_pipeline.config import settings
from privacyguard_pipeline.pipeline import PrivacyGuardPipeline

_TEXT = 'идёт Проект Заря, писать на support@corp.example'


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


def _post(client: TestClient, **extra: Any) -> Any:
    return client.post('/v1/anonymize', json={'text': _TEXT, **extra})


def test_deny_list_masks_custom_string(client: TestClient) -> None:
    response = _post(client, deny_list=['Проект Заря'])

    assert response.status_code == 200
    assert 'Заря' not in response.json()['anonymized_text']


def test_allow_list_leaves_email_unmasked(client: TestClient) -> None:
    baseline = _post(client).json()['anonymized_text']
    assert 'support@corp.example' not in baseline

    response = _post(client, allow_list=['support@corp.example'])

    assert 'support@corp.example' in response.json()['anonymized_text']


def test_request_lists_do_not_affect_next_request(
    client: TestClient,
) -> None:
    _post(
        client,
        allow_list=['support@corp.example'],
        deny_list=['Проект Заря'],
    )

    later = _post(client).json()['anonymized_text']

    assert 'support@corp.example' not in later


@pytest.mark.parametrize(
    'extra',
    [
        {'deny_list': ['']},
        {'allow_list': ['   ']},
        {'deny_list': ['а' * 201]},
        {'deny_list': ['x'] * 501},
    ],
)
def test_invalid_entries_are_rejected_with_422(
    client: TestClient,
    extra: dict[str, Any],
) -> None:
    response = _post(client, **extra)

    assert response.status_code == 422


def test_error_does_not_echo_entry_text(client: TestClient) -> None:
    secret = 'секретная запись'
    response = _post(client, deny_list=[secret, ''])

    assert response.status_code == 422
    assert secret not in response.text


def test_pdf_endpoint_rejects_invalid_list_before_reading_file(
    client: TestClient,
) -> None:
    response = client.post(
        '/v1/anonymize/pdf',
        files={'file': ('x.pdf', b'%PDF-1.4', 'application/pdf')},
        data={'deny_list': ['']},
    )

    assert response.status_code == 422
