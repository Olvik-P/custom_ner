"""allow/deny-списки в PrivacyGuardPipeline и PDFAnonymizer."""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest

from privacyguard_pipeline.detection import PIIDetector
from privacyguard_pipeline.exceptions import InvalidListEntryError
from privacyguard_pipeline.masker import Masker
from privacyguard_pipeline.pdf import PDFAnonymizer
from privacyguard_pipeline.pipeline import PrivacyGuardPipeline

_TEXT = 'идёт проект Заря, писать на support@corp.example'


@pytest.fixture
def pipeline(
    detector: PIIDetector,
    monkeypatch: pytest.MonkeyPatch,
) -> PrivacyGuardPipeline:
    async def _echo(
        self: PrivacyGuardPipeline,
        anonymized_text: str,
        system_prompt: str | None = None,
    ) -> tuple[str, bool, str | None]:
        return anonymized_text, True, None

    monkeypatch.setattr(PrivacyGuardPipeline, '_call_llm', _echo)
    return PrivacyGuardPipeline(detector=detector, masker=Masker())


async def test_pipeline_applies_lists_and_stats_hold_no_entries(
    pipeline: PrivacyGuardPipeline,
) -> None:
    result = await pipeline.process(
        _TEXT,
        allow_list=['support@corp.example'],
        deny_list=['Проект Заря'],
    )

    assert 'Заря' not in result['anonymized_text']
    assert 'support@corp.example' in result['anonymized_text']
    dumped = json.dumps(result['stats'], ensure_ascii=False)
    assert 'Заря' not in dumped
    assert 'support@corp.example' not in dumped
    # Ответ LLM (эхо) демаскирован обратно в исходный текст.
    assert result['llm_response'] == _TEXT


async def test_pipeline_lists_do_not_leak_into_next_call(
    pipeline: PrivacyGuardPipeline,
) -> None:
    await pipeline.process(_TEXT, deny_list=['Проект Заря'])

    later = await pipeline.process(_TEXT)

    assert '<CUSTOM_' not in later['anonymized_text']


async def test_pipeline_rejects_invalid_list_before_llm(
    pipeline: PrivacyGuardPipeline,
) -> None:
    with pytest.raises(InvalidListEntryError):
        await pipeline.process(_TEXT, deny_list=[''])


def _make_pdf(path: Path, font: str, text: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), text, fontfile=font, fontname='F1')
    doc.save(str(path))
    doc.close()


def test_pdf_deny_list_redacts_entry(
    tmp_path: Path,
    cyrillic_font_path: str,
    detector: PIIDetector,
) -> None:
    source = tmp_path / 'in.pdf'
    _make_pdf(source, cyrillic_font_path, 'Внутри проекта Заря идут работы')

    result = PDFAnonymizer(detector=detector).anonymize(
        source,
        tmp_path / 'out.pdf',
        deny_list=['Проект Заря'],
    )

    assert result.success
    assert result.redacted_by_type
    with fitz.open(str(tmp_path / 'out.pdf')) as out:
        assert 'Заря' not in out[0].get_text()


def test_pdf_invalid_list_is_caller_error_not_failed_result(
    tmp_path: Path,
    cyrillic_font_path: str,
    detector: PIIDetector,
) -> None:
    source = tmp_path / 'in.pdf'
    _make_pdf(source, cyrillic_font_path, 'текст')
    out = tmp_path / 'out.pdf'

    with pytest.raises(InvalidListEntryError):
        PDFAnonymizer(detector=detector).anonymize(
            source,
            out,
            allow_list=['   '],
        )

    assert not out.exists()
