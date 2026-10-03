"""Новые типы идентификаторов в общих механизмах пакета."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from privacyguard_pipeline.detection import PIIDetector
from privacyguard_pipeline.masker import Masker
from privacyguard_pipeline.pdf import PDFAnonymizer


def _found(
    detector: PIIDetector,
    text: str,
    **kwargs: object,
) -> list[tuple[str, str]]:
    result = detector.detect(text, **kwargs)  # type: ignore[arg-type]
    return [(s.text, s.entity_type) for s in result.spans]


def test_masking_round_trip_uses_the_new_token_type(
    detector: PIIDetector,
) -> None:
    text = 'машина А123ВС77 у подъезда'
    masker = Masker()
    masked = masker.mask(text, detector.detect(text).spans)

    assert '<CAR_PLATE_' in masked
    assert 'А123ВС77' not in masked
    assert masker.demask(masked) == text


def test_threshold_zero_adds_candidates_default_threshold_hides(
    detector: PIIDetector,
) -> None:
    text = 'отправление 12345678901234567890 передано'

    assert _found(detector, text) == []
    assert ('12345678901234567890', 'BANK_ACCOUNT') in _found(
        detector,
        text,
        min_confidence=0.0,
    )


def test_allow_list_suppresses_a_new_type(detector: PIIDetector) -> None:
    text = 'машина А123ВС77 у подъезда'

    assert _found(detector, text, allow_list=['А123ВС77']) == []


def test_deny_list_wins_over_allow_list_for_a_new_type(
    detector: PIIDetector,
) -> None:
    got = _found(
        detector,
        'машина А123ВС77 у подъезда',
        allow_list=['А123ВС77'],
        deny_list=['А123ВС77'],
    )

    assert got == [('А123ВС77', 'CUSTOM')]


@pytest.mark.parametrize(
    ('text', 'entity_type'),
    [
        ('паспорт 45 10 123456', 'PASSPORT'),
        ('ИНН 7707083893', 'INN'),
        ('СНИЛС 112-233-445 95', 'SNILS'),
        ('карта 4111 1111 1111 1111', 'CARD'),
        ('тел. 8 916 123-45-67', 'PHONE'),
        ('почта a.b@example.com', 'EMAIL'),
        ('ОГРН 1027700132195', 'OGRN'),
    ],
)
def test_existing_types_are_unchanged(
    detector: PIIDetector,
    text: str,
    entity_type: str,
) -> None:
    got = _found(detector, text)

    assert entity_type in {t for _, t in got}


def _make_pdf(path: Path, font: str, lines: list[str]) -> None:
    doc = fitz.open()
    page = doc.new_page()
    y = 100
    for line in lines:
        page.insert_text((72, y), line, fontfile=font, fontname='F1')
        y += 30
    doc.save(str(path))
    doc.close()


def test_pdf_entity_types_filter_accepts_new_type(
    tmp_path: Path,
    cyrillic_font_path: str,
    detector: PIIDetector,
) -> None:
    source = tmp_path / 'in.pdf'
    _make_pdf(
        source,
        cyrillic_font_path,
        ['машина А123ВС77 у подъезда', 'тел. 8 916 123-45-67'],
    )

    result = PDFAnonymizer(detector=detector).anonymize(
        source,
        tmp_path / 'out.pdf',
        entity_types=['CAR_PLATE'],
    )

    assert result.success
    assert result.redacted_by_type == {'CAR_PLATE': 1}
    with fitz.open(str(tmp_path / 'out.pdf')) as out:
        text = out[0].get_text()
    assert 'А123ВС77' not in text
    assert '916' in text
