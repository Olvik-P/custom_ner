"""Тесты для privacyguard_pipeline.mcp_server.server (бизнес-логика).

Тестируют функции-обработчики напрямую (mask_text, demask_text,
anonymize_pdf_bytes, detect_text_file, detect_pdf_file), а не через
MCP-протокол/Context — вне активной MCP-сессии Context недоступен (см.
докстринг mcp_server/server.py), а вся содержательная логика уже
вынесена в эти функции.
"""

from __future__ import annotations

import base64
import subprocess
import sys
from pathlib import Path

import fitz
import pytest
from mcp.server.mcpserver.exceptions import ToolError
from privacyguard_pipeline.detection import PIIDetector
from privacyguard_pipeline.mcp_server.registry import MaskRegistry
from privacyguard_pipeline.mcp_server.server import (
    anonymize_pdf_bytes,
    demask_text,
    detect_file,
    detect_pdf_file,
    detect_text_file,
    mask_text,
)

# ---------------------------------------------------------------------------
# mask_text / demask_text
# ---------------------------------------------------------------------------


def test_mask_then_demask_round_trip(detector: PIIDetector) -> None:
    registry = MaskRegistry(ttl_seconds=60)
    text = 'Меня зовут Иван Петров, мой телефон +7 999 123 45 67.'

    masked = mask_text(text, detector, registry)
    assert 'Иван' not in masked['masked_text']
    assert 'Петров' not in masked['masked_text']
    assert '999' not in masked['masked_text']

    restored = demask_text(masked['handle'], masked['masked_text'], registry)
    assert restored['text'] == text


def test_mask_without_pii_leaves_text_unchanged(detector: PIIDetector) -> None:
    registry = MaskRegistry(ttl_seconds=60)
    text = 'Это обычный текст без персональных данных.'

    masked = mask_text(text, detector, registry)
    assert masked['masked_text'] == text

    restored = demask_text(masked['handle'], masked['masked_text'], registry)
    assert restored['text'] == text


def test_demask_with_unknown_handle_raises_tool_error() -> None:
    registry = MaskRegistry(ttl_seconds=60)

    with pytest.raises(ToolError):
        demask_text('never-issued', 'some text', registry)


def test_demask_twice_with_same_handle_raises_second_time(
    detector: PIIDetector,
) -> None:
    registry = MaskRegistry(ttl_seconds=60)
    masked = mask_text('Иван Петров', detector, registry)

    demask_text(masked['handle'], masked['masked_text'], registry)
    with pytest.raises(ToolError):
        demask_text(masked['handle'], masked['masked_text'], registry)


# ---------------------------------------------------------------------------
# anonymize_pdf_bytes
# ---------------------------------------------------------------------------


def _build_pdf_bytes(font: str, lines: list[str]) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    y = 100
    for line in lines:
        page.insert_text((72, y), line, fontfile=font, fontname='F1')
        y += 30
    pdf_bytes: bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_anonymize_pdf_bytes_redacts_pii(
    cyrillic_font_path: str,
    detector: PIIDetector,
) -> None:
    pdf_bytes = _build_pdf_bytes(
        cyrillic_font_path,
        ['Меня зовут Иван Петров, мой телефон +7 999 123 45 67.'],
    )

    result = anonymize_pdf_bytes(pdf_bytes, detector=detector)

    assert result['redacted_by_type'].get('PER') == 1
    assert result['redacted_by_type'].get('PHONE') == 1

    redacted_bytes = base64.b64decode(result['redacted_pdf_base64'])
    extracted = fitz.open(stream=redacted_bytes, filetype='pdf')[0].get_text()
    assert 'Иван' not in extracted
    assert '999' not in extracted


def test_anonymize_pdf_bytes_raises_tool_error_without_pdf_extra(
    tmp_path: Path,
) -> None:
    """Имитирует отсутствие экстры "pdf" в изолированном подпроцессе.

    Тот же приём, что и test_core_package_imports_without_pdf_extra в
    tests/test_pdf_anonymizer.py: подставляет None в sys.modules для
    fitz/pytesseract, чтобы вызвать реальный путь PDFDependencyError, не
    удаляя ничего из общего venv этого окружения.
    """
    script = (
        'import sys\n'
        "sys.modules['fitz'] = None\n"
        "sys.modules['pytesseract'] = None\n"
        'from mcp.server.mcpserver.exceptions import ToolError\n'
        'from privacyguard_pipeline.mcp_server.server import (\n'
        '    anonymize_pdf_bytes,\n'
        ')\n'
        'try:\n'
        "    anonymize_pdf_bytes(b'irrelevant')\n"
        'except ToolError:\n'
        "    print('OK')\n"
        'else:\n'
        "    print('FAIL: no ToolError raised')\n"
    )
    proc = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr
    assert 'OK' in proc.stdout, proc.stdout


# ---------------------------------------------------------------------------
# detect_text_file / detect_pdf_file
# ---------------------------------------------------------------------------


def test_detect_text_file_counts_entities_without_leaking_values(
    tmp_path: Path,
    detector: PIIDetector,
) -> None:
    path = tmp_path / 'note.txt'
    path.write_text(
        'Меня зовут Иван Петров, мой телефон +7 999 123 45 67.',
        encoding='utf-8',
    )

    result = detect_text_file(path, detector)

    assert result == {'entity_counts': {'PER': 1, 'PHONE': 1}}


def test_detect_text_file_without_pii_returns_empty_counts(
    tmp_path: Path,
    detector: PIIDetector,
) -> None:
    path = tmp_path / 'note.txt'
    path.write_text('Обычный текст без персональных данных.', encoding='utf-8')

    result = detect_text_file(path, detector)

    assert result == {'entity_counts': {}}


def test_detect_pdf_file_counts_entities_per_page(
    tmp_path: Path,
    cyrillic_font_path: str,
    detector: PIIDetector,
) -> None:
    path = tmp_path / 'doc.pdf'
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text(
        (72, 100),
        'Меня зовут Иван Петров.',
        fontfile=cyrillic_font_path,
        fontname='F1',
    )
    page2 = doc.new_page()
    page2.insert_text(
        (72, 100),
        'Мой телефон +7 999 123 45 67.',
        fontfile=cyrillic_font_path,
        fontname='F1',
    )
    doc.save(str(path))
    doc.close()

    result = detect_pdf_file(path, detector)

    assert result == {
        'pages': {
            '1': {'PER': 1},
            '2': {'PHONE': 1},
        },
    }


# ---------------------------------------------------------------------------
# detect_file (диспетчеризация по расширению)
# ---------------------------------------------------------------------------


def test_detect_file_missing_path_raises_tool_error(
    tmp_path: Path,
    detector: PIIDetector,
) -> None:
    missing = tmp_path / 'does_not_exist.txt'

    with pytest.raises(ToolError):
        detect_file(str(missing), detector)


def test_detect_file_directory_path_raises_tool_error(
    tmp_path: Path,
    detector: PIIDetector,
) -> None:
    with pytest.raises(ToolError):
        detect_file(str(tmp_path), detector)


def test_detect_file_dispatches_by_extension(
    tmp_path: Path,
    cyrillic_font_path: str,
    detector: PIIDetector,
) -> None:
    txt_path = tmp_path / 'note.TXT'
    txt_path.write_text('Обычный текст.', encoding='utf-8')
    assert detect_file(str(txt_path), detector) == {'entity_counts': {}}

    # Страница с настоящим текстовым слоем (а не совсем пустая) — у
    # совсем пустой страницы нет текстового слоя вовсе, что заставило бы
    # detect_pdf_file уйти в обязательный путь OCR (см. spec
    # pdf-anonymization) вместо проверки самой диспетчеризации по
    # расширению.
    pdf_path = tmp_path / 'plain.PDF'
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 100),
        'Обычный текст без персональных данных.',
        fontfile=cyrillic_font_path,
        fontname='F1',
    )
    doc.save(str(pdf_path))
    doc.close()
    assert detect_file(str(pdf_path), detector) == {'pages': {'1': {}}}
