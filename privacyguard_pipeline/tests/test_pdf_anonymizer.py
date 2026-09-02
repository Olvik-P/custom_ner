"""Tests for privacyguard_pipeline.pdf (PDFAnonymizer)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import fitz
import pytest
from privacyguard_pipeline.pdf import PDFAnonymizer


def _build_pdf(path: Path, font: str, lines: list[str]) -> None:
    """Write a one-page PDF with each line in a Cyrillic-capable font."""
    doc = fitz.open()
    page = doc.new_page()
    y = 100
    for line in lines:
        page.insert_text((72, y), line, fontfile=font, fontname='F1')
        y += 30
    doc.save(str(path))
    doc.close()


def test_pii_removed_from_extracted_text(
    tmp_path: Path,
    cyrillic_font_path: str,
) -> None:
    input_pdf = tmp_path / 'input.pdf'
    output_pdf = tmp_path / 'output.pdf'
    _build_pdf(
        input_pdf,
        cyrillic_font_path,
        [
            'Меня зовут Иван Петров, мой телефон +7 999 123 45 67.',
            'Это обычный текст без персональных данных.',
        ],
    )

    result = PDFAnonymizer().anonymize(input_pdf, output_pdf)

    assert result.success, result.error_message
    assert result.redacted_by_type.get('PER') == 1
    assert result.redacted_by_type.get('PHONE') == 1

    extracted = fitz.open(str(output_pdf))[0].get_text()
    assert 'Иван' not in extracted
    assert 'Петров' not in extracted
    assert '999' not in extracted
    assert 'обычный' in extracted
    assert 'персональных' in extracted


def test_no_pii_leaves_page_unchanged(
    tmp_path: Path,
    cyrillic_font_path: str,
) -> None:
    input_pdf = tmp_path / 'input.pdf'
    output_pdf = tmp_path / 'output.pdf'
    text = 'Это обычный текст без персональных данных.'
    _build_pdf(input_pdf, cyrillic_font_path, [text])

    result = PDFAnonymizer().anonymize(input_pdf, output_pdf)

    assert result.success, result.error_message
    assert result.total_spans_redacted == 0

    extracted = fitz.open(str(output_pdf))[0].get_text()
    assert extracted.replace('\xa0', ' ').strip() == text.strip()


def test_entity_type_filter_limits_redaction(
    tmp_path: Path,
    cyrillic_font_path: str,
) -> None:
    input_pdf = tmp_path / 'input.pdf'
    output_pdf = tmp_path / 'output.pdf'
    _build_pdf(
        input_pdf,
        cyrillic_font_path,
        ['Меня зовут Иван Петров, мой телефон +7 999 123 45 67.'],
    )

    result = PDFAnonymizer().anonymize(
        input_pdf,
        output_pdf,
        entity_types=['PHONE'],
    )

    assert result.success, result.error_message
    assert result.redacted_by_type == {'PHONE': 1}

    extracted = fitz.open(str(output_pdf))[0].get_text()
    assert 'Иван' in extracted
    assert '999' not in extracted


def test_url_kept_by_default_but_redactable_on_request(
    tmp_path: Path,
    cyrillic_font_path: str,
) -> None:
    input_pdf = tmp_path / 'input.pdf'
    line = 'Ссылка: https://example.com/report-12345 конец.'

    default_output = tmp_path / 'default.pdf'
    _build_pdf(input_pdf, cyrillic_font_path, [line])
    default_result = PDFAnonymizer().anonymize(input_pdf, default_output)

    assert default_result.success, default_result.error_message
    assert 'URL' not in default_result.redacted_by_type
    default_text = fitz.open(str(default_output))[0].get_text()
    assert 'example.com' in default_text

    redact_output = tmp_path / 'redact_urls.pdf'
    redact_result = PDFAnonymizer().anonymize(
        input_pdf,
        redact_output,
        redact_urls=True,
    )

    assert redact_result.success, redact_result.error_message
    assert redact_result.redacted_by_type.get('URL') == 1
    redacted_text = fitz.open(str(redact_output))[0].get_text()
    assert 'example.com' not in redacted_text

    explicit_output = tmp_path / 'explicit_url.pdf'
    explicit_result = PDFAnonymizer().anonymize(
        input_pdf,
        explicit_output,
        entity_types=['URL'],
    )

    assert explicit_result.success, explicit_result.error_message
    assert explicit_result.redacted_by_type == {'URL': 1}
    explicit_text = fitz.open(str(explicit_output))[0].get_text()
    assert 'example.com' not in explicit_text


def test_missing_input_raises_before_processing(tmp_path: Path) -> None:
    missing = tmp_path / 'does_not_exist.pdf'

    with pytest.raises(FileNotFoundError):
        PDFAnonymizer().anonymize(missing)


def test_core_package_imports_without_pdf_extra() -> None:
    """Simulate a missing "pdf" extra in an isolated subprocess.

    Sets fitz/pytesseract to None in sys.modules (the standard trick for
    forcing ImportError on an optional dependency) so the core package's
    own import path is exercised without actually uninstalling anything
    from this environment's shared venv.
    """
    script = (
        'import sys\n'
        "sys.modules['fitz'] = None\n"
        "sys.modules['pytesseract'] = None\n"
        'import privacyguard_pipeline\n'
        'from privacyguard_pipeline.exceptions import PDFDependencyError\n'
        'try:\n'
        '    privacyguard_pipeline.PDFAnonymizer\n'
        'except PDFDependencyError:\n'
        "    print('OK')\n"
        'else:\n'
        "    print('FAIL: no error raised')\n"
    )
    proc = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr
    assert 'OK' in proc.stdout, proc.stdout


def test_ocr_fallback_for_scanned_page(
    tmp_path: Path,
    cyrillic_font_path: str,
) -> None:
    pytest.importorskip('pytesseract')
    import pytesseract

    try:
        pytesseract.get_tesseract_version()
    except Exception:
        pytest.skip('System Tesseract binary not available')

    input_pdf = tmp_path / 'scanned.pdf'
    output_pdf = tmp_path / 'scanned_redacted.pdf'

    text_doc = fitz.open()
    text_page = text_doc.new_page()
    text_page.insert_text(
        (72, 100),
        'Мой телефон +7 999 123 45 67.',
        fontfile=cyrillic_font_path,
        fontname='F1',
    )
    zoom = 3
    pix = text_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    text_doc.close()

    scanned_doc = fitz.open()
    page = scanned_doc.new_page(
        width=pix.width / zoom,
        height=pix.height / zoom,
    )
    page.insert_image(page.rect, pixmap=pix)
    scanned_doc.save(str(input_pdf))
    scanned_doc.close()

    result = PDFAnonymizer().anonymize(input_pdf, output_pdf)

    assert result.success, result.error_message
    assert result.redacted_by_type.get('PHONE', 0) >= 1
