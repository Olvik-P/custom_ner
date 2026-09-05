"""Тесты для privacyguard_pipeline.pdf (PDFAnonymizer)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import fitz
import pytest
from privacyguard_pipeline.exceptions import PDFDependencyError
from privacyguard_pipeline.pdf import PDFAnonymizer, ocr


def _build_pdf(path: Path, font: str, lines: list[str]) -> None:
    """Записывает одностраничный PDF со строками в кириллическом шрифте."""
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
    """Имитирует отсутствие экстры "pdf" в изолированном подпроцессе.

    Устанавливает fitz/pytesseract в None в sys.modules (стандартный
    приём для принудительного ImportError на опциональной зависимости),
    чтобы проверить собственный путь импорта основного пакета, не
    удаляя на самом деле ничего из общего venv этого окружения.
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


def test_ocr_runs_on_mixed_page_with_stamp_and_image_pii(
    tmp_path: Path,
    cyrillic_font_path: str,
) -> None:
    """Маленький настоящий штамп не должен останавливать OCR страницы.

    page_has_text_layer() раньше работал по принципу "всё или ничего":
    любой извлекаемый текст (даже однословный штамп номера страницы)
    направлял всю страницу через извлечение только текста, поэтому PII,
    запечённый в сопутствующем изображении, никогда не проходил OCR и
    не редактировался.
    """
    pytest.importorskip('pytesseract')
    import pytesseract

    try:
        pytesseract.get_tesseract_version()
    except Exception:
        pytest.skip('System Tesseract binary not available')

    input_pdf = tmp_path / 'mixed.pdf'
    output_pdf = tmp_path / 'mixed_redacted.pdf'

    # Рендерим текст PII в изображение — та же техника, что и в тесте
    # OCR для чисто сканированной страницы выше.
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

    mixed_doc = fitz.open()
    page = mixed_doc.new_page(width=pix.width / zoom, height=pix.height / zoom)
    page.insert_image(page.rect, pixmap=pix)
    # Настоящий, напрямую извлекаемый текстовый штамп в другом месте
    # страницы (голый номер страницы — однозначный, поэтому сам не
    # будет помечен как PII) — раньше одного этого хватало, чтобы
    # page_has_text_layer() полностью пропустил OCR изображения.
    page.insert_text(
        (72, page.rect.height - 20),
        '1',
        fontfile=cyrillic_font_path,
        fontname='F2',
    )
    mixed_doc.save(str(input_pdf))
    mixed_doc.close()

    result = PDFAnonymizer().anonymize(input_pdf, output_pdf)

    assert result.success, result.error_message
    assert result.redacted_by_type == {'PHONE': 1}

    extracted = fitz.open(str(output_pdf))[0].get_text()
    assert '1' in extracted  # сам штамп не тронут
    assert (
        '999' not in extracted
    )  # PII из изображения прошёл OCR и отредактирован


def test_ocr_unavailable_does_not_fail_a_page_with_a_text_layer(
    tmp_path: Path,
    cyrillic_font_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OCR необязателен, если на странице уже есть настоящий текстовый слой.

    Страница с извлекаемым текстом (редактируется через путь текстового
    слоя независимо от OCR) плюс не связанное с ним встроенное
    изображение (например, логотип на бланке) всё равно должна успешно
    отредактироваться, когда Tesseract не установлен — OCR здесь лишь
    подхватывает *дополнительный* PII, который может находиться в
    изображении, а не единственный источник PII на странице, поэтому
    отсутствующий движок OCR не должен провалить всю страницу.
    """
    input_pdf = tmp_path / 'logo.pdf'
    output_pdf = tmp_path / 'logo_redacted.pdf'

    # Крошечное изображение вроде логотипа, не связанное с текстом PII ниже.
    logo_doc = fitz.open()
    logo_page = logo_doc.new_page(width=40, height=40)
    logo_page.draw_rect(logo_page.rect, color=(0, 0, 1), fill=(0, 0, 1))
    logo_pix = logo_page.get_pixmap()
    logo_doc.close()

    doc = fitz.open()
    page = doc.new_page()
    page.insert_image(fitz.Rect(72, 40, 112, 80), pixmap=logo_pix)
    page.insert_text(
        (72, 120),
        'Меня зовут Иван Петров, мой телефон +7 999 123 45 67.',
        fontfile=cyrillic_font_path,
        fontname='F1',
    )
    doc.save(str(input_pdf))
    doc.close()

    def _raise_dependency_error(*args: object, **kwargs: object) -> None:
        raise PDFDependencyError('System Tesseract OCR binary not found')

    monkeypatch.setattr(
        ocr,
        'extract_page_text_blocks_ocr',
        _raise_dependency_error,
    )

    result = PDFAnonymizer().anonymize(input_pdf, output_pdf)

    assert result.success, result.error_message
    assert result.redacted_by_type.get('PER') == 1
    assert result.redacted_by_type.get('PHONE') == 1

    extracted = fitz.open(str(output_pdf))[0].get_text()
    assert 'Иван' not in extracted
    assert '999' not in extracted


def test_ocr_unavailable_still_fails_an_image_only_page(
    tmp_path: Path,
    cyrillic_font_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OCR обязателен (не опционален) на странице без текстового слоя.

    В отличие от смешанного случая выше, у полностью отсканированной
    страницы нет другого способа найти свой PII — молчаливый пропуск
    OCR там оставил бы PII неотредактированным без какого-либо сигнала,
    а спека это явно запрещает.
    """
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
    pix = text_page.get_pixmap(matrix=fitz.Matrix(3, 3))
    text_doc.close()

    scanned_doc = fitz.open()
    page = scanned_doc.new_page(width=pix.width / 3, height=pix.height / 3)
    page.insert_image(page.rect, pixmap=pix)
    scanned_doc.save(str(input_pdf))
    scanned_doc.close()

    def _raise_dependency_error(*args: object, **kwargs: object) -> None:
        raise PDFDependencyError('System Tesseract OCR binary not found')

    monkeypatch.setattr(
        ocr,
        'extract_page_text_blocks_ocr',
        _raise_dependency_error,
    )

    result = PDFAnonymizer().anonymize(input_pdf, output_pdf)

    assert not result.success
    assert 'Tesseract' in (result.error_message or '')
