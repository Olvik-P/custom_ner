"""OCR-фолбэк для страниц PDF без извлекаемого текстового слоя.

Рендерит страницу в изображение, запускает Tesseract через
``pytesseract`` и переносит bounding box'ы распознанных слов из
пиксельного пространства обратно в координаты страницы PDF —
производя ту же форму TextBlock/WordBox, что и ``text_extractor.py``,
чтобы детекция и редактирование использовали один и тот же путь кода
независимо от того, какой источник извлечения использовался для
страницы.
"""

from __future__ import annotations

from typing import Any, Sequence, cast

import fitz
import pytesseract
from PIL import Image

from privacyguard_pipeline.constants import PDF_OCR_LANGUAGE, PDF_OCR_ZOOM
from privacyguard_pipeline.exceptions import PDFDependencyError
from privacyguard_pipeline.pdf.text_extractor import (
    BBox,
    RawWord,
    TextBlock,
    group_words_into_blocks,
)

_LANG_ERROR_HINTS = ('lang', 'traineddata')


def extract_page_text_blocks_ocr(
    page: fitz.Page,
    lang: str = PDF_OCR_LANGUAGE,
    exclude_bboxes: Sequence[BBox] = (),
) -> list[TextBlock]:
    """Прогоняет OCR по странице и возвращает её как TextBlock'и по блокам.

    Args:
        page: Страница PyMuPDF для OCR (используется, когда у неё нет
            текстового слоя, либо в дополнение к текстовому слою, когда
            на странице также есть изображения).
        lang: Языковой код Tesseract (русский по умолчанию).
        exclude_bboxes: Bounding box'ы слов, уже покрытые извлекаемым
            текстовым слоем (координаты страницы). OCR-слово, чей центр
            попадает внутрь одного из них, отбрасывается, чтобы
            страница, смешивающая настоящий текст с изображением, не
            получила одно и то же слово задетектированным — и
            потенциально дважды отредактированным.

    Returns:
        Список TextBlock той же формы, что и
        text_extractor.extract_page_text_blocks.

    Raises:
        PDFDependencyError: Если системный бинарник Tesseract или
            запрошенный языковой пакет недоступны.
    """
    matrix = fitz.Matrix(PDF_OCR_ZOOM, PDF_OCR_ZOOM)
    pixmap = page.get_pixmap(matrix=matrix)
    image = Image.frombytes(
        'RGB',
        (pixmap.width, pixmap.height),
        pixmap.samples,
    )

    data = _run_tesseract(image, lang)
    raw_words = _to_raw_words(data, ~matrix)
    if exclude_bboxes:
        raw_words = [
            word
            for word in raw_words
            if not _center_in_any_bbox(word.bbox, exclude_bboxes)
        ]
    return group_words_into_blocks(page.number, raw_words)


def _center_in_any_bbox(bbox: BBox, others: Sequence[BBox]) -> bool:
    """Находится ли центр bbox внутри одного из `others`."""
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    return any(
        ox0 <= cx <= ox1 and oy0 <= cy <= oy1 for ox0, oy0, ox1, oy1 in others
    )


def _run_tesseract(image: Image.Image, lang: str) -> dict[str, list[Any]]:
    """Запускает pytesseract и переводит его ошибки в PDFDependencyError.

    Args:
        image: Отрендеренное изображение страницы для OCR.
        lang: Языковой код Tesseract.

    Returns:
        Словарь данных pytesseract по каждому слову (Output.DICT).

    Raises:
        PDFDependencyError: Если Tesseract или его языковой пакет
            отсутствуют.
    """
    try:
        return cast(
            'dict[str, list[Any]]',
            pytesseract.image_to_data(
                image,
                lang=lang,
                output_type=pytesseract.Output.DICT,
            ),
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise PDFDependencyError(
            'System Tesseract OCR binary not found on PATH. Install '
            'Tesseract (with the Russian language pack, '
            "e.g. 'tesseract-ocr-rus') to anonymize scanned PDF pages.",
        ) from exc
    except pytesseract.TesseractError as exc:
        message = str(exc).lower()
        if any(hint in message for hint in _LANG_ERROR_HINTS):
            raise PDFDependencyError(
                f"Tesseract language pack '{lang}' not found. Install "
                f"'{lang}.traineddata' for your Tesseract installation.",
            ) from exc
        raise


def _to_raw_words(
    data: dict[str, list[Any]],
    inverse_matrix: fitz.Matrix,
) -> list[RawWord]:
    """Конвертирует данные слов pytesseract в независимые от пикселей RawWord.

    Args:
        data: Вывод pytesseract.image_to_data (Output.DICT).
        inverse_matrix: Обратная матрица к той, что использовалась для
            рендеринга страницы, переводящая пиксельные координаты
            обратно в координаты страницы PDF.

    Returns:
        Список RawWord с bbox в координатах страницы PDF, пропуская
        пустые записи OCR (строки уровня строки/абзаца/блока без
        текста).
    """
    raw_words: list[RawWord] = []
    for i, word_text in enumerate(data['text']):
        stripped = word_text.strip()
        if not stripped:
            continue

        left, top = data['left'][i], data['top'][i]
        width, height = data['width'][i], data['height'][i]
        top_left = fitz.Point(left, top) * inverse_matrix
        bottom_right = fitz.Point(left + width, top + height) * (
            inverse_matrix
        )

        raw_words.append(
            RawWord(
                text=stripped,
                block_no=data['block_num'][i],
                line_no=data['line_num'][i],
                word_no=data['word_num'][i],
                bbox=(
                    top_left.x,
                    top_left.y,
                    bottom_right.x,
                    bottom_right.y,
                ),
            ),
        )
    return raw_words
