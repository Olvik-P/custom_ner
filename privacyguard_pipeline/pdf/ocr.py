"""OCR fallback for PDF pages without an extractable text layer.

Renders a page to an image, runs Tesseract via ``pytesseract``, and maps
the recognized words' pixel-space bounding boxes back to PDF page
coordinates — producing the same TextBlock/WordBox shape as
``text_extractor.py`` so detection and redaction share one downstream
code path regardless of which extraction source a page used.
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
    """OCR a page and return it as block-level TextBlocks.

    Args:
        page: PyMuPDF page to OCR (used when it has no text layer, or in
            addition to the text layer when the page also has images).
        lang: Tesseract language code (Russian by default).
        exclude_bboxes: Word bounding boxes already covered by an
            extractable text layer (page coordinates). An OCR word whose
            center falls inside one of these is dropped, so a page that
            mixes real text with an image doesn't get the same text
            detected — and potentially double-redacted — twice.

    Returns:
        List of TextBlock, in the same shape as
        text_extractor.extract_page_text_blocks.

    Raises:
        PDFDependencyError: If the system Tesseract binary or the
            requested language pack is not available.
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
    """Whether bbox's center point falls inside any of `others`."""
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    return any(
        ox0 <= cx <= ox1 and oy0 <= cy <= oy1 for ox0, oy0, ox1, oy1 in others
    )


def _run_tesseract(image: Image.Image, lang: str) -> dict[str, list[Any]]:
    """Run pytesseract and translate its errors into PDFDependencyError.

    Args:
        image: Rendered page image to OCR.
        lang: Tesseract language code.

    Returns:
        pytesseract's per-word data dict (Output.DICT).

    Raises:
        PDFDependencyError: If Tesseract or its language pack is missing.
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
    """Convert pytesseract's word data into pixel-independent RawWords.

    Args:
        data: pytesseract.image_to_data output (Output.DICT).
        inverse_matrix: Inverse of the matrix used to render the page,
            mapping pixel coordinates back to PDF page coordinates.

    Returns:
        List of RawWord with bboxes in PDF page coordinates, skipping
        blank OCR entries (line/paragraph/block-level rows with no text).
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
