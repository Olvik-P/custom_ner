"""Text-layer extraction for PDF pages, with offset-to-bbox mapping.

Reconstructs contiguous block-level text from ``page.get_text("words")``
(rather than raw font-run spans from ``get_text("dict")``) so that
``PIIDetector`` sees enough surrounding context for its NER and
contextual-disambiguation layers to work as designed, while keeping an
exact character-offset -> word-bbox mapping for redaction.

``group_words_into_blocks`` is shared with the OCR fallback (``ocr.py``) so
both extraction paths produce the same TextBlock/WordBox shape for
downstream detection and redaction code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import fitz

BBox = tuple[float, float, float, float]

# Index positions in a page.get_text("words") tuple:
# (x0, y0, x1, y1, word, block_no, line_no, word_no)
_WORD_TEXT_INDEX = 4
_WORD_BLOCK_INDEX = 5
_WORD_LINE_INDEX = 6
_WORD_NO_INDEX = 7


class RawWord(NamedTuple):
    """A single recognized word before block/line grouping.

    Produced by either the PyMuPDF text layer or the OCR fallback — both
    feed the same `group_words_into_blocks`.
    """

    text: str
    block_no: int
    line_no: int
    word_no: int
    bbox: BBox


@dataclass(frozen=True)
class WordBox:
    """A single word with its bounding box and offset in the block text.

    Attributes:
        text: The word's own text.
        start: Start character offset within the owning TextBlock.text.
        end: End character offset within the owning TextBlock.text.
        bbox: Word bounding box (x0, y0, x1, y1) in page coordinates.
    """

    text: str
    start: int
    end: int
    bbox: BBox


@dataclass(frozen=True)
class TextBlock:
    """Reconstructed block-level text, ready for PIIDetector.detect().

    Attributes:
        page_num: Zero-based page index this block belongs to.
        text: Contiguous text reconstructed from the block's words.
        words: Word-level bounding boxes, offset-aligned with text.
    """

    page_num: int
    text: str
    words: list[WordBox]


def page_has_text_layer(page: 'fitz.Page') -> bool:
    """Check whether a page has an extractable text layer.

    Args:
        page: PyMuPDF page to inspect.

    Returns:
        True if the page has at least one extractable word, False if the
        page is image-only (scanned) and needs the OCR fallback.
    """
    return bool(page.get_text('words'))


def extract_page_text_blocks(page: 'fitz.Page') -> list[TextBlock]:
    """Extract block-level text with a per-word offset-to-bbox mapping.

    Args:
        page: PyMuPDF page to extract text from.

    Returns:
        List of TextBlock, one per text block on the page.
    """
    raw_words = [
        RawWord(
            text=word[_WORD_TEXT_INDEX],
            block_no=word[_WORD_BLOCK_INDEX],
            line_no=word[_WORD_LINE_INDEX],
            word_no=word[_WORD_NO_INDEX],
            bbox=(word[0], word[1], word[2], word[3]),
        )
        for word in page.get_text('words')
    ]
    return group_words_into_blocks(page.number, raw_words)


def group_words_into_blocks(
    page_num: int,
    raw_words: list[RawWord],
) -> list[TextBlock]:
    """Group raw words into per-block TextBlock objects with offset maps.

    Words are grouped by block index (paragraph-level grouping), ordered
    by line then word position within the line, and joined into one
    contiguous string per block — newline between lines, single space
    between words on the same line — so PIIDetector sees full-sentence or
    full-paragraph context instead of isolated fragments.

    Args:
        page_num: Zero-based page index the words belong to.
        raw_words: Recognized words for the page, in any order.

    Returns:
        List of TextBlock, one per distinct block_no in raw_words.
    """
    if not raw_words:
        return []

    words_by_block: dict[int, list[RawWord]] = {}
    for word in raw_words:
        words_by_block.setdefault(word.block_no, []).append(word)

    blocks: list[TextBlock] = []
    for block_no in sorted(words_by_block):
        block_words = sorted(
            words_by_block[block_no],
            key=lambda w: (w.line_no, w.word_no),
        )
        blocks.append(_join_block_words(page_num, block_words))
    return blocks


def _join_block_words(
    page_num: int,
    block_words: list[RawWord],
) -> TextBlock:
    """Join one block's words into contiguous text with a bbox map.

    Args:
        page_num: Zero-based page index the words belong to.
        block_words: Words for one block, pre-sorted by reading order.

    Returns:
        A TextBlock with joined text and offset-aligned WordBox entries.
    """
    text_parts: list[str] = []
    words: list[WordBox] = []
    cursor = 0
    prev_line_no: int | None = None

    for word in block_words:
        line_changed = (
            prev_line_no is not None and word.line_no != prev_line_no
        )
        # A word wrapped across the line break (e.g. "Моск-" / "ва") is
        # split by PyMuPDF into two word tokens whose text each keeps
        # its own half of the hyphenated original. Detecting on the two
        # halves joined by "-\n" would hide the value from NER/regex, so
        # when the previous line ends in a hyphen attached to a real
        # word (not a standalone dash token), drop that hyphen and join
        # directly with no separator instead of inserting "\n".
        prev_word = words[-1] if words else None
        hyphen_wrap = (
            line_changed
            and prev_word is not None
            and len(prev_word.text) > 1
            and prev_word.text.endswith('-')
        )

        if hyphen_wrap:
            assert prev_word is not None  # for type-checkers
            text_parts[-1] = text_parts[-1][:-1]
            cursor -= 1
            words[-1] = WordBox(
                text=prev_word.text[:-1],
                start=prev_word.start,
                end=prev_word.end - 1,
                bbox=prev_word.bbox,
            )
        elif prev_line_no is not None:
            separator = '\n' if line_changed else ' '
            text_parts.append(separator)
            cursor += len(separator)

        start = cursor
        text_parts.append(word.text)
        cursor += len(word.text)
        words.append(
            WordBox(text=word.text, start=start, end=cursor, bbox=word.bbox),
        )
        prev_line_no = word.line_no

    return TextBlock(page_num=page_num, text=''.join(text_parts), words=words)
