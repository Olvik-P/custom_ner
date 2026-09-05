"""Извлечение текстового слоя страниц PDF с маппингом offset->bbox.

Восстанавливает непрерывный текст на уровне блоков из
``page.get_text("words")`` (а не сырые font-run спаны из
``get_text("dict")``), чтобы ``PIIDetector`` видел достаточно
окружающего контекста для работы слоёв NER и контекстного разрешения
неоднозначности так, как задумано, при этом сохраняя точный маппинг
символьного offset -> bbox слова для редактирования.

``group_words_into_blocks`` используется совместно с OCR-фолбэком
(``ocr.py``), поэтому оба пути извлечения производят одинаковую форму
TextBlock/WordBox для последующего кода детекции и редактирования.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import fitz

BBox = tuple[float, float, float, float]

# Позиции индексов в кортеже page.get_text("words"):
# (x0, y0, x1, y1, word, block_no, line_no, word_no)
_WORD_TEXT_INDEX = 4
_WORD_BLOCK_INDEX = 5
_WORD_LINE_INDEX = 6
_WORD_NO_INDEX = 7


class RawWord(NamedTuple):
    """Одно распознанное слово до группировки по блокам/строкам.

    Производится либо текстовым слоем PyMuPDF, либо OCR-фолбэком — оба
    подают результат в один и тот же `group_words_into_blocks`.
    """

    text: str
    block_no: int
    line_no: int
    word_no: int
    bbox: BBox


@dataclass(frozen=True)
class WordBox:
    """Одно слово с его bounding box и offset в тексте блока.

    Attributes:
        text: Текст самого слова.
        start: Начальный символьный offset внутри TextBlock.text,
            которому принадлежит слово.
        end: Конечный символьный offset внутри TextBlock.text.
        bbox: Bounding box слова (x0, y0, x1, y1) в координатах
            страницы.
    """

    text: str
    start: int
    end: int
    bbox: BBox


@dataclass(frozen=True)
class TextBlock:
    """Восстановленный текст на уровне блока, готовый для PIIDetector.detect().

    Attributes:
        page_num: Индекс страницы (с нуля), которой принадлежит блок.
        text: Непрерывный текст, восстановленный из слов блока.
        words: Bounding box'ы на уровне слов, выровненные по offset с
            текстом.
    """

    page_num: int
    text: str
    words: list[WordBox]


def page_has_text_layer(page: 'fitz.Page') -> bool:
    """Проверяет, есть ли у страницы извлекаемый текстовый слой.

    Args:
        page: Страница PyMuPDF для проверки.

    Returns:
        True, если на странице есть хотя бы одно извлекаемое слово,
        False, если страница состоит только из изображения
        (отсканирована) и нужен OCR-фолбэк.
    """
    return bool(page.get_text('words'))


def extract_page_text_blocks(page: 'fitz.Page') -> list[TextBlock]:
    """Извлекает текст на уровне блоков с маппингом offset->bbox по словам.

    Args:
        page: Страница PyMuPDF для извлечения текста.

    Returns:
        Список TextBlock, по одному на текстовый блок страницы.
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
    """Группирует сырые слова в объекты TextBlock по блокам с offset-маппингом.

    Слова группируются по индексу блока (группировка на уровне
    абзаца), упорядочиваются по строке, затем по позиции слова внутри
    строки, и объединяются в одну непрерывную строку на блок —
    перевод строки между строками, один пробел между словами в одной
    строке — так, чтобы PIIDetector видел контекст целого предложения
    или абзаца вместо изолированных фрагментов.

    Args:
        page_num: Индекс страницы (с нуля), которой принадлежат слова.
        raw_words: Распознанные слова страницы в произвольном порядке.

    Returns:
        Список TextBlock, по одному на каждый отдельный block_no в
        raw_words.
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
    """Объединяет слова одного блока в непрерывный текст с картой bbox.

    Args:
        page_num: Индекс страницы (с нуля), которой принадлежат слова.
        block_words: Слова одного блока, предварительно отсортированные
            по порядку чтения.

    Returns:
        TextBlock с объединённым текстом и выровненными по offset
        записями WordBox.
    """
    text_parts: list[str] = []
    words: list[WordBox] = []
    cursor = 0
    prev_line_no: int | None = None

    for word in block_words:
        line_changed = (
            prev_line_no is not None and word.line_no != prev_line_no
        )
        # Слово, перенесённое через границу строки (например, "Моск-" /
        # "ва"), разбивается PyMuPDF на два токена-слова, каждый из
        # которых хранит свою половину дефисного оригинала. Детекция по
        # двум половинам, соединённым через "-\n", скрыла бы значение от
        # NER/regex, поэтому, если предыдущая строка заканчивается
        # дефисом, прикреплённым к реальному слову (а не отдельным
        # токеном-тире), этот дефис убирается, и слова соединяются
        # напрямую без разделителя вместо вставки "\n".
        prev_word = words[-1] if words else None
        hyphen_wrap = (
            line_changed
            and prev_word is not None
            and len(prev_word.text) > 1
            and prev_word.text.endswith('-')
        )

        if hyphen_wrap:
            assert prev_word is not None  # для type-checker'ов
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
