"""Настоящее редактирование PDF — удаляет содержимое, а не просто рисует
поверх.

Использует redaction-аннотации PyMuPDF (``add_redact_annot`` +
``apply_redactions``), которые вырезают лежащие в основе текстовые
объекты и обнуляют пиксели изображения под прямоугольником
редактирования, вместо рисования косметического чёрного прямоугольника
поверх содержимого, которое осталось бы извлекаемым под ним.

``apply_redactions()`` PyMuPDF удаляет текст с гранулярностью PDF
"span" (непрерывный прогон одного шрифта в потоке содержимого,
например одна выключенная строка, нарисованная одним оператором вывода
текста) — прямоугольник редактирования, пересекающий любую часть
span'а, может уронить символы по всему этому span'у, а не только в
части под прямоугольником. ``redact_text_layer`` решает это, редактируя
*весь* затронутый span целиком, а затем заново вставляя не-PII
("выжившие") слова этого span'а исходным встроенным шрифтом, размером
и базовой линией, так что теряется только сам текст PII.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass

import fitz

from privacyguard_pipeline.constants import PDF_REDACTION_PADDING
from privacyguard_pipeline.pdf.text_extractor import BBox, WordBox

logger = logging.getLogger(__name__)

# Шрифты, используемые для восстановления не-PII текста, когда
# собственный встроенный шрифт PDF нельзя переиспользовать (см.
# font_for_text). Упорядочены по предпочтению.
_FALLBACK_FONT_CANDIDATES = (
    '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    'C:/Windows/Fonts/times.ttf',
    'C:/Windows/Fonts/arial.ttf',
)


@dataclass(frozen=True)
class TextSpan:
    """Текстовый span PyMuPDF: один однородный прогон шрифта/стиля.

    Это наименьшая единица, которую ``apply_redactions()`` может
    безопасно удалить, не рискуя повредить соседний текст (см.
    докстринг модуля).

    Attributes:
        bbox: Bounding box span'а (x0, y0, x1, y1) в координатах
            страницы.
        font: Базовое имя встроенного шрифта (например,
            "BAAAAA+LiberationSerif").
        size: Размер шрифта в пунктах.
        color: Цвет текста как 24-битное целое sRGB (0xRRGGBB).
        origin_y: Y-координата базовой линии этого span'а
            (горизонтальный текст имеет одну постоянную базовую линию
            на span).
    """

    bbox: BBox
    font: str
    size: float
    color: int
    origin_y: float


class FontCache:
    """Извлекает встроенные шрифты во временные файлы, кэшируя по шрифту.

    ``insert_text()`` принимает только *путь к файлу* шрифта, а не
    сырые байты, поэтому каждый отдельный встроенный шрифт,
    используемый для повторной вставки, один раз записывается во
    временный файл и переиспользуется. По завершении вызовите
    ``cleanup()``.
    """

    def __init__(self, doc: fitz.Document) -> None:
        self._doc = doc
        self._tmpdir = tempfile.mkdtemp(prefix='privacyguard_pdf_fonts_')
        # Ключ — xref (глобальный для документа id конкретного объекта
        # встроенного шрифта), а не font_name: разные подмножества
        # одноимённого шрифта на двух страницах (например, обе
        # "LiberationSerif", но каждая встраивает только используемые
        # на этой странице глифы, под разными xref) не должны делить
        # одну запись кэша — переиспользование подмножества одной
        # страницы для другой молча роняет не-PII слова, для которых у
        # неверного подмножества нет глифов (см. font_for_text/
        # _font_covers).
        self._paths: dict[int, str | None] = {}

    def path_for(self, page: fitz.Page, font_name: str) -> str | None:
        """Возвращает локальный путь к встроенному шрифту, извлекая один раз.

        Args:
            page: Страница, на которой используется шрифт (для
                разрешения её xref).
            font_name: Базовое имя шрифта span'а для поиска.

        Returns:
            Путь к извлечённому файлу шрифта, либо None, если его не
            удалось извлечь (например, невстраиваемый шрифт Type3/CID)
            — в этом случае вызывающий код должен пропустить повторную
            вставку, а не гадать.
        """
        resolved = self._resolve_xref(page, font_name)
        if resolved is None:
            return None
        xref, ext = resolved

        if xref in self._paths:
            return self._paths[xref]

        path = self._extract(xref, ext)
        self._paths[xref] = path
        return path

    @staticmethod
    def _resolve_xref(
        page: fitz.Page,
        font_name: str,
    ) -> tuple[int, str] | None:
        # get_text("dict") сообщает span['font'] БЕЗ префикса подмножества
        # PDF (например, "LiberationSerif"), а get_fonts() сообщает сырое
        # имя ресурса С ним (например, "BAAAAA+LiberationSerif") —
        # префикс по спецификации PDF всегда ровно 6 заглавных букв + "+",
        # поэтому перед сравнением он отрезается. Этот проход дешёвый,
        # только по метаданным (байты шрифта не читаются), и выполняется
        # при каждом поиске — кэшируется только извлечение байтов ниже.
        for xref, ext, _subtype, basefont, *_rest in page.get_fonts(
            full=True,
        ):
            if _strip_subset_prefix(basefont) == font_name:
                return xref, ext
        return None

    def _extract(self, xref: int, ext: str) -> str | None:
        try:
            _name, real_ext, _kind, buffer = self._doc.extract_font(xref)
        except Exception:
            logger.warning(
                'Could not extract embedded font (xref %d) for reinsertion',
                xref,
            )
            return None
        if not buffer:
            return None
        path = os.path.join(
            self._tmpdir,
            f'{xref}.{real_ext or ext or "ttf"}',
        )
        with open(path, 'wb') as fh:
            fh.write(buffer)
        return path

    def cleanup(self) -> None:
        """Удаляет все извлечённые временные файлы шрифтов."""
        shutil.rmtree(self._tmpdir, ignore_errors=True)


def font_for_text(fontfile: str | None, text: str) -> str | None:
    """Подбирает файл шрифта, способный отрисовать каждый символ текста.

    PDF почти всегда встраивают *подмножественные* шрифты, которые
    обычно не несут пригодной для использования Unicode-таблицы
    символов (PDF сам сопоставляет символы глифам), поэтому извлечённый
    встроенный шрифт часто вообще не может отрисовать новый текст — он
    молча выдаёт пустые прямоугольники .notdef. В этом случае
    происходит откат на системный шрифт.

    Args:
        fontfile: Предпочитаемый файл шрифта (извлечённый из PDF), если
            есть.
        text: Текст, который предстоит вставить.

    Returns:
        Путь к файлу шрифта, покрывающему text, либо None, если ни один
        не подходит (в этом случае вызывающий код должен пропустить
        повторную вставку, а не выводить мусор).
    """
    if fontfile is not None and _font_covers(fontfile, text):
        return fontfile

    fallback = _system_fallback_font()
    if fallback is not None and _font_covers(fallback, text):
        return fallback
    return None


def _font_covers(fontfile: str, text: str) -> bool:
    """Проверяет, есть ли в файле шрифта глиф для каждого символа text."""
    font = _load_font(fontfile)
    if font is None:
        return False
    return all(font.has_glyph(ord(ch)) for ch in text if not ch.isspace())


@functools.lru_cache(maxsize=16)
def _load_font(fontfile: str) -> fitz.Font | None:
    """Загружает (и кэширует) fitz.Font для проверки покрытия глифами."""
    try:
        return fitz.Font(fontfile=fontfile)
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def _system_fallback_font() -> str | None:
    """Ищет системный шрифт, пригодный для восстановления не-PII текста.

    Returns:
        Путь к первому существующему кандидату шрифта, либо None — на
        минимальном образе Linux ни один может быть не установлен
        (установите, например, fonts-liberation или fonts-dejavu).
    """
    for path in _FALLBACK_FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    logger.warning(
        'No system fallback font found — non-PII text on redacted lines '
        'cannot be restored. Install fonts-liberation or fonts-dejavu.',
    )
    return None


def redact_page_boxes(
    page: fitz.Page,
    boxes: list[BBox],
    padding: float = PDF_REDACTION_PADDING,
) -> None:
    """Необратимо редактирует набор bounding box'ов на одной странице.

    Используется для пути OCR-фолбэка, где редактируемое содержимое —
    это пиксели изображения (а не span текста PDF), поэтому вставлять
    нечего и риска протечки span'а нет.

    Args:
        page: Страница PyMuPDF для редактирования.
        boxes: Bounding box'ы (x0, y0, x1, y1) в координатах страницы
            для редактирования. Ничего не делает, если пусто.
        padding: Дополнительный отступ вокруг каждого box'а, чтобы
            полностью покрыть выносные элементы букв и неточность OCR
            bbox.
    """
    if not boxes:
        return

    for x0, y0, x1, y1 in boxes:
        rect = fitz.Rect(
            x0 - padding,
            y0 - padding,
            x1 + padding,
            y1 + padding,
        )
        page.add_redact_annot(rect, fill=(0, 0, 0))

    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS)
    logger.debug(
        'Applied %d redaction(s) on page %d',
        len(boxes),
        page.number,
    )


def redact_text_layer(
    page: fitz.Page,
    matched_words: list[WordBox],
    all_words: list[WordBox],
    font_cache: FontCache,
    padding: float = PDF_REDACTION_PADDING,
) -> None:
    """Редактирует PII на странице с текстом, не повреждая соседний текст.

    Редактирует каждый затронутый span PDF целиком (единственная
    гранулярность, которую ``apply_redactions()`` может безопасно
    удалить — см. докстринг модуля), затем заново вставляет не-PII
    слова этого span'а исходным встроенным шрифтом/размером/базовой
    линией, так что теряется только сам текст PII.

    Args:
        page: Страница PyMuPDF для редактирования (изменяется на
            месте).
        matched_words: Слова, пересекающиеся с обнаруженным PII-спаном.
        all_words: Все слова на странице (совпавшие и нет),
            используются для поиска выживших (не-PII) слов каждого
            затронутого span'а.
        font_cache: Общий экстрактор/кэш встроенных шрифтов для
            повторной вставки.
        padding: Дополнительный отступ вокруг собственного bbox
            каждого span'а.
    """
    if not matched_words:
        return

    spans = _get_page_spans(page)
    matched_ids = {id(w) for w in matched_words}

    affected_spans: dict[TextSpan, None] = {}
    unmapped_boxes: list[BBox] = []
    for word in matched_words:
        span = _find_span_for_word(spans, word.bbox)
        if span is None:
            unmapped_boxes.append(word.bbox)
            continue
        affected_spans[span] = None

    # Редактирование БЕЗ заливки: весь span должен быть удалён (это и
    # есть гранулярность, на которой работает apply_redactions), но
    # закрашивание его чёрным скрыло бы не-PII текст, вставленный ниже
    # заново. Чёрные полосы рисуются позже, только поверх PII-слов.
    for x0, y0, x1, y1 in unmapped_boxes:
        rect = fitz.Rect(
            x0 - padding, y0 - padding, x1 + padding, y1 + padding
        )
        page.add_redact_annot(rect)

    survivors_by_span: dict[TextSpan, list[WordBox]] = {}
    fontfile_by_span: dict[TextSpan, str | None] = {}
    for span in affected_spans:
        # Извлекаем шрифт ДО apply_redactions() — как только
        # единственное использование шрифта в span'е отредактировано,
        # PyMuPDF может убрать этот ресурс шрифта со страницы, и
        # page.get_fonts() перестаёт его перечислять.
        fontfile_by_span[span] = font_cache.path_for(page, span.font)

        # Без отступа здесь: span.bbox уже плотно охватывает весь
        # диапазон глифов строки (он берётся из собственных метрик
        # шрифта PyMuPDF, а не из приблизительного bbox слова), а
        # отступ на соседних, плотно расположенных строках может
        # залезть в bbox соседнего span'а — из-за чего весь этот
        # соседний span тоже будет стёрт (та самая протечка span'а,
        # ради избежания которой существует эта функция), при этом не
        # будучи здесь зарегистрированным для восстановления.
        page.add_redact_annot(fitz.Rect(*span.bbox))
        survivors_by_span[span] = [
            w
            for w in all_words
            if id(w) not in matched_ids
            and _find_span_for_word(spans, w.bbox) is span
        ]

    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS)

    restored, skipped = 0, 0
    for span, survivors in survivors_by_span.items():
        if not survivors:
            continue
        color = _int_to_rgb(span.color)
        for word in survivors:
            fontfile = font_for_text(fontfile_by_span[span], word.text)
            if fontfile is None:
                skipped += 1
                continue
            page.insert_text(
                (word.bbox[0], span.origin_y),
                word.text,
                fontsize=span.size,
                fontname=_font_alias(fontfile),
                fontfile=fontfile,
                color=color,
            )
            restored += 1

    # Только косметика — сам текст PII уже удалён из потока содержимого;
    # эти полосы лишь отмечают, где он был.
    for word in matched_words:
        x0, y0, x1, y1 = word.bbox
        page.draw_rect(
            fitz.Rect(x0 - padding, y0 - padding, x1 + padding, y1 + padding),
            color=None,
            fill=(0, 0, 0),
        )

    logger.debug(
        'Redacted %d span(s) on page %d; restored %d non-PII word(s), '
        'skipped %d (font not reinsertable)',
        len(affected_spans),
        page.number,
        restored,
        skipped,
    )


def _get_page_spans(page: fitz.Page) -> list[TextSpan]:
    """Извлекает собственные границы span'ов и метаданные шрифтов страницы.

    Args:
        page: Страница PyMuPDF для анализа.

    Returns:
        По одному TextSpan на каждый прогон шрифта, сообщённый
        ``get_text("dict")``.
    """
    spans: list[TextSpan] = []
    text_dict = page.get_text('dict')
    for block in text_dict.get('blocks', []):
        if block.get('type') != 0:
            continue
        for line in block.get('lines', []):
            for span in line.get('spans', []):
                spans.append(
                    TextSpan(
                        bbox=tuple(span['bbox']),
                        font=span['font'],
                        size=span['size'],
                        color=span['color'],
                        origin_y=span['origin'][1],
                    ),
                )
    return spans


def _find_span_for_word(
    spans: list[TextSpan],
    word_bbox: BBox,
    tolerance: float = 1.0,
) -> TextSpan | None:
    """Находит, в какой span попадает центр bbox слова.

    Args:
        spans: Кандидаты span'ов (из _get_page_spans).
        word_bbox: Bounding box слова.
        tolerance: Дополнительный отступ (в пунктах) на погрешность
            округления по краям.

    Returns:
        Содержащий TextSpan, либо None, если ни один span не подошёл.
    """
    cx = (word_bbox[0] + word_bbox[2]) / 2
    cy = (word_bbox[1] + word_bbox[3]) / 2
    for span in spans:
        x0, y0, x1, y1 = span.bbox
        if (
            x0 - tolerance <= cx <= x1 + tolerance
            and y0 - tolerance <= cy <= y1 + tolerance
        ):
            return span
    return None


def _font_alias(fontfile: str) -> str:
    """Стабильный псевдоним ресурса PDF для файла шрифта.

    Без явного псевдонима insert_text() откатывается на "helv"
    (базовый шрифт base-14 без кириллицы) и игнорирует переданный
    fontfile.
    """
    digest = hashlib.sha1(fontfile.encode('utf-8')).hexdigest()[:8]
    return f'PG{digest}'


def _strip_subset_prefix(font_name: str) -> str:
    """Отрезает префикс подмножества шрифта PDF (6 заглавных букв + "+")."""
    prefix, sep, rest = font_name.partition('+')
    if sep and len(prefix) == 6 and prefix.isupper() and prefix.isalpha():
        return rest
    return font_name


def _int_to_rgb(color: int) -> tuple[float, float, float]:
    """Конвертирует 24-битное целое sRGB (0xRRGGBB) в кортеж float RGB 0-1."""
    return (
        ((color >> 16) & 255) / 255,
        ((color >> 8) & 255) / 255,
        (color & 255) / 255,
    )
