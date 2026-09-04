"""True PDF redaction — removes content, not just visual overlay.

Uses PyMuPDF's redaction annotations (``add_redact_annot`` +
``apply_redactions``), which strip the underlying text objects and blank
image pixels under the redaction box, instead of drawing a cosmetic black
rectangle on top of content that would remain extractable underneath.

PyMuPDF's ``apply_redactions()`` removes text at the granularity of a PDF
"span" (a contiguous same-font run in the content stream, e.g. one
justified line drawn as a single text-showing operator) — a redaction
rectangle overlapping any part of a span can drop characters throughout
that whole span, not just the part under the rectangle. ``redact_text_layer``
handles this by redacting the *whole* affected span and then re-inserting
the span's non-PII ("surviving") words using the original embedded font,
size and baseline, so only the actual PII text is lost.
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

# Fonts used to restore non-PII text when the PDF's own embedded font
# cannot be reused (see font_for_text). Ordered by preference.
_FALLBACK_FONT_CANDIDATES = (
    '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    'C:/Windows/Fonts/times.ttf',
    'C:/Windows/Fonts/arial.ttf',
)


@dataclass(frozen=True)
class TextSpan:
    """A PyMuPDF text span: one homogeneous font/style run.

    This is the smallest unit ``apply_redactions()`` can safely remove
    without risking damage to neighboring text (see module docstring).

    Attributes:
        bbox: Span bounding box (x0, y0, x1, y1) in page coordinates.
        font: Embedded font's base name (e.g. "BAAAAA+LiberationSerif").
        size: Font size in points.
        color: Text color as a 24-bit sRGB integer (0xRRGGBB).
        origin_y: Baseline y-coordinate for this span (horizontal text
            has one constant baseline per span).
    """

    bbox: BBox
    font: str
    size: float
    color: int
    origin_y: float


class FontCache:
    """Extracts embedded fonts to temp files, cached per font name.

    ``insert_text()`` only accepts a font *file path*, not raw bytes, so
    each distinct embedded font used for reinsertion is written to a
    temp file once and reused. Call ``cleanup()`` when done.
    """

    def __init__(self, doc: fitz.Document) -> None:
        self._doc = doc
        self._tmpdir = tempfile.mkdtemp(prefix='privacyguard_pdf_fonts_')
        # Keyed by xref (a document-global id for the specific embedded
        # font *object*), not by font_name: two pages' different subsets
        # of a same-named font (e.g. both "LiberationSerif" but each
        # embedding only that page's used glyphs, under different xrefs)
        # must not share a cache entry — reusing one page's subset for
        # another silently drops non-PII words the wrong subset lacks
        # glyphs for (see font_for_text/_font_covers).
        self._paths: dict[int, str | None] = {}

    def path_for(self, page: fitz.Page, font_name: str) -> str | None:
        """Get a local file path for an embedded font, extracting once.

        Args:
            page: Page the font is used on (to resolve its xref).
            font_name: Span's font base name to look up.

        Returns:
            Path to the extracted font file, or None if it could not be
            extracted (e.g. a non-embeddable Type3/CID font) — callers
            should skip reinsertion for that font rather than guess.
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
        # get_text("dict") reports span['font'] WITHOUT the PDF subset
        # prefix (e.g. "LiberationSerif"), while get_fonts() reports the
        # raw resource name WITH it (e.g. "BAAAAA+LiberationSerif") — the
        # prefix is always exactly 6 uppercase letters + "+" per the PDF
        # spec, so strip it before comparing. This scan is cheap
        # metadata-only (no font bytes read) and runs on every lookup —
        # only the byte extraction below is cache-guarded.
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
        """Remove all extracted font temp files."""
        shutil.rmtree(self._tmpdir, ignore_errors=True)


def font_for_text(fontfile: str | None, text: str) -> str | None:
    """Pick a font file able to render every character in text.

    PDFs almost always embed *subset* fonts, which usually carry no
    usable Unicode cmap (the PDF maps characters to glyphs itself), so
    an extracted embedded font often cannot render new text at all —
    it silently produces .notdef boxes. Falls back to a system font
    when that happens.

    Args:
        fontfile: Preferred font file (extracted from the PDF), if any.
        text: The text about to be inserted.

    Returns:
        A font file path that covers text, or None if none is available
        (caller should then skip reinsertion rather than emit garbage).
    """
    if fontfile is not None and _font_covers(fontfile, text):
        return fontfile

    fallback = _system_fallback_font()
    if fallback is not None and _font_covers(fallback, text):
        return fallback
    return None


def _font_covers(fontfile: str, text: str) -> bool:
    """Check a font file has a glyph for every character in text."""
    font = _load_font(fontfile)
    if font is None:
        return False
    return all(font.has_glyph(ord(ch)) for ch in text if not ch.isspace())


@functools.lru_cache(maxsize=16)
def _load_font(fontfile: str) -> fitz.Font | None:
    """Load (and cache) a fitz.Font for glyph-coverage checks."""
    try:
        return fitz.Font(fontfile=fontfile)
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def _system_fallback_font() -> str | None:
    """Find a system font usable for reinserting non-PII text.

    Returns:
        Path to the first existing candidate font, or None — on a
        minimal Linux image none may be installed (install e.g.
        fonts-liberation or fonts-dejavu).
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
    """Irreversibly redact a set of bounding boxes on one page.

    Used for the OCR fallback path, where redacted content is image
    pixels (not a PDF text span), so there is nothing to reinsert and no
    span-bleed risk.

    Args:
        page: PyMuPDF page to redact.
        boxes: Bounding boxes (x0, y0, x1, y1) in page coordinates to
            redact. No-op if empty.
        padding: Extra margin added around each box to fully cover
            ascenders/descenders and OCR bbox imprecision.
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
    """Redact PII on a text-layer page without corrupting neighboring text.

    Redacts each affected PDF span in full (the only granularity
    ``apply_redactions()`` can safely remove — see module docstring),
    then reinserts that span's non-PII words using the original embedded
    font/size/baseline so only the actual PII text is lost.

    Args:
        page: PyMuPDF page to redact (mutated in place).
        matched_words: Words overlapping a detected PII span.
        all_words: Every word on the page (matched and not), used to
            find each affected span's surviving (non-PII) words.
        font_cache: Shared embedded-font extractor/cache for reinsertion.
        padding: Extra margin added around each span's own bbox.
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

    # Redact WITHOUT a fill: the whole span has to be removed (that is
    # the granularity apply_redactions works at), but painting it black
    # would hide the non-PII text reinserted below it. The black bars
    # are drawn afterwards, over the PII words only.
    for x0, y0, x1, y1 in unmapped_boxes:
        rect = fitz.Rect(
            x0 - padding, y0 - padding, x1 + padding, y1 + padding
        )
        page.add_redact_annot(rect)

    survivors_by_span: dict[TextSpan, list[WordBox]] = {}
    fontfile_by_span: dict[TextSpan, str | None] = {}
    for span in affected_spans:
        # Extract the font BEFORE apply_redactions() — once a span's
        # only usage of a font is redacted, PyMuPDF may drop that font
        # resource from the page, and page.get_fonts() stops listing it.
        fontfile_by_span[span] = font_cache.path_for(page, span.font)

        # No padding here: span.bbox already tightly encloses the whole
        # line's glyph extent (it comes from PyMuPDF's own font metrics,
        # not an approximate word box), and padding on adjacent, closely
        # spaced lines can cross into the neighboring span's bbox — which
        # gets that whole neighboring span wiped too (same span-bleed
        # this function exists to avoid), without it being registered
        # here for restoration.
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

    # Cosmetic only — the PII text itself is already gone from the
    # content stream; these bars just mark where it was.
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
    """Extract PyMuPDF's own span boundaries and font metadata for a page.

    Args:
        page: PyMuPDF page to inspect.

    Returns:
        One TextSpan per font run reported by ``get_text("dict")``.
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
    """Find which span a word's bbox center falls inside.

    Args:
        spans: Candidate spans (from _get_page_spans).
        word_bbox: The word's bounding box.
        tolerance: Extra margin (points) for rounding slack at edges.

    Returns:
        The containing TextSpan, or None if no span matches.
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
    """Stable PDF resource alias for a font file.

    Without an explicit alias, insert_text() falls back to "helv" (a
    base-14 font with no Cyrillic) and ignores the supplied fontfile.
    """
    digest = hashlib.sha1(fontfile.encode('utf-8')).hexdigest()[:8]
    return f'PG{digest}'


def _strip_subset_prefix(font_name: str) -> str:
    """Strip a PDF font subset prefix (6 uppercase letters + "+"), if any."""
    prefix, sep, rest = font_name.partition('+')
    if sep and len(prefix) == 6 and prefix.isupper() and prefix.isalpha():
        return rest
    return font_name


def _int_to_rgb(color: int) -> tuple[float, float, float]:
    """Convert a 24-bit sRGB integer (0xRRGGBB) to a 0-1 float RGB tuple."""
    return (
        ((color >> 16) & 255) / 255,
        ((color >> 8) & 255) / 255,
        (color & 255) / 255,
    )
