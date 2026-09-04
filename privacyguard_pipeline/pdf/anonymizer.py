"""Public API for anonymizing PII directly inside PDF files.

Orchestrates, per page: choose text-layer extraction or OCR fallback ->
PIIDetector.detect() on the reconstructed block text -> map matched spans
to word bounding boxes -> irreversible redaction via renderer.py.

For text-layer pages, redaction goes through renderer.redact_text_layer,
which redacts whole PDF spans (the granularity PyMuPDF's apply_redactions
actually operates at) and reinserts each span's non-PII words using its
original embedded font — see renderer.py's module docstring for why.

Usage:
    from privacyguard_pipeline.pdf import PDFAnonymizer

    anonymizer = PDFAnonymizer()
    result = anonymizer.anonymize("contract.pdf", "contract_redacted.pdf")
    print(result.stats)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import fitz

from privacyguard_pipeline.detection import PIIDetector
from privacyguard_pipeline.exceptions import PDFDependencyError
from privacyguard_pipeline.pdf import ocr, renderer, text_extractor
from privacyguard_pipeline.pdf.text_extractor import TextBlock, WordBox

logger = logging.getLogger(__name__)

# Links (e.g. to public government portals/КНМ registries) are often
# needed by whoever reads a redacted document, so — unlike other PII
# types — URL is excluded from redaction by default. Pass
# redact_urls=True to include it, or list "URL" explicitly in
# entity_types to redact only URLs.
_URL_ENTITY_TYPE = 'URL'


@dataclass
class PDFAnonymizationResult:
    """Result of anonymizing one PDF file.

    Never carries the original PII values — only counts by entity type,
    matching the project's audit invariant (see AuditLogger).

    Attributes:
        input_path: Path to the source PDF.
        output_path: Path the redacted PDF was (or would be) saved to.
        pages_processed: Number of pages in the source document.
        total_spans_redacted: Total number of PII spans redacted.
        redacted_by_type: Count of redacted spans per PII entity type.
        success: Whether anonymization completed without errors.
        error_message: Error description if success is False.
    """

    input_path: str
    output_path: str
    pages_processed: int = 0
    total_spans_redacted: int = 0
    redacted_by_type: dict[str, int] = field(default_factory=dict)
    success: bool = False
    error_message: str | None = None


class PDFAnonymizer:
    """Public API for anonymizing PII in PDF files.

    Usage:
        anonymizer = PDFAnonymizer()
        result = anonymizer.anonymize("input.pdf", "output_redacted.pdf")
    """

    def __init__(self, detector: PIIDetector | None = None) -> None:
        self.detector = detector or PIIDetector()
        logger.info('PDFAnonymizer initialized')

    def anonymize(
        self,
        input_pdf: str | Path,
        output_pdf: str | Path | None = None,
        entity_types: list[str] | None = None,
        redact_urls: bool = False,
    ) -> PDFAnonymizationResult:
        """Anonymize PII in a PDF file.

        Args:
            input_pdf: Path to the source PDF. Must exist.
            output_pdf: Path for the redacted PDF. If None, appends
                "_redacted" to the input filename.
            entity_types: Optional list of PII entity types to redact
                (e.g. ["PER", "PHONE"]). If None, all detected types are
                redacted except URL (see redact_urls). Listing "URL"
                here explicitly always redacts it, regardless of
                redact_urls.
            redact_urls: When entity_types is None, also redact detected
                URLs (kept visible by default — readers often need a
                document's own links, e.g. a public registry entry).
                Has no effect when entity_types is given explicitly.

        Returns:
            PDFAnonymizationResult with statistics. On failure, success is
            False and no redacted file is left at output_path.

        Raises:
            FileNotFoundError: If input_pdf does not exist.
        """
        input_path = Path(input_pdf)
        if not input_path.exists():
            raise FileNotFoundError(f'PDF not found: {input_pdf}')

        output_path = (
            Path(output_pdf)
            if output_pdf is not None
            else input_path.with_name(
                f'{input_path.stem}_redacted{input_path.suffix}',
            )
        )
        result = PDFAnonymizationResult(
            input_path=str(input_path),
            output_path=str(output_path),
        )

        tmp_path = output_path.with_name(output_path.name + '.tmp')
        # Missing Tesseract has the same cause and consequence on every
        # page of a document — warn about it once, not once per page.
        ocr_unavailable_warned = [False]
        try:
            with fitz.open(str(input_path)) as doc:
                result.pages_processed = len(doc)
                font_cache = renderer.FontCache(doc)
                try:
                    for page in doc:
                        self._anonymize_page(
                            page,
                            entity_types,
                            redact_urls,
                            result,
                            font_cache,
                            ocr_unavailable_warned,
                        )
                    doc.save(str(tmp_path))
                finally:
                    font_cache.cleanup()
            tmp_path.replace(output_path)
            result.success = True
            logger.info(
                'PDF anonymization complete: %d spans redacted across '
                '%d pages',
                result.total_spans_redacted,
                result.pages_processed,
            )
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            result.error_message = str(exc)
            logger.error('PDF anonymization failed: %s', exc)

        return result

    def _anonymize_page(
        self,
        page: fitz.Page,
        entity_types: list[str] | None,
        redact_urls: bool,
        result: PDFAnonymizationResult,
        font_cache: renderer.FontCache,
        ocr_unavailable_warned: list[bool],
    ) -> None:
        """Detect and redact PII on a single page, updating result stats.

        Args:
            page: PyMuPDF page to process (mutated in place).
            entity_types: Optional entity-type filter (see anonymize()).
            redact_urls: Whether to include URLs (see anonymize()).
            result: Result object to accumulate statistics into.
            font_cache: Shared embedded-font cache for text reinsertion.
            ocr_unavailable_warned: Single-element flag shared across all
                pages of this document, so a missing OCR engine is
                logged once per anonymize() call, not once per page.
        """
        has_text_layer = text_extractor.page_has_text_layer(page)
        has_images = bool(page.get_images(full=False))

        text_blocks: list[TextBlock] = (
            text_extractor.extract_page_text_blocks(page)
            if has_text_layer
            else []
        )

        ocr_blocks: list[TextBlock] = []
        if not has_text_layer or has_images:
            # Additive, not exclusive: a page can have both a (partial)
            # text layer and image content with PII of its own (e.g. a
            # small real-text date stamp on an otherwise-scanned page).
            # Only running the text path when any text exists would
            # never OCR that image — see pdf-anonymization spec's OCR
            # fallback requirement.
            exclude_bboxes = [
                word.bbox for block in text_blocks for word in block.words
            ]
            try:
                ocr_blocks = ocr.extract_page_text_blocks_ocr(
                    page,
                    exclude_bboxes=exclude_bboxes,
                )
            except PDFDependencyError as exc:
                if not has_text_layer:
                    # OCR is the *only* way to find PII on this page —
                    # silently skipping it would leave PII unredacted
                    # with no signal, which the spec explicitly forbids.
                    raise
                # The page already has a real text layer being redacted
                # below; OCR here is only picking up extra PII that may
                # be sitting in an embedded image (e.g. a logo, a scanned
                # signature) alongside it. Missing Tesseract shouldn't
                # fail redaction of the text this page definitely has —
                # degrade to text-layer-only and say so. Expected/
                # actionable (a missing optional dependency), so log a
                # short message rather than a stack trace, and only
                # once per document rather than once per page.
                if not ocr_unavailable_warned[0]:
                    logger.warning(
                        'OCR unavailable (%s) — image content on '
                        'text-layer pages will not be checked for PII; '
                        'text-layer PII is still redacted normally.',
                        exc,
                    )
                    ocr_unavailable_warned[0] = True

        all_text_words: list[WordBox] = []
        matched_text_words: list[WordBox] = []
        for block in text_blocks:
            all_text_words.extend(block.words)
            matched_text_words.extend(
                self._collect_matched_words(
                    block,
                    entity_types,
                    redact_urls,
                    result,
                ),
            )

        matched_ocr_words: list[WordBox] = []
        for block in ocr_blocks:
            matched_ocr_words.extend(
                self._collect_matched_words(
                    block,
                    entity_types,
                    redact_urls,
                    result,
                ),
            )

        if text_blocks:
            renderer.redact_text_layer(
                page,
                matched_text_words,
                all_text_words,
                font_cache,
            )
        if matched_ocr_words:
            renderer.redact_page_boxes(
                page,
                [word.bbox for word in matched_ocr_words],
            )

    def _collect_matched_words(
        self,
        block: TextBlock,
        entity_types: list[str] | None,
        redact_urls: bool,
        result: PDFAnonymizationResult,
    ) -> list[WordBox]:
        """Detect PII in a block and map matches to words.

        Args:
            block: Reconstructed block-level text with its word bbox map.
            entity_types: Optional entity-type filter (see anonymize()).
            redact_urls: Whether to include URLs (see anonymize()).
            result: Result object to accumulate statistics into.

        Returns:
            Every word overlapping a matched PII span.
        """
        detection = self.detector.detect(block.text)
        matched: list[WordBox] = []

        for pii_span in detection.spans:
            if entity_types is not None:
                if pii_span.entity_type not in entity_types:
                    continue
            elif pii_span.entity_type == _URL_ENTITY_TYPE and not redact_urls:
                continue

            span_words = [
                word
                for word in block.words
                if word.start < pii_span.end and word.end > pii_span.start
            ]
            if not span_words:
                continue

            matched.extend(span_words)
            result.redacted_by_type[pii_span.entity_type] = (
                result.redacted_by_type.get(pii_span.entity_type, 0) + 1
            )
            result.total_spans_redacted += 1

        return matched
