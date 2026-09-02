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
    ) -> None:
        """Detect and redact PII on a single page, updating result stats.

        Args:
            page: PyMuPDF page to process (mutated in place).
            entity_types: Optional entity-type filter (see anonymize()).
            redact_urls: Whether to include URLs (see anonymize()).
            result: Result object to accumulate statistics into.
            font_cache: Shared embedded-font cache for text reinsertion.
        """
        is_text_layer = text_extractor.page_has_text_layer(page)
        blocks = (
            text_extractor.extract_page_text_blocks(page)
            if is_text_layer
            else ocr.extract_page_text_blocks_ocr(page)
        )

        all_words: list[WordBox] = []
        matched_words: list[WordBox] = []
        for block in blocks:
            all_words.extend(block.words)
            matched_words.extend(
                self._collect_matched_words(
                    block,
                    entity_types,
                    redact_urls,
                    result,
                ),
            )

        if is_text_layer:
            renderer.redact_text_layer(
                page,
                matched_words,
                all_words,
                font_cache,
            )
        else:
            renderer.redact_page_boxes(
                page,
                [word.bbox for word in matched_words],
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
