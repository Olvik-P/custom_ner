"""Публичный API для анонимизации PII прямо внутри файлов PDF.

Оркеструет по каждой странице: выбор извлечения текстового слоя или
OCR-фолбэка -> PIIDetector.detect() на восстановленном тексте блока ->
маппинг совпавших спанов на bounding box'ы слов -> необратимое
редактирование через renderer.py.

Для страниц с текстовым слоем редактирование идёт через
renderer.redact_text_layer, который редактирует целые span'ы PDF
(гранулярность, на которой реально работает apply_redactions PyMuPDF)
и заново вставляет не-PII слова каждого span'а исходным встроенным
шрифтом — почему именно так, см. докстринг модуля renderer.py.

Использование:
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

# Ссылки (например, на публичные государственные порталы/реестры КНМ)
# часто нужны тому, кто читает отредактированный документ, поэтому — в
# отличие от других типов PII — URL по умолчанию исключён из
# редактирования. Передайте redact_urls=True, чтобы включить его, или
# явно укажите "URL" в entity_types, чтобы редактировать только URL.
_URL_ENTITY_TYPE = 'URL'


@dataclass
class PDFAnonymizationResult:
    """Результат анонимизации одного файла PDF.

    Никогда не несёт исходные значения PII — только счётчики по типу
    сущности, в соответствии с инвариантом аудита проекта (см.
    AuditLogger).

    Attributes:
        input_path: Путь к исходному PDF.
        output_path: Путь, по которому был (или будет) сохранён
            отредактированный PDF.
        pages_processed: Число страниц в исходном документе.
        total_spans_redacted: Общее число отредактированных PII-спанов.
        redacted_by_type: Число отредактированных спанов по каждому
            типу PII-сущности.
        success: Завершилась ли анонимизация без ошибок.
        error_message: Описание ошибки, если success is False.
    """

    input_path: str
    output_path: str
    pages_processed: int = 0
    total_spans_redacted: int = 0
    redacted_by_type: dict[str, int] = field(default_factory=dict)
    success: bool = False
    error_message: str | None = None


class PDFAnonymizer:
    """Публичный API для анонимизации PII в файлах PDF.

    Использование:
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
        """Анонимизирует PII в файле PDF.

        Args:
            input_pdf: Путь к исходному PDF. Должен существовать.
            output_pdf: Путь для отредактированного PDF. Если None,
                добавляет "_redacted" к имени исходного файла.
            entity_types: Опциональный список типов PII-сущностей для
                редактирования (например, ["PER", "PHONE"]). Если None,
                редактируются все обнаруженные типы, кроме URL (см.
                redact_urls). Явное указание "URL" здесь всегда
                редактирует его, независимо от redact_urls.
            redact_urls: Когда entity_types is None, также редактировать
                обнаруженные URL (по умолчанию оставлены видимыми —
                читателям часто нужны собственные ссылки документа,
                например запись в публичном реестре). Не имеет эффекта,
                если entity_types задан явно.

        Returns:
            PDFAnonymizationResult со статистикой. При сбое success
            равен False и по output_path не остаётся
            отредактированного файла.

        Raises:
            FileNotFoundError: Если input_pdf не существует.
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
        # Отсутствие Tesseract имеет одну и ту же причину и следствие на
        # каждой странице документа — предупреждаем об этом один раз, а
        # не на каждой странице.
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
        """Обнаруживает и редактирует PII на странице, обновляя статистику.

        Args:
            page: Страница PyMuPDF для обработки (изменяется на месте).
            entity_types: Опциональный фильтр по типу сущности (см.
                anonymize()).
            redact_urls: Включать ли URL (см. anonymize()).
            result: Объект результата для накопления статистики.
            font_cache: Общий кэш встроенных шрифтов для повторной
                вставки текста.
            ocr_unavailable_warned: Флаг из одного элемента, общий для
                всех страниц этого документа, чтобы отсутствие движка
                OCR логировалось один раз за вызов anonymize(), а не на
                каждой странице.
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
            # Дополняющее, а не взаимоисключающее: на странице может
            # быть одновременно (частичный) текстовый слой и
            # содержимое-изображение со своим собственным PII (например,
            # маленький настоящий текстовый штамп с датой на в остальном
            # отсканированной странице). Если запускать текстовый путь
            # только при наличии текста, это изображение никогда не
            # попадёт в OCR — см. требование OCR-фолбэка в спеке
            # pdf-anonymization.
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
                    # OCR — *единственный* способ найти PII на этой
                    # странице — молчаливый пропуск оставил бы PII
                    # неотредактированным без какого-либо сигнала, а
                    # спека это явно запрещает.
                    raise
                # На этой странице уже есть настоящий текстовый слой,
                # который редактируется ниже; OCR здесь лишь подхватывает
                # дополнительный PII, который может находиться во
                # встроенном изображении (например, логотип,
                # отсканированная подпись) рядом с ним. Отсутствие
                # Tesseract не должно провалить редактирование текста,
                # который на этой странице точно есть — деградируем до
                # редактирования только текстового слоя и сообщаем об
                # этом. Ожидаемо/действенно (отсутствует опциональная
                # зависимость), поэтому логируем короткое сообщение, а
                # не трассировку стека, и только один раз на документ, а
                # не на каждую страницу.
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
        """Обнаруживает PII в блоке и сопоставляет совпадения словам.

        Args:
            block: Восстановленный текст на уровне блока с картой bbox
                его слов.
            entity_types: Опциональный фильтр по типу сущности (см.
                anonymize()).
            redact_urls: Включать ли URL (см. anonymize()).
            result: Объект результата для накопления статистики.

        Returns:
            Каждое слово, пересекающееся с совпавшим PII-спаном.
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
