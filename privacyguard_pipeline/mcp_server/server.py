"""MCP-сервер PrivacyGuard: инструменты mask/demask/anonymize_pdf/detect.

Локальный сервер поверх stdio (см. __main__.py), защищающий вызывающего
MCP-агента от прямого доступа к сырым значениям PII — в отличие от
http-api, который обезличивает вокруг вызова внешнего LLM API,
инструменты здесь обезличивают вокруг того, что видит сама
модель-агент. Ни один инструмент не обращается к внешнему LLM API
(OpenAI-совместимому или Claude) — содержательную обработку
обезличенного текста между ``mask`` и ``demask`` выполняет сам
вызывающий агент.

Бизнес-логика каждого инструмента вынесена в отдельные функции
(``mask_text``, ``demask_text``, ``anonymize_pdf_to_path``,
``detect_text_file``, ``detect_pdf_file``), принимающие свои
зависимости (``PIIDetector``, ``MaskRegistry``) явными параметрами, а не
через MCP ``Context`` — сам ``Context`` недоступен вне активной
MCP-сессии, поэтому тестировать поведение инструментов проще напрямую
через эти функции, не поднимая полную MCP-сессию (см.
tests/mcp_server/).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from privacyguard_pipeline import __version__
from privacyguard_pipeline.config import settings
from privacyguard_pipeline.constants import (
    MCP_REGISTRY_SWEEP_INTERVAL_SECONDS,
)
from privacyguard_pipeline.detection import PIIDetector
from privacyguard_pipeline.exceptions import PDFDependencyError
from privacyguard_pipeline.masker import Masker
from privacyguard_pipeline.mcp_server.registry import (
    HandleNotFoundError,
    MaskRegistry,
)

logger = logging.getLogger(__name__)

_INSTRUCTIONS = (
    'Обезличивает PII в тексте/файлах до того, как их содержимое '
    'попадёт в контекст вызывающего агента. mask/demask работают парой '
    'в рамках одного диалога: mask возвращает обезличенный текст и '
    'handle, содержательная обработка выполняется самим агентом над '
    'обезличенным текстом, затем demask с тем же handle восстанавливает '
    'исходные значения в получившемся тексте. anonymize_pdf необратимо '
    'редактирует PII в PDF (без demask — редактирование необратимо). '
    'detect читает локальный файл на стороне сервера и возвращает '
    'только сводку по типам/количествам найденных сущностей, не '
    'раскрывая содержимое файла.'
)


@dataclass
class AppContext:
    """Тяжёлые синглтоны и session-scoped состояние сервера.

    Создаётся один раз при старте процесса (см. _lifespan), а не на
    каждый вызов инструмента — по аналогии с app.state в api/app.py.
    Только PIIDetector стоит держать синглтоном (загрузка моделей
    Natasha не бесплатна); PDFAnonymizer, наоборот, дёшево создать
    заново на каждый вызов anonymize_pdf/detect, если ему передать уже
    готовый detector.
    """

    detector: PIIDetector
    registry: MaskRegistry


async def _sweep_loop(registry: MaskRegistry) -> None:
    """Периодически чистит реестр от просроченных по TTL handle'ов.

    Дополняет немедленную проверку TTL внутри register()/pop() (см.
    MaskRegistry.sweep) на случай, если долго не происходит ни одного
    нового вызова mask/demask — без этой фоновой задачи забытый handle
    жил бы в памяти неограниченно долго (design.md - Decision 2).
    """
    while True:
        await asyncio.sleep(MCP_REGISTRY_SWEEP_INTERVAL_SECONDS)
        registry.sweep()


@asynccontextmanager
async def _lifespan(
    _server: MCPServer[AppContext],
) -> AsyncIterator[AppContext]:
    app_context = AppContext(
        detector=PIIDetector(),
        registry=MaskRegistry(ttl_seconds=settings.mcp_handle_ttl_seconds),
    )
    sweep_task = asyncio.create_task(_sweep_loop(app_context.registry))
    logger.info('PrivacyGuard MCP server started')
    try:
        yield app_context
    finally:
        sweep_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sweep_task
        logger.info('PrivacyGuard MCP server shut down')


server: MCPServer[AppContext] = MCPServer(
    'privacyguard-pipeline',
    version=__version__,
    instructions=_INSTRUCTIONS,
    lifespan=_lifespan,
)


def _app_context(ctx: Context[AppContext, Any]) -> AppContext:
    return ctx.request_context.lifespan_context


# ---------------------------------------------------------------------------
# mask / demask
# ---------------------------------------------------------------------------


def mask_text(
    text: str,
    detector: PIIDetector,
    registry: MaskRegistry,
) -> dict[str, str]:
    """Обнаруживает и маскирует PII, регистрируя Masker под новым handle.

    Реальные значения PII не входят в возвращаемое значение — только
    обезличенный текст и handle, нужный для последующего demask_text().
    """
    detection = detector.detect(text)
    masker = Masker()
    masked_text = masker.mask(text, detection.spans)
    handle = registry.register(masker)
    return {'masked_text': masked_text, 'handle': handle}


def demask_text(
    handle: str,
    text: str,
    registry: MaskRegistry,
) -> dict[str, str]:
    """Восстанавливает PII по handle и немедленно снимает запись из реестра.

    Raises:
        ToolError: Если handle никогда не выдавался, уже был
            использован предыдущим demask_text(), либо истёк по TTL.
    """
    try:
        masker = registry.pop(handle)
    except HandleNotFoundError as exc:
        raise ToolError(
            'Unknown mask handle: it was never issued, was already '
            'consumed by a previous demask call, or expired after '
            f'{settings.mcp_handle_ttl_seconds}s of inactivity.',
        ) from exc

    restored = masker.demask(text)
    masker.clear()
    return {'text': restored}


@server.tool()
async def mask(text: str, ctx: Context[AppContext, Any]) -> dict[str, str]:
    """Обнаруживает PII в тексте и заменяет его токенами.

    Возвращает обезличенный текст и handle. Реальные значения PII не
    возвращаются. Передайте тот же handle в demask вместе с текстом,
    который вы напишете поверх обезличенного текста, чтобы восстановить
    исходные значения.
    """
    app_context = _app_context(ctx)
    return mask_text(text, app_context.detector, app_context.registry)


@server.tool()
async def demask(
    handle: str,
    text: str,
    ctx: Context[AppContext, Any],
) -> dict[str, str]:
    """Восстанавливает PII в тексте по handle, ранее полученному от mask.

    Соответствие для этого handle удаляется сразу после успешного
    вызова — повторный вызов demask с тем же handle завершится ошибкой.
    """
    app_context = _app_context(ctx)
    return demask_text(handle, text, app_context.registry)


# ---------------------------------------------------------------------------
# anonymize_pdf
# ---------------------------------------------------------------------------


def anonymize_pdf_to_path(
    input_path: Path,
    output_path: Path | None,
    detector: PIIDetector | None = None,
) -> dict[str, Any]:
    """Необратимо редактирует PII в PDF-файле и пишет результат на диск.

    Ни исходное, ни отредактированное содержимое не проходят через эту
    функцию как байты в памяти дольше, чем нужно самому
    PDFAnonymizer — оба пути на диске передаются ему напрямую.

    Raises:
        ToolError: Если input_path не существует или недоступен, если
            опциональные PDF-зависимости не установлены, либо
            редактирование не удалось.
    """
    if not input_path.is_file():
        raise ToolError(
            f'File not found or not a regular file: {input_path}',
        )

    try:
        from privacyguard_pipeline import PDFAnonymizer
    except PDFDependencyError as exc:
        raise ToolError(str(exc)) from exc

    result = PDFAnonymizer(detector=detector).anonymize(
        input_pdf=input_path,
        output_pdf=output_path,
    )
    if not result.success:
        raise ToolError(result.error_message or 'PDF anonymization failed')

    return {
        'output_path': result.output_path,
        'pages_processed': result.pages_processed,
        'total_spans_redacted': result.total_spans_redacted,
        'redacted_by_type': result.redacted_by_type,
    }


@server.tool()
async def anonymize_pdf(
    input_path: str,
    ctx: Context[AppContext, Any],
    output_path: str | None = None,
) -> dict[str, Any]:
    """Необратимо редактирует PII на каждой странице PDF-файла.

    Принимает путь к исходному PDF-файлу и, опционально, путь для
    результата (если не задан — сохраняется рядом с исходником, как и у
    программного PDFAnonymizer.anonymize() по умолчанию). Сервер сам
    читает и пишет файлы — ни исходное, ни отредактированное содержимое
    не передаются в ответе, прочитайте результат по возвращённому
    output_path самостоятельно. В отличие от mask/demask, здесь нет пары
    для восстановления — редактирование необратимо.
    """
    app_context = _app_context(ctx)
    return anonymize_pdf_to_path(
        Path(input_path),
        Path(output_path) if output_path else None,
        detector=app_context.detector,
    )


# ---------------------------------------------------------------------------
# detect
# ---------------------------------------------------------------------------


def detect_text_file(path: Path, detector: PIIDetector) -> dict[str, Any]:
    """Читает текстовый файл и считает найденные PII-сущности по типу.

    Ни содержимое файла, ни сами значения PII не возвращаются.
    """
    try:
        text = path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolError(f'Could not read file: {exc}') from exc

    counts: dict[str, int] = {}
    for span in detector.detect(text).spans:
        counts[span.entity_type] = counts.get(span.entity_type, 0) + 1
    return {'entity_counts': counts}


def detect_pdf_file(path: Path, detector: PIIDetector) -> dict[str, Any]:
    """Читает PDF-файл и считает найденные PII-сущности по типу постранично.

    Ни содержимое файла, ни сами значения PII не возвращаются.

    Raises:
        ToolError: Если опциональные PDF-зависимости не установлены,
            либо файл не удалось обработать как PDF.
    """
    try:
        from privacyguard_pipeline.pdf import detect_page_entity_counts
    except PDFDependencyError as exc:
        raise ToolError(str(exc)) from exc
    import fitz

    pages: dict[str, dict[str, int]] = {}
    ocr_unavailable_warned = [False]
    try:
        with fitz.open(str(path)) as doc:
            for page in doc:
                pages[str(page.number + 1)] = detect_page_entity_counts(
                    page,
                    detector,
                    ocr_unavailable_warned,
                )
    except PDFDependencyError as exc:
        raise ToolError(str(exc)) from exc
    except Exception as exc:
        raise ToolError(f'Could not process PDF: {exc}') from exc

    return {'pages': pages}


def detect_file(file_path: str, detector: PIIDetector) -> dict[str, Any]:
    """Определяет тип файла по расширению и делегирует нужной ветке detect.

    Raises:
        ToolError: Если file_path не существует или не является
            обычным файлом.
    """
    path = Path(file_path)
    if not path.is_file():
        raise ToolError(f'File not found or not a regular file: {file_path}')

    if path.suffix.lower() == '.pdf':
        return detect_pdf_file(path, detector)
    return detect_text_file(path, detector)


@server.tool()
async def detect(
    file_path: str,
    ctx: Context[AppContext, Any],
) -> dict[str, Any]:
    """Читает локальный файл и возвращает сводку найденного PII по типу.

    Сервер сам читает файл по переданному пути — его содержимое, как и
    сами значения найденного PII, никогда не передаются вызывающему
    агенту, только количество сущностей по каждому типу: одним блоком
    для текстового файла, постранично для PDF.

    Рекомендуется вызывать этот инструмент первым для любого локального
    файла, который может содержать PII, прежде чем читать его напрямую:
    если entity_counts/pages пусты, файл можно безопасно читать обычным
    чтением файла (полная точность, включая вёрстку PDF); если найден
    PII — используйте mask_file вместо прямого чтения файла.
    """
    app_context = _app_context(ctx)
    return detect_file(file_path, app_context.detector)


# ---------------------------------------------------------------------------
# mask_file / close_masked_file
# ---------------------------------------------------------------------------


def _write_masked_file(path: Path, masked_text: str) -> Path:
    """Пишет маскированный текст рядом с исходным файлом.

    Всегда `.txt`, независимо от типа исходного файла (design.md -
    Decision 6) — маскированный результат всегда плоский текст, а не
    файл того же формата, что и оригинал.
    """
    masked_path = path.with_name(f'{path.stem}_masked.txt')
    masked_path.write_text(masked_text, encoding='utf-8')
    return masked_path


def mask_text_file(
    path: Path,
    detector: PIIDetector,
    registry: MaskRegistry,
) -> dict[str, str]:
    """Маскирует содержимое текстового файла, регистрируя Masker под handle.

    Raises:
        ToolError: Если файл не удалось прочитать как текст.
    """
    try:
        text = path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolError(f'Could not read file: {exc}') from exc

    masker = Masker()
    masked_text = masker.mask(text, detector.detect(text).spans)
    masked_path = _write_masked_file(path, masked_text)
    handle = registry.register(masker, masked_file_path=masked_path)
    return {'masked_path': str(masked_path), 'handle': handle}


def mask_pdf_file(
    path: Path,
    detector: PIIDetector,
    registry: MaskRegistry,
) -> dict[str, str]:
    """Маскирует извлечённый текст всех страниц PDF под одним handle.

    Один общий Masker на весь документ (design.md - Decision 3) — все
    страницы делят один handle, страницы в результате разделены
    заголовком `--- Страница N ---`.

    Raises:
        ToolError: Если опциональные PDF-зависимости не установлены,
            либо файл не удалось обработать как PDF.
    """
    try:
        from privacyguard_pipeline.pdf import extract_page_text
    except PDFDependencyError as exc:
        raise ToolError(str(exc)) from exc
    import fitz

    masker = Masker()
    ocr_unavailable_warned = [False]
    parts: list[str] = []
    try:
        with fitz.open(str(path)) as doc:
            for page in doc:
                page_text = extract_page_text(page, ocr_unavailable_warned)
                masked_page_text = masker.mask(
                    page_text,
                    detector.detect(page_text).spans,
                )
                parts.append(
                    f'--- Страница {page.number + 1} ---\n{masked_page_text}',
                )
    except PDFDependencyError as exc:
        raise ToolError(str(exc)) from exc
    except Exception as exc:
        raise ToolError(f'Could not process PDF: {exc}') from exc

    masked_path = _write_masked_file(path, '\n\n'.join(parts))
    handle = registry.register(masker, masked_file_path=masked_path)
    return {'masked_path': str(masked_path), 'handle': handle}


def mask_file_contents(
    file_path: str,
    detector: PIIDetector,
    registry: MaskRegistry,
) -> dict[str, str]:
    """Определяет тип файла по расширению и делегирует нужной ветке mask.

    Raises:
        ToolError: Если file_path не существует или не является
            обычным файлом.
    """
    path = Path(file_path)
    if not path.is_file():
        raise ToolError(f'File not found or not a regular file: {file_path}')

    if path.suffix.lower() == '.pdf':
        return mask_pdf_file(path, detector, registry)
    return mask_text_file(path, detector, registry)


def close_masked_file_handle(handle: str, registry: MaskRegistry) -> None:
    """Завершает работу с handle от mask_file, не восстанавливая PII.

    Удаляет маскированный файл, связанный с handle (через `registry.pop`,
    см. `MaskRegistry.pop`), и делает handle недействительным для
    последующих вызовов `demask`/`close_masked_file` — без необходимости
    передавать какой-либо текст (design.md - Decision 4). Используется,
    когда итоговый ответ агента не должен содержать реальные значения
    PII, поэтому `demask` не нужен, а файл всё равно нужно убрать раньше
    TTL.

    Raises:
        ToolError: Если handle никогда не выдавался, уже был закрыт
            (через demask или предыдущий close_masked_file), либо истёк
            по TTL.
    """
    try:
        registry.pop(handle)
    except HandleNotFoundError as exc:
        raise ToolError(
            'Unknown mask handle: it was never issued, was already '
            'closed by a previous demask/close_masked_file call, or '
            f'expired after {settings.mcp_handle_ttl_seconds}s of '
            'inactivity.',
        ) from exc


@server.tool()
async def mask_file(
    file_path: str,
    ctx: Context[AppContext, Any],
) -> dict[str, str]:
    """Маскирует содержимое локального файла (текстового или PDF).

    Сервер сам читает файл по переданному пути, заменяет найденный PII
    токенами того же вида, что и mask, и возвращает путь к
    маскированной копии (всегда плоский текст) вместе с handle —
    содержимое файла в ответе на этот вызов не передаётся, читайте
    результат напрямую по masked_path. Передайте тот же handle в demask
    вместе с текстом вашего ответа, чтобы восстановить реальные значения
    только там, где это нужно, либо в close_masked_file, если реальные
    значения в ответе не нужны вовсе.

    Рекомендуется вызывать detect первым: используйте mask_file вместо
    прямого чтения файла только если detect показал ненулевые счётчики
    сущностей для этого файла.
    """
    app_context = _app_context(ctx)
    return mask_file_contents(
        file_path,
        app_context.detector,
        app_context.registry,
    )


@server.tool()
async def close_masked_file(
    handle: str,
    ctx: Context[AppContext, Any],
) -> dict[str, str]:
    """Завершает работу с маскированным файлом от mask_file без demask.

    Удаляет маскированный файл с диска и делает handle недействительным.
    Не восстанавливает и не принимает никакой текст — используйте этот
    инструмент вместо demask, когда итоговый ответ не должен содержать
    реальные значения PII, но маскированный файл всё равно пора убрать,
    не дожидаясь TTL.
    """
    close_masked_file_handle(handle, _app_context(ctx).registry)
    return {'status': 'closed'}
