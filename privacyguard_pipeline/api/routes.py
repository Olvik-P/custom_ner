"""HTTP-маршруты для API PrivacyGuard."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from privacyguard_pipeline.api.auth import require_api_key
from privacyguard_pipeline.api.schemas import (
    AnonymizeRequest,
    AnonymizeResponse,
    HealthResponse,
    StatsResponse,
)
from privacyguard_pipeline.audit import audit_logger
from privacyguard_pipeline.constants import MAX_PDF_UPLOAD_BYTES
from privacyguard_pipeline.exceptions import PDFDependencyError
from privacyguard_pipeline.masker import Masker
from privacyguard_pipeline.pipeline import PrivacyGuardPipeline

if TYPE_CHECKING:
    from privacyguard_pipeline.detection import PIIDetector
    from privacyguard_pipeline.llm_proxy import LLMProxy

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get('/health', response_model=HealthResponse)
async def health() -> HealthResponse:
    """Проверка живости. Без аутентификации, без PII на входе/выходе."""
    return HealthResponse()


@router.post(
    '/v1/anonymize',
    response_model=AnonymizeResponse,
    dependencies=[Depends(require_api_key)],
)
async def anonymize(
    body: AnonymizeRequest,
    request: Request,
) -> AnonymizeResponse:
    """Прогоняет пайплайн анонимизация -> LLM -> деанонимизация.

    Собирает привязанный к запросу PrivacyGuardPipeline: детектор и
    LLM-прокси общие (создаются один раз при старте приложения — см.
    app.py), но Masker создаётся заново на каждый запрос, чтобы
    соответствие токенов одного запроса никогда не могло быть прочитано
    или очищено другим параллельным запросом.
    """
    detector: PIIDetector = request.app.state.detector
    llm_proxy: LLMProxy = request.app.state.llm_proxy
    pipeline = PrivacyGuardPipeline(
        detector=detector,
        masker=Masker(),
        llm_proxy=llm_proxy,
    )
    result = await pipeline.process(body.text, body.system_prompt)
    return AnonymizeResponse(**result)


@router.get(
    '/v1/stats',
    response_model=StatsResponse,
    dependencies=[Depends(require_api_key)],
)
async def stats() -> StatsResponse:
    """Статистика сессии: только типы/количества сущностей, никогда PII."""
    return StatsResponse.model_validate(audit_logger.get_stats())


@router.post(
    '/v1/anonymize/pdf',
    dependencies=[Depends(require_api_key)],
)
async def anonymize_pdf(file: UploadFile) -> FileResponse:
    """Анонимизирует PII в загруженном PDF, возвращая отредактированный файл.

    Возвращает понятную ошибку (а не общий 500), когда опциональная
    группа зависимостей "pdf" не установлена — в соответствии с уже
    существующим паттерном PDFDependencyError, используемым
    CLI/программным API.
    """
    try:
        from privacyguard_pipeline import PDFAnonymizer
    except PDFDependencyError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        ) from exc

    contents = await file.read()
    if len(contents) > MAX_PDF_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f'PDF exceeds the {MAX_PDF_UPLOAD_BYTES} byte upload limit'
            ),
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = Path(tmp_dir) / 'input.pdf'
        input_path.write_bytes(contents)

        # Выходной файл живёт вне TemporaryDirectory (который удаляется
        # в конце этого блока "with", до того как FileResponse успеет
        # его застримить) - он удаляется явно через BackgroundTask
        # после отправки ответа.
        output_fd, output_name = tempfile.mkstemp(suffix='.pdf')
        os.close(output_fd)
        output_path = Path(output_name)

        result = PDFAnonymizer().anonymize(
            input_pdf=input_path,
            output_pdf=output_path,
        )
        if not result.success:
            output_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=result.error_message or 'PDF anonymization failed',
            )

    stats_headers = {
        'X-Pages-Processed': str(result.pages_processed),
        'X-Total-Spans-Redacted': str(result.total_spans_redacted),
        'X-Redacted-By-Type': json.dumps(result.redacted_by_type),
    }
    return FileResponse(
        output_path,
        media_type='application/pdf',
        filename='redacted.pdf',
        headers=stats_headers,
        background=BackgroundTask(output_path.unlink, missing_ok=True),
    )
