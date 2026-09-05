"""Подпакет HTTP API для PrivacyGuard Pipeline.

Отдаёт PrivacyGuardPipeline.process() (а при установленной экстре
"pdf" — и PDFAnonymizer) через аутентифицированный HTTP-интерфейс, как
альтернативную точку входа к существующему CLI/программному API.

Требует опциональную группу зависимостей "api" (fastapi, uvicorn,
python-multipart). Этот подпакет никогда не импортируется остальной
верхнеуровневой поверхностью privacyguard_pipeline, поэтому
отсутствующая экстра "api" ломает только запуск самого сервера, а не
базовый импорт.
"""

from __future__ import annotations
