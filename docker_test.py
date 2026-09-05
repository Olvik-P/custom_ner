"""Ручной смоук-тест: дёргает POST /v1/anonymize у HTTP API внутри
Docker-контейнера вместо локального окружения.

Монтирует репозиторий в образ ner-pipeline и запускает там API-сервер
(из /workspace, то есть тестируется текущий код с диска, а не то, что
было запечено в образ при последней сборке `docker build`), затем
вызывает /health и /v1/anonymize по HTTP так же, как это делал бы
реальный клиент.

Требует, чтобы образ уже существовал:
    docker build -t ner-pipeline .

Использование:
    python docker_test.py
"""

from __future__ import annotations

import json
import secrets
import sys
import urllib.request

from docker_test_common import (
    PORT,
    start_container,
    stop_container,
    wait_for_health,
)

CONTAINER_NAME = 'ner-pipeline-smoketest'
API_KEY = secrets.token_hex(16)
SAMPLE_TEXT = 'Пациент Иванов Пётр Сергеевич, тел. +7(916)123-45-67'


def _call_anonymize() -> dict:
    payload = json.dumps({'text': SAMPLE_TEXT}).encode('utf-8')
    request = urllib.request.Request(
        f'http://localhost:{PORT}/v1/anonymize',
        data=payload,
        headers={
            'Content-Type': 'application/json',
            'X-API-Key': API_KEY,
        },
        method='POST',
    )
    with urllib.request.urlopen(request, timeout=60) as resp:
        return json.loads(resp.read())


def main() -> int:
    start_container(CONTAINER_NAME, API_KEY)
    try:
        wait_for_health()
        result = _call_anonymize()
    finally:
        stop_container(CONTAINER_NAME)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
