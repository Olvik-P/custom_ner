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
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

IMAGE = 'ner-pipeline'
CONTAINER_NAME = 'ner-pipeline-smoketest'
PORT = 8420
API_KEY = secrets.token_hex(16)
SAMPLE_TEXT = 'Пациент Иванов Пётр Сергеевич, тел. +7(916)123-45-67'

REPO_ROOT = Path(__file__).resolve().parent
ENV_FILE = REPO_ROOT / 'privacyguard_pipeline' / '.env'


def _wait_for_health(timeout: float = 30.0) -> None:
    """Опрашивает /health, пока сервер не ответит или не истечёт таймаут."""
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f'http://localhost:{PORT}/health',
                timeout=2,
            ) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError) as exc:
            last_error = exc
        time.sleep(1)
    raise TimeoutError(
        f'API did not become healthy within {timeout}s: {last_error}',
    )


def _start_container() -> None:
    subprocess.run(
        ['docker', 'rm', '-f', CONTAINER_NAME],
        capture_output=True,
        check=False,
    )

    cmd = [
        'docker', 'run', '-d', '--rm', '--name', CONTAINER_NAME,
        '-v', f'{REPO_ROOT.as_posix()}:/workspace',
        '-w', '/workspace',
        '-p', f'{PORT}:{PORT}',
    ]
    if ENV_FILE.exists():
        cmd += ['--env-file', ENV_FILE.as_posix()]
    else:
        print(
            f'Warning: {ENV_FILE} not found - the LLM call step will '
            'run without an API key (see .env.example).',
            file=sys.stderr,
        )
    # -e имеет приоритет над --env-file, поэтому этот фиксированный
    # тестовый ключ всегда побеждает над тем, что задано в .env
    # (API_KEY/API_KEY_REQUIRED, если они там есть).
    cmd += ['-e', f'API_KEY={API_KEY}', '-e', 'API_KEY_REQUIRED=true']
    cmd += [IMAGE, 'python', '-m', 'privacyguard_pipeline.api']

    subprocess.run(cmd, check=True)


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
    _start_container()
    try:
        _wait_for_health()
        result = _call_anonymize()
    finally:
        subprocess.run(
            ['docker', 'stop', CONTAINER_NAME],
            capture_output=True,
            check=False,
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
