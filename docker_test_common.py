"""Общие хелперы для ручных смоук-тестов docker_test.py/docker_test_pdf.py.

Оба скрипта поднимают контейнер ner-pipeline с кодом текущего диска
(volume-mount /workspace) и дожидаются готовности HTTP API по одной и
той же схеме - вынесено сюда, чтобы не дублировать между ними. Этот
файл не часть тестового набора (testpaths = tests/) и не импортируется
пакетом privacyguard_pipeline.
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

IMAGE = 'ner-pipeline'
PORT = 8420

REPO_ROOT = Path(__file__).resolve().parent
ENV_FILE = REPO_ROOT / 'privacyguard_pipeline' / '.env'


def wait_for_health(timeout: float = 30.0) -> None:
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


def start_container(
    container_name: str,
    api_key: str,
    *,
    warn_missing_env: bool = True,
) -> None:
    """Поднимает контейнер IMAGE с кодом текущего диска и заданным ключом."""
    subprocess.run(
        ['docker', 'rm', '-f', container_name],
        capture_output=True,
        check=False,
    )

    cmd = [
        'docker',
        'run',
        '-d',
        '--rm',
        '--name',
        container_name,
        '-v',
        f'{REPO_ROOT.as_posix()}:/workspace',
        '-w',
        '/workspace',
        '-p',
        f'{PORT}:{PORT}',
    ]
    if ENV_FILE.exists():
        cmd += ['--env-file', ENV_FILE.as_posix()]
    elif warn_missing_env:
        print(
            f'Warning: {ENV_FILE} not found - the LLM call step will '
            'run without an API key (see .env.example).',
            file=sys.stderr,
        )
    # -e имеет приоритет над --env-file, поэтому этот фиксированный
    # тестовый ключ всегда побеждает над тем, что задано в .env
    # (API_KEY/API_KEY_REQUIRED, если они там есть).
    cmd += ['-e', f'API_KEY={api_key}', '-e', 'API_KEY_REQUIRED=true']
    cmd += [IMAGE, 'python', '-m', 'privacyguard_pipeline.api']

    subprocess.run(cmd, check=True)


def stop_container(container_name: str) -> None:
    subprocess.run(
        ['docker', 'stop', container_name],
        capture_output=True,
        check=False,
    )
