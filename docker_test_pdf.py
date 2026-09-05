"""Ручной смоук-тест: дёргает POST /v1/anonymize/pdf у HTTP API внутри
Docker-контейнера вместо локального окружения.

Монтирует репозиторий в образ ner-pipeline и запускает там API-сервер
(из /workspace, то есть тестируется текущий код с диска) - особенно
полезно для OCR-фолбэка, так как в контейнере уже есть Tesseract +
языковой пакет "rus" и не нужна системная установка/настройка PATH, в
отличие от запуска PDFAnonymizer напрямую на Windows. Загружает
настоящий PDF по HTTP так же, как это делал бы реальный клиент,
используя curl (идёт в комплекте с Windows 10/11) для multipart-загрузки.

Требует, чтобы образ уже существовал:
    docker build -t ner-pipeline .

Использование:
    python docker_test_pdf.py [input.pdf] [output.pdf]
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

IMAGE = 'ner-pipeline'
CONTAINER_NAME = 'ner-pipeline-pdf-smoketest'
PORT = 8420
API_KEY = 'docker-test-pdf-smoketest-key'
DEFAULT_INPUT = Path('docs') / 'Решение (безбумажное).pdf'

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
    # -e имеет приоритет над --env-file, поэтому этот фиксированный
    # тестовый ключ всегда побеждает над тем, что задано в .env
    # (API_KEY/API_KEY_REQUIRED, если они там есть).
    cmd += ['-e', f'API_KEY={API_KEY}', '-e', 'API_KEY_REQUIRED=true']
    cmd += [IMAGE, 'python', '-m', 'privacyguard_pipeline.api']

    subprocess.run(cmd, check=True)


def main(argv: list[str]) -> int:
    input_pdf = Path(argv[0]) if len(argv) > 0 else DEFAULT_INPUT
    output_pdf = (
        Path(argv[1])
        if len(argv) > 1
        else input_pdf.with_name(
            f'{input_pdf.stem}_redacted{input_pdf.suffix}',
        )
    )

    if not input_pdf.exists():
        print(f'Входной файл не найден: {input_pdf}', file=sys.stderr)
        return 1

    print(f'Входной файл:  {input_pdf}')
    print(f'Выходной файл: {output_pdf}')
    print()

    _start_container()
    try:
        _wait_for_health()
        headers_file = output_pdf.with_suffix('.headers.txt')
        curl_cmd = [
            'curl', '-s', '-w', '\nHTTP:%{http_code}\n',
            '-X', 'POST', f'http://localhost:{PORT}/v1/anonymize/pdf',
            '-H', f'X-API-Key: {API_KEY}',
            '-F', f'file=@{input_pdf};type=application/pdf',
            '-D', str(headers_file),
            '-o', str(output_pdf),
        ]
        result = subprocess.run(
            curl_cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    finally:
        subprocess.run(
            ['docker', 'stop', CONTAINER_NAME],
            capture_output=True,
            check=False,
        )

    print(result.stdout.strip())
    if headers_file.exists():
        for line in headers_file.read_text(encoding='utf-8').splitlines():
            if line.lower().startswith('x-'):
                print(line)
        headers_file.unlink()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
