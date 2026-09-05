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
from pathlib import Path

from docker_test_common import (
    PORT,
    start_container,
    stop_container,
    wait_for_health,
)

CONTAINER_NAME = 'ner-pipeline-pdf-smoketest'
API_KEY = 'docker-test-pdf-smoketest-key'
DEFAULT_INPUT = Path('docs') / 'Решение (безбумажное).pdf'


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

    start_container(CONTAINER_NAME, API_KEY, warn_missing_env=False)
    try:
        wait_for_health()
        headers_file = output_pdf.with_suffix('.headers.txt')
        curl_cmd = [
            'curl',
            '-s',
            '-w',
            '\nHTTP:%{http_code}\n',
            '-X',
            'POST',
            f'http://localhost:{PORT}/v1/anonymize/pdf',
            '-H',
            f'X-API-Key: {API_KEY}',
            '-F',
            f'file=@{input_pdf};type=application/pdf',
            '-D',
            str(headers_file),
            '-o',
            str(output_pdf),
        ]
        result = subprocess.run(
            curl_cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    finally:
        stop_container(CONTAINER_NAME)

    print(result.stdout.strip())
    if headers_file.exists():
        for line in headers_file.read_text(encoding='utf-8').splitlines():
            if line.lower().startswith('x-'):
                print(line)
        headers_file.unlink()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
