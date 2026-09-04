"""Manual smoke test: exercises the HTTP API's POST /v1/anonymize
inside the Docker container instead of the local environment.

Bind-mounts the repo into the privacyguard-pipeline image and starts
the API server there (from /workspace, so it's the current on-disk
code, not whatever was baked in at `docker build` time), then calls
/health and /v1/anonymize over HTTP like any real client would.

Requires the image to exist first:
    docker build -t privacyguard-pipeline .

Usage:
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

IMAGE = 'privacyguard-pipeline'
CONTAINER_NAME = 'privacyguard-api-smoketest'
PORT = 8420
API_KEY = secrets.token_hex(16)
SAMPLE_TEXT = 'Пациент Иванов Пётр Сергеевич, тел. +7(916)123-45-67'

REPO_ROOT = Path(__file__).resolve().parent
ENV_FILE = REPO_ROOT / 'privacyguard_pipeline' / '.env'


def _wait_for_health(timeout: float = 30.0) -> None:
    """Poll /health until the server responds or timeout elapses."""
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
    # -e overrides --env-file, so this fixed test key always wins over
    # whatever API_KEY/API_KEY_REQUIRED (if any) is in .env.
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
