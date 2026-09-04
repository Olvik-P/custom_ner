"""Manual smoke test: runs test.py's pipeline demo inside the Docker
container instead of the local environment.

Bind-mounts the repo into the privacyguard-pipeline image and runs
test.py there, so the same step-by-step detect -> mask -> LLM ->
demask demo exercises the containerized Python/Tesseract/Natasha
stack. Source always comes from the mount (not whatever was baked in
at `docker build` time), so this reflects the current working tree.

Requires the image to exist first:
    docker build -t privacyguard-pipeline .

Usage:
    python docker_test.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

IMAGE = 'privacyguard-pipeline'
REPO_ROOT = Path(__file__).resolve().parent
ENV_FILE = REPO_ROOT / 'privacyguard_pipeline' / '.env'


def main() -> int:
    if not ENV_FILE.exists():
        print(
            f'Warning: {ENV_FILE} not found - the LLM call step will '
            'run without an API key (see .env.example).',
            file=sys.stderr,
        )

    cmd = [
        'docker', 'run', '--rm',
        '-v', f'{REPO_ROOT.as_posix()}:/workspace',
        '-w', '/workspace',
    ]
    if ENV_FILE.exists():
        cmd += ['--env-file', ENV_FILE.as_posix()]
    cmd += [IMAGE, 'python', 'test.py']

    return subprocess.run(cmd).returncode


if __name__ == '__main__':
    sys.exit(main())
