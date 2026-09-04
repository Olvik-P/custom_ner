"""Manual smoke test: runs test_pdf.py's PDF anonymization demo inside
the Docker container instead of the local environment.

Bind-mounts the repo into the privacyguard-pipeline image and runs
test_pdf.py there - useful in particular for the OCR fallback path,
since the container has Tesseract + the "rus" language pack baked in
and needs no system install/PATH setup, unlike running test_pdf.py
directly on Windows.

Requires the image to exist first:
    docker build -t privacyguard-pipeline .

Usage:
    python docker_test_pdf.py [input.pdf] [output.pdf]

Arguments are forwarded to test_pdf.py as-is (repo-relative paths,
since the whole repo is mounted at /workspace inside the container).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

IMAGE = 'privacyguard-pipeline'
REPO_ROOT = Path(__file__).resolve().parent


def main(argv: list[str]) -> int:
    cmd = [
        'docker', 'run', '--rm',
        '-v', f'{REPO_ROOT.as_posix()}:/workspace',
        '-w', '/workspace',
        IMAGE, 'python', 'test_pdf.py',
        *argv,
    ]
    return subprocess.run(cmd).returncode


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
