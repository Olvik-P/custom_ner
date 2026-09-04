# PrivacyGuard Pipeline runtime image.
#
# Not a service - a portable environment for the CLI / programmatic
# pipeline and the PDF anonymizer, with Tesseract (+ Russian language
# data) and all Python dependencies baked in. Run directly:
#
#   docker build -t privacyguard-pipeline .
#   docker run --env-file privacyguard_pipeline/.env privacyguard-pipeline \
#       python main.py "Text with PII"
#   docker run -it privacyguard-pipeline bash
#
# or use as a base image (`FROM privacyguard-pipeline`) for other
# projects that need this pipeline available.

FROM python:3.12-slim

# System dependency: Tesseract OCR + Russian trained data, required by
# the PDF anonymizer's OCR fallback for scanned pages
# (privacyguard_pipeline/pdf/ocr.py). Installed via apt so pytesseract
# finds it and the "rus" language data at the standard system path -
# no PATH/TESSDATA_PREFIX configuration needed, unlike the manual
# Windows setup this image replaces.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-rus \
    && rm -rf /var/lib/apt/lists/*

# Non-root runtime user.
RUN groupadd --gid 1000 privacyguard \
    && useradd --uid 1000 --gid privacyguard --create-home --shell /bin/bash privacyguard

WORKDIR /app

# The package's pyproject.toml lives inside privacyguard_pipeline/
# itself (packages.find's `where = [".."]` resolves against the repo
# root from there), so its own source must already be present to
# build/install it - a separate "copy manifest, install deps, then
# copy source" split isn't possible here. This still keeps the
# expensive apt layer above cached independently of Python source
# changes.
COPY privacyguard_pipeline/ ./privacyguard_pipeline/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "./privacyguard_pipeline[pdf]" \
    && rm -rf /app/privacyguard_pipeline/build /app/*.egg-info

RUN mkdir -p /app/privacyguard_pipeline/logs \
    && chown -R privacyguard:privacyguard /app

USER privacyguard
WORKDIR /app/privacyguard_pipeline

# No service/entrypoint - default to printing CLI usage. Override the
# command to actually run something, e.g.:
#   docker run <image> python main.py "text with PII"
CMD ["python", "main.py", "--help"]
