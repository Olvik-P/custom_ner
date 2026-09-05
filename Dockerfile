# Рантайм-образ PrivacyGuard Pipeline.
#
# По умолчанию слушает HTTP API (см. раздел "HTTP API" в README.md) —
# чтобы другие проекты, на любом языке/стеке, обращались к пайплайну
# по сети, а не встраивали его как Python-зависимость. CLI остаётся
# доступным в том же образе через переопределение CMD:
#
#   docker build -t ner-pipeline .
#   docker run --env-file privacyguard_pipeline/.env -p 8420:8420 \
#       ner-pipeline
#   docker run --env-file privacyguard_pipeline/.env ner-pipeline \
#       python main.py "Text with PII"
#   docker run -it ner-pipeline bash
#
# или использовать как базовый образ (`FROM ner-pipeline`) для других
# проектов, которым нужен этот пайплайн.

FROM python:3.12-slim

# Системная зависимость: Tesseract OCR + русская обученная модель,
# нужны OCR-фолбэку PDF-анонимайзера для сканированных страниц
# (privacyguard_pipeline/pdf/ocr.py). Ставится через apt, чтобы
# pytesseract нашёл и сам бинарник, и языковые данные "rus" по
# стандартному системному пути - никакой настройки PATH/TESSDATA_PREFIX
# не требуется, в отличие от ручной установки на Windows, которую этот
# образ заменяет.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-rus \
    && rm -rf /var/lib/apt/lists/*

# Непривилегированный пользователь для рантайма.
RUN groupadd --gid 1000 privacyguard \
    && useradd --uid 1000 --gid privacyguard --create-home --shell /bin/bash privacyguard

WORKDIR /app

# pyproject.toml пакета лежит внутри самого privacyguard_pipeline/
# (packages.find с `where = [".."]` резолвится относительно корня
# репозитория оттуда), поэтому исходники пакета должны уже присутствовать
# к моменту сборки/установки - классический слоёный вариант "скопировать
# манифест, поставить зависимости, потом скопировать исходники" здесь
# невозможен. Это всё ещё позволяет держать дорогой apt-слой выше
# закэшированным независимо от изменений в Python-исходниках.
COPY privacyguard_pipeline/ ./privacyguard_pipeline/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "./privacyguard_pipeline[pdf,api]" \
    && rm -rf /app/privacyguard_pipeline/build /app/*.egg-info

RUN mkdir -p /app/privacyguard_pipeline/logs \
    && chown -R privacyguard:privacyguard /app

USER privacyguard
WORKDIR /app/privacyguard_pipeline

EXPOSE 8420

# По умолчанию: запускается HTTP API-сервер (эндпоинты и обязательный
# заголовок X-API-Key - см. раздел "HTTP API" в README.md). Чтобы вместо
# этого запустить CLI, переопределите команду, например:
#   docker run <image> python main.py "text with PII"
CMD ["python", "-m", "privacyguard_pipeline.api"]
