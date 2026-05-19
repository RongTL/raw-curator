# syntax=docker/dockerfile:1.7
FROM docker.io/nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=600 \
    PIP_RETRIES=10 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_REQUESTS_TIMEOUT=600 \
    POETRY_HTTP_BASIC_RETRIES=10

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev python3-pip \
      build-essential pkg-config git curl ca-certificates \
      libraw-dev libjpeg-turbo8-dev libtiff-dev libpng-dev \
      libopenexr-dev libheif-dev libwebp-dev \
      libgl1 libglib2.0-0 \
      exiftool darktable \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

RUN pip install --break-system-packages --no-cache-dir poetry==1.8.5

WORKDIR /app

COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root --without dev

COPY app/ ./app/
COPY db/ ./db/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY alembic.ini README.md ./
RUN poetry install --only-root

ENV RAWCURATOR_CACHE=/data/cache \
    RAWCURATOR_MODELS=/data/models \
    RAWCURATOR_PHOTOS=/data/photos \
    RAWCURATOR_XMP=/data/xmp \
    HF_HOME=/data/models/hf \
    TORCH_HOME=/data/models/torch

CMD ["raw-curator", "--help"]
