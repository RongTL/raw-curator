# syntax=docker/dockerfile:1.7
FROM docker.io/nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DEFAULT_TIMEOUT=600 \
    PIP_RETRIES=10 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_REQUESTS_TIMEOUT=600 \
    POETRY_HTTP_BASIC_RETRIES=10 \
    POETRY_INSTALLER_PARALLEL=true \
    POETRY_INSTALLER_MAX_WORKERS=8 \
    MAKEFLAGS="-j8" \
    MAX_JOBS=8 \
    CMAKE_BUILD_PARALLEL_LEVEL=8

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache && \
    apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev python3-pip \
      build-essential pkg-config git curl ca-certificates \
      libraw-dev libjpeg-turbo8-dev libtiff-dev libpng-dev \
      libopenexr-dev libheif-dev libwebp-dev \
      libgl1 libglib2.0-0 \
      exiftool darktable \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --break-system-packages poetry==1.8.5

WORKDIR /app

COPY pyproject.toml poetry.lock* ./
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/pypoetry \
    poetry install --no-root --without dev

COPY app/ ./app/
COPY db/ ./db/
COPY scripts/ ./scripts/
COPY alembic.ini README.md ./
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/pypoetry \
    poetry install --only-root

ENV RAWCURATOR_CACHE=/data/cache \
    RAWCURATOR_MODELS=/data/models \
    RAWCURATOR_PHOTOS=/data/photos \
    RAWCURATOR_XMP=/data/xmp \
    HF_HOME=/data/models/hf \
    TORCH_HOME=/data/models/torch

CMD ["raw-curator", "--help"]
