# ============================================================================ #
# Dockerfile para Genoly-GPU
# 
# Construcción condicional:
#   --build-arg GENOLY_GPU_MODE=cpu   → imagen sin CUDA (ligera)
#   --build-arg GENOLY_GPU_MODE=gpu   → imagen con CUDA 12.6 (para GPU NVIDIA)
#
# Detección automática en tiempo de build (via script):
#   docker build --build-arg GENOLY_GPU_MODE=$(python docker/detect_gpu.py) .
# ============================================================================ #

ARG GENOLY_GPU_MODE=cpu

# ============================================================================ #
# STAGE 1: build del frontend React (siempre igual)
# ============================================================================ #
FROM node:22-alpine AS frontend-build

WORKDIR /app/ui/frontend
COPY ui/frontend/package.json ui/frontend/package-lock.json ./
RUN npm ci
COPY ui/frontend/ ./
RUN npm run build

# ============================================================================ #
# STAGE 2: backend (CPU o GPU)
# ============================================================================ #

# --- Rama CPU (por defecto, imagen ligera) ---
FROM python:3.12-slim AS backend-cpu

ARG GENOLY_PORT=8000
ENV GENOLY_PORT=$GENOLY_PORT \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    GENOLY_DEVICE=cpu

WORKDIR /app

# Dependencias base. torch va primero desde el índice CPU: la build que
# sirve PyPI por defecto incluye CUDA (varios GB) y aquí no se usa; con
# esto la imagen CPU se queda en unos cientos de MB.
COPY requirements.txt .
COPY ui/backend/requirements.txt ./ui/backend/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt -r ui/backend/requirements.txt

# Código. El paquete se instala con pip (--no-deps: las dependencias ya
# están en requirements.txt, así no se tira de PyPI otra vez) para que
# `import Genoly` funcione desde cualquier sitio, no solo desde /app.
COPY setup.py pyproject.toml README.md ./
COPY Genoly ./Genoly
RUN pip install --no-cache-dir --no-deps .

# Código de la API y frontend compilado
COPY ui/backend ./ui/backend
COPY --from=frontend-build /app/ui/frontend/dist ./ui/frontend/dist

RUN useradd --create-home --shell /usr/sbin/nologin genoly \
    && chown -R genoly:genoly /app
USER genoly

EXPOSE $GENOLY_PORT

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"GENOLY_PORT\"]}/api/health')" || exit 1

CMD uvicorn ui.backend.main:app --host 0.0.0.0 --port $GENOLY_PORT

# --- Rama GPU (con CUDA 12.6) ---
# Ubuntu 24.04: su repositorio incluye python3.12 (en 22.04 no existe y el build falla)
FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04 AS backend-gpu

ARG GENOLY_PORT=8000
ENV GENOLY_PORT=$GENOLY_PORT \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    GENOLY_DEVICE=cuda \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    CUDA_LAUNCH_BLOCKING=0

WORKDIR /app

# Instalar Python 3.12 y dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3-pip \
    python3.12-venv \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# PyTorch con CUDA 12.6 (build probada con RTX 3050)
COPY requirements.txt .
COPY ui/backend/requirements.txt ./ui/backend/requirements.txt
RUN pip3 install --no-cache-dir 'torch>=2.0.0' 'torchvision>=0.15.0' \
    --index-url https://download.pytorch.org/whl/cu126 && \
    pip3 install --no-cache-dir -r requirements.txt -r ui/backend/requirements.txt

# Código (ver rama CPU: se instala el paquete, sin volver a bajar deps)
COPY setup.py pyproject.toml README.md ./
COPY Genoly ./Genoly
RUN pip3 install --no-cache-dir --no-deps .

# Código de la API y frontend compilado
COPY ui/backend ./ui/backend
COPY --from=frontend-build /app/ui/frontend/dist ./ui/frontend/dist

# Usuario no root
RUN useradd --create-home --shell /usr/sbin/nologin genoly \
    && chown -R genoly:genoly /app
USER genoly

EXPOSE $GENOLY_PORT

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"GENOLY_PORT\"]}/api/health')" || exit 1

# Nota: en modo GPU, si el contenedor no tiene acceso a la GPU (falta nvidia-container-toolkit),
# PyTorch caerá a CPU automáticamente gracias a Genoly.core.device.
CMD uvicorn ui.backend.main:app --host 0.0.0.0 --port $GENOLY_PORT

# ============================================================================ #
# STAGE FINAL: selección condicional de la imagen base
# ============================================================================ #
FROM backend-${GENOLY_GPU_MODE} AS final
