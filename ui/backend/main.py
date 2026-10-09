"""
Genoly-GPU UI: API REST del backend.

Sirve los módulos de Genoly (GPU/CUDA) a través de FastAPI y, en
producción, el frontend React compilado (ui/frontend/dist).

Arranque (desde cualquier directorio, incluido ui/backend):
    uvicorn main:app --host 0.0.0.0 --port 8000
o desde la raíz del proyecto:
    python -m uvicorn ui.backend.main:app --host 0.0.0.0 --port 8000

Nota: usar --workers 1 (por defecto). El estado de los trabajos de fondo
y sus streams SSE vive en memoria del proceso, y los trabajos GPU ya se
serializan internamente; varios procesos competirían por la misma VRAM.
"""

import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

# Bootstrap de imports: añade la raíz del proyecto a sys.path para que
# `Genoly` y `ui.backend.*` resuelvan aunque uvicorn se ejecute desde
# ui/backend (o desde otro directorio de trabajo).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from Genoly import __version__
from Genoly.core.gpu_setup import GpuSetup
from ui.backend import security
from ui.backend.routers import (
    annotation,
    dataset,
    device,
    downstream,
    gblup,
    jobs,
    kmer,
    pblup,
    qc,
    qdata,
    quantitative,
    report,
    upload,
    variants,
)
from ui.backend.routers import (
    map as map_router,
)

logger = logging.getLogger("genoly.ui")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

#: Intervalo (segundos) de la limpieza periódica de derrames huérfanos.
CLEANUP_INTERVAL = int(os.environ.get("GENOLY_CLEANUP_INTERVAL", "600"))


def _spill_cleaner_loop() -> None:
    """Elimina periódicamente los directorios de derrame de trabajadores
    GPU muertos (OOM/segfault/kill). Nunca toca un derrame en uso."""
    from Genoly.kmer.kmers import cleanup_orphan_spills
    while True:
        time.sleep(CLEANUP_INTERVAL)
        try:
            removed = cleanup_orphan_spills()
            if removed:
                logger.info("Limpieza de derrames: %d directorios huérfanos "
                            "eliminados", removed)
        except Exception as exc:  # nunca deja caer el servidor
            logger.warning("Error en la limpieza de derrames: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    security.warn_if_open()
    threading.Thread(target=_spill_cleaner_loop, daemon=True).start()
    threading.Thread(target=_upload_cleaner_loop, daemon=True).start()
    yield


def _upload_cleaner_loop() -> None:
    """Aplica el TTL de subidas, datasets y extracciones (si está activo)."""
    from ui.backend.maintenance import UPLOAD_TTL_SECONDS, run_cleanup
    while True:
        time.sleep(CLEANUP_INTERVAL)
        if not UPLOAD_TTL_SECONDS:
            continue
        try:
            removed = run_cleanup()
            if removed:
                logger.info("Limpieza por TTL: %d elementos eliminados",
                            removed)
        except Exception as exc:  # nunca deja caer el servidor
            logger.warning("Error en la limpieza por TTL: %s", exc)


app = FastAPI(
    title="Genoly-GPU API",
    description="API REST de análisis genómico acelerado por GPU (NVIDIA/CUDA)",
    version=__version__,
    lifespan=lifespan,
)

# CORS: solo los orígenes declarados (por defecto el dev server de Vite).
# En producción el frontend se sirve desde este mismo FastAPI, así que no
# hace falta ningún origen: lista vacía = sin CORS.
_origins = security.cors_origins()
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key", "Authorization"],
    )


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """Exige la API key en /api/* si GENOLY_API_KEY está definida."""
    path = request.url.path
    if path.startswith("/api/") and not security.is_authorized(
            request.headers, request.query_params, path):
        return JSONResponse(
            status_code=401,
            content={"detail": "Falta o no coincide la clave de API "
                               "(cabecera X-API-Key o ?api_key=)"},
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await call_next(request)

app.include_router(device.router)
app.include_router(qc.router)
app.include_router(kmer.router)
app.include_router(variants.router)
app.include_router(quantitative.router)
app.include_router(gblup.router)
app.include_router(pblup.router)
app.include_router(upload.router)
app.include_router(jobs.router)
app.include_router(dataset.router)
app.include_router(map_router.router)
app.include_router(downstream.router)
app.include_router(annotation.router)
app.include_router(report.router)
app.include_router(qdata.router)


@app.get("/api/health")
def health() -> dict:
    """Comprobación de salud del servicio."""
    return {"status": "ok", "version": __version__}


@app.get("/api/setup")
def setup_status() -> dict:
    """Estado de la GPU y build de PyTorch (compatibilidad CUDA)."""
    info = GpuSetup.detect_nvidia()
    status = GpuSetup.torch_status()
    tag = GpuSetup.recommend_cuda_tag(info.cuda_version) if info.available else None
    return {
        "nvidia": info.__dict__,
        "torch": status,
        "recommended_cuda_tag": tag,
        "install_command": GpuSetup.install_command(tag) if tag else None,
    }


# Montar el frontend compilado si existe (producción)
FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ui.backend.main:app", host="0.0.0.0", port=8000, reload=True)