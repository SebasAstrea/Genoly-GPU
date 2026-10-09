"""
Carga robusta de datos para genética cuantitativa (CSV/Excel sucios).

- POST /api/qdata/preview -> perfil del archivo subido (delimitador,
  cabecera, columnas con tipo y datos perdidos, filas de ejemplo).
- GET  /api/qdata/{id}/grid-full   -> matriz completa como JSON (para la
  interfaz interactiva estilo hoja de cálculo).
- POST /api/qdata/apply-filters    -> envía el resultado del filtrado
  interactivo; se aplica clean_grid y se guarda con un ``clean_id``.
- POST /api/qdata/clean            -> limpieza automática (parámetros fijos) en proceso aislado.
- GET  /api/qdata/{id}             -> matriz limpia {phenotypes, genotypes,
  markers, report} para enviar a los modelos.
"""

import json
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ui.backend import jobs
from ui.backend.uploads import UPLOAD_DIR, create_upload, get_upload, upload_path

router = APIRouter(prefix="/api/qdata", tags=["qdata"])


class QDataPreviewRequest(BaseModel):
    upload_id: str
    max_rows: int = 2000


class GridFullRequest(BaseModel):
    """Pide la grid completa de un archivo ya subido."""
    upload_id: str
    max_rows: Optional[int] = None


class ApplyFiltersRequest(BaseModel):
    """Envía una grid filtrada/limpiada por el cliente + opciones de clean_grid."""
    rows: List[List[Optional[str]]]  # filas que quedaron tras filtro del frontend
    phenotype_col: int = 0
    impute_method: str = "media"
    max_column_missingness: float = 1.0
    min_individuals: int = 5
    min_markers: int = 2
    min_maf: float = 0.0
    hwe_p: Optional[float] = None


class QDataCleanRequest(BaseModel):
    upload_id: str
    phenotype_col: int = 0
    impute_method: str = "media"
    max_column_missingness: float = 1.0
    min_individuals: int = 5
    min_markers: int = 2
    min_maf: float = 0.0
    hwe_p: Optional[float] = None


class QDataJobResponse(BaseModel):
    job_id: str
    events_url: str
    clean_id: str


class AnalysisRequest(BaseModel):
    genotypes: List[List[float]]
    phenotypes: Optional[List[float]] = None
    kinship: str = "vanraden"
    n_folds: int = 5
    n_repeats: int = 1
    seed: int = 0
    min_maf: float = 0.01
    top_relatives: int = 10


class QAnalysisJobResponse(BaseModel):
    job_id: str
    events_url: str


@router.post("/preview")
def preview(req: QDataPreviewRequest) -> dict:
    """Perfil del archivo subido para decidir la limpieza."""
    from Genoly.quantitative.preprocess import load_grid, profile_grid
    path = upload_path(req.upload_id)
    try:
        grid = load_grid(path, max_rows=req.max_rows)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Archivo ilegible: {exc}")

    profile = profile_grid(grid)
    sample = [row[:min(len(row), 12)] for row in grid[:15]]
    return {
        "rows_preview": len(grid),
        "sample_rows": sample,
        "columns": profile["columns"],
        "column_names": profile["column_names"],
        "header_detected": profile["header_detected"],
    }


@router.post("/{upload_id}/grid-full")
def grid_full(upload_id: str, req: GridFullRequest) -> dict:
    """Devuelve la grid completa para filtrado interactivo en el navegador."""
    from Genoly.quantitative.preprocess import load_grid, profile_grid

    path = upload_path(upload_id)
    try:
        grid = load_grid(path, max_rows=req.max_rows)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Archivo ilegible: {exc}")

    profile = profile_grid(grid)
    return {
        "upload_id": upload_id,
        "rows_total": len(grid),
        "columns": profile["columns"],
        "column_names": profile["column_names"],
        "header_detected": profile["header_detected"],
        "grid": grid,  # <-- TODAS las filas como matriz de strings/nulls
    }


@router.post("/apply-filters", response_model=QDataJobResponse)
async def apply_filters(req: ApplyFiltersRequest) -> QDataJobResponse:
    """Filtra/limpia la grid enviada por el frontend con clean_grid."""
    from Genoly.quantitative.preprocess import clean_grid

    if req.impute_method not in ("media", "moda"):
        raise HTTPException(status_code=422,
                            detail="impute_method debe ser 'media' o 'moda'")

    clean_id = uuid4().hex
    clean_path = UPLOAD_DIR / f"{clean_id}.json"
    create_upload(clean_id, "qdata", Path(clean_path), "limpio.json", 0)

    job = jobs.manager.create("apply_filters")
    jobs.manager.submit_process(job, {
        "kind": "apply_filters",
        "grid_rows": req.rows,
        "out_path": str(clean_path),
        "out_upload_id": clean_id,
        "phenotype_col": req.phenotype_col,
        "impute_method": req.impute_method,
        "max_column_missingness": req.max_column_missingness,
        "min_individuals": req.min_individuals,
        "min_markers": req.min_markers,
        "min_maf": req.min_maf,
        "hwe_p": req.hwe_p,
    })
    return QDataJobResponse(job_id=job.id,
                            events_url=f"/api/jobs/{job.id}/events",
                            clean_id=clean_id)


@router.post("/clean", response_model=QDataJobResponse)
async def clean(req: QDataCleanRequest) -> QDataJobResponse:
    """Limpia, imputa y filtra la matriz en un proceso aislado."""
    path = upload_path(req.upload_id)
    if req.impute_method not in ("media", "moda"):
        raise HTTPException(status_code=422,
                            detail="impute_method debe ser 'media' o 'moda'")

    clean_id = uuid4().hex
    clean_path = UPLOAD_DIR / f"{clean_id}.json"
    create_upload(clean_id, "qdata", Path(clean_path), "limpio.json", 0)

    job = jobs.manager.create("qdata_clean")
    jobs.manager.submit_process(job, {
        "kind": "qdata_clean",
        "path": str(path),
        "out_path": str(clean_path),
        "out_upload_id": clean_id,
        "phenotype_col": req.phenotype_col,
        "impute_method": req.impute_method,
        "max_column_missingness": req.max_column_missingness,
        "min_individuals": req.min_individuals,
        "min_markers": req.min_markers,
        "min_maf": req.min_maf,
        "hwe_p": req.hwe_p,
    })
    return QDataJobResponse(job_id=job.id,
                            events_url=f"/api/jobs/{job.id}/events",
                            clean_id=clean_id)


@router.get("/{clean_id}")
def get_clean(clean_id: str) -> dict:
    """Matriz limpia (fenotipos, genotipos, marcadores, reporte)."""
    meta = get_upload(clean_id)
    if meta is None or meta.get("kind") != "qdata":
        raise HTTPException(status_code=404, detail="Datos limpios no encontrados")
    path = Path(meta["path"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="El archivo ya no existe")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise HTTPException(status_code=500,
                            detail="No se pudo leer la matriz limpia")
    data["clean_id"] = clean_id
    data["filename"] = meta.get("filename")
    return data


def _launch(kind: str, req) -> QAnalysisJobResponse:
    job = jobs.manager.create(kind)
    jobs.manager.submit_process(
        job, {"kind": kind, **req.model_dump(exclude_none=True)})
    return QAnalysisJobResponse(job_id=job.id,
                                events_url=f"/api/jobs/{job.id}/events")


@router.post("/crossval", response_model=QAnalysisJobResponse)
async def crossval(req: AnalysisRequest) -> QAnalysisJobResponse:
    """Validación cruzada K-fold de la predicción genómica."""
    if req.phenotypes is None:
        raise HTTPException(status_code=422, detail="Se necesitan fenotipos")
    if len(req.phenotypes) != len(req.genotypes):
        raise HTTPException(status_code=422,
                            detail="fenotipos y genotipos deben tener igual nº de filas")
    return _launch("crossval", req)


@router.post("/gwas", response_model=QAnalysisJobResponse)
async def gwas(req: AnalysisRequest) -> QAnalysisJobResponse:
    """GWAS de marcador único (EMMAX-like) con Manhattan plot."""
    if req.phenotypes is None:
        raise HTTPException(status_code=422, detail="Se necesitan fenotipos")
    return _launch("gwas", req)


@router.post("/kinship", response_model=QAnalysisJobResponse)
async def kinship(req: AnalysisRequest) -> QAnalysisJobResponse:
    """Matriz de parentesco: heatmap, PCA de población y pares relacionados."""
    return _launch("kinship", req)