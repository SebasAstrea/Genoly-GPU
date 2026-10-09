"""
Carga robusta de datos para genética cuantitativa (CSV/Excel sucios).

- POST /api/qdata/preview         -> perfil del archivo subido (delimitador,
  cabecera, columnas con tipo y datos perdidos, filas de ejemplo).
- POST /api/qdata/preview-clean   -> vista previa de la limpieza (dry-run):
  impacto por regla, errores bloqueantes y señales de calidad, sin persistir.
- POST /api/qdata/{id}/grid-page  -> página de filas de la grid para la
  tabla interactiva (filtros por definición, paginación en servidor).
- POST /api/qdata/{id}/column-values -> valores únicos de una columna
  (para los popovers de filtro).
- POST /api/qdata/apply-filters    -> filtra/limpia en servidor a partir de
  definiciones de filtro y ediciones; se aplica clean_grid y se guarda.
- POST /api/qdata/clean            -> limpieza automática en proceso aislado.
- GET  /api/qdata/{id}             -> matriz limpia {phenotypes, genotypes,
  markers, pedigree, report} para enviar a los modelos.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ui.backend import jobs
from ui.backend.uploads import UPLOAD_DIR, create_upload, get_upload, upload_path

router = APIRouter(prefix="/api/qdata", tags=["qdata"])

# límite de valores únicos devueltos por columna en los popovers de filtro
_MAX_COLUMN_VALUES = 5000


class CleanOptions(BaseModel):
    """Opciones comunes de ``clean_grid`` (limpieza y filtrado)."""
    phenotype_col: int = 0
    impute_method: str = "media"
    max_column_missingness: float = 1.0
    min_individuals: int = 5
    min_markers: int = 2
    min_maf: float = 0.0
    hwe_p: Optional[float] = None
    id_col: Optional[int] = None
    sire_col: Optional[int] = None
    dam_col: Optional[int] = None


class QDataPreviewRequest(BaseModel):
    upload_id: str
    max_rows: int = 2000


class ColumnFilter(BaseModel):
    """Filtro por definición: valores permitidos + si se conservan vacíos."""
    vals: List[str] = Field(default_factory=list)
    vacios: bool = True


class GridPageRequest(BaseModel):
    """Página de la grid para la tabla interactiva (paginación en servidor)."""
    upload_id: str
    offset: int = 0
    limit: int = 500
    filters: Dict[int, ColumnFilter] = Field(default_factory=dict)


class ColumnValuesRequest(BaseModel):
    """Valores únicos de una columna, opcionalmente tras aplicar filtros."""
    upload_id: str
    col: int
    filters: Dict[int, ColumnFilter] = Field(default_factory=dict)


class CellEdit(BaseModel):
    row: int  # índice global de fila de datos (sin cabecera)
    col: int
    value: str


class ApplyFiltersRequest(CleanOptions):
    """Filtrado/ediciones en servidor + opciones de clean_grid."""
    upload_id: str
    filters: Dict[int, ColumnFilter] = Field(default_factory=dict)
    edits: List[CellEdit] = Field(default_factory=list)
    header: bool = True


class QDataPreviewCleanRequest(CleanOptions):
    """Dry-run de la limpieza: mismos parámetros que /clean."""
    upload_id: str
    max_rows: Optional[int] = None


class QDataCleanRequest(CleanOptions):
    upload_id: str


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


def _filters_as_dict(filters: Dict[int, ColumnFilter]) -> dict:
    """Serializa los filtros a la forma {col: {vals, vacios}} del worker."""
    return {str(col): {"vals": list(f.vals), "vacios": bool(f.vacios)}
            for col, f in filters.items()}


def _cell_missing(value) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in {"", "na", "nan", "null", "none", "-", "nd"}


def _passes_filters(row, filters: dict) -> bool:
    """Semántica del filtrado por definición (equivalente a FiltableGrid)."""
    for key, f in filters.items():
        col = int(key)
        raw = row[col] if col < len(row) else None
        if _cell_missing(raw):
            if not f.get("vacios", True):
                return False
        else:
            allowed = f.get("vals")
            if allowed is not None and str(raw).strip() not in set(allowed):
                return False
    return True


def _load_grid_or_400(upload_id: str, max_rows: Optional[int] = None):
    from Genoly.quantitative.preprocess import load_grid
    path = upload_path(upload_id)
    try:
        return load_grid(path, max_rows=max_rows)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Archivo ilegible: {exc}")


@router.post("/preview")
def preview(req: QDataPreviewRequest) -> dict:
    """Perfil del archivo subido para decidir la limpieza."""
    from Genoly.quantitative.preprocess import profile_grid
    grid = _load_grid_or_400(req.upload_id, req.max_rows)

    profile = profile_grid(grid)
    sample = [row[:min(len(row), 12)] for row in grid[:15]]
    return {
        "rows_preview": len(grid),
        "sample_rows": sample,
        "columns": profile["columns"],
        "column_names": profile["column_names"],
        "header_detected": profile["header_detected"],
    }


@router.post("/preview-clean")
def preview_clean(req: QDataPreviewCleanRequest) -> dict:
    """
    Dry-run de la limpieza: mismos parámetros que /clean pero sin persistir.

    Devuelve el impacto por regla (filas/marcadores proyectados), los
    errores bloqueantes y las señales de calidad (``qc_flags``).
    """
    from Genoly.quantitative.preprocess import clean_grid
    grid = _load_grid_or_400(req.upload_id, req.max_rows)
    try:
        phenotypes, genotypes, markers, pedigree, report = clean_grid(
            grid,
            phenotype_col=req.phenotype_col,
            impute_method=req.impute_method,
            max_column_missingness=req.max_column_missingness,
            min_individuals=req.min_individuals,
            min_markers=req.min_markers,
            min_maf=req.min_maf,
            hwe_p=req.hwe_p,
            id_col=req.id_col,
            sire_col=req.sire_col,
            dam_col=req.dam_col,
            dry_run=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {
        "report": report,
        "blocking_errors": report.get("blocking_errors", []),
        "qc_flags": report.get("qc_flags", []),
        "projected": {
            "rows": len(phenotypes),
            "markers": len(markers),
            "pedigree": pedigree is not None,
        },
    }


@router.post("/{upload_id}/grid-page")
def grid_page(upload_id: str, req: GridPageRequest) -> dict:
    """Página de la grid (sólo filas de datos) tras los filtros activos."""
    from Genoly.quantitative.preprocess import profile_grid

    grid = _load_grid_or_400(upload_id)
    profile = profile_grid(grid)
    body = grid[1:] if profile["header_detected"] else grid
    filters = {str(k): f.model_dump() for k, f in req.filters.items()}
    if filters:
        kept = [(i, row) for i, row in enumerate(body)
                if _passes_filters(row, filters)]
    else:
        kept = list(enumerate(body))
    total = len(kept)
    offset = max(0, req.offset)
    page = kept[offset:offset + max(1, min(req.limit, 5000))]
    return {
        "upload_id": upload_id,
        "total_matching": total,
        "rows_total": len(body),
        "offset": offset,
        "rows": [{"index": i, "cells": row} for i, row in page],
        "column_names": profile["column_names"],
        "column_types": [c["type"] for c in profile["columns"]],
        "header_detected": profile["header_detected"],
    }


@router.post("/{upload_id}/column-values")
def column_values(upload_id: str, req: ColumnValuesRequest) -> dict:
    """Valores únicos de una columna (popovers de filtro de la tabla)."""
    from Genoly.quantitative.preprocess import profile_grid

    grid = _load_grid_or_400(upload_id)
    profile = profile_grid(grid)
    if not 0 <= req.col < len(profile["column_names"]):
        raise HTTPException(status_code=422, detail="Columna inválida")
    body = grid[1:] if profile["header_detected"] else grid
    filters = {str(k): f.model_dump() for k, f in req.filters.items()}
    values = set()
    has_missing = False
    for row in body:
        if filters and not _passes_filters(row, filters):
            continue
        raw = row[req.col] if req.col < len(row) else None
        if _cell_missing(raw):
            has_missing = True
        else:
            values.add(str(raw).strip())
    ordered = sorted(values)
    truncated = len(ordered) > _MAX_COLUMN_VALUES
    return {
        "values": ordered[:_MAX_COLUMN_VALUES],
        "truncated": truncated,
        "has_missing": has_missing,
        "n_values": len(ordered),
    }


@router.post("/apply-filters", response_model=QDataJobResponse)
async def apply_filters(req: ApplyFiltersRequest) -> QDataJobResponse:
    """Filtra/limpia la grid en servidor a partir de filtros y ediciones."""
    if req.impute_method not in ("media", "moda"):
        raise HTTPException(status_code=422,
                            detail="impute_method debe ser 'media' o 'moda'")

    path = upload_path(req.upload_id)
    clean_id = uuid4().hex
    clean_path = UPLOAD_DIR / f"{clean_id}.json"
    create_upload(clean_id, "qdata", Path(clean_path), "limpio.json", 0)

    job = jobs.manager.create("apply_filters")
    jobs.manager.submit_process(job, {
        "kind": "apply_filters",
        "path": str(path),
        "out_path": str(clean_path),
        "out_upload_id": clean_id,
        "filters": _filters_as_dict(req.filters),
        "edits": [e.model_dump() for e in req.edits],
        "header": req.header,
        "phenotype_col": req.phenotype_col,
        "impute_method": req.impute_method,
        "max_column_missingness": req.max_column_missingness,
        "min_individuals": req.min_individuals,
        "min_markers": req.min_markers,
        "min_maf": req.min_maf,
        "hwe_p": req.hwe_p,
        "id_col": req.id_col,
        "sire_col": req.sire_col,
        "dam_col": req.dam_col,
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
        "id_col": req.id_col,
        "sire_col": req.sire_col,
        "dam_col": req.dam_col,
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