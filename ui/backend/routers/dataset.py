"""
Datasets NCBI: registro de un dataset completo (carpeta o .zip) con todos
sus ensamblajes FASTA, para analizarlos en un solo flujo.

- POST /api/dataset/register {path} -> registra el dataset, descubre los
  FASTA y calcula sus estadísticas en segundo plano.
- GET  /api/dataset/{id}       -> estado de cada archivo del dataset.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ui.backend import datasets as ds

router = APIRouter(prefix="/api/dataset", tags=["dataset"])


class DatasetFile(BaseModel):
    filename: str
    size: int
    stats_status: str = "pending"
    records: Optional[int] = None
    total_bases: Optional[int] = None
    first: Optional[dict] = None
    error: Optional[str] = None


class DatasetResponse(BaseModel):
    dataset_id: str
    kind: str
    status: str = "pending"
    file_count: int
    files: List[DatasetFile]
    error: Optional[str] = None


class DatasetRegisterRequest(BaseModel):
    path: str


def _public(meta: dict) -> DatasetResponse:
    return DatasetResponse(
        dataset_id=meta["dataset_id"],
        kind=meta.get("kind", "dir"),
        status=meta.get("status", "pending"),
        file_count=len(meta["files"]),
        files=[
            DatasetFile(
                filename=f["filename"],
                size=f["size"],
                stats_status=f["stats_status"],
                records=f.get("records"),
                total_bases=f.get("total_bases"),
                first=f.get("first"),
                error=f.get("error"),
            )
            for f in meta["files"]
        ],
        error=meta.get("error"),
    )


@router.post("/register", response_model=DatasetResponse)
def register_dataset(req: DatasetRegisterRequest) -> DatasetResponse:
    """Registra un dataset NCBI (carpeta o .zip) y descubre sus FASTA."""
    return _public(ds.register_dataset(req.path))


@router.get("/{dataset_id}", response_model=DatasetResponse)
def dataset_status(dataset_id: str) -> DatasetResponse:
    """Estado del dataset: archivos FASTA descubiertos y sus estadísticas."""
    meta = ds.get_dataset(dataset_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Dataset no encontrado")
    return _public(meta)