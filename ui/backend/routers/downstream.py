"""
Análisis descendente (PCA, t-SNE, K-Means).

- POST /api/downstream -> encola el análisis como trabajo aislado (proceso
  aparte); el progreso y el resultado llegan por el SSE de ``events_url``.
"""

from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ui.backend import jobs

router = APIRouter(prefix="/api/downstream", tags=["downstream"])


class DownstreamRequest(BaseModel):
    matrix: List[List[float]]      # individuos x rasgos (dosis, expresión...)
    pca_components: int = 2
    run_tsne: bool = True
    tsne_perplexity: float = 30.0
    tsne_iter: int = 500
    k: int = 3
    standardize: bool = True


class DownstreamJobResponse(BaseModel):
    job_id: str
    events_url: str


@router.post("", response_model=DownstreamJobResponse)
async def run_downstream(req: DownstreamRequest) -> DownstreamJobResponse:
    """PCA + t-SNE + K-Means sobre la matriz de rasgos, en proceso aislado."""
    if len(req.matrix) < 3:
        raise HTTPException(status_code=422,
                            detail="Se necesitan al menos 3 individuos")
    if not req.matrix or not req.matrix[0]:
        raise HTTPException(status_code=422, detail="La matriz está vacía")
    width = len(req.matrix[0])
    if any(len(row) != width for row in req.matrix):
        raise HTTPException(status_code=422,
                            detail="Todas las filas deben tener la misma longitud")

    job = jobs.manager.create("downstream")
    jobs.manager.submit_process(job, {
        "kind": "downstream",
        "matrix": req.matrix,
        "pca_components": req.pca_components,
        "run_tsne": req.run_tsne,
        "tsne_perplexity": req.tsne_perplexity,
        "tsne_iter": req.tsne_iter,
        "k": req.k,
        "standardize": req.standardize,
    })
    return DownstreamJobResponse(job_id=job.id,
                                 events_url=f"/api/jobs/{job.id}/events")