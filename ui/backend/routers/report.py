"""
Visualización y reporte: heatmap, volcano y resumen descargable.

- POST /api/report -> genera figuras SVG y un reporte Markdown/JSON en un
  proceso aislado; el Markdown queda como subida descargable.
"""

from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ui.backend import jobs
from ui.backend.uploads import UPLOAD_DIR, create_upload

router = APIRouter(prefix="/api/report", tags=["report"])


class ReportRequest(BaseModel):
    title: str = "Reporte de análisis"
    matrix: Optional[List[List[float]]] = None
    row_labels: Optional[List[str]] = None
    col_labels: Optional[List[str]] = None
    fold_changes: Optional[List[float]] = None
    p_values: Optional[List[float]] = None
    fc_threshold: float = 1.0
    p_threshold: float = 0.05
    sections: Optional[Dict[str, dict]] = None


class ReportJobResponse(BaseModel):
    job_id: str
    events_url: str


@router.post("", response_model=ReportJobResponse)
async def build(req: ReportRequest) -> ReportJobResponse:
    """Genera heatmap/volcano y el reporte, en proceso aislado."""
    if not req.matrix and not (req.fold_changes and req.p_values) \
            and not req.sections:
        raise HTTPException(
            status_code=422,
            detail="Proporciona una matriz, datos de volcano o secciones")

    md_id = uuid4().hex
    md_path = UPLOAD_DIR / f"{md_id}.md"
    create_upload(md_id, "report", Path(md_path), "report.md", 0)

    job = jobs.manager.create("report")
    jobs.manager.submit_process(job, {
        "kind": "report",
        "title": req.title,
        "matrix": req.matrix,
        "row_labels": req.row_labels,
        "col_labels": req.col_labels,
        "fold_changes": req.fold_changes,
        "p_values": req.p_values,
        "fc_threshold": req.fc_threshold,
        "p_threshold": req.p_threshold,
        "sections": req.sections or {},
        "md_path": str(md_path),
        "md_upload_id": md_id,
    })
    return ReportJobResponse(job_id=job.id,
                             events_url=f"/api/jobs/{job.id}/events")