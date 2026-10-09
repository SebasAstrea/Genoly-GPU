"""
Anotación funcional: mapeo de variantes a genes/regiones y términos GO.

- POST /api/annotation -> anota una lista de variantes con un GFF/GTF
  (proceso aislado; progreso y resultado por SSE).
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ui.backend import jobs
from ui.backend.uploads import upload_path

router = APIRouter(prefix="/api/annotation", tags=["annotation"])


class VariantItem(BaseModel):
    sequence: str
    position: int
    ref: Optional[str] = None
    alt: Optional[str] = None
    type: Optional[str] = None
    freq: Optional[float] = None


class AnnotationRequest(BaseModel):
    gff_upload_id: str          # GFF3/GTF (upload o ruta registrada)
    variants: List[VariantItem]  # variantes (p. ej. del VCF)


class AnnotationJobResponse(BaseModel):
    job_id: str
    events_url: str


@router.post("", response_model=AnnotationJobResponse)
async def annotate(req: AnnotationRequest) -> AnnotationJobResponse:
    """Anota las variantes contra la anotación, en proceso aislado."""
    if not req.variants:
        raise HTTPException(status_code=422, detail="No hay variantes que anotar")
    gff_path = upload_path(req.gff_upload_id)

    job = jobs.manager.create("annotation")
    jobs.manager.submit_process(job, {
        "kind": "annotation",
        "gff_path": str(gff_path),
        "variants": [v.model_dump() for v in req.variants],
    })
    return AnnotationJobResponse(job_id=job.id,
                                 events_url=f"/api/jobs/{job.id}/events")