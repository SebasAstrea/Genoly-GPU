from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter
from pydantic import BaseModel

from Genoly.variants.caller import Read, VariantCaller
from ui.backend import jobs
from ui.backend.uploads import UPLOAD_DIR, create_upload, upload_path

router = APIRouter(prefix="/api/variants", tags=["variants"])


class ReadModel(BaseModel):
    sequence: str
    start: int
    strand: str = "+"


class VariantRequest(BaseModel):
    reference: str
    reads: List[ReadModel]
    qualities: Optional[List[str]] = None
    min_depth: int = 10
    min_alt_freq: float = 0.2


class VariantResponse(BaseModel):
    variants: List[dict]
    total_variants: int
    snvs: int
    deletions: int
    mean_depth: float


@router.post("/call", response_model=VariantResponse)
def call_variants(req: VariantRequest) -> VariantResponse:
    """Llamada de variantes (SNV y deleciones) sobre GPU."""
    vc = VariantCaller()
    reads = [Read(sequence=r.sequence, start=r.start, strand=r.strand)
             for r in req.reads]

    variants = vc.call_variants(
        req.reference, reads, req.qualities,
        min_depth=req.min_depth, min_alt_freq=req.min_alt_freq,
    )

    # Profundidad media
    pile = vc.pileup(req.reference, reads, req.qualities)
    depth = pile.depth
    mean_depth = float(depth.sum().item()) / max(1, len(depth))

    return VariantResponse(
        variants=[v.__dict__ for v in variants],
        total_variants=len(variants),
        snvs=sum(1 for v in variants if v.type == "SNV"),
        deletions=sum(1 for v in variants if v.type == "DEL"),
        mean_depth=round(mean_depth, 2),
    )


class VariantFileRequest(BaseModel):
    ref_upload_id: str       # FASTA de referencia
    sam_upload_id: str       # SAM/BAM de lecturas mapeadas
    min_depth: int = 10
    min_alt_freq: float = 0.2
    region_size: int = 50_000_000


class VariantFileJobResponse(BaseModel):
    job_id: str
    events_url: str


@router.post("/call-file", response_model=VariantFileJobResponse)
async def call_variants_file(req: VariantFileRequest) -> VariantFileJobResponse:
    """
    Llama variantes de un SAM/BAM contra una referencia (proceso aislado).

    Procesa por regiones de ``region_size`` bases (VRAM acotada) y exporta
    el VCF en una subida descargable. El progreso y el resultado llegan
    por el stream SSE de ``events_url``.
    """
    ref_path = upload_path(req.ref_upload_id)
    sam_path = upload_path(req.sam_upload_id)

    vcf_id = uuid4().hex
    vcf_path = UPLOAD_DIR / f"{vcf_id}.vcf"
    create_upload(vcf_id, "vcf", Path(vcf_path), "variants.vcf", 0)

    job = jobs.manager.create("variant_call")
    jobs.manager.submit_process(job, {
        "kind": "variant_call",
        "ref_path": str(ref_path),
        "sam_path": str(sam_path),
        "out_vcf_path": str(vcf_path),
        "out_vcf_upload_id": vcf_id,
        "min_depth": req.min_depth,
        "min_alt_freq": req.min_alt_freq,
        "region_size": req.region_size,
    })
    return VariantFileJobResponse(job_id=job.id,
                                  events_url=f"/api/jobs/{job.id}/events")