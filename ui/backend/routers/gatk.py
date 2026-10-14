"""
Router GATK: HaplotypeCaller, BQSR y MarkDuplicates sobre archivos
subidos (SAM/BAM + referencia FASTA), como trabajos de fondo con
progreso SSE.

- ``POST /api/gatk/call-file``  -> job ``haplotype``  (VCF descargable)
- ``POST /api/gatk/bqsr``       -> job ``bqsr``       (tabla JSON + SAM)
- ``POST /api/gatk/markdups``   -> job ``markdups``   (SAM con flags)
"""

from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ui.backend import jobs
from ui.backend.uploads import UPLOAD_DIR, create_upload, upload_path

router = APIRouter(prefix="/api/gatk", tags=["gatk"])


class HaplotypeRequest(BaseModel):
    """Llamada de variantes estilo GATK HaplotypeCaller."""
    ref_upload_id: str
    sam_upload_id: str
    min_mapq: int = Field(20, ge=0, le=60)
    min_depth: int = Field(5, ge=1)
    min_qual: float = Field(10.0, ge=0.0)
    min_af: float = Field(0.15, ge=0.0, le=1.0)
    window: int = Field(50, ge=10, le=1000)
    extension: int = Field(50, ge=0, le=1000)
    max_haplotypes: int = Field(32, ge=2, le=256)
    ploidy: int = Field(2, ge=1, le=4)
    region_size: int = Field(50_000_000, ge=1000)
    sample_name: str = "SAMPLE"
    produce_vcf: bool = True


class BqsrRequest(BaseModel):
    """Recalibración de calidades de base (GATK BaseRecalibrator)."""
    ref_upload_id: str
    sam_upload_id: str
    min_base_quality: int = Field(0, ge=0, le=93)
    min_mapq: int = Field(0, ge=0, le=60)
    emit_sam: bool = True
    sample_name: str = "SAMPLE"


class MarkdupsRequest(BaseModel):
    """Marcado de duplicados de PCR/ópticos (GATK MarkDuplicates)."""
    sam_upload_id: str
    emit_sam: bool = True
    sample_name: str = "SAMPLE"


class GatkJobResponse(BaseModel):
    job_id: str
    events_url: str
    vcf_upload_id: Optional[str] = None
    sam_upload_id: Optional[str] = None
    table_upload_id: Optional[str] = None


@router.post("/call-file", response_model=GatkJobResponse)
async def call_haplotype_file(req: HaplotypeRequest) -> GatkJobResponse:
    """Encola la llamada de variantes estilo HaplotypeCaller."""
    ref_path = upload_path(req.ref_upload_id)
    sam_path = upload_path(req.sam_upload_id)

    vcf_id = None
    spec = {
        "kind": "haplotype",
        "ref_path": str(ref_path),
        "sam_path": str(sam_path),
        "min_mapq": req.min_mapq,
        "min_depth": req.min_depth,
        "min_qual": req.min_qual,
        "min_af": req.min_af,
        "window": req.window,
        "extension": req.extension,
        "max_haplotypes": req.max_haplotypes,
        "ploidy": req.ploidy,
        "region_size": req.region_size,
        "sample_name": req.sample_name,
        "out_vcf_path": None,
        "out_vcf_upload_id": None,
    }
    if req.produce_vcf:
        vcf_id = uuid4().hex
        spec["out_vcf_path"] = str(UPLOAD_DIR / f"{vcf_id}.vcf")
        spec["out_vcf_upload_id"] = vcf_id
        create_upload(vcf_id, "vcf", Path(spec["out_vcf_path"]),
                      "haplotype.vcf", 0)

    job = jobs.manager.create("haplotype")
    jobs.manager.submit_process(job, spec)
    return GatkJobResponse(
        job_id=job.id, events_url=f"/api/jobs/{job.id}/events",
        vcf_upload_id=vcf_id)


@router.post("/bqsr", response_model=GatkJobResponse)
async def run_bqsr(req: BqsrRequest) -> GatkJobResponse:
    """Encola la recalibración de calidades (BQSR)."""
    ref_path = upload_path(req.ref_upload_id)
    sam_path = upload_path(req.sam_upload_id)

    table_id = uuid4().hex
    sam_id = uuid4().hex if req.emit_sam else None
    spec = {
        "kind": "bqsr",
        "ref_path": str(ref_path),
        "sam_path": str(sam_path),
        "min_base_quality": req.min_base_quality,
        "min_mapq": req.min_mapq,
        "sample_name": req.sample_name,
        "out_table_path": str(UPLOAD_DIR / f"{table_id}.json"),
        "out_table_upload_id": table_id,
        "out_sam_path": str(UPLOAD_DIR / f"{sam_id}.sam") if sam_id else None,
        "out_sam_upload_id": sam_id,
    }
    create_upload(table_id, "report", Path(spec["out_table_path"]),
                  "recal_table.json", 0)
    if sam_id:
        create_upload(sam_id, "sam", Path(spec["out_sam_path"]),
                      "recalibrated.sam", 0)

    job = jobs.manager.create("bqsr")
    jobs.manager.submit_process(job, spec)
    return GatkJobResponse(
        job_id=job.id, events_url=f"/api/jobs/{job.id}/events",
        table_upload_id=table_id, sam_upload_id=sam_id)


@router.post("/markdups", response_model=GatkJobResponse)
async def run_markdups(req: MarkdupsRequest) -> GatkJobResponse:
    """Encola el marcado de duplicados."""
    sam_path = upload_path(req.sam_upload_id)

    sam_id = uuid4().hex if req.emit_sam else None
    spec = {
        "kind": "markdups",
        "sam_path": str(sam_path),
        "sample_name": req.sample_name,
        "out_sam_path": str(UPLOAD_DIR / f"{sam_id}.sam") if sam_id else None,
        "out_sam_upload_id": sam_id,
    }
    if sam_id:
        create_upload(sam_id, "sam", Path(spec["out_sam_path"]),
                      "markdups.sam", 0)

    job = jobs.manager.create("markdups")
    jobs.manager.submit_process(job, spec)
    return GatkJobResponse(
        job_id=job.id, events_url=f"/api/jobs/{job.id}/events",
        sam_upload_id=sam_id)
