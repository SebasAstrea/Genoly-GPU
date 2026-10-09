"""
Conteo de k-mers sobre GPU.

- POST /api/kmer/count       -> síncrono (FastAPI lo ejecuta en el
  threadpool; no bloquea el event loop). Usa el pipeline de streaming
  para archivos subidos.
- POST /api/kmer/count-async -> encola un trabajo de fondo y devuelve un
  job_id; el progreso se consulta por SSE en /api/jobs/{id}/events.

Los archivos subidos se procesan con el pipeline de streaming completo:
lectura por bloques de disco (64 KiB) -> ventanas con solape k-1 ->
lotes de RAM -> micro-lotes adaptativos de GPU según la VRAM libre
(``KmerCounter.count_fasta``), con RAM y VRAM acotadas sea cual sea el
tamaño del archivo.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from Genoly.io.fasta import FastaReader
from Genoly.kmer.kmers import KmerCounter
from ui.backend import jobs
from ui.backend.datasets import get_dataset
from ui.backend.uploads import get_upload, upload_path

router = APIRouter(prefix="/api/kmer", tags=["kmer"])


class KmerRequest(BaseModel):
    sequences: List[str] = []
    k: int = 21
    canonical: bool = True
    min_abundance: int = 1
    top: int = 20
    upload_id: Optional[str] = None  # archivo FASTA subido (streaming)
    dataset_id: Optional[str] = None  # dataset NCBI (varios FASTA a la vez)


class KmerResponse(BaseModel):
    k: int
    total_unique: int
    total_kmers: int
    top_kmers: List[dict]
    spectrum: dict
    genome_estimate: Optional[float] = None


class KmerJobResponse(BaseModel):
    job_id: str
    events_url: str


def _validate(req: KmerRequest) -> None:
    if not 1 <= req.k <= 31:
        raise HTTPException(status_code=422,
                            detail="k debe estar entre 1 y 31")
    if not req.upload_id and not req.dataset_id and not req.sequences:
        raise HTTPException(status_code=422,
                            detail="Indica secuencias, un archivo subido o un dataset")


def _load_dataset(dataset_id: str) -> dict:
    dataset = get_dataset(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset no encontrado")
    return dataset


def _dataset_totals(dataset: dict) -> dict:
    """Totales de registros y bases del dataset (para el % de progreso)."""
    total_records = 0
    total_bases = 0
    for finfo in dataset["files"]:
        if finfo.get("records") is None:
            # estadísticas aún no listas: escaneo ligero del archivo
            stats = FastaReader(finfo["path"]).scan_stats()
            total_records += stats.records
            total_bases += stats.total_bases
            continue
        total_records += finfo["records"]
        total_bases += finfo["total_bases"]
    return {"total_records": total_records, "total_bases": total_bases}


def _build_payload(kc: KmerCounter, req: KmerRequest,
                   values, counts) -> dict:
    top_kmers = [
        {"kmer": kc.decode_kmer(int(v), req.k), "count": int(c)}
        for v, c in zip(values[:req.top].tolist(), counts[:req.top].tolist())
    ]

    spectrum = KmerCounter.spectrum_from_counts(counts)

    estimate = None
    if req.min_abundance <= 1 and values.numel() > 0:
        size, _ = KmerCounter.estimate_from_counts(counts)
        estimate = round(size, 0) if size > 0 else None

    return {
        "k": req.k,
        "total_unique": len(values),
        "total_kmers": int(counts.sum().item()) if len(counts) else 0,
        "top_kmers": top_kmers,
        "spectrum": spectrum,
        "genome_estimate": estimate,
    }


def _finalize_estimate(payload: dict, req: KmerRequest) -> dict:
    """Añade la estimación de genoma al payload del modo agregado."""
    estimate = None
    if req.min_abundance <= 1 and payload["total_unique"] > 0:
        size, _ = KmerCounter.estimate_from_spectrum(payload["spectrum"])
        estimate = round(size, 0) if size > 0 else None
    payload["genome_estimate"] = estimate
    return payload


def _count_upload(kc: KmerCounter, path, req: KmerRequest,
                  on_progress=None) -> dict:
    """
    Conteo de k-mers de un FASTA subido con RAM y disco acotados.

    Usa el acumulador particionado con derrame a disco
    (``count_fasta_aggregated``): genomas completos con millones de
    k-mers únicos no materializan jamás el resultado completo en RAM.
    """
    payload = kc.count_fasta_aggregated(
        path, k=req.k, canonical=req.canonical,
        min_abundance=req.min_abundance, top=req.top,
        on_progress=on_progress)
    return _finalize_estimate(payload, req)


def _count_dataset(kc: KmerCounter, dataset: dict, req: KmerRequest,
                   on_progress=None) -> dict:
    """
    Conteo de k-mers de todos los FASTA de un dataset NCBI en un solo
    acumulador: el resultado combinado es exacto (los k-mers compartidos
    entre ensamblajes no se duplican).
    """
    paths = [finfo["path"] for finfo in dataset["files"]]
    payload = kc.count_fastas_aggregated(
        paths, k=req.k, canonical=req.canonical,
        min_abundance=req.min_abundance, top=req.top,
        on_progress=on_progress)
    return _finalize_estimate(payload, req)


def _count_sync(req: KmerRequest) -> dict:
    """Conteo síncrono con el pipeline de streaming para uploads/datasets."""
    kc = KmerCounter()
    if req.upload_id:
        path = upload_path(req.upload_id)
        return _count_upload(kc, path, req)
    if req.dataset_id:
        return _count_dataset(kc, _load_dataset(req.dataset_id), req)
    values, counts = kc.count(
        req.sequences, k=req.k, canonical=req.canonical,
        min_abundance=req.min_abundance)
    return _build_payload(kc, req, values, counts)


@router.post("/count", response_model=KmerResponse)
def count_kmers(req: KmerRequest) -> KmerResponse:
    """Conteo de k-mers sobre GPU (bloqueante, en threadpool)."""
    _validate(req)
    return KmerResponse(**_count_sync(req))


@router.post("/count-async", response_model=KmerJobResponse)
async def count_kmers_async(req: KmerRequest) -> KmerJobResponse:
    """
    Encola el conteo como trabajo de fondo y devuelve el job_id.

    El trabajo se ejecuta en un PROCESO SEPARADO (aislamiento): si revienta
    (OOM, segfault), la API sigue viva y el job queda en error. Consume el
    stream SSE en ``/api/jobs/{job_id}/events`` para recibir el progreso
    (registros y bases leídas, micro-lotes GPU) y el resultado final.
    """
    _validate(req)

    spec = {
        "kind": "kmer",
        "k": req.k,
        "canonical": req.canonical,
        "min_abundance": req.min_abundance,
        "top": req.top,
        "sequences": req.sequences,
    }
    totals: dict = {}

    if req.upload_id:
        path = upload_path(req.upload_id)  # valida y lanza 404/400 antes
        spec["path"] = str(path)
        # Totales para el % de avance. La subida ya calculó las estadísticas
        # en segundo plano (registry): se reutilizan y se evita un escaneo
        # completo (una lectura de 50-100 GB). Solo se rescanea si faltan.
        meta = get_upload(req.upload_id)
        if meta and meta.get("records") is not None and \
                meta.get("total_bases") is not None:
            totals = {"total_records": meta["records"],
                      "total_bases": meta["total_bases"]}
        else:
            stats = FastaReader(path).scan_stats()
            totals = {"total_records": stats.records,
                      "total_bases": stats.total_bases}
    elif req.dataset_id:
        dataset = _load_dataset(req.dataset_id)
        spec["paths"] = [finfo["path"] for finfo in dataset["files"]]
        totals = _dataset_totals(dataset)

    job = jobs.manager.create("kmer")
    jobs.manager.submit_process(job, spec, totals=totals)
    return KmerJobResponse(job_id=job.id,
                           events_url=f"/api/jobs/{job.id}/events")
