"""
Subida de archivos FASTA en streaming asíncrono.

El archivo se recibe como un stream de chunks (nunca en memoria) y se
escribe a disco en un hilo de fondo para no bloquear el event loop.
Las estadísticas (nº de registros, bases totales, primer registro) se
calculan en un HILO DE FONDO tras la carga, de modo que la respuesta no
bloquea esperando un escaneo completo del archivo (que en un FASTA de
50-100 GB puede tardar minutos): se consultan en ``GET /api/upload/{id}``
mientras ``stats_status`` sea ``pending``.

Además de la carga multipart, existe ``POST /api/upload/register`` para
registrar una ruta del servidor sin transmitir el archivo por HTTP, la
vía recomendada para genomas de decenas de GB.
"""

import hashlib
import threading
from pathlib import Path
from typing import AsyncIterator, Optional
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel

from Genoly.io.fasta import FastaReader
from Genoly.io.fastq import FastqReader
from Genoly.quantitative.preprocess import scan_grid_stats
from ui.backend.maintenance import MAX_UPLOAD_BYTES, check_upload_size
from ui.backend.uploads import (
    UPLOAD_DIR,
    check_registrable,
    create_upload,
    get_upload,
    set_bytes,
    set_checksum,
    set_stats,
)

router = APIRouter(prefix="/api/upload", tags=["upload"])

CHUNK_BYTES = 1024 * 1024

#: Tamaño máximo por chunk en la subida por partes. La escritura es en
#: streaming (nunca se bufferiza el chunk completo en RAM), pero el límite
#: evita que un cliente envíe cuerpos gigantes de una sola vez.
MAX_CHUNK_PUT = 256 * 1024 * 1024

_FASTA_SUFFIXES = (".fasta", ".fa", ".fna", ".txt")
_FASTQ_SUFFIXES = (".fastq", ".fq")
_TABULAR_SUFFIXES = (".csv", ".tsv", ".xlsx", ".xls")


def _upload_kind(filename: str) -> str:
    """Tipo de subida según la extensión: 'fastq', 'tabular' o 'fasta'."""
    suffix = Path(filename).suffix.lower()
    if suffix in _FASTQ_SUFFIXES:
        return "fastq"
    if suffix in _TABULAR_SUFFIXES:
        return "tabular"
    return "fasta"


def _file_ext(kind: str, filename: str = "") -> str:
    """Extensión del archivo en disco: preserva .xlsx/.xls de los tabulares."""
    if kind == "fastq":
        return "fastq"
    if kind == "tabular":
        suffix = Path(filename).suffix.lower()
        if suffix in (".xlsx", ".xls"):
            return suffix[1:]
        return "csv"
    return "fasta"


async def _iter_chunks(file: UploadFile,
                       chunk_size: int = CHUNK_BYTES) -> AsyncIterator[bytes]:
    """
    Itera los chunks del archivo subido de forma asíncrona.

    Usa ``file.chunks()`` (async iterator nativo) si la versión de
    Starlette lo ofrece; en caso contrario cae a ``await file.read(n)``.
    """
    chunks = getattr(file, "chunks", None)
    if callable(chunks):
        async for chunk in chunks():
            if chunk:
                yield chunk
        return
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            return
        yield chunk


class UploadedMeta(BaseModel):
    id: str
    description: Optional[str] = None
    length: int


class UploadResponse(BaseModel):
    upload_id: str
    filename: str
    kind: str = "multipart"
    stats_status: str = "pending"  # pending | ready | error
    records: Optional[int] = None
    total_bases: Optional[int] = None
    first: Optional[UploadedMeta] = None
    bytes_on_disk: int = 0
    sha256: Optional[str] = None
    error: Optional[str] = None


class RegisterRequest(BaseModel):
    path: str


class ChunkedInitRequest(BaseModel):
    filename: str
    size: Optional[int] = None  # tamaño esperado (opcional, informativo)


class ChunkedInitResponse(BaseModel):
    upload_id: str
    filename: str
    received: int = 0


class ChunkedPutResponse(BaseModel):
    upload_id: str
    offset: int
    received: int


class ChunkedCompleteRequest(BaseModel):
    size: Optional[int] = None
    expected_sha256: Optional[str] = None


def _chunked_dest(upload_id: str) -> Path:
    """Resuelve el destino de una subida por partes validando el id."""
    if not upload_id or not all(c in "0123456789abcdef" for c in upload_id) \
            or len(upload_id) != 32:
        raise HTTPException(status_code=400,
                            detail="Identificador de subida inválido")
    meta = get_upload(upload_id)
    if meta is None:
        raise HTTPException(status_code=404,
                            detail="Subida por partes no encontrada")
    dest = Path(meta["path"])
    if not dest.is_file():
        raise HTTPException(status_code=404,
                            detail="Subida por partes no encontrada")
    return dest


def _sha256_file(path: Path) -> str:
    """SHA-256 en streaming del archivo (una pasada, sin cargarlo en RAM)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK_BYTES), b""):
            h.update(block)
    return h.hexdigest()


def _background_stats(upload_id: str, dest: Path) -> None:
    """Escanea las estadísticas del archivo en un hilo de fondo y actualiza
    el registro. Los archivos ilegibles o sin registros FASTA se borran
    (solo vía multipart; las rutas registradas se dejan intactas) y el
    estado queda en ``error``."""
    kind = _upload_kind(dest.name)
    try:
        if kind == "fastq":
            stats = FastqReader(dest).scan_stats()
            records = stats.reads
            total_bases = stats.total_bases
            first = ({"id": stats.first_id or "", "description": None,
                      "length": stats.first_length}
                     if stats.first_id is not None else None)
        elif kind == "tabular":
            stats = scan_grid_stats(dest)
            records = stats["rows"]
            total_bases = records * stats["columns"]
            first = None
        else:
            stats = FastaReader(dest).scan_stats()
            records = stats.records
            total_bases = stats.total_bases
            first = ({"id": stats.first_id or "",
                      "description": stats.first_description,
                      "length": stats.first_length}
                     if stats.first_id is not None else None)
    except Exception as exc:
        set_stats(upload_id, 0, 0, error=str(exc) or exc.__class__.__name__)
        return

    if records == 0:
        set_stats(upload_id, 0, 0,
                  error="El archivo no contiene registros FASTA/FASTQ")
        if dest.parent == UPLOAD_DIR:
            dest.unlink(missing_ok=True)
        return

    set_stats(upload_id, records, total_bases, first)


def _response(meta: dict) -> UploadResponse:
    first = meta.get("first")
    if isinstance(first, dict):
        first = UploadedMeta(**first)
    return UploadResponse(
        upload_id=meta["upload_id"],
        filename=meta["filename"],
        kind=meta["kind"],
        stats_status=meta["stats_status"],
        records=meta.get("records"),
        total_bases=meta.get("total_bases"),
        first=first,
        bytes_on_disk=meta.get("bytes_on_disk", 0),
        sha256=meta.get("sha256"),
        error=meta.get("error"),
    )


@router.post("", response_model=UploadResponse)
async def upload_fasta(file: UploadFile = File(...)) -> UploadResponse:
    """Guarda el FASTA/FASTQ en disco (streaming) y devuelve un
    identificador para analizarlo; las estadísticas llegan en segundo
    plano."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Archivo sin nombre")

    kind = _upload_kind(file.filename)
    upload_id = uuid4().hex
    dest = UPLOAD_DIR / f"{upload_id}.{_file_ext(kind, file.filename)}"
    bytes_on_disk = 0
    sha = hashlib.sha256()

    try:
        with dest.open("wb") as fh:
            # Escritura en hilo de fondo: el event loop no se bloquea.
            # El checksum se calcula en streaming, sin releer el archivo.
            async for chunk in _iter_chunks(file):
                bytes_on_disk += await run_in_threadpool(fh.write, chunk)
                sha.update(chunk)
                if MAX_UPLOAD_BYTES and bytes_on_disk > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="El archivo supera el límite de "
                               f"{MAX_UPLOAD_BYTES // (1024 ** 3)} GiB "
                               "(GENOLY_MAX_UPLOAD_BYTES)")
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    except Exception:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500,
                            detail="No se pudo guardar el archivo")

    if bytes_on_disk == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="El archivo está vacío")

    create_upload(upload_id, kind, dest,
                  Path(file.filename).name, bytes_on_disk)
    set_checksum(upload_id, sha.hexdigest())
    threading.Thread(target=_background_stats, args=(upload_id, dest),
                     daemon=True).start()
    return _response(get_upload(upload_id))


# ---------------------------------------------------------------------- #
# Subida por partes (reanudable) para archivos de decenas de GB
# ---------------------------------------------------------------------- #
@router.post("/chunked", response_model=ChunkedInitResponse)
def chunked_init(req: ChunkedInitRequest) -> ChunkedInitResponse:
    """
    Inicia una subida por partes: crea el archivo vacío y devuelve el id.
    Los chunks se envían después con ``PUT /api/upload/chunked/{id}``.
    """
    if not req.filename.strip():
        raise HTTPException(status_code=400, detail="Archivo sin nombre")
    if req.size is not None:
        if req.size < 0:
            raise HTTPException(status_code=400, detail="Tamaño inválido")
        check_upload_size(0, req.size)

    kind = _upload_kind(req.filename)
    upload_id = uuid4().hex
    dest = UPLOAD_DIR / f"{upload_id}.{_file_ext(kind, req.filename)}"
    dest.touch()
    create_upload(upload_id, kind, dest,
                  Path(req.filename).name, 0)
    return ChunkedInitResponse(upload_id=upload_id,
                               filename=Path(req.filename).name,
                               received=0)


@router.get("/chunked/{upload_id}", response_model=ChunkedPutResponse)
def chunked_status(upload_id: str) -> ChunkedPutResponse:
    """Bytes ya recibidos (para reanudar una subida interrumpida)."""
    dest = _chunked_dest(upload_id)
    return ChunkedPutResponse(upload_id=upload_id,
                              offset=0,
                              received=dest.stat().st_size)


@router.put("/chunked/{upload_id}", response_model=ChunkedPutResponse)
async def chunked_put(upload_id: str, request: Request,
                      offset: int = Query(0, ge=0)) -> ChunkedPutResponse:
    """
    Escribe un chunk en la posición ``offset`` (streaming, sin bufferizar).

    El cliente indica el offset absoluto en el archivo; escribir en la
    misma posición de nuevo reanuda/sobrescribe de forma idempotente.
    ``offset`` no puede adelantarse al final del archivo (evita huecos
    dispersos), el chunk no puede superar :data:`MAX_CHUNK_PUT` y el total
    no puede superar ``GENOLY_MAX_UPLOAD_BYTES``.
    """
    dest = _chunked_dest(upload_id)
    current = dest.stat().st_size
    if offset > current:
        raise HTTPException(
            status_code=409,
            detail=f"Offset {offset} por delante del final del archivo "
                   f"({current} bytes recibidos)")
    declared = request.headers.get("content-length")
    if declared and declared.isdigit():
        check_upload_size(offset, int(declared))
        if int(declared) > MAX_CHUNK_PUT:
            raise HTTPException(
                status_code=413,
                detail=f"Chunk de {int(declared)} bytes: el máximo es "
                       f"{MAX_CHUNK_PUT // (1024 ** 2)} MiB")

    received = offset
    written = 0
    try:
        with dest.open("r+b") as fh:
            fh.seek(offset)
            async for chunk in request.stream():
                written += len(chunk)
                if written > MAX_CHUNK_PUT:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Chunk de más de "
                               f"{MAX_CHUNK_PUT // (1024 ** 2)} MiB")
                check_upload_size(offset, written)
                await run_in_threadpool(fh.write, chunk)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500,
                            detail="No se pudo escribir el chunk")
    if written == 0:
        raise HTTPException(status_code=400, detail="Chunk vacío")

    received = dest.stat().st_size
    set_bytes(upload_id, received)
    return ChunkedPutResponse(upload_id=upload_id, offset=offset,
                              received=received)


@router.post("/chunked/{upload_id}/complete", response_model=UploadResponse)
def chunked_complete(upload_id: str,
                     req: ChunkedCompleteRequest) -> UploadResponse:
    """
    Cierra la subida por partes: valida el tamaño (y el checksum, si se
    indicó) y lanza el cálculo de estadísticas en segundo plano.
    """
    dest = _chunked_dest(upload_id)
    received = dest.stat().st_size
    check_upload_size(0, received)
    if req.size is not None and received != req.size:
        raise HTTPException(
            status_code=400,
            detail=f"Subida incompleta: {received} de {req.size} bytes")
    if received == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="El archivo está vacío")

    sha = None
    if req.expected_sha256:
        sha = _sha256_file(dest)
        if sha != req.expected_sha256.lower():
            dest.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400,
                detail=f"Checksum no coincide: {sha} != {req.expected_sha256}")

    set_checksum(upload_id, sha)
    set_bytes(upload_id, received)
    threading.Thread(target=_background_stats, args=(upload_id, dest),
                     daemon=True).start()
    return _response(get_upload(upload_id))


@router.post("/register", response_model=UploadResponse)
def register_fasta(req: RegisterRequest) -> UploadResponse:
    """
    Registra un archivo FASTA/FASTQ ya presente en el servidor (sin
    copiarlo ni transmitirlo por HTTP). Recomendado para archivos de
    50-100 GB: se valida la ruta y las estadísticas se calculan en
    segundo plano.
    """
    if not req.path.strip():
        raise HTTPException(status_code=400, detail="Indica una ruta")
    src = check_registrable(Path(req.path))
    kind = _upload_kind(src.name)

    upload_id = uuid4().hex
    create_upload(upload_id, kind, src, src.name,
                  bytes_on_disk=int(src.stat().st_size))
    threading.Thread(target=_background_stats, args=(upload_id, src),
                     daemon=True).start()
    return _response(get_upload(upload_id))


@router.get("/{upload_id}", response_model=UploadResponse)
def upload_status(upload_id: str) -> UploadResponse:
    """Estado de una subida: metadatos y estadísticas (si ya están)."""
    meta = get_upload(upload_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Subida no encontrada")
    return _response(meta)


@router.get("/{upload_id}/download")
def download_upload(upload_id: str) -> FileResponse:
    """Descarga el archivo de una subida (SAM/VCF/FASTA/FASTQ...)."""
    meta = get_upload(upload_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Subida no encontrada")
    path = Path(meta["path"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="El archivo ya no existe")
    return FileResponse(path, filename=meta["filename"],
                        media_type="application/octet-stream")