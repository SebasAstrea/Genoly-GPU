"""
Gestión de datasets NCBI (carpeta descomprimida o .zip).

Un dataset NCBI típico contiene ``ncbi_dataset/data/<ACCESION>/`` con los
FASTA de cada ensamblaje (p. ej. GCA y GCF del mismo genoma). Este módulo
registra el dataset, descubre todos los FASTA (``.fna``/``.fasta``/``.fa``)
de forma recursiva y calcula las estadísticas de cada archivo en segundo
plano, de modo que todos los ensamblajes puedan analizarse en un solo
flujo (QC y k-mers combinados).

Los datasets se guardan en un registro JSON persistente (patrón idéntico
al de ``uploads.py``, con candado **entre procesos**: el proceso
trabajador escribe el mismo fichero, ver :mod:`ui.backend.registry`). Los
.zip se extraen a disco (nunca se cargan en RAM) en
``GENOLY_DATASET_DIR/extract/<id>`` con guarda contra zip-slip. Las rutas
registradas deben caer dentro de los directorios de datos permitidos
(:mod:`ui.backend.security`).
"""

import os
import re
import shutil
import threading
import time
import zipfile
from pathlib import Path
from tempfile import gettempdir
from typing import List, Optional

from fastapi import HTTPException

from Genoly.io.fasta import FastaReader
from ui.backend.registry import JsonRegistry
from ui.backend.security import check_data_path

#: Extensiones de archivos FASTA soportados en el descubrimiento.
FASTA_SUFFIXES = (".fna", ".fasta", ".fa", ".faa", ".ffn")

_DATASETS_DIR = Path(
    os.environ.get("GENOLY_DATASET_DIR")
    or (Path(gettempdir()) / "genoly_datasets")
).expanduser()
_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
_DATASETS_DIR.joinpath("extract").mkdir(parents=True, exist_ok=True)

_REGISTRY_PATH = _DATASETS_DIR / "registry.json"

_DATASET_RE = re.compile(r"^[0-9a-f]{32}$")

#: Registro compartido con los procesos trabajadores (candado inter-proceso).
_registry = JsonRegistry(_REGISTRY_PATH)


def get_dataset(dataset_id: str) -> Optional[dict]:
    """Metadatos de un dataset, o None si no existe."""
    if not _DATASET_RE.match(dataset_id or ""):
        raise HTTPException(status_code=400,
                            detail="Identificador de dataset inválido")
    return _registry.get(dataset_id)


def _validate_source(src: Path) -> Path:
    """Valida que la fuente exista y respete el allowlist de datos."""
    return check_data_path(src, extra=(_DATASETS_DIR,))


def discover_fastas(path) -> List[Path]:
    """
    Encuentra los archivos FASTA de una fuente (directorio o .zip extraído).

    Args:
        path: Directorio (se recorre recursivamente) o carpeta ya extraída.

    Returns:
        Lista de rutas a archivos FASTA, ordenadas por nombre.
    """
    path = Path(path)
    fastas: List[Path] = []
    if path.is_file():
        if path.suffix.lower() in FASTA_SUFFIXES:
            fastas.append(path)
        return fastas
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.suffix.lower() in FASTA_SUFFIXES:
            fastas.append(p)
    return fastas


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extrae un .zip a disco con guarda contra zip-slip (no se toca RAM)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = dest_dir.joinpath(member.filename).resolve()
            if not target.is_relative_to(dest_dir.resolve()):
                raise HTTPException(status_code=400,
                                    detail="El .zip contiene rutas inseguras")
            if member.is_dir():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def _file_meta(path: Path) -> dict:
    return {
        "path": str(path),
        "filename": path.name,
        "size": int(path.stat().st_size),
        "created_at": time.time(),
        "stats_status": "pending",   # pending | ready | error
        "records": None,
        "total_bases": None,
        "first": None,
        "error": None,
    }


def register_dataset(source: str) -> dict:
    """
    Registra un dataset NCBI (directorio o .zip) descubriendo sus FASTA.

    - Directorio: se recorren recursivamente los ``.fna/.fasta/.fa``.
    - .zip: se extrae a disco en ``extract/<id>`` y se descubren los FASTA.

    Las estadísticas de cada archivo se calculan en un hilo de fondo; se
    consultan en ``GET /api/dataset/{id}``.
    """
    if not source or not source.strip():
        raise HTTPException(status_code=400, detail="Indica una ruta")
    src = _validate_source(Path(source))

    from uuid import uuid4
    dataset_id = uuid4().hex

    if src.is_file() and src.suffix.lower() == ".zip":
        extract_dir = _DATASETS_DIR / "extract" / dataset_id
        try:
            _extract_zip(src, extract_dir)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="El .zip es inválido")
        base = extract_dir
    else:
        base = src

    fastas = discover_fastas(base)
    if not fastas:
        raise HTTPException(
            status_code=400,
            detail=f"No se encontraron archivos FASTA en la ruta "
                   f"({FASTA_SUFFIXES})")

    files = [_file_meta(p) for p in fastas]
    meta = {
        "dataset_id": dataset_id,
        "source": str(src),
        "kind": "zip" if src.suffix.lower() == ".zip" else "dir",
        "files": files,
        "status": "pending",   # pending | ready | error
        "error": None,
    }
    _registry.set(dataset_id, meta)

    threading.Thread(target=_scan_dataset, args=(dataset_id,),
                     daemon=True).start()
    return dict(meta)


def _scan_dataset(dataset_id: str) -> None:
    """Escanea las estadísticas de cada FASTA del dataset en un hilo."""
    meta = get_dataset(dataset_id)
    if meta is None:
        return
    for i, finfo in enumerate(meta["files"]):
        try:
            stats = FastaReader(finfo["path"]).scan_stats()
            first = None
            if stats.first_id is not None:
                first = {
                    "id": stats.first_id or "",
                    "description": stats.first_description,
                    "length": stats.first_length,
                }
            _set_file_stats(dataset_id, i, stats.records,
                            stats.total_bases, first)
        except Exception as exc:
            _set_file_stats(dataset_id, i, 0, 0, None,
                            error=str(exc) or exc.__class__.__name__)


def _set_file_stats(dataset_id: str, index: int, records: int,
                    total_bases: int, first: Optional[dict],
                    *, error: Optional[str] = None) -> None:
    def _apply(meta: dict) -> None:
        if index >= len(meta["files"]):
            return
        finfo = meta["files"][index]
        finfo["records"] = records
        finfo["total_bases"] = total_bases
        finfo["first"] = first
        finfo["stats_status"] = "error" if error else "ready"
        finfo["error"] = error
        meta["status"] = "error" if any(
            f["stats_status"] == "error" for f in meta["files"]) else (
            "ready" if all(f["stats_status"] != "pending"
                           for f in meta["files"]) else "pending")

    _registry.update_entry(dataset_id, _apply)


def all_datasets() -> dict:
    """Copia del registro completo de datasets (para mantenimiento)."""
    return dict(_registry.locked_read())


def delete_dataset(dataset_id: str) -> None:
    """Elimina la entrada de un dataset del registro."""
    _registry.update(lambda data: data.pop(dataset_id, None))