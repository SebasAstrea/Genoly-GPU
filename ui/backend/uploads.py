"""
Gestión de archivos FASTA subidos por la UI.

Hay dos vías de carga, ambas con memoria acotada:

1. Multipart (``POST /api/upload``): el archivo se recibe como stream y
   se escribe a disco por bloques, nunca en RAM.
2. Registro de ruta del servidor (``POST /api/upload/register``): para
   archivos de decenas de GB (50-100 GB) lo correcto es apuntar a un
   archivo ya presente en el servidor en lugar de transmitirlo por HTTP.
   No se copia nada: el análisis lee la ruta original en streaming.

Las estadísticas (registros, bases totales, primer registro) se calculan
en un hilo de fondo tras la carga, de modo que la respuesta de la carga
no bloquea esperando un escaneo completo del archivo; el estado se
consulta en ``GET /api/upload/{id}``.

Toda la información de cada subida vive en un registro JSON
(``registry.json``) bajo ``UPLOAD_DIR``, persistente entre reinicios y
protegido con un candado ENTRE PROCESOS (flock): el proceso trabajador que
calcula las estadísticas escribe el mismo fichero que el servidor, así que
un candado de hilos no bastaba. Ver :mod:`ui.backend.registry`.

Directorio de uploads: ``GENOLY_UPLOAD_DIR`` (por defecto, el temporal
del sistema; para archivos de decenas de GB apúntalo a un disco real,
p. ej. ``/var/tmp/genoly/uploads``). El derrame del conteo de k-mers se
configura por separado con ``GENOLY_SPILL_DIR`` (ver Genoly.kmer.kmers).

Las rutas registradas con ``POST /api/upload/register`` deben caer dentro
de los directorios de datos permitidos (ver
:data:`GENOLY_ALLOWED_DATA_DIRS` en :mod:`ui.backend.security`).
"""

import os
import re
import time
from pathlib import Path
from tempfile import gettempdir
from typing import Optional

from fastapi import HTTPException

from ui.backend.registry import JsonRegistry
from ui.backend.security import check_data_path

UPLOAD_DIR = Path(
    os.environ.get("GENOLY_UPLOAD_DIR")
    or (Path(gettempdir()) / "genoly_uploads")
).expanduser()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_REGISTRY_PATH = UPLOAD_DIR / "registry.json"

_UPLOAD_RE = re.compile(r"^[0-9a-f]{32}$")

#: Registro compartido con los procesos trabajadores (candado inter-proceso).
_registry = JsonRegistry(_REGISTRY_PATH)


def get_upload(upload_id: str) -> Optional[dict]:
    """Metadatos de una subida, o None si no existe."""
    if not _UPLOAD_RE.match(upload_id or ""):
        raise HTTPException(status_code=400,
                            detail="Identificador de subida inválido")
    return _registry.get(upload_id)


def create_upload(upload_id: str, kind: str, path: Path, filename: str,
                  bytes_on_disk: int = 0) -> dict:
    """Registra una subida nueva (multipart o ruta registrada) con
    estadísticas pendientes."""
    meta = {
        "upload_id": upload_id,
        "kind": kind,                  # multipart | registered
        "path": str(path),
        "filename": filename,
        "bytes_on_disk": bytes_on_disk,
        "created_at": time.time(),
        "stats_status": "pending",     # pending | ready | error
        "records": None,
        "total_bases": None,
        "first": None,
        "sha256": None,
        "error": None,
    }
    _registry.set(upload_id, meta)
    return dict(meta)


def set_checksum(upload_id: str, sha256: Optional[str]) -> None:
    """Guarda el checksum SHA-256 (hex) de una subida ya registrada."""
    _registry.update_entry(upload_id,
                           lambda meta: meta.__setitem__("sha256", sha256))


def set_bytes(upload_id: str, bytes_on_disk: int) -> None:
    """Actualiza los bytes escritos de una subida en curso."""
    _registry.update_entry(
        upload_id,
        lambda meta: meta.__setitem__("bytes_on_disk", bytes_on_disk))


def set_stats(upload_id: str, records: int, total_bases: int, first=None,
              *, error: Optional[str] = None) -> None:
    """Actualiza las estadísticas (o el error) de una subida ya registrada."""
    def _apply(meta: dict) -> None:
        meta["records"] = records
        meta["total_bases"] = total_bases
        meta["first"] = first
        meta["stats_status"] = "error" if error else "ready"
        meta["error"] = error

    _registry.update_entry(upload_id, _apply)


def upload_path(upload_id: str) -> Path:
    """
    Resuelve la ruta real del archivo de una subida (multipart o
    registrada) validando el identificador. Si no está en el registro,
    la subida ya no existe (404).
    """
    if not _UPLOAD_RE.match(upload_id or ""):
        raise HTTPException(status_code=400,
                            detail="Identificador de subida inválido")
    meta = get_upload(upload_id)
    if meta is None:
        raise HTTPException(status_code=404,
                            detail="El archivo subido ya no existe")
    path = Path(meta["path"])
    if not path.is_file():
        raise HTTPException(status_code=404,
                            detail="El archivo subido ya no existe")
    return path


def check_registrable(path: Path) -> Path:
    """
    Valida una ruta del servidor para el registro de subidas.

    - Debe existir y ser un archivo regular legible.
    - Debe estar dentro de un directorio de datos permitido
      (``GENOLY_ALLOWED_DATA_DIRS``; por defecto la raíz del proyecto y el
      propio directorio de subidas).
    - No puede ser una ruta sensible del sistema (/etc, ~/.ssh, .env, ...).

    Devuelve la ruta resuelta y absoluta.
    """
    src = check_data_path(path, extra=(UPLOAD_DIR,))
    if not os.access(src, os.R_OK):
        raise HTTPException(status_code=400,
                            detail="El archivo no es legible")
    return src


def all_uploads() -> dict:
    """Copia del registro completo de subidas (para mantenimiento)."""
    return dict(_registry.locked_read())


def delete_upload(upload_id: str) -> None:
    """Elimina la entrada de una subida del registro."""
    _registry.update(lambda data: data.pop(upload_id, None))