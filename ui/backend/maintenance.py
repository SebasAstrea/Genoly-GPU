"""
Límites de tamaño y caducidad (TTL) de los archivos de trabajo.

Dos controles contra el llenado de disco:

- **Límite por archivo** (``GENOLY_MAX_UPLOAD_BYTES``, 50 GiB por defecto):
  se aplica en la carga multipart, en la carga por partes (tamaño declarado,
  offset y total) y al cerrarla. Sin este límite, ``PUT /upload/chunked/{id}``
  aceptaba cualquier ``offset`` y cualquier número de chunks.

- **TTL** (``GENOLY_UPLOAD_TTL_HOURS``, 7 días por defecto, ``0`` para
  desactivarlo): borra las subidas y las extracciones de datasets antiguas
  junto con su entrada de registro, más los ficheros huérfanos (cargas
  interrumpidas y temporales de registro). Los archivos *registrados* por el
  usuario no se tocan: son datos suyos, no de la aplicación.

``run_cleanup`` está pensado para llamarse desde el hilo de limpieza de
``ui.backend.main``.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("genoly.ui.maintenance")

_GIB = 1024 ** 3


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except ValueError:
        logger.warning("%s no es un entero; se usa %s", name, default)
        return default
    return max(0, value)


#: Tamaño máximo por archivo (multipart o por partes). 0 = sin límite.
MAX_UPLOAD_BYTES = _positive_int("GENOLY_MAX_UPLOAD_BYTES", 50 * _GIB)

#: Caducidad de subidas y extracciones en horas. 0 = desactivado.
UPLOAD_TTL_SECONDS = _positive_int("GENOLY_UPLOAD_TTL_HOURS", 168) * 3600


def check_upload_size(current: int, incoming: int = 0) -> None:
    """
    Lanza 413 si ``current + incoming`` supera :data:`MAX_UPLOAD_BYTES`.

    Args:
        current: Bytes ya escritos del archivo.
        incoming: Bytes que se van a añadir en esta petición.
    """
    if not MAX_UPLOAD_BYTES:
        return
    if current + incoming > MAX_UPLOAD_BYTES:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=413,
            detail=f"El archivo supera el límite de "
                   f"{MAX_UPLOAD_BYTES // _GIB} GiB "
                   f"(GENOLY_MAX_UPLOAD_BYTES)")


def _age(timestamp: Optional[float], path: Optional[Path]) -> float:
    """Edad en segundos de un elemento (preferencia por el sello propio)."""
    now = time.time()
    if timestamp:
        return max(0.0, now - timestamp)
    if path is not None:
        try:
            return max(0.0, now - path.stat().st_mtime)
        except OSError:
            return 0.0
    return 0.0


def _expired(timestamp: Optional[float], path: Optional[Path]) -> bool:
    if not UPLOAD_TTL_SECONDS:
        return False
    return _age(timestamp, path) > UPLOAD_TTL_SECONDS


def _cleanup_uploads() -> int:
    """Borra subidas caducadas (solo las gestionadas por la aplicación)."""
    from ui.backend import uploads

    removed = 0
    registry = uploads.all_uploads()
    for upload_id, meta in list(registry.items()):
        path = Path(meta.get("path", ""))
        created = meta.get("created_at")
        owned = path.parent == uploads.UPLOAD_DIR
        if not _expired(created, path if path.exists() else None):
            continue
        if owned:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("No se pudo borrar %s: %s", path, exc)
                continue
            removed += 1
        uploads.delete_upload(upload_id)

    # Ficheros huérfanos: cargas interrumpidas sin entrada de registro.
    for path in uploads.UPLOAD_DIR.iterdir():
        if not path.is_file():
            continue
        name = path.name
        if name in ("registry.json", "registry.json.lock"):
            continue
        if name.endswith(".tmp") or ".corrupt-" in name:
            if _expired(None, path):
                path.unlink(missing_ok=True)
                removed += 1
            continue
        if any(p["path"] == str(path) for p in registry.values()):
            continue
        if _expired(None, path):
            try:
                path.unlink(missing_ok=True)
                removed += 1
            except OSError as exc:
                logger.warning("No se pudo borrar %s: %s", path, exc)
    return removed


def _cleanup_datasets() -> int:
    """Borra extracciones de datasets caducadas y su registro."""
    from ui.backend import datasets

    removed = 0
    registry = datasets.all_datasets()
    for dataset_id, meta in list(registry.items()):
        kind = meta.get("kind")
        if kind != "zip":
            continue  # un directorio registrado es datos del usuario
        extract_dir = datasets._DATASETS_DIR / "extract" / dataset_id
        created = meta.get("created_at")
        if not _expired(created, extract_dir if extract_dir.exists() else None):
            continue
        if extract_dir.exists():
            import shutil
            try:
                shutil.rmtree(extract_dir, ignore_errors=True)
            except OSError as exc:  # pragma: no cover
                logger.warning("No se pudo borrar %s: %s", extract_dir, exc)
                continue
            removed += 1
        datasets.delete_dataset(dataset_id)

    extract_root = datasets._DATASETS_DIR / "extract"
    if extract_root.is_dir():
        known = {d for d in registry}
        for child in extract_root.iterdir():
            if not child.is_dir() or child.name in known:
                continue
            if _expired(None, child):
                import shutil
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
    return removed


def run_cleanup() -> int:
    """
    Aplica el TTL a subidas y datasets.

    Returns:
        Número de elementos eliminados.
    """
    if not UPLOAD_TTL_SECONDS:
        return 0
    total = 0
    try:
        total += _cleanup_uploads()
    except Exception as exc:  # pragma: no cover - nunca tumbar el servidor
        logger.warning("Limpieza de subidas fallida: %s", exc)
    try:
        total += _cleanup_datasets()
    except Exception as exc:  # pragma: no cover
        logger.warning("Limpieza de datasets fallida: %s", exc)
    return total
