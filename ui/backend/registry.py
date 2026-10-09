"""
Registro JSON persistente compartido entre procesos.

El servidor FastAPI y los procesos trabajadores (``ui.backend.worker``,
lanzados con ``spawn``) leen y escriben el mismo ``registry.json``. Un
candado de hilos no protege ese fichero: cada proceso tiene el suyo, así que
dos escrituras simultáneas hacían lectura-modificación-escritura y se
perdían actualizaciones. Este módulo serializa el acceso con ``flock`` (un
candado de fichero operativo entre procesos) además del candado de hilos,
y escribe siempre de forma atómica (temporal + ``os.replace``).

Si el fichero está corrupto no se descarta en silencio: se aparta con un
sufijo de fecha para que no haya pérdida de metadatos silenciosa.
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:  # pragma: no cover - POSIX; en Windows se degrada a candado de hilo
    import fcntl
    _HAS_FCNTL = True
except ImportError:  # pragma: no cover
    fcntl = None
    _HAS_FCNTL = False

logger = logging.getLogger("genoly.ui.registry")


class JsonRegistry:
    """Registro JSON con candado entre procesos y escritura atómica."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._thread_lock = threading.Lock()
        self._lock_path = self.path.with_name(self.path.name + ".lock")

    # ------------------------------------------------------------------ #
    # Candados
    # ------------------------------------------------------------------ #
    def _acquire_file(self):
        """Abre y bloquea el candado de fichero (best-effort)."""
        if not _HAS_FCNTL:
            return None
        fh = open(self._lock_path, "a+")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        except OSError:  # pragma: no cover
            fh.close()
            return None
        return fh

    @staticmethod
    def _release_file(fh) -> None:
        if fh is None:
            return
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:  # pragma: no cover
            pass
        finally:
            fh.close()

    # ------------------------------------------------------------------ #
    # Lectura / escritura
    # ------------------------------------------------------------------ #
    def read(self) -> Dict[str, Any]:
        """Carga el registro. Un fichero corrupto se aparta y se devuelve {}."""
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            logger.warning("Registro %s ilegible (%s); se aparta el fichero",
                           self.path, exc)
            self._quarantine()
            return {}

    def _quarantine(self) -> None:
        """Renombra el registro corrupto para no perderlo silenciosamente."""
        try:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            self.path.replace(
                self.path.with_name(f"{self.path.name}.corrupt-{stamp}"))
        except OSError as exc:  # pragma: no cover
            logger.warning("No se pudo apartar el registro corrupto: %s", exc)

    def write(self, data: Dict[str, Any]) -> None:
        """Escribe el registro de forma atómica (sin bloquear)."""
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(self.path)

    def update(self, mutator: Callable[[Dict[str, Any]], Any]) -> Any:
        """
        Ejecuta ``mutator(datos)`` con el registro cargado y guarda el
        resultado. La lectura-modificación-escritura va entera dentro del
        candado de hilo y del candado de fichero, por lo que es atómica
        respecto a otros procesos trabajadores.

        Returns:
            Lo que devuelva ``mutator``.
        """
        with self._thread_lock:
            fh = self._acquire_file()
            try:
                data = self.read()
                result = mutator(data)
                self.write(data)
                return result
            finally:
                self._release_file(fh)

    def locked_read(self) -> Dict[str, Any]:
        """Lectura consistente bajo candado (frente a escrituras ajenas)."""
        with self._thread_lock:
            fh = self._acquire_file()
            try:
                return self.read()
            finally:
                self._release_file(fh)

    def get(self, key: str) -> Optional[dict]:
        """Entrada del registro (lectura bajo candado)."""
        return self.locked_read().get(key)

    def set(self, key: str, value: Any) -> None:
        """Añade o reemplaza una entrada bajo candado."""
        self.update(lambda data: data.__setitem__(key, value))

    def update_entry(self, key: str,
                     mutator: Callable[[dict], Any]) -> Any:
        """Modifica una entrada existente bajo candado (no-op si falta)."""
        def _apply(data: Dict[str, Any]) -> Any:
            entry = data.get(key)
            if entry is None:
                return None
            return mutator(entry)

        with self._thread_lock:
            fh = self._acquire_file()
            try:
                data = self.read()
                entry = data.get(key)
                if entry is None:
                    return None
                result = mutator(entry)
                self.write(data)
                return result
            finally:
                self._release_file(fh)
