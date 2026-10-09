"""
Seguridad de la API: autenticación, CORS y rutas de datos permitidas.

Tres controles configurables por variable de entorno:

- ``GENOLY_API_KEY``: si está definida, TODA ruta ``/api/*`` exige la clave
  (cabecera ``X-API-Key``, ``Authorization: Bearer`` o, para EventSource y
  los enlaces ``<a download>`` que no pueden enviar cabeceras, el parámetro
  ``?api_key=``). Si no está definida la API queda abierta (modo local) y se
  registra un aviso al arrancar.

- ``GENOLY_CORS_ORIGINS``: orígenes permitidos en CORS, separados por comas.
  Por defecto solo el dev server de Vite; en producción el frontend se sirve
  desde el propio FastAPI (mismo origen) y no hace falta CORS.

- ``GENOLY_ALLOWED_DATA_DIRS``: directorios desde los que se pueden registrar
  rutas de datos del servidor (``/api/upload/register`` y
  ``/api/dataset/register``). Por defecto el propio directorio de subidas, el
  de datasets y la raíz del proyecto. Fuera de ahí se responde 403; hay
  además una lista negra de rutas sensibles del sistema que se bloquea
  siempre, aunque el allowlist las cubra.
"""

import hmac
import logging
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional

logger = logging.getLogger("genoly.ui.security")

#: Raíz del proyecto (para el allowlist por defecto de rutas de datos).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Rutas exentas de API key: la comprobación de salud que usa el
#: HEALTHCHECK de Docker no puede enviar cabeceras.
PUBLIC_PATHS = frozenset({"/api/health"})

#: Prefijos del sistema que nunca se pueden registrar ni descargar.
_FORBIDDEN_PREFIXES = (
    "/etc", "/proc", "/sys", "/dev", "/run", "/boot", "/root",
    "/var/log", "/var/spool", "/var/lib", "/usr", "/bin", "/sbin",
    "/lib", "/lib64", "/snap",
)

#: Directorios dentro del home que nunca se pueden registrar.
_FORBIDDEN_HOME_DIRS = (".ssh", ".aws", ".gnupg", ".kube", ".docker",
                        ".gnupg", ".password-store", ".local/share/Keyring")

#: Ficheros con secretos bloqueados por nombre, estén donde estén.
_SENSITIVE_NAME_RE = re.compile(
    r"(^|/)\.env(\..+)?$"
    r"|(^|/)id_(rsa|dsa|ecdsa|ed25519|ecdsa_sk|ed25519_sk)(\..+)?$"
    r"|(\.pem|\.key|\.p12|\.pfx|\.jks|\.keystore|\.crt|\.cer)$"
    r"|(^|/)(credentials|credentials\.json|\.netrc|\.npmrc|\.pypirc|"
    r"settings\.xml|authorized_keys|known_hosts|shadow|passwd|group|"
    r"htpasswd|id_rsa\.pub)$",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Autenticación
# --------------------------------------------------------------------------- #
def api_key() -> Optional[str]:
    """Clave de API configurada, o None si la API está en modo abierto."""
    key = os.environ.get("GENOLY_API_KEY", "").strip()
    return key or None


def extract_api_key(headers, query=None) -> Optional[str]:
    """
    Extrae la clave enviada por el cliente.

    Soporta cabecera ``X-API-Key``, cabecera ``Authorization: Bearer`` y
    parámetro de consulta ``api_key`` (necesario para ``EventSource`` y los
    enlaces de descarga, que no permiten cabeceras personalizadas).
    """
    value = headers.get("x-api-key")
    if value:
        return value.strip()
    auth = headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[7:].strip()
    if query is not None:
        q = query.get("api_key")
        if isinstance(q, str) and q:
            return q.strip()
    return None


def is_authorized(headers, query=None, path: str = "") -> bool:
    """True si la petición pasa la comprobación de API key."""
    expected = api_key()
    if expected is None:
        return True
    if path in PUBLIC_PATHS:
        return True
    provided = extract_api_key(headers, query)
    if provided is None:
        return False
    return hmac.compare_digest(provided.encode(), expected.encode())


# --------------------------------------------------------------------------- #
# CORS
# --------------------------------------------------------------------------- #
DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
)


def cors_origins() -> List[str]:
    """Orígenes CORS permitidos (``GENOLY_CORS_OGINS`` o los de Vite)."""
    raw = os.environ.get("GENOLY_CORS_ORIGINS", "").strip()
    if not raw:
        return list(DEFAULT_CORS_ORIGINS)
    return [o.strip() for o in raw.split(",") if o.strip()]


# --------------------------------------------------------------------------- #
# Rutas de datos del servidor
# --------------------------------------------------------------------------- #
def allowed_data_dirs(extra: Iterable[Path] = ()) -> List[Path]:
    """
    Directorios desde los que se pueden registrar rutas de datos.

    Prioridad: ``GENOLY_ALLOWED_DATA_DIRS`` (separada por comas) si está
    definida; en caso contrario los directorios de trabajo de la aplicación
    más la raíz del proyecto. ``extra`` permite añadir los directorios
    propios de cada módulo (subidas, datasets).
    """
    raw = os.environ.get("GENOLY_ALLOWED_DATA_DIRS", "").strip()
    if raw:
        dirs = [Path(p).expanduser() for p in raw.split(",") if p.strip()]
    else:
        dirs = [PROJECT_ROOT]
    dirs.extend(Path(p) for p in extra)
    seen, out = set(), []
    for d in dirs:
        try:
            resolved = d.resolve()
        except OSError:  # pragma: no cover - ruta no resoluble
            resolved = d.absolute()
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


def forbidden_reason(src: Path) -> Optional[str]:
    """Motivo por el que la ruta está bloqueada, o None si no lo está."""
    text = str(src)
    lowered = text.lower()
    for prefix in _FORBIDDEN_PREFIXES:
        if text == prefix or text.startswith(prefix + "/"):
            return f"la ruta queda dentro de {prefix}"
    home = Path.home()
    try:
        rel_home = src.relative_to(home)
    except ValueError:
        rel_home = None
    if rel_home is not None:
        parts = rel_home.parts
        for i in range(len(parts)):
            prefix = "/".join(parts[: i + 1])
            if prefix in _FORBIDDEN_HOME_DIRS:
                return f"la ruta queda dentro de ~/{prefix}"
    if _SENSITIVE_NAME_RE.search(text.replace("\\", "/")):
        return "el nombre del archivo sugiere que contiene credenciales"
    if lowered.endswith(".git/config") or "/.git/" in lowered:
        return "la ruta queda dentro de un repositorio .git"
    return None


def check_data_path(path: Path, extra: Iterable[Path] = ()) -> Path:
    """
    Valida una ruta de datos del servidor y devuelve la ruta resuelta.

    Exige que el archivo exista, que esté dentro de un directorio permitido
    y que no sea una ruta sensible del sistema.

    Raises:
        HTTPException: 400 si no existe, 403 si está fuera del allowlist o
            es una ruta sensible.
    """
    from fastapi import HTTPException

    src = path.expanduser()
    try:
        src = src.resolve()
    except OSError as exc:
        raise HTTPException(status_code=400,
                            detail=f"No se pudo resolver la ruta: {exc}")

    if not src.exists():
        raise HTTPException(status_code=400, detail="La ruta no existe")

    reason = forbidden_reason(src)
    if reason is not None:
        raise HTTPException(
            status_code=403,
            detail=f"Ruta no permitida: {reason}. "
                   "Configura GENOLY_ALLOWED_DATA_DIRS para ampliar los "
                   "directorios de datos.")

    allowed = allowed_data_dirs(extra)
    if not any(_is_within(src, base) for base in allowed):
        raise HTTPException(
            status_code=403,
            detail="La ruta está fuera de los directorios de datos "
                   "permitidos (GENOLY_ALLOWED_DATA_DIRS)")

    if src.is_dir():
        return src
    if not src.is_file():
        raise HTTPException(status_code=400,
                            detail="La ruta no es un archivo ni un directorio")
    return src


def _is_within(src: Path, base: Path) -> bool:
    """True si ``src`` es ``base`` o está dentro (con symlinks resueltos)."""
    try:
        return src == base or src.is_relative_to(base)
    except (ValueError, OSError):  # pragma: no cover
        return False


def warn_if_open() -> None:
    """Registra un aviso si la API queda sin autenticación."""
    if api_key() is None:
        logger.warning(
            "GENOLY_API_KEY no está definida: la API es accesible sin "
            "autenticación (modo local). Define GENOLY_API_KEY para exigir "
            "la cabecera X-API-Key en /api/*.")
