"""Routers de la API.

Cada submódulo declara un ``APIRouter`` que ``ui.backend.main`` registra
con ``include_router``. Se importan como ``from ui.backend.routers import
qdata`` (import explícito de submódulo); no hace falta re-exportar nada
aquí.
"""
