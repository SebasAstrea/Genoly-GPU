"""Configuración compartida de la suite de tests.

El repositorio no está instalado como paquete durante los tests, así que
la raíz del proyecto se añade al ``sys.path`` una sola vez aquí (antes
cada fichero repetía el mismo ``sys.path.insert``).
"""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
