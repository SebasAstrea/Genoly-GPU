"""Punto de entrada para `pip install -e .` y builds con la extensión C.

Toda la configuración del paquete vive en `pyproject.toml` (versión,
dependencias, clasificadores); este fichero añade la extensión NATIVA
opcional del PairHMM (``csrc/pairhmm.c``). Si no hay toolchain de C en
la máquina, la compilación se salta con un aviso: Genoly sigue
funcionando con los backends PyTorch/NumPy del PairHMM.
"""

import sys
import warnings

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class optional_build_ext(build_ext):
    """Compila las extensiones; si falla el toolchain, continúa sin
    ellas (el paquete funciona igualmente con los backends Python)."""

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as exc:  # pragma: no cover - depende del entorno
            warnings.warn(
                f"No se pudo compilar la extensión nativa {ext.name} "
                f"({exc}). Genoly usará los backends PyTorch/NumPy del "
                "PairHMM.", RuntimeWarning)


if sys.platform == 'win32':
    _c_args = ['/O2']
    _link_args = []
else:
    _c_args = ['-O3', '-std=c99']
    _link_args = ['-lm']

ext_modules = [
    Extension(
        'Genoly.gatk._pairhmm_c',
        sources=['csrc/pairhmm.c'],
        extra_compile_args=_c_args,
        extra_link_args=_link_args,
        optional=True,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={'build_ext': optional_build_ext},
)
