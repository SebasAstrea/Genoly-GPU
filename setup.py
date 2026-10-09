"""Punto de entrada legado para `pip install -e .`.

Toda la configuración del paquete vive en `pyproject.toml` (versión,
dependencias, clasificadores); este fichero solo queda para pip/setuptools
antiguos que aún invocan `setup.py`.
"""

from setuptools import setup

setup()
