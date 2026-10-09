"""
Descarga datos genómicos reales para probar los módulos de Genoly-GPU.

Todo se deja en la carpeta local `datos_reales/` (ignorada por git):

    - Genomas FASTA de NCBI: E. coli K-12 MG1655 (NC_000913.3) y
      SARS-CoV-2 Wuhan-Hu-1 (NC_045512.2).
    - Lecturas FASTQ reales de Illumina del dataset de prueba de nf-core
      (SARS-CoV-2, parejas R1/R2).
    - Datos cuantitativos reales del panel de diversidad de maíz de GAPIT:
      fenotipos (mdp_traits.txt), genotipos numéricos (mdp_numeric.txt)
      y matriz de parentesco publicada (KSN.txt).

Uso:
    python descargar_datos_test.py
"""

import gzip
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Tuple

RAIZ = Path(__file__).resolve().parent
DESTINO = RAIZ / "datos_reales"
CUANTITATIVO = DESTINO / "cuantitativo"

URL_NCBI = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            "?db=nuccore&id={accession}&rettype=fasta&retmode=text")
URL_FASTQ = ("https://raw.githubusercontent.com/nf-core/test-datasets/modules/"
             "data/genomics/sarscov2/illumina/fastq/{lectura}.fastq.gz")
URL_GAPIT = "https://zzlab.net/GAPIT/GAPIT_Tutorial_Data.zip"

GENOMAS: List[Tuple[str, str]] = [
    ("e_coli_k12.fasta", URL_NCBI.format(accession="NC_000913.3")),
    ("sars_cov_2.fasta", URL_NCBI.format(accession="NC_045512.2")),
]
LECTURAS: List[Tuple[str, str]] = [
    ("sars_reads_R1.fastq", "test_1"),
    ("sars_reads_R2.fastq", "test_2"),
]
FICHEROS_CUANTITATIVOS: List[str] = ["mdp_traits.txt", "mdp_numeric.txt", "KSN.txt"]


def descargar(url: str, destino: Path) -> None:
    """Descarga una URL a un archivo local por bloques."""
    with urllib.request.urlopen(url, timeout=120) as respuesta:
        if getattr(respuesta, "status", 200) != 200:
            raise RuntimeError(f"HTTP {respuesta.status} en {url}")
        with open(destino, "wb") as fh:
            while True:
                bloque = respuesta.read(1 << 20)
                if not bloque:
                    break
                fh.write(bloque)


def validar_fasta(ruta: Path) -> int:
    """Comprueba que el FASTA descargado es válido y devuelve su longitud total."""
    cabecera_ok = False
    total = 0
    with open(ruta, encoding="utf-8") as fh:
        for linea in fh:
            linea = linea.strip()
            if linea.startswith(">"):
                cabecera_ok = True
            else:
                total += len(linea)
    if not cabecera_ok or total < 1000:
        raise ValueError(f"FASTA sospechoso: {ruta} ({total} pb)")
    return total


def descargar_fastq(nombre: str, lectura: str) -> int:
    """Descarga un FASTQ comprimido de nf-core, lo descomprime y borra el .gz."""
    ruta_gz = DESTINO / f"{lectura}.fastq.gz"
    ruta_fastq = DESTINO / nombre
    descargar(URL_FASTQ.format(lectura=lectura), ruta_gz)
    n_lecturas = 0
    with gzip.open(ruta_gz, "rt", encoding="utf-8") as origen, \
            open(ruta_fastq, "w", encoding="utf-8") as salida:
        for i, linea in enumerate(origen):
            salida.write(linea)
            if i % 4 == 0:
                n_lecturas += 1
    ruta_gz.unlink()
    return n_lecturas


def extraer_cuantitativos(zip_ruta: Path) -> None:
    """Extrae del zip de GAPIT solo los ficheros cuantitativos necesarios."""
    CUANTITATIVO.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_ruta) as zf:
        for nombre in FICHEROS_CUANTITATIVOS:
            miembro = f"GAPIT_Tutorial_Data/{nombre}"
            with zf.open(miembro) as origen, \
                    open(CUANTITATIVO / nombre, "wb") as salida:
                shutil.copyfileobj(origen, salida)


def main() -> int:
    """Punto de entrada: descarga y valida todos los datos reales."""
    DESTINO.mkdir(parents=True, exist_ok=True)
    print(f"Destino: {DESTINO}\n")

    print("== Genomas FASTA (NCBI) ==")
    for nombre, url in GENOMAS:
        ruta = DESTINO / nombre
        print(f"Descargando {nombre} ...")
        descargar(url, ruta)
        pb = validar_fasta(ruta)
        print(f"  OK: {pb:,} pb".replace(",", "."))

    print("\n== Lecturas FASTQ reales (nf-core test datasets) ==")
    for nombre, lectura in LECTURAS:
        n = descargar_fastq(nombre, lectura)
        print(f"  OK: {nombre} ({n} lecturas)")

    print("\n== Datos cuantitativos reales (panel de maíz GAPIT) ==")
    zip_ruta = DESTINO / "GAPIT_Tutorial_Data.zip"
    print("Descargando GAPIT_Tutorial_Data.zip ...")
    descargar(URL_GAPIT, zip_ruta)
    extraer_cuantitativos(zip_ruta)
    zip_ruta.unlink()
    for nombre in FICHEROS_CUANTITATIVOS:
        tam_kb = (CUANTITATIVO / nombre).stat().st_size / 1024
        print(f"  OK: cuantitativo/{nombre} ({tam_kb:.0f} KB)")

    print("\nTodos los datos se han descargado correctamente.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
