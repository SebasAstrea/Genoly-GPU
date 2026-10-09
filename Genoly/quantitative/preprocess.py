"""
Carga y preprocesamiento de tablas de fenotipos y genotipos (CSV o Excel).

Replica en el núcleo el preprocesamiento que aplica la interfaz web antes
de ajustar un modelo: detección de cabecera y delimitador, conversión de
decimales con coma, eliminación de columnas no numéricas y filas sin
fenotipo, e imputación de dosis perdidas por media o moda del marcador.
"""

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

IMPUTE_METHODS = ("media", "moda")
_MISSING_TOKENS = {"", "na", "nan", "null", "none", "-", "nd"}
_DECIMAL_COMMA = re.compile(r"^-?\d+,\d+$")


@dataclass
class PreprocessReport:
    """Resumen de las operaciones de limpieza aplicadas a una tabla."""
    archivo: Optional[str] = None
    rows_read: int = 0
    header_detected: bool = False
    columns_total: int = 0
    dropped_columns: List[str] = field(default_factory=list)
    dropped_rows_no_phenotype: int = 0
    imputed_cells: int = 0
    impute_method: str = "media"
    final_rows: int = 0
    final_markers: int = 0


def _sniff_delimiter(line: str) -> str:
    best, best_count = ",", -1
    for candidate in (",", ";", "\t"):
        count = line.count(candidate)
        if count > best_count:
            best, best_count = candidate, count
    return best


def _to_number(cell: Optional[str]) -> Optional[float]:
    """Convierte una celda a número: None si falta, NaN si no es numérica."""
    if cell is None or cell.strip().lower() in _MISSING_TOKENS:
        return None
    s = cell.strip()
    if _DECIMAL_COMMA.match(s):
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _is_non_numeric(cell: Optional[str]) -> bool:
    v = _to_number(cell)
    return v is not None and v != v


def _effective_suffix(path: Path) -> str:
    """
    Extensión real del archivo para elegir el lector.

    Además de la extensión declarada, detecta XLSX por magic bytes
    (``PK\\x03\\x04``, cabecera ZIP/OPC): así un .xlsx guardado con la
    extensión equivocada (p. ej. ``.csv`` por la subida de la UI) no se
    descuadra al leerlo como texto.
    """
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return suffix
    try:
        with open(path, "rb") as fh:
            if fh.read(4) == b"PK\x03\x04":
                return ".xlsx"
    except OSError:
        pass
    return suffix


def load_grid(path, max_rows: Optional[int] = None) -> List[List[Optional[str]]]:
    """
    Lee un archivo CSV/TSV/TXT o Excel y devuelve una matriz de celdas.

    Lectura en streaming por líneas (o ``iter_rows`` en Excel), de modo
    que archivos grandes no se cargan completos en RAM salvo que se pida.
    Para CSV se detecta el delimitador (coma, punto y coma o tabulador);
    para Excel (.xlsx/.xls) se requiere el paquete opcional openpyxl.

    Args:
        path: Ruta del archivo de datos.
        max_rows: Límite de filas a leer (sin contar la cabecera). Útil
            para previsualizar archivos grandes.

    Returns:
        Matriz de cadenas (celdas vacías como cadena vacía).

    Raises:
        ValueError: Si la extensión no está soportada o el archivo está vacío.
        ImportError: Si se lee Excel sin openpyxl instalado.
    """
    p = Path(path)
    suffix = _effective_suffix(p)

    if suffix in (".xlsx", ".xls"):
        try:
            import openpyxl
        except ImportError as exc:
            raise ImportError(
                "Instala openpyxl para leer archivos de Excel: pip install openpyxl"
            ) from exc
        # File handle en vez de ruta: openpyxl valida la extensión del
        # nombre solo con rutas; con stream comprueba realmente el ZIP,
        # así un XLSX guardado como .csv se lee igual.
        with open(p, "rb") as fh:
            workbook = openpyxl.load_workbook(fh, read_only=True, data_only=True)
            sheet = workbook.active
            grid = []
            for row in sheet.iter_rows(values_only=True):
                grid.append([("" if cell is None else str(cell).strip())
                             for cell in row])
                if max_rows and len(grid) >= max_rows:
                    break
            workbook.close()
    elif suffix in (".csv", ".tsv", ".txt"):
        lines = []
        with open(p, "r", encoding="utf-8-sig", errors="replace") as fh:
            for line in fh:
                if line.strip():
                    lines.append(line)
                    if max_rows and len(lines) >= max_rows:
                        break
        if not lines:
            raise ValueError("El archivo no contiene datos")
        delimiter = _sniff_delimiter(lines[0])
        grid = [[cell.strip() for cell in row]
                for row in csv.reader(lines, delimiter=delimiter)]
    else:
        raise ValueError(
            f"Formato no soportado: '{suffix}'. Usa .csv, .tsv, .txt, .xlsx o .xls"
        )

    grid = [row for row in grid if any(str(c).strip() for c in row)]
    if not grid:
        raise ValueError("El archivo no contiene filas de datos")
    width = max(len(row) for row in grid)
    return [row + [""] * (width - len(row)) for row in grid]


def scan_grid_stats(path) -> dict:
    """
    Conteo rápido en streaming de un CSV/Excel (filas, columnas, cabecera).

    Pensado para las estadísticas de subida: no materializa la matriz.
    """
    p = Path(path)
    suffix = _effective_suffix(p)
    if suffix in (".xlsx", ".xls"):
        try:
            import openpyxl
        except ImportError:
            return {"rows": 0, "columns": 0, "header_detected": False}
        with open(p, "rb") as fh:
            workbook = openpyxl.load_workbook(fh, read_only=True)
            sheet = workbook.active
            rows = 0
            cols = 0
            for row in sheet.iter_rows(values_only=True):
                rows += 1
                cols = max(cols, len(row))
            workbook.close()
        return {"rows": max(0, rows - 1), "columns": cols,
                "header_detected": True}

    rows = 0
    max_cols = 0
    first = None
    delimiter = ","
    with open(p, "r", encoding="utf-8-sig", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            rows += 1
            if first is None:
                first = line
                delimiter = _sniff_delimiter(line)
            max_cols = max(max_cols, len(line.split(delimiter)))

    header_detected = False
    if first:
        cells = first.split(delimiter)
        header_detected = any(_is_non_numeric(c) for c in cells)
    return {
        "rows": max(0, rows - 1) if header_detected else rows,
        "columns": max_cols,
        "header_detected": header_detected,
    }


def profile_grid(grid: List[List[Optional[str]]]) -> dict:
    """
    Perfila una matriz de celdas: tipo y datos perdidos por columna.

    Args:
        grid: Matriz de celdas (cadenas), con cabecera o sin ella.

    Returns:
        Dict con ``header_detected``, ``column_names`` y ``columns``
        (por columna: índice, nombre, tipo, missing, total, missing_pct).
    """
    if not grid:
        return {"header_detected": False, "column_names": [],
                "columns": []}
    width = max(len(row) for row in grid)
    grid = [row + [""] * (width - len(row)) for row in grid]

    header_detected = any(_is_non_numeric(c) for c in grid[0])
    if header_detected:
        column_names = [
            str(c).strip() if str(c).strip() else f"col_{j + 1}"
            for j, c in enumerate(grid[0])
        ]
        body = grid[1:]
    else:
        column_names = [f"col_{j + 1}" for j in range(width)]
        body = grid

    columns = []
    for j in range(width):
        values = []
        for row in body:
            v = _to_number(row[j]) if j < len(row) else None
            values.append(v)
        total = len(values)
        missing = sum(1 for v in values if v is None)
        numeric = sum(1 for v in values if v is not None and v == v)
        if numeric == total:
            ctype = "numeric"
        elif numeric == 0:
            ctype = "text"
        else:
            ctype = "mixed"
        columns.append({
            "index": j,
            "name": column_names[j],
            "type": ctype,
            "missing": missing,
            "total": total,
            "missing_pct": round(missing / total * 100, 1) if total else 100.0,
        })
    return {
        "header_detected": header_detected,
        "column_names": column_names,
        "columns": columns,
    }


def _marker_qc(genotypes: List[List[float]],
               min_maf: float = 0.0,
               hwe_p: Optional[float] = None
               ) -> Tuple[List[int], List[int], List[int]]:
    """
    Filtros de calidad por marcador: MAF y equilibrio de Hardy-Weinberg.

    Args:
        genotypes: Matriz de dosis (n x m) ya imputada.
        min_maf: Frecuencia alélica menor mínima (0-0.5).
        hwe_p: Umbral de p-valor del test chi-cuadrado de HWE; los
            marcadores con p menor se descartan.

    Returns:
        Tupla (índices_conservados, índices_baja_maf, índices_hwe).
    """
    if min_maf < 0 or min_maf > 0.5:
        raise ValueError("min_maf debe estar en [0, 0.5]")
    m = len(genotypes[0])
    keep = []
    low_maf = []
    bad_hwe = []
    for j in range(m):
        col = [row[j] for row in genotypes]
        n = len(col)
        freq = sum(col) / (2 * n)
        maf = min(freq, 1.0 - freq)
        if maf < min_maf:
            low_maf.append(j)
            continue
        if hwe_p is not None:
            n11 = sum(1 for v in col if v >= 1.999)
            n01 = sum(1 for v in col if 0.999 < v < 1.999)
            n00 = n - n11 - n01
            p = (2 * n00 + n01) / (2 * n)
            exp00 = n * p * p
            exp01 = 2 * n * p * (1 - p)
            exp11 = n * (1 - p) ** 2
            chi2 = 0.0
            for o, e in zip((n00, n01, n11), (exp00, exp01, exp11)):
                if e > 1e-9:
                    chi2 += (o - e) ** 2 / e
            from scipy import stats as sps
            p_hwe = float(sps.chi2.sf(chi2, 1))
            if p_hwe < hwe_p:
                bad_hwe.append(j)
                continue
        keep.append(j)
    return keep, low_maf, bad_hwe


def clean_grid(grid: List[List[Optional[str]]],
               phenotype_col: int = 0,
               impute_method: str = "media",
               max_column_missingness: float = 1.0,
               min_individuals: int = 5,
               min_markers: int = 2,
               min_maf: float = 0.0,
               hwe_p: Optional[float] = None,
               ) -> Tuple[List[float], List[List[float]], List[str], dict]:
    """
    Limpieza robusta de una matriz sucia para los modelos cuantitativos.

    Aplica, en orden:
    1. Detección de cabecera y delimitador (ya hechos en ``load_grid``).
    2. Descarte de columnas no numéricas (salvo el fenotipo) o con
       proporción de datos perdidos mayor que ``max_column_missingness``.
    3. Descarte de filas sin fenotipo numérico.
    4. Imputación de dosis perdidas por media o moda del marcador.

    Args:
        grid: Matriz de celdas crudas.
        phenotype_col: Índice de la columna de fenotipo.
        impute_method: 'media' o 'moda'.
        max_column_missingness: Proporción máxima de celdas vacías por
            marcador (0-1) para conservarlo.
        min_individuals: Mínimo de individuos tras la limpieza.
        min_markers: Mínimo de marcadores útiles tras la limpieza.

    Returns:
        Tupla (fenotipos, genotipos, nombres_de_marcadores, reporte).
    """
    if impute_method not in IMPUTE_METHODS:
        raise ValueError(
            f"impute_method debe ser uno de: {', '.join(IMPUTE_METHODS)}")
    if not 0 <= max_column_missingness <= 1:
        raise ValueError("max_column_missingness debe estar en [0, 1]")

    grid = [list(row) for row in grid if any(str(c).strip() for c in row)]
    if not grid:
        raise ValueError("El archivo no contiene filas de datos")
    rows_read = len(grid)
    width = max(len(row) for row in grid)
    grid = [row + [""] * (width - len(row)) for row in grid]

    header_detected = any(_is_non_numeric(c) for c in grid[0])
    if header_detected:
        column_names = [
            str(c).strip() if str(c).strip() else f"col_{j + 1}"
            for j, c in enumerate(grid[0])
        ]
        grid = grid[1:]
    else:
        column_names = [f"col_{j + 1}" for j in range(width)]

    if not 0 <= phenotype_col < width:
        raise ValueError("Índice de columna de fenotipo inválido")

    body = [[_to_number(c) for c in row] for row in grid]

    kept_cols = []
    dropped_columns = []
    for j in range(width):
        if j == phenotype_col:
            kept_cols.append(j)
            continue
        values = [row[j] for row in body if row[j] is not None]
        n = len(values)
        missing_pct = 1.0 if not body else (len(body) - n) / len(body)
        all_numeric = all(v == v for v in values)
        if n > 0 and all_numeric and missing_pct <= max_column_missingness:
            kept_cols.append(j)
        else:
            if not all_numeric:
                reason = "columna no numérica"
            elif n == 0:
                reason = "sin datos"
            else:
                reason = f"datos perdidos {missing_pct:.0%}"
            dropped_columns.append({"name": column_names[j], "reason": reason})

    phenotypes: List[float] = []
    genotypes: List[List[Optional[float]]] = []
    dropped_rows_no_phenotype = 0
    for row in body:
        pheno = row[phenotype_col]
        if pheno is None or pheno != pheno:
            dropped_rows_no_phenotype += 1
            continue
        phenotypes.append(pheno)
        genotypes.append([row[j] for j in kept_cols if j != phenotype_col])

    if len(phenotypes) < min_individuals:
        raise ValueError(
            f"Tras la limpieza quedan {len(phenotypes)} individuos; "
            f"se necesitan al menos {min_individuals}")
    if len(kept_cols) - 1 < min_markers:
        raise ValueError(
            f"Tras la limpieza quedan {len(kept_cols) - 1} marcadores; "
            f"se necesitan al menos {min_markers}")

    imputed_cells = 0
    n_markers = len(genotypes[0])
    for j in range(n_markers):
        observed = [row[j] for row in genotypes if row[j] is not None]
        if not observed:
            continue
        if impute_method == "moda":
            fill = max(set(observed), key=observed.count)
        else:
            fill = sum(observed) / len(observed)
        for row in genotypes:
            if row[j] is None:
                row[j] = fill
                imputed_cells += 1

    # QC de marcadores: MAF y equilibrio de Hardy-Weinberg
    keep_idx, low_maf_idx, hwe_idx = _marker_qc(
        genotypes, min_maf=min_maf, hwe_p=hwe_p)
    dropped_maf = [column_names[kept_cols[k + 1]]
                   for k in low_maf_idx]
    dropped_hwe = [column_names[kept_cols[k + 1]]
                   for k in hwe_idx]
    # subconjunto de marcadores conservados
    kept_marker_positions = keep_idx
    genotypes = [[row[k] for k in kept_marker_positions]
                 for row in genotypes]
    n_markers = len(genotypes[0])
    markers = [column_names[kept_cols[k + 1]]
               for k in kept_marker_positions]

    if n_markers < min_markers:
        raise ValueError(
            f"Tras la limpieza quedan {n_markers} marcadores; "
            f"se necesitan al menos {min_markers}")

    report = {
        "rows_read": rows_read,
        "header_detected": header_detected,
        "columns_total": width,
        "dropped_columns": dropped_columns,
        "dropped_rows_no_phenotype": dropped_rows_no_phenotype,
        "imputed_cells": imputed_cells,
        "impute_method": impute_method,
        "final_rows": len(phenotypes),
        "final_markers": n_markers,
        "phenotype_column": column_names[phenotype_col],
        "dropped_by_maf": dropped_maf,
        "dropped_by_hwe": dropped_hwe,
    }
    return phenotypes, genotypes, markers, report


def preprocess_grid(grid: List[List[Optional[str]]],
                    impute_method: str = "media",
                    ) -> Tuple[List[float], List[List[float]], PreprocessReport]:
    """
    Limpia e imputa una matriz de celdas para los análisis cuantitativos.

    Pasos aplicados: detección de cabecera, eliminación de columnas no
    numéricas (salvo la primera, usada como fenotipo), descarte de filas sin
    fenotipo y rellenado de dosis perdidas con la media o la moda de cada
    marcador.

    Args:
        grid: Matriz de celdas crudas (cadenas).
        impute_method: 'media' o 'moda'.

    Returns:
        Tupla (fenotipos, genotipos_imputados, informe).

    Raises:
        ValueError: Si el método es inválido, quedan menos de 5 individuos
                    o menos de 2 marcadores útiles.
    """
    if impute_method not in IMPUTE_METHODS:
        raise ValueError(f"impute_method debe ser uno de: {', '.join(IMPUTE_METHODS)}")

    grid = [list(row) for row in grid]
    rows_read = len(grid)
    if not grid:
        raise ValueError("El archivo no contiene filas de datos")
    width = max(len(row) for row in grid)

    header_detected = any(_is_non_numeric(c) for c in grid[0])
    if header_detected:
        column_names = [
            str(c).strip() if str(c).strip() else f"col_{j + 1}"
            for j, c in enumerate(grid[0])
        ]
        grid = grid[1:]
    else:
        column_names = [f"col_{j + 1}" for j in range(width)]

    body = [[_to_number(c) for c in row] for row in grid]

    valid_columns: List[int] = []
    dropped_columns: List[str] = []
    for j in range(width):
        if j == 0:
            valid_columns.append(0)
            continue
        values = [row[j] for row in body if row[j] is not None]
        all_numeric = all(v == v for v in values)
        if values and all_numeric:
            valid_columns.append(j)
        else:
            dropped_columns.append(column_names[j])

    phenotypes: List[float] = []
    genotypes: List[List[Optional[float]]] = []
    dropped_rows = 0
    for row in body:
        pheno = row[0]
        if pheno is None or pheno != pheno:
            dropped_rows += 1
            continue
        phenotypes.append(pheno)
        genotypes.append([row[j] for j in valid_columns[1:]])

    if len(phenotypes) < 5:
        raise ValueError(
            f"Tras la limpieza quedan {len(phenotypes)} individuos; "
            "se necesitan al menos 5"
        )
    if len(valid_columns) < 3:
        raise ValueError(
            f"Tras la limpieza quedan {len(valid_columns) - 1} marcadores; "
            "se necesitan al menos 2"
        )

    imputed_cells = 0
    n_markers = len(genotypes[0])
    for j in range(n_markers):
        observed = [row[j] for row in genotypes if row[j] is not None]
        if not observed:
            continue
        if impute_method == "moda":
            fill = max(set(observed), key=observed.count)
        else:
            fill = sum(observed) / len(observed)
        for row in genotypes:
            if row[j] is None:
                row[j] = fill
                imputed_cells += 1

    report = PreprocessReport(
        rows_read=rows_read,
        header_detected=header_detected,
        columns_total=width,
        dropped_columns=dropped_columns,
        dropped_rows_no_phenotype=dropped_rows,
        imputed_cells=imputed_cells,
        impute_method=impute_method,
        final_rows=len(phenotypes),
        final_markers=n_markers,
    )
    return phenotypes, genotypes, report


def prepare_quantitative_data(path,
                              impute_method: str = "media",
                              ) -> Tuple[List[float], List[List[float]], PreprocessReport]:
    """
    Carga un archivo CSV/Excel, lo limpia y lo deja listo para los modelos.

    Args:
        path: Ruta del archivo (.csv, .tsv, .txt, .xlsx o .xls). Primera
              columna = fenotipo, resto = dosis alélicas por marcador.
        impute_method: 'media' o 'moda' para las dosis perdidas.

    Returns:
        Tupla (fenotipos, genotipos_imputados, PreprocessReport).
    """
    grid = load_grid(path)
    phenotypes, genotypes, report = preprocess_grid(grid, impute_method=impute_method)
    report.archivo = Path(path).name
    return phenotypes, genotypes, report
