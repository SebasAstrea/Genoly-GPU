"""
Matriz de parentesco pedigree (A) y su inversa para PBLUP.

Construye la matriz de parentesco aditiva A a partir de un pedigrí
(individuos, padres, madres) y su inversa A⁻¹. El orden de aparición
se normaliza topológicamente para que cada individuo aparezca después
de sus padres. Para pedigrees muy grandes, sustituir la inversión
directa por la recurrencia de Meuwissen & Luo (1992) sin cambiar la
interfaz.
"""

from typing import List, Optional, Tuple

import torch

__all__ = ['PedigreeError', 'normalize_pedigree', 'build_a_matrix',
           'build_a_inverse']

_MISSING_TOKENS = {"", "0", "na", "nan", "null", "none", "-", "nd"}


class PedigreeError(ValueError):
    """Error de estructura del pedigrí (ciclos, IDs inválidos, etc.)."""


def _clean_id(value) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if s.lower() in _MISSING_TOKENS:
        return None
    return s


def normalize_pedigree(individuals: List, sires: List, dams: List
                       ) -> Tuple[List[str], List[Optional[str]],
                                  List[Optional[str]], dict]:
    """
    Valida y normaliza un pedigrí crudo.

    - Los tokens ausentes (``''``, ``'0'``, ``'na'``, …) se convierten en None.
    - Los IDs se ordenan topológicamente: cada individuo va después de sus
      padres si están listados.
    - Los padres no listados se dejan como None y se cuentan en
      ``info['unknown_parents']``.
    - Se rechazan: longitudes distintas, IDs vacíos/duplicados, auto-padre
      (sire==dam==el propio ID) y ciclos.

    Returns:
        (individuals, sires, dams, info) con IDs en orden topológico.
        ``info`` contiene ``n``, ``founders`` (índices sin padres listados)
        y ``unknown_parents`` (nº de referencias a padres no listados).

    Raises:
        PedigreeError: Si la estructura del pedigrí es inválida.
    """
    if len(individuals) != len(sires) or len(individuals) != len(dams):
        raise PedigreeError(
            "individuals, sires y dams deben tener la misma longitud")
    n = len(individuals)
    if n == 0:
        raise PedigreeError("El pedigrí está vacío")

    ids = [_clean_id(v) for v in individuals]
    for i, v in enumerate(ids):
        if v is None:
            raise PedigreeError(f"Individuo {i + 1} sin identificador")
    if len(set(ids)) != n:
        raise PedigreeError("Los identificadores del pedigrí deben ser únicos")

    sr = [_clean_id(v) for v in sires]
    dm = [_clean_id(v) for v in dams]

    id_set = set(ids)
    for i in range(n):
        if sr[i] == ids[i] or dm[i] == ids[i]:
            raise PedigreeError(
                f"{ids[i]} no puede ser su propio padre/madre")

    pos = {v: i for i, v in enumerate(ids)}
    unknown_refs = 0
    for i in range(n):
        if sr[i] is not None and sr[i] not in id_set:
            unknown_refs += 1
        if dm[i] is not None and dm[i] not in id_set:
            unknown_refs += 1

    # Orden topológico (Kahn): padres antes que hijos.
    children_of = {}
    indegree = [0] * n
    for i in range(n):
        for p in (sr[i], dm[i]):
            if p is not None and p in pos:
                children_of.setdefault(pos[p], []).append(i)
                indegree[i] += 1

    queue = [i for i in range(n) if indegree[i] == 0]
    order: List[int] = []
    while queue:
        i = queue.pop()
        order.append(i)
        for c in children_of.get(i, []):
            indegree[c] -= 1
            if indegree[c] == 0:
                queue.append(c)
    if len(order) != n:
        raise PedigreeError("El pedigrí contiene ciclos de parentesco")

    ids_o = [ids[i] for i in order]
    sr_o = [sr[i] for i in order]
    dm_o = [dm[i] for i in order]

    founders = [i for i in range(n) if sr_o[i] is None and dm_o[i] is None]
    info = {
        "n": n,
        "founders": founders,
        "unknown_parents": unknown_refs,
    }
    return ids_o, sr_o, dm_o, info


def build_a_matrix(individuals: List, sires: List, dams: List
                   ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Construye la matriz de parentesco aditiva A y el endogamia F.

    Fórmulas (modelo animal):
        A[i,i] = 1 + 0.5 * A[sire_i, dam_i]   (0 si alguno desconocido)
        A[i,j] = 0.5 * (A[sire_i, j] + A[dam_i, j])

    Returns:
        (A, F, info) donde A es (n,n) float64, F = diag(A) - 1 y
        ``info`` es el de ``normalize_pedigree`` más ``inbreeding``
        (lista de F por individuo).
    """
    ids, sr, dm, info = normalize_pedigree(individuals, sires, dams)
    n = info["n"]
    pos = {v: i for i, v in enumerate(ids)}

    A = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        si = pos.get(sr[i]) if sr[i] is not None else None
        di = pos.get(dm[i]) if dm[i] is not None else None
        # fila i con padres conocidos
        if si is not None:
            A[i, :] += 0.5 * A[si, :]
        if di is not None:
            A[i, :] += 0.5 * A[di, :]
        # diagonal
        if si is not None and di is not None:
            A[i, i] = 1.0 + 0.5 * A[si, di]
        else:
            A[i, i] = 1.0
        # simetría: copiar fila i a la columna i para los ya asignados
        A[:i, i] = A[i, :i].clone()

    F = torch.diagonal(A).clone() - 1.0
    info["inbreeding"] = F.tolist()
    info["individuals"] = ids
    return A, F, info


def build_a_inverse(individuals: List, sires: List, dams: List
                    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Inversa de A para el modelo animal.

    Construye A con ``build_a_matrix`` y la invierte directamente
    (``torch.linalg.inv``). Para pedigrees de campo (n hasta miles) el
    coste O(n³) es despreciable frente al resto del ajuste REML; si se
    necesitan pedigrees muy grandes, sustituir por la recurrencia
    triangular de Meuwissen & Luo (1992) sin cambiar la interfaz.

    Returns:
        (A_inv, F, info). Se verifica A @ A⁻¹ ≈ I en los tests.
    """
    A, F, info = build_a_matrix(individuals, sires, dams)
    Ainv = torch.linalg.inv(A)
    return Ainv, F, info
