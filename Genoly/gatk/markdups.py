"""
MarkDuplicates al estilo GATK.

Port de la heurística de duplicados de GATK MarkDuplicates:

- Clave de un extremo: ``(referencia, hebra, posición 5' sin clips)``,
  con la posición 5' calculada sobre el CIGAR (soft/hard clips
  incluidos), igual que el ``optical duplicate`` key de GATK.
- Para lecturas apareadas con mate conocido, la clave es el par de
  extremos normalizado (orden independiente del extremo que llegue
  primero), como en el modelo de duplicados por pares de GATK.
- Puntuación de una lectura: nº de bases con calidad >= 15 (regla de
  GATK); el de mayor puntuación de cada clave no se marca, el resto sí.
"""

from typing import Dict, List, Optional, Set, Tuple

from Genoly.gatk.reads import FLAG_DUPLICATE, GATKRead

#: Calidad mínima de base para contar en la puntuación (GATK: 15).
MIN_SCORE_QUAL = 15

PairKey = Tuple[Tuple[str, str, int], ...]


def _end_key(read: GATKRead) -> Tuple[str, str, int]:
    return (read.rname or '*', read.strand, read.five_prime())


def duplicate_keys(read: GATKRead) -> Optional[PairKey]:
    """Clave del duplicado; ``None`` si la lectura no aporta información."""
    if read.is_unmapped or not read.sequence:
        return None
    end = _end_key(read)
    if read.is_paired and read.mpos is not None:
        mate_strand = '-' if read.mate_is_reverse else '+'
        mate = (read.rname or '*', mate_strand, read.mpos)
        return tuple(sorted([end, mate]))
    return (end,)


def read_score(read: GATKRead) -> int:
    """Puntuación GATK: bases con calidad >= 15."""
    return sum(1 for q in read.phred_scores() if q >= MIN_SCORE_QUAL)


def mark_duplicates(reads: List[GATKRead],
                    already_marked: bool = True) -> Dict[str, int]:
    """
    Marca duplicados y devuelve estadísticas.

    Returns:
        Dict con ``total``, ``duplicates``, ``duplicate_rate``,
        ``libraries`` (nº de claves) y ``qnames`` (los marcados).
    """
    best: Dict[PairKey, Tuple[int, str]] = {}
    for read in reads:
        if already_marked and read.is_duplicate:
            continue
        key = duplicate_keys(read)
        if key is None:
            continue
        score = read_score(read)
        current = best.get(key)
        if current is None or score > current[0]:
            best[key] = (score, read.qname)

    by_key: Dict[PairKey, List[GATKRead]] = {}
    for read in reads:
        if already_marked and read.is_duplicate:
            continue
        key = duplicate_keys(read)
        if key is None:
            continue
        by_key.setdefault(key, []).append(read)

    marked: Set[str] = set()
    for key, group in by_key.items():
        winner = best[key][1]
        for read in group:
            if read.qname != winner:
                marked.add(read.qname)

    total = len(reads)
    duplicates = len({r.qname for r in reads} & marked)
    return {
        "total": total,
        "duplicates": duplicates,
        "duplicate_rate": duplicates / total if total else 0.0,
        "libraries": len(by_key),
        "qnames": sorted(marked),
    }


def apply_duplicate_flags(reads: List[GATKRead],
                          marked: Optional[Set[str]] = None) -> List[GATKRead]:
    """
    Devuelve copias de las lecturas con el flag 0x400 (duplicado) activo
    para las lecturas marcadas (por defecto, las que indique
    :func:`mark_duplicates`).
    """
    import dataclasses

    if marked is None:
        marked = set(mark_duplicates(reads)["qnames"])
    out: List[GATKRead] = []
    for read in reads:
        if read.qname in marked and not read.is_duplicate:
            read = dataclasses.replace(read,
                                       flag=read.flag | FLAG_DUPLICATE)
        out.append(read)
    return out
