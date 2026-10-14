"""
Lecturas alineadas y filtros al estilo GATK (Read Filters).

Port del modelo de lectura y del conjunto de filtros que HaplotypeCaller
aplica antes de genotipar (MappingQualityAvailable, NotDuplicate,
NotSecondaryAlignment, NotSupplementaryAlignment, MinMappingQuality...).

La convención de coordenadas es la de SAM: ``start`` es 0-based en la
hebra forward de la referencia y ``sequence``/``qualities`` están siempre
en orientación forward (tal y como htslib los almacena), independiente-
mente de la hebra de mapeo.
"""

import re
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

#: Offset Phred+33 (Sanger / Illumina 1.8+).
PHRED_OFFSET = 33

#: Calidad asignada a las lecturas sin calidades (``None`` o ``*``).
PHRED_MISSING = 93

#: Clips al inicio del CIGAR (``2S6M...`` / ``2H6M...``).
_CIGAR_HEAD = re.compile(r'^(\d+)([SH])')

#: Clips al final del CIGAR (``...6M1H`` / ``...6M2S``).
_CIGAR_TAIL = re.compile(r'(\d+)([SH])$')

FLAG_PAIRED = 1
FLAG_PROPER_PAIR = 2
FLAG_UNMAPPED = 4
FLAG_REVERSE = 16
FLAG_MATE_REVERSE = 32
FLAG_SECONDARY = 256
FLAG_DUPLICATE = 512
FLAG_SUPPLEMENTARY = 2048


@dataclass
class GATKRead:
    """Lectura alineada con los campos del modelo GATK/SAM."""
    qname: str
    sequence: str
    start: int                       # 0-based, hebra forward
    strand: str = '+'                # '+' o '-'
    qualities: Optional[str] = None  # Phred+33 en orientación forward
    mapq: int = 60
    flag: int = 0
    cigar: Optional[str] = None
    rname: Optional[str] = None
    mpos: Optional[int] = None       # posición 0-based del mate (si aparea)

    @property
    def is_paired(self) -> bool:
        return bool(self.flag & FLAG_PAIRED)

    @property
    def is_proper_pair(self) -> bool:
        return bool(self.flag & FLAG_PROPER_PAIR)

    @property
    def is_unmapped(self) -> bool:
        return bool(self.flag & FLAG_UNMAPPED)

    @property
    def is_reverse(self) -> bool:
        return bool(self.flag & FLAG_REVERSE)

    @property
    def is_duplicate(self) -> bool:
        return bool(self.flag & FLAG_DUPLICATE)

    @property
    def is_secondary(self) -> bool:
        return bool(self.flag & FLAG_SECONDARY)

    @property
    def is_supplementary(self) -> bool:
        return bool(self.flag & FLAG_SUPPLEMENTARY)

    @property
    def mate_is_reverse(self) -> bool:
        return bool(self.flag & FLAG_MATE_REVERSE)

    def phred_scores(self) -> List[int]:
        """Calidades Phred numéricas (sin calidades -> ``PHRED_MISSING``)."""
        if not self.qualities:
            return [PHRED_MISSING] * len(self.sequence)
        return [ord(c) - PHRED_OFFSET for c in self.qualities]

    def _clip_ends(self) -> Tuple[int, int]:
        """``(clip5, clip3)`` de hard/soft clips según el CIGAR."""
        if not self.cigar or self.cigar == '*':
            return 0, 0
        clip5 = clip3 = 0
        m = _CIGAR_HEAD.match(self.cigar)
        if m:
            clip5 = int(m.group(1))
        m = _CIGAR_TAIL.search(self.cigar)
        if m:
            clip3 = int(m.group(1))
        return clip5, clip3

    def ref_span(self) -> int:
        """Bases de referencia consumidas por el CIGAR (M/D/N/=/X)."""
        if not self.cigar or self.cigar == '*':
            return len(self.sequence)
        span, num = 0, ''
        for ch in self.cigar:
            if ch.isdigit():
                num += ch
                continue
            if ch in 'MDN=X':
                span += int(num or 0)
            num = ''
        return span

    def unclipped_start(self) -> int:
        """Extremo 5' sin clips, en coordenadas forward de la referencia."""
        clip5, _ = self._clip_ends()
        return self.start - clip5

    def unclipped_end(self) -> int:
        """Extremo 3' sin clips (incluye clips del final)."""
        _, clip3 = self._clip_ends()
        return self.start + self.ref_span() + clip3

    def five_prime(self) -> int:
        """Coordenada forward del extremo 5' (regla de MarkDuplicates)."""
        if self.strand == '-':
            return self.unclipped_end()
        return self.unclipped_start()

    def overlaps(self, start: int, end: int) -> bool:
        """¿Solapa el intervalo ``[start, end)`` de la referencia?"""
        return self.start < end and self.start + len(self.sequence) > start


@dataclass
class FilterStats:
    """Resultado de :func:`filter_reads` con el desglose por filtro."""
    reads: List[GATKRead]
    total: int
    kept: int
    removed: Dict[str, int]


def filter_reads(reads: Sequence[GATKRead],
                 *,
                 min_mapq: int = 20,
                 drop_unmapped: bool = True,
                 drop_duplicates: bool = True,
                 drop_secondary: bool = True,
                 drop_supplementary: bool = True,
                 require_proper_pair: bool = False,
                 drop_missing_mate: bool = False) -> FilterStats:
    """
    Aplica el conjunto de filtros de lectura al estilo GATK.

    Se descartan, en este orden: no mapeadas, MAPQ por debajo de
    ``min_mapq`` (o MAPQ no disponible = 0 con ``min_mapq > 0``),
    secundarias, suplementarias, duplicados marcados y, opcionalmente,
    lecturas que no formen pareja propia o carezcan de mate.
    """
    kept: List[GATKRead] = []
    removed: Dict[str, int] = {}
    total = len(reads)

    def drop(reason: str) -> None:
        removed[reason] = removed.get(reason, 0) + 1

    for read in reads:
        if drop_unmapped and read.is_unmapped:
            drop('unmapped')
            continue
        if min_mapq > 0 and read.mapq < min_mapq:
            drop('low_mapq')
            continue
        if drop_secondary and read.is_secondary:
            drop('secondary')
            continue
        if drop_supplementary and read.is_supplementary:
            drop('supplementary')
            continue
        if drop_duplicates and read.is_duplicate:
            drop('duplicate')
            continue
        if require_proper_pair and read.is_paired and not read.is_proper_pair:
            drop('not_proper_pair')
            continue
        if drop_missing_mate and read.is_paired and read.mpos is None:
            drop('missing_mate')
            continue
        kept.append(read)

    return FilterStats(reads=kept, total=total, kept=len(kept),
                       removed=removed)


def reads_from_sam(path, rname: Optional[str] = None) -> List[GATKRead]:
    """
    Carga un SAM/BAM y devuelve lecturas alineadas como ``GATKRead``.

    Args:
        path: Ruta del SAM/BAM (pysam si está instalado; si no, SAM texto).
        rname: Si se indica, solo lecturas mapeadas contra esa secuencia.
    """
    from Genoly.io.sam import iter_sam

    out: List[GATKRead] = []
    for sr in iter_sam(path):
        if not sr.mapped:
            continue
        if rname is not None and sr.rname != rname:
            continue
        qual = sr.qual if sr.qual and sr.qual != '*' else None
        out.append(GATKRead(
            qname=sr.qname, sequence=sr.seq, start=sr.pos,
            strand=sr.strand, qualities=qual, mapq=sr.mapq,
            flag=sr.flag, cigar=sr.cigar, rname=sr.rname,
            mpos=sr.pnext if sr.pnext is not None and sr.pnext >= 0 else None,
        ))
    return out


def with_qualities(read: GATKRead, qualities: str) -> GATKRead:
    """Copia de la lectura con nuevas calidades (Phred+33)."""
    return replace(read, qualities=qualities)
