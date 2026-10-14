"""
Lectura y escritura de archivos SAM (y BAM, vía pysam si está disponible).

- ``write_sam``: escribe el resultado del mapeo (hits + lecturas) como un
  SAM de texto en streaming.
- ``iter_sam``: recorre un SAM/BAM/gzip usando pysam (htslib) cuando está
  instalado; si no, un parser mínimo de SAM de texto.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Union

try:
    import pysam  # type: ignore
    _HAS_PYSAM = True
except ImportError:  # pragma: no cover
    pysam = None
    _HAS_PYSAM = False

CIGAR_CODES = {'M': 0, 'I': 1, 'D': 2, 'N': 3, 'S': 4, 'H': 5,
               'P': 6, '=': 7, 'X': 8}


@dataclass
class SamRead:
    """Registro SAM/BAM."""
    qname: str
    flag: int
    rname: Optional[str]
    pos: int             # 0-based
    mapq: int
    cigar: Optional[str]
    seq: str
    qual: Optional[str]
    pnext: int = -1      # posición 0-based del mate (-1 si no aplica)

    @property
    def mapped(self) -> bool:
        return self.rname is not None and not (self.flag & 4)

    @property
    def strand(self) -> str:
        return '-' if (self.flag & 16) else '+'


def reverse_complement(sequence: str) -> str:
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N', 'U': 'A'}
    return ''.join(complement.get(b, 'N') for b in reversed(sequence.upper()))


def write_sam(path: Union[str, Path], ref_seqs: List, pairs: Iterator,
              include_unmapped: bool = True) -> None:
    """
    Escribe un SAM de texto a partir de pares (lectura, hit de mapeo).

    En SAM, SEQ se almacena siempre en la hebra forward de la referencia:
    para las lecturas mapeadas en hebra '-' se escribe el reverse
    complement de la lectura (y su calidad invertida), de modo que la
    secuencia coincide base a base con la referencia en ``POS``.

    Args:
        path: Ruta de salida.
        ref_seqs: Secuencias de la referencia (objetos con ``.name`` y
            ``.seq`` bytes) para las líneas ``@SQ``.
        pairs: Iterable de tuplas ``(FastqRecord, MappingHit)``.
        include_unmapped: Incluir lecturas sin mapeo (FLAG 4).
    """
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("@HD\tVN:1.6\tSO:unsorted\n")
        for rs in ref_seqs:
            fh.write(f"@SQ\tSN:{rs.name}\tLN:{len(rs.seq)}\n")
        for read, hit in pairs:
            if hit.mapped:
                flag = 16 if hit.strand == "-" else 0
                pos = hit.ref_start + 1  # SAM es 1-based
                cigar = hit.cigar or "*"
                if cigar != "*" and (hit.clip5 or hit.clip3):
                    # soft-clips para que el CIGAR consuma toda la lectura
                    cigar = f"{hit.clip5}S{cigar}{hit.clip3}S"
                seq = read.sequence
                qual = read.quality
                if hit.strand == "-":
                    seq = reverse_complement(seq)
                    qual = qual[::-1] if qual else qual
                fh.write(f"{read.id}\t{flag}\t{hit.ref_name}\t{pos}\t60\t"
                         f"{cigar}\t*\t0\t0\t{seq}\t{qual}\n")
            elif include_unmapped:
                fh.write(f"{read.id}\t4\t*\t0\t0\t*\t*\t0\t0\t"
                         f"{read.sequence}\t{read.quality}\n")


def iter_sam(path: Union[str, Path]) -> Iterator[SamRead]:
    """Recorre un SAM/BAM (pysam) o un SAM de texto (fallback)."""
    if _HAS_PYSAM:
        with pysam.AlignmentFile(str(path), "r") as fh:
            for r in fh:
                yield SamRead(
                    qname=r.query_name,
                    flag=r.flag,
                    rname=r.reference_name,
                    pos=r.reference_start,
                    mapq=r.mapping_quality,
                    cigar=r.cigarstring,
                    seq=r.query_sequence or "",
                    qual=r.qual,
                    pnext=r.next_reference_start if r.is_paired else -1,
                )
        return

    # Parser mínimo de SAM de texto (sin pysam)
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("@"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 11:
                continue
            flag = int(parts[1])
            rname = None if parts[2] == "*" else parts[2]
            pos = int(parts[3]) - 1
            try:
                pnext = int(parts[8]) - 1 if parts[8] not in ("0", "*") else -1
            except (ValueError, IndexError):
                pnext = -1
            yield SamRead(
                qname=parts[0], flag=flag, rname=rname, pos=pos,
                mapq=int(parts[4]), cigar=None if parts[5] == "*" else parts[5],
                seq=parts[9], qual=parts[10], pnext=pnext,
            )