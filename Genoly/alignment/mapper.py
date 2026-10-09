"""
Mapeo de lecturas a un genoma de referencia (seed-and-extend).

Estrategia (estilo BWA/minimap2, simplificada):

1. **Índice de la referencia**: se recorren los FASTA de la referencia por
   bloques de disco y se indexan k-mers muestreados cada ``stride`` bases
   (rolling hash base-4). Los seeds se guardan en arrays ordenados
   (código, secuencia, posición) con búsqueda binaria, de modo que el
   consumo de memoria es O(bases / stride) en vez de O(bases).
2. **Seeding**: cada lectura (y su reverso complementario) aporta sus
   k-mers muestreados; los aciertos se agrupan por diagonal
   (``pos_ref - pos_read``) para encontrar la región candidata.
3. **Extensión**: la mejor diagonal se extiende con el alineador local
   nativo (parasail, SIMD) contra el tramo de referencia, produciendo el
   alineamiento, la coordenada, la hebra, la identidad y el CIGAR.

Los objetos de índice viven en memoria (secuencias de la referencia como
bytes): adecuado para referencias de hasta ~1 GB; para genomas completos
de varios GB, aumente ``stride`` o indexe un subconjunto (el índice de
minimizers con derrame a disco es una mejora posterior).
"""

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from Genoly.alignment.alignment import GPUSequenceAligner
from Genoly.io.fasta import FastaReader

#: Tabla byte -> valor base-4 (A=0, C=1, G=2, T=3; el resto -1).
_BASE_VAL = np.full(256, -1, dtype=np.int32)
for _i, _ch in enumerate("ACGT"):
    _BASE_VAL[ord(_ch)] = _i
    _BASE_VAL[ord(_ch.lower())] = _i


def reverse_complement(sequence: str) -> str:
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N', 'U': 'A'}
    return ''.join(complement.get(b, 'N') for b in reversed(sequence.upper()))


@dataclass
class RefSequence:
    """Secuencia de la referencia indexada (bytes en mayúsculas)."""
    name: str
    seq: bytes


class RefIndex:
    """Índice de seeds k-mer de la referencia (búsqueda binaria)."""

    def __init__(self, k: int, stride: int):
        self.k = k
        self.stride = stride
        self.sequences: List[RefSequence] = []
        self.codes: np.ndarray = np.empty(0, dtype=np.int64)
        self.seq_idxs: np.ndarray = np.empty(0, dtype=np.int32)
        self.positions: np.ndarray = np.empty(0, dtype=np.int32)

    def lookup(self, code: int) -> Iterator[Tuple[int, int]]:
        """Aciertos (seq_idx, pos) de un código k-mer, por búsqueda binaria."""
        i = int(np.searchsorted(self.codes, code))
        n = self.codes.shape[0]
        while i < n and self.codes[i] == code:
            yield int(self.seq_idxs[i]), int(self.positions[i])
            i += 1


class ReadMapper:
    """
    Mapea lecturas (FASTQ) contra una referencia FASTA con seed-and-extend.
    """

    def __init__(self, k: int = 15, stride: int = 1,
                 min_seeds: int = 2, overhang: int = 80,
                 max_seed_hits: int = 200_000, device: str = "cpu"):
        if not 1 <= k <= 31:
            raise ValueError("k debe estar entre 1 y 31")
        if stride < 1:
            raise ValueError("stride debe ser >= 1")
        self.k = k
        self.stride = stride
        self.min_seeds = min_seeds
        self.overhang = overhang
        self.max_seed_hits = max_seed_hits
        self.aligner = GPUSequenceAligner(device=device)
        self._mask = (1 << (2 * k)) - 1

    # ------------------------------------------------------------------ #
    # Índice de la referencia
    # ------------------------------------------------------------------ #
    def build_index(self, ref_path: str,
                    on_progress: Optional[Callable[[dict], None]] = None
                    ) -> RefIndex:
        """
        Construye el índice de seeds de la referencia en streaming.

        Args:
            ref_path: Ruta del FASTA de referencia.
            on_progress: Callback tras cada secuencia con
                ``{"sequences", "bases", "seeds"}``.

        Returns:
            :class:`RefIndex` con las secuencias (bytes) y los seeds
            ordenados por código.
        """
        index = RefIndex(self.k, self.stride)
        codes: List[int] = []
        seq_idxs: List[int] = []
        positions: List[int] = []
        total_bases = 0

        for sid, record in enumerate(FastaReader(ref_path).records()):
            seq = record.sequence.upper().encode()
            index.sequences.append(RefSequence(record.id, seq))
            n = len(seq)
            total_bases += n
            self._collect_seeds(seq, sid, codes, seq_idxs, positions)
            if on_progress is not None:
                on_progress({"sequences": sid + 1, "bases": total_bases,
                             "seeds": len(codes)})

        codes_arr = np.asarray(codes, dtype=np.int64)
        order = np.argsort(codes_arr, kind="stable")
        index.codes = codes_arr[order]
        index.seq_idxs = np.asarray(seq_idxs, dtype=np.int32)[order]
        index.positions = np.asarray(positions, dtype=np.int32)[order]
        return index

    def _collect_seeds(self, seq: bytes, sid: int,
                       codes: List[int], seq_idxs: List[int],
                       positions: List[int]) -> None:
        """K-mers muestreados cada ``stride`` posiciones (rolling hash)."""
        k = self.k
        mask = self._mask
        n = len(seq)
        if n < k:
            return
        val = _BASE_VAL
        bad = 0
        code = 0
        for j in range(k):
            v = int(val[seq[j]])
            if v < 0:
                bad += 1
                v = 0
            code = (code << 2) | v
        if bad == 0:
            codes.append(code)
            seq_idxs.append(sid)
            positions.append(0)
        for i in range(1, n - k + 1):
            v_out = int(val[seq[i - 1]])
            if v_out < 0:
                bad -= 1
                v_out = 0
            v_in = int(val[seq[i + k - 1]])
            if v_in < 0:
                bad += 1
                v_in = 0
            code = ((code << 2) | v_in) & mask
            if bad == 0 and i % self.stride == 0:
                codes.append(code)
                seq_idxs.append(sid)
                positions.append(i)

    # ------------------------------------------------------------------ #
    # Mapeo de una lectura
    # ------------------------------------------------------------------ #
    def _read_seeds(self, seq: bytes) -> List[Tuple[int, int]]:
        """K-mers muestreados de una lectura: (pos_read, código)."""
        k = self.k
        n = len(seq)
        if n < k:
            return []
        val = _BASE_VAL
        bad = 0
        code = 0
        seeds: List[Tuple[int, int]] = []
        for j in range(k):
            v = int(val[seq[j]])
            if v < 0:
                bad += 1
                v = 0
            code = (code << 2) | v
        if bad == 0:
            seeds.append((0, code))
        for i in range(1, n - k + 1):
            v_out = int(val[seq[i - 1]])
            if v_out < 0:
                bad -= 1
            v_in = int(val[seq[i + k - 1]])
            if v_in < 0:
                bad += 1
                v_in = 0
            code = ((code << 2) | v_in) & self._mask
            if bad == 0 and i % self.stride == 0:
                seeds.append((i, code))
        return seeds

    def map_read(self, read, index: RefIndex) -> "MappingHit":
        """
        Mapea una lectura contra el índice y devuelve el mejor acierto.

        Args:
            read: :class:`~Genoly.io.fastq.FastqRecord`.
            index: Índice de la referencia.

        Returns:
            :class:`MappingHit` (``mapped=False`` si no hay acierto).
        """
        unmapped = MappingHit(read_id=read.id, mapped=False)
        seq = read.sequence.upper().encode()
        if len(seq) < self.k:
            return unmapped

        best: Optional[MappingHit] = None
        for strand, query in (("+", seq), ("-", reverse_complement(read.sequence).encode())):
            seeds = self._read_seeds(query)
            if not seeds:
                continue
            # agrupar aciertos por diagonal (pos_ref - pos_read)
            counts: dict = {}
            total_hits = 0
            for rp, code in seeds:
                for sid, rpos in index.lookup(code):
                    diag = rpos - rp
                    counts[(sid, diag)] = counts.get((sid, diag), 0) + 1
                    total_hits += 1
                    if total_hits > self.max_seed_hits:
                        break
                if total_hits > self.max_seed_hits:
                    break
            if not counts:
                continue

            (sid, diag), n_seeds = max(counts.items(), key=lambda kv: kv[1])
            if n_seeds < self.min_seeds:
                continue

            hit = self._extend(query, index, sid, diag, read, strand, n_seeds)
            if hit is None:
                continue
            if best is None or hit.score > best.score:
                best = hit

        return best if best is not None else unmapped

    def _extend(self, query: bytes, index: RefIndex, sid: int, diag: int,
                read, strand: str, n_seeds: int) -> Optional["MappingHit"]:
        """Extiende la diagonal con alineamiento local (parasail)."""
        ref = index.sequences[sid].seq
        start = max(0, diag - self.overhang)
        end = min(len(ref), diag + len(query) + self.overhang)
        if end - start < self.k:
            return None
        ref_sub = ref[start:end].decode("ascii", errors="replace")
        aln = self.aligner.align_pair(query.decode("ascii", errors="replace"),
                                      ref_sub)
        if aln.alignment_length == 0:
            return None

        aq = aln.aligned_query
        at = aln.aligned_target
        n_ref = sum(1 for c in at if c != "-")
        if aln.ref_end is not None and n_ref > 0:
            # coordenada 0-based del inicio de la alineación en la referencia
            ref_start = start + aln.ref_end - (n_ref - 1)
        else:
            # fallback (ruta Python): desplazamiento por gaps iniciales
            i = 0
            while i < len(aq) and aq[i] == "-":
                i += 1
            ref_start = start + sum(1 for c in at[:i] if c != "-")

        # soft-clips: bases de la lectura no alineadas (CIGAR SAM válido)
        clip5 = clip3 = 0
        if aln.query_end is not None:
            query_consumed = sum(1 for c in aq if c != "-")
            q_start = aln.query_end - (query_consumed - 1)
            clip5 = max(0, q_start)
            clip3 = max(0, len(query) - 1 - aln.query_end)

        return MappingHit(
            read_id=read.id,
            ref_name=index.sequences[sid].name,
            ref_start=ref_start,
            strand=strand,
            score=round(aln.score, 1),
            identity_percent=round(aln.identity_percent, 2),
            cigar=aln.cigar_string,
            mapped=True,
            n_seeds=n_seeds,
            clip5=clip5,
            clip3=clip3,
        )

    def map_reads(self, reads: Iterable, index: RefIndex,
                  on_progress: Optional[Callable[[dict], None]] = None,
                  progress_every: int = 10_000) -> List["MappingHit"]:
        """
        Mapea un iterable de lecturas (progreso cada ``progress_every``).

        Returns:
            Lista de :class:`MappingHit`, en el mismo orden que las lecturas.
        """
        return [hit for _, hit in self.iter_mapped(
            reads, index, on_progress, progress_every)]

    def iter_mapped(self, reads: Iterable, index: RefIndex,
                    on_progress: Optional[Callable[[dict], None]] = None,
                    progress_every: int = 10_000
                    ) -> Iterator[Tuple["object", "MappingHit"]]:
        """
        Genera pares ``(lectura, hit)`` a medida que se mapea, para poder
        escribir el SAM en streaming sin guardar todas las lecturas.
        """
        done = 0
        for read in reads:
            hit = self.map_read(read, index)
            yield read, hit
            done += 1
            if on_progress is not None and done % progress_every == 0:
                on_progress({"reads": done})


@dataclass
class MappingHit:
    """Resultado del mapeo de una lectura."""
    read_id: str
    mapped: bool = False
    ref_name: Optional[str] = None
    ref_start: int = -1
    strand: Optional[str] = None
    score: float = 0.0
    identity_percent: float = 0.0
    cigar: Optional[str] = None
    n_seeds: int = 0
    #: Soft-clips (bases de la lectura no alineadas en cada extremo), para
    #: construir un CIGAR SAM válido cuya longitud coincida con la lectura.
    clip5: int = 0
    clip3: int = 0