from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from Genoly.core.device import DeviceManager
from Genoly.encoding.encoder import SequenceEncoder

#: Offset Phred+33 (Sanger / Illumina 1.8+): ASCII de la calidad menos 33.
PHRED_OFFSET = 33

#: Calidad asignada a las lecturas sin calidades (``None`` o ``*`` en SAM):
#: el más alto posible para que ninguna base se descarte por calidad, ya que
#: "calidad no disponible" no es lo mismo que "calidad mala".
PHRED_MISSING = 93


@dataclass
class Read:
    """Lectura alineada contra una referencia."""
    sequence: str
    start: int              # Posición 0-based en la referencia
    strand: str = '+'       # '+' o '-'


@dataclass
class Variant:
    """Variante llamada entre lecturas y la referencia."""
    position: int           # Posición 1-based en la referencia
    ref: str
    alt: str
    type: str               # 'SNV' o 'DEL'
    depth: int
    alt_count: int
    freq: float
    bases: Dict[str, int] = field(default_factory=dict)
    #: Base de la referencia inmediatamente anterior a la deleción; permite
    #: escribir la representación anclada que exige el VCF 4.2. ``None``
    #: cuando la deleción empieza en la primera base de la secuencia.
    anchor: Optional[str] = None


@dataclass
class PileupResult:
    """Resultado del pileup: cobertura y conteos de bases por posición."""
    reference: str
    depth: torch.Tensor
    base_counts: torch.Tensor    # (L, 4) en orden A,C,G,T
    consensus: str


class VariantCaller:
    """
    Pileup y llamada de variantes acelerado por GPU.

    Acumula los conteos de bases de todas las lecturas sobre cada posición
    de la referencia usando operaciones de dispersión (scatter) sobre CUDA,
    y llama variantes comparando el consenso con la referencia.
    """

    def __init__(self, device: Optional[str] = None,
                 min_base_quality: Optional[int] = None):
        """
        Args:
            device: 'cuda', 'cpu' o None para auto-detectar.
            min_base_quality: Score Phred mínimo por base; las bases por
                              debajo se ignoran en el pileup.
        """
        self.manager = DeviceManager(device)
        self.device = self.manager.device
        self.encoder = SequenceEncoder(device)
        self.min_base_quality = min_base_quality

    @staticmethod
    def reverse_complement(sequence: str) -> str:
        """Reverse complement de una secuencia."""
        complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A',
                      'N': 'N', 'U': 'A'}
        return ''.join(complement.get(b, 'N') for b in reversed(sequence.upper()))

    def _quality_scores(self, qualities: List[Optional[str]], batch: int,
                        max_len: int) -> torch.Tensor:
        """Matriz ``(batch, max_len)`` de calidades Phred en el dispositivo.

        Las lecturas sin calidades (``None`` o cadena vacía) se rellenan con
        :data:`PHRED_MISSING` para que el filtro por calidad las conserve en
        lugar de descartarlas.
        """
        qual = torch.full((batch, max_len), PHRED_MISSING, dtype=torch.long,
                          device=self.device)
        for i, q in enumerate(qualities[:batch]):
            if not q:
                continue
            row = torch.tensor([ord(c) - PHRED_OFFSET for c in q],
                               dtype=torch.long, device=self.device)
            n = min(row.numel(), max_len)
            qual[i, :n] = row[:n]
        return qual

    # ------------------------------------------------------------------ #
    # Pileup en GPU
    # ------------------------------------------------------------------ #
    def pileup(self, reference: str,
               reads: List[Read],
               qualities: Optional[List[str]] = None) -> PileupResult:
        """
        Calcula la cobertura y los conteos de bases por posición.

        Args:
            reference: Secuencia de referencia.
            reads: Lecturas alineadas (secuencia + posición 0-based + strand).
            qualities: Cadenas de calidad Phred opcionales, una por lectura,
                       para filtrar bases de baja calidad.

        Returns:
            PileupResult con profundidad y conteos de bases.
        """
        if not reads:
            return PileupResult(reference, torch.zeros(0, device=self.device),
                                torch.zeros((0, 4), device=self.device), '')

        ref_len = len(reference)

        # Normalizar lecturas según la hebra
        sequences = []
        for read in reads:
            seq = read.sequence
            if read.strand == '-':
                seq = self.reverse_complement(seq)
            sequences.append(seq)

        encoded, lengths = self.encoder.encode(sequences)
        b, max_len = encoded.shape

        # Posición en la referencia de cada base
        positions = torch.arange(max_len, device=self.device).unsqueeze(0) \
            + torch.tensor([r.start for r in reads], device=self.device).unsqueeze(1)

        # Máscara válida: base canónica, posición dentro de la referencia y
        # dentro de la longitud real de la lectura
        valid = encoded < 4
        valid &= (positions >= 0) & (positions < ref_len)
        valid &= torch.arange(max_len, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)

        # Filtro por calidad de base
        if self.min_base_quality is not None and qualities is not None:
            qual = self._quality_scores(qualities, b, max_len)
            valid &= qual >= self.min_base_quality

        # Acumular conteos de bases (A=0, C=1, G=2, T=3) con scatter
        counts = torch.zeros((ref_len, 4), dtype=torch.float32, device=self.device)
        if valid.any():
            flat_pos = positions[valid]
            flat_base = encoded[valid]
            one_hot = F.one_hot(flat_base, num_classes=4).float()
            counts.index_add_(0, flat_pos, one_hot)

        depth = counts.sum(dim=1).long()

        # Consenso por posición
        consensus_chars = []
        max_counts, max_bases = counts.max(dim=1)
        bases = ['A', 'C', 'G', 'T']
        for i in range(ref_len):
            if depth[i].item() > 0:
                consensus_chars.append(bases[int(max_bases[i].item())])
            else:
                consensus_chars.append('N')
        consensus = ''.join(consensus_chars)

        return PileupResult(reference, depth, counts, consensus)

    # ------------------------------------------------------------------ #
    # Llamada de variantes
    # ------------------------------------------------------------------ #
    def call_variants(self, reference: str,
                      reads: List[Read],
                      qualities: Optional[List[str]] = None,
                      min_depth: int = 10,
                      min_alt_freq: float = 0.2) -> List[Variant]:
        """
        Llama variantes comparando el consenso de lecturas con la referencia.

        Args:
            reference: Secuencia de referencia.
            reads: Lecturas alineadas.
            qualities: Calidades Phred opcionales por lectura.
            min_depth: Cobertura mínima para considerar una posición.
            min_alt_freq: Frecuencia mínima del alelo alternativo (0-1).

        Returns:
            Lista de variantes (SNV y DEL).
        """
        pile = self.pileup(reference, reads, qualities)
        variants = []

        counts = pile.base_counts
        depth = pile.depth

        # SNVs: consenso != referencia con cobertura suficiente
        for i, (ref_base, cons_base) in enumerate(zip(reference, pile.consensus)):
            pos = i + 1  # coordenadas 1-based en la salida
            ref_base = ref_base.upper()
            if ref_base == 'N':
                continue
            d = depth[i].item()
            if d < min_depth:
                continue
            if cons_base == ref_base or cons_base == 'N':
                continue

            base_counts = {
                b: int(counts[i, j].item())
                for j, b in enumerate(['A', 'C', 'G', 'T'])
            }
            alt_count = base_counts[cons_base]
            freq = alt_count / d if d > 0 else 0.0
            if freq < min_alt_freq:
                continue

            variants.append(Variant(
                position=pos,
                ref=ref_base,
                alt=cons_base,
                type='SNV',
                depth=d,
                alt_count=alt_count,
                freq=freq,
                bases=base_counts,
            ))

        # Deleciones: regiones sin cobertura interior flanqueadas por lecturas
        variants.extend(self._call_deletions(reference, depth))

        variants.sort(key=lambda v: v.position)
        return variants

    def _call_deletions(self, reference: str, depth: torch.Tensor,
                        min_del_len: int = 2,
                        min_flank_depth: int = 2) -> List[Variant]:
        """
        Detecta deleciones: regiones de referencia completamente sin
        cobertura (depth == 0) flanqueadas por posiciones con lecturas.

        Es una heurística conservadora: solo se reporta una deleción cuando
        existe un tramo interior con cobertura nula de al menos `min_del_len`
        bases, con posiciones a ambos lados cubiertas.

        Implementado con la versión vectorizada (:meth:`_call_deletions_vec`),
        sin ``max_del_len`` para conservar el comportamiento histórico.
        """
        return self._call_deletions_vec(
            reference, depth, min_del_len=min_del_len,
            max_del_len=max(len(reference), 1),
            min_flank_depth=min_flank_depth)

    # ------------------------------------------------------------------ #
    # Llamada de variantes desde SAM/BAM (vectorizada, por regiones)
    # ------------------------------------------------------------------ #
    def _pileup_counts(self, reference: str, reads: List[Read],
                       qualities: Optional[List[str]] = None
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Conteos de bases (L, 4) y profundidad (L,) en GPU, vectorizado.

        Las lecturas deben venir en orientación de la referencia (forward):
        al leer de un SAM, ``seq`` ya está en la hebra forward y ``start``
        es la coordenada forward, así que no se re-orienta nada.
        """
        ref_len = len(reference)
        if not reads:
            return (torch.zeros((ref_len, 4), dtype=torch.float32,
                                device=self.device),
                    torch.zeros(ref_len, dtype=torch.long, device=self.device))

        sequences = [read.sequence for read in reads]

        encoded, lengths = self.encoder.encode(sequences)
        b, max_len = encoded.shape
        positions = torch.arange(max_len, device=self.device).unsqueeze(0) \
            + torch.tensor([r.start for r in reads],
                           device=self.device).unsqueeze(1)

        valid = encoded < 4
        valid &= (positions >= 0) & (positions < ref_len)
        valid &= torch.arange(max_len, device=self.device).unsqueeze(0) \
            < lengths.unsqueeze(1)

        if self.min_base_quality is not None and qualities is not None:
            qual = self._quality_scores(qualities, b, max_len)
            valid &= qual >= self.min_base_quality

        counts = torch.zeros((ref_len, 4), dtype=torch.float32,
                             device=self.device)
        if valid.any():
            flat_pos = positions[valid]
            flat_base = encoded[valid]
            one_hot = F.one_hot(flat_base, num_classes=4).float()
            counts.index_add_(0, flat_pos, one_hot)

        depth = counts.sum(dim=1).long()
        return counts, depth

    def _call_deletions_vec(self, reference: str, depth: torch.Tensor,
                            min_del_len: int = 2,
                            max_del_len: int = 1000,
                            min_flank_depth: int = 2) -> List[Variant]:
        """Deleciones (tramos con depth == 0 flanqueados por cobertura),
        vectorizado: detecta los runs de ceros con operaciones de shift.

        ``max_del_len`` acota la longitud: un hueco de cobertura enorme es
        casi siempre una región repetitiva mal mapeada, no una deleción
        real (una deleción biológica de miles de bases sin ninguna lectura
        en los flancos es improbable en un experimento con buena
        cobertura)."""
        zero = depth == 0
        if zero.numel() > len(reference):
            zero = zero[:len(reference)]
        n = zero.numel()
        if not zero.any():
            return []
        start = zero & ~torch.cat(
            [torch.zeros(1, dtype=torch.bool, device=zero.device), zero[:-1]])
        end = zero & ~torch.cat(
            [zero[1:], torch.zeros(1, dtype=torch.bool, device=zero.device)])
        variants = []
        for s, e in zip(start.nonzero().squeeze(1).tolist(),
                        end.nonzero().squeeze(1).tolist()):
            run_len = e - s + 1
            if run_len < min_del_len or run_len > max_del_len:
                continue
            left_ok = s == 0 or int(depth[s - 1].item()) >= min_flank_depth
            right_ok = e == n - 1 or int(depth[e + 1].item()) >= min_flank_depth
            if left_ok and right_ok:
                variants.append(Variant(position=s + 1, ref=reference[s:e + 1],
                                        alt='-', type='DEL', depth=0,
                                        alt_count=0, freq=1.0,
                                        anchor=reference[s - 1] if s > 0
                                        else None))
        return variants

    def _call_region(self, reference: str, reads: List[Read],
                     min_depth: int = 10, min_alt_freq: float = 0.2,
                     min_flank_depth: int = 2,
                     max_del_len: int = 1000,
                     qualities: Optional[List[str]] = None) -> List[Variant]:
        """Llama variantes de una región con detección vectorizada.

        ``qualities`` es opcional: una cadena Phred por lectura, en el mismo
        orden que ``reads``. Si el caller tiene ``min_base_quality`` definido,
        las bases por debajo de ese umbral se excluyen del pileup (si no se
        pasan calidades el filtro no puede aplicarse).
        """
        counts, depth = self._pileup_counts(reference, reads, qualities)
        length = len(reference)
        if length == 0:
            return []

        ref_codes = self.encoder.encode([reference])[0][0]
        base_names = ['A', 'C', 'G', 'T']
        variants: List[Variant] = []

        # Para cada posición, el alelo alternativo es la base NO-ref más
        # frecuente (llamada por frecuencia, no por consenso): así se
        # detectan heterocigotos donde el alelo ref sigue siendo el
        # mayoritario.
        for i in range(length):
            d = int(depth[i].item())
            if d < min_depth:
                continue
            ref_code = int(ref_codes[i].item())
            if ref_code >= 4:
                continue  # referencia con N en esa posición
            row = counts[i]
            # enmascara el alelo de referencia y toma el mayor de los otros
            masked = row.clone()
            masked[ref_code] = -1.0
            alt_code = int(torch.argmax(masked).item())
            alt_count = int(row[alt_code].item())
            freq = alt_count / d if d > 0 else 0.0
            if freq < min_alt_freq:
                continue
            variants.append(Variant(
                position=i + 1,
                ref=reference[i],
                alt=base_names[alt_code],
                type='SNV',
                depth=d,
                alt_count=alt_count,
                freq=freq,
                bases={b: int(row[j].item())
                       for j, b in enumerate(base_names)},
            ))
        variants.extend(self._call_deletions_vec(
            reference, depth, min_del_len=2, max_del_len=max_del_len,
            min_flank_depth=min_flank_depth))
        variants.sort(key=lambda v: v.position)
        return variants

    def call_variants_from_sam(self, sam_path, ref_path,
                               min_depth: int = 10,
                               min_alt_freq: float = 0.2,
                               region_size: int = 50_000_000,
                               on_progress=None) -> dict:
        """
        Llama variantes de un SAM/BAM contra una referencia FASTA.

        Procesa cada secuencia de la referencia por regiones de
        ``region_size`` bases (memoria/VRAM acotada) y devuelve las
        variantes agrupadas por secuencia.

        Args:
            sam_path: Ruta del SAM/BAM (pysam).
            ref_path: Ruta del FASTA de referencia.
            min_depth: Cobertura mínima.
            min_alt_freq: Frecuencia mínima del alelo alternativo.
            region_size: Bases por región (0 = secuencia completa).
            on_progress: Callback ``fn(info)`` por región.

        Returns:
            Dict con ``by_sequence`` (secuencia -> lista de Variant),
            ``sequences`` (resumen por secuencia), ``total_variants``,
            ``snvs`` y ``deletions``.
        """
        from Genoly.io.fasta import FastaReader
        from Genoly.io.sam import iter_sam

        refs: Dict[str, str] = {}
        for record in FastaReader(ref_path).records():
            refs[record.id] = record.sequence.upper()

        reads_by_ref: Dict[str, list] = {}
        for sr in iter_sam(sam_path):
            if not sr.mapped or sr.rname is None:
                continue
            # ``*`` significa "calidad no disponible" en SAM: no filtrar.
            qual = sr.qual if sr.qual and sr.qual != '*' else None
            reads_by_ref.setdefault(sr.rname, []).append(
                (sr.seq, sr.pos, sr.strand, qual))

        by_sequence: Dict[str, List[Variant]] = {}
        sequences_summary: List[dict] = []
        total_variants = 0
        snvs = 0
        deletions = 0

        for name, refseq in refs.items():
            recs = reads_by_ref.get(name)
            if not recs:
                sequences_summary.append(
                    {"name": name, "length": len(refseq), "reads": 0,
                     "variants": 0})
                continue

            length = len(refseq)
            step = region_size or length
            seq_variants: List[Variant] = []
            n_reads = 0

            # Ordenar por posición permite descartar con búsqueda binaria las
            # lecturas que no pueden solapar la región (antes se recorría la
            # lista completa por cada región -> O(lecturas x regiones)).
            recs.sort(key=lambda t: t[1])
            positions = [t[1] for t in recs]
            max_read_len = max((len(t[0]) for t in recs), default=0)

            for ri, r0 in enumerate(range(0, length, step)):
                r1 = min(r0 + step, length)
                lo = bisect_left(positions, r0 - max_read_len)
                hi = bisect_left(positions, r1)
                region_reads: List[Read] = []
                region_quals: List[Optional[str]] = []
                for (seq, pos, strand, qual) in recs[lo:hi]:
                    if pos + len(seq) > r0:
                        region_reads.append(Read(sequence=seq,
                                                 start=pos - r0,
                                                 strand=strand))
                        region_quals.append(qual)
                if not region_reads:
                    continue
                n_reads += len(region_reads)
                local = self._call_region(refseq[r0:r1], region_reads,
                                          min_depth, min_alt_freq,
                                          qualities=region_quals)
                for v in local:
                    v.position += r0
                seq_variants.extend(local)
                if on_progress is not None:
                    on_progress({"sequence": name, "region": ri + 1,
                                 "regions": len(range(0, length, step)),
                                 "variants": len(seq_variants)})

            if seq_variants:
                seq_variants.sort(key=lambda v: v.position)
                by_sequence[name] = seq_variants
                snvs += sum(1 for v in seq_variants if v.type == 'SNV')
                deletions += sum(1 for v in seq_variants if v.type == 'DEL')
                total_variants += len(seq_variants)
            sequences_summary.append(
                {"name": name, "length": length, "reads": n_reads,
                 "variants": len(seq_variants)})

        return {
            "by_sequence": by_sequence,
            "sequences": sequences_summary,
            "total_variants": total_variants,
            "snvs": snvs,
            "deletions": deletions,
        }