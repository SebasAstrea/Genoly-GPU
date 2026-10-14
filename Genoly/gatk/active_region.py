"""
Detección de regiones activas (GATK HaplotypeCaller).

Port del filtrado previo de HaplotypeCaller: solo las regiones con
evidencia de variación (mismatches/indels por encima de umbral de
frecuencia y profundidad, o huecos de cobertura) pasan al ensamblaje y
al PairHMM, lo que reduce drásticamente el coste frente a genotipar todo
el genoma.

La evidencia se calcula con un pileup vectorizado en GPU (scatter sobre
tensores torch, igual que el resto de Genoly). Las lecturas se asumen en
orientación forward (convención SAM); los indels en las lecturas se
manifiestan como rastros de mismatches en el pileup sin huecos, lo que
dispara la actividad de la región igual que en GATK (donde el
``AlignmentStateMachine`` sí los contabiliza explícitamente).
"""

from typing import List, Optional, Sequence, Tuple

import torch

from Genoly.core.device import DeviceManager
from Genoly.encoding.encoder import SequenceEncoder
from Genoly.gatk.reads import PHRED_MISSING, PHRED_OFFSET, GATKRead


class ActiveRegionDetector:
    """Detecta ventanas con evidencia de variantes (estilo GATK)."""

    def __init__(self, window: int = 50, extension: int = 50,
                 min_depth: int = 3, min_af: float = 0.15,
                 min_alt_count: int = 2, min_base_quality: int = 10,
                 max_gap: int = 50, device: Optional[str] = None):
        """
        Args:
            window: tamaño de la ventana deslizante (GATK: 50).
            extension: padding alrededor de la ventana activa.
            min_depth: cobertura mínima para mirar una posición.
            min_af: frecuencia mínima del alelo no-ref para activar.
            min_alt_count: conteo mínimo del alelo no-ref.
            min_base_quality: bases por debajo no cuentan como evidencia.
            max_gap: longitud máxima de hueco de cobertura considerado
                deleción (por encima es región repetitiva, no activa).
            device: 'cuda', 'cpu' o None (auto).
        """
        self.window = max(1, window)
        self.extension = max(0, extension)
        self.min_depth = max(1, min_depth)
        self.min_af = min_af
        self.min_alt_count = max(1, min_alt_count)
        self.min_base_quality = min_base_quality
        self.max_gap = max_gap
        self.manager = DeviceManager(device)
        self.device = self.manager.device
        self.encoder = SequenceEncoder(device)

    def _pileup(self, reference: str, reads: Sequence[GATKRead]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Conteos de bases (L, 4) y profundidad (L,) en el dispositivo."""
        ref_len = len(reference)
        counts = torch.zeros((ref_len, 4), dtype=torch.float32,
                             device=self.device)
        if not reads:
            return counts, torch.zeros(ref_len, dtype=torch.long,
                                       device=self.device)

        sequences = [r.sequence for r in reads]
        encoded, lengths = self.encoder.encode(sequences)
        b, max_len = encoded.shape
        starts = torch.tensor([r.start for r in reads], device=self.device)
        positions = torch.arange(max_len, device=self.device).unsqueeze(0) \
            + starts.unsqueeze(1)

        valid = encoded < 4
        valid &= (positions >= 0) & (positions < ref_len)
        valid &= (torch.arange(max_len, device=self.device).unsqueeze(0)
                  < lengths.unsqueeze(1))

        if self.min_base_quality > 0 and any(r.qualities for r in reads):
            qual = torch.full((b, max_len), PHRED_MISSING, dtype=torch.long,
                              device=self.device)
            for i, r in enumerate(reads):
                if not r.qualities:
                    continue
                row = torch.tensor(
                    [ord(c) - PHRED_OFFSET for c in r.qualities],
                    dtype=torch.long, device=self.device)
                n = min(row.numel(), max_len)
                qual[i, :n] = row[:n]
            valid &= qual >= self.min_base_quality

        if bool(valid.any()):
            flat_pos = positions[valid]
            flat_base = encoded[valid]
            one_hot = torch.nn.functional.one_hot(
                flat_base, num_classes=4).float()
            counts.index_add_(0, flat_pos, one_hot)

        depth = counts.sum(dim=1).long()
        return counts, depth

    def detect(self, reference: str,
               reads: Sequence[GATKRead]) -> List[Tuple[int, int]]:
        """
        Devuelve intervalos ``[start, end)`` activos, fusionados y con
        padding de ``extension``, en coordenadas de ``reference``.
        """
        ref_len = len(reference)
        if ref_len == 0 or not reads:
            return []

        counts, depth = self._pileup(reference, reads)
        # encode devuelve (códigos (1, L), longitudes): tomar la fila 0
        ref_codes = self.encoder.encode([reference])[0][0][:ref_len]

        # evidencia por posición: alelo no-ref dominante con soporte
        total = counts.sum(dim=1)
        masked = counts.clone()
        ref_c = ref_codes.long()
        valid_ref = (ref_c >= 0) & (ref_c < 4)
        idx = torch.arange(ref_len, device=self.device)
        masked[idx[valid_ref], ref_c[valid_ref]] = -1.0
        alt_count, alt_base = masked.max(dim=1)
        enough = (total >= self.min_depth) & valid_ref \
            & (alt_count >= self.min_alt_count) \
            & (alt_count / total.clamp(min=1) >= self.min_af)

        # huecos de cobertura cortos flanqueados = deleciones
        zero = depth == 0
        interesting = enough.clone()
        if bool(zero.any()) and ref_len > 1:
            start = zero & ~torch.cat(
                [torch.zeros(1, dtype=torch.bool, device=self.device),
                 zero[:-1]])
            end = zero & ~torch.cat(
                [zero[1:],
                 torch.zeros(1, dtype=torch.bool, device=self.device)])
            left = torch.cat([torch.zeros(1, dtype=torch.bool,
                                          device=self.device), depth[:-1]])
            right = torch.cat([depth[1:],
                               torch.zeros(1, dtype=torch.bool,
                                           device=self.device)])
            for s, e in zip(start.nonzero().squeeze(1).tolist(),
                            end.nonzero().squeeze(1).tolist()):
                run = e - s + 1
                if 1 <= run <= self.max_gap \
                        and int(left[s]) >= self.min_depth \
                        and int(right[e]) >= self.min_depth:
                    interesting[s:e + 1] = True

        if not bool(interesting.any()):
            return []

        # ventanas con evidencia -> fusión + padding
        idx_hit = interesting.nonzero().squeeze(1)
        windows = set()
        for pos in idx_hit.tolist():
            w0 = (pos // self.window) * self.window
            windows.add(w0)
        intervals: List[Tuple[int, int]] = []
        for w0 in sorted(windows):
            a = max(0, w0 - self.extension)
            b = min(ref_len, w0 + self.window + self.extension)
            if intervals and a <= intervals[-1][1]:
                intervals[-1] = (intervals[-1][0], max(intervals[-1][1], b))
            else:
                intervals.append((a, b))
        return intervals


def active_regions(reads: Sequence[GATKRead], reference: str,
                   region: Optional[Tuple[int, int]] = None, *,
                   window: int = 50, extension: int = 50,
                   device: Optional[str] = None, **kwargs
                   ) -> List[Tuple[int, int]]:
    """
    Conveniencia: detecta regiones activas en una sola llamada.

    Args:
        reads: lecturas alineadas (coordenadas sobre ``reference``).
        reference: secuencia forward de la referencia.
        region: opcional ``(start, end)``; solo se devuelven intervalos
            que la solapan.
        window / extension / device / kwargs: parámetros del detector
            (``min_af``, ``min_alt_count``, ``min_depth``...).

    Returns:
        Lista de intervalos ``[start, end)`` activos y fusionados.
    """
    detector = ActiveRegionDetector(window=window, extension=extension,
                                    device=device, **kwargs)
    intervals = detector.detect(reference, reads)
    if region is None:
        return intervals
    start, end = region
    return [(a, b) for a, b in intervals if a < end and b > start]
