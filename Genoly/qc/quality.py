from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from Genoly.core.device import DeviceManager
from Genoly.encoding.encoder import SequenceEncoder
from Genoly.io.fastq import FastqRecord


@dataclass
class QualityReport:
    """Resumen de calidad de un conjunto de lecturas."""
    num_reads: int
    total_bases: int
    mean_read_length: float
    mean_quality: float
    gc_content_percent: float
    base_composition: Dict[str, int] = field(default_factory=dict)
    quality_by_position: List[float] = field(default_factory=list)


class QualityAnalyzer:
    """
    Control de calidad y filtrado de lecturas acelerado por GPU.

    Calcula contenido GC, composición de bases y distribución de calidad
    Phred operando sobre tensores CUDA, y ofrece trimming/filtrado.
    """

    def __init__(self, device: Optional[str] = None,
                 phred_offset: int = 33):
        """
        Args:
            device: 'cuda', 'cpu' o None para auto-detectar.
            phred_offset: Offset Phred (33 para Sanger/Illumina 1.8+).
        """
        self.manager = DeviceManager(device)
        self.device = self.manager.device
        self.phred_offset = phred_offset
        self.encoder = SequenceEncoder(device)

        # Índices de bases canónicas
        self.idx_A = self.encoder.to_idx['A']
        self.idx_C = self.encoder.to_idx['C']
        self.idx_G = self.encoder.to_idx['G']
        self.idx_T = self.encoder.to_idx['T']
        self.idx_N = self.encoder.to_idx['N']

    # ------------------------------------------------------------------ #
    # GC content y composición (GPU)
    # ------------------------------------------------------------------ #
    def gc_content(self, sequences: List[str]) -> torch.Tensor:
        """
        Contenido GC por secuencia, calculado sobre GPU.

        Args:
            sequences: Lista de secuencias.

        Returns:
            Tensor (N,) con fracción GC en [0, 1] por secuencia.
        """
        encoded, lengths = self.encoder.encode(sequences)
        valid = encoded < self.idx_N  # solo A/C/G/T
        gc = ((encoded == self.idx_C) | (encoded == self.idx_G)) & valid

        gc_count = gc.sum(dim=1).float()
        total = valid.sum(dim=1).float()
        return torch.where(total > 0, gc_count / total, torch.zeros_like(total))

    def gc_content_percent(self, sequences: List[str]) -> torch.Tensor:
        """Contenido GC en porcentaje (0-100)."""
        return self.gc_content(sequences) * 100

    def base_composition(self, sequences: List[str]) -> Dict[str, int]:
        """
        Composición total de bases del conjunto de secuencias.

        Cuenta A, C, G y T por su índice; todo lo demás (N y los códigos
        IUPAC de ambigüedad R, Y, S, W, K, M, B, D, H, V) se acumula en
        ``N``. El relleno de las secuencias cortas no se cuenta.

        Args:
            sequences: Lista de secuencias.

        Returns:
            Diccionario base -> número de ocurrencias.
        """
        encoded, lengths = self.encoder.encode(sequences)
        # Fuera del relleno y de las bases canónicas -> clase N
        canonical = torch.where(encoded < self.idx_N, encoded,
                                torch.full_like(encoded, self.idx_N))
        positions = torch.arange(encoded.shape[1], device=self.device)
        real = positions.unsqueeze(0) < lengths.unsqueeze(1)
        flat = canonical[real].cpu()
        counts = torch.bincount(flat, minlength=5).tolist()
        bases = ['A', 'C', 'G', 'T', 'N']
        return {base: int(counts[i]) for i, base in enumerate(bases)}

    # ------------------------------------------------------------------ #
    # Calidad Phred
    # ------------------------------------------------------------------ #
    def quality_scores(self, records: List[FastqRecord]) -> torch.Tensor:
        """
        Convierte las cadenas de calidad ASCII de un lote a scores Phred.

        Args:
            records: Lista de lecturas FASTQ.

        Returns:
            Tensor (B, L) de scores enteros con padding a 0.
        """
        if not records:
            return torch.zeros((0, 0), dtype=torch.long, device=self.device)

        max_len = max(len(r.quality) for r in records)
        scores = torch.zeros((len(records), max_len), dtype=torch.long, device=self.device)

        for i, record in enumerate(records):
            row = torch.tensor(
                [ord(c) - self.phred_offset for c in record.quality],
                dtype=torch.long, device=self.device,
            )
            scores[i, :len(row)] = row
        return scores

    def quality_distribution(self, records: List[FastqRecord],
                             max_position: Optional[int] = None) -> List[float]:
        """
        Media de calidad Phred por posición de base.

        Args:
            records: Lista de lecturas FASTQ.
            max_position: Posición máxima a considerar.

        Returns:
            Lista con la media de calidad por posición.
        """
        if not records:
            return []

        max_len = max(len(r.quality) for r in records)
        if max_position:
            max_len = min(max_len, max_position)

        scores = self.quality_scores(records)[:, :max_len]
        nonzero = (scores > 0).float()
        means = scores.float().sum(dim=0) / nonzero.sum(dim=0).clamp(min=1)
        return means.cpu().tolist()

    # ------------------------------------------------------------------ #
    # Trimming y filtrado
    # ------------------------------------------------------------------ #
    def trim_by_quality(self, records: List[FastqRecord],
                        min_quality: int = 20,
                        window_size: int = 5) -> List[FastqRecord]:
        """
        Recorta las lecturas desde el extremo 3' usando ventana deslizante.

        El algoritmo elimina las posiciones desde el primer punto en el que
        la media de calidad de la ventana cae por debajo del umbral.

        Args:
            records: Lista de lecturas FASTQ.
            min_quality: Umbral mínimo de calidad media por ventana.
            window_size: Tamaño de la ventana deslizante.

        Returns:
            Lista de lecturas recortadas.
        """
        trimmed = []
        for record in records:
            scores = [ord(c) - self.phred_offset for c in record.quality]
            trim_pos = len(scores)

            for i in range(len(scores) - window_size + 1):
                window = scores[i:i + window_size]
                if sum(window) / len(window) < min_quality:
                    trim_pos = i
                    break

            if trim_pos > 0:
                trimmed.append(FastqRecord(
                    id=record.id,
                    sequence=record.sequence[:trim_pos],
                    quality=record.quality[:trim_pos],
                    plus=record.plus,
                ))
        return trimmed

    def filter_by_quality(self, records: List[FastqRecord],
                          min_mean_quality: float = 20.0,
                          min_length: int = 20,
                          max_n_ratio: float = 0.1) -> List[FastqRecord]:
        """
        Filtra lecturas por calidad media, longitud mínima y proporción de N.

        Args:
            records: Lista de lecturas FASTQ.
            min_mean_quality: Calidad media mínima requerida.
            min_length: Longitud mínima tras el recorte.
            max_n_ratio: Proporción máxima de bases N permitida (0-1).

        Returns:
            Lista de lecturas que superan los filtros.
        """
        kept = []
        for record in records:
            scores = [ord(c) - self.phred_offset for c in record.quality]
            if not scores:
                continue
            mean_q = sum(scores) / len(scores)
            if mean_q < min_mean_quality:
                continue
            if len(record.sequence) < min_length:
                continue
            n_ratio = record.sequence.upper().count('N') / len(record.sequence)
            if n_ratio > max_n_ratio:
                continue
            kept.append(record)
        return kept

    # ------------------------------------------------------------------ #
    # Reporte completo
    # ------------------------------------------------------------------ #
    def report(self, records: List[FastqRecord]) -> QualityReport:
        """
        Genera un reporte completo de calidad del conjunto de lecturas.

        Args:
            records: Lista de lecturas FASTQ.

        Returns:
            QualityReport con métricas agregadas.
        """
        if not records:
            return QualityReport(num_reads=0, total_bases=0, mean_read_length=0,
                                 mean_quality=0, gc_content_percent=0)

        sequences = [r.sequence for r in records]
        total_bases = sum(len(s) for s in sequences)

        composition = self.base_composition(sequences)
        # GC ponderado por bases: media de fracciones por lectura sesga el
        # resultado cuando las lecturas tienen longitudes distintas.
        acgt = sum(composition[b] for b in "ACGT")
        gc = ((composition["C"] + composition["G"]) / acgt * 100.0
              if acgt else 0.0)

        all_scores = []
        for r in records:
            all_scores.extend(ord(c) - self.phred_offset for c in r.quality)
        mean_q = sum(all_scores) / len(all_scores) if all_scores else 0.0

        return QualityReport(
            num_reads=len(records),
            total_bases=total_bases,
            mean_read_length=total_bases / len(records),
            mean_quality=mean_q,
            gc_content_percent=gc,
            base_composition=composition,
            quality_by_position=self.quality_distribution(records),
        )

    def summarize(self, records: List[FastqRecord]) -> None:
        """Imprime un resumen legible del control de calidad."""
        report = self.report(records)
        print("=" * 60)
        print("REPORTE DE CONTROL DE CALIDAD")
        print("=" * 60)
        print(f"Lecturas: {report.num_reads}")
        print(f"Bases totales: {report.total_bases}")
        print(f"Longitud media: {report.mean_read_length:.1f} pb")
        print(f"Calidad media (Phred): {report.mean_quality:.1f}")
        print(f"Contenido GC: {report.gc_content_percent:.2f}%")
        print(f"Composición: {report.base_composition}")
        print("=" * 60)

    # ------------------------------------------------------------------ #
    # Análisis en streaming (archivos FASTQ de decenas de GB)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _scores_array(quality: str) -> np.ndarray:
        """Scores Phred (int32) de una cadena de calidad, sin padding."""
        if not quality:
            return np.zeros(0, dtype=np.int32)
        return (np.frombuffer(quality.encode("latin-1"), np.uint8)
                .astype(np.int32) - 33)

    @staticmethod
    def _trim_pos(scores: np.ndarray, min_quality: int,
                  window_size: int) -> int:
        """
        Posición de recorte 3' con ventana deslizante (vectorizado).

        Devuelve la primera posición donde la media de la ventana cae por
        debajo del umbral; 0 si hay que descartar la lectura; len(scores)
        si no hay ningún punto de recorte.
        """
        n = int(scores.shape[0])
        if n <= window_size:
            return n
        cs = np.cumsum(scores)
        win_sums = np.empty(n - window_size + 1, dtype=np.int64)
        win_sums[0] = int(cs[window_size - 1])
        win_sums[1:] = cs[window_size:] - cs[: -window_size]
        bad = np.flatnonzero(win_sums < min_quality * window_size)
        return int(bad[0]) if bad.size else n

    def trim_by_quality_batch(self, records: List[FastqRecord],
                              min_quality: int = 20,
                              window_size: int = 5) -> List[FastqRecord]:
        """
        Recorta las lecturas desde el extremo 3' (ventana deslizante),
        vectorizado con numpy (cumsum). Lecturas recortadas a 0 se
        descartan. Equivalente a :meth:`trim_by_quality`.
        """
        trimmed: List[FastqRecord] = []
        for record in records:
            scores = self._scores_array(record.quality)
            pos = self._trim_pos(scores, min_quality, window_size)
            if pos > 0:
                trimmed.append(FastqRecord(
                    id=record.id,
                    sequence=record.sequence[:pos],
                    quality=record.quality[:pos],
                    plus=record.plus,
                ))
        return trimmed

    def filter_by_quality_batch(self, records: List[FastqRecord],
                                min_mean_quality: float = 20.0,
                                min_length: int = 20,
                                max_n_ratio: float = 0.1
                                ) -> List[FastqRecord]:
        """Equivalente a :meth:`filter_by_quality`, vectorizado con numpy."""
        kept: List[FastqRecord] = []
        for record in records:
            scores = self._scores_array(record.quality)
            if scores.size == 0:
                continue
            if float(scores.mean()) < min_mean_quality:
                continue
            if len(record.sequence) < min_length:
                continue
            if record.sequence.upper().count("N") / len(record.sequence) \
                    > max_n_ratio:
                continue
            kept.append(record)
        return kept

    def analyze_stream(self, records: Iterator[FastqRecord],
                       batch_size: int = 4096,
                       max_position: int = 300,
                       histogram_bins: int = 100,
                       on_progress: Optional[Callable[[dict], None]] = None,
                       ) -> dict:
        """
        Control de calidad FastQC-like de un FASTQ en streaming.

        Recorre un iterable de lecturas por lotes de ``batch_size`` (RAM
        acotada) y agrega: lecturas, bases, longitud media/mínima/máxima,
        calidad media, contenido GC, composición, calidad media por
        posición (hasta ``max_position``) e histograma de longitudes.

        Args:
            records: Iterador perezoso de lecturas.
            batch_size: Lecturas por lote de RAM.
            max_position: Posiciones consideradas en la curva de calidad.
            histogram_bins: Nº de bins del histograma de longitudes.
            on_progress: Callback ``fn(info)`` tras cada lote con
                ``{"reads", "bases"}``.

        Returns:
            Dict con el reporte (num_reads, total_bases, mean_read_length,
            mean_quality, gc_content_percent, base_composition,
            quality_by_position, length_histogram, length_bin_size).
        """
        num_reads = 0
        total_bases = 0
        gc_bases = 0
        acgt_bases = 0
        composition = {base: 0 for base in "ACGTN"}
        qsum = np.zeros(max_position, dtype=np.float64)
        qcount = np.zeros(max_position, dtype=np.float64)
        q_total = 0.0
        q_n = 0
        lengths_sum = 0
        min_len = None
        max_len = 0
        # histograma de longitudes (bins de ancho fijo, cola en el último)
        bin_size = max(1, max_position // histogram_bins)
        hist = np.zeros(histogram_bins, dtype=np.int64)

        batch: List[FastqRecord] = []

        def flush() -> None:
            nonlocal num_reads, total_bases, gc_bases, acgt_bases
            nonlocal q_total, q_n, lengths_sum, min_len, max_len
            seqs = [r.sequence for r in batch]
            quals = [r.quality for r in batch]
            b_bases = sum(len(s) for s in seqs)
            num_reads += len(batch)
            total_bases += b_bases
            lengths_sum += b_bases

            comp = self.base_composition(seqs)
            for base in composition:
                composition[base] += comp[base]
            # GC ponderado por bases canónicas: la media de fracciones por
            # lectura (y usar bases totales, con N) sesga el resultado
            # cuando las lecturas tienen longitudes variables.
            acgt_bases += sum(comp[b] for b in "ACGT")
            gc_bases += comp["C"] + comp["G"]

            for q in quals:
                a = self._scores_array(q)
                if a.size == 0:
                    continue
                q_total += float(a.sum())
                q_n += int(a.size)
                cov = min(a.size, max_position)
                qsum[:cov] += a[:cov]
                qcount[:cov] += 1.0
                L = len(q)
                min_len = L if min_len is None else min(min_len, L)
                max_len = max(max_len, L)
                hist[min(L // bin_size, histogram_bins - 1)] += 1

            batch.clear()
            if on_progress is not None:
                on_progress({"reads": num_reads, "bases": total_bases})

        for record in records:
            batch.append(record)
            if len(batch) >= batch_size:
                flush()
        if batch:
            flush()

        if num_reads == 0:
            return {
                "num_reads": 0, "total_bases": 0, "mean_read_length": 0.0,
                "mean_quality": 0.0, "gc_content_percent": 0.0,
                "base_composition": {}, "quality_by_position": [],
                "length_histogram": [], "length_bin_size": bin_size,
            }

        # recortar posiciones sin cobertura de la curva de calidad
        covered = qcount > 0
        last = int(np.flatnonzero(covered).max()) + 1 if covered.any() else 0
        quality_by_position = [
            round(float(v), 2)
            for v in (qsum[:last] / qcount[:last].clip(min=1))
        ]

        return {
            "num_reads": num_reads,
            "total_bases": total_bases,
            "mean_read_length": round(lengths_sum / num_reads, 1),
            "min_length": min_len,
            "max_length": max_len,
            "mean_quality": round(q_total / q_n, 2) if q_n else 0.0,
            "gc_content_percent": round(
                gc_bases / acgt_bases * 100.0, 4) if acgt_bases else 0.0,
            "base_composition": composition,
            "quality_by_position": quality_by_position,
            "length_histogram": [int(v) for v in hist],
            "length_bin_size": bin_size,
        }

    def process_stream(self, records: Iterator[FastqRecord], out_fh,
                       min_quality: int = 20,
                       window_size: int = 5,
                       min_length: int = 20,
                       min_mean_quality: float = 20.0,
                       max_n_ratio: float = 0.1,
                       batch_size: int = 4096,
                       on_progress: Optional[Callable[[dict], None]] = None,
                       ) -> Tuple[int, int, int, int]:
        """
        Pipeline de preprocesamiento en streaming: trim por calidad (3'),
        filtro por calidad media/longitud/N, y escritura del FASTQ limpio.

        Args:
            records: Iterador perezoso de lecturas de entrada.
            out_fh: Fichero abierto en modo texto donde escribir el resultado.
            min_quality: Umbral de la ventana deslizante de recorte.
            window_size: Tamaño de la ventana de recorte.
            min_length: Longitud mínima tras el recorte.
            min_mean_quality: Calidad media mínima de la lectura.
            max_n_ratio: Proporción máxima de bases N (0-1).
            batch_size: Lecturas por lote de RAM.
            on_progress: Callback ``fn(info)`` con ``{"reads_in"}``.

        Returns:
            Tupla (lecturas_in, lecturas_out, bases_in, bases_out).
        """
        reads_in = 0
        reads_out = 0
        bases_in = 0
        bases_out = 0
        batch: List[FastqRecord] = []

        def flush() -> None:
            nonlocal reads_in, reads_out, bases_in, bases_out
            reads_in += len(batch)
            bases_in += sum(len(r) for r in batch)
            trimmed = self.trim_by_quality_batch(
                batch, min_quality, window_size)
            kept = self.filter_by_quality_batch(
                trimmed, min_mean_quality, min_length, max_n_ratio)
            for record in kept:
                out_fh.write(f"@{record.id}\n{record.sequence}\n"
                             f"{record.plus or '+'}\n{record.quality}\n")
                bases_out += len(record)
            reads_out += len(kept)
            batch.clear()
            if on_progress is not None:
                on_progress({"reads_in": reads_in})

        for record in records:
            batch.append(record)
            if len(batch) >= batch_size:
                flush()
        if batch:
            flush()
        return reads_in, reads_out, bases_in, bases_out