"""
Base Quality Score Recalibration (BQSR) al estilo GATK.

Port del modelo de covariables de GATK BaseRecalibrator/ApplyBQSR:

- Se acumulan, por cada base alineada contra la referencia (columnas
  M/=/X del CIGAR; sin CIGAR se asume todo mapeado), los conteos
  (observaciones, errores) de cada covariable
  ``(calidad reportada, ciclo direccional, hebra)``.
- El ciclo direccional es 1..L en hebra forward y L..1 en reverse
  (``CycleCovariate`` de GATK).
- La calidad empírica de cada covariable se estima con suavizado
  bayesiano hacia la tasa global de error, y ApplyBQSR sustituye la
  calidad reportada por la empírica (entera, recortada a [0, 93]).

Los conteos se acumulan con ``index_add_`` sobre tensores torch, por lo
que la estimación corre en GPU cuando hay CUDA disponible.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from Genoly.core.device import DeviceManager
from Genoly.gatk.reads import PHRED_OFFSET, GATKRead

CovKey = Tuple[int, int, int]  # (calidad, ciclo, hebra)


@dataclass
class RecalTable:
    """Tabla de recalibración: covariable -> [observaciones, errores]."""
    counts: Dict[CovKey, List[int]] = field(default_factory=dict)
    n_bases: int = 0
    n_errors: int = 0
    max_cycle: int = 0

    @property
    def global_rate(self) -> float:
        """Tasa global de error (suavizada con pseudo-observaciones)."""
        return (self.n_errors + 1.0) / (self.n_bases + 2.0)

    def empirical_quality(self, qual: int, cycle: int, strand: int,
                          smoothing: float = 10.0) -> float:
        """Calidad empírica Phred de una covariable (GATK-style)."""
        obs, errs = self.counts.get((qual, cycle, strand), (0, 0))
        if obs == 0:
            return float(qual)
        rate = (errs + smoothing * self.global_rate) / (obs + smoothing)
        rate = min(max(rate, 1e-6), 0.5)
        return -10.0 * math.log10(rate)

    def to_dict(self) -> dict:
        return {
            "n_bases": self.n_bases,
            "n_errors": self.n_errors,
            "max_cycle": self.max_cycle,
            "global_rate": self.global_rate,
            "covariates": [
                {"qual": q, "cycle": c, "strand": s,
                 "observed": v[0], "mismatches": v[1]}
                for (q, c, s), v in sorted(self.counts.items())
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RecalTable":
        table = cls(n_bases=int(data.get("n_bases", 0)),
                    n_errors=int(data.get("n_errors", 0)),
                    max_cycle=int(data.get("max_cycle", 0)))
        for item in data.get("covariates", []):
            key = (int(item["qual"]), int(item["cycle"]), int(item["strand"]))
            table.counts[key] = [int(item["observed"]),
                                 int(item["mismatches"])]
        return table


def _match_columns(read: GATKRead) -> List[Tuple[int, int]]:
    """Pares ``(índice de base en la lectura, posición en la referencia)``
    de las columnas de alineamiento (M/=/X); sin CIGAR, todos los pares
    sin huecos desde ``start``."""
    if not read.cigar or read.cigar == '*':
        return [(i, read.start + i) for i in range(len(read.sequence))]
    pairs: List[Tuple[int, int]] = []
    qpos = 0
    rpos = read.start
    num = ''
    for ch in read.cigar:
        if ch.isdigit():
            num += ch
            continue
        n = int(num or 0)
        num = ''
        if ch in 'M=X':
            for _ in range(n):
                pairs.append((qpos, rpos))
                qpos += 1
                rpos += 1
        elif ch == 'I':
            qpos += n
        elif ch in 'DN':
            rpos += n
        elif ch == 'S':
            qpos += n
        elif ch == 'H':
            pass
    return pairs


class BaseRecalibrator:
    """Estimación de la tabla de recalibración (GATK BaseRecalibrator)."""

    def __init__(self, device: Optional[str] = None,
                 min_base_quality: int = 0,
                 max_cycle: int = 1000,
                 min_mapq: int = 0):
        self.manager = DeviceManager(device)
        self.device = self.manager.device
        self.min_base_quality = min_base_quality
        self.max_cycle = max_cycle
        self.min_mapq = min_mapq

    def estimate(self, reference: str, reads: List[GATKRead],
                 on_progress=None) -> RecalTable:
        """
        Acumula las covariables de todas las lecturas alineadas contra
        ``reference`` (posiciones 0-based de la referencia).
        """
        ref_len = len(reference)
        quals: List[int] = []
        cycles: List[int] = []
        strands: List[int] = []
        mismatches: List[int] = []

        for ri, read in enumerate(reads):
            if self.min_mapq and read.mapq < self.min_mapq:
                continue
            seq = read.sequence.upper()
            qlist = read.phred_scores()
            length = len(seq)
            for qpos, rpos in _match_columns(read):
                if rpos < 0 or rpos >= ref_len or qpos >= length:
                    continue
                q = qlist[qpos] if qpos < len(qlist) else 0
                if q < self.min_base_quality:
                    continue
                base = seq[qpos]
                ref_base = reference[rpos]
                if base not in 'ACGT' or ref_base not in 'ACGT':
                    continue
                cycle = qpos + 1 if read.strand != '-' else length - qpos
                cycle = min(max(cycle, 1), self.max_cycle)
                quals.append(min(max(q, 0), 93))
                cycles.append(cycle)
                strands.append(0 if read.strand != '-' else 1)
                mismatches.append(1 if base != ref_base else 0)
            if on_progress is not None and (ri + 1) % 200 == 0:
                on_progress({"stage": "bqsr", "reads": ri + 1,
                             "bases": len(quals)})

        table = RecalTable(max_cycle=self.max_cycle)
        if not quals:
            return table

        q_t = torch.tensor(quals, dtype=torch.long, device=self.device)
        c_t = torch.tensor(cycles, dtype=torch.long, device=self.device)
        s_t = torch.tensor(strands, dtype=torch.long, device=self.device)
        m_t = torch.tensor(mismatches, dtype=torch.long, device=self.device)

        n_q = 94
        n_c = self.max_cycle + 1
        flat = (q_t * n_c + c_t) * 2 + s_t
        obs = torch.zeros(n_q * n_c * 2, dtype=torch.long,
                          device=self.device)
        err = torch.zeros_like(obs)
        obs.index_add_(0, flat, torch.ones_like(flat))
        err.index_add_(0, flat, m_t)

        obs_cpu = obs.cpu().numpy()
        err_cpu = err.cpu().numpy()
        n_slot = n_q * n_c
        for idx in range(obs_cpu.size):
            o = int(obs_cpu[idx])
            if o == 0:
                continue
            s = idx % 2
            slot = idx // 2
            if slot >= n_slot:
                continue
            q = slot // n_c
            c = slot % n_c
            table.counts[(q, c, s)] = [o, int(err_cpu[idx])]
        table.n_bases = int(obs.sum().item())
        table.n_errors = int(err.sum().item())
        return table


def apply_recalibration(reads: List[GATKRead], table: RecalTable,
                        smoothing: float = 10.0) -> List[GATKRead]:
    """
    ApplyBQSR: devuelve copias de las lecturas con calidades
    recalibradas (Phred+33).
    """
    import dataclasses

    out: List[GATKRead] = []
    for read in reads:
        if not read.qualities:
            out.append(read)
            continue
        length = len(read.sequence)
        scores = read.phred_scores()
        new_scores: List[int] = []
        for qpos, q in enumerate(scores):
            cycle = qpos + 1 if read.strand != '-' else length - qpos
            cycle = min(max(cycle, 1), table.max_cycle or cycle)
            emp = table.empirical_quality(min(max(q, 0), 93), cycle,
                                          0 if read.strand != '-' else 1,
                                          smoothing=smoothing)
            new_scores.append(min(93, max(0, int(round(emp)))))
        quals = ''.join(chr(q + PHRED_OFFSET) for q in new_scores)
        out.append(dataclasses.replace(read, qualities=quals))
    return out


def write_recal_sam(path, reads: List[GATKRead],
                    ref_lengths: Dict[str, int],
                    sample: str = 'SAMPLE') -> None:
    """Escribe un SAM mínimo con las lecturas (y calidades) indicadas."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("@HD\tVN:1.6\tSO:unsorted\n")
        for name, length in ref_lengths.items():
            fh.write(f"@SQ\tSN:{name}\tLN:{length}\n")
        fh.write(f"@RG\tID:1\tSM:{sample}\n")
        for read in reads:
            rname = read.rname or '*'
            pos = read.start + 1 if rname != '*' else 0
            cigar = read.cigar or '*'
            qual = read.qualities or '*'
            fh.write(f"{read.qname}\t{read.flag}\t{rname}\t{pos}\t"
                     f"{read.mapq}\t{cigar}\t*\t0\t0\t{read.sequence}\t"
                     f"{qual}\n")
