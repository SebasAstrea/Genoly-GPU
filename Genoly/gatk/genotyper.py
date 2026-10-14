"""
HaplotypeCaller al estilo GATK sobre GPU.

Orquesta el pipeline de genotipado de GATK adaptado a Genoly:

    SAM/BAM -> filtros de lectura -> regiones activas (GPU)
        -> ensamblaje local (de Bruijn) -> PairHMM (C / GPU)
        -> likelihoods de genotipo (PL/GQ/AD/DP/QUAL) -> VCF

La heavy-duty (pileup de actividad, likelihoods) corre en tensores
PyTorch sobre CUDA cuando hay GPU; el PairHMM usa el kernel nativo C si
está compilado o el lotado de PyTorch en caso contrario.
"""

from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from Genoly.gatk.active_region import ActiveRegionDetector
from Genoly.gatk.assembly import LocalAssembler
from Genoly.gatk.likelihoods import (
    GenotypeResult,
    allele_log_likelihoods,
    genotype_likelihoods,
    haplotype_sites,
)
from Genoly.gatk.pairhmm import (
    DEFAULT_GAP_EXTEND,
    DEFAULT_GAP_OPEN,
    HAP_QUALITY,
    pairhmm_log10,
)
from Genoly.gatk.reads import FilterStats, GATKRead, filter_reads


@dataclass
class GATKVariant:
    """Variante con genotipo completo al estilo GATK/VCF."""
    position: int            # POS VCF (1-based)
    ref: str                 # REF (anclada en indels)
    alts: List[str]          # ALT (ancladas en indels)
    type: str                # SNV / DEL / INS (del primer alt)
    qual: float              # QUAL phred
    filter: str              # PASS / LowQual / LowDepth
    dp: int                  # lecturas informativas
    ad: List[int]            # REF, ALT1, ...
    pl: List[int]            # Phred-scaled genotype likelihoods
    gq: int                  # calidad del genotipo
    gt: str                  # '0/1', '1/1', '0/0'...
    af: float                # frecuencia del primer alt
    genotypes: List[Tuple[int, ...]] = field(default_factory=list)

    @property
    def alt(self) -> str:
        return ','.join(self.alts)

    @property
    def alt_count(self) -> int:
        return self.ad[1] if len(self.ad) > 1 else 0

    @property
    def freq(self) -> float:
        return self.af


class HaplotypeCaller:
    """Llamada de variantes con reensamblaje local (estilo GATK)."""

    def __init__(self, device: Optional[str] = None,
                 *,
                 min_mapq: int = 20,
                 min_base_quality: int = 10,
                 min_depth: int = 5,
                 min_qual: float = 10.0,
                 min_af: float = 0.15,
                 window: int = 50,
                 extension: int = 50,
                 k: Optional[int] = None,
                 max_haplotypes: int = 32,
                 ploidy: int = 2,
                 hap_quality: float = HAP_QUALITY,
                 gap_open: float = DEFAULT_GAP_OPEN,
                 gap_extend: float = DEFAULT_GAP_EXTEND,
                 max_reads_per_region: int = 2000,
                 backend: str = 'auto',
                 emit_hom_ref: bool = False):
        """
        Args:
            device: 'cuda', 'cpu' o None (auto-detección).
            min_mapq: MAPQ mínimo de las lecturas (filtro GATK).
            min_base_quality: calidad mínima de base para el pileup.
            min_depth: profundidad mínima para emitir un genotipo.
            min_qual: QUAL mínimo; por debajo -> filtro LowQual.
            min_af / window / extension: parámetros del detector de
                regiones activas.
            k: k del ensamblaje (None = automático).
            max_haplotypes: tope del pool de haplotipos.
            ploidy: ploidía del genotipado (2 por defecto).
            hap_quality / gap_open / gap_extend: parámetros del PairHMM.
            max_reads_per_region: tope de lecturas por región activa.
            backend: backend del PairHMM ('auto', 'native', 'torch',
                'numpy').
            emit_hom_ref: emitir también genotipos 0/0 (para depuración).
        """
        self.min_mapq = min_mapq
        self.min_base_quality = min_base_quality
        self.min_depth = min_depth
        self.min_qual = min_qual
        self.ploidy = ploidy
        self.hap_quality = hap_quality
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        self.max_reads_per_region = max_reads_per_region
        self.backend = backend
        self.emit_hom_ref = emit_hom_ref
        self.detector = ActiveRegionDetector(
            window=window, extension=extension, min_af=min_af,
            min_base_quality=min_base_quality, device=device)
        self.assembler = LocalAssembler(k=k, max_haplotypes=max_haplotypes)

    # ------------------------------------------------------------------ #
    # Genotipado de una región
    # ------------------------------------------------------------------ #
    def call_region(self, reference: str,
                    reads: Sequence[GATKRead]) -> List[GATKVariant]:
        """
        Genotipa una región de referencia con las lecturas alineadas.

        Args:
            reference: secuencia de la región (forward).
            reads: lecturas con ``start`` relativo al inicio de la región.

        Returns:
            Variantes con genotipo (POS relativo a ``reference``).
        """
        stats: FilterStats = filter_reads(reads, min_mapq=self.min_mapq)
        kept = stats.reads
        if not kept:
            return []
        active = self.detector.detect(reference, kept)
        if not active:
            return []

        variants: Dict[Tuple, GATKVariant] = {}
        for a0, a1 in active:
            region_reads = [r for r in kept if r.overlaps(a0, a1)]
            if not region_reads:
                continue
            if len(region_reads) > self.max_reads_per_region:
                region_reads = sorted(
                    region_reads, key=lambda r: -r.mapq
                )[:self.max_reads_per_region]
            for v in self._call_active(reference[a0:a1], region_reads):
                v.position += a0
                key = (v.position, v.ref, v.alt)
                prev = variants.get(key)
                if prev is None or v.qual > prev.qual:
                    variants[key] = v
        out = sorted(variants.values(), key=lambda v: v.position)
        return out

    def _call_active(self, sub_ref: str, reads: Sequence[GATKRead]
                     ) -> List[GATKVariant]:
        """Genotipa una región activa (ensamblaje + PairHMM + PL)."""
        seqs = [r.sequence for r in reads]
        quals = [r.qualities for r in reads]
        haplotypes = self.assembler.assemble(sub_ref, seqs)
        if len(haplotypes) < 2:
            return []

        read_hap = pairhmm_log10(
            list(zip(seqs, quals)), haplotypes,
            hap_quality=self.hap_quality, gap_open=self.gap_open,
            gap_extend=self.gap_extend, backend=self.backend)

        sites, per_hap = haplotype_sites(sub_ref, haplotypes)
        out: List[GATKVariant] = []
        for site in sites:
            n_alleles = 1 + len(site.alts)
            if n_alleles < 2:
                continue
            allele_ll = allele_log_likelihoods(read_hap, per_hap,
                                               site, n_alleles)
            geno: GenotypeResult = genotype_likelihoods(
                allele_ll, ploidy=self.ploidy)
            if geno.dp == 0:
                continue
            if geno.gt == '/'.join(['0'] * self.ploidy) \
                    and not self.emit_hom_ref:
                continue
            filt = 'PASS'
            if geno.dp < self.min_depth:
                filt = 'LowDepth'
            elif geno.qual < self.min_qual:
                filt = 'LowQual'
            alt_total = sum(geno.ad[1:])
            af = alt_total / geno.dp if geno.dp else 0.0
            out.append(GATKVariant(
                position=site.position, ref=site.ref, alts=list(site.alts),
                type=site.types[0] if site.types else 'SNV',
                qual=round(geno.qual, 2), filter=filt, dp=geno.dp,
                ad=list(geno.ad), pl=list(geno.pl), gq=geno.gq,
                gt=geno.gt, af=round(af, 4), genotypes=geno.genotypes))
        return out

    # ------------------------------------------------------------------ #
    # Pipeline completo sobre SAM/BAM + FASTA
    # ------------------------------------------------------------------ #
    def call_from_sam(self, sam_path, ref_path,
                      region_size: int = 50_000_000,
                      sample_name: str = 'SAMPLE',
                      on_progress: Optional[Callable[[dict], None]] = None
                      ) -> dict:
        """
        Llama variantes de un SAM/BAM contra una referencia FASTA.

        Procesa cada secuencia por bloques de ``region_size`` bases
        (memoria/VRAM acotada), como el resto del pipeline de Genoly.

        Returns:
            Dict con ``by_sequence``, ``sequences``, ``total_variants``,
            ``snvs``, ``indels``, ``pass_variants``, ``sample_name``.
        """
        from Genoly.gatk.reads import reads_from_sam
        from Genoly.io.fasta import FastaReader

        refs: Dict[str, str] = {}
        for record in FastaReader(ref_path).records():
            refs[record.id] = record.sequence.upper()

        all_reads = reads_from_sam(sam_path)
        reads_by_ref: Dict[str, List[GATKRead]] = {}
        for r in all_reads:
            if r.rname is None:
                continue
            reads_by_ref.setdefault(r.rname, []).append(r)

        by_sequence: Dict[str, List[GATKVariant]] = {}
        sequences_summary: List[dict] = []
        total = snvs = indels = passed = 0

        for name, refseq in refs.items():
            recs = reads_by_ref.get(name)
            if not recs:
                sequences_summary.append(
                    {"name": name, "length": len(refseq), "reads": 0,
                     "variants": 0})
                continue
            length = len(refseq)
            step = region_size or length
            recs.sort(key=lambda r: r.start)
            starts = [r.start for r in recs]
            max_read_len = max((len(r.sequence) for r in recs), default=0)

            seq_variants: List[GATKVariant] = []
            n_reads = 0
            n_regions = len(range(0, length, step))
            for ri, r0 in enumerate(range(0, length, step)):
                r1 = min(r0 + step, length)
                lo = bisect_left(starts, r0 - max_read_len)
                hi = bisect_left(starts, r1)
                region_reads: List[GATKRead] = []
                for r in recs[lo:hi]:
                    if r.overlaps(r0, r1):
                        shifted = GATKRead(
                            qname=r.qname, sequence=r.sequence,
                            start=r.start - r0, strand=r.strand,
                            qualities=r.qualities, mapq=r.mapq,
                            flag=r.flag, cigar=r.cigar, rname=r.rname,
                            mpos=r.mpos)
                        region_reads.append(shifted)
                if not region_reads:
                    continue
                n_reads += len(region_reads)
                local = self.call_region(refseq[r0:r1], region_reads)
                for v in local:
                    v.position += r0
                seq_variants.extend(local)
                if on_progress is not None:
                    on_progress({"sequence": name, "region": ri + 1,
                                 "regions": n_regions,
                                 "variants": len(seq_variants)})

            if seq_variants:
                seq_variants.sort(key=lambda v: v.position)
                by_sequence[name] = seq_variants
                total += len(seq_variants)
                snvs += sum(1 for v in seq_variants if v.type == 'SNV')
                indels += sum(1 for v in seq_variants if v.type != 'SNV')
                passed += sum(1 for v in seq_variants if v.filter == 'PASS')
            sequences_summary.append(
                {"name": name, "length": length, "reads": n_reads,
                 "variants": len(seq_variants)})

        return {
            "by_sequence": by_sequence,
            "sequences": sequences_summary,
            "total_variants": total,
            "snvs": snvs,
            "indels": indels,
            "pass_variants": passed,
            "sample_name": sample_name,
        }
