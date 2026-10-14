"""
Algoritmos de GATK portados a Genoly-GPU (Python + C + GPU).

Implementación propia, inspirada en el diseño de GATK 4.x (Broad
Institute), de los algoritmos centrales del pipeline de genómica:

- :class:`~Genoly.gatk.genotyper.HaplotypeCaller`: llamada de variantes
  con regiones activas, reensamblaje local (de Bruijn), PairHMM y
  genotipado con PL/GQ/AD/DP/QUAL.
- :mod:`Genoly.gatk.pairhmm`: motor de likelihoods P(lectura |
  haplotipo) — el mismo rol que el PairHMM nativo C++ de GATK — con
  kernel en C (``csrc/pairhmm.c``) y respaldo lotado en PyTorch (GPU).
- :mod:`Genoly.gatk.bqsr`: BaseRecalibrator/ApplyBQSR (covariables
  calidad × ciclo × hebra, estimación en GPU).
- :mod:`Genoly.gatk.markdups`: MarkDuplicates (claves 5' sin clips,
  pares normalizados, puntuación por calidades).
- :mod:`Genoly.gatk.active_region`: detección de regiones activas.
- :mod:`Genoly.gatk.assembly`: reensamblaje local de haplotipos.
- :mod:`Genoly.gatk.vcf`: escritura VCF con FORMAT GT/AD/DP/GQ/PL.

La API es independiente de GATK (no hay código Java): los algoritmos se
reimplementaron en Python/C sobre los tensores GPU de Genoly.
"""

from Genoly.gatk.active_region import ActiveRegionDetector, active_regions
from Genoly.gatk.assembly import LocalAssembler
from Genoly.gatk.bqsr import (
    BaseRecalibrator,
    RecalTable,
    apply_recalibration,
    write_recal_sam,
)
from Genoly.gatk.genotyper import GATKVariant, HaplotypeCaller
from Genoly.gatk.likelihoods import (
    GenotypeResult,
    VariantSite,
    allele_log_likelihoods,
    genotype_likelihoods,
    haplotype_sites,
)
from Genoly.gatk.markdups import (
    apply_duplicate_flags,
    duplicate_keys,
    mark_duplicates,
    read_score,
)
from Genoly.gatk.pairhmm import has_native, pairhmm_log10
from Genoly.gatk.reads import (
    FilterStats,
    GATKRead,
    filter_reads,
    reads_from_sam,
    with_qualities,
)
from Genoly.gatk.vcf import write_gatk_vcf

__all__ = [
    'ActiveRegionDetector',
    'active_regions',
    'LocalAssembler',
    'BaseRecalibrator',
    'RecalTable',
    'apply_recalibration',
    'write_recal_sam',
    'HaplotypeCaller',
    'GATKVariant',
    'GenotypeResult',
    'VariantSite',
    'genotype_likelihoods',
    'haplotype_sites',
    'allele_log_likelihoods',
    'mark_duplicates',
    'apply_duplicate_flags',
    'duplicate_keys',
    'read_score',
    'pairhmm_log10',
    'has_native',
    'GATKRead',
    'FilterStats',
    'filter_reads',
    'reads_from_sam',
    'with_qualities',
    'write_gatk_vcf',
]
