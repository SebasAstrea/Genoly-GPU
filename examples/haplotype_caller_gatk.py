"""
Llamada de variantes estilo GATK HaplotypeCaller con Genoly-GPU.

Demuestra el pipeline completo portado de GATK 4.x:

  1. HaplotypeCaller: regiones activas (GPU) -> reensamblaje local
     (de Bruijn) -> PairHMM (kernel C o torch GPU) -> genotipado
     PL/GQ/AD/DP/QUAL -> VCF 4.2.
  2. MarkDuplicates: claves 5' sin clips y puntuación por calidades,
     con flags 0x400 en el SAM de salida.
  3. BQSR: covariables (calidad x ciclo x hebra) y recalibracion de
     las calidades de base (tabla JSON + SAM recalibrado).

Los datos son sinteticos (referencia de 2 kb con un SNV heterocigoto
y una deleción de 3 pb simulados en las lecturas).

Uso:
    python examples/haplotype_caller_gatk.py
"""

import os
import random
import sys
import tempfile

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from Genoly.gatk import (  # noqa: E402
    BaseRecalibrator,
    GATKRead,
    HaplotypeCaller,
    active_regions,
    apply_duplicate_flags,
    apply_recalibration,
    duplicate_keys,
    has_native,
    mark_duplicates,
    write_gatk_vcf,
    write_recal_sam,
)

SEED = 11
REF_LEN = 2000
SNV_POS = 750          # 0-based
DEL_START = 1200       # 0-based, deleción de 3 pb
DEL_LEN = 3
READ_LEN = 90
STRIDE = 3             # solape denso alrededor de cada variante


def simular_referencia():
    """Referencia pseudoaleatoria determinista."""
    rng = random.Random(SEED)
    return "".join(rng.choice("ACGT") for _ in range(REF_LEN))


def simular_lecturas(ref):
    """Lecturas densas (tiling) con un SNV heterocigoto y una deleción.

    Los starts se reparten a intervalos fijos alrededor de cada variante
    para que las lecturas la flanqueen con holgura (necesario para que el
    ensamblaje local reconstruya ambos alelos).
    """
    rng = random.Random(SEED + 1)
    snv_alt = "ACGT"[("ACGT".index(ref[SNV_POS]) + 1) % 4]
    del_hap = ref[:DEL_START] + ref[DEL_START + DEL_LEN:]

    reads = []

    def add(qname, seq, start):
        if not (0 <= start and start + len(seq) <= len(ref)):
            return
        qual = "".join(chr(33 + rng.randint(35, 40)) for _ in seq)
        reads.append(GATKRead(
            qname=qname, sequence=seq, qualities=qual, flag=0,
            rname="chrSim", start=start, mapq=60,
            cigar=f"{len(seq)}M"))

    # fondo de lecturas de referencia (tiling de todo el genoma)
    for i, start in enumerate(range(0, REF_LEN - READ_LEN, READ_LEN // 2)):
        add(f"bg{i}", ref[start:start + READ_LEN], start)

    # alelo de referencia alrededor del SNV (flanqueo >= 25 pb cada lado)
    snv_starts = range(SNV_POS - 60, SNV_POS + 1, STRIDE)
    for i, start in enumerate(snv_starts):
        add(f"snvref{i}", ref[start:start + READ_LEN], start)
    # alelo alternativo (SNV heterocigoto 1:1)
    alt = ref[:SNV_POS] + snv_alt + ref[SNV_POS + 1:]
    for i, start in enumerate(snv_starts):
        add(f"snvalt{i}", alt[start:start + READ_LEN], start)

    # deleción de 3 pb: lecturas cortas del haplótipo con la deleción
    del_starts = range(DEL_START - 60, DEL_START + 1, STRIDE)
    for i, start in enumerate(del_starts):
        add(f"del{i}", del_hap[start:start + READ_LEN], start)
    return reads, snv_alt, del_hap


def main():
    ref = simular_referencia()
    reads, snv_alt, del_hap = simular_lecturas(ref)
    print(f"Referencia: {REF_LEN} pb | lecturas: {len(reads)} "
          f"(PairHMM nativo C: {'sí' if has_native() else 'no'})")

    # --- 1. Región activa (la etapa previa que filtra posiciones) -------
    activas = active_regions(reads, ref, window=50, extension=50)
    print(f"Regiones activas con evidencia de variantes: {len(activas)}")
    for a, b in activas[:5]:
        print(f"  [{a}, {b})")

    # --- 2. HaplotypeCaller -------------------------------------------
    caller = HaplotypeCaller(min_mapq=20, min_depth=5, min_qual=10.0,
                             min_af=0.15, max_haplotypes=32)
    variantes = caller.call_region(ref, reads)
    print(f"\nVariantes llamadas: {len(variantes)}")
    for v in variantes:
        print(f"  {v.position}\t{v.ref}>{v.alt}\t{v.type}\t"
              f"GT={v.gt}\tDP={v.dp}\tAD={v.ad}\tGQ={v.gq}\t"
              f"QUAL={v.qual:.1f}\t{v.filter}")

    snvs = [v for v in variantes if v.type == "SNV"
            and v.position == SNV_POS + 1]
    dels = [v for v in variantes if v.type == "DEL"
            and v.position == DEL_START]
    assert snvs and snvs[0].gt == "0/1", "el SNV heterocigoto debe llamarse"
    assert dels, "la deleción debe llamarse"

    # --- 3. VCF (estilo GATK: GT:AD:DP:GQ:PL) -------------------------
    with tempfile.TemporaryDirectory() as tmp:
        vcf = os.path.join(tmp, "haplotype.vcf")
        n = write_gatk_vcf(vcf, {"chrSim": variantes},
                           [{"name": "chrSim", "length": REF_LEN}],
                           sample="SIM")
        print(f"\nVCF escrito: {n} variantes en {vcf}")
        with open(vcf, encoding="utf-8") as fh:
            for line in fh:
                if not line.startswith("#"):
                    print(" ", line.rstrip())

    # --- 4. MarkDuplicates -------------------------------------------
    stats = mark_duplicates(reads)
    flagged = apply_duplicate_flags(reads, set(stats["qnames"]))
    print(f"\nDuplicados: {stats['duplicates']}/{stats['total']} "
          f"({stats['duplicate_rate'] * 100:.1f}%) en "
          f"{stats['libraries']} claves")
    ej = duplicate_keys(reads[0])
    print(f"  Ejemplo de clave 5': {ej}")

    # --- 5. BQSR ------------------------------------------------------
    table = BaseRecalibrator().estimate(ref, reads)
    recal = apply_recalibration(flagged, table)
    print(f"\nBQSR: {table.n_bases} bases, {table.n_errors} errores, "
          f"tasa global {table.global_rate:.5f}")
    print(f"  Qual ciclo 1 reportada=40 -> "
          f"empírica={table.empirical_quality(40, 1, 0):.1f}")
    with tempfile.TemporaryDirectory() as tmp:
        sam = os.path.join(tmp, "recalibrated.sam")
        write_recal_sam(sam, recal, {"chrSim": REF_LEN}, sample="SIM")
        print(f"  SAM recalibrado escrito: {sam}")

    print("\nPipeline GATK completado.")


if __name__ == "__main__":
    main()
