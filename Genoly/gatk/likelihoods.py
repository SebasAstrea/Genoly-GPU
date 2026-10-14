"""
Likelihoods de genotipo al estilo GATK (desde el PairHMM hasta PL/GQ).

Flujo (equivalente al de HaplotypeCaller, en orden):

1. Cada haplotipo ensamblado se compara con la referencia (opcodes de
   ``difflib``: sustituciones = SNVs, borrados = DELs, inserciones = INSs)
   y se extraen los sitios variantes con alelos en representación VCF
   anclada.
2. La likelihood de un alelo para una lectura es la suma de
   probabilidades (log-sum-exp) sobre los haplotipos que llevan ese
   alelo — la marginalización de haplotipos de GATK.
3. Las likelihoods de genotipo (log10) para cada combinación de ploidía
   se combinan como mezcla 1/2 (heterocigoto) y se normalizan a Phred
   (PL), con GQ, AD, DP y QUAL (phred de que el sitio NO sea hom-ref),
   como en el modelo de genotipado de GATK.
"""

import difflib
import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

LOG_HALF = math.log10(0.5)
MAX_QUAL = 999.0


@dataclass
class VariantSite:
    """Sitio variante con alelos en representación VCF (anclada)."""
    key: Tuple                  # ('snv', pos0) | ('del', s, e) | ('ins', a)
    position: int               # POS VCF (1-based)
    ref: str                    # alelo REF (con ancla en indels)
    alts: List[str]             # alelos ALT (con ancla en indels)
    types: List[str] = field(default_factory=list)  # por alt: SNV/DEL/INS


@dataclass
class GenotypeResult:
    """Genotipo llamado para un sitio (modelo diploide por defecto)."""
    gt: str                     # '0/1', '1/1', ...
    gl: List[float]             # log10 P(datos | genotipo), sin normalizar
    pl: List[int]               # Phred-scaled (el mejor vale 0)
    gq: int                     # calidad del genotipo
    ad: List[int]               # profundidad por alelo (REF, ALT...)
    dp: int                     # lecturas informativas
    qual: float                 # QUAL del sitio (phred P(no es hom-ref))
    genotypes: List[Tuple[int, ...]]


def _compare_haplotype(reference: str, haplotype: str) -> List[dict]:
    """Eventos variantes de un haplotipo vs la referencia."""
    matcher = difflib.SequenceMatcher(None, reference, haplotype,
                                      autojunk=False)
    events: List[dict] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        if tag == 'replace' and (i2 - i1) == (j2 - j1):
            for off in range(i2 - i1):
                events.append({'kind': 'snv', 'pos': i1 + off,
                               'ref': reference[i1 + off],
                               'alt': haplotype[j1 + off]})
            continue
        if tag in ('replace', 'delete') and i2 > i1 and i1 > 0:
            events.append({'kind': 'del', 'start': i1, 'end': i2,
                           'anchor': reference[i1 - 1],
                           'seq': reference[i1:i2]})
        if tag in ('replace', 'insert') and j2 > j1 and i1 > 0:
            events.append({'kind': 'ins', 'anchor': i1 - 1,
                           'seq': haplotype[j1:j2],
                           'anchor_base': reference[i1 - 1]})
    return events


def _site_from_event(event: dict) -> Tuple[Tuple, int, str, str, str]:
    """``(key, position, ref, alt, type)`` en representación VCF."""
    if event['kind'] == 'snv':
        pos0 = event['pos']
        return (('snv', pos0), pos0 + 1, event['ref'], event['alt'], 'SNV')
    if event['kind'] == 'del':
        start, end = event['start'], event['end']
        return (('del', start, end), start,
                event['anchor'] + event['seq'], event['anchor'], 'DEL')
    anchor = event['anchor']
    return (('ins', anchor), anchor + 1,
            event['anchor_base'], event['anchor_base'] + event['seq'], 'INS')


def haplotype_sites(reference: str, haplotypes: Sequence[str]
                    ) -> Tuple[List[VariantSite], List[Dict[Tuple, int]]]:
    """
    Extrae los sitios variantes del pool de haplotipos.

    Returns:
        ``(sites, per_hap_alleles)`` donde ``per_hap_alleles[i]`` asigna
        a cada ``site.key`` el índice de alelo que lleva ese haplotipo
        (0 = REF). Un haplotipo que no toca el sitio queda en REF.
    """
    # key -> {alelo_str: tipo}; y por hap, key -> alelo_str
    alleles: Dict[Tuple, Dict[str, str]] = {}
    hap_allele_str: List[Dict[Tuple, str]] = []
    for hap in haplotypes:
        per_key: Dict[Tuple, str] = {}
        for event in _compare_haplotype(reference, hap):
            key, _pos, ref, alt, typ = _site_from_event(event)
            per_key[key] = alt
            bucket = alleles.setdefault(key, {})
            bucket.setdefault(ref, 'REF')
            bucket.setdefault(alt, typ)
        hap_allele_str.append(per_key)

    sites: List[VariantSite] = []
    key_to_index: Dict[Tuple, int] = {}
    for key in sorted(alleles, key=lambda k: (str(k[0]), k[1:])):
        if key[0] == 'snv':
            pos = key[1] + 1
        elif key[0] == 'del':
            pos = key[1]          # POS del ancla (1-based)
        else:
            pos = key[1] + 1
        ref = ''
        for a, t in alleles[key].items():
            if t == 'REF':
                ref = a
                break
        alts = [a for a, t in alleles[key].items() if t != 'REF']
        types = [alleles[key][a] for a in alts]
        key_to_index[key] = len(sites)
        sites.append(VariantSite(key=key, position=pos, ref=ref,
                                 alts=alts, types=types))

    per_hap: List[Dict[Tuple, int]] = []
    for mapping in hap_allele_str:
        idx_map: Dict[Tuple, int] = {}
        for key, alt in mapping.items():
            site = sites[key_to_index[key]]
            try:
                idx_map[key] = 1 + site.alts.index(alt)
            except ValueError:
                idx_map[key] = 0
        per_hap.append(idx_map)
    return sites, per_hap


def allele_log_likelihoods(read_hap_ll: np.ndarray,
                           per_hap_alleles: Sequence[Dict[Tuple, int]],
                           site: VariantSite, n_alleles: int) -> np.ndarray:
    """``(n_lecturas, n_alelos)`` marginalizando haplotipos por alelo."""
    n_reads = read_hap_ll.shape[0]
    out = np.full((n_reads, n_alleles), -np.inf, dtype=np.float64)
    for a in range(n_alleles):
        cols = [h for h, mapping in enumerate(per_hap_alleles)
                if mapping.get(site.key, 0) == a]
        if not cols:
            continue
        block = read_hap_ll[:, cols]
        m = block.max(axis=1, keepdims=True)
        sums = m[:, 0] + np.log10(np.power(10.0, block - m).sum(axis=1))
        out[:, a] = np.where(np.isfinite(m[:, 0]), sums, -np.inf)
    return out


def genotype_likelihoods(allele_ll: np.ndarray, ploidy: int = 2) -> GenotypeResult:
    """
    Combina las likelihoods por alelo en un genotipo (modelo GATK).

    - Heterocigoto: mezcla equiprobable 1/2 de los dos alelos.
    - PL: Phred-scaled y normalizado (el mejor genotipo vale 0).
    - GQ: PL del segundo mejor (tope 99).
    - QUAL: phred de P(no ser hom-ref) con prior uniforme.
    """
    n_alleles = allele_ll.shape[1]
    if n_alleles < 2:
        raise ValueError("hace falta al menos un alelo REF y uno ALT")
    if ploidy == 1:
        genotypes = [(a,) for a in range(n_alleles)]
    elif ploidy == 2:
        # orden VCF de PL (Number=G): (0,0), (0,1), (1,1), (0,2), ...
        genotypes = [(i, j) for j in range(n_alleles) for i in range(j + 1)]
    else:
        genotypes = list(itertools.combinations_with_replacement(
            range(n_alleles), ploidy))
    gl = np.full(len(genotypes), -np.inf, dtype=np.float64)
    informative = np.isfinite(allele_ll).any(axis=1)
    used = allele_ll[informative]
    for g, combo in enumerate(genotypes):
        if len(set(combo)) == 1:
            gl[g] = float(used[:, combo[0]].sum())
        else:
            a, b = combo[0], combo[1]
            m = np.maximum(used[:, a], used[:, b])
            mix = m + np.log10(
                0.5 * np.power(10.0, used[:, a] - m)
                + 0.5 * np.power(10.0, used[:, b] - m))
            gl[g] = float(mix.sum())
    dp = int(informative.sum())
    best = int(np.argmax(gl))
    norm = gl - gl[best]
    pl = np.clip(np.round(-10.0 * norm), 0, None).astype(int).tolist()
    pl[best] = 0
    gq = int(min(99, sorted(pl)[1])) if len(pl) > 1 else 99

    ad = np.zeros(n_alleles, dtype=int)
    if dp > 0:
        winner = np.argmax(used, axis=1)
        for a in range(n_alleles):
            ad[a] = int((winner == a).sum())

    # P(hom-ref | datos) con prior uniforme sobre genotipos
    ref_gt = tuple(sorted((0,) * ploidy))
    try:
        ref_idx = genotypes.index(ref_gt)
    except ValueError:
        ref_idx = None
    if ref_idx is None or not np.isfinite(gl[ref_idx]):
        qual = MAX_QUAL
    else:
        shifted = gl - gl.max()
        denom = np.power(10.0, shifted).sum()
        p_homref = float(np.power(10.0, shifted[ref_idx]) / denom)
        qual = min(MAX_QUAL, -10.0 * math.log10(max(p_homref, 1e-99)))

    gt = '/'.join(str(a) for a in sorted(genotypes[best]))
    return GenotypeResult(gt=gt, gl=gl.tolist(), pl=pl, gq=gq,
                          ad=ad.tolist(), dp=dp, qual=qual,
                          genotypes=genotypes)
