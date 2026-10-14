"""
Exportación de variantes genotipadas a VCF 4.2 (estilo GATK).

Escribe el esquema de HaplotypeCaller: cabeceras FILTER (LowQual,
LowDepth), FORMAT (GT/AD/DP/GQ/PL) y columnas de muestra, con las
variantes ancladas según VCF 4.2 para indels.
"""

from pathlib import Path
from typing import Dict, Iterable, Union


def _format_header() -> str:
    return (
        '##FILTER=<ID=LowQual,Description="Low quality genotype">\n'
        '##FILTER=<ID=LowDepth,Description="Insufficient depth">\n'
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##FORMAT=<ID=AD,Number=R,Type=Integer,'
        'Description="Allelic depths for the ref and alt alleles in the '
        'order listed">\n'
        '##FORMAT=<ID=DP,Number=1,Type=Integer,'
        'Description="Approximate read depth over the site">\n'
        '##FORMAT=<ID=GQ,Number=1,Type=Integer,'
        'Description="Genotype Quality">\n'
        '##FORMAT=<ID=PL,Number=G,Type=Integer,'
        'Description="Normalized, Phred-scaled genotype likelihoods">\n'
    )


def _info_fields(v) -> str:
    fields = [f"DP={v.dp}", f"AF={','.join(f'{a:.4f}' for a in _alt_afs(v))}"]
    if v.type in ('DEL', 'INS'):
        end = v.position + len(v.ref) - 1
        fields.append(f"END={end}")
    return ";".join(fields)


def _alt_afs(v):
    total = sum(v.ad) if v.ad else max(v.dp, 1)
    if not v.ad or len(v.ad) < 2:
        return [v.af]
    return [a / total if total else 0.0 for a in v.ad[1:]]


def write_gatk_vcf(path: Union[str, Path],
                   by_sequence: Dict[str, list],
                   sequences: Iterable[dict] = (),
                   sample: str = 'SAMPLE') -> int:
    """
    Escribe un VCF 4.2 con genotipos (FORMAT + columna de muestra).

    Args:
        path: ruta de salida.
        by_sequence: ``{secuencia: [GATKVariant, ...]}``.
        sequences: resumen por secuencia (para los ``##contig``).
        sample: nombre de la muestra (columna final).

    Returns:
        Número de variantes escritas.
    """
    contigs = sequences or ({'name': k} for k in by_sequence)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("##fileformat=VCFv4.2\n")
        fh.write("##source=Genoly-GPU HaplotypeCaller (GATK-inspired)\n")
        fh.write(_format_header())
        fh.write('##INFO=<ID=DP,Number=1,Type=Integer,'
                 'Description="Total depth">\n')
        fh.write('##INFO=<ID=AF,Number=A,Type=Float,'
                 'Description="Allele frequency in genotypes">\n')
        fh.write('##INFO=<ID=END,Number=1,Type=Integer,'
                 'Description="End position of the variant">\n')
        for c in contigs:
            name = c.get('name', c) if isinstance(c, dict) else c
            length = c.get('length') if isinstance(c, dict) else None
            if length:
                fh.write(f'##contig=<ID={name},length={length}>\n')
            else:
                fh.write(f'##contig=<ID={name}>\n')
        fh.write('#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t'
                 f'{sample}\n')
        for name, variants in by_sequence.items():
            for v in variants:
                alt = ','.join(v.alts) if getattr(v, 'alts', None) \
                    else v.alt
                fmt = 'GT:AD:DP:GQ:PL'
                sample_col = ':'.join([
                    v.gt,
                    ','.join(str(x) for x in v.ad),
                    str(v.dp),
                    str(v.gq),
                    ','.join(str(x) for x in v.pl),
                ])
                fh.write(f"{name}\t{v.position}\t.\t{v.ref}\t{alt}\t"
                         f"{v.qual:.2f}\t{v.filter}\t{_info_fields(v)}\t"
                         f"{fmt}\t{sample_col}\n")
                count += 1
    return count
