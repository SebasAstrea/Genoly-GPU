"""
Exportación de variantes a formato VCF 4.2.
"""

from pathlib import Path
from typing import Dict, Iterable, Union


def _vcf_info(variant) -> str:
    """Bloque INFO conforme a la especificación VCF 4.2."""
    fields = [f"DP={variant.depth}", f"AF={variant.freq:.4f}"]
    if variant.type == 'DEL':
        # END es obligatorio para el alelo simbólico <DEL> y coincide con
        # POS+len(REF)-1 en la representación anclada.
        fields.append(f"END={variant.position + len(variant.ref) - 1}")
    else:
        ref_count = variant.bases.get(variant.ref) if variant.bases else None
        if ref_count is None:
            ref_count = max(variant.depth - variant.alt_count, 0)
        fields.append(f"AD={ref_count},{variant.alt_count}")
    return ";".join(fields)


def _vcf_alleles(variant):
    """``(POS, REF, ALT)`` válidos para el registro.

    Las deleciones se escriben en representación mínima anclada a la base
    anterior (``POS = inicio - 1``, ``REF = anclaje + bases borradas``,
    ``ALT = anclaje``), que es lo que exige VCF 4.2: ``ALT='-'`` no es un
    alelo válido. Si la deleción empieza en la primera base de la secuencia
    no hay anclaje y se recurre al alelo simbólico ``<DEL>``.
    """
    if variant.type != 'DEL':
        return variant.position, variant.ref, variant.alt

    anchor = getattr(variant, 'anchor', None)
    if anchor:
        return variant.position - 1, anchor + variant.ref, anchor
    return variant.position, variant.ref[:1] or 'N', '<DEL>'


def write_vcf(path: Union[str, Path],
              by_sequence: Dict[str, list],
              sequences: Iterable[dict] = ()) -> int:
    """
    Escribe un VCF 4.2 con las variantes agrupadas por secuencia.

    Args:
        path: Ruta de salida.
        by_sequence: ``{nombre_secuencia: [Variant, ...]}``.
        sequences: Resumen por secuencia (para los contigs @contig).

    Returns:
        Número de variantes escritas.
    """
    contigs = sequences or ({'name': k} for k in by_sequence)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("##fileformat=VCFv4.2\n")
        fh.write('##source=Genoly-GPU\n')
        fh.write('##INFO=<ID=DP,Number=1,Type=Integer,Description="Total depth">\n')
        fh.write('##INFO=<ID=AD,Number=R,Type=Integer,Description="Allelic depths (REF,ALT)">\n')
        fh.write('##INFO=<ID=AF,Number=A,Type=Float,Description="Alt allele frequency">\n')
        fh.write('##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the variant">\n')
        fh.write('##ALT=<ID=DEL,Description="Deletion">\n')
        for c in contigs:
            name = c.get('name', c) if isinstance(c, dict) else c
            fh.write(f'##contig=<ID={name}>\n')
        fh.write('#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n')
        for name, variants in by_sequence.items():
            for v in variants:
                pos, ref, alt = _vcf_alleles(v)
                fh.write(f"{name}\t{pos}\t.\t{ref}\t{alt}\t."
                         f"\tPASS\t{_vcf_info(v)}\n")
                count += 1
    return count