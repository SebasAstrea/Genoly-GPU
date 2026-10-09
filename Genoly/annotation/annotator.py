"""
Anotación funcional: mapeo de variantes a genes/regiones y ontologías.

- ``load_gff``: parser de GFF3/GTF en streaming (intervalos por secuencia
  y tipo de feature), con atributos genéricos (gene_id, Name, GO...).
- ``Annotator``: anota variantes (posición 1-based + secuencia) con el gen
  o feature que las contiene, la región relativa (exón/intrón/UTR...) y
  sus términos GO.
- ``summarize_go``: agrega términos GO de las variantes anotadas.

Pensado para funcionar con anotaciones reales (Ensembl/NCBI GFF3) sin
cargar el archivo completo en memoria: se conservan solo los intervalos
de los tipos de feature solicitados.
"""

import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

#: Tipos de feature que se anotan por defecto (SO/GFF3).
DEFAULT_FEATURE_TYPES = (
    "gene", "mRNA", "transcript", "exon", "CDS",
    "five_prime_UTR", "three_prime_UTR", "UTR",
)

#: Orden de prioridad para describir la región de una variante.
_REGION_PRIORITY = {
    "CDS": 5, "exon": 4, "five_prime_UTR": 3, "three_prime_UTR": 3,
    "UTR": 3, "mRNA": 2, "transcript": 2, "gene": 1,
}


@dataclass
class Feature:
    """Feature genómica anotada (gen, transcrito, exón, CDS...)."""
    seqid: str
    type: str
    start: int          # 1-based inclusive (GFF)
    end: int            # 1-based inclusive
    strand: str = "."
    attributes: Dict[str, str] = field(default_factory=dict)

    def contains(self, position: int) -> bool:
        return self.start <= position <= self.end

    @property
    def gene_id(self) -> Optional[str]:
        for key in ("gene_id", "gene", "Parent", "ID", "Name"):
            if key in self.attributes:
                return self.attributes[key].split(",")[0]
        return None

    @property
    def name(self) -> Optional[str]:
        return self.attributes.get("Name") or self.attributes.get("gene_name")

    @property
    def go_terms(self) -> List[str]:
        terms: List[str] = []
        for key in ("Ontology_term", "go_term", "GO"):
            if key in self.attributes:
                terms.extend(self.attributes[key].split(","))
        return [t for t in terms if t]


def _open_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _parse_attributes(field: str, fmt: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    if fmt == "gtf":
        for part in field.strip().strip(";").split(";"):
            part = part.strip()
            if not part or " " not in part:
                continue
            key, value = part.split(" ", 1)
            attrs[key.strip()] = value.strip().strip('"')
    else:  # GFF3
        for part in field.strip().split(";"):
            part = part.strip()
            if not part or "=" not in part:
                continue
            key, value = part.split("=", 1)
            attrs[key.strip()] = value.strip()
    return attrs


def load_gff(path: Union[str, Path],
             feature_types: Iterable[str] = DEFAULT_FEATURE_TYPES,
             on_progress=None) -> Dict[str, List[Feature]]:
    """
    Carga features de un GFF3/GTF (o .gz) agrupadas por secuencia.

    Args:
        path: Ruta del archivo de anotación.
        feature_types: Tipos de feature a conservar.
        on_progress: Callback ``fn({"features": n})`` cada 100k líneas.

    Returns:
        Dict ``{seqid: [Feature, ...]}`` ordenadas por inicio.
    """
    path = Path(path)
    wanted = set(feature_types)
    by_seq: Dict[str, List[Feature]] = {}
    n = 0
    fmt = "gtf" if path.suffix.lower() in (".gtf", ".gtf.gz") else "gff"

    with _open_maybe_gzip(path) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            ftype = parts[2]
            if ftype not in wanted:
                continue
            try:
                start, end = int(parts[3]), int(parts[4])
            except ValueError:
                continue
            feature = Feature(
                seqid=parts[0], type=ftype, start=start, end=end,
                strand=parts[6], attributes=_parse_attributes(parts[8], fmt),
            )
            by_seq.setdefault(feature.seqid, []).append(feature)
            n += 1
            if on_progress is not None and n % 100_000 == 0:
                on_progress({"features": n})

    for seqid in by_seq:
        by_seq[seqid].sort(key=lambda f: (f.start, f.end))
    return by_seq


def _find_features(features: List[Feature], starts: List[int],
                   position: int) -> List[Feature]:
    """
    Features que contienen ``position``.

    Los intervalos pueden estar anidados (gen > transcrito > exón) y no
    siguen un orden estricto por ``end``, así que la búsqueda binaria por
    ``end`` no es válida: se localiza por inicio con ``bisect`` (los
    features están ordenados por ``start``) y se filtra hacia atrás.
    """
    import bisect
    hi = bisect.bisect_right(starts, position)
    found = []
    for i in range(hi):
        f = features[i]
        if f.end >= position:
            found.append(f)
    return found


class Annotator:
    """Anota variantes con genes, región relativa y términos GO."""

    def __init__(self, feature_types: Iterable[str] = DEFAULT_FEATURE_TYPES):
        self.feature_types = tuple(feature_types)
        self.by_seq: Dict[str, List[Feature]] = {}
        self._starts: Dict[str, List[int]] = {}

    def load(self, path, on_progress=None) -> int:
        """Carga la anotación y devuelve el número de features."""
        self.by_seq = load_gff(path, self.feature_types, on_progress)
        self._starts = {seqid: [f.start for f in feats]
                        for seqid, feats in self.by_seq.items()}
        return sum(len(v) for v in self.by_seq.values())

    def annotate(self, seqid: str, position: int) -> Optional[dict]:
        """
        Anota una variante (``position`` 1-based) en la secuencia ``seqid``.

        Returns:
            Dict con ``gene``, ``gene_name``, ``region`` (CDS/exon/intrón/
            gen/UTR), ``strand``, ``go_terms`` y los features solapados;
            ``None`` si no cae en ningún feature anotado.
        """
        features = self.by_seq.get(seqid)
        if not features:
            return None
        hits = _find_features(features, self._starts.get(seqid, []), position)
        if not hits:
            return None

        best = max(hits, key=lambda f: _REGION_PRIORITY.get(f.type, 0))
        gene = next((f for f in hits if f.type == "gene"), None)
        gene_id = (gene.gene_id if gene else None) or best.gene_id
        gene_name = (gene.name if gene else None) or best.name

        # región: si el mejor feature es de nivel gen/transcrito, la variante
        # cae en un exón/CDS (ya sería el mejor por prioridad) o en un
        # intrón si el transcrito tiene exones anotados.
        region = best.type
        if best.type in ("gene", "mRNA", "transcript"):
            has_exon = any(f.type in ("exon", "CDS") for f in features
                           if f.start <= position <= f.end
                           or (f.start <= best.end and f.end >= best.start))
            region = "intron" if has_exon else best.type

        go_terms: List[str] = []
        for f in hits:
            go_terms.extend(f.go_terms)

        return {
            "seqid": seqid,
            "position": position,
            "gene": gene_id,
            "gene_name": gene_name,
            "region": region,
            "strand": best.strand,
            "go_terms": sorted(set(go_terms)),
            "features": sorted({f.type for f in hits}),
        }

    def annotate_variants(self, variants: Iterable, on_progress=None) -> dict:
        """
        Anota una lista de variantes (dicts con ``sequence``/``position``,
        o de un VCF con ``CHROM``/``POS``).

        Returns:
            Dict con ``annotated`` (lista), ``by_gene`` (conteo por gen),
            ``by_region`` (conteo por región), ``go_terms`` (conteo por
            término GO), ``total`` y ``annotated_count``.
        """
        annotated: List[dict] = []
        by_gene: Dict[str, int] = {}
        by_region: Dict[str, int] = {}
        go_counts: Dict[str, int] = {}
        total = 0

        for v in variants:
            total += 1
            if isinstance(v, dict):
                seqid = v.get("sequence") or v.get("seqid") or v.get("CHROM")
                pos = v.get("position") or v.get("POS")
                extra = {k: v.get(k) for k in ("ref", "alt", "type", "freq")
                         if k in v}
            else:
                seqid = getattr(v, "sequence", None)
                pos = getattr(v, "position", None)
                extra = {"ref": getattr(v, "ref", None),
                         "alt": getattr(v, "alt", None),
                         "type": getattr(v, "type", None)}
            if seqid is None or pos is None:
                continue
            hit = self.annotate(str(seqid), int(pos))
            if hit is None:
                continue
            hit.update(extra)
            annotated.append(hit)
            if hit["gene"]:
                by_gene[hit["gene"]] = by_gene.get(hit["gene"], 0) + 1
            by_region[hit["region"]] = by_region.get(hit["region"], 0) + 1
            for term in hit["go_terms"]:
                go_counts[term] = go_counts.get(term, 0) + 1
            if on_progress is not None and len(annotated) % 1000 == 0:
                on_progress({"annotated": len(annotated), "total": total})

        return {
            "annotated": annotated,
            "by_gene": dict(sorted(by_gene.items(),
                                   key=lambda kv: -kv[1])[:100]),
            "by_region": by_region,
            "go_terms": dict(sorted(go_counts.items(),
                                    key=lambda kv: -kv[1])[:100]),
            "total": total,
            "annotated_count": len(annotated),
        }


#: Nombres legibles de términos GO frecuentes (subconjunto; el resto se
#: muestra por su ID). Sirve como etiqueta cuando no hay ontología local.
GO_NAMES = {
    "GO:0008150": "proceso biológico",
    "GO:0003674": "función molecular",
    "GO:0005575": "componente celular",
    "GO:0006412": "traducción",
    "GO:0006355": "regulación de la transcripción",
    "GO:0005515": "interacción proteína-proteína",
    "GO:0005524": "unión a ATP",
    "GO:0004672": "actividad proteína quinasa",
    "GO:0007165": "transducción de señales",
    "GO:0006950": "respuesta a estrés",
    "GO:0055085": "transporte transmembrana",
    "GO:0003824": "actividad catalítica",
    "GO:0016020": "membrana",
    "GO:0005634": "núcleo",
    "GO:0005737": "citoplasma",
}


def summarize_go(go_counts: Dict[str, int]) -> List[dict]:
    """Convierte el conteo de términos GO en una lista etiquetada."""
    return [
        {"id": term, "name": GO_NAMES.get(term, term), "count": count}
        for term, count in sorted(go_counts.items(),
                                  key=lambda kv: -kv[1])
    ]