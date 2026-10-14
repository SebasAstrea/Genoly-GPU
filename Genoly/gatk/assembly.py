"""
Ensamblaje local de haplotipos (grafo de De Bruijn).

Port simplificado y determinista de la etapa de reensamblaje de
HaplotypeCaller: a partir de las lecturas que caen en una región activa
se construye un grafo de k-mers, se ancla en los k-mers del extremo de la
referencia y se enumeran caminos de origen a destino como haplotipos
candidatos. La referencia se incluye siempre (GATK garantiza que el
camino de referencia esté en el pool).

Diferencias con GATK (documentadas a propósito):
- GATK adapta k en varios pases (10..100) con poda de grafos; aquí un k
  fijo por región (por defecto automático) con conteo de soporte.
- GATK puntúa caminos con un modelo complejo; aquí se ordenan por soporte
  de aristas (nº de lecturas que sostienen el k-mer siguiente).
"""

from typing import Dict, List, Optional, Sequence

#: Soporte mínimo de una arista para no podarla (1 = no podar nada).
MIN_EDGE_SUPPORT = 1

_ACGT = frozenset('ACGT')


class LocalAssembler:
    """Ensamblador local de haplotipos sobre un grafo de k-mers."""

    def __init__(self, k: Optional[int] = None,
                 max_haplotypes: int = 32,
                 max_paths: int = 20_000,
                 min_edge_support: int = MIN_EDGE_SUPPORT):
        """
        Args:
            k: longitud de k-mer; None = automática según lecturas.
            max_haplotypes: tope de haplotipos devueltos (incluye ref).
            max_paths: tope de caminos explorados (cota de coste).
            min_edge_support: poda de aristas con menos soporte.
        """
        self.k = k
        self.max_haplotypes = max(2, max_haplotypes)
        self.max_paths = max_paths
        self.min_edge_support = max(1, min_edge_support)

    def choose_k(self, reference: str, reads: Sequence[str]) -> int:
        """Elige k automáticamente: menor que toda lectura y que la ref."""
        if self.k is not None:
            return self.k
        min_read = min((len(r) for r in reads if r), default=0)
        limit = min(len(reference), min_read) - 1
        return max(5, min(25, limit))

    def assemble(self, reference: str, reads: Sequence[str]) -> List[str]:
        """
        Ensambla haplotipos candidatos para ``reference`` (la secuencia
        de la región, en orientación forward).

        Returns:
            Lista de haplotipos (la referencia es siempre el primero).
        """
        reference = reference.upper()
        if not reference or len(reference) < 3:
            return [reference] if reference else []
        k = self.choose_k(reference, reads)
        if k < 5 or k >= len(reference):
            return [reference]

        succ: Dict[str, Dict[str, int]] = {}

        def add_kmers(sequence: str, weight: int = 1) -> None:
            for i in range(len(sequence) - k + 1):
                kmer = sequence[i:i + k]
                if not _ACGT.issuperset(kmer):
                    continue
                nxt = sequence[i + k] if i + k < len(sequence) else None
                if nxt is None or nxt not in _ACGT:
                    continue
                bucket = succ.setdefault(kmer, {})
                bucket[nxt] = bucket.get(nxt, 0) + weight

        for read in reads:
            if read:
                add_kmers(read.upper())
        # la referencia siempre conectada (peso alto: no se poda su camino)
        add_kmers(reference, weight=10_000)

        start = reference[:k]
        sink = reference[-k:]
        if start not in succ and sink not in succ and len(succ) == 0:
            return [reference]

        haplotypes: List[str] = [reference]
        seen = {reference}
        # DFS: (kmer_actual, camino_de_bases_extra)
        stack: List[tuple] = [(start, start)]
        paths = 0
        while stack and len(haplotypes) < self.max_haplotypes \
                and paths < self.max_paths:
            kmer, path = stack.pop()
            paths += 1
            edges = succ.get(kmer)
            if not edges:
                continue
            ordered = sorted(edges.items(),
                             key=lambda kv: (-kv[1], kv[0]))
            for base, support in ordered:
                if support < self.min_edge_support:
                    continue
                nxt = (kmer + base)[1:]
                new_path = path + base
                if len(new_path) > len(reference) + 64:
                    continue  # ampliación descontrolada (repeats)
                if nxt == sink:
                    # alcanzar el sink = el haplotipo termina en el final
                    # de la región; si es más corto que la referencia es un
                    # alelo con deleción interna (se conserva tal cual)
                    if new_path not in seen and len(new_path) >= k:
                        seen.add(new_path)
                        haplotypes.append(new_path)
                        if len(haplotypes) >= self.max_haplotypes:
                            break
                    continue
                if nxt in kmer:
                    continue  # lazo simple
                stack.append((nxt, new_path))
        return haplotypes
