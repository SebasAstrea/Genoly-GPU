import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

# Motor nativo opcional (parasail, SIMD/C): sustituye el Smith-Waterman
# escalar de Python, que es O(n^3) e inutilizable (una alineación 100x100
# tarda ~34 s; con parasail, ~0.02 ms). Si no está instalado, se usa el
# alineador Python como fallback.
try:
    import parasail as _parasail
except ImportError:  # pragma: no cover
    _parasail = None

#: Escala entera para parasail (SIMD exige scoring entero). x2 preserva los
#: parámetros por defecto del módulo (match=2, mismatch=-1, open=-2, ext=-0.5).
_NATIVE_SCALE = 2

@dataclass
class AlignmentResult:
    """Resultado estructurado de un alineamiento"""
    score: float
    aligned_query: str
    aligned_target: str
    alignment_length: int
    identity_percent: float
    gaps: int
    mismatches: int
    cigar_string: Optional[str] = None
    #: Extremos 0-based de la alineación en la referencia/query (nativo
    #: parasail); None en la ruta Python escalar. Útiles para localizar la
    #: coordenada de un alineamiento local dentro de un tramo mayor.
    ref_end: Optional[int] = None
    query_end: Optional[int] = None

class GPUSequenceAligner:
    """
    Alineador de secuencias genómicas con aceleración GPU.

    Unifica funcionalidades de alineamiento y análisis básico. El núcleo de
    Smith-Waterman usa el motor nativo parasail (SIMD) cuando está
    disponible, y cae al algoritmo Python escalar en caso contrario.
    """

    def __init__(self,
                 match_score: float = 2.0,
                 mismatch_penalty: float = -1.0,
                 gap_open: float = -2.0,
                 gap_extend: float = -0.5,
                 device: Optional[str] = None):
        """
        Inicializa el alineador con parámetros de scoring.

        Args:
            match_score: Puntuación por coincidencia (default: 2.0)
            mismatch_penalty: Penalización por mismatch (default: -1.0)
            gap_open: Penalización por apertura de gap (default: -2.0)
            gap_extend: Penalización por extensión de gap (default: -0.5)
            device: 'cuda', 'cpu', o None para auto-detectar
        """
        self.match_score = match_score
        self.mismatch_penalty = mismatch_penalty
        self.gap_open = gap_open
        self.gap_extend = gap_extend

        # Configurar dispositivo
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Mapeo de nucleótidos
        self.nucleotides = ['A', 'C', 'G', 'T', 'N']
        self.nuc_to_idx = {nuc: i for i, nuc in enumerate(self.nucleotides)}
        self.idx_to_nuc = {i: nuc for nuc, i in self.nuc_to_idx.items()}

        # Motor nativo parasail (matriz + penalizaciones enteras escaladas)
        self.engine = 'parasail' if _parasail is not None else 'python'
        self._native_matrix = None
        self._native_open = 0
        self._native_gap = 0
        if _parasail is not None:
            self._native_matrix = _parasail.matrix_create(
                'ACGTN',
                int(round(match_score * _NATIVE_SCALE)),
                int(round(mismatch_penalty * _NATIVE_SCALE)),
            )
            # parasail espera penalizaciones positivas (las resta internamente)
            self._native_open = int(round(-gap_open * _NATIVE_SCALE))
            self._native_gap = int(round(-gap_extend * _NATIVE_SCALE))

        # Precomputar matriz de scoring
        self._init_scoring_matrix()

        # Estado
        self.batch_size = 0
        self.max_sequence_length = 0

        print(f"GPUSequenceAligner inicializado en: {self.device} "
              f"(motor: {self.engine})")
    
    def _init_scoring_matrix(self):
        """Inicializa la matriz de scoring en el dispositivo adecuado"""
        size = len(self.nucleotides)
        self.scoring_matrix = torch.zeros((size, size), device=self.device)
        
        # Llenar matriz
        for i in range(size):
            for j in range(size):
                if i == j:
                    self.scoring_matrix[i, j] = self.match_score
                else:
                    self.scoring_matrix[i, j] = self.mismatch_penalty
    
    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """
        Codifica una secuencia de ADN/RNA a tensor numérico.
        
        Args:
            sequence: Secuencia como string
            
        Returns:
            Tensor codificado en el dispositivo
        """
        encoded = torch.zeros(len(sequence), dtype=torch.long, device=self.device)
        
        for i, nuc in enumerate(sequence.upper()):
            encoded[i] = self.nuc_to_idx.get(nuc, self.nuc_to_idx['N'])
        
        return encoded
    
    def encode_batch(self, sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Codifica un batch de secuencias con padding.
        
        Args:
            sequences: Lista de secuencias
            
        Returns:
            Tuple: (tensor codificado, tensor de longitudes)
        """
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths)
        batch_size = len(sequences)
        
        # Tensor con padding
        encoded = torch.full((batch_size, max_len), 
                            self.nuc_to_idx['N'], 
                            dtype=torch.long,
                            device=self.device)
        
        # Codificar cada secuencia
        for i, seq in enumerate(sequences):
            seq_encoded = self.encode_sequence(seq)
            encoded[i, :len(seq_encoded)] = seq_encoded
        
        lengths_tensor = torch.tensor(lengths, device=self.device)
        
        return encoded, lengths_tensor
    
    def align_pair(self, query: str, target: str) -> AlignmentResult:
        """
        Alinea un par de secuencias usando algoritmo Smith-Waterman.

        Args:
            query: Secuencia query
            target: Secuencia target/referencia

        Returns:
            AlignmentResult con toda la información
        """
        if not query or not target:
            return AlignmentResult(
                score=0.0, aligned_query='', aligned_target='',
                alignment_length=0, identity_percent=0.0,
                gaps=0, mismatches=0, cigar_string='')

        if self.engine == 'parasail':
            score, aligned_q, aligned_t, ref_end, query_end = \
                self._align_native(query, target)
        else:
            # Codificar secuencias
            query_encoded = self.encode_sequence(query)
            target_encoded = self.encode_sequence(target)
            score, aligned_q, aligned_t = self._smith_waterman(
                query_encoded, target_encoded)
            ref_end = query_end = None

        # Calcular métricas
        alignment_len = len(aligned_q)
        identity = sum(1 for q, t in zip(aligned_q, aligned_t) if q == t and q != '-')
        identity_pct = (identity / alignment_len * 100) if alignment_len > 0 else 0

        gaps = aligned_q.count('-') + aligned_t.count('-')
        mismatches = sum(1 for q, t in zip(aligned_q, aligned_t)
                        if q != t and q != '-' and t != '-')

        # Generar CIGAR string simplificado
        cigar = self._generate_cigar(aligned_q, aligned_t)

        return AlignmentResult(
            score=score,
            aligned_query=aligned_q,
            aligned_target=aligned_t,
            alignment_length=alignment_len,
            identity_percent=identity_pct,
            gaps=gaps,
            mismatches=mismatches,
            cigar_string=cigar,
            ref_end=ref_end,
            query_end=query_end,
        )

    def _align_native(self, query: str, target: str
                      ) -> Tuple[float, str, str, Optional[int], Optional[int]]:
        """
        Smith-Waterman local con el motor nativo parasail (SIMD).

        Las penalizaciones y el match/mismatch se escalan x2 a enteros; el
        score se devuelve en la escala original (dividido por la escala).
        Devuelve además los extremos 0-based (ref_end, query_end) de la
        alineación en cada secuencia.
        """
        res = _parasail.sw_trace_scan_sat(
            query, target, self._native_open, self._native_gap,
            self._native_matrix)
        tb = res.get_traceback()
        return (res.score / _NATIVE_SCALE, tb.query, tb.ref,
                int(res.end_ref) if res.pointer else None,
                int(res.end_query) if res.pointer else None)
    
    def align_batch(self, queries: List[str], targets: List[str]) -> List[AlignmentResult]:
        """
        Alinea múltiples pares de secuencias en paralelo.
        
        Args:
            queries: Lista de secuencias query
            targets: Lista de secuencias target (misma longitud que queries)
            
        Returns:
            Lista de AlignmentResult
        """
        if len(queries) != len(targets):
            raise ValueError("queries y targets deben tener la misma longitud")
        
        results = []

        # El motor nativo (parasail) es O(n^2) SIMD: procesar en serie es
        # suficiente para miles de pares por segundo.
        for q, t in zip(queries, targets):
            results.append(self.align_pair(q, t))

        return results
    
    def find_variants(self, aligned_query: str, aligned_target: str) -> List[Dict[str, Any]]:
        """
        Encuentra variantes entre dos secuencias alineadas.
        
        Args:
            aligned_query: Secuencia query alineada
            aligned_target: Secuencia target alineada
            
        Returns:
            Lista de variantes encontradas
        """
        variants = []
        ref_pos = 0
        
        for i, (q_base, t_base) in enumerate(zip(aligned_query, aligned_target)):
            # Incrementar posición en referencia si no es gap
            if t_base != '-':
                ref_pos += 1
            
            # Detectar variantes (SNVs, inserciones, deleciones)
            if q_base != t_base:
                variant = {
                    'position': ref_pos,
                    'ref': t_base if t_base != '-' else '',
                    'alt': q_base if q_base != '-' else '',
                    'type': self._classify_variant(q_base, t_base)
                }
                variants.append(variant)
        
        return variants
    
    def analyze_mutations(self, 
                         query: str, 
                         target: str,
                         known_variants: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Análisis completo de mutaciones entre dos secuencias.
        
        Args:
            query: Secuencia query
            target: Secuencia target
            known_variants: Variantes conocidas para comparar
            
        Returns:
            Diccionario con análisis completo
        """
        # 1. Alinear
        alignment = self.align_pair(query, target)
        
        # 2. Encontrar variantes
        variants = self.find_variants(alignment.aligned_query, alignment.aligned_target)
        
        # 3. Comparar con variantes conocidas
        known_matches = []
        if known_variants:
            known_matches = self._match_known_variants(variants, known_variants)
        
        # 4. Calcular estadísticas
        stats = {
            'total_variants': len(variants),
            'snvs': sum(1 for v in variants if v['type'] == 'SNV'),
            'insertions': sum(1 for v in variants if v['type'] == 'INSERTION'),
            'deletions': sum(1 for v in variants if v['type'] == 'DELETION'),
            'known_matches': len(known_matches),
            'alignment_score': alignment.score,
            'identity_percent': alignment.identity_percent
        }
        
        return {
            'alignment': alignment,
            'variants': variants,
            'known_matches': known_matches,
            'statistics': stats
        }
    
    def _smith_waterman(self, query: torch.Tensor, target: torch.Tensor) -> Tuple[float, str, str]:
        """
        Implementación de Smith-Waterman en PyTorch.
        
        Args:
            query: Tensor de query codificado
            target: Tensor de target codificado
            
        Returns:
            Tuple: (score, aligned_query, aligned_target)
        """
        m, n = len(query), len(target)
        
        # Matrices de scoring
        H = torch.zeros((m + 1, n + 1), device=self.device)
        traceback = torch.zeros((m + 1, n + 1), dtype=torch.long, device=self.device)
        
        # Llenar matrices
        max_score = 0.0
        max_i, max_j = 0, 0
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Calcular scores
                match_score = H[i-1, j-1] + self.scoring_matrix[query[i-1], target[j-1]]
                
                # Buscar mejor gap en query (deleción en query = inserción en target)
                best_gap_query = float('-inf')
                for k in range(1, i):
                    gap_score = H[i-k, j] + self.gap_open + self.gap_extend * (k-1)
                    best_gap_query = max(best_gap_query, gap_score)
                
                # Buscar mejor gap en target (inserción en query = deleción en target)
                best_gap_target = float('-inf')
                for k in range(1, j):
                    gap_score = H[i, j-k] + self.gap_open + self.gap_extend * (k-1)
                    best_gap_target = max(best_gap_target, gap_score)
                
                # Elegir mejor opción
                scores = torch.tensor([0.0, match_score, best_gap_query, best_gap_target], 
                                     device=self.device)
                H[i, j] = torch.max(scores)
                traceback[i, j] = torch.argmax(scores).item()
                
                # Actualizar máximo
                if H[i, j] > max_score:
                    max_score = H[i, j].item()
                    max_i, max_j = i, j
        
        # Traceback
        i, j = max_i, max_j
        aligned_query = []
        aligned_target = []
        
        while i > 0 and j > 0 and H[i, j] > 0:
            move = traceback[i, j]
            
            if move == 1:  # Diagonal (match/mismatch)
                aligned_query.append(self.idx_to_nuc[query[i-1].item()])
                aligned_target.append(self.idx_to_nuc[target[j-1].item()])
                i -= 1
                j -= 1
            elif move == 2:  # Gap en query
                aligned_query.append(self.idx_to_nuc[query[i-1].item()])
                aligned_target.append('-')
                i -= 1
            elif move == 3:  # Gap en target
                aligned_query.append('-')
                aligned_target.append(self.idx_to_nuc[target[j-1].item()])
                j -= 1
            else:
                break
        
        # Revertir y convertir a strings
        aligned_query_str = ''.join(reversed(aligned_query))
        aligned_target_str = ''.join(reversed(aligned_target))
        
        return max_score, aligned_query_str, aligned_target_str
    
    def _classify_variant(self, query_base: str, target_base: str) -> str:
        """Clasifica el tipo de variante"""
        if query_base == '-' and target_base != '-':
            return 'DELETION'
        elif query_base != '-' and target_base == '-':
            return 'INSERTION'
        elif query_base != target_base and query_base != '-' and target_base != '-':
            return 'SNV'
        return 'UNKNOWN'
    
    def _match_known_variants(self, found_variants: List[Dict], 
                             known_variants: List[Dict]) -> List[Dict]:
        """Compara variantes encontradas con base de datos conocida"""
        matches = []
        
        for found in found_variants:
            for known in known_variants:
                # Comparación simple (en realidad sería más complejo)
                if (found['position'] == known.get('position') and
                    found['ref'] == known.get('ref') and
                    found['alt'] == known.get('alt')):
                    match = found.copy()
                    match['known_data'] = known
                    matches.append(match)
        
        return matches
    
    def _generate_cigar(self, aligned_query: str, aligned_target: str) -> str:
        """Genera string CIGAR simplificado"""
        cigar = []
        count = 1
        current_op = None
        
        for q, t in zip(aligned_query, aligned_target):
            if q == t:
                op = 'M'
            elif q == '-':
                op = 'D'
            elif t == '-':
                op = 'I'
            else:
                op = 'X'
            
            if op == current_op:
                count += 1
            else:
                if current_op is not None:
                    cigar.append(f"{count}{current_op}")
                current_op = op
                count = 1
        
        if current_op is not None:
            cigar.append(f"{count}{current_op}")
        
        return ''.join(cigar)
    
    def benchmark(self, sequence_length: int = 100, num_pairs: int = 100) -> Dict[str, float]:
        """
        Ejecuta benchmark de rendimiento.
        
        Args:
            sequence_length: Longitud de secuencias de prueba
            num_pairs: Número de pares a alinear
            
        Returns:
            Diccionario con métricas de rendimiento
        """
        # Generar secuencias aleatorias
        import random
        bases = ['A', 'C', 'G', 'T']
        
        queries = [''.join(random.choices(bases, k=sequence_length)) 
                  for _ in range(num_pairs)]
        targets = queries.copy()  # Para benchmark, alinear secuencias iguales
        
        # Benchmark GPU
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start = time.time()
        
        self.align_batch(queries, targets)
        
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        gpu_time = time.time() - start
        
        # Benchmark CPU (si estamos en GPU)
        cpu_time = None
        if self.device.type == 'cuda':
            cpu_aligner = GPUSequenceAligner(device='cpu')
            start = time.time()
            cpu_aligner.align_batch(queries, targets)
            cpu_time = time.time() - start
        
        return {
            'gpu_time': gpu_time,
            'cpu_time': cpu_time,
            'pairs_per_second': num_pairs / gpu_time if gpu_time > 0 else 0,
            'speedup': cpu_time / gpu_time if cpu_time and gpu_time > 0 else None
        }