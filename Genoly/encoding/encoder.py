from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch

from Genoly.core.device import DeviceManager
from Genoly.core.vram import VRAMManager, retry_on_oom

# Alfabeto canónico para cómputo núcleo
CANONICAL = ['A', 'C', 'G', 'T', 'N']

# Alfabeto IUPAC completo
IUPAC = ['A', 'C', 'G', 'T', 'N', 'R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H', 'V']


@dataclass
class EncodedBatch:
    """
    Micro-lote codificado producido por :meth:`SequenceEncoder.encode_stream`.

    Attributes:
        tensor: Tensor en el dispositivo: (n, L_max) int64 o
            (n, L_max, C) float32 si es one-hot.
        lengths: Longitud real de cada secuencia (sin padding).
        indices: Posición de cada secuencia dentro del lote de RAM
            original (para restaurar el orden si se reordena).
    """
    tensor: torch.Tensor
    lengths: List[int]
    indices: List[int]


class SequenceEncoder:
    """
    Codificación de secuencias de ADN/ARN a tensores numéricos.

    Transforma secuencias de texto en tensores enteros o one-hot listos
    para operar sobre GPU (CUDA/NVIDIA), acelerando el resto del pipeline.
    """

    def __init__(self, device: Optional[str] = None,
                 include_iupac: bool = True):
        """
        Args:
            device: 'cuda', 'cpu' o None para auto-detectar.
            include_iupac: incluir códigos de ambigüedad IUPAC en el alfabeto.
        """
        self.manager = DeviceManager(device)
        self.device = self.manager.device

        self.alphabet = IUPAC if include_iupac else CANONICAL
        self.to_idx: Dict[str, int] = {b: i for i, b in enumerate(self.alphabet)}
        self.idx_to_base: Dict[int, str] = {i: b for b, i in self.to_idx.items()}

        # U -> T (ARN a ADN) para simplificar el cómputo
        self.to_idx.setdefault('U', self.to_idx['T'])
        self.num_classes = len(self.alphabet)

        # Tabla ASCII (256) -> índice, para codificar sin bucles Python
        self._lookup = self._build_lookup()

    def _build_lookup(self) -> np.ndarray:
        """Tabla de traducción vectorizada ASCII -> índice (256 entradas)."""
        lookup = np.full(256, self.to_idx['N'], dtype=np.int64)
        for i in range(256):
            ch = chr(i)
            idx = self.to_idx.get(ch)
            if idx is None:
                idx = self.to_idx.get(ch.upper())
            if idx is None:
                idx = self.to_idx['N']
            lookup[i] = idx
        return lookup

    # ------------------------------------------------------------------ #
    # Codificación a enteros
    # ------------------------------------------------------------------ #
    def encode_char(self, char: str) -> int:
        """Codifica un carácter a índice entero (IUPAC desconocido -> N)."""
        return self.to_idx.get(char.upper(), self.to_idx['N'])

    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """
        Codifica una secuencia a tensor 1D en el dispositivo.

        La codificación es vectorizada (tabla ASCII -> índice), sin
        bucles Python por base, por lo que soporta secuencias de
        cientos de megabases sin degradación catastrófica.

        Args:
            sequence: Secuencia de ADN/ARN.

        Returns:
            Tensor de tipo long con la secuencia codificada.
        """
        if not sequence:
            return torch.zeros(0, dtype=torch.long, device=self.device)
        data = sequence.encode('latin-1', errors='replace')
        arr = np.frombuffer(data, dtype=np.uint8)
        return torch.from_numpy(self._lookup[arr]).to(self.device)

    def encode(self, sequences: List[str],
               padding: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Codifica un lote de secuencias con padding.

        La codificación es vectorizada (tabla ASCII -> índice), sin
        bucles Python por base, para soportar secuencias gigantes.

        Args:
            sequences: Lista de secuencias.
            padding: Índice de padding (por defecto el de 'N').

        Returns:
            Tuple (tensor (B, L), tensor de longitudes).
        """
        if not sequences:
            raise ValueError("La lista de secuencias está vacía")

        lengths = [len(s) for s in sequences]
        max_len = max(lengths)
        pad_idx = self.to_idx['N'] if padding is None else padding

        encoded = torch.full((len(sequences), max_len), pad_idx,
                             dtype=torch.long, device=self.device)

        for i, seq in enumerate(sequences):
            if not seq:
                continue
            data = seq.encode('latin-1', errors='replace')
            arr = np.frombuffer(data, dtype=np.uint8)
            codes = torch.from_numpy(self._lookup[arr]).to(self.device)
            encoded[i, :codes.shape[0]] = codes

        lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=self.device)
        return encoded, lengths_tensor

    # ------------------------------------------------------------------ #
    # Streaming con gestión dinámica de VRAM
    # ------------------------------------------------------------------ #
    def bytes_per_base(self, one_hot: bool = False) -> int:
        """
        Coste estimado en bytes por base del tensor codificado (peor caso).

        Documentación del cálculo:

        - Entera: destino (B, L) int64 (8 B/base) + transitorios de la
          tabla ASCII y de la copia CPU→GPU (16 B/base) => 24 B/base.
        - One-hot: intermedio int64 (8 B/base) + destino (B, L, C)
          float32 (C*4 B/base) + transitorios de máscara y scatter
          (2*C*4 B/base) => 8 + 3*C*4 (68 B/base con C=5).

        Ver ``Genoly.core.vram`` para el modelo completo.
        """
        if one_hot:
            # encode_one_hot genera siempre las 5 clases canónicas
            # (A/C/G/T/N), no el alfabeto IUPAC completo
            return 8 + 3 * 5 * 4
        return 24

    def _encode_group(self, seqs: List[str], lengths: List[int],
                      group: List[int], one_hot: bool):
        """Codifica un micro-lote y devuelve (tensor, longitudes reales).

        Aislado para poder envolverlo en ``retry_on_oom`` desde
        :meth:`encode_stream`.
        """
        if one_hot:
            return self.encode_one_hot(seqs), [lengths[i] for i in group]
        tensor, lengths_tensor = self.encode(seqs)
        return (tensor,
                [int(v) for v in lengths_tensor.cpu().tolist()])

    def encode_stream(self,
                      batches: Iterable[List[str]],
                      one_hot: bool = False,
                      vram: Optional[VRAMManager] = None
                      ) -> Iterator[EncodedBatch]:
        """
        Convierte lotes de RAM en micro-lotes adaptativos de GPU.

        Cada lote de RAM (lista de secuencias) se divide con
        ``VRAMManager.plan_micro_batches`` de modo que el tensor
        resultante quepa en una fracción segura de la VRAM libre:

            bytes(micro-lote) = n * L_max * bytes_per_base <= presupuesto

        Tras ceder cada micro-lote se fuerza la recolección de basura y
        el vaciado de la caché de CUDA (si ``VRAMManager.release_after_each``),
        devolviendo la memoria al driver y evitando fragmentación.

        Args:
            batches: Iterables de lotes de RAM (listas de secuencias),
                p. ej. ``[[s1, s2, ...], [s3, ...]]``.
            one_hot: Si True, codifica one-hot (B, L, C) float32.
            vram: Gestor de VRAM; por defecto se crea uno sobre el
                dispositivo del encoder.

        Yields:
            :class:`EncodedBatch` con el tensor en el dispositivo.
        """
        manager = vram if vram is not None else VRAMManager(self.device)
        cost = self.bytes_per_base(one_hot)

        for batch in batches:
            if not batch:
                continue
            lengths = [len(s) for s in batch]
            for group in manager.plan_micro_batches(lengths, cost):
                seqs = [batch[i] for i in group]
                tensor, chunk_lengths = retry_on_oom(
                    lambda: self._encode_group(seqs, lengths, group,
                                               one_hot),
                    manager=manager)
                yield EncodedBatch(
                    tensor=tensor,
                    lengths=chunk_lengths,
                    indices=list(group),
                )
                del tensor
                if manager.release_after_each:
                    manager.release()

    # ------------------------------------------------------------------ #
    # Codificación one-hot
    # ------------------------------------------------------------------ #
    def encode_one_hot(self, sequences: List[str],
                       include_n: bool = True) -> torch.Tensor:
        """
        Codifica un lote de secuencias a one-hot (B, L, C).

        Args:
            sequences: Lista de secuencias.
            include_n: Incluir columna para 'N' (5 clases canónicas).
                      Si False, usa solo A/C/G/T (4 clases).

        Returns:
            Tensor de tipo float32.
        """
        if not sequences:
            raise ValueError("La lista de secuencias está vacía")

        encoded, lengths = self.encode(sequences)
        # A/C/G/T conservan su índice. N y cualquier código IUPAC de
        # ambigüedad (R, Y, S, W, K, M, B, D, H, V) se agrupan en la clase N.
        # Antes se usaba ``encoded % 5``, que mapeaba R->A, Y->C, S->G, W->T,
        # K->A, M->C... y enmascaraba las bases ambiguas como canónicas.
        canonical_idx = torch.where(encoded < 4, encoded,
                                    torch.full_like(encoded, 4))
        num_classes = 5 if include_n else 4

        one_hot = torch.zeros(
            (*encoded.shape, num_classes), dtype=torch.float32, device=self.device
        )
        valid_class = canonical_idx < num_classes
        canonical_idx = canonical_idx.clamp(0, num_classes - 1)
        one_hot.scatter_(-1, canonical_idx.unsqueeze(-1), 1.0)
        # Ceros en posiciones padding (más allá de la longitud real) y en las
        # bases que no tienen clase asignada (N con include_n=False).
        positions = torch.arange(encoded.shape[1], device=self.device)
        real = positions.unsqueeze(0) < lengths.unsqueeze(1)
        return one_hot * (valid_class & real).unsqueeze(-1).float()

    # ------------------------------------------------------------------ #
    # Decodificación
    # ------------------------------------------------------------------ #
    def decode(self, tensor: torch.Tensor,
               lengths: Optional[torch.Tensor] = None) -> List[str]:
        """
        Convierte un tensor de índices de vuelta a secuencias de texto.

        Args:
            tensor: Tensor (B, L) o (L,) de índices enteros.
            lengths: Longitudes reales por secuencia (para recortar padding).

        Returns:
            Lista de secuencias decodificadas.
        """
        if tensor.dim() == 2:
            decoded = []
            for i in range(tensor.shape[0]):
                length = lengths[i].item() if lengths is not None else tensor.shape[1]
                row = tensor[i, :length]
                decoded.append(self._row_to_string(row))
            return decoded

        return [self._row_to_string(tensor)]

    def _row_to_string(self, row: torch.Tensor) -> str:
        return "".join(self.idx_to_base.get(int(v), 'N') for v in row.tolist())


def encode_to_tensor(sequences: List[str],
                     device: Optional[str] = None,
                     one_hot: bool = False) -> torch.Tensor:
    """
    Función de conveniencia para codificar secuencias a tensores.

    Args:
        sequences: Lista de secuencias.
        device: 'cuda', 'cpu' o None.
        one_hot: Si True devuelve one-hot, si False devuelve enteros.

    Returns:
        Tensor codificado.
    """
    encoder = SequenceEncoder(device)
    if one_hot:
        return encoder.encode_one_hot(sequences)
    encoded, _ = encoder.encode(sequences)
    return encoded