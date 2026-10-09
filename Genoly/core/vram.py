"""
Gestión dinámica de VRAM (Dynamic VRAM Management) para el pipeline Genoly.

Este módulo implementa el arbitraje de memoria entre RAM y GPU:

1. Mide la memoria libre del dispositivo con ``torch.cuda.mem_get_info()``
   (o la RAM del sistema si no hay CUDA).
2. Aplica una fracción de seguridad sobre esa memoria para no agotarla:
   el propio proceso, el driver y otros procesos necesitan margen.
3. Calcula el peso en bytes de los tensores que generará la siguiente
   etapa y divide el lote de RAM en micro-lotes (micro-batching
   adaptativo) que caben en ese presupuesto.
4. Fuerza la recolección de basura y el vaciado de la caché de PyTorch
   tras cada micro-lote para devolver la memoria al driver y evitar la
   fragmentación del alocador de CUDA.

Límites matemáticos
-------------------
Para un micro-lote de ``n`` secuencias con longitud máxima de padding
``L`` y un coste de ``b`` bytes por base, el peor caso es:

    bytes(micro-lote) = n * L * b

El planificador (``plan_micro_batches``) agrupa las secuencias de forma
gulosa, en orden de llegada, cerrando cada micro-lote en cuanto:

    (n + 1) * max(L_1 .. L_{n+1}) * b  >  presupuesto

donde::

    presupuesto = memoria_libre * fraccion_seguridad   (>= min_budget_bytes)

Una secuencia individual que por sí sola exceda el presupuesto viaja
sola en su propio micro-lote: es responsabilidad de la etapa de
ventanado (``FastaReader.iter_windows`` / ``KmerCounter.split_sequence_windows``)
garantizar que ninguna unidad individual quede por debajo del límite.

Coste por base por etapa (todos los tensores intermedios del conteo de
k-mers son (n, L) en int64 = 8 bytes/elemento, más máscaras bool = 1
byte/elemento). Con la implementación optimizada de este paquete:

    k-mers canónico (int64):  b = 8 * (2 + 2 + 4 + 1 + 2) + 1  =  89 B/base
                              (codificado+base4, transitorios de Horner,
                               transitorios del revcomp, comprimido,
                               transitorios de ``torch.unique``)
    k-mers directo (int64):   b = 8 * (2 + 2 + 1 + 2) + 1      =  57 B/base
    codificación entera:      b = 8 * 3                        =  24 B/base
                              (destino int64 + transitorios CPU→GPU)
    codificación one-hot:     b = 8 + 3 * C * 4                =  68 B/base
                              (intermedio int64 + one-hot float32 con
                               transitorios de máscara/scatter, C = 5)

El modelo incluye un margen del ~30 % sobre el pico teórico para cubrir
los transitorios de ``torch.unique`` (ordenación) y de ``scatter``.
"""

from __future__ import annotations

import ctypes
import gc
import logging
import sys
from typing import Callable, Dict, Iterator, List, Sequence, TypeVar

import torch

from Genoly.core.device import DeviceManager

logger = logging.getLogger("genoly.vram")

T = TypeVar("T")

#: Fracción de la memoria libre que se puede usar como presupuesto de
#: micro-lote. 0.25 deja margen para el driver, otros procesos y los
#: picos transitorios que no captura el modelo lineal.
DEFAULT_SAFETY_FRACTION = 0.25

#: Suelo del presupuesto: evita micro-lotes degenerados en GPUs muy
#: ocupadas (el kernel puede seguir fallando por OOM si la VRAM real
#: disponible es menor que lo que pide una sola unidad).
DEFAULT_MIN_BUDGET_BYTES = 32 * 1024 * 1024  # 32 MiB

#: Techo de elementos por micro-lote: acota el coste del padding y el
#: tiempo de cómputo por llamada al kernel.
DEFAULT_MAX_ITEMS = 65_536

#: Objetivo de bytes por micro-lote para sugerir ventanas de secuencia.
WINDOW_TARGET_BYTES = 128 * 1024 * 1024  # 128 MiB
WINDOW_MIN_BASES = 65_536                # 64 kpb
WINDOW_MAX_BASES = 8 * 1024 * 1024       # 8 Mpb

#: Mínimo al que puede reducirse la fracción de seguridad tras un OOM.
MIN_SAFETY_FRACTION = 0.01

#: Mínimo al que puede reducirse el suelo del presupuesto (1 MiB).
MIN_BUDGET_FLOOR_BYTES = 1024 * 1024


def is_cuda_oom(exc: BaseException) -> bool:
    """
    True si ``exc`` es un agotamiento de memoria de CUDA (o de su caché).

    Cubre ``torch.cuda.OutOfMemoryError`` (PyTorch >= 1.13), el
    ``torch.OutOfMemoryError`` global (>= 2) y el ``RuntimeError`` con el
    mensaje clásico del alocador, para que el reintento funcione también
    con builds antiguos.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    oom_type = getattr(torch, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    if isinstance(exc, MemoryError):
        return True
    return (isinstance(exc, RuntimeError)
            and "out of memory" in str(exc).lower())


def retry_on_oom(fn: Callable[[], T], *,
                 manager: "VRAMManager | None" = None,
                 attempts: int = 3,
                 shrink: float = 0.5) -> T:
    """
    Ejecuta ``fn()`` y, si revienta por falta de memoria de GPU, libera
    memoria, reduce el presupuesto del ``VRAMManager`` y reintenta.

    El suelo de la VRAM libre puede ser menor que el presupuesto mínimo
    (otro proceso puede estar usando la GPU), así que sin este reintento
    el lote primero falla siempre y el trabajo muere sin llegarse a
    procesar. Si se agotan los intentos se relanza la última excepción.

    Args:
        fn: Operación que puede agotar la VRAM (sin argumentos).
        manager: Gestor cuyo presupuesto se reduce en cada reintento.
        attempts: Intentos totales (>= 1).
        shrink: Factor por el que se reduce el presupuesto en cada fallo.

    Returns:
        El valor devuelto por ``fn()``.
    """
    if attempts < 1:
        raise ValueError("attempts debe ser >= 1")
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last = attempt + 1 >= attempts
            if last or not is_cuda_oom(exc):
                raise
            logger.warning(
                "CUDA sin memoria (intento %d/%d); se libera caché y se "
                "reduce el presupuesto ×%s", attempt + 1, attempts, shrink)
            if manager is not None:
                manager.shrink(shrink)
                manager.release()
            else:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    raise AssertionError("inalcanzable")  # pragma: no cover


def _system_free_bytes() -> int:
    """Memoria RAM libre del sistema (fallback cuando no hay CUDA)."""
    try:
        import psutil  # dependencia opcional

        return int(psutil.virtual_memory().available)
    except Exception:
        pass

    if sys.platform.startswith("win"):
        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(
                    ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", "r") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except Exception:
            pass

    return 2 * 1024 * 1024 * 1024  # 2 GiB de cortesía si nada funciona


def estimate_kmer_bytes_per_base(k: int, canonical: bool = True,
                                 itemsize: int = 8) -> int:
    """
    Coste estimado en bytes por base del conteo de k-mers (peor caso).

    Modelo por tensores simultáneos de forma (n, L), itemsize por elemento:

    - 2 tensores fijos (codificación entera + dígitos base-4): ``2``
    - 2 transitorios del bucle de Horner (producto + suma): ``2``
    - comprimido ``codes[valid]`` previo a ``unique``: ``1``
    - 2 transitorios de ``torch.unique`` (ordenación + conteos): ``2``
    - 4 transitorios del reverse complement en streaming: ``4`` (canónico)
    - máscara bool de validez: ``+1`` byte

        b = itemsize * (2 + 2 + 1 + 2 [+ 4]) + 1

    El coste de ``torch.unique`` sobre k-mers es proporcional al número
    de k-mers distintos, acotado por min(4^k, n*L); el término +2 cubre
    el peor caso.
    """
    terms = 2 + 2 + 1 + 2 + (4 if canonical else 0)
    return itemsize * terms + 1


def estimate_encode_bytes_per_base(num_classes: int = 5,
                                   one_hot: bool = False,
                                   itemsize: int = 8) -> int:
    """
    Coste estimado en bytes por base de la codificación a tensores.

    - Entera (int64): destino (n, L) + transitorios numpy/CPU→GPU: 3 * 8.
    - One-hot (n, L, C) en float32: intermedio int64 + destino
      (n, L, C) * 4 + transitorios de máscara y scatter (2 * C * 4):

        b = itemsize + 3 * C * 4
    """
    if one_hot:
        return itemsize + 3 * num_classes * 4
    return 3 * itemsize


class VRAMManager:
    """
    Gestor dinámico de VRAM: presupuesto, micro-batching adaptativo y
    liberación agresiva de memoria entre micro-lotes.

    Args:
        device: 'cuda', 'cpu' o None para auto-detectar.
        safety_fraction: fracción de la memoria libre usable como
            presupuesto por micro-lote (0 < f <= 1).
        min_budget_bytes: suelo del presupuesto en bytes.
        max_items: máximo de secuencias por micro-lote.
        release_after_each: si True (por defecto), los consumidores deben
            llamar a ``release()`` tras cada micro-lote. Forzar
            ``gc.collect() + torch.cuda.empty_cache()`` devuelve la
            memoria reservada al driver y evita la fragmentación, a costa
            de re-alocar bloques CUDA en el siguiente micro-lote; con
            micro-lotes grandes (>= 64 MiB) ese coste es despreciable
            frente al cómputo.
        release_every: liberar cada N micro-lotes en lugar de tras cada
            uno (solo surte efecto con ``release_after_each=True``). Los
            consumidores respetan este ajuste con un contador; valores
            de 4-8 reducen el coste fijo ``gc + empty_cache`` (~33 ms
            por llamada) en pipelines de muchos micro-lotes pequeños.
            El presupuesto se recalcula con la VRAM no liberada, así que
            los micro-lotes intermedios son algo menores (auto-equilibrado,
            nunca más agresivo que con N=1).
    """

    def __init__(self,
                 device=None,
                 safety_fraction: float = DEFAULT_SAFETY_FRACTION,
                 min_budget_bytes: int = DEFAULT_MIN_BUDGET_BYTES,
                 max_items: int = DEFAULT_MAX_ITEMS,
                 release_after_each: bool = True,
                 release_every: int = 1):
        if not 0.0 < safety_fraction <= 1.0:
            raise ValueError("safety_fraction debe estar en (0, 1]")
        if min_budget_bytes < 1:
            raise ValueError("min_budget_bytes debe ser >= 1")
        if max_items < 1:
            raise ValueError("max_items debe ser >= 1")
        if release_every < 1:
            raise ValueError("release_every debe ser >= 1")

        self.manager = DeviceManager(device)
        self.device = self.manager.device
        self.safety_fraction = float(safety_fraction)
        self.min_budget_bytes = int(min_budget_bytes)
        self.max_items = int(max_items)
        self.release_after_each = bool(release_after_each)
        self.release_every = int(release_every)

    # ------------------------------------------------------------------ #
    # Medición de memoria
    # ------------------------------------------------------------------ #
    def free_bytes(self) -> int:
        """Memoria libre del dispositivo (VRAM global o RAM del sistema)."""
        if self.manager.is_cuda:
            free, _ = torch.cuda.mem_get_info(self.manager.cuda_index)
            return int(free)
        return _system_free_bytes()

    def total_bytes(self) -> int:
        """Memoria total del dispositivo."""
        if self.manager.is_cuda:
            _, total = torch.cuda.mem_get_info(self.manager.cuda_index)
            return int(total)
        return self.free_bytes()

    def budget_bytes(self) -> int:
        """
        Presupuesto de bytes para un micro-lote::

            presupuesto = max(memoria_libre * fraccion_seguridad,
                              min_budget_bytes)
        """
        return max(
            int(self.free_bytes() * self.safety_fraction),
            self.min_budget_bytes,
        )

    # ------------------------------------------------------------------ #
    # Micro-batching adaptativo
    # ------------------------------------------------------------------ #
    def plan_micro_batches(self, lengths: Sequence[int],
                           bytes_per_base: int) -> Iterator[List[int]]:
        """
        Divide ``lengths`` en grupos de índices que caben en el presupuesto.

        Cada grupo cumple ``n * max(L_i del grupo) * bytes_per_base <=
        presupuesto``. Las secuencias viajan en orden de llegada (sin
        reordenar), lo que hace el plan compatible con streams. Si una
        sola secuencia excede el presupuesto, viaja sola: la etapa
        correspondiente debe haberla acotado por ventanado.

        Args:
            lengths: longitud en bases de cada secuencia del lote de RAM.
            bytes_per_base: coste en bytes por base (ver las funciones
                ``estimate_*_bytes_per_base`` de este módulo).

        Yields:
            Listas de índices sobre ``lengths`` (micro-lotes).
        """
        if bytes_per_base < 1:
            raise ValueError("bytes_per_base debe ser >= 1")

        group: List[int] = []
        max_len = 0

        for i, length in enumerate(lengths):
            # Se recalcula en cada paso: si un micro-lote revienta por OOM,
            # el consumidor llama a shrink() y los siguientes grupos salen
            # ya con el presupuesto reducido (budget_bytes se lee aquí).
            budget = self.budget_bytes()
            candidate_max = max_len if length <= max_len else length
            candidate_bytes = (len(group) + 1) * candidate_max * bytes_per_base
            if group and (candidate_bytes > budget
                          or len(group) >= self.max_items):
                yield group
                group = [i]
                max_len = length
            else:
                group.append(i)
                max_len = candidate_max

        if group:
            yield group

    def suggest_window_bases(self, bytes_per_base: int,
                             target_bytes: int = WINDOW_TARGET_BYTES) -> int:
        """
        Tamaño de ventana (en bases) para que un micro-lote de ~1 unidad
        quepa holgadamente en el presupuesto.

        ``min(target_bytes, presupuesto) / bytes_per_base``, redondeado a
        potencia de dos por debajo y acotado a [64 kpb, 8 Mpb].
        """
        target = min(self.budget_bytes(), int(target_bytes))
        bases = max(1, target // max(1, bytes_per_base))
        bases = 1 << (bases.bit_length() - 1)  # potencia de 2 por debajo
        return max(WINDOW_MIN_BASES, min(WINDOW_MAX_BASES, bases))

    # ------------------------------------------------------------------ #
    # Liberación de memoria
    # ------------------------------------------------------------------ #
    def release(self) -> None:
        """
        Fuerza la recolección de basura y vacía la caché de PyTorch.

        Llamar tras cada micro-lote devuelve los bloques reservados al
        driver, de modo que la siguiente medición de ``free_bytes()``
        refleja la realidad y se evita la fragmentación del alocador.
        """
        gc.collect()
        self.manager.empty_cache()

    def shrink(self, factor: float = 0.5) -> None:
        """
        Reduce el presupuesto tras un OOM para el siguiente micro-lote.

        Multiplica ``safety_fraction`` y ``min_budget_bytes`` por
        ``factor``, con suelo en :data:`MIN_SAFETY_FRACTION` /
        :data:`MIN_BUDGET_FLOOR_BYTES` para que el presupuesto nunca
        colapse a cero (``plan_micro_batches`` lo exige >= 1).
        """
        if not 0.0 < factor < 1.0:
            raise ValueError("factor debe estar en (0, 1)")
        self.safety_fraction = max(MIN_SAFETY_FRACTION,
                                   self.safety_fraction * factor)
        self.min_budget_bytes = max(
            MIN_BUDGET_FLOOR_BYTES,
            int(self.min_budget_bytes * factor))

    def stats(self) -> Dict[str, object]:
        """Instantánea del estado de memoria (para telemetría/logs)."""
        free = self.free_bytes()
        total = self.total_bytes()
        return {
            "device": str(self.device),
            "free_bytes": free,
            "total_bytes": total,
            "budget_bytes": self.budget_bytes(),
            "safety_fraction": self.safety_fraction,
        }

    def __repr__(self) -> str:  # pragma: no cover - diagnóstico
        stats = self.stats()
        return (f"VRAMManager(device={stats['device']}, "
                f"free={stats['free_bytes'] / 1e9:.2f} GB, "
                f"budget={stats['budget_bytes'] / 1e9:.2f} GB)")
