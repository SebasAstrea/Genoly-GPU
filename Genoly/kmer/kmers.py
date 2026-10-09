import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, NamedTuple, Optional, Tuple

import numpy as np
import torch

from Genoly.core.device import DeviceManager
from Genoly.core.vram import (
    VRAMManager,
    estimate_kmer_bytes_per_base,
    is_cuda_oom,
    retry_on_oom,
)
from Genoly.encoding.encoder import SequenceEncoder
from Genoly.io.fasta import (
    DEFAULT_BLOCK_SIZE,
    FASTA_CODE_INVALID,
    FastaReader,
)

logger = logging.getLogger("genoly.kmers")

#: Unidades (registros/ventanas) agrupadas por lote de RAM antes de
#: planificar los micro-lotes de GPU.
DEFAULT_RAM_BATCH_SIZE = 64

#: Elementos acumulados en las partes (valores, conteos) antes de
#: compactar la agregación en CPU (~4M elementos * 16 B = 64 MiB).
MERGE_THRESHOLD_ELEMENTS = 4_000_000

#: Particiones del acumulador con derrame a disco. Los k-mers se reparten
#: por sus bits bajos (valor & (P-1)): dos copias del mismo k-mer caen
#: siempre en la misma partición, así que agregar cada partición de forma
#: independiente produce el resultado exacto global.
N_PARTITIONS = 256

#: Filas bufferizadas en RAM por partición antes de derramar a .npy.
SPILL_ROWS_PER_PARTITION = 2_000_000

#: Filas por llamada a ``torch.unique`` en el finalize (acota la VRAM del
#: dedup final; un micro-lote tipico de 32M filas ocupa ~1.3 GB de VRAM).
GPU_BATCH_ROWS = 32_000_000


def _env_spill_dir() -> Optional[str]:
    """
    Directorio base del derrame k-mer configurable (``GENOLY_SPILL_DIR``).

    El derrame de un genoma completo con k grande puede ocupar decenas de
    GB (p. ej. ~34 GB para el salmón con k=21). El directorio temporal
    del sistema suele ser tmpfs (RAM), que no tiene capacidad para eso:
    con esta variable se apunta a un directorio en disco. Si no se define,
    se usa el directorio temporal del sistema.
    """
    env = os.environ.get("GENOLY_SPILL_DIR")
    if not env:
        return None
    base = Path(env).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    return str(base)


def _pid_alive(pid: int) -> bool:
    """True si existe un proceso con ese PID (o no se puede comprobar)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cleanup_orphan_spills(spill_dir: Optional[str] = None,
                          min_age_seconds: int = 300) -> int:
    """
    Elimina los directorios de derrame k-mer huérfanos.

    Cuando un trabajador GPU muere abruptamente (OOM del kernel, segfault,
    ``kill -9``), su ``TemporaryDirectory`` no llega a limpiarse y queda
    un ``genoly_kmer_<pid>_<aleatorio>/`` con decenas de GB en disco. El
    prefijo incluye el PID del proceso que lo creó: un directorio es
    huérfano si su PID ya no existe y supera la edad mínima.

    Args:
        spill_dir: Base de derrame a escanear (por defecto, la configurada
            con ``GENOLY_SPILL_DIR`` o el directorio temporal del sistema).
        min_age_seconds: Edad mínima del directorio antes de eliminarlo
            (evita borrar un directorio recién creado por un job activo en
            caso de reutilización de PID).

    Returns:
        Número de directorios eliminados.
    """
    base = Path(spill_dir) if spill_dir else Path(
        _env_spill_dir() or tempfile.gettempdir())
    if not base.is_dir():
        return 0

    removed = 0
    now = time.time()
    for entry in base.iterdir():
        if not entry.is_dir() or not entry.name.startswith("genoly_kmer_"):
            continue
        try:
            pid = int(entry.name.split("_")[2])
        except (IndexError, ValueError):
            pid = None  # nombre inesperado: se borra solo si es muy antiguo
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        if pid is not None and _pid_alive(pid):
            continue  # un job vivo lo está usando
        if now - mtime < min_age_seconds:
            continue
        try:
            shutil.rmtree(entry)
            removed += 1
        except OSError:
            pass
    return removed

#: Coste en bytes por base de la ruta numérica (uint8 H2D + dígitos
#: int32 + palabras int32 + transitorios de Horner/revcomp + combine
#: int64 + máscara bool), para el planificador de micro-lotes.
CODES_BYTES_PER_BASE = 64

#: Dígitos base-4 que caben en un int32 con signo: 4^15 = 2^30 < 2^31.
#: Los códigos de k-mer se construyen en int32 (tasa plena en GPU; int64
#: rinde ~2x peor por ancho de banda de memoria) y solo se combinan a
#: int64 al final del micro-lote. Para k <= 15 basta una palabra; para
#: 16 <= k <= 30 se usan dos (lo + hi). k = 31 necesita 62 bits y cae a
#: la ruta int64 clásica.
INT32_DIGITS = 15


class _KmerWords(NamedTuple):
    """
    Códigos de k-mer de un micro-lote en palabras int32.

    lo: (B, W) int32 — los 15 dígitos menos significativos del código
        (o el código completo si k <= 15, o int64 si k = 31).
    hi: (B, W) int32 con los dígitos restantes (código = hi * 4^15 + lo),
        o None si k <= 15 o k = 31.
    valid: (B, W) bool — ventanas con los k dígitos canónicos (A/C/G/T)
        y dentro de la longitud real de cada secuencia.
    """
    lo: torch.Tensor
    hi: Optional[torch.Tensor]
    valid: torch.Tensor


class _PartitionedKmerAccumulator:
    """
    Acumulador de (k-mer, conteo) particionado con derrame a disco.

    Motivación: en genomas reales con k grande, casi todos los k-mers son
    únicos, de modo que el resultado exacto puede ocupar varios GB (p. ej.
    el cromosoma 1 humano con k=21: ~246M de k-mers únicos x 12 B ≈ 3 GB).
    Materializarlo en RAM no es viable en máquinas modestas.

    Estrategia (externa al RAM, como Jellyfish/KMC):

    1. ``add`` reparte cada micro-lote (ya agregado con ``torch.unique``)
       en ``n_partitions`` cubetas por ``valor & (P-1)`` — barato y
       uniforme: el rango de códigos es disjunto entre particiones.
    2. Cuando el buffer de una partición supera ``spill_rows``, se
       derrama a un par de ficheros ``.npy`` (valores int64, conteos
       int32) y se libera la RAM.
    3. ``finalize`` acumula particiones consecutivas hasta reunir
       ``group_rows`` filas, deduplica cada grupo en una sola llamada
       GPU (``torch.unique`` + ``index_add_``) y cede el resultado
       parcial. Como cada grupo contiene particiones completas (dos
       copias del mismo k-mer caen siempre en la misma partición), el
       resultado es exacto. El pico de RAM es
       O(filas_del_grupo) ≈ total/P × particiones_por_grupo, nunca el
       total.

    El consumidor debe llamar a ``close()`` (limpia los temporizables),
    idealmente en un ``finally``.
    """

    def __init__(self,
                 n_partitions: int = N_PARTITIONS,
                 spill_rows: int = SPILL_ROWS_PER_PARTITION,
                 spill_dir: Optional[str] = None,
                 device=None,
                 group_rows: int = GPU_BATCH_ROWS):
        if n_partitions < 1 or (n_partitions & (n_partitions - 1)) != 0:
            raise ValueError("n_partitions debe ser potencia de 2")
        if spill_rows < 1:
            raise ValueError("spill_rows debe ser >= 1")
        if group_rows < 1:
            raise ValueError("group_rows debe ser >= 1")

        self.n_partitions = n_partitions
        self.spill_rows = spill_rows
        self.group_rows = group_rows
        self.device = device
        self._owns_dir = spill_dir is None
        if spill_dir is None:
            # base configurable con GENOLY_SPILL_DIR (disco real, no tmpfs).
            # El prefijo incluye el PID del proceso: permite a la limpieza
            # periódica detectar directorios huérfanos de trabajadores
            # muertos (ver cleanup_orphan_spills).
            self._tmp = tempfile.TemporaryDirectory(
                prefix=f"genoly_kmer_{os.getpid()}_",
                dir=_env_spill_dir())
        else:
            self._tmp = None
        self.spill_dir = Path(spill_dir) if spill_dir else Path(self._tmp.name)
        self._buffer: List[List[Tuple[torch.Tensor, torch.Tensor]]] = [
            [] for _ in range(n_partitions)]
        self._buffered = [0] * n_partitions
        self._files: List[List[Tuple[Path, Path]]] = [
            [] for _ in range(n_partitions)]

    def add(self, values: torch.Tensor, counts: torch.Tensor) -> None:
        """
        Reparte un micro-lote de k-mers únicos (GPU o CPU) en particiones.

        Los conteos se guardan en int32: un micro-lote no puede contener
        más de 2^31 instancias (está acotado por la VRAM del presupuesto),
        y el merge final los eleva a int64.

        ``argsort(pid, stable=True)`` es OBLIGATORIO: un sort inestable
        reordenaría los valores dentro de cada partición y los runs del
        derrame dejarían de estar ordenados (la fusión final O(n) los
        exige ordenados).

        Fase 3: en lugar de un ``.cpu()`` sincronizante por partición
        (256 transferencias D2H por micro-lote), se hace UN único D2H
        del array ya ordenado (valores int64 + conteos int32) y el
        troceado por partición se hace en CPU con numpy (bincount sobre
        los valores ordenados: ``valor & (P-1)`` es el pid de cada fila).

        Las rebanadas se COPAN al guardarlas en el buffer: una vista numpy
        retiene vivo el array D2H padre (``sv``/``sc``, que puede ocupar
        varios GB por micro-lote). Sin la copia, los buffers de 256
        particiones mantienen decenas de arrays padre a la vez y el pico de
        RAM crece sin relación con las filas realmente retenidas (causó
        OOM al contar dos genomas completos con k=21). El coste de la copia
        es O(spill_rows * 12 B) por partición.
        """
        if values.numel() == 0:
            return

        order = torch.argsort(values & (self.n_partitions - 1),
                              stable=True)
        sv = values[order].cpu().numpy()
        sc = counts[order].to(torch.int32).cpu().numpy()
        sizes = np.bincount(sv & (self.n_partitions - 1),
                            minlength=self.n_partitions)

        offset = 0
        for p, size in enumerate(sizes.tolist()):
            if size == 0:
                continue
            # copias independientes: liberan sv/sc al terminar el reparto
            self._buffer[p].append((sv[offset:offset + size].copy(),
                                    sc[offset:offset + size].copy()))
            offset += int(size)

            self._buffered[p] += int(size)
            if self._buffered[p] >= self.spill_rows:
                self._spill(p)

    def _spill(self, p: int) -> None:
        # cada slice bufferizada es su propio fichero: las slices de
        # micro-lotes distintos NO forman un run ordenado si se
        # concatenan (los rangos de valores se intercalan)
        for v, c in self._buffer[p]:
            base = self.spill_dir / f"p{p}_{len(self._files[p])}"
            np.save(base.with_suffix(".v.npy"), v)
            np.save(base.with_suffix(".c.npy"), c)
            self._files[p].append((base.with_suffix(".v.npy"),
                                   base.with_suffix(".c.npy")))
        self._buffer[p].clear()
        self._buffered[p] = 0

    def finalize(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Cede (valores únicos, conteos) agregados, por grupos de
        particiones consecutivas (no en orden global), en el DISPOSITIVO
        del acumulador (CPU solo cuando ``device`` es None).

        La deduplicación se hace en el dispositivo (``torch.unique`` +
        ``index_add_``). Fase 3: se acumulan particiones consecutivas
        sin rebasar ``group_rows`` filas y se deduplica cada grupo en
        una sola llamada (antes: una llamada por partición, ~52 ms de
        H2D+unique+D2H cada una). Es exacto porque cada grupo contiene
        particiones COMPLETAS: dos copias del mismo k-mer caen siempre
        en la misma partición, y el resultado de ``torch.unique`` es
        global del grupo. El consumidor solo suma estadísticas y top-k,
        así que el orden entre cedas no afecta al resultado final.

        Para grupos mayores que ``group_rows`` (particiones sesgadas o
        genomas enormes) el grupo se procesa por trozos y los trozos
        resultantes (ordenados y únicos) se fusionan en CPU con
        :meth:`_merge_two_runs`; en ese caso la ceda del grupo vuelve
        como tensores CPU.

        El pico de RAM es O(filas_del_grupo) y el de VRAM
        O(min(filas_grupo, group_rows) * ~44 B).
        """
        try:
            mmaps: List[np.ndarray] = []
            runs: List[Tuple[np.ndarray, np.ndarray]] = []
            pending: List[int] = []
            group_rows_sum = 0

            def close_mmaps() -> None:
                # Windows bloquea los ficheros con mmaps abiertos: hay
                # que cerrarlos antes de que _release_all borre los
                # derrames. Los arrays cedidos siempre tienen datos
                # propios (salida de concatenate/merge), nunca vistas de
                # mmap, así que cerrarlos tras procesar el grupo es seguro.
                for arr in mmaps:
                    mmap = getattr(arr, "_mmap", None)
                    if mmap is not None:
                        mmap.close()
                mmaps.clear()

            def clear_pending() -> None:
                for q in pending:
                    self._buffer[q].clear()
                    self._buffered[q] = 0
                pending.clear()

            def flush() -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
                nonlocal runs, pending, group_rows_sum
                try:
                    yield from self._dedupe_runs(runs)
                finally:
                    close_mmaps()
                    clear_pending()
                    runs = []
                    pending = []
                    group_rows_sum = 0

            for p in range(self.n_partitions):
                if not self._buffer[p] and not self._files[p]:
                    continue

                # abrir los derrames de p (mmap perezoso: solo cabecera),
                # AUN sin registrarlos: si p desborda el grupo, el flush
                # de abajo cierra los mmaps del grupo ANTERIOR y los de p
                # deben sobrevivirle (leer un mmap cerrado = access
                # violation en Windows)
                nuevos_mmaps: List[np.ndarray] = []
                nuevos: List[Tuple[np.ndarray, np.ndarray]] = []
                for path_v, path_c in self._files[p]:
                    # mmap copy-on-write: escribible para torch (evita el
                    # aviso de arrays read-only y las copias preventivas),
                    # con lecturas zero-copy paginadas desde disco
                    av = np.load(path_v, mmap_mode="c")
                    ac = np.load(path_c, mmap_mode="c")
                    nuevos_mmaps.extend((av, ac))
                    nuevos.append((av, ac))
                # cada slice bufferizada es su propio run (ordenada
                # dentro de si misma; concatenarlas no lo estaria)
                for v_t, c_t in self._buffer[p]:
                    nuevos.append((v_t, c_t))
                p_rows = sum(int(r[0].shape[0]) for r in nuevos)

                # flush ANTES de añadir p si desbordaría el grupo: así
                # cada grupo cabe en un único trozo de GPU (sin fusión)
                if runs and group_rows_sum + p_rows > self.group_rows:
                    yield from flush()

                mmaps.extend(nuevos_mmaps)
                runs.extend(nuevos)
                pending.append(p)
                group_rows_sum += p_rows

            if runs:
                yield from flush()
        finally:
            close_mmaps()
            self._release_all()

    def _dedupe_runs(
            self, runs: List[Tuple[np.ndarray, np.ndarray]]
            ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Deduplica una lista de runs (ordenados individualmente) en el
        dispositivo por trozos de ``group_rows`` filas, fusionando los
        resultados parciales con :meth:`_merge_two_runs` y cediendo el
        resultado global del grupo (ordenado y único) en el dispositivo.

        Con un solo trozo (el caso normal, grupo ``<= group_rows``) el
        resultado no baja de la GPU; con varios trozos la fusión se hace
        en CPU y la ceda del grupo es un tensor CPU.
        """
        chunks = list(self._chunks_of_rows(runs, self.group_rows))
        if len(chunks) == 1:
            # montar el lote DIRECTAMENTE en GPU (un H2D por run, a un
            # buffer prealocado): evita el concatenate en RAM (~500
            # copias dispersas + 384 MB intermedios) y el doble toque de
            # los derrames en disco
            chunk = chunks[0]
            total = sum(int(r[0].shape[0]) for r in chunk)
            tv = torch.empty(total, dtype=torch.int64, device=self.device)
            tc = torch.empty(total, dtype=torch.int32, device=self.device)
            pos = 0
            for av, ac in chunk:
                n = int(av.shape[0])
                tv[pos:pos + n].copy_(torch.from_numpy(np.asarray(av)))
                tc[pos:pos + n].copy_(torch.from_numpy(np.asarray(ac)))
                pos += n
            uv, inverse = torch.unique(tv, return_inverse=True)
            uc = torch.zeros(
                uv.numel(), dtype=torch.int64,
                device=self.device).index_add_(
                    0, inverse, tc.to(torch.int64))
            del tv, tc, inverse
            yield uv, uc
            return

        out_v: Optional[np.ndarray] = None
        out_c: Optional[np.ndarray] = None
        for chunk in chunks:
            v = np.concatenate([r[0] for r in chunk])
            c = np.concatenate([r[1] for r in chunk])
            tv = torch.from_numpy(v).to(self.device)
            tc = torch.from_numpy(c).to(self.device)
            uv, inverse = torch.unique(tv, return_inverse=True)
            uc = torch.zeros(
                uv.numel(), dtype=torch.int64,
                device=self.device).index_add_(
                    0, inverse, tc.to(torch.int64))
            del tv, tc, inverse
            chunk_out = (uv.cpu().numpy(), uc.cpu().numpy())
            del uv, uc
            if out_v is None:
                out_v, out_c = chunk_out
            else:
                out_v, out_c = self._merge_two_runs(
                    out_v, out_c, chunk_out[0], chunk_out[1])
        if out_v is not None:
            yield (torch.from_numpy(np.ascontiguousarray(out_v)),
                   torch.from_numpy(np.ascontiguousarray(out_c)))

    @staticmethod
    def _chunks_of_rows(runs: List[Tuple[np.ndarray, np.ndarray]],
                        max_rows: int
                        ) -> Iterator[List[Tuple[np.ndarray, np.ndarray]]]:
        """Agrupa runs en lotes cuyo concatenado no supere ``max_rows``."""
        chunk: List[Tuple[np.ndarray, np.ndarray]] = []
        rows = 0
        for run in runs:
            size = int(run[0].shape[0])
            if chunk and rows + size > max_rows:
                yield chunk
                chunk = []
                rows = 0
            chunk.append(run)
            rows += size
        if chunk:
            yield chunk

    @staticmethod
    def _merge_two_runs(av: np.ndarray, ac: np.ndarray,
                        bv: np.ndarray, bc: np.ndarray,
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fusiona dos runs ordenados y sin duplicados internos, sumando los
        conteos de los valores presentes en ambos. O(n_a + n_b) con
        búsqueda binaria e inserción vectorizada (sin re-ordenar).
        """
        if av.size == 0:
            return bv, bc
        if bv.size == 0:
            return av, ac

        pos = np.searchsorted(av, bv)
        safe = np.minimum(pos, av.size - 1)
        dup = av[safe] == bv

        # conteos de los duplicados repartidos en las posiciones de a
        add = np.bincount(pos[dup], weights=bc[dup].astype(np.float64),
                          minlength=av.size)
        a_counts = ac + add.astype(np.int64)

        keep = ~dup
        nb = int(np.count_nonzero(keep))
        out_v = np.empty(av.size + nb, dtype=np.int64)
        out_c = np.empty(av.size + nb, dtype=np.int64)
        if nb:
            # cada b conservado se inserta en su posicion de a desplazada
            # por los b anteriores ya insertados (pos es no-decreciente)
            slots = pos[keep] + np.arange(nb, dtype=np.int64)
            mask = np.ones(out_v.size, dtype=bool)
            mask[slots] = False
            out_v[mask] = av
            out_c[mask] = a_counts
            out_v[~mask] = bv[keep]
            out_c[~mask] = bc[keep]
        else:
            out_v[:] = av
            out_c[:] = a_counts
        return out_v, out_c

    def _release_all(self) -> None:
        for files in self._files:
            for path_v, path_c in files:
                path_v.unlink(missing_ok=True)
                path_c.unlink(missing_ok=True)
        self._files = [[] for _ in range(self.n_partitions)]
        self._buffer = [[] for _ in range(self.n_partitions)]
        self._buffered = [0] * self.n_partitions

    def close(self) -> None:
        """Elimina los ficheros de derrame (y el directorio, si es propio)."""
        self._release_all()
        if self._owns_dir and self._tmp is not None:
            self._tmp.cleanup()
        # con spill_dir externo solo se eliminan los ficheros, no el dir


class _SourceCounter:
    """
    Proxy de iteración que cuenta registros y bases de la fuente a medida
    que se consumen, para el reporte de progreso del streaming.
    """

    __slots__ = ("_iter", "records", "bases")

    def __init__(self, records: Iterable):
        self._iter = iter(records)
        self.records = 0
        self.bases = 0

    def __iter__(self) -> "_SourceCounter":
        return self

    def __next__(self) -> str:
        record = next(self._iter)
        self.records += 1
        self.bases += len(record)
        return record.sequence


class KmerCounter:
    """
    Conteo de k-mers y análisis de espectro acelerado por GPU.

    Codifica cada k-mer como un entero en base 4 (A=0, C=1, G=2, T=3)
    mediante aritmética entera vectorizada (Horner) en int64, exacta para
    todo k <= 31, sobre el dispositivo detectado.

    La capa de streaming (`count_stream` / `count_records` / `count_fasta`)
    procesa fuentes arbitrariamente grandes con RAM y VRAM acotadas:

    - RAM: el FASTA se lee por bloques de disco (64 KiB) y las unidades
      (registros o ventanas) se agrupan en lotes de `ram_batch_size`.
    - VRAM: cada lote se divide en micro-lotes adaptativos según la
      memoria libre (`VRAMManager.plan_micro_batches`) y, tras cada
      micro-lote, se fuerza `gc.collect()` y `torch.cuda.empty_cache()`.
    - Agregación: los conteos parciales de `torch.unique` se fusionan en
      CPU, compactando cuando superan `MERGE_THRESHOLD_ELEMENTS`.
    """

    def __init__(self, device: Optional[str] = None):
        """
        Args:
            device: 'cuda', 'cpu' o None para auto-detectar.
        """
        self.manager = DeviceManager(device)
        self.device = self.manager.device
        self.encoder = SequenceEncoder(device)

    # ------------------------------------------------------------------ #
    # Codificación de k-mers (memoria optimizada)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_k(k: int) -> None:
        if k < 1:
            raise ValueError("k debe ser >= 1")
        if k > 31:
            raise ValueError("k > 31 no cabe en enteros de 64 bits")

    def _horner_words(self, digits: torch.Tensor, n_ventanas: int,
                      k: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Horner base-4 sobre dígitos ya en el dispositivo (int32 o int64),
        devolviendo una palabra (k <= 15), dos (16 <= k <= 30) o int64
        (k = 31). Ver :class:`_KmerWords` para la composición del código.
        """
        shape = (digits.shape[0], n_ventanas)
        device = digits.device

        if k <= INT32_DIGITS:
            lo = torch.zeros(shape, dtype=torch.int32, device=device)
            for j in range(k):
                lo = lo * 4 + digits[:, j:j + n_ventanas]
            return lo, None

        if k <= 2 * INT32_DIGITS:
            h = k - INT32_DIGITS
            lo = torch.zeros(shape, dtype=torch.int32, device=device)
            hi = torch.zeros(shape, dtype=torch.int32, device=device)
            for j in range(k):
                sl = digits[:, j:j + n_ventanas]
                if j < h:
                    hi = hi * 4 + sl
                else:
                    lo = lo * 4 + sl
            return lo, hi

        # k = 31: 62 bits no caben en dos int32 con signo; Horner int64
        digits64 = digits.to(torch.int64)
        codes = torch.zeros(shape, dtype=torch.int64, device=device)
        for j in range(k):
            codes = codes * 4 + digits64[:, j:j + n_ventanas]
        return codes, None

    def _encode_kmers(self, sequences: List[str],
                      k: int) -> _KmerWords:
        """
        Codifica todos los k-mers de un lote de secuencias (ruta por
        cadenas; la ruta numérica usa :meth:`_count_codes_micro_batch`).

        Los códigos se construyen con Horner en **int32** (tasa plena en
        GPU; int64 mueve el doble de bytes por operación y rinde ~2x peor)
        repartiendo los dígitos base-4 en una o dos palabras.

        Las ventanas con dígitos no canónicos (N/IUPAC) producen códigos
        basura pero van enmascaradas por ``valid``: nunca se cuentan.
        Memoria pico por base: máscara bool + digits int32 + palabras
        int32 + transitorios de Horner (ver ``estimate_kmer_bytes_per_base``).

        Args:
            sequences: Lista de secuencias (todas de longitud >= k).
            k: Longitud del k-mer.

        Returns:
            :class:`_KmerWords` con lo/hi (B, W) y la máscara válida.
        """
        self._check_k(k)
        if not sequences:
            raise ValueError("La lista de secuencias está vacía")

        encoded, lengths = self.encoder.encode(sequences)

        # Validez por dígito (bool): una ventana es válida si sus k
        # dígitos lo son; se acumula con AND sobre vistas.
        valid_digits = encoded < 4

        n_ventanas = max(int((lengths - k + 1).max()), 0)
        digits32 = encoded.to(torch.int32)
        del encoded

        lo, hi = self._horner_words(digits32, n_ventanas, k)
        del digits32
        words = _KmerWords(lo, hi, None)

        valid = valid_digits[:, 0:n_ventanas].clone()
        for j in range(1, k):
            valid &= valid_digits[:, j:j + n_ventanas]
        del valid_digits

        # Ventanas que caben dentro de la longitud real de cada secuencia
        lens = (lengths - k + 1).clamp(min=0)
        pos = torch.arange(n_ventanas, device=lo.device).unsqueeze(0)
        valid &= pos < lens.unsqueeze(1)

        return words._replace(valid=valid)

    def _reverse_complement_codes(self, codes: torch.Tensor,
                                  k: int) -> torch.Tensor:
        """
        Reverse complement de códigos int64 (ruta k = 31), en streaming
        por operaciones de bits: extrae los dígitos base-4 uno a uno (del
        menos significativo al más significativo) y reconstruye el código
        del revcomp sobre la marcha, con consumo extra O(1) tensores.

        Los dígitos de un código válido (0 <= code < 4^k) se obtienen con
        ``(code >> 2j) & 3``; para códigos inválidos (ventanas con N) el
        resultado es basura acotada a [0, 4^k), pero esas ventanas van
        enmascaradas por la máscara de validez y nunca se cuentan.
        """
        rc = torch.zeros_like(codes)
        for j in range(k):
            digit = (codes >> (2 * j)) & 3
            rc = rc * 4 + (3 - digit)
        return rc

    def _revcomp_words(self, words: _KmerWords,
                       k: int) -> _KmerWords:
        """
        Reverse complement de los códigos en palabras int32.

        Con lo = últimos 15 dígitos y hi = los restantes, el revcomp del
        k-mer (dígitos d_{k-1}..d_0 complementados e invertidos) se
        reconstruye así:

        - ``rc_lo`` (15 dígitos menos significativos del revcomp) son los
          dígitos globales d_14..d_0 complementados: se extraen en orden
          MSB->LSB, primero los que viven en ``lo`` (d_14..d_h) y luego
          los de ``hi`` (d_{h-1}..d_0).
        - ``rc_hi`` (h dígitos) son d_{k-1}..d_{k-h} complementados: los
          h dígitos menos significativos de ``lo``, extraídos LSB->MSB.

        Todo con operaciones de bits en streaming (pico O(1) tensores).
        Para ventanas inválidas el resultado es basura enmascarada.
        """
        lo, hi = words.lo, words.hi

        if hi is None:
            if lo.dtype == torch.int64:  # ruta k = 31
                return _KmerWords(
                    self._reverse_complement_codes(lo, k), None, words.valid)
            rc_lo = torch.zeros_like(lo)
            for j in range(k):
                digit = (lo >> (2 * j)) & 3
                rc_lo = rc_lo * 4 + (3 - digit)
            return _KmerWords(rc_lo, None, words.valid)

        h = k - INT32_DIGITS
        rc_lo = torch.zeros_like(lo)
        # dígitos d_14..d_h (viven en lo, de MSB a LSB entre ellos)
        for j in range(INT32_DIGITS - 1, h - 1, -1):
            digit = (lo >> (2 * (INT32_DIGITS - 1 + h - j))) & 3
            rc_lo = rc_lo * 4 + (3 - digit)
        # dígitos d_{h-1}..d_0 (viven en hi, del LSB al MSB:
        # (hi >> 2i) & 3 = d_{h-1-i})
        for i in range(h):
            digit = (hi >> (2 * i)) & 3
            rc_lo = rc_lo * 4 + (3 - digit)

        # rc_hi: últimos h dígitos del k-mer complementados (LSB de lo)
        rc_hi = torch.zeros_like(hi)
        for i in range(h):
            digit = (lo >> (2 * i)) & 3
            rc_hi = rc_hi * 4 + (3 - digit)

        return _KmerWords(rc_lo, rc_hi, words.valid)

    @staticmethod
    def _min_words(a: _KmerWords, b: _KmerWords) -> _KmerWords:
        """
        Mínimo canónico lexicográfico de dos códigos en palabras:
        compara ``hi`` y, en empate, ``lo`` (equivale a comparar el
        código global en int64 sin materializarlo).
        """
        if a.hi is None:
            return a._replace(lo=torch.minimum(a.lo, b.lo))

        min_hi = torch.minimum(a.hi, b.hi)
        min_lo = torch.where(
            a.hi < b.hi, a.lo,
            torch.where(a.hi > b.hi, b.lo, torch.minimum(a.lo, b.lo)))
        return _KmerWords(min_lo, min_hi, a.valid)

    @staticmethod
    def _combine_words(words: _KmerWords) -> torch.Tensor:
        """Combina las palabras en el código global int64 (una sola vez
        por micro-lote, para ``unique`` y el acumulador)."""
        if words.hi is None:
            if words.lo.dtype == torch.int64:
                return words.lo
            return words.lo.to(torch.int64)
        return words.hi.to(torch.int64).mul_(4 ** INT32_DIGITS) \
            .add_(words.lo.to(torch.int64))

    # ------------------------------------------------------------------ #
    # Conteo en memoria (API clásica)
    # ------------------------------------------------------------------ #
    def count(self, sequences: List[str], k: int,
              canonical: bool = True,
              min_abundance: int = 1,
              chunk_size: Optional[int] = None,
              window_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cuenta la frecuencia de cada k-mer en las secuencias.

        Args:
            sequences: Lista de secuencias.
            k: Longitud del k-mer.
            canonical: Si True, cuenta cada k-mer y su reverse complement
                      como una sola entidad (usa el código mínimo).
            min_abundance: Frecuencia mínima para incluir un k-mer.
            chunk_size: Procesar en lotes de este tamaño para limitar memoria.
            window_size: Máximo de bases por ventana. Las secuencias más
                         largas se parten en ventanas con solape de k-1
                         bases para limitar el consumo de VRAM/RAM al
                         procesar genomas completos. ``None`` o ``0`` =
                         sin ventanado (como el resto de la API);
                         ``> 0`` = ese tamaño.

        Returns:
            Tuple (valores únicos, conteos) ordenados por frecuencia
            descendente (tensores CPU).
        """
        if not sequences:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

        # 0 y None significan "sin ventanado", igual que en count_stream
        if window_size:
            expanded = []
            for seq in sequences:
                expanded.extend(self.split_sequence_windows(seq, k, window_size))
            sequences = expanded

        # Las secuencias más cortas que k no aportan ningún k-mer y romperían
        # la convolución (kernel > entrada)
        sequences = [s for s in sequences if len(s) >= k]
        if not sequences:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

        if chunk_size is None:
            chunk_size = len(sequences)

        # Este camino no planifica micro-lotes: si un lote no cabe en la
        # VRAM libre, en lugar de morir se parte el chunk por la mitad y
        # se reintenta (con la caché liberada) hasta que quepa.
        manager = (VRAMManager(self.device)
                   if getattr(self.device, "type", None) == "cuda" else None)

        all_values = []
        all_counts = []

        size = max(1, int(chunk_size))
        start = 0
        while start < len(sequences):
            chunk = sequences[start:start + size]
            try:
                values, counts = self._count_micro_batch(chunk, k, canonical)
            except Exception as exc:
                if manager is None or len(chunk) < 2 or not is_cuda_oom(exc):
                    raise
                size = max(1, len(chunk) // 2)
                manager.release()
                logger.warning(
                    "CUDA sin memoria con %d secuencias; se parte el lote "
                    "y se reintenta con %d", len(chunk), size)
                continue
            if values is not None:
                # Acumular en CPU acota la VRAM al micro-lote actual: en GPU
                # el pico llegaba a ~3x el tamaño final (chunks + cat + sort).
                all_values.append(values.to("cpu"))
                all_counts.append(counts.to("cpu"))
            start += len(chunk)

        if not all_values:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

        values = torch.cat(all_values)
        counts = torch.cat(all_counts)

        # Agregar conteos de k-mers repetidos entre chunks
        values, counts = self._aggregate(values, counts)

        # Filtrar por abundancia mínima
        keep = counts >= min_abundance
        values, counts = values[keep], counts[keep]

        # Ordenar por frecuencia descendente
        order = torch.argsort(counts, descending=True)
        return values[order].cpu(), counts[order].cpu()

    # ------------------------------------------------------------------ #
    # Conteo en streaming (RAM y VRAM acotadas)
    # ------------------------------------------------------------------ #
    def _count_micro_batch(self, sequences: List[str], k: int,
                           canonical: bool
                           ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Contea los k-mers de un micro-lote y devuelve (valores, conteos)
        en el dispositivo (únicos, ordenados), o (None, None) si no hay
        k-mers válidos.

        El cómputo pesado (Horner, revcomp) se hace en int32; el código
        global se combina a int64 una única vez, justo antes del
        ``unique``.
        """
        words = self._encode_kmers(sequences, k)

        if canonical:
            rc = self._revcomp_words(words, k)
            words = self._min_words(words, rc)
            del rc

        codes = self._combine_words(words)
        valid = words.valid
        del words
        flat = codes[valid]
        del codes, valid
        if flat.numel() == 0:
            return None, None

        values, counts = torch.unique(flat, return_counts=True)
        del flat
        return values, counts

    def _count_codes_micro_batch(self, codes_u8: torch.Tensor, k: int,
                                 canonical: bool
                                 ) -> Tuple[Optional[torch.Tensor],
                                            Optional[torch.Tensor]]:
        """
        Conteo de k-mers sobre un micro-lote **ya codificado** en el
        dispositivo (ruta numérica; ver
        :meth:`FastaReader.iter_windows_codes`).

        Args:
            codes_u8: (n, L) uint8 en el dispositivo con 0..3 = A/C/G/T y
                255 = dígito inválido. El padding de 255 en las colas
                cortas reproduce la semántica "ventana dentro del
                registro": cualquier ventana que toque padding contiene
                un dígito inválido y va enmascarada, así que no hace
                falta la máscara posicional de la ruta por cadenas.
            k: Longitud del k-mer.
            canonical: Contear cada k-mer junto a su reverse complement.
        """
        self._check_k(k)
        L = codes_u8.shape[1]
        n_ventanas = L - k + 1
        if n_ventanas <= 0:
            return None, None

        digits32 = codes_u8.to(torch.int32)
        valid_digits = digits32 < 4

        valid = valid_digits[:, 0:n_ventanas].clone()
        for j in range(1, k):
            valid &= valid_digits[:, j:j + n_ventanas]
        del valid_digits

        lo, hi = self._horner_words(digits32, n_ventanas, k)
        del digits32
        words = _KmerWords(lo, hi, valid)

        if canonical:
            rc = self._revcomp_words(words, k)
            words = self._min_words(words, rc)
            del rc

        codes = self._combine_words(words)
        del words
        flat = codes[valid]
        del codes, valid
        if flat.numel() == 0:
            return None, None

        values, counts = torch.unique(flat, return_counts=True)
        del flat
        return values, counts

    def _stream_codes_micro_batches(self,
                                    windows: Iterable[np.ndarray],
                                    k: int,
                                    canonical: bool,
                                    ram_batch_size: int,
                                    window_size: int,
                                    manager: VRAMManager,
                                    on_progress: Optional[Callable[[dict], None]],
                                    ) -> Iterator[Tuple[Optional[torch.Tensor],
                                                        Optional[torch.Tensor]]]:
        """
        Driver del pipeline numérico: agrupa ventanas de códigos (arrays
        uint8) en lotes de RAM, planifica micro-lotes de GPU y los sube
        en un único tensor uint8 por micro-lote (8x menos bytes de H2D
        que la ruta por cadenas; el padding a 255 marca los dígitos
        inválidos).
        """
        stats = {"units_done": 0, "bases_done": 0, "micro_batches": 0,
                 "window_size": int(window_size)}
        micro_index = 0  # persiste entre lotes de RAM para release_every

        def emit_progress() -> None:
            if on_progress is not None:
                info = dict(stats)
                info["stage"] = "kmer"
                on_progress(info)

        def process(batch: List[np.ndarray]
                    ) -> Iterator[Tuple[Optional[torch.Tensor],
                                        Optional[torch.Tensor]]]:
            nonlocal micro_index
            lengths = [a.size for a in batch]
            for group in manager.plan_micro_batches(lengths,
                                                    CODES_BYTES_PER_BASE):
                l_max = max(lengths[i] for i in group)
                codes = np.full((len(group), l_max), FASTA_CODE_INVALID,
                                dtype=np.uint8)
                for r, i in enumerate(group):
                    a = batch[i]
                    codes[r, :a.size] = a
                stats["micro_batches"] += 1
                stats["units_done"] += len(group)
                stats["bases_done"] += sum(lengths[i] for i in group)
                def _run_batch(codes=codes):
                    codes_t = torch.from_numpy(codes).to(self.device)
                    try:
                        return self._count_codes_micro_batch(
                            codes_t, k, canonical)
                    finally:
                        del codes_t

                values, counts = retry_on_oom(_run_batch, manager=manager)
                del codes
                micro_index += 1
                if manager.release_after_each and (
                        micro_index % manager.release_every == 0):
                    manager.release()
                emit_progress()
                yield values, counts

        batch: List[np.ndarray] = []
        for arr in windows:
            if arr.size >= k:  # colas sin k-mers: descartadas
                batch.append(arr)
                if len(batch) >= ram_batch_size:
                    yield from process(batch)
                    batch = []
        if batch:
            yield from process(batch)

    def _stream_micro_batches(self,
                              sequences: Iterable[str],
                              k: int,
                              canonical: bool,
                              ram_batch_size: int,
                              window_size: Optional[int],
                              manager: VRAMManager,
                              on_progress: Optional[Callable[[dict], None]],
                              ) -> Iterator[Tuple[Optional[torch.Tensor],
                                                  Optional[torch.Tensor]]]:
        """
        Generador compartido del pipeline de streaming: consume unidades
        (secuencias/ventanas), las agrupa en lotes de RAM, planifica
        micro-lotes de GPU y cede (valores, conteos) por micro-lote.

        El diccionario de progreso incluye ``window_size`` efectiva.
        """
        bytes_per_base = estimate_kmer_bytes_per_base(k, canonical)

        if window_size == 0:
            units: Iterable[str] = sequences
        else:
            if window_size is None:
                window_size = manager.suggest_window_bases(bytes_per_base)
            units = self._iter_windowed(sequences, k, window_size)

        stats = {"units_done": 0, "bases_done": 0, "micro_batches": 0,
                 "window_size": int(window_size or 0)}
        micro_index = 0  # persiste entre lotes de RAM para release_every

        def emit_progress() -> None:
            if on_progress is not None:
                info = dict(stats)
                info["stage"] = "kmer"
                on_progress(info)

        def process(batch: List[str]) -> Iterator[Tuple[
                Optional[torch.Tensor], Optional[torch.Tensor]]]:
            nonlocal micro_index
            lengths = [len(u) for u in batch]
            for group in manager.plan_micro_batches(lengths, bytes_per_base):
                seqs = [batch[i] for i in group]
                values, counts = retry_on_oom(
                    lambda s=seqs: self._count_micro_batch(s, k, canonical),
                    manager=manager)
                stats["micro_batches"] += 1
                stats["units_done"] += len(group)
                stats["bases_done"] += sum(lengths[i] for i in group)
                micro_index += 1
                if manager.release_after_each and (
                        micro_index % manager.release_every == 0):
                    manager.release()
                emit_progress()
                yield values, counts

        batch: List[str] = []
        for unit in units:
            if len(unit) >= k:
                batch.append(unit)
                if len(batch) >= ram_batch_size:
                    yield from process(batch)
                    batch = []
        if batch:
            yield from process(batch)

    def count_stream(self,
                     sequences: Iterable[str],
                     k: int,
                     canonical: bool = True,
                     min_abundance: int = 1,
                     ram_batch_size: int = DEFAULT_RAM_BATCH_SIZE,
                     window_size: Optional[int] = None,
                     vram: Optional[VRAMManager] = None,
                     on_progress: Optional[Callable[[dict], None]] = None,
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Conteo de k-mers en streaming sobre un iterable perezoso de
        secuencias, con RAM y VRAM acotadas independientemente del tamaño
        de la fuente.

        Niveles de batching:

        1. RAM: las unidades (secuencias o ventanas) se agrupan en lotes
           de ``ram_batch_size``.
        2. VRAM: cada lote se divide en micro-lotes adaptativos con
           ``VRAMManager.plan_micro_batches`` usando el coste
           ``estimate_kmer_bytes_per_base(k, canonical)``; tras cada
           ``release_every``-ésimo micro-lote se fuerza ``gc.collect()``
           y ``torch.cuda.empty_cache()``.
        3. Agregación: los pares (valores, conteos) parciales se fusionan
           en CPU y se compactan al superar ``MERGE_THRESHOLD_ELEMENTS``.

        Límite de memoria: el resultado completo vive en RAM
        (``total_kmers_unicos * 16 B`` aprox.). Para fuentes con millones
        de k-mers únicos (genomas completos) usa
        :meth:`count_fasta_aggregated`, que acota la RAM derramando a
        disco y devuelve solo estadísticas.

        Args:
            sequences: Iterable perezoso de secuencias (no una lista
                completa); p. ej. ``FastaReader(path).iter_sequences()``.
            k: Longitud del k-mer (1 <= k <= 31).
            canonical: Contear cada k-mer junto a su reverse complement.
            min_abundance: Frecuencia mínima del k-mer final.
            ram_batch_size: Unidades por lote de RAM.
            window_size: Bases por ventana para secuencias largas.
                ``None`` = sugerida automáticamente por el presupuesto de
                VRAM (recomendado); ``0`` = sin ventanado; ``> 0`` = ese
                tamaño. Las ventanas usan solape de k-1 bases, por lo que
                el conteo agregado es idéntico al de la secuencia entera.
            vram: Gestor de VRAM; por defecto uno sobre el dispositivo.
            on_progress: Callback ``fn(info: dict)`` invocado tras cada
                micro-lote con ``{"stage", "units_done", "bases_done",
                "micro_batches", "window_size"}``.

        Returns:
            Tuple (valores únicos, conteos) ordenados por frecuencia
            descendente (tensores CPU).
        """
        self._check_k(k)
        if ram_batch_size < 1:
            raise ValueError("ram_batch_size debe ser >= 1")

        manager = vram if vram is not None else VRAMManager(self.device)

        parts_values: List[torch.Tensor] = []
        parts_counts: List[torch.Tensor] = []
        part_elems = 0

        def merge_if_needed() -> None:
            nonlocal part_elems, parts_values, parts_counts
            if part_elems <= MERGE_THRESHOLD_ELEMENTS:
                return
            values = torch.cat(parts_values)
            counts = torch.cat(parts_counts)
            values, counts = self._aggregate(values, counts)
            parts_values = [values]
            parts_counts = [counts]
            part_elems = int(values.numel())

        for values, counts in self._stream_micro_batches(
                sequences, k, canonical, ram_batch_size, window_size,
                manager, on_progress):
            if values is not None:
                parts_values.append(values.cpu())
                parts_counts.append(counts.cpu())
                part_elems += int(values.numel())
                merge_if_needed()

        if not parts_values:
            return (torch.tensor([], dtype=torch.long),
                    torch.tensor([], dtype=torch.long))

        values = torch.cat(parts_values)
        counts = torch.cat(parts_counts)
        values, counts = self._aggregate(values, counts)

        keep = counts >= min_abundance
        values, counts = values[keep], counts[keep]

        order = torch.argsort(counts, descending=True)
        return values[order].cpu(), counts[order].cpu()

    def count_fasta_aggregated(self,
                               path,
                               k: int,
                               canonical: bool = True,
                               min_abundance: int = 1,
                               top: int = 20,
                               ram_batch_size: int = DEFAULT_RAM_BATCH_SIZE,
                               window_size: Optional[int] = None,
                               vram: Optional[VRAMManager] = None,
                               on_progress: Optional[Callable[[dict], None]] = None,
                                n_partitions: int = N_PARTITIONS,
                                spill_rows: int = SPILL_ROWS_PER_PARTITION,
                                spill_dir: Optional[str] = None,
                                block_size: int = DEFAULT_BLOCK_SIZE,
                                release_every: int = 4,
                                ) -> Dict[str, object]:
        """Conteo de k-mers de un único FASTA; ver :meth:`count_fastas_aggregated`."""
        return self.count_fastas_aggregated(
            [path], k, canonical=canonical, min_abundance=min_abundance,
            top=top, ram_batch_size=ram_batch_size, window_size=window_size,
            vram=vram, on_progress=on_progress, n_partitions=n_partitions,
            spill_rows=spill_rows, spill_dir=spill_dir,
            block_size=block_size, release_every=release_every)

    def count_fastas_aggregated(self,
                                paths,
                                k: int,
                                canonical: bool = True,
                                min_abundance: int = 1,
                                top: int = 20,
                                ram_batch_size: int = DEFAULT_RAM_BATCH_SIZE,
                                window_size: Optional[int] = None,
                                vram: Optional[VRAMManager] = None,
                                on_progress: Optional[Callable[[dict], None]] = None,
                                 n_partitions: int = N_PARTITIONS,
                                 spill_rows: int = SPILL_ROWS_PER_PARTITION,
                                 spill_dir: Optional[str] = None,
                                 block_size: int = DEFAULT_BLOCK_SIZE,
                                 release_every: int = 4,
                                 ) -> Dict[str, object]:
        """
        Conteo de k-mers sobre **uno o varios** archivos FASTA con un único
        acumulador particionado: el resultado combinado es exacto (los
        k-mers compartidos entre archivos no se duplican).

        Pensado para datasets NCBI con varios ensamblajes del mismo genoma
        (p. ej. GCA + GCF): se procesan todos en una sola pasada por
        acumulador, con RAM, VRAM y disco acotadas, devolviendo solo
        estadísticas agregadas (ver :meth:`count_fasta_aggregated`).

        El progreso incluye ``file``/``file_index``/``files`` para saber qué
        ensamblaje se está procesando.

        Args:
            paths: Iterable de rutas a archivos FASTA.
            k: Longitud del k-mer (1 <= k <= 31).
            canonical: Contear cada k-mer junto a su reverse complement.
            min_abundance: Frecuencia mínima del k-mer.
            top: Número de k-mers más frecuentes a devolver.
            ram_batch_size: Ventanas por lote de RAM.
            window_size: Bases por ventana; ``None`` = sugerida por el
                presupuesto de VRAM, ``0`` = registros completos.
            vram: Gestor de VRAM; por defecto uno sobre el dispositivo.
            on_progress: Callback ``fn(info: dict)`` con el progreso.
            n_partitions: Particiones del acumulador (potencia de 2).
            spill_rows: Filas por partición antes de derramar a disco.
            spill_dir: Directorio de derrame.
            block_size: Tamaño del bloque de lectura de disco.
            release_every: Liberar VRAM/RAM cada N micro-lotes.

        Returns:
            Dict con ``k``, ``total_unique``, ``total_kmers``,
            ``spectrum`` y ``top_kmers``.
        """
        self._check_k(k)
        if top < 0:
            raise ValueError("top debe ser >= 0")
        if release_every < 1:
            raise ValueError("release_every debe ser >= 1")

        manager = vram if vram is not None else VRAMManager(
            self.device, release_every=release_every)
        accumulator = _PartitionedKmerAccumulator(
            n_partitions=n_partitions, spill_rows=spill_rows,
            spill_dir=spill_dir, device=self.device)

        total_unique = 0
        total_kmers = 0
        spectrum: Dict[int, int] = {}
        top_list: List[Tuple[int, int]] = []  # (conteo, código)

        # Ventana única para todos los archivos (teselado consistente).
        if window_size is None:
            window_size = manager.suggest_window_bases(
                estimate_kmer_bytes_per_base(k, canonical))
        if window_size != 0 and window_size < k:
            raise ValueError(
                "window_size debe ser >= k para la ruta numérica")
        effective_window = window_size

        try:
            paths = list(paths)
            n_files = len(paths)
            for idx, path in enumerate(paths):
                path = Path(path)
                fname = path.name

                def on_file(info: dict, _idx=idx, _name=fname,
                            _n=n_files) -> None:
                    if on_progress is not None:
                        on_progress({**info, "file": _name,
                                     "file_index": _idx, "files": _n})

                if window_size == 0:
                    # Registros completos como unidades (RAM acotada por el
                    # mayor registro del archivo): ruta por cadenas.
                    iterable: Iterable[str] = FastaReader(
                        path, block_size).iter_sequences()
                    for values, counts in self._stream_micro_batches(
                            iterable, k, canonical, ram_batch_size, 0,
                            manager, on_file):
                        if values is not None:
                            accumulator.add(values, counts)
                            del values, counts
                else:
                    # Ruta numérica: ventanas de códigos base-4 leídas por
                    # bloques de disco (sin cadenas de Python).
                    windows = FastaReader(path, block_size).iter_windows_codes(
                        window_size, overlap=k - 1)
                    for values, counts in self._stream_codes_micro_batches(
                            windows, k, canonical, ram_batch_size,
                            effective_window, manager, on_file):
                        if values is not None:
                            accumulator.add(values, counts)
                            del values, counts

            for u_v, u_c in accumulator.finalize():
                # finalize cede en el dispositivo (CPU en la fusión por
                # trozos): normalizar para que las estadísticas y el
                # top-k se calculen en GPU, no en CPU
                u_v = u_v.to(self.device)
                u_c = u_c.to(self.device)
                if min_abundance > 1:
                    keep = u_c >= min_abundance
                    u_v, u_c = u_v[keep], u_c[keep]
                if u_v.numel() == 0:
                    continue

                total_unique += int(u_v.numel())
                total_kmers += int(u_c.sum().item())

                mult, freq = torch.unique(u_c, return_counts=True)
                for m, f in zip(mult.tolist(), freq.tolist()):
                    spectrum[int(m)] = spectrum.get(int(m), 0) + int(f)

                if top > 0:
                    t = min(top, u_c.shape[0])
                    # topk no define el orden de los empates (arbitrario
                    # en CUDA) y con finalize agrupado ya no hay rondas
                    # intermedias que los re-ordenen: se usa argsort
                    # estable descendente, que deja los empates en orden
                    # de código ascendente (u_v viene ordenado de unique),
                    # el mismo contrato del re-ordenado de top_list.
                    top_idx = torch.argsort(-u_c, stable=True)[:t]
                    top_list.extend(zip(u_c[top_idx].tolist(),
                                        u_v[top_idx].tolist()))
                    top_list.sort(key=lambda x: (-x[0], x[1]))
                    del top_list[top:]
        finally:
            accumulator.close()

        top_kmers = [
            {"kmer": self.decode_kmer(code, k), "count": int(count)}
            for count, code in top_list
        ]

        return {
            "k": k,
            "total_unique": total_unique,
            "total_kmers": total_kmers,
            "spectrum": spectrum,
            "top_kmers": top_kmers,
        }

    def _iter_windowed(self, sequences: Iterable[str], k: int,
                       window_size: int) -> Iterator[str]:
        """
        Divide perezosamente cada secuencia en ventanas con solape de
        k-1 bases (ver :meth:`split_sequence_windows`).
        """
        step = max(1, window_size - k + 1)
        for seq in sequences:
            if len(seq) <= window_size:
                yield seq
                continue
            for pos in range(0, len(seq), step):
                yield seq[pos:pos + window_size]

    def count_records(self,
                      records: Iterable,
                      k: int,
                      canonical: bool = True,
                      min_abundance: int = 1,
                      ram_batch_size: int = DEFAULT_RAM_BATCH_SIZE,
                      window_size: Optional[int] = None,
                      vram: Optional[VRAMManager] = None,
                      on_progress: Optional[Callable[[dict], None]] = None,
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Conteo de k-mers en streaming sobre un iterable de registros FASTA
        (:class:`~Genoly.io.fasta.FastaRecord`), con reporte de progreso
        de la fuente (registros y bases leídos).

        Args:
            records: Iterable perezoso de registros, p. ej.
                ``FastaReader(path).records()``.
            (resto de parámetros): iguales que :meth:`count_stream`.

        Returns:
            Tuple (valores únicos, conteos) ordenados por frecuencia
            descendente (tensores CPU).
        """
        source = _SourceCounter(records)

        def progress(info: dict) -> None:
            if on_progress is not None:
                payload = dict(info)
                payload["records"] = source.records
                payload["bases"] = source.bases
                on_progress(payload)

        return self.count_stream(
            source, k,
            canonical=canonical,
            min_abundance=min_abundance,
            ram_batch_size=ram_batch_size,
            window_size=window_size,
            vram=vram,
            on_progress=progress,
        )

    def count_fasta(self,
                    path,
                    k: int,
                    canonical: bool = True,
                    min_abundance: int = 1,
                    ram_batch_size: int = DEFAULT_RAM_BATCH_SIZE,
                    window_size: Optional[int] = None,
                    vram: Optional[VRAMManager] = None,
                    on_progress: Optional[Callable[[dict], None]] = None,
                    block_size: int = DEFAULT_BLOCK_SIZE,
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Conteo de k-mers sobre un archivo FASTA de cualquier tamaño con
        RAM y VRAM acotadas.

        A diferencia de :meth:`count_records`, las ventanas se generan
        directamente desde el flujo de bloques de disco
        (:meth:`~Genoly.io.fasta.FastaReader.iter_windows`), de modo que
        ni siquiera un registro de varios GB llega a materializarse en
        RAM. El consumo queda acotado por O(ram_batch_size * window_size)
        en RAM y por el presupuesto de VRAM del micro-lote.

        Args:
            path: Ruta del archivo FASTA.
            k: Longitud del k-mer (1 <= k <= 31).
            canonical: Contear cada k-mer junto a su reverse complement.
            min_abundance: Frecuencia mínima del k-mer final.
            ram_batch_size: Ventanas por lote de RAM.
            window_size: Bases por ventana; ``None`` = sugerida por el
                presupuesto de VRAM (recomendado), ``0`` = registros
                completos (la RAM queda acotada por el mayor registro).
            vram: Gestor de VRAM; por defecto uno sobre el dispositivo.
            on_progress: Callback ``fn(info: dict)`` con progreso.
            block_size: Tamaño del bloque de lectura de disco.

        Returns:
            Tuple (valores únicos, conteos) ordenados por frecuencia
            descendente (tensores CPU).
        """
        self._check_k(k)
        reader = FastaReader(path, block_size)

        if window_size == 0:
            return self.count_records(
                reader.records(), k,
                canonical=canonical,
                min_abundance=min_abundance,
                ram_batch_size=ram_batch_size,
                window_size=0,
                vram=vram,
                on_progress=on_progress,
            )

        if window_size is None:
            manager = vram if vram is not None else VRAMManager(self.device)
            window_size = manager.suggest_window_bases(
                estimate_kmer_bytes_per_base(k, canonical))

        windows = reader.iter_windows(window_size, overlap=k - 1)
        return self.count_stream(
            windows, k,
            canonical=canonical,
            min_abundance=min_abundance,
            ram_batch_size=ram_batch_size,
            window_size=0,
            vram=vram,
            on_progress=on_progress,
        )

    # ------------------------------------------------------------------ #
    # Utilidades
    # ------------------------------------------------------------------ #
    @staticmethod
    def _aggregate(values: torch.Tensor, counts: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Suma los conteos de valores duplicados entre chunks.

        Implementación sort-based (pico ~20 B/fila frente a los ~40 B de
        ``unique(return_inverse) + scatter_add``): ordena (valor, conteo)
        conjuntamente, marca los inicios de segmento de valores iguales y
        suma los conteos por segmento con ``index_add_``.
        """
        v, order = torch.sort(values)
        c = counts[order]
        del values, counts, order

        if v.numel() == 0:
            return v, c.to(torch.int64)

        # Todo debe vivir en el dispositivo de entrada: la ruta en RAM
        # (count) acumula en CPU para no retener VRAM, y la de streaming
        # cede los micro-lotes en CPU.
        device = v.device
        new = torch.empty(v.shape[0], dtype=torch.bool, device=device)
        new[0] = True
        if v.numel() > 1:
            new[1:] = v[1:] != v[:-1]

        u_v = v[new]
        seg = torch.cumsum(new.to(torch.int64), dim=0) - 1
        u_c = torch.zeros(u_v.shape[0], dtype=torch.int64, device=device)
        u_c.index_add_(0, seg, c.to(torch.int64))
        return u_v, u_c

    def decode_kmer(self, code: int, k: int) -> str:
        """Convierte un código entero de k-mer a su secuencia de texto."""
        digits = []
        for _ in range(k):
            digits.append(code % 4)
            code //= 4
        bases = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
        return ''.join(bases[d] for d in reversed(digits))

    @staticmethod
    def split_sequence_windows(sequence: str, k: int,
                               window_size: int) -> List[str]:
        """
        Divide una secuencia en ventanas con solape de k-1 bases.

        El solape garantiza que ningún k-mer de la secuencia original quede
        partido entre dos ventanas, por lo que los conteos agregados son
        idénticos a procesar la secuencia completa.
        """
        if window_size <= 0:
            raise ValueError("window_size debe ser >= 1")
        length = len(sequence)
        if length <= window_size:
            return [sequence]
        step = max(1, window_size - k + 1)
        windows = []
        pos = 0
        while pos < length:
            windows.append(sequence[pos:pos + window_size])
            pos += step
        return windows

    @staticmethod
    def spectrum_from_counts(counts: torch.Tensor) -> Dict[int, int]:
        """Espectro (multiplicidad -> nº de k-mers) a partir de conteos agregados."""
        if counts.numel() == 0:
            return {}
        multiplicities, freq = torch.unique(counts, return_counts=True)
        return {int(m): int(f) for m, f in zip(multiplicities.tolist(), freq.tolist())}

    @staticmethod
    def estimate_from_counts(counts: torch.Tensor) -> Tuple[float, float]:
        """Tamaño de genoma y cobertura a partir de conteos agregados (Lander-Waterman)."""
        if counts.numel() == 0:
            return 0.0, 0.0
        total_kmers = int(counts.sum().item())
        multiplicities, freq = torch.unique(counts, return_counts=True)
        peak_index = int(freq.argmax().item())
        coverage = float(multiplicities[peak_index].item())
        genome_size = total_kmers / coverage if coverage > 0 else 0.0
        return genome_size, coverage

    @staticmethod
    def estimate_from_spectrum(spectrum: Dict[int, int]) -> Tuple[float, float]:
        """
        Tamaño de genoma y cobertura a partir del espectro (Lander-Waterman).

        Equivalente a :meth:`estimate_from_counts` pero sin necesidad de
        materializar los conteos individuales: usable con el resultado de
        :meth:`count_fasta_aggregated`.

        Args:
            spectrum: Diccionario multiplicidad -> nº de k-mers.

        Returns:
            Tuple (tamaño estimado del genoma, cobertura media estimada).
        """
        if not spectrum:
            return 0.0, 0.0
        total_kmers = sum(mult * freq for mult, freq in spectrum.items())
        # la multiplicidad con más k-mers (primera en empate, como argmax)
        coverage = float(max(spectrum.items(), key=lambda kv: kv[1])[0])
        genome_size = total_kmers / coverage if coverage > 0 else 0.0
        return genome_size, coverage

    def spectrum(self, sequences: List[str], k: int,
                 min_abundance: int = 1,
                 chunk_size: Optional[int] = None,
                 window_size: Optional[int] = None) -> Dict[int, int]:
        """
        Espectro de k-mers: distribución del número de k-mers por multiplicidad.

        Args:
            sequences: Lista de secuencias.
            k: Longitud del k-mer.
            min_abundance: Frecuencia mínima para incluir un k-mer.
            chunk_size: Procesar en lotes para limitar memoria.
            window_size: Máximo de bases por ventana (genomas completos).

        Returns:
            Diccionario multiplicidad -> número de k-mers con esa multiplicidad.
        """
        _, counts = self.count(sequences, k, canonical=True,
                               min_abundance=min_abundance,
                               chunk_size=chunk_size, window_size=window_size)
        return self.spectrum_from_counts(counts)

    def estimate_genome_size(self, sequences: List[str], k: int,
                             min_abundance: int = 2,
                             chunk_size: Optional[int] = None,
                             window_size: Optional[int] = None) -> Tuple[float, float]:
        """
        Estima el tamaño del genoma a partir del espectro k-mer.

        Usa la aproximación de Lander-Waterman: G = N_k / C, donde N_k es el
        número total de k-mers (tras filtrar errores) y C es la cobertura
        media estimada como la multiplicidad pico del espectro.

        Args:
            sequences: Lista de secuencias.
            k: Longitud del k-mer.
            min_abundance: Abundancia mínima para descartar k-mers con errores.
            chunk_size: Procesar en lotes para limitar memoria.
            window_size: Máximo de bases por ventana (genomas completos).

        Returns:
            Tuple (tamaño estimado del genoma, cobertura media estimada).
        """
        _, counts = self.count(sequences, k, canonical=True,
                               min_abundance=min_abundance,
                               chunk_size=chunk_size, window_size=window_size)
        return self.estimate_from_counts(counts)
