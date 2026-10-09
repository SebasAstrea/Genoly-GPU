"""
Trabajos de análisis de larga duración con progreso en tiempo real.

Los análisis sobre archivos multi-GB pueden tardar minutos: ejecutarlos
en el event loop de FastAPI bloquearía la API y la UI. Este módulo los
ejecuta en hilos de fondo (ThreadPoolExecutor) y emite el progreso por
una cola asyncio, que el endpoint SSE (/api/jobs/{id}/events) reenvía al
frontend en tiempo real (Server-Sent Events).

El hilo trabajador publica eventos con ``job.publish(...)``, seguro para
hilos: usa ``loop.call_soon_threadsafe`` para programar el ``put_nowait``
en el event loop sin bloquearlo.
"""

import asyncio
import multiprocessing as mp
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import HTTPException

from ui.backend import worker as worker_mod

#: Máximo de trabajos terminados retenidos en memoria (evicción FIFO).
MAX_FINISHED_JOBS = 50

#: Hilos de trabajo GPU del proceso (variable de entorno GENOLY_MAX_WORKERS).
#: El valor por defecto, 1, serializa los análisis: los micro-lotes de
#: VRAM asumen un único consumidor de la GPU del proceso.
MAX_JOB_WORKERS = max(1, int(os.environ.get("GENOLY_MAX_WORKERS", "1")))

#: Frecuencia máxima de publicación de eventos de progreso (segundos).
PROGRESS_INTERVAL_SECONDS = 0.5

_JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")


@dataclass
class Job:
    """Estado de un trabajo de análisis y su cola de eventos SSE."""

    id: str
    kind: str
    created_at: float
    status: str = "pending"  # pending | running | done | error
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    loop: Optional[asyncio.AbstractEventLoop] = field(
        default=None, repr=False)
    #: Colas de eventos, una por suscriptor SSE. Solo se tocan en el hilo
    #: del event loop (publicación vía ``call_soon_threadsafe``), de modo
    #: que varios clientes pueden seguir el mismo trabajo sin quedarse cada
    #: uno con una parte de los eventos.
    subscribers: list = field(default_factory=list, repr=False)
    #: Último evento terminal (done/error) para poder reenviarlo a un
    #: cliente que se suscribe cuando el trabajo ya terminó.
    terminal: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def _deliver(self, event: Dict[str, Any]) -> None:
        """Entrega el evento a todos los suscriptores (hilo del loop)."""
        if event.get("type") in ("done", "error"):
            self.terminal = event
        for queue in list(self.subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:  # pragma no cubierto - colas ilimitadas
                pass

    def publish(self, event: Dict[str, Any]) -> None:
        """
        Publica un evento en todas las colas de suscriptores (thread-safe).

        Los productores deben limitar la frecuencia (throttle) para no
        inundar las colas con archivos enormes.
        """
        loop = self.loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self._deliver, dict(event))

    def subscribe(self, queue: "asyncio.Queue[Dict[str, Any]]") -> None:
        """Registra un nuevo suscriptor SSE (solo en el hilo del loop)."""
        if queue not in self.subscribers:
            self.subscribers.append(queue)

    def unsubscribe(self, queue: "asyncio.Queue[Dict[str, Any]]") -> None:
        """Retira un suscriptor SSE; nunca falla si ya no estaba."""
        try:
            self.subscribers.remove(queue)
        except ValueError:
            pass

    def snapshot(self) -> Dict[str, Any]:
        """Estado serializable del trabajo (para GET /api/jobs/{id})."""
        return {
            "job_id": self.id,
            "kind": self.kind,
            "status": self.status,
            "error": self.error,
            "result": self.result,
        }


class JobManager:
    """
    Registro y ejecución de trabajos de fondo.

    ``max_workers=1`` por defecto SERIALIZA los trabajos GPU: dos
    análisis simultáneos competirían por la misma VRAM y anularían el
    presupuesto del micro-batching adaptativo.
    """

    def __init__(self, max_workers: int = 1,
                 max_finished: int = MAX_FINISHED_JOBS):
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="genoly-job")
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._max_finished = max_finished

    def create(self, kind: str) -> Job:
        """
        Crea un trabajo. Debe llamarse desde el event loop (necesita
        ``asyncio.get_running_loop()`` para publicar eventos).
        """
        job = Job(
            id=uuid4().hex,
            kind=kind,
            created_at=time.time(),
            loop=asyncio.get_running_loop(),
        )
        with self._lock:
            self._evict_locked()
            self._jobs[job.id] = job
        return job

    def submit_process(self, job: Job, spec: Dict[str, Any],
                       totals: Optional[Dict[str, Any]] = None) -> Job:
        """
        Ejecuta un trabajo en un PROCESO SEPARADO (``spawn``).

        El hilo del executor serializa los trabajos GPU (max_workers=1);
        cada uno corre en su propio proceso, de modo que un crash del
        trabajo (OOM del kernel, segfault, ...) NUNCA mata el servidor: el
        proceso hijo muere, el job se marca como error y la API sigue viva.

        El progreso llega por una Pipe multiprocessing y se publica como
        eventos SSE (fusionando ``totals`` y con throttling). Si el proceso
        muere sin enviar mensaje (EOF), el job se marca como error con el
        código de salida.

        Args:
            job: Trabajo creado con ``manager.create``.
            spec: Especificación picklable del trabajo (ver ui.backend.worker).
            totals: Totales (registros/bases) fusionados en cada evento de
                progreso para que la UI calcule el porcentaje de avance.
        """
        throttle = {"t": 0.0}

        def runner() -> None:
            proc: Optional[mp.Process] = None
            parent: Optional["mp.connection.Connection"] = None
            job.status = "running"
            job.publish({"type": "start", "kind": job.kind})
            try:
                parent, child = mp.get_context("spawn").Pipe(duplex=False)
                proc = mp.get_context("spawn").Process(
                    target=worker_mod.run_job, args=(spec, child))
                proc.start()
                child.close()

                result: Optional[Dict[str, Any]] = None
                while True:
                    try:
                        msg = parent.recv()
                    except (EOFError, OSError):
                        # el proceso murió sin terminar (OOM/segfault/exit)
                        if proc is not None:
                            proc.join(timeout=3)
                        raise RuntimeError(
                            f"El proceso trabajador murió durante el análisis "
                            f"[exit={proc.exitcode if proc else None}]")
                    kind, data = msg
                    if kind == "progress":
                        now = time.monotonic()
                        if now - throttle["t"] >= PROGRESS_INTERVAL_SECONDS:
                            throttle["t"] = now
                            job.publish({"type": "progress",
                                         **(totals or {}), **data})
                    elif kind == "result":
                        result = data
                        break
                    elif kind == "error":
                        raise RuntimeError(data)

                proc.join()
                job.result = result
                job.status = "done"
                job.publish({"type": "done", "result": job.result})
            except Exception as exc:  # el mensaje viaja al cliente vía SSE
                job.status = "error"
                job.error = str(exc) or exc.__class__.__name__
                job.publish({"type": "error", "detail": job.error})
                if proc is not None and proc.is_alive():
                    proc.kill()
                    proc.join(timeout=2)
            finally:
                if proc is not None and proc.is_alive():
                    proc.terminate()
                if parent is not None:
                    try:
                        parent.close()
                    except Exception:
                        pass

        self._executor.submit(runner)
        return job

    def get(self, job_id: str) -> Job:
        """Devuelve el trabajo o lanza 404 si no existe."""
        if not _JOB_ID_RE.match(job_id or ""):
            raise HTTPException(status_code=400,
                                detail="Identificador de trabajo inválido")
        job = self._jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404,
                                detail="Trabajo no encontrado")
        return job

    def _evict_locked(self) -> None:
        finished = [j for j in self._jobs.values()
                    if j.status in ("done", "error")]
        excess = len(finished) - self._max_finished
        for job in finished[:max(0, excess)]:
            self._jobs.pop(job.id, None)


#: Singleton del proceso: todos los routers comparten el mismo ejecutor.
manager = JobManager(max_workers=MAX_JOB_WORKERS)
