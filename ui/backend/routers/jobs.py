"""
Endpoints de seguimiento de trabajos de fondo.

- GET /api/jobs/{id}         -> estado y resultado (JSON, polling).
- GET /api/jobs/{id}/events  -> progreso en tiempo real vía SSE.

El stream SSE envía cada evento como ``data: {json}\\n\\n`` con keep-alives
cada 15 s para atravesar proxies. Cada cliente recibe su propia cola: el
evento ``done``/``error`` se reenvía a todos los suscriptores, y un cliente
que se suscribe cuando el trabajo ya terminó recibe el evento terminal de
inmediato (sin colgar en keep-alives infinitos).
"""

import asyncio
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ui.backend import jobs

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

KEEP_ALIVE_SECONDS = 15.0

#: Tipos de evento que cierran el stream.
_TERMINAL = ("done", "error")


def _terminal_event(job) -> dict:
    """Evento terminal sintetizado desde el snapshot del trabajo."""
    if job.status == "error":
        return {"type": "error", "detail": job.error or "El trabajo falló"}
    return {"type": "done", "result": job.result}


def _fmt(event: dict) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


@router.get("/{job_id}")
async def job_status(job_id: str) -> dict:
    """Estado y, si terminó, resultado del trabajo (para polling)."""
    return jobs.manager.get(job_id).snapshot()


@router.get("/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    """Stream Server-Sent Events con el progreso del trabajo."""
    job = jobs.manager.get(job_id)

    async def event_stream():
        # Suscribirse antes de mirar el estado: no hay `await` entre la
        # suscripción y la comprobación, así que no puede perderse un
        # evento terminal en ese hueco. Todo corre en el hilo del loop.
        queue: "asyncio.Queue[dict]" = asyncio.Queue()
        job.subscribe(queue)
        try:
            snapshot = job.snapshot()
            yield _fmt({"type": "status", "status": snapshot["status"],
                        "kind": snapshot["kind"]})
            if snapshot["status"] in _TERMINAL:
                yield _fmt(job.terminal or _terminal_event(job))
                return

            while True:
                try:
                    event = await asyncio.wait_for(queue.get(),
                                                   timeout=KEEP_ALIVE_SECONDS)
                except asyncio.TimeoutError:
                    if job.status in _TERMINAL:
                        yield _fmt(job.terminal or _terminal_event(job))
                        return
                    yield ": keep-alive\n\n"
                    continue
                yield _fmt(event)
                if event.get("type") in _TERMINAL:
                    return
        finally:
            job.unsubscribe(queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
