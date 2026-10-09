"""
Tests del aislamiento de trabajos GPU en procesos separados.

Un trabajo (k-mers) se ejecuta en un proceso ``spawn`` aparte: si el
trabajo revienta (error controlado, segfault, os._exit) el proceso del
servidor (este test) sobrevive y el trabajo queda en ``error``.

Ejecutar desde la raiz del proyecto:
    python -m pytest tests/test_worker.py -v
o directamente:
    python tests/test_worker.py
"""

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

from Genoly import FastaRecord, write_fasta
from ui.backend import jobs


def _run_job(spec, totals=None):
    """Lanza un trabajo aislado y espera a que termine (done/error)."""
    async def main():
        job = jobs.manager.create(spec.get("kind", "kmer"))
        jobs.manager.submit_process(job, spec, totals=totals)
        deadline = time.monotonic() + 120
        while job.status not in ("done", "error"):
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"El job {job.id} no terminó en 120 s")
            await asyncio.sleep(0.05)
        return job
    return asyncio.run(main())


class TestProcessIsolation(unittest.TestCase):
    def test_job_kmer_proceso_completa(self):
        """Un job real de k-mers se ejecuta en un proceso hijo y da resultado."""
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        p = os.path.join(d, 'a.fasta')
        write_fasta(p, [FastaRecord(id='s', sequence='ACGT' * 10)])
        job = _run_job({"kind": "kmer", "path": p, "k": 5, "canonical": True,
                        "min_abundance": 1, "top": 3, "sequences": []})
        self.assertEqual(job.status, "done", job.error)
        self.assertEqual(job.result["total_kmers"], 36)   # 40 pb, k=5 -> 36 ventanas
        self.assertEqual(job.result["total_unique"], 2)   # 4 k-mers distintos, colapsados por canónico

    def test_job_error_controlado_aislado(self):
        """Un error lanzado en el hijo se convierte en job 'error'."""
        job = _run_job({"kind": "crash_test", "mode": "raise"})
        self.assertEqual(job.status, "error")
        self.assertIn("crash simulado", job.error)

    def test_job_segv_aislado_no_mata_el_proceso(self):
        """Un segfault del hijo no mata al proceso del servidor (este test)."""
        job = _run_job({"kind": "crash_test", "mode": "segv"})
        self.assertEqual(job.status, "error")
        self.assertIn("murió", job.error)  # detectado como crash, no como error
        # el proceso padre sigue vivo (este test sigue ejecutándose)

    def test_job_exit_aislado_no_mata_el_proceso(self):
        """Un os._exit(1) del hijo (como un OOM) no mata al servidor."""
        job = _run_job({"kind": "crash_test", "mode": "exit"})
        self.assertEqual(job.status, "error")
        self.assertIn("murió", job.error)
        self.assertIn("exit=1", job.error)

    def test_cleanup_derrames_huerfanos(self):
        """La limpieza borra los derrames de PIDs muertos y conserva los vivos."""
        from Genoly.kmer.kmers import cleanup_orphan_spills
        base = tempfile.mkdtemp(prefix='spill_clean_')
        self.addCleanup(shutil.rmtree, base, ignore_errors=True)
        p = subprocess.Popen([sys.executable, '-c', 'pass'])
        dead_pid = p.pid
        p.wait()  # pid garantizado muerto
        old = time.time() - 3600
        fresh = time.time()

        orphan_old = os.path.join(base, f'genoly_kmer_{dead_pid}_a')
        live = os.path.join(base, f'genoly_kmer_{os.getpid()}_b')
        orphan_fresh = os.path.join(base, f'genoly_kmer_{dead_pid}_c')
        for d in (orphan_old, live, orphan_fresh):
            os.makedirs(d)
        os.utime(orphan_old, (old, old))
        os.utime(orphan_fresh, (fresh, fresh))
        os.utime(live, (old, old))  # viejo pero PID vivo: se conserva

        removed = cleanup_orphan_spills(spill_dir=base, min_age_seconds=600)
        self.assertEqual(removed, 1)
        self.assertFalse(os.path.exists(orphan_old))
        self.assertTrue(os.path.exists(live))
        self.assertTrue(os.path.exists(orphan_fresh))
        shutil.rmtree(base, ignore_errors=True)

    def test_progreso_fusionado_con_totales(self):
        """Los eventos de progreso llevan los totales fusionados."""
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        p = os.path.join(d, 'b.fasta')
        write_fasta(p, [FastaRecord(id='s', sequence='ACGT' * 100)])
        events = []
        async def main():
            job = jobs.manager.create("kmer")
            real = job.publish
            job.publish = lambda ev: (events.append(dict(ev)), real(ev)) and None
            jobs.manager.submit_process(
                job, {"kind": "kmer", "path": p, "k": 5, "canonical": True,
                      "min_abundance": 1, "top": 3, "sequences": []},
                totals={"total_records": 1, "total_bases": 400})
            deadline = time.monotonic() + 120
            while job.status not in ("done", "error"):
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"El job {job.id} no terminó en 120 s")
                await asyncio.sleep(0.05)
            return job
        job = asyncio.run(main())
        self.assertEqual(job.status, "done", job.error)
        self.assertTrue(any(e["type"] == "progress" and e.get("total_bases") == 400
                            for e in events), events)


if __name__ == "__main__":
    unittest.main(verbosity=2)
