"""
Tests del hito FASTQ: lector por bloques, QC en streaming y
preprocesamiento (trim/filtro), incluida la ejecución aislada en proceso.

Ejecutar desde la raiz del proyecto:
    python -m pytest tests/test_fastq_pipeline.py -v
"""

import asyncio
import os
import random
import shutil
import tempfile
import time
import unittest

from Genoly.io.fastq import FastqReader, FastqRecord, write_fastq
from Genoly.qc.quality import QualityAnalyzer
from ui.backend import jobs


def _run_job(spec):
    async def main():
        job = jobs.manager.create(spec.get("kind", "kmer"))
        jobs.manager.submit_process(job, spec)
        deadline = time.monotonic() + 120
        while job.status not in ("done", "error"):
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"El job {job.id} no terminó en 120 s")
            await asyncio.sleep(0.05)
        return job
    return asyncio.run(main())


def _rnd_read(rng, n, q_base=35, n_prob=0.01):
    seq = ''.join('N' if rng.random() < n_prob else rng.choice('ACGT')
                  for _ in range(n))
    qual = ''.join(chr(min(93, 33 + q_base + rng.randint(-5, 5)))
                   for _ in range(n))
    return FastqRecord(id=f'r{rng.randint(1, 10 ** 6)}',
                       sequence=seq, quality=qual)


class TestFastqReader(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='fastq_')
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)

    def test_lector_por_bloques_y_scan_stats(self):
        rng = random.Random(1)
        records = [_rnd_read(rng, rng.randint(50, 150)) for _ in range(3000)]
        path = os.path.join(self.tmpdir, 'reads.fastq')
        write_fastq(path, records)
        loaded = list(FastqReader(path).records())
        self.assertEqual(len(loaded), 3000)
        st = FastqReader(path).scan_stats()
        self.assertEqual(st.reads, 3000)
        self.assertEqual(st.total_bases, sum(len(r) for r in records))
        self.assertEqual(st.first_id, loaded[0].id)
        self.assertEqual(st.first_length, len(loaded[0]))

    def test_multiline_illumina(self):
        path = os.path.join(self.tmpdir, 'ml.fastq')
        with open(path, 'w') as fh:
            fh.write('@r1 desc\nACGTACGT\n+\nIIIIIIII\n')
            fh.write('@r2\nTTTT\n+\n!!!!\n')
        recs = list(FastqReader(path).records())
        self.assertEqual([r.id for r in recs], ['r1', 'r2'])
        self.assertEqual(recs[0].sequence, 'ACGTACGT')
        self.assertEqual(recs[1].quality, '!!!!')

    def test_archivo_invalido(self):
        path = os.path.join(self.tmpdir, 'bad.fastq')
        with open(path, 'w') as fh:
            fh.write('esto no es fastq\n')
        with self.assertRaises(ValueError):
            list(FastqReader(path).records())


class TestFastqQcStream(unittest.TestCase):
    def setUp(self):
        self.qa = QualityAnalyzer('cpu')
        rng = random.Random(7)
        self.records = [_rnd_read(rng, rng.randint(50, 150))
                        for _ in range(500)]

    def test_analyze_stream_equivale_a_report(self):
        stream = self.qa.analyze_stream(iter(self.records), batch_size=64,
                                        max_position=150)
        rep = self.qa.report(self.records)
        self.assertEqual(stream['num_reads'], 500)
        self.assertAlmostEqual(stream['mean_quality'], rep.mean_quality,
                               delta=0.5)
        self.assertAlmostEqual(stream['gc_content_percent'],
                               rep.gc_content_percent, delta=0.1)
        self.assertAlmostEqual(stream['mean_read_length'],
                               rep.mean_read_length, delta=0.5)
        self.assertEqual(len(stream['quality_by_position']),
                         len(rep.quality_by_position))

    def test_analyze_stream_progreso(self):
        events = []
        self.qa.analyze_stream(iter(self.records), batch_size=64,
                               on_progress=events.append)
        self.assertTrue(events)
        self.assertEqual(events[-1]['reads'], 500)

    def test_process_stream_trim_y_filtro(self):
        _td = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, _td, ignore_errors=True)
        out = os.path.join(_td, 'clean.fastq')
        rin, rout, bin_, bout = self.qa.process_stream(
            iter(self.records), open(out, 'w'),
            min_quality=20, window_size=5, min_length=30,
            min_mean_quality=20, max_n_ratio=0.05)
        self.assertEqual(rin, 500)
        self.assertLessEqual(rout, 500)
        self.assertLessEqual(bout, bin_)

    def test_lecturas_de_baja_calidad_se_descartan(self):
        bad = [FastqRecord(id=f'b{i}', sequence='A' * 100,
                           quality='#' * 100) for i in range(50)]
        _td = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, _td, ignore_errors=True)
        out = os.path.join(_td, 'bad.fastq')
        rin, rout, _, _ = self.qa.process_stream(
            iter(bad), open(out, 'w'), min_quality=20, window_size=5,
            min_length=0, min_mean_quality=0, max_n_ratio=1)
        self.assertEqual(rout, 0)


class TestFastqJobAislado(unittest.TestCase):
    """El QC y el procesamiento FASTQ corren en un proceso aislado."""

    def setUp(self):
        rng = random.Random(9)
        self.tmpdir = tempfile.mkdtemp(prefix='fastqjob_')
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        records = [_rnd_read(rng, rng.randint(50, 120)) for _ in range(800)]
        self.path = os.path.join(self.tmpdir, 'reads.fastq')
        write_fastq(self.path, records)

    def test_fastq_qc_en_proceso(self):
        job = _run_job({"kind": "fastq_qc", "path": self.path,
                        "batch_size": 128, "max_position": 120})
        self.assertEqual(job.status, "done", job.error)
        self.assertEqual(job.result["num_reads"], 800)

    def test_fastq_process_en_proceso(self):
        out = os.path.join(self.tmpdir, 'clean.fastq')
        job = _run_job({"kind": "fastq_process", "path": self.path,
                        "out_path": out, "out_upload_id": None,
                        "min_quality": 20, "window_size": 5,
                        "min_length": 30, "min_mean_quality": 20,
                        "max_n_ratio": 0.05})
        self.assertEqual(job.status, "done", job.error)
        self.assertEqual(job.result["reads_in"], 800)
        self.assertGreater(job.result["reads_out"], 0)
        self.assertTrue(os.path.isfile(out))


if __name__ == "__main__":
    unittest.main(verbosity=2)
