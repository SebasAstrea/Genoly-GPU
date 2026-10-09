"""
Tests del mapeo de lecturas a referencia (seed-and-extend), incluida la
ejecución aislada en proceso.

Ejecutar desde la raiz del proyecto:
    python -m pytest tests/test_mapper.py -v
"""

import asyncio
import os
import random
import shutil
import tempfile
import time
import unittest

from Genoly.alignment.mapper import ReadMapper, reverse_complement
from Genoly.io.fasta import FastaRecord, write_fasta
from Genoly.io.fastq import FastqRecord, write_fastq
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


def _mkref(tmpdir, n=50_000, name='chr_test'):
    rng = random.Random(21)
    seq = ''.join(rng.choice('ACGT') for _ in range(n))
    path = os.path.join(tmpdir, 'ref.fasta')
    write_fasta(path, [FastaRecord(id=name, sequence=seq)], line_width=80)
    return seq, path


def _mkreads(seq, tmpdir, n=200, rate=0.05, length=150):
    rng = random.Random(7)
    reads = []
    truth = []
    for i in range(n):
        start = rng.randint(0, len(seq) - length - 1)
        seg = list(seq[start:start + length])
        for j in range(len(seg)):
            if rng.random() < rate:
                seg[j] = rng.choice([b for b in 'ACGT' if b != seg[j]])
        seg = ''.join(seg)
        rev = rng.random() < 0.5
        if rev:
            seg = reverse_complement(seg)
        truth.append((start, rev))
        reads.append(FastqRecord(id=f'r{i}', sequence=seg, quality='I' * length))
    path = os.path.join(tmpdir, 'reads.fastq')
    write_fastq(path, reads)
    return reads, truth, path


class TestReadMapper(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='mapper_')
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.ref, self.ref_path = _mkref(self.tmpdir)
        self.mapper = ReadMapper(k=15, stride=1, min_seeds=3)

    def test_mapeo_exacto_posicion_y_hebra(self):
        idx = self.mapper.build_index(self.ref_path)
        self.assertEqual(len(idx.sequences), 1)
        self.assertGreater(idx.codes.shape[0], 40_000)

        reads, truth, _ = _mkreads(self.ref, self.tmpdir)
        hits = self.mapper.map_reads(reads, idx)
        mapped = [(h, truth[i]) for i, h in enumerate(hits) if h.mapped]
        self.assertGreaterEqual(len(mapped), len(reads) * 0.95)

        ok = sum(1 for h, (s, rev) in mapped if abs(h.ref_start - s) <= 2)
        strand_ok = sum(1 for h, (s, rev) in mapped
                        if h.strand == ('-' if rev else '+'))
        self.assertGreaterEqual(ok, len(mapped) * 0.95)
        self.assertGreaterEqual(strand_ok, len(mapped) * 0.95)

    def test_mapeo_hebra_reversa_convencion_sam(self):
        idx = self.mapper.build_index(self.ref_path)
        start = 1000
        rev_read = reverse_complement(self.ref[start:start + 150])
        hit = self.mapper.map_read(
            FastqRecord(id='x', sequence=rev_read, quality='I' * 150), idx)
        self.assertTrue(hit.mapped)
        self.assertEqual(hit.strand, '-')
        self.assertLessEqual(abs(hit.ref_start - start), 2)  # POS = inicio forward

    def test_lectura_corta_no_mapea(self):
        idx = self.mapper.build_index(self.ref_path)
        hit = self.mapper.map_read(
            FastqRecord(id='x', sequence='ACGT', quality='IIII'), idx)
        self.assertFalse(hit.mapped)


class TestMapJobAislado(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='mapjob_')
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.ref, self.ref_path = _mkref(self.tmpdir, n=30_000)
        _, _, self.reads_path = _mkreads(self.ref, self.tmpdir, n=150)

    def test_map_en_proceso(self):
        job = _run_job({"kind": "map", "ref_path": self.ref_path,
                        "reads_path": self.reads_path, "k": 15,
                        "stride": 1, "min_seeds": 2, "sample_limit": 100})
        self.assertEqual(job.status, "done", job.error)
        r = job.result
        self.assertEqual(r["total_reads"], 150)
        self.assertGreaterEqual(r["mapping_rate"], 90.0)
        self.assertGreater(r["mean_identity"], 90.0)
        self.assertTrue(r["hits"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
