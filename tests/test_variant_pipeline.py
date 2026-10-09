"""
Tests del hito 3: SAM/BAM + llamada de variantes + VCF.

Cubre el flujo mapeo -> SAM (pysam) -> variantes -> VCF, incluida la
ejecución aislada en proceso.

Ejecutar desde la raiz del proyecto:
    python -m pytest tests/test_variant_pipeline.py -v
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
from Genoly.io.sam import iter_sam, write_sam
from Genoly.variants.caller import VariantCaller
from Genoly.variants.vcf import write_vcf
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


def _make_dataset(tmpdir, n_ref=50_000, n_var=40, cov=40, rate=0.45,
                  seed=33, region=(10_000, 13_000)):
    rng = random.Random(seed)
    ref = ''.join(rng.choice('ACGT') for _ in range(n_ref))
    ref_path = os.path.join(tmpdir, 'ref.fasta')
    write_fasta(ref_path, [FastaRecord(id='chr1', sequence=ref)], line_width=80)

    h0, h1 = region
    variants = {}
    for p in rng.sample(range(h0 + 100, h1 - 100), n_var):
        variants[p] = (ref[p], rng.choice([b for b in 'ACGT' if b != ref[p]]))

    reads = []
    n_reads = (h1 - h0) * cov // 150
    for i in range(n_reads):
        start = h0 + rng.randint(0, h1 - h0 - 150)
        seg = list(ref[start:start + 150])
        for j in range(len(seg)):
            ap = start + j
            if ap in variants and rng.random() < rate:
                seg[j] = variants[ap][1]
            elif rng.random() < 0.002:
                seg[j] = rng.choice([b for b in 'ACGT' if b != seg[j]])
        seg = ''.join(seg)
        if rng.random() < 0.5:
            seg = reverse_complement(seg)
        reads.append(FastqRecord(id=f'r{i}', sequence=seg, quality='I' * 150))
    reads_path = os.path.join(tmpdir, 'reads.fastq')
    write_fastq(reads_path, reads)
    return ref, ref_path, reads_path, variants, reads


class TestVariantPipeline(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='var_')
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.ref, self.ref_path, self.reads_path, self.variants, self.reads = \
            _make_dataset(self.tmpdir)
        self.mapper = ReadMapper(k=15, stride=1, min_seeds=3)
        self.index = self.mapper.build_index(self.ref_path)
        self.sam_path = os.path.join(self.tmpdir, 'map.sam')
        write_sam(self.sam_path, self.index.sequences,
                  self.mapper.iter_mapped(self.reads, self.index))

    def test_sam_valido_y_hebra_forward(self):
        srs = list(iter_sam(self.sam_path))
        self.assertEqual(len(srs), len(self.reads))
        self.assertTrue(all(s.mapped for s in srs))
        # SEQ en el SAM debe alinearse a la referencia en POS (>=90% match:
        # las lecturas llevan mutaciones y ruido)
        for s in srs[:50]:
            ref_seg = self.ref[s.pos:s.pos + len(s.seq)]
            matches = sum(1 for a, b in zip(s.seq, ref_seg) if a == b)
            self.assertGreaterEqual(matches, len(s.seq) * 0.9)

    def test_variantes_snv_exactas(self):
        vc = VariantCaller('cpu')
        res = vc.call_variants_from_sam(self.sam_path, self.ref_path,
                                        min_depth=10, min_alt_freq=0.2)
        bypos = {v.position - 1: v for v in res['by_sequence']['chr1']
                 if v.type == 'SNV'}
        tp = sum(1 for p, (r, a) in self.variants.items()
                 if p in bypos and bypos[p].ref == r and bypos[p].alt == a)
        fp = sum(1 for p in bypos if p not in self.variants)
        self.assertGreaterEqual(tp, len(self.variants) * 0.9)
        self.assertLessEqual(fp, 3)

    def test_vcf_escrito(self):
        vc = VariantCaller('cpu')
        res = vc.call_variants_from_sam(self.sam_path, self.ref_path)
        vcf = os.path.join(self.tmpdir, 'out.vcf')
        n = write_vcf(vcf, res['by_sequence'], res['sequences'])
        self.assertEqual(n, res['total_variants'])
        text = open(vcf).read()
        self.assertIn('##fileformat=VCFv4.2', text)
        self.assertIn('#CHROM\tPOS', text)
        self.assertGreater(n, 0)


class TestVariantJobAislado(unittest.TestCase):
    def test_variant_call_en_proceso(self):
        tmpdir = tempfile.mkdtemp(prefix='varjob_')
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        ref, ref_path, reads_path, variants, reads = _make_dataset(
            tmpdir, n_ref=30_000, n_var=20, region=(5_000, 8_000))
        mapper = ReadMapper(k=15, stride=1, min_seeds=3)
        index = mapper.build_index(ref_path)
        sam_path = os.path.join(tmpdir, 'map.sam')
        write_sam(sam_path, index.sequences,
                  mapper.iter_mapped(reads, index))
        vcf_path = os.path.join(tmpdir, 'out.vcf')

        job = _run_job({"kind": "variant_call", "ref_path": ref_path,
                        "sam_path": sam_path, "out_vcf_path": vcf_path,
                        "out_vcf_upload_id": None, "min_depth": 10,
                        "min_alt_freq": 0.2, "region_size": 50_000_000})
        self.assertEqual(job.status, "done", job.error)
        self.assertGreaterEqual(job.result['total_variants'],
                                len(variants) * 0.8)
        self.assertTrue(os.path.isfile(vcf_path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
