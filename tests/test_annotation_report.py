"""
Tests de los hitos 5 y 6: anotación funcional (GFF/GO) y reporte
(heatmap/volcano/Markdown), incluida la ejecución aislada en proceso.

Ejecutar desde la raiz del proyecto:
    python -m pytest tests/test_annotation_report.py -v
"""

import asyncio
import os
import shutil
import tempfile
import time
import unittest

from Genoly.annotation.annotator import Annotator, summarize_go
from Genoly.report.plots import build_report, heatmap_svg, volcano_svg
from ui.backend import jobs

GFF = """##gff-version 3
chr1\t.\tgene\t100\t5000\t.\t+\t.\tID=gene1;Name=BRCA1
chr1\t.\tmRNA\t100\t5000\t.\t+\t.\tID=tx1;Parent=gene1
chr1\t.\texon\t100\t1000\t.\t+\t.\tParent=tx1
chr1\t.\tCDS\t200\t800\t.\t+\t.\tParent=tx1;Ontology_term=GO:0004672,GO:0005524
chr1\t.\tCDS\t2000\t3000\t.\t+\t.\tParent=tx1;Ontology_term=GO:0004672
chr1\t.\tgene\t8000\t9000\t.\t-\t.\tID=gene2;Name=TP53
chr2\t.\tgene\t1\t1000\t.\t+\t.\tID=gene3
"""


def _run_job(spec):
    async def main():
        job = jobs.manager.create(spec.get("kind", "annotation"))
        jobs.manager.submit_process(job, spec)
        deadline = time.monotonic() + 120
        while job.status not in ("done", "error"):
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"El job {job.id} no terminó en 120 s")
            await asyncio.sleep(0.05)
        return job
    return asyncio.run(main())


class TestAnnotator(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='ann_')
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.gff = os.path.join(self.tmpdir, 'ann.gff3')
        with open(self.gff, 'w') as fh:
            fh.write(GFF)
        self.an = Annotator()
        self.an.load(self.gff)

    def test_region_cds_intron_gen(self):
        self.assertEqual(self.an.annotate('chr1', 300)['region'], 'CDS')
        self.assertEqual(self.an.annotate('chr1', 1500)['region'], 'intron')
        self.assertEqual(self.an.annotate('chr1', 8500)['region'], 'gene')
        self.assertIsNone(self.an.annotate('chr1', 7000))

    def test_gene_y_go(self):
        hit = self.an.annotate('chr1', 300)
        self.assertEqual(hit['gene'], 'gene1')
        self.assertEqual(hit['gene_name'], 'BRCA1')
        self.assertIn('GO:0004672', hit['go_terms'])
        self.assertEqual(self.an.annotate('chr1', 8500)['strand'], '-')

    def test_annotate_variants_resumen(self):
        variants = [
            {'sequence': 'chr1', 'position': 300, 'ref': 'A', 'alt': 'T'},
            {'sequence': 'chr1', 'position': 310, 'ref': 'C', 'alt': 'G'},
            {'sequence': 'chr1', 'position': 1500},
            {'sequence': 'chr1', 'position': 8500},
            {'sequence': 'chr1', 'position': 7000},  # sin anotación
        ]
        res = self.an.annotate_variants(variants)
        self.assertEqual(res['annotated_count'], 4)
        self.assertEqual(res['by_region']['CDS'], 2)
        self.assertEqual(res['by_region']['intron'], 1)
        go = summarize_go(res['go_terms'])
        self.assertTrue(any(g['id'] == 'GO:0004672' for g in go))


class TestPlots(unittest.TestCase):
    def test_heatmap_svg(self):
        matrix = [[float(i + j) for j in range(5)] for i in range(6)]
        svg = heatmap_svg(matrix, row_labels=[f'r{i}' for i in range(6)],
                          col_labels=[f'c{j}' for j in range(5)],
                          title='Expresión')
        self.assertIn('<svg', svg)
        self.assertIn('Expresión', svg)
        self.assertGreaterEqual(svg.count('<rect'), 30)  # fondo + celdas

    def test_volcano_svg_significativos(self):
        fc = [-3, -0.2, 0.1, 2.5, 0.05, 4.0]
        pv = [0.001, 0.8, 0.9, 0.01, 0.5, 0.0001]
        svg = volcano_svg(fc, pv)
        self.assertIn('<svg', svg)
        # 3 significativos coloreados en acento
        self.assertEqual(svg.count('#40e0b2'), 3)

    def test_build_report_markdown(self):
        rep = build_report('Mi análisis', {
            'QC': {'lecturas': 1000, 'gc': 41.2},
            'Mapeo': {'mapeadas': 990, 'tasa': 99.0},
        })
        self.assertIn('# Mi análisis', rep.markdown)
        self.assertIn('**lecturas**: 1000', rep.markdown)
        self.assertEqual(len(rep.json['sections']), 2)


class TestAnnotationJobAislado(unittest.TestCase):
    def test_annotation_en_proceso(self):
        tmpdir = tempfile.mkdtemp(prefix='annjob_')
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        gff = os.path.join(tmpdir, 'ann.gff3')
        with open(gff, 'w') as fh:
            fh.write(GFF)
        job = _run_job({"kind": "annotation", "gff_path": gff, "variants": [
            {"sequence": "chr1", "position": 300, "ref": "A", "alt": "T"},
            {"sequence": "chr1", "position": 1500},
        ]})
        self.assertEqual(job.status, "done", job.error)
        self.assertEqual(job.result['annotated_count'], 2)
        self.assertEqual(job.result['features_loaded'], 7)


class TestReportJobAislado(unittest.TestCase):
    def test_report_en_proceso(self):
        tmpdir = tempfile.mkdtemp(prefix='repjob_')
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        md = os.path.join(tmpdir, 'report.md')
        job = _run_job({"kind": "report", "title": "Test",
                        "matrix": [[1.0, 2.0], [3.0, 4.0], [2.0, 1.0]],
                        "row_labels": None, "col_labels": None,
                        "fold_changes": [2.0, -3.0, 0.1],
                        "p_values": [0.01, 0.001, 0.9],
                        "fc_threshold": 1.0, "p_threshold": 0.05,
                        "sections": {"QC": {"lecturas": 100}},
                        "md_path": md, "md_upload_id": None})
        self.assertEqual(job.status, "done", job.error)
        self.assertIn('<svg', job.result['heatmap_svg'])
        self.assertIn('<svg', job.result['volcano_svg'])
        self.assertIn('# Test', job.result['markdown'])
        self.assertTrue(os.path.isfile(md))


if __name__ == "__main__":
    unittest.main(verbosity=2)
