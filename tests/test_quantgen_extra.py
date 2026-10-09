"""
Tests de los módulos extra de genética cuantitativa: validación cruzada,
GWAS (EMMAX), parentesco/GRM y QC de marcadores (MAF/HWE).

Ejecutar desde la raiz del proyecto:
    python -m pytest tests/test_quantgen_extra.py -v
"""

import math
import os
import random
import unittest

import torch

from Genoly.quantitative.crossval import crossval
from Genoly.quantitative.gwas import gwas
from Genoly.quantitative.popgen import grm_analysis
from Genoly.quantitative.preprocess import _marker_qc, clean_grid, load_grid


def _synthetic(n=200, p=800, qtl=100, seed=11):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    G = [[rng.choice([0, 1, 2]) for _ in range(p)] for _ in range(n)]
    Gt = torch.tensor(G, dtype=torch.float64)
    poly = rng.sample(range(p), 60)
    u = Gt[:, poly] @ (torch.randn(60, dtype=torch.float64) * 0.4) \
        + Gt[:, qtl] * 1.6
    u = (u - u.mean()) / u.std()
    y = (u * math.sqrt(0.5) + torch.randn(n) * math.sqrt(0.5)).tolist()
    return G, y


class TestCrossval(unittest.TestCase):
    def test_prediccion_positiva(self):
        G, y = _synthetic()
        cv = crossval(y, G, n_folds=5, n_repeats=2, seed=1)
        self.assertEqual(cv['n_individuals'], 200)
        self.assertEqual(len(cv['per_fold']), 10)
        self.assertEqual(len(cv['per_individual']), 200 * 2)  # 2 repeticiones
        self.assertGreater(cv['mean_r'], 0.2)
        self.assertGreater(cv['mean_accuracy'], 0.0)

    def test_errores(self):
        with self.assertRaises(ValueError):
            crossval([1, 2], [[0], [1]], n_folds=1)


class TestGwas(unittest.TestCase):
    def test_detecta_qtl(self):
        G, y = _synthetic()
        res = gwas(y, G, min_maf=0.05)
        rank = next(i for i, m in enumerate(res['markers'])
                    if m['index'] == 100)
        self.assertLess(rank, 5)
        self.assertIn('<svg', res['manhattan_svg'])
        self.assertGreaterEqual(res['n_tested'], 100)


class TestGrm(unittest.TestCase):
    def test_analisis(self):
        G, _ = _synthetic(n=60, p=300)
        res = grm_analysis(G)
        self.assertEqual(len(res['pca']), 60)
        self.assertIn('<svg', res['heatmap_svg'])
        self.assertTrue(res['top_pairs'])


class TestMarkerQc(unittest.TestCase):
    def test_maf_y_hwe(self):
        # marcador monomórfico -> MAF 0
        mono = [0, 0, 0, 0, 0, 0, 0, 0]
        # marcador en equilibrio
        seg = [0, 1, 2, 1, 0, 2, 1, 0]
        geno = [list(r) for r in zip(mono, seg)]
        keep, low_maf, _ = _marker_qc(geno, min_maf=0.05)
        self.assertNotIn(0, keep)          # monomórfico descartado
        self.assertEqual(low_maf, [0])
        self.assertIn(1, keep)

    def test_clean_grid_filtra_maf(self):
        path = os.path.join(
            os.path.dirname(__file__), '_tmp_maf.csv')
        with open(path, 'w') as fh:
            fh.write('feno;m1;m2\n')
            for i in range(20):
                fh.write(f'{i};0;{i % 3}\n')
        grid = load_grid(path)
        _, geno, markers, rep = clean_grid(
            grid, phenotype_col=0, min_maf=0.05, min_markers=1)
        # m1 es monomórfico (todos 0) -> descartado
        self.assertEqual(markers, ['m2'])
        self.assertEqual(rep['dropped_by_maf'], ['m1'])
        os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)