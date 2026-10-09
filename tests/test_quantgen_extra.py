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
from Genoly.quantitative.pblup import PedigreeBLUP
from Genoly.quantitative.pedigree import (
    PedigreeError,
    build_a_inverse,
    build_a_matrix,
    normalize_pedigree,
)
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
        _, geno, markers, _, rep = clean_grid(
            grid, phenotype_col=0, min_maf=0.05, min_markers=1)
        # m1 es monomórfico (todos 0) -> descartado
        self.assertEqual(markers, ['m2'])
        self.assertEqual(rep['dropped_by_maf'], ['m1'])
        os.unlink(path)


class TestPedigreeClean(unittest.TestCase):
    """clean_grid con columnas de pedigrí (id/sire/dam), dry_run, qc_flags."""

    def _pedigree_grid(self):
        return [
            ['id', 'feno', 'sire', 'dam', 'm1'],
            ['A1', '10.5', 'P1', 'M1', '0'],
            ['A2', '11.0', 'P1', 'M1', '1'],
            ['A3', '12.0', 'P2', 'M2', '2'],
            ['A4', '13.5', 'P2', 'M2', '1'],
            ['A5', '14.0', 'P3', 'M3', '0'],
            ['A6', '15.0', 'P3', 'M3', '2'],
        ]

    def test_preserva_columnas_pedigree_no_numericas(self):
        pheno, geno, markers, pedigree, rep = clean_grid(
            self._pedigree_grid(), phenotype_col=1,
            id_col=0, sire_col=2, dam_col=3,
            min_individuals=5, min_markers=1)
        self.assertEqual(len(pheno), 6)
        self.assertEqual(markers, ['m1'])
        self.assertEqual(pedigree['individuals'], ['A1', 'A2', 'A3', 'A4', 'A5', 'A6'])
        self.assertEqual(pedigree['sires'], ['P1', 'P1', 'P2', 'P2', 'P3', 'P3'])
        self.assertEqual(pedigree['dams'], ['M1', 'M1', 'M2', 'M2', 'M3', 'M3'])
        self.assertEqual(rep['pedigree_columns']['id'], 'id')
        self.assertEqual(rep['pedigree_columns']['sire'], 'sire')
        self.assertEqual(rep['pedigree_columns']['dam'], 'dam')
        # las columnas de pedigrí no deben aparecer como descartadas
        dropped = [dc['name'] for dc in rep['dropped_columns']]
        self.assertNotIn('id', dropped)
        self.assertNotIn('sire', dropped)
        self.assertNotIn('dam', dropped)

    def test_solo_pedigree_cero_marcadores(self):
        grid = [
            ['id', 'feno', 'sire', 'dam'],
            ['A1', '10.5', 'P1', 'M1'],
            ['A2', '11.0', 'P1', 'M1'],
            ['A3', '12.0', 'P2', 'M2'],
            ['A4', '13.5', 'P2', 'M2'],
            ['A5', '14.0', 'P3', 'M3'],
        ]
        pheno, geno, markers, pedigree, rep = clean_grid(
            grid, phenotype_col=1, id_col=0, sire_col=2, dam_col=3,
            min_individuals=5, min_markers=0)
        self.assertEqual(len(pheno), 5)
        self.assertEqual(markers, [])
        self.assertEqual(geno, [[], [], [], [], []])
        self.assertEqual(pedigree['individuals'], ['A1', 'A2', 'A3', 'A4', 'A5'])
        self.assertEqual(rep['final_markers'], 0)

    def test_solo_id_sin_sire_dam_pedigree_sire_dam_none(self):
        grid = [
            ['id', 'feno', 'm1'],
            ['A1', '10', '0'],
            ['A2', '11', '1'],
            ['A3', '12', '2'],
            ['A4', '13', '0'],
            ['A5', '14', '1'],
        ]
        pheno, geno, markers, pedigree, rep = clean_grid(
            grid, phenotype_col=1, id_col=0,
            min_individuals=5, min_markers=1)
        self.assertEqual(pedigree['individuals'], ['A1', 'A2', 'A3', 'A4', 'A5'])
        self.assertEqual(pedigree['sires'], [None] * 5)
        self.assertEqual(pedigree['dams'], [None] * 5)

    def test_ids_ausentes_se_autorenumeran(self):
        grid = [
            ['id', 'feno', 'm1'],
            ['', '10', '0'],
            ['A2', '11', '1'],
            ['', '12', '2'],
            ['A4', '13', '0'],
            ['A5', '14', '1'],
        ]
        pheno, geno, markers, pedigree, rep = clean_grid(
            grid, phenotype_col=1, id_col=0,
            min_individuals=5, min_markers=1)
        self.assertEqual(pedigree['individuals'],
                         ['IND-0001', 'A2', 'IND-0003', 'A4', 'A5'])

    def test_dry_run_no_lanza_y_rellena_blocking_errors(self):
        grid = [
            ['feno', 'm1'],
            ['10', '0'],
            ['11', '1'],
        ]
        with self.assertRaises(ValueError):
            clean_grid(grid, phenotype_col=0, min_individuals=5, min_markers=1)
        pheno, geno, markers, pedigree, rep = clean_grid(
            grid, phenotype_col=0, min_individuals=5, min_markers=1,
            dry_run=True)
        self.assertTrue(rep['blocking_errors'])
        self.assertTrue(rep['dry_run'])
        self.assertEqual(len(pheno), 2)

    def test_dry_run_indices_solapados_sigue_lanzando(self):
        grid = [
            ['id', 'feno', 'm1'],
            ['A1', '10', '0'],
        ]
        with self.assertRaises(ValueError):
            clean_grid(grid, phenotype_col=1, id_col=1, dry_run=True)

    def test_qc_flags_sire_eq_dam_y_dosis_fuera_rango(self):
        grid = [
            ['id', 'feno', 'sire', 'dam', 'm1'],
            ['A1', '10', 'P', 'P', '0'],
            ['A2', '11', 'P1', 'M1', '5'],
            ['A3', '12', 'P2', 'M2', '1'],
            ['A4', '13', 'P2', 'M2', '2'],
            ['A5', '14', 'P3', 'M3', '0'],
        ]
        pheno, geno, markers, pedigree, rep = clean_grid(
            grid, phenotype_col=1, id_col=0, sire_col=2, dam_col=3,
            min_individuals=5, min_markers=1)
        codes = [f['code'] for f in rep['qc_flags']]
        self.assertIn('sire_eq_dam', codes)
        self.assertIn('dosis_fuera_de_rango', codes)
        flag = next(f for f in rep['qc_flags'] if f['code'] == 'sire_eq_dam')
        self.assertEqual(flag['count'], 1)

    def test_qc_flags_ids_duplicados(self):
        grid = [
            ['id', 'feno', 'm1'],
            ['A1', '10', '0'],
            ['A1', '11', '1'],
            ['A2', '12', '2'],
            ['A3', '13', '0'],
            ['A4', '14', '1'],
        ]
        pheno, geno, markers, pedigree, rep = clean_grid(
            grid, phenotype_col=1, id_col=0,
            min_individuals=5, min_markers=1)
        codes = [f['code'] for f in rep['qc_flags']]
        self.assertIn('ids_duplicados', codes)
        flag = next(f for f in rep['qc_flags'] if f['code'] == 'ids_duplicados')
        self.assertEqual(flag['count'], 1)

    def test_validaciones_indices_solapados(self):
        grid = [
            ['id', 'feno', 'm1'],
            ['A1', '10', '0'],
        ]
        with self.assertRaises(ValueError):
            clean_grid(grid, phenotype_col=1, id_col=2, sire_col=2)
        with self.assertRaises(ValueError):
            clean_grid(grid, phenotype_col=0, id_col=0)
        with self.assertRaises(ValueError):
            clean_grid(grid, phenotype_col=1, sire_col=1)


class TestFilterSemantica(unittest.TestCase):
    """La semántica de filtros por definición (worker) replica la del JS."""

    def test_passes_filters_equivalencia(self):
        from ui.backend.worker import _passes_filters
        row = ['10.5', 'A1', '', '0']
        # vacios=False excluye celdas missing
        self.assertFalse(_passes_filters(row, {'2': {'vals': [], 'vacios': False}}))
        # vacios=True las incluye
        self.assertTrue(_passes_filters(row, {'2': {'vals': [], 'vacios': True}}))
        # valor presente debe estar en vals
        self.assertTrue(_passes_filters(row, {'1': {'vals': ['A1'], 'vacios': False}}))
        self.assertFalse(_passes_filters(row, {'1': {'vals': ['A2'], 'vacios': True}}))
        # vals vacío + vacios=True = sólo se permiten celdas missing
        self.assertFalse(_passes_filters(row, {'0': {'vals': [], 'vacios': True}}))
        # sin entrada de filtro para la columna no se restringe
        self.assertTrue(_passes_filters(row, {}))


class TestPedigree(unittest.TestCase):
    """Matriz A, inversa A⁻¹ y normalización topológica del pedigrí."""

    def test_fundador(self):
        A, F, info = build_a_matrix(['A'], [None], [None])
        self.assertEqual(A.shape, (1, 1))
        self.assertAlmostEqual(float(A[0, 0]), 1.0)
        self.assertAlmostEqual(float(F[0]), 0.0)
        self.assertEqual(info['founders'], [0])

    def test_hijo_un_padre(self):
        # C hijo de S (dam desconocida): A[C,S]=0.5, A[C,C]=1, F=0
        ids = ['S', 'D', 'C']
        sr = [None, None, 'S']
        dm = [None, None, None]
        A, F, info = build_a_matrix(ids, sr, dm)
        names = info['individuals']
        pos = {v: i for i, v in enumerate(names)}
        self.assertAlmostEqual(float(A[pos['C'], pos['S']]), 0.5)
        self.assertAlmostEqual(float(A[pos['C'], pos['C']]), 1.0)
        self.assertAlmostEqual(float(A[pos['C'], pos['D']]), 0.0)

    def test_hermanos_completos(self):
        # A[O1,O2] = 0.5 (comparten 50%)
        ids = ['P', 'M', 'O1', 'O2']
        sr = [None, None, 'P', 'P']
        dm = [None, None, 'M', 'M']
        A, _F, info = build_a_matrix(ids, sr, dm)
        names = info['individuals']
        pos = {v: i for i, v in enumerate(names)}
        self.assertAlmostEqual(float(A[pos['O1'], pos['O2']]), 0.5)

    def test_medios_hermanos(self):
        # A[H1,H2] = 0.25 (mismo padre, madres distintas no emparentadas)
        ids = ['P', 'M1', 'M2', 'H1', 'H2']
        sr = [None, None, None, 'P', 'P']
        dm = [None, None, None, 'M1', 'M2']
        A, _F, info = build_a_matrix(ids, sr, dm)
        names = info['individuals']
        pos = {v: i for i, v in enumerate(names)}
        self.assertAlmostEqual(float(A[pos['H1'], pos['H2']]), 0.25)

    def test_endogamia_f(self):
        # E hijo de hermanos completos C,D: F(E)=0.5*A[C,D]=0.25
        ids = ['P', 'M', 'C', 'D', 'E']
        sr = [None, None, 'P', 'P', 'C']
        dm = [None, None, 'M', 'M', 'D']
        _A, F, info = build_a_matrix(ids, sr, dm)
        names = info['individuals']
        self.assertAlmostEqual(float(F[names.index('E')]), 0.25)

    def test_orden_irrelevante(self):
        # El mismo pedigrí en distinto orden da el mismo F(E)
        ids = ['E', 'D', 'C', 'M', 'P']
        sr = ['C', 'P', 'P', None, None]
        dm = ['D', 'M', 'M', None, None]
        _A, F, info = build_a_matrix(ids, sr, dm)
        self.assertAlmostEqual(
            float(F[info['individuals'].index('E')]), 0.25)

    def test_a_por_ainv_identidad(self):
        ids = ['P', 'M', 'C', 'D', 'E']
        sr = [None, None, 'P', 'P', 'C']
        dm = [None, None, 'M', 'M', 'D']
        A, _F, _info = build_a_matrix(ids, sr, dm)
        Ainv, _F2, _i2 = build_a_inverse(ids, sr, dm)
        prod = A @ Ainv
        eye = torch.eye(5, dtype=torch.float64)
        self.assertTrue(torch.allclose(prod, eye, atol=1e-10))

    def test_ciclo_error(self):
        with self.assertRaises(PedigreeError):
            normalize_pedigree(['A', 'B'], ['B', 'A'], [None, None])

    def test_autopadre_error(self):
        with self.assertRaises(PedigreeError):
            normalize_pedigree(['A'], ['A'], [None])

    def test_longitudes_error(self):
        with self.assertRaises(PedigreeError):
            normalize_pedigree(['A', 'B'], ['A'], [None, None])

    def test_ids_duplicados_error(self):
        with self.assertRaises(PedigreeError):
            normalize_pedigree(['A', 'A'], [None, None], [None, None])

    def test_padres_no_listados(self):
        _ids, _sr, _dm, info = normalize_pedigree(['C'], ['X'], ['Y'])
        self.assertEqual(info['unknown_parents'], 2)


class TestPedigreeBLUP(unittest.TestCase):
    """PBLUP con varianzas dadas y REML, fiabilidades y equivalencia GBLUP."""

    def _pedigree(self):
        ids = ['P', 'M', 'C', 'D', 'E', 'F']
        sr = [None, None, 'P', 'P', 'C', 'D']
        dm = [None, None, 'M', 'M', 'D', 'M']
        y = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        X = [[1.0] for _ in range(6)]
        return ids, sr, dm, y, X

    def test_varianzas_dadas(self):
        ids, sr, dm, y, X = self._pedigree()
        model = PedigreeBLUP(device='cpu')
        r = model.fit_pedigree(y, X, ids, sr, dm,
                               genetic_variance=1.0, residual_variance=1.0)
        self.assertEqual(r.variance_source, 'dadas')
        self.assertEqual(r.breeding_values.shape[0], 6)
        self.assertTrue((model.reliabilities() >= 0).all())
        self.assertTrue((model.reliabilities() <= 1).all())

    def test_reml(self):
        ids, sr, dm, y, X = self._pedigree()
        model = PedigreeBLUP(device='cpu')
        r = model.fit_pedigree(y, X, ids, sr, dm)
        self.assertEqual(r.variance_source, 'reml')
        self.assertGreater(r.genetic_variance, 0)
        self.assertGreater(r.residual_variance, 0)
        self.assertTrue((model.reliabilities() >= 0).all())
        self.assertTrue((model.reliabilities() <= 1).all())

    def test_equivalencia_gblup(self):
        # PedigreeBLUP(y,X,A) == GenomicBLUP(y,X,A) cuando y,X están en el
        # mismo orden topológico que A
        from Genoly.quantitative.gblup import GenomicBLUP
        ids, sr, dm, y, X = self._pedigree()
        A, F, info = build_a_matrix(ids, sr, dm)
        names = info['individuals']
        pos = {v: i for i, v in enumerate(ids)}
        idx = [pos[v] for v in names]
        y_ord = [y[i] for i in idx]
        X_ord = [[1.0] for _ in range(6)]
        g = GenomicBLUP(device='cpu').fit(
            y_ord, X_ord, A, genetic_variance=1.0, residual_variance=1.0)
        p = PedigreeBLUP(device='cpu').fit(
            y_ord, X_ord, A, genetic_variance=1.0, residual_variance=1.0,
            inbreeding=F.detach().clone(), individuals=names)
        self.assertTrue(torch.allclose(
            g.breeding_values, p.breeding_values, atol=1e-8))
        self.assertTrue(torch.allclose(
            g.reliabilities, p.reliabilities, atol=1e-8))

    def test_inbreeding_en_resultado(self):
        ids, sr, dm, y, X = self._pedigree()
        model = PedigreeBLUP(device='cpu')
        r = model.fit_pedigree(y, X, ids, sr, dm,
                               genetic_variance=1.0, residual_variance=1.0)
        f = model.inbreeding()
        self.assertEqual(f.shape[0], 6)
        # F[E]=0.25 (hijo de hermanos completos C,D)
        self.assertAlmostEqual(
            float(f[r.individuals.index('E')]), 0.25, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)