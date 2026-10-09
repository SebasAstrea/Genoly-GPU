"""
Tests del hito 4: análisis descendente (PCA, t-SNE, K-Means).

Ejecutar desde la raiz del proyecto:
    python -m pytest tests/test_downstream.py -v
"""

import asyncio
import time
import unittest

import torch

from Genoly.downstream.reduction import (
    DownstreamAnalyzer,
    KMeans,
    PCAReducer,
    TSNEReducer,
)
from ui.backend import jobs


def _run_job(spec):
    async def main():
        job = jobs.manager.create(spec.get("kind", "downstream"))
        jobs.manager.submit_process(job, spec)
        deadline = time.monotonic() + 120
        while job.status not in ("done", "error"):
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"El job {job.id} no terminó en 120 s")
            await asyncio.sleep(0.05)
        return job
    return asyncio.run(main())


def _clustered_matrix(groups=3, per_group=50, dim=20, spread=0.5, seed=1):
    torch.manual_seed(seed)
    centers = [torch.randn(dim) * 5 + g * 12 for g in range(groups)]
    X = torch.cat([c + torch.randn(per_group, dim) * spread
                   for c in centers])
    truth = [g for g in range(groups) for _ in range(per_group)]
    return X, truth


def _purity(labels, truth):
    from collections import Counter
    k = max(labels) + 1
    total = 0
    for c in range(k):
        members = [truth[i] for i, lab in enumerate(labels) if lab == c]
        if members:
            total += Counter(members).most_common(1)[0][1]
    return total / len(truth)


class TestReduction(unittest.TestCase):
    def test_pca_varianza_y_forma(self):
        X, _ = _clustered_matrix()
        res = PCAReducer('cpu').fit(X, n_components=3)
        self.assertEqual(tuple(res.scores.shape), (150, 3))
        ratio = res.explained_variance_ratio.tolist()
        self.assertGreater(ratio[0], 0.7)
        self.assertAlmostEqual(sum(ratio), 1.0, delta=0.01)

    def test_kmeans_pureza(self):
        X, truth = _clustered_matrix()
        res = KMeans('cpu').fit(X, k=3, n_init=5)
        self.assertEqual(len(res['labels']), 150)
        self.assertGreaterEqual(_purity(res['labels'], truth), 0.95)

    def test_tsne_separa_grupos(self):
        X, truth = _clustered_matrix(groups=2, per_group=40, dim=10)
        emb = TSNEReducer('cpu').fit(X, perplexity=15, n_iter=200)
        self.assertEqual(tuple(emb.shape), (80, 2))
        # los dos grupos deben quedar separados (distancia intermedia mayor
        # que la intra-grupo)
        a = emb[:40].mean(dim=0)
        b = emb[40:].mean(dim=0)
        self.assertGreater(float(torch.dist(a, b)), 0.5)

    def test_analyzer_completo(self):
        X, truth = _clustered_matrix()
        res = DownstreamAnalyzer('cpu').run(
            X, pca_components=2, run_tsne=True, tsne_perplexity=20,
            tsne_iter=200, k=3)
        self.assertEqual(res['n'], 150)
        self.assertEqual(len(res['pca']['scores']), 150)
        self.assertEqual(len(res['tsne']), 150)
        self.assertEqual(len(res['clusters']['labels']), 150)
        self.assertGreaterEqual(
            _purity(res['clusters']['labels'], truth), 0.95)


class TestDownstreamJobAislado(unittest.TestCase):
    def test_downstream_en_proceso(self):
        X, truth = _clustered_matrix(per_group=30)
        job = _run_job({"kind": "downstream", "matrix": X.tolist(),
                        "pca_components": 2, "run_tsne": False,
                        "k": 3, "standardize": True})
        self.assertEqual(job.status, "done", job.error)
        self.assertEqual(job.result['n'], 90)
        self.assertGreaterEqual(
            _purity(job.result['clusters']['labels'], truth), 0.9)


if __name__ == "__main__":
    unittest.main(verbosity=2)