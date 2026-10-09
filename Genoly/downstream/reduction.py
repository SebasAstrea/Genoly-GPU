"""
Análisis descendente: reducción de dimensionalidad y clustering.

Opera sobre una matriz de rasgos (individuos x marcadores/features), p. ej.
dosis alélicas, conteos de expresión o cualquier perfil ómico numérico:

- **PCA** por SVD (con estandarización opcional), sobre torch/GPU.
- **t-SNE** (Barnes-Hut aproximado por vecinos k, afinidad gaussiana y
  descenso de gradiente), sobre torch/GPU.
- **K-Means++** con inicialización k-means++ y múltiples reinicios.

Los tensores viven en el dispositivo detectado (CUDA si hay GPU); los
resultados se devuelven como listas Python listas para JSON.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from Genoly.core.device import DeviceManager


def _as_tensor(X, device) -> torch.Tensor:
    """Convierte una matriz (lista de listas) a tensor float32."""
    if isinstance(X, torch.Tensor):
        return X.to(device=device, dtype=torch.float32)
    return torch.tensor(X, dtype=torch.float32, device=device)


@dataclass
class PCAResult:
    scores: torch.Tensor       # (n, n_components)
    loadings: torch.Tensor     # (n_components, p)
    explained_variance: torch.Tensor
    explained_variance_ratio: torch.Tensor


class PCAReducer:
    """Análisis de componentes principales por SVD."""

    def __init__(self, device: Optional[str] = None):
        self.device = DeviceManager(device).device

    def fit(self, X, n_components: int = 2, standardize: bool = True,
            on_progress=None) -> PCAResult:
        """
        Ajusta PCA sobre la matriz ``X`` (n individuos x p rasgos).

        Args:
            X: Matriz de rasgos (lista de listas o tensor).
            n_components: Número de componentes a retener.
            standardize: Estandarizar cada rasgo (media 0, desviación 1).
            on_progress: Callback opcional ``fn(info)``.

        Returns:
            :class:`PCAResult` con scores, loadings y varianza explicada.
        """
        M = _as_tensor(X, self.device)
        if M.dim() != 2:
            raise ValueError("X debe ser una matriz 2D")
        n, p = M.shape
        if n_components < 1 or n_components > min(n, p):
            raise ValueError(
                f"n_components debe estar en [1, {min(n, p)}]")

        if on_progress:
            on_progress({"stage": "pca", "n": n, "p": p})

        mean = M.mean(dim=0, keepdim=True)
        Mc = M - mean
        if standardize:
            std = Mc.std(dim=0, unbiased=False).clamp(min=1e-8)
            Mc = Mc / std

        # SVD: Mc = U S V^T -> scores = U S
        U, S, Vh = torch.linalg.svd(Mc, full_matrices=False)
        scores = U[:, :n_components] * S[:n_components]
        loadings = Vh[:n_components, :]
        var = (S ** 2) / max(1, n - 1)
        total = var.sum().clamp(min=1e-12)
        return PCAResult(
            scores=scores.cpu(),
            loadings=loadings.cpu(),
            explained_variance=var[:n_components].cpu(),
            explained_variance_ratio=(var[:n_components] / total).cpu(),
        )


class TSNEReducer:
    """t-SNE aproximado (afinidad gaussiana + descenso de gradiente)."""

    def __init__(self, device: Optional[str] = None):
        self.device = DeviceManager(device).device

    def fit(self, X, n_components: int = 2, perplexity: float = 30.0,
            learning_rate: float = 200.0, n_iter: int = 500,
            standardize: bool = True, seed: int = 0,
            on_progress=None) -> torch.Tensor:
        """
        Calcula la incrustación t-SNE de ``X``.

        Implementación estándar (van der Maaten & Hinton): afinidades
        gaussianas en el espacio original (con ``perplexity``), colas
        pesadas t-Student en el espacio latente y descenso de gradiente
        con momento. La afinidad usa los ``3*perplexity`` vecinos más
        cercanos por distancia euclídea.

        Returns:
            Tensor (n, n_components) en CPU.
        """
        M = _as_tensor(X, self.device)
        if M.dim() != 2:
            raise ValueError("X debe ser una matriz 2D")
        n = M.shape[0]
        if n < 3:
            raise ValueError("t-SNE requiere al menos 3 individuos")

        if standardize:
            Mc = M - M.mean(dim=0, keepdim=True)
            Mc = Mc / Mc.std(dim=0, unbiased=False).clamp(min=1e-8)
        else:
            Mc = M

        n_neighbors = min(n - 1, max(3, int(3 * perplexity)))

        # distancias y vecinos
        dist = torch.cdist(Mc, Mc)
        knn_d, knn_i = torch.topk(dist, n_neighbors + 1, largest=False)
        knn_d = knn_d[:, 1:]          # sin el propio punto
        knn_i = knn_i[:, 1:]

        # sigma por punto: búsqueda binaria para que la entropía ~ perplexity
        target = torch.log(torch.tensor(perplexity, device=self.device))
        sigma = torch.ones(n, device=self.device)
        for _ in range(20):
            P = torch.exp(-knn_d / (2 * sigma.unsqueeze(1) ** 2))
            P = P / P.sum(dim=1, keepdim=True).clamp(min=1e-12)
            entropy = -(P * torch.log(P.clamp(min=1e-12))).sum(dim=1)
            sigma = sigma * torch.exp((entropy - target) * 0.5)
            sigma = sigma.clamp(min=1e-3, max=1e3)

        # matriz P simétrica y normalizada
        Pmat = torch.zeros(n, n, device=self.device)
        Pmat.scatter_(1, knn_i, P)
        Pmat = (Pmat + Pmat.t()) / (2 * n)
        Pmat = Pmat.clamp(min=1e-12)

        # inicialización aleatoria (reproducible: siembra CPU y CUDA; el
        # generador no servía porque torch.randn en GPU exige un
        # generator del mismo device)
        torch.manual_seed(seed)
        Y = torch.randn(n, n_components, device=self.device) * 1e-4
        velocity = torch.zeros_like(Y)
        momentum = 0.5

        for it in range(n_iter):
            if it == 250:
                momentum = 0.8
            # matriz Q con colas t-Student (grado 1)
            num = 1.0 / (1.0 + torch.cdist(Y, Y) ** 2)
            num.fill_diagonal_(0.0)
            q = num / num.sum().clamp(min=1e-12)
            q = q.clamp(min=1e-12)

            # gradiente: 4 * (P - Q) ponderado por num
            pq = (Pmat - q) * num
            grad = 4.0 * (pq.sum(dim=1, keepdim=True) * Y
                          - pq @ Y)
            velocity = momentum * velocity - learning_rate * grad
            Y = Y + velocity
            Y = Y - Y.mean(dim=0, keepdim=True)  # centrar
            if on_progress and (it % max(1, n_iter // 10) == 0):
                on_progress({"stage": "tsne", "iter": it, "n_iter": n_iter})

        return Y.cpu()


class KMeans:
    """K-Means con inicialización k-means++ y múltiples reinicios."""

    def __init__(self, device: Optional[str] = None):
        self.device = DeviceManager(device).device

    def _kmeans_plusplus(self, X: torch.Tensor, k: int,
                         gen: torch.Generator) -> torch.Tensor:
        n = X.shape[0]
        Xc = X.cpu()
        centers = [Xc[torch.randint(n, (1,), generator=gen).item()]]
        for _ in range(1, k):
            d2 = torch.cdist(Xc, torch.stack(centers)).min(dim=1).values ** 2
            probs = d2 / d2.sum().clamp(min=1e-12)
            idx = torch.multinomial(probs, 1, generator=gen).item()
            centers.append(Xc[idx])
        return torch.stack(centers).to(X.device)

    def fit(self, X, k: int = 3, n_init: int = 10, max_iter: int = 300,
            seed: int = 0, on_progress=None) -> Dict[str, object]:
        """
        Agrupa ``X`` en ``k`` clusters.

        Returns:
            Dict con ``labels`` (lista), ``centroids`` (lista de listas),
            ``inertia`` y ``n_iter``.
        """
        M = _as_tensor(X, self.device)
        if M.dim() != 2:
            raise ValueError("X debe ser una matriz 2D")
        n = M.shape[0]
        if k < 1 or k > n:
            raise ValueError(f"k debe estar en [1, {n}]")

        gen = torch.Generator(device="cpu").manual_seed(seed)
        best = None
        for init in range(max(1, n_init)):
            centers = self._kmeans_plusplus(M, k, gen)
            labels = torch.zeros(n, dtype=torch.long, device=self.device)
            for it in range(max_iter):
                d = torch.cdist(M, centers)
                new_labels = d.argmin(dim=1)
                if torch.equal(new_labels, labels) and it > 0:
                    break
                labels = new_labels
                for c in range(k):
                    mask = labels == c
                    if mask.any():
                        centers[c] = M[mask].mean(dim=0)
            inertia = float(torch.cdist(M, centers).min(dim=1).values
                            .pow(2).sum().item())
            if best is None or inertia < best["inertia"]:
                best = {"labels": labels.cpu(), "centroids": centers.cpu(),
                        "inertia": inertia, "n_iter": it + 1,
                        "init": init}
            if on_progress:
                on_progress({"stage": "kmeans", "init": init + 1,
                             "n_init": n_init, "inertia": inertia})

        return {
            "labels": [int(v) for v in best["labels"].tolist()],
            "centroids": best["centroids"].tolist(),
            "inertia": round(best["inertia"], 4),
            "n_iter": best["n_iter"],
        }


class DownstreamAnalyzer:
    """Fachada de los análisis descendentes sobre una matriz de rasgos."""

    def __init__(self, device: Optional[str] = None):
        self.device = DeviceManager(device).device
        self.pca = PCAReducer(device)
        self.tsne = TSNEReducer(device)
        self.kmeans = KMeans(device)

    def run(self, X, pca_components: int = 2, run_tsne: bool = True,
            tsne_perplexity: float = 30.0, tsne_iter: int = 500,
            k: int = 3, standardize: bool = True,
            on_progress=None) -> Dict[str, object]:
        """
        Ejecuta PCA, t-SNE (opcional) y K-Means sobre ``X``.

        Returns:
            Dict con ``pca`` (scores, varianza), ``tsne`` (coords),
            ``clusters`` (labels, centroides, inercia) y ``n``/``p``.
        """
        M = _as_tensor(X, self.device)
        n, p = M.shape
        result: Dict[str, object] = {"n": n, "p": p}

        pca_res = self.pca.fit(M, n_components=pca_components,
                               standardize=standardize, on_progress=on_progress)
        result["pca"] = {
            "scores": pca_res.scores.tolist(),
            "explained_variance_ratio": [
                round(float(v), 6)
                for v in pca_res.explained_variance_ratio.tolist()],
            "loadings": pca_res.loadings.tolist(),
        }

        # clustering sobre las componentes PCA (reduce ruido y coste)
        feats = pca_res.scores
        if feats.shape[1] < 2:
            feats = _as_tensor(M, self.device)
        clusters = self.kmeans.fit(feats, k=k, on_progress=on_progress)
        result["clusters"] = clusters

        if run_tsne:
            emb = self.tsne.fit(M, perplexity=tsne_perplexity,
                                n_iter=tsne_iter, standardize=standardize,
                                on_progress=on_progress)
            result["tsne"] = emb.tolist()

        return result