"""
GBLUP: predicción genómica de valores de cría acelerada por GPU.

Fachada sobre el núcleo BLUP compartido (``blup.fit_blup``) con matriz de
parentesco genómica (GRM). La matemática, el cálculo de fiabilidades y la
estimación REML viven en ``Genoly.quantitative.blup``; aquí sólo se
empaqueta el resultado con la API histórica de GBLUP.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from Genoly.core.device import DeviceManager
from Genoly.quantitative.blup import fit_blup

__all__ = ['GBLUPResult', 'GenomicBLUP']


@dataclass
class GBLUPResult:
    """Resultado de la predicción genómica GBLUP."""
    breeding_values: torch.Tensor
    fixed_effects: torch.Tensor
    reliabilities: torch.Tensor
    genetic_variance: float
    residual_variance: float
    log_likelihood: Optional[float]
    variance_source: str


class GenomicBLUP:
    """
    Predicción genómica GBLUP sobre GPU.

    A diferencia de LinearMixedModel (que siempre estima las componentes de
    varianza), permite fijarlas directamente: si se conocen de estudios
    previos la solución es directa, sin iterar, e incluye la fiabilidad y la
    precisión de cada valor de cría. Si no se proporcionan, se estiman por
    REML antes de resolver.

    Los efectos fijos se ESTIMAN por BLUE (beta_hat) y los valores de cría
    son u = G Z' V^-1 (y - X beta_hat). Las fiabilidades usan el PEV del BLUP
    con efectos fijos estimados, equivalente al bloque inferior derecho de la
    inversa de las ecuaciones del modelo mixto de Henderson:

        PEV(u) = diag(G - G Z' P Z G)
        P = V^-1 - V^-1 X (X' V^-1 X)^-1 X' V^-1,  G = sigma_g^2 K

    y reliability_i = 1 - PEV(u)_i / (sigma_g^2 * k_ii), truncado a [0, 1].
    La diferencia entre T1 y T2 del PEV se calcula fusionada dentro de P para
    evitar la cancelación catastrófica de restar diagonales grandes por
    separado.
    """

    def __init__(self, device: Optional[str] = None):
        """
        Args:
            device: 'cuda', 'cpu' o None para auto-detectar.
        """
        self.manager = DeviceManager(device)
        self.device = self.manager.device
        self.dtype = torch.float64
        self.result_: Optional[GBLUPResult] = None

    def fit(self, y, X, K, Z=None,
            genetic_variance: Optional[float] = None,
            residual_variance: Optional[float] = None,
            max_iter: int = 100, tol: float = 1e-8) -> GBLUPResult:
        """
        Ajusta el modelo y predice los valores de cría.

        Args:
            y: Fenotipos (n,).
            X: Diseño de efectos fijos (n, p), incluido el intercepto si procede.
            K: Matriz de parentesco (q, q); usar build_kinship para obtenerla.
            Z: Diseño de efectos aleatorios (n, q); None usa la identidad.
            genetic_variance: Varianza genética conocida (σ_g²). Si se indica,
                              debe indicarse también residual_variance.
            residual_variance: Varianza residual conocida (σ_e²).
            max_iter: Iteraciones máximas si hay que estimar por REML.
            tol: Tolerancia de convergencia si hay que estimar por REML.

        Returns:
            GBLUPResult con valores de cría, efectos fijos, fiabilidades y
            varianzas empleadas.

        Raises:
            ValueError: Si se proporciona una sola varianza o las formas no
                        son consistentes.
        """
        core, X_cpu, Z_cpu = fit_blup(
            y, X, K, Z,
            genetic_variance=genetic_variance,
            residual_variance=residual_variance,
            max_iter=max_iter, tol=tol,
            device=self.device, dtype=self.dtype)
        self.result_ = GBLUPResult(
            breeding_values=core.breeding_values,
            fixed_effects=core.fixed_effects,
            reliabilities=core.reliabilities,
            genetic_variance=core.genetic_variance,
            residual_variance=core.residual_variance,
            log_likelihood=core.log_likelihood,
            variance_source=core.variance_source,
        )
        self._X_cpu = X_cpu
        self._Z_cpu = Z_cpu
        return self.result_

    def _require_fitted(self) -> None:
        if self.result_ is None:
            raise RuntimeError("El modelo debe ajustarse con fit() antes de usar esta operación")

    def blup(self) -> torch.Tensor:
        """
        Devuelve los valores de cría genómicos predichos (GEBV).

        Returns:
            Tensor (q,) en CPU.
        """
        self._require_fitted()
        return self.result_.breeding_values

    def blue(self) -> torch.Tensor:
        """
        Devuelve los estimadores de los efectos fijos (β).

        Returns:
            Tensor (p,) en CPU.
        """
        self._require_fitted()
        return self.result_.fixed_effects

    def reliabilities(self) -> torch.Tensor:
        """
        Devuelve la fiabilidad de cada valor de cría (1 - PEV/Var(u)).

        Returns:
            Tensor (q,) en [0, 1], en CPU.
        """
        self._require_fitted()
        return self.result_.reliabilities

    def accuracies(self) -> torch.Tensor:
        """
        Devuelve la precisión de cada valor de cría (raíz de la fiabilidad).

        Returns:
            Tensor (q,) en [0, 1], en CPU.
        """
        self._require_fitted()
        return torch.sqrt(self.result_.reliabilities)

    def predict(self) -> torch.Tensor:
        """
        Devuelve los valores ajustados del modelo (Xβ + Zu).

        Returns:
            Tensor (n,) en CPU.
        """
        self._require_fitted()
        fitted = self._X_cpu @ self.result_.fixed_effects.unsqueeze(1) \
            + self._Z_cpu @ self.result_.breeding_values.unsqueeze(1)
        return fitted.reshape(-1)
