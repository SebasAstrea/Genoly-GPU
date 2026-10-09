"""
PBLUP: predicción de valores de cría por modelo animal (pedigrí).

Igual que GBLUP pero con matriz de parentesco aditiva A construida a
partir del pedigrí (sire/dam) en lugar de una GRM genómica. Requiere
que los datos hayan pasado por la limpieza con columnas de pedigrí.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from Genoly.core.device import DeviceManager
from Genoly.quantitative.blup import fit_blup
from Genoly.quantitative.pedigree import build_a_matrix

__all__ = ['PBLUPResult', 'PedigreeBLUP']


@dataclass
class PBLUPResult:
    """Resultado de la predicción por modelo animal (PBLUP)."""
    breeding_values: torch.Tensor
    fixed_effects: torch.Tensor
    reliabilities: torch.Tensor
    genetic_variance: float
    residual_variance: float
    log_likelihood: Optional[float]
    variance_source: str
    inbreeding: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    founders: List[int] = field(default_factory=list)
    unknown_parents: int = 0
    individuals: List[str] = field(default_factory=list)


class PedigreeBLUP:
    """
    Predicción por modelo animal (PBLUP) sobre GPU.

    Construye la matriz de parentesco aditiva A a partir del pedigrí
    (individuos, sires, dams) y resuelve el mismo modelo mixto que
    GenomicBLUP, sólo que con K = A en lugar de la GRM.

    La fiabilidad usa el PEV del BLUP con efectos fijos estimados:

        PEV(u) = diag(A - A Z' P Z A)
        P = V^-1 - V^-1 X (X' V^-1 X)^-1 X' V^-1,  V = σ_g² Z A Z' + σ_e² I

    y reliability_i = 1 - PEV(u)_i / (σ_g² A_ii), truncado a [0, 1].
    """

    def __init__(self, device: Optional[str] = None):
        self.manager = DeviceManager(device)
        self.device = self.manager.device
        self.dtype = torch.float64
        self.result_: Optional[PBLUPResult] = None

    def fit(self, y, X, A, Z=None,
            genetic_variance: Optional[float] = None,
            residual_variance: Optional[float] = None,
            max_iter: int = 100, tol: float = 1e-8,
            inbreeding: Optional[torch.Tensor] = None,
            founders: Optional[List[int]] = None,
            unknown_parents: int = 0,
            individuals: Optional[List[str]] = None) -> PBLUPResult:
        """
        Ajusta el modelo animal y predice los valores de cría.

        Args:
            y: Fenotipos (n,).
            X: Diseño de efectos fijos (n, p).
            A: Matriz de parentesco aditiva (n, n); ver build_a_matrix.
            Z: Diseño de efectos aleatorios (n, n); None usa la identidad.
            genetic_variance: σ_g² conocida; requiere también residual_variance.
            residual_variance: σ_e² conocida.
            max_iter: Iteraciones REML máximas.
            tol: Tolerancia REML.
            inbreeding: F por individuo (opcional, se propaga al resultado).
            founders: Índices de fundadores (opcional).
            unknown_parents: Nº de referencias a padres no listados.
            individuals: IDs de los individuos en el mismo orden que A.

        Returns:
            PBLUPResult con valores de cría, fiabilidades, endogamia y
            metadatos del pedigrí.
        """
        core, X_cpu, Z_cpu = fit_blup(
            y, X, A, Z,
            genetic_variance=genetic_variance,
            residual_variance=residual_variance,
            max_iter=max_iter, tol=tol,
            device=self.device, dtype=self.dtype)
        self.result_ = PBLUPResult(
            breeding_values=core.breeding_values,
            fixed_effects=core.fixed_effects,
            reliabilities=core.reliabilities,
            genetic_variance=core.genetic_variance,
            residual_variance=core.residual_variance,
            log_likelihood=core.log_likelihood,
            variance_source=core.variance_source,
            inbreeding=inbreeding if inbreeding is not None else torch.empty(0),
            founders=founders or [],
            unknown_parents=unknown_parents,
            individuals=individuals or [],
        )
        self._X_cpu = X_cpu
        self._Z_cpu = Z_cpu
        return self.result_

    def fit_pedigree(self, y, X, individuals, sires, dams, Z=None,
                     genetic_variance: Optional[float] = None,
                     residual_variance: Optional[float] = None,
                     max_iter: int = 100, tol: float = 1e-8) -> PBLUPResult:
        """
        Construye A desde el pedigrí crudo y ajusta el modelo animal.

        Args:
            y: Fenotipos (n,).
            X: Diseño de efectos fijos (n, p).
            individuals: IDs de los n individuos (en el mismo orden que y).
            sires: IDs de los padres (None si desconocido).
            dams: IDs de las madres (None si desconocido).
            Z, genetic_variance, residual_variance, max_iter, tol: igual que fit.

        Returns:
            PBLUPResult.

        Raises:
            PedigreeError: Si el pedigrí es inválido (ciclos, IDs duplicados…).
        """
        A, F, info = build_a_matrix(individuals, sires, dams)
        # reordenar y para coincidir con el orden topológico de A
        order = info["individuals"]
        pos = {v: i for i, v in enumerate(individuals)}
        idx = [pos[v] for v in order]
        if isinstance(y, torch.Tensor):
            y_ordered = y[idx]
        else:
            y_ordered = [y[i] for i in idx]
        if Z is None:
            Z_ordered = None
        else:
            Z_t = Z if isinstance(Z, torch.Tensor) else torch.tensor(Z, dtype=self.dtype)
            Z_ordered = Z_t[idx]
        X_t = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=self.dtype)
        X_ordered = X_t[idx]
        return self.fit(
            y_ordered, X_ordered, A, Z=Z_ordered,
            genetic_variance=genetic_variance,
            residual_variance=residual_variance,
            max_iter=max_iter, tol=tol,
            inbreeding=F.detach().clone() if isinstance(F, torch.Tensor)
            else torch.tensor(F, dtype=self.dtype),
            founders=info["founders"],
            unknown_parents=info["unknown_parents"],
            individuals=order,
        )

    def _require_fitted(self) -> None:
        if self.result_ is None:
            raise RuntimeError("El modelo debe ajustarse con fit() antes de usar esta operación")

    def blup(self) -> torch.Tensor:
        """Valores de cría predichos (BLUP del modelo animal). Tensor (n,) en CPU."""
        self._require_fitted()
        return self.result_.breeding_values

    def blue(self) -> torch.Tensor:
        """Efectos fijos estimados (β). Tensor (p,) en CPU."""
        self._require_fitted()
        return self.result_.fixed_effects

    def reliabilities(self) -> torch.Tensor:
        """Fiabilidad de cada valor de cría. Tensor (n,) en [0, 1], en CPU."""
        self._require_fitted()
        return self.result_.reliabilities

    def accuracies(self) -> torch.Tensor:
        """Precisión de cada valor de cría (√fiabilidad). Tensor (n,) en [0, 1]."""
        self._require_fitted()
        return torch.sqrt(self.result_.reliabilities)

    def inbreeding(self) -> torch.Tensor:
        """Coeficiente de endogamia F de cada individuo. Tensor (n,) en CPU."""
        self._require_fitted()
        return self.result_.inbreeding

    def predict(self) -> torch.Tensor:
        """Valores ajustados del modelo (Xβ + Zu). Tensor (n,) en CPU."""
        self._require_fitted()
        fitted = self._X_cpu @ self.result_.fixed_effects.unsqueeze(1) \
            + self._Z_cpu @ self.result_.breeding_values.unsqueeze(1)
        return fitted.reshape(-1)
