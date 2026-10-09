"""
Núcleo BLUP compartido por GBLUP y PBLUP.

Dadas las componentes de varianza (fijadas o estimadas por REML), resuelve
el modelo mixto y calcula fiabilidades por el PEV:

    û = σ_g² K Z' V⁻¹ (y - Xβ̂),   V = σ_g² Z K Z' + σ_e² I
    PEV(u) = σ_g² K - σ_g⁴ K Z' P Z K
    fiabilidad_i = 1 - PEV_i / (σ_g² K_ii)

GBLUP y PBLUP sólo difieren en K (GRM genómica vs A de pedigrí).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from Genoly.quantitative.reml import solve_variance_components
from Genoly.quantitative.utils import cholesky_regularized, prepare_model_inputs

__all__ = ['BLUPResult', 'fit_blup']


@dataclass
class BLUPResult:
    """Resultado del BLUP genético (compartido GBLUP/PBLUP)."""
    breeding_values: torch.Tensor
    fixed_effects: torch.Tensor
    reliabilities: torch.Tensor
    genetic_variance: float
    residual_variance: float
    log_likelihood: Optional[float]
    variance_source: str


def fit_blup(y, X, K, Z=None,
             genetic_variance: Optional[float] = None,
             residual_variance: Optional[float] = None,
             max_iter: int = 100, tol: float = 1e-8,
             device: Optional[str] = None,
             dtype: torch.dtype = torch.float64) -> Tuple[BLUPResult, torch.Tensor, torch.Tensor]:
    """
    Ajusta el modelo mixto y predice los valores de cría.

    Args:
        y: Fenotipos (n,).
        X: Diseño de efectos fijos (n, p).
        K: Matriz de parentesco (q, q); GRM en GBLUP, A en PBLUP.
        Z: Diseño de efectos aleatorios (n, q); None usa la identidad.
        genetic_variance: σ_g² conocida; requiere también residual_variance.
        residual_variance: σ_e² conocida.
        max_iter: Iteraciones REML máximas.
        tol: Tolerancia REML.
        device: 'cuda', 'cpu' o None para auto-detectar.
        dtype: Tipo de punto flotante.

    Returns:
        (BLUPResult, X_cpu, Z_cpu) — los tensores de diseño en CPU se
        devuelven junto al resultado para poder calcular fitted = Xβ + Zu.

    Raises:
        ValueError: Una sola varianza dada, o formas inconsistentes.
    """
    from Genoly.core.device import DeviceManager

    given = (genetic_variance is not None, residual_variance is not None)
    if any(given) and not all(given):
        raise ValueError(
            "Proporciona ambas varianzas (genética y residual) o ninguna "
            "para estimarlas por REML"
        )

    manager = DeviceManager(device)
    dev = manager.device

    y_t, X_t, K_t, Z_t = prepare_model_inputs(
        y, X, K, Z, device=dev, dtype=dtype)

    loglik = None
    if genetic_variance is None:
        est = solve_variance_components(y_t, X_t, K_t, Z_t,
                                        method="reml",
                                        max_iter=max_iter,
                                        tol=tol)
        var_g = est.genetic_variance
        var_e = est.residual_variance
        loglik = est.log_likelihood
        source = "reml"
    else:
        var_g = float(genetic_variance)
        var_e = float(residual_variance)
        source = "dadas"

    n = y_t.shape[0]
    q = K_t.shape[0]
    eye_n = torch.eye(n, dtype=dtype, device=dev)

    V = var_g * (Z_t @ K_t @ Z_t.T) + var_e * eye_n
    L = cholesky_regularized(V)
    Vinv = torch.cholesky_solve(eye_n, L)
    Vinv_X = Vinv @ X_t
    XtViX = X_t.T @ Vinv_X
    L_fix = cholesky_regularized(XtViX)
    beta = torch.cholesky_solve(X_t.T @ (Vinv @ y_t.unsqueeze(1)),
                                L_fix).reshape(-1)
    resid = y_t.unsqueeze(1) - X_t @ beta.unsqueeze(1)
    breeding_values = var_g * (K_t @ (Z_t.T @ (Vinv @ resid))).reshape(-1)

    proj = Vinv - Vinv_X @ torch.cholesky_solve(Vinv_X.T, L_fix)
    kz = K_t @ Z_t.T
    if var_g > 0.0:
        pev = var_g * torch.diagonal(K_t) \
            - var_g ** 2 * torch.diagonal(kz @ proj @ Z_t @ K_t)
        k_diag = torch.diagonal(K_t).clamp(min=1e-12)
        reliability = (1.0 - pev / (var_g * k_diag)).clamp(0.0, 1.0)
    else:
        reliability = torch.zeros(q, dtype=dtype, device=dev)

    result = BLUPResult(
        breeding_values=breeding_values.detach().cpu(),
        fixed_effects=beta.detach().cpu(),
        reliabilities=reliability.detach().cpu(),
        genetic_variance=float(var_g),
        residual_variance=float(var_e),
        log_likelihood=loglik,
        variance_source=source,
    )
    manager.empty_cache()
    return result, X_t.detach().cpu(), Z_t.detach().cpu()
