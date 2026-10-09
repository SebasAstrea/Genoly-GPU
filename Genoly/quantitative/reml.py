"""
Estimación de componentes de varianza por máxima verosimilitud (restringida).

Motor de optimización compartido por los modelos del paquete: ascenso por
puntuación de Fisher con búsqueda de línea sobre las componentes σ_g² y σ_e²
del modelo mixto univariante. El algoritmo EM se descartó por converger
demasiado lento.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch

from Genoly.quantitative.utils import cholesky_regularized, prepare_model_inputs

_LOG_2PI = math.log(2.0 * math.pi)


@dataclass
class VarianceComponents:
    """Componentes de varianza estimadas para un modelo lineal mixto."""
    genetic_variance: float
    residual_variance: float
    log_likelihood: float
    iterations: int
    converged: bool
    fixed_effects: torch.Tensor
    random_effects: torch.Tensor


def solve_variance_components(y_t: torch.Tensor,
                              X_t: torch.Tensor,
                              K_t: torch.Tensor,
                              Z_t: torch.Tensor,
                              method: str = "reml",
                              max_iter: int = 100,
                              tol: float = 1e-8) -> VarianceComponents:
    """
    Estima las componentes de varianza sobre tensores ya preparados.

    Args:
        y_t: Fenotipos (n,) en el dispositivo de cálculo.
        X_t: Diseño de efectos fijos (n, p).
        K_t: Matriz de parentesco (q, q), simétrica.
        Z_t: Diseño de efectos aleatorios (n, q).
        method: 'reml' o 'ml'.
        max_iter: Máximo de iteraciones de puntuación de Fisher.
        tol: Tolerancia de convergencia sobre el cambio relativo de las
             componentes de varianza.

    Returns:
        VarianceComponents con las varianzas estimadas y los efectos
        ajustados (BLUE/BLUP) en CPU.

    Raises:
        ValueError: Si el método no es 'reml' ni 'ml'.
        RuntimeError: Si alguna factorización de Cholesky falla.
    """
    if method not in ("reml", "ml"):
        raise ValueError("method debe ser 'reml' o 'ml'")
    reml = method == "reml"

    n = y_t.shape[0]
    p = X_t.shape[1]

    eye_n = torch.eye(n, dtype=y_t.dtype, device=y_t.device)
    W = Z_t @ K_t @ Z_t.T

    var_total = max(float(y_t.var(unbiased=False).item()), 1e-6)
    var_g = var_total / 2.0
    var_e = var_total / 2.0
    floor = 1e-10

    def solve_state(g: float, e: float) -> Dict[str, object]:
        V = g * W + e * eye_n
        L = cholesky_regularized(V)
        Vinv = torch.cholesky_solve(eye_n, L)
        Vinv_X = Vinv @ X_t
        XtViX = X_t.T @ Vinv_X
        L_fix = cholesky_regularized(XtViX)
        beta = torch.cholesky_solve(X_t.T @ (Vinv @ y_t.unsqueeze(1)),
                                    L_fix).reshape(-1)
        resid = y_t.unsqueeze(1) - X_t @ beta.unsqueeze(1)
        t = Vinv @ resid
        quad = float((resid * t).sum().item())
        logdet_v = 2.0 * torch.log(torch.diagonal(L)).sum().item()
        logdet_fix = 2.0 * torch.log(torch.diagonal(L_fix)).sum().item()
        df = n - p if reml else n
        extra = logdet_fix if reml else 0.0
        loglik = -0.5 * (df * _LOG_2PI + logdet_v + extra + quad)
        u = g * (K_t @ (Z_t.T @ t)).reshape(-1)
        proj = Vinv - Vinv_X @ torch.cholesky_solve(Vinv_X.T, L_fix) if reml else Vinv
        return {"beta": beta, "u": u, "loglik": loglik, "t": t, "proj": proj}

    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        state = solve_state(var_g, var_e)
        proj = state["proj"]
        t = state["t"]

        proj_w = proj @ W
        tr_pw = float((proj * W).sum().item())
        tr_p = float(proj.diagonal().sum().item())
        score_g = -0.5 * (tr_pw - float((t * (W @ t)).sum().item()))
        score_e = -0.5 * (tr_p - float((t * t).sum().item()))

        info_gg = 0.5 * float((proj_w * proj_w).sum().item())
        info_ge = 0.5 * float((proj_w * proj).sum().item())
        info_ee = 0.5 * float((proj * proj).sum().item())

        det = info_gg * info_ee - info_ge * info_ge
        if abs(det) < 1e-12 * max(info_gg * info_ee, 1.0):
            ridge = 1e-8 * max(info_gg + info_ee, 1.0)
            info_gg += ridge
            info_ee += ridge
            det = info_gg * info_ee - info_ge * info_ge

        delta_g = (info_ee * score_g - info_ge * score_e) / det
        delta_e = (info_gg * score_e - info_ge * score_g) / det

        base_loglik = float(state["loglik"])
        step = 1.0
        improved = False
        new_g, new_e = var_g, var_e
        while step >= 1e-6:
            new_g = var_g + step * delta_g
            new_e = var_e + step * delta_e
            if new_g > floor and new_e > floor:
                trial = solve_state(new_g, new_e)
                if float(trial["loglik"]) >= base_loglik - 1e-12:
                    improved = True
                    break
            step *= 0.5

        if not improved:
            converged = True
            break

        rel_change = max(abs(new_g - var_g) / (abs(var_g) + 1e-12),
                         abs(new_e - var_e) / (abs(var_e) + 1e-12))
        var_g, var_e = new_g, new_e
        if rel_change < tol:
            converged = True
            break

    final = solve_state(var_g, var_e)
    return VarianceComponents(
        genetic_variance=float(var_g),
        residual_variance=float(var_e),
        log_likelihood=float(final["loglik"]),
        iterations=iterations,
        converged=converged,
        fixed_effects=final["beta"].detach().cpu(),
        random_effects=final["u"].detach().cpu(),
    )


def estimate_variance_components(y, X, K, Z=None,
                                 method: str = "reml",
                                 max_iter: int = 100,
                                 tol: float = 1e-8,
                                 device: Optional[str] = None,
                                 dtype: torch.dtype = torch.float64,
                                 ) -> VarianceComponents:
    """
    Estima las componentes de varianza desde entradas de usuario.

    Convierte y valida las entradas (ver prepare_model_inputs) antes de
    delegar en solve_variance_components.

    Args:
        y: Fenotipos (n,).
        X: Diseño de efectos fijos (n, p), incluido el intercepto si procede.
        K: Matriz de parentesco (q, q); usar build_kinship para obtenerla.
        Z: Diseño de efectos aleatorios (n, q); None usa la identidad.
        method: 'reml' (por defecto) o 'ml'.
        max_iter: Máximo de iteraciones de puntuación de Fisher.
        tol: Tolerancia de convergencia relativa.
        device: 'cuda', 'cpu' o None para auto-detectar.
        dtype: Tipo de dato para los cálculos internos.

    Returns:
        VarianceComponents con las varianzas estimadas y los efectos
        ajustados en CPU.
    """
    y_t, X_t, K_t, Z_t = prepare_model_inputs(y, X, K, Z, device=device, dtype=dtype)
    return solve_variance_components(y_t, X_t, K_t, Z_t,
                                     method=method, max_iter=max_iter, tol=tol)
