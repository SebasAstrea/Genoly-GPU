from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from Genoly.quantitative.gblup import GenomicBLUP
from Genoly.quantitative.lmm import build_kinship
from Genoly.quantitative.reml import estimate_variance_components

router = APIRouter(prefix="/api/gblup", tags=["gblup"])


class GBLUPRequest(BaseModel):
    phenotypes: List[float]
    genotypes: List[List[Optional[float]]]
    kinship_method: str = "vanraden"
    genetic_variance: Optional[float] = None
    residual_variance: Optional[float] = None
    max_iter: int = 100


class GBLUPResponse(BaseModel):
    n_individuals: int
    n_markers: int
    genetic_variance: float
    residual_variance: float
    variance_source: str
    log_likelihood: Optional[float]
    iterations: Optional[int]
    converged: Optional[bool]
    fixed_effects: List[float]
    blup: List[dict]


@router.post("/predict", response_model=GBLUPResponse)
def predict_gblup(req: GBLUPRequest) -> GBLUPResponse:
    """Predicción genómica GBLUP con fiabilidad y precisión por individuo."""
    if len(req.phenotypes) != len(req.genotypes):
        raise HTTPException(
            status_code=400,
            detail=f"El número de fenotipos ({len(req.phenotypes)}) no coincide con "
                   f"el número de individuos ({len(req.genotypes)})",
        )
    if len(req.phenotypes) < 5:
        raise HTTPException(
            status_code=400,
            detail="Se necesitan al menos 5 individuos para ajustar el modelo",
        )
    if req.kinship_method not in ("vanraden", "gcta"):
        raise HTTPException(
            status_code=400,
            detail="kinship_method debe ser 'vanraden' o 'gcta'",
        )
    given = (req.genetic_variance is not None, req.residual_variance is not None)
    if any(given) and not all(given):
        raise HTTPException(
            status_code=400,
            detail="Proporciona ambas varianzas (genética y residual) o ninguna "
                   "para estimarlas por REML",
        )

    genotypes = [
        [float("nan") if v is None else float(v) for v in row]
        for row in req.genotypes
    ]
    n_markers = len(genotypes[0])
    fixed_design = [[1.0] for _ in req.phenotypes]
    kinship = build_kinship(genotypes, method=req.kinship_method)

    if req.genetic_variance is not None:
        log_likelihood = None
        iterations = None
        converged = None
    else:
        est = estimate_variance_components(
            req.phenotypes, fixed_design, kinship, max_iter=req.max_iter,
        )
        log_likelihood = round(est.log_likelihood, 4)
        iterations = est.iterations
        converged = est.converged

    model = GenomicBLUP()
    result = model.fit(
        req.phenotypes,
        fixed_design,
        kinship,
        genetic_variance=req.genetic_variance,
        residual_variance=req.residual_variance,
        max_iter=req.max_iter,
    )

    values = model.blup().tolist()
    reliabilities = model.reliabilities().tolist()
    accuracies = model.accuracies().tolist()

    return GBLUPResponse(
        n_individuals=len(req.phenotypes),
        n_markers=n_markers,
        genetic_variance=round(result.genetic_variance, 6),
        residual_variance=round(result.residual_variance, 6),
        variance_source=result.variance_source,
        log_likelihood=log_likelihood,
        iterations=iterations,
        converged=converged,
        fixed_effects=[round(v, 6) for v in model.blue().tolist()],
        blup=[
            {
                "individual": i + 1,
                "value": round(v, 6),
                "reliability": round(r, 6),
                "accuracy": round(a, 6),
            }
            for i, (v, r, a) in enumerate(zip(values, reliabilities, accuracies))
        ],
    )
