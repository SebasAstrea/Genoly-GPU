from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from Genoly.quantitative.lmm import LinearMixedModel, build_kinship

router = APIRouter(prefix="/api/quantitative", tags=["quantitative"])


class QuantitativeRequest(BaseModel):
    phenotypes: List[float]
    genotypes: List[List[Optional[float]]]
    method: str = "reml"
    max_iter: int = 100
    kinship_method: str = "vanraden"


class QuantitativeResponse(BaseModel):
    n_individuals: int
    n_markers: int
    genetic_variance: float
    residual_variance: float
    heritability: float
    log_likelihood: float
    iterations: int
    converged: bool
    fixed_effects: List[float]
    blup: List[dict]


@router.post("/fit", response_model=QuantitativeResponse)
def fit_lmm(req: QuantitativeRequest) -> QuantitativeResponse:
    """Ajusta un modelo lineal mixto (REML/ML) y predice valores de cría (BLUP)."""
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

    genotypes = [
        [float("nan") if v is None else float(v) for v in row]
        for row in req.genotypes
    ]
    n_markers = len(genotypes[0])

    kinship = build_kinship(genotypes, method=req.kinship_method)
    model = LinearMixedModel()
    result = model.fit(
        req.phenotypes,
        [[1.0] for _ in req.phenotypes],
        kinship,
        method=req.method,
        max_iter=req.max_iter,
    )

    return QuantitativeResponse(
        n_individuals=len(req.phenotypes),
        n_markers=n_markers,
        genetic_variance=round(result.genetic_variance, 6),
        residual_variance=round(result.residual_variance, 6),
        heritability=round(result.heritability, 6),
        log_likelihood=round(result.log_likelihood, 4),
        iterations=result.iterations,
        converged=result.converged,
        fixed_effects=[round(v, 6) for v in model.blue().tolist()],
        blup=[
            {"individual": i + 1, "value": round(v, 6)}
            for i, v in enumerate(model.blup().tolist())
        ],
    )
