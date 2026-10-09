from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from Genoly.quantitative.pblup import PedigreeBLUP
from Genoly.quantitative.pedigree import PedigreeError, build_a_matrix
from Genoly.quantitative.reml import estimate_variance_components

router = APIRouter(prefix="/api/pblup", tags=["pblup"])


class PBLUPRequest(BaseModel):
    phenotypes: List[float]
    individuals: List[str]
    sires: List[Optional[str]]
    dams: List[Optional[str]]
    genetic_variance: Optional[float] = None
    residual_variance: Optional[float] = None
    max_iter: int = 100


class PBLUPResponse(BaseModel):
    n_individuals: int
    genetic_variance: float
    residual_variance: float
    heritability: float
    variance_source: str
    log_likelihood: Optional[float]
    iterations: Optional[int]
    converged: Optional[bool]
    fixed_effects: List[float]
    founders: List[int]
    unknown_parents: int
    blup: List[dict]


@router.post("/predict", response_model=PBLUPResponse)
def predict_pblup(req: PBLUPRequest) -> PBLUPResponse:
    """Predicción por modelo animal (PBLUP) con matriz A de pedigrí."""
    n = len(req.phenotypes)
    if n != len(req.individuals) or n != len(req.sires) or n != len(req.dams):
        raise HTTPException(
            status_code=400,
            detail="phenotypes, individuals, sires y dams deben tener la "
                   "misma longitud",
        )
    if n < 5:
        raise HTTPException(
            status_code=400,
            detail="Se necesitan al menos 5 individuos para ajustar el modelo",
        )
    given = (req.genetic_variance is not None, req.residual_variance is not None)
    if any(given) and not all(given):
        raise HTTPException(
            status_code=400,
            detail="Proporciona ambas varianzas (genética y residual) o ninguna "
                   "para estimarlas por REML",
        )

    try:
        A, _F, info = build_a_matrix(req.individuals, req.sires, req.dams)
    except PedigreeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # reordenar fenotipos al orden topológico de A
    order = info["individuals"]
    pos = {v: i for i, v in enumerate(req.individuals)}
    idx = [pos[v] for v in order]
    y_ordered = [req.phenotypes[i] for i in idx]
    fixed_design = [[1.0] for _ in range(n)]

    if req.genetic_variance is not None:
        log_likelihood = None
        iterations = None
        converged = None
    else:
        est = estimate_variance_components(
            y_ordered, fixed_design, A, max_iter=req.max_iter,
        )
        log_likelihood = round(est.log_likelihood, 4)
        iterations = est.iterations
        converged = est.converged

    model = PedigreeBLUP()
    result = model.fit(
        y_ordered,
        fixed_design,
        A,
        genetic_variance=req.genetic_variance,
        residual_variance=req.residual_variance,
        max_iter=req.max_iter,
        inbreeding=_F,
        founders=info["founders"],
        unknown_parents=info["unknown_parents"],
        individuals=order,
    )

    values = model.blup().tolist()
    reliabilities = model.reliabilities().tolist()
    accuracies = model.accuracies().tolist()
    f_vals = model.inbreeding().tolist()

    h2 = (result.genetic_variance
          / (result.genetic_variance + result.residual_variance))

    return PBLUPResponse(
        n_individuals=n,
        genetic_variance=round(result.genetic_variance, 6),
        residual_variance=round(result.residual_variance, 6),
        heritability=round(h2, 6),
        variance_source=result.variance_source,
        log_likelihood=log_likelihood,
        iterations=iterations,
        converged=converged,
        fixed_effects=[round(v, 6) for v in model.blue().tolist()],
        founders=info["founders"],
        unknown_parents=info["unknown_parents"],
        blup=[
            {
                "index": i,
                "individual": order[i],
                "value": round(v, 6),
                "reliability": round(r, 6),
                "accuracy": round(a, 6),
                "inbreeding": round(f_vals[i], 6) if f_vals else 0.0,
            }
            for i, (v, r, a) in enumerate(zip(values, reliabilities, accuracies))
        ],
    )
