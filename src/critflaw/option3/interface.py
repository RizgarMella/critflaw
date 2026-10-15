"""The user-facing side of the experimental Option 3 emulator (theory manual, sections 6.4.1 and 6.4.7):

- ``sigma0_from_proof_stress``: the law's reference stress ``sigma_0`` for a user's 0.2 % proof stress (the FE,
  the emulator and the declared limit load refer ``Lr`` to ``sigma_0``; using ``Rp0.2`` in its place would shift
  the ``Lr`` axis by 4-52 % over the box);
- ``assessment_curve``: the guard — an Option 3 curve that is not assessment-grade is refused for assessments
  unless the caller declares ``experimental=True``, and the returned curve keeps the label."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from critflaw.fe.warp_law import WarpDeformationLaw
from critflaw.option3.emulator import Prediction

FloatArray = NDArray[np.float64]


def sigma0_from_proof_stress(
    rp02: ArrayLike, modulus: ArrayLike, n: ArrayLike, nu: float = 0.3, iterations: int = 80
) -> FloatArray:
    """``sigma_0`` such that ``WarpDeformationLaw(E, nu, sigma_0, n).proof_stress_0p2() == rp02``, vectorised
    (bisection on ``sigma_0`` in (rp02 / 10, rp02); the ratio lies in (1, 10) for the box)."""
    rp, e, nn = np.broadcast_arrays(np.asarray(rp02, float), np.asarray(modulus, float), np.asarray(n, float))
    if np.any(rp <= 0.0) or np.any(e <= 0.0) or np.any(nn <= 1.0):
        raise ValueError("need rp02 > 0, E > 0, n > 1")
    lo, hi = rp / 10.0, rp.copy()
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        implied = np.asarray(WarpDeformationLaw(e, nu, mid, nn).proof_stress_0p2(), dtype=float)
        lo, hi = np.where(implied < rp, mid, lo), np.where(implied < rp, hi, mid)
    return np.asarray(0.5 * (lo + hi), dtype=float)


class NotAssessmentGrade(RuntimeError):
    """The Option 3 curve is experimental (unverified FE data, deepest point only) and was not declared so."""


@dataclass(frozen=True)
class AssessmentCurve:
    lr: FloatArray
    kr_draws: FloatArray
    lr_max: float
    experimental: bool
    labels: tuple[str, ...]


def assessment_curve(prediction: Prediction, lr_max: float, experimental: bool = False) -> AssessmentCurve:
    """The Option 3 curve for an assessment, cut at ``min(Lr_end, Lr,max)``. An experimental prediction requires
    ``experimental=True``; the labels travel with the curve. A prediction outside the box is refused (its status
    is reported by the caller as an undecided class, theory manual section 6.4.6)."""
    if not prediction.assessment_grade and not experimental:
        raise NotAssessmentGrade(
            "the Option 3 emulator is experimental (verification_status 'unverified', deepest point only): pass "
            "experimental=True to use it, and report the result as experimental"
        )
    if prediction.status != "supported":
        raise ValueError(f"no Option 3 curve: the realisation is {prediction.status}")
    keep = prediction.lr <= lr_max + 1e-12
    labels = (
        f"verification_status={prediction.verification_status}",
        "deepest point only (the surface position is unresolved: mesh-sensitive)",
        "experimental" if experimental else "assessment-grade",
    )
    return AssessmentCurve(prediction.lr[keep], prediction.kr_draws[:, keep], lr_max, experimental, labels)


__all__ = ["sigma0_from_proof_stress", "assessment_curve", "AssessmentCurve", "NotAssessmentGrade"]
