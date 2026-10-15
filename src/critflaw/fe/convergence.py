"""D7 mesh-convergence formulas (theory manual, section 6.2): under ``J(h) = J* + C h^p`` with four levels of
representative sizes ``h_1 > h_2 > h_3 > h_4`` (``h_2 = h_1/r21``, ``h_3 = h_2/r32``, ``h_4 = h_3/r43``) and
differences ``d12 = J1 - J2`` etc.: the observed order from levels 1-3 (non-constant ratios), the level-4
prediction that makes the asymptotic test non-vacuous, and the coarse-grid (production, level 1) error with the
GCI safety factor. Matched to an independent reference implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

P_MIN, P_MAX = 0.05, 6.0


def difference_ratio_model(p: float, r21: float, r32: float) -> float:
    """``d12 / d23`` under the model: ``r21^p (1 - r21^-p) / (1 - r32^-p)``."""
    return float(r21**p * (1.0 - r21 ** (-p)) / (1.0 - r32 ** (-p)))


def observed_order(d12: float, d23: float, r21: float, r32: float, iterations: int = 200) -> float:
    """``p`` solving ``d12 / d23 = model(p)`` by bisection on ``[P_MIN, P_MAX]``; NaN when the measured ratio
    lies outside the model's range on that interval (no solution: a clause-(b) failure, never a loop) or when
    ``d23`` is zero."""
    if d23 == 0.0 or not np.isfinite(d12 / d23):
        return float("nan")
    target = d12 / d23
    lo, hi = P_MIN, P_MAX
    f_lo, f_hi = difference_ratio_model(lo, r21, r32) - target, difference_ratio_model(hi, r21, r32) - target
    if np.sign(f_lo) == np.sign(f_hi):
        return float("nan")
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        f_mid = difference_ratio_model(mid, r21, r32) - target
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def predicted_ratio(p: float, r32: float, r43: float) -> float:
    """``d34 / d23`` predicted by the model from ``p``: ``r32^-p (1 - r43^-p) / (1 - r32^-p)``."""
    return float(r32 ** (-p) * (1.0 - r43 ** (-p)) / (1.0 - r32 ** (-p)))


def coarse_error(j1: float, j2: float, p: float, r21: float, safety_factor: float = 1.25) -> float:
    """Estimated absolute error of the coarse (production) level: ``Fs |J1 - J2| r21^p / (r21^p - 1)``."""
    return float(safety_factor * abs(j1 - j2) * r21**p / (r21**p - 1.0))


@dataclass(frozen=True)
class ConvergenceVerdict:
    p: float
    p_234: float
    predicted_d34_over_d23: float
    realised_d34_over_d23: float
    coarse_error_abs: float
    coarse_error_rel: float  # relative to J2
    monotone: bool
    order_in_range: bool
    prediction_within: bool
    order_stable: bool

    @property
    def asymptotic(self) -> bool:
        return self.monotone and self.order_in_range and self.prediction_within and self.order_stable


def assess_four_levels(
    j: tuple[float, float, float, float],
    ratios: tuple[float, float, float] = (1.25, 1.2, 4.0 / 3.0),
    p_range: tuple[float, float] = (0.8, 3.0),
    prediction_tolerance: float = 0.2,
    order_stability: float = 0.5,
) -> ConvergenceVerdict:
    """The D7 asymptotic test (a)-(d) and the production-error estimate from four levels. ``ratios`` are the
    refinement ratios r21, r32, r43; pass those of the levels actually run (``fecase.LEVELS``: 4/3, 5/4, 6/5)."""
    j1, j2, j3, j4 = j
    r21, r32, r43 = ratios
    d12, d23, d34 = j1 - j2, j2 - j3, j3 - j4
    monotone = bool(np.sign(d12) == np.sign(d23) == np.sign(d34) != 0)
    p = observed_order(d12, d23, r21, r32)
    p_234 = observed_order(d23, d34, r32, r43)
    pred = predicted_ratio(p, r32, r43) if np.isfinite(p) else float("nan")
    real = d34 / d23 if d23 != 0.0 else float("nan")
    order_ok = bool(np.isfinite(p) and p_range[0] <= p <= p_range[1])
    pred_ok = bool(np.isfinite(pred) and np.isfinite(real) and abs(real - pred) <= prediction_tolerance * abs(pred))
    stable = bool(np.isfinite(p) and np.isfinite(p_234) and abs(p_234 - p) <= order_stability)
    err = coarse_error(j1, j2, p, r21) if np.isfinite(p) else float("nan")
    return ConvergenceVerdict(
        p, p_234, pred, real, err, err / abs(j2) if j2 != 0.0 else float("nan"), monotone, order_ok, pred_ok, stable
    )
