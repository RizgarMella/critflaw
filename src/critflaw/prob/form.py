"""Generic FORM and SORM for a smooth limit-state function in standard normal space.

These routines are for user-supplied smooth limit states (theory manual, section 4.4.3).
They are not applied to failure-assessment-diagram problems, whose failure set is a
union of guarded, non-smooth events for which no applicability rule proved reliable.

Convention (theory manual, section 4.4.3): alpha = grad g / |grad g| at the design point and u* = -beta alpha.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import special

FloatArray = NDArray[np.float64]
LimitState = Callable[[FloatArray], FloatArray]  # (m, n) -> (m,)


class FormNotApplicable(ValueError):
    pass


@dataclass(frozen=True)
class FormResult:
    beta: float
    design_point: FloatArray  # u*
    alpha: FloatArray
    pf: float  # Phi(-beta): a first-order approximation, not an estimate with an error bound
    iterations: int
    starts_converged: int
    competing_design_points: int = 0  # other distinct converged points with beta within 0.5 of the smallest


def _gradient(g: LimitState, u: FloatArray, step: float) -> FloatArray:
    n = u.size
    points = np.repeat(u[None, :], 2 * n, axis=0)
    points[np.arange(n), np.arange(n)] += step
    points[n + np.arange(n), np.arange(n)] -= step
    values = np.asarray(g(points), dtype=float)
    return np.asarray((values[:n] - values[n:]) / (2.0 * step), dtype=float)


def _hlrf(g: LimitState, start: FloatArray, scale: float, step: float, max_iter: int) -> tuple[FloatArray, int, bool]:
    """Improved HL-RF: Newton direction with an Armijo line search on a merit function."""
    u = start.copy()
    for iteration in range(1, max_iter + 1):
        value = float(g(u[None, :])[0])
        grad = _gradient(g, u, step)
        norm2 = float(grad @ grad)
        if norm2 == 0.0 or not np.isfinite(norm2) or not np.isfinite(value):
            return u, iteration, False
        target = (float(grad @ u) - value) / norm2 * grad
        direction = target - u
        penalty = max(2.0 * float(np.linalg.norm(u)) / np.sqrt(norm2), 10.0)

        def merit(point: FloatArray, c: float = penalty) -> float:
            return 0.5 * float(point @ point) + c * abs(float(g(point[None, :])[0]))

        base, t = merit(u), 1.0
        while t > 1e-6 and merit(u + t * direction) > base - 1e-4 * t * float(direction @ direction):
            t *= 0.5
        new = u + t * direction
        moved = float(np.linalg.norm(new - u))
        u = new
        if abs(float(g(u[None, :])[0])) < 1e-8 * scale and moved < 1e-6:
            return u, iteration, True
    return u, max_iter, False


def form(
    g: LimitState, dimension: int, starts: int = 5, seed: int = 0, step: float = 1e-4, max_iter: int = 200
) -> FormResult:
    """First-order reliability method from several starting points.

    Raises :class:`FormNotApplicable` if the origin is in the failure region (g(0) <= 0),
    or if no start converges. The design point returned is the converged point nearest
    the origin.
    """
    origin = np.zeros(dimension)
    g0 = float(g(origin[None, :])[0])
    if not np.isfinite(g0):
        raise FormNotApplicable("the limit state is not finite at the origin")
    if g0 <= 0.0:
        raise FormNotApplicable("the origin is in the failure region (g(0) <= 0); FORM is not used")
    rng = np.random.default_rng(seed)
    candidates = [origin] + [rng.standard_normal(dimension) for _ in range(max(starts - 1, 0))]
    best: FloatArray | None = None
    total_iterations = converged = 0
    found: list[FloatArray] = []
    for start in candidates:
        point, iterations, ok = _hlrf(g, start, g0, step, max_iter)
        total_iterations += iterations
        if ok:
            converged += 1
            found.append(point)
            if best is None or np.linalg.norm(point) < np.linalg.norm(best):
                best = point
    if best is None:
        raise FormNotApplicable("FORM did not converge from any starting point")
    nearest = float(np.linalg.norm(best))
    distinct: list[FloatArray] = []
    for point in found:
        far_from_best = np.linalg.norm(point - best) > 0.1
        new = all(np.linalg.norm(point - other) > 0.1 for other in distinct)
        if far_from_best and new and np.linalg.norm(point) - nearest < 0.5:
            distinct.append(point)
    grad = _gradient(g, best, step)
    alpha = grad / np.linalg.norm(grad)
    beta = float(np.linalg.norm(best))
    return FormResult(beta, best, alpha, float(special.ndtr(-beta)), total_iterations, converged, len(distinct))


def sorm_breitung(g: LimitState, result: FormResult, step: float = 1e-3) -> float:
    """Breitung's second-order probability Phi(-beta) prod (1 + beta kappa_i)^-1/2.

    Raises :class:`FormNotApplicable` when any 1 + beta kappa_i < 0.5: the asymptotic formula
    is verified only for moderate curvature and becomes badly wrong as that term approaches zero.
    """
    n = result.design_point.size
    if n < 2:
        return result.pf
    # Orthonormal basis whose last vector is -alpha (pointing from the origin to u*).
    basis = np.linalg.qr(np.column_stack([-result.alpha, np.eye(n)[:, : n - 1]]))[0]
    tangent = basis[:, 1:]
    grad_norm = float(np.linalg.norm(_gradient(g, result.design_point, 1e-4)))
    u = result.design_point
    g_at = float(g(u[None, :])[0])
    hessian = np.empty((n - 1, n - 1))
    for i in range(n - 1):
        for j in range(i, n - 1):
            ei, ej = tangent[:, i] * step, tangent[:, j] * step
            pts = np.vstack([u + ei + ej, u + ei - ej, u - ei + ej, u - ei - ej])
            v = np.asarray(g(pts), dtype=float)
            hessian[i, j] = hessian[j, i] = (v[0] - v[1] - v[2] + v[3]) / (4.0 * step**2)
    del g_at
    # curvatures of the limit-state surface seen from the origin (positive: curving away from it)
    kappa = np.linalg.eigvalsh(hessian / grad_norm)
    terms = 1.0 + result.beta * kappa
    if np.any(terms < 0.5):
        raise FormNotApplicable("SORM (Breitung) is not applicable: 1 + beta * curvature is below 0.5")
    return float(special.ndtr(-result.beta) / np.sqrt(np.prod(terms)))
