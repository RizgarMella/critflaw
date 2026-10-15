"""Acceptance test and radial margin (theory manual, sections 1.6 and 1.7)."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int8]

# Where the ray from the origin leaves the acceptable region.
EXIT_CURVE = 0  # through the assessment line
EXIT_CUTOFF = 1  # through the vertical line Lr = Lr,max
EXIT_NONE = 2  # origin: the ray never leaves (F = +inf)
EXIT_STEP = 3  # through a vertical step of the assessment line
EXIT_NAMES = {EXIT_CURVE: "curve", EXIT_CUTOFF: "cutoff", EXIT_NONE: "none", EXIT_STEP: "step"}


class FADCurve(Protocol):
    """A non-increasing assessment line with f(0) = 1 and f <= 1."""

    @property
    def lr_max(self) -> FloatArray: ...

    def __call__(self, lr: ArrayLike) -> FloatArray: ...


def acceptable(lr: ArrayLike, kr: ArrayLike, curve: FADCurve) -> BoolArray:
    """Direct acceptance test: Kr < f(Lr) and Lr < Lr,max (a point on the line is unacceptable)."""
    x = np.asarray(lr, dtype=float)
    y = np.asarray(kr, dtype=float)
    lr_max = curve.lr_max
    inside_cutoff = x < lr_max
    # f is only evaluated inside the cut-off; outside it the point is unacceptable anyway.
    x_eval = np.where(inside_cutoff, x, 0.0)
    return np.asarray(inside_cutoff & (y < curve(x_eval)), dtype=bool)


def radial_margin(
    lr: ArrayLike, kr: ArrayLike, curve: FADCurve, iterations: int = 80
) -> tuple[FloatArray, IntArray]:
    """Radial margin F: first exit from the acceptable region along the ray from the origin.

    Returns ``(F, exit_code)``. F = +inf at the origin (and where 1/Kr overflows on the Kr
    axis). Requires a non-increasing curve with f <= 1, for which the acceptable set along
    the ray is the interval t < F. F is accurate to a few ulp, so a point within rounding of
    the boundary can have F == 1.0 whichever side it is on: acceptance must be taken from
    :func:`acceptable`, never from F (theory manual, section 1.6).
    """
    x, y, lr_max = np.broadcast_arrays(
        np.asarray(lr, dtype=float), np.asarray(kr, dtype=float), curve.lr_max
    )
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))) or np.any(x < 0.0) or np.any(y < 0.0):
        raise ValueError("Lr and Kr must be finite and non-negative")

    with np.errstate(divide="ignore", over="ignore"):
        t_cutoff = np.where(x > 0.0, lr_max / np.where(x > 0.0, x, 1.0), np.inf)
        t_kr_one = np.where(y > 0.0, 1.0 / np.where(y > 0.0, y, 1.0), np.inf)
    # Rigorous upper bound on F because f <= 1 and f = 0 at and beyond the cut-off.
    hi = np.minimum(t_cutoff, t_kr_one)

    origin = (x == 0.0) & (y == 0.0)
    on_kr_axis = (x == 0.0) & (y > 0.0)  # F = f(0)/Kr exactly (f(0) = 1 except for some Option 2 laws)
    on_lr_axis = (y == 0.0) & (x > 0.0)  # F = Lr,max/Lr exactly
    # A bracket that overflows (subnormal point) is returned as +inf without searching.
    search = ~(origin | on_kr_axis | on_lr_axis) & np.isfinite(hi)

    lo = np.zeros_like(hi)
    hi_work = np.where(search, hi, 1.0)
    for _ in range(iterations):
        t = lo + 0.5 * (hi_work - lo)  # overflow-safe midpoint
        # t < hi <= Lr,max/Lr, so t*Lr stays inside the cut-off; guard against rounding.
        trial_lr = np.minimum(t * x, np.nextafter(lr_max, 0.0))
        inside = t * y < curve(np.where(search, trial_lr, 0.0))
        lo = np.where(inside, t, lo)
        hi_work = np.where(inside, hi_work, t)
    with np.errstate(over="ignore"):
        on_axis_value = np.where(on_kr_axis, hi * curve(np.zeros_like(x)), hi)
    f_margin = np.where(search, lo + 0.5 * (hi_work - lo), on_axis_value)

    exit_code = np.full(f_margin.shape, EXIT_CURVE, dtype=np.int8)
    reaches_cutoff = np.isfinite(t_cutoff) & (f_margin >= t_cutoff * (1.0 - 1e-12))
    exit_code[reaches_cutoff] = EXIT_CUTOFF
    for location in getattr(curve, "step_locations", ()):
        # Exit through a step: the exit point is at the step abscissa and lies strictly below
        # the curve value just left of it (so it is on the vertical segment, not on the curve).
        f_finite = np.where(search, f_margin, 0.0)
        at_step = search & ~reaches_cutoff & (np.abs(f_finite * x - location) <= 1e-9 * location)
        left = curve(np.where(at_step, np.nextafter(location, 0.0), 0.0))
        exit_code[at_step & (f_finite * y < left * (1.0 - 1e-9))] = EXIT_STEP
    exit_code[origin] = EXIT_NONE
    return np.asarray(f_margin, dtype=float), exit_code
