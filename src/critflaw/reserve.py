"""Reserve factors and limiting flaw size by scanning for all crossings (theory manual, section 1.8).

Acceptability is not assumed to be monotone in the scanned parameter: residual
stress relaxation, the plasticity interaction term and through-wall stress
variation can all produce more than one crossing. Each function scans a stated
range on a stated grid, polishes every change of acceptability by bisection,
reports the smallest value at which the assessment is not acceptable, and says
what it could not resolve. These functions take a scalar assessment.

A parameter value at which a solution is outside its validity range counts as
"not acceptable" for the scan, but is reported separately (``limited_by_validity``)
because it is not a predicted failure.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from critflaw.assessment import Assessment
from critflaw.model import SurfaceFlaw, ThroughFlaw

# status codes returned by the probe
_OK, _FAIL, _INVALID = 0, 1, 2


@dataclass(frozen=True)
class Crossing:
    """Result of a scan.

    value:  smallest parameter value found at which the assessment is not acceptable
            (the not-acceptable side of the crossing); +inf if none was found in range.
    bounded: False if no such value was found inside the scanned range.
    unique: False if acceptability changes more than once over the scanned range.
    crossings: every change of acceptability found, in increasing order.
    below_range: True if the assessment is not acceptable even at the smallest value
            probed (``range[0]`` extended downward); ``value`` is then only an upper
            bound on the true smallest value, which may be zero.
    limited_by_validity: True if ``value`` is where a solution leaves its validity
            range rather than where the assessment point reaches the assessment line.
    range, points, log_spaced: the scan actually performed. Regions of non-acceptability
            narrower than the grid spacing can be missed.
    unacceptable_at_zero: (load reserve only) True if the assessment is not acceptable with
            exactly no primary load. ``value`` is then 0. The source procedure treats "no
            primary load" differently from a vanishing one, so the assessment can be
            acceptable at small positive loads; those crossings are still listed.
    """

    value: float
    bounded: bool
    unique: bool
    crossings: tuple[float, ...]
    below_range: bool
    limited_by_validity: bool
    range: tuple[float, float]
    points: int
    log_spaced: bool
    unacceptable_at_zero: bool = False


def _scan(
    status: Callable[[float], int],
    lo: float,
    hi: float,
    points: int,
    log: bool,
    extend_down_decades: int = 12,
) -> Crossing:
    def ok(x: float) -> bool:
        return status(x) == _OK

    def polish(left: float, right: float, left_ok: bool) -> float:
        """Bisect a change of acceptability; return its not-acceptable side."""
        for _ in range(200):
            mid = 0.5 * (left + right)
            if mid in (left, right):
                break
            if ok(mid) == left_ok:
                left = mid
            else:
                right = mid
        return right if left_ok else left

    grid = np.geomspace(lo, hi, points) if log else np.linspace(lo, hi, points)
    states = [ok(float(v)) for v in grid]
    crossings = [
        polish(float(grid[i - 1]), float(grid[i]), states[i - 1])
        for i in range(1, points)
        if states[i] != states[i - 1]
    ]

    below_range = False
    if states[0]:
        first = crossings[0] if crossings else float("inf")
    else:
        # Not acceptable at the low end: look below the range for where it starts.
        probe, found = float(grid[0]), False
        for _ in range(extend_down_decades):
            lower = probe / 10.0
            if ok(lower):
                first = polish(lower, probe, True)
                crossings.insert(0, first)
                found = True
                break
            probe = lower
        if not found:
            first, below_range = probe, True
    limited = bool(np.isfinite(first)) and status(first) == _INVALID
    return Crossing(
        value=first,
        bounded=bool(np.isfinite(first)),
        unique=len(crossings) <= 1,
        crossings=tuple(crossings),
        below_range=below_range,
        limited_by_validity=limited,
        range=(float(grid[0]), float(grid[-1])),
        points=points,
        log_spaced=log,
    )


def _require_scalar(problem: Assessment) -> None:
    if problem.run(on_invalid="mask").acceptable.size != 1:
        raise ValueError("reserve factors and limiting flaw size take a scalar assessment (one sample)")


def _status(problem: Assessment) -> int:
    result = problem.run(on_invalid="mask")
    if result.acceptable.size != 1:
        raise ValueError("reserve factors and limiting flaw size take a scalar assessment (one sample)")
    if not bool(result.valid.item()):
        return _INVALID
    return _OK if bool(result.acceptable.item()) else _FAIL


def reserve_factor_on_load(
    problem: Assessment, min_factor: float = 1e-3, max_factor: float = 100.0, points: int = 400
) -> Crossing:
    """Smallest factor on primary stresses (secondary stresses held) that is not acceptable.

    Exactly zero primary load is assessed first: if that is not acceptable the result has
    ``value = 0`` and ``unacceptable_at_zero=True`` (no reserve on load exists), with any
    crossings at positive load still listed.
    """

    def status(factor: float) -> int:
        loads = problem.loads
        scaled = loads.with_(pm=np.asarray(loads.pm) * factor, pb=np.asarray(loads.pb) * factor)
        return _status(problem.with_(loads=scaled))

    scan = _scan(status, min_factor, max_factor, points, log=True)
    if status(0.0) != _OK:
        return dataclasses.replace(scan, value=0.0, bounded=True, below_range=False, unacceptable_at_zero=True)
    return scan


def reserve_factor_on_toughness(
    problem: Assessment, min_factor: float = 1e-3, max_factor: float = 1000.0, points: int = 400
) -> Crossing:
    """Smallest divisor on fracture toughness that is not acceptable.

    ``below_range=True`` means the assessment is not acceptable however high the
    toughness (for example plastic collapse governs): no reserve on toughness exists.
    """

    def status(factor: float) -> int:
        material = problem.material
        return _status(problem.with_(material=material.with_(toughness=np.asarray(material.toughness) / factor)))

    return _scan(status, min_factor, max_factor, points, log=True)


def limiting_flaw_size(problem: Assessment, points: int = 400) -> Crossing:
    """Smallest flaw size that is not acceptable, scaling the flaw at fixed shape.

    Surface flaw: depth ``a`` scanned up to the wall thickness at fixed a/c.
    Through-thickness flaw: total length ``2a`` scanned up to the plate width.
    Check ``limited_by_validity``: if True, ``value`` is where the stress intensity
    factor solution stops being valid, and no failure is predicted below it.
    """
    _require_scalar(problem)
    flaw = problem.flaw
    if isinstance(flaw, SurfaceFlaw):
        a0 = float(np.asarray(flaw.a).item())
        aspect = float(np.asarray(flaw.two_c).item()) / a0
        upper = float(np.asarray(problem.component.thickness).item())

        def status(size: float) -> int:
            return _status(problem.with_(flaw=SurfaceFlaw(a=size, two_c=aspect * size)))

    elif isinstance(flaw, ThroughFlaw):
        upper = float(np.asarray(problem.component.width).item())

        def status(size: float) -> int:
            return _status(problem.with_(flaw=ThroughFlaw(two_a=size)))

    else:  # pragma: no cover
        raise TypeError(f"unsupported flaw type {type(flaw).__name__}")
    return _scan(status, upper * 1e-4, upper, points, log=False)
