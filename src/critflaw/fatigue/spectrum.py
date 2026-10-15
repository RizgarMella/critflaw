"""Cyclic loading and the extreme (fracture-check) load (theory manual, sections 5.2 and 5.4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from critflaw.model import _Replaceable
from critflaw.prob.distributions import Distribution

FloatArray = NDArray[np.float64]
Value = Any

Order = Literal["as_given", "largest_first", "smallest_first"]


@dataclass(frozen=True)
class Spectrum(_Replaceable):
    """Blocks of stress range ``(membrane, bending, cycles)``.

    ``kind="repeating"``: the blocks are one pass of a pattern that repeats; the rate is the
    expectation over the pass (theory manual, section 5.2). ``kind="ordered"``: the blocks are applied once, in the
    declared order, each at its own rate (a lifetime histogram); their cycles must sum to the life.
    Membrane and bending ranges of a block are in phase. ``load_scale`` multiplies every range
    (may be a distribution). ``ktm``, ``ktb``, ``km`` are the stress concentration and
    misalignment factors applied to the ranges.
    """

    membrane: FloatArray
    bending: FloatArray
    cycles: FloatArray
    cycles_per_year: float
    kind: Literal["repeating", "ordered"] = "repeating"
    order: Order | None = None
    load_scale: Value = 1.0
    ktm: float = 1.0
    ktb: float = 1.0
    km: float = 1.0
    life_factor: float = 1.0  # deterministic runs only: the source's halving of a variable-amplitude life
    note: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        m, b, n = (np.asarray(v, dtype=float).ravel() for v in (self.membrane, self.bending, self.cycles))
        if not (m.size == b.size == n.size >= 1):
            raise ValueError("blocks need the same number of membrane ranges, bending ranges and cycle counts")
        if np.any(m < 0.0) or np.any(b < 0.0) or np.any(n <= 0.0) or not np.all(np.isfinite(m + b + n)):
            raise ValueError("ranges must be non-negative and cycle counts positive and finite")
        if not self.cycles_per_year > 0.0:
            raise ValueError("cycles_per_year must be positive")
        if self.kind == "ordered" and self.order is None:
            raise ValueError(
                "an ordered spectrum needs order='as_given' | 'largest_first' | 'smallest_first' (no default)"
            )
        if self.kind == "repeating" and self.order is not None:
            raise ValueError("order applies to an ordered spectrum only")
        if not self.life_factor >= 1.0:
            raise ValueError("life_factor must be at least 1 (2 halves the calculated life)")
        if not isinstance(self.load_scale, Distribution) and not np.all(np.asarray(self.load_scale, dtype=float) > 0.0):
            raise ValueError("load_scale must be positive")
        object.__setattr__(self, "membrane", m)
        object.__setattr__(self, "bending", b)
        object.__setattr__(self, "cycles", n)

    @classmethod
    def repeating(cls, blocks: ArrayLike, cycles_per_year: float, **factors: Any) -> Spectrum:
        arr = np.atleast_2d(np.asarray(blocks, dtype=float))
        return cls(arr[:, 0], arr[:, 1], arr[:, 2], cycles_per_year, "repeating", None, **factors)

    @classmethod
    def ordered(cls, blocks: ArrayLike, cycles_per_year: float, order: Order, **factors: Any) -> Spectrum:
        arr = np.atleast_2d(np.asarray(blocks, dtype=float))
        if order == "largest_first":
            arr = arr[np.argsort(-(arr[:, 0] + arr[:, 1]), kind="stable")]
        elif order == "smallest_first":
            arr = arr[np.argsort(arr[:, 0] + arr[:, 1], kind="stable")]
        elif order != "as_given":
            raise ValueError("order must be 'as_given', 'largest_first' or 'smallest_first'")
        return cls(arr[:, 0], arr[:, 1], arr[:, 2], cycles_per_year, "ordered", order, **factors)

    @property
    def pass_cycles(self) -> float:
        return float(np.sum(self.cycles))

    @property
    def weights(self) -> FloatArray:
        """Fraction of the pass in each block (repeating spectra)."""
        return np.asarray(self.cycles / self.pass_cycles, dtype=float)

    def block_end_times(self) -> FloatArray:
        """Ordered spectra: the time (years) at which each block ends."""
        return np.asarray(np.cumsum(self.cycles) / self.cycles_per_year, dtype=float)

    def total_years(self) -> float:
        return float(np.sum(self.cycles) / self.cycles_per_year)


@dataclass(frozen=True)
class ExtremeLoad(_Replaceable):
    """Distribution of the maximum of the fracture-check load over ``reference_period_years``
    (for example the annual maximum of the primary membrane stress), applied to the stress
    component ``component`` of the problem. Step maxima over a process step of length ``delta``
    have distribution function F^(delta / reference period) (independent sub-period maxima)."""

    distribution: Distribution
    reference_period_years: float = 1.0
    component: Literal["pm", "pb"] = "pm"

    def __post_init__(self) -> None:
        if not self.reference_period_years > 0.0:
            raise ValueError("reference_period_years must be positive")
        if self.distribution.support()[0] < 0.0:
            raise ValueError("the extreme load must be non-negative; truncate the distribution explicitly")

    def step_maximum(self, u: ArrayLike, delta_years: float) -> FloatArray:
        """Maximum over a step of length ``delta_years`` from standard normal scores: F^-1(Phi(u)^(T/delta)),
        evaluated through the upper tail so that large ``u`` keeps its precision."""
        from scipy import special

        u_ = np.asarray(u, dtype=float)
        ratio = self.reference_period_years / delta_years
        # P(step max <= x) = F(x)^(delta/T)  =>  x = F^-1(p^ratio), p = Phi(u); 1 - p^ratio from log1p for accuracy
        log_p = ratio * special.log_ndtr(u_)  # log of Phi(u)^ratio
        lower = self.distribution.ppf(np.exp(np.minimum(log_p, 0.0)))  # small probabilities: from below
        upper = self.distribution.isf(np.clip(-np.expm1(log_p), 0.0, 1.0))  # near one: from above
        return np.asarray(np.where(u_ <= 0.0, lower, upper), dtype=float)
