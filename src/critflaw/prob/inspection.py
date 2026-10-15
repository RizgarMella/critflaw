"""Flaw-size distributions conditioned on an inspection outcome (theory manual, section 4.10).

* ``undetected(prior, pod)``: density proportional to prior(a) (1 - PoD(a)); the probability of
  non-detection P(ND) is returned with it.
* ``detected(a_measured, sizing_error, prior, pod)``: density proportional to
  prior(a) PoD(a) p_err(a_measured - a).

Both are built with respect to the prior probability measure: for a continuous prior the
weight is integrated in the prior's normal score z from both ends, so that tail probabilities
keep relative accuracy; for a discrete prior the masses are reweighted.

A conditioned distribution that lies about 35.5 or more standard normal scores into a tail of
the prior (for example a measured size 36 prior standard deviations from the prior mean) cannot
be integrated in double arithmetic and is refused with an error; it is never returned cut off.

A probability-of-detection model is a callable ``pod(a)`` with values in [0, 1]. It may also
provide ``non_detection(a)``, the value of 1 - PoD(a) computed directly (used where the PoD is
so close to one that ``1 - pod(a)`` would round to zero), and ``discontinuities``, the flaw
sizes at which it jumps. A callable with a jump that does not declare it loses accuracy near
the jump (up to about 1e-3, with a cumulative distribution that is not monotone at that level).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from critflaw.prob.distributions import Distribution, Empirical

FloatArray = NDArray[np.float64]

_Z_MAX = 37.0
_GL_NODES = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
_GL_WEIGHTS = np.array([5.0, 8.0, 5.0]) / 9.0


def _level(level: float) -> None:
    if not 0.0 <= level <= 1.0:  # also rejects NaN
        raise ValueError("a probability of detection must lie in [0, 1]")


@dataclass(frozen=True)
class LogLogisticPoD:
    """PoD(a) = 1 / (1 + (a50 / a)^beta); 0 for a <= 0."""

    a50: float
    beta: float
    discontinuities: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not (0.0 < self.a50 < np.inf and 0.0 < self.beta < np.inf):
            raise ValueError("a50 and beta must be positive and finite")

    def _log_odds(self, a: ArrayLike) -> FloatArray:
        """beta ln(a / a50), -inf for a <= 0. Working with the logarithm avoids forming the power,
        which overflows long before the probabilities stop being representable."""
        a_ = np.asarray(a, dtype=float)
        positive = a_ > 0.0
        log_ratio = np.log(np.where(positive, a_, 1.0)) - np.log(self.a50)
        return np.asarray(np.where(positive, self.beta * log_ratio, -np.inf))

    def __call__(self, a: ArrayLike) -> FloatArray:
        x = self._log_odds(a)
        with np.errstate(under="ignore"):
            return np.asarray(np.exp(-np.logaddexp(0.0, -x)), dtype=float)

    def non_detection(self, a: ArrayLike) -> FloatArray:
        """1 - PoD(a) = 1 / (1 + (a / a50)^beta), without subtraction from one."""
        x = self._log_odds(a)
        with np.errstate(under="ignore"):
            return np.asarray(np.exp(-np.logaddexp(0.0, x)), dtype=float)


@dataclass(frozen=True)
class StepPoD:
    """PoD(a) = ``level`` for a >= ``threshold``, otherwise 0."""

    threshold: float
    level: float = 1.0

    def __post_init__(self) -> None:
        _level(self.level)
        if not np.isfinite(self.threshold):
            raise ValueError("the threshold must be finite")

    @property
    def discontinuities(self) -> tuple[float, ...]:
        return (self.threshold,)

    def __call__(self, a: ArrayLike) -> FloatArray:
        return np.asarray(np.where(np.asarray(a, dtype=float) >= self.threshold, self.level, 0.0), dtype=float)

    def non_detection(self, a: ArrayLike) -> FloatArray:
        return np.asarray(np.where(np.asarray(a, dtype=float) >= self.threshold, 1.0 - self.level, 1.0), dtype=float)


@dataclass(frozen=True)
class ConstantPoD:
    level: float
    discontinuities: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        _level(self.level)

    def __call__(self, a: ArrayLike) -> FloatArray:
        return np.asarray(np.full(np.shape(a), self.level), dtype=float)

    def non_detection(self, a: ArrayLike) -> FloatArray:
        return np.asarray(np.full(np.shape(a), 1.0 - self.level), dtype=float)


def _non_detection(pod: Callable[[ArrayLike], FloatArray]) -> Callable[[FloatArray], FloatArray]:
    direct = getattr(pod, "non_detection", None)
    if direct is not None:
        return lambda a: np.asarray(direct(a), dtype=float)
    return lambda a: np.asarray(1.0 - pod(a), dtype=float)


def _checked(weight: FloatArray, bound: float) -> FloatArray:
    if not np.all(np.isfinite(weight)) or np.any(weight < 0.0) or np.any(weight > bound * (1.0 + 1e-12)):
        raise ValueError("the reweighting function must be finite, non-negative and (for a PoD) at most 1")
    return weight


class _Reweighted(Distribution):
    """A continuous prior reweighted by ``weight(a)`` with values in [0, bound].

    ``breaks`` are flaw sizes at which the weight jumps; they are made cell boundaries.
    ``refine`` are further flaw sizes made cell boundaries (for a sizing-error likelihood: the
    measured size less the quantiles of the error, so that a likelihood much narrower than the
    prior, or with a narrow core and long tails, is resolved and not stepped over).
    """

    def __init__(
        self,
        prior: Distribution,
        weight: Callable[[FloatArray], FloatArray],
        breaks: tuple[float, ...],
        name: str | None,
        bound: float = np.inf,
        refine: FloatArray | None = None,
    ) -> None:
        if prior.is_discrete:
            raise TypeError("use the discrete construction for a discrete prior")
        self.prior, self.name = prior, name
        self._weight, self._bound = weight, bound
        lo_a, hi_a = prior.support()

        def score(a: float) -> float:
            return float(np.clip(prior.normal_score(np.clip(a, lo_a, hi_a)), -_Z_MAX, _Z_MAX))

        z = np.linspace(-_Z_MAX, _Z_MAX, 8001)
        extra = [score(b) for b in breaks if lo_a < b < hi_a]
        if refine is not None:
            inside = refine[(refine > lo_a) & (refine < hi_a)]
            extra += list(np.clip(prior.normal_score(inside), -_Z_MAX, _Z_MAX))
        z = np.unique(np.concatenate([z, extra])) if extra else z
        # 3-point Gauss-Legendre in each cell of z with the standard normal density as measure:
        # no cancellation, so tail masses keep relative accuracy
        a, raw, measure = self._nodes(z[:-1], z[1:])
        cell = np.sum(raw * measure, axis=-1)
        self._z = z
        self._cum_low = np.concatenate([[0.0], np.cumsum(cell)])
        self._cum_up = np.concatenate([np.cumsum(cell[::-1])[::-1], [0.0]])
        self.mass = float(self._cum_low[-1])
        if not self.mass > 0.0:
            raise ValueError("the conditioned distribution has zero probability")
        outer = (z[1:] <= -_Z_MAX + 1.0) | (z[:-1] >= _Z_MAX - 1.0)  # the last unit of normal score at each end
        if float(np.sum(cell[outer])) > 1e-9 * self.mass:
            raise ValueError(
                "the conditioned distribution lies about 35.5 or more standard normal scores into a tail of the "
                "prior, beyond what can be integrated in double arithmetic; an observation that far outside "
                "the prior means the prior does not describe this flaw"
            )
        # effective support: the cells that carry mass (a weight that is zero beyond a declared
        # jump ends the support there)
        occupied = np.nonzero(cell > 0.0)[0]
        self._z_lo, self._z_hi = float(z[occupied[0]]), float(z[occupied[-1] + 1])
        # moments from the same nodes as the masses (not from differences of the cumulative
        # table, which lose the tails, and not from cell midpoints)
        p = raw * measure / self.mass
        finite = np.isfinite(a) & (p > 0.0)
        a_ = np.where(finite, a, 0.0)
        with np.errstate(over="ignore", invalid="ignore"):
            mean = float(np.sum(np.where(finite, p * a_, 0.0)))
            var = float(np.sum(np.where(finite, p * (a_ - mean) ** 2, 0.0)))
        self._moments = (mean, float(np.sqrt(var)))

    def _nodes(self, z_from: FloatArray, z_to: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Gauss-Legendre nodes in each interval: flaw size, weight there, and quadrature measure
        (normal density evaluated directly, times the node weight and half the interval)."""
        z_from, z_to = np.asarray(z_from, dtype=float), np.asarray(z_to, dtype=float)
        half = 0.5 * (z_to - z_from)
        zg = (0.5 * (z_to + z_from))[..., None] + half[..., None] * _GL_NODES
        a = self.prior.from_normal_score(zg)
        raw = _checked(np.asarray(self._weight(a), dtype=float), self._bound)
        measure = np.exp(-0.5 * zg**2) / np.sqrt(2.0 * np.pi) * _GL_WEIGHTS * half[..., None]
        return a, raw, measure

    def _partial(self, z_from: FloatArray, z_to: FloatArray) -> FloatArray:
        """Integral of weight * normal density over [z_from, z_to] (within one cell)."""
        _, raw, measure = self._nodes(z_from, z_to)
        return np.asarray(np.sum(raw * measure, axis=-1), dtype=float)

    def _score(self, x: ArrayLike) -> FloatArray:
        return np.clip(self.prior.normal_score(np.asarray(x, dtype=float)), -_Z_MAX, _Z_MAX)

    def _cell(self, zx: FloatArray) -> NDArray[np.intp]:
        return np.clip(np.searchsorted(self._z, zx, side="right") - 1, 0, self._z.size - 2)

    def _lower_at(self, zx: FloatArray) -> FloatArray:
        """Unnormalised mass below zx: the table value at the cell start plus the partial cell."""
        i = self._cell(zx)
        return np.asarray(self._cum_low[i] + self._partial(self._z[i], zx), dtype=float)

    def _upper_at(self, zx: FloatArray) -> FloatArray:
        """Unnormalised mass above zx, built from the upper side so that tails keep relative accuracy."""
        i = self._cell(zx)
        return np.asarray(self._cum_up[i + 1] + self._partial(zx, self._z[i + 1]), dtype=float)

    def cdf(self, x: ArrayLike) -> FloatArray:
        return np.asarray(np.clip(self._lower_at(self._score(x)) / self.mass, 0.0, 1.0), dtype=float)

    def sf(self, x: ArrayLike) -> FloatArray:
        return np.asarray(np.clip(self._upper_at(self._score(x)) / self.mass, 0.0, 1.0), dtype=float)

    def _invert(self, target: FloatArray, lower: bool) -> FloatArray:
        """z at which the unnormalised lower (or upper) mass equals ``target``: the cell from the
        table, then a safeguarded Newton iteration inside the cell on the exact partial-cell integral. The result is
        kept inside the effective support, so a run of empty cells is never returned."""
        target = np.asarray(target, dtype=float)
        if target.size == 0:
            return target.copy()
        if lower:
            i = np.clip(np.searchsorted(self._cum_low, target, side="right") - 1, 0, self._z.size - 2)
        else:
            position = self._z.size - 1 - np.searchsorted(self._cum_up[::-1], target, side="left")
            i = np.clip(position, 0, self._z.size - 2)
            i = np.where(self._cum_up[i] < target, np.maximum(i - 1, 0), i)
        lo, hi = self._z[i].copy(), self._z[i + 1].copy()
        z = 0.5 * (lo + hi)
        for _ in range(60):  # Newton steps kept inside a shrinking bracket; a bisection step otherwise
            if lower:
                excess = self._cum_low[i] + self._partial(self._z[i], z) - target
            else:
                excess = target - (self._cum_up[i + 1] + self._partial(z, self._z[i + 1]))
            lo, hi = np.where(excess < 0.0, z, lo), np.where(excess < 0.0, hi, z)
            slope = np.asarray(self._weight(self.prior.from_normal_score(z)), dtype=float)
            slope = slope * np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
            with np.errstate(divide="ignore", invalid="ignore"):
                step = z - excess / slope
            settled = np.abs(step - z) <= 4e-16 * np.maximum(1.0, np.abs(z))  # correction at rounding level
            inside = np.isfinite(step) & (step > lo) & (step < hi)
            new = np.where(settled, z, np.where(inside, step, 0.5 * (lo + hi)))
            moved = float(np.max(np.abs(new - z)))
            z = new
            if moved == 0.0:
                break
        return np.asarray(np.clip(z, self._z_lo, self._z_hi), dtype=float)

    def _quantile(self, q_lower: FloatArray, q_upper: FloatArray) -> FloatArray:
        """Value with probability q_lower below it (q_upper above it; the smaller of the two is used)."""
        q_lower, q_upper = np.broadcast_arrays(np.asarray(q_lower, dtype=float), np.asarray(q_upper, dtype=float))
        z = np.full(q_lower.shape, np.nan)
        valid = (q_lower >= 0.0) & (q_upper >= 0.0)  # False for NaN and for probabilities outside [0, 1]
        from_below = valid & (q_lower <= q_upper)
        from_above = valid & ~from_below
        z[from_below] = self._invert(q_lower[from_below] * self.mass, True)  # each value inverted on one side only
        z[from_above] = self._invert(q_upper[from_above] * self.mass, False)
        z = np.where(valid & (q_lower <= 0.0), self._z_lo, np.where(valid & (q_upper <= 0.0), self._z_hi, z))
        out = np.full(q_lower.shape, np.nan)
        out[valid] = self.prior.from_normal_score(z[valid])
        return out

    def ppf(self, q: ArrayLike) -> FloatArray:
        q_ = np.asarray(q, dtype=float)
        return self._quantile(q_, 1.0 - q_)

    def isf(self, q: ArrayLike) -> FloatArray:
        q_ = np.asarray(q, dtype=float)
        return self._quantile(1.0 - q_, q_)

    def pdf(self, x: ArrayLike) -> FloatArray:
        raise NotImplementedError("the density of a conditioned distribution is not provided")

    def support(self) -> tuple[float, float]:
        """Where the conditioned distribution has mass in double arithmetic: the prior's support,
        cut where the weight is zero (beyond a detection threshold) or underflows to zero (many
        sizing-error standard deviations from the measured size)."""
        p_lo, p_hi = self.prior.support()
        lo = p_lo if self._z_lo <= -_Z_MAX else float(self.prior.from_normal_score(self._z_lo))
        hi = p_hi if self._z_hi >= _Z_MAX else float(self.prior.from_normal_score(self._z_hi))
        return lo, hi

    def mean(self) -> float:
        return self._moments[0]

    def std(self) -> float:
        return self._moments[1]


def _build(
    prior: Distribution,
    weight: Callable[[FloatArray], FloatArray],
    breaks: tuple[float, ...],
    name: str | None,
    bound: float = np.inf,
    refine: FloatArray | None = None,
) -> tuple[Distribution, float]:
    """A reweighted continuous prior, or reweighted masses of a discrete one, with the total mass."""
    if isinstance(prior, Empirical):
        w = _checked(np.asarray(weight(prior.values), dtype=float), bound) * prior.weights
        mass = float(np.sum(w))
        if not mass > 0.0:
            raise ValueError("the conditioned distribution has zero probability")
        keep = w > 0.0
        return Empirical(prior.values[keep], w[keep], name=name), mass
    dist = _Reweighted(prior, weight, breaks, name, bound, refine)
    return dist, dist.mass


def undetected(
    prior: Distribution, pod: Callable[[ArrayLike], FloatArray], name: str | None = None
) -> tuple[Distribution, float]:
    """Flaw size given that inspection did not detect it. Returns ``(distribution, P(ND))``.

    A sizing error does not apply: an undetected flaw has no measured size.
    """
    breaks = tuple(getattr(pod, "discontinuities", ()))
    return _build(prior, _non_detection(pod), breaks, name or "undetected flaw size", bound=1.0)


def detected(
    a_measured: float,
    sizing_error: Distribution,
    prior: Distribution,
    pod: Callable[[ArrayLike], FloatArray],
    name: str | None = None,
) -> tuple[Distribution, float]:
    """Flaw size given that inspection detected it and reported ``a_measured``.

    ``sizing_error`` is the distribution of (measured - true). Returns the posterior and its
    normalising constant, the integral of prior(a) PoD(a) p_err(a_measured - a) over a.
    """
    if sizing_error.is_discrete:
        raise TypeError("the sizing error must be a continuous distribution")
    if not np.isfinite(a_measured):
        raise ValueError("the measured size must be finite")
    breaks = tuple(getattr(pod, "discontinuities", ()))

    def weight(a: FloatArray) -> FloatArray:
        with np.errstate(divide="ignore"):
            density = np.asarray(sizing_error.pdf(a_measured - a), dtype=float)
        # a density that is unbounded at the end of its support (integrable): the single point is dropped
        return np.asarray(pod(a) * np.where(np.isposinf(density), 0.0, density), dtype=float)

    # true = measured - error. Cell boundaries at the measured size less the error's quantiles, evenly
    # spaced in the error's normal score out to 1e-18 on each side: every part of the likelihood (a
    # narrow core, long tails, a density that is unbounded at the end of its support) gets its
    # share of cells whatever its shape and however narrow it is compared with the prior.
    # (coarser beyond that, so that far tails of the posterior keep relative accuracy)
    scores = np.concatenate([np.linspace(-37.0, -8.8, 800), np.linspace(-8.8, 8.8, 4001), np.linspace(8.8, 37.0, 800)])
    refine = a_measured - sizing_error.from_normal_score(scores)
    ends = np.array([a_measured - e for e in sizing_error.support() if np.isfinite(e)])
    refine = np.concatenate([refine[np.isfinite(refine)], ends])
    return _build(prior, weight, breaks, name or "detected flaw size", refine=refine)


def per_component_upper_bound(flaws_per_component: float, p_not_detected: float, pf_per_flaw: float) -> float:
    """1 - exp(-nu P(ND) Pf|flaw), with nu the expected number of flaws per component **before**
    inspection (Poisson occurrence).

    This is an upper bound on the per-component failure probability. It is exact only if the
    flaws share no random variable; when they share a load or material property the exact value
    1 - E[exp(-nu P(ND|theta) Pf|flaw(theta))] is smaller (Jensen's inequality).
    """
    ok = 0.0 <= flaws_per_component < np.inf and 0.0 <= p_not_detected <= 1.0 and 0.0 <= pf_per_flaw <= 1.0
    if not ok:
        raise ValueError("need a finite non-negative rate and probabilities in [0, 1]")
    return float(-np.expm1(-flaws_per_component * p_not_detected * pf_per_flaw))
