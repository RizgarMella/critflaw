"""Random variables (theory manual, section 4.1).

Distribution objects are compared by identity: one object used in two places of a
problem is one random variable. All functions are vectorised.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import integrate, optimize, special, stats

FloatArray = NDArray[np.float64]


def _phi_cdf(z: ArrayLike) -> FloatArray:
    return np.asarray(special.ndtr(np.asarray(z, dtype=float)), dtype=float)


class Distribution:
    """Base class. Subclasses implement ``cdf``, ``sf``, ``ppf``, ``isf`` and ``pdf``."""

    is_discrete = False
    name: str | None = None

    def cdf(self, x: ArrayLike) -> FloatArray:
        raise NotImplementedError

    def sf(self, x: ArrayLike) -> FloatArray:
        raise NotImplementedError

    def ppf(self, q: ArrayLike) -> FloatArray:
        raise NotImplementedError

    def isf(self, q: ArrayLike) -> FloatArray:
        raise NotImplementedError

    def pdf(self, x: ArrayLike) -> FloatArray:
        raise NotImplementedError

    def support(self) -> tuple[float, float]:
        raise NotImplementedError

    def mean(self) -> float:
        raise NotImplementedError

    def std(self) -> float:
        raise NotImplementedError

    def from_normal_score(self, z: ArrayLike) -> FloatArray:
        """x = F^-1(Phi(z)), evaluated through the survival function in the upper tail so that
        Phi(z) rounding to 1 does not give an infinite value."""
        z_ = np.asarray(z, dtype=float)
        lower = self.ppf(_phi_cdf(np.minimum(z_, 0.0)))
        upper = self.isf(_phi_cdf(-np.maximum(z_, 0.0)))
        return np.asarray(np.where(z_ <= 0.0, lower, upper), dtype=float)

    def normal_score(self, x: ArrayLike) -> FloatArray:
        """z = Phi^-1(F(x)), tail-safe. For discrete variables the mid-distribution score."""
        x_ = np.asarray(x, dtype=float)
        cdf, sf = self.cdf(x_), self.sf(x_)
        return np.asarray(np.where(cdf <= 0.5, special.ndtri(cdf), -special.ndtri(sf)), dtype=float)

    def sample(self, rng: np.random.Generator, size: int) -> FloatArray:
        return self.from_normal_score(rng.standard_normal(size))

    def __repr__(self) -> str:
        label = f" {self.name!r}" if self.name else ""
        return f"<{type(self).__name__}{label} mean={self.mean():.6g} sd={self.std():.6g}>"


class _Continuous(Distribution):
    """A continuous distribution backed by a frozen scipy.stats distribution."""

    def __init__(self, frozen: Any, name: str | None = None) -> None:
        self._d = frozen
        self.name = name

    def cdf(self, x: ArrayLike) -> FloatArray:
        return np.asarray(self._d.cdf(x), dtype=float)

    def sf(self, x: ArrayLike) -> FloatArray:
        return np.asarray(self._d.sf(x), dtype=float)

    def ppf(self, q: ArrayLike) -> FloatArray:
        return np.asarray(self._d.ppf(q), dtype=float)

    def isf(self, q: ArrayLike) -> FloatArray:
        return np.asarray(self._d.isf(q), dtype=float)

    def pdf(self, x: ArrayLike) -> FloatArray:
        return np.asarray(self._d.pdf(x), dtype=float)

    def support(self) -> tuple[float, float]:
        lo, hi = self._d.support()
        return float(lo), float(hi)

    def mean(self) -> float:
        return float(self._d.mean())

    def std(self) -> float:
        return float(self._d.std())


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class Normal(_Continuous):
    def __init__(self, mean: float, sd: float, name: str | None = None) -> None:
        _check(np.isfinite(mean) and np.isfinite(sd) and sd > 0.0, "Normal needs a finite mean and sd > 0")
        super().__init__(stats.norm(loc=mean, scale=sd), name)


class LogNormal(_Continuous):
    """Lognormal by median and coefficient of variation (or ``from_mean_sd``).

    sigma_ln = sqrt(ln(1 + cov^2)),  mu_ln = ln(mean) - sigma_ln^2 / 2.
    """

    def __init__(self, median: float, cov: float, name: str | None = None) -> None:
        ok = median > 0.0 and cov > 0.0 and np.isfinite(median) and np.isfinite(cov)
        _check(bool(ok), "LogNormal needs median, cov > 0")
        self.sigma_ln = float(np.sqrt(np.log1p(cov**2)))
        self.mu_ln = float(np.log(median))
        super().__init__(stats.lognorm(s=self.sigma_ln, scale=median), name)

    @classmethod
    def from_mean_sd(cls, mean: float, sd: float, name: str | None = None) -> LogNormal:
        _check(mean > 0.0 and sd > 0.0, "LogNormal.from_mean_sd needs mean, sd > 0")
        cov = sd / mean
        return cls(median=mean / np.sqrt(1.0 + cov**2), cov=cov, name=name)


class Weibull(_Continuous):
    """Weibull with scale ``theta``, shape ``k`` and optional location (three-parameter form)."""

    def __init__(self, scale: float, shape: float, loc: float = 0.0, name: str | None = None) -> None:
        _check(scale > 0.0 and shape > 0.0 and np.isfinite(loc), "Weibull needs scale, shape > 0 and a finite loc")
        self.scale, self.shape, self.loc = float(scale), float(shape), float(loc)
        super().__init__(stats.weibull_min(c=shape, scale=scale, loc=loc), name)

    @classmethod
    def from_mean_sd(cls, mean: float, sd: float, loc: float = 0.0, name: str | None = None) -> Weibull:
        """Solve the shape from the coefficient of variation of (X - loc)."""
        _check(mean > loc and sd > 0.0, "Weibull.from_mean_sd needs mean > loc and sd > 0")
        target = (sd / (mean - loc)) ** 2

        def cov2(k: float) -> float:
            g1, g2 = special.gammaln(1.0 + 1.0 / k), special.gammaln(1.0 + 2.0 / k)
            return float(np.expm1(g2 - 2.0 * g1) - target)

        lo, hi = 0.02, 500.0
        while cov2(hi) > 0.0 and hi < 1e9:  # small coefficients of variation need a large shape
            hi *= 10.0
        if cov2(lo) < 0.0 or cov2(hi) > 0.0:
            raise ValueError("no Weibull shape reproduces this coefficient of variation")
        shape = optimize.brentq(cov2, lo, hi, xtol=1e-14, rtol=1e-14)
        scale = (mean - loc) / special.gamma(1.0 + 1.0 / shape)
        return cls(scale=scale, shape=shape, loc=loc, name=name)


class Gumbel(_Continuous):
    """Gumbel distribution of maxima with mode ``mode`` and scale ``scale``."""

    def __init__(self, mode: float, scale: float, name: str | None = None) -> None:
        _check(np.isfinite(mode) and scale > 0.0, "Gumbel needs a finite mode and scale > 0")
        super().__init__(stats.gumbel_r(loc=mode, scale=scale), name)

    @classmethod
    def from_mean_sd(cls, mean: float, sd: float, name: str | None = None) -> Gumbel:
        scale = sd * np.sqrt(6.0) / np.pi
        return cls(mode=mean - np.euler_gamma * scale, scale=scale, name=name)


class Uniform(_Continuous):
    def __init__(self, lo: float, hi: float, name: str | None = None) -> None:
        _check(np.isfinite(lo) and np.isfinite(hi) and hi > lo, "Uniform needs finite lo < hi")
        super().__init__(stats.uniform(loc=lo, scale=hi - lo), name)


class Exponential(_Continuous):
    """Exponential distribution starting at ``loc`` with mean ``mean`` (so the scale is mean - loc)."""

    def __init__(self, mean: float, loc: float = 0.0, name: str | None = None) -> None:
        _check(mean > loc and np.isfinite(mean), "Exponential needs mean > loc")
        super().__init__(stats.expon(loc=loc, scale=mean - loc), name)


class Truncated(Distribution):
    """``dist`` conditioned on lo <= X <= hi. Tail-safe: uses whichever of cdf / sf is small."""

    def __init__(self, dist: Distribution, lo: float = -np.inf, hi: float = np.inf, name: str | None = None) -> None:
        _check(not dist.is_discrete, "Truncated needs a continuous distribution")
        _check(hi > lo, "Truncated needs lo < hi")
        self.dist, self.lo, self.hi = dist, float(lo), float(hi)
        self.name = name if name is not None else dist.name
        self._f_lo, self._s_lo = float(dist.cdf(lo)), float(dist.sf(lo))
        self._f_hi, self._s_hi = float(dist.cdf(hi)), float(dist.sf(hi))
        # mass kept, computed without cancellation on whichever side is in the tail
        self._mass = self._f_hi - self._f_lo if self._f_lo < 0.5 else self._s_lo - self._s_hi
        _check(self._mass > 0.0, "truncation interval has zero probability")
        self._moments: tuple[float, float] | None = None

    def support(self) -> tuple[float, float]:
        lo, hi = self.dist.support()
        return max(lo, self.lo), min(hi, self.hi)

    def cdf(self, x: ArrayLike) -> FloatArray:
        x_ = np.clip(np.asarray(x, dtype=float), self.lo, self.hi)
        if self._f_lo < 0.5:
            raw = (self.dist.cdf(x_) - self._f_lo) / self._mass
        else:
            raw = (self._s_lo - self.dist.sf(x_)) / self._mass
        return np.asarray(np.clip(raw, 0.0, 1.0), dtype=float)

    def sf(self, x: ArrayLike) -> FloatArray:
        x_ = np.clip(np.asarray(x, dtype=float), self.lo, self.hi)
        if self._s_hi < 0.5:
            raw = (self.dist.sf(x_) - self._s_hi) / self._mass
        else:
            raw = (self._f_hi - self.dist.cdf(x_)) / self._mass
        return np.asarray(np.clip(raw, 0.0, 1.0), dtype=float)

    def ppf(self, q: ArrayLike) -> FloatArray:
        q_ = np.asarray(q, dtype=float)
        if self._f_lo < 0.5:
            x = self.dist.ppf(self._f_lo + q_ * self._mass)
        else:
            x = self.dist.isf(self._s_lo - q_ * self._mass)
        return np.asarray(np.clip(x, self.lo, self.hi), dtype=float)

    def isf(self, q: ArrayLike) -> FloatArray:
        q_ = np.asarray(q, dtype=float)
        if self._s_hi < 0.5:
            x = self.dist.isf(self._s_hi + q_ * self._mass)
        else:
            x = self.dist.ppf(self._f_hi - q_ * self._mass)
        return np.asarray(np.clip(x, self.lo, self.hi), dtype=float)

    def pdf(self, x: ArrayLike) -> FloatArray:
        x_ = np.asarray(x, dtype=float)
        inside = (x_ >= self.lo) & (x_ <= self.hi)
        return np.asarray(np.where(inside, self.dist.pdf(x_) / self._mass, 0.0), dtype=float)

    def _mean_std(self) -> tuple[float, float]:
        if self._moments is None:
            # Integrate over the two halves in the probability measured from the nearer end, so that
            # both tails are resolved; the variance is the centred second moment (no cancellation).
            def lower(q: float) -> float:
                return float(self.ppf(q))

            def upper(q: float) -> float:
                return float(self.isf(q))

            def integral(fn: Any) -> float:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", integrate.IntegrationWarning)
                    return float(integrate.quad(fn, 0.0, 0.5, limit=400, epsabs=0.0, epsrel=1e-10)[0])

            m1 = integral(lower) + integral(upper)
            var = integral(lambda q: (lower(q) - m1) ** 2) + integral(lambda q: (upper(q) - m1) ** 2)
            self._moments = (m1, float(np.sqrt(var)))
        return self._moments

    def mean(self) -> float:
        return self._mean_std()[0]

    def std(self) -> float:
        return self._mean_std()[1]


class Empirical(Distribution):
    """Discrete distribution with probability masses at given values (masses are real: a mass
    exactly on the assessment line counts as failure)."""

    is_discrete = True

    def __init__(self, values: ArrayLike, weights: ArrayLike | None = None, name: str | None = None) -> None:
        v = np.asarray(values, dtype=float).ravel()
        w = np.full(v.size, 1.0 / v.size) if weights is None else np.asarray(weights, dtype=float).ravel()
        _check(v.size >= 1 and v.size == w.size, "Empirical needs matching values and weights")
        ok = bool(np.all(np.isfinite(v))) and bool(np.all(w > 0.0))
        _check(ok, "Empirical needs finite values and positive weights")
        order = np.argsort(v, kind="stable")
        v, w = v[order], w[order] / np.sum(w)
        self.values, inverse = np.unique(v, return_inverse=True)
        self.weights = np.bincount(inverse, weights=w)
        self._cum = np.cumsum(self.weights)
        self._cum[-1] = 1.0
        # P(X > value_i), accumulated from the top so that tiny upper-tail masses are not lost
        self._surv = np.concatenate([np.cumsum(self.weights[::-1])[::-1][1:], [0.0]])
        self.name = name

    def cdf(self, x: ArrayLike) -> FloatArray:
        idx = np.searchsorted(self.values, np.asarray(x, dtype=float), side="right")
        return np.asarray(np.concatenate([[0.0], self._cum])[idx], dtype=float)

    def sf(self, x: ArrayLike) -> FloatArray:
        idx = np.searchsorted(self.values, np.asarray(x, dtype=float), side="right")
        return np.asarray(np.concatenate([[1.0], self._surv])[idx], dtype=float)

    def ppf(self, q: ArrayLike) -> FloatArray:
        idx = np.searchsorted(self._cum, np.asarray(q, dtype=float), side="left")
        return np.asarray(self.values[np.minimum(idx, self.values.size - 1)], dtype=float)

    def isf(self, q: ArrayLike) -> FloatArray:
        """Smallest value whose survival probability does not exceed q (computed from the top)."""
        q_ = np.asarray(q, dtype=float)
        # _surv is non-increasing; count entries strictly greater than q
        idx = np.sum(self._surv[None, ...] > q_[..., None], axis=-1) if q_.ndim else int(np.sum(self._surv > q_))
        return np.asarray(self.values[np.minimum(idx, self.values.size - 1)], dtype=float)

    def pdf(self, x: ArrayLike) -> FloatArray:
        raise TypeError("a discrete distribution has no density")

    def normal_score(self, x: ArrayLike) -> FloatArray:
        """Mid-distribution score Phi^-1(F(x-) + P(X = x) / 2), from whichever tail is smaller."""
        x_ = np.asarray(x, dtype=float)
        below = np.concatenate([[0.0], self._cum])[np.searchsorted(self.values, x_, side="left")]
        mid_lower = 0.5 * (below + self.cdf(x_))
        above = self.sf(x_)
        at_or_above = np.concatenate([[1.0], self._surv])[np.searchsorted(self.values, x_, side="left")]
        mid_upper = 0.5 * (above + at_or_above)
        return np.asarray(np.where(mid_lower <= 0.5, special.ndtri(mid_lower), -special.ndtri(mid_upper)), dtype=float)

    def support(self) -> tuple[float, float]:
        return float(self.values[0]), float(self.values[-1])

    def mean(self) -> float:
        return float(np.sum(self.values * self.weights))

    def std(self) -> float:
        return float(np.sqrt(np.sum(self.weights * (self.values - self.mean()) ** 2)))


class Deterministic(Empirical):
    def __init__(self, value: float, name: str | None = None) -> None:
        super().__init__([value], name=name)
