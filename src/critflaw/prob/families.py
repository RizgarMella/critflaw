"""Alternative distribution families with the same mean, standard deviation and support
(theory manual, section 4.8)."""

from __future__ import annotations

import numpy as np
from scipy import optimize

from critflaw.prob.distributions import Distribution, LogNormal, Normal, Truncated, Weibull

FAMILIES = ("normal", "lognormal", "weibull")


class NoMatch(ValueError):
    """No distribution of the requested family has these moments on this support."""


def family_of(dist: Distribution) -> str | None:
    """'normal', 'lognormal' or 'weibull' for a (possibly truncated) distribution of that family."""
    base = dist.dist if isinstance(dist, Truncated) else dist
    if isinstance(base, Normal):
        return "normal"
    if isinstance(base, LogNormal):
        return "lognormal"
    if isinstance(base, Weibull) and base.loc == 0.0:
        return "weibull"
    return None


def _make(family: str, p0: float, p1: float) -> Distribution:
    if family == "normal":
        return Normal(p0, float(np.exp(p1)))
    if family == "lognormal":
        return LogNormal(float(np.exp(p0)), float(np.exp(p1)))
    return Weibull(float(np.exp(p0)), float(np.exp(p1)))


def moment_matched(dist: Distribution, family: str) -> Distribution:
    """A distribution of ``family`` with the same support, mean and standard deviation as ``dist``.

    The moments are matched **after** truncation to the support. Raises :class:`NoMatch` if the
    family cannot live on that support (a lognormal or Weibull needs a non-negative support) or
    cannot reach the moments (a normal truncated at zero only reaches a coefficient of
    variation below 1).
    """
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    mean, sd = dist.mean(), dist.std()
    lo, hi = dist.support()
    natural_lo = -np.inf if family == "normal" else 0.0
    if lo < natural_lo:
        raise NoMatch(f"a {family} distribution cannot have support below zero")
    needs_truncation = lo > natural_lo or np.isfinite(hi)
    if family == "normal" and np.isfinite(lo) != np.isfinite(hi):
        # a normal truncated on one side only: the standard deviation is below the distance of the mean from the bound
        if sd >= (mean - lo if np.isfinite(lo) else hi - mean):
            raise NoMatch(
                f"no normal distribution truncated to [{lo:g}, {hi:g}] has mean {mean:g} and sd {sd:g}: "
                "the standard deviation must be smaller than the distance of the mean from the bound"
            )
    if not needs_truncation:
        if family == "normal":
            return Normal(mean, sd, name=dist.name)
        if family == "lognormal":
            return LogNormal.from_mean_sd(mean, sd, name=dist.name)
        return Weibull.from_mean_sd(mean, sd, name=dist.name)

    def residual(p: np.ndarray) -> np.ndarray:
        try:
            t = Truncated(_make(family, float(p[0]), float(p[1])), lo=lo, hi=hi)
            return np.array([(t.mean() - mean) / sd, (t.std() - sd) / sd])
        except (ValueError, OverflowError, FloatingPointError):
            return np.array([1e6, 1e6])

    if family == "normal":
        start = np.array([mean, np.log(sd)])
    elif family == "lognormal":
        cov = sd / mean
        start = np.array([np.log(mean / np.sqrt(1.0 + cov**2)), np.log(cov)])
    else:
        guess = Weibull.from_mean_sd(mean, sd)
        start = np.array([np.log(guess.scale), np.log(guess.shape)])
    fit = optimize.least_squares(residual, start, xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=400)
    if not np.all(np.abs(fit.fun) < 1e-8):
        raise NoMatch(
            f"no truncated {family} distribution on [{lo:g}, {hi:g}] with mean {mean:g} and sd {sd:g} was found "
            "(none exists, or the moments are within about 1% of the limit of what the family can reach, "
            "where the search does not converge)"
        )
    return Truncated(_make(family, float(fit.x[0]), float(fit.x[1])), lo=lo, hi=hi, name=dist.name)
