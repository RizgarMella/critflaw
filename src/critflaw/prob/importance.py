"""Importance sampling as an optional refinement of a rare-event estimate (theory manual, section 4.4.2).

The proposal is a defensive mixture q = w0 phi + sum_j w_j phi(. - c_j) in standard normal
space. Because q >= w0 phi the weights are bounded by 1/w0, so the estimator has finite
variance whatever the centres miss, and an interval that holds for any limit state can be
given (it is wide at small probabilities). The usual asymptotic interval is also reported,
labelled for what it assumes: that the proposal covers every failure region.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import special, stats

from critflaw.prob.intervals import Interval, empirical_bernstein
from critflaw.prob.solvers import LimitState

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ImportanceResult:
    pf: float
    cov: float  # coefficient of variation of the estimator (sample estimate)
    effective_failures: float  # (sum w)^2 / sum w^2 over failing samples
    conditional_interval: Interval | None  # asymptotic; valid only if every failure region is covered
    guaranteed_interval: Interval  # empirical Bernstein; holds whatever was missed
    component_shares: tuple[float, ...]  # share of the estimate from samples nearest each centre
    defensive_share: float  # share from samples drawn by the defensive (nominal) component
    samples: int


def importance_sampling(
    limit_state: LimitState,
    dimension: int,
    centres: FloatArray,
    samples: int,
    seed: int,
    defensive_weight: float = 0.2,
    confidence: float = 0.99,
) -> ImportanceResult:
    """Mixture importance sampling centred on ``centres`` (k, dimension), unit covariance.

    The conditional interval is given only when the effective number of failing samples is at
    least 100. That gate guards against a noisy estimate; it says nothing about a failure
    region the centres miss (the estimate is then low with a small reported scatter), which is
    why only the guaranteed interval may be relied on without knowledge of the regions.
    """
    c = np.atleast_2d(np.asarray(centres, dtype=float))
    if c.shape[1] != dimension or c.shape[0] < 1 or not np.all(np.isfinite(c)):
        raise ValueError("centres must be finite and have shape (k, dimension) with k >= 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if not 0.0 < defensive_weight < 1.0:
        raise ValueError("defensive_weight must lie in (0, 1)")
    if samples < 2:
        raise ValueError("need at least two samples")
    rng = np.random.default_rng(seed)
    k = c.shape[0]
    w = np.concatenate([[defensive_weight], np.full(k, (1.0 - defensive_weight) / k)])
    component = rng.choice(k + 1, size=samples, p=w)
    shift = np.vstack([np.zeros(dimension), c])[component]
    u = rng.standard_normal((samples, dimension)) + shift

    # log of q / phi = log sum_j w_j exp(u . c_j - |c_j|^2 / 2), with c_0 = 0
    exponents = u @ c.T - 0.5 * np.sum(c**2, axis=1)
    log_ratio = special.logsumexp(np.column_stack([np.zeros(samples), exponents]), b=w, axis=1)
    weights = np.exp(-log_ratio)  # phi / q, bounded by 1 / defensive_weight
    fail = limit_state(u).fail
    terms = np.where(fail, weights, 0.0)

    pf = float(np.mean(terms))
    variance = float(np.var(terms, ddof=1))
    std_error = float(np.sqrt(variance / samples))
    cov = std_error / pf if pf > 0.0 else float("inf")
    total, total_sq = float(np.sum(terms)), float(np.sum(terms**2))
    ess = total**2 / total_sq if total_sq > 0.0 else 0.0

    conditional = None
    if pf > 0.0 and ess >= 100.0:
        z = float(stats.norm.ppf(0.5 + confidence / 2.0))
        sigma_ln = float(np.sqrt(np.log1p(cov**2)))
        conditional = Interval(
            pf * float(np.exp(-z * sigma_ln)),
            min(pf * float(np.exp(z * sigma_ln)), 1.0),
            confidence,
            "asymptotic; conditional on the proposal covering every failure region",
        )
    bernstein = empirical_bernstein(pf, variance, samples, 1.0 / defensive_weight, confidence)
    guaranteed = Interval(bernstein.lower, min(bernstein.upper, 1.0), confidence, bernstein.kind)

    nearest = np.argmin(((u[:, None, :] - c[None, :, :]) ** 2).sum(axis=2), axis=1)
    shares = tuple(float(np.sum(terms[nearest == j]) / total) if total > 0.0 else 0.0 for j in range(k))
    defensive = float(np.sum(terms[component == 0]) / total) if total > 0.0 else 0.0
    return ImportanceResult(pf, cov, ess, conditional, guaranteed, shares, defensive, samples)
