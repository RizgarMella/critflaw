"""Intervals for estimated probabilities (theory manual, section 4.4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float
    confidence: float
    kind: str  # how it was obtained and what it assumes

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper


def clopper_pearson(failures: int, samples: int, confidence: float = 0.99) -> Interval:
    """Exact two-sided binomial interval for independent samples of the nominal distribution.

    Coverage is at least ``confidence`` for every true probability. With zero failures the
    upper bound is 1 - (alpha/2)^(1/N); with every sample failing the lower bound is
    (alpha/2)^(1/N). (A one-sided bound at zero failures combined with two-sided intervals
    elsewhere would not have guaranteed coverage.)
    """
    if samples <= 0 or not 0 <= failures <= samples:
        raise ValueError("need samples > 0 and 0 <= failures <= samples")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    alpha = 1.0 - confidence
    if failures == 0:
        return Interval(0.0, float(-np.expm1(np.log(alpha / 2.0) / samples)), confidence, "Clopper-Pearson, two-sided")
    if failures == samples:
        return Interval(float((alpha / 2.0) ** (1.0 / samples)), 1.0, confidence, "Clopper-Pearson, two-sided")
    lower = float(stats.beta.ppf(alpha / 2.0, failures, samples - failures + 1))
    upper = float(stats.beta.isf(alpha / 2.0, failures + 1, samples - failures))
    return Interval(lower, upper, confidence, "Clopper-Pearson, two-sided")


def empirical_bernstein(
    values_mean: float, values_variance: float, samples: int, bound: float, confidence: float = 0.99
) -> Interval:
    """Interval for the mean of i.i.d. summands in [0, bound], valid for any distribution.

    Half-width sqrt(2 V ln(3/delta) / N) + 3 bound ln(3/delta) / N
    (Audibert, Munos and Szepesvari 2009, Theorem 1).
    """
    if samples <= 0 or bound <= 0.0:
        raise ValueError("need samples > 0 and bound > 0")
    log_term = float(np.log(3.0 / (1.0 - confidence)))
    half = float(np.sqrt(2.0 * values_variance * log_term / samples) + 3.0 * bound * log_term / samples)
    return Interval(max(values_mean - half, 0.0), values_mean + half, confidence, "empirical Bernstein (guaranteed)")
