"""Fracture toughness estimates (theory manual, chapter 03, sections 3.2 to 3.4).

Public functions take and return SI units (Pa, m, Pa m^0.5, J/m^2, joules, degrees C).
The correlations themselves are written in the units of their sources (MPa m^0.5, mm, J);
the conversions are local to each function.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

_MPA_SQRT_M = 1.0e6
K_MIN_MASTER_CURVE = 20.0 * _MPA_SQRT_M  # lower bound of the Master Curve distribution
CHARPY_UPPER_LIMIT_MAX_YIELD = 480.0e6


def _f(x: ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=float)


def k_from_j(j_mat: ArrayLike, modulus: ArrayLike, poisson: float = 0.3) -> FloatArray:
    """Kmat = sqrt(E J / (1 - nu^2))  [provenance: toughness.k_from_j]."""
    return np.asarray(np.sqrt(_f(modulus) * _f(j_mat) / (1.0 - poisson**2)), dtype=float)


def ctod_constraint_factor(test_yield: ArrayLike, test_uts: ArrayLike) -> FloatArray:
    """m = 1.517 (sy/su)^-0.3188 for 0.3 < sy/su < 0.98, otherwise 1.5  [provenance: toughness.k_from_ctod]."""
    ratio = _f(test_yield) / _f(test_uts)
    inside = (ratio > 0.3) & (ratio < 0.98)
    return np.asarray(np.where(inside, 1.517 * np.where(inside, ratio, 1.0) ** -0.3188, 1.5), dtype=float)


def k_from_ctod(
    ctod: ArrayLike, test_yield: ArrayLike, test_uts: ArrayLike, modulus: ArrayLike, poisson: float = 0.3
) -> FloatArray:
    """Kmat = sqrt(m sy delta E / (1 - nu^2))  [provenance: toughness.k_from_ctod].

    ``test_yield`` and ``test_uts`` are the tensile properties of the material in which CTOD
    was measured, at the test temperature. They are not the (possibly sampled) properties
    used in the assessment.
    """
    m = ctod_constraint_factor(test_yield, test_uts)
    return np.asarray(np.sqrt(m * _f(test_yield) * _f(ctod) * _f(modulus) / (1.0 - poisson**2)), dtype=float)


def k_jc_limit(ligament: ArrayLike, test_yield: ArrayLike, modulus: ArrayLike, poisson: float = 0.3) -> FloatArray:
    """Small-scale-yielding limit K = sqrt(E b0 sy / (30 (1 - nu^2)))  [provenance: toughness.subsize]."""
    return np.asarray(np.sqrt(_f(modulus) * _f(ligament) * _f(test_yield) / (30.0 * (1.0 - poisson**2))), dtype=float)


def subsize_correction(k_small: ArrayLike, specimen_thickness: ArrayLike, component_thickness: ArrayLike) -> FloatArray:
    """K_B = 20 + (K_b - 20) (b / B)^0.25 in MPa m^0.5  [provenance: toughness.subsize].

    For results ending in cleavage from specimens thinner than the component. The caller must
    have checked the two conditions of the source: limited prior ductile extension, and
    K_b below :func:`k_jc_limit` (a value above it is replaced by the limit).
    """
    kb = _f(k_small) / _MPA_SQRT_M
    out = 20.0 + (kb - 20.0) * (_f(specimen_thickness) / _f(component_thickness)) ** 0.25
    return np.asarray(out * _MPA_SQRT_M, dtype=float)


def master_curve(
    probability: ArrayLike,
    temperature: ArrayLike,
    t0: ArrayLike,
    crack_front_length: ArrayLike,
    thickness: ArrayLike,
) -> FloatArray:
    """Master Curve toughness at cumulative probability ``probability``  [provenance: toughness.master_curve].

    K = 20 + [11 + 77 exp(0.019 (T - T0))] (25 / l)^0.25 [-ln(1 - p)]^0.25   (MPa m^0.5, l in mm)

    ``crack_front_length`` l is in metres and is capped at twice the section thickness.
    Temperatures in degrees C. The distribution has a lower bound of 20 MPa m^0.5.
    """
    p = _f(probability)
    if np.any((p < 0.0) | (p >= 1.0)):
        raise ValueError("probability must lie in [0, 1)")
    length_mm = np.minimum(_f(crack_front_length), 2.0 * _f(thickness)) * 1.0e3
    if np.any(length_mm <= 0.0):
        raise ValueError("crack front length must be positive")
    shape = 11.0 + 77.0 * np.exp(0.019 * (_f(temperature) - _f(t0)))
    k = 20.0 + shape * (25.0 / length_mm) ** 0.25 * (-np.log1p(-p)) ** 0.25
    return np.asarray(k * _MPA_SQRT_M, dtype=float)


def t0_from_charpy(transition_temperature: ArrayLike, energy: int = 27) -> FloatArray:
    """T0 = T27J - 18 C or T40J - 24 C (standard deviation 15 C)  [provenance: toughness.charpy_t0]."""
    if energy == 27:
        return np.asarray(_f(transition_temperature) - 18.0, dtype=float)
    if energy == 40:
        return np.asarray(_f(transition_temperature) - 24.0, dtype=float)
    raise ValueError("energy must be 27 or 40 (joules)")


T0_CHARPY_STANDARD_DEVIATION = 15.0  # degrees C


def charpy_master_curve(
    temperature: ArrayLike,
    t0: ArrayLike,
    thickness: ArrayLike,
    probability: float = 0.05,
    temperature_margin: float = 25.0,
) -> FloatArray:
    """Charpy-based Master Curve estimate  [provenance: toughness.charpy_master_curve].

    K = 20 + {11 + 77 exp[0.019 (T - T0 - TK)]} (25 / B)^0.25 [ln(1 / (1 - Pf))]^0.25
    with B the material thickness in mm and TK the margin for scatter of the Charpy
    correlation (+25 C for 90% confidence at 15 C standard deviation). Not for ductile behaviour.
    """
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie in (0, 1)")
    b_mm = _f(thickness) * 1.0e3
    if np.any(b_mm <= 0.0) or not np.all(np.isfinite(b_mm)):
        raise ValueError("thickness must be finite and positive")
    shape = 11.0 + 77.0 * np.exp(0.019 * (_f(temperature) - _f(t0) - temperature_margin))
    k = 20.0 + shape * (25.0 / b_mm) ** 0.25 * np.log(1.0 / (1.0 - probability)) ** 0.25
    return np.asarray(k * _MPA_SQRT_M, dtype=float)


def charpy_lower_bound(charpy_energy: ArrayLike, thickness: ArrayLike) -> FloatArray:
    """K = [12 sqrt(Cv) - 20] (25 / B)^0.25 + 20  (Cv in J, B in mm)  [provenance: toughness.charpy_lower_bound].

    Lower shelf and transition behaviour, Charpy energy at the service temperature.
    """
    cv = _f(charpy_energy)
    if np.any(cv <= 0.0):
        raise ValueError("Charpy energy must be positive")
    b_mm = _f(thickness) * 1.0e3
    if np.any(b_mm <= 0.0) or not np.all(np.isfinite(b_mm)):
        raise ValueError("thickness must be finite and positive")
    k = (12.0 * np.sqrt(cv) - 20.0) * (25.0 / b_mm) ** 0.25 + 20.0
    if np.any(k <= 0.0):
        raise ValueError("the Charpy lower-bound correlation gives a non-positive toughness for these inputs")
    return np.asarray(k * _MPA_SQRT_M, dtype=float)


def charpy_upper_limit(charpy_energy: ArrayLike, yield_strength: ArrayLike) -> FloatArray:
    """Cap K = 0.54 Cv + 55 on Charpy-based estimates  [provenance: toughness.charpy_upper_limit].

    Restricted to ferritic steels with yield strength below 480 MPa.
    """
    if np.any(_f(yield_strength) >= CHARPY_UPPER_LIMIT_MAX_YIELD):
        raise ValueError("the Charpy upper limit applies only to yield strengths below 480 MPa")
    return np.asarray((0.54 * _f(charpy_energy) + 55.0) * _MPA_SQRT_M, dtype=float)


def charpy_upper_shelf(upper_shelf_energy: ArrayLike, modulus: ArrayLike, poisson: float = 0.3) -> FloatArray:
    """Initiation toughness at 0.2 mm tearing from upper-shelf Charpy energy
    [provenance: toughness.charpy_upper_shelf].

    K = sqrt( E (0.53 Cv^1.28) (0.2^(0.133 Cv^0.256)) / (1000 (1 - nu^2)) )   (E in MPa, Cv in J)
    """
    cv = _f(upper_shelf_energy)
    e_mpa = _f(modulus) / 1.0e6
    j = 0.53 * cv**1.28 * 0.2 ** (0.133 * cv**0.256)
    return np.asarray(np.sqrt(e_mpa * j / (1000.0 * (1.0 - poisson**2))) * _MPA_SQRT_M, dtype=float)


def subsize_charpy_temperature_shift(specimen_thickness: ArrayLike) -> FloatArray:
    """Upward shift of the transition temperature for a sub-size Charpy specimen
    [provenance: toughness.charpy_subsize].

    dT = -51.4 ln[2 (B_ss / 10)^0.25 - 1]   (B_ss in mm)
    """
    b_mm = _f(specimen_thickness) * 1.0e3
    if np.any((b_mm <= 10.0 / 16.0) | (b_mm > 10.0)):
        raise ValueError("sub-size Charpy thickness must lie in (0.625, 10] mm")
    return np.asarray(-51.4 * np.log(2.0 * (b_mm / 10.0) ** 0.25 - 1.0), dtype=float)


def minimum_of_three_equivalent(values: ArrayLike) -> float:
    """Kmat from 3 to 15 results as the minimum-of-three equivalent  [provenance: toughness.mote].

    Number of results -> rank of the value used (1 = lowest): 3-5: lowest; 6-10: second
    lowest; 11-15: third lowest.
    """
    v = np.sort(_f(values).ravel())
    if not np.all(np.isfinite(v)):
        raise ValueError("toughness values must be finite")
    if not 3 <= v.size <= 15:
        raise ValueError("the minimum-of-three equivalent is defined for 3 to 15 results")
    rank = 1 if v.size <= 5 else 2 if v.size <= 10 else 3
    return float(v[rank - 1])


def tolerance_factor(count: int, percentile: float = 0.20, confidence: float = 0.90) -> float:
    """One-sided tolerance factor k for a normal sample  [provenance: toughness.lower_bound].

    With mean m and standard deviation S (divisor n - 1) of ``count`` results, m - k S is
    below the ``percentile`` quantile of the population with probability ``confidence``.
    Exact value from the non-central t distribution: k = t'_{n-1, delta}(confidence) / sqrt(n),
    delta = z_{1 - percentile} sqrt(n). This reproduces the source's table for 3 to 20 results
    to its printed precision, except its entry for 16 results (see docs/unsourced_items.md).
    """
    from scipy import stats

    if count < 2:
        raise ValueError("at least two results are needed")
    if not (0.0 < percentile < 1.0 and 0.0 < confidence < 1.0):
        raise ValueError("percentile and confidence must lie in (0, 1)")
    delta = stats.norm.isf(percentile) * np.sqrt(count)
    return float(stats.nct.ppf(confidence, count - 1, delta) / np.sqrt(count))


def lower_bound_toughness(
    values: ArrayLike, percentile: float = 0.20, confidence: float = 0.90, lognormal: bool = False
) -> float:
    """Lower-bound toughness Kbar - k S from a set of results  [provenance: toughness.lower_bound].

    ``lognormal=True`` applies the same treatment to ln(K), which the source prefers for
    cleavage data in the transition region (a normal fit can give very low or negative values).
    """
    v = _f(values).ravel()
    if v.size < 3:
        raise ValueError("at least three results are needed")
    if lognormal:
        if np.any(v <= 0.0):
            raise ValueError("toughness values must be positive")
        v = np.log(v)
    if not np.all(np.isfinite(v)):
        raise ValueError("toughness values must be finite")
    bound = float(np.mean(v) - tolerance_factor(v.size, percentile, confidence) * np.std(v, ddof=1))
    if lognormal:
        return float(np.exp(bound))
    if bound <= 0.0:
        raise ValueError(
            "the normal lower bound is not positive for these data; use lognormal=True (preferred for cleavage data)"
        )
    return bound
