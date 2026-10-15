"""Fatigue crack growth laws (theory manual, section 5.1).

Internal units are SI: rate in m/cycle, stress intensity factor range in Pa sqrt(m). The published
coefficients are tabulated for mm/cycle and N/mm^(3/2); the conversion depends on the exponent and
is done once, at construction  [provenance: fatigue.paris_law, fatigue.recommended_laws,
fatigue.thresholds].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from critflaw.model import _Replaceable
from critflaw.prob.distributions import Distribution, LogNormal

FloatArray = NDArray[np.float64]
Value = Any  # float, array or Distribution

# 1 Pa sqrt(m) in N/mm^(3/2); 1 m in mm
_DK_NMM_PER_SI = 1e-6 * np.sqrt(1000.0)
_RATE_M_PER_MM = 1e-3


def coefficient_to_si(a_nmm: ArrayLike, m: ArrayLike) -> FloatArray:
    """A for m/cycle and Pa sqrt(m) from A for mm/cycle and N/mm^(3/2): A 10^-3 (10^-6 sqrt(1000))^m."""
    return np.asarray(np.asarray(a_nmm, dtype=float) * _RATE_M_PER_MM * _DK_NMM_PER_SI ** np.asarray(m, dtype=float))


def dk_to_si(dk_nmm: ArrayLike) -> FloatArray:
    """Pa sqrt(m) from N/mm^(3/2)."""
    return np.asarray(np.asarray(dk_nmm, dtype=float) / _DK_NMM_PER_SI, dtype=float)


def transition_point(a_a: ArrayLike, m_a: ArrayLike, a_b: ArrayLike, m_b: ArrayLike) -> FloatArray:
    """Intersection of the two stage lines, in the units of the coefficients."""
    aa, ma, ab, mb = (np.asarray(v, dtype=float) for v in (a_a, m_a, a_b, m_b))
    return np.asarray((ab / aa) ** (1.0 / (ma - mb)), dtype=float)


def _log10_normal(a_median_si: float, sd_log10: float, name: str) -> LogNormal:
    """A with log10 A normal: lognormal with the mean law as median."""
    sigma_ln = sd_log10 * np.log(10.0)
    return LogNormal(a_median_si, float(np.sqrt(np.expm1(sigma_ln**2))), name=name)


@dataclass(frozen=True)
class GrowthLaw(_Replaceable):
    """Rate = 0 below ``threshold``; ``a_coefficient dk^m``; with a second stage the minimum of the two
    stage lines (transition by intersection; requires ``m > m_b``). SI units. Coefficients may be
    distributions (log10 A normal: the mean law with the published scatter), and are then realised
    per sample by the probabilistic layer.

    ``shallow_cap`` is a (depth, threshold) pair: for a surface flaw shallower than ``depth`` the
    threshold used is the smaller of ``threshold`` and the cap.
    """

    a_coefficient: Value
    m: float
    threshold: Value = 0.0
    a_coefficient_b: Value | None = None
    m_b: float | None = None
    stage_correlation: float | None = None  # correlation of log A between the stages (random laws)
    shallow_cap: tuple[float, float] | None = None
    name: str = "growth law"
    validity: str = ""  # the catalogue's statement of applicability; carried as text
    effective_range: bool = False  # the source's dK_eff form near the threshold: not implemented (no constants)

    def __post_init__(self) -> None:
        if self.a_coefficient_b is not None:
            if self.m_b is None or not self.m > self.m_b:
                raise ValueError("a two-stage law needs m > m_b (the lines must cross once, stage A below the knee)")
            random = isinstance(self.a_coefficient, Distribution) or isinstance(self.a_coefficient_b, Distribution)
            if random and self.stage_correlation is None:
                raise ValueError("a two-stage law with random coefficients needs stage_correlation (no default)")
        if self.stage_correlation is not None and not -1.0 <= self.stage_correlation <= 1.0:
            raise ValueError("stage_correlation must lie in [-1, 1]")
        if not self.m > 0.0:
            raise ValueError("m must be positive")
        if self.shallow_cap is not None and not (self.shallow_cap[0] > 0.0 and self.shallow_cap[1] >= 0.0):
            raise ValueError("shallow_cap must be (depth > 0, threshold >= 0)")
        for name in ("a_coefficient", "a_coefficient_b", "threshold"):
            v = getattr(self, name)
            if v is None or isinstance(v, Distribution):
                continue
            if not np.all(np.asarray(v, dtype=float) >= 0.0):
                raise ValueError(f"{name} must be non-negative")

    @property
    def two_stage(self) -> bool:
        return self.a_coefficient_b is not None

    @property
    def is_random(self) -> bool:
        return any(
            isinstance(getattr(self, n), Distribution) for n in ("a_coefficient", "a_coefficient_b", "threshold")
        )

    def effective_threshold(self, depth: ArrayLike | None = None) -> FloatArray:
        thr = np.asarray(self.threshold, dtype=float)
        if depth is None or self.shallow_cap is None:
            return np.asarray(thr + 0.0, dtype=float)
        shallow = np.asarray(depth, dtype=float) < self.shallow_cap[0]
        return np.asarray(np.where(shallow, np.minimum(thr, self.shallow_cap[1]), thr), dtype=float)

    def stage_rates(self, dk: ArrayLike) -> tuple[FloatArray, FloatArray | None]:
        """The two stage lines (without threshold or minimum), for event logic."""
        dk_ = np.maximum(np.asarray(dk, dtype=float), 0.0)
        ra = np.asarray(self.a_coefficient, dtype=float) * dk_**self.m
        if not self.two_stage:
            return np.asarray(ra, dtype=float), None
        rb = np.asarray(self.a_coefficient_b, dtype=float) * dk_ ** float(self.m_b)  # type: ignore[arg-type]
        return np.asarray(ra, dtype=float), np.asarray(rb, dtype=float)

    def rate(self, dk: ArrayLike, depth: ArrayLike | None = None) -> FloatArray:
        """Growth per cycle (m/cycle) for a range ``dk`` (Pa sqrt(m)); zero below the threshold."""
        dk_ = np.asarray(dk, dtype=float)
        ra, rb = self.stage_rates(dk_)
        r = ra if rb is None else np.minimum(ra, rb)
        above = dk_ >= self.effective_threshold(depth)
        return np.asarray(np.where(above & (dk_ > 0.0), r, 0.0), dtype=float)

    def transition(self) -> FloatArray | None:
        if self.a_coefficient_b is None or self.m_b is None:
            return None
        return transition_point(self.a_coefficient, self.m, self.a_coefficient_b, self.m_b)


# --------------------------------------------------------------------------------------
# Catalogue (values from the independent transcription; primary reference as cited by the source)
# --------------------------------------------------------------------------------------

Curve = Literal["mean", "mean-fixed", "upper"]
StressRatio = Literal["low", "high"]  # R < 0.5, R >= 0.5

# environment -> stress ratio class -> (stage A: A_mean, m, A_upper, sd_log10; stage B: same), N/mm units
_TABLE: dict[str, dict[str, tuple[tuple[float, float, float, float], tuple[float, float, float, float]]]] = {
    "air": {
        "low": ((1.21e-26, 8.16, 4.37e-26, 0.279), (3.98e-13, 2.88, 6.77e-13, 0.115)),
        "high": ((4.80e-18, 5.10, 2.10e-17, 0.320), (5.86e-13, 2.88, 1.29e-12, 0.171)),
    },
    "marine-free": {
        "low": ((3.0e-14, 3.42, 8.55e-14, 0.227), (1.27e-7, 1.30, 1.93e-7, 0.091)),
        "high": ((5.37e-14, 3.42, 1.72e-13, 0.253), (5.67e-7, 1.11, 7.48e-7, 0.060)),
    },
    "marine-cp850": {
        "low": ((1.21e-26, 8.16, 4.37e-26, 0.279), (5.16e-12, 2.67, 1.32e-11, 0.204)),
        "high": ((4.80e-18, 5.10, 2.10e-17, 0.320), (6.0e-12, 2.67, 2.02e-11, 0.264)),
    },
    "marine-cp1100": {
        "low": ((1.21e-26, 8.16, 4.37e-26, 0.279), (5.51e-8, 1.40, 9.24e-8, 0.112)),
        "high": ((4.80e-18, 5.10, 2.10e-17, 0.320), (5.25e-8, 1.40, 1.02e-7, 0.144)),
    },
}
# printed transition points (mean, upper) per environment and class, N/mm^(3/2): used only in tests
PRINTED_TRANSITIONS = {
    ("air", "low"): (363.0, 315.0),
    ("air", "high"): (196.0, 144.0),
    ("marine-free", "low"): (1336.0, 993.0),
    ("marine-free", "high"): (1098.0, 748.0),
    ("marine-cp850", "low"): (462.0, 434.0),
    ("marine-cp850", "high"): (323.0, 290.0),
    ("marine-cp1100", "low"): (576.0, 514.0),
    ("marine-cp1100", "high"): (517.0, 415.0),
}
_VALIDITY = {
    "air": "steels (ferritic, austenitic, duplex) with yield or 0.2% proof strength <= 700 N/mm^2; air or other "
    "non-aggressive environments up to 100 C",
    "marine-free": "steels (not austenitic or duplex) with yield or 0.2% proof strength <= 600 N/mm^2; marine "
    "environment up to 20 C, freely corroding; data at 5-20 C and 0.17-0.5 Hz",
    "marine-cp850": "as marine-free, with cathodic protection at -850 mV (Ag/AgCl)",
    "marine-cp1100": "as marine-free, with cathodic protection at -1100 mV (Ag/AgCl)",
}
THRESHOLD_WELDED_NMM = {"air": 63.0, "marine-cp": 63.0, "marine-free": 0.0, "aluminium-air": 21.0}
SHALLOW_CAP = (1.0e-3, float(dk_to_si(63.0)))  # surface flaws less than 1 mm deep: threshold at most 63 N/mm^(3/2)
SIMPLIFIED_A_NMM = {"air": 5.21e-13, "marine": 2.3e-12}


def table_entry(environment: str, stress_ratio: StressRatio) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return _TABLE[environment][stress_ratio]


def recommended_law(
    environment: str,
    stress_ratio: StressRatio = "high",
    curve: Curve = "upper",
    stage_correlation: float | None = None,
    threshold_nmm: float | None = None,
    welded: bool = True,
) -> GrowthLaw:
    """Two-stage law from the catalogue. ``curve="upper"``: mean + 2 SD, fixed (the recommendation for
    welded joints at R >= 0.5); ``"mean"``: the mean law with log10 A random (needs
    ``stage_correlation``); ``"mean-fixed"``: the mean law with fixed coefficients.

    The threshold defaults to the welded-joint value for the environment (63 N/mm^(3/2) in air or
    with cathodic protection, 0 freely corroding); for a surface flaw shallower than 1 mm the
    threshold is capped at 63 N/mm^(3/2).
    """
    if environment not in _TABLE:
        raise ValueError(f"environment must be one of {sorted(_TABLE)}")
    (aa, ma, aa_up, sd_a), (ab, mb, ab_up, sd_b) = _TABLE[environment][stress_ratio]
    if threshold_nmm is None:
        threshold_nmm = THRESHOLD_WELDED_NMM["marine-free" if environment == "marine-free" else "air"]
    label = f"{environment}, R {'>= 0.5' if stress_ratio == 'high' else '< 0.5'}, {curve}"
    if curve == "upper":
        a1: Value = float(coefficient_to_si(aa_up, ma))
        a2: Value = float(coefficient_to_si(ab_up, mb))
    elif curve == "mean-fixed":
        a1, a2 = float(coefficient_to_si(aa, ma)), float(coefficient_to_si(ab, mb))
    elif curve == "mean":
        if stage_correlation is None:
            raise ValueError("the mean law has random coefficients: stage_correlation is required (no default)")
        a1 = _log10_normal(float(coefficient_to_si(aa, ma)), sd_a, f"log A stage A ({label})")
        a2 = _log10_normal(float(coefficient_to_si(ab, mb)), sd_b, f"log A stage B ({label})")
    else:
        raise ValueError("curve must be 'upper', 'mean' or 'mean-fixed'")
    return GrowthLaw(
        a1,
        ma,
        float(dk_to_si(threshold_nmm)),
        a2,
        mb,
        stage_correlation if curve == "mean" else None,
        SHALLOW_CAP if welded else None,
        name=label,
        validity=_VALIDITY[environment],
    )


def steel_in_air(
    stress_ratio: StressRatio = "high", curve: Curve = "upper", stage_correlation: float | None = None
) -> GrowthLaw:
    return recommended_law("air", stress_ratio, curve, stage_correlation)


def steel_in_seawater(
    protection: Literal["free", "cp850", "cp1100"] = "free",
    stress_ratio: StressRatio = "high",
    curve: Curve = "upper",
    stage_correlation: float | None = None,
) -> GrowthLaw:
    return recommended_law(
        {"free": "marine-free", "cp850": "marine-cp850", "cp1100": "marine-cp1100"}[protection],
        stress_ratio,
        curve,
        stage_correlation,
    )


def simplified_law(environment: Literal["air", "marine"] = "air", modulus_ratio: float = 1.0) -> GrowthLaw:
    """m = 3 upper-bound laws; ``modulus_ratio`` = E_steel / E (or E_RT / E_ET): A scales with its cube
    and the threshold with its inverse."""
    a = SIMPLIFIED_A_NMM[environment] * modulus_ratio**3
    thr = THRESHOLD_WELDED_NMM["air" if environment == "air" else "marine-free"] / modulus_ratio
    return GrowthLaw(
        float(coefficient_to_si(a, 3.0)),
        3.0,
        float(dk_to_si(thr)),
        shallow_cap=SHALLOW_CAP,
        name=f"simplified, {environment}",
        validity="steels with yield or 0.2% proof strength <= 600 N/mm^2",
    )


def threshold_unwelded_steel(stress_ratio: ArrayLike) -> FloatArray:
    """Threshold for unwelded steel (air, or marine with cathodic protection, up to 20 C) as a function
    of the stress ratio R, in Pa sqrt(m): 63 for R >= 0.5, 170 - 214 R for 0 <= R < 0.5, 170 for R < 0."""
    r = np.asarray(stress_ratio, dtype=float)
    nmm = np.where(r >= 0.5, 63.0, np.where(r >= 0.0, 170.0 - 214.0 * r, 170.0))
    return dk_to_si(nmm)
