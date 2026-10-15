"""Failure assessment curves Kr = f(Lr) (theory manual, sections 1.2 and 1.3).

All quantities are SI (Pa). Curve parameters may be arrays (one curve per
sample); they broadcast against the ``lr`` argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
Yielding = Literal["continuous", "discontinuous", "unknown"]

_MPA = 1.0e6
# Upper yield-strength limit for which the Lueders strain estimate may be used
# with the discontinuous-yield curve [provenance: material.luders_strain].
LUDERS_ESTIMATE_MAX_YIELD = 946.0 * _MPA


def lr_cutoff(yield_strength: ArrayLike, uts: ArrayLike) -> FloatArray:
    """Plastic collapse cut-off Lr,max = (sy + su) / (2 sy)  [provenance: fad.cutoff]."""
    sy = np.asarray(yield_strength, dtype=float)
    su = np.asarray(uts, dtype=float)
    return np.asarray((sy + su) / (2.0 * sy), dtype=float)


def luders_strain_estimate(yield_strength: ArrayLike) -> FloatArray:
    """Lueders strain 0.0375 (1 - 0.001 sy[MPa])  [provenance: material.luders_strain]."""
    sy = np.asarray(yield_strength, dtype=float)
    if np.any(sy >= LUDERS_ESTIMATE_MAX_YIELD):
        raise ValueError(
            "Lueders strain estimate is only applicable for yield strength below 946 MPa; "
            "supply luders_strain explicitly"
        )
    return np.asarray(0.0375 * (1.0 - 0.001 * sy / _MPA), dtype=float)


@dataclass(frozen=True)
class Option1Curve:
    """Option 1 assessment line  [provenance: fad.option1.continuous / .discontinuous].

    Continuous yielding:
        f = (1 + Lr^2/2)^-1/2 [0.3 + 0.7 exp(-mu Lr^6)]        Lr <= 1
        f = f(1) Lr^((N-1)/(2N))                               1 < Lr < Lr,max
        f = 0                                                  Lr >= Lr,max
        mu = min(0.001 E/sy, 0.6),  N = 0.3 (1 - sy/su)

    Discontinuous yielding:
        f = (1 + Lr^2/2)^-1/2                                  Lr < 1
        f = (lambda + 1/(2 lambda))^-1/2                       Lr = 1
        f = f(1) Lr^((N-1)/(2N))                               1 < Lr < Lr,max
        f = 0                                                  Lr >= Lr,max
        lambda = 1 + E * luders_strain / sy

    ``yielding="unknown"`` follows the source rule for uncertainty (no plateau below
    Lr = 1, a plateau from Lr = 1), taken as the pointwise minimum of the two forms so
    that the curve is non-increasing for every input: the continuous form for Lr < 1,
    and the power law from the smaller of the two f(1) values for Lr >= 1.
    """

    yield_strength: ArrayLike
    uts: ArrayLike
    modulus: ArrayLike
    yielding: Yielding
    luders_strain: ArrayLike | None = None

    def __post_init__(self) -> None:
        sy = np.asarray(self.yield_strength, dtype=float)
        su = np.asarray(self.uts, dtype=float)
        e = np.asarray(self.modulus, dtype=float)
        if self.yielding not in ("continuous", "discontinuous", "unknown"):
            raise ValueError("yielding must be 'continuous', 'discontinuous' or 'unknown'")
        if np.any(~np.isfinite(sy)) or np.any(~np.isfinite(su)) or np.any(~np.isfinite(e)):
            raise ValueError("yield strength, tensile strength and modulus must be finite")
        if np.any(sy <= 0.0) or np.any(e <= 0.0):
            raise ValueError("yield strength and modulus must be positive")
        if np.any(su <= sy):
            raise ValueError("tensile strength must exceed yield strength")
        if self.luders_strain is not None:
            strain = np.asarray(self.luders_strain, dtype=float)
            if np.any(~np.isfinite(strain)) or np.any(strain <= 0.0):
                raise ValueError("luders_strain must be finite and positive")
        elif self.yielding != "continuous":
            luders_strain_estimate(sy)  # raises if the estimate is not applicable

    @property
    def lr_max(self) -> FloatArray:
        return lr_cutoff(self.yield_strength, self.uts)

    @property
    def step_locations(self) -> tuple[float, ...]:
        """Lr values at which the curve may drop discontinuously."""
        return () if self.yielding == "continuous" else (1.0,)

    @property
    def hardening_n(self) -> FloatArray:
        sy = np.asarray(self.yield_strength, dtype=float)
        su = np.asarray(self.uts, dtype=float)
        return np.asarray(0.3 * (1.0 - sy / su), dtype=float)

    @property
    def f_at_one(self) -> FloatArray:
        """Value of the curve at Lr = 1 (the value after the step, where there is one)."""
        sy = np.asarray(self.yield_strength, dtype=float)
        e = np.asarray(self.modulus, dtype=float)
        mu = np.minimum(0.001 * e / sy, 0.6)
        continuous = np.asarray(1.5**-0.5 * (0.3 + 0.7 * np.exp(-mu)), dtype=float)
        if self.yielding == "continuous":
            return continuous
        strain = (
            luders_strain_estimate(sy) if self.luders_strain is None else np.asarray(self.luders_strain, dtype=float)
        )
        lam = 1.0 + e * strain / sy
        plateau = np.asarray((lam + 0.5 / lam) ** -0.5, dtype=float)
        if self.yielding == "discontinuous":
            return plateau
        return np.asarray(np.minimum(continuous, plateau), dtype=float)

    def __call__(self, lr: ArrayLike) -> FloatArray:
        x = np.asarray(lr, dtype=float)
        if np.any(np.isnan(x)) or np.any(x < 0.0):
            raise ValueError("Lr must be non-negative and not NaN")
        sy = np.asarray(self.yield_strength, dtype=float)
        e = np.asarray(self.modulus, dtype=float)
        n = self.hardening_n
        f1 = self.f_at_one
        lr_max = self.lr_max

        below = np.minimum(x, 1.0)
        base = (1.0 + 0.5 * below**2) ** -0.5
        if self.yielding == "discontinuous":
            low = np.where(x < 1.0, base, f1)
        else:
            mu = np.minimum(0.001 * e / sy, 0.6)
            low = np.where(x < 1.0, base * (0.3 + 0.7 * np.exp(-mu * below**6)), f1)
        above = f1 * np.maximum(x, 1.0) ** ((n - 1.0) / (2.0 * n))
        f = np.where(x <= 1.0, low, above)
        return np.asarray(np.where(x >= lr_max, 0.0, f), dtype=float)
