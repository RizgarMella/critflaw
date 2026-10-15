"""Fracture ratio Kr with plasticity interaction (theory manual, section 1.5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

CHI_SIMPLIFIED_LIMIT = 4.0  # above this the simplified route is over-conservative


def rho_simplified(chi: ArrayLike, lr: ArrayLike) -> FloatArray:
    """Additive plasticity interaction term rho  [provenance: fad.rho.simplified].

    rho1 = 0.1 chi^0.714 - 0.007 chi^2 + 0.00003 chi^5   (chi < 5.2),  0.25 otherwise
    rho  = rho1                    Lr <= 0.8
         = 4 rho1 (1.05 - Lr)      0.8 < Lr < 1.05
         = 0                       Lr >= 1.05
    """
    x = np.asarray(chi, dtype=float)
    load = np.asarray(lr, dtype=float)
    xc = np.minimum(x, 5.2)
    rho1 = np.where(x < 5.2, 0.1 * xc**0.714 - 0.007 * xc**2 + 0.00003 * xc**5, 0.25)
    rho = np.where(load <= 0.8, rho1, np.where(load < 1.05, 4.0 * rho1 * (1.05 - load), 0.0))
    return np.asarray(rho, dtype=float)


def v_simplified(chi: ArrayLike, lr: ArrayLike) -> FloatArray:
    """Multiplying plasticity interaction factor V  [provenance: fad.v.simplified].

    V = min(1 + 0.2 Lr + 0.02 chi (1 + 2 Lr), 3.1 - 2 Lr)   Lr < 1.05,   1 otherwise
    """
    x = np.asarray(chi, dtype=float)
    load = np.asarray(lr, dtype=float)
    with np.errstate(invalid="ignore"):
        v = np.fmin(1.0 + 0.2 * load + 0.02 * x * (1.0 + 2.0 * load), 3.1 - 2.0 * load)
    return np.asarray(np.where(load < 1.05, v, 1.0), dtype=float)


def kj_secondary(
    k_secondary: ArrayLike, crack_size: ArrayLike, yield_strength: ArrayLike, plane_stress: bool = True
) -> FloatArray:
    """Plastic-zone-corrected secondary K  [provenance: fad.kjs.plastic_zone].

    a_eff = a + (1 / (2 pi beta)) (Ks / sy)^2,   KJs = (a_eff / a)^0.5 Ks
    beta = 1 for plane stress, 3 for plane strain.
    """
    ks = np.asarray(k_secondary, dtype=float)
    a = np.asarray(crack_size, dtype=float)
    sy = np.asarray(yield_strength, dtype=float)
    beta = 1.0 if plane_stress else 3.0
    a_eff = a + (ks / sy) ** 2 / (2.0 * np.pi * beta)
    return np.asarray(np.sqrt(a_eff / a) * ks, dtype=float)


@dataclass(frozen=True)
class FractureRatio:
    kr: FloatArray
    chi: FloatArray  # Ks * Lr / Kp; NaN where undefined (no primary load)
    interaction: FloatArray  # rho or V actually applied (0 or 1 where not applicable)
    secondary_only: BoolArray
    chi_above_limit: BoolArray  # simplified route used outside its preferred range


def fracture_ratio(
    k_primary: ArrayLike,
    k_secondary: ArrayLike,
    lr: ArrayLike,
    k_mat: ArrayLike,
    crack_size: ArrayLike,
    yield_strength: ArrayLike,
    method: Literal["rho", "V"] = "rho",
    plane_stress: bool = True,
    k_self_balancing: ArrayLike = 0.0,
) -> FractureRatio:
    """Kr by the decision table of theory manual section 1.5  [provenance: fad.kr].

    rho route:  Kr = (Kp + Ks) / Kmat + rho
    V route:    Kr = (Kp + V Ks) / Kmat
    A negative Ks is set to zero. With secondary stress acting alone
    Kr = KJs / Kmat. The self-balancing term is added outside the interaction.
    """
    if method not in ("rho", "V"):
        raise ValueError("method must be 'rho' or 'V'")
    kp, ks, load, kmat, a, sy, ksb = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (k_primary, k_secondary, lr, k_mat, crack_size,
                                                yield_strength, k_self_balancing))
    )
    for name, arr in (("k_primary", kp), ("k_secondary", ks), ("lr", load), ("k_mat", kmat), ("k_self_balancing", ksb)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be finite")
    if np.any(kp < 0.0) or np.any(load < 0.0):
        raise ValueError("primary stress intensity factor and Lr must be non-negative")
    if np.any(kmat <= 0.0):
        raise ValueError("fracture toughness must be positive")
    if np.any(ksb < 0.0):
        # Where the self-balancing K is negative it is conservative to take it as zero; the
        # caller must do that explicitly so that Kr >= 0 always holds.
        raise ValueError("k_self_balancing must be non-negative")

    ks = np.maximum(ks, 0.0)
    has_secondary = ks > 0.0
    # "No primary load" means Lr = 0 and Kp = 0. A primary load that gives Kp = 0 at this
    # crack-front position (Lr > 0) is not the secondary-only case: chi is then unbounded
    # and the interaction takes its limiting value (rho1 = 0.25, V = 3.1 - 2 Lr).
    has_primary = (kp > 0.0) | (load > 0.0)
    both = has_secondary & has_primary
    secondary_only = has_secondary & ~has_primary

    chi = np.full(kp.shape, np.nan)
    np.divide(ks * load, kp, out=chi, where=kp > 0.0)
    chi = np.where(has_primary & (kp == 0.0), np.where(has_secondary, np.inf, 0.0), chi)

    chi_safe = np.where(both, chi, 0.0)
    if method == "rho":
        interaction = np.where(both, rho_simplified(chi_safe, load), 0.0)
        kr = (kp + ks) / kmat + interaction
    else:
        interaction = np.where(both, v_simplified(chi_safe, load), 1.0)
        kr = (kp + interaction * ks) / kmat

    kjs = kj_secondary(ks, a, sy, plane_stress=plane_stress)
    kr = np.where(secondary_only, kjs / kmat, kr)
    kr = kr + ksb / kmat

    return FractureRatio(
        kr=np.asarray(kr, dtype=float),
        chi=chi,
        interaction=np.asarray(interaction, dtype=float),
        secondary_only=secondary_only,
        chi_above_limit=both & (chi_safe > CHI_SIMPLIFIED_LIMIT),
    )
