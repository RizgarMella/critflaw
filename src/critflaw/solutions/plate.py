"""Stress intensity factor and reference stress solutions for flat plates.

Symbols: B plate thickness, W plate width, a flaw depth (surface flaw) or
half-length (through-thickness flaw), c surface-flaw half-length, theta the
parametric angle (0 at the surface, pi/2 at the deepest point).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

THETA_DEEPEST = np.pi / 2.0
THETA_SURFACE = 0.0
FW_SURFACE_MAX_2C_OVER_W = 0.8


def _arrays(*values: ArrayLike) -> list[FloatArray]:
    return [np.asarray(b, dtype=float) for b in np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in values))]


# --------------------------------------------------------------------------------------
# Surface flaw  [provenance: k.plate.surface]
# --------------------------------------------------------------------------------------


def surface_flaw_valid(a: ArrayLike, c: ArrayLike, thickness: ArrayLike, width: ArrayLike) -> BoolArray:
    """Validity of the surface-flaw solution.

    0 < a/2c <= 1;  a/B < 1.25 (a/c + 0.6) for a/2c <= 0.1, otherwise a/B < 1;
    finite-width factor applicable up to 2c/W = 0.8.
    """
    a_, c_, b_, w_ = _arrays(a, c, thickness, width)
    a_2c = a_ / (2.0 * c_)
    a_b = a_ / b_
    depth_ok = np.where(a_2c <= 0.1, a_b < 1.25 * (a_ / c_ + 0.6), a_b < 1.0)
    return np.asarray(
        (a_ > 0.0) & (c_ > 0.0) & (a_2c <= 1.0) & depth_ok & (2.0 * c_ / w_ <= FW_SURFACE_MAX_2C_OVER_W),
        dtype=bool,
    )


def surface_flaw_fw(a: ArrayLike, c: ArrayLike, thickness: ArrayLike, width: ArrayLike) -> FloatArray:
    """Finite width factor fw = {sec[(pi c / W) (a/B)^0.5]}^0.5."""
    a_, c_, b_, w_ = _arrays(a, c, thickness, width)
    return np.asarray(np.cos(np.pi * c_ / w_ * np.sqrt(a_ / b_)) ** -0.5, dtype=float)


def surface_flaw_mm_mb(
    a: ArrayLike, c: ArrayLike, thickness: ArrayLike, theta: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Membrane and bending magnification factors Mm, Mb for a surface flaw in a plate.

    Mm = [M1 + M2 (a/B)^2 + M3 (a/B)^4] g f_theta / Phi,   Mb = H Mm,
    H = H1 + (H2 - H1) sin^q(theta).
    Two coefficient sets: a/c <= 1 (a/2c <= 0.5) and a/c > 1 (0.5 < a/2c <= 1).
    """
    a_, c_, b_, th = _arrays(a, c, thickness, theta)
    ac = a_ / c_
    ab = a_ / b_
    shallow = ac <= 1.0
    # Evaluate each coefficient set on inputs clipped into its own range so that
    # the unused branch never produces overflow or invalid values.
    r = np.where(shallow, ac, 1.0)  # a/c in (0, 1]
    s = np.where(shallow, 1.0, 1.0 / ac)  # c/a in (0, 1)
    sin_t = np.sin(th)
    cos_t = np.cos(th)

    m1 = np.where(shallow, 1.13 - 0.09 * r, np.sqrt(s) * (1.0 + 0.04 * s))
    m2 = np.where(shallow, 0.89 / (0.2 + r) - 0.54, 0.2 * s**4)
    m3 = np.where(shallow, 0.5 - 1.0 / (0.65 + r) + 14.0 * (1.0 - r) ** 24, -0.11 * s**4)
    g = np.where(
        shallow,
        1.0 + (0.1 + 0.35 * ab**2) * (1.0 - sin_t) ** 2,
        1.0 + (0.1 + 0.35 * s * ab**2) * (1.0 - sin_t) ** 2,
    )
    f_theta = np.where(
        shallow,
        (r**2 * cos_t**2 + sin_t**2) ** 0.25,
        (s**2 * sin_t**2 + cos_t**2) ** 0.25,
    )
    phi = np.where(shallow, np.sqrt(1.0 + 1.464 * r**1.65), np.sqrt(1.0 + 1.464 * s**1.65))
    mm = (m1 + m2 * ab**2 + m3 * ab**4) * g * f_theta / phi

    q = np.where(shallow, 0.2 + r + 0.6 * ab, 0.2 + s + 0.6 * ab)
    h1 = np.where(
        shallow,
        1.0 - 0.34 * ab - 0.11 * r * ab,
        1.0 - (0.04 + 0.41 * s) * ab + (0.55 - 1.93 * s**0.75 + 1.38 * s**1.5) * ab**2,
    )
    g1 = np.where(shallow, -1.22 - 0.12 * r, -2.11 + 0.77 * s)
    g2 = np.where(
        shallow,
        0.55 - 1.05 * r**0.75 + 0.47 * r**1.5,
        0.55 - 0.72 * s**0.75 + 0.14 * s**1.5,
    )
    h2 = 1.0 + g1 * ab + g2 * ab**2
    h = h1 + (h2 - h1) * np.clip(sin_t, 0.0, 1.0) ** q
    return np.asarray(mm, dtype=float), np.asarray(h * mm, dtype=float)


def surface_flaw_reference_stress(
    pm: ArrayLike,
    pb: ArrayLike,
    a: ArrayLike,
    c: ArrayLike,
    thickness: ArrayLike,
    width: ArrayLike,
    pin_jointed: bool = False,
) -> FloatArray:
    """Reference stress for a surface flaw in a plate  [provenance: sref.plate.surface].

    Normal bending restraint:
        sref = {Pb + [Pb^2 + 9 Pm^2 (1 - al)^2]^0.5} / [3 (1 - al)^2]
    Negligible bending restraint (pin jointed):
        sref = {Pb + 3 Pm al + [(Pb + 3 Pm al)^2 + 9 Pm^2 (1 - al)^2]^0.5} / [3 (1 - al)^2]
    al = (a/B) / (1 + B/c)   for W >= 2 (c + B),   al = (a/B)(2c/W)   otherwise.
    """
    pm_, pb_, a_, c_, b_, w_ = _arrays(pm, pb, a, c, thickness, width)
    alpha = np.where(w_ >= 2.0 * (c_ + b_), (a_ / b_) / (1.0 + b_ / c_), (a_ / b_) * (2.0 * c_ / w_))
    bend = pb_ + 3.0 * pm_ * alpha if pin_jointed else pb_
    sref = (bend + np.sqrt(bend**2 + 9.0 * pm_**2 * (1.0 - alpha) ** 2)) / (3.0 * (1.0 - alpha) ** 2)
    return np.asarray(sref, dtype=float)


# --------------------------------------------------------------------------------------
# Through-thickness flaw  [provenance: k.plate.through, sref.plate.through]
# --------------------------------------------------------------------------------------


def through_flaw_valid(a: ArrayLike, width: ArrayLike) -> BoolArray:
    a_, w_ = _arrays(a, width)
    return np.asarray((a_ > 0.0) & (2.0 * a_ < w_), dtype=bool)


def through_flaw_fw(a: ArrayLike, width: ArrayLike) -> FloatArray:
    """Finite width factor fw = [sec(pi a / W)]^0.5 with M = Mm = Mb = 1."""
    a_, w_ = _arrays(a, width)
    return np.asarray(np.cos(np.pi * a_ / w_) ** -0.5, dtype=float)


def through_flaw_reference_stress(pm: ArrayLike, pb: ArrayLike, a: ArrayLike, width: ArrayLike) -> FloatArray:
    """sref = [Pb + (Pb^2 + 9 Pm^2)^0.5] / {3 [1 - (2a/W)]}."""
    pm_, pb_, a_, w_ = _arrays(pm, pb, a, width)
    return np.asarray((pb_ + np.sqrt(pb_**2 + 9.0 * pm_**2)) / (3.0 * (1.0 - 2.0 * a_ / w_)), dtype=float)


# --------------------------------------------------------------------------------------
# General form  [provenance: k.general]
# --------------------------------------------------------------------------------------


def stress_intensity(
    crack_size: ArrayLike,
    mm: ArrayLike,
    mb: ArrayLike,
    fw: ArrayLike,
    pm: ArrayLike,
    pb: ArrayLike,
    qm: ArrayLike,
    qb: ArrayLike,
    bulging: ArrayLike = 1.0,
    ktm: ArrayLike = 1.0,
    ktb: ArrayLike = 1.0,
    km: ArrayLike = 1.0,
    mkm: ArrayLike = 1.0,
    mkb: ArrayLike = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """Primary and secondary stress intensity factors, K = (Y sigma) (pi a)^0.5.

    (Y sigma)_p = M fw {ktm Mkm Mm Pm + ktb Mkb Mb [Pb + (km - 1) Pm]}
    (Y sigma)_s = Mm Qm + Mb Qb
    """
    a_ = np.asarray(crack_size, dtype=float)
    mm_, mb_, fw_ = (np.asarray(v, dtype=float) for v in (mm, mb, fw))
    pm_, pb_, qm_, qb_ = (np.asarray(v, dtype=float) for v in (pm, pb, qm, qb))
    m_, ktm_, ktb_, km_, mkm_, mkb_ = (np.asarray(v, dtype=float) for v in (bulging, ktm, ktb, km, mkm, mkb))
    ys_p = m_ * fw_ * (ktm_ * mkm_ * mm_ * pm_ + ktb_ * mkb_ * mb_ * (pb_ + (km_ - 1.0) * pm_))
    ys_s = mm_ * qm_ + mb_ * qb_
    root = np.sqrt(np.pi * a_)
    return np.asarray(ys_p * root, dtype=float), np.asarray(ys_s * root, dtype=float)
