"""Stress intensity factor and reference stress solutions for embedded elliptical flaws in flat plates
(theory manual, section 2.3.3)  [provenance: k.plate.embedded, sref.plate.embedded].

Symbols: 2a flaw height (through-thickness), 2c flaw length, p ligament to the nearer surface,
B' = 2a + 2p the effective thickness, B plate thickness, W plate width, theta the parametric angle
(0 at the ends of the major axis, pi/2 at the minor-axis end nearest the surface).

Bending (M_b) is given by the source at the minor-axis ends only; the assessment of a bending-loaded
embedded flaw is refused by the model layer. The functions below evaluate M_b where
it is defined so that the path is tested and ready.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from critflaw.solutions.plate import BoolArray, FloatArray, _arrays

THETA_NEAR_END = np.pi / 2.0  # minor-axis end nearest the surface
THETA_MAJOR_AXIS = 0.0
WIDTH_LIMIT_2C_OVER_W = 0.5  # the stricter of the two printed limits (a deliberate choice)
FW_NOTE_2C_OVER_W = 0.8  # the finite-width note's own limit, recorded
MB_THRESHOLD_P_OVER_B = 0.1841
MB_THRESHOLD_A_OVER_B = 0.125


def effective_thickness(a: ArrayLike, p: ArrayLike) -> FloatArray:
    """B' = 2a + 2p."""
    a_, p_ = _arrays(a, p)
    return np.asarray(2.0 * a_ + 2.0 * p_, dtype=float)


def embedded_flaw_valid(a: ArrayLike, c: ArrayLike, p: ArrayLike, thickness: ArrayLike, width: ArrayLike) -> BoolArray:
    """Validity of the embedded-flaw membrane solution and geometry (theory manual, section 2.3.3).

    p > 0;  2a + 2p <= B (p is the nearer ligament);  0 < a/2c <= 1;
    a/B' < 0.625 (a/c + 0.6) for a/2c <= 0.1 (no depth limit is printed for a/2c > 0.1);
    2c/W < 0.5.
    """
    a_, c_, p_, b_, w_ = _arrays(a, c, p, thickness, width)
    b_eff = 2.0 * a_ + 2.0 * p_
    a_2c = a_ / (2.0 * c_)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth_ok = np.where(a_2c <= 0.1, a_ / b_eff < 0.625 * (a_ / c_ + 0.6), True)
    return np.asarray(
        (a_ > 0.0)
        & (c_ > 0.0)
        & (p_ > 0.0)
        & (b_eff <= b_)
        & (a_2c <= 1.0)
        & depth_ok
        & (2.0 * c_ / w_ < WIDTH_LIMIT_2C_OVER_W),
        dtype=bool,
    )


def embedded_flaw_fw(a: ArrayLike, c: ArrayLike, p: ArrayLike, width: ArrayLike) -> FloatArray:
    """Finite width factor fw = {sec[(pi c / W) (2a/B')^0.5]}^0.5."""
    a_, c_, p_, w_ = _arrays(a, c, p, width)
    b_eff = 2.0 * a_ + 2.0 * p_
    return np.asarray(np.cos(np.pi * c_ / w_ * np.sqrt(2.0 * a_ / b_eff)) ** -0.5, dtype=float)


def embedded_flaw_mm(a: ArrayLike, c: ArrayLike, p: ArrayLike, theta: ArrayLike) -> FloatArray:
    """Membrane magnification factor M_m of an embedded flaw.

    Mm = [M1 + M2 (2a/B')^2 + M3 (2a/B')^4] g f_theta / Phi
    M1 = 1 (a/2c <= 0.5), (c/a)^0.5 (a/2c > 0.5);  M2 = 0.05 / (0.11 + (a/c)^1.5);  M3 = 0.29 / (0.23 + (a/c)^1.5)
    g = 1 - {(2a/B')^4 [2.6 - 4a/B']^0.5 / (1 + 4 a/c)} |cos theta|
    f_theta and Phi with the a/c forms for a/2c <= 0.5 and the c/a forms for a/2c > 0.5 (two branches at a/c = 1,
    as for the surface flaw).
    """
    a_, c_, p_, th = _arrays(a, c, p, theta)
    b_eff = 2.0 * a_ + 2.0 * p_
    ac = a_ / c_
    x = 2.0 * a_ / b_eff  # 2a/B'
    shallow = ac <= 1.0
    r = np.where(shallow, ac, 1.0)  # a/c on its branch
    s = np.where(shallow, 1.0, 1.0 / ac)  # c/a on its branch
    sin_t, cos_t = np.sin(th), np.cos(th)
    m1 = np.where(shallow, 1.0, np.sqrt(s))
    m2 = 0.05 / (0.11 + ac**1.5)
    m3 = 0.29 / (0.23 + ac**1.5)
    g = 1.0 - (x**4 * np.sqrt(np.maximum(2.6 - 2.0 * x, 0.0)) / (1.0 + 4.0 * ac)) * np.abs(cos_t)  # 4a/B' = 2x
    f_theta = np.where(shallow, (r**2 * cos_t**2 + sin_t**2) ** 0.25, (s**2 * sin_t**2 + cos_t**2) ** 0.25)
    phi = np.where(shallow, np.sqrt(1.0 + 1.464 * r**1.65), np.sqrt(1.0 + 1.464 * s**1.65))
    return np.asarray((m1 + m2 * x**2 + m3 * x**4) * g * f_theta / phi, dtype=float)


def embedded_flaw_mb_valid(a: ArrayLike, c: ArrayLike) -> BoolArray:
    """The bending solution is given for 0 < a/2c <= 0.5 at the minor-axis ends only."""
    a_, c_ = _arrays(a, c)
    return np.asarray((a_ > 0.0) & (c_ > 0.0) & (a_ / (2.0 * c_) <= 0.5), dtype=bool)


def embedded_flaw_mb(a: ArrayLike, c: ArrayLike, p: ArrayLike, thickness: ArrayLike) -> FloatArray:
    """Bending magnification factor M_b at the minor-axis ends (full thickness B, not B').

    Mb = [l1 + l2 (p/B) + l3 (a/B) + l4 (p a / B^2)] / Phi with three coefficient sets:
    p/B <= 0.1841; p/B > 0.1841 and a/B <= 0.125; p/B > 0.1841 and a/B > 0.125.
    """
    a_, c_, p_, b_ = _arrays(a, c, p, thickness)
    pb = p_ / b_
    ab = a_ / b_
    ac = a_ / c_
    first = pb <= MB_THRESHOLD_P_OVER_B
    second = ~first & (ab <= MB_THRESHOLD_A_OVER_B)
    l1 = np.where(first, 1.044, np.where(second, 0.94, 1.06))
    l2 = np.where(first, -2.44, np.where(second, -1.875, -2.20))
    l3 = np.where(first, 0.0, np.where(second, -0.1146, -0.6666))
    l4 = np.where(first, -3.166, np.where(second, -1.844, -0.6666))
    phi = np.sqrt(1.0 + 1.464 * np.minimum(ac, 1.0) ** 1.65)  # a/2c <= 0.5 on the valid range
    return np.asarray((l1 + l2 * pb + l3 * ab + l4 * pb * ab) / phi, dtype=float)


def embedded_flaw_reference_stress(
    pm: ArrayLike, pb: ArrayLike, a: ArrayLike, c: ArrayLike, p: ArrayLike, thickness: ArrayLike, width: ArrayLike
) -> FloatArray:
    """Reference stress of an embedded flaw under combined tension and bending  [provenance: sref.plate.embedded].

    sref = {Pb + 3 Pm al + [(Pb + 3 Pm al)^2 + 9 Pm^2 ((1 - al)^2 + 4 p al / B)]^0.5} / {3 [(1 - al)^2 + 4 p al / B]}
    al = (2a/B) / (1 + B/c)   for W >= 2 (c + B),   al = (2a/B)(2c/W)   otherwise.
    """
    pm_, pb_, a_, c_, p_, b_, w_ = _arrays(pm, pb, a, c, p, thickness, width)
    alpha = np.where(w_ >= 2.0 * (c_ + b_), (2.0 * a_ / b_) / (1.0 + b_ / c_), (2.0 * a_ / b_) * (2.0 * c_ / w_))
    denom = (1.0 - alpha) ** 2 + 4.0 * p_ * alpha / b_
    bend = pb_ + 3.0 * pm_ * alpha
    return np.asarray((bend + np.sqrt(bend**2 + 9.0 * pm_**2 * denom)) / (3.0 * denom), dtype=float)


def extended_embedded_reference_stress(
    pm: ArrayLike, pb: ArrayLike, a: ArrayLike, p: ArrayLike, thickness: ArrayLike
) -> FloatArray:
    """Reference stress of an extended embedded flaw (full width; no K solution exists for it)
    [provenance: sref.plate.embedded_extended].

    sref = {[2 (lam + al) + (4 (lam + al)^2 + c3)^0.5] / c3} Pm   for al <= al2,
    lam = Pb / (6 Pm), al = a/B, k = (B/2 - p - a)/B, al2 = k/2, c3 = 1 - 8 al k - 4 al^2;  NaN beyond al2.
    """
    pm_, pb_, a_, p_, b_ = _arrays(pm, pb, a, p, thickness)
    alpha = a_ / b_
    k = (0.5 * b_ - p_ - a_) / b_
    alpha2 = 0.5 * k
    # the printed form has Pm in a denominator; the equivalent form with t = Pb/3 + 2 Pm al is stable at Pm = 0
    c3 = 1.0 - 8.0 * alpha * k - 4.0 * alpha**2
    t_ = pb_ / 3.0 + 2.0 * pm_ * alpha
    with np.errstate(divide="ignore", invalid="ignore"):
        sref = (t_ + np.sqrt(t_**2 + pm_**2 * c3)) / c3
    return np.asarray(np.where((alpha <= alpha2) & (c3 > 0.0), sref, np.nan), dtype=float)
