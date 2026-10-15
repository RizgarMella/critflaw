"""Reference stress and limit load solutions for flaws in cylinders (theory manual, section 2.5.2)
[provenance: sref.cylinder.axial_thin, sref.cylinder.axial_pressure, sref.cylinder.axial_thick_bending,
sref.cylinder.circumferential_thin, sref.cylinder.circumferential_thick].

Symbols: a flaw depth (part-wall) or half-length (through-wall), c half-length, B wall, r_i, r_o,
r_m = r_i + B/2, W cylinder length, P_m membrane and P_b through-wall bending stress, sigma_Y yield strength.
The source's internal inconsistencies are resolved by one implemented reading each, recorded in
``docs/unsourced_items.md`` (the external-flaw radius substitution is read literally: the outer radius
replaces the inner one in M_T).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from critflaw.solutions.plate import BoolArray, FloatArray, _arrays

# ---- axial flaws, thin wall, membrane and through-wall bending (P.9.1 to P.9.5) ----------------------


def axial_through_thin(
    pm: ArrayLike, pb: ArrayLike, a: ArrayLike, r_i: ArrayLike, thickness: ArrayLike, width: ArrayLike
) -> FloatArray:
    """sref = 1.2 M_T Pm + 2 Pb / [3 (1 - 2a/W)],  M_T = [1 + 1.6 a^2 / (r_i B)]^0.5 (half-length a)."""
    pm_, pb_, a_, r_, b_, w_ = _arrays(pm, pb, a, r_i, thickness, width)
    m_t = np.sqrt(1.0 + 1.6 * a_**2 / (r_ * b_))
    return np.asarray(1.2 * m_t * pm_ + 2.0 * pb_ / (3.0 * (1.0 - 2.0 * a_ / w_)), dtype=float)


def _alpha_axial(a: FloatArray, c: FloatArray, b: FloatArray, w: FloatArray) -> FloatArray:
    return np.asarray(np.where(w >= 2.0 * (c + b), (a / b) / (1.0 + b / c), (a / b) * (2.0 * c / w)), dtype=float)


def axial_surface_thin(
    pm: ArrayLike, pb: ArrayLike, a: ArrayLike, c: ArrayLike, radius: ArrayLike, thickness: ArrayLike, width: ArrayLike
) -> FloatArray:
    """Axial surface flaw: sref = 1.2 M_s Pm + 2 Pb / [3 (1 - al)^2],
    M_s = [1 - a/(B M_T)] / [1 - a/B],  M_T = [1 + 1.6 c^2 / (radius B)]^0.5,
    al = (a/B)/(1 + B/c) for W >= 2 (c + B), (a/B)(2c/W) otherwise.
    ``radius`` is r_i for an internal flaw and r_o for an external one (the source's substitution, read literally)."""
    pm_, pb_, a_, c_, r_, b_, w_ = _arrays(pm, pb, a, c, radius, thickness, width)
    m_t = np.sqrt(1.0 + 1.6 * c_**2 / (r_ * b_))
    m_s = (1.0 - a_ / (b_ * m_t)) / (1.0 - a_ / b_)
    alpha = _alpha_axial(a_, c_, b_, w_)
    return np.asarray(1.2 * m_s * pm_ + 2.0 * pb_ / (3.0 * (1.0 - alpha) ** 2), dtype=float)


def axial_extended_thin(
    pm: ArrayLike,
    pb: ArrayLike,
    a: ArrayLike,
    radius: ArrayLike,
    thickness: ArrayLike,
    width: ArrayLike,
    external: bool,
) -> FloatArray:
    """Extended axial flaw: internal: the surface form with al = a/B and c = W/2 (in M_T, with r_i);
    external: M_s = 1/(1 - a/B), al = a/B (radius unused)."""
    pm_, pb_, a_, r_, b_, w_ = _arrays(pm, pb, a, radius, thickness, width)
    alpha = a_ / b_
    if external:
        m_s = 1.0 / (1.0 - alpha)
    else:
        m_t = np.sqrt(1.0 + 1.6 * (0.5 * w_) ** 2 / (r_ * b_))
        m_s = (1.0 - a_ / (b_ * m_t)) / (1.0 - alpha)
    return np.asarray(1.2 * m_s * pm_ + 2.0 * pb_ / (3.0 * (1.0 - alpha) ** 2), dtype=float)


# ---- axial flaws, thick wall under internal pressure: limit pressures (P.9.6, P.9.7) ----------------


def axial_internal_limit_pressure(
    a: ArrayLike, c: ArrayLike, r_i: ArrayLike, thickness: ArrayLike, sigma_y: ArrayLike, extended: bool = False
) -> FloatArray:
    """p_L for an internal axial surface flaw in a thick wall (defect-face pressure included):
    p_L/sY = [1/(h1c + 1)] [h1c ln(1 + eta) + f_ps (1 + al eta) ln((1 + eta)/(1 + al eta))],  eta = B/r_i, al = a/B,
    Phi = a/c, M_in = [1 + 1.4 al eta / (Phi^2 (1 + al eta))]^0.5, f_ps = 1/(1 + al eta/2) (Phi <= 1) or
    Phi/(Phi + al eta/2) (Phi > 1), h1c as printed. Extended: Phi = 0 and 1/M_in = 0 (the limit h1c -> 0)."""
    a_, c_, r_, b_, sy = _arrays(a, c, r_i, thickness, sigma_y)
    alpha, eta = a_ / b_, b_ / r_
    ln1 = np.log(1.0 + eta)
    ln_ratio = np.log((1.0 + eta) / (1.0 + alpha * eta))
    if extended:
        f_ps = 1.0 / (1.0 + 0.5 * alpha * eta)
        h1c = np.zeros_like(alpha)
    else:
        phi = a_ / c_
        m_in = np.sqrt(1.0 + 1.4 * alpha * eta / (phi**2 * (1.0 + alpha * eta)))
        f_ps = np.where(phi <= 1.0, 1.0 / (1.0 + 0.5 * alpha * eta), phi / (phi + 0.5 * alpha * eta))
        ln2 = np.log(1.0 + alpha * eta)
        num = (1.0 - alpha) * f_ps * (ln2 - alpha * eta * ln_ratio)
        den = m_in * (ln1 - f_ps * ((1.0 / m_in) * ln2 + (1.0 + alpha * eta * (1.0 - 1.0 / m_in)) * ln_ratio))
        h1c = num / den
    ratio = (1.0 / (h1c + 1.0)) * (h1c * ln1 + f_ps * (1.0 + alpha * eta) * ln_ratio)
    return np.asarray(ratio * sy, dtype=float)


def axial_external_limit_pressure(
    a: ArrayLike, c: ArrayLike, r_i: ArrayLike, thickness: ArrayLike, sigma_y: ArrayLike, extended: bool = False
) -> FloatArray:
    """p_L for an external axial surface flaw in a thick wall:
    p_L/sY = [1/(h2c + 1)] [h2c ln(1 + eta) + ln(1 + eta - al eta)],  h2c = (1 - al)/(M_ex - 1),
    M_ex = [1 + 1.4 al eta / ((1 + eta) Phi^2)]^0.5. Extended: 1/M_ex = 0, h2c = 0."""
    a_, c_, r_, b_, sy = _arrays(a, c, r_i, thickness, sigma_y)
    alpha, eta = a_ / b_, b_ / r_
    if extended:
        h2c = np.zeros_like(alpha)
    else:
        phi = a_ / c_
        m_ex = np.sqrt(1.0 + 1.4 * alpha * eta / ((1.0 + eta) * phi**2))
        h2c = (1.0 - alpha) / (m_ex - 1.0)
    ratio = (1.0 / (h2c + 1.0)) * (h2c * np.log(1.0 + eta) + np.log(1.0 + eta - alpha * eta))
    return np.asarray(ratio * sy, dtype=float)


# ---- external axial flaw, thick wall, membrane and bending: normalised limit stresses (P.9.8) -------


def axial_external_thick_limits(
    pm: ArrayLike, pb: ArrayLike, a: ArrayLike, c: ArrayLike, thickness: ArrayLike, width: ArrayLike
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """(L_N, L_b, valid): normalised limit force and moment of an external axial surface flaw in a thick wall,
    L_N = sigma_limit,m / sY, L_b = (2/3) sigma_limit,b / sY, with lam = Pb/(6 Pm), al = a/B, psi = c/B,
    beta = 2c/W and validity beta <= psi/(1 + psi); psi = 0 is the extended flaw. Pure bending (Pm = 0) is the
    limit lam -> inf: al0 = (psi + 1)/(psi + 2) and L_N -> 0 (returned as 0)."""
    pm_, pb_, a_, c_, b_, w_ = _arrays(pm, pb, a, c, thickness, width)
    alpha, psi, beta = a_ / b_, c_ / b_, 2.0 * c_ / w_
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = pb_ / (6.0 * pm_)
    pure_bending = pm_ == 0.0
    lam_f = np.where(pure_bending, 0.0, lam)
    alpha0 = np.where(
        pure_bending,
        (psi + 1.0) / (psi + 2.0),
        -(lam_f - 0.5) + np.sqrt((lam_f - 0.5) ** 2 + 2.0 * (1.0 + psi) * lam_f / (2.0 + psi)),
    )
    x = alpha * psi / (1.0 + psi)
    d1 = (1.0 - x) ** 2 + 2.0 * alpha**2 * psi / (1.0 + psi) ** 2
    d2 = (1.0 - x) * (1.0 - psi * (1.0 - alpha)) + 2.0 * alpha * (1.0 - alpha) * psi / (1.0 + psi)
    low = alpha <= alpha0
    den1 = 2.0 * lam_f + x + np.sqrt((2.0 * lam_f + x) ** 2 + d1)
    den2 = 2.0 * lam_f + psi * (1.0 - alpha) + np.sqrt((2.0 * lam_f + psi * (1.0 - alpha)) ** 2 + (1.0 + psi) * d2)
    l_n = np.where(low, d1 / den1, d2 / den2)
    l_b = np.where(low, 4.0 * lam_f * d1 / den1, 4.0 * lam_f * d2 / den2)
    # pure bending: L_b = lim 4 lam d / (2 lam + ... + sqrt((2 lam + ...)^2 + ...)) = d
    l_b = np.where(pure_bending, np.where(low, d1, d2), l_b)
    l_n = np.where(pure_bending, 0.0, l_n)
    valid = beta <= psi / (1.0 + psi) + 1e-15
    return np.asarray(l_n, dtype=float), np.asarray(l_b, dtype=float), np.asarray(valid, dtype=bool)


# ---- circumferential flaws, thin wall (P.10.1 to P.10.5) ----------------------------------------------


def circumferential_through_thin(
    pm_axial: ArrayLike,
    pm_pressure: ArrayLike,
    pm_moment: ArrayLike,
    pb_local: ArrayLike,
    a: ArrayLike,
    r_i: ArrayLike,
    thickness: ArrayLike,
) -> FloatArray:
    """Through-wall circumferential flaw of half-length a (at mid-thickness):
    sref = pi (Pma + Pmp) / [pi - a/r_i - 2 arcsin(sin(a/r_i)/2)]
         + pi Pmb (r_o^4 - r_i^4) / {[pi - a/r_i - 2 sin^2(a/r_i)/(pi - a/r_i) - sin(2a/r_i)/2] 4 r_o r_m^2 B}
         + 2 Pbl / [3 (1 - 2a/(pi r_i))]."""
    pma, pmp, pmb, pbl, a_, r_, b_ = _arrays(pm_axial, pm_pressure, pm_moment, pb_local, a, r_i, thickness)
    x = a_ / r_
    r_o = r_ + b_
    r_m = r_ + 0.5 * b_
    term1 = np.pi * (pma + pmp) / (np.pi - x - 2.0 * np.arcsin(0.5 * np.sin(x)))
    bracket = np.pi - x - 2.0 * np.sin(x) ** 2 / (np.pi - x) - 0.5 * np.sin(2.0 * x)
    term2 = np.pi * pmb * (r_o**4 - r_**4) / (bracket * 4.0 * r_o * r_m**2 * b_)
    term3 = 2.0 * pbl / (3.0 * (1.0 - 2.0 * a_ / (np.pi * r_)))
    return np.asarray(term1 + term2 + term3, dtype=float)


def circumferential_through_thin_valid(a: ArrayLike, r_i: ArrayLike, thickness: ArrayLike) -> BoolArray:
    a_, r_, b_ = _arrays(a, r_i, thickness)
    return np.asarray((a_ >= 0.0) & (2.0 * a_ / (np.pi * r_) <= 0.5) & (b_ / r_ > 0.0) & (b_ / r_ <= 0.2), dtype=bool)


def circumferential_surface_thin(
    pm: ArrayLike, pb: ArrayLike, a: ArrayLike, c: ArrayLike, radius: ArrayLike, thickness: ArrayLike
) -> FloatArray:
    """Circumferential surface flaw (internal: radius = r_i; external: r_o substituted):
    sref = Pm [pi (1 - a/B) + 2 (a/B) sin(c/r)] / {(1 - a/B) [pi - (c/r)(a/B)]} + 2 Pb / [3 (1 - al)^2],
    al = (a/B)/(1 + B/c) for pi r >= c + B, (a/B)(c/(pi r)) otherwise. Fully circumferential: c = pi r, al = a/B."""
    pm_, pb_, a_, c_, r_, b_ = _arrays(pm, pb, a, c, radius, thickness)
    ab = a_ / b_
    alpha = np.where(np.pi * r_ >= c_ + b_, ab / (1.0 + b_ / c_), ab * (c_ / (np.pi * r_)))
    membrane = pm_ * (np.pi * (1.0 - ab) + 2.0 * ab * np.sin(c_ / r_)) / ((1.0 - ab) * (np.pi - (c_ / r_) * ab))
    return np.asarray(membrane + 2.0 * pb_ / (3.0 * (1.0 - alpha) ** 2), dtype=float)


def circumferential_fully_thin(
    pm: ArrayLike, pb: ArrayLike, a: ArrayLike, radius: ArrayLike, thickness: ArrayLike
) -> FloatArray:
    """Fully circumferential surface flaw: the surface form with c = pi r and al = a/B."""
    pm_, pb_, a_, r_, b_ = _arrays(pm, pb, a, radius, thickness)
    ab = a_ / b_
    membrane = pm_ / (1.0 - ab)  # sin(pi) = 0 and pi - pi ab = pi (1 - ab)
    return np.asarray(membrane + 2.0 * pb_ / (3.0 * (1.0 - ab) ** 2), dtype=float)


def circumferential_surface_thin_valid(c: ArrayLike, radius: ArrayLike) -> BoolArray:
    c_, r_ = _arrays(c, radius)
    return np.asarray((c_ > 0.0) & (c_ / (np.pi * r_) <= 0.8), dtype=bool)


# ---- circumferential flaws, thick wall, tension and global bending: net-section limit load (P.11) ---


def _f_functions(alpha: FloatArray, eta: FloatArray, external: bool) -> tuple[FloatArray, ...]:
    s = -1.0 if external else 1.0  # sign of the eta-odd terms (internal: +, external: -)
    f_b = 1.0 + eta**2 / 12.0
    f_a = 1.0 - s * 0.5 * eta + s * 0.5 * alpha * eta
    f_c = 1.0 - s * eta + eta**2 / 4.0 + s * alpha * eta - 0.5 * alpha * eta**2 + alpha**2 * eta**2 / 3.0
    f_d = (1.0 - alpha) * (1.0 + s * alpha * eta - alpha * eta**2 / 6.0 + alpha**2 * eta**2 / 3.0 + eta**2 / 12.0) / f_b
    f_e = 1.0 - alpha + s * 0.5 * alpha * eta - s * 0.5 * alpha**2 * eta
    return f_a, f_b, f_c, f_d, f_e


def circumferential_thick_limit(
    a: ArrayLike,
    theta: ArrayLike,
    r_m: ArrayLike,
    thickness: ArrayLike,
    load_ratio: ArrayLike,
    external: bool = False,
    iterations: int = 100,
) -> tuple[FloatArray, FloatArray, FloatArray, BoolArray]:
    """(L_N, L_b, beta, branch_ii) for a circumferential flaw of depth a and half-angle theta in a thick wall
    under tension and global bending with lam = L_b / ((pi/2) L_N) given (``load_ratio``; NaN or inf for pure
    bending, 0 for pure tension). Solves the source's two equation systems (branch (i) theta + beta <= pi,
    (ii) otherwise) for the neutral-axis half-angle beta in [0, pi/2] by bisection. Through-wall: a = B;
    fully circumferential: theta = pi. Dimensional limits: F = 2 pi r_m B sY L_N, M = 4 r_m^2 B sY L_b."""
    a_, th, r_, b_, lam = _arrays(a, theta, r_m, thickness, load_ratio)
    alpha, eta = a_ / b_, b_ / r_
    f_a, f_b, f_c, f_d, f_e = _f_functions(alpha, eta, external)
    pure_bending = ~np.isfinite(lam)
    pure_tension = lam == 0.0
    k = np.where(pure_bending, 0.0, (np.pi / 2.0) * np.where(np.isfinite(lam), lam, 0.0))  # L_b = k L_N

    def residual(beta: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, BoolArray]:
        branch_ii = th + beta > np.pi
        l_n_i = 1.0 - f_a * alpha * th / np.pi - 2.0 * beta / np.pi
        l_b_i = f_b * np.sin(beta) - 0.5 * alpha * f_c * np.sin(th)
        l_n_ii = 2.0 * f_e * (1.0 - beta / np.pi) - 1.0 + (1.0 - f_e) * th / np.pi
        l_b_ii = f_b * (f_d * np.sin(beta) + 0.5 * (1.0 - f_d) * np.sin(th))
        l_n = np.where(branch_ii, l_n_ii, l_n_i)
        l_b = np.where(branch_ii, l_b_ii, l_b_i)
        # pure tension: L_b = 0 defines beta; pure bending: L_N = 0 defines beta; otherwise L_b = k L_N
        res = np.where(pure_tension, l_b, np.where(pure_bending, l_n, l_b - k * l_n))
        return res, l_n, l_b, branch_ii

    lo = np.zeros_like(alpha)
    hi = np.full_like(alpha, 0.5 * np.pi)  # the physical neutral-axis root lies in [0, pi/2]
    r_lo = residual(lo)[0]
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        r_mid = residual(mid)[0]
        same = np.sign(r_mid) == np.sign(r_lo)
        lo = np.where(same, mid, lo)
        r_lo = np.where(same, r_mid, r_lo)
        hi = np.where(same, hi, mid)
    beta = 0.5 * (lo + hi)
    _, l_n, l_b, branch_ii = residual(beta)
    return (
        np.asarray(np.maximum(l_n, 0.0), dtype=float),
        np.asarray(np.maximum(l_b, 0.0), dtype=float),
        np.asarray(beta, dtype=float),
        np.asarray(branch_ii, dtype=bool),
    )
