"""FAD curve, acceptance, radial margin and Kr: limiting cases and invariants (spec S1.5 kind b)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from critflaw.fad import (
    Option1Curve,
    acceptable,
    fracture_ratio,
    kj_secondary,
    lr_cutoff,
    radial_margin,
    rho_simplified,
    v_simplified,
)
from critflaw.fad.margins import EXIT_CURVE, EXIT_CUTOFF, EXIT_NONE, EXIT_STEP

SY, SU, E = 355e6, 490e6, 207e9
CONT = Option1Curve(SY, SU, E, "continuous")
DISC = Option1Curve(SY, SU, E, "discontinuous")

yield_st = st.floats(200e6, 945e6)
ratio_st = st.floats(1.02, 2.0)
modulus_st = st.floats(60e9, 220e9)


# ---- curves -------------------------------------------------------------------------


def test_cutoff() -> None:
    assert lr_cutoff(SY, SU) == pytest.approx((355 + 490) / (2 * 355))


@pytest.mark.parametrize("curve", [CONT, DISC])
def test_f_at_zero_is_one_and_zero_at_cutoff(curve: Option1Curve) -> None:
    assert curve(0.0) == 1.0
    assert curve(curve.lr_max) == 0.0
    assert curve(curve.lr_max + 5.0) == 0.0
    assert curve(np.nextafter(curve.lr_max, 0.0)) > 0.0


def test_continuous_curve_is_continuous_at_one() -> None:
    assert CONT(np.nextafter(1.0, 2.0)) == pytest.approx(float(CONT(1.0)), rel=1e-12)
    assert CONT(np.nextafter(1.0, 0.0)) == pytest.approx(float(CONT(1.0)), rel=1e-12)


def test_regression_distinguishes_this_edition_from_older_fixed_coefficient_curve() -> None:
    mu = min(0.001 * E / SY, 0.6)
    hand = (1 + 0.5) ** -0.5 * (0.3 + 0.7 * np.exp(-mu))
    assert CONT(1.0) == pytest.approx(hand, rel=1e-14)
    older = (1 - 0.14) * (0.3 + 0.7 * np.exp(-0.65))
    assert abs(float(CONT(1.0)) - older) > 5e-3


def test_material_dependence_of_mu_cap() -> None:
    # 0.001 E / sy exceeds 0.6 for low-strength material, so mu is capped.
    low = Option1Curve(200e6, 400e6, 207e9, "continuous")
    assert low(1.0) == pytest.approx(1.5**-0.5 * (0.3 + 0.7 * np.exp(-0.6)), rel=1e-14)


def test_discontinuous_step() -> None:
    assert DISC(np.nextafter(1.0, 0.0)) == pytest.approx(1.5**-0.5, rel=1e-12)
    strain = 0.0375 * (1 - 0.001 * 355)
    lam = 1 + E * strain / SY
    assert DISC(1.0) == pytest.approx((lam + 0.5 / lam) ** -0.5, rel=1e-14)
    assert DISC(1.0) < DISC(np.nextafter(1.0, 0.0))
    explicit = Option1Curve(SY, SU, E, "discontinuous", luders_strain=strain)
    assert explicit(1.05) == pytest.approx(float(DISC(1.05)), rel=1e-14)


def test_power_law_branch() -> None:
    n = 0.3 * (1 - SY / SU)
    assert CONT(1.1) == pytest.approx(float(CONT(1.0)) * 1.1 ** ((n - 1) / (2 * n)), rel=1e-14)


def test_invalid_curve_inputs() -> None:
    with pytest.raises(ValueError):
        Option1Curve(500e6, 400e6, E, "continuous")
    with pytest.raises(ValueError):
        CONT(-0.1)
    with pytest.raises(ValueError):
        CONT(np.nan)
    with pytest.raises(ValueError):
        Option1Curve(950e6, 1100e6, E, "discontinuous")(1.0)


@given(sy=yield_st, ratio=ratio_st, e=modulus_st, kind=st.sampled_from(["continuous", "discontinuous", "unknown"]))
@settings(max_examples=200, deadline=None)
def test_curve_is_non_increasing_and_bounded(sy: float, ratio: float, e: float, kind: str) -> None:
    curve = Option1Curve(sy, sy * ratio, e, kind)
    lr = np.concatenate([np.linspace(0.0, float(curve.lr_max) * 1.2, 2001), [1.0, float(curve.lr_max)]])
    lr.sort()
    f = curve(lr)
    assert np.all(np.diff(f) <= 1e-13)
    assert np.all((f >= 0.0) & (f <= 1.0))


def test_curve_broadcasts_one_curve_per_sample() -> None:
    sy = np.array([300e6, 355e6, 460e6])
    curve = Option1Curve(sy, sy * 1.3, E, "continuous")
    out = curve(np.array([0.9, 1.0, 1.1]))
    for i in range(3):
        assert out[i] == pytest.approx(float(Option1Curve(sy[i], sy[i] * 1.3, E, "continuous")([0.9, 1.0, 1.1][i])), rel=1e-14)


# ---- acceptance and radial margin ----------------------------------------------------


def test_point_on_the_line_is_unacceptable() -> None:
    f = float(CONT(0.7))
    assert not acceptable(0.7, f, CONT)
    assert acceptable(0.7, np.nextafter(f, 0.0), CONT)
    assert not acceptable(CONT.lr_max, 0.0, CONT)


def test_radial_margin_axes_and_origin() -> None:
    lr_max = float(CONT.lr_max)
    f, code = radial_margin([0.005, 0.0, 0.0], [0.0, 0.004, 0.0], CONT)
    assert f[0] == pytest.approx(lr_max / 0.005, rel=1e-14) and code[0] == EXIT_CUTOFF
    assert f[1] == pytest.approx(250.0, rel=1e-14) and code[1] == EXIT_CURVE
    assert np.isinf(f[2]) and code[2] == EXIT_NONE


def test_radial_margin_cutoff_governed_ray() -> None:
    # Below the ray through the corner (Lr,max, f(Lr,max-)) the cut-off governs and F = Lr,max / Lr.
    lr_max = float(CONT.lr_max)
    corner_slope = float(CONT(np.nextafter(lr_max, 0.0))) / lr_max
    f, code = radial_margin(0.5, 0.5 * corner_slope * 0.5, CONT)
    assert f == pytest.approx(lr_max / 0.5, rel=1e-12) and code == EXIT_CUTOFF


def test_radial_margin_through_the_step() -> None:
    # A ray with f(1) < Kr/Lr < f(1-) leaves through the vertical step: F = 1 / Lr.
    slope = 0.5 * (float(DISC(1.0)) + 1.5**-0.5)
    f, code = radial_margin(0.4, 0.4 * slope, DISC)
    assert f == pytest.approx(1.0 / 0.4, rel=1e-9) and code == EXIT_STEP
    # the same ray on the continuous curve leaves through the curve, and a steeper one leaves before the step
    assert radial_margin(0.4, 0.4 * slope, CONT)[1] == EXIT_CURVE
    assert radial_margin(0.4, 0.4 * 0.9, DISC)[1] == EXIT_CURVE
    # "unknown" has a step only where the plateau value is below the continuous value
    unknown = Option1Curve(SY, SU, E, "unknown")
    slope_u = 0.5 * (float(unknown(1.0)) + float(CONT(1.0)))
    assert radial_margin(0.4, 0.4 * slope_u, unknown)[1] == EXIT_STEP


def test_radial_margin_of_unacceptable_points() -> None:
    f, _ = radial_margin([1.0, 1.3, 0.2], [0.9, 0.1, 1.5], CONT)
    assert np.all(f < 1.0)
    assert f[1] == pytest.approx(float(CONT.lr_max) / 1.3, rel=1e-12)


class _RaisingCurve:
    """Curve double that refuses evaluation at or beyond its cut-off (spec S2.4)."""

    lr_max = np.asarray(1.2)

    def __call__(self, lr: object) -> np.ndarray:
        x = np.asarray(lr, dtype=float)
        if np.any(x >= 1.2) or np.any(x < 0.0):
            raise AssertionError("curve evaluated outside its domain")
        return np.asarray(1.0 - 0.5 * x)


def test_curve_is_never_evaluated_outside_its_domain() -> None:
    curve = _RaisingCurve()
    lr = np.array([1.0, 0.3, 1.19, 2.0, 0.0, 0.9])
    kr = np.array([0.1, 0.2, 0.0, 0.3, 0.5, 5.0])
    f, _ = radial_margin(lr, kr, curve)
    acceptable(lr, kr, curve)
    assert np.all(np.isfinite(f))


@given(
    lr=st.floats(0.0, 2.0),
    kr=st.floats(0.0, 2.0),
    sy=yield_st,
    ratio=ratio_st,
    kind=st.sampled_from(["continuous", "discontinuous", "unknown"]),
)
@settings(max_examples=400, deadline=None)
def test_margin_is_consistent_with_direct_acceptance(lr: float, kr: float, sy: float, ratio: float, kind: str) -> None:
    curve = Option1Curve(sy, sy * ratio, E, kind)
    f, _ = radial_margin(lr, kr, curve)
    f = float(f)
    ok = bool(acceptable(lr, kr, curve))
    if lr == 0.0 and kr == 0.0:
        assert np.isinf(f) and ok
        return
    if np.isinf(f):  # the margin overflows for a subnormal assessment point
        assert ok and max(lr, kr) < 1e-300
        return
    # Away from the boundary the two must agree exactly.
    if abs(f - 1.0) > 1e-9:
        assert ok == (f > 1.0)
    # Scaling the point to just inside / just outside the margin flips acceptance.
    assert acceptable(lr * f * (1 - 1e-7), kr * f * (1 - 1e-7), curve)
    assert not acceptable(lr * f * (1 + 1e-7), kr * f * (1 + 1e-7), curve)


# ---- plasticity interaction ----------------------------------------------------------


def test_rho_breakpoints() -> None:
    assert rho_simplified(0.0, 0.5) == 0.0
    chi = 2.0
    rho1 = 0.1 * chi**0.714 - 0.007 * chi**2 + 0.00003 * chi**5
    assert rho_simplified(chi, 0.5) == pytest.approx(rho1, rel=1e-14)
    assert rho_simplified(chi, 0.8) == pytest.approx(rho1, rel=1e-14)
    assert rho_simplified(chi, np.nextafter(0.8, 1.0)) == pytest.approx(rho1, rel=1e-12)  # continuous
    assert rho_simplified(chi, 0.9) == pytest.approx(4 * rho1 * 0.15, rel=1e-12)
    assert rho_simplified(chi, 1.05) == 0.0
    assert rho_simplified(chi, 1.2) == 0.0
    assert rho_simplified(6.0, 0.5) == 0.25
    # The polynomial meets the plateau at chi = 5.2 to within its rounding.
    assert rho_simplified(np.nextafter(5.2, 0.0), 0.5) == pytest.approx(0.25, abs=1e-3)


def test_v_breakpoints() -> None:
    assert v_simplified(0.0, 0.0) == 1.0
    assert v_simplified(2.0, 0.5) == pytest.approx(1 + 0.1 + 0.02 * 2.0 * 2.0, rel=1e-14)
    assert v_simplified(2.0, 1.0) == pytest.approx(min(1.2 + 0.02 * 2 * 3, 1.1), rel=1e-14)
    assert v_simplified(3.0, np.nextafter(1.05, 0.0)) == pytest.approx(1.0, abs=1e-12)  # continuous at 1.05
    assert v_simplified(3.0, 1.05) == 1.0
    assert v_simplified(3.0, 1.5) == 1.0


def test_kr_decision_table() -> None:
    kmat, a, sy = 100e6, 5e-3, 355e6
    # no secondary stress
    r = fracture_ratio(40e6, 0.0, 0.5, kmat, a, sy)
    assert r.kr == pytest.approx(0.4) and not r.secondary_only
    # negative secondary K is set to zero
    r = fracture_ratio(40e6, -30e6, 0.5, kmat, a, sy)
    assert r.kr == pytest.approx(0.4) and r.interaction == 0.0
    # primary and secondary, rho route
    chi = 30e6 * 0.5 / 40e6
    r = fracture_ratio(40e6, 30e6, 0.5, kmat, a, sy)
    assert r.kr == pytest.approx(0.7 + float(rho_simplified(chi, 0.5)), rel=1e-14)
    assert r.chi == pytest.approx(chi) and not r.chi_above_limit
    # V route gives a different Kr
    rv = fracture_ratio(40e6, 30e6, 0.5, kmat, a, sy, method="V")
    assert rv.kr == pytest.approx((40e6 + float(v_simplified(chi, 0.5)) * 30e6) / kmat, rel=1e-14)
    assert rv.kr != pytest.approx(float(r.kr), rel=1e-6)
    # chi above 4 is flagged
    assert fracture_ratio(10e6, 100e6, 0.5, kmat, a, sy).chi_above_limit
    # secondary stress alone: plastic-zone-corrected K, no rho
    r = fracture_ratio(0.0, 30e6, 0.0, kmat, a, sy)
    a_eff = a + (30e6 / sy) ** 2 / (2 * np.pi)
    assert r.secondary_only and r.kr == pytest.approx(np.sqrt(a_eff / a) * 30e6 / kmat, rel=1e-14)
    assert kj_secondary(30e6, a, sy, plane_stress=False) < kj_secondary(30e6, a, sy, plane_stress=True)
    # self-balancing term is added outside the interaction
    r2 = fracture_ratio(40e6, 30e6, 0.5, kmat, a, sy, k_self_balancing=5e6)
    assert r2.kr == pytest.approx(0.7 + float(rho_simplified(chi, 0.5)) + 0.05, rel=1e-14)


def test_kr_vectorised_matches_scalar() -> None:
    kp = np.array([40e6, 0.0, 40e6, 10e6])
    ks = np.array([0.0, 30e6, -5e6, 100e6])
    lr = np.array([0.5, 0.0, 0.9, 0.3])
    r = fracture_ratio(kp, ks, lr, 100e6, 5e-3, 355e6)
    for i in range(4):
        assert r.kr[i] == pytest.approx(float(fracture_ratio(kp[i], ks[i], lr[i], 100e6, 5e-3, 355e6).kr), rel=1e-14)


def test_radial_margin_does_not_overflow_for_tiny_points() -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        f, _ = radial_margin(1e-310, 1e-308, CONT)
    assert 9e307 < float(f) < 1e308
