"""Phase 4 gate (i), the material law: the WARP3D deformation law as an Option 2 material (design note phase 4,
section 2.1). Kind-(b) pins of every branch and the Option 2 integration; the partner oracle comparison is added
when oracle 1 (arc, GCI) is received."""

from __future__ import annotations

import numpy as np
import pytest

from critflaw.fad.option2 import Option2Curve, PowerLaw
from critflaw.fe import WarpDeformationLaw, normalised_strain, transition_arc
from critflaw.fe.warp_law import K1

E, S0 = 207e9, 355e6


def test_arc_is_tangent_to_both_branches_and_unique_numbers_match_the_reviewers() -> None:
    # tangency: the arc meets the line at (K1, K1) with slope 1 and the power branch at (K2^n, K2) with its slope
    for n, k2_ref, r_ref in ((3, 1.0225, 0.295), (5, 1.0137, 0.231), (10, 1.0069, 0.197), (20, 1.0034, 0.183)):
        k2, xc, yc, r = (float(v) for v in transition_arc(n))
        assert k2 == pytest.approx(k2_ref, abs=6e-5) and r == pytest.approx(r_ref, abs=6e-4)  # red team B's values
        assert (K1 - xc) ** 2 + (K1 - yc) ** 2 == pytest.approx(r * r, rel=1e-12)  # passes through (K1, K1)
        x2 = k2**n
        assert (x2 - xc) ** 2 + (k2 - yc) ** 2 == pytest.approx(r * r, rel=1e-9)  # passes through the contact point
        # tangent directions: the radius is normal to y = x at (K1, K1) and to y = x^(1/n) at the contact point
        assert (xc - K1) == pytest.approx(-(yc - K1), rel=1e-12)
        g = (1.0 / n) * x2 ** (1.0 / n - 1.0)  # curve slope dy/dx
        assert (xc - x2) * 1.0 + (yc - k2) * g == pytest.approx(0.0, abs=1e-9 * r)  # (centre - point) . tangent = 0
        assert k2 > 1.0 and yc > 0.0
    # vectorised and scalar agree
    assert transition_arc(np.array([3.0, 10.0]))[0].tolist() == pytest.approx(
        [transition_arc(3)[0], transition_arc(10)[0]]
    )


def test_strain_branches_are_continuous_and_match_the_closed_forms() -> None:
    n = 10
    k2 = float(transition_arc(n)[0])
    y = np.array([0.5, K1, 0.5 * (K1 + k2), k2, 1.2])
    x = normalised_strain(y, n)
    assert x[0] == 0.5 and x[1] == pytest.approx(K1, rel=1e-12)
    assert x[3] == pytest.approx(k2**n, rel=1e-9) and x[4] == pytest.approx(1.2**n, rel=1e-12)
    assert K1 < x[2] < k2**n
    # continuity across the two junctions
    for yj in (K1, k2):
        below, above = normalised_strain(np.nextafter(yj, 0.0), n), normalised_strain(np.nextafter(yj, 2.0), n)
        assert float(above) == pytest.approx(float(below), rel=1e-6)
    # monotone in y
    grid = np.linspace(0.01, 2.0, 2000)
    assert np.all(np.diff(normalised_strain(grid, n)) > 0.0)


def test_option2_protocol_q_is_zero_then_non_decreasing_and_the_curve_takes_the_law() -> None:
    law = WarpDeformationLaw(E, 0.3, S0, 10)
    s = np.linspace(0.0, 1.6, 1601)
    q = law.q(s)
    assert np.all(q[s <= K1] == 0.0) and np.all(np.diff(q) >= -1e-18)
    assert np.array_equal(law.q_hat(s), q)
    assert law.step_locations == () and law.s_end == float("inf") and not law.declared_plateau
    # beyond K2 the plastic strain is eps_0 (s^n - s): q = eps_0 (s^(n-1) - 1)
    assert float(law.q(np.asarray(1.3))) == pytest.approx(S0 / E * (1.3**9 - 1.0), rel=1e-12)
    assert law.hardened_enough(np.array([E / S0]), np.array([2.0]))[0]
    curve = Option2Curve.from_warp_law(law, 490e6)
    assert float(curve.kappa) == pytest.approx(E / S0)
    with pytest.raises(ValueError, match="from_warp_law"):
        Option2Curve(law, 300e6, 490e6, E)  # a different yield strength than the law's sigma_0
    assert Option2Curve(PowerLaw(0.002, 10), 300e6, 490e6, E).kappa == pytest.approx(E / 300e6)  # other laws unaffected
    # per-sample n is supported
    laws = WarpDeformationLaw(E, 0.3, S0, np.array([5.0, 10.0]))
    assert laws.parameter_shape == (2,) and laws.q(np.asarray(1.3)).shape == (2,)


def test_reported_quantities_and_refusals() -> None:
    law = WarpDeformationLaw(E, 0.3, S0, 10)
    rp = float(law.proof_stress_0p2())
    assert float(law.strain(rp) - rp / E) == pytest.approx(0.002, rel=1e-6)
    assert 1.0 < rp / S0 < 1.2  # red team B: 1.02-1.52 over the envelope; n = 10, sigma_0/E = 1.7e-3 is near 1.08
    cs = float(law.considere_stress())
    s = cs / S0
    assert s**10 == pytest.approx(1.0 / (10 * S0 / E), rel=1e-12)
    for bad in (dict(E=-1.0), dict(nu=0.5), dict(sigma_0=0.0), dict(n=1.0)):
        kw = dict(E=E, nu=0.3, sigma_0=S0, n=10.0)
        kw.update(bad)
        with pytest.raises(ValueError):
            WarpDeformationLaw(**kw)


# ---- partner oracle 1 (arc and D7 formulas), received 2026-10-07 ------------------------------------------------

from pathlib import Path as _Path  # noqa: E402

from critflaw.fe import convergence as CV  # noqa: E402


def _load_oracle():  # type: ignore[no-untyped-def]
    import importlib.util

    path = _Path(__file__).parent / "oracles" / "partner_reference_phase4_arc_gci.py"
    spec = importlib.util.spec_from_file_location("partner_reference_phase4_arc_gci", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_partner_oracle_arc_and_gci_agree() -> None:
    oracle = _load_oracle()
    for n in (3, 5, 10, 20):
        mine = tuple(float(v) for v in transition_arc(n))
        theirs = tuple(float(v) for v in oracle.arc(n))
        assert mine == pytest.approx(theirs, rel=1e-10), n
    for y, n in ((0.9, 10), (0.97, 10), (1.0, 10), (1.05, 10), (1.2, 5)):
        assert float(normalised_strain(y, n)) == pytest.approx(float(oracle.strain(y, n)), rel=1e-10)
    law = WarpDeformationLaw(E, 0.3, S0, 10)
    for s in (0.5, 0.99, 1.3):
        # the oracle's q has kappa factored out: q_lib = eps_0 * q_oracle
        assert float(law.q(np.asarray(s))) == pytest.approx(
            S0 / E * float(oracle.plastic_compliance(s, 10)), rel=1e-10, abs=1e-30
        )
    for p in (0.8, 2.0, 3.0):
        seq = oracle.synthetic_sequence(p)
        j = tuple(float(v) for v in seq["J"])
        v = CV.assess_four_levels(j)
        assert v.p == pytest.approx(p, abs=1e-9) and v.p == pytest.approx(float(seq["observed_order"]), abs=1e-9)
        assert v.predicted_d34_over_d23 == pytest.approx(float(seq["predicted_ratio"]), rel=1e-12)
        assert v.realised_d34_over_d23 == pytest.approx(v.predicted_d34_over_d23, rel=1e-9)
        assert v.coarse_error_abs == pytest.approx(float(seq["coarse_error"]), rel=1e-9) == pytest.approx(5.0, rel=1e-9)
        assert v.monotone and v.prediction_within and v.order_stable
        assert v.order_in_range or p in (0.8, 3.0)  # these cases sit on the range boundaries (bisection rounding)
    assert np.isnan(CV.observed_order(1.0, 1.0, 1.25, 1.2)) and np.isnan(
        float(oracle.observed_order(1.0, 1.0, 1.25, 1.2))
    )


def test_d7_regression_the_partners_round_3_counter_example_fails_correctly() -> None:
    # J* = 100, J1 = 104, J2 = 102.56, J3 = 101.6384 at constant ratio 1.25 with p = 2: the old formula gave the
    # finer level's error (2.56, a false pass at 2.5 % of J2); the coarse-grid estimate is 5.00 = 4.88 % of J2
    p = CV.observed_order(104 - 102.56, 102.56 - 101.6384, 1.25, 1.25)
    assert p == pytest.approx(2.0, abs=1e-9)
    assert CV.coarse_error(104.0, 102.56, p, 1.25, safety_factor=1.0) == pytest.approx(4.0, rel=1e-9)
    assert CV.coarse_error(104.0, 102.56, p, 1.25) / 102.56 == pytest.approx(0.0488, abs=2e-4)
    # an oscillatory sequence fails (a); a flat sequence fails (b)
    assert not CV.assess_four_levels((104.0, 102.0, 103.0, 102.5)).monotone
    assert not CV.assess_four_levels((104.0, 103.0, 102.0, 101.0)).order_in_range  # ratio 1 below the infimum
