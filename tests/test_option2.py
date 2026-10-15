"""Option 2 assessment line: laws, monotonicity rule, tabulated data handling (design note D1-D5)."""

from __future__ import annotations

import numpy as np
import pytest

from critflaw.fad import acceptable, radial_margin
from critflaw.fad.margins import EXIT_CUTOFF, EXIT_STEP
from critflaw.fad.option2 import (
    Option2Curve,
    PowerLaw,
    StressStrainDataError,
    TabulatedLaw,
    hardening_exponent_lower_bound,
)

SY, SU, E = 355e6, 490e6, 207e9
KAPPA = E / SY
ULP = 5e-15


def d1(s: np.ndarray, e: np.ndarray) -> np.ndarray:
    return (e / s + s**3 / (2 * e)) ** -0.5


# ---- analytic laws -----------------------------------------------------------------------


def test_elastic_law_gives_the_elastic_line() -> None:
    curve = Option2Curve(PowerLaw(offset=0.0, n=1.0), 400e6, 500e6, E)
    lr = np.linspace(0.0, 1.12, 50)
    np.testing.assert_allclose(curve(lr), (1 + lr**2 / 2) ** -0.5, rtol=1e-14)
    assert curve(0.0) == 1.0 and curve(1.125) == 0.0 and bool(curve.valid)


def test_power_law_matches_the_source_expression() -> None:
    n = float(hardening_exponent_lower_bound(SY, SU))
    assert n == pytest.approx(1 / (0.3 * (1 - 355 / 490)), rel=1e-14)
    curve = Option2Curve(PowerLaw(offset=0.002, n=n), SY, SU, E)
    lr = np.linspace(1e-3, float(curve.lr_max) * 0.999, 400)
    e = lr + KAPPA * 0.002 * lr**n
    np.testing.assert_allclose(curve(lr), d1(lr, e), rtol=1e-13)
    np.testing.assert_allclose(curve(lr), curve.unmodified(lr), rtol=1e-13)  # already monotone: no correction
    assert curve(0.0) == 1.0 and np.all(np.diff(curve(lr)) <= ULP)
    assert curve.lr_max == pytest.approx((355 + 490) / (2 * 355))
    # at Lr = 1 with a 0.2% proof strength: e = 1 + 0.002 E / sy
    r1 = 1 + 0.002 * KAPPA
    assert curve(1.0) == pytest.approx((r1 + 0.5 / r1) ** -0.5, rel=1e-13)


@pytest.mark.parametrize(("sy", "offset"), [(300e6, 0.0), (300e6, 0.002), (355e6, 0.0), (900e6, 0.0)])
def test_luders_step_line_is_monotone_and_never_above_the_source_expression(sy: float, offset: float) -> None:
    su, delta = sy * 1.3, 0.0375 * (1 - 0.001 * sy / 1e6)
    n = float(hardening_exponent_lower_bound(sy, su))
    curve = Option2Curve(PowerLaw(offset, n, luders_strain=delta), sy, su, E)
    lr = np.unique(np.concatenate([np.linspace(0.0, 1.1499, 20001), [np.nextafter(1.0, 0), 1.0]]))
    h, f = curve(lr), curve.unmodified(lr)
    assert np.max(np.diff(h)) <= ULP and np.max(h - f) <= ULP
    r_hat = 1 + (E / sy) * (offset + delta)
    assert curve(1.0) == pytest.approx((r_hat + 0.5 / r_hat) ** -0.5, rel=1e-13)
    assert curve(np.nextafter(1.0, 0)) > 1.15 * float(curve(1.0))  # a real step at the yield strength
    assert curve.step_locations == (1.0,)
    correction = curve.largest_correction()
    assert (0.005 < correction < 0.04) if offset == 0.0 else correction < 0.004


def test_power_law_validation() -> None:
    with pytest.raises(ValueError):
        PowerLaw(offset=0.002, n=0.8)
    with pytest.raises(ValueError):
        PowerLaw(offset=-0.001, n=5.0)
    with pytest.raises(ValueError):
        PowerLaw(offset=0.002, n=5.0, luders_strain=0.0)
    with pytest.raises(ValueError):
        Option2Curve(PowerLaw(0.002, 5.0), 500e6, 400e6, E)


def test_one_curve_per_sample_keeps_the_offset_definition_of_yield() -> None:
    sy = np.array([300e6, 380e6, 460e6])
    su = sy * np.array([1.5, 1.3, 1.2])
    n = hardening_exponent_lower_bound(sy, su)
    vec = Option2Curve(PowerLaw(0.002, n), sy, su, E)
    lr = np.array([0.9, 1.0, 1.05])
    out = vec(lr)
    for i in range(3):
        one = Option2Curve(PowerLaw(0.002, float(n[i])), float(sy[i]), float(su[i]), E)
        assert out[i] == pytest.approx(float(one(lr[i])), rel=1e-14)
    # plastic strain at yield is 0.2% for every sample: e(1) = 1 + 0.002 E / sy
    r1 = 1 + 0.002 * E / sy
    np.testing.assert_allclose(vec(np.ones(3)), (r1 + 0.5 / r1) ** -0.5, rtol=1e-13)


def test_validity_condition_rejects_a_law_still_elastic_beyond_root_two() -> None:
    elastic = Option2Curve(PowerLaw(0.0, 1.0), 200e6, 600e6, E)  # Lr,max = 2
    assert not bool(elastic.valid)
    hardened = Option2Curve(PowerLaw(0.002, 5.0), 200e6, 600e6, E)
    assert bool(hardened.valid)
    assert np.max(np.diff(hardened(np.linspace(0, 1.999, 4000)))) <= ULP


# ---- tabulated laws ----------------------------------------------------------------------


def ramberg_osgood_table(sy: float, n: float, offset: float = 0.002, s_max: float = 1.25, points: int = 60) -> tuple:
    """True stress and strain of eps = sigma/E + offset (sigma/sy)^n."""
    s = np.concatenate([np.linspace(0.0, 0.6, 5), np.linspace(0.65, s_max, points)])
    return s * sy, s * sy / E + offset * s**n


def test_tabulated_law_reproduces_the_power_law_it_was_sampled_from() -> None:
    n = 8.0
    stress, strain = ramberg_osgood_table(SY, n)
    law = TabulatedLaw.from_true(stress, strain, SY, E, yielding="continuous")
    tab = Option2Curve(law, SY, SU, E)
    ref = Option2Curve(PowerLaw(0.002, n), SY, SU, E)
    lr = np.linspace(0.0, float(tab.lr_max) * 0.999, 500)
    # interpolation error of a 60-point table, plus plastic strain below 2% of yield strain being ignored
    assert np.max(np.abs(tab(lr) - ref(lr))) < 0.015
    assert np.max(np.abs(tab(lr[lr > 0.75]) - ref(lr[lr > 0.75]))) < 3e-3
    assert tab(0.0) == 1.0 and np.max(tab(lr)) <= 1.0 and np.max(np.diff(tab(lr))) <= ULP
    assert law.step_locations == () and "declared yielding: continuous" in law.notes


def engineering_plateau_record(sy: float, delta: float, n: float, s_max: float = 1.2) -> tuple:
    """Engineering curve: elastic to yield, a flat plateau of ``delta``, then power-law hardening."""
    eps_y = sy / E
    elastic = np.linspace(0.0, eps_y, 12)
    plateau = eps_y + np.linspace(0.0, delta, 12)[1:]
    s_true = np.linspace(1.0, s_max, 40)[1:]
    eps_true = s_true * sy / E + np.log1p(eps_y + delta) - np.log1p(eps_y) + 0.002 * (s_true**n - 1)
    eps_eng = np.expm1(eps_true)
    stress = np.concatenate([elastic * E, np.full(plateau.size, sy), s_true * sy / (1 + eps_eng)])
    return stress, np.concatenate([elastic, plateau, eps_eng])


def test_declared_plateau_gives_a_step_line() -> None:
    delta, n = 0.02, 12.0
    stress, strain = engineering_plateau_record(SY, delta, n)
    law = TabulatedLaw.from_engineering(stress, strain, SY, E, yielding="plateau", luders_strain=delta)
    curve = Option2Curve(law, SY, SU, E)
    assert law.step_locations == (1.0,) and "plateau points removed: 11" in law.notes
    jump = np.log1p(SY / E + delta) - np.log1p(SY / E)
    r_hat = 1 + KAPPA * jump
    assert curve(1.0) == pytest.approx((r_hat + 0.5 / r_hat) ** -0.5, rel=1e-9)
    assert curve(np.nextafter(1.0, 0)) == pytest.approx(1.5**-0.5, rel=1e-9)
    lr = np.unique(np.concatenate([np.linspace(0, 1.15, 5000), [np.nextafter(1.0, 0), 1.0]]))
    assert np.max(np.diff(curve(lr))) <= ULP and np.max(curve(lr) - curve.unmodified(lr)) <= ULP
    # the same record declared continuous has no step: the flat becomes a steep true-stress ramp, and
    # the line just below the yield strength stays above the step line (the documented hazard)
    ramp = Option2Curve(TabulatedLaw.from_engineering(stress, strain, SY, E, yielding="continuous"), SY, SU, E)
    assert ramp.step_locations == () and float(ramp(1.0)) > 2 * float(curve(1.0))
    # ... and only there: once the true stress has passed the end of the flat the two lines agree
    assert float(ramp(1.03)) == pytest.approx(float(curve(1.03)), abs=0.02)
    # exit through the step is reported by the radial margin
    slope = 0.5 * (float(curve(1.0)) + 1.5**-0.5)
    f, code = radial_margin(0.4, 0.4 * slope, curve)
    assert f == pytest.approx(2.5, rel=1e-9) and code == EXIT_STEP


def test_plateau_declaration_rules() -> None:
    stress, strain = engineering_plateau_record(SY, 0.02, 12.0)
    with pytest.raises(ValueError, match="required declaration"):
        TabulatedLaw.from_engineering(stress, strain, SY, E, yielding="detect")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="needs luders_strain"):
        TabulatedLaw.from_engineering(stress, strain, SY, E, yielding="plateau")
    with pytest.raises(ValueError, match="only meaningful"):
        TabulatedLaw.from_engineering(stress, strain, SY, E, yielding="continuous", luders_strain=0.02)
    # a declared plateau longer than the real one reaches into the hardening part: not flat
    with pytest.raises(StressStrainDataError, match="not flat"):
        TabulatedLaw.from_engineering(stress, strain, SY, E, yielding="plateau", luders_strain=0.12)
    with pytest.raises(StressStrainDataError, match="not flat"):
        TabulatedLaw.from_engineering(stress, strain, SY * 1.08, E, yielding="plateau", luders_strain=0.02)


def test_low_hardening_continuous_curves_are_accepted_without_a_step() -> None:
    # a detector would call these plateaus; the library does not detect
    stress, strain = ramberg_osgood_table(690e6, 30.0, s_max=1.08)
    law = TabulatedLaw.from_true(stress, strain, 690e6, E, yielding="continuous")
    curve = Option2Curve(law, 690e6, 760e6, E)
    ref = Option2Curve(PowerLaw(0.002, 30.0), 690e6, 760e6, E)
    assert law.step_locations == () and curve(1.0) == pytest.approx(float(ref(1.0)), abs=5e-3)
    # the partner's bilinear engineering curve: elastic to 0.002, then 1000 MPa per unit strain
    eps = np.array([0.0, 0.001, 0.002, 0.003, 0.004, 0.01, 0.03, 0.06])
    sig = np.where(eps <= 0.002, 200e9 * eps, 400e6 + 1000e6 * (eps - 0.002))
    bilinear = TabulatedLaw.from_engineering(sig, eps, 400e6, 200e9, yielding="continuous")
    line = Option2Curve(bilinear, 400e6, 460e6, 200e9)
    assert bilinear.step_locations == ()
    assert line(np.nextafter(1.0, 0)) == pytest.approx(1.5**-0.5, rel=1e-3)  # no artificial strain increment


def test_noisy_data_are_accepted_and_move_the_line_by_a_bounded_amount() -> None:
    rng = np.random.default_rng(5)
    stress, strain = ramberg_osgood_table(SY, 8.0, points=60)
    clean = Option2Curve(TabulatedLaw.from_true(stress, strain, SY, E, yielding="continuous"), SY, SU, E)
    lr = np.linspace(0.0, 1.18, 400)
    worst = 0.0
    for _ in range(50):
        noisy_stress = np.maximum(stress + rng.normal(0.0, 1e6, stress.size), 0.0)
        noisy_strain = np.maximum(strain + rng.normal(0.0, 5e-5, strain.size), 0.0)
        noisy_stress[0] = noisy_strain[0] = 0.0
        law = TabulatedLaw.from_true(noisy_stress, noisy_strain, SY, E, yielding="continuous")
        curve = Option2Curve(law, SY, SU, E)
        values = curve(lr)
        assert np.max(values) <= 1.0 and np.max(np.diff(values)) <= ULP and curve(0.0) == 1.0
        worst = max(worst, float(np.max(np.abs(values - clean(lr)))))
    assert worst < 0.05


def test_bad_data_are_rejected() -> None:
    stress, strain = ramberg_osgood_table(SY, 8.0)
    with pytest.raises(StressStrainDataError, match="modulus is too low"):
        TabulatedLaw.from_true(stress, strain, SY, E * 0.5, yielding="continuous")
    with pytest.raises(StressStrainDataError, match="elastic range"):
        TabulatedLaw.from_true(stress[40:], strain[40:], SY, E, yielding="continuous")
    dropping = stress.copy()
    dropping[30:36] -= 0.2 * SY  # a sustained load drop is not noise
    with pytest.raises(StressStrainDataError, match="not a monotonic"):
        TabulatedLaw.from_true(dropping, strain, SY, E, yielding="continuous")
    with pytest.raises(StressStrainDataError, match="beyond the yield strength"):
        TabulatedLaw.from_true(stress[:20], strain[:20], SY, E, yielding="continuous")


def test_table_that_ends_before_the_collapse_cut_off_ends_the_line_there() -> None:
    stress, strain = ramberg_osgood_table(SY, 8.0, s_max=1.10)
    curve = Option2Curve(TabulatedLaw.from_true(stress, strain, SY, E, yielding="continuous"), SY, SU, E)
    assert curve.lr_max == pytest.approx(1.10) and bool(curve.ends_at_data)
    assert curve(1.10) == 0.0 and curve(1.0999) > 0.0
    assert not acceptable(1.10, 0.01, curve)
    assert radial_margin(0.5, 1e-6, curve) == (pytest.approx(2.2, rel=1e-12), EXIT_CUTOFF)


def test_running_maximum_from_knots_is_exact_and_the_line_is_monotone_and_conservative() -> None:
    rng = np.random.default_rng(0)
    checked = rejected = 0
    for trial in range(3000):
        kappa = rng.uniform(300.0, 900.0)
        sy = E / kappa
        lr_max = rng.uniform(1.02, 2.0)
        k = int(rng.integers(4, 40))
        knots = np.sort(np.concatenate([[rng.uniform(0.3, 0.95)], rng.uniform(0.3, lr_max * 1.05, k), [lr_max * 1.06]]))
        inc = rng.exponential(rng.choice([1e-4, 1e-3, 1e-2]), knots.size) * (rng.random(knots.size) > 0.3)
        inc[0] = 0.0
        if trial % 7 == 0:  # a sharp rise followed by constant plastic strain
            inc[2:] = 0.0
        law = TabulatedLaw(np.concatenate([[0.0], knots]), np.concatenate([[0.0], np.cumsum(inc)]), None, ())
        curve = Option2Curve(law, sy, sy * (2 * lr_max - 1), E)
        s = np.sort(np.concatenate([np.linspace(1e-3, lr_max * (1 - 1e-12), 1500), knots[knots < lr_max]]))
        brute = np.maximum.accumulate(law.q(s))
        np.testing.assert_allclose(law.q_hat(s), brute, rtol=0, atol=1e-15)
        if not bool(curve.valid):
            rejected += 1
            continue
        checked += 1
        h = curve(s)
        assert np.max(np.diff(h)) <= ULP and np.max(h - curve.unmodified(s)) <= ULP
    assert checked > 2000 and rejected > 50


def test_partner_counter_example_is_rejected_by_the_validity_condition() -> None:
    # elastic to s = 2, then e rises to 2 sqrt(2) over 1e-5, then constant plastic strain
    kappa = 700.0
    ep = (2 * np.sqrt(2) - 2.00001) / kappa
    law = TabulatedLaw(np.array([0.0, 2.0, 2.00001, 3.2]), np.array([0.0, 0.0, ep, ep]), None, ())
    curve = Option2Curve(law, E / kappa, 5 * E / kappa, E)  # Lr,max = 3
    assert not bool(curve.valid)


def test_plateau_below_the_declared_yield_strength_puts_the_step_at_the_plateau_stress() -> None:
    # e.g. an upper yield strength entered as yield: the plateau sits 4% lower
    delta, n, true_sy = 0.02, 12.0, SY
    stress, strain = engineering_plateau_record(true_sy, delta, n)
    declared = true_sy / 0.96
    law = TabulatedLaw.from_engineering(stress, strain, declared, E, yielding="plateau", luders_strain=delta)
    assert law.step_locations == (pytest.approx(0.96, abs=1e-9),)
    assert any("step placed there" in note for note in law.notes)
    curve = Option2Curve(law, declared, SU / 0.96, E)
    # the line must not stay elastic between the plateau stress and the declared yield strength
    assert float(curve(0.97)) < 0.5 and float(curve(0.955)) > 0.8
    lr = np.unique(np.concatenate([np.linspace(0, 1.1, 4000), [np.nextafter(0.96, 0), 0.96]]))
    assert np.max(np.diff(curve(lr))) <= ULP
    # a plateau at or above the declared yield strength keeps the step at 1
    at = TabulatedLaw.from_engineering(stress, strain, true_sy, E, yielding="plateau", luders_strain=delta)
    assert at.step_locations == (1.0,)

# ---- added after the phase-2 code reviews ---------------------------------------------------


def test_flat_stretch_keeps_the_strain_reached_at_that_stress() -> None:
    # true-stress record with a flat stretch at yield, declared continuous (reviewer's case)
    stress = np.array([0.0, 0.5, 1.0, 1.0, 1.0, 1.05, 1.15, 1.25]) * SY
    strain = np.array([0.0, 0.5, 1.0, 6.0, 12.0, 14.0, 20.0, 30.0]) * SY / E
    curve = Option2Curve(TabulatedLaw.from_true(stress, strain, SY, E, yielding="continuous"), SY, SU, E)
    r = 1 + (strain[4] * E / SY - 1.0)  # e/s at the end of the flat
    assert float(curve(1.0)) == pytest.approx((r + 0.5 / r) ** -0.5, rel=1e-9)
    # both ends of the flat are kept: below it the data say elastic, and the line is elastic there
    assert float(curve(0.75)) == pytest.approx((1 + 0.75**2 / 2) ** -0.5, rel=1e-12)
    assert float(curve(np.nextafter(1.0, 0))) == pytest.approx(1.5**-0.5, rel=1e-9)
    assert curve.step_locations == (1.0,) and not curve.law.declared_plateau
    assert any("flat" in note for note in curve.law.notes)
    # Stresses rounded to 10 MPa (3% of yield) give a stepped line: between two stress levels the data
    # cannot say where the strain was reached. The error stays below 0.02.
    s_true, e_true = ramberg_osgood_table(SY, 8.0, points=120)
    exact = Option2Curve(TabulatedLaw.from_true(s_true, e_true, SY, E, yielding="continuous"), SY, SU, E)
    rounded = Option2Curve(
        TabulatedLaw.from_true(np.round(s_true / 10e6) * 10e6, e_true, SY, E, yielding="continuous"), SY, SU, E
    )
    lr = np.linspace(0.0, 1.18, 300)
    assert np.max(np.abs(rounded(lr) - exact(lr))) < 0.02


def test_equivalent_engineering_and_true_plateau_records_agree() -> None:
    eng_stress = np.array([0, 200e6, 400e6, 400e6, 400e6, 500e6, 560e6])
    eng_strain = np.array([0, 0.001, 0.002, 0.042, 0.082, 0.12, 0.18])
    args = (400e6, 200e9)
    a = TabulatedLaw.from_engineering(eng_stress, eng_strain, *args, yielding="plateau", luders_strain=0.08)
    b = TabulatedLaw.from_true(
        eng_stress * (1 + eng_strain), np.log1p(eng_strain), *args, yielding="plateau", luders_strain=0.08
    )
    assert a.step_locations == b.step_locations == (1.0,)
    np.testing.assert_allclose(a.knots, b.knots, rtol=1e-12)
    np.testing.assert_allclose(a.eps_p, b.eps_p, rtol=1e-12, atol=1e-15)


def test_zero_offset_law_does_not_overflow() -> None:
    curve = Option2Curve(PowerLaw(0.0, 10000.0), SY, SU, E)
    assert curve(1.1) == pytest.approx((1 + 1.1**2 / 2) ** -0.5, rel=1e-14)
    mixed = Option2Curve(PowerLaw(np.array([0.0, 0.002]), 10.0), SY, SU, E)
    out = mixed(np.array([1.1, 1.1]))
    assert out[0] == pytest.approx((1 + 1.1**2 / 2) ** -0.5, rel=1e-14) and out[1] < out[0]
    assert mixed.law.parameter_shape == (2,)


def test_power_law_details_pinned() -> None:
    # jump applies at s = 1 itself; the running maximum includes the offset
    law = PowerLaw(0.002, 2.0, luders_strain=0.01)
    assert law.q(np.array([1.0]))[0] == pytest.approx(0.012) and law.q(np.array([np.nextafter(1.0, 0)]))[0] < 0.0021
    assert law.q_hat(np.array([1.02]))[0] == pytest.approx(0.012)  # held at offset + luders
    assert law.q(np.array([1.02]))[0] < 0.012
    # validity must hold at every point beyond sqrt(2), not just one
    weak = Option2Curve(PowerLaw(1e-7, 2.0), 200e6, 600e6, E)
    assert not bool(weak.valid)
    # offset 0.002 table reproduces the power-law line closely where plastic strain is above the noise threshold
    s_true, e_true = ramberg_osgood_table(SY, 8.0, points=400)
    tab = Option2Curve(TabulatedLaw.from_true(s_true, e_true, SY, E, yielding="continuous"), SY, SU, E)
    ref = Option2Curve(PowerLaw(0.002, 8.0), SY, SU, E)
    lr = np.linspace(0.8, 1.18, 200)
    assert np.max(np.abs(tab(lr) - ref(lr))) < 2e-4


def test_tabulated_validity_fails_when_a_single_interior_knot_is_too_elastic() -> None:
    kappa = 500.0
    knots = np.array([0.0, 0.9, 1.5, 1.7, 1.9, 2.1])
    good = np.array([0.0, 0.0, 0.004, 0.006, 0.008, 0.012])
    bad = good.copy()
    bad[3] = 0.004  # e(1.7) = 1.7 + 500 * 0.004 = 3.7 >= 1.7^2 / sqrt(2) = 2.04: still fine
    assert bool(Option2Curve(TabulatedLaw(knots, bad, None, ()), E / kappa, 3.3 * E / kappa, E).valid)
    barely = np.array([0.0, 0.0, 1e-5, 2e-5, 0.008, 0.012])  # e(1.7) = 1.71 < 2.04 at one interior knot
    assert not bool(Option2Curve(TabulatedLaw(knots, barely, None, ()), E / kappa, 3.3 * E / kappa, E).valid)
    assert bool(Option2Curve(TabulatedLaw(knots, barely, None, ()), E / kappa, 1.7 * E / kappa, E).valid)  # Lr,max 1.35


def test_notes_record_data_cleaning() -> None:
    stress, strain = ramberg_osgood_table(SY, 8.0)
    law = TabulatedLaw.from_true(stress, strain, SY, E * 0.93, yielding="continuous")  # modulus a little low
    assert any("negative plastic strain" in n for n in law.notes)