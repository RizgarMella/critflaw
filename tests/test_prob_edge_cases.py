"""Tests added after the Phase 2 code reviews: one per finding and per surviving mutant."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import critflaw as cf
from critflaw import prob
from critflaw.prob import solvers
from critflaw.prob.solvers import Evaluation, SubsetResult, replicate_interval, subset_replicates, subset_simulation

PLATE = cf.FlatPlate(thickness=25e-3, width=500e-3)
FLAW = cf.SurfaceFlaw(a=5e-3, two_c=40e-3)
STEEL = cf.Material(355e6, 490e6, 207e9, 100e6, yielding="continuous")
MPA = 1e6


# ---- engine: model factors, Master Curve, Option 2 ------------------------------------------


def test_model_factors_scale_the_right_quantities() -> None:
    base = cf.Assessment(PLATE, FLAW, STEEL, cf.StressState(pm=150e6, qm=80e6)).run()
    s = cf.Assessment(PLATE, FLAW, STEEL, cf.StressState(pm=150e6, qm=80e6), sref_model=1.2).run()
    assert s.lr == pytest.approx(1.2 * float(base.lr), rel=1e-14)
    assert s.reference_stress == pytest.approx(1.2 * float(base.reference_stress), rel=1e-14)
    np.testing.assert_allclose(s.k_primary, base.k_primary, rtol=1e-14)
    k = cf.Assessment(PLATE, FLAW, STEEL, cf.StressState(pm=150e6, qm=80e6), k_model=1.3).run()
    np.testing.assert_allclose(k.k_primary, 1.3 * base.k_primary, rtol=1e-14)
    np.testing.assert_allclose(k.k_secondary, 1.3 * base.k_secondary, rtol=1e-14)
    assert k.lr == pytest.approx(float(base.lr), rel=1e-14)
    # an array-valued factor is no longer listed as "not accounted for"
    arr = cf.Assessment(PLATE, FLAW, STEEL, cf.StressState(pm=150e6), k_model=np.array([1.0, 1.1])).run()
    assert not any("stress intensity factor" in t for t in arr.trace.excluded_uncertainty)
    assert any("stress intensity factor" in t for t in base.trace.excluded_uncertainty)


@pytest.mark.parametrize(
    ("flaw", "front"),
    [
        (cf.SurfaceFlaw(5e-3, 40e-3), 40e-3),  # surface flaw: 2c
        (cf.SurfaceFlaw(5e-3, 80e-3), 50e-3),  # 2c above twice the thickness: capped at 2B
        (cf.ThroughFlaw(30e-3), 50e-3),  # through-thickness: 2B
    ],
)
def test_master_curve_crack_front_length_in_the_engine(flaw: cf.SurfaceFlaw | cf.ThroughFlaw, front: float) -> None:
    material = cf.Material(355e6, 490e6, 207e9, cf.MasterCurve(t0=-50.0, temperature=-20.0, probability=0.3),
                           yielding="continuous")
    r = cf.Assessment(PLATE, flaw, material, cf.StressState(pm=120e6)).run()
    shape = 11 + 77 * np.exp(0.019 * 30.0)
    expected = (20 + shape * (25 / (front * 1e3)) ** 0.25 * (-np.log(0.7)) ** 0.25) * MPA
    np.testing.assert_allclose(r.k_primary / r.kr_by_position, expected, rtol=1e-12)


def option2_material(law: object, yielding: str = "continuous", su: float = 490e6) -> cf.Material:
    return cf.Material(355e6, su, 207e9, 100e6, yielding=yielding, fad_option=2, law=law)  # type: ignore[arg-type]


def table(s_max: float, n: float = 8.0, points: int = 80) -> tuple[np.ndarray, np.ndarray]:
    s = np.concatenate([np.linspace(0.0, 0.6, 5), np.linspace(0.65, s_max, points)])
    return s * 355e6, s * 355e6 / 207e9 + 0.002 * s**n


def test_exit_domain_end_only_when_the_table_ends_before_the_cut_off() -> None:
    short = cf.TabulatedLaw.from_true(*table(1.10), 355e6, 207e9, yielding="continuous")
    long = cf.TabulatedLaw.from_true(*table(1.25), 355e6, 207e9, yielding="continuous")
    loads = cf.StressState(pm=200e6)
    tough = dict(toughness=1e12)
    r_short = cf.Assessment(PLATE, FLAW, option2_material(short).with_(**tough), loads).run()
    r_long = cf.Assessment(PLATE, FLAW, option2_material(long).with_(**tough), loads).run()
    assert r_short.exit.item() == "domain_end" and r_long.exit.item() == "cutoff"
    assert r_short.lr_max == pytest.approx(1.10) and r_long.lr_max == pytest.approx((355 + 490) / 710)
    assert "stress_strain_data_end_before_collapse_cut_off" in r_short.trace.flags
    assert "stress_strain_data_end_before_collapse_cut_off" not in r_long.trace.flags
    assert r_short.trace.options["material_law"] == "TabulatedLaw"
    assert "declared yielding: continuous" in r_short.trace.options["material_law_notes"]
    assert float(r_short.trace.options["option2_monotone_correction"]) < 1e-6


def test_option2_validity_condition_reaches_the_engine() -> None:
    elastic = option2_material(cf.PowerLaw(0.0, 1.0), su=3.3 * 355e6)  # Lr,max = 2.15, still elastic
    problem = cf.Assessment(PLATE, FLAW, elastic, cf.StressState(pm=100e6))
    with pytest.raises(cf.OutOfValidityError, match="strained plastically"):
        problem.run()
    r = problem.run(on_invalid="mask")
    assert not r.valid and not r.acceptable and np.isnan(r.radial_margin)
    assert "option2_law_too_elastic_beyond_root_two_samples_present" in r.trace.flags
    ok = cf.Assessment(PLATE, FLAW, option2_material(cf.PowerLaw(0.002, 5.0), su=3.3 * 355e6), cf.StressState(pm=100e6))
    assert ok.run().valid


def test_margin_on_the_kr_axis_uses_f_at_zero() -> None:
    # offset > 0 with n = 1 gives e/s = 1 + kappa * offset everywhere, so f(0) < 1
    material = option2_material(cf.PowerLaw(0.002, 1.0))
    r = cf.Assessment(PLATE, FLAW, material, cf.StressState(qm=150e6)).run()
    f0 = (1 + 207e9 / 355e6 * 0.002) ** -0.5
    assert r.lr == 0.0 and f0 < 0.75
    assert r.radial_margin == pytest.approx(f0 / float(r.kr), rel=1e-12)
    assert bool(r.acceptable) == (float(r.kr) < f0)


def test_tabulated_correction_limit_is_enforced() -> None:
    law = cf.TabulatedLaw.from_true([0, 200e6, 280e6, 1e9], [0, 0.001, 0.0054, 0.009], 400e6, 200e9, yielding="continuous")
    material = cf.Material(400e6, 1.2e9, 200e9, 100e6, yielding="continuous", fad_option=2, law=law)
    with pytest.raises(cf.OutOfValidityError, match="data are suspect"):
        cf.Assessment(PLATE, FLAW, material, cf.StressState(pm=100e6)).run()


def test_correction_limit_is_scanned_over_the_sampled_range_and_applied_per_sample() -> None:
    # the correction peaks at a mid-range E / yield (0.07 at kappa = 100) and is small at both ends
    stress = 400e6 * np.array([0, 0.25, 0.5, 1.4])
    law = cf.TabulatedLaw.from_true(stress, stress / 200e9 + np.array([0, 0, 0.005, 0.005]), 400e6, 200e9,
                                    yielding="continuous")
    modulus = 400e6 * np.array([1.0, 100.0, 1e8])
    material = cf.Material(400e6, 640e6, modulus, 1e12, yielding="continuous", fad_option=2, law=law)
    problem = cf.Assessment(PLATE, FLAW, material, cf.StressState(pm=100e6))
    with pytest.raises(cf.OutOfValidityError, match="data are suspect"):
        problem.run()
    r = problem.run(on_invalid="mask")
    assert list(r.valid) == [True, False, True]
    assert "option2_tabulated_correction_above_limit_samples_present" in r.trace.flags
    assert float(r.trace.options["option2_monotone_correction"]) > 0.05


def test_invalid_realisations_met_only_by_subset_simulation_still_trigger_the_validity_bracket() -> None:
    # out-of-validity mass of about 1e-4: a 2000-sample Monte Carlo channel usually sees none of it
    depth = prob.Truncated(prob.LogNormal(4e-3, 0.45, name="a"), lo=1e-4, hi=60e-3, name="a")
    problem = cf.Assessment(PLATE, cf.SurfaceFlaw(a=depth, two_c=120e-3), STEEL.with_(toughness=1e12),
                            cf.StressState(pm=20e6))
    pa = cf.ProbabilisticAssessment(problem)
    for seed in range(20):
        res = pa.solve(seed=seed, n_explore=2000, samples_per_level=1000)
        if res.reliability.monte_carlo.invalid == 0 and res.reliability.subset is not None:
            assert any(run.invalid_seen > 0 for run in res.reliability.subset.runs)
            assert res.reliability_valid_only is not None
            break
    else:  # pragma: no cover
        pytest.fail("no seed with a Monte Carlo channel free of invalid samples")


def test_array_valued_law_parameters_broadcast() -> None:
    law = cf.PowerLaw(np.array([0.001, 0.002]), 8.0, luders_strain=None)
    r = cf.Assessment(PLATE, FLAW, option2_material(law), cf.StressState(pm=250e6)).run()
    assert r.radial_margin.shape == (2,) and r.radial_margin[0] > r.radial_margin[1]
    for i, offset in enumerate((0.001, 0.002)):
        one = cf.Assessment(PLATE, FLAW, option2_material(cf.PowerLaw(offset, 8.0)), cf.StressState(pm=250e6)).run()
        assert r.radial_margin[i] == pytest.approx(float(one.radial_margin), rel=1e-12)
    assert r.trace.options["option2_monotone_correction"] == "nan"  # not reduced for per-sample laws


# ---- validation of random inputs ---------------------------------------------------------------


def test_boundary_atoms_and_strength_ordering_are_rejected_at_construction() -> None:
    with pytest.raises(ValueError):
        cf.SurfaceFlaw(a=prob.Empirical([0.0, 0.001]), two_c=0.01)
    cf.SurfaceFlaw(a=prob.Empirical([0.001, 0.002]), two_c=0.01)
    with pytest.raises(ValueError):
        cf.Ratio(prob.Empirical([1.0, 1.3]))
    with pytest.raises(ValueError):
        cf.MasterCurve(0.0, 0.0, prob.Empirical([0.5, 1.0]))
    with pytest.raises(ValueError):
        cf.MasterCurve(0.0, 0.0, prob.Uniform(0.0, 2.0))
    cf.MasterCurve(0.0, 0.0, prob.Uniform(0.0, 1.0))
    # exactly one random strength: ordering must be guaranteed by the supports
    with pytest.raises(ValueError, match="Ratio"):
        cf.Material(prob.LogNormal(380e6, 0.07), 490e6, 207e9, 1e8, yielding="continuous")
    with pytest.raises(ValueError, match="Ratio"):
        cf.Material(355e6, prob.LogNormal(490e6, 0.05), 207e9, 1e8, yielding="continuous")
    cf.Material(prob.Truncated(prob.LogNormal(380e6, 0.07), hi=480e6), 490e6, 207e9, 1e8, yielding="continuous")
    cf.Material(355e6, prob.Truncated(prob.LogNormal(490e6, 0.05), lo=360e6), 207e9, 1e8, yielding="continuous")


# ---- probabilistic assessment: invalid realisations and modes ---------------------------------


def deep_flaw_problem() -> cf.ProbabilisticAssessment:
    """Flaw depth can exceed the validity limit of the K solution (a/B < 1)."""
    depth = prob.Truncated(prob.LogNormal(12e-3, 0.6, name="a"), lo=1e-4, hi=60e-3, name="a")
    problem = cf.Assessment(PLATE, cf.SurfaceFlaw(a=depth, two_c=120e-3), STEEL.with_(toughness=400e6),
                            cf.StressState(pm=60e6))
    return cf.ProbabilisticAssessment(problem)


def test_limit_state_bounds_for_realisations_outside_validity() -> None:
    pa = deep_flaw_problem()
    u = np.random.default_rng(0).standard_normal((50_000, 1))
    failed, safe = pa.limit_state("failed")(u), pa.limit_state("safe")(u)
    assert failed.invalid is not None and 0.01 < failed.invalid.mean() < 0.5
    inv = failed.invalid
    assert np.all(failed.fail[inv]) and np.all(np.isneginf(failed.y[inv]))
    assert not np.any(safe.fail[inv]) and np.all(np.isposinf(safe.y[inv]))
    np.testing.assert_array_equal(failed.fail[~inv], safe.fail[~inv])
    np.testing.assert_array_equal(failed.y[~inv], safe.y[~inv])
    with pytest.raises(ValueError):
        pa.limit_state("ignore")  # type: ignore[arg-type]


def test_out_of_validity_mass_is_reported_from_the_main_run() -> None:
    res = deep_flaw_problem().solve(seed=0, n_explore=100_000)
    assert res.p_out_of_validity > 0.01 and res.p_out_of_validity_interval.contains(res.p_out_of_validity)
    assert res.reliability_valid_only is not None
    assert res.reliability_valid_only.guaranteed.upper < res.reliability.guaranteed.lower  # the bracket is real
    assert res.modes is not None and res.modes.out_of_validity > 0.9
    text = res.summary()
    assert "outside the validity" in text and "Counted as safe" in text
    again = deep_flaw_problem().solve(seed=0, n_explore=100_000)
    assert again.reliability.monte_carlo.failures == res.reliability.monte_carlo.failures  # reproducible


def test_mode_split_counts_cut_off_exits_as_collapse() -> None:
    load = prob.Truncated(prob.Normal(420e6, 30e6, name="pm"), lo=0.0, name="pm")
    problem = cf.Assessment(PLATE, FLAW, STEEL.with_(toughness=1e12), cf.StressState(pm=load))
    res = cf.ProbabilisticAssessment(problem).solve(seed=1, n_explore=50_000)
    assert res.reliability.status == "resolved" and res.modes is not None
    assert res.modes.collapse == pytest.approx(1.0) and res.modes.fracture == pytest.approx(0.0)
    assert res.reliability_valid_only is None and res.p_out_of_validity == 0.0


# ---- solvers -----------------------------------------------------------------------------------


def linear(beta: float):
    def g(u: np.ndarray) -> Evaluation:
        y = beta - u[:, 0]
        return Evaluation(y, y <= 0.0)

    return g


def test_inconsistent_replicates_leave_only_the_guaranteed_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    runs = tuple(subset_simulation(linear(4.0), 1, s, samples_per_level=1000) for s in range(2))
    fake = SubsetResult("replicates inconsistent", 1.325e-5, None, (1e-6, 1e-6, 1e-6, 5e-5), runs)
    monkeypatch.setattr(solvers, "subset_replicates", lambda *a, **k: fake)
    res = solvers.solve(linear(4.0), 1, seed=0, n_explore=10_000)
    assert res.status == "replicates inconsistent" and res.pf == res.monte_carlo.pf
    assert res.conditional_estimate is None and "disagree too much" in res.summary()
    # a single replicate cannot give an interval either
    one = subset_replicates(linear(4.0), 1, 0, replicates=1, samples_per_level=1000)
    assert one.status == "replicates inconsistent" and one.interval is None


def test_level_cap_propagates_to_the_combined_result(monkeypatch: pytest.MonkeyPatch) -> None:
    capped = subset_replicates(linear(12.0), 1, 0, replicates=2, samples_per_level=1000, max_levels=2)
    assert capped.status == "level cap" and np.isnan(capped.pf) and capped.interval is None
    monkeypatch.setattr(solvers, "subset_replicates", lambda *a, **k: capped)
    res = solvers.solve(linear(12.0), 1, seed=0, n_explore=10_000)
    assert res.status == "level cap reached; unresolved" and res.conditional_estimate is None


def test_replicate_interval_formula() -> None:
    x = np.array([1.0e-6, 1.4e-6, 0.8e-6, 1.1e-6, 1.3e-6])
    iv = replicate_interval(x)
    m, s = x.mean(), x.std(ddof=1)
    half = stats.t.ppf(0.995, 4) * s / (m * np.sqrt(5))
    assert iv is not None and iv.lower == pytest.approx(m * np.exp(-half), rel=1e-12)
    assert iv.upper == pytest.approx(m * np.exp(half), rel=1e-12)
    assert replicate_interval(np.array([0.5, 0.9, 0.99])).upper <= 1.0  # type: ignore[union-attr]
    assert replicate_interval(np.array([1e-6, 1e-6, 1e-6])) is None  # identical: no information on spread
    res = subset_replicates(linear(3.0), 1, 3, replicates=4, samples_per_level=1000)
    assert res.pf == pytest.approx(float(np.mean(res.replicate_estimates)), rel=1e-14)


def test_subset_simulation_bookkeeping() -> None:
    # stops at the first level when at least p0 N samples already fail, with the plain fraction
    easy = subset_simulation(linear(1.0), 1, 0, samples_per_level=2000)
    assert len(easy.level_fractions) == 1 and easy.evaluations == 2000
    assert easy.pf == pytest.approx(stats.norm.sf(1.0), rel=0.1)
    # one conditional level costs exactly N minus the number of seeds
    two = subset_simulation(linear(2.0), 1, 0, samples_per_level=2000)
    assert len(two.level_fractions) == 2 and two.samples_per_level == 2000
    assert two.evaluations == 2000 + (2000 - int(round(two.level_fractions[0] * 2000)))
    # chain-correlation factors are finite at every level, also when seeds exceed p0 N
    deep = subset_simulation(linear(4.75), 1, 0, samples_per_level=2000)
    assert len(deep.gammas) == len(deep.level_fractions) - 1 and np.all(np.isfinite(deep.gammas))
    assert np.isfinite(deep.cov_independent_levels) and deep.cov_fully_correlated >= deep.cov_independent_levels
    # p (1 - p) / N with N the samples per level, for the first level
    p0 = deep.level_fractions[0]
    assert deep._level_cov2()[0] == pytest.approx((1 - p0) / (p0 * 2000), rel=1e-12)


def test_plateau_trigger_is_half_the_population() -> None:
    def atom(mass: float):
        cut = stats.norm.ppf(mass)

        def g(u: np.ndarray) -> Evaluation:
            y = np.where(u[:, 0] < cut, 0.7, 6.0 - u[:, 0])  # `mass` of the samples share the lowest value
            return Evaluation(y, y <= 0.0)

        return g

    assert subset_simulation(atom(0.7), 1, 0, samples_per_level=2000).status == "plateau"
    # an atom of 30% at the lowest value is taken in one level (counted fraction 0.3), not a plateau
    run = subset_simulation(atom(0.3), 1, 1, samples_per_level=2000, max_levels=1)
    assert run.status in ("level cap", "plateau") and (not run.level_fractions or run.level_fractions[0] <= 0.5)


def test_driving_variable_check_is_two_sided_and_rejects_nan() -> None:
    with pytest.raises(RuntimeError):
        solvers.consistent_driving_variable(np.array([-0.5]), np.array([False]))
    with pytest.raises(RuntimeError):
        solvers.consistent_driving_variable(np.array([np.nan]), np.array([False]))


# ---- small pinned values -----------------------------------------------------------------------


def test_pinned_details() -> None:
    assert cf.toughness.ctod_constraint_factor(0.98e8, 1e8) == 1.5  # the correlation needs sy/su < 0.98
    assert cf.toughness.ctod_constraint_factor(0.979e8, 1e8) == pytest.approx(1.517 * 0.979**-0.3188)
    # Master Curve cap at twice the thickness, against an independent expression
    capped = cf.toughness.master_curve(0.5, 0.0, 0.0, 0.2, 0.03)
    assert capped == pytest.approx((20 + 88 * (25 / 60) ** 0.25 * np.log(2) ** 0.25) * MPA, rel=1e-12)
    values = np.array([110.0, 95.0, 130.0, 120.0, 105.0]) * MPA
    log_bound = cf.toughness.lower_bound_toughness(values, lognormal=True)
    logs = np.log(values)
    assert log_bound == pytest.approx(np.exp(logs.mean() - cf.toughness.tolerance_factor(5) * logs.std(ddof=1)))
    assert 50 * MPA < log_bound < 95 * MPA
    with pytest.raises(ValueError, match="lognormal=True"):
        cf.toughness.lower_bound_toughness(np.array([5.0, 100.0, 300.0]) * MPA)
    with pytest.raises(ValueError):
        cf.toughness.charpy_lower_bound(0.5, 0.001)  # (12 sqrt(0.5) - 20)(25/1)^0.25 + 20 < 0
    with pytest.raises(ValueError):
        cf.toughness.tolerance_factor(5, percentile=1.5)
    # normal-lognormal correlation has a closed form: rho0 * sigma_ln / sqrt(exp(sigma_ln^2) - 1)
    ln = prob.LogNormal(100.0, 0.5)
    exact = 0.6 * ln.sigma_ln / np.sqrt(np.expm1(ln.sigma_ln**2))
    assert prob.latent_to_pearson(prob.Normal(0, 1), ln, 0.6) == pytest.approx(exact, abs=1e-9)
    # SORM is refused when the curvature term is small, not only when it is negative
    res = prob.FormResult(3.0, np.array([3.0, 0.0]), np.array([-1.0, 0.0]), float(stats.norm.cdf(-3)), 1, 1)
    with pytest.raises(prob.FormNotApplicable):
        prob.sorm_breitung(lambda u: 3.0 - u[:, 0] - 0.5 * 0.2 * u[:, 1] ** 2, res)  # 1 + beta kappa = 0.4
    two = prob.form(lambda u: np.minimum(3.0 - u[:, 0], 3.2 - u[:, 1]), 2, starts=16, seed=1)
    assert two.competing_design_points >= 1


def test_correction_is_evaluated_over_each_realisations_own_domain() -> None:
    sy, e = 400e6, 400e9
    stress = sy * np.array([0, 0.5, 1.1, 2.2])
    law = cf.TabulatedLaw.from_true(stress, stress / e + np.array([0, 0, 0.005, 0.005]), sy, e, yielding="continuous")
    # cut-offs 1.01 and 2.0: the first realisation's line never reaches the region that needs correcting
    material = cf.Material(sy, np.array([408e6, 1200e6]), e, 1e12, yielding="continuous", fad_option=2, law=law)
    r = cf.Assessment(PLATE, FLAW, material, cf.StressState(pm=50e6)).run(on_invalid="mask")
    assert list(r.valid) == [True, False]
    for i, su in enumerate((408e6, 1200e6)):
        one = cf.Assessment(PLATE, FLAW, material.with_(uts=su), cf.StressState(pm=50e6)).run(on_invalid="mask")
        assert bool(one.valid) == bool(r.valid[i])
    curve = cf.Option2Curve(law, sy, np.array([408e6, 1200e6]), e)
    correction = curve.correction()
    assert correction[0] == 0.0 and correction[1] == pytest.approx(0.0837, abs=2e-3)
    # a step beyond the cut-off is outside the assessment domain and must not be counted
    knots = np.array([0, 0.5, 1, 1.2, 1.2, 1.8, 1.8, 2.0])
    stepped = cf.TabulatedLaw(knots, np.array([0, 0, 0, 0, 0.005, 0.005, 0.01, 0.01]), None, ())
    assert stepped.step_locations == (1.2, 1.8)
    inside = cf.Material(sy, 480e6, e, 1e12, yielding="continuous", fad_option=2, law=stepped)  # cut-off 1.1
    ok = cf.Assessment(PLATE, FLAW, inside, cf.StressState(pm=50e6)).run()
    assert ok.valid and float(ok.trace.options["option2_monotone_correction"]) == 0.0


def test_correction_uses_each_realisations_own_stiffness_ratio() -> None:
    # widely spread E / yield: no interpolation between realisations may hide a large correction
    sy = 400e6
    stress = sy * np.array([0, 0.25, 0.5, 1.4])
    law = cf.TabulatedLaw.from_true(stress, stress / 200e9 + np.array([0, 0, 0.005, 0.005]), sy, 200e9,
                                    yielding="continuous")
    moduli = sy * np.array([1.0, 1000.0, 1e240])
    batch = cf.Option2Curve(law, sy, 640e6, moduli).correction()
    for i, modulus in enumerate(moduli):
        assert batch[i] == pytest.approx(float(cf.Option2Curve(law, sy, 640e6, float(modulus)).correction()), rel=1e-12)
    assert batch[1] == pytest.approx(0.1458, abs=2e-3) and batch[0] < 0.004 and batch[2] < 0.004
    material = cf.Material(sy, 640e6, moduli, 1e12, yielding="continuous", fad_option=2, law=law)
    r = cf.Assessment(PLATE, FLAW, material, cf.StressState(pm=50e6)).run(on_invalid="mask")
    assert list(r.valid) == [True, False, True]
