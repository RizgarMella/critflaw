"""End-to-end probabilistic assessment through the deterministic engine."""

from __future__ import annotations

import numpy as np
import pytest

import critflaw as cf
from critflaw import prob
from critflaw.probabilistic import collect, realise

PLATE = cf.FlatPlate(thickness=25e-3, width=500e-3)


def six_variable_problem() -> tuple[cf.Assessment, dict[str, prob.Distribution]]:
    v = {
        "sy": prob.LogNormal(380e6, 0.07, name="sy"),
        "ratio": prob.Truncated(prob.LogNormal(1.35, 0.04), lo=1.02, name="uts/sy"),
        "kmat": prob.Weibull.from_mean_sd(110e6, 25e6, name="kmat"),
        "a": prob.Truncated(prob.LogNormal(4e-3, 0.4), lo=1e-4, hi=20e-3, name="a"),
        "pm": prob.Truncated(prob.Gumbel.from_mean_sd(170e6, 20e6), lo=0.0, name="pm"),
        "k_model": prob.LogNormal(1.0, 0.1, name="k_model"),
    }
    material = cf.Material(v["sy"], cf.Ratio(v["ratio"]), 207e9, v["kmat"], yielding="continuous")
    problem = cf.Assessment(
        PLATE,
        cf.SurfaceFlaw(a=v["a"], two_c=60e-3),
        material,
        cf.StressState(pm=v["pm"], residual=cf.Residual.as_welded()),
        k_model=v["k_model"],
    )
    return problem, v


def test_collect_and_realise() -> None:
    problem, v = six_variable_problem()
    found = collect(problem)
    assert len(found) == 6 and all(any(f is d for d in v.values()) for f in found)
    values = {id(d): np.full(3, d.mean()) for d in found}
    realised = realise(problem, values)
    assert collect(realised) == [] and realised.random_fields() == []
    assert np.allclose(realised.material.tensile_strength(), v["sy"].mean() * v["ratio"].mean())
    # one object used twice is one variable
    shared = prob.LogNormal(1e8, 0.1)
    twice = cf.Assessment(PLATE, cf.SurfaceFlaw(3e-3, 15e-3),
                          cf.Material(355e6, 490e6, 207e9, 1e8, yielding="continuous"), cf.StressState(pm=shared, qm=shared))
    assert len(collect(twice)) == 1


def test_a_problem_with_random_variables_cannot_be_run_deterministically() -> None:
    problem, _ = six_variable_problem()
    assert "material.yield_strength" in problem.random_fields() and "assessment.k_model" in problem.random_fields()
    with pytest.raises(TypeError, match="ProbabilisticAssessment"):
        problem.run()
    with pytest.raises(ValueError, match="no random variables"):
        cf.ProbabilisticAssessment(realise(problem, {id(d): np.array([d.mean()]) for d in collect(problem)}))


def test_unbounded_distributions_on_positive_quantities_are_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="truncation explicitly"):
        cf.SurfaceFlaw(a=prob.Normal(4e-3, 1e-3), two_c=60e-3)
    with pytest.raises(ValueError, match="truncate"):
        cf.StressState(pm=prob.Normal(150e6, 30e6))
    cf.StressState(pm=prob.Truncated(prob.Normal(150e6, 30e6), lo=0.0))  # the explicit form is accepted
    with pytest.raises(ValueError, match="Ratio"):
        cf.Material(prob.LogNormal(380e6, 0.07), prob.LogNormal(520e6, 0.05), 207e9, 1e8, yielding="continuous")
    with pytest.raises(ValueError, match="above 1"):
        cf.Ratio(prob.LogNormal(1.3, 0.2))


def test_monte_carlo_through_the_engine_matches_a_direct_vectorised_run() -> None:
    problem, v = six_variable_problem()
    pa = cf.ProbabilisticAssessment(problem)
    assert pa.names == ("a", "sy", "uts/sy", "kmat", "pm", "k_model") or set(pa.names) == set(d.name for d in v.values())
    rng = np.random.default_rng(11)
    u = rng.standard_normal((200_000, 6))
    ev = pa.limit_state("failed")(u)
    # the same samples run through the engine by hand
    x = {d.name: d.from_normal_score(u[:, i]) for i, d in enumerate(pa.variables)}
    direct = cf.Assessment(
        PLATE,
        cf.SurfaceFlaw(a=x["a"], two_c=60e-3),
        cf.Material(x["sy"], x["sy"] * x["uts/sy"], 207e9, x["kmat"], yielding="continuous"),
        cf.StressState(pm=x["pm"], residual=cf.Residual.as_welded()),
        k_model=x["k_model"],
    ).run(on_invalid="mask")
    assert np.array_equal(ev.fail, ~direct.acceptable)
    assert 1e-3 < ev.fail.mean() < 0.5
    # driving variable agrees in sign with the indicator
    assert np.all(ev.y[ev.fail] < 0) and np.all(ev.y[~ev.fail] > 0)


def test_solve_resolved_and_conditional_agree_with_a_large_reference_run() -> None:
    problem, v = six_variable_problem()
    pa = cf.ProbabilisticAssessment(problem)
    reference = prob.solvers.monte_carlo(pa.limit_state("failed"), 6, 400_000, seed=99)
    assert reference.failures > 500
    res = pa.solve(seed=1, n_explore=200_000)
    assert res.reliability.status == "resolved" and res.reliability.guaranteed.contains(reference.pf)
    text = res.summary()
    assert "per flaw" in text and "Failure modes" in text and res.modes is not None
    assert res.modes.fracture + res.modes.collapse + res.modes.out_of_validity == pytest.approx(1.0)
    # a tougher, lightly loaded variant: rare, so the result is conditional and carries both statements
    rare = problem.with_(loads=cf.StressState(pm=prob.Truncated(prob.Gumbel.from_mean_sd(90e6, 9e6), lo=0.0)))
    rare_pa = cf.ProbabilisticAssessment(rare)
    big = prob.solvers.monte_carlo(rare_pa.limit_state("failed"), 6, 1_000_000, seed=5)
    out = rare_pa.solve(seed=2, n_explore=20_000, samples_per_level=2000)
    assert out.reliability.guaranteed.contains(big.pf)
    if out.reliability.status == "conditional":
        assert big.interval.lower * 0.3 < out.reliability.pf < big.interval.upper * 3.0
        assert "cannot be excluded" in out.summary()


def test_correlation_between_problem_variables() -> None:
    problem, v = six_variable_problem()
    pa = cf.ProbabilisticAssessment(problem).correlate(v["sy"], v["kmat"], -0.5)
    u = np.random.default_rng(0).standard_normal((100_000, 6))
    realised = pa.realise(u)
    corr = np.corrcoef(np.asarray(realised.material.yield_strength), np.asarray(realised.material.toughness))[0, 1]
    assert corr == pytest.approx(-0.5, abs=0.01)


def test_option2_and_master_curve_through_the_probabilistic_engine() -> None:
    sy = prob.LogNormal(380e6, 0.07, name="sy")
    p = prob.Uniform(0.0, 1.0, name="mc_p")
    t0 = prob.Normal(-60.0, 15.0, name="T0")
    n = float(cf.hardening_exponent_lower_bound(380e6, 520e6))
    material = cf.Material(
        sy, cf.Ratio(1.37), 207e9, cf.MasterCurve(t0=t0, temperature=-10.0, probability=p),
        yielding="continuous", fad_option=2, law=cf.PowerLaw(0.002, n),
    )
    problem = cf.Assessment(PLATE, cf.SurfaceFlaw(5e-3, 40e-3), material, cf.StressState(pm=180e6, qm=100e6))
    pa = cf.ProbabilisticAssessment(problem)
    u = np.random.default_rng(4).standard_normal((50_000, 3))
    r = pa.evaluate(u)
    assert r.valid.all() and "toughness.master_curve" in r.trace.equations and "fad.option2" in r.trace.equations
    # toughness never below the Master Curve lower bound and adjusted for the 40 mm crack front
    x = pa.vector.to_physical(u)
    names = list(pa.names)
    expected = cf.toughness.master_curve(x[:, names.index("mc_p")], -10.0, x[:, names.index("T0")], 40e-3, 25e-3)
    assert expected.min() >= cf.toughness.K_MIN_MASTER_CURVE
    one = realise(problem, {id(v): x[:1, i] for i, v in enumerate(pa.variables)}).run()
    kr_secondary_free = float(one.k_primary[0][0] + one.k_secondary[0][0]) / float(expected[0])
    assert float(one.kr_by_position[0][0]) >= kr_secondary_free  # rho >= 0 on top of (Kp + Ks) / Kmat
    assert 0.0 < (~r.acceptable).mean() < 1.0


def test_fad_model_factor_is_applied_as_an_equivalent_acceptance_test() -> None:
    steel = cf.Material(355e6, 490e6, 207e9, 80e6, yielding="continuous")
    base = cf.Assessment(PLATE, cf.SurfaceFlaw(6e-3, 40e-3), steel, cf.StressState(pm=200e6))
    r0 = base.run()
    for m in (0.8, 1.0, 1.3, 2.0):
        r = base.with_(fad_model=m).run()
        curve = base.curve()
        direct = bool(float(r0.kr) < m * float(curve(float(r0.lr))) and float(r0.lr) < float(curve.lr_max))
        assert bool(r.acceptable) == direct
        # the result reports the actual Kr; the factor enters the acceptance test and the margin
        assert r.kr == pytest.approx(float(r0.kr), rel=1e-14) and r.lr == pytest.approx(float(r0.lr))
        if float(r0.lr) == 0.0:
            assert r.radial_margin == pytest.approx(m * float(r0.radial_margin), rel=1e-9)
    # a zero-Lr point with Kr above 1 is acceptable when the factor is large enough (reviewer's case)
    sec = cf.Assessment(PLATE, cf.SurfaceFlaw(6e-3, 40e-3), steel.with_(toughness=20e6), cf.StressState(qm=300e6))
    r1 = sec.run()
    assert float(r1.kr) > 1.0 and not r1.acceptable
    r2 = sec.with_(fad_model=float(r1.kr) * 1.5).run()
    assert r2.acceptable and r2.radial_margin == pytest.approx(1.5, rel=1e-9)
    with pytest.raises(ValueError):
        base.with_(fad_model=0.0)
