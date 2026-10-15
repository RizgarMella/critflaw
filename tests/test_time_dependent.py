"""Phase 3, Part A: time-dependent reliability, inspection and the component update
(design note phase 3, F10 to F17; benchmarks 10 to 17)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate

import critflaw as cf
from critflaw import prob
from critflaw.fatigue import ExtremeLoad, GrowthLaw, Inspection, Spectrum, TimeDependentAssessment, steel_in_air
from critflaw.fatigue.laws import coefficient_to_si
from critflaw.prob.inspection import StepPoD

B, W = 25e-3, 0.5
STEEL = cf.Material(355e6, 490e6, 207e9, 90e6, yielding="continuous")
NO_GROWTH = GrowthLaw(1e-40, 3.0, 1e9)  # threshold far above any range: a dormant flaw
SPECTRUM = Spectrum.repeating([(50e6, 0.0, 1.0)], cycles_per_year=1e4)


def extreme(mean: float = 150e6, sd: float = 15e6) -> ExtremeLoad:
    return ExtremeLoad(prob.Truncated(prob.Gumbel.from_mean_sd(mean, sd), lo=0.0), 1.0)


def critical_load(problem: cf.Assessment, lo: float = 1e6, hi: float = 2e9) -> float:
    """Membrane stress at which the (deterministic) assessment stops being acceptable."""
    return float(critical_loads(problem, 1, lo, hi)[0])


def critical_loads(problem: cf.Assessment, n: int, lo: float = 1e6, hi: float = 2e9) -> np.ndarray:
    """Vectorised: the problem's fields may be arrays of length n."""
    lo_, hi_ = np.full(n, lo), np.full(n, hi)
    for _ in range(60):
        mid = 0.5 * (lo_ + hi_)
        ok = problem.with_(loads=problem.loads.with_(pm=mid)).run(on_invalid="mask").acceptable
        lo_, hi_ = np.where(ok, mid, lo_), np.where(ok, hi_, mid)
    return 0.5 * (lo_ + hi_)


def test_no_growth_with_the_load_as_the_only_random_variable() -> None:
    # benchmark 10: cumulative 1 - (1 - p)^k with constant hazard p; both pairings identical
    problem = cf.Assessment(cf.FlatPlate(B, W), cf.SurfaceFlaw(6e-3, 60e-3), STEEL, cf.StressState(pm=100e6))
    crit = critical_load(problem)
    ext = extreme(0.88 * crit, 0.1 * crit)
    p = float(ext.distribution.sf(crit))
    assert 0.02 < p < 0.3
    run = TimeDependentAssessment(problem, NO_GROWTH, SPECTRUM, ext, times=range(1, 11), leak_is_failure=False)
    assert run.dimension == 10  # no time-invariant random variable, one per step
    res = run.solve(seed=0, samples=20_000)
    k = np.arange(1, 11)
    exact = 1 - (1 - p) ** k
    for i in range(10):
        assert res.cumulative_upper.lower[i] <= exact[i] <= res.cumulative_upper.upper[i]
        assert res.cumulative_lower.value[i] == res.cumulative_upper.value[i]  # no growth: no discordance
        assert res.hazard_upper.lower[i] <= p <= res.hazard_upper.upper[i]
        # without growth the conservative pair collapses onto the within-pairing value
        assert res.hazard_conservative_lower[i] == pytest.approx(res.hazard_upper.value[i], abs=1e-12)
        assert res.hazard_conservative_upper[i] == pytest.approx(res.hazard_upper.value[i], abs=1e-12)
    assert np.all(res.discordant == 0) and not np.any(res.refine) and res.assumption_violations == 0
    assert res.modes["fracture"] + res.modes["collapse"] == pytest.approx(res.cumulative_upper.value[-1])
    assert res.events == {
        "leak": 0.0,
        "leak_assumed": 0.0,
        "surface_breaking": 0.0,
        "ended_outside_validity": 0.0,
        "repaired": 0.0,
    }
    assert res.events_any == {
        **res.events,
        "removed_flaw_had_leaked": 0.0,
        "removed_flaw_had_ended": 0.0,
        "by_trigger": {},
    }
    assert "assumption A" in res.summary()


def test_no_growth_with_a_random_resistance_is_not_the_independent_formula() -> None:
    # benchmark 11: E[1 - (1 - p(R))^k] by quadrature over the toughness; the hazard falls with time
    kmat = prob.LogNormal.from_mean_sd(90e6, 20e6, name="kmat")
    steel = cf.Material(355e6, 490e6, 207e9, kmat, yielding="continuous")
    problem = cf.Assessment(cf.FlatPlate(B, W), cf.SurfaceFlaw(6e-3, 60e-3), steel, cf.StressState(pm=100e6))
    crit_median = critical_load(problem.with_(material=steel.with_(toughness=90e6)))
    ext = extreme(0.85 * crit_median, 0.1 * crit_median)
    run = TimeDependentAssessment(problem, NO_GROWTH, SPECTRUM, ext, times=range(1, 21), leak_is_failure=False)
    res = run.solve(seed=3, samples=20_000)
    # reference: p(kmat) on a quantile grid of kmat, then the expectation
    q = (np.arange(4000) + 0.5) / 4000
    crit = critical_loads(problem.with_(material=steel.with_(toughness=kmat.ppf(q))), 4000)
    p_r = ext.distribution.sf(crit)
    years = np.arange(1, 21)
    exact = np.array([np.mean(1 - (1 - p_r) ** k) for k in years])
    independent = 1 - (1 - np.mean(p_r)) ** years
    assert independent[-1] > 1.1 * exact[-1]  # the independent formula over-predicts
    for i in (0, 9, 19):  # pointwise 99% intervals: a few points, not all twenty
        assert res.cumulative_upper.lower[i] <= exact[i] <= res.cumulative_upper.upper[i]
    hazard_exact = np.array([np.mean(p_r * (1 - p_r) ** (k - 1)) / np.mean((1 - p_r) ** (k - 1)) for k in years])
    assert hazard_exact[-1] < 0.8 * hazard_exact[0]
    assert res.hazard_upper.lower[0] <= hazard_exact[0] <= res.hazard_upper.upper[0]
    assert res.hazard_upper.lower[-1] <= hazard_exact[-1] <= res.hazard_upper.upper[-1]


def test_growth_brackets_the_continuous_time_limit() -> None:
    # benchmark 12: deterministic growth, random annual maxima; the pairings bracket
    # 1 - exp(integral of ln F(x_crit(s)) ds), and the bracket narrows with a finer step
    law = steel_in_air()
    sp = Spectrum.repeating([(80e6, 0.0, 1.0)], cycles_per_year=3e4)
    problem = cf.Assessment(cf.FlatPlate(B, W), cf.SurfaceFlaw(3e-3, 30e-3), STEEL, cf.StressState(pm=100e6))
    crit0 = critical_load(problem)
    ext = extreme(0.75 * crit0, 0.08 * crit0)
    run = TimeDependentAssessment(problem, law, sp, ext, times=range(1, 6), leak_is_failure=False)
    # the deterministic growth path from the integrator itself, at a fine grid
    from critflaw.fatigue import Grower, GrowthState

    st = GrowthState.start(np.array([3e-3]), np.array([15e-3]), np.array([B]), np.array([W]), 1)
    g = Grower(law, sp, 1, 5.0)
    t_grid = np.linspace(0.0, 5.0, 251)
    ln_f = []
    for t in t_grid:
        if t > 0:
            g.advance(st, float(t))
        assert st.kind[0] == 0, "the case must stay a surface flaw"
        crit = critical_load(problem.with_(flaw=cf.SurfaceFlaw(float(st.a[0]), 2 * float(st.c[0]))))
        ln_f.append(float(np.log(ext.distribution.cdf(crit))))
    limit = 1 - np.exp(integrate.trapezoid(ln_f, t_grid))
    assert 0.03 < limit < 0.9
    res = run.solve(seed=2, samples=20_000)
    assert res.cumulative_lower.lower[-1] <= limit <= res.cumulative_upper.upper[-1]
    assert res.cumulative_lower.value[-1] < res.cumulative_upper.value[-1]
    fine = TimeDependentAssessment(problem, law, sp, ext, times=range(1, 6), leak_is_failure=False, delta=0.25)
    res_fine = fine.solve(seed=2, samples=20_000)
    assert res_fine.cumulative_lower.lower[-1] <= limit <= res_fine.cumulative_upper.upper[-1]
    width_coarse = res.cumulative_upper.value[-1] - res.cumulative_lower.value[-1]
    width_fine = res_fine.cumulative_upper.value[-1] - res_fine.cumulative_lower.value[-1]
    assert width_fine < 0.5 * width_coarse  # first order in the step


def test_leak_definition_moves_the_leak_mass_exactly() -> None:
    law = steel_in_air()
    sp = Spectrum.repeating([(120e6, 20e6, 1.0)], cycles_per_year=1e5)
    a0 = prob.Truncated(prob.LogNormal.from_mean_sd(2e-3, 0.8e-3), hi=10e-3, name="a0")
    problem = cf.Assessment(cf.FlatPlate(B, W), cf.SurfaceFlaw(a0, 12e-3), STEEL, cf.StressState(pm=50e6))
    ext = extreme(30e6, 3e6)  # a low load: leaks happen before fracture for most samples
    kw = dict(times=range(1, 6), inspections=())
    without = TimeDependentAssessment(problem, law, sp, ext, leak_is_failure=False, **kw).solve(seed=3, samples=3000)
    with_leak = TimeDependentAssessment(problem, law, sp, ext, leak_is_failure=True, **kw).solve(seed=3, samples=3000)
    assert 0.1 < without.events_any["leak"] and with_leak.events_any == without.events_any  # same paths
    # a through flaw that grows to the width limit is out of validity: counted as failed in the headline,
    # as safe in the valid-only curve; the leak definition is seen on the valid-only curve
    assert without.cumulative_valid_only.value[-1] < 0.9
    assert with_leak.cumulative_valid_only.value[-1] > without.cumulative_valid_only.value[-1]
    assert with_leak.modes["leak"] > 0.0 and without.modes.get("leak", 0.0) == 0.0
    # the whole leak mass is in the headline when leak is failure
    assert with_leak.cumulative_upper.value[-1] >= with_leak.events["leak"] - 1e-12
    assert with_leak.cumulative_valid_only.value[-1] >= with_leak.events["leak"] - 1e-12
    with pytest.raises(ValueError, match="leak"):
        TimeDependentAssessment(problem, law, sp, ext, leak_is_failure=True, **kw).solve(
            seed=0, samples=50, subset=True
        )
    with pytest.raises(ValueError, match="explicitly"):
        TimeDependentAssessment(problem, law, sp, ext, leak_is_failure=None, **kw)  # type: ignore[arg-type]


def two_point_problem() -> tuple[cf.Assessment, ExtremeLoad, GrowthLaw, Spectrum]:
    """Two equally likely initial flaws; deterministic growth; a nearly deterministic load that fails
    the large flaw once it has grown (about year 7) and never the small one within 10 years."""
    a0 = prob.Empirical([1.0e-3, 4.0e-3], name="a0")
    problem = cf.Assessment(cf.FlatPlate(B, W), cf.SurfaceFlaw(a0, 40e-3), STEEL, cf.StressState(pm=100e6))
    law = GrowthLaw(float(coefficient_to_si(5.86e-13, 2.88)), 2.88, 0.0)
    sp = Spectrum.repeating([(70e6, 0.0, 1.0)], cycles_per_year=1e5)  # large flaw: 8 mm at 4.5 years; small: never
    # the load fails a flaw once it is 8 mm deep (the large flaw gets there in a few years)
    crit = critical_load(problem.with_(flaw=cf.SurfaceFlaw(8e-3, 40e-3)))
    ext = ExtremeLoad(prob.Uniform(crit, crit * (1 + 1e-6)), 1.0)
    return problem, ext, law, sp


def test_inspection_with_repair_on_a_two_point_population() -> None:
    # benchmark 16 (hand calculation): without inspection half the population fails at the year the
    # large flaw reaches its critical size; a step PoD at year 3 that sees the large flaw only, with
    # repair, removes that failure
    problem, ext, law, sp = two_point_problem()
    plain = TimeDependentAssessment(problem, law, sp, ext, times=range(1, 11), leak_is_failure=False)
    res = plain.solve(seed=0, samples=2000)
    assert res.cumulative_upper.value[-1] == pytest.approx(0.5, abs=0.03)
    fail_year = int(np.argmax(res.cumulative_upper.value > 0.1)) + 1
    assert 3 < fail_year <= 10
    insp = Inspection(3.0, StepPoD(2.5e-3), action="repair")
    repaired = TimeDependentAssessment(
        problem, law, sp, ext, times=range(1, 11), leak_is_failure=False, inspections=[insp]
    ).solve(seed=0, samples=2000)
    assert repaired.events["repaired"] == pytest.approx(0.5, abs=0.03)
    assert repaired.cumulative_upper.value[-1] == 0.0
    assert repaired.inspections[0]["p_detect_all"] == pytest.approx(0.5, abs=0.03)
    recorded = TimeDependentAssessment(
        problem,
        law,
        sp,
        ext,
        times=range(1, 11),
        leak_is_failure=False,
        inspections=[Inspection(3.0, StepPoD(2.5e-3), action="none")],
    ).solve(seed=0, samples=2000)
    assert recorded.cumulative_upper.value[-1] == pytest.approx(0.5, abs=0.03) and recorded.events["repaired"] == 0.0
    with pytest.raises(ValueError, match="boundaries"):
        TimeDependentAssessment(
            problem,
            law,
            sp,
            ext,
            times=range(1, 11),
            leak_is_failure=False,
            inspections=[Inspection(3.5, StepPoD(2.5e-3))],
        )


def test_component_update_from_an_observed_inspection_outcome() -> None:
    # benchmark 17: PoD 0.5 for both flaws (ConstantPoD); without the update, half the large flaws are repaired
    problem, ext, law, sp = two_point_problem()
    from critflaw.prob.inspection import ConstantPoD

    pod = ConstantPoD(0.5)
    base = TimeDependentAssessment(
        problem, law, sp, ext, times=range(1, 11), leak_is_failure=False, inspections=[Inspection(3.0, pod)]
    ).solve(seed=0, samples=4000)
    assert base.cumulative_upper.value[-1] == pytest.approx(0.25, abs=0.03)
    # observed "not detected": no repair happened; the large flaw fails for certain: Pf after = 0.5
    nd = TimeDependentAssessment(
        problem,
        law,
        sp,
        ext,
        times=range(1, 11),
        leak_is_failure=False,
        inspections=[Inspection(3.0, pod, observed="not_detected")],
    ).solve(seed=0, samples=4000)
    assert nd.conditioning is not None and not nd.weighted
    assert nd.conditioning["pf_after_upper"] == pytest.approx(0.5, abs=0.03)
    assert nd.conditioning["p_conditioning_event_upper"] == pytest.approx(0.5, abs=0.03)
    assert nd.conditioning["interval_upper"].contains(0.5)
    lo, up = nd.conditioning["conservative_pair"]
    # with a deterministic load both pairings coincide, and so does the conservative pair
    assert lo == pytest.approx(nd.conditioning["pf_after_upper"], abs=1e-12) and up == pytest.approx(lo, abs=1e-12)
    # observed "detected" (and repaired): Pf after = 0
    det = TimeDependentAssessment(
        problem,
        law,
        sp,
        ext,
        times=range(1, 11),
        leak_is_failure=False,
        inspections=[Inspection(3.0, pod, observed="detected")],
    ).solve(seed=0, samples=4000)
    assert det.conditioning is not None and det.conditioning["pf_after_upper"] == 0.0
    # a sized outcome: weighted and conditional; a measured size near the large flaw puts the mass there
    sized = TimeDependentAssessment(
        problem,
        law,
        sp,
        ext,
        times=range(1, 11),
        leak_is_failure=False,
        inspections=[
            Inspection(3.0, pod, sizing_error=prob.Normal(0.0, 0.3e-3), action="none", observed=("sized", 4.3e-3))
        ],
    ).solve(seed=0, samples=4000)
    assert sized.weighted and sized.conditioning is not None
    value = sized.conditioning["pf_after_upper"]
    assert value is None or value > 0.95


def test_growth_law_coefficients_are_random_variables_of_the_run() -> None:
    law = steel_in_air(curve="mean", stage_correlation=0.5)
    sp = Spectrum.repeating([(80e6, 10e6, 1.0)], cycles_per_year=5e4)
    problem = cf.Assessment(cf.FlatPlate(B, W), cf.SurfaceFlaw(2e-3, 20e-3), STEEL, cf.StressState(pm=100e6))
    run = TimeDependentAssessment(problem, law, sp, extreme(), times=range(1, 4), leak_is_failure=False)
    assert run.n_invariant == 2 and run.dimension == 5
    assert run.vector is not None and run.vector.latent_correlation[0, 1] == pytest.approx(0.5)
    res = run.solve(seed=0, samples=500)
    assert "stage correlation: 0.5" in "\n".join(res.assumptions)
    with pytest.raises(ValueError, match="shared_scale"):
        TimeDependentAssessment(
            problem, law, sp.with_(load_scale=prob.LogNormal(1.0, 0.1)), extreme(), times=[1, 2], leak_is_failure=False
        )
