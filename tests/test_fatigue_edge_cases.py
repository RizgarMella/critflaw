"""Phase 3, Part A: tests added for the round-1 code review findings (findings P-xx and C3-xx)."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from scipy import stats

import critflaw as cf
from critflaw import prob
from critflaw.fatigue import ExtremeLoad, Grower, GrowthLaw, GrowthState, Inspection, Spectrum, TimeDependentAssessment
from critflaw.fatigue import growth as G
from critflaw.fatigue.laws import coefficient_to_si, dk_to_si, steel_in_air
from critflaw.prob.inspection import ConstantPoD

HERE = Path(__file__).parent
NMM = float(dk_to_si(1.0))
B, W = 25e-3, 0.5
STEEL = cf.Material(355e6, 490e6, 207e9, 90e6, yielding="continuous")


def _load(name: str) -> ModuleType:
    module = ModuleType(name)
    exec(
        compile((HERE / "oracles" / f"{name}.py").read_text(encoding="utf-8"), name, "exec"), module.__dict__
    )  # reviewed
    return module


# ---- the partner's fixed-step integrator (benchmark 5, 1e-8 row) ------------------------------


def fixed_heun(
    law: GrowthLaw, sp: Spectrum, a0: float, c0: float, thickness: float, width: float, h: float, n: int, kind: int
) -> tuple[float, float]:
    """The production Heun step applied at a fixed step with the active set frozen (event-free case)."""
    st = GrowthState.start(np.array([a0]), np.array([c0]), np.array([thickness]), np.array([width]), 1)
    st.kind[:] = kind
    g = Grower(law, sp, 1, 1.0)
    idx = np.array([0])
    for _ in range(n):
        dk = g.ranges(idx, st.a, st.c, st.kind, st.thickness, st.width)
        above, stage_b, _ = g.active_set(idx, dk, st.a, st.c, st.kind)
        va0, vc0, _ = g.velocities(idx, st.a, st.c, st.kind, st.t, st.thickness, st.width, above, stage_b)
        _, _, a2, c2 = g.heun(st, idx, np.array([h]), above, stage_b, (va0, vc0))
        st.a[:], st.c[:] = a2, c2
        st.t += h
    return float(st.a[0]), float(st.c[0])


def test_partner_fixed_step_integrator_agrees_to_1e_8() -> None:
    oracle = _load("partner_reference_fatigue_integrator")
    table = {r["case"]: r for r in oracle.tabulate()}
    cases = {
        "case_1": (2e-3, 5e-3, 25e-3, 0.5, 100e6, 20e6, 1.4097e-28, 2.88, 2e5, 1e-3, 500),
        "case_2": (1e-3, 6e-3, 12e-3, 0.3, 120e6, 0.0, 1.4097e-28, 2.88, 1e5, 5e-3, 400),
        "case_3": (3e-3, 3e-3, 25e-3, 0.5, 80e6, 40e6, 1.4097e-28, 2.88, 5e4, 2e-3, 1000),
    }
    for name, (a0, c0, thick, width, dsm, dsb, a_si, m, nu, h, n) in cases.items():
        law = GrowthLaw(a_si, m, 0.0)
        sp = Spectrum.repeating([(dsm, dsb, 1.0)], nu)
        a, c = fixed_heun(law, sp, a0, c0, thick, width, h, n, G.SURFACE)
        want = table[name]["value"]
        assert table[name]["all_applicable_end_state_limits_satisfied"]
        assert a == pytest.approx(want[0], rel=1e-8) and c == pytest.approx(want[1], rel=1e-8)
    law = GrowthLaw(1.64754666094773e-29, 3.0, 0.0)
    _, c = fixed_heun(law, Spectrum.repeating([(100e6, 0.0, 1.0)], 1e6), 1.0, 5e-3, 1.0, 0.5, 1e-3, 100, G.THROUGH)
    assert c == pytest.approx(table["case_4"]["value"][0], rel=1e-8)


# ---- integrator events (P-01, P-12, P-13, C3-03, C3-08, C3-09, C3-11, benchmark 9) ------------


def test_threshold_cap_event_precedes_breakthrough() -> None:
    # P-01: the 1 mm cap raises the threshold above every range; the flaw must arrest at 1 mm, not break through
    law = GrowthLaw(1e-30, 3.0, 1e12, shallow_cap=(1.0e-3, 0.0))
    sp = Spectrum.repeating([(100e6, 0.0, 1.0)], 1e6)
    st = GrowthState.start(np.array([0.999e-3]), np.array([2e-3]), np.array([1.00001e-3]), np.array([0.5]), 1)
    g = Grower(law, sp, 1, 100.0)
    g.advance(st, 100.0)
    assert st.kind[0] == G.SURFACE and not np.isfinite(st.t_leak[0])
    assert st.a[0] == pytest.approx(1.0e-3, rel=1e-6) and st.dormant[0]


def test_predictor_outside_corrector_inside_is_a_plain_step() -> None:
    # C3-09: a terminal step is taken only when the corrector leaves the boundary set
    law = GrowthLaw(float(coefficient_to_si(1e-12, 3.0)), 3.0, 0.0)
    sp = Spectrum.repeating([(100e6, 0.0, 1.0)], 1e6)
    st = GrowthState.start(np.array([1.0]), np.array([0.2]), np.array([1.0]), np.array([0.5]), 1)
    st.kind[:] = G.THROUGH
    g = Grower(law, sp, 1, 1.0)
    g._step(st, np.array([0]), 1.0)
    assert 2 * st.c[0] / 0.5 <= 0.9 + 1e-12  # inside or exactly on the zone boundary


def test_long_flaw_depth_limit_is_in_the_boundary_set() -> None:
    # P-12 / benchmark 9: a long shallow flaw under bending reaches the depth limit a/B = 1.25 (a/c + 0.6)
    law = GrowthLaw(float(coefficient_to_si(1e-13, 3.0)), 3.0, 0.0)
    sp = Spectrum.repeating([(60e6, 60e6, 1.0)], 1e5)
    st = GrowthState.start(np.array([0.0222]), np.array([0.2]), np.array([0.025]), np.array([1.0]), 1)
    g = Grower(law, sp, 1, 100.0)
    a_clip, _ = g.clip_inside(
        np.array([0.0225]), np.array([0.2]), np.array([G.SURFACE]), np.array([0.025]), np.array([1.0])
    )
    assert a_clip[0] == pytest.approx(0.025 * 1.25 * (0.0222 / 0.2 + 0.6) * (1 + 1e-3), rel=5e-3)
    g.advance(st, 100.0)
    assert st.leak_trigger[0] == 2 and np.isfinite(st.t_leak_assumed[0]) and st.kind[0] >= G.THROUGH
    assert st.depth_growing_at_recharacterisation[0]


def test_invalid_initial_flaw_is_reported_not_clipped() -> None:
    # C3-11
    st = GrowthState.start(
        np.array([30e-3, 5e-3]), np.array([40e-3, 0.3]), np.array([25e-3, 25e-3]), np.array([0.5, 0.5]), 1
    )
    assert list(st.invalid_start) == [True, True] and list(st.kind) == [G.ENDED, G.ENDED]


def cycle_by_cycle_bending(
    a: float, c: float, thickness: float, width: float, dsb: float, law: GrowthLaw, cycles: int
) -> tuple[float, float, bool]:
    """One cycle per step, pure bending; returns the final sizes and whether the deepest point ever stopped and resumed."""
    g = Grower(law, Spectrum.repeating([(0.0, dsb, 1.0)], 1.0), 1, 1.0)
    idx = np.array([0])
    kind = np.array([G.SURFACE])
    arrested = resumed = False
    for _ in range(cycles):
        dk = g.ranges(idx, np.array([a]), np.array([c]), kind, np.array([thickness]), np.array([width]))[0, 0]
        ra, rc = float(law.rate(dk[0], np.array([a]))), float(law.rate(dk[1], np.array([a])))
        if ra == 0.0:
            arrested = True
        elif arrested:
            resumed = True
        a, c = a + ra, c + rc
    return a, c, resumed


def test_sliding_at_the_threshold_matches_the_cycle_by_cycle_reference() -> None:
    # C3-03: pure bending on a deep long flaw: depth growth lowers dK at the deepest point below the
    # threshold while length growth raises it; the cycle-by-cycle reference chatters on the threshold
    law = GrowthLaw(float(coefficient_to_si(5.86e-13, 2.88)), 2.88, 0.0)
    thickness, width, dsb = 25e-3, 0.5, 80e6
    found = None
    g = Grower(law, Spectrum.repeating([(0.0, dsb, 1.0)], 1.0), 1, 1.0)
    for ab in np.linspace(0.55, 0.9, 15):
        for ac in np.linspace(0.12, 0.5, 20):
            a, c = ab * thickness, ab * thickness / ac
            if 2 * c > 0.8 * width:
                continue
            dk = g.ranges(
                np.array([0]), np.array([a]), np.array([c]), np.array([0]), np.array([thickness]), np.array([width])
            )[0, 0]
            if dk[0] < dk[1]:
                found = (a, c, float(dk[0]))
                break
        if found:
            break
    assert found is not None
    a0, c0, dk_deep = found
    thr = dk_deep * 1.0001  # the deepest point is just below threshold
    law = law.with_(threshold=thr)
    cycles = 400_000
    a_ref, c_ref, resumed = cycle_by_cycle_bending(a0, c0, thickness, width, dsb, law, cycles)
    st = GrowthState.start(np.array([a0]), np.array([c0]), np.array([thickness]), np.array([width]), 1)
    gg = Grower(law, Spectrum.repeating([(0.0, dsb, 1.0)], 1e5), 1, cycles / 1e5)
    gg.advance(st, cycles / 1e5)
    assert st.a[0] == pytest.approx(a_ref, rel=5e-3) and st.c[0] == pytest.approx(c_ref, rel=5e-3)
    assert bool(st.slid[0]) == resumed  # the integrator's sliding state mirrors the reference's arrest-and-resume


def test_sliding_state_is_persisted_and_left() -> None:
    st = GrowthState.start(np.array([5e-3]), np.array([20e-3]), np.array([25e-3]), np.array([0.5]), 1)
    st.sliding[0, 0, 0] = True
    law = GrowthLaw(float(coefficient_to_si(5.86e-13, 2.88)), 2.88, 0.0)
    g = Grower(law, Spectrum.repeating([(100e6, 0.0, 1.0)], 1e5), 1, 1.0)
    g._step(st, np.array([0]), 1.0)
    # membrane loading: holding the deepest point at the threshold is impossible (residual < 0): the state is left
    assert not st.sliding[0, 0, 0]


# ---- pairings and counts (P-05, P-07, P-16, C3-01, C3-06, benchmark 17d) ----------------------


def test_conservative_pair_on_the_reviewers_two_class_cases() -> None:
    # the fourth design review's two-class case: assumption A holds; the within-pairing values lie on one
    # side of the truth while the conservative pair brackets it (exact expected counts, n = 1e6)
    pair = TimeDependentAssessment._conservative_pair
    rng = np.random.default_rng(0)
    n = 1_000_000
    weak = rng.uniform(size=n) < 0.1
    load = stats.gumbel_r(loc=180e6 - 0.5772 * 18e6 * np.sqrt(6) / np.pi, scale=18e6 * np.sqrt(6) / np.pi)
    m1, m2 = load.rvs(n, random_state=1), load.rvs(n, random_state=2)
    # year 1: weak 300 MPa until t = 0.99 then 150; strong: no failure. year 2: weak 150; strong 300 until 0.01 then 200
    fp_u = np.full(n, 2)
    fp_l = np.full(n, 2)
    up1 = np.where(weak, m1 > 150e6, False)
    lo1 = np.where(weak, m1 > 300e6, False)
    up2 = np.where(weak, m2 > 150e6, m2 > 200e6)
    lo2 = np.where(weak, m2 > 150e6, m2 > 300e6)
    fp_u[up2] = 1
    fp_u[up1] = 0
    fp_l[lo2] = 1
    fp_l[lo1] = 0
    ub, lb = pair(fp_u, fp_l, 0, 1, np.ones(n, dtype=bool))
    # truth: survival of the weak class to year 1 is F(300)^0.99 F(150)^0.01 under independent sub-period maxima
    s_weak = load.cdf(300e6) ** 0.99 * load.cdf(150e6) ** 0.01
    p_weak, p_strong = load.sf(150e6), 1 - load.cdf(300e6) ** 0.01 * load.cdf(200e6) ** 0.99
    truth = (0.1 * s_weak * p_weak + 0.9 * p_strong) / (0.1 * s_weak + 0.9)
    assert lb <= truth <= ub
    within_upper = np.sum((fp_u > 0) & (fp_u <= 1)) / np.sum(fp_u > 0)
    within_lower = np.sum((fp_l > 0) & (fp_l <= 1)) / np.sum(fp_l > 0)
    assert within_upper < truth and within_lower < truth  # both on the same side: not a bracket
    # the partner's two-type case (annual maxima uniform on [0, 1])
    u1, u2 = rng.uniform(size=n), rng.uniform(size=n)
    a_type = rng.uniform(size=n) < 0.5
    fp_u = np.full(n, 2)
    fp_l = np.full(n, 2)
    fp_u[np.where(a_type, u2 > 0.08, u2 > 0.1)] = 1
    fp_l[np.where(a_type, u2 > 0.08, u2 > 0.1)] = 1
    fp_u[np.where(a_type, u1 > 0.08, u1 > 0.1)] = 0
    fp_l[np.where(a_type, u1 > 0.8, u1 > 1.0)] = 0
    ub, lb = pair(fp_u, fp_l, 0, 1, np.ones(n, dtype=bool))
    assert lb <= 0.916693 <= ub and ub == pytest.approx(0.99089, abs=0.003) and lb == pytest.approx(0.49939, abs=0.003)


def test_pairings_take_the_most_and_least_severe_of_all_checkpoints() -> None:
    # P-05 / C3-01: a step in which the surface flaw breaks through: the pre-breakthrough state (a = B,
    # evaluated just inside validity) is the most severe; the post-event through flaw is less severe
    law = steel_in_air()
    sp = Spectrum.repeating([(120e6, 20e6, 1.0)], cycles_per_year=1e4)
    problem = cf.Assessment(cf.FlatPlate(B, W), cf.SurfaceFlaw(23e-3, 40e-3), STEEL, cf.StressState(pm=50e6))
    ext = ExtremeLoad(prob.Uniform(100e6, 100.001e6), 1.0)
    run = TimeDependentAssessment(problem, law, sp, ext, times=[1, 2], leak_is_failure=False)
    r = run.run(np.zeros((1, run.dimension)))
    assert np.isfinite(r["state"].t_leak[0]) and r["state"].t_leak[0] <= 2.0
    k = int(r["leak_step"][0])
    assert np.isfinite(r["g_min"][0, k]) and r["g_min"][0, k] < r["g_max"][0, k]
    assert r["ended_step"][0] != k and not r["any_invalid"][0, k]  # the boundary state is evaluated inside validity


def test_valid_failure_is_kept_when_the_step_later_leaves_validity() -> None:
    # P-07 at the counting level
    run = TimeDependentAssessment.__new__(TimeDependentAssessment)
    run.leak_is_failure = False
    run.times = np.array([1.0, 2.0])
    run.spectrum = Spectrum.repeating([(1.0, 0.0, 1.0)], 1.0)
    run.law = GrowthLaw(1e-30, 3.0, 0.0)
    run.shared_scale = False
    run.inspections = ()
    state = GrowthState.start(
        np.array([1e-3, 1e-3]), np.array([2e-3, 2e-3]), np.array([25e-3, 25e-3]), np.array([0.5, 0.5]), 1
    )
    fake = {
        "g_min": np.array([[-1.0, 1.0], [1.0, 1.0]]),
        "g_max": np.array([[0.5, 1.0], [1.0, 1.0]]),
        "any_invalid": np.array([[True, False], [False, True]]),
        "mode": np.full((2, 2), "curve", dtype=object),
        "leak_step": np.array([-1, -1]),
        "ended_step": np.array([-1, 1]),
        "violations": 0,
        "violating_samples": 0,
        "inspections": [],
        "repaired": np.zeros(2, dtype=bool),
        "t_repair": np.full(2, np.nan),
        "removed_leaked": np.zeros(2, dtype=bool),
        "removed_ended": np.zeros(2, dtype=bool),
        "state": state,
        "problem": None,
        "invalid_start": 0.0,
        "grid": np.array([1.0, 2.0]),
        "n": 2,
    }
    res = run._assemble(fake, np.array([1.0, 2.0]), 0)
    # sample 0 failed at a valid checkpoint in step 0: counted in every treatment; sample 1 is invalid in step 1 only
    assert list(res.cumulative_upper.value) == [0.5, 1.0] and list(res.cumulative_valid_only.value) == [0.5, 0.5]
    assert list(res.cumulative_lower.value) == [0.5, 1.0]
    assert res.modes["fracture"] == 0.5 and res.modes["out_of_validity"] == 0.5


# ---- inspections (P-08, P-09, P-10, C3-04) ------------------------------------------------------


def two_point_problem() -> tuple[cf.Assessment, ExtremeLoad, GrowthLaw, Spectrum]:
    a0 = prob.Empirical([1.0e-3, 4.0e-3], name="a0")
    problem = cf.Assessment(cf.FlatPlate(B, W), cf.SurfaceFlaw(a0, 40e-3), STEEL, cf.StressState(pm=100e6))
    law = GrowthLaw(float(coefficient_to_si(5.86e-13, 2.88)), 2.88, 0.0)
    sp = Spectrum.repeating([(70e6, 0.0, 1.0)], cycles_per_year=1e5)
    crit = problem.with_(flaw=cf.SurfaceFlaw(8e-3, 40e-3))
    lo, hi = 1e6, 2e9
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        ok = bool(crit.with_(loads=crit.loads.with_(pm=mid)).run(on_invalid="mask").acceptable)
        lo, hi = (mid, hi) if ok else (lo, mid)
    return problem, ExtremeLoad(prob.Uniform(lo, lo * (1 + 1e-6)), 1.0), law, sp


def test_sizing_error_has_its_own_random_variable() -> None:
    # P-08 / C3-04: PoD 0.5 and acceptance at the true size with a zero-mean error: P(repair) = 0.25
    problem, ext, law, sp = two_point_problem()
    insp = Inspection(2.0, ConstantPoD(0.5), sizing_error=prob.Normal(0.0, 1e-4), accept_below=4.0e-3, action="repair")
    run = TimeDependentAssessment(problem, law, sp, ext, times=range(1, 4), leak_is_failure=False, inspections=[insp])
    assert insp.draws == 2 and run.dimension == 1 + 3 + 2
    res = run.solve(seed=0, samples=8000, stage_sensitivity=False)
    # only the large flaw (4 mm, grown to about 5 mm by year 2) is above the acceptance size: P(repair) = 0.5 x 0.5
    assert res.inspections[0]["p_repair_all"] == pytest.approx(0.25, abs=0.02)
    assert res.inspections[0]["p_detect_all"] == pytest.approx(0.5, abs=0.02)


def test_failed_components_are_not_inspected_and_denominators_are_survivors() -> None:
    # P-09: every sample fails in the first step (load above the critical load); an inspection at its end
    # finds nothing to inspect
    problem = cf.Assessment(cf.FlatPlate(B, W), cf.SurfaceFlaw(6e-3, 60e-3), STEEL, cf.StressState(pm=100e6))
    ext = ExtremeLoad(prob.Uniform(1.5e9, 1.5001e9), 1.0)
    run = TimeDependentAssessment(
        problem,
        GrowthLaw(1e-40, 3.0, 1e9),
        Spectrum.repeating([(1e6, 0, 1)], 1.0),
        ext,
        times=[1, 2],
        leak_is_failure=False,
        inspections=[Inspection(1.0, ConstantPoD(1.0))],
    )
    res = run.solve(seed=0, samples=50, stage_sensitivity=False)
    assert res.cumulative_upper.value[0] == 1.0
    assert res.inspections[0]["p_detect_all"] == 0.0 and res.inspections[0]["survivors_lower"] == 0
    assert np.isnan(res.inspections[0]["p_detect_flawed_survivors"]) and res.events["repaired"] == 0.0


def test_update_after_an_earlier_policy_inspection() -> None:
    # P-10 / benchmark 17b: a repair inspection at year 1 (flaw-free mass), then this component's own
    # "not detected" at year 2: the flaw-free samples are retained, the large flaws that escaped repair are not
    problem, ext, law, sp = two_point_problem()
    pod = ConstantPoD(0.5)
    run = TimeDependentAssessment(
        problem,
        law,
        sp,
        ext,
        times=range(1, 11),
        leak_is_failure=False,
        inspections=[Inspection(1.0, pod, action="repair"), Inspection(2.0, pod, observed="not_detected")],
    )
    res = run.solve(seed=0, samples=8000, stage_sensitivity=False)
    assert res.conditioning is not None and not res.weighted
    # survivors at year 2: unrepaired (3/4 of each size) with PoD 0.5 kept with probability 1/2, repaired kept always
    # later failure: unrepaired large flaws only. P(fail | retained) = (0.5 x 0.5 x 0.5) / (0.5 x 0.5 + 0.5 x 0.5 x 0.5 x 2)
    expected = (0.5 * 0.5 * 0.5) / (0.5 + 0.5 * 0.5)
    assert res.conditioning["pf_after_upper"] == pytest.approx(expected, abs=0.03)
    assert res.conditioning["label_upper"].startswith("exact for the upper pairing")
    assert "refine_at_conditioning_time" in res.conditioning


# ---- refinement, sensitivity, grow, compare_orders (P-17, P-18, P-19, P-21) ----------------------


def test_default_step_is_the_reference_period_and_refinement_keeps_the_coarse_maximum() -> None:
    problem = cf.Assessment(cf.FlatPlate(B, W), cf.SurfaceFlaw(6e-3, 60e-3), STEEL, cf.StressState(pm=100e6))
    ext = ExtremeLoad(prob.Truncated(prob.Gumbel.from_mean_sd(150e6, 15e6), lo=0.0), 1.0)
    run = TimeDependentAssessment(
        problem,
        GrowthLaw(1e-40, 3.0, 1e9),
        Spectrum.repeating([(1e6, 0, 1)], 1.0),
        ext,
        times=[5, 10],
        leak_is_failure=False,
    )
    assert run.delta == 1.0 and run.n_steps == 10  # P-17
    rng = np.random.default_rng(0)
    u = rng.standard_normal((2000, run.dimension))
    fine, grid = run._refined(u, run.grid, rng)
    assert grid.size == 20 and grid[0] == 0.5
    coarse = ext.step_maximum(u[:, run.n_invariant : run.n_invariant + 10], 1.0)
    sub = ext.step_maximum(fine[:, run.n_invariant : run.n_invariant + 20], 0.5)
    np.testing.assert_allclose(np.maximum(sub[:, 0::2], sub[:, 1::2]), coarse, rtol=1e-9)
    # the sub-step maxima have the half-step distribution F^(1/2)
    assert stats.kstest(ext.distribution.cdf(sub.ravel()) ** 0.5, "uniform").pvalue > 0.01


def test_refine_runs_when_the_bracket_is_wide() -> None:
    law = steel_in_air()
    sp = Spectrum.repeating([(80e6, 0.0, 1.0)], cycles_per_year=3e4)
    problem = cf.Assessment(cf.FlatPlate(B, W), cf.SurfaceFlaw(3e-3, 30e-3), STEEL, cf.StressState(pm=100e6))
    ext = ExtremeLoad(prob.Truncated(prob.Gumbel.from_mean_sd(240e6, 25e6), lo=0.0), 1.0)
    run = TimeDependentAssessment(problem, law, sp, ext, times=range(1, 6), leak_is_failure=False)
    res = run.solve(seed=2, samples=3000, refine=True, stage_sensitivity=False)
    assert res.refinements >= 1 and res.delta < 1.0
    plain = run.solve(seed=2, samples=3000, stage_sensitivity=False)
    width_plain = plain.cumulative_upper.value[-1] - plain.cumulative_lower.value[-1]
    width_fine = res.cumulative_upper.value[-1] - res.cumulative_lower.value[-1]
    assert width_fine < width_plain


def test_stage_sensitivity_and_knee_shift() -> None:
    law = steel_in_air(curve="mean", stage_correlation=0.5)
    sp = Spectrum.repeating([(80e6, 10e6, 1.0)], cycles_per_year=5e4)
    problem = cf.Assessment(cf.FlatPlate(B, W), cf.SurfaceFlaw(2e-3, 20e-3), STEEL, cf.StressState(pm=100e6))
    ext = ExtremeLoad(prob.Truncated(prob.Gumbel.from_mean_sd(150e6, 15e6), lo=0.0), 1.0)
    run = TimeDependentAssessment(problem, law, sp, ext, times=range(1, 4), leak_is_failure=False)
    res = run.solve(seed=0, samples=300)
    assert res.sensitivity is not None and set(res.sensitivity) == {"stage_correlation=0", "stage_correlation=1"}
    s_a, s_b = 0.320, 0.171
    assert res.knee_shift_log10 == pytest.approx(
        -math.sqrt((s_a**2 + s_b**2 - 2 * 0.5 * s_a * s_b) / (2 * math.pi)), rel=1e-6
    )
    assert "validity:" in "\n".join(res.assumptions)


def test_grow_and_compare_orders_on_a_deterministic_problem() -> None:
    problem = cf.Assessment(cf.FlatPlate(B, W), cf.SurfaceFlaw(3e-3, 30e-3), STEEL, cf.StressState(pm=100e6))
    law = steel_in_air()
    ext = ExtremeLoad(prob.Uniform(200e6, 200.001e6), 1.0)
    blocks = [(120e6, 0.0, 6e4), (40e6, 0.0, 6e4)]
    sp = Spectrum.ordered(blocks, 6e4, "as_given")
    run = TimeDependentAssessment(problem, law, sp, ext, times=[1, 2], leak_is_failure=False)
    hist = run.grow(load=200e6)
    assert hist.times[-1] == 2.0 and hist.a[-1] > 3e-3 and np.isfinite(hist.g).all()
    orders = run.compare_orders(load=200e6)
    assert set(orders) == {"as_given", "largest_first", "smallest_first"}
    assert (
        orders["largest_first"].a[-1] > orders["smallest_first"].a[-1]
    )  # the large block first grows more by the same time
    assert all(np.isfinite(h.cycles_applied) for h in orders.values())
    random_problem = problem.with_(flaw=cf.SurfaceFlaw(prob.Uniform(2e-3, 4e-3), 30e-3))
    with pytest.raises(ValueError, match="deterministic"):
        TimeDependentAssessment(
            random_problem,
            law,
            Spectrum.repeating([(120e6, 0.0, 1.0)], 6e4, life_factor=2.0),
            ext,
            times=[1],
            leak_is_failure=False,
        )
    with pytest.raises(NotImplementedError, match="TODO"):
        Grower(GrowthLaw(1e-12, 3.0, 0.0, effective_range=True), sp, 1, 1.0)


def test_leak_mass_moves_exactly_between_definitions() -> None:
    law = steel_in_air()
    sp = Spectrum.repeating([(60e6, 10e6, 1.0)], cycles_per_year=1e5)
    a0 = prob.Truncated(prob.LogNormal.from_mean_sd(2e-3, 0.8e-3), hi=10e-3, name="a0")
    problem = cf.Assessment(cf.FlatPlate(B, W), cf.SurfaceFlaw(a0, 12e-3), STEEL, cf.StressState(pm=50e6))
    ext = ExtremeLoad(prob.Uniform(20e6, 20.001e6), 1.0)  # a load that never fails anything: only leaks count
    kw = dict(times=range(1, 6), stage_sensitivity=False)
    without = TimeDependentAssessment(problem, law, sp, ext, times=range(1, 6), leak_is_failure=False).solve(
        seed=3, samples=2000, stage_sensitivity=False
    )
    with_leak = TimeDependentAssessment(problem, law, sp, ext, times=range(1, 6), leak_is_failure=True).solve(
        seed=3, samples=2000, stage_sensitivity=False
    )
    assert without.cumulative_valid_only.value[-1] == 0.0
    assert with_leak.cumulative_valid_only.value[-1] == pytest.approx(
        with_leak.events_any["leak"] + with_leak.events_any["leak_assumed"], abs=1e-12
    )
    assert with_leak.modes["leak"] == pytest.approx(with_leak.cumulative_valid_only.value[-1], abs=1e-12)
    del kw


# ---- round 3 (D3C-05 to D3C-13, P-04, P-09, P-26) ---------------------------------------------


def test_sizing_decision_direction_and_independence_from_detection() -> None:
    # PoD 0.5; the acceptance size 0.5 sd above the true size: P(repair) = 0.5 x P(error > 0.5 sd) = 0.154
    # (an inverted comparison gives 0.346; a sizing error drawn from the detection variable gives 0)
    problem, ext, law, sp = two_point_problem()
    sd = 1.0e-3
    plain = TimeDependentAssessment(problem, law, sp, ext, times=range(1, 3), leak_is_failure=False)
    probe = np.zeros((2, plain.dimension))
    probe[:, 0] = (-3.0, 3.0)  # both atoms of the initial flaw size
    big = float(np.max(plain.run(probe)["state"].a))  # the large flaw's depth at year 2
    insp = Inspection(
        2.0, ConstantPoD(0.5), sizing_error=prob.Normal(0.0, sd), accept_below=big + 0.5 * sd, action="repair"
    )
    run = TimeDependentAssessment(problem, law, sp, ext, times=range(1, 3), leak_is_failure=False, inspections=[insp])
    res = run.solve(seed=1, samples=8000, stage_sensitivity=False)
    expected = 0.5 * 0.5 * float(stats.norm.sf(0.5))  # half the population is the large flaw
    assert res.inspections[0]["p_repair_all"] == pytest.approx(expected, abs=0.015)
    assert res.inspections[0]["p_detect_all"] == pytest.approx(0.5, abs=0.02)
    assert "denominators" in res.inspections[0] and res.inspections[0]["survivors_upper"] == 8000


def test_catalogue_defaults_and_through_flaw_detection() -> None:
    assert steel_in_air().threshold == pytest.approx(63.0 * NMM) and steel_in_air().shallow_cap is not None
    from critflaw.fatigue.laws import steel_in_seawater

    assert steel_in_seawater("free").threshold == 0.0 and steel_in_seawater("cp850").threshold == pytest.approx(
        63.0 * NMM
    )
    # a through flaw at an inspection: detected with probability 1 by default, by pod_through if given
    law = steel_in_air()
    sp = Spectrum.repeating([(120e6, 20e6, 1.0)], cycles_per_year=1e4)
    problem = cf.Assessment(cf.FlatPlate(B, W), cf.SurfaceFlaw(23e-3, 40e-3), STEEL, cf.StressState(pm=50e6))
    ext = ExtremeLoad(prob.Uniform(10e6, 10.001e6), 1.0)  # too low to fail anything
    for pod_through, expected in ((None, 1.0), (ConstantPoD(0.0), 0.0)):
        run = TimeDependentAssessment(
            problem,
            law,
            sp,
            ext,
            times=[1, 2, 3],
            leak_is_failure=False,
            inspections=[Inspection(3.0, ConstantPoD(0.0), pod_through=pod_through, action="none")],
        )
        res = run.solve(seed=0, samples=4, stage_sensitivity=False)
        assert res.events_any["leak"] == 1.0 and res.inspections[0]["p_detect_all"] == expected


def test_ordered_finish_and_checkpoint_sizes_against_quadrature() -> None:
    from scipy import integrate

    law = GrowthLaw(float(coefficient_to_si(1e-12, 3.0)), 3.0, 0.0)
    nu = 1e6

    def inv_v(c: float) -> float:
        dk = math.cos(math.pi * c / 0.5) ** -0.5 * 100e6 * math.sqrt(math.pi * c)
        return 1.0 / (float(law.rate(dk)) * nu)

    t_total = integrate.quad(inv_v, 0.225, 0.25, epsabs=0, epsrel=1e-12, limit=400)[0]
    for sp in (Spectrum.repeating([(100e6, 0.0, 1.0)], nu), Spectrum.ordered([(100e6, 0.0, nu)], nu, "as_given")):
        st = GrowthState.start(np.array([1.0]), np.array([0.225]), np.array([1.0]), np.array([0.5]), 1)
        st.kind[:] = G.THROUGH
        g = Grower(law, sp, 1, 1.0)
        t_half = 0.5 * t_total  # a checkpoint inside the finish window: the table's size inverts the integral
        g.advance(st, t_half)
        c_ref = float(integrate.quad(inv_v, 0.225, st.c[0], epsabs=0, epsrel=1e-12)[0])
        assert c_ref == pytest.approx(t_half, rel=1e-6) and st.kind[0] == G.THROUGH
        g.advance(st, 1.0)
        assert st.kind[0] == G.ENDED and st.t_ended[0] == pytest.approx(t_total, rel=1e-8)


def test_block_end_corrector_against_the_closed_form() -> None:
    # alternating blocks on a through flaw with m = 2 and Y = 1: c = c0 exp(A pi sum(dsigma_i^2 N_i)), order-free
    a_si = 1e-22
    law = GrowthLaw(a_si, 2.0, 0.0)
    blocks = [(100e6, 0.0, 1e4), (50e6, 0.0, 1e4)] * 5
    sp = Spectrum.ordered(blocks, 1e5, "as_given")
    st = GrowthState.start(np.array([1.0]), np.array([5e-3]), np.array([1.0]), np.array([1e9]), len(blocks))
    st.kind[:] = G.THROUGH
    g = Grower(law, sp, 1, 1.0)
    g.advance(st, sp.total_years())
    exponent = a_si * math.pi * sum(ds**2 * n for ds, _, n in blocks)
    assert st.c[0] == pytest.approx(5e-3 * math.exp(exponent), rel=1e-5)


def test_sized_update_withheld_and_numeric_cases() -> None:
    problem, ext, law, sp = two_point_problem()
    pod = ConstantPoD(0.5)
    plain = TimeDependentAssessment(problem, law, sp, ext, times=range(1, 4), leak_is_failure=False)
    probe = np.zeros((2, plain.dimension))
    probe[:, 0] = (-3.0, 3.0)
    big = float(np.max(plain.run(probe)["state"].a))  # the large flaw at year 3
    insp = Inspection(3.0, pod, sizing_error=prob.Normal(0.0, 1e-3), action="none", observed=("sized", big))
    small = TimeDependentAssessment(
        problem, law, sp, ext, times=range(1, 11), leak_is_failure=False, inspections=[insp]
    ).solve(seed=0, samples=120, stage_sensitivity=False)
    assert small.weighted and small.conditioning["withheld_upper"] and small.conditioning["pf_after_upper"] is None
    assert small.conditioning["samples_needed_upper"] > 120
    large = TimeDependentAssessment(
        problem, law, sp, ext, times=range(1, 11), leak_is_failure=False, inspections=[insp]
    ).solve(seed=0, samples=2000, stage_sensitivity=False)
    assert not large.conditioning["withheld_upper"]
    assert large.conditioning["pf_after_upper"] == pytest.approx(1.0, abs=0.02)  # the weight sits on the large flaw
    assert large.conditioning["label_upper"] == "conditional; weighted; no guaranteed interval"


def test_repair_flaw_must_be_valid_and_resets_the_integrator_state() -> None:
    problem, ext, law, sp = two_point_problem()
    bad = Inspection(1.0, ConstantPoD(1.0), repair_flaw=(30e-3, 40e-3))
    with pytest.raises(ValueError, match="validity"):
        TimeDependentAssessment(problem, law, sp, ext, times=[1, 2], leak_is_failure=False, inspections=[bad]).solve(
            seed=0, samples=4, stage_sensitivity=False
        )
    good = Inspection(1.0, ConstantPoD(1.0), repair_flaw=(0.5e-3, 5e-3))
    res = TimeDependentAssessment(
        problem, law, sp, ext, times=range(1, 11), leak_is_failure=False, inspections=[good]
    ).solve(seed=0, samples=200, stage_sensitivity=False)
    assert res.events_any["repaired"] == 1.0 and res.cumulative_upper.value[-1] == 0.0  # the repair flaw stays small


def test_sensitivity_with_unshared_scale_and_after_refinement() -> None:
    law = steel_in_air(curve="mean", stage_correlation=0.5)
    sp = Spectrum.repeating([(80e6, 10e6, 1.0)], cycles_per_year=5e4, load_scale=prob.Uniform(0.9, 1.1))
    problem = cf.Assessment(cf.FlatPlate(B, W), cf.SurfaceFlaw(2e-3, 20e-3), STEEL, cf.StressState(pm=100e6))
    ext = ExtremeLoad(prob.Truncated(prob.Gumbel.from_mean_sd(150e6, 15e6), lo=0.0), 1.0)
    run = TimeDependentAssessment(problem, law, sp, ext, times=range(1, 3), leak_is_failure=False, shared_scale=False)
    res = run.solve(seed=0, samples=200, refine=True)
    assert res.sensitivity is not None and len(res.sensitivity) == 2
    sub = run.solve(seed=0, samples=200, subset=True, replicates=2, samples_per_level=100, curve_times=[1.0])
    assert sub.subset is not None and sub.subset_curve is not None and 1.0 in sub.subset_curve["separate_runs"]


def test_sample_at_a_block_end_is_not_skipped_to_the_next_block() -> None:
    law = GrowthLaw(1e-22, 2.0, 0.0)
    sp = Spectrum.ordered([(100e6, 0.0, 1e4), (100e6, 0.0, 1e4)], 1e5, "as_given")
    st = GrowthState.start(np.array([1.0]), np.array([5e-3]), np.array([1.0]), np.array([1e9]), 2)
    st.kind[:] = G.THROUGH
    st.t[:] = 0.1 * (1 - 1e-10)  # within 1e-9 below the end of block 1
    g = Grower(law, sp, 1, 1.0)
    g.advance(st, 0.2)
    assert st.c[0] == pytest.approx(5e-3 * math.exp(1e-22 * math.pi * 100e6**2 * 1e4), rel=1e-5)  # block 2 integrated


# ---- round 4 (P-27, P-04, P-09, P-14, P-20, P-23) ------------------------------------------------


def test_through_flaw_beyond_the_zone_entry_is_assessed_at_its_actual_size() -> None:
    # P-27: a through flaw at 2c/W = 0.96 fails; the terminal-zone entry size (0.9) would not
    problem = cf.Assessment(cf.FlatPlate(B, W), cf.SurfaceFlaw(1e-3, 4e-3), STEEL, cf.StressState(pm=30e6))
    run = TimeDependentAssessment(
        problem,
        GrowthLaw(1e-40, 3.0, 1e9),
        Spectrum.repeating([(1e6, 0, 1)], 1.0),
        ExtremeLoad(prob.Uniform(30e6, 30.001e6), 1.0),
        times=[1],
        leak_is_failure=False,
    )
    grower = Grower(run.law, run.spectrum, 2, 1.0)
    snap = (np.array([B, B]), np.array([0.24, 0.225]), np.array([G.THROUGH, G.THROUGH]))
    g, valid, _ = run._driving(problem, grower, snap, np.array([30e6, 30e6]))
    direct = problem.with_(flaw=cf.ThroughFlaw(two_a=0.48), loads=problem.loads.with_(pm=30e6)).run(on_invalid="mask")
    assert valid[0] and g[0] <= 0.0 and not bool(direct.acceptable)
    assert g[0] == pytest.approx(float(direct.radial_margin - 1.0), rel=1e-9)
    assert g[1] > 0.0  # the entry size itself is still acceptable: the clip would have hidden the failure


def test_stencil_stays_inside_the_long_flaw_depth_limit() -> None:
    # P-14: on the long-flaw depth boundary the forward a-point is outside the boundary set
    law = GrowthLaw(float(coefficient_to_si(1e-13, 3.0)), 3.0, 0.0)
    g = Grower(law, Spectrum.repeating([(60e6, 60e6, 1.0)], 1e5), 1, 1.0)
    a, c, thick, width = np.array([0.0222222222222222]), np.array([0.2]), np.array([0.025]), np.array([1.0])
    assert np.all(g.boundary_excess(a, c, np.array([0]), thick, width) <= 1e-9)
    dk_a, dk_c = g._derivatives(np.array([0]), a, c, np.array([0]), thick, width)
    # one-sided: the derivative equals the backward difference computed here
    h = 1e-6 * a
    back = (
        g.ranges(np.array([0]), a, c, np.array([0]), thick, width)
        - g.ranges(np.array([0]), a - h, c, np.array([0]), thick, width)
    ) / h[:, None, None]
    np.testing.assert_allclose(dk_a, back, rtol=1e-9)
    assert np.all(np.isfinite(dk_c))


def test_sequential_sized_then_not_detected_update_stays_weighted() -> None:
    # benchmark 17c: a sized observation (weights) followed by "not detected" (rejection): the population
    # stays weighted; with equal PoD the later outcome leaves the relative weights unchanged
    problem, ext, law, sp = two_point_problem()
    pod = ConstantPoD(0.5)
    plain = TimeDependentAssessment(problem, law, sp, ext, times=range(1, 3), leak_is_failure=False)
    probe = np.zeros((2, plain.dimension))
    probe[:, 0] = (-3.0, 3.0)
    sizes = np.sort(plain.run(probe)["state"].a)  # small and large flaw at year 2
    a_m = 0.5 * (sizes[0] + sizes[1]) + 0.3 * (sizes[1] - sizes[0])  # nearer the large flaw
    sd = 0.5 * (sizes[1] - sizes[0])
    insps = [
        Inspection(2.0, pod, sizing_error=prob.Normal(0.0, sd), action="none", observed=("sized", a_m)),
        Inspection(3.0, pod, action="none", observed="not_detected"),
    ]
    res = TimeDependentAssessment(
        problem, law, sp, ext, times=range(1, 11), leak_is_failure=False, inspections=insps
    ).solve(seed=0, samples=4000, stage_sensitivity=False)
    assert res.weighted and res.conditioning["pf_after_upper"] is not None
    w_small, w_large = stats.norm.pdf(a_m - sizes[0], scale=sd), stats.norm.pdf(a_m - sizes[1], scale=sd)
    expected = w_large / (w_large + w_small)  # only the large flaw fails later; the retention is 1/2 for both
    assert res.conditioning["pf_after_upper"] == pytest.approx(expected, abs=0.03)
    assert len(res.conditioning["effective_sizes_after_each_outcome"]) == 2
    assert "no guaranteed interval" in res.conditioning["label_upper"]


def test_assumption_violation_path_with_a_prescribed_severity_dip(monkeypatch: pytest.MonkeyPatch) -> None:
    # benchmark 12b (bookkeeping path): the driving variable dips inside a step and recovers; the violation
    # is counted, the upper pairing uses the dip and the lower pairing the least severe checkpoint
    problem = cf.Assessment(cf.FlatPlate(B, W), cf.SurfaceFlaw(1e-3, 4e-3), STEEL, cf.StressState(pm=30e6))
    run = TimeDependentAssessment(
        problem,
        GrowthLaw(1e-40, 3.0, 1e9),
        Spectrum.repeating([(1e6, 0, 1)], 1.0),
        ExtremeLoad(prob.Uniform(30e6, 30.001e6), 1.0),
        times=[1, 2],
        leak_is_failure=False,
    )
    calls = {"n": 0}
    profile = [0.5, 0.2, -0.1, 0.2, 0.5, 0.5]  # start, four interior, end: a dip in the middle

    def fake(self: Any, problem: Any, grower: Any, snap: Any, m: Any, only: Any = None) -> Any:
        n = snap[0].size
        value = profile[calls["n"] % 6]
        calls["n"] += 1
        return np.full(n, value), np.ones(n, dtype=bool), np.full(n, "curve", dtype=object)

    monkeypatch.setattr(TimeDependentAssessment, "_driving", fake)
    r = run.run(np.zeros((3, run.dimension)))
    assert r["violations"] == 6 and r["violating_samples"] == 3  # both steps, every sample
    profile[:] = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]  # monotone non-increasing: no violation
    calls["n"] = 0
    r0 = run.run(np.zeros((2, run.dimension)))
    assert r0["violations"] == 0 and r0["violating_samples"] == 0
    np.testing.assert_allclose(r0["g_min"], 0.0)
    np.testing.assert_allclose(r0["g_max"], 0.5)
    np.testing.assert_allclose(r["g_min"], -0.1)
    np.testing.assert_allclose(r["g_max"], 0.5)
    res = run._assemble(r, run.grid, 0)
    assert list(res.cumulative_upper.value) == [1.0, 1.0] and list(res.cumulative_lower.value) == [0.0, 0.0]
    assert res.assumption_violations == 6 and "violated it" in "\n".join(res.assumptions)


def test_subset_in_time_coverage_and_shares_on_a_no_growth_case() -> None:
    # benchmark 14 (reduced): constant hazard p per year over 5 years; Pf(0, 5) = 1 - (1 - p)^5 with p ~ 2e-3
    problem = cf.Assessment(cf.FlatPlate(B, W), cf.SurfaceFlaw(6e-3, 60e-3), STEEL, cf.StressState(pm=100e6))
    lo, hi = 1e6, 2e9
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        ok = bool(problem.with_(loads=problem.loads.with_(pm=mid)).run(on_invalid="mask").acceptable)
        lo, hi = (mid, hi) if ok else (lo, mid)
    crit = 0.5 * (lo + hi)
    ext = ExtremeLoad(prob.Truncated(prob.Gumbel.from_mean_sd(0.54 * crit, 0.08 * crit), lo=0.0), 1.0)
    p = float(ext.distribution.sf(crit))
    assert 1e-4 < p < 6e-4  # lifetime probability about 1.5e-3: below what 1000 Monte Carlo samples resolve
    exact = 1 - (1 - p) ** 5
    run = TimeDependentAssessment(
        problem,
        GrowthLaw(1e-40, 3.0, 1e9),
        Spectrum.repeating([(1e6, 0, 1)], 1.0),
        ext,
        times=range(1, 6),
        leak_is_failure=False,
    )
    covered = 0
    shares = []
    for seed in range(6):
        res = run.solve(
            seed=seed, samples=2000, subset=True, replicates=4, samples_per_level=1000, stage_sensitivity=False
        )
        rel = res.subset
        assert rel is not None and rel.subset is not None
        interval = rel.subset.interval
        if interval is not None:
            covered += interval.contains(exact)
        if res.subset_curve and "share_mean" in res.subset_curve:
            shares.append(res.subset_curve["share_mean"])
            assert res.subset_curve["resolution_floor"] > 0.0
    assert covered >= 5  # replicate interval covers in at least 5 of 6 seeds
    # constant hazard: the share of first passages by year k is close to k / 5 (exactly (1-(1-p)^k)/(1-(1-p)^5))
    mean_share = np.mean(shares, axis=0)
    expected = (1 - (1 - p) ** np.arange(1, 6)) / exact
    np.testing.assert_allclose(mean_share, expected, atol=0.1)


# ---- round 4, findings E3C-01, E3C-03 to E3C-07 and the named mutation survivors ----------------


def _through_reference(law: GrowthLaw, blocks: list[tuple[float, float]], w: float) -> Any:
    """1/v(c) for a through flaw under blocks (stress range, cycles per year), finite-width corrected."""

    def inv_v(c: float) -> float:
        v = 0.0
        for ds, nu in blocks:
            dk = math.cos(math.pi * c / w) ** -0.5 * ds * math.sqrt(math.pi * c)
            v += float(law.rate(dk)) * nu
        return 1.0 / v

    return inv_v


def test_finish_with_a_threshold_crossing_inside_the_window() -> None:
    # E3C-01: a block whose range crosses the threshold inside the finish window makes the integrand jump;
    # the panels never straddle the kink, so the end time and the mid-window read agree with scipy to 1e-12
    from scipy import integrate

    w, nu = 0.5, 1e6
    a_si = float(coefficient_to_si(1e-12, 3.0))
    for frac in (0.02, 0.21, 0.5, 0.79):
        c_cross = 0.225 + frac * (0.25 - 0.225)
        thr = math.cos(math.pi * c_cross / w) ** -0.5 * 40e6 * math.sqrt(math.pi * c_cross)
        law = GrowthLaw(a_si, 3.0, thr)
        inv_v = _through_reference(law, [(100e6, 0.5 * nu), (40e6, 0.5 * nu)], w)
        t_total = integrate.quad(inv_v, 0.225, 0.25, points=[c_cross], epsabs=0, epsrel=1e-13, limit=500)[0]
        sp = Spectrum.repeating([(100e6, 0.0, 1.0), (40e6, 0.0, 1.0)], nu)
        st = GrowthState.start(np.array([1.0]), np.array([0.225]), np.array([1.0]), np.array([w]), 2)
        st.kind[:] = G.THROUGH
        g = Grower(law, sp, 1, 1.0)
        g.advance(st, 0.5 * t_total)
        t_mid = integrate.quad(inv_v, 0.225, float(st.c[0]), points=[c_cross], epsabs=0, epsrel=1e-13, limit=500)[0]
        assert t_mid == pytest.approx(0.5 * t_total, rel=1e-12)
        g.advance(st, 1.0)
        assert st.kind[0] == G.ENDED and st.t_ended[0] == pytest.approx(t_total, rel=1e-12)


def test_ordered_multi_block_finish_against_quadrature() -> None:
    # E3C-03 / C3-02: the block-by-block finish; two unequal blocks spanning the zone against scipy at 1e-9,
    # and five identical ordered blocks against the repeating form
    from scipy import integrate, optimize

    w, nu = 0.5, 1e6
    law = GrowthLaw(float(coefficient_to_si(1e-12, 3.0)), 3.0, 0.0)
    inv_1 = _through_reference(law, [(100e6, nu)], w)
    inv_2 = _through_reference(law, [(60e6, nu)], w)
    t1 = 0.5 * integrate.quad(inv_1, 0.225, 0.25, epsabs=0, epsrel=1e-13)[0]  # block 1 ends half-way
    c_star = optimize.brentq(
        lambda c: integrate.quad(inv_1, 0.225, c, epsabs=0, epsrel=1e-13)[0] - t1, 0.225, 0.25, xtol=1e-15
    )
    t_total = t1 + integrate.quad(inv_2, c_star, 0.25, epsabs=0, epsrel=1e-13)[0]
    sp = Spectrum.ordered([(100e6, 0.0, t1 * nu), (60e6, 0.0, 10 * t_total * nu)], nu, "as_given")
    st = GrowthState.start(np.array([1.0]), np.array([0.225]), np.array([1.0]), np.array([w]), 2)
    st.kind[:] = G.THROUGH
    g = Grower(law, sp, 1, 1.0)
    g.advance(st, t1)
    assert st.c[0] == pytest.approx(c_star, rel=1e-9) and st.kind[0] == G.THROUGH
    g.advance(st, sp.total_years())
    assert st.kind[0] == G.ENDED and st.t_ended[0] == pytest.approx(t_total, rel=1e-9)
    # the block end crossed inside one advance (the path of run()'s fixed-fraction checkpoints)
    c_mid = optimize.brentq(
        lambda c: integrate.quad(inv_2, c_star, c, epsabs=0, epsrel=1e-13)[0] - 0.5 * t1, c_star, 0.25, xtol=1e-15
    )
    one = GrowthState.start(np.array([1.0]), np.array([0.225]), np.array([1.0]), np.array([w]), 2)
    one.kind[:] = G.THROUGH
    g.advance(one, 1.5 * t1)
    assert one.c[0] == pytest.approx(c_mid, rel=1e-9)
    g.advance(one, sp.total_years())
    assert one.t_ended[0] == pytest.approx(t_total, rel=1e-9)
    # five identical blocks: the same end time as the repeating form
    rep = GrowthState.start(np.array([1.0]), np.array([0.225]), np.array([1.0]), np.array([w]), 1)
    rep.kind[:] = G.THROUGH
    Grower(law, Spectrum.repeating([(100e6, 0.0, 1.0)], nu), 1, 1.0).advance(rep, 1.0)
    t_rep = float(rep.t_ended[0])
    five = Spectrum.ordered([(100e6, 0.0, 0.3 * t_rep * nu)] * 5, nu, "as_given")
    st5 = GrowthState.start(np.array([1.0]), np.array([0.225]), np.array([1.0]), np.array([w]), 5)
    st5.kind[:] = G.THROUGH
    Grower(law, five, 1, 1.0).advance(st5, five.total_years())
    assert st5.kind[0] == G.ENDED and st5.t_ended[0] == pytest.approx(t_rep, rel=1e-9)


def test_finish_entered_from_outside_the_zone_keeps_the_time_already_spent() -> None:
    # G20: the Heun steps to the zone boundary and the finish from there add up to the whole time
    from scipy import integrate

    w, nu = 0.5, 1e6
    law = GrowthLaw(float(coefficient_to_si(1e-12, 3.0)), 3.0, 0.0)
    inv_v = _through_reference(law, [(100e6, nu)], w)
    t_total = integrate.quad(inv_v, 0.2, 0.25, epsabs=0, epsrel=1e-13)[0]
    t_zone = integrate.quad(inv_v, 0.2, 0.225, epsabs=0, epsrel=1e-13)[0]
    assert t_zone > 0.3 * t_total  # the pre-zone part is a large share of the life
    st = GrowthState.start(np.array([1.0]), np.array([0.2]), np.array([1.0]), np.array([w]), 1)
    st.kind[:] = G.THROUGH
    Grower(law, Spectrum.repeating([(100e6, 0.0, 1.0)], nu), 1, 1.0).advance(st, 1.0)
    assert st.t_ended[0] == pytest.approx(t_total, rel=1e-3)  # Heun at tol 2e-4 to the zone, then the finish


def test_predictor_outside_corrector_inside_on_a_relaxing_boundary() -> None:
    # G15 / D3C-07: a long shallow flaw just below the long-flaw depth limit; the Euler predictor crosses
    # the limit but the corrector (whose larger c relaxes the limit) is inside: a plain step, no event
    law = GrowthLaw(float(coefficient_to_si(1e-12, 3.0)), 3.0, 0.0)
    thick, width = 0.025, 1.0
    g = Grower(law, Spectrum.repeating([(0.0, 150e6, 1.0)], 1e5), 1, 1.0)
    c0 = 0.2
    a0 = float(g._long_limit(np.array([c0]), np.array([thick]))[0]) * (1 - 1e-2)
    st = GrowthState.start(np.array([a0]), np.array([c0]), np.array([thick]), np.array([width]), 1)
    idx = np.array([0])
    dk = g.ranges(idx, st.a, st.c, st.kind, st.thickness, st.width)
    above, stage_b, _ = g.active_set(idx, dk, st.a, st.c, st.kind)
    va0, vc0, _ = g.velocities(idx, st.a, st.c, st.kind, st.t, st.thickness, st.width, above, stage_b, None)
    h = 0.004256696454607871
    a1, c1, a2, c2 = g.heun(st, idx, np.array([h]), above, stage_b, (va0, vc0))
    assert np.any(g.boundary_excess(a1, c1, st.kind, st.thickness, st.width) > 0.0)  # predictor outside
    assert np.all(g.boundary_excess(a2, c2, st.kind, st.thickness, st.width) < 0.0)  # corrector inside
    st.h_next[0] = h
    g._step(st, idx, 1.0)
    assert st.event_count[0] == 0 and st.kind[0] == G.SURFACE and st.steps[0] == 1
    assert st.a[0] == a2[0] and st.c[0] == c2[0] and st.t[0] == pytest.approx(h)


def test_sensitivity_curves_follow_the_refined_grid() -> None:
    # E3C-04: a case that refines; the sensitivity variants are run on the refined grid
    law = steel_in_air(curve="mean", stage_correlation=0.5)
    sp = Spectrum.repeating([(80e6, 0.0, 1.0)], cycles_per_year=1e5, load_scale=prob.Uniform(0.9, 1.1))
    problem = cf.Assessment(cf.FlatPlate(B, W), cf.SurfaceFlaw(3e-3, 30e-3), STEEL, cf.StressState(pm=100e6))
    ext = ExtremeLoad(prob.Truncated(prob.Gumbel.from_mean_sd(240e6, 25e6), lo=0.0), 1.0)
    run = TimeDependentAssessment(problem, law, sp, ext, times=range(1, 6), leak_is_failure=False, shared_scale=False)
    res = run.solve(seed=2, samples=400, refine=True)
    assert res.refinements >= 1 and res.delta < 1.0
    assert res.sensitivity is not None and len(res.sensitivity) == 2
    for curve in res.sensitivity.values():
        assert curve.value.size == res.cumulative_upper.value.size
        assert np.all(np.diff(curve.value) >= 0.0)


def test_life_factor_is_refused_before_any_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    # T47 / E3C-07
    _, ext, law, sp = two_point_problem()
    problem = cf.Assessment(cf.FlatPlate(B, W), cf.SurfaceFlaw(1e-3, 40e-3), STEEL, cf.StressState(pm=100e6))
    run = TimeDependentAssessment(problem, law, sp.with_(life_factor=2.0), ext, times=[1, 2], leak_is_failure=False)
    assert np.isfinite(run.grow().a).all()  # deterministic runs accept the factor
    with pytest.raises(ValueError, match="life_factor"):
        run.solve(seed=0, samples=10)
    with pytest.raises(ValueError, match="life_factor"):  # refused before run() is reached
        monkeypatch.setattr(TimeDependentAssessment, "run", lambda *a, **k: pytest.fail("run() was called"))
        run.solve(seed=0, samples=10)


def test_repair_clears_the_removed_flaws_record_and_keeps_it_as_history() -> None:
    # E3C-05: a flaw that leaked at 0.94 years is repaired at year 2 by the same flaw again; the repair
    # flaw's breakthrough at 2.94 years is the sample's leak event (step 2); the first leak is history
    law = steel_in_air()
    sp = Spectrum.repeating([(120e6, 20e6, 1.0)], cycles_per_year=1e4)
    problem = cf.Assessment(cf.FlatPlate(B, W), cf.SurfaceFlaw(23e-3, 40e-3), STEEL, cf.StressState(pm=50e6))
    ext = ExtremeLoad(prob.Uniform(10e6, 10.001e6), 1.0)  # too low to fail anything structurally
    insp = Inspection(2.0, ConstantPoD(1.0), repair_flaw=(23e-3, 40e-3))
    plain = TimeDependentAssessment(problem, law, sp, ext, times=range(1, 6), leak_is_failure=False)
    t_first = float(plain.run(np.zeros((1, plain.dimension)))["state"].t_leak[0])
    assert 0.9 < t_first < 1.0
    run = TimeDependentAssessment(problem, law, sp, ext, times=range(1, 6), leak_is_failure=False, inspections=[insp])
    r = run.run(np.zeros((2, run.dimension)))
    assert r["removed_leaked"].all() and r["leak_step"].tolist() == [2, 2]
    np.testing.assert_allclose(r["state"].t_leak, 2.0 + t_first, rtol=1e-6)
    assert np.all(np.isnan(r["state"].t_ended))  # the removed flaw's end is not the repair flaw's
    res = run._assemble(r, run.grid, 0)
    assert res.events_any["removed_flaw_had_leaked"] == 1.0 and res.events_any["repaired"] == 1.0
    assert res.events_any["leak"] == 1.0 and res.events_any["ended_outside_validity"] == 0.0
    assert res.cumulative_upper.value.tolist() == [0.0] * 5
    # a small repair flaw that never leaks again: the removed flaw's leak is history only
    small_repair = Inspection(2.0, ConstantPoD(1.0), repair_flaw=(1e-3, 10e-3))
    run_s = TimeDependentAssessment(
        problem, law, sp, ext, times=range(1, 6), leak_is_failure=False, inspections=[small_repair]
    )
    r_s = run_s.run(np.zeros((1, run_s.dimension)))
    res_s = run_s._assemble(r_s, run_s.grid, 0)
    assert res_s.events_any["leak"] == 0.0 and res_s.events_any["removed_flaw_had_leaked"] == 1.0
    assert r_s["leak_step"].tolist() == [-1]
    # a flaw that never leaked before its repair: no history
    never = problem.with_(flaw=cf.SurfaceFlaw(1e-3, 10e-3))
    run_n = TimeDependentAssessment(never, law, sp, ext, times=range(1, 6), leak_is_failure=False, inspections=[insp])
    res_n = run_n._assemble(run_n.run(np.zeros((1, run_n.dimension))), run_n.grid, 0)
    assert res_n.events_any["repaired"] == 1.0 and res_n.events_any["removed_flaw_had_leaked"] == 0.0
    assert "P(detect)" in res.summary() and "removed_flaw_had_leaked" in res.summary()
    # when leaks count, the first leak fails the sample in step 0 and it is never inspected
    counted = TimeDependentAssessment(
        problem, law, sp, ext, times=range(1, 6), leak_is_failure=True, inspections=[insp]
    )
    rc = counted.run(np.zeros((1, counted.dimension)))
    assert rc["leak_step"].tolist() == [0] and not rc["repaired"].any() and not rc["removed_leaked"].any()
    assert counted._assemble(rc, counted.grid, 0).cumulative_upper.value.tolist() == [1.0] * 5


def test_inspection_repair_resets_the_integrator_flags() -> None:
    # T42: the sliding, carried-step, dormancy and override flags of a repaired sample are reset
    problem, ext, law, sp = two_point_problem()
    run = TimeDependentAssessment(problem, law, sp, ext, times=[1, 2], leak_is_failure=False)
    state = GrowthState.start(np.array([2e-3, 2e-3]), np.array([20e-3, 20e-3]), np.array([B, B]), np.array([W, W]), 1)
    state.sliding[0] = True
    state.h_next[:] = 0.1
    state.dormant[0] = True
    state.has_override[0] = True
    state.zero_cuts[0] = 3
    state.force[0] = True
    insp = Inspection(1.0, ConstantPoD(1.0), repair_flaw=(0.5e-3, 5e-3))
    info = run._inspect(state, insp, np.zeros((2, 1)), np.array([True, False]), np.array([True, False]))
    assert info["repaired_now"].tolist() == [True, False]  # the failed sample is not inspected
    assert not state.sliding[0].any() and not state.dormant[0] and not state.has_override[0]
    assert state.zero_cuts[0] == 0 and not state.force[0] and np.isnan(state.h_next[0])
    assert state.a[0] == 0.5e-3 and state.h_next[1] == 0.1 and state.a[1] == 2e-3


def test_conditioning_time_flag_drives_refinement() -> None:
    # T44 / P-11: discordance at the conditioning time alone requests a refinement
    problem, ext, law, sp = two_point_problem()
    run = TimeDependentAssessment(problem, law, sp, ext, times=[1, 2], leak_is_failure=False)
    res = run.solve(seed=0, samples=20, stage_sensitivity=False)
    assert not run._needs_refinement(res)
    flagged = dataclasses.replace(res, conditioning={"refine_at_conditioning_time": True})
    assert run._needs_refinement(flagged)


def test_block_weights_with_blocks_shorter_than_a_thousandth_of_a_year() -> None:
    # G45 / D3C-11: a relative tolerance at the block boundaries
    law = GrowthLaw(1e-22, 2.0, 0.0)
    sp = Spectrum.ordered([(100e6, 0.0, 10.0)] * 5, 1e5, "as_given")  # blocks of 1e-4 year
    g = Grower(law, sp, 1, 1.0)
    ends = g.block_ends
    assert ends is not None and ends[0] == pytest.approx(1e-4)
    t = np.array([0.5e-4, 1e-4 * (1 - 1e-13), 1e-4, 2.5e-4, 5e-4 * (1 - 1e-6)])
    w = g.block_weights(t)
    assert np.argmax(w, axis=1).tolist() == [0, 1, 1, 2, 4]  # the end of a block belongs to the next
    assert np.all(w.sum(axis=1) == 1e5)  # S3: one block at a time, never equal weights


def test_range_at_the_threshold_grows_and_a_negative_range_is_clamped() -> None:
    # G6: equality at the threshold counts as growing; G4: a negative combined range is zero
    thr = 50e6 * NMM
    law = GrowthLaw(float(coefficient_to_si(1e-12, 3.0)), 3.0, thr)
    g = Grower(law, Spectrum.repeating([(100e6, 0.0, 1.0)], 1.0), 1, 1.0)
    idx, kind = np.array([0]), np.array([G.SURFACE])
    dk = np.full((1, 1, 2), thr)
    above, _, _ = g.active_set(idx, dk, np.array([2e-3]), np.array([20e-3]), kind)
    assert above.all()
    above, _, _ = g.active_set(idx, dk * (1 - 1e-12), np.array([2e-3]), np.array([20e-3]), kind)
    assert not above.any()
    # pure bending on a deep flaw: the deep-point bending factor is negative, the range is clamped to zero
    from critflaw.solutions import plate

    a_deep, c_deep = np.array([0.9 * B]), np.array([1.8 * B])
    _, mb = plate.surface_flaw_mm_mb(a_deep, c_deep, np.array([B]), np.pi / 2)
    assert mb[0] < 0.0
    g_bend = Grower(law, Spectrum.repeating([(0.0, 100e6, 1.0)], 1.0), 1, 1.0)
    rng = g_bend.ranges(idx, a_deep, c_deep, kind, np.array([B]), np.array([W]))
    assert rng[0, 0, 0] == 0.0 and rng[0, 0, 1] > 0.0


def test_growth_law_validation_boundaries() -> None:
    # L5, L6, L9
    with pytest.raises(ValueError, match="m > m_b"):
        GrowthLaw(1e-12, 3.0, 0.0, a_coefficient_b=1e-12, m_b=3.0)
    with pytest.raises(ValueError, match="shallow_cap"):
        GrowthLaw(1e-12, 3.0, 0.0, shallow_cap=(0.0, 1.0))
    cap = (1e-3, 63.0 * NMM)
    law = GrowthLaw(1e-12, 3.0, 100.0 * NMM, shallow_cap=cap)
    g = Grower(law, Spectrum.repeating([(100e6, 0.0, 1.0)], 1.0), 1, 1.0)
    idx, kind = np.array([0]), np.array([G.SURFACE])
    assert g.threshold_at(idx, np.array([1e-3 * (1 - 1e-12)]), kind)[0] == pytest.approx(63.0 * NMM)
    assert g.threshold_at(idx, np.array([1e-3]), kind)[0] == pytest.approx(100.0 * NMM)  # 1 mm is not shallow


def test_events_before_the_first_failure_include_the_failure_step() -> None:
    # T38: a leak in the step of the first failure (leaks not counted as failure) is an event "before"
    run = TimeDependentAssessment.__new__(TimeDependentAssessment)
    run.leak_is_failure = False
    run.times = np.array([1.0, 2.0])
    run.spectrum = Spectrum.repeating([(1.0, 0.0, 1.0)], 1.0)
    run.law = GrowthLaw(1e-30, 3.0, 0.0)
    run.shared_scale = False
    run.inspections = ()
    state = GrowthState.start(np.array([1e-3, 1e-3]), np.array([2e-3, 2e-3]), np.array([B, B]), np.array([W, W]), 1)
    state.t_leak[:] = (1.5, 1.5)
    fake = {
        "g_min": np.array([[1.0, -1.0], [1.0, 1.0]]),
        "g_max": np.array([[1.0, -0.5], [1.0, 1.0]]),
        "any_invalid": np.zeros((2, 2), dtype=bool),
        "mode": np.full((2, 2), "curve", dtype=object),
        "leak_step": np.array([1, 1]),
        "ended_step": np.array([-1, -1]),
        "violations": 0,
        "violating_samples": 0,
        "inspections": [],
        "repaired": np.zeros(2, dtype=bool),
        "t_repair": np.full(2, np.nan),
        "removed_leaked": np.zeros(2, dtype=bool),
        "removed_ended": np.zeros(2, dtype=bool),
        "state": state,
        "problem": None,
        "invalid_start": 0.0,
        "grid": np.array([1.0, 2.0]),
        "n": 2,
    }
    res = run._assemble(fake, np.array([1.0, 2.0]), 0)
    assert res.events["leak"] == 1.0 and res.events_any["leak"] == 1.0 and res.modes["fracture"] == 0.5


# ---- round 5 (partner P-28, P-04, P-09, P-20) --------------------------------------------------


def _critical_load(problem: cf.Assessment, lo: float = 1e6, hi: float = 2e9) -> float:
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        ok = bool(problem.with_(loads=problem.loads.with_(pm=mid)).run(on_invalid="mask").acceptable)
        lo, hi = (mid, hi) if ok else (lo, mid)
    return 0.5 * (lo + hi)


def test_grow_counts_the_end_of_validity_as_failure_and_stops_the_cycles_there() -> None:
    # P-28: a flaw that reaches the end of the growth solution's validity before the first checkpoint
    law = GrowthLaw(float(coefficient_to_si(1e-10, 3.0)), 3.0, 0.0)
    problem = cf.Assessment(cf.FlatPlate(B, W), cf.SurfaceFlaw(2e-3, 20e-3), STEEL, cf.StressState(pm=50e6))
    sp = Spectrum.ordered([(150e6, 0.0, 1e7)], 1e7, "as_given")
    run = TimeDependentAssessment(
        problem, law, sp, ExtremeLoad(prob.Uniform(50e6, 50.001e6), 1.0), times=[1], leak_is_failure=False
    )
    h = run.grow(load=50e6)
    assert h.g[0] > 0.0 and np.all(np.isnan(h.g[1:]))  # acceptable at the start, ended before 0.2 years
    assert 0.0 < h.t_leak < h.t_ended < 0.2
    assert h.first_failure == h.t_ended and h.cycles_applied == pytest.approx(1e7 * h.t_ended)
    assert h.life == h.first_failure


def test_grow_checks_both_sides_of_a_breakthrough() -> None:
    # P-28: the pre-breakthrough state is the most severe in its step; a load that fails it but none of
    # the plain checkpoints gives the breakthrough time as the deterministic life
    law = steel_in_air()
    sp = Spectrum.repeating([(120e6, 20e6, 1.0)], cycles_per_year=1e4)
    problem = cf.Assessment(cf.FlatPlate(B, W), cf.SurfaceFlaw(23e-3, 40e-3), STEEL, cf.StressState(pm=50e6))
    ext = ExtremeLoad(prob.Uniform(100e6, 100.001e6), 1.0)
    run = TimeDependentAssessment(problem, law, sp, ext, times=[1, 2], leak_is_failure=False)
    one = TimeDependentAssessment(problem, law, sp, ext, times=[1], leak_is_failure=False)
    state = one.run(np.zeros((1, one.dimension)))["state"]  # the breakthrough is in the first step
    assert state.event_count[0] == 1
    pre = (
        state.event_pre[0:1, 0, 0].copy(),
        state.event_pre[0:1, 0, 1].copy(),
        state.event_pre[0:1, 0, 2].astype(np.intp),
    )
    grower = Grower(law, sp, 1, 2.0)

    def crit(snap: Any) -> float:
        lo, hi = 1e6, 2e9
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            g_s, v_s, _ = run._driving(problem, grower, snap, np.array([mid]))
            lo, hi = (mid, hi) if (v_s[0] and g_s[0] > 0.0) else (lo, mid)
        return 0.5 * (lo + hi)

    safe = run.grow(load=50e6)
    plain_snaps = [(safe.a[i : i + 1], safe.c[i : i + 1], safe.kind[i : i + 1]) for i in range(safe.times.size)]
    crit_plain = min(crit(s) for s in plain_snaps)
    crit_pre = crit(pre)
    assert crit_pre < crit_plain  # the boundary state is the most severe
    load = 0.5 * (crit_pre + crit_plain)
    h = run.grow(load=load)
    assert np.all(h.g[np.isfinite(h.g)] > 0.0)  # no plain checkpoint fails
    assert h.first_failure == pytest.approx(h.t_leak) and h.cycles_applied == pytest.approx(1e4 * h.t_leak)
    assert np.isnan(run.grow(load=0.99 * crit_pre).first_failure)  # below the boundary state's critical load: safe
    assert run.grow(load=1.01 * crit_plain).first_failure < h.t_leak


def test_grow_and_compare_orders_refuse_inspections() -> None:
    # P-28: inspection policies are not simulated deterministically
    law = GrowthLaw(1e-22, 2.0, 0.0)
    problem = cf.Assessment(cf.FlatPlate(B, W), cf.SurfaceFlaw(2e-3, 20e-3), STEEL, cf.StressState(pm=50e6))
    sp = Spectrum.ordered([(100e6, 0.0, 1e4), (50e6, 0.0, 1e4)], 1e4, "as_given")
    insp = [Inspection(1.0, ConstantPoD(1.0))]
    run = TimeDependentAssessment(
        problem,
        law,
        sp,
        ExtremeLoad(prob.Uniform(50e6, 50.001e6), 1.0),
        times=[1, 2],
        leak_is_failure=False,
        inspections=insp,
    )
    with pytest.raises(ValueError, match="inspections"):
        run.grow(load=50e6)
    with pytest.raises(ValueError, match="inspections"):
        run.compare_orders(load=50e6)


def test_controller_reduction_below_the_minimum_step_enters_the_finish_for_a_through_flaw() -> None:
    # P-04 (the partner's counterexample): c = 0.1, W = 0.5, m = 8.16, initial velocity 1e9 m/year under an
    # ordered spectrum: the size-limited first step is 2e-12 years and the controller would reduce it
    # below the minimum; the finish takes over and the end time agrees with scipy
    from scipy import integrate

    w, nu = 0.5, 1e6
    dk0 = math.cos(math.pi * 0.1 / w) ** -0.5 * 100e6 * math.sqrt(math.pi * 0.1)
    law = GrowthLaw(1e9 / (dk0**8.16 * nu), 8.16, 0.0)
    sp = Spectrum.ordered([(100e6, 0.0, nu)], nu, "as_given")
    st = GrowthState.start(np.array([1.0]), np.array([0.1]), np.array([1.0]), np.array([w]), 1)
    st.kind[:] = G.THROUGH
    g = Grower(law, sp, 1, 1.0)
    assert g.eta * 0.1 / 1e9 < 100.0 * g.min_step  # the first step is already within the finish band
    g.advance(st, 1.0)
    ref = integrate.quad(_through_reference(law, [(100e6, nu)], w), 0.1, 0.25, epsabs=0, epsrel=1e-12, limit=400)[0]
    assert st.kind[0] == G.ENDED and st.t_ended[0] == pytest.approx(ref, rel=1e-10)
    # a surface flaw in the same situation still raises (the rate is not resolvable)
    st_s = GrowthState.start(np.array([1e-3]), np.array([0.1]), np.array([B]), np.array([w]), 1)
    with pytest.raises(RuntimeError, match="minimum step"):
        Grower(GrowthLaw(1e30, 8.16, 0.0), sp, 1, 1.0).advance(st_s, 1.0)


def test_inspection_reports_both_pairings_and_the_interval_probability() -> None:
    # P-09: the F13 values before and after an inspection under both pairings
    problem, ext, law, sp = two_point_problem()
    insp = Inspection(2.0, ConstantPoD(0.5), action="repair")
    run = TimeDependentAssessment(problem, law, sp, ext, times=range(1, 5), leak_is_failure=False, inspections=[insp])
    res = run.solve(seed=0, samples=400, stage_sensitivity=False)
    info = res.inspections[0]
    k = int(info["step"])
    for pairing, curve in (("upper", res.cumulative_upper), ("lower", res.cumulative_lower)):
        assert info[f"cumulative_{pairing}_before"] == pytest.approx(curve.value[k])
        assert info[f"cumulative_{pairing}_after_next_step"] == pytest.approx(curve.value[k + 1])
    assert info["hazard_upper_next_step"] == pytest.approx(res.hazard_upper.value[k + 1])
    assert info["hazard_lower_next_step"] == pytest.approx(res.hazard_lower.value[k + 1])
    assert info["interval_probability_next_step"] == pytest.approx(res.interval_probability.value[k + 1])
    lo, hi = info["hazard_conservative_next_step"]
    assert lo <= info["hazard_lower_next_step"] + 1e-12 and hi >= info["hazard_upper_next_step"] - 1e-12
    text = res.summary()
    assert "over all N samples, eligibility under the lower pairing" in text and "next-step hazard" in text


def test_subset_two_mode_case_and_below_resolution_reporting() -> None:
    # P-20 / benchmark 14 (two-mode, reduced): early failures from large initial flaws (1% of the
    # population, hazard p1 per year) and late failures from the small flaws (hazard p2); no growth, so
    # Pf(0, t) = 0.01 (1 - (1 - p1)^t) + 0.99 (1 - (1 - p2)^t) exactly
    small = cf.Assessment(cf.FlatPlate(B, W), cf.SurfaceFlaw(3e-3, 40e-3), STEEL, cf.StressState(pm=100e6))
    big = small.with_(flaw=cf.SurfaceFlaw(10e-3, 40e-3))
    crit_s, crit_b = _critical_load(small), _critical_load(big)
    assert crit_b < crit_s
    gumbel = prob.Gumbel.from_mean_sd(0.75 * crit_b, 0.08 * crit_b)
    ext = ExtremeLoad(prob.Truncated(gumbel, lo=0.0), 1.0)
    p1, p2 = float(ext.distribution.sf(crit_b)), float(ext.distribution.sf(crit_s))
    assert 0.005 < p1 < 0.02 and p2 < 0.1 * p1
    exact = [0.01 * (1 - (1 - p1) ** t) + 0.99 * (1 - (1 - p2) ** t) for t in range(1, 6)]
    problem = small.with_(flaw=cf.SurfaceFlaw(prob.Empirical([3e-3, 10e-3], [0.99, 0.01], name="a0"), 40e-3))
    run = TimeDependentAssessment(
        problem,
        GrowthLaw(1e-40, 3.0, 1e9),
        Spectrum.repeating([(1e6, 0, 1)], 1.0),
        ext,
        times=range(1, 6),
        leak_is_failure=False,
    )
    covered, shares = 0, []
    for seed in range(4):
        res = run.solve(
            seed=seed, samples=1000, subset=True, replicates=4, samples_per_level=1000, stage_sensitivity=False
        )
        assert res.subset is not None and res.subset.subset is not None and res.subset_curve is not None
        interval = res.subset.subset.interval
        assert interval is not None
        covered += interval.contains(exact[-1])
        assert res.subset.pf == pytest.approx(exact[-1], rel=0.5)
        shares.append(res.subset_curve["share_mean"])
    assert covered >= 3
    expected = np.array(exact) / exact[-1]
    mean_share = np.mean(shares, axis=0)
    assert np.all(mean_share / expected < 1.5) and np.all(expected / mean_share < 1.5)  # all shares exceed 0.02
    # below resolution: a bounded load that cannot fail the flaw before it has grown: the share by year 1
    # is exactly zero and is reported as below resolution, with the curve value withheld
    law = steel_in_air()
    sp = Spectrum.repeating([(80e6, 0.0, 1.0)], cycles_per_year=3e4)
    grown = cf.Assessment(cf.FlatPlate(B, W), cf.SurfaceFlaw(3e-3, 30e-3), STEEL, cf.StressState(pm=100e6))
    plain = TimeDependentAssessment(
        grown, law, sp, ExtremeLoad(prob.Uniform(1e6, 1.001e6), 1.0), times=range(1, 6), leak_is_failure=False
    )
    sizes = plain.grow()
    year1 = grown.with_(flaw=cf.SurfaceFlaw(float(sizes.a[5]), float(2 * sizes.c[5])))
    crit_1 = _critical_load(year1)
    bounded = ExtremeLoad(
        prob.Truncated(prob.Gumbel.from_mean_sd(0.7 * crit_1, 0.05 * crit_1), lo=0.0, hi=0.999 * crit_1), 1.0
    )
    run2 = TimeDependentAssessment(grown, law, sp, bounded, times=range(1, 6), leak_is_failure=False)
    res2 = run2.solve(seed=0, samples=500, subset=True, replicates=2, samples_per_level=500, stage_sensitivity=False)
    assert res2.subset_curve is not None and res2.subset_curve["below_resolution"][0]
    assert np.isnan(res2.subset_curve["curve"][0]) and res2.subset_curve["share_mean"][0] == 0.0
    assert "below resolution (curve value withheld) at t = 1" in res2.summary()


def test_surface_flaw_controller_step_between_one_and_a_hundred_minimum_steps_advances() -> None:
    # P-29: the minimum-step error for surface flaws is at min_step (F8 item 5), not at the through-flaw
    # finish band of 100 min_step; a rejected trial that reduces to ~20 min_step advances
    a0, c0 = 2e-3, 20e-3
    nu = 1e6
    sp = Spectrum.ordered([(100e6, 0.0, nu)], nu, "as_given")
    probe = Grower(GrowthLaw(1e-12, 3.0, 0.0), sp, 1, 1.0)
    dk = probe.ranges(
        np.array([0]), np.array([a0]), np.array([c0]), np.array([G.SURFACE]), np.array([B]), np.array([W])
    )
    dk_a = float(dk[0, 0, 0])
    for h_target, outcome in ((30.0 * probe.min_step, "advances"), (0.3 * probe.min_step, "raises")):
        v_a = probe.eta * a0 / h_target  # the size-limited first step is h_target
        law = GrowthLaw(v_a / (dk_a**3.0 * nu), 3.0, 0.0)
        st = GrowthState.start(np.array([a0]), np.array([c0]), np.array([B]), np.array([W]), 1)
        g = Grower(law, sp, 1, 1.0)
        if outcome == "advances":
            g.advance(st, 2.0 * h_target)
            assert st.steps[0] >= 1 and st.a[0] > a0 and st.kind[0] == G.SURFACE
        else:
            with pytest.raises(RuntimeError, match="minimum step"):
                g.advance(st, 2.0 * h_target)


def test_pass_expectation_weights_of_a_repeating_spectrum_with_unequal_cycles() -> None:
    # S4.5 (E4C-03): the velocity is the cycle-weighted expectation over the pass: 1 : 9 cycles
    from scipy import integrate

    w, nu = 0.5, 1e6
    law = GrowthLaw(float(coefficient_to_si(1e-12, 3.0)), 3.0, 0.0)
    sp = Spectrum.repeating([(100e6, 0.0, 1.0), (40e6, 0.0, 9.0)], nu)
    g = Grower(law, sp, 1, 1.0)
    idx, kind = np.array([0]), np.array([G.THROUGH])
    c = np.array([0.2])
    dk = g.ranges(idx, np.array([1.0]), c, kind, np.array([1.0]), np.array([w]))
    above, sb, _ = g.active_set(idx, dk, np.array([1.0]), c, kind)
    _, vc, _ = g.velocities(
        idx, np.array([1.0]), c, kind, np.array([0.0]), np.array([1.0]), np.array([w]), above, sb, None
    )
    expected = sum(float(law.rate(float(dk[0, b, 1]))) * nu * n for b, n in ((0, 0.1), (1, 0.9)))
    assert vc[0] == pytest.approx(expected, rel=1e-12)
    # and through the finish against scipy with the same weights
    inv_v = _through_reference(law, [(100e6, 0.1 * nu), (40e6, 0.9 * nu)], w)
    t_total = integrate.quad(inv_v, 0.225, 0.25, epsabs=0, epsrel=1e-13)[0]
    st = GrowthState.start(np.array([1.0]), np.array([0.225]), np.array([1.0]), np.array([w]), 2)
    st.kind[:] = G.THROUGH
    g.advance(st, 1.0)
    assert st.t_ended[0] == pytest.approx(t_total, rel=1e-10)


def test_lower_pairing_uses_the_event_checkpoints() -> None:
    # T40 (E4C-04a): an event-side value above every plain checkpoint sets g_max
    problem = cf.Assessment(cf.FlatPlate(B, W), cf.SurfaceFlaw(23e-3, 40e-3), STEEL, cf.StressState(pm=50e6))
    law = steel_in_air()
    sp = Spectrum.repeating([(120e6, 20e6, 1.0)], cycles_per_year=1e4)
    ext = ExtremeLoad(prob.Uniform(100e6, 100.001e6), 1.0)
    run = TimeDependentAssessment(problem, law, sp, ext, times=[1], leak_is_failure=False)
    grower = Grower(law, sp, 1, 1.0)
    m = ext.step_maximum(np.zeros(1), 1.0)  # the load run() uses for u = 0
    r = run.run(np.zeros((1, run.dimension)))
    state = r["state"]
    post = (
        state.event_post[0:1, 0, 0].copy(),
        state.event_post[0:1, 0, 1].copy(),
        state.event_post[0:1, 0, 2].astype(np.intp),
    )
    g_post = float(run._driving(problem, grower, post, m)[0][0])
    hist = run.grow(load=float(m[0]))  # the plain checkpoints of the step
    assert g_post > np.nanmax(hist.g)  # the post-breakthrough through flaw at the event time is the least severe
    assert r["g_max"][0, 0] == pytest.approx(g_post, rel=1e-12)


def test_catalogue_shallow_cap_depth_boundary() -> None:
    # L9 (E4C-04d): the catalogue's cap applies below 1 mm, not at it
    from critflaw.fatigue import laws as L

    law = steel_in_air()
    assert L.SHALLOW_CAP[0] == 1.0e-3
    thr = law.effective_threshold([1e-3 * (1 - 1e-12), 1e-3, 2e-3])
    np.testing.assert_allclose(thr / NMM, [63.0, 63.0, 63.0])  # the air threshold is 63 already: the cap is inert
    marine = GrowthLaw(1e-12, 3.0, 100.0 * NMM, shallow_cap=L.SHALLOW_CAP)
    np.testing.assert_allclose(marine.effective_threshold([1e-3 * (1 - 1e-12), 1e-3]) / NMM, [63.0, 100.0])
