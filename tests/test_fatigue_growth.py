"""Phase 3, Part A: growth laws and the growth integrator (design note phase 3, F1 to F9, benchmarks 1 to 9)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from critflaw.fatigue import laws as L
from critflaw.fatigue.growth import ENDED, SURFACE, THROUGH, Grower, GrowthState
from critflaw.fatigue.spectrum import ExtremeLoad, Spectrum
from critflaw.prob import distributions as prob


def _load_oracle() -> ModuleType:
    module = ModuleType("partner_reference_fatigue")
    source = (Path(__file__).parent / "oracles" / "partner_reference_fatigue.py").read_text(encoding="utf-8")
    exec(compile(source, "partner_reference_fatigue", "exec"), module.__dict__)  # reviewed: math only
    return module


oracle = _load_oracle()
ORACLE_CASES = json.loads(
    (Path(__file__).parent.parent / "tests/oracles/partner_oracle_fatigue.json").read_text(encoding="utf-8")
)["cases"]
NMM = float(L.dk_to_si(1.0))  # one N/mm^(3/2) in Pa sqrt(m)


# ---- laws ------------------------------------------------------------------------------------


def test_transition_by_intersection_reproduces_the_printed_points() -> None:
    for (env, sr), (mean_printed, upper_printed) in L.PRINTED_TRANSITIONS.items():
        (aa, ma, aup, _), (ab, mb, bup, _) = L.table_entry(env, sr)
        assert float(L.transition_point(aa, ma, ab, mb)) == pytest.approx(mean_printed, rel=0.015)
        assert float(L.transition_point(aup, ma, bup, mb)) == pytest.approx(upper_printed, rel=0.015)
        law = L.recommended_law(env, sr, "upper")
        assert float(law.transition()) / NMM == pytest.approx(upper_printed, rel=0.015)  # type: ignore[arg-type]


def test_upper_bound_coefficients_follow_from_mean_and_scatter() -> None:
    for env in ("air", "marine-free", "marine-cp850", "marine-cp1100"):
        for sr in ("low", "high"):
            for a_mean, _, a_upper, sd in L.table_entry(env, sr):
                assert a_mean * 10 ** (2 * sd) == pytest.approx(a_upper, rel=0.003)  # printed to 3 figures


def test_laws_against_the_partner_oracle() -> None:
    for case in ORACLE_CASES:
        fn, args, expected = case["inputs"]["function"], case["inputs"]["arguments"], case["expected"]
        if fn == "transition_point":
            got = float(L.transition_point(*args))
        elif fn == "upper_from_mean":
            got = args[0] * 10 ** (2 * args[1])
        elif fn == "to_si":
            got = float(L.coefficient_to_si(*args))
        elif fn == "threshold_unwelded_steel":
            got = float(L.threshold_unwelded_steel(args[0])) / NMM
        elif fn == "rate_mm_per_cycle":
            dk, stages, dk0 = args
            law = L.GrowthLaw(
                float(L.coefficient_to_si(stages[0][0], stages[0][1])),
                stages[0][1],
                dk0 * NMM,
                float(L.coefficient_to_si(stages[1][0], stages[1][1])) if len(stages) > 1 else None,
                stages[1][1] if len(stages) > 1 else None,
            )
            got = float(law.rate(dk * NMM)) / 1e-3
        elif fn == "cycles_through_crack":
            ai, af, ds, a_si, m = args
            got = cycles_closed_form(ai, af, ds, a_si, m)
        else:
            assert "error_type" in expected  # life_fixed_step: the partner declared it underdetermined
            continue
        want = expected["value"]
        if case["name"] == "transition_freely_corroding_high_upper":
            # The partner's hand-computed table entry is an arithmetic slip (it used 7.638 for
            # log10(7.48e-7) - log10(1.72e-13) = 6.638); its own transition_point() gives 10^2.8738 = 747.7,
            # which is the printed transition point 748. The value from its code is used.
            want = math.log10(oracle.transition_point(*args))
        if expected.get("transform") == "log10":
            assert math.log10(got) == pytest.approx(want, abs=1e-7), case["name"]  # the table is hand-computed
        else:
            assert got == pytest.approx(want, rel=1e-9, abs=1e-12), case["name"]


def test_unit_conversion_gives_the_same_rate_from_both_unit_systems() -> None:
    for env in ("air", "marine-free"):
        (aa, ma, _, _), (ab, mb, _, _) = L.table_entry(env, "high")
        law = L.recommended_law(env, "high", "mean-fixed", threshold_nmm=0.0, welded=False)
        for dk_nmm in (80.0, 300.0, 2000.0):
            mm_per_cycle = min(aa * dk_nmm**ma, ab * dk_nmm**mb)
            assert float(law.rate(dk_nmm * NMM)) == pytest.approx(mm_per_cycle * 1e-3, rel=1e-12)
    assert float(L.dk_to_si(63.0)) == pytest.approx(1.992e6, rel=1e-3)


def test_law_construction_rules_and_thresholds() -> None:
    with pytest.raises(ValueError, match="m > m_b"):
        L.GrowthLaw(1e-12, 2.0, 0.0, 1e-13, 3.0)
    with pytest.raises(ValueError, match="stage_correlation"):
        L.steel_in_air(curve="mean")
    random = L.steel_in_air(curve="mean", stage_correlation=0.5)
    assert random.is_random and isinstance(random.a_coefficient, prob.LogNormal)
    # log10 A normal with the published SD: the lognormal's sigma_ln
    sd = L.table_entry("air", "high")[0][3]
    assert random.a_coefficient.sigma_ln == pytest.approx(sd * np.log(10.0), rel=1e-12)
    fixed = L.steel_in_air()
    assert not fixed.is_random and fixed.stage_correlation is None
    # the shallow cap: threshold 63 for a < 1 mm even when a larger threshold is declared
    law = L.GrowthLaw(1e-30, 3.0, 100.0 * NMM, shallow_cap=L.SHALLOW_CAP)
    np.testing.assert_allclose(law.effective_threshold(np.array([0.5e-3, 2e-3])) / NMM, [63.0, 100.0])
    np.testing.assert_allclose(
        L.threshold_unwelded_steel(np.array([-1.0, 0.0, 0.25, 0.5, 0.9])) / NMM, [170, 170, 116.5, 63, 63]
    )
    # stage correlation: SD of ln(transition) from the two scatters
    for rho in (0.0, 0.5, 1.0):
        law = L.steel_in_air(curve="mean", stage_correlation=rho)
        s_a, s_b = L.table_entry("air", "high")[0][3], L.table_entry("air", "high")[1][3]
        expected = np.log(10.0) * np.sqrt(s_a**2 + s_b**2 - 2 * rho * s_a * s_b) / (law.m - law.m_b)  # type: ignore[operator]
        rng = np.random.default_rng(1)
        z1, z2 = rng.standard_normal(200_000), rng.standard_normal(200_000)
        la = law.a_coefficient.from_normal_score(z1)
        lb = law.a_coefficient_b.from_normal_score(rho * z1 + np.sqrt(1 - rho**2) * z2)  # type: ignore[union-attr]
        sample = np.std(np.log(L.transition_point(la, law.m, lb, law.m_b)))
        assert sample == pytest.approx(expected, rel=0.01)


# ---- integrator ------------------------------------------------------------------------------


def cycles_closed_form(a_i: float, a_f: float, ds: float, a_si: float, m: float) -> float:
    if m == 2.0:
        return math.log(a_f / a_i) / (a_si * (ds * math.sqrt(math.pi)) ** 2)
    return (a_f ** (1 - m / 2) - a_i ** (1 - m / 2)) / ((1 - m / 2) * a_si * (ds * math.sqrt(math.pi)) ** m)


def grow_until(grower: Grower, state: GrowthState, size: float, which: str, t_max: float) -> float:
    """Time at which ``a`` (or ``c``) first reaches ``size`` on sample 0, by stepping and interpolation."""
    idx = np.array([0])
    while state.t[0] < t_max and state.kind[0] <= THROUGH:
        t0, s0 = state.t[0], getattr(state, which)[0]
        grower._step(state, idx, t_max)
        s1 = getattr(state, which)[0]
        if s1 >= size:
            return float(t0 + (state.t[0] - t0) * (size - s0) / (s1 - s0))
    raise AssertionError("size not reached")


@pytest.mark.parametrize(("m", "a_nmm"), [(3.0, 5.21e-13), (2.0, 1e-10), (4.0, 1e-15)])
def test_through_flaw_life_against_the_closed_form(m: float, a_nmm: float) -> None:
    a_si = float(L.coefficient_to_si(a_nmm, m))
    ds, a_i, a_f = 100e6, 2e-3, 10e-3
    exact = cycles_closed_form(a_i, a_f, ds, a_si, m)
    nu = 1e6
    for eta, tol, worst in ((0.02, 2e-4, 5e-3), (0.01, 1e-4, 5e-3)):
        law = L.GrowthLaw(a_si, m, 0.0)
        state = GrowthState.start(np.array([1.0]), np.array([a_i]), np.array([1.0]), np.array([1e9]), 1)  # width: Y = 1
        state.kind[:] = THROUGH
        grower = Grower(law, Spectrum.repeating([(ds, 0.0, 1.0)], nu), 1, 10.0, eta=eta, tol=tol)
        t = grow_until(grower, state, a_f, "c", 10.0)
        assert t * nu == pytest.approx(exact, rel=worst)
    # convergence: with the controller on the error is linear in tol
    errors = []
    for tol in (4e-4, 1e-4):
        law = L.GrowthLaw(a_si, m, 0.0)
        state = GrowthState.start(np.array([1.0]), np.array([a_i]), np.array([1.0]), np.array([1e9]), 1)
        state.kind[:] = THROUGH
        grower = Grower(law, Spectrum.repeating([(ds, 0.0, 1.0)], nu), 1, 10.0, eta=1.0, tol=tol)
        errors.append(abs(grow_until(grower, state, a_f, "c", 10.0) * nu / exact - 1.0))
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.5)


def test_surface_flaw_ranges_against_the_partner_k_solution() -> None:
    law = L.steel_in_air()
    sp = Spectrum.repeating([(100e6, 30e6, 1.0)], 1e5)
    for a, c in ((2e-3, 5e-3), (8e-3, 9e-3), (15e-3, 40e-3)):
        st = GrowthState.start(np.array([a]), np.array([c]), np.array([25e-3]), np.array([0.3]), 1)
        g = Grower(law, sp, 1, 1.0)
        dk = g.ranges(np.array([0]), st.a, st.c, st.kind, st.thickness, st.width)[0, 0]
        deep = oracle.surface_flaw_dk(a, c, 25e-3, 0.3, 100e6, 30e6, math.pi / 2)
        surf = oracle.surface_flaw_dk(a, c, 25e-3, 0.3, 100e6, 30e6, 0.0)
        assert dk[0] == pytest.approx(deep, rel=1e-12) and dk[1] == pytest.approx(surf, rel=1e-12)


def cycle_by_cycle(
    a: float,
    c: float,
    B: float,
    W: float,
    dsm: float,
    dsb: float,
    stages: list,
    dk0: float,
    targets: list[float],
    max_cycles: int = 3_000_000,
) -> tuple[list[float], float | None]:
    """One cycle per step with the partner's rates (SI): cycles to each target depth, and to breakthrough."""
    out: list[float] = []
    n = 0
    while n < max_cycles:
        va, vc = oracle.surface_growth_rates_si(a, c, B, W, dsm, dsb, stages, dk0)
        if va == 0.0 and vc == 0.0:
            break
        a, c, n = a + va, c + vc, n + 1
        while len(out) < len(targets) and a >= targets[len(out)]:
            out.append(n)
        if a >= B:
            return out, n
    return out, None


def test_surface_flaw_growth_against_a_cycle_by_cycle_reference() -> None:
    # two stages, a threshold the surface point starts below, coupling of a and c, breakthrough
    stages = [[4.8e-18, 5.10], [5.86e-13, 2.88]]  # air, mean, R >= 0.5 (mm, N units)
    dk0 = 63.0
    a0, c0, B, W = 0.5e-3, 6.0e-3, 8e-3, 0.3
    dsm, dsb = 120e6, 0.0
    targets = [1e-3, 2e-3, 4e-3]
    ref, ref_break = cycle_by_cycle(a0, c0, B, W, dsm, dsb, stages, dk0, targets, max_cycles=6_000_000)
    assert ref_break is not None and len(ref) == 3
    law = L.recommended_law("air", "high", "mean-fixed", threshold_nmm=dk0, welded=False)
    nu = 1e5
    st = GrowthState.start(np.array([a0]), np.array([c0]), np.array([B]), np.array([W]), 1)
    g = Grower(law, Spectrum.repeating([(dsm, dsb, 1.0)], nu), 1, 100.0)
    idx = np.array([0])
    # the surface point is below threshold at the start and wakes later
    dk = g.ranges(idx, st.a, st.c, st.kind, st.thickness, st.width)[0, 0]
    assert dk[1] < dk0 * NMM < dk[0]
    times = []
    for target in targets:
        times.append(grow_until(g, st, target, "a", 100.0))
    while st.kind[0] == SURFACE:
        g._step(st, idx, 100.0)
    for t, n in zip(times, ref, strict=True):
        assert t * nu == pytest.approx(n, rel=5e-3)
    assert st.t_leak[0] * nu == pytest.approx(ref_break, rel=5e-3)
    assert st.kind[0] == THROUGH and st.c[0] > c0 and st.leak_trigger[0] == 1
    assert st.event_count[0] == 1 and tuple(st.event_pre[0, 0, :2]) == pytest.approx((B, st.c[0]), rel=1e-9)


def test_dormant_flaw_and_wake_up_by_length_growth() -> None:
    law = L.GrowthLaw(float(L.coefficient_to_si(5.86e-13, 2.88)), 2.88, 63.0 * NMM)
    sp = Spectrum.repeating([(5e6, 0.0, 1.0)], 1e5)
    st = GrowthState.start(np.array([1e-3]), np.array([2e-3]), np.array([25e-3]), np.array([0.5]), 1)
    g = Grower(law, sp, 1, 10.0)
    g.advance(st, 10.0)
    assert st.dormant[0] and st.t[0] == 10.0 and st.a[0] == 1e-3
    # a deep point below threshold that wakes when c grows: pure bending, long flaw
    law2 = L.GrowthLaw(float(L.coefficient_to_si(1e-12, 3.0)), 3.0, 50.0 * NMM)
    sp2 = Spectrum.repeating([(0.0, 60e6, 1.0)], 1e5)
    # a deep long flaw under bending: the deepest point's bending factor is small or negative
    found = None
    for ab in np.linspace(0.5, 0.9, 9):
        for ac in np.linspace(0.15, 0.5, 8):
            a, c = ab * 25e-3, ab * 25e-3 / ac
            if 2 * c > 0.8 * 0.5:
                continue
            st2 = GrowthState.start(np.array([a]), np.array([c]), np.array([25e-3]), np.array([0.5]), 1)
            g2 = Grower(law2, sp2, 1, 50.0)
            dk = g2.ranges(np.array([0]), st2.a, st2.c, st2.kind, st2.thickness, st2.width)[0, 0]
            if dk[0] < 50.0 * NMM < dk[1]:
                found = (a, c)
                break
        if found:
            break
    assert found is not None  # deepest point arrested, surface point growing
    st2 = GrowthState.start(np.array([found[0]]), np.array([found[1]]), np.array([25e-3]), np.array([0.5]), 1)
    g2 = Grower(law2, sp2, 1, 50.0)
    g2.advance(st2, 50.0)
    assert st2.c[0] > found[1] and not st2.dormant[0]


def test_terminal_steps_land_on_the_boundaries_and_the_through_flaw_finishes_by_quadrature() -> None:
    law = L.steel_in_air()
    sp = Spectrum.repeating([(100e6, 20e6, 1.0)], 2e5)
    # just below breakthrough
    st = GrowthState.start(np.array([24.99e-3]), np.array([40e-3]), np.array([25e-3]), np.array([0.5]), 1)
    g = Grower(law, sp, 1, 40.0)
    g._step(st, np.array([0]), 40.0)
    assert st.kind[0] == THROUGH and st.a[0] == 25e-3 and np.isfinite(st.t_leak[0]) and st.steps[0] == 1
    # just below the width limit of the surface solution: leak_assumed with the trigger recorded
    st = GrowthState.start(np.array([5e-3]), np.array([0.3999 * 0.5]), np.array([25e-3]), np.array([0.5]), 1)
    g = Grower(law, sp, 1, 40.0)
    g.advance(st, 40.0)
    assert np.isfinite(st.t_leak_assumed[0]) and st.leak_trigger[0] == 4
    # a through flaw reaching the width limit: finite time, status ended, for both exponents
    for m, a_nmm in ((3.0, 1e-12), (8.16, 3e-29)):  # the second: about 1e-6 m per cycle at the start
        law_m = L.GrowthLaw(float(L.coefficient_to_si(a_nmm, m)), m, 0.0)
        st = GrowthState.start(np.array([1.0]), np.array([0.1]), np.array([1.0]), np.array([0.5]), 1)
        st.kind[:] = THROUGH
        g = Grower(law_m, Spectrum.repeating([(100e6, 0.0, 1.0)], 1e6), 1, 1e5)  # a life far longer than the stage
        g.advance(st, 1e5)
        assert st.kind[0] == ENDED and np.isfinite(st.t_ended[0]) and st.c[0] == 0.25
        # reference: size quadrature of 1 / v from 0.1 to W/2 with scipy
        from scipy import integrate

        def inv_v(c: float, law_m: L.GrowthLaw = law_m) -> float:
            dk = (math.cos(math.pi * c / 0.5)) ** -0.5 * 100e6 * math.sqrt(math.pi * c)
            return 1.0 / (float(law_m.rate(dk)) * 1e6)

        ref = integrate.quad(inv_v, 0.1, 0.25, epsabs=0, epsrel=1e-10, limit=400)[0]
        assert st.t_ended[0] == pytest.approx(ref, rel=2e-3)


def test_spectrum_order_cases_from_the_review() -> None:
    # normalised: rate 0.001 dK^2, dK_H = 2 sqrt(a), dK_L = 0.9 sqrt(a), threshold 1; 100 high then 100 low cycles
    law = L.GrowthLaw(0.001, 2.0, 1.0)
    ds_h, ds_l = 2.0 / math.sqrt(math.pi), 0.9 / math.sqrt(math.pi)

    def final_size(spectrum: Spectrum) -> float:
        st = GrowthState.start(np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1e12]), 2)
        st.kind[:] = THROUGH
        g = Grower(law, spectrum, 1, 1.0)
        g.advance(st, 100.0 if spectrum.kind == "repeating" else spectrum.total_years())  # 200 cycles
        return float(st.c[0])

    high_first = final_size(Spectrum.ordered([(ds_h, 0, 100), (ds_l, 0, 100)], 200.0, "as_given"))
    low_first = final_size(Spectrum.ordered([(ds_h, 0, 100), (ds_l, 0, 100)], 200.0, "smallest_first"))
    assert high_first == pytest.approx(math.exp(0.4) * math.exp(0.081), rel=5e-4)
    assert low_first == pytest.approx(math.exp(0.4), rel=5e-4)  # the low block is dormant first
    mixed = final_size(Spectrum.repeating([(ds_h, 0, 1), (ds_l, 0, 1)], 2.0))
    assert mixed == pytest.approx(1.5500, rel=2e-3)
    assert Spectrum.ordered([(1, 0, 1), (2, 0, 1)], 1.0, "largest_first").membrane[0] == 2.0
    with pytest.raises(ValueError, match="order"):
        Spectrum.ordered([(1, 0, 1)], 1.0, "worst")  # type: ignore[arg-type]
    # a pass that is too long for the expectation form is refused
    long = Spectrum.repeating([(ds_h, 0, 100)], 200.0)
    st = GrowthState.start(np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1e12]), 1)
    st.kind[:] = THROUGH
    with pytest.raises(ValueError, match="one pass"):
        Grower(law, long, 1, 1.0).advance(st, 0.5)


def test_extreme_load_step_maximum() -> None:
    ext = ExtremeLoad(prob.Truncated(prob.Gumbel.from_mean_sd(180e6, 18e6), lo=0.0), 1.0)
    u = np.array([-3.0, 0.0, 2.0, 8.0])
    # a one-year step: the distribution itself
    np.testing.assert_allclose(ext.step_maximum(u, 1.0), ext.distribution.from_normal_score(u), rtol=1e-10)
    # a quarter-year step: F(x)^(1/4) = Phi(u)  =>  F(x) = Phi(u)^4
    from scipy import stats

    x = ext.step_maximum(u, 0.25)
    np.testing.assert_allclose(ext.distribution.cdf(x), stats.norm.cdf(u) ** 4, rtol=1e-9)
    assert np.all(np.diff(x) > 0) and np.isfinite(x[-1])
    with pytest.raises(ValueError):
        ExtremeLoad(prob.Normal(180e6, 18e6))
