"""Phase 6: the reliability core on the synthetic curve family of the Phase 5 tests — batch bounds equal the
single-realisation bounds; the classification order (collapse first); the Pf bound pair brackets the true Pf of
the synthetic 'FE'; the crossed ANOVA components are non-negative and the interval is labelled."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from critflaw.option3 import emulator as EM
from critflaw.option3 import reliability as R
from critflaw.option3.design import LOWER, UPPER, sobol_points
from test_option3_emulator import schedule, synth_curve, true_ratio  # the same synthetic 'FE'

DIM = 7  # five emulator inputs, then a load factor and a toughness variable


def to_x(u: np.ndarray) -> np.ndarray:
    """The first five standard normals mapped into the box (a central sub-box, so all realisations are inside)."""
    p = stats.norm.cdf(u[:, :5])
    return LOWER + (0.15 + 0.7 * p) * (UPPER - LOWER)


def problem(u: np.ndarray):  # type: ignore[no-untyped-def]
    x = to_x(u)
    lr = 0.75 * np.exp(0.12 * u[:, 5])  # load
    kr = 0.55 * np.exp(0.15 * u[:, 6])  # applied K over toughness
    lr_max = np.full(lr.size, 1.15)
    return x, lr, kr, lr_max


def true_failed(u: np.ndarray) -> np.ndarray:
    x, lr, kr, lr_max = problem(u)
    out = lr >= lr_max
    for j in np.flatnonzero(~out):
        c = synth_curve(x[j])
        if lr[j] > c.lr[-1]:
            out[j] = True  # beyond the support counts as failed in the headline (undecided -> failed)
            continue
        f = np.interp(lr[j], c.lr, c.j_over_je) ** -0.5  # the synthetic truth is smooth: interpolation is exact enough
        out[j] = kr[j] >= f
    return out


@pytest.fixture(scope="module")
def draws() -> EM.FunctionDraws:
    em = EM.train([synth_curve(x) for x in sobol_points(48, 1)], LOWER, UPPER, schedule, n_starts=3)
    return EM.function_draws(em, 24, np.random.default_rng(1), n_features=1000)


def test_batch_bounds_equal_single_bounds(draws: EM.FunctionDraws) -> None:
    rng = np.random.default_rng(2)
    u = rng.standard_normal((40, DIM))
    x, lr, _kr, _lm = problem(u)
    x[0] = [0.1, 0.5, 10.0, 2e-3, 0.0]  # outside the box
    lo, hi, st = R.batch_bounds(draws, 3, x, lr)
    for j in range(40):
        lo1, hi1, st1 = draws.bounds(3, x[j], float(lr[j]))
        assert st[j] == st1
        if st1 == "supported":
            assert lo[j] == pytest.approx(lo1, rel=1e-9) and hi[j] == pytest.approx(hi1, rel=1e-9)


def test_pf_bound_pair_brackets_the_synthetic_truth(draws: EM.FunctionDraws) -> None:
    rng = np.random.default_rng(3)
    # the truth by plain MC on the synthetic 'FE' (moderate Pf so the test is cheap)
    u_ref = rng.standard_normal((40000, DIM))
    pf_true = float(true_failed(u_ref).mean())
    pops = [R.population(problem, DIM, 4000, np.zeros(DIM), rng, alpha=1.0) for _ in range(3)]
    res = R.pf_crossed(draws, pops, experimental=True)
    se = np.sqrt(pf_true * (1 - pf_true) / 4000)
    assert res.pf_low.mean() - 4 * se <= pf_true <= res.pf_high.mean() + 4 * se
    assert res.draw_component >= 0.0 and res.population_component >= 0.0
    rows = res.pf_high.mean(axis=1)
    assert rows.min() <= res.interval[0] <= res.interval[1] <= rows.max() and rows.min() <= res.pf <= rows.max()
    assert any("surrogate-only" in lab for lab in res.labels) and any("experimental" in lab for lab in res.labels)
    assert set(res.undecided_mass) == set(R.UNDECIDED)


def test_collapse_is_classified_before_the_emulator(draws: EM.FunctionDraws) -> None:
    def collapsed(u: np.ndarray):  # type: ignore[no-untyped-def]
        x, lr, kr, _ = problem(u)
        return x, lr, kr, np.zeros(lr.size)  # Lr,max = 0: every realisation collapses

    pop = R.population(collapsed, DIM, 500, np.zeros(DIM), np.random.default_rng(4), alpha=1.0)
    res = R.pf_crossed(draws, [pop], experimental=True)
    assert np.allclose(res.pf_high, 1.0) and np.allclose(res.pf_low, 1.0)


def test_defensive_weights_are_unbiased() -> None:
    rng = np.random.default_rng(5)
    shift = np.r_[np.zeros(5), 2.0, 1.5]
    pop = R.population(problem, DIM, 200000, shift, rng, alpha=0.3)
    assert pop.weight.mean() == pytest.approx(1.0, abs=0.01)  # E_q[p/q] = 1
    _ = true_ratio  # imported for the synthetic family's definition


def test_round1_fixes_gate_nonfinite_mixture_share_monotone_mode(draws: EM.FunctionDraws) -> None:
    """Phase 6 code round 1 (red team A): P-07 the experimental gate; P-02 non-finite assessment values are
    `invalid` (in Pf_high, not Pf_low), after a valid collapse; P-03 the density uses the realised nominal share;
    P-01 the rigorous monotone mode brackets the convex mode."""
    rng = np.random.default_rng(6)
    pop = R.population(problem, DIM, 400, np.zeros(DIM), rng, alpha=1.0)
    with pytest.raises(PermissionError):
        R.pf_crossed(draws, [pop])
    kr = pop.kr.copy()
    kr[:10] = np.nan
    lr_max = pop.lr_max.copy()
    lr_max[10:12] = 0.0  # collapsed: failed even though Kr is finite
    bad = R.Population(pop.u, pop.weight, pop.x, pop.lr, kr, lr_max)
    assert bad.invalid_point[:10].all() and not bad.always_failed[:10].any() and bad.always_failed[10:12].all()
    res = R.pf_crossed(draws, [bad], experimental=True)
    assert res.undecided_mass["invalid"] >= 10 / 400 - 1e-12
    assert np.all(res.pf_high >= res.pf_low + 10 / 400 - 1e-12)
    # P-03: alpha = 0.33 with n = 10 realises 3/10 nominal; the weights are exact for the realised share
    p3 = R.population(problem, DIM, 200000, np.r_[np.zeros(5), 1.0, 1.0], rng, alpha=0.33)
    assert p3.weight.mean() == pytest.approx(1.0, abs=0.01)
    # P-01: the monotone pair is the wider bracket
    mono = R.pf_crossed(draws, [pop], experimental=True, convex=False)
    conv = R.pf_crossed(draws, [pop], experimental=True)
    assert np.all(mono.pf_low <= conv.pf_low + 1e-15) and np.all(mono.pf_high >= conv.pf_high - 1e-15)
    assert any("monotone (rigorous)" in lab for lab in mono.labels) and any("convex" in lab for lab in conv.labels)
