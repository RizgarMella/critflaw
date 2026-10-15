"""Reliability solvers on abstract limit states with exact answers (design note section 7).

These are the quick versions (few seeds). The full coverage runs over 1000 seeds are in
benchmarks/prob_interval_coverage.py (results written to benchmarks/out/).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy import stats

from critflaw.prob.solvers import (
    Evaluation,
    consistent_driving_variable,
    monte_carlo,
    replicate_interval,
    solve,
    subset_replicates,
    subset_simulation,
)

N_LEVEL = 2000


def linear(beta: float, dim: int):
    a = np.ones(dim) / np.sqrt(dim)

    def g(u: np.ndarray) -> Evaluation:
        y = beta - u @ a
        return Evaluation(y, y <= 0.0)

    return g


def two_branch(u: np.ndarray) -> Evaluation:
    y = np.minimum(3.0 - u[:, 0], 3.2 - u[:, 1])
    return Evaluation(y, y <= 0.0)


def two_sided(u: np.ndarray) -> Evaluation:
    y = 5.0 - np.abs(u[:, 0])
    return Evaluation(y, y <= 0.0)


def missed_region(u: np.ndarray) -> Evaluation:
    """The driving variable gives no hint of the left failure region (partner's construction)."""
    y = np.where(u[:, 0] <= -4.265, -1.0, 5.0 - u[:, 0])
    return Evaluation(y, y <= 0.0)


MISSED_EXACT = stats.norm.cdf(-4.265) + stats.norm.sf(5.0)


def plateau(u: np.ndarray) -> Evaluation:
    y = np.minimum(0.3, 5.0 - u[:, 0])
    return Evaluation(y, y <= 0.0)


def atom(u: np.ndarray) -> Evaluation:
    y = np.where(u[:, 0] > stats.norm.isf(0.01), -1.0, 1.0)
    return Evaluation(y, y <= 0.0)


# ---- Monte Carlo ---------------------------------------------------------------------------


def test_monte_carlo_interval_covers_the_exact_value() -> None:
    exact = stats.norm.cdf(-3.0)
    hits = 0
    for seed in range(40):
        r = monte_carlo(linear(3.0, 3), 3, 200_000, seed)
        hits += r.interval.contains(exact)
        assert r.failure_points.shape[1] == 3 and r.failures == pytest.approx(200_000 * exact, rel=0.5)
    assert hits >= 38
    # batching does not change the result
    a = monte_carlo(linear(3.0, 3), 3, 50_000, 7, batch=50_000)
    b = monte_carlo(linear(3.0, 3), 3, 50_000, 7, batch=1_000)
    assert a.failures == b.failures


def test_monte_carlo_with_the_origin_in_the_failure_region() -> None:
    def g(u: np.ndarray) -> Evaluation:
        y = u[:, 0] - 2.0
        return Evaluation(y, y <= 0.0)

    r = monte_carlo(g, 1, 100_000, 1)
    assert r.interval.contains(stats.norm.cdf(2.0)) and r.pf > 0.97


def test_discrete_mass_on_the_boundary_counts_as_failure() -> None:
    # y is exactly 0 with probability 0.25 (a point on the assessment line): failure
    def g(u: np.ndarray) -> Evaluation:
        y = np.where(u[:, 0] > stats.norm.isf(0.25), 0.0, 1.0)
        return Evaluation(y, y <= 0.0)

    r = monte_carlo(g, 1, 400_000, 3)
    assert r.pf == pytest.approx(0.25, abs=0.003)


def test_driving_variable_consistency() -> None:
    y = consistent_driving_variable(np.array([1e-15, -1e-15, 2.0, -3.0]), np.array([True, False, False, True]))
    assert y[0] < 0 < y[1] and y[2] == 2.0 and y[3] == -3.0
    with pytest.raises(RuntimeError):
        consistent_driving_variable(np.array([0.5]), np.array([True]))


# ---- subset simulation ---------------------------------------------------------------------


@pytest.mark.parametrize(("beta", "dim"), [(3.0902, 2), (4.7534, 2), (4.7534, 50)])
def test_subset_simulation_on_a_linear_limit_state(beta: float, dim: int) -> None:
    exact = stats.norm.cdf(-beta)
    estimates = []
    for seed in range(12):
        run = subset_simulation(linear(beta, dim), dim, seed, samples_per_level=N_LEVEL)
        assert run.status == "ok" and all(0.1 <= f <= 0.5 for f in run.level_fractions[:-1])
        # each conditional level costs N minus its seeds (about 0.9 N; duplicates from rejected moves vary it)
        assert run.evaluations == pytest.approx(N_LEVEL + (len(run.level_fractions) - 1) * N_LEVEL * 0.9, rel=0.01)
        assert run.failure_points.shape[0] >= 0.1 * N_LEVEL
        estimates.append(run.pf)
    # the product estimator, not the last-level fraction (which is of order 0.1 to 1)
    assert np.mean(estimates) == pytest.approx(exact, rel=0.25)
    assert min(run.level_fractions) >= 0.1 and run.level_fractions[-1] > 100 * exact


def test_subset_replicate_interval_covers_on_smooth_cases() -> None:
    cases = [(linear(3.0902, 2), 2, stats.norm.cdf(-3.0902)), (two_branch, 2, 1 - stats.norm.cdf(3.0) * stats.norm.cdf(3.2))]
    cases.append((two_sided, 1, 2 * stats.norm.sf(5.0)))
    for fn, dim, exact in cases:
        hits = 0
        for seed in range(8):
            res = subset_replicates(fn, dim, seed, replicates=8, samples_per_level=N_LEVEL)
            assert res.status == "ok" and res.interval is not None
            hits += res.interval.contains(exact)
            assert res.pf == pytest.approx(exact, rel=0.5)
        assert hits >= 6, (exact, hits)


def test_plateau_is_detected_and_no_estimate_is_given() -> None:
    for fn in (plateau, atom):
        for seed in range(5):
            run = subset_simulation(fn, 1, seed, samples_per_level=N_LEVEL)
            assert run.status == "plateau" and np.isnan(run.pf)
    res = solve(atom, 1, seed=0, n_explore=100_000, samples_per_level=N_LEVEL)
    assert res.status == "resolved" and res.guaranteed.contains(0.01)  # Monte Carlo handles the atom
    res = solve(plateau, 1, seed=0, n_explore=20_000, samples_per_level=N_LEVEL)
    assert res.status == "plateau" and res.conditional_estimate is None
    assert res.guaranteed.contains(stats.norm.sf(5.0)) and "no estimate" in res.summary()


def test_partial_atom_below_the_trigger_is_handled() -> None:
    # 30% of the mass sits at the top value; the rest is a continuous tail down to failure
    def g(u: np.ndarray) -> Evaluation:
        y = np.minimum(3.5 - u[:, 0], 3.5 - stats.norm.ppf(0.3))
        return Evaluation(y, y <= 0.0)

    exact = stats.norm.sf(3.5)
    estimates = [subset_simulation(g, 1, seed, samples_per_level=N_LEVEL).pf for seed in range(12)]
    assert np.mean(estimates) == pytest.approx(exact, rel=0.3)


def test_strongly_dependent_chains_are_reported_through_gamma() -> None:
    free = subset_simulation(linear(3.0902, 2), 2, 0, samples_per_level=N_LEVEL)
    stuck = subset_simulation(linear(3.0902, 2), 2, 0, samples_per_level=N_LEVEL, fixed_correlation=0.98)
    # chains have 10 states, which caps the factor; 0.98 correlation gives about 5, free adaptation about 1.5
    assert stuck.gammas[0] > 4.0 > 2.5 > free.gammas[0]
    assert stuck.cov_fully_correlated > stuck.cov_independent_levels > free.cov_independent_levels


def test_level_cap_gives_no_estimate_and_no_bound() -> None:
    run = subset_simulation(linear(12.0, 2), 2, 0, samples_per_level=N_LEVEL, max_levels=3)
    assert run.status == "level cap" and np.isnan(run.pf)


# ---- replicate interval --------------------------------------------------------------------


def test_replicate_interval_coverage_by_exact_enumeration() -> None:
    # each replicate is 0.5 p or 1.5 p with equal probability; the mean is exactly p
    covered = 0
    for combo in itertools.product((0.5, 1.5), repeat=8):
        iv = replicate_interval(np.array(combo))
        covered += iv is not None and iv.contains(1.0)
    assert covered / 256 == pytest.approx(0.9609, abs=0.001) and covered / 256 >= 0.95


def test_replicate_interval_is_withheld_for_inconsistent_replicates() -> None:
    assert replicate_interval(np.array([1e-6, 0.0, 2e-6, 1e-6])) is None
    assert replicate_interval(np.array([1e-6, 1e-6, 1e-6, 5e-5])) is None  # CoV above 1
    assert replicate_interval(np.array([1e-6, np.nan])) is None
    iv = replicate_interval(np.array([1.0e-6, 1.2e-6, 0.9e-6, 1.1e-6]))
    assert iv is not None and iv.lower < 1.05e-6 < iv.upper and "approximate" in iv.kind


# ---- combined result -----------------------------------------------------------------------


def test_status_resolved_when_monte_carlo_sees_enough_failures() -> None:
    res = solve(linear(3.0, 2), 2, seed=1, n_explore=200_000)
    assert res.status == "resolved" and res.subset is None
    assert res.guaranteed.contains(stats.norm.cdf(-3.0)) and res.pf == res.monte_carlo.pf


def test_status_conditional_shows_both_statements() -> None:
    exact = stats.norm.cdf(-4.7534)  # 1e-6
    res = solve(linear(4.7534, 2), 2, seed=2, n_explore=20_000, samples_per_level=N_LEVEL)
    assert res.status == "conditional" and res.pf == pytest.approx(exact, rel=0.6)
    assert res.guaranteed.lower <= res.pf <= res.guaranteed.upper
    text = res.summary()
    assert "cannot be verified" in text and "cannot be excluded" in text and "Guaranteed" in text


def test_missed_region_is_never_presented_with_false_confidence() -> None:
    # Subset simulation finds only the right-hand region (2.9e-7); the truth is 1.03e-5.
    sub = subset_replicates(missed_region, 1, 0, samples_per_level=N_LEVEL)
    assert sub.interval is not None and not sub.interval.contains(MISSED_EXACT)  # the hazard, recorded
    statuses = []
    for seed in range(12):
        res = solve(missed_region, 1, seed=seed, n_explore=1_000_000, samples_per_level=N_LEVEL)
        statuses.append(res.status)
        assert res.guaranteed.contains(MISSED_EXACT) or seed == -1
        k = res.monte_carlo.failures
        if res.status == "conditional":
            assert k < 10 and res.guaranteed.lower <= res.pf <= res.guaranteed.upper
        elif res.status == "contradicted":
            assert not res.guaranteed.contains(res.subset.pf) and res.pf == res.monte_carlo.pf  # type: ignore[union-attr]
            assert "rejected" in res.summary()
        else:
            assert res.status == "resolved" and k >= 10
    assert "resolved" in statuses and "contradicted" in statuses
    # with a small exploration budget the result is conditional and says what cannot be excluded
    small = solve(missed_region, 1, seed=3, n_explore=10_000, samples_per_level=N_LEVEL)
    assert small.status in ("conditional", "contradicted") and small.guaranteed.upper > MISSED_EXACT


@pytest.mark.parametrize("k", [0, 1, 2])
def test_few_monte_carlo_failures_with_estimate_inside_the_interval_are_conditional(k: int) -> None:
    from critflaw.prob.intervals import clopper_pearson

    interval = clopper_pearson(k, 1_000_000)
    assert interval.contains(stats.norm.sf(5.0))  # the missed-region estimate lies inside: conditional is correct
