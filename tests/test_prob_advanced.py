"""Phase 2, second part: partial factors, importance sampling, diagnostics, family sensitivity,
inspection-conditioned flaw sizes (design note D16, D18-D22)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate, special, stats

import critflaw as cf
from critflaw import prob
from critflaw.prob.diagnostics import association_with_failure, cluster_failure_points, most_likely_failure_points
from critflaw.prob.families import NoMatch, family_of, moment_matched
from critflaw.prob.importance import importance_sampling
from critflaw.prob.inspection import (
    ConstantPoD,
    LogLogisticPoD,
    StepPoD,
    detected,
    per_component_upper_bound,
    undetected,
)
from critflaw.prob.partial_factors import NotTabulated, implied_partial_factor, published_partial_factors
from critflaw.prob.solvers import Evaluation, monte_carlo

Z99 = 2.5758293035489004  # two-sided 99% point of the standard normal, typed in

# ---- published partial factors -------------------------------------------------------------


def test_published_partial_factors_lookup() -> None:
    f = published_partial_factors(7e-5, cov_stress=0.2, cov_flaw_size=0.1, cov_toughness=0.2)
    assert (f.beta, f.stress, f.flaw_size, f.toughness, f.yield_strength) == (3.8, 1.35, 1.5, 2.6, 1.1)
    g = published_partial_factors(1e-7, 0.3, 0.5, 0.1)
    assert (g.stress, g.flaw_size, g.toughness, g.yield_strength) == (1.8, 2.5, 2.0, 1.5)
    h = published_partial_factors(2.3e-1, 0.1, 0.3, 0.3)
    assert (h.stress, h.flaw_size, h.toughness, h.yield_strength) == (1.05, 1.08, 1.0, 1.0)
    assert published_partial_factors(1e-3, 0.1, 0.1, 0.3).toughness == 2.85
    with pytest.raises(NotTabulated, match="not permitted"):
        published_partial_factors(1e-5, 0.2, 0.1, 0.3)
    with pytest.raises(NotTabulated, match="no interpolation"):
        published_partial_factors(1e-5, 0.15, 0.1, 0.2)
    with pytest.raises(NotTabulated, match="not tabulated"):
        published_partial_factors(1e-4, 0.1, 0.1, 0.2)
    with pytest.raises(NotTabulated):
        published_partial_factors(1e-5, 0.1, 0.1, 0.2, cov_yield=0.07)
    # the reliability indices quoted with the targets are consistent with them
    for target, beta in prob.partial_factors.TARGETS.items():
        assert stats.norm.sf(beta) == pytest.approx(target, rel=0.05)
    for near_miss in (0.12, 0.29, 0.1 + 1e-6):
        with pytest.raises(NotTabulated):
            published_partial_factors(1e-5, near_miss, 0.1, 0.2)
    for target in (8e-5, 1.1e-3, 1e-5 * (1 + 1e-6)):
        with pytest.raises(NotTabulated):
            published_partial_factors(target, 0.1, 0.1, 0.2)
    for design, reference in ((50.0, -10.0), (1.0, 0.0), (-1.0, 2.0)):
        with pytest.raises(ValueError, match="positive"):
            implied_partial_factor(design, reference, "resistance")
    assert implied_partial_factor(180.0, 150.0, "load") == pytest.approx(1.2)
    assert implied_partial_factor(60.0, 90.0, "resistance") == pytest.approx(1.5)
    with pytest.raises(ValueError):
        implied_partial_factor(1.0, 1.0, "other")


# The whole published table, typed from the independent transcription of the source (not from
# the code). Columns: targets 2.3e-1, 1e-3, 7e-5, 1e-5, 1e-7. None is "not permitted".
PUBLISHED = {
    "stress": {0.1: (1.05, 1.2, 1.25, 1.3, 1.4), 0.2: (1.1, 1.25, 1.35, 1.4, 1.55), 0.3: (1.12, 1.4, 1.5, 1.6, 1.8)},
    "flaw_size": {
        0.1: (1.0, 1.4, 1.5, 1.7, 2.1),
        0.2: (1.05, 1.45, 1.55, 1.8, 2.2),
        0.3: (1.08, 1.5, 1.65, 1.9, 2.3),
        0.5: (1.15, 1.7, 1.85, 2.1, 2.5),
    },
    "toughness": {0.1: (1.0, 1.3, 1.5, 1.7, 2.0), 0.2: (1.0, 1.8, 2.6, 3.2, 5.5), 0.3: (1.0, 2.85, None, None, None)},
    "yield_strength": {0.1: (1.0, 1.05, 1.1, 1.2, 1.5)},
}
PUBLISHED_TARGETS = ((2.3e-1, 0.739), (1e-3, 3.09), (7e-5, 3.8), (1e-5, 4.27), (1e-7, 5.2))


def test_every_cell_of_the_published_table() -> None:
    cells = 0
    for column, (target, beta) in enumerate(PUBLISHED_TARGETS):
        for cs, stress in PUBLISHED["stress"].items():
            for ca, flaw in PUBLISHED["flaw_size"].items():
                for ck, toughness in PUBLISHED["toughness"].items():
                    if toughness[column] is None:
                        with pytest.raises(NotTabulated, match="not permitted"):
                            published_partial_factors(target, cs, ca, ck)
                        continue
                    f = published_partial_factors(target, cs, ca, ck)
                    got = (f.target_pf, f.beta, f.stress, f.flaw_size, f.toughness, f.yield_strength)
                    expected = (target, beta, stress[column], flaw[column], toughness[column])
                    assert got == (*expected, PUBLISHED["yield_strength"][0.1][column])
                    cells += 1
    assert cells == 5 * 3 * 4 * 3 - 3 * 3 * 4  # three "not permitted" cells, each met with 12 combinations


# ---- importance sampling -------------------------------------------------------------------


def linear(beta: float):
    def g(u: np.ndarray) -> Evaluation:
        y = beta - u[:, 0]
        return Evaluation(y, y <= 0.0)

    return g


def two_sided(u: np.ndarray) -> Evaluation:
    y = 5.0 - np.abs(u[:, 0])
    return Evaluation(y, y <= 0.0)


def test_importance_sampling_with_a_covering_proposal() -> None:
    exact = stats.norm.sf(4.0)
    hits = 0
    for seed in range(20):
        r = importance_sampling(linear(4.0), 2, np.array([[4.0, 0.0]]), 20_000, seed)
        assert r.conditional_interval is not None and r.effective_failures > 100 and r.cov < 0.05
        hits += r.conditional_interval.contains(exact)
        assert r.guaranteed_interval.contains(exact) and r.pf == pytest.approx(exact, rel=0.1)
        assert r.component_shares == (pytest.approx(1.0),) and 0.0 <= r.defensive_share < 0.2
    assert hits >= 18
    both = importance_sampling(two_sided, 1, np.array([[5.0], [-5.0]]), 40_000, 1)
    assert both.conditional_interval is not None and both.conditional_interval.contains(2 * stats.norm.sf(5.0))
    assert both.component_shares[0] == pytest.approx(0.5, abs=0.05)


def test_importance_sampling_with_a_missed_region_is_only_safe_through_the_guaranteed_interval() -> None:
    exact = 2 * stats.norm.sf(5.0)
    misses = 0
    for seed in range(20):
        r = importance_sampling(two_sided, 1, np.array([[5.0]]), 20_000, seed)  # the left region is not covered
        assert r.guaranteed_interval.contains(exact)
        assert r.conditional_interval is not None and "conditional" in r.conditional_interval.kind
        misses += not r.conditional_interval.contains(exact)
        assert r.pf == pytest.approx(0.5 * exact, rel=0.1)  # half the truth, with a small reported CoV
    assert misses >= 18  # the hazard: the asymptotic interval alone would be wrong almost always
    # the guaranteed half-width is sqrt(2 V ln(3/delta) / N) + 3 ln(3/delta) / (w0 N)
    width = r.guaranteed_interval.upper - r.pf
    assert width > 3 * np.log(300.0) / (0.2 * 20_000)


def test_importance_sampling_against_a_recomputation_from_the_same_draws() -> None:
    centres, n, w0 = np.array([[3.0, 0.0], [0.0, 3.5]]), 40_000, 0.2

    def g(u: np.ndarray) -> Evaluation:
        y = np.minimum(3.0 - u[:, 0], 3.5 - u[:, 1])
        return Evaluation(y, y <= 0.0)

    r = importance_sampling(g, 2, centres, n, seed=11)
    # the same draws, with the densities written out in full
    rng = np.random.default_rng(11)
    component = rng.choice(3, size=n, p=[w0, 0.4, 0.4])
    u = rng.standard_normal((n, 2)) + np.vstack([np.zeros(2), centres])[component]
    phi = stats.multivariate_normal(np.zeros(2)).pdf(u)
    q = w0 * phi + sum(0.4 * stats.multivariate_normal(c).pdf(u) for c in centres)
    terms = np.where(g(u).fail, phi / q, 0.0)
    pf = float(np.mean(terms))
    sd = float(np.sqrt(np.sum((terms - pf) ** 2) / (n - 1)))  # sample standard deviation (n - 1)
    assert r.pf == pytest.approx(pf, rel=1e-12)
    assert r.cov == pytest.approx(sd / np.sqrt(n) / pf, rel=1e-10)
    assert r.effective_failures == pytest.approx(np.sum(terms) ** 2 / np.sum(terms**2), rel=1e-10)
    # asymptotic interval: pf exp(-+ z sqrt(ln(1 + cov^2))) with the two-sided 99% z
    sigma_ln = np.sqrt(np.log(1.0 + r.cov**2))
    assert r.conditional_interval is not None
    assert r.conditional_interval.lower == pytest.approx(pf * np.exp(-Z99 * sigma_ln), rel=1e-9)
    assert r.conditional_interval.upper == pytest.approx(pf * np.exp(Z99 * sigma_ln), rel=1e-9)
    # guaranteed interval: half-width sqrt(2 V ln 300 / N) + 3 (1 / w0) ln 300 / N
    half = np.sqrt(2.0 * sd**2 * np.log(300.0) / n) + 15.0 * np.log(300.0) / n
    assert r.guaranteed_interval.upper == pytest.approx(pf + half, rel=1e-10)
    assert r.guaranteed_interval.lower == pytest.approx(max(pf - half, 0.0), abs=1e-15)
    # shares by nearest centre, with centres of unequal weight
    p1, p2 = stats.norm.sf(3.0), stats.norm.sf(3.5)
    assert r.component_shares[0] == pytest.approx(p1 / (p1 + p2 - p1 * p2), abs=0.02)
    assert sum(r.component_shares) == pytest.approx(1.0)
    nearest_first = np.sum((u - centres[0]) ** 2, axis=1) <= np.sum((u - centres[1]) ** 2, axis=1)
    assert r.component_shares[0] == pytest.approx(np.sum(terms[nearest_first]) / np.sum(terms), rel=1e-10)
    assert r.defensive_share == pytest.approx(np.sum(terms[component == 0]) / np.sum(terms), rel=1e-10)


def test_importance_sampling_withholds_the_asymptotic_interval_below_100_effective_failures() -> None:
    for seed in range(6):
        r = importance_sampling(linear(4.0), 2, np.array([[4.0, 0.0]]), 600, seed)
        assert r.pf > 0.0 and 50.0 < r.effective_failures < 100.0
        assert r.conditional_interval is None and r.guaranteed_interval.contains(stats.norm.sf(4.0))


def test_importance_sampling_arguments() -> None:
    with pytest.raises(ValueError, match="shape"):
        importance_sampling(linear(3.0), 2, np.array([[3.0]]), 100, 0)
    with pytest.raises(ValueError, match="shape"):
        importance_sampling(linear(3.0), 2, np.empty((0, 2)), 100, 0)
    for samples in (0, 1):
        with pytest.raises(ValueError, match="two samples"):
            importance_sampling(linear(3.0), 1, np.array([[3.0]]), samples, 0)
    assert importance_sampling(linear(0.0), 1, np.array([[0.0]]), 2, 0).guaranteed_interval.upper <= 1.0
    with pytest.raises(ValueError, match="finite"):
        importance_sampling(linear(3.0), 1, np.array([[np.nan]]), 100, 0)
    for confidence in (0.0, 1.0):
        with pytest.raises(ValueError, match="confidence"):
            importance_sampling(linear(3.0), 1, np.array([[3.0]]), 100, 0, confidence=confidence)
    # a probability near one: the asymptotic upper bound would exceed 1 and is capped there
    likely = importance_sampling(linear(-2.0), 1, np.array([[0.0]]), 150, 0)
    assert likely.conditional_interval is not None and likely.conditional_interval.upper == 1.0
    assert likely.pf * np.exp(Z99 * np.sqrt(np.log(1.0 + likely.cov**2))) > 1.0
    with pytest.raises(ValueError):
        importance_sampling(linear(3.0), 1, np.array([[3.0]]), 100, 0, defensive_weight=1.0)
    none = importance_sampling(linear(40.0), 1, np.array([[3.0]]), 1000, 0)
    assert none.pf == 0.0 and none.conditional_interval is None and none.guaranteed_interval.lower == 0.0


# ---- diagnostics ---------------------------------------------------------------------------


def two_branch(u: np.ndarray) -> Evaluation:
    y = np.minimum(3.0 - u[:, 0], 3.2 - u[:, 1])
    return Evaluation(y, y <= 0.0)


def test_most_likely_failure_points_find_both_branches() -> None:
    mc = monte_carlo(two_branch, 2, 3_000_000, seed=2)
    labels = cluster_failure_points(mc.failure_points, two_branch)
    assert len(np.unique(labels)) == 2
    by_first = mc.failure_points[:, 0] >= 3.0  # the corner (both) is a share of 1e-3 of the samples
    assert np.mean((labels == 0) == by_first) > 0.995
    points = most_likely_failure_points(two_branch, mc.failure_points)
    assert len(points) == 2 and all(p.dimension == 2 and p.label == "approximate" for p in points)
    p1, p2 = stats.norm.sf(3.0), stats.norm.sf(3.2)
    assert points[0].share == pytest.approx(p1 / (p1 + p2), abs=0.05)
    assert points[0].distance == pytest.approx(3.0, abs=0.08) and points[1].distance == pytest.approx(3.2, abs=0.08)
    for p in points:
        # on the limit state to the accuracy of the bisection, not of the scan
        assert bool(two_branch(p.u[None, :]).fail[0]) and not bool(two_branch((1 - 1e-9) * p.u[None, :]).fail[0])
    assert most_likely_failure_points(two_branch, np.empty((0, 2))) == []
    # a single mode stays one cluster
    one = monte_carlo(lambda u: Evaluation(3.0 - u[:, 0], u[:, 0] >= 3.0), 2, 1_000_000, seed=3)
    assert len(most_likely_failure_points(lambda u: Evaluation(3.0 - u[:, 0], u[:, 0] >= 3.0), one.failure_points)) == 1


def beyond(rng: np.random.Generator, beta: float, n: int) -> np.ndarray:
    """Exact samples of a standard normal variable conditional on exceeding ``beta``."""
    return np.asarray(-special.ndtri(rng.uniform(size=n) * stats.norm.sf(beta)))


@pytest.mark.parametrize("dim", [2, 6, 10, 20])
@pytest.mark.parametrize("beta", [1.0, 2.0, 3.0, 4.0])
def test_one_failure_region_is_one_cluster_in_any_number_of_variables(dim: int, beta: float) -> None:
    rng = np.random.default_rng(100 * dim + int(beta))
    points = rng.standard_normal((4000, dim))
    points[:, 0] = beyond(rng, beta, 4000)
    # direction cones alone split this region (the reason the limit state is used)
    if dim >= 6:
        assert len(np.unique(cluster_failure_points(points))) > 1
    assert len(np.unique(cluster_failure_points(points, linear(beta)))) == 1
    found = most_likely_failure_points(linear(beta), points)
    assert len(found) == 1 and found[0].share == 1.0 and found[0].dimension == dim
    assert found[0].distance == pytest.approx(beta, abs=0.01 * beta)  # the design point, not a sample
    assert found[0].u[0] / found[0].distance > 0.995
    assert ("indicative" in found[0].label) == (dim > 10)


@pytest.mark.parametrize("dim", [2, 6, 20])
def test_two_failure_regions_in_many_variables(dim: int) -> None:
    rng = np.random.default_rng(dim)
    p1, p2 = stats.norm.sf(3.0), stats.norm.sf(3.2)
    first = rng.uniform(size=6000) < p1 / (p1 + p2)  # the overlap of the two (1e-3 of the samples) is ignored
    points = rng.standard_normal((6000, dim))
    points[first, 0] = beyond(rng, 3.0, int(first.sum()))
    points[~first, 1] = beyond(rng, 3.2, int((~first).sum()))
    found = most_likely_failure_points(two_branch, points)
    assert len(found) == 2
    assert found[0].share == pytest.approx(np.mean(points[:, 0] >= 3.0), abs=0.01)
    # the governing region comes first, with its own point
    assert found[0].distance == pytest.approx(3.0, abs=0.03) and found[0].u[0] > 2.97
    assert found[1].distance == pytest.approx(3.2, abs=0.03) and found[1].u[1] > 3.17


def two_half_spaces(rng: np.random.Generator, dim: int, angle: float, betas: tuple[float, float], n: int):
    """Failure samples of two half-spaces whose normals are ``angle`` degrees apart (a mixture in
    proportion to their probabilities; the overlap is ignored), the limit state, and the share
    of the samples that violate the first one more."""
    normal = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
    p1, p2 = stats.norm.sf(betas[0]), stats.norm.sf(betas[1])
    first = rng.uniform(size=n) < p1 / (p1 + p2)
    points = rng.standard_normal((n, dim))
    points[first, 0] = beyond(rng, betas[0], int(first.sum()))
    along, across = beyond(rng, betas[1], int((~first).sum())), rng.standard_normal(int((~first).sum()))
    points[~first, 0] = along * normal[0] - across * normal[1]
    points[~first, 1] = along * normal[1] + across * normal[0]

    def g(u: np.ndarray) -> Evaluation:
        y = np.minimum(betas[0] - u[:, 0], betas[1] - u[:, :2] @ normal)
        return Evaluation(y, y <= 0.0)

    return points, g, float(np.mean((betas[0] - points[:, 0]) <= (betas[1] - points[:, :2] @ normal)))


@pytest.mark.parametrize("angle", [90.0, 180.0])
@pytest.mark.parametrize("dim", [15, 20, 30])
def test_a_region_with_a_twentieth_of_the_samples_is_found_in_many_variables(dim: int, angle: float) -> None:
    points, g, share = two_half_spaces(np.random.default_rng(dim + int(angle)), dim, angle, (2.0, 3.0), 20_000)
    found = most_likely_failure_points(g, points)
    assert len(found) == 2 and 0.04 < 1.0 - share < 0.07
    assert found[1].share == pytest.approx(1.0 - share, abs=0.02)
    assert found[0].distance == pytest.approx(2.0, rel=0.01) and found[1].distance == pytest.approx(3.0, rel=0.015)


@pytest.mark.parametrize("dim", [2, 10, 20])
@pytest.mark.parametrize("betas", [(3.0, 3.0), (3.0, 3.2)])
def test_regions_sixty_degrees_apart_that_meet_further_out_stay_two(dim: int, betas: tuple[float, float]) -> None:
    # groups near the corner are linked to both regions: the order (from the origin outward) decides
    points, g, share = two_half_spaces(np.random.default_rng(dim), dim, 60.0, betas, 6000)
    found = sorted(most_likely_failure_points(g, points), key=lambda p: -p.u[0])
    assert len(found) == 2
    assert found[0].distance == pytest.approx(betas[0], rel=0.01) and found[0].u[0] > 0.99 * betas[0]
    assert found[1].distance == pytest.approx(betas[1], rel=0.01)
    assert found[1].u[:2] @ np.array([0.5, np.sqrt(0.75)]) > 0.99 * betas[1]
    assert found[0].share == pytest.approx(share, abs=0.05)
    assert found[0].share + found[1].share == pytest.approx(1.0)


def test_a_safe_gap_between_two_radii_separates_regions() -> None:
    # a patch at radius 3 and a complete ring at radius 5: the arc at radius 5 fails everywhere,
    # but the way out from the patch to that radius crosses safe ground
    def rings(u: np.ndarray) -> Evaluation:
        r = np.linalg.norm(u, axis=1)
        fail = ((r >= 2.9) & (r <= 3.1) & (u[:, 0] > 0) & (np.abs(u[:, 1]) <= 0.1)) | ((r >= 5.0) & (r <= 5.1))
        return Evaluation(np.where(fail, -1.0, 1.0), fail)

    found = most_likely_failure_points(rings, np.array([[3.0, 0.0], [0.0, 5.05]] * 4))
    assert [(p.share, round(p.distance, 6)) for p in found] == [(0.5, 2.9), (0.5, 5.0)]

    # a band 1.3% of the line wide in front of the main region is not stepped over by the scan
    def banded(u: np.ndarray) -> Evaluation:
        fail = ((u[:, 0] >= 2.94) & (u[:, 0] <= 3.0)) | (u[:, 0] >= 4.0)
        return Evaluation(np.where(fail, -1.0, 1.0), fail)

    (point,) = most_likely_failure_points(banded, np.array([[4.5, 0.0], [4.6, 0.1], [4.2, -0.1]]))
    assert point.distance == pytest.approx(2.94, abs=0.002)


def test_one_failure_sample_and_identical_samples() -> None:
    # reachable through the engine: a small exploratory run can end with a single failure sample
    for points in (np.array([[3.5, 0.2]]), np.array([[3.5, 0.2]] * 3), np.array([[3.5, 0.2], [3.6, -0.1]])):
        assert list(cluster_failure_points(points, linear(3.0))) == [0] * points.shape[0]
        (point,) = most_likely_failure_points(linear(3.0), points)
        assert point.share == 1.0 and point.u[0] == pytest.approx(3.0, abs=1e-9)
        assert point.distance == pytest.approx(3.0 * np.hypot(3.5, 0.2) / 3.5, rel=1e-3) or points.shape[0] == 2


def test_a_sphere_is_one_region() -> None:
    def sphere(u: np.ndarray) -> Evaluation:
        y = 3.0 - np.linalg.norm(u, axis=1)
        return Evaluation(y, y <= 0.0)

    mc = monte_carlo(sphere, 4, 400_000, seed=7)
    (point,) = most_likely_failure_points(sphere, mc.failure_points)
    assert point.share == 1.0 and point.distance == pytest.approx(3.0, abs=1e-6)


def test_small_regions_opposite_regions_and_a_failing_origin() -> None:
    # a distinct region holding 1% of the failure samples is kept
    points = np.vstack([np.tile([3.0, 0.0], (99, 1)), [[0.0, 5.0]]])

    def cross(u: np.ndarray) -> Evaluation:
        fail = (u[:, 0] >= 3.0) | (u[:, 1] >= 5.0)
        return Evaluation(np.where(fail, -1.0, 1.0), fail)

    assert list(np.bincount(cluster_failure_points(points))) == [99, 1]
    found = most_likely_failure_points(cross, points)
    assert [(p.share, round(p.distance, 6)) for p in found] == [(0.99, 3.0), (0.01, 5.0)]
    # opposite regions (no arc between them), in one and in several variables
    rng = np.random.default_rng(5)
    for dim in (1, 8):
        both = rng.standard_normal((4000, dim))
        both[:, 0] = beyond(rng, 5.0, 4000) * np.where(rng.uniform(size=4000) < 0.5, 1.0, -1.0)
        found = most_likely_failure_points(two_sided, both)
        assert len(found) == 2 and found[0].share == pytest.approx(0.5, abs=0.03)
        assert all(p.distance == pytest.approx(5.0, abs=0.02) for p in found)

    # the origin itself fails: it is the most likely failure point, at distance zero
    def band(u: np.ndarray) -> Evaluation:
        fail = (np.abs(u[:, 0]) < 0.001) | (u[:, 0] >= 1.0)
        return Evaluation(np.where(fail, -1.0, 1.0), fail)

    (only,) = most_likely_failure_points(band, np.array([[1.1]]))
    assert only.distance == 0.0 and only.share == 1.0 and "origin" in only.label
    # a sample at the origin has no direction: a cluster of its own, without disturbing the others
    labels = cluster_failure_points(np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 2.0], [0.0, 2.0]]))
    assert len(np.unique(labels)) == 4 and labels[3] == labels[4] and labels[0] not in labels[1:]
    assert cluster_failure_points(np.empty((0, 3))).size == 0


def test_more_regions_than_the_maximum_is_stated() -> None:
    def three(u: np.ndarray) -> Evaluation:
        angles = np.radians([0.0, 120.0, 240.0])
        y = np.min(3.0 - (u[:, :1] * np.cos(angles) + u[:, 1:2] * np.sin(angles)), axis=1)
        return Evaluation(y, y <= 0.0)

    mc = monte_carlo(three, 2, 2_000_000, seed=6)
    assert [p.label for p in most_likely_failure_points(three, mc.failure_points)] == ["approximate"] * 3
    limited = most_likely_failure_points(three, mc.failure_points, max_clusters=2)
    assert len(limited) == 2 and all("more than 2 failure regions" in p.label for p in limited)
    assert sum(p.share for p in limited) == pytest.approx(1.0)
    assert limited[0].share == pytest.approx(2 / 3, abs=0.03)  # the dropped region went to one of the kept two

    # not symmetric: regions at 0 and 100 degrees (index 3.0) and a smaller one at 150 degrees (index 3.4);
    # limited to two, the small one joins the region at 100 degrees, the nearer in direction
    def lopsided(u: np.ndarray) -> Evaluation:
        angles = np.radians([0.0, 100.0, 150.0])
        y = np.min(np.array([3.0, 3.0, 3.4]) - (u[:, :1] * np.cos(angles) + u[:, 1:2] * np.sin(angles)), axis=1)
        return Evaluation(y, y <= 0.0)

    mc = monte_carlo(lopsided, 2, 3_000_000, seed=8)
    assert len(most_likely_failure_points(lopsided, mc.failure_points)) == 3
    first, second = most_likely_failure_points(lopsided, mc.failure_points, max_clusters=2)
    small = stats.norm.sf(3.4) / (2 * stats.norm.sf(3.0) + stats.norm.sf(3.4))
    assert first.share == pytest.approx(0.5 * (1 - small) + small, abs=0.03) and first.u[1] > 2.5
    assert second.share == pytest.approx(0.5 * (1 - small), abs=0.03) and second.u[0] > 2.9


def test_association_formula_and_its_guards() -> None:
    rng = np.random.default_rng(8)
    vector = prob.RandomVector([prob.Normal(0, 1, name="x"), prob.Normal(0, 1, name="y"), prob.Empirical([1.0, 2.0])])
    u = rng.standard_normal((40, 3))
    u[:, 0] = beyond(rng, 3.0, 40)  # conditional variance far below 1
    out = {r.name: r for r in association_with_failure(vector, u, ("x", "y", "atoms"))}
    for name, column in (("x", 0), ("y", 1)):
        m = float(np.mean(u[:, column]))
        v = float(np.sum((u[:, column] - m) ** 2) / 39)
        assert out[name].mean_shift == pytest.approx(m, rel=1e-12) and out[name].spread == pytest.approx(np.sqrt(v))
        assert out[name].divergence == pytest.approx(0.5 * (m * m + v - 1.0 - np.log(v)), rel=1e-10)
    assert out["x"].spread < 0.5 and out["x"].divergence > 4.0
    # fewer than the minimum number of failure samples: nothing is reported
    names = ("x", "y", "atoms")
    assert association_with_failure(vector, u[:29], names) == []
    assert len(association_with_failure(vector, u[:30], names)) == 3
    assert len(association_with_failure(vector, u[:5], names, minimum_samples=5)) == 3
    # a discrete variable with every failure sample on one atom: listed last, no number
    one_atom = u.copy()
    one_atom[:, 2] = -1.0
    ranked = association_with_failure(vector, one_atom, names)
    assert [r.name for r in ranked][-1] == "atoms" and np.isnan(ranked[-1].divergence) and ranked[-1].spread == 0.0
    assert all(np.isfinite(r.divergence) for r in ranked[:-1])


def test_association_ranks_a_two_sided_cause_above_an_inert_variable() -> None:
    # design note, benchmark 7: at least 990 of 1000 runs
    rng = np.random.default_rng(9)
    vector = prob.RandomVector([prob.Normal(0, 1), prob.Normal(0, 1)])
    right = 0
    for _ in range(1000):
        u = rng.standard_normal((200, 2))
        u[:, 0] = beyond(rng, 5.0, 200) * np.where(rng.uniform(size=200) < 0.5, 1.0, -1.0)
        right += association_with_failure(vector, u, ("cause", "inert"))[0].name == "cause"
    assert right >= 990


def test_association_sees_two_sided_causes_and_is_not_causal() -> None:
    a, b, c = prob.Normal(0, 1, name="two-sided cause"), prob.Normal(0, 1, name="inert"), prob.Normal(0, 1, name="echo")
    vector = prob.RandomVector([a, b, c]).correlate(a, c, 0.9)

    def g(u: np.ndarray) -> Evaluation:
        x = vector.to_physical(u)
        y = 3.5 - np.abs(x[:, 0])
        return Evaluation(y, y <= 0.0)

    mc = monte_carlo(g, 3, 2_000_000, seed=4)
    out = {r.name: r for r in association_with_failure(vector, mc.failure_points, ("two-sided cause", "inert", "echo"))}
    assert abs(out["two-sided cause"].mean_shift) < 0.3 and out["two-sided cause"].spread > 3.0
    assert out["two-sided cause"].divergence > 20 * out["inert"].divergence
    # the variable correlated with the cause also ranks high although it causes nothing
    assert out["echo"].divergence > 5 * out["inert"].divergence
    det = prob.RandomVector([a, prob.Deterministic(2.0)])
    pair = np.array([[3.6, 0.0], [3.7, 0.1]])
    assert [r.name for r in association_with_failure(det, pair, ("x", "fixed"), minimum_samples=2)] == ["x"]


# ---- distribution families -----------------------------------------------------------------


def test_moment_matching_after_truncation() -> None:
    declared = prob.LogNormal.from_mean_sd(1.0, 0.5, name="k")
    assert family_of(declared) == "lognormal" and family_of(prob.Gumbel(0, 1)) is None
    for family in ("normal", "weibull"):
        alt = moment_matched(declared, family)
        assert alt.mean() == pytest.approx(1.0, rel=1e-7) and alt.std() == pytest.approx(0.5, rel=1e-7)
        assert alt.support()[0] == 0.0 and alt.name == "k" and family_of(alt) == family
    normal_alt = moment_matched(declared, "normal")
    assert isinstance(normal_alt, prob.Truncated) and normal_alt.dist.mean() != pytest.approx(1.0, abs=1e-3)
    # a normal truncated at zero cannot reach a coefficient of variation above 1
    with pytest.raises(NoMatch):
        moment_matched(prob.LogNormal.from_mean_sd(1.0, 1.2), "normal")
    # an unbounded normal has no lognormal or Weibull counterpart on the same support
    for family in ("lognormal", "weibull"):
        with pytest.raises(NoMatch, match="below zero"):
            moment_matched(prob.Normal(10.0, 1.0), family)
    # a truncated declared distribution keeps its bounds
    bounded = prob.Truncated(prob.Normal(100.0, 20.0), lo=50.0, hi=160.0)
    alt = moment_matched(bounded, "lognormal")
    assert alt.support() == (50.0, 160.0)
    assert alt.mean() == pytest.approx(bounded.mean(), rel=1e-7) and alt.std() == pytest.approx(bounded.std(), rel=1e-7)


def moments_from_the_density(dist: prob.Distribution, lo: float, hi: float) -> tuple[float, float]:
    """Mean and standard deviation by integrating the density (independent of mean() and std())."""
    points = dist.ppf(np.array([1e-12, 1e-6, 0.01, 0.25, 0.5, 0.75, 0.99, 1 - 1e-6, 1 - 1e-12]))
    edges = np.unique(np.concatenate([[lo], points, [hi]]))
    edges = edges[np.isfinite(edges)]

    def moment(k: int) -> float:
        return float(sum(
            integrate.quad(lambda x: x**k * float(dist.pdf(x)), a, b, epsabs=0, epsrel=1e-11, limit=200)[0]
            for a, b in zip(edges[:-1], edges[1:], strict=True)
        ))

    total, mean = moment(0), moment(1)
    assert total == pytest.approx(1.0, abs=1e-9)
    return mean, float(np.sqrt(moment(2) - mean**2))


def test_moment_matching_checked_by_integrating_the_density() -> None:
    cases = [
        # declared distribution, family, expected type, expected support
        (prob.Truncated(prob.Normal(100.0, 30.0), lo=0.0, name="v"), "lognormal", prob.LogNormal, (0.0, np.inf)),
        (prob.Truncated(prob.Normal(100.0, 30.0), lo=0.0, name="v"), "weibull", prob.Weibull, (0.0, np.inf)),
        (prob.Gumbel.from_mean_sd(10.0, 2.0, name="v"), "normal", prob.Normal, (-np.inf, np.inf)),
        (prob.LogNormal.from_mean_sd(1.0, 0.5, name="v"), "normal", prob.Truncated, (0.0, np.inf)),
        (prob.Truncated(prob.LogNormal.from_mean_sd(100.0, 30.0), hi=150.0, name="v"), "weibull", prob.Truncated, (0.0, 150.0)),
        (prob.Truncated(prob.Normal(100.0, 20.0), lo=50.0, hi=160.0, name="v"), "lognormal", prob.Truncated, (50.0, 160.0)),
    ]
    for declared, family, kind, support in cases:
        alt = moment_matched(declared, family)
        assert type(alt) is kind and alt.support() == support and alt.name == "v" and family_of(alt) == family
        lo = max(support[0], float(alt.ppf(1e-15)))
        hi = min(support[1], float(alt.isf(1e-15)))
        mean, sd = moments_from_the_density(alt, lo, hi)
        want_mean, want_sd = moments_from_the_density(declared, *(float(declared.ppf(1e-15)), float(declared.isf(1e-15))))
        assert mean == pytest.approx(want_mean, rel=2e-7) and sd == pytest.approx(want_sd, rel=2e-7)
        assert mean == pytest.approx(declared.mean(), rel=2e-7) and sd == pytest.approx(declared.std(), rel=2e-7)
    # a Weibull with a shift is not of the plain Weibull family
    assert family_of(prob.Weibull(2.0, 1.5, loc=1.0)) is None and family_of(prob.Weibull(2.0, 1.5)) == "weibull"
    # a normal truncated on one side cannot have a standard deviation above the mean's distance from the bound
    for cov in (1.01, 1.2, 3.0):
        with pytest.raises(NoMatch, match="distance of the mean"):
            moment_matched(prob.LogNormal.from_mean_sd(100.0, 100.0 * cov), "normal")
    assert moment_matched(prob.LogNormal.from_mean_sd(100.0, 90.0), "normal").std() == pytest.approx(90.0, rel=1e-7)
    # close to the limit a match still exists and is found (the early refusal applies only beyond the limit)
    close = moment_matched(prob.LogNormal.from_mean_sd(100.0, 97.0), "normal")
    assert close.mean() == pytest.approx(100.0, rel=1e-7) and close.std() == pytest.approx(97.0, rel=1e-7)
    with pytest.raises(ValueError, match="family must be"):
        moment_matched(prob.Normal(0.0, 1.0), "gumbel")


def test_family_sensitivity_keeps_correlations_and_is_reproducible() -> None:
    def build(toughness: prob.Distribution) -> tuple[cf.ProbabilisticAssessment, prob.Distribution]:
        load = prob.Truncated(prob.Gumbel.from_mean_sd(190e6, 19e6), lo=0.0, name="pm")
        steel = cf.Material(355e6, 490e6, 207e9, toughness, yielding="continuous")
        problem = cf.Assessment(cf.FlatPlate(25e-3, 0.5), cf.SurfaceFlaw(6e-3, 60e-3), steel, cf.StressState(pm=load))
        return cf.ProbabilisticAssessment(problem).correlate(toughness, load, 0.6, kind="latent"), load

    kmat = prob.LogNormal.from_mean_sd(100e6, 25e6, name="kmat")
    settings = dict(seed=3, n_explore=100_000, replicates=2, samples_per_level=2000)
    pa, _ = build(kmat)
    rows = {(r.variable, r.family): r for r in pa.family_sensitivity(**settings)}
    again = {(r.variable, r.family): r for r in pa.family_sensitivity(**settings)}
    for family in ("normal", "weibull"):
        by_hand, _ = build(moment_matched(kmat, family))
        expected = by_hand.solve(**settings).reliability
        assert rows[("kmat", family)].pf == expected.pf and rows[("kmat", family)].guaranteed == expected.guaranteed
        assert again[("kmat", family)].pf == expected.pf
        # the correlation matters here: without it the answer is different
        loose, _ = build(moment_matched(kmat, family))
        loose.vector._latent = np.eye(2)
        assert loose.solve(**settings).reliability.pf != expected.pf
    # a positive correlation of toughness with load lowers the failure probability
    alone = cf.ProbabilisticAssessment(pa.problem).solve(**settings).reliability.pf
    assert pa.solve(**settings).reliability.pf < 0.7 * alone


def test_family_sensitivity_reports_a_family_that_cannot_match() -> None:
    kmat = prob.LogNormal.from_mean_sd(100e6, 120e6, name="kmat")  # coefficient of variation above 1
    steel = cf.Material(355e6, 490e6, 207e9, kmat, yielding="continuous")
    problem = cf.Assessment(cf.FlatPlate(25e-3, 0.5), cf.SurfaceFlaw(6e-3, 60e-3), steel, cf.StressState(pm=150e6))
    rows = cf.ProbabilisticAssessment(problem).family_sensitivity(seed=0, n_explore=20_000, replicates=2)
    by = {r.family: r for r in rows}
    assert by["normal"].status == "no moment-matched alternative" and by["normal"].pf is None
    assert by["weibull"].pf is not None and by["weibull"].pf > 0.0


def test_family_sensitivity_through_the_engine() -> None:
    kmat = prob.LogNormal.from_mean_sd(100e6, 25e6, name="kmat")
    load = prob.Truncated(prob.Gumbel.from_mean_sd(190e6, 19e6), lo=0.0, name="pm")
    steel = cf.Material(355e6, 490e6, 207e9, kmat, yielding="continuous")
    problem = cf.Assessment(cf.FlatPlate(25e-3, 0.5), cf.SurfaceFlaw(6e-3, 60e-3), steel, cf.StressState(pm=load))
    pa = cf.ProbabilisticAssessment(problem)
    base = pa.solve(seed=0, n_explore=400_000)
    assert 1e-5 < base.reliability.pf < 0.05
    rows = pa.family_sensitivity(seed=0, n_explore=400_000)
    by = {(r.variable, r.family): r for r in rows}
    assert by[("pm", "Truncated")].status == "no alternative family defined"
    assert set(k for k in by if k[0] == "kmat") == {("kmat", "normal"), ("kmat", "weibull")}
    for family in ("normal", "weibull"):
        row = by[("kmat", family)]
        assert row.pf is not None and row.guaranteed is not None and row.status in ("resolved", "conditional")
    # the lower tail of toughness drives failure: the family changes the answer materially
    values = [by[("kmat", f)].pf for f in ("normal", "weibull")] + [base.reliability.pf]
    assert max(values) / min(values) > 1.3


def test_failure_point_and_association_through_the_engine() -> None:
    kmat = prob.LogNormal.from_mean_sd(100e6, 25e6, name="kmat")
    load = prob.Truncated(prob.Gumbel.from_mean_sd(170e6, 20e6), lo=0.0, name="pm")
    inert = prob.Normal(0.0, 1.0, name="inert")
    steel = cf.Material(355e6, 490e6, 207e9, kmat, yielding="continuous")
    problem = cf.Assessment(cf.FlatPlate(25e-3, 0.5), cf.SurfaceFlaw(6e-3, 60e-3), steel,
                            cf.StressState(pm=load, qm=100e6, qb=inert))
    pa = cf.ProbabilisticAssessment(problem)
    res = pa.solve(seed=0, n_explore=200_000)
    points = pa.most_likely_failure_points(res)
    assert points and points[0].physical is not None
    values = dict(zip(pa.names, points[0].physical, strict=True))
    assert values["kmat"] < kmat.mean() and values["pm"] > load.mean()  # low toughness, high load
    ranked = [a.name for a in pa.association(res)]
    assert ranked[-1] == "inert" and set(ranked[:2]) == {"kmat", "pm"}
    # one mechanism: one point, at the distance the failure probability implies
    assert len(points) == 1 and points[0].share == 1.0 and points[0].dimension == 3
    assert points[0].distance == pytest.approx(-stats.norm.ppf(res.reliability.pf), abs=0.25)
    # implied partial factors from that point against given reference values
    factors = pa.implied_partial_factors(points[0], {"kmat": (75e6, "resistance"), "pm": (170e6, "load")})
    assert factors == {"kmat": pytest.approx(75e6 / values["kmat"]), "pm": pytest.approx(values["pm"] / 170e6)}
    with pytest.raises(ValueError, match="not random variables"):
        pa.implied_partial_factors(points[0], {"sy": (355e6, "resistance")})
    bare = most_likely_failure_points(pa.limit_state("failed"), pa.failure_samples(res)[:200])[0]
    with pytest.raises(ValueError, match="no physical coordinates"):
        pa.implied_partial_factors(bare, {"kmat": (75e6, "resistance")})
    # two variables with one name: which one is meant cannot be told
    twins = cf.Assessment(
        cf.FlatPlate(25e-3, 0.5), cf.SurfaceFlaw(6e-3, 60e-3),
        cf.Material(355e6, 490e6, 207e9, prob.LogNormal.from_mean_sd(100e6, 25e6, name="x"), yielding="continuous"),
        cf.StressState(pm=prob.Truncated(prob.Gumbel.from_mean_sd(170e6, 20e6), lo=0.0, name="x")),
    )
    with pytest.raises(ValueError, match="not unique"):
        cf.ProbabilisticAssessment(twins).implied_partial_factors(points[0], {"x": (75e6, "resistance")})
    for bad in (np.inf, np.nan):
        with pytest.raises(ValueError, match="finite"):
            implied_partial_factor(bad, 1.0, "load")


def test_diagnostics_use_the_subset_samples_when_the_subset_channel_gives_the_headline() -> None:
    kmat = prob.LogNormal.from_mean_sd(100e6, 15e6, name="kmat")
    load = prob.Truncated(prob.Gumbel.from_mean_sd(150e6, 10e6), lo=0.0, name="pm")
    steel = cf.Material(355e6, 490e6, 207e9, kmat, yielding="continuous")
    problem = cf.Assessment(cf.FlatPlate(25e-3, 0.5), cf.SurfaceFlaw(4e-3, 40e-3), steel, cf.StressState(pm=load))
    pa = cf.ProbabilisticAssessment(problem)
    res = pa.solve(seed=0, n_explore=20_000)
    assert res.reliability.status == "conditional" and res.reliability.monte_carlo.failure_points.shape[0] == 0
    samples = pa.failure_samples(res)
    assert samples.shape[0] > 5000 and bool(np.all(pa.limit_state("failed")(samples[:500]).fail))
    (point,) = pa.most_likely_failure_points(res)
    assert point.distance == pytest.approx(-stats.norm.ppf(res.reliability.pf), abs=0.15)


def test_failure_point_when_failure_is_leaving_the_validity_range() -> None:
    # a tiny load: the only "failure" is a flaw deeper than the plate is thick (outside validity)
    depth = prob.Uniform(1e-3, 30e-3, name="a")
    steel = cf.Material(355e6, 490e6, 207e9, 300e6, yielding="continuous")
    problem = cf.Assessment(cf.FlatPlate(25e-3, 0.5), cf.SurfaceFlaw(depth, 80e-3), steel, cf.StressState(pm=5e6))
    pa = cf.ProbabilisticAssessment(problem)
    res = pa.solve(seed=0, n_explore=20_000)
    assert res.modes is not None and res.modes.out_of_validity == 1.0
    (point,) = pa.most_likely_failure_points(res)
    assert point.physical is not None and point.physical[0] == pytest.approx(25e-3, rel=1e-9)  # the crossing itself


# ---- inspection-conditioned flaw sizes -----------------------------------------------------


def test_undetected_with_constant_pod_is_the_prior() -> None:
    prior = prob.Exponential(mean=2.0e-3)
    post, p_nd = undetected(prior, ConstantPoD(0.3))
    assert p_nd == pytest.approx(0.7, rel=1e-9)
    x = prior.ppf(np.array([1e-9, 0.01, 0.5, 0.99, 1 - 1e-9]))
    np.testing.assert_allclose(post.cdf(x), prior.cdf(x), rtol=1e-6)
    np.testing.assert_allclose(post.sf(x), prior.sf(x), rtol=1e-6)
    np.testing.assert_allclose(post.ppf(prior.cdf(x)), x, rtol=1e-6)
    assert post.mean() == pytest.approx(2.0e-3, rel=1e-10) and post.std() == pytest.approx(2.0e-3, rel=1e-10)
    assert post.support() == prior.support()
    # far tails (beyond a normal score of 8) keep relative accuracy
    for q in (1e-20, 1e-40, 1e-100):
        assert post.sf(prior.isf(q)) == pytest.approx(q, rel=1e-6)
        assert post.isf(q) == pytest.approx(prior.isf(q), rel=1e-6)
    # heavy tails: most of the second moment lies beyond a normal score of 5
    for heavy in (prob.LogNormal(1.0, 5.0), prob.LogNormal(3.0, 40.0), prob.Weibull(1.0, 0.4)):
        same, _ = undetected(heavy, ConstantPoD(0.3))
        assert same.mean() == pytest.approx(heavy.mean(), rel=1e-8) and same.std() == pytest.approx(heavy.std(), rel=1e-6)


def test_undetected_with_a_step_pod_is_the_truncated_prior() -> None:
    prior = prob.Exponential(mean=1.0)
    post, p_nd = undetected(prior, StepPoD(2.0))
    assert p_nd == pytest.approx(1 - np.exp(-2.0), rel=1e-7)
    a = np.array([1e-6, 0.5, 1.0, 1.9, 1.999999])
    np.testing.assert_allclose(post.cdf(a), (1 - np.exp(-a)) / (1 - np.exp(-2.0)), rtol=1e-6)
    assert post.cdf(2.0) == pytest.approx(1.0, abs=1e-9) and post.sf(5.0) == 0.0
    assert post.isf(1e-9) == pytest.approx(2.0, rel=1e-6)
    # the support ends at the threshold, and so do the end quantiles
    assert post.support() == (0.0, pytest.approx(2.0, rel=1e-12))
    for top in (post.ppf(1.0), post.isf(0.0), post.from_normal_score(38.0)):
        assert top == pytest.approx(2.0, rel=1e-12)
    assert post.ppf(0.0) == pytest.approx(0.0, abs=1e-200) and post.isf(1.0) == pytest.approx(0.0, abs=1e-200)
    assert post.mean() == pytest.approx((1 - 3 * np.exp(-2.0)) / (1 - np.exp(-2.0)), rel=1e-9)
    partial, p_partial = undetected(prior, StepPoD(2.0, level=0.9))
    assert p_partial == pytest.approx(1 - 0.9 * np.exp(-2.0), rel=1e-7)
    assert partial.sf(3.0) == pytest.approx(0.1 * np.exp(-3.0) / p_partial, rel=1e-6)  # relative accuracy in the tail


def test_undetected_against_independent_integrals() -> None:
    prior = prob.LogNormal(2.0e-3, 0.8)
    pod = LogLogisticPoD(a50=3.0e-3, beta=4.0)
    post, p_nd = undetected(prior, pod)

    def integrand(a: float) -> float:
        return float(prior.pdf(a) * (1.0 - pod(a)))

    total = integrate.quad(integrand, 0.0, np.inf, epsabs=0, epsrel=1e-12, limit=400)[0]
    assert p_nd == pytest.approx(total, rel=1e-7)
    for a in (2e-4, 1e-3, 3e-3, 8e-3, 3e-2, 1e-1):
        lower = integrate.quad(integrand, 0.0, a, epsabs=0, epsrel=1e-12, limit=400)[0] / total
        upper = integrate.quad(integrand, a, np.inf, epsabs=0, epsrel=1e-12, limit=400)[0] / total
        assert post.cdf(a) == pytest.approx(lower, rel=1e-6)
        assert post.sf(a) == pytest.approx(upper, rel=1e-6)  # down to about 1e-12
    assert post.sf(1e-1) < 1e-9
    q = np.array([1e-10, 1e-4, 0.3, 0.9])
    np.testing.assert_allclose(post.cdf(post.ppf(q)), q, rtol=1e-6)
    np.testing.assert_allclose(post.sf(post.isf(q)), q, rtol=1e-6)
    # an undetected population is smaller than the prior
    assert post.mean() < prior.mean()


def test_non_detection_probabilities_too_small_for_one_minus_pod() -> None:
    pod = LogLogisticPoD(a50=1.0, beta=4.0)
    np.testing.assert_allclose(pod.non_detection(np.array([1e5, 2e5])), [1e-20, 6.25e-22], rtol=1e-12)
    assert list(pod(np.array([-1.0, 0.0]))) == [0.0, 0.0] and list(pod.non_detection(np.array([-1.0, 0.0]))) == [1.0, 1.0]
    a = np.array([0.3, 1.0, 2.5])
    np.testing.assert_allclose(pod(a) + pod.non_detection(a), 1.0, rtol=1e-15)
    np.testing.assert_allclose(pod(a), 1.0 / (1.0 + (1.0 / a) ** 4.0), rtol=1e-15)
    # sizes at which the power (a / a50)^beta overflows although the probability is representable
    assert pod.non_detection(np.array([1e80]))[0] == pytest.approx(1e-320, rel=1e-3) and pod(np.array([1e80]))[0] == 1.0
    assert pod.non_detection(np.array([1e-80]))[0] == 1.0 and pod(np.array([1e-80]))[0] == pytest.approx(1e-320, rel=1e-3)
    huge, p_huge = undetected(prob.Empirical([1e80]), pod)
    assert p_huge == pytest.approx(1e-320, rel=1e-3) and list(huge.values) == [1e80]
    post, p_nd = undetected(prob.Empirical([1e5, 2e5]), pod)
    assert p_nd == pytest.approx(5.3125e-21, rel=1e-12)
    np.testing.assert_allclose(post.weights, [16 / 17, 1 / 17], rtol=1e-12)
    # continuous prior concentrated where the PoD rounds to one
    prior = prob.Uniform(1e5, 2e5)
    post, p_nd = undetected(prior, pod)
    assert p_nd == pytest.approx((1e-15 - 0.125e-15) / 3 / 1e5, rel=1e-9)  # integral of a^-4 / 1e5
    assert post.cdf(1.5e5) == pytest.approx((1 - 1.5**-3) / (1 - 2.0**-3), rel=1e-9)
    # a model without a direct form still works where 1 - PoD is representable
    plain, p_plain = undetected(prob.Exponential(mean=1.0), lambda a: np.where(np.asarray(a) >= 2.0, 0.9, 0.0))
    assert p_plain == pytest.approx(1 - 0.9 * np.exp(-2.0), rel=1e-3)


def test_inspection_models_reject_impossible_inputs() -> None:
    prior = prob.Exponential(mean=1.0)
    for make in (lambda: ConstantPoD(-0.5), lambda: ConstantPoD(1.5), lambda: ConstantPoD(float("nan")),
                 lambda: StepPoD(1.0, level=1.2), lambda: StepPoD(float("nan")), lambda: LogLogisticPoD(0.0, 2.0),
                 lambda: LogLogisticPoD(1.0, -1.0)):
        with pytest.raises(ValueError):
            make()
    # nothing can remain undetected: zero probability, for continuous and discrete priors
    for pod in (ConstantPoD(1.0), StepPoD(-1.0)):
        for population in (prior, prob.Empirical([1.0, 2.0])):
            with pytest.raises(ValueError, match="zero probability"):
                undetected(population, pod)
    with pytest.raises(ValueError, match="zero probability"):
        detected(1.0, prob.Normal(0.0, 0.1), prob.Uniform(50.0, 60.0), ConstantPoD(1.0))
    # a "PoD" outside [0, 1]
    for bad in (lambda a: 1.5 + 0.0 * np.asarray(a), lambda a: -0.5 + 0.0 * np.asarray(a), lambda a: np.nan * np.asarray(a)):
        for population in (prior, prob.Empirical([1.0, 2.0])):
            with pytest.raises(ValueError, match="non-negative"):
                undetected(population, bad)
    with pytest.raises(ValueError, match="finite"):
        detected(float("nan"), prob.Normal(0.0, 0.1), prior, ConstantPoD(1.0))
    # an atom exactly at a detection threshold is detected (PoD applies from the threshold on)
    post, p_nd = undetected(prob.Empirical([1.0, 2.0, 3.0]), StepPoD(2.0))
    assert list(post.values) == [1.0] and p_nd == pytest.approx(1 / 3)


def test_discrete_prior_is_reweighted() -> None:
    prior = prob.Empirical([1.0, 2.0])
    post, p_nd = undetected(prior, StepPoD(1.5))
    assert p_nd == pytest.approx(0.5) and list(post.values) == [1.0] and post.is_discrete
    soft, p_soft = undetected(prior, ConstantPoD(0.2))
    assert p_soft == pytest.approx(0.8) and list(soft.weights) == [0.5, 0.5]


def test_detected_posterior_matches_the_conjugate_normal_result() -> None:
    prior = prob.Normal(5.0, 2.0)
    error = prob.Normal(0.0, 1.0)  # measured minus true
    post, _ = detected(6.5, error, prior, ConstantPoD(1.0))
    var = 1.0 / (1 / 4.0 + 1 / 1.0)
    mean = var * (5.0 / 4.0 + 6.5 / 1.0)
    ref = stats.norm(mean, np.sqrt(var))
    x = np.array([mean - 3, mean - 1, mean, mean + 1, mean + 3])
    np.testing.assert_allclose(post.cdf(x), ref.cdf(x), rtol=1e-6)
    np.testing.assert_allclose(post.sf(x), ref.sf(x), rtol=1e-6)
    assert post.mean() == pytest.approx(mean, rel=1e-10) and post.std() == pytest.approx(np.sqrt(var), rel=1e-10)
    # a detection threshold removes posterior mass below it: the conjugate posterior truncated at 6
    cut, _ = detected(6.5, error, prior, StepPoD(6.0))
    assert cut.cdf(5.999) == pytest.approx(0.0, abs=1e-12) and cut.mean() > mean
    for x in (6.0 + 1e-6, 6.001, 6.3, 8.0):
        assert cut.cdf(x) == pytest.approx((ref.cdf(x) - ref.cdf(6.0)) / ref.sf(6.0), rel=1e-7)
    assert cut.support()[0] == pytest.approx(6.0, rel=1e-12) and cut.ppf(0.0) == pytest.approx(6.0, rel=1e-12)
    # a flaw that can only have been found above 2 mm is positive, and the model accepts it
    found, _ = detected(4e-3, prob.Normal(0.0, 0.5e-3), prob.Normal(3e-3, 2e-3), StepPoD(2e-3))
    assert cf.SurfaceFlaw(a=found, two_c=80e-3).a is found and found.support()[0] == pytest.approx(2e-3)


def test_detected_far_tails_and_an_observation_the_prior_cannot_hold() -> None:
    post, _ = detected(6.5, prob.Normal(0.0, 0.1), prob.Normal(5.0, 2.0), ConstantPoD(1.0))
    var = 1.0 / (1 / 4.0 + 100.0)
    ref = stats.norm(var * (5.0 / 4.0 + 650.0), np.sqrt(var))
    # relative comparisons only (abs=0): the values are far below pytest's default absolute tolerance
    for k in (5.0, 10.0, 12.0, 20.0):  # beyond 8.8 sizing-error standard deviations too
        upper, lower = ref.sf(ref.mean() + k * ref.std()), ref.cdf(ref.mean() - k * ref.std())
        assert float(post.sf(ref.mean() + k * ref.std())) == pytest.approx(upper, rel=1e-6, abs=0.0)
        assert float(post.cdf(ref.mean() - k * ref.std())) == pytest.approx(lower, rel=1e-6, abs=0.0)
    narrow, _ = detected(6.5, prob.Normal(0.0, 0.01), prob.Normal(5.0, 2.0), ConstantPoD(1.0))
    var = 1.0 / (1 / 4.0 + 1e4)
    ref = stats.norm(var * (5.0 / 4.0 + 6.5e4), np.sqrt(var))
    for k in (10.0, 20.0):
        assert float(narrow.sf(ref.mean() + k * ref.std())) == pytest.approx(ref.sf(ref.mean() + k * ref.std()), rel=1e-6, abs=0.0)
        assert float(narrow.cdf(ref.mean() - k * ref.std())) == pytest.approx(ref.cdf(ref.mean() - k * ref.std()), rel=1e-6, abs=0.0)
    # 30 prior standard deviations out: still exact
    far, constant = detected(70.0, prob.Normal(0.0, 0.1), prob.Normal(40.0, 1.0), ConstantPoD(1.0))
    assert constant == pytest.approx(stats.norm(40.0, np.sqrt(1.01)).pdf(70.0), rel=1e-9, abs=0.0)
    assert far.mean() == pytest.approx(40.0 + 30.0 / 1.01, rel=1e-10)
    # 38 out: most of the posterior lies beyond what can be integrated; refused, not truncated
    for measured in (78.0, 76.8, 2.0, 3.2):  # either tail, and close enough to the edge to lose only the outer tail
        with pytest.raises(ValueError, match="tail of the prior"):
            detected(measured, prob.Normal(0.0, 0.1), prob.Normal(40.0, 1.0), ConstantPoD(1.0))
    with pytest.raises(ValueError, match="tail of the prior"):  # a wider error: only 1e-5 of the posterior is cut
        detected(78.2, prob.Normal(0.0, 0.3), prob.Normal(40.0, 1.0), ConstantPoD(1.0))
    for function in (most_likely_failure_points, lambda g, pts: cluster_failure_points(pts, g)):
        with pytest.raises(ValueError, match="finite"):
            function(linear(3.0), np.array([[3.5, np.nan]]))
    # quantiles of an array are converged one by one (round trip at rounding level)
    q = np.array([1e-12, 1e-3, 0.2, 0.5, 0.8, 0.999])
    np.testing.assert_allclose(post.cdf(post.ppf(q)), q, rtol=1e-12)
    np.testing.assert_allclose(post.sf(post.isf(q)), q, rtol=1e-12)
    # quantiles next to a hard end of the support do not go backwards
    cut, _ = undetected(prob.Exponential(mean=1.0), StepPoD(2.0))
    assert bool(np.all(np.diff(cut.ppf(np.linspace(0.999999, 1.0, 2001))) >= 0.0))


def test_detected_with_a_biased_sizing_error() -> None:
    # error = measured - true with mean +0.7: the instrument oversizes, so the flaw is smaller than measured
    prior = prob.Normal(5.0, 2.0)
    post, constant = detected(6.5, prob.Normal(0.7, 1.0), prior, ConstantPoD(1.0))
    var = 1.0 / (1 / 4.0 + 1 / 1.0)
    assert post.mean() == pytest.approx(var * (5.0 / 4.0 + (6.5 - 0.7) / 1.0), rel=1e-10)
    assert constant == pytest.approx(stats.norm(5.0 + 0.7, np.sqrt(5.0)).pdf(6.5), rel=1e-10)
    # a one-sided error: the flaw is never larger than measured
    under, _ = detected(6.5, prob.Exponential(mean=0.5), prior, ConstantPoD(1.0))
    assert under.support()[1] == pytest.approx(6.5, rel=1e-9) and under.cdf(6.5) == pytest.approx(1.0, abs=1e-12)
    assert under.sf(6.0) > 0.5


@pytest.mark.parametrize("ratio", [1.0, 0.1, 0.01, 1e-3, 1e-4, 1e-6])
def test_detected_with_a_sizing_error_much_narrower_than_the_prior(ratio: float) -> None:
    s = 2.0 * ratio
    post, constant = detected(6.5, prob.Normal(0.0, s), prob.Normal(5.0, 2.0), ConstantPoD(1.0))
    var = 1.0 / (1 / 4.0 + 1 / s**2)
    mean = var * (5.0 / 4.0 + 6.5 / s**2)
    ref = stats.norm(mean, np.sqrt(var))
    assert constant == pytest.approx(stats.norm(5.0, np.sqrt(4.0 + s**2)).pdf(6.5), rel=1e-9)
    assert post.mean() == pytest.approx(mean, rel=1e-10) and post.std() == pytest.approx(np.sqrt(var), rel=1e-8)
    x = mean + np.sqrt(var) * np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    np.testing.assert_allclose(post.cdf(x), ref.cdf(x), rtol=1e-8)
    np.testing.assert_allclose(post.sf(x), ref.sf(x), rtol=1e-8)


@pytest.mark.parametrize(
    ("error", "tolerance"),
    [
        (prob.Exponential(mean=2.0e-4), 1e-9),  # one-sided and 10,000 times narrower than the prior
        (prob.Normal(6.0e-4, 2.0e-4), 1e-9),  # biased by three of its own standard deviations
        (prob.LogNormal(0.001, 3.0), 1e-9),  # a narrow core with a long tail
        (prob.LogNormal(0.01, 1.0), 1e-9),
        (prob.Weibull(0.5, 0.7), 1e-8),  # density unbounded at zero error
        (prob.Weibull(0.01, 0.4), 2e-5),
    ],
)
def test_detected_with_skewed_and_one_sided_sizing_errors(error: prob.Distribution, tolerance: float) -> None:
    prior = prob.Normal(5.0, 2.0)
    post, constant = detected(6.5, error, prior, ConstantPoD(1.0))

    # reference in the error's own normal score: constant = E[prior density at (measured - error)]
    def integrand(z: float) -> float:
        return float(prior.pdf(6.5 - error.from_normal_score(z)) * stats.norm.pdf(z))

    def integral(lo: float, hi: float) -> float:
        return float(integrate.quad(integrand, lo, hi, epsabs=0, epsrel=1e-13, limit=400)[0])

    total = integral(-9.0, 9.0)
    assert constant == pytest.approx(total, rel=tolerance)
    for q in (0.1, 0.5, 0.9):  # P(true size below measured - q-quantile of the error) = P(error above it)
        size = 6.5 - float(error.ppf(q))
        z = float(stats.norm.ppf(q))
        assert float(post.cdf(size)) == pytest.approx(integral(z, 9.0) / total, rel=10 * tolerance, abs=1e-12)


def test_quantiles_outside_the_unit_interval_are_not_numbers() -> None:
    post, _ = undetected(prob.Exponential(mean=1.0), LogLogisticPoD(1.0, 4.0))
    out = post.ppf(np.array([-0.1, 1.1, np.nan, 0.5]))
    assert list(np.isnan(out)) == [True, True, True, False] and bool(np.all(np.isnan(post.isf(np.array([-0.1, 1.1])))))
    assert bool(np.isnan(post.from_normal_score(np.nan)))
    # sampling agrees with the distribution (both sides of the median are inverted separately)
    sample = post.from_normal_score(np.random.default_rng(3).standard_normal(100_000))
    assert float(np.mean(sample)) == pytest.approx(post.mean(), abs=4 * post.std() / np.sqrt(100_000))
    assert float(np.mean(sample <= post.ppf(0.9))) == pytest.approx(0.9, abs=0.004)
    np.testing.assert_allclose(post.cdf(post.ppf(np.array([1e-12, 0.2, 0.5, 0.8]))), [1e-12, 0.2, 0.5, 0.8], rtol=1e-9)
    np.testing.assert_allclose(post.sf(post.isf(np.array([1e-12, 0.2, 0.5]))), [1e-12, 0.2, 0.5], rtol=1e-9)


def test_narrow_sizing_error_on_a_skewed_prior_with_a_pod() -> None:
    prior, pod, s = prob.LogNormal(2.0e-3, 0.8), LogLogisticPoD(a50=3.0e-3, beta=4.0), 1.0e-6
    post, constant = detected(4.0e-3, prob.Normal(0.0, s), prior, pod)

    def integrand(a: float) -> float:
        return float(prior.pdf(a) * pod(a) * stats.norm.pdf(4.0e-3 - a, scale=s))

    edges = 4.0e-3 + s * np.arange(-10.0, 11.0, 2.0)
    pieces = [integrate.quad(integrand, a, b, epsabs=0, epsrel=1e-12)[0] for a, b in zip(edges[:-1], edges[1:], strict=True)]
    assert constant == pytest.approx(sum(pieces), rel=1e-9)
    assert post.std() == pytest.approx(s, rel=1e-5) and post.mean() == pytest.approx(4.0e-3, rel=1e-6)
    assert post.cdf(4.0e-3 + s) == pytest.approx(sum(pieces[:5]) / sum(pieces) + integrate.quad(
        integrand, 4.0e-3, 4.0e-3 + s, epsabs=0, epsrel=1e-12)[0] / sum(pieces), rel=1e-8)


def test_a_conditioned_flaw_size_can_be_given_a_pearson_correlation() -> None:
    depth, _ = undetected(prob.Exponential(mean=2.0e-3), LogLogisticPoD(a50=3.0e-3, beta=4.0), name="a")
    kmat = prob.LogNormal.from_mean_sd(100e6, 25e6, name="kmat")
    pearson = prob.latent_to_pearson(depth, kmat, 0.3)
    # the same coefficient by Gauss-Hermite quadrature over the two latent normals, written out here
    nodes, weights = np.polynomial.hermite_e.hermegauss(80)
    weights = weights / np.sum(weights)
    z1, z2 = np.meshgrid(nodes, nodes, indexing="ij")
    x = depth.from_normal_score(z1.ravel()).reshape(z1.shape)
    y = kmat.from_normal_score((0.3 * z1 + np.sqrt(1 - 0.3**2) * z2).ravel()).reshape(z1.shape)
    w = weights[:, None] * weights[None, :]
    covariance = float(np.sum(w * (x - np.sum(w * x)) * (y - np.sum(w * y))))
    assert pearson == pytest.approx(covariance / (depth.std() * kmat.std()), abs=1e-4)
    assert prob.pearson_to_latent(depth, kmat, pearson) == pytest.approx(0.3, abs=1e-6)
    # moments against independent integrals
    pod, prior = LogLogisticPoD(a50=3.0e-3, beta=4.0), prob.Exponential(mean=2.0e-3)

    def moment(k: int) -> float:
        return integrate.quad(lambda a: a**k * float(prior.pdf(a) * pod.non_detection(a)), 0.0, 0.2,
                              epsabs=0, epsrel=1e-12, limit=400, points=[1e-3, 3e-3, 1e-2])[0]

    mean = moment(1) / moment(0)
    assert depth.mean() == pytest.approx(mean, rel=1e-9)
    assert depth.std() == pytest.approx(np.sqrt(moment(2) / moment(0) - mean**2), rel=1e-8)
    with pytest.raises(TypeError):
        detected(1.0, prob.Empirical([0.0]), prior, ConstantPoD(1.0))


def test_inspected_flaw_in_an_assessment() -> None:
    depth, p_nd = undetected(prob.Exponential(mean=2.0e-3), LogLogisticPoD(a50=3.0e-3, beta=4.0), name="a")
    steel = cf.Material(355e6, 490e6, 207e9, 90e6, yielding="continuous")
    problem = cf.Assessment(cf.FlatPlate(25e-3, 0.5), cf.SurfaceFlaw(a=depth, two_c=80e-3), steel,
                            cf.StressState(pm=200e6, residual=cf.Residual.as_welded()))
    res = cf.ProbabilisticAssessment(problem).solve(seed=0, n_explore=200_000)
    assert 0.0 < p_nd < 1.0 and res.per == "flaw"
    bound = per_component_upper_bound(3.0, p_nd, res.reliability.guaranteed.upper)
    assert 0.0 <= bound <= 1.0
    # the same probability by rejection sampling of the undetected population, outside the library's conditioning
    rng = np.random.default_rng(12)
    a = rng.exponential(2.0e-3, 400_000)
    a = a[rng.uniform(size=a.size) < 1.0 / (1.0 + (a / 3.0e-3) ** 4.0)]
    assert a.size / 400_000 == pytest.approx(p_nd, abs=0.003)
    direct = cf.Assessment(cf.FlatPlate(25e-3, 0.5), cf.SurfaceFlaw(a=a, two_c=80e-3), steel,
                           cf.StressState(pm=200e6, residual=cf.Residual.as_welded())).run(on_invalid="mask")
    pf = float(np.mean(~direct.acceptable | ~direct.valid))
    assert 0.015 < pf < 0.025 and res.reliability.pf == pytest.approx(pf, abs=0.0017)  # four standard errors


def test_per_component_value_is_an_upper_bound_when_flaws_share_a_load() -> None:
    # one expected undetected flaw; a shared load state makes every flaw fail with probability 1/2
    assert per_component_upper_bound(1.0, 1.0, 0.5) == pytest.approx(1 - np.exp(-0.5))
    exact = 0.5 * (1 - np.exp(-1.0))
    assert per_component_upper_bound(1.0, 1.0, 0.5) > exact
    assert per_component_upper_bound(0.0, 1.0, 0.5) == 0.0
    with pytest.raises(ValueError):
        per_component_upper_bound(1.0, 1.2, 0.5)
    for bad in ((1.0, 0.5, 1.2), (1.0, 0.5, -0.1), (-1.0, 0.5, 0.5), (float("nan"), 0.5, 0.5), (float("inf"), 0.0, 0.5)):
        with pytest.raises(ValueError):
            per_component_upper_bound(*bad)
    # inspection that misses 40% of flaws: 1 - exp(-2 x 0.4 x 0.3), worked by hand
    assert per_component_upper_bound(2.0, 0.4, 0.3) == pytest.approx(0.2133721389334830, rel=1e-12)
    # against simulation with a shared load: every flaw fails if the load is high (probability 0.3)
    rng = np.random.default_rng(13)
    undetected_flaws = rng.poisson(2.0 * 0.4, 400_000)
    high_load = rng.uniform(size=400_000) < 0.3
    simulated = float(np.mean(high_load & (undetected_flaws > 0)))
    assert simulated == pytest.approx(0.3 * (1 - np.exp(-0.8)), abs=0.003)
    assert simulated < per_component_upper_bound(2.0, 0.4, 0.3) - 0.04
