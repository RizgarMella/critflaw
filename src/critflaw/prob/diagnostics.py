"""Diagnostics from failure samples: most likely failure points and association with failure
(theory manual, section 4.7).

Nothing here is a probability estimate. The most likely failure point is indicative (its
direction degrades with the number of variables) and the association measure is not causal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from critflaw.prob.dependence import RandomVector
from critflaw.prob.solvers import LimitState

FloatArray = NDArray[np.float64]


# --------------------------------------------------------------------------------------
# Failure regions
# --------------------------------------------------------------------------------------

_GROUPS = 64  # groups of samples formed before they are joined into regions
_PATH_POINTS = 15  # points tested on each arc and each radial line between two groups
_SCAN_POINTS = 200  # points of the outward scan along a line from the origin
_BATCH = 200_000  # largest number of points in one call of the limit state


def _unit(v: FloatArray) -> FloatArray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.asarray(v / np.where(norm > 0.0, norm, 1.0), dtype=float)


def _fails(limit_state: LimitState, points: FloatArray) -> NDArray[np.bool_]:
    if points.shape[0] == 0:  # a single group has no pairs to test
        return np.zeros(0, dtype=bool)
    parts = [limit_state(points[i : i + _BATCH]).fail for i in range(0, points.shape[0], _BATCH)]
    return np.asarray(np.concatenate(parts), dtype=bool)


def _cones(points: FloatArray, cone_degrees: float, limit: int) -> NDArray[np.intp]:
    """Direction cones: the unassigned sample nearest the origin becomes a centre and takes every
    unassigned sample within ``cone_degrees`` of its direction; at most ``limit`` centres, the
    rest go to the nearest centre direction. A sample at the origin has no direction and forms
    a cluster of its own."""
    n = points.shape[0]
    labels = np.full(n, -1, dtype=np.intp)
    norms = np.linalg.norm(points, axis=1)
    unit = _unit(points)
    cos_limit = float(np.cos(np.radians(cone_degrees)))
    centres: list[FloatArray] = []
    directed = norms > 0.0
    while len(centres) < limit:
        free = np.nonzero((labels < 0) & directed)[0]
        if free.size == 0:
            break
        centre = unit[free[np.argmin(norms[free])]]
        centres.append(centre)
        labels[free[unit[free] @ centre >= cos_limit]] = len(centres) - 1
    free = np.nonzero((labels < 0) & directed)[0]
    if free.size:
        labels[free] = np.argmax(unit[free] @ np.asarray(centres).T, axis=1)
    labels[~directed] = len(centres)
    return labels


def _groups(points: FloatArray, limit: int) -> NDArray[np.intp]:
    """Compact groups of samples (k-means, deterministic): the first centre is the sample nearest
    the origin, each further one the sample farthest from all centres so far, followed by 25
    rounds of reassignment. Samples of a small separate region are far from the centres of the
    large one, so the region gets centres of its own. Splitting one region into many groups
    does no harm: groups of one region are joined again using the limit state."""
    squared = np.sum(points**2, axis=1)
    chosen = [int(np.argmin(squared))]
    gap = np.sum((points - points[chosen[0]]) ** 2, axis=1)
    while len(chosen) < min(limit, points.shape[0]):
        j = int(np.argmax(gap))
        if gap[j] <= 0.0:
            break
        chosen.append(j)
        gap = np.minimum(gap, np.sum((points - points[j]) ** 2, axis=1))
    centres = points[chosen].copy()
    labels = np.zeros(points.shape[0], dtype=np.intp)
    for step in range(25):
        distance = squared[:, None] - 2.0 * points @ centres.T + np.sum(centres**2, axis=1)[None, :]
        nearest = np.argmin(distance, axis=1)
        if step > 0 and np.array_equal(nearest, labels):
            break
        labels = nearest
        count = np.bincount(labels, minlength=centres.shape[0])
        total = np.zeros_like(centres)
        np.add.at(total, labels, points)
        centres = np.where(count[:, None] > 0, total / np.maximum(count, 1)[:, None], centres)
    return np.asarray(np.unique(labels, return_inverse=True)[1], dtype=np.intp)


def _first_crossings(limit_state: LimitState, targets: FloatArray) -> tuple[FloatArray, NDArray[np.bool_]]:
    """For each row of ``targets``: the first change from acceptable to failing on the straight line
    from the origin to it (scanned at ``_SCAN_POINTS`` points, then bisected), and whether any
    scanned point failed. All lines are evaluated together."""
    m, d = targets.shape
    t = np.linspace(0.0, 1.0, _SCAN_POINTS + 1)[1:]
    failing = _fails(limit_state, (t[None, :, None] * targets[:, None, :]).reshape(-1, d)).reshape(m, t.size)
    found = np.any(failing, axis=1)
    first = np.argmax(failing, axis=1)
    hi = t[first]
    lo = np.where(first > 0, t[np.maximum(first - 1, 0)], 0.0)
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        fail = _fails(limit_state, mid[:, None] * targets)
        hi, lo = np.where(fail, mid, hi), np.where(fail, lo, mid)
    return np.asarray(hi[:, None] * targets, dtype=float), found


def _crossings(limit_state: LimitState, points: FloatArray, labels: NDArray[np.intp], count: int) -> FloatArray:
    """One crossing point per group: the nearer to the origin of the crossings on the line to the
    group's innermost sample and on the line along the mean direction of its samples. The mean
    direction averages out the scatter of single samples, which grows with the number of variables."""
    norms = np.linalg.norm(points, axis=1)
    unit = _unit(points)
    inner = np.empty((count, points.shape[1]))
    along = np.empty_like(inner)
    for j in range(count):
        mine = np.nonzero(labels == j)[0]
        inner[j] = points[mine[np.argmin(norms[mine])]]
        mean = np.mean(unit[mine], axis=0)
        along[j] = _unit(mean) * float(np.max(norms[mine])) if np.linalg.norm(mean) > 1e-6 else inner[j]
    crossing, found = _first_crossings(limit_state, np.vstack([inner, along]))
    # a failing sample always gives a crossing on its own line; the guard keeps a point in any case
    first = np.where(found[:count, None], crossing[:count], inner)
    second = crossing[count:]
    better = found[count:] & (np.linalg.norm(second, axis=1) < np.linalg.norm(first, axis=1))
    return np.asarray(np.where(better[:, None], second, first), dtype=float)


def _linked(limit_state: LimitState, crossing: FloatArray) -> NDArray[np.bool_]:
    """Matrix of pairs of crossing points with no barrier between them. Two points are linked when
    every tested point fails (a) on the radial line from the inner point out to the larger of
    the two radii and (b) on the arc between the two directions at (just beyond) that radius.
    Nearly opposite directions have no arc and are not linked."""
    k, d = crossing.shape
    first, second = np.triu_indices(k, 1)
    s = np.linspace(0.0, 1.0, _PATH_POINTS + 2)
    radius = np.linalg.norm(crossing, axis=1)
    ua, ub = _unit(crossing[first]), _unit(crossing[second])
    cosine = np.clip(np.sum(ua * ub, axis=1), -1.0, 1.0)
    theta = np.arccos(cosine)
    with np.errstate(divide="ignore", invalid="ignore"):
        wa = np.sin((1.0 - s)[None, :] * theta[:, None]) / np.sin(theta)[:, None]
        wb = np.sin(s[None, :] * theta[:, None]) / np.sin(theta)[:, None]
    same = (theta < 1e-9)[:, None]
    wa, wb = np.where(same, 1.0 - s[None, :], wa), np.where(same, s[None, :], wb)
    has_arc = (cosine > -0.999) & (radius[first] > 0.0) & (radius[second] > 0.0)
    big = 1.0001 * np.maximum(radius[first], radius[second])
    small = 1.0001 * np.minimum(radius[first], radius[second])
    arc = big[:, None, None] * _unit(wa[:, :, None] * ua[:, None, :] + wb[:, :, None] * ub[:, None, :])
    inner_direction = np.where((radius[first] <= radius[second])[:, None], ua, ub)
    lift = small[:, None] + (big - small)[:, None] * s[None, :]
    radial = lift[:, :, None] * inner_direction[:, None, :]
    path = np.where(has_arc[:, None, None], np.concatenate([arc, radial], axis=1), 0.0)
    fail = _fails(limit_state, path.reshape(-1, d)).reshape(first.size, path.shape[1])
    out = np.eye(k, dtype=bool)
    out[first, second] = out[second, first] = np.all(fail, axis=1) & has_arc
    return out


def _by_size(labels: NDArray[np.intp]) -> NDArray[np.intp]:
    """Relabel 0..k-1 in order of decreasing size."""
    values, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    rank = np.empty(values.size, dtype=np.intp)
    rank[np.argsort(-counts, kind="stable")] = np.arange(values.size)
    return np.asarray(rank[inverse], dtype=np.intp)


def _regions(
    points: FloatArray, limit_state: LimitState, max_clusters: int
) -> tuple[NDArray[np.intp], FloatArray, bool]:
    """Labels 0..k-1 in order of decreasing size, one point per region (the crossing nearest the
    origin found in it), and whether more than ``max_clusters`` regions were found."""
    group = _groups(points, _GROUPS)
    count = int(group.max()) + 1
    crossing = _crossings(limit_state, points, group, count)
    linked = _linked(limit_state, crossing)
    radius = np.linalg.norm(crossing, axis=1)
    direction = _unit(crossing)
    region = np.full(count, -1, dtype=np.intp)
    done: list[int] = []
    for i in np.argsort(radius, kind="stable"):  # from the origin outward
        below = [j for j in done if linked[i, j]]
        if below:  # joins the region of the nearest (in direction) group it can descend to
            region[i] = region[below[int(np.argmax(direction[below] @ direction[i]))]]
        else:  # nothing nearer the origin can be reached without a barrier: a region of its own
            region[i] = i
        done.append(int(i))
    labels = _by_size(region[group])
    found = int(labels.max()) + 1
    truncated = found > max_clusters
    if truncated:  # the smallest regions join the kept region nearest in mean direction
        means = np.array([_unit(np.mean(_unit(points[labels == j]), axis=0)) for j in range(max_clusters)])
        for j in range(max_clusters, found):
            mine = labels == j
            labels[mine] = int(np.argmax(means @ np.mean(_unit(points[mine]), axis=0)))
        labels = _by_size(labels)
    # the point of a region: the nearest crossing among its groups and along its own mean direction
    k = int(labels.max()) + 1
    whole = _crossings(limit_state, points, labels, k)
    best = np.empty((k, points.shape[1]))
    for j in range(k):
        members = np.unique(group[labels == j])
        candidates = np.vstack([whole[j][None, :], crossing[members]])
        best[j] = candidates[np.argmin(np.linalg.norm(candidates, axis=1))]
    return labels, best, truncated


def cluster_failure_points(
    points: FloatArray,
    limit_state: LimitState | None = None,
    max_clusters: int = 6,
    cone_degrees: float = 45.0,
) -> NDArray[np.intp]:
    """Group failure samples (standard normal space) into failure regions; labels are in order
    of decreasing size.

    With ``limit_state`` given (the reliable form): the samples are first split into up to 64
    compact groups, and each group gets a crossing point on the limit state (see
    ``most_likely_failure_points``). The groups are then joined into regions, a region being the
    set of groups from which one point of locally least distance is reached without a barrier.
    Taking the groups from the origin outward, a group joins the region of the nearest (in
    direction) group closer to the origin that it is linked to, and starts a region of its own
    if there is none. Two groups are linked when every tested point fails on the radial line
    from the inner crossing point out to the larger of the two radii and on the arc between the
    two directions at that radius. Groups are not joined merely because their regions meet
    further out (two limit states that intersect remain two regions).

    Limits. A region whose nearest points form a ring or shell (a strongly curved limit state)
    is reported as several regions at the same distance, because the arc between two points of
    the ring passes inside it. A separate region with a very small share of the samples in
    many variables can fail to get a group of its own and is then absorbed (in the cases tried,
    a share of 5% at 90 degrees or more from the main region was found up to 30 variables).
    Regions close in direction are harder the lower the reliability index and the more
    variables: at an index of 3, regions 60 degrees apart were separated up to 20 variables and
    regions 45 degrees apart or closer were reported as one from about 20 variables; at an index
    of 2, regions 60 degrees apart were merged in one run in five to ten from 15 variables, and a
    region with 13% of the samples 60 degrees from the main one was absorbed from 10 variables.
    A merged pair is reported at a distance up to 15% too large. Nearly opposite groups are
    never joined.

    Without ``limit_state`` samples are grouped by direction only (cones of ``cone_degrees``
    around the unassigned sample nearest the origin, at most ``max_clusters``). That splits a
    single region as soon as there are about six variables, because the failure samples of one
    flat limit state spread over an angle of about atan(sqrt(n - 1) / beta).
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if pts.shape[0] == 0:
        return np.empty(0, dtype=np.intp)
    if not np.all(np.isfinite(pts)):
        raise ValueError("failure samples must be finite")
    if limit_state is None:
        return _by_size(_cones(pts, cone_degrees, max_clusters))
    if bool(limit_state(np.zeros((1, pts.shape[1]))).fail[0]):
        return np.zeros(pts.shape[0], dtype=np.intp)  # the origin fails: no regions to tell apart
    return _regions(pts, limit_state, max_clusters)[0]


# --------------------------------------------------------------------------------------
# Most likely failure point
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FailurePoint:
    u: FloatArray  # in independent standard normal space
    distance: float  # |u|
    share: float  # share of the failure samples in this region
    physical: FloatArray | None = None  # in physical variables, if a random vector was given
    dimension: int = 0  # number of random variables
    label: str = "approximate"  # what the point is and how far it can be trusted


def most_likely_failure_points(
    limit_state: LimitState,
    failure_points: FloatArray,
    vector: RandomVector | None = None,
    max_clusters: int = 6,
) -> list[FailurePoint]:
    """One indicative point per failure region found among the failure samples, **sorted by share
    (largest first)**. The point nearest the origin is the one with the smallest ``distance``; it
    is not always the first.

    Lines from the origin are scanned outward (200 points each, so a failing band thinner than
    1/200 of the line can be stepped over) and the first change from acceptable to failing is
    polished by bisection. The lines are those to the innermost sample and along the mean
    direction of each group of samples and of the region as a whole (see
    ``cluster_failure_points``); the crossing nearest the origin is the region's point. It is
    approximate; with more than about ten variables its direction is only indicative, and the
    label says so. If the origin itself fails, it is returned alone with distance zero.
    """
    pts = np.atleast_2d(np.asarray(failure_points, dtype=float))
    if pts.shape[0] == 0:
        return []
    if not np.all(np.isfinite(pts)):
        raise ValueError("failure samples must be finite")
    d = pts.shape[1]

    def physical(point: FloatArray) -> FloatArray | None:
        return vector.to_physical(point[None, :])[0] if vector is not None else None

    origin = np.zeros(d)
    if bool(limit_state(origin[None, :]).fail[0]):
        label = "the origin (every variable at its median) is itself in the failure region"
        return [FailurePoint(origin, 0.0, 1.0, physical(origin), d, label)]
    labels, best, truncated = _regions(pts, limit_state, max_clusters)
    label = "approximate"
    if d > 10:
        label += f"; {d} variables: the direction is only indicative above about ten"
    if truncated:
        label += f"; more than {max_clusters} failure regions were found and the smallest were joined to the nearest"
    shares = np.bincount(labels) / pts.shape[0]
    return [
        FailurePoint(best[j], float(np.linalg.norm(best[j])), float(shares[j]), physical(best[j]), d, label)
        for j in range(best.shape[0])
    ]


# --------------------------------------------------------------------------------------
# Association with failure
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Association:
    name: str
    mean_shift: float  # mean of the marginal normal score conditional on failure
    spread: float  # its standard deviation conditional on failure (1 if unaffected)
    divergence: float  # Kullback-Leibler divergence of N(mean_shift, spread^2) from N(0, 1); NaN if no spread


def association_with_failure(
    vector: RandomVector, failure_points: FloatArray, names: tuple[str, ...], minimum_samples: int = 30
) -> list[Association]:
    """Per-variable association with failure: 0.5 (m^2 + v - 1 - ln v) for the marginal normal
    score conditional on failure. Deterministic variables are excluded; discrete ones use
    mid-distribution scores.

    This measures association, not causation: a variable correlated with an important one
    ranks high without being important, and sampling noise gives small non-zero values to
    inert variables. It is reported for information and is not used to select anything.

    With fewer than ``minimum_samples`` failure samples nothing is returned (the values would
    be noise). A variable with no spread at all among the failure samples (a discrete variable
    with every failure sample on one atom) has an infinite divergence by the formula; it is
    listed last with divergence NaN, because that says more about the sample than the variable.
    """
    pts = np.atleast_2d(np.asarray(failure_points, dtype=float))
    if pts.shape[0] < max(minimum_samples, 2):
        return []
    x = vector.to_physical(pts)
    out = []
    for i, marginal in enumerate(vector.marginals):
        if marginal.std() == 0.0:
            continue
        score = marginal.normal_score(x[:, i])
        score = score[np.isfinite(score)]
        if score.size < 2:
            continue
        m, v = float(np.mean(score)), float(np.var(score, ddof=1))
        divergence = 0.5 * (m * m + v - 1.0 - float(np.log(v))) if v > 0.0 else float("nan")
        out.append(Association(names[i], m, float(np.sqrt(v)), divergence))
    return sorted(out, key=lambda a: -a.divergence if np.isfinite(a.divergence) else np.inf)
