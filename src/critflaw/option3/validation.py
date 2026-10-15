"""V9 validation of the experimental emulator (theory manual, section 6.5), criteria fixed before the held-out
runs: held-out Kr RMSE <= 0.02 over supported steps; 95 % interval coverage per point (within the increment
floor's resolution) with its curve-level standard error; the energy score of whole supported curves against a
mean-only baseline (a multivariate proper score) and the variogram score (dependence diagnostic); the OpenTURNS
kriging cross-check (held-out predictions inside the GP's 95 % intervals at >= 90 % of points)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from critflaw.option3.curve import Option3Curve
from critflaw.option3.emulator import COVERAGE_RESOLUTION, Emulator, map_inputs

FloatArray = NDArray[np.float64]

RMSE_TARGET = 0.02
OT_AGREEMENT_TARGET = 0.9


def energy_score(draws: FloatArray, obs: FloatArray) -> float:
    """ES = E||X - y|| - 0.5 E||X - X'|| (lower is better; proper for the joint distribution)."""
    d1 = np.linalg.norm(draws - obs, axis=1).mean()
    half = draws.shape[0] // 2
    d2 = np.linalg.norm(draws[:half] - draws[half : 2 * half], axis=1).mean()
    return float(d1 - 0.5 * d2)


def variogram_score(draws: FloatArray, obs: FloatArray, p: float = 0.5) -> float:
    """VS_p = sum_{i<j} (|y_i - y_j|^p - E|X_i - X_j|^p)^2 (sensitive to the dependence between steps)."""
    n = obs.size
    s = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            ex = np.mean(np.abs(draws[:, i] - draws[:, j]) ** p)
            s += (abs(obs[i] - obs[j]) ** p - ex) ** 2
    return float(s)


@dataclass(frozen=True)
class ValidationReport:
    n_curves: int
    n_points: int
    rmse: float
    coverage: float  # the mean of the per-curve coverages (the primary, curve-level statistic)
    coverage_se_curves: float  # its standard error from the spread of per-curve coverage
    coverage_points: float  # point-weighted coverage (information; no standard error claimed)
    energy: float  # mean energy score of the emulator
    energy_baseline: float  # mean energy score of a mean-only (degenerate) forecast
    variogram: float
    rmse_pass: bool
    support_overruns: int = 0  # held-out curves whose predicted support exceeds their own usable support

    def summary(self) -> dict[str, object]:
        return dict(self.__dict__)


def validate(
    emulator: Emulator, heldout: Sequence[Option3Curve], n_draws: int = 1000, seed: int = 0
) -> ValidationReport:
    rng = np.random.default_rng(seed)
    errs, per_curve, es, es0, vs = [], [], [], [], []
    overruns = 0
    for c in heldout:
        p = emulator.predict(c.x, n_draws, rng)
        if p.lr.size > c.kr.size:  # the emulator advertised steps the FE case could not use (small-strain cap)
            overruns += 1
        k = min(p.lr.size, c.kr.size)
        draws, obs = p.kr_draws[:, :k], c.kr[:k]
        lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
        inside = (obs >= lo - COVERAGE_RESOLUTION) & (obs <= hi + COVERAGE_RESOLUTION)
        per_curve.append(float(inside.mean()))
        errs.append(draws.mean(axis=0) - obs)
        es.append(energy_score(draws, obs))
        mean_only = np.tile(draws.mean(axis=0), (2, 1))
        es0.append(energy_score(mean_only, obs))
        vs.append(variogram_score(draws, obs))
    e = np.concatenate(errs)
    rmse = float(np.sqrt(np.mean(e**2)))
    cov = np.asarray(per_curve)
    n_points = int(e.size)
    weights = np.array([x.size for x in errs], dtype=float)
    coverage_points = float(np.sum(cov * weights) / weights.sum())
    coverage = float(cov.mean())
    se = float(np.std(cov, ddof=1) / np.sqrt(cov.size)) if cov.size > 1 else float("nan")
    return ValidationReport(
        len(heldout), n_points, rmse, coverage, se, coverage_points, float(np.mean(es)), float(np.mean(es0)),
        float(np.mean(vs)),
        rmse <= RMSE_TARGET and overruns == 0, overruns,
    )  # fmt: skip


def openturns_agreement(emulator: Emulator, train: Sequence[Option3Curve], heldout: Sequence[Option3Curve]) -> float:
    """Fraction of held-out increments where an independent OpenTURNS kriging (Matérn-5/2, constant trend, its own
    hyperparameters) on the same training increments lies inside the GP's 95 % interval of the increment."""
    import openturns as ot  # type: ignore[import-untyped]

    # the same target and input space as the GP: increments minus the prior, mapped inputs (the earlier version
    # compared OpenTURNS on raw increments with the GP's prior-relative component)
    em = emulator
    lo = np.r_[map_inputs(em.lower, em.log_inputs), 2.0]
    hi = np.r_[map_inputs(em.upper, em.log_inputs), 11.0]
    rows, ys = [], []
    for c in train:
        y = c.increments() - (em.prior(c.x, c.lr) if em.prior is not None else 0.0)
        for i, yk in enumerate(y):
            rows.append(np.r_[map_inputs(c.x, em.log_inputs), float(i + 2)])
            ys.append(yk)
    z = (np.asarray(rows) - lo) / (hi - lo)
    algo = ot.KrigingAlgorithm(
        ot.Sample(z), ot.Sample(np.asarray(ys)[:, None]), ot.MaternModel([0.5] * 6, [1.0], 2.5),
        ot.ConstantBasisFactory(6).build(),
    )  # fmt: skip
    algo.run()
    meta = algo.getResult().getMetaModel()
    hits, total = 0, 0
    for c in heldout:
        y = c.increments()
        zq = em.gp_input(c.x, np.arange(2, y.size + 2))
        m, v = em.gp.predict(zq)
        zs = (zq - lo) / (hi - lo)
        m_ot = np.array(meta(ot.Sample(zs))).ravel()
        hits += int(np.sum(np.abs(m_ot - m) <= 1.96 * np.sqrt(v) + 1e-9))
        total += y.size
    return hits / max(total, 1)


__all__ = ["validate", "openturns_agreement", "ValidationReport", "energy_score", "variogram_score", "RMSE_TARGET"]
