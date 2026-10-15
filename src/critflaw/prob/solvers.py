"""Reliability solvers on a limit state in independent standard normal space (theory manual, section 4.4).

A limit state is a function ``u (m, n) -> Evaluation`` giving, per sample, the authoritative
failure indicator and a driving variable ``y`` that is negative for failed samples and positive
otherwise (``-inf`` / ``+inf`` are allowed, e.g. for samples outside a solution's validity).

What a result claims (theory manual, section 4.0):

* guaranteed: a Clopper-Pearson interval from plain Monte Carlo, valid whatever the limit state;
* conditional: a subset-simulation estimate, valid only if the driving variable led the sampler
  to every failure region that matters, which cannot be verified from the samples.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from critflaw.prob.intervals import Interval, clopper_pearson

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class Evaluation:
    y: FloatArray  # driving variable; sign agrees with ``fail``
    fail: BoolArray  # authoritative failure indicator
    invalid: BoolArray | None = None  # samples outside the validity of a solution, if the problem has any


LimitState = Callable[[FloatArray], Evaluation]

Status = Literal[
    "resolved", "contradicted", "conditional", "plateau", "level cap reached; unresolved", "replicates inconsistent"
]


def consistent_driving_variable(y: FloatArray, fail: BoolArray) -> FloatArray:
    """Force the sign of ``y`` to agree with the indicator where they differ only by rounding.

    A disagreement larger than 1e-9 is an internal error, so this cannot hide a defect.
    """
    y_ = np.asarray(y, dtype=float)
    if np.any(np.isnan(y_)):
        raise RuntimeError("the driving variable is NaN for some samples")
    bad = (fail & (y_ > 1e-9)) | (~fail & (y_ < -1e-9))
    if np.any(bad):
        raise RuntimeError("driving variable and failure indicator disagree beyond rounding")
    return np.asarray(np.where(fail, np.minimum(y_, -1e-12), np.maximum(y_, 1e-12)), dtype=float)


# --------------------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MonteCarloResult:
    failures: int
    samples: int
    interval: Interval
    failure_points: FloatArray = field(repr=False)  # u of failing samples (capped)
    invalid: int = 0  # samples outside validity among ``samples``

    @property
    def pf(self) -> float:
        return self.failures / self.samples

    @property
    def few_failures(self) -> bool:
        return self.failures < 10


def monte_carlo(
    limit_state: LimitState, dimension: int, samples: int, seed: int, batch: int = 200_000, keep: int = 20_000
) -> MonteCarloResult:
    """Plain Monte Carlo with a two-sided 99% Clopper-Pearson interval."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(seed)
    failures, done, invalid = 0, 0, 0
    kept: list[FloatArray] = []
    kept_count = 0
    while done < samples:
        m = min(batch, samples - done)
        u = rng.standard_normal((m, dimension))
        evaluation = limit_state(u)
        fail = evaluation.fail
        failures += int(np.sum(fail))
        if evaluation.invalid is not None:
            invalid += int(np.sum(evaluation.invalid))
        if kept_count < keep and np.any(fail):
            take = u[fail][: keep - kept_count]
            kept.append(take)
            kept_count += take.shape[0]
        done += m
    points = np.vstack(kept) if kept else np.empty((0, dimension))
    return MonteCarloResult(failures, samples, clopper_pearson(failures, samples), points, invalid)


# --------------------------------------------------------------------------------------
# Subset simulation
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SubsetRun:
    status: Literal["ok", "plateau", "level cap"]
    pf: float  # product of counted level fractions (NaN unless status is "ok")
    level_fractions: tuple[float, ...]
    thresholds: tuple[float, ...]
    gammas: tuple[float, ...]  # chain-correlation factors per conditional level (approximate)
    evaluations: int
    samples_per_level: int
    failure_points: FloatArray = field(repr=False)
    invalid_seen: int = 0  # evaluations that were outside the validity of a solution

    @property
    def cov_independent_levels(self) -> float:
        """Approximate single-run CoV assuming independent levels (not a bound)."""
        return float(np.sqrt(np.sum(self._level_cov2())))

    @property
    def cov_fully_correlated(self) -> float:
        """Approximate single-run CoV assuming fully correlated levels (not a bound)."""
        return float(np.sum(np.sqrt(self._level_cov2())))

    def _level_cov2(self) -> FloatArray:
        p = np.asarray(self.level_fractions, dtype=float)
        gamma = np.concatenate([[0.0], np.asarray(self.gammas, dtype=float)])[: p.size]
        return np.asarray((1.0 - p) / (p * self.samples_per_level) * (1.0 + gamma), dtype=float)


def _chain_gamma(indicator: BoolArray, chain_index: NDArray[np.intp]) -> float:
    """Chain-correlation factor from the autocorrelation of an indicator along Markov chains.

    ``chain_index[c, k]`` is the position in ``indicator`` of state k of chain c, or -1 past the
    end of a shorter chain."""
    present = chain_index >= 0
    values = np.where(present, indicator[np.clip(chain_index, 0, None)], False).astype(float)
    total = int(np.sum(present))
    p = float(np.sum(values) / total)
    if chain_index.shape[1] < 2 or p in (0.0, 1.0):
        return 0.0
    var = p * (1.0 - p)
    length = chain_index.shape[1]
    gamma = 0.0
    for lag in range(1, length):
        pairs = present[:, :-lag] & present[:, lag:]
        count = int(np.sum(pairs))
        if count == 0:
            break
        r = float(np.sum(values[:, :-lag] * values[:, lag:] * pairs) / count) - p * p
        gamma += 2.0 * (1.0 - lag / length) * r / var
    return max(gamma, 0.0)

def subset_simulation(
    limit_state: LimitState,
    dimension: int,
    seed: int,
    samples_per_level: int = 5000,
    p0: float = 0.1,
    max_levels: int = 20,
    fixed_correlation: float | None = None,
) -> SubsetRun:
    """Subset simulation with order-statistic thresholds and counted level fractions (theory manual, section 4.4.1).

    Stops with status "plateau" when more than half of a level's samples share the threshold
    value: the driving variable then carries no information and no estimate is given.
    ``fixed_correlation`` fixes the proposal correlation (for testing strongly dependent chains).
    """
    n = samples_per_level
    n_seed_target = int(round(p0 * n))
    if n_seed_target < 2 or n % n_seed_target != 0:
        raise ValueError("samples_per_level * p0 must be an integer of at least 2 that divides samples_per_level")
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n, dimension))
    ev = limit_state(u)
    y = consistent_driving_variable(ev.y, ev.fail)
    fail = ev.fail
    evaluations = n
    fractions: list[float] = []
    thresholds: list[float] = []
    gammas: list[float] = []
    lam = 0.6
    chain_index: NDArray[np.intp] | None = None  # (n_chains, max length) positions in the level, -1 if absent
    invalid_seen = 0 if ev.invalid is None else int(np.sum(ev.invalid))

    def finish(status: Literal["ok", "plateau", "level cap"], pf: float) -> SubsetRun:
        return SubsetRun(
            status, pf, tuple(fractions), tuple(thresholds), tuple(gammas), evaluations, n, u[fail], invalid_seen
        )



    for _level in range(max_levels + 1):
        n_fail = int(np.sum(fail))
        if n_fail >= n_seed_target:
            if chain_index is not None:
                gammas.append(_chain_gamma(fail, chain_index))
            fractions.append(n_fail / n)
            return finish("ok", float(np.prod(fractions)))
        if _level == max_levels:
            break
        threshold = float(np.partition(y, n_seed_target - 1)[n_seed_target - 1])
        below = y <= threshold
        fraction = float(np.mean(below))
        if fraction > 0.5:
            return finish("plateau", float("nan"))
        if chain_index is not None:
            gammas.append(_chain_gamma(below, chain_index))
        fractions.append(fraction)
        thresholds.append(threshold)

        seeds_u, seeds_y, seeds_fail = u[below], y[below], fail[below]
        n_seeds = seeds_u.shape[0]
        lengths = np.full(n_seeds, n // n_seeds)
        extra = n - int(lengths.sum())
        if extra:
            lengths[rng.choice(n_seeds, size=extra, replace=False)] += 1
        chain_index = np.full((n_seeds, int(lengths.max())), -1, dtype=np.intp)
        chain_index[:, 0] = np.arange(n_seeds)
        position = n_seeds

        sigma_hat = np.maximum(seeds_u.std(axis=0), 1e-6)
        new_u, new_y, new_fail = [seeds_u], [seeds_y], [seeds_fail]
        owner = [np.arange(n_seeds)]
        cur_u, cur_y, cur_fail = seeds_u.copy(), seeds_y.copy(), seeds_fail.copy()
        for step in range(1, int(lengths.max())):
            active = lengths > step
            idx = np.nonzero(active)[0]
            if fixed_correlation is None:
                sigma = np.minimum(1.0, lam * sigma_hat)
                rho = np.sqrt(1.0 - sigma**2)
            else:
                rho = np.full(dimension, fixed_correlation)
                sigma = np.sqrt(1.0 - rho**2)
            cand = rho * cur_u[idx] + sigma * rng.standard_normal((idx.size, dimension))
            cev = limit_state(cand)
            cand_y = consistent_driving_variable(cev.y, cev.fail)
            evaluations += idx.size
            if cev.invalid is not None:
                invalid_seen += int(np.sum(cev.invalid))
            accept = cand_y <= threshold
            moved = idx[accept]
            cur_u[moved], cur_y[moved], cur_fail[moved] = cand[accept], cand_y[accept], cev.fail[accept]
            new_u.append(cur_u[idx].copy())
            new_y.append(cur_y[idx].copy())
            new_fail.append(cur_fail[idx].copy())
            owner.append(idx)
            chain_index[idx, step] = position + np.arange(idx.size)
            position += idx.size
            if fixed_correlation is None:  # adapt the proposal spread toward an acceptance rate of 0.44
                lam = float(np.clip(lam * np.exp((float(np.mean(accept)) - 0.44) / np.sqrt(step)), 1e-3, 1e3))
        u, y, fail = np.vstack(new_u), np.concatenate(new_y), np.concatenate(new_fail)

    return finish("level cap", float("nan"))


@dataclass(frozen=True)
class SubsetResult:
    status: Literal["ok", "plateau", "level cap", "replicates inconsistent"]
    pf: float  # arithmetic mean of replicate estimates (NaN if not "ok" or "replicates inconsistent")
    interval: Interval | None  # approximate; None when withheld
    replicate_estimates: tuple[float, ...]
    runs: tuple[SubsetRun, ...] = field(repr=False)

    @property
    def evaluations(self) -> int:
        return sum(r.evaluations for r in self.runs)


def replicate_interval(estimates: FloatArray, confidence: float = 0.99) -> Interval | None:
    """Approximate interval for the mean of independent replicate estimates (theory manual, section 4.4).

    m * exp(+- t s / (m sqrt(n))). Returns None (interval withheld) if any estimate is zero or
    not finite, or if the replicate coefficient of variation exceeds 1.
    """
    x = np.asarray(estimates, dtype=float)
    if x.size < 2 or not np.all(np.isfinite(x)) or np.any(x <= 0.0):
        return None
    m, s = float(np.mean(x)), float(np.std(x, ddof=1))
    if s / m > 1.0 or s == 0.0:
        return None  # inconsistent replicates, or identical ones (no information on the spread)
    t = float(stats.t.ppf(0.5 + confidence / 2.0, x.size - 1))
    half = t * s / (m * np.sqrt(x.size))
    kind = "approximate (subset-simulation replicates)"
    return Interval(m * float(np.exp(-half)), min(m * float(np.exp(half)), 1.0), confidence, kind)


def subset_replicates(
    limit_state: LimitState,
    dimension: int,
    seed: int,
    replicates: int = 8,
    samples_per_level: int = 5000,
    p0: float = 0.1,
    max_levels: int = 20,
    fixed_correlation: float | None = None,
) -> SubsetResult:
    seeds = np.random.SeedSequence(seed).spawn(replicates)
    runs = tuple(
        subset_simulation(
            limit_state, dimension, int(s.generate_state(1)[0]), samples_per_level, p0, max_levels, fixed_correlation
        )
        for s in seeds
    )
    for bad in ("plateau", "level cap"):
        if any(r.status == bad for r in runs):
            return SubsetResult(bad, float("nan"), None, tuple(r.pf for r in runs), runs)
    estimates = np.array([r.pf for r in runs])
    interval = replicate_interval(estimates)
    status: Literal["ok", "replicates inconsistent"] = "ok" if interval is not None else "replicates inconsistent"
    return SubsetResult(status, float(np.mean(estimates)), interval, tuple(estimates.tolist()), runs)


# --------------------------------------------------------------------------------------
# Combined result
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReliabilityResult:
    """Failure probability with a guaranteed and, where needed, a conditional statement."""

    status: Status
    pf: float  # headline value
    guaranteed: Interval  # Clopper-Pearson from the Monte Carlo channel; assumes nothing
    monte_carlo: MonteCarloResult = field(repr=False)
    subset: SubsetResult | None = field(repr=False, default=None)

    @property
    def conditional_estimate(self) -> float | None:
        if self.subset is None or self.subset.status != "ok" or not np.isfinite(self.subset.pf):
            return None
        return self.subset.pf

    def summary(self) -> str:
        g = self.guaranteed
        lines = [f"Status: {self.status}"]
        mc = self.monte_carlo
        lines.append(
            f"Guaranteed (plain Monte Carlo, {mc.samples:,} samples, {mc.failures} failures): "
            f"{int(round(g.confidence * 100))}% interval [{g.lower:.3g}, {g.upper:.3g}]"
        )
        if self.status == "resolved":
            lines.append(f"Failure probability: {self.pf:.3g}")
        elif self.status == "conditional":
            sub = self.subset
            assert sub is not None
            approx = (
                f", approximate interval [{sub.interval.lower:.3g}, {sub.interval.upper:.3g}]"
                if sub.interval is not None
                else ", no interval (replicates inconsistent)"
            )
            lines.append(f"Rare-event estimate (subset simulation, conditional): {sub.pf:.3g}{approx}")
            lines.append(
                "The rare-event estimate is valid only if no failure region was missed, which cannot be verified "
                f"from the samples. Values up to the guaranteed upper bound {g.upper:.3g} cannot be excluded."
            )
        elif self.status == "contradicted":
            sub = self.subset
            assert sub is not None
            lines.append(
                f"The rare-event estimate {sub.pf:.3g} lies outside the guaranteed interval and is rejected. "
                "Use the guaranteed interval."
            )
        elif self.status == "replicates inconsistent":
            sub = self.subset
            assert sub is not None
            spread = ", ".join(f"{v:.2g}" for v in sub.replicate_estimates)
            lines.append(
                f"Subset-simulation replicates disagree too much to give an estimate ({spread}). "
                "Only the guaranteed interval is available."
            )
        else:
            lines.append(
                f"Subset simulation gave no estimate ({self.status}). Only the guaranteed interval is available."
            )
        return "\n".join(lines)


def solve(
    limit_state: LimitState,
    dimension: int,
    seed: int = 0,
    n_explore: int = 1_000_000,
    replicates: int = 8,
    samples_per_level: int = 5000,
) -> ReliabilityResult:
    """Monte Carlo first; subset simulation only if Monte Carlo saw fewer than 10 failures (section 0)."""
    children = np.random.SeedSequence(seed).spawn(2)
    mc = monte_carlo(limit_state, dimension, n_explore, int(children[0].generate_state(1)[0]))
    if not mc.few_failures:
        return ReliabilityResult("resolved", mc.pf, mc.interval, mc)
    sub_seed = int(children[1].generate_state(1)[0])
    sub = subset_replicates(limit_state, dimension, sub_seed, replicates, samples_per_level)
    if sub.status == "plateau":
        return ReliabilityResult("plateau", mc.pf, mc.interval, mc, sub)
    if sub.status == "level cap":
        return ReliabilityResult("level cap reached; unresolved", mc.pf, mc.interval, mc, sub)
    if sub.status == "replicates inconsistent":
        # no conditional estimate is given: only the guaranteed interval stands
        return ReliabilityResult("replicates inconsistent", mc.pf, mc.interval, mc, sub)
    if not mc.interval.contains(sub.pf):
        return ReliabilityResult("contradicted", mc.pf, mc.interval, mc, sub)
    return ReliabilityResult("conditional", sub.pf, mc.interval, mc, sub)
