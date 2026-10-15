"""Active learning of the emulator for Pf (theory manual, section 7.6).

Each iteration: train the emulator on the curves so far; draw D coherent functions; estimate Pf over the fixed S
populations (crossed); stop when the sampling CoV of ``Pf_high`` is below its bound **and then** the 90 % surrogate
interval is narrower than ``width_ratio * Pf_high`` (an absolute tolerance near zero; zero spread never counts);
otherwise run the oracle (an FE case, or the synthetic 'FE' of V10) at the next input: the U criterion is a
**ranking proxy** computed from the coherent draws — over the band realisations, ``m (1 - m) w`` with ``m`` the
fraction of draws classifying a realisation as failed — and a declared share of the calls (default 15 %) goes to
exploration: the nominal-density realisation whose GP posterior variance is the largest fraction of the prior
variance. The **explored mass** reports the physical probability of band realisations where the
posterior variance still exceeds 0.9 x the prior variance."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from critflaw.option3 import emulator as EM
from critflaw.option3 import reliability as R
from critflaw.option3.curve import Option3Curve

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class StoppingRule:
    sampling_cov: float = 0.10
    width_ratio: float = 0.5
    abs_tol: float = 1e-9
    budget: int = 20  # oracle calls


DEFAULT_RULE = StoppingRule()


@dataclass
class Iteration:
    n_curves: int
    pf: float
    pf_low: float
    interval: tuple[float, float]
    sampling_cov: float
    draw_component: float
    unexplored_mass: float
    undecided_mass: dict[str, float]
    added: str  # "acquisition" | "exploration" | "stopped" | "budget"


@dataclass
class ActiveResult:
    history: list[Iteration] = field(default_factory=list)
    converged: bool = False
    final: R.PfResult | None = None
    curves: list[Option3Curve] = field(default_factory=list)  # the training curves at the end


def unexplored_mass(
    em: EM.Emulator, pop: R.Population, ratio: float = 0.9, fd: EM.FunctionDraws | None = None, convex: bool = True
) -> float:
    """Physical probability (weighted) of the **failure-relevant** band realisations — those that at least one draw
    classifies failed or undecided (otherwise the bulk safe mass would otherwise dominate) — where the GP's posterior
    variance at the middle supported step exceeds ``ratio`` x its prior variance."""
    band = np.flatnonzero(~pop.always_failed & ~pop.invalid_point)
    if band.size == 0:
        return 0.0
    if fd is not None:
        relevant = np.zeros(band.size, dtype=bool)
        for i in range(min(fd.n_draws, 8)):
            lo, hi, st = R.batch_bounds(fd, i, pop.x[band], pop.lr[band], convex)
            relevant |= (st != "supported") | (pop.kr[band] >= lo)
        band = band[relevant]
    mass = 0.0
    prior_var = em.gp.variance * em.gp.y_scale**2
    for j in band:
        x = pop.x[j]
        if np.any(x < em.lower) or np.any(x > em.upper):
            continue
        k = max(2, em.support(x).size // 2)
        _m, v = em.gp.predict(em.gp_input(x, np.array([k])))
        if v[0] > ratio * prior_var:
            mass += pop.weight[j]
    return float(mass / pop.u.shape[0])


def sampling_cov(res: R.PfResult) -> float:
    """The CoV of the population-averaged Pf_high from the within-population weighted variance (not from the
    spread of a few population means); a zero standard error never counts (inf)."""
    if res.pf <= 0.0 or not np.isfinite(res.sampling_se) or res.sampling_se <= 0.0:
        return float("inf")
    return float(res.sampling_se / res.pf)


def run(
    problem: R.Problem,
    dim: int,
    oracle: Callable[[FloatArray], Option3Curve],
    curves: list[Option3Curve],
    lower: FloatArray,
    upper: FloatArray,
    schedule_lr: Callable[[FloatArray], FloatArray],
    design_point: FloatArray,
    rng: np.random.Generator,
    n_draws: int = 24,
    n_pops: int = 3,
    pop_size: int = 4000,
    explore_share: float = 0.15,
    rule: StoppingRule = DEFAULT_RULE,
    prior: Callable[[FloatArray, FloatArray], FloatArray] | None = None,
    n_features: int = 1000,
    experimental: bool = False,
    convex: bool = True,
) -> ActiveResult:
    if not experimental:
        raise PermissionError("active learning uses the experimental Option 3 emulator: pass experimental=True")
    pops = [R.population(problem, dim, pop_size, design_point, rng) for _ in range(n_pops)]
    nominal = R.population(problem, dim, 500, np.zeros(dim), rng, alpha=1.0)
    curves = list(curves)
    out = ActiveResult()
    calls = 0
    n_explored = 0
    while True:
        em = EM.train(curves, lower, upper, schedule_lr, n_starts=3, prior=prior)
        fd = EM.function_draws(em, n_draws, rng, n_features)
        res = R.pf_crossed(fd, pops, experimental=True, convex=convex)
        cov = sampling_cov(res)
        # the stopping width from the interaction-adjusted draw component (90 % normal width), not the raw
        # percentile interval (kept for reporting); zero estimated variance never counts as convergence
        width = 2.0 * 1.645 * float(np.sqrt(res.draw_component))
        band_empty = not any(np.any(~p.always_failed & ~p.invalid_point) for p in pops)
        unexp = unexplored_mass(em, pops[0], fd=fd, convex=convex)
        converged = (
            cov <= rule.sampling_cov
            and res.pf > rule.abs_tol
            and width > 0.0  # zero spread never counts as convergence
            and width <= rule.width_ratio * res.pf
        )
        # every realisation decided without the emulator: Pf does not depend on the draws (terminal, flagged)
        status = (
            "draw-independent"
            if band_empty
            else ("stopped" if converged else ("budget" if calls >= rule.budget else ""))
        )
        it = Iteration(len(curves), res.pf, float(res.pf_low.mean()), res.interval, cov, res.draw_component, unexp,
                       res.undecided_mass, status)  # fmt: skip
        out.history.append(it)
        if status:
            out.converged = converged
            out.final = res
            out.curves = curves
            return out
        explore = explore_share > 0 and n_explored < explore_share * (calls + 1) - 1e-12
        x_next = _explore(em, nominal) if explore else _acquire(fd, pops[0], convex)
        curves.append(oracle(x_next))
        it.added = "exploration" if explore else "acquisition"
        n_explored += int(explore)
        calls += 1


def _acquire(fd: EM.FunctionDraws, pop: R.Population, convex: bool = True) -> FloatArray:
    band = np.flatnonzero(~pop.always_failed & ~pop.invalid_point)
    if band.size == 0:
        raise ValueError("no band realisation to acquire at")  # guarded by the draw-independent stop
    fails = np.zeros(band.size)
    sup_any = np.zeros(band.size, dtype=bool)
    for i in range(fd.n_draws):
        lo, hi, st = R.batch_bounds(fd, i, pop.x[band], pop.lr[band], convex)
        sup = st == "supported"
        mid = 0.5 * (lo + hi)
        fails += sup & (pop.kr[band] >= mid)
        sup_any |= sup
    m = fails / fd.n_draws
    score = np.where(sup_any, m * (1.0 - m) * pop.weight[band], -1.0)
    return np.asarray(pop.x[band[int(np.argmax(score))]], dtype=float)


def _explore(em: EM.Emulator, nominal: R.Population) -> FloatArray:
    prior_var = em.gp.variance * em.gp.y_scale**2
    best, best_ratio = nominal.x[0], -1.0
    for x in nominal.x:
        if np.any(x < em.lower) or np.any(x > em.upper):
            continue
        k = max(2, em.support(x).size // 2)
        _m, v = em.gp.predict(em.gp_input(x, np.array([k])))
        if v[0] / prior_var > best_ratio:
            best, best_ratio = x, float(v[0] / prior_var)
    return np.asarray(best, dtype=float)


__all__ = ["run", "StoppingRule", "ActiveResult", "Iteration", "unexplored_mass", "sampling_cov"]
