"""Probability of failure with the emulator (theory manual, chapter 07).

A problem maps standard-normal samples ``u`` to ``(x, lr, kr, lr_max)``: the emulator inputs, the assessment point
(Lr referred to sigma_0) and the collapse limit. Each realisation is classified in this order (section 7.1):
collapse (``Lr >= Lr,max``, failed, the emulator not queried); ``Kr >= 1`` (failed for every draw, since every
emulated curve has ``Kr <= 1``); otherwise by the draw's monotone between-step bounds (safe below the lower bound,
failed at or above the upper bound, ``between_steps`` in between) or a non-supported status. Undecided
realisations give the bound pair ``Pf_low`` (undecided -> safe) and ``Pf_high`` (undecided -> failed).

Populations: defensive importance sampling, a mixture ``alpha N(0, I) + (1 - alpha) N(u*, I)`` with exact weights,
``u*`` the design point of a cheap proxy limit state (the Option 2 curve of the same law, by the caller). Crossed
design: D function draws x S independent populations; a two-way ANOVA separates the draw (surrogate) component,
the population (sampling) component and the residual (interaction plus within-cell noise).

Every result is ``experimental``: the emulator is trained on unverified FE data at the deepest point only."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from critflaw.option3.curve import MAX_STEP
from critflaw.option3.emulator import FunctionDraws, map_inputs, reference_strain, step_bounds

FloatArray = NDArray[np.float64]
Problem = Callable[[FloatArray], tuple[FloatArray, FloatArray, FloatArray, FloatArray]]  # u -> (x, lr, kr, lr_max)

UNDECIDED = ("outside_box", "invalid", "beyond_support", "cap_uncertain", "between_steps")


def batch_bounds(
    fd: FunctionDraws, i: int, x: FloatArray, lr: FloatArray, convex: bool = True
) -> tuple[FloatArray, FloatArray, NDArray[np.str_]]:
    """Draw ``i``'s monotone bounds ``(Kr_lower, Kr_upper)`` at each realisation's ``lr`` (M,), vectorised over the
    M inputs ``x`` (M, 5); statuses for realisations the emulator does not support."""
    em = fd.emulator
    m = x.shape[0]
    lo_b = np.full(m, np.nan)
    hi_b = np.full(m, np.nan)
    status = np.full(m, "supported", dtype="<U16")
    finite = np.all(np.isfinite(x), axis=1)
    inside = finite & np.all((x >= em.lower - 1e-12) & (x <= em.upper + 1e-12), axis=1)
    status[~finite] = "invalid"
    status[finite & ~inside] = "outside_box"
    idx = np.flatnonzero(inside)
    if idx.size == 0:
        return lo_b, hi_b, status
    # per-realisation step Lr (schedule), support length, cap cut
    lrs = np.full((idx.size, 12), np.nan)
    n_sup = np.zeros(idx.size, dtype=int)
    n_sched = np.zeros(idx.size, dtype=int)
    for j, r in enumerate(idx):
        s = em.support(x[r])
        n_sched[j] = s.size
        ref = reference_strain(x[r], s)
        cut = int(np.argmax(ref >= em.cap_strain)) if np.any(ref >= em.cap_strain) else s.size
        n_sup[j] = cut
        lrs[j, : s.size] = s
    k = np.arange(2, MAX_STEP + 1)
    rows = np.column_stack([np.repeat(map_inputs(x[idx], em.log_inputs), k.size, axis=0), np.tile(k, idx.size)])
    y = fd.paths.one(i, rows).reshape(idx.size, k.size)
    if em.prior is not None:
        for j, r in enumerate(idx):
            s = lrs[j, : n_sched[j]]
            if s.size >= 2:
                y[j, : s.size - 1] += em.prior(x[r], s)
    d = em.discrepancy
    u = (x[idx] - d.lower) / (d.upper - d.lower)
    a = np.column_stack([np.ones(idx.size), u])
    mean = a @ d.coef.T  # (M_in, MAX_STEP - 1)
    lev = np.einsum("mi,kij,mj->mk", a, d.gram_inv, a)
    y = y - fd.r[i] * (mean + d.sd[None, :] * np.sqrt(1.0 + lev) * fd.common[i])
    inc = np.exp(y)
    valid = np.arange(1, MAX_STEP)[None, :] < n_sup[:, None]  # increments 2..K of each realisation's support
    inc = np.where(valid, inc, 0.0)
    ratio = 1.0 + np.concatenate([np.zeros((idx.size, 1)), np.cumsum(inc, axis=1)], axis=1)
    ratio = 1.0 + (ratio - 1.0) * (1.0 + fd.far[i])
    kr = ratio**-0.5  # (M_in, MAX_STEP); columns beyond the support are not used
    for j, r in enumerate(idx):
        if n_sup[j] < 2:
            status[r] = "cap_uncertain"
            continue
        s = lrs[j, : n_sup[j]]
        q = lr[r]
        if q > s[-1] + 1e-12:
            status[r] = "cap_uncertain" if n_sched[j] > n_sup[j] else "beyond_support"
            continue
        lo_b[r], hi_b[r] = step_bounds(s, kr[j, : n_sup[j]], float(q), convex)
    return lo_b, hi_b, status


@dataclass(frozen=True)
class Population:
    u: FloatArray  # (N, dim) samples in standard-normal space
    weight: FloatArray  # density ratio nominal / mixture
    x: FloatArray
    lr: FloatArray
    kr: FloatArray
    lr_max: FloatArray

    @property
    def invalid_point(self) -> NDArray[np.bool_]:
        """Non-finite assessment values (Lr, Kr or Lr,max): undecided (`invalid`), never silently safe."""
        bad = ~(np.isfinite(self.lr) & np.isfinite(self.kr) & np.isfinite(self.lr_max))
        return np.asarray(bad & ~self.collapsed, dtype=bool)

    @property
    def collapsed(self) -> NDArray[np.bool_]:
        with np.errstate(invalid="ignore"):
            return np.asarray(np.isfinite(self.lr) & np.isfinite(self.lr_max) & (self.lr >= self.lr_max), dtype=bool)

    @property
    def always_failed(self) -> NDArray[np.bool_]:
        with np.errstate(invalid="ignore"):
            kr_fail = np.isfinite(self.kr) & (self.kr >= 1.0)
        return np.asarray((self.collapsed | kr_fail) & ~self.invalid_point, dtype=bool)


def population(problem: Problem, dim: int, n: int, design_point: FloatArray, rng: np.random.Generator,
               alpha: float = 0.3) -> Population:  # fmt: skip
    """Defensive importance sampling: a share ``alpha`` from N(0, I), the rest from N(u*, I); exact weights."""
    n0 = int(round(alpha * n))
    alpha = n0 / n  # the realised nominal share of the stratified sample (exact weights)
    u = np.vstack([rng.standard_normal((n0, dim)), design_point + rng.standard_normal((n - n0, dim))])
    log_nom = stats.norm.logpdf(u).sum(axis=1)
    log_shift = stats.norm.logpdf(u - design_point).sum(axis=1)
    if alpha >= 1.0:
        mix = log_nom  # nominal sampling only
    elif alpha <= 0.0:
        mix = log_shift
    else:
        mix = np.logaddexp(np.log(alpha) + log_nom, np.log1p(-alpha) + log_shift)
    weight = np.exp(log_nom - mix)
    x, lr, kr, lr_max = problem(u)
    return Population(u, weight, x, lr, kr, lr_max)


@dataclass(frozen=True)
class PfResult:
    pf_high: FloatArray  # (D, S) per draw and population, undecided -> failed
    pf_low: FloatArray  # (D, S) undecided -> safe
    undecided_mass: dict[str, float]  # by class, averaged over draws and populations
    draw_component: float  # variance of the draw effect (net of the residual), for the stopping rule
    population_component: float
    residual: float
    interval: tuple[float, float]  # the 90 % surrogate-only interval of Pf_high (draw means over populations)
    sampling_se: float = float("nan")  # standard error of the population-averaged Pf_high (within-population)
    labels: tuple[str, ...] = (
        "surrogate-only interval: excludes model uncertainty, distribution-parameter uncertainty, FE-versus-reality,"
        " GP hyperparameter uncertainty",
        "experimental: FE data without family-level verification, deepest point only",
    )

    @property
    def pf(self) -> float:
        return float(self.pf_high.mean())


def pf_crossed(fd: FunctionDraws, pops: list[Population], experimental: bool = False, convex: bool = True) -> PfResult:
    """Pf over D draws x S populations with the bound pair, undecided masses and the two-way ANOVA. The
    emulator is experimental: ``experimental=True`` is required. ``convex=True`` tightens the between-step
    bounds under the stated assumption that J/J_e is convex in Lr above the floor (all 20 FE curves are; checked
    locally per draw); ``convex=False`` gives the rigorous monotone pair. The mode is recorded in the labels."""
    if not experimental:
        raise PermissionError("the Pf uses the experimental Option 3 emulator: pass experimental=True")
    n_d, n_s = fd.n_draws, len(pops)
    high = np.zeros((n_d, n_s))
    low = np.zeros((n_d, n_s))
    var_high = np.zeros((n_d, n_s))  # estimator variance per cell
    masses = {k: 0.0 for k in UNDECIDED}
    for s, pop in enumerate(pops):
        n = pop.u.shape[0]
        fail_all = pop.always_failed
        invalid = pop.invalid_point
        band = np.flatnonzero(~fail_all & ~invalid)
        inv_w = float(np.sum(pop.weight[invalid]))
        base = float(np.sum(pop.weight[fail_all]))
        for i in range(n_d):
            lo_b, hi_b, status = batch_bounds(fd, i, pop.x[band], pop.lr[band], convex)
            kr = pop.kr[band]
            sup = status == "supported"
            failed = sup & (kr >= hi_b)
            between = sup & (kr >= lo_b) & (kr < hi_b)
            w = pop.weight[band]
            undecided = (~sup) | between
            high[i, s] = (base + inv_w + float(np.sum(w[failed | undecided]))) / n
            contrib = np.zeros(n)
            contrib[fail_all | invalid] = pop.weight[fail_all | invalid]
            contrib[band[failed | undecided]] = w[failed | undecided]
            var_high[i, s] = float(np.var(contrib)) / n
            low[i, s] = (base + float(np.sum(w[failed]))) / n
            for k in UNDECIDED:
                cls = between if k == "between_steps" else (status == k)
                masses[k] += float(np.sum(w[cls])) / n / (n_d * n_s)
            masses["invalid"] += inv_w / n / (n_d * n_s)
    grand = high.mean()
    row = high.mean(axis=1)
    col = high.mean(axis=0)
    ms_draw = n_s * float(np.sum((row - grand) ** 2)) / max(n_d - 1, 1)
    ms_pop = n_d * float(np.sum((col - grand) ** 2)) / max(n_s - 1, 1)
    resid = high - row[:, None] - col[None, :] + grand
    ms_res = float(np.sum(resid**2)) / max((n_d - 1) * (n_s - 1), 1)
    draw_c = max((ms_draw - ms_res) / n_s, 0.0)
    pop_c = max((ms_pop - ms_res) / n_d, 0.0)
    lo_q, hi_q = np.percentile(row, [5.0, 95.0])
    mode = (
        "between-step bounds: convex (assumption: J/J_e convex in Lr above 1e-3, as all 20 FE curves; checked per draw)"
        if convex
        else "between-step bounds: monotone (rigorous)"
    )
    se = float(np.mean(np.sqrt(var_high.sum(axis=1)) / n_s))  # SE of the population mean, averaged over draws
    return PfResult(
        high, low, masses, draw_c, pop_c, ms_res, (float(lo_q), float(hi_q)),
        labels=(*PfResult.labels, mode), sampling_se=se,
    )  # fmt: skip


__all__ = ["Population", "PfResult", "population", "pf_crossed", "batch_bounds", "UNDECIDED"]
