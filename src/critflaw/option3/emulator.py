"""Experimental emulator of the deepest-point Option 3 curve (theory manual, sections 6.3 and 6.4).

One joint GP over ``(x, k)`` (x the five inputs, k the 1-based FE load step) for the increments
``y_k = log(J/J_e(k) - J/J_e(k-1))``; the curve is ``J/J_e(k) = 1 + sum_{i<=k} exp(y_i)``, ``Kr = (J/J_e)^(-1/2)``,
so every draw has ``Kr <= 1`` and is non-increasing in ``Lr`` by construction. The discretisation discrepancy is
the Richardson-scaled level-1-minus-level-2 difference (``R`` sampled, ``d`` a per-step mean with its spread from
the paired cases) plus a one-sided far-field allowance; both enter every draw. Predictions never raise for a
realisation: each carries a status (``supported``, ``outside_box``, ``beyond_support``).

Every prediction is ``verification_status = "unverified"``, ``assessment_grade = False``."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from critflaw.option3 import gp as G
from critflaw.option3.curve import LR_SUPPORT, MAX_STEP, Option3Curve

FloatArray = NDArray[np.float64]

RICHARDSON_RATIO = 4.0 / 3.0  # r_21 of the D7 levels
ORDER_RANGE = (0.9, 1.8)  # the observed deepest-point order of the convergence study; R = r^p / (r^p - 1) in [2.5, 4.4]
FAR_FIELD_MAX = 0.0025  # one-sided relative allowance on J/J_e - 1 (the far-field check)
COVERAGE_RESOLUTION = 1e-4  # Kr resolution of the increment floor: coverage is counted within it (validation)


REGRESSION_MIN_PAIRS = 8  # fewer pairs at a step: intercept only (a linear fit in five inputs needs dof)


@dataclass(frozen=True)
class Discrepancy:
    """Per step k = 2..MAX_STEP, d = y_L1 - y_L2 regressed on the scaled inputs: ``coef[k-2] = (b0, b1..b5)``
    (intercept only where fewer than ``REGRESSION_MIN_PAIRS`` pairs reach the step), ``sd[k-2]`` the residual
    standard deviation (with the degrees of freedom of the fit). Inputs are scaled by ``lower``/``upper``."""

    coef: FloatArray  # (MAX_STEP - 1, 6)
    sd: FloatArray
    n_pairs: int
    lower: FloatArray = field(default_factory=lambda: np.zeros(5))
    upper: FloatArray = field(default_factory=lambda: np.ones(5))
    gram_inv: FloatArray = field(default_factory=lambda: np.zeros((MAX_STEP - 1, 6, 6)))  # (A^T A)^-1 per step

    def leverage_at(self, x: FloatArray, k: NDArray[np.int64]) -> FloatArray:
        """a(x)^T (A^T A)^-1 a(x) per step: the regression mean's variance in units of the residual variance."""
        u = (np.asarray(x, dtype=float) - self.lower) / (self.upper - self.lower)
        a = np.r_[1.0, u]
        return np.asarray(np.einsum("i,kij,j->k", a, self.gram_inv[k - 2], a), dtype=float)

    def mean_at(self, x: FloatArray, k: NDArray[np.int64]) -> FloatArray:
        u = (np.asarray(x, dtype=float) - self.lower) / (self.upper - self.lower)
        c = self.coef[k - 2]
        return np.asarray(c[:, 0] + c[:, 1:] @ u, dtype=float)

    @staticmethod
    def from_pairs(
        pairs: Sequence[tuple[Option3Curve, Option3Curve]], lower: FloatArray, upper: FloatArray
    ) -> Discrepancy:
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        coef = np.zeros((MAX_STEP - 1, 6))
        sd = np.zeros(MAX_STEP - 1)
        gram_inv = np.zeros((MAX_STEP - 1, 6, 6))
        for i in range(MAX_STEP - 1):
            us, ds = [], []
            for l1, l2 in pairs:
                y1, y2 = l1.increments(), l2.increments()
                if i < min(y1.size, y2.size):
                    us.append((l1.x - lower) / (upper - lower))
                    ds.append(float(y1[i] - y2[i]))
            if not ds:
                continue
            d = np.asarray(ds)
            if len(ds) >= REGRESSION_MIN_PAIRS:
                a = np.column_stack([np.ones(len(ds)), np.asarray(us)])
                b, *_ = np.linalg.lstsq(a, d, rcond=None)
                resid = d - a @ b
                dof = max(len(ds) - a.shape[1], 1)
                coef[i] = b
                sd[i] = float(np.sqrt(resid @ resid / dof))
                gram_inv[i] = np.linalg.pinv(a.T @ a)
            else:
                coef[i, 0] = float(d.mean())
                sd[i] = float(d.std(ddof=1)) if len(ds) > 1 else abs(float(d[0]))
                gram_inv[i, 0, 0] = 1.0 / len(ds)  # the mean's variance: sd^2 / m
        return Discrepancy(coef, sd, len(pairs), lower, upper, gram_inv)

    @staticmethod
    def none() -> Discrepancy:
        return Discrepancy(np.zeros((MAX_STEP - 1, 6)), np.zeros(MAX_STEP - 1), 0)


@dataclass(frozen=True)
class Prediction:
    x: FloatArray
    status: str  # supported | outside_box | invalid | cap_uncertain; per lr query also beyond_support
    lr: FloatArray  # supported step Lr values
    kr_mean: FloatArray
    kr_draws: FloatArray  # (n_draws, n_steps)
    schedule_end_lr: float = 0.0  # the support the schedule rule alone would give (beyond lr[-1] if cap-cut)
    verification_status: str = "unverified"
    assessment_grade: bool = False

    def kr_at(self, lr: FloatArray) -> tuple[FloatArray, NDArray[np.str_]]:
        """Kr draws at arbitrary ``lr`` (linear in J/J_e between steps, which keeps monotonicity) and a status per
        query point; beyond the support the draw is NaN and the status ``beyond_support`` (never clipped)."""
        lr = np.atleast_1d(np.asarray(lr, dtype=float))
        if self.status != "supported" or self.lr.size == 0:
            return np.full((self.kr_draws.shape[0], lr.size), np.nan), np.full(lr.size, self.status, dtype="<U16")
        beyond = "cap_uncertain" if self.schedule_end_lr > self.lr[-1] + 1e-12 else "beyond_support"
        status = np.where(
            lr <= self.lr[-1] + 1e-12,
            "supported",
            np.where(lr <= self.schedule_end_lr + 1e-12, beyond, "beyond_support"),
        ).astype("<U16")
        ratio = self.kr_draws**-2.0
        out = np.full((ratio.shape[0], lr.size), np.nan)
        ok = status == "supported"
        for i in range(ratio.shape[0]):
            out[i, ok] = np.interp(lr[ok], self.lr, ratio[i]) ** -0.5
        return out, status


@dataclass
class Emulator:
    lower: FloatArray
    upper: FloatArray
    schedule_lr: Callable[[FloatArray], FloatArray]  # x -> Lr of the 12 FE steps
    gp: G.GPFit
    discrepancy: Discrepancy = field(default_factory=Discrepancy.none)
    far_field_max: float = FAR_FIELD_MAX
    prior: Callable[[FloatArray, FloatArray], FloatArray] | None = None  # (x, step lr) -> prior log increments
    log_inputs: tuple[int, ...] = ()  # input columns entering the GP as logarithms (multiplicative effects)
    cap_strain: float = float("inf")  # the smallest reference strain eps_0 Lr^n at which a training case's cap
    # cut its support (inf when none was cut): a case-specific executable rule for a new input

    def gp_input(self, x: FloatArray, k: NDArray[np.int64]) -> FloatArray:
        """The GP input rows (mapped x, step) for one x at the steps ``k``."""
        return np.column_stack([np.tile(map_inputs(x, self.log_inputs), (k.size, 1)), k.astype(float)])

    def support(self, x: FloatArray) -> FloatArray:
        lr = self.schedule_lr(x)
        keep = (np.arange(lr.size) < MAX_STEP) & (lr <= LR_SUPPORT + 1e-12)
        k_end = int(np.argmin(keep)) if not keep.all() else lr.size
        return lr[:k_end]

    def predict(self, x: FloatArray, n_draws: int = 200, rng: np.random.Generator | None = None) -> Prediction:
        x = np.asarray(x, dtype=float).ravel()
        rng = rng if rng is not None else np.random.default_rng(0)
        empty = np.zeros(0)
        if x.size != self.lower.size or not np.all(np.isfinite(x)):
            return Prediction(x, "invalid", empty, empty, np.zeros((n_draws, 0)))
        if np.any(x < self.lower - 1e-12) or np.any(x > self.upper + 1e-12):
            return Prediction(x, "outside_box", empty, empty, np.zeros((n_draws, 0)))
        lr = self.support(x)
        schedule_end = float(lr[-1])
        # the small-strain cap: a step whose reference strain eps_0 Lr^n reaches the smallest reference strain at
        # which the cap cut a training case is not advertised (the cap is driven by the remote strain level, which
        # the reference strain measures); the kept steps are supported, an Lr query beyond them cap_uncertain
        ref = reference_strain(x, lr)
        if np.any(ref >= self.cap_strain):
            cut = int(np.argmax(ref >= self.cap_strain))
            if cut < 2:  # no increment would remain: the whole curve is cap-uncertain, nothing is advertised
                empty = np.zeros(0)
                return Prediction(x, "cap_uncertain", empty, empty, np.zeros((n_draws, 0)), schedule_end)
            lr = lr[:cut]
        k = np.arange(2, lr.size + 1)
        z = self.gp_input(x, k)
        y = self.gp.draw(z, n_draws, rng)  # (n_draws, K - 1)
        if self.prior is not None:
            y = y + self.prior(x, lr)
        d = self.discrepancy
        p_order = rng.uniform(*ORDER_RANGE, (n_draws, 1))
        rp = RICHARDSON_RATIO**p_order
        r = rp / (rp - 1.0)
        # one mesh pair makes the level difference: fully correlated along the curve; the residual
        # spread widened by the regression's own uncertainty at x, sqrt(1 + leverage)
        common = rng.standard_normal((n_draws, 1))
        dk = d.mean_at(x, k) + d.sd[k - 2] * np.sqrt(1.0 + d.leverage_at(x, k)) * common
        y = y - r * dk  # Richardson: y* = y_L1 + R (y_L2 - y_L1) = y_L1 - R d, d = y_L1 - y_L2
        ratio = 1.0 + np.concatenate([np.zeros((n_draws, 1)), np.cumsum(np.exp(y), axis=1)], axis=1)
        far = rng.uniform(0.0, self.far_field_max, (n_draws, 1))
        ratio = 1.0 + (ratio - 1.0) * (1.0 + far)
        kr = ratio**-0.5
        return Prediction(x, "supported", lr, kr.mean(axis=0), kr, schedule_end)


CONVEX_FLOOR = 1e-3  # J/J_e - 1 above which convexity in Lr is used (all 20 FE curves are convex above it)


def step_bounds(lr: FloatArray, kr: FloatArray, q: float, convex: bool = True) -> tuple[float, float]:
    """Bounds ``(Kr_lower, Kr_upper)`` of a curve known at the steps ``(lr, kr)`` at ``q`` inside the support
    (theory manual, section 7.2). Always valid: monotone, ``Kr_{k+1} <= f(q) <= Kr_k``. Where ``J/J_e = Kr^-2``
    is convex locally (the left step above ``CONVEX_FLOOR`` and non-decreasing secant slopes of the interval and its
    neighbours) the bracket is tightened: the chord bounds ``J/J_e`` from above (``Kr`` from below) and the
    neighbouring secants bound it from below (``Kr`` from above). This is valid **under the assumption** that
    J/J_e is convex in Lr between the steps above the floor (supported by every accepted FE curve);
    the slope check only catches violations visible at the steps,
    it cannot prove convexity between them. ``convex=False`` gives the rigorous monotone bracket."""
    p = int(np.searchsorted(lr, q, side="left"))
    if p >= lr.size:  # within rounding above the last step: the last step's value
        return float(kr[-1]), float(kr[-1])
    if p < lr.size and abs(lr[p] - q) <= 1e-12:
        return float(kr[p]), float(kr[p])
    if p == 0:  # below the first step: J = J_e
        return float(kr[0]), 1.0
    lo, hi = float(kr[p]), float(kr[p - 1])  # monotone
    r = kr**-2.0
    a, b = p - 1, p
    if not convex or r[a] - 1.0 < CONVEX_FLOOR:  # convex=False: the rigorous monotone bracket only
        return lo, hi
    s = np.diff(r) / np.diff(lr)
    s_in = s[a]
    left = s[a - 1] if a - 1 >= 0 and r[a - 1] - 1.0 >= CONVEX_FLOOR else None  # the segment above the floor
    right = s[b] if b < s.size else None
    if (left is not None and left > s_in) or (right is not None and right < s_in):
        return lo, hi  # not convex here: monotone bounds only
    chord = r[a] + s_in * (q - lr[a])
    below = r[a]
    if left is not None:
        below = max(below, r[a] + left * (q - lr[a]))
    if right is not None:
        below = max(below, r[b] - right * (lr[b] - q))
    return max(lo, float(chord**-0.5)), min(hi, float(below**-0.5))


def map_inputs(x: FloatArray, log_inputs: tuple[int, ...]) -> FloatArray:
    """x with the declared columns replaced by their logarithms (the GP's input space)."""
    z = np.array(x, dtype=float)
    for j in log_inputs:
        z[..., j] = np.log(z[..., j])
    return z


def reference_strain(x: FloatArray, lr: FloatArray) -> FloatArray:
    """The law's reference strain eps_0 Lr^n at each Lr (x = (a/B, a/c, n, sigma0/E, beta)): the quantity the
    schedule caps at 2 % and the measure of the remote strain level."""
    return np.asarray(x[3] * np.asarray(lr, dtype=float) ** x[2], dtype=float)


def train(
    curves: Sequence[Option3Curve],
    lower: FloatArray,
    upper: FloatArray,
    schedule_lr: Callable[[FloatArray], FloatArray],
    pairs: Sequence[tuple[Option3Curve, Option3Curve]] = (),
    n_starts: int = 6,
    prior: Callable[[FloatArray, FloatArray], FloatArray] | None = None,
    log_inputs: tuple[int, ...] = (),
) -> Emulator:
    """Fit the joint GP on the increments of the level-1 curves (masked steps are simply absent); with a
    ``prior`` the GP models the increments minus the prior's (``critflaw.option3.prior``)."""
    rows, ys = [], []
    for c in curves:
        y = c.increments()
        if prior is not None:
            y = y - prior(c.x, c.lr)
        for i, yk in enumerate(y):
            rows.append(np.r_[map_inputs(c.x, log_inputs), float(i + 2)])
            ys.append(yk)
    z = np.asarray(rows, dtype=float)
    lo = np.r_[map_inputs(np.asarray(lower, dtype=float), log_inputs), 2.0]
    hi = np.r_[map_inputs(np.asarray(upper, dtype=float), log_inputs), float(MAX_STEP)]
    fit = G.fit(z, np.asarray(ys), lo, hi, n_starts=n_starts)
    disc = Discrepancy.from_pairs(pairs, lower, upper) if pairs else Discrepancy.none()
    em = Emulator(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float), schedule_lr, fit, disc)
    cut = [c for c in curves if c.n_steps < em.support(c.x).size]
    cap = min((float(reference_strain(c.x, em.support(c.x))[c.n_steps]) for c in cut), default=float("inf"))
    return Emulator(em.lower, em.upper, schedule_lr, fit, disc, FAR_FIELD_MAX, prior, log_inputs, cap)


@dataclass(frozen=True)
class FunctionDraws:
    """Coherent function draws of the emulator (theory manual, section 7.3): each draw fixes the GP path
    (pathwise conditioning), the Richardson factor R, the discrepancy's common normal and the far-field factor
    once, so a draw is one curve-valued function of x. At a single x the distribution equals ``predict``'s."""

    emulator: Emulator
    paths: G.PathwiseDraws
    r: FloatArray  # (n_draws,)
    common: FloatArray  # (n_draws,)
    far: FloatArray  # (n_draws,)

    @property
    def n_draws(self) -> int:
        return int(self.r.size)

    def curve(self, i: int, x: FloatArray) -> tuple[FloatArray, FloatArray, str]:
        """(Lr at the supported steps, Kr of draw ``i`` there, status) for one input x."""
        em = self.emulator
        x = np.asarray(x, dtype=float).ravel()
        empty = np.zeros(0)
        if x.size != em.lower.size or not np.all(np.isfinite(x)):
            return empty, empty, "invalid"
        if np.any(x < em.lower - 1e-12) or np.any(x > em.upper + 1e-12):
            return empty, empty, "outside_box"
        lr = em.support(x)
        ref = reference_strain(x, lr)
        if np.any(ref >= em.cap_strain):
            cut = int(np.argmax(ref >= em.cap_strain))
            if cut < 2:
                return empty, empty, "cap_uncertain"
            lr = lr[:cut]
        k = np.arange(2, lr.size + 1)
        z = em.gp_input(x, k)
        y = self.paths.one(i, z)
        if em.prior is not None:
            y = y + em.prior(x, lr)
        d = em.discrepancy
        y = y - self.r[i] * (d.mean_at(x, k) + d.sd[k - 2] * np.sqrt(1.0 + d.leverage_at(x, k)) * self.common[i])
        ratio = 1.0 + np.r_[0.0, np.cumsum(np.exp(y))]
        ratio = 1.0 + (ratio - 1.0) * (1.0 + self.far[i])
        return lr, np.asarray(ratio**-0.5, dtype=float), "supported"

    def bounds(self, i: int, x: FloatArray, lr_query: float) -> tuple[float, float, str]:
        """Monotone bounds of draw ``i``'s curve at ``lr_query`` (theory manual, section 7.2):
        ``(Kr_lower, Kr_upper, status)`` with ``Kr_lower = Kr_{k+1} <= f(Lr) <= Kr_k = Kr_upper``; the status is
        the realisation's (``supported`` inside the support)."""
        lr, kr, status = self.curve(i, x)
        if status != "supported":
            return float("nan"), float("nan"), status
        if lr_query > lr[-1] + 1e-12:
            beyond = (
                "cap_uncertain"
                if self.emulator.support(np.asarray(x, dtype=float)).size > lr.size
                else "beyond_support"
            )
            return float("nan"), float("nan"), beyond
        lo, hi = step_bounds(lr, kr, float(lr_query))
        return lo, hi, "supported"


def function_draws(em: Emulator, n_draws: int, rng: np.random.Generator, n_features: int = 2000) -> FunctionDraws:
    p_order = rng.uniform(*ORDER_RANGE, n_draws)
    rp = RICHARDSON_RATIO**p_order
    return FunctionDraws(
        em, G.pathwise(em.gp, n_draws, rng, n_features), rp / (rp - 1.0), rng.standard_normal(n_draws),
        rng.uniform(0.0, em.far_field_max, n_draws),
    )  # fmt: skip


__all__ = ["Emulator", "Prediction", "Discrepancy", "FunctionDraws", "function_draws", "train"]
