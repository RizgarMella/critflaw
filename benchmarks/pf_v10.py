"""Phase 6 V10 (design note phase 6, section 7; spec S6.5): the whole adaptive loop on a synthetic 'FE' with a
known Pf ~ 1e-5 — the closed-form curve family of the Phase 5 tests (steps, input-dependent support) as the
oracle; the truth by a large importance-sampling run on the true curves. One seed per call (light: one process,
no solver); results appended to benchmarks/out/pf_v10.jsonl. Pass (over 20 seeds): the 90 % surrogate
interval of Pf_high covers the truth in >= 15 of 20; premature stops (converged and not covered) <= 2 of 20;
the RFF feature check (F = 1000 vs 2000) changes Pf by <= 5 % in every seed. Isolated variant: the region is
found (a training curve inside it) or reported unresolved (unexplored mass >= Pf).

    uv run python benchmarks/pf_v10.py <seed> [isolated]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from critflaw.option3 import active as A  # noqa: E402
from critflaw.option3 import emulator as EM  # noqa: E402
from critflaw.option3 import reliability as R  # noqa: E402
from critflaw.option3.curve import Option3Curve  # noqa: E402
from critflaw.option3.design import LOWER, UPPER, sobol_points  # noqa: E402
from test_option3_emulator import schedule, synth_curve  # noqa: E402

DIM = 7
OUT = Path("benchmarks/out/pf_v10.jsonl")
ISO_CENTRE = np.array([0.25, 0.9, 10.5, 1.5e-3, 0.8])  # the isolated region (in x), away from the main failure region
ISO_RADIUS = 0.14  # in unit-box distance: a 5-ball of volume ~1e-4 of the box (CC-02)
ISO_STRENGTH = 6.0  # J/J_e - 1 grows 6x faster inside: region term 6.3e-6, the size of the main Pf (doubles it)


SUB_UPPER = np.array([UPPER[0], UPPER[1], 12.0, UPPER[3], UPPER[4]])  # n <= 12: the support reaches Lr >= 1.12


def to_x(u: np.ndarray) -> np.ndarray:
    return LOWER + stats.norm.cdf(u[:, :5]) * (SUB_UPPER - LOWER)


def problem(u: np.ndarray):  # type: ignore[no-untyped-def]
    x = to_x(u)
    lr = 0.96 * np.exp(0.025 * u[:, 5])  # the failure region sits at Lr ~ 0.97, Kr ~ 0.91: the curve decides
    kr = 0.535 * np.exp(0.11 * u[:, 6])
    return x, lr, kr, np.full(lr.size, 1.5)


def isolated(x: np.ndarray) -> bool:
    return bool(np.linalg.norm((x - ISO_CENTRE) / (SUB_UPPER - LOWER)) < ISO_RADIUS)


def oracle_factory(iso: bool):  # type: ignore[no-untyped-def]
    def oracle(x: np.ndarray) -> Option3Curve:
        c = synth_curve(x)
        if iso and isolated(x):  # a hidden weak region: J/J_e grows ISO_STRENGTH times faster
            return Option3Curve(c.x, c.level, "iso", c.lr, 1.0 + ISO_STRENGTH * (c.j_over_je - 1.0))
        return c

    return oracle


def region_term(rng: np.random.Generator, n: int = 200000) -> float:
    """The isolated region's extra Pf: P(region) x [P(fail | region, weak curve) - P(fail | region, normal curve)].
    x is uniform in the sub-box (to_x maps standard normals through their CDF), so P(region) is the ball's volume
    fraction (inside the box); the conditional probabilities by sampling x uniformly in the ball and the load and
    toughness variables from their nominal laws."""
    lo, hi = LOWER, SUB_UPPER
    centre = (ISO_CENTRE - lo) / (hi - lo)
    pts = []
    while sum(len(p) for p in pts) < n:  # uniform in the 5-ball (unit-box coordinates), kept inside the box
        v = centre + rng.uniform(-ISO_RADIUS, ISO_RADIUS, (4 * n, 5))
        keep = (np.linalg.norm(v - centre, axis=1) < ISO_RADIUS) & np.all((v >= 0) & (v <= 1), axis=1)
        pts.append(v[keep])
    unit = np.vstack(pts)[:n]
    ball_fraction = (8.0 * np.pi**2 / 15.0) * ISO_RADIUS**5 * float(np.mean(np.all((unit >= 0) & (unit <= 1), axis=1)))
    x = lo + unit * (hi - lo)
    u56 = rng.standard_normal((n, 2))
    lr = 0.96 * np.exp(0.025 * u56[:, 0])
    kr = 0.535 * np.exp(0.11 * u56[:, 1])
    weak = oracle_factory(True)
    diff = 0
    for j in range(n):
        c_weak = weak(x[j])
        c_norm = synth_curve(x[j])
        f_w = lr[j] > c_weak.lr[-1] or kr[j] >= np.interp(lr[j], c_weak.lr, c_weak.j_over_je) ** -0.5
        f_n = lr[j] > c_norm.lr[-1] or kr[j] >= np.interp(lr[j], c_norm.lr, c_norm.j_over_je) ** -0.5
        diff += int(f_w) - int(f_n)
    return ball_fraction * diff / n


def truth(iso: bool, rng: np.random.Generator, n: int = 200000) -> float:
    """Pf_high of the true curves (undecided -> failed, as the headline) by importance sampling; with the isolated
    region, its extra contribution is added exactly in probability (region_term)."""
    shift = design_point()
    pop = R.population(problem, DIM, n, shift, rng, alpha=0.3)
    oracle = oracle_factory(False)
    failed = pop.always_failed.copy()
    for j in np.flatnonzero(~failed):
        c = oracle(pop.x[j])
        if pop.lr[j] > c.lr[-1]:
            failed[j] = True
            continue
        failed[j] = pop.kr[j] >= np.interp(pop.lr[j], c.lr, c.j_over_je) ** -0.5
    main = float(np.sum(pop.weight[failed]) / n)
    return main + (region_term(rng) if iso else 0.0)


def design_point() -> np.ndarray:
    """The design point over all seven inputs (min |u| subject to failure of the synthetic curves), multi-start:
    the dominant failures combine tails in x (weak curves) with high load/toughness ratio, so a centre fixed at the
    box's median would miss them (the S5.9 hazard). In an application the proxy is the Option 2 curve."""

    def g(v: np.ndarray) -> float:
        x, lr, kr, _ = problem(v[None, :])
        c = synth_curve(x[0])
        if lr[0] > c.lr[-1]:
            return -1.0
        return float(np.interp(lr[0], c.lr, c.j_over_je) ** -0.5 - kr[0])

    best = None
    for start in (
        np.r_[np.zeros(5), 0.5, 4.4],
        np.r_[-1.0, 0.0, -2.0, 0.0, 0.0, 0.5, 3.0],
        np.r_[1.0, 0.0, -2.5, 1.0, 0.0, 0.5, 2.5],
    ):
        r = optimize.minimize(lambda v: v @ v, start, constraints=[{"type": "ineq", "fun": lambda v: -g(v)}])
        if r.success and g(r.x) <= 1e-6 and (best is None or r.fun < best.fun):
            best = r
    assert best is not None
    return np.asarray(best.x, dtype=float)


def main(seed: int, iso: bool) -> None:
    rng = np.random.default_rng(seed)
    t0 = time.time()
    pf_true = truth(iso, np.random.default_rng(10_000 + int(iso)))
    initial = [oracle_factory(iso)(x) for x in sobol_points(16, seed + 1)]
    res = A.run(problem, DIM, oracle_factory(iso), initial, LOWER, UPPER, schedule, design_point(), rng,
                n_draws=24, n_pops=3, pop_size=3000, rule=A.StoppingRule(budget=14), experimental=True)  # fmt: skip
    last = res.history[-1]
    assert res.final is not None

    def bracket_of(r: R.PfResult) -> tuple[float, float]:
        # the reported uncertainty: validity (undecided -> safe / failed) and surrogate (draws), at 90 %
        return (float(np.percentile(r.pf_low.mean(axis=1), 5.0)), float(np.percentile(r.pf_high.mean(axis=1), 95.0)))

    bracket = bracket_of(res.final)
    # the final emulator re-evaluated: the rigorous monotone bounds, and the RFF feature-count check (F vs 2F)
    em = EM.train(res.curves, LOWER, UPPER, schedule, n_starts=3)
    pops = [R.population(problem, DIM, 3000, design_point(), np.random.default_rng(seed + 500 + k)) for k in range(3)]
    fd1 = EM.function_draws(em, 24, np.random.default_rng(seed + 900), n_features=1000)
    fd2 = EM.function_draws(em, 24, np.random.default_rng(seed + 900), n_features=2000)
    mono = R.pf_crossed(fd1, pops, experimental=True, convex=False)
    f1 = R.pf_crossed(fd1, pops, experimental=True)
    f2 = R.pf_crossed(fd2, pops, experimental=True)
    bracket_mono = bracket_of(mono)
    # the design's criterion (CC-01): the 90 % surrogate interval of Pf_high covers the truth (Pf_high on the true
    # curves is the truth's analogue); the wider validity bracket is recorded as information
    covered = last.interval[0] <= pf_true <= last.interval[1]
    rec = {
        "seed": seed, "isolated": iso, "pf_true": pf_true, "pf_high": last.pf, "pf_low": last.pf_low,
        "interval": last.interval, "bracket": bracket,
        "covered": covered, "covered_bracket": bracket[0] <= pf_true <= bracket[1],
        "premature_stop": bool(res.converged and not covered),
        "bracket_monotone": bracket_mono, "covered_monotone": bracket_mono[0] <= pf_true <= bracket_mono[1],
        "pf_F1000": f1.pf, "pf_F2000": f2.pf, "features_rel_change": abs(f2.pf - f1.pf) / max(f1.pf, 1e-300),
        "bracketed": last.pf_low <= pf_true <= last.pf, "converged": res.converged, "n_curves": last.n_curves,
        "unexplored_mass": last.unexplored_mass, "sampling_cov": last.sampling_cov,
        "iso_found": any(isolated(c.x) for c in res.curves) if iso else None,
        "iso_reported_unresolved": bool(iso and last.unexplored_mass >= pf_true) if iso else None,
        "wall_s": time.time() - t0,
    }  # fmt: skip
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    print(json.dumps(rec, default=float))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0, len(sys.argv) > 2 and sys.argv[2] == "isolated")
