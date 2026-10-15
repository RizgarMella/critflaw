"""Coverage of the reliability intervals over many seeds (design note phase 2, section 7).

For limit states with exact answers: how often the 99% Clopper-Pearson interval of the Monte
Carlo channel and the approximate replicate interval of subset simulation contain the exact
failure probability. Criteria from the design note: Monte Carlo at least 97.5%, subset
simulation at least 95%, mean of the subset estimates within 4% of exact.

    uv run python benchmarks/prob_interval_coverage.py --seeds 1000 --out benchmarks/out/prob_interval_coverage.json

Cases are run one after another in a single process; results are written after each case, so
an interrupted run keeps what it finished (and skips it when started again).
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy import stats

from critflaw.prob.solvers import Evaluation, monte_carlo, subset_replicates

LimitState = Callable[[np.ndarray], Evaluation]


def linear(beta: float, dim: int) -> LimitState:
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


def hundred_events(u: np.ndarray) -> Evaluation:
    """One event at an index of 3 and 99 at 4 (independent)."""
    y = np.minimum(3.0 - u[:, 0], np.min(4.0 - u[:, 1:], axis=1))
    return Evaluation(y, y <= 0.0)


B3, B6 = float(stats.norm.isf(1e-3)), float(stats.norm.isf(1e-6))

# name -> (kind, limit state, dimension, exact probability, options)
CASES: dict[str, tuple[str, LimitState, int, float, dict[str, float]]] = {
    "1 linear 1e-3, Monte Carlo 200,000": ("mc", linear(B3, 2), 2, 1e-3, {"samples": 200_000}),
    "1 linear 1e-3, subset": ("subset", linear(B3, 2), 2, 1e-3, {}),
    "1 linear 1e-6, subset": ("subset", linear(B6, 2), 2, 1e-6, {}),
    "6 two branches, Monte Carlo 200,000": (
        "mc", two_branch, 2, float(1 - stats.norm.cdf(3.0) * stats.norm.cdf(3.2)), {"samples": 200_000},
    ),
    "6 two branches, subset": ("subset", two_branch, 2, float(1 - stats.norm.cdf(3.0) * stats.norm.cdf(3.2)), {}),
    "7 two-sided |U| > 5, subset": ("subset", two_sided, 1, float(2 * stats.norm.sf(5.0)), {}),
    "15 hundred events, Monte Carlo 200,000": (
        "mc", hundred_events, 100, float(1 - stats.norm.cdf(3.0) * stats.norm.cdf(4.0) ** 99), {"samples": 200_000},
    ),
    "15 hundred events, subset": (
        "subset", hundred_events, 100, float(1 - stats.norm.cdf(3.0) * stats.norm.cdf(4.0) ** 99), {},
    ),
    "16 linear 1e-3, subset, proposal correlation fixed at 0.98": (
        "subset", linear(B3, 2), 2, 1e-3, {"fixed_correlation": 0.98},
    ),
}


def run_case(name: str, seeds: int) -> dict[str, object]:
    kind, g, dim, exact, options = CASES[name]
    started = time.time()
    covered = given = 0
    estimates = []
    statuses: dict[str, int] = {}
    for seed in range(seeds):
        if kind == "mc":
            mc = monte_carlo(g, dim, int(options["samples"]), seed)
            estimates.append(mc.failures / mc.samples)
            given += 1
            covered += mc.interval.contains(exact)
        else:
            res = subset_replicates(g, dim, seed, fixed_correlation=options.get("fixed_correlation"))
            statuses[res.status] = statuses.get(res.status, 0) + 1
            if res.interval is not None:
                given += 1
                covered += res.interval.contains(exact)
                estimates.append(res.pf)
    mean = float(np.mean(estimates)) if estimates else float("nan")
    return {
        "kind": kind,
        "exact": exact,
        "seeds": seeds,
        "intervals_given": given,
        "covered": int(covered),
        "coverage_of_given": covered / given if given else None,
        "mean_estimate_over_exact": mean / exact,
        "statuses": statuses,
        "criterion": 0.975 if kind == "mc" else 0.95,
        "meets_criterion": bool(given and covered / given >= (0.975 if kind == "mc" else 0.95)),
        "seconds": round(time.time() - started, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/out/prob_interval_coverage.json"))
    parser.add_argument("--only", default="", help="run only cases whose name contains this text")
    args = parser.parse_args()
    results: dict[str, object] = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    for name in CASES:
        if args.only not in name:
            continue
        done = results.get(name)
        if isinstance(done, dict) and done.get("seeds") == args.seeds:
            continue
        results[name] = run_case(name, args.seeds)
        args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(name, json.dumps(results[name]), flush=True)


if __name__ == "__main__":
    main()
