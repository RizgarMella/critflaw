"""Gate (ii) D7: nonlinear mesh convergence at the two design points (design note phase 4, D7 [v4]/[v5]): the
four levels, the observed order from levels 1-3 with the non-constant-ratio procedure, the level-4 prediction,
the order stability, the numerical-resolution prerequisite (D9: the solver-tolerance sensitivity measured on
the same levels, 10x tighter residual tolerance) and the production (level_1) error with the coarse-grid GCI,
per step above Lr = 0.8 at the deepest point and at phi_s. Writes fe/runs_gate2/d7_<point>.json.

    uv run python benchmarks/fe_d7_convergence.py [mid|corner] [n_levels: 4 (default) or 3]

Levels (n_rings, n_sectors, n_front, n_front_s, n_blend): (6,12,12,3,3), (8,16,16,4,4), (10,20,20,5,5),
(12,24,24,6,6); exact ratios 4/3, 5/4, 6/5 in every direction (sectors must be a multiple of 4, which the
design's first set (10,15,20,5,5) violated; its (16,24,32,8,8) also exceeded the 90-min threshold).

Prerequisite (design D7 [v5]): abs(J_2 - J_3) and abs(J_3 - J_4) >= 5x the tolerance sensitivity of the levels
involved; if not met, all levels are re-run at the tightened tolerance (their sensitivity measured against a
further 10x) and the test repeated once; if still not met, D7 is unresolved (no fallback). The run of each
level is cached by the runner's solve key through the stored job directories; a level already run is read back.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from critflaw.fe import convergence as CV
from critflaw.fe import crack_mesh as CM
from critflaw.fe.plastic_case import PlasticCase, PlasticResult, run_case
from critflaw.fe.runner import SolveCache, WarpRunner

LEVELS = ((6, 12, 12, 3, 3), (8, 16, 16, 4, 4), (10, 20, 20, 5, 5), (12, 24, 24, 6, 6))
RATIOS = (4.0 / 3.0, 1.25, 1.2)
TOLERANCES = (0.002, 0.0002, 0.00002)  # the production Newton tolerance and its two 10x tightenings
SENSITIVITY_FACTOR = 5.0
POINTS = {
    "mid": dict(a=8e-3, c=20e-3, B=20e-3, W=200e-3, L_far=400e-3, E=207e9, nu=0.3, sigma_0=355e6, n=10.0, beta=0.0),
    "corner": dict(
        a=12e-3, c=12e-3, B=20e-3, W=240e-3, L_far=480e-3, E=207e9, nu=0.3, sigma_0=1.035e9, n=20.0, beta=0.0
    ),
}


# explicit policy (design 2.5 [impl]): a stored attempt killed by the host or timed out is re-solved and superseded,
# the earlier attempt kept under attempts/ — the benchmark chain is restarted after memory kills
CACHE = SolveCache(Path("fe/runs_gate2/cache"), retry_incomplete=True)


def result_for(case: PlasticCase, work: Path, runner: WarpRunner) -> PlasticResult:
    """Solve or read back through the solve cache (keyed by the generated inputs and the execution identity:
    a changed case, far field, tolerance or image is a miss); the stored record and run QA apply."""
    return run_case(case, work, runner, CACHE)


def level_results(point: str, tol_index: int, runner: WarpRunner, levels: tuple[int, ...]) -> list[PlasticResult]:
    base = POINTS[point]
    suffix = "" if tol_index == 0 else f"_tol{tol_index}"
    results = []
    for idx in levels:
        level = LEVELS[idx]
        case = PlasticCase(**base, level=level, norm_res_tol=TOLERANCES[tol_index])  # type: ignore[arg-type]
        t0 = time.time()
        res = result_for(case, Path(f"fe/runs_gate2/d7q_{point}_L{level[0]}{suffix}"), runner)
        rec = res.record
        RUN_LOG.append(
            {
                "level": level,
                "tolerance": TOLERANCES[tol_index],
                "status": rec.status,
                "solver_wall_s": rec.wall_seconds,
                "steps_completed": rec.steps_completed,
                "iterations_per_step": list(res.iterations),
                "nonconverged_steps": list(rec.nonconverged_steps),
            }  # fmt: skip
        )
        print(f"level {level} tol {TOLERANCES[tol_index]}: {time.time() - t0:.0f}s ({rec.status})", flush=True)
        results.append(res)
    return results


RUN_LOG: list[dict[str, object]] = []  # per level and tolerance: status, wall time, iterations (G2-C-05/07)


def station(res: PlasticResult, name: str) -> int:
    """The station index of a named position on a level's own front (the levels have different station
    counts; a node sits exactly at phi_s on every level, design 2.1)."""
    if name == "deepest":
        return int(res.phi.size - 1)
    i = int(np.argmin(np.abs(res.phi - np.radians(10.0))))
    assert abs(res.phi[i] - np.radians(10.0)) < 1e-9
    return i


def assess(
    results: list[PlasticResult], tightened: list[PlasticResult], steps: list[int], positions: tuple[str, ...]
) -> dict[str, list[dict[str, object]]]:
    """Per position and step: the four-level verdict (or, with three levels, the clauses (a), (b) and the GCI
    only: the level-4 prediction (c) and the order stability (d) are not evaluable and the asymptotic test is
    then never passed), the tolerance sensitivities of the levels re-run and the prerequisite
    (abs(J_k - J_k+1) >= 5 max(s_k, s_k+1) for every consecutive pair from level 2)."""
    out: dict[str, list[dict[str, object]]] = {}
    n = len(results)
    for name in positions:
        rows: list[dict[str, object]] = []
        idx = [station(r, name) for r in results]
        idx_t = [station(t, name) for t in tightened]
        for k in steps:
            j = tuple(float(r.j[k, i]) for r, i in zip(results, idx, strict=True))
            s = tuple(abs(float(t.j[k, i]) - j[m + 1]) for m, (t, i) in enumerate(zip(tightened, idx_t, strict=True)))
            prereq = all(
                abs(j[m + 1] - j[m + 2]) >= SENSITIVITY_FACTOR * max(s[m], s[m + 1]) for m in range(len(s) - 1)
            )
            if n == 4:
                v = CV.assess_four_levels((j[0], j[1], j[2], j[3]), RATIOS)
                row = {
                    "p": v.p, "p_234": v.p_234, "monotone": v.monotone, "prediction_within": v.prediction_within,
                    "order_stable": v.order_stable, "order_in_range": v.order_in_range, "asymptotic": v.asymptotic,
                    "level1_error_rel": v.coarse_error_rel,
                }  # fmt: skip
            else:
                d12, d23 = j[0] - j[1], j[1] - j[2]
                p = CV.observed_order(d12, d23, RATIOS[0], RATIOS[1])
                err = CV.coarse_error(j[0], j[1], p, RATIOS[0]) / abs(j[1]) if np.isfinite(p) else float("nan")
                row = {
                    "p": p, "p_234": float("nan"), "monotone": bool(np.sign(d12) == np.sign(d23) != 0),
                    "prediction_within": None, "order_stable": None,
                    "order_in_range": bool(np.isfinite(p) and 0.8 <= p <= 3.0), "asymptotic": False,
                    "level1_error_rel": err,
                }  # fmt: skip
            rows.append(
                {
                    "step": k + 1,
                    "lr": float(results[0].lr[k]),
                    "j": j,
                    "tolerance_sensitivity_from_L2": s,
                    "resolution_prerequisite": prereq,
                    **row,
                }
            )
        out[name] = rows
    return out


def main(point: str, n_levels: int = 4) -> None:
    runner = WarpRunner(threads=4, memory="6g", timeout_seconds=2900)  # the lighter settings of the restarts
    base_levels = tuple(range(n_levels))
    results = level_results(point, 0, runner, base_levels)
    lr = results[0].lr
    steps = list(PlasticCase(**POINTS[point], level=LEVELS[0]).assessed_steps())  # type: ignore[arg-type]
    positions = ("deepest", "phi_s")
    sizes = []
    for level in LEVELS[:n_levels]:
        m = CM.build(PlasticCase(**POINTS[point], level=level).params()).mesh  # type: ignore[arg-type]
        sizes.append((m.n_nodes, m.n_elements))
    report: dict[str, object] = {
        "point": point,
        "levels": LEVELS[:n_levels],
        "level_4_note": None
        if n_levels == 4
        else "level_4 beyond the 90-min threshold on this machine with the "
        "converged far field (fe/runs_gate2/level4_log.txt): clauses (c), (d) not evaluable, D7 cannot pass",
        "ratios": RATIOS,
        "nodes_elements": sizes,
        "lr": lr.tolist(),
        "steps_assessed": steps,
    }
    passes: list[dict[str, object]] = []
    verdict = "unresolved"
    for tol_index in (0, 1):
        if tol_index == 1:  # all levels re-run at the tightened tolerance (levels 2.. are read back)
            results = level_results(point, 1, runner, base_levels)
        tightened = level_results(point, tol_index + 1, runner, base_levels[1:])
        table = assess(results, tightened, steps, positions)
        prereq = all(bool(r["resolution_prerequisite"]) for rows in table.values() for r in rows)
        asym = all(bool(r["asymptotic"]) for rows in table.values() for r in rows)
        err_ok = all(float(r["level1_error_rel"]) <= 0.03 for rows in table.values() for r in rows)  # type: ignore[arg-type]
        passes.append({"tolerance": TOLERANCES[tol_index], "prerequisite": prereq, "positions": table})
        if prereq:
            verdict = "pass" if asym and err_ok else "unresolved"
            break
    report["passes"] = passes
    report["verdict"] = verdict
    report["runs"] = RUN_LOG
    Path(f"fe/runs_gate2/d7_{point}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(point, verdict)
    for name, rows in passes[-1]["positions"].items():  # type: ignore[attr-defined]
        for r in rows:
            head = f"  {name} Lr={r['lr']:.3f} p={r['p']:.2f} p234={r['p_234']:.2f} asym={r['asymptotic']}"
            print(head + f" prereq={r['resolution_prerequisite']} err1={r['level1_error_rel'] * 100:.2f}%")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mid", int(sys.argv[2]) if len(sys.argv) > 2 else 4)
