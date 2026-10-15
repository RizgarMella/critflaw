"""Phase 5 FE campaign (design note phase 5, section 5): the training design (48 level-1 cases, 12 level-2 pairs),
the 12 held-out cases and the c/W check pair, solved one at a time through ``WarpJSolver`` with the solve cache
(restartable after memory kills: the cache's explicit retry policy re-solves killed attempts). Each result is
written as a JSON summary to fe/runs_p5/results/<role>_<case id>.json; a failed case is recorded, not skipped.

    uv run python benchmarks/option3_campaign.py [train|heldout|check|all]
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from critflaw.fe.fecase import WarpJSolver, result_summary
from critflaw.fe.runner import SolveCache, WarpRunner
from critflaw.option3.design import C_OVER_W, DesignPoint, heldout_design, training_design

ROOT = Path("fe/runs_p5")
INFRASTRUCTURE = ("did not complete: failed", "did not complete: timeout")  # with exit 137 / docker errors


def points(which: str) -> list[tuple[DesignPoint, float]]:
    out: list[tuple[DesignPoint, float]] = []
    if which in ("train", "all"):
        out += [(p, C_OVER_W) for p in training_design()]
    if which in ("heldout", "all"):
        out += [(p, C_OVER_W) for p in heldout_design()]
    if which in ("check", "all"):
        mid = heldout_design(1)[0]
        mid = DesignPoint(mid.x.copy(), "check", mid.level)
        out += [(mid, C_OVER_W), (mid, 0.025)]
    return out


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_lock() -> Path:
    """One campaign at a time: a killed shell can leave its python process running, and two solvers at once
    exhaust the host's memory. A live holder makes a second instance exit; a stale lock is taken over."""
    lock = ROOT / "campaign.lock"
    ROOT.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        try:
            holder = int(lock.read_text(encoding="utf-8").strip())
        except ValueError:
            holder = -1
        if holder > 0 and holder != os.getpid() and _pid_alive(holder):
            sys.exit(f"another campaign (pid {holder}) is running")
    lock.write_text(str(os.getpid()), encoding="utf-8")
    return lock


def resolve_with_retry(runner: WarpRunner, attempts: int = 10, wait: float = 30.0) -> None:
    """The Docker daemon restarts under memory pressure; its image lookup then fails for a while."""
    for i in range(attempts):
        try:
            runner.resolve()
            return
        except RuntimeError:
            if i == attempts - 1:
                raise
            time.sleep(wait)


def main(which: str) -> None:
    lock = acquire_lock()
    runner = WarpRunner(threads=4, memory="4g", timeout_seconds=3600)  # lighter after host-memory kills
    resolve_with_retry(runner)
    solver = WarpJSolver(ROOT, runner, SolveCache(ROOT / "cache", retry_incomplete=True))
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    for p, c_w in points(which):
        case = p.case(c_w)
        tag = f"{p.role}_L{p.level[0]}_cW{c_w:.3f}_{case.case_id}"
        dest = results / f"{tag}.json"
        if dest.exists():
            continue
        t0 = time.time()
        res = solver.run(case)
        if res.status == "failed" and any(code in (res.failing_check or "") for code in INFRASTRUCTURE):
            # a host/container kill is not a property of the case: not recorded, retried on the next pass
            print(f"{tag}: infrastructure failure ({res.failing_check}), not recorded", flush=True)
            continue
        summary = result_summary(res)
        summary["design"] = {"role": p.role, "x": p.x.tolist(), "c_over_W": c_w}
        dest.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"{tag}: {res.status} {res.verification_status} {res.failing_check} {time.time() - t0:.0f}s", flush=True)
    lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
