"""Gate (ii) D7b: far-field refinement check at the mid-box design point (design note phase 4, D7b [v4]): the
production level_1 run against (1) level_1 with the far-field divisions doubled (the first far layer halved,
hence every z layer above the block, and the in-plane divisions from the outer offset curve to the plate
edges) and (2) level_1 with L_far doubled. Pass: J changes < 0.5 % at every step and interior position; else
the far field joins the refined region. Writes fe/runs_gate2/d7b_mid.json.

    uv run python benchmarks/fe_d7b_convergence.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fe_d7_convergence import LEVELS, POINTS, result_for  # noqa: E402

from critflaw.fe.plastic_case import PlasticCase  # noqa: E402
from critflaw.fe.runner import WarpRunner  # noqa: E402

BAND = 0.005


def main() -> None:
    runner = WarpRunner(threads=4, memory="6g", timeout_seconds=2900)
    base = PlasticCase(**POINTS["mid"], level=LEVELS[0])  # type: ignore[arg-type]
    ref = result_for(base, Path("fe/runs_gate2/d7q_mid_L6"), runner)
    variants = {
        "far_divisions_doubled": PlasticCase(
            **{**base.__dict__, "far_first": 0.5 * base.far_first, "n_outer": 2 * base.n_outer}
        ),
        "L_far_doubled": PlasticCase(**{**base.__dict__, "L_far": 2.0 * base.L_far}),
    }
    report: dict[str, object] = {"point": "mid", "level": LEVELS[0], "band": BAND, "variants": {}}
    passed = True
    for name, case in variants.items():
        res = result_for(case, Path(f"fe/runs_gate2/d7bq_mid_{name}"), runner)
        assert np.allclose(res.lr, ref.lr, rtol=1e-6)
        rel = res.j[:, 1:] / ref.j[:, 1:] - 1.0  # interior stations
        worst = float(np.max(np.abs(rel)))
        ok = worst < BAND
        passed = passed and ok
        report["variants"][name] = {  # type: ignore[index]
            "worst_rel_change": worst,
            "per_step_max": np.max(np.abs(rel), axis=1).tolist(),
            "pass": ok,
            "solver_s": res.record.wall_seconds,
        }
        print(f"{name}: worst |dJ/J| = {worst * 100:.3f} % -> {'pass' if ok else 'FAIL'}", flush=True)
    report["verdict"] = "pass" if passed else "fail"
    Path("fe/runs_gate2/d7b_mid.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("D7b", report["verdict"])


if __name__ == "__main__":
    main()
