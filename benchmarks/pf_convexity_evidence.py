"""Evidence for the convexity assumption of the between-step bounds (design note phase 6, section 1 [impl]): for
every accepted Phase 5 FE curve, the secant slopes of J/J_e over Lr are non-decreasing wherever J/J_e - 1 >= 1e-3
(the floor of `emulator.step_bounds`). Reports each curve and every violation with its position. Pure
post-processing of the stored campaign results. Writes fe/runs_p5/convexity_evidence.json.

    uv run python benchmarks/pf_convexity_evidence.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from critflaw.option3.curve import CurveQAError, from_summary
from critflaw.option3.emulator import CONVEX_FLOOR

RESULTS = Path("fe/runs_p5/results")


def main() -> None:
    rows = []
    for f in sorted(RESULTS.glob("train_L6_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        try:
            c = from_summary(d)
        except CurveQAError:
            continue
        lr = np.asarray(d["lr"], dtype=float)
        r = np.asarray(d["kr"], dtype=float)[:, -1] ** -2.0
        s = np.diff(r) / np.diff(lr)
        viol = []
        for k in range(1, s.size):
            above = r[k - 1] - 1.0 >= CONVEX_FLOOR  # both segments start above the floor
            if above and s[k] < s[k - 1] * (1.0 - 1e-9):
                viol.append({"lr": float(lr[k]), "slopes": [float(s[k - 1]), float(s[k])]})
        rows.append({"case_id": c.case_id, "x": c.x.tolist(), "n_steps": int(lr.size), "violations_above_floor": viol})
    n_ok = sum(not r["violations_above_floor"] for r in rows)
    out = {"floor": CONVEX_FLOOR, "n_curves": len(rows), "n_convex_above_floor": n_ok, "curves": rows}
    Path("fe/runs_p5/convexity_evidence.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{n_ok} of {len(rows)} accepted FE curves convex above J/J_e - 1 = {CONVEX_FLOOR:g}")
    for r in rows:
        if r["violations_above_floor"]:
            print(r["case_id"], r["violations_above_floor"])


if __name__ == "__main__":
    main()
