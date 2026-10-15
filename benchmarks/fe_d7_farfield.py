"""Gate (ii) D7 far-field check (G2-C-R2-03): levels 1 and 2 at the mid-box point re-run with the D7b-refined far
field (first far layer halved, ``n_outer`` 20) against the production far field; per assessed step and position
the level-1-to-2 difference ``d_12`` on both far fields (unchanged = the level differences are near-field) and the
far-field shift of ``J_1`` (the one-sided term added to the near-field GCI). Writes
fe/runs_gate2/d7_farfield_check.txt.

    uv run python benchmarks/fe_d7_farfield.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fe_d7_convergence import LEVELS, POINTS, result_for, station  # noqa: E402

from critflaw.fe import crack_mesh as CM  # noqa: E402
from critflaw.fe.plastic_case import PlasticCase  # noqa: E402
from critflaw.fe.runner import WarpRunner  # noqa: E402


def main() -> None:
    runner = WarpRunner(threads=4, memory="6g", timeout_seconds=3000)
    base = [result_for(PlasticCase(**POINTS["mid"], level=lv), Path(f"fe/runs_gate2/d7q_mid_L{lv[0]}"), runner)  # type: ignore[arg-type]
            for lv in LEVELS[:2]]  # fmt: skip
    lines = []
    fine = []
    for lv in LEVELS[:2]:
        c = PlasticCase(**POINTS["mid"], level=lv, far_first=0.5, n_outer=20)  # type: ignore[arg-type]
        lines.append(f"{lv} nodes {CM.build(c.params()).mesh.n_nodes}")
        fine.append(result_for(c, Path(f"fe/runs_gate2/d7ff_mid_L{lv[0]}"), runner))
    steps = list(PlasticCase(**POINTS["mid"], level=LEVELS[0]).assessed_steps())  # type: ignore[arg-type]
    for name in ("deepest", "phi_s"):
        ib = [station(r, name) for r in base]
        i_f = [station(r, name) for r in fine]
        for k in steps:
            d12b = base[0].j[k, ib[0]] - base[1].j[k, ib[1]]
            d12f = fine[0].j[k, i_f[0]] - fine[1].j[k, i_f[1]]
            shift = 100 * (fine[0].j[k, i_f[0]] / base[0].j[k, ib[0]] - 1)
            lines.append(
                f"{name} Lr={base[0].lr[k]:.3f} d12 production far field {d12b:+.2f}  refined far field {d12f:+.2f}  "
                f"ratio {d12f / d12b:.3f}  far-field shift L1 {shift:+.3f}%"
            )
    Path("fe/runs_gate2/d7_farfield_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
