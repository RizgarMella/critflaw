"""Gate (ii) D3: the elastic design over the box a/B in [0.2, 0.6], a/c in [0.4, 1.0]: 25 tension cases on a
5 x 5 grid at c/W = 0.05, 5 bending cases along the diagonal, and the four geometry corners at c/W = 0.1; each
run in the pinned image with the production (level_1) mesh; per case and front position the signed K_I against
the Phase 1 plate solution at the same parametric angle, the ring spread, and the mesh QA (positive Jacobians,
scaled Jacobian band). Writes fe/runs_gate2/d3_design.txt (the gate report's table).

    uv run python benchmarks/fe_d3_elastic_k.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from critflaw.fe import crack_mesh as CM
from critflaw.fe import mesh as M
from critflaw.fe.elastic_case import ElasticCase, run_case
from critflaw.fe.runner import WarpRunner

B = 20e-3
E, NU = 207e9, 0.3


def main() -> None:
    cases: list[tuple[float, float, float, float]] = []  # (a/B, a/c, beta, c/W)
    for a_b in (0.2, 0.3, 0.4, 0.5, 0.6):
        for a_c in (0.4, 0.55, 0.7, 0.85, 1.0):
            cases.append((a_b, a_c, 0.0, 0.05))
    for a_b, a_c in ((0.2, 0.4), (0.3, 0.55), (0.4, 0.7), (0.5, 0.85), (0.6, 1.0)):
        cases.append((a_b, a_c, 1.0, 0.05))
    for a_b, a_c in ((0.2, 0.4), (0.2, 1.0), (0.6, 0.4), (0.6, 1.0)):  # the c/W = 0.1 edge of the box, four corners
        cases.append((a_b, a_c, 0.0, 0.1))
    runner = WarpRunner(threads=4, memory="6g", timeout_seconds=1800)
    out = Path("fe/runs_gate2/d3_design.txt")
    header = (
        "a_B a_c beta c_W nodes elems min_sJ_ordinary worst_interior_pct at_phi_s_pct deepest_pct "
        "max_spread_pct status wall_s"
    )
    lines = [header]
    for a_b, a_c, beta, c_w in cases:
        a = a_b * B
        c = a / a_c
        w = c / c_w
        pm, pb = (100e6, 0.0) if beta == 0.0 else (0.0, 100e6)
        case = ElasticCase(a=a, c=c, B=B, W=w, L_far=CM.remote_length(B, w), E=E, nu=NU, pm=pm, pb=pb)
        p = case.params()
        cb = CM.build(p)
        m = cb.mesh
        j = M.jacobians(m)
        tube = np.zeros(m.n_elements, bool)
        tube[cb.block_elements[: p.n_rings * p.n_sectors * p.n_front] - 1] = True
        sj = M.scaled_jacobian(m)
        assert np.all(j > 0.0)
        t0 = time.time()
        work = Path(f"fe/runs_gate2/d3_design/aB{a_b:.2f}_ac{a_c:.2f}_b{beta:.0f}_cW{c_w:.2f}")
        try:
            res = run_case(case, work, runner)
            d = res.k_fe[1:] / res.k_ref[1:] - 1.0
            status = "pass" if np.all(np.abs(d) < 0.03) else ("unresolved" if np.all(np.abs(d) < 0.06) else "FAIL")
            lines.append(
                f"{a_b:.2f} {a_c:.2f} {beta:.0f} {c_w:.2f} {m.n_nodes} {m.n_elements} {sj[~tube].min():.3f} "
                f"{float(np.max(np.abs(d))) * 100:.2f} {d[case.n_front_s - 1] * 100:+.2f} {d[-1] * 100:+.2f} "
                f"{float(res.k_spread[1:].max()) * 100:.2f} {status} {time.time() - t0:.0f}"
            )
        except Exception as exc:  # noqa: BLE001 - the report records the failure
            head = f"{a_b:.2f} {a_c:.2f} {beta:.0f} {c_w:.2f} {m.n_nodes} {m.n_elements} {sj[~tube].min():.3f}"
            lines.append(f"{head} - - - - ERROR:{str(exc)[:60]} {time.time() - t0:.0f}")
        print(lines[-1], flush=True)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
