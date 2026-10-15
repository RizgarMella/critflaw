"""Phase 5 V9 validation without further FE (design note phase 5, section 9): leave-one-curve-out over every
accepted training curve, and the two existing D7 level-1 runs (mid-box and corner points) held out; the final
emulator is fitted on all accepted training curves with the Option 2 prior. Pure post-processing of stored
outputs — never launches a solver. Writes fe/runs_p5/v9.json.

    uv run python benchmarks/option3_v9.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from option3_loco import load, loco  # noqa: E402

from critflaw.fe.parser import parse_output  # noqa: E402
from critflaw.fe.plastic_case import PlasticCase, evaluate, write_inputs  # noqa: E402
from critflaw.fe.runner import RunRecord  # noqa: E402
from critflaw.option3 import emulator as EM  # noqa: E402
from critflaw.option3.curve import LR_SUPPORT, MAX_STEP, Option3Curve  # noqa: E402
from critflaw.option3.design import LOWER, UPPER, schedule_lr  # noqa: E402
from critflaw.option3.prior import option2_log_increments  # noqa: E402
from critflaw.option3.validation import openturns_agreement, validate  # noqa: E402

E = 207e9
D7 = {  # the D7 design points (benchmarks/fe_d7_convergence.py POINTS) at the production level, stored outputs only
    "mid": (dict(a=8e-3, c=20e-3, B=20e-3, W=200e-3, L_far=400e-3, E=E, nu=0.3, sigma_0=355e6, n=10.0, beta=0.0),
            Path("fe/runs_gate2/d7q_mid_L6")),
    "corner": (dict(a=12e-3, c=12e-3, B=20e-3, W=240e-3, L_far=480e-3, E=E, nu=0.3, sigma_0=1.035e9, n=20.0, beta=0.0),
               Path("fe/runs_gate2/d7q_corner_L6")),
}  # fmt: skip


def stored_curve(name: str) -> Option3Curve:
    kw, work = D7[name]
    case = PlasticCase(**kw)  # type: ignore[arg-type]
    out, out_e = work / "job" / "out.txt", work / "job_elastic" / "out.txt"
    if not (out.exists() and out_e.exists()):
        raise FileNotFoundError(f"{work}: stored outputs missing (this script never solves)")
    with tempfile.TemporaryDirectory() as tmp:
        cb, _inputs, domains = write_inputs(case, Path(tmp))
    text = out.read_text(encoding="utf-8", errors="replace")
    packets = (work / "job" / "packets.bpf").read_bytes()
    n = len(case.schedule())
    res = evaluate(case, cb, parse_output(text), text, domains, RunRecord("complete", 0, 0.0, n, n, True, (), (), ()),
                   parse_output(out_e.read_text(encoding="utf-8", errors="replace")), packets=packets)  # fmt: skip
    keep = (np.arange(res.lr.size) < MAX_STEP) & (res.lr <= LR_SUPPORT + 1e-12)
    k = int(np.argmin(keep)) if not keep.all() else res.lr.size
    ratio = np.maximum.accumulate(res.kr[:k, -1] ** -2.0)
    x = np.array([kw["a"] / kw["B"], kw["a"] / kw["c"], kw["n"], kw["sigma_0"] / E, kw["beta"]])
    return Option3Curve(x, (6, 12, 12, 3, 3), f"d7_{name}", res.lr[:k], ratio)


def main() -> None:
    curves = load()
    print(f"{len(curves)} accepted training curves", flush=True)
    loco_report = loco(curves, option2_log_increments)
    print("LOCO:", loco_report, flush=True)
    em = EM.train(curves, LOWER, UPPER, schedule_lr, n_starts=4, prior=option2_log_increments)
    held = [stored_curve("mid"), stored_curve("corner")]
    rep = validate(em, held, n_draws=1000)
    agreement = openturns_agreement(em, curves, held)
    out = {
        "n_training_curves": len(curves),
        "loco": loco_report,
        "heldout_d7": rep.summary(),
        "openturns_agreement": agreement,
        "criteria": {"rmse_le": 0.02, "openturns_agreement_ge": 0.9, "support_overruns": 0},
        "verification_status": "unverified",
        "assessment_grade": False,
    }
    Path("fe/runs_p5/v9.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
