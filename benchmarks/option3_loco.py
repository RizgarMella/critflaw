"""Phase 5: leave-one-curve-out (LOCO) accuracy of the emulator on the campaign data collected so far, with and
without the Option 2 physics prior (design note phase 5, sections 3 and 6). Each level-1 training curve is left
out in turn, the emulator refitted on the rest and the left-out Kr curve predicted; reports the Kr RMSE, the
curve-level coverage of the 95 % intervals and the energy score. Pure post-processing (no solver runs).

    uv run python benchmarks/option3_loco.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from critflaw.option3 import emulator as EM
from critflaw.option3.curve import CurveQAError, from_summary
from critflaw.option3.design import LOWER, UPPER, schedule_lr
from critflaw.option3.emulator import COVERAGE_RESOLUTION
from critflaw.option3.prior import option2_log_increments
from critflaw.option3.validation import energy_score

RESULTS = Path("fe/runs_p5/results")


def load(role: str = "train", level: int = 6) -> list:  # type: ignore[type-arg]
    curves = []
    for f in sorted(RESULTS.glob(f"{role}_L{level}_*.json")):
        try:
            curves.append(from_summary(json.loads(f.read_text(encoding="utf-8"))))
        except CurveQAError as exc:
            print(f"{f.name}: excluded ({exc})")
    return curves


def loco(curves: list, prior: object) -> dict[str, float]:  # type: ignore[type-arg]
    errs, cov, es = [], [], []
    rng = np.random.default_rng(0)
    for i, c in enumerate(curves):
        rest = curves[:i] + curves[i + 1 :]
        em = EM.train(rest, LOWER, UPPER, schedule_lr, n_starts=3, prior=prior)  # type: ignore[arg-type]
        p = em.predict(c.x, 600, rng)
        k = min(p.lr.size, c.kr.size)
        d, o = p.kr_draws[:, :k], c.kr[:k]
        lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
        cov.append(float(np.mean((o >= lo - COVERAGE_RESOLUTION) & (o <= hi + COVERAGE_RESOLUTION))))
        errs.append(d.mean(axis=0) - o)
        es.append(energy_score(d, o))
    e = np.concatenate(errs)
    return {"rmse": float(np.sqrt(np.mean(e**2))), "max_abs": float(np.max(np.abs(e))),
            "coverage": float(np.mean(cov)), "energy": float(np.mean(es))}  # fmt: skip


def main() -> None:
    curves = load()
    print(f"{len(curves)} level-1 training curves")
    out = {
        "n_curves": len(curves),
        "no_prior": loco(curves, None),
        "option2_prior": loco(curves, option2_log_increments),
    }
    print(json.dumps(out, indent=1))
    Path("fe/runs_p5/loco.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
