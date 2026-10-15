"""The deepest-point Option 3 curve of one FE case (theory manual, section 6.3): ``J/J_e`` at the FE load steps,
its support (step <= 11, Lr <= 1.2, inside the small-strain cap) and the emulated increments
``y_k = log(J/J_e(k) - J/J_e(k-1))`` (k >= 2). Works on ``FEResult`` summaries (``critflaw.fe.fecase.result_summary``)
so that the training database is the record of truth.

Every curve is ``verification_status = "unverified"``, ``assessment_grade = False`` (theory manual, section 6.4.6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

MAX_STEP = 11  # 1-based: the strain-capped last step (12) is not accepted as evidence
LR_SUPPORT = 1.2  # the range of the accepted D7 evidence
INCREMENT_FLOOR = 1e-5  # elastic-range increments of J/J_e below this are set to it before the log
MONOTONE_TOLERANCE = 2e-4  # a decrease of J/J_e larger than this (relative) is a QA failure of the case: the
# elastic-range J is resolved to ~1e-4 relative by the Newton tolerance (the FE runs show dips of 1e-4 at
# J/J_e = 1.0001 -> 1.0000), so smaller dips are noise and are removed by the running maximum


class CurveQAError(ValueError):
    """The FE curve violates a property the emulator relies on (J/J_e non-decreasing, J_e consistent)."""


@dataclass(frozen=True)
class Option3Curve:
    x: FloatArray  # (a/B, a/c, n, sigma0/E, beta)
    level: tuple[int, ...]
    case_id: str
    lr: FloatArray  # the supported steps' Lr (1-based steps 1..K)
    j_over_je: FloatArray  # J/J_e at those steps (1 at step 1)
    verification_status: str = "unverified"
    assessment_grade: bool = False

    @property
    def kr(self) -> FloatArray:
        return np.asarray(self.j_over_je**-0.5, dtype=float)

    @property
    def n_steps(self) -> int:
        return int(self.lr.size)

    def increments(self) -> FloatArray:
        """``y_k`` for k = 2..K (length K - 1)."""
        d = np.diff(self.j_over_je)
        return np.asarray(np.log(np.maximum(d, INCREMENT_FLOOR)), dtype=float)


def from_summary(summary: dict[str, Any]) -> Option3Curve:
    """The curve of one ``result_summary`` record (the deepest point is the last station)."""
    if summary.get("verification_status") == "fe-failed" or summary.get("status") != "complete":
        raise CurveQAError(f"case {summary.get('case_id')}: not a complete, QA-passing FE result")
    lr = np.asarray(summary["lr"], dtype=float)
    kr = np.asarray(summary["kr"], dtype=float)[:, -1]
    usable = ~np.asarray(summary["outside_small_strain"], dtype=bool)
    if lr.ndim != 1 or kr.shape != lr.shape or usable.shape != lr.shape:
        raise CurveQAError(f"case {summary.get('case_id')}: inconsistent array shapes")
    if not (np.all(np.isfinite(kr)) and np.all(kr > 0.0) and np.all(np.isfinite(lr)) and np.all(np.diff(lr) > 0.0)):
        raise CurveQAError(f"case {summary.get('case_id')}: non-finite or non-positive Kr, or Lr not increasing")
    ratio = kr**-2.0
    if abs(ratio[0] - 1.0) > 1e-9:
        raise CurveQAError(f"case {summary.get('case_id')}: J/J_e at step 1 is {ratio[0]}, not 1")
    keep = (np.arange(lr.size) < MAX_STEP) & (lr <= LR_SUPPORT + 1e-12) & usable
    # the support is a prefix: the first excluded step ends it
    k_end = int(np.argmin(keep)) if not keep.all() else lr.size
    if k_end < 2:
        raise CurveQAError(f"case {summary.get('case_id')}: fewer than two supported steps")
    lr, ratio = lr[:k_end], ratio[:k_end]
    if np.any(np.diff(ratio) < -MONOTONE_TOLERANCE * ratio[1:]):
        raise CurveQAError(f"case {summary.get('case_id')}: J/J_e decreases between supported steps")
    ratio = np.maximum.accumulate(ratio)  # sub-tolerance decreases (rounding) removed
    design = summary.get("design")
    if not design or "x" not in design:
        raise CurveQAError(f"case {summary.get('case_id')}: no design inputs in the record")
    x = np.asarray(design["x"], dtype=float)
    if x.shape != (5,) or not np.all(np.isfinite(x)):
        raise CurveQAError(f"case {summary.get('case_id')}: design inputs malformed")
    level = tuple(summary["case"]["level"])
    return Option3Curve(x, level, str(summary["case_id"]), lr, ratio)


__all__ = ["Option3Curve", "CurveQAError", "from_summary", "MAX_STEP", "LR_SUPPORT", "INCREMENT_FLOOR"]
