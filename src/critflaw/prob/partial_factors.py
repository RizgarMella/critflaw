"""Published partial safety factors for fracture assessment (theory manual, section 4.9).

A discrete lookup: targets and coefficients of variation exactly as tabulated, no
interpolation, "not permitted" entries raise  [provenance: partial_factors.table].

The factors were derived by first-order second-moment analysis for the continuous-yield
Option 1 line, independent variables, normal stress and flaw size, Weibull or lognormal
toughness and lognormal yield strength. They apply to these reference values:
stress and flaw height: the mean; toughness: mean minus one standard deviation;
yield strength: mean minus two standard deviations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

# target probability of failure -> reliability index quoted with it (the factor table is headed
# 'probability of failure'; the source's separate table of example targets gives the same values per year)
TARGETS: dict[float, float] = {2.3e-1: 0.739, 1e-3: 3.09, 7e-5: 3.8, 1e-5: 4.27, 1e-7: 5.2}

_NP = float("nan")  # "not permitted"
_STRESS = {0.1: (1.05, 1.2, 1.25, 1.3, 1.4), 0.2: (1.1, 1.25, 1.35, 1.4, 1.55), 0.3: (1.12, 1.4, 1.5, 1.6, 1.8)}
_FLAW = {
    0.1: (1.0, 1.4, 1.5, 1.7, 2.1),
    0.2: (1.05, 1.45, 1.55, 1.8, 2.2),
    0.3: (1.08, 1.5, 1.65, 1.9, 2.3),
    0.5: (1.15, 1.7, 1.85, 2.1, 2.5),
}
_TOUGHNESS = {0.1: (1.0, 1.3, 1.5, 1.7, 2.0), 0.2: (1.0, 1.8, 2.6, 3.2, 5.5), 0.3: (1.0, 2.85, _NP, _NP, _NP)}
_YIELD = {0.1: (1.0, 1.05, 1.1, 1.2, 1.5)}


class NotTabulated(ValueError):
    pass


@dataclass(frozen=True)
class PartialFactors:
    target_pf: float
    beta: float
    stress: float  # multiplies the mean stress
    flaw_size: float  # multiplies the mean flaw height
    toughness: float  # divides (mean - 1 SD) toughness
    yield_strength: float  # divides (mean - 2 SD) yield strength


def _pick(table: Mapping[float, tuple[float, ...]], cov: float, column: int, what: str) -> float:
    key = next((k for k in table if abs(k - cov) < 1e-12), None)
    if key is None:
        raise NotTabulated(
            f"no published factor for {what} at a coefficient of variation of {cov:g}; "
            f"tabulated values are {sorted(table)} (no interpolation)"
        )
    value = table[key][column]
    if np.isnan(value):
        raise NotTabulated(f"the published table marks {what} at CoV {cov:g} as not permitted for this target")
    return value


def published_partial_factors(
    target_pf: float, cov_stress: float, cov_flaw_size: float, cov_toughness: float, cov_yield: float = 0.1
) -> PartialFactors:
    """Look up the published partial factors. Raises :class:`NotTabulated` for anything not in the table."""
    targets = list(TARGETS)
    column = next((i for i, t in enumerate(targets) if abs(t - target_pf) <= 1e-9 * t), None)
    if column is None:
        raise NotTabulated(f"target {target_pf:g} is not tabulated; tabulated targets are {targets}")
    return PartialFactors(
        target_pf=targets[column],
        beta=TARGETS[targets[column]],
        stress=_pick(_STRESS, cov_stress, column, "stress"),
        flaw_size=_pick(_FLAW, cov_flaw_size, column, "flaw size"),
        toughness=_pick(_TOUGHNESS, cov_toughness, column, "toughness"),
        yield_strength=_pick(_YIELD, cov_yield, column, "yield strength"),
    )


def implied_partial_factor(design_value: float, reference_value: float, kind: str) -> float:
    """Partial factor implied by a design point against the published reference value (theory manual, section 4.9).

    ``kind="load"`` (stress, flaw size): design / reference. ``kind="resistance"`` (toughness,
    yield strength): reference / design.
    """
    if kind not in ("load", "resistance"):
        raise ValueError("kind must be 'load' or 'resistance'")
    if not (0.0 < design_value < np.inf and 0.0 < reference_value < np.inf):
        raise ValueError(
            "design and reference values must be positive and finite (mean minus two standard deviations is not "
            "positive for a coefficient of variation of 0.5 or more: no factor is defined)"
        )
    if kind == "load":
        return design_value / reference_value
    return reference_value / design_value
