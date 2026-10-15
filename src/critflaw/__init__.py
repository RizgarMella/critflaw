"""critflaw: probabilistic engineering critical assessment of flawed metallic structures.

WARNING: research software in development. Not verified, not validated, not certified. Do not use it for any
real engineering, safety, design, fitness-for-service, regulatory or commercial decision. See DISCLAIMER.md.
"""

from critflaw import prob, toughness
from critflaw.assessment import Assessment, OutOfValidityError, Result, Trace
from critflaw.fad.curves import Option1Curve
from critflaw.fad.option2 import Option2Curve, PowerLaw, TabulatedLaw, hardening_exponent_lower_bound
from critflaw.model import (
    Cylinder,
    CylinderSurfaceFlaw,
    CylinderThroughFlaw,
    EmbeddedFlaw,
    ExtendedFlaw,
    FlatPlate,
    MasterCurve,
    Material,
    PressureLoad,
    Ratio,
    Residual,
    StressState,
    SurfaceFlaw,
    ThroughFlaw,
)
from critflaw.probabilistic import ProbabilisticAssessment, ProbabilisticResult
from critflaw.reserve import (
    Crossing,
    limiting_flaw_size,
    reserve_factor_on_load,
    reserve_factor_on_toughness,
)

__version__ = "0.0.1"

__all__ = [
    "Assessment",
    "Crossing",
    "FlatPlate",
    "MasterCurve",
    "Material",
    "Option1Curve",
    "Option2Curve",
    "OutOfValidityError",
    "PowerLaw",
    "ProbabilisticAssessment",
    "ProbabilisticResult",
    "Ratio",
    "Residual",
    "Result",
    "TabulatedLaw",
    "StressState",
    "Cylinder",
    "CylinderSurfaceFlaw",
    "CylinderThroughFlaw",
    "EmbeddedFlaw",
    "ExtendedFlaw",
    "PressureLoad",
    "SurfaceFlaw",
    "ThroughFlaw",
    "Trace",
    "hardening_exponent_lower_bound",
    "limiting_flaw_size",
    "prob",
    "reserve_factor_on_load",
    "reserve_factor_on_toughness",
    "toughness",
]

from critflaw import fatigue  # noqa: E402
from critflaw.fatigue import ExtremeLoad, Inspection, Spectrum, TimeDependentAssessment  # noqa: E402

__all__ += ["ExtremeLoad", "Inspection", "Spectrum", "TimeDependentAssessment", "fatigue"]
