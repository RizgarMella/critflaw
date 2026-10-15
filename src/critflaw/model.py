"""Problem definition: geometry, flaw, material and loads (immutable; SI units).

Any numeric field may be a scalar, a NumPy array (one value per sample), or a
:class:`critflaw.prob.Distribution` (a random variable; theory manual, section 4.1).
A problem containing distributions is assessed through
:class:`critflaw.prob.ProbabilisticAssessment`, which substitutes sampled arrays and
re-runs every validation below on the realised values.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import ArrayLike

from critflaw.prob.distributions import Distribution

# A field value: a number, an array, or a random variable.
Value = ArrayLike | Distribution


def is_random(value: Any) -> bool:
    return isinstance(value, Distribution) or (isinstance(value, Ratio) and isinstance(value.value, Distribution))


def _numeric(name: str, value: Any) -> None:
    if isinstance(value, (str, bytes, bool, np.bool_)):
        raise TypeError(f"{name} must be a number, an array of numbers or a distribution")


def _positive(name: str, value: Any) -> None:
    _numeric(name, value)
    if isinstance(value, Distribution):
        # Decided from the support alone: an unbounded distribution on a
        # positive quantity must be truncated explicitly by the user; nothing is dropped silently.
        lowest = value.support()[0]
        # a continuous distribution may start at zero (no mass there); a discrete one may not
        if lowest < 0.0 or (value.is_discrete and lowest <= 0.0):
            raise ValueError(
                f"{name} must be positive but its distribution has support at or below zero; "
                "write the truncation explicitly, e.g. Truncated(Normal(...), lo=0)"
            )
        return
    arr = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"{name} must be finite and positive")


def _finite(name: str, value: Any) -> None:
    _numeric(name, value)
    if isinstance(value, Distribution):
        return
    if np.any(~np.isfinite(np.asarray(value, dtype=float))):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class Ratio:
    """Tensile strength given as ``yield strength * value`` with ``value > 1`` (theory manual, section 3.5), so that
    the ordering of yield and tensile strength holds by construction when both are random."""

    value: Value

    def __post_init__(self) -> None:
        if isinstance(self.value, Distribution):
            lowest = self.value.support()[0]
            if lowest < 1.0 or (self.value.is_discrete and lowest <= 1.0):
                raise ValueError("the support of a tensile-to-yield ratio must lie above 1")
        elif np.any(np.asarray(self.value, dtype=float) <= 1.0):
            raise ValueError("a tensile-to-yield ratio must exceed 1")


@dataclass(frozen=True)
class MasterCurve:
    """Master Curve toughness: the cumulative probability ``probability`` (normally a
    Uniform(0, 1) random variable) at temperature ``temperature`` for reference temperature
    ``t0`` (degrees C). The toughness itself is computed in the engine from the crack front
    length of the flaw being assessed (theory manual, section 2.6)."""

    t0: Value
    temperature: Value
    probability: Value

    def __post_init__(self) -> None:
        _finite("t0", self.t0)
        _finite("temperature", self.temperature)
        if isinstance(self.probability, Distribution):
            lo, hi = self.probability.support()
            if lo < 0.0 or hi > 1.0 or (self.probability.is_discrete and hi >= 1.0):
                raise ValueError("probability must have support within [0, 1), with no mass at 1")
        else:
            p = np.asarray(self.probability, dtype=float)
            if np.any((p < 0.0) | (p >= 1.0)):
                raise ValueError("probability must lie in [0, 1)")


class _Replaceable:
    def with_(self, **changes: Any) -> Self:
        """Return a modified copy; problems are never mutated."""
        return dataclasses.replace(self, **changes)  # type: ignore[type-var]


@dataclass(frozen=True)
class FlatPlate(_Replaceable):
    thickness: Value  # B
    width: Value  # W

    def __post_init__(self) -> None:
        _positive("thickness", self.thickness)
        _positive("width", self.width)


@dataclass(frozen=True)
class Cylinder(_Replaceable):
    """Cylinder of inner radius ``r_i``, wall ``thickness`` and ``length`` (the W of the finite-width rule).
    ``k_route`` selects the stress intensity factor model explicitly (theory manual, section 2.4): "pressure_only"
    (plate factors with a bulging factor; refuses mechanical primary stresses) or "tabulated" (the tabulated
    factors, interpolated). ``sref_route`` selects the reference stress family: "thin" (the thin-wall
    reference stresses), "thick_pressure" (the thick-wall limit pressures for axial surface flaws; needs a
    ``PressureLoad``) or "thick" (the thick-wall limit loads: axial external under membrane and bending,
    circumferential under tension and global bending)."""

    r_i: Value
    thickness: Value  # B
    length: Value  # W
    k_route: Literal["pressure_only", "tabulated"]
    sref_route: Literal["thin", "thick_pressure", "thick"] = "thin"

    def __post_init__(self) -> None:
        _positive("r_i", self.r_i)
        _positive("thickness", self.thickness)
        _positive("length", self.length)
        if self.k_route not in ("pressure_only", "tabulated"):
            raise ValueError(
                "k_route must be 'pressure_only' or 'tabulated' (no default: the routes are different models)"
            )
        if self.sref_route not in ("thin", "thick_pressure", "thick"):
            raise ValueError("sref_route must be 'thin', 'thick_pressure' or 'thick'")

    @property
    def width(self) -> Value:
        return self.length

    @property
    def r_m(self) -> Any:
        return np.asarray(self.r_i, dtype=float) + 0.5 * np.asarray(self.thickness, dtype=float)


@dataclass(frozen=True)
class CylinderSurfaceFlaw(_Replaceable):
    """Semi-elliptical surface flaw in a cylinder wall: depth ``a``, length ``two_c``, ``orientation``
    "axial" | "circumferential", ``position`` "internal" | "external"."""

    a: Value
    two_c: Value
    orientation: Literal["axial", "circumferential"]
    position: Literal["internal", "external"]

    def __post_init__(self) -> None:
        _positive("a", self.a)
        _positive("two_c", self.two_c)
        if self.orientation not in ("axial", "circumferential") or self.position not in ("internal", "external"):
            raise ValueError("orientation must be 'axial' or 'circumferential'; position 'internal' or 'external'")


@dataclass(frozen=True)
class CylinderThroughFlaw(_Replaceable):
    """Through-wall flaw of length ``two_a`` (circumferential: measured at mid-thickness)."""

    two_a: Value
    orientation: Literal["axial", "circumferential"]

    def __post_init__(self) -> None:
        _positive("two_a", self.two_a)
        if self.orientation not in ("axial", "circumferential"):
            raise ValueError("orientation must be 'axial' or 'circumferential'")


@dataclass(frozen=True)
class ExtendedFlaw(_Replaceable):
    """Extended flaw of uniform depth ``a``: fully circumferential, or extended axial (the full length)."""

    a: Value
    orientation: Literal["axial", "circumferential"]
    position: Literal["internal", "external"]

    def __post_init__(self) -> None:
        _positive("a", self.a)
        if self.orientation not in ("axial", "circumferential") or self.position not in ("internal", "external"):
            raise ValueError("orientation must be 'axial' or 'circumferential'; position 'internal' or 'external'")


@dataclass(frozen=True)
class PressureLoad(_Replaceable):
    """Internal pressure with the user's pressure-to-stress factors (theory manual, section 2.4.1): the primary
    membrane stress due to pressure is ``hoop_per_pressure x pressure`` for axial flaws and
    ``axial_per_pressure x pressure`` for circumferential flaws. The same ``pressure`` feeds the thick-wall
    limit pressures, so K and Lr always describe one loading. The source gives no conversion and its K
    solutions exclude crack-face pressure."""

    pressure: Value
    hoop_per_pressure: Value
    axial_per_pressure: Value = 0.0

    def __post_init__(self) -> None:
        _finite("pressure", self.pressure)
        _positive("hoop_per_pressure", self.hoop_per_pressure)
        _finite("axial_per_pressure", self.axial_per_pressure)


@dataclass(frozen=True)
class SurfaceFlaw(_Replaceable):
    """Semi-elliptical surface flaw of depth ``a`` and surface length ``two_c``."""

    a: Value
    two_c: Value

    def __post_init__(self) -> None:
        _positive("a", self.a)
        _positive("two_c", self.two_c)

    @classmethod
    def from_a_over_c(cls, a: ArrayLike, a_over_c: ArrayLike) -> SurfaceFlaw:
        _positive("a_over_c", a_over_c)
        depth = np.asarray(a, dtype=float)
        return cls(a=depth, two_c=2.0 * depth / np.asarray(a_over_c, dtype=float))


@dataclass(frozen=True)
class ThroughFlaw(_Replaceable):
    """Through-thickness flaw of total length ``two_a``."""

    two_a: Value

    def __post_init__(self) -> None:
        _positive("two_a", self.two_a)


@dataclass(frozen=True)
class EmbeddedFlaw(_Replaceable):
    """Embedded elliptical flaw of height ``two_a`` (through-thickness), length ``two_c`` and ligament ``p``
    to the **nearer** surface (theory manual, section 2.3.3). ``near_surface`` names that surface as the one on
    which the declared bending stress is tensile ("tension") or compressive ("compression"); it is required
    whenever a bending component is present. The geometric validation ``2a + 2p <= B`` (p is the smaller
    ligament) is applied by the assessment, at construction for fixed values and per realisation otherwise.
    """

    two_a: Value
    two_c: Value
    p: Value
    near_surface: Literal["tension", "compression"] | None = None

    def __post_init__(self) -> None:
        _positive("two_a", self.two_a)
        _positive("two_c", self.two_c)
        _positive("p", self.p)
        if self.near_surface not in (None, "tension", "compression"):
            raise ValueError("near_surface must be 'tension', 'compression' or None")

    @classmethod
    def from_ligaments(
        cls, two_a: ArrayLike, two_c: ArrayLike, p_1: ArrayLike, p_2: ArrayLike, **kw: Any
    ) -> EmbeddedFlaw:
        """The flaw described by its two ligaments (fixed values); ``p`` is the smaller one."""
        p1, p2 = np.asarray(p_1, dtype=float), np.asarray(p_2, dtype=float)
        return cls(two_a=two_a, two_c=two_c, p=np.minimum(p1, p2), **kw)


Flaw = SurfaceFlaw | ThroughFlaw | EmbeddedFlaw | CylinderSurfaceFlaw | CylinderThroughFlaw | ExtendedFlaw
Component = FlatPlate | Cylinder


@dataclass(frozen=True)
class Material(_Replaceable):
    """Tensile and fracture properties.

    ``uts`` may be a :class:`Ratio`. ``toughness`` is Kmat in Pa m^0.5, or a :class:`MasterCurve`.
    For ``fad_option=2`` a material ``law`` (``critflaw.fad.option2.PowerLaw`` or
    ``TabulatedLaw``) is required and ``yielding`` must agree with it.
    """

    yield_strength: Value
    uts: Value | Ratio
    modulus: Value
    toughness: Value | MasterCurve
    # Required, no default (theory manual, section 1.3): a wrong assumption changes f(1) by a factor of about two.
    yielding: Literal["continuous", "discontinuous", "unknown"]
    fad_option: int = 1
    luders_strain: ArrayLike | None = None
    law: Any = None

    def __post_init__(self) -> None:
        if self.yielding not in ("continuous", "discontinuous", "unknown"):
            raise ValueError("yielding must be 'continuous', 'discontinuous' or 'unknown'")
        _positive("yield_strength", self.yield_strength)
        _positive("modulus", self.modulus)
        if not isinstance(self.uts, Ratio):
            _positive("uts", self.uts)
            if not (is_random(self.uts) or is_random(self.yield_strength)):
                if np.any(np.asarray(self.uts, dtype=float) <= np.asarray(self.yield_strength, dtype=float)):
                    raise ValueError("uts must exceed yield_strength")
            else:
                # Ordering must be guaranteed by the supports whenever either strength is random
                # (a fixed value is a single-point support).
                def bounds(value: Any) -> tuple[float, float]:
                    if isinstance(value, Distribution):
                        return value.support()
                    arr = np.asarray(value, dtype=float)
                    return float(np.min(arr)), float(np.max(arr))

                if bounds(self.uts)[0] <= bounds(self.yield_strength)[1]:
                    raise ValueError(
                        "these yield and tensile strength distributions can give tensile <= yield; "
                        "give the tensile strength as Ratio(...) of the yield strength, or bound the distributions"
                    )
        if not isinstance(self.toughness, MasterCurve):
            _positive("toughness", self.toughness)
        if self.fad_option not in (1, 2):
            raise NotImplementedError("FAD Options 1 and 2 are implemented")
        if self.fad_option == 2:
            if self.law is None:
                raise ValueError("fad_option=2 needs a material law (PowerLaw or TabulatedLaw)")
            if not (hasattr(self.law, "q_hat") and hasattr(self.law, "declared_plateau")):
                raise TypeError("law must be a PowerLaw or a TabulatedLaw")
            if self.yielding == "unknown" or bool(self.law.declared_plateau) != (self.yielding == "discontinuous"):
                raise ValueError(
                    "for Option 2 'yielding' must be 'discontinuous' if the law has a yield plateau and "
                    "'continuous' otherwise"
                )
            if self.luders_strain is not None:
                raise ValueError("for Option 2 the Lueders strain belongs to the material law")
            return
        if self.law is not None:
            raise ValueError("a material law is only used with fad_option=2")
        if self.luders_strain is not None:
            if self.yielding == "continuous":
                raise ValueError("luders_strain has no meaning for continuous yielding")
            _positive("luders_strain", self.luders_strain)
        elif self.yielding != "continuous" and not is_random(self.yield_strength):
            if np.any(np.asarray(self.yield_strength, dtype=float) >= 946.0e6):
                raise ValueError(
                    "the Lueders strain estimate applies only below 946 MPa yield strength; supply luders_strain"
                )

    def tensile_strength(self) -> ArrayLike:
        """Realised tensile strength (resolves a :class:`Ratio`)."""
        if isinstance(self.uts, Ratio):
            ratio = np.asarray(self.uts.value, dtype=float)
            return np.asarray(np.asarray(self.yield_strength, dtype=float) * ratio, dtype=float)
        return np.asarray(self.uts, dtype=float)


@dataclass(frozen=True)
class Residual(_Replaceable):
    """Uniform (membrane) welding residual stress  [provenance: residual.as_welded, residual.pwht].

    ``fraction`` of the appropriate yield strength; ``appropriate_yield`` defaults
    to the material yield strength. With ``relaxation`` the as-welded value is the
    lower of sy' and (1.4 - sref/sf') sy', with sref limited to [0.4 sf', sf'].
    ``room_temperature_yield`` replaces sy' in the unrelaxed term only, for
    assessment temperatures below ambient.
    """

    fraction: float = 1.0
    appropriate_yield: ArrayLike | None = None
    appropriate_uts: ArrayLike | None = None
    relaxation: bool = False
    room_temperature_yield: ArrayLike | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.relaxation, bool):
            raise TypeError("relaxation must be a bool")
        if not 0.0 < self.fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        if self.appropriate_uts is not None:
            if not self.relaxation:
                raise ValueError("appropriate_uts is only used with relaxation")
            reference = self.appropriate_yield
            if reference is not None and np.any(
                np.asarray(self.appropriate_uts, dtype=float) <= np.asarray(reference, dtype=float)
            ):
                raise ValueError("appropriate_uts must exceed appropriate_yield")
        if self.relaxation and self.fraction != 1.0:
            raise ValueError("relaxation with primary load is defined for the as-welded condition only")
        if self.room_temperature_yield is not None:
            _positive("room_temperature_yield", self.room_temperature_yield)
        if self.appropriate_yield is not None:
            _positive("appropriate_yield", self.appropriate_yield)
        if self.appropriate_uts is not None:
            _positive("appropriate_uts", self.appropriate_uts)

    @classmethod
    def as_welded(
        cls,
        appropriate_yield: ArrayLike | None = None,
        appropriate_uts: ArrayLike | None = None,
        relaxation: bool = False,
        room_temperature_yield: ArrayLike | None = None,
    ) -> Residual:
        return cls(1.0, appropriate_yield, appropriate_uts, relaxation, room_temperature_yield)

    @classmethod
    def pwht(
        cls,
        flaw_orientation: Literal["parallel", "transverse"],
        furnace_range: Literal["580-620C", "550-580C"] = "580-620C",
        appropriate_yield: ArrayLike | None = None,
    ) -> Residual:
        """Post-weld heat treated: 20% (flaw parallel to weld); 30% or 40% (flaw transverse)."""
        if flaw_orientation == "parallel":
            fraction = 0.2
        elif flaw_orientation == "transverse":
            fraction = 0.3 if furnace_range == "580-620C" else 0.4
        else:
            raise ValueError("flaw_orientation must be 'parallel' or 'transverse'")
        if furnace_range not in ("580-620C", "550-580C"):
            raise ValueError("furnace_range must be '580-620C' or '550-580C'")
        return cls(fraction, appropriate_yield, None, False)


@dataclass(frozen=True)
class StressState(_Replaceable):
    """Primary (pm, pb) and secondary (qm, qb) membrane and bending stresses, Pa."""

    pm: Value = 0.0
    pb: Value = 0.0
    qm: Value = 0.0
    qb: Value = 0.0
    residual: Residual | None = None
    # cylinders: pressure as one declared record; the membrane stress from a global moment (M.25,
    # at the flaw position farthest from the bending axis); the surface on which the through-wall bending
    # ``pb`` is tensile (the source's convention is the outside)
    pressure: PressureLoad | None = None
    pm_moment: Value = 0.0
    bending_tensile_surface: Literal["outside", "inside"] = "outside"

    def __post_init__(self) -> None:
        for name in ("pm", "pb", "qm", "qb", "pm_moment"):
            _finite(name, getattr(self, name))
        if self.bending_tensile_surface not in ("outside", "inside"):
            raise ValueError("bending_tensile_surface must be 'outside' or 'inside'")
        for name in ("pm", "pb"):
            value = getattr(self, name)
            if isinstance(value, Distribution):
                negative = value.support()[0] < 0.0
            else:
                negative = bool(np.any(np.asarray(value, dtype=float) < 0.0))
            if negative:
                raise ValueError(
                    "primary stresses must be non-negative (tensile); truncate an unbounded distribution explicitly"
                )
