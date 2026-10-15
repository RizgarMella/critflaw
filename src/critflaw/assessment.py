"""The assessment engine: Lr, Kr, acceptance and margins (theory manual, chapter 01)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from critflaw import toughness as toughness_models
from critflaw.fad.curves import Option1Curve
from critflaw.fad.kr import fracture_ratio
from critflaw.fad.margins import EXIT_CUTOFF, EXIT_NAMES, acceptable, radial_margin
from critflaw.fad.option2 import MAX_TABULATED_CORRECTION, Option2Curve, TabulatedLaw
from critflaw.kb.status import kb_status
from critflaw.model import (
    Component,
    Cylinder,
    CylinderSurfaceFlaw,
    CylinderThroughFlaw,
    EmbeddedFlaw,
    ExtendedFlaw,
    Flaw,
    MasterCurve,
    Material,
    StressState,
    SurfaceFlaw,
    ThroughFlaw,
    Value,
    _positive,
    _Replaceable,
    is_random,
)
from critflaw.provenance import EQUATIONS, cite
from critflaw.solutions import cylinder_dispatch
from critflaw.solutions import embedded as embedded_solutions
from critflaw.solutions import plate as plate_solutions

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.intp]
StrArray = NDArray[np.str_]

EDITION = "2013+A1"
EXIT_INVALID = "invalid"
EXIT_DOMAIN_END = "domain_end"  # a tabulated Option 2 law ends before the plastic collapse cut-off

# Uncertainty sources that a deterministic result does not account for (theory manual, chapter 01, section 1.9;
# chapter 04, section 4.11).
EXCLUDED_UNCERTAINTY: tuple[str, ...] = (
    "model uncertainty of the stress intensity factor solution",
    "model uncertainty of the reference stress solution",
    "model uncertainty of the failure assessment line",
    "scatter and statistical uncertainty of every input (single values were used)",
)


class OutOfValidityError(ValueError):
    """A solution was requested outside its validity range (never extrapolated)."""


@dataclass(frozen=True)
class Trace:
    """What a result depended on, and what it does not account for (theory manual, chapter 01, section 1.9)."""

    edition: str
    equations: tuple[str, ...]
    options: dict[str, str]
    flags: tuple[str, ...]
    excluded_uncertainty: tuple[str, ...] = EXCLUDED_UNCERTAINTY

    def citations(self) -> list[str]:
        return [cite(e) for e in self.equations]

    def dependencies(self) -> list[dict[str, Any]]:
        """Verification status of every equation the result used."""
        out = []
        for e in self.equations:
            record = EQUATIONS[e]
            out.append(
                {
                    "id": e,
                    "citation": cite(e),
                    "checked_against_printed_source": record.pdf_checked,
                    "checked_against_primary_reference": record.checked_against_primary,
                    "self_consistent_only": record.self_consistent_only,
                    "kb_status": self.kb_status(e),  # linked evidence; suspect-cell reads downgrade
                }
            )
        return out

    def kb_status(self, equation: str) -> str:
        """The knowledge-base status of an equation in this result: from linked evidence, and ``unverified`` for a
        tabulated cylinder factor when samples were interpolated through a suspect cell (theory manual,
        chapter 08, section 8.3.2)."""
        if equation.startswith("k.cylinder.tables.") and any("suspect_cell" in f for f in self.flags):
            return "unverified"
        return kb_status(equation)

    def below_cross_checked(self) -> list[str]:
        """Equations whose knowledge-base status (linked evidence, theory manual 08) is below ``cross-checked``."""
        return [e for e in self.equations if self.kb_status(e) in ("unverified", "self-consistent")]

    def self_consistent_only(self) -> list[str]:
        return [e for e in self.equations if EQUATIONS[e].self_consistent_only]

    def not_checked_against_primary(self) -> list[str]:
        return [e for e in self.equations if EQUATIONS[e].checked_against_primary != "yes"]

    def report(self) -> str:
        options = ", ".join(f"{k}={v}" for k, v in self.options.items())
        lines = [f"Source edition: {self.edition}", f"Options: {options}"]
        lines.append("Flags: " + (", ".join(self.flags) if self.flags else "none"))
        lines.append("Equations used:")
        for dep in self.dependencies():
            status = []
            status.append(f"knowledge base: {dep['kb_status']}")
            if dep["self_consistent_only"]:
                status.append("SELF-CONSISTENT ONLY (no oracle independent of the source text)")
            if dep["checked_against_primary_reference"] != "yes":
                status.append("not checked against a primary reference")
            if not dep["checked_against_printed_source"]:
                status.append("NOT checked against the printed source")
            lines.append(f"  - {dep['citation']}" + (f"  [{'; '.join(status)}]" if status else ""))
        lines.append("Not accounted for:")
        lines += [f"  - {s}" for s in self.excluded_uncertainty]
        return "\n".join(lines)


@dataclass(frozen=True)
class Result:
    """Outcome of an assessment. Every array has the common broadcast shape of the inputs,
    except the ``*_by_position`` and ``k_*`` arrays, which have a leading crack-front-position axis.
    For samples outside a solution's validity range: ``valid`` and ``acceptable`` are False,
    floats are NaN, ``governing_position`` is -1 and ``exit`` is "invalid".
    """

    lr: FloatArray
    kr: FloatArray  # at the governing crack-front position
    radial_margin: FloatArray  # minimum over positions
    acceptable: BoolArray  # direct test at every position (theory manual, section 1.6)
    valid: BoolArray
    governing_position: IntArray  # index into ``positions``
    exit: StrArray  # "curve", "step", "cutoff", "domain_end", "none" (origin) or "invalid"
    positions: tuple[str, ...]
    kr_by_position: FloatArray
    k_primary: FloatArray
    k_secondary: FloatArray
    reference_stress: FloatArray
    qm: FloatArray  # membrane secondary stress actually used (after any relaxation)
    lr_max: FloatArray
    trace: Trace = field(repr=False)

    def report(self) -> str:
        """Human-readable summary including verification status and exclusions (theory manual, section 1.9)."""
        n = int(np.size(self.valid))
        head = [
            f"Evaluations: {n}; acceptable: {int(np.sum(self.acceptable))}; "
            f"outside validity: {int(np.sum(~self.valid))}"
        ]
        if n == 1 and bool(np.all(self.valid)):
            head.append(
                f"Lr = {self.lr.item():.6g}, Kr = {self.kr.item():.6g}, radial margin = "
                f"{self.radial_margin.item():.6g} (exit: {self.exit.item()}), "
                f"governing position: {self.positions[int(self.governing_position.item())]}"
            )
        return "\n".join(head) + "\n" + self.trace.report()


def _f(value: Any) -> FloatArray:
    return np.asarray(value, dtype=float)


@dataclass(frozen=True)
class Assessment(_Replaceable):
    component: Component
    flaw: Flaw
    material: Material
    loads: StressState
    bending_restraint: Literal["normal", "pin_jointed"] = "normal"
    interaction: Literal["rho", "V"] = "rho"
    plane_stress: bool = True  # for the plastic-zone correction when secondary stress acts alone
    # Multiplicative model-uncertainty factors (theory manual, section 1.1); 1.0 means "not accounted for".
    k_model: Value = 1.0  # on primary and secondary stress intensity factors
    sref_model: Value = 1.0  # on the reference stress
    fad_model: Value = 1.0  # on the assessment line (the cut-off is unaffected)

    def __post_init__(self) -> None:
        for name in ("k_model", "sref_model", "fad_model"):
            _positive(name, getattr(self, name))
        if self.bending_restraint not in ("normal", "pin_jointed"):
            raise ValueError("bending_restraint must be 'normal' or 'pin_jointed'")
        if self.interaction not in ("rho", "V"):
            raise ValueError("interaction must be 'rho' or 'V'")
        if not isinstance(self.plane_stress, bool):
            raise TypeError("plane_stress must be a bool")

    def curve(self) -> Option1Curve | Option2Curve:
        m = self.material
        if m.fad_option == 2:
            return Option2Curve(m.law, _f(m.yield_strength), m.tensile_strength(), _f(m.modulus))
        return Option1Curve(_f(m.yield_strength), m.tensile_strength(), _f(m.modulus), m.yielding, m.luders_strain)

    def random_fields(self) -> list[str]:
        """Names of fields that hold random variables (empty for a realised problem)."""
        found = []

        def holds_random(value: Any) -> bool:
            if is_random(value):
                return True
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                return any(holds_random(getattr(value, g.name)) for g in dataclasses.fields(value))
            return False

        for owner, obj in (
            ("component", self.component),
            ("flaw", self.flaw),
            ("material", self.material),
            ("loads", self.loads),
            ("assessment", self),
        ):
            for f in dataclasses.fields(obj):
                if owner == "assessment" and f.name in ("component", "flaw", "material", "loads"):
                    continue
                if holds_random(getattr(obj, f.name)):
                    found.append(f"{owner}.{f.name}")
        return found

    def run(self, on_invalid: Literal["raise", "mask"] = "raise", strict: bool = False) -> Result:
        """Evaluate the assessment.

        ``on_invalid="raise"`` raises :class:`OutOfValidityError` if any sample is
        outside a solution's validity range. ``"mask"`` returns those samples blanked
        (see :class:`Result`), for the caller to account for explicitly (theory manual, section 4.6).
        ``strict=True`` refuses a result that used any equation whose knowledge-base status
        (test-linked evidence) is below ``cross-checked`` (theory manual, chapter 08, section 8.4);
        by default the status of every equation is recorded in the trace.
        """
        if on_invalid not in ("raise", "mask"):
            raise ValueError("on_invalid must be 'raise' or 'mask'")
        random = self.random_fields()
        if random:
            raise TypeError(
                f"the problem contains random variables ({', '.join(random)}); "
                "assess it with critflaw.ProbabilisticAssessment"
            )
        m = self.material
        loads = self.loads
        res = loads.residual
        sy, su = _f(m.yield_strength), _f(m.tensile_strength())
        k_model, sref_model, fad_model = _f(self.k_model), _f(self.sref_model), _f(self.fad_model)
        b, w = _f(self.component.thickness), _f(self.component.width)
        pm, pb = _f(loads.pm), _f(loads.pb)
        surface = isinstance(self.flaw, SurfaceFlaw)
        embedded = isinstance(self.flaw, EmbeddedFlaw)
        cylindrical = isinstance(self.flaw, (CylinderSurfaceFlaw, CylinderThroughFlaw, ExtendedFlaw))
        if embedded and isinstance(self.component, Cylinder):
            raise NotImplementedError("TODO(verify): embedded flaws in cylinders (the plate solution with M = 1)")
        if cylindrical != isinstance(self.component, Cylinder):
            raise TypeError("cylinder flaws need a Cylinder component and plate flaws a FlatPlate")
        cyl: cylinder_dispatch.CylinderFactors | None = None
        equations = ["fad.cutoff", "fad.kr", "k.general"]
        if m.fad_option == 2:
            equations += ["fad.option2", "fad.option2.monotone"]
        else:
            if m.yielding != "discontinuous":
                equations.append("fad.option1.continuous")
            if m.yielding != "continuous":
                equations.append("fad.option1.discontinuous")
                if m.luders_strain is None:
                    equations.append("material.luders_strain")
        if isinstance(m.toughness, MasterCurve):
            equations.append("toughness.master_curve")
        flags: list[str] = []

        if isinstance(self.flaw, SurfaceFlaw):
            a = _f(self.flaw.a)
            c = 0.5 * _f(self.flaw.two_c)
            valid_raw = plate_solutions.surface_flaw_valid(a, c, b, w)
            positions: tuple[str, ...] = ("deepest", "surface")
            thetas: tuple[float, ...] = (plate_solutions.THETA_DEEPEST, plate_solutions.THETA_SURFACE)
            equations += ["k.plate.surface", "sref.plate.surface"]
        elif isinstance(self.flaw, ThroughFlaw):
            a = 0.5 * _f(self.flaw.two_a)
            c = a
            valid_raw = plate_solutions.through_flaw_valid(a, w)
            positions = ("tip",)
            thetas = (0.0,)
            equations += ["k.plate.through", "sref.plate.through"]
        elif isinstance(self.flaw, EmbeddedFlaw):
            a = 0.5 * _f(self.flaw.two_a)
            c = 0.5 * _f(self.flaw.two_c)
            p_lig = _f(self.flaw.p)
            # The source gives no bending magnification at the major-axis ends: a bending-loaded embedded
            # flaw is refused rather than assessed with an invented M_b = 0 (theory manual, section 2.3.3).
            bending_present = bool(np.any(pb != 0.0)) or bool(np.any(_f(loads.qb) != 0.0))
            if bending_present:
                raise NotImplementedError(
                    "TODO(verify): M_b at the major-axis ends of an embedded flaw (the source gives the bending "
                    "magnification at the minor-axis ends only); a bending-loaded embedded flaw is not assessed"
                )
            valid_raw = embedded_solutions.embedded_flaw_valid(a, c, p_lig, b, w)
            positions = ("near_end", "major_axis")
            thetas = (embedded_solutions.THETA_NEAR_END, embedded_solutions.THETA_MAJOR_AXIS)
            equations += ["k.plate.embedded", "sref.plate.embedded"]
            flags.append("embedded_recharacterisation_relaxation_not_applied")
        elif cylindrical:
            assert isinstance(self.component, Cylinder)
            cyl = cylinder_dispatch.cylinder_factors(self.component, self.flaw, loads, pm, pb)
            flags += cyl.flags  # ineligible realisations are out of validity (masked or raised below), never dropped
            a = cyl.crack_size
            c = a
            valid_raw = cyl.valid
            positions = cyl.positions
            thetas = tuple(float(i) for i in range(len(positions)))  # position index, not an angle
            equations += cyl.equations
            if bool(np.any(cyl.suspect)):
                flags.append("tabulated_factor_interpolated_through_a_suspect_cell_samples_present")
        else:  # pragma: no cover - exhaustive over Flaw
            raise TypeError(f"unsupported flaw type {type(self.flaw).__name__}")

        # Common sample shape of every numeric input.
        participants: list[Any] = [sy, su, b, w, pm, pb, a, c, m.modulus, loads.qm, loads.qb]
        if embedded:
            participants.append(p_lig)
        if cyl is not None:
            participants += [cyl.sref, cyl.pm_total, cyl.fw, cyl.bulging, *cyl.mm, *cyl.mb]
        participants += [k_model, sref_model, fad_model]
        if isinstance(m.toughness, MasterCurve):
            participants += [m.toughness.t0, m.toughness.temperature, m.toughness.probability]
        else:
            participants.append(m.toughness)
        if m.luders_strain is not None:
            participants.append(m.luders_strain)
        if res is not None:
            participants += [
                v for v in (res.appropriate_yield, res.appropriate_uts, res.room_temperature_yield) if v is not None
            ]
        law_shape = tuple(getattr(m.law, "parameter_shape", ())) if m.fad_option == 2 else ()
        shape = np.broadcast_shapes(law_shape, *(np.shape(v) for v in participants))
        curve = self.curve()
        valid: BoolArray = np.broadcast_to(valid_raw, shape)
        if isinstance(curve, Option2Curve):
            # a law that has not strained plastically beyond Lr = sqrt(2) is outside the validity of
            # the monotone-line construction (theory manual, section 1.4)
            law_ok = np.broadcast_to(curve.valid, shape)
            if not np.all(law_ok | ~valid):
                if on_invalid == "raise":
                    raise OutOfValidityError(
                        "the stress-strain law has not strained plastically beyond Lr = 1.41 "
                        "(e(s) < s^2/sqrt(2)); the Option 2 line is not defined for it"
                    )
                flags.append("option2_law_too_elastic_beyond_root_two_samples_present")
            valid = valid & law_ok
            if isinstance(m.law, TabulatedLaw):
                # A tabulated line lowered by more than the limit means the data are suspect. The
                # correction is evaluated for each realisation over its own domain.
                per_sample = np.broadcast_to(curve.correction(), shape)
                too_large = per_sample > MAX_TABULATED_CORRECTION
                if np.any(too_large & valid):
                    if on_invalid == "raise":
                        raise OutOfValidityError(
                            f"the monotonicity rule lowers this tabulated Option 2 line by "
                            f"{float(np.max(per_sample)):.3f}, more than {MAX_TABULATED_CORRECTION}: "
                            "the stress-strain data are suspect"
                        )
                    flags.append("option2_tabulated_correction_above_limit_samples_present")
                valid = valid & ~too_large
        if not np.all(valid):
            if on_invalid == "raise":
                raise OutOfValidityError(
                    f"{int(np.sum(~valid))} of {valid.size} evaluation(s) are outside the validity "
                    "range of the stress intensity factor solution; solutions are not extrapolated"
                )
            flags.append("out_of_validity_samples_present")
        # Invalid samples are evaluated on a stand-in geometry (always inside validity) so that
        # no warnings or NaNs arise, and every one of their outputs is blanked below.
        a_eval = np.where(valid, a, np.minimum(0.1 * b, 0.05 * w))
        c_eval = np.where(valid, c, np.minimum(0.2 * b, 0.05 * w)) if (surface or embedded) else a_eval
        if embedded:
            a_eval = np.where(valid, a, np.minimum(0.05 * b, 0.025 * w))
            p_eval = np.where(valid, p_lig, 0.2 * b)

        if surface:
            sref = plate_solutions.surface_flaw_reference_stress(
                pm, pb, a_eval, c_eval, b, w, pin_jointed=self.bending_restraint == "pin_jointed"
            )
            fw = plate_solutions.surface_flaw_fw(a_eval, c_eval, b, w)
        elif embedded:
            sref = embedded_solutions.embedded_flaw_reference_stress(pm, pb, a_eval, c_eval, p_eval, b, w)
            fw = embedded_solutions.embedded_flaw_fw(a_eval, c_eval, p_eval, w)
        elif cyl is not None:
            sref = np.where(valid, cyl.sref, 1.0)
            fw = np.where(valid, cyl.fw, 1.0)
        else:
            sref = plate_solutions.through_flaw_reference_stress(pm, pb, a_eval, w)
            fw = plate_solutions.through_flaw_fw(a_eval, w)
        sref = sref * sref_model
        lr = np.broadcast_to(sref / sy, shape)

        # Fracture toughness; the Master Curve value depends on the crack front length (theory manual, section 2.6).
        if isinstance(m.toughness, MasterCurve):
            front = 2.0 * c_eval if surface else (4.0 * c_eval if embedded else 2.0 * b)  # theory manual, section 2.6
            if cyl is not None:
                front = np.minimum(cyl.front_length, 2.0 * b)
            # a sampled probability can round to exactly 1; the Master Curve needs p < 1
            p_mc = np.minimum(_f(m.toughness.probability), np.nextafter(1.0, 0.0))
            k_mat = toughness_models.master_curve(p_mc, _f(m.toughness.temperature), _f(m.toughness.t0), front, b)
        else:
            k_mat = _f(m.toughness)

        # Secondary membrane stress: user value plus any welding residual stress.
        qm = np.broadcast_to(_f(loads.qm), shape)
        if res is not None:
            sy_app = sy if res.appropriate_yield is None else _f(res.appropriate_yield)
            sy_unrelaxed = sy_app if res.room_temperature_yield is None else _f(res.room_temperature_yield)
            q_res = res.fraction * sy_unrelaxed
            if res.relaxation:
                su_app = su if res.appropriate_uts is None else _f(res.appropriate_uts)
                flow = 0.5 * (sy_app + su_app)
                q_res = np.minimum(q_res, (1.4 - np.clip(sref / flow, 0.4, 1.0)) * sy_app)
                flags.append("residual_stress_relaxation_applied")
            qm = qm + q_res
            equations.append("residual.as_welded" if res.fraction == 1.0 else "residual.pwht")

        kp_list, ks_list, kr_list, f_list, ok_list, exit_list = [], [], [], [], [], []
        any_secondary = any_secondary_only = any_chi_high = any_negative_kp = False
        for theta in thetas:
            if surface:
                mm, mb = plate_solutions.surface_flaw_mm_mb(a_eval, c_eval, b, theta)
            elif embedded:
                mm = embedded_solutions.embedded_flaw_mm(a_eval, c_eval, p_eval, theta)
                mb = np.zeros(shape)  # no bending is present (refused above); M_b is not evaluated here
            elif cyl is not None:
                i_pos = int(theta)
                mm = np.where(valid, cyl.mm[i_pos], 1.0)
                mb = np.where(valid, cyl.mb[i_pos], 1.0)
            else:
                mm = mb = np.ones(shape)
            pm_k = cyl.pm_total * cyl.membrane_multiplier if cyl is not None else pm
            bulging = cyl.bulging if cyl is not None else 1.0
            kp, ks = plate_solutions.stress_intensity(a_eval, mm, mb, fw, pm_k, pb, qm, _f(loads.qb), bulging=bulging)
            kp, ks = kp * k_model, ks * k_model
            # The bending magnification factor is negative at the deepest point of deep flaws.
            # A crack closed by the primary load is assessed with Kp = 0 (conservative; theory manual, section 1.5).
            any_negative_kp |= bool(np.any((kp < 0.0) & valid))
            kp = np.maximum(kp, 0.0)
            ratio = fracture_ratio(
                kp, ks, lr, k_mat, a_eval, sy, method=self.interaction, plane_stress=self.plane_stress
            )
            # A factor m on the assessment line is applied by assessing (Lr, Kr / m) against the
            # unmodified line: the same acceptance test, with the margin bracket still valid.
            kr_assessed = ratio.kr / fad_model
            margin, exit_code = radial_margin(lr, kr_assessed, curve)
            kp_list.append(np.broadcast_to(kp, shape))
            ks_list.append(np.broadcast_to(ks, shape))
            kr_list.append(np.broadcast_to(ratio.kr, shape))
            f_list.append(np.broadcast_to(margin, shape))
            ok_list.append(np.broadcast_to(acceptable(lr, kr_assessed, curve), shape))
            exit_list.append(np.broadcast_to(exit_code, shape))
            sec_only = np.broadcast_to(ratio.secondary_only, shape)
            any_secondary |= bool(np.any((np.broadcast_to(ks, shape) > 0.0) & ~sec_only & valid))
            any_secondary_only |= bool(np.any(sec_only & valid))
            any_chi_high |= bool(np.any(np.broadcast_to(ratio.chi_above_limit, shape) & valid))

        if any_negative_kp:
            flags.append("negative_primary_k_set_to_zero")
        if any_secondary:  # at least one valid sample combines primary and secondary stress
            equations.append("fad.rho.simplified" if self.interaction == "rho" else "fad.v.simplified")
        if any_secondary_only:
            equations.append("fad.kjs.plastic_zone")
            flags.append("secondary_stress_acting_alone")
        if any_chi_high:
            flags.append("chi_above_4_simplified_interaction_over_conservative")

        # Governing position: smallest margin; where margins tie (both positions leave through the
        # cut-off or a step) the position with the larger Kr governs.
        f_stack, kr_stack = np.stack(f_list), np.stack(kr_list)
        f_min = np.min(f_stack, axis=0)
        tied = f_stack <= f_min * (1.0 + 1e-12)
        governing = np.argmax(np.where(tied, kr_stack, -np.inf), axis=0)

        def take(stack: list[NDArray[Any]]) -> NDArray[Any]:
            picked: NDArray[Any] = np.take_along_axis(np.stack(stack), governing[np.newaxis, ...], axis=0)[0]
            return picked

        exit_codes = take(exit_list)
        exit_names = np.full(shape, EXIT_INVALID, dtype="<U10")
        for code, name in EXIT_NAMES.items():
            exit_names[(exit_codes == code) & valid] = name
        if isinstance(curve, Option2Curve):
            ends = np.broadcast_to(curve.ends_at_data, shape)
            exit_names[(exit_codes == EXIT_CUTOFF) & valid & ends] = EXIT_DOMAIN_END
            if np.any(ends & valid):
                flags.append("stress_strain_data_end_before_collapse_cut_off")
        law_options: dict[str, str] = {}
        if isinstance(curve, Option2Curve):
            correction = curve.largest_correction()
            law_options["option2_monotone_correction"] = f"{correction:.4f}"
            notes = tuple(getattr(m.law, "notes", ()))
            if notes:
                law_options["material_law_notes"] = "; ".join(notes)
        excluded = tuple(
            text
            for text, factor in zip(EXCLUDED_UNCERTAINTY, (k_model, sref_model, fad_model, None), strict=True)
            if factor is None or np.ndim(factor) == 0
        )

        cyl_options = dict(cyl.notes) if cyl is not None else {}
        trace = Trace(
            edition=EDITION,
            equations=tuple(dict.fromkeys(equations)),
            options={
                **cyl_options,
                "fad_option": str(m.fad_option),
                "material_law": type(m.law).__name__ if m.fad_option == 2 else "none",
                "yielding": m.yielding,
                "interaction": self.interaction,
                "bending_restraint": self.bending_restraint,
                "plastic_zone_correction": "plane stress" if self.plane_stress else "plane strain",
                **law_options,
            },
            flags=tuple(flags),
            excluded_uncertainty=excluded,
        )
        if strict and trace.below_cross_checked():
            raise ValueError(
                "strict assessment refused: equations below 'cross-checked' in the knowledge base: "
                + ", ".join(trace.below_cross_checked())
            )
        nan = np.nan
        return Result(
            lr=np.where(valid, lr, nan),
            kr=np.where(valid, take(kr_list), nan),
            radial_margin=np.where(valid, take(f_list), nan),
            acceptable=np.all(np.stack(ok_list), axis=0) & valid,
            valid=np.array(valid, dtype=bool),
            governing_position=np.where(valid, governing, -1),
            exit=exit_names,
            positions=positions,
            kr_by_position=np.where(valid, np.stack(kr_list), nan),
            k_primary=np.where(valid, np.stack(kp_list), nan),
            k_secondary=np.where(valid, np.stack(ks_list), nan),
            reference_stress=np.where(valid, np.broadcast_to(sref, shape), nan),
            qm=np.where(valid, qm, nan),
            lr_max=np.where(valid, np.broadcast_to(curve.lr_max, shape), nan),
            trace=trace,
        )
