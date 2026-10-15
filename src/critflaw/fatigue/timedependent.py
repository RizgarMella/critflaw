"""Time-dependent reliability under fatigue crack growth (theory manual, chapter 05, sections 5.4 to 5.7).

The three time-dependent probabilities (section 5.5) from the Monte Carlo channel, with every step bracketed
between the upper and lower pairings of the step's extreme load with the flaw configurations (section 5.4),
inspection and repair policies (section 5.6.1), the update of one component from its own inspection outcome
(section 5.6.2), coupled refinement of the load-process step, the stage-correlation sensitivity and subset
simulation in time.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special, stats

from critflaw.assessment import Assessment
from critflaw.fatigue.growth import (
    EMBEDDED,
    ENDED,
    EVENT_CAPACITY,
    EXTENDED,
    GROWING,
    NONE,
    SURFACE,
    THROUGH,
    CylinderSpec,
    Grower,
    GrowthState,
)
from critflaw.fatigue.laws import GrowthLaw
from critflaw.fatigue.spectrum import ExtremeLoad, Spectrum
from critflaw.model import (
    Cylinder,
    CylinderSurfaceFlaw,
    CylinderThroughFlaw,
    EmbeddedFlaw,
    ExtendedFlaw,
    SurfaceFlaw,
    ThroughFlaw,
    is_random,
)
from critflaw.prob.dependence import RandomVector
from critflaw.prob.distributions import Distribution
from critflaw.prob.intervals import clopper_pearson
from critflaw.prob.solvers import Evaluation, ReliabilityResult, consistent_driving_variable, solve
from critflaw.probabilistic import collect, realise
from critflaw.solutions import plate

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]
BoolArray = NDArray[np.bool_]

INTERIOR = (0.2, 0.4, 0.6, 0.8)  # interior checkpoints of a load-process step (fractions of the step)
MAX_REFINEMENTS = 4
KISH_GATE = 100.0


@dataclass(frozen=True)
class Inspection:
    """An inspection at ``time`` (a step boundary). ``pod`` is a probability-of-detection model of the
    flaw depth; a through flaw is detected with probability 1 unless ``pod_through`` (of its length)
    is given. ``sizing_error`` (measured minus true depth) with ``accept_below`` makes repair conditional
    on the measured depth (one further random variable per inspection). ``action`` on a detected and
    not accepted flaw: "repair" removes it (a ``repair_flaw`` as (a, 2c) in metres grows from then on),
    "none" records only. A repair clears the removed flaw's event record (leak, re-characterisation, end,
    sliding) so that the repair flaw's own breakthrough is a new leak event (and a failure when leaks
    count); that a removed flaw had leaked or ended is reported in ``events_any`` as history.
    ``observed`` conditions on this component's own outcome: "not_detected", "detected" or
    ("sized", a_measured); sizing of a through flaw is not defined."""

    time: float
    pod: Callable[[ArrayLike], FloatArray]
    sizing_error: Distribution | None = None
    accept_below: float | None = None
    action: Literal["repair", "none"] = "repair"
    repair_flaw: tuple[float, float] | None = None
    pod_through: Callable[[ArrayLike], FloatArray] | None = None
    pod_embedded: Callable[[ArrayLike], FloatArray] | None = None  # of the height 2a
    observed: Any = None

    def __post_init__(self) -> None:
        if self.accept_below is not None and self.sizing_error is None:
            raise ValueError("accept_below needs a sizing_error (the decision is taken on the measured size)")
        if self.action not in ("repair", "none"):
            raise ValueError("action must be 'repair' or 'none'")
        ok = self.observed in (None, "not_detected", "detected") or (
            isinstance(self.observed, tuple) and len(self.observed) == 2 and self.observed[0] == "sized"
        )
        if not ok:
            raise ValueError("observed must be None, 'not_detected', 'detected' or ('sized', a_measured)")
        if isinstance(self.observed, tuple) and self.sizing_error is None:
            raise ValueError("a sized outcome needs a sizing_error")
        if self.repair_flaw is not None and not (self.repair_flaw[0] > 0 and self.repair_flaw[1] > 0):
            raise ValueError("repair_flaw must be (a, 2c) in metres, both positive")

    @property
    def draws(self) -> int:
        """Random variables of this inspection: the detection score, plus a sizing draw when needed."""
        return 2 if (self.sizing_error is not None and not isinstance(self.observed, tuple)) else 1


@dataclass(frozen=True)
class _Invariants:
    problem: Assessment
    law: GrowthLaw
    spectrum: Spectrum


@dataclass(frozen=True)
class Curve:
    """One probability against time with pointwise 99% Clopper-Pearson intervals."""

    times: FloatArray
    value: FloatArray
    lower: FloatArray
    upper: FloatArray
    kind: str


@dataclass(frozen=True)
class GrowthHistory:
    """Deterministic growth (``grow``): sizes at the checkpoint times, events and the driving variable."""

    times: FloatArray
    a: FloatArray
    c: FloatArray
    kind: IntArray
    g: FloatArray  # driving variable at each checkpoint under the given load (NaN without a load)
    t_leak: float
    t_leak_assumed: float
    t_ended: float
    leak_trigger: int
    first_failure: float  # first failing or invalid checkpoint, end of validity, or leak if it counts; NaN if none
    cycles_applied: float

    @property
    def life(self) -> float:
        return self.first_failure


@dataclass(frozen=True)
class TimeDependentResult:
    times: FloatArray
    leak_is_failure: bool
    cumulative_upper: Curve  # headline: upper pairing, out-of-validity counted as failed
    cumulative_lower: Curve
    cumulative_valid_only: Curve  # upper pairing, out-of-validity counted as safe (the valid-only bound)
    interval_probability: Curve  # upper pairing (approximation, not a bound)
    hazard_upper: Curve  # within the upper pairing: exact for that discretised model (sensitivity)
    hazard_lower: Curve  # within the lower pairing
    hazard_conservative_upper: FloatArray  # conservative pair (bracket under assumption A)
    hazard_conservative_lower: FloatArray
    discordant: IntArray
    refine: BoolArray
    modes: dict[str, float]
    events: dict[str, float]  # path events before the sample's first failure (upper pairing)
    events_any: dict[str, Any]  # path events regardless of failure (and "by_trigger": shares per boundary code)
    assumption_violations: int
    assumptions: tuple[str, ...]
    inspections: tuple[dict[str, Any], ...]
    samples: int
    delta: float
    refinements: int = 0
    subset: ReliabilityResult | None = None
    subset_curve: dict[str, Any] | None = None
    conditioning: dict[str, Any] | None = None
    weighted: bool = False
    sensitivity: dict[str, Curve] | None = None  # headline at stage correlation 0 and 1
    knee_shift_log10: float | None = None
    invalid_start: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Time-dependent failure probability per flaw ({self.samples} samples; leak counted as failure: "
            f"{self.leak_is_failure}; load-process step {self.delta:g} years, refined {self.refinements} times)",
            "cumulative Pf(0, t): bracket under assumption A [lower, upper pairing], each with its 99% interval;",
            "hazard per step: within each pairing (exact for that discretised model) and the conservative pair",
        ]
        for i, t in enumerate(self.times):
            lines.append(
                f"  t = {t:g}: Pf [{self.cumulative_lower.value[i]:.3g}, {self.cumulative_upper.value[i]:.3g}]  "
                f"upper interval [{self.cumulative_upper.lower[i]:.3g}, {self.cumulative_upper.upper[i]:.3g}]  "
                f"valid-only {self.cumulative_valid_only.value[i]:.3g}  "
                f"hazard [{self.hazard_lower.value[i]:.3g}, {self.hazard_upper.value[i]:.3g}]  "
                f"conservative pair [{self.hazard_conservative_lower[i]:.3g}, {self.hazard_conservative_upper[i]:.3g}]"
                + ("  (bracket under assumption A is wide; refine the load-process step)" if self.refine[i] else "")
            )
        lines.append("first failure mode (upper pairing): " + ", ".join(f"{k} {v:.3g}" for k, v in self.modes.items()))
        lines.append("events before the first failure: " + ", ".join(f"{k} {v:.3g}" for k, v in self.events.items()))
        lines.append(
            "events at any time: "
            + ", ".join(f"{k} {v:.3g}" for k, v in self.events_any.items() if isinstance(v, float))
            + (f"; by trigger code {self.events_any['by_trigger']}" if self.events_any.get("by_trigger") else "")
        )
        if self.invalid_start:
            lines.append(f"initial flaws outside the solution's validity: {self.invalid_start:.3g}")
        nan = float("nan")
        for insp in self.inspections:
            lines.append(
                f"inspection at t = {insp.get('time', nan):g}: P(detect) over all N samples, eligibility under the "
                f"lower pairing {insp.get('p_detect_all', nan):.3g} (upper {insp.get('p_detect_all_upper', nan):.3g}), "
                f"over flawed survivors lower {insp.get('p_detect_flawed_survivors', nan):.3g} "
                f"(upper {insp.get('p_detect_flawed_survivors_upper', nan):.3g}); "
                f"P(repair) all {insp.get('p_repair_all', nan):.3g} (upper {insp.get('p_repair_all_upper', nan):.3g}); "
                f"Pf before [{insp.get('cumulative_lower_before', nan):.3g}, "
                f"{insp.get('cumulative_upper_before', nan):.3g}], "
                f"after the next step [{insp.get('cumulative_lower_after_next_step', nan):.3g}, "
                f"{insp.get('cumulative_upper_after_next_step', nan):.3g}], "
                f"next-step hazard [{insp.get('hazard_lower_next_step', nan):.3g}, "
                f"{insp.get('hazard_upper_next_step', nan):.3g}], "
                f"interval probability {insp.get('interval_probability_next_step', nan):.3g}"
            )
        if self.sensitivity:
            lines.append(
                "stage-correlation sensitivity (final time, upper pairing): "
                + ", ".join(f"{k} {v.value[-1]:.3g}" for k, v in self.sensitivity.items())
                + (f"; knee shift {self.knee_shift_log10:.3g} decades" if self.knee_shift_log10 is not None else "")
            )
        if self.subset_curve:
            lines.append(f"subset simulation: {self.subset_curve.get('label', 'see subset_curve')}")
            below = self.subset_curve.get("below_resolution")
            if below is not None:
                grid = self.subset_curve["grid"]
                under = [f"{float(grid[i]):g}" for i in np.nonzero(below)[0]]
                lines.append(
                    f"  resolution floor {self.subset_curve['resolution_floor']:.3g} of the lifetime probability; "
                    + (
                        f"below resolution (curve value withheld) at t = {', '.join(under)}"
                        if under
                        else "no time below resolution"
                    )
                )
        lines.extend(self.assumptions)
        if self.conditioning:
            c = self.conditioning
            lines.append(f"conditioning: {c.get('conditioning')}; {c.get('label_upper', '')}")
            if c.get("pf_after_upper") is not None:
                lines.append(
                    f"  Pf after conditioning (upper) {c['pf_after_upper']:.3g}"
                    + (f", lower {c['pf_after_lower']:.3g}" if c.get("pf_after_lower") is not None else "")
                )
            if c.get("withheld_upper"):
                lines.append(f"  withheld: {c.get('samples_needed_upper')} samples needed")
        return "\n".join(lines)


def _cylinder_spec(problem: Assessment) -> CylinderSpec | None:
    comp, flaw = problem.component, problem.flaw
    if not isinstance(comp, Cylinder):
        return None
    position = "internal" if isinstance(flaw, CylinderThroughFlaw) else flaw.position  # type: ignore[union-attr]
    return CylinderSpec(
        float(np.asarray(comp.r_i, dtype=float)),
        flaw.orientation,  # type: ignore[union-attr]
        position,
        comp.k_route,
        problem.loads.bending_tensile_surface,
    )


def _start_state(problem: Assessment, n: int, blocks: int) -> GrowthState:
    """The initial growth state of n samples for any supported flaw type."""
    flaw = problem.flaw
    thickness = np.broadcast_to(np.asarray(problem.component.thickness, dtype=float), (n,))
    width = np.broadcast_to(np.asarray(problem.component.width, dtype=float), (n,))
    if isinstance(flaw, EmbeddedFlaw):
        a0 = np.broadcast_to(0.5 * np.asarray(flaw.two_a, dtype=float), (n,)).copy()
        c0 = np.broadcast_to(0.5 * np.asarray(flaw.two_c, dtype=float), (n,)).copy()
        p0 = np.broadcast_to(np.asarray(flaw.p, dtype=float), (n,)).copy()
        return GrowthState.start(a0, c0, thickness, width, blocks, ligament=p0)
    if isinstance(flaw, SurfaceFlaw | CylinderSurfaceFlaw):
        a0 = np.broadcast_to(np.asarray(flaw.a, dtype=float), (n,)).copy()
        c0 = np.broadcast_to(0.5 * np.asarray(flaw.two_c, dtype=float), (n,)).copy()
        state = GrowthState.start(a0, c0, thickness, width, blocks)
        if isinstance(flaw, CylinderSurfaceFlaw):
            # start validity: the cylinder assessment's own range decides (the plate check of start() is not it)
            res = problem.run(on_invalid="mask")
            valid = np.broadcast_to(np.asarray(res.valid, dtype=bool), (n,))
            state.kind[:] = np.where(valid, SURFACE, ENDED)
            state.invalid_start[:] = ~valid
            state.t_ended[:] = np.where(valid, np.nan, 0.0)
        return state
    if isinstance(flaw, CylinderThroughFlaw):
        c0 = np.broadcast_to(0.5 * np.asarray(flaw.two_a, dtype=float), (n,)).copy()
        state = GrowthState.start(thickness.copy(), c0, thickness, width, blocks)
        res = problem.run(on_invalid="mask")
        valid = np.broadcast_to(np.asarray(res.valid, dtype=bool), (n,))
        state.kind[:] = np.where(valid, THROUGH, ENDED)
        state.invalid_start[:] = ~valid
        state.t_ended[:] = np.where(valid, np.nan, 0.0)
        return state
    assert isinstance(flaw, ExtendedFlaw)
    a0 = np.broadcast_to(np.asarray(flaw.a, dtype=float), (n,)).copy()
    state = GrowthState.start(a0, width.copy(), thickness, width, blocks)
    res = problem.run(on_invalid="mask")
    valid = np.broadcast_to(np.asarray(res.valid, dtype=bool), (n,))
    state.kind[:] = np.where(valid, EXTENDED, ENDED)
    state.invalid_start[:] = ~valid
    state.t_ended[:] = np.where(valid, np.nan, 0.0)
    return state


class TimeDependentAssessment:
    """An :class:`Assessment` whose flaw grows by fatigue, checked against an extreme load per step.

    ``problem``: the static assessment (its flaw is the initial flaw; the stress component named by
    ``extreme`` is replaced per step by the step maximum). ``times``: reporting times (years), multiples
    of the load-process step ``delta`` (default: the extreme load's reference period).
    ``leak_is_failure`` is required. ``shared_scale`` is required when ``spectrum.load_scale`` is random.
    """

    def __init__(
        self,
        problem: Assessment,
        law: GrowthLaw,
        spectrum: Spectrum,
        extreme: ExtremeLoad,
        times: Sequence[float],
        leak_is_failure: bool,
        inspections: Sequence[Inspection] = (),
        delta: float | None = None,
        shared_scale: bool | None = None,
        eta: float = 0.02,
        tol: float = 2e-4,
    ) -> None:
        cylinder_flaw = isinstance(problem.flaw, CylinderSurfaceFlaw | CylinderThroughFlaw | ExtendedFlaw)
        if not isinstance(problem.flaw, SurfaceFlaw | EmbeddedFlaw) and not cylinder_flaw:
            raise NotImplementedError("TODO(verify): fatigue growth is implemented for plate and cylinder flaws")
        if cylinder_flaw != isinstance(problem.component, Cylinder):
            raise TypeError("cylinder flaws need a Cylinder component and plate flaws a FlatPlate")
        if cylinder_flaw:
            assert isinstance(problem.component, Cylinder)
            if is_random(problem.component.r_i):
                raise NotImplementedError("TODO(verify): a random inner radius in fatigue growth (not vectorised)")
        if isinstance(problem.flaw, EmbeddedFlaw) and cylinder_flaw:
            raise NotImplementedError("TODO(verify): embedded flaws in cylinders (the plate solution with M = 1)")
        if isinstance(problem.flaw, EmbeddedFlaw):
            # bending-loaded embedded flaws are refused in every path: no bending range in the
            # spectrum and no bending component in the fracture-check loads
            if bool(np.any(spectrum.bending != 0.0)) or extreme.component in ("pb", "qb"):
                raise NotImplementedError(
                    "TODO(verify): M_b at the major-axis ends of an embedded flaw; growth under bending is not assessed"
                )
        if not isinstance(leak_is_failure, bool):
            raise ValueError("leak_is_failure must be given explicitly (True or False)")
        t = np.asarray(sorted(float(x) for x in times), dtype=float)
        if t.size == 0 or t[0] <= 0.0 or np.any(np.diff(t) <= 0.0):
            raise ValueError("times must be positive and increasing")
        self.delta = float(delta) if delta is not None else float(extreme.reference_period_years)
        if not 0.0 < self.delta < np.inf:
            raise ValueError("delta must be positive and finite")
        grid = np.arange(1, int(round(t[-1] / self.delta)) + 1) * self.delta
        if not np.all(np.isclose(t[:, None], grid[None, :], rtol=1e-9).any(axis=1)):
            raise ValueError("every reporting time must be a multiple of the load-process step")
        self.times = t
        self.grid: FloatArray = np.asarray(grid, dtype=float)
        self.problem, self.law, self.spectrum, self.extreme = problem, law, spectrum, extreme
        self.leak_is_failure = leak_is_failure
        self.eta, self.tol = eta, tol
        if spectrum.kind == "ordered" and not np.isclose(spectrum.total_years(), t[-1], rtol=1e-9):
            raise ValueError("the blocks of an ordered spectrum must sum to the last requested time")
        random_scale = isinstance(spectrum.load_scale, Distribution)
        if random_scale and shared_scale is None:
            raise ValueError("shared_scale (True or False) is required when the load scale is random")
        self.shared_scale = bool(shared_scale) if random_scale else False
        self._random_scale = random_scale
        for insp in inspections:
            if not np.any(np.isclose(insp.time, grid, rtol=1e-9)):
                raise ValueError("inspections must be at load-process step boundaries")
        self.inspections = tuple(sorted(inspections, key=lambda i: i.time))
        self.invariants = _Invariants(problem, law, spectrum)
        self.variables = collect(self.invariants)
        self.vector = RandomVector(self.variables) if self.variables else None
        if self.vector is not None and law.stage_correlation is not None and law.is_random:
            assert law.a_coefficient_b is not None
            rho = float(np.clip(law.stage_correlation, -(1 - 1e-9), 1 - 1e-9))  # +-1 exactly is singular
            self.vector.correlate(law.a_coefficient, law.a_coefficient_b, rho, kind="latent")
        if spectrum.life_factor != 1.0 and self.variables:
            raise ValueError(
                "life_factor is for deterministic runs only (it would stack a fixed factor on sampled scatter)"
            )
        self.n_invariant = len(self.variables)
        self.n_steps = grid.size
        self.n_inspection_draws = sum(i.draws for i in self.inspections)
        self.dimension = self.n_invariant + self.n_steps + self.n_inspection_draws

    def correlate(
        self, x: Distribution, y: Distribution, rho: float, kind: Literal["pearson", "latent"] = "pearson"
    ) -> TimeDependentAssessment:
        if self.vector is None:
            raise ValueError("no random variables")
        self.vector.correlate(x, y, rho, kind)
        return self

    def _realised(self, u_inv: FloatArray) -> tuple[Assessment, GrowthLaw, Spectrum]:
        if self.vector is None:
            return self.problem, self.law, self.spectrum
        x = self.vector.to_physical(u_inv)
        values = {id(v): x[:, i] for i, v in enumerate(self.variables)}
        inv: _Invariants = realise(self.invariants, values)
        return inv.problem, inv.law, inv.spectrum

    # ---- the run: growth, checkpoints, both pairings ----------------------------------------

    def run(self, u: FloatArray, steps: FloatArray | None = None) -> dict[str, Any]:
        """Grow every sample through the life and evaluate every checkpoint under both pairings."""
        u = np.atleast_2d(np.asarray(u, dtype=float))
        n = u.shape[0]
        grid = self.grid if steps is None else np.asarray(steps, dtype=float)
        K = grid.size
        delta = float(grid[0])
        if u.shape[1] != self.n_invariant + K + self.n_inspection_draws:
            raise ValueError(f"expected {self.n_invariant + K + self.n_inspection_draws} columns")
        u_inv = u[:, : self.n_invariant]
        u_steps = u[:, self.n_invariant : self.n_invariant + K]
        u_insp = u[:, self.n_invariant + K :]
        problem, law, spectrum = self._realised(u_inv)
        state = _start_state(problem, n, spectrum.membrane.size)
        a0, c0 = state.a.copy(), state.c.copy()  # stand-ins for samples without an event
        grower = Grower(law, spectrum, n, float(grid[-1]), self.eta, self.tol, cylinder=_cylinder_spec(problem))
        grower.bind(state)
        scale = np.broadcast_to(np.asarray(spectrum.load_scale, dtype=float), (n,)) if self.shared_scale else np.ones(n)
        g_min = np.full((n, K), np.inf)  # most severe valid checkpoint
        g_max = np.full((n, K), -np.inf)  # least severe valid checkpoint
        any_invalid = np.zeros((n, K), dtype=bool)
        mode = np.full((n, K), "", dtype=object)  # exit at the most severe checkpoint
        leak_step = np.full(n, -1)
        ended_step = np.full(n, -1)
        violations = 0
        violating_samples = np.zeros(n, dtype=bool)
        inspected: list[dict[str, Any]] = []
        repaired = np.zeros(n, dtype=bool)
        t_repair = np.full(n, np.nan)
        removed_leaked = np.zeros(n, dtype=bool)  # history of flaws removed by a repair
        codes_seen = np.zeros((n, 32), dtype=bool)  # trigger codes of every event of every sample
        removed_ended = np.zeros(n, dtype=bool)
        fail_lower_sofar = np.zeros(n, dtype=bool)  # first passage under the lower pairing, invalid as failed
        fail_upper_sofar = np.zeros(n, dtype=bool)
        draw = 0
        for k in range(K):
            t_start = float(grid[k - 1]) if k > 0 else 0.0
            t_end = float(grid[k])
            state.new_segment()
            snapshots = [state.snapshot()]
            for frac in INTERIOR:
                grower.advance(state, t_start + frac * (t_end - t_start))
                snapshots.append(state.snapshot())
            grower.advance(state, t_end)
            snapshots.append(state.snapshot())
            for e in range(EVENT_CAPACITY):
                has_e = state.event_count > e
                codes_seen[np.nonzero(has_e)[0], state.event_trigger[has_e, e]] = True
            m = self.extreme.step_maximum(u_steps[:, k], delta) * scale
            evaluated = [self._driving(problem, grower, snap, m) for snap in snapshots]
            n_plain = len(evaluated)
            # both sides of every flaw-type change in the step
            for e in range(EVENT_CAPACITY):
                has = state.event_count > e
                if not np.any(has):
                    break
                idx = np.nonzero(has)[0]
                for side in (state.event_pre, state.event_post):
                    snap = (
                        np.where(has, side[:, e, 0], a0),
                        np.where(has, side[:, e, 1], c0),
                        np.where(has, side[:, e, 2], 0).astype(np.intp),
                    )
                    g_e, v_e, x_e = self._driving(problem, grower, snap, m, only=idx)
                    g_e[~has], v_e[~has] = np.nan, True  # samples without this event are untouched
                    evaluated.append((g_e, v_e, x_e))
            g_stack = np.stack([g for g, _, _ in evaluated], axis=1)
            v_stack = np.stack([v for _, v, _ in evaluated], axis=1)
            x_stack = np.stack([x for _, _, x in evaluated], axis=1)
            usable = v_stack & ~np.isnan(g_stack)  # +inf (no flaw) is a usable, safe value
            g_min[:, k] = np.min(np.where(usable, g_stack, np.inf), axis=1)
            g_max[:, k] = np.max(np.where(usable, g_stack, -np.inf), axis=1)
            present = np.ones_like(v_stack)
            present[:, n_plain:] = ~np.isnan(g_stack[:, n_plain:]) | ~v_stack[:, n_plain:]
            any_invalid[:, k] = np.any(~v_stack & present, axis=1)
            most = np.argmin(np.where(usable, g_stack, np.inf), axis=1)
            mode[:, k] = x_stack[np.arange(n), most]
            # assumption A: severity non-increasing between consecutive plain checkpoints; an event in
            # the step breaks the sequence (both sides of it are checkpoints)
            seq = g_stack[:, :n_plain]
            with np.errstate(invalid="ignore"):
                rises = np.diff(seq, axis=1) > 1e-9 * np.maximum(np.abs(seq[:, :-1]), 1.0)
            rises &= np.isfinite(seq[:, :-1]) & np.isfinite(seq[:, 1:])
            bad = np.any(rises, axis=1) & (state.event_count == 0)
            violations += int(np.sum(bad))
            violating_samples |= bad
            for t_ev in (state.t_leak, state.t_leak_assumed):
                in_step = (t_ev > t_start) & (t_ev <= t_end) & (leak_step < 0)
                leak_step[in_step] = k
            in_step = (state.t_ended > t_start) & (state.t_ended <= t_end) & (ended_step < 0)
            ended_step[in_step] = k
            fail_lower_k = (g_max[:, k] <= 0.0) | any_invalid[:, k]
            if self.leak_is_failure:
                fail_lower_k |= leak_step == k
            fail_lower_sofar |= fail_lower_k
            fail_upper_k = (g_min[:, k] <= 0.0) | any_invalid[:, k]
            if self.leak_is_failure:
                fail_upper_k |= leak_step == k
            fail_upper_sofar |= fail_upper_k
            # inspections at the end of this step: fracture check first, then inspection, then repair,
            # on the samples alive under the lower pairing (the superset of survivors)
            for insp in self.inspections:
                if not np.isclose(insp.time, t_end, rtol=1e-9):
                    continue
                cols = u_insp[:, draw : draw + insp.draws]
                draw += insp.draws
                info = self._inspect(state, insp, cols, ~fail_lower_sofar, ~fail_upper_sofar)
                repaired_now = info.pop("repaired_now")
                repaired |= repaired_now
                t_repair[repaired_now] = t_end
                removed_leaked |= repaired_now & (np.isfinite(state.t_leak) | np.isfinite(state.t_leak_assumed))
                removed_ended |= repaired_now & np.isfinite(state.t_ended)
                # the removed flaw's record is cleared: the repair flaw's own breakthrough is a new event
                for arr in (state.t_leak, state.t_leak_assumed, state.t_ended):
                    arr[repaired_now] = np.nan
                state.leak_trigger[repaired_now] = 0
                state.depth_growing_at_recharacterisation[repaired_now] = False
                state.slid[repaired_now] = False
                state.clamped[repaired_now] = False
                leak_step[repaired_now] = -1
                ended_step[repaired_now] = -1
                info["step"] = k
                inspected.append(info)
        return {
            "g_min": g_min,
            "g_max": g_max,
            "any_invalid": any_invalid,
            "mode": mode,
            "leak_step": leak_step,
            "ended_step": ended_step,
            "violations": violations,
            "violating_samples": int(np.sum(violating_samples)),
            "inspections": inspected,
            "repaired": repaired,
            "t_repair": t_repair,
            "removed_leaked": removed_leaked,
            "codes_seen": codes_seen,
            "removed_ended": removed_ended,
            "state": state,
            "problem": problem,
            "invalid_start": float(np.mean(state.invalid_start)),
            "grid": grid,
            "n": n,
        }

    def _driving(
        self,
        problem: Assessment,
        grower: Grower,
        snap: tuple[FloatArray, FloatArray, IntArray],
        m: FloatArray,
        only: IntArray | None = None,
    ) -> tuple[FloatArray, BoolArray, NDArray[Any]]:
        """Driving variable (radial margin - 1, sign consistent with the acceptance test), validity and
        exit code of every sample at a configuration under the step maximum ``m``. Boundary states are
        evaluated just inside the engine's open validity limits. Repaired samples: +inf and valid."""
        a, c, kind = snap
        n = a.size
        g = np.full(n, np.inf)
        valid = np.ones(n, dtype=bool)
        exit_code = np.full(n, "none", dtype=object)
        thickness = np.broadcast_to(np.asarray(problem.component.thickness, dtype=float), (n,))
        width = np.broadcast_to(np.asarray(problem.component.width, dtype=float), (n,))
        b_eff = grower.b_eff if grower.b_eff.shape == a.shape else None
        a, c = grower.inside_state(a, c, kind, thickness, width, b_eff)
        loads = problem.loads.with_(**{self.extreme.component: m})
        base = problem.flaw
        for which, flaw_kind in (
            (SURFACE, "surface"),
            (THROUGH, "through"),
            (EMBEDDED, "embedded"),
            (EXTENDED, "extended"),
        ):
            sel = kind == which
            if only is not None:
                mask = np.zeros(n, dtype=bool)
                mask[only] = True
                sel &= mask
            if not np.any(sel):
                continue
            if flaw_kind == "surface" and isinstance(base, CylinderSurfaceFlaw | CylinderThroughFlaw | ExtendedFlaw):
                flaw: Any = CylinderSurfaceFlaw(
                    a=np.where(sel, a, 1e-3),
                    two_c=2.0 * np.where(sel, c, 2e-3),
                    orientation=base.orientation,
                    position=getattr(base, "position", "internal"),
                )
            elif flaw_kind == "through" and isinstance(base, CylinderSurfaceFlaw | CylinderThroughFlaw | ExtendedFlaw):
                flaw = CylinderThroughFlaw(two_a=np.where(sel, 2.0 * c, 2e-3), orientation=base.orientation)
            elif flaw_kind == "extended":
                if not isinstance(base, ExtendedFlaw):
                    continue
                flaw = ExtendedFlaw(a=np.where(sel, a, 1e-3), orientation=base.orientation, position=base.position)
            elif flaw_kind == "surface":
                flaw = SurfaceFlaw(a=np.where(sel, a, 1e-3), two_c=2.0 * np.where(sel, c, 2e-3))
            elif flaw_kind == "embedded":
                assert b_eff is not None
                p_lig = np.maximum(0.5 * (b_eff - 2.0 * a), 0.0)
                p_eval = np.where(sel, np.maximum(p_lig, 1e-12), 0.2 * thickness)
                flaw = EmbeddedFlaw(two_a=2.0 * np.where(sel, a, 1e-3), two_c=2.0 * np.where(sel, c, 4e-3), p=p_eval)
            else:
                flaw = ThroughFlaw(two_a=np.where(sel, 2.0 * c, 2e-3))
            res = dataclasses.replace(problem, flaw=flaw, loads=loads).run(on_invalid="mask")
            fail = ~res.acceptable & res.valid
            y = consistent_driving_variable(np.where(res.valid, res.radial_margin - 1.0, 0.0), fail)
            g[sel] = np.where(res.valid, y, np.nan)[sel]
            valid[sel] = res.valid[sel]
            exit_code[sel] = np.asarray(res.exit, dtype=object)[sel]
        ended = kind == ENDED
        g[ended], valid[ended], exit_code[ended] = np.nan, False, "invalid"
        return g, valid, exit_code

    def _inspect(
        self, state: GrowthState, insp: Inspection, cols: FloatArray, alive: BoolArray, alive_upper: BoolArray
    ) -> dict[str, Any]:
        n = state.a.size
        has_flaw = np.isin(state.kind, GROWING) & alive
        surf = ((state.kind == SURFACE) | (state.kind == EXTENDED)) & alive  # inspected by depth
        thr = (state.kind == THROUGH) & alive
        emb = (state.kind == EMBEDDED) & alive
        if np.any(emb):
            if insp.pod_embedded is None:
                raise ValueError("an inspection of embedded flaws needs pod_embedded (a PoD of the flaw height 2a)")
        if (isinstance(insp.observed, tuple) or insp.accept_below is not None) and np.any(thr):
            raise NotImplementedError("TODO(verify): sizing of a through flaw (the measured quantity is not defined)")
        pod = np.zeros(n)
        pod[surf] = np.clip(insp.pod(state.a[surf]), 0.0, 1.0)
        if np.any(emb):
            assert insp.pod_embedded is not None
            pod[emb] = np.clip(insp.pod_embedded(2.0 * state.a[emb]), 0.0, 1.0)
        pod[thr] = 1.0 if insp.pod_through is None else np.clip(insp.pod_through(2.0 * state.c[thr]), 0.0, 1.0)
        score = stats.norm.cdf(cols[:, 0])
        simulated = has_flaw & (score < pod)
        detected = simulated.copy()
        retained = alive.copy()
        weights = None
        if insp.observed == "not_detected":
            retained = alive & ~simulated  # flaw-free samples are never detected
            detected = np.zeros(n, dtype=bool)
        elif insp.observed == "detected":
            retained = alive & simulated
            detected = has_flaw.copy()
        elif isinstance(insp.observed, tuple):
            detected = has_flaw.copy()
            a_m = float(insp.observed[1])
            assert insp.sizing_error is not None
            inspected = np.where(state.kind == EMBEDDED, 2.0 * state.a, state.a)  # height for embedded flaws
            weights = np.where(has_flaw, pod * insp.sizing_error.pdf(a_m - inspected), 0.0)  # flaw-free: 0
        to_repair = detected & (insp.action == "repair")
        if insp.accept_below is not None and np.any(to_repair):
            assert insp.sizing_error is not None
            if isinstance(insp.observed, tuple):
                measured = np.full(n, float(insp.observed[1]))
            else:
                inspected = np.where(state.kind == EMBEDDED, 2.0 * state.a, state.a)
                measured = inspected + insp.sizing_error.from_normal_score(cols[:, 1])  # a separate draw
            to_repair &= measured >= insp.accept_below
        if np.any(to_repair):
            for name in ("sliding", "last_change", "ignore_branch", "zero_cuts", "force", "has_override"):
                getattr(state, name)[to_repair] = 0
            state.h_next[to_repair] = np.nan
            state.dormant[to_repair] = False
            if insp.repair_flaw is None:
                state.kind[to_repair] = NONE
            else:
                ok = plate.surface_flaw_valid(
                    insp.repair_flaw[0], 0.5 * insp.repair_flaw[1], state.thickness[to_repair], state.width[to_repair]
                )
                if not np.all(ok):
                    raise ValueError("repair_flaw lies outside the surface solution's validity for this component")
                state.kind[to_repair] = SURFACE
                state.a[to_repair] = insp.repair_flaw[0]
                state.c[to_repair] = 0.5 * insp.repair_flaw[1]
                state.dormant[to_repair] = False
                state.h_next[to_repair] = np.nan
        flawed_alive = int(np.sum(has_flaw))
        flawed_upper = int(np.sum(has_flaw & alive_upper))
        return {
            "time": insp.time,
            "observed": insp.observed,
            "p_detect_all": float(np.mean(detected)),
            "p_repair_all": float(np.mean(to_repair)),
            "p_detect_all_upper": float(np.mean(detected & alive_upper)),
            "p_repair_all_upper": float(np.mean(to_repair & alive_upper)),
            "p_detect_flawed_survivors": float(np.sum(detected) / flawed_alive) if flawed_alive else float("nan"),
            "p_repair_flawed_survivors": float(np.sum(to_repair) / flawed_alive) if flawed_alive else float("nan"),
            "p_detect_flawed_survivors_upper": float(np.sum(detected & alive_upper) / flawed_upper)
            if flawed_upper
            else float("nan"),
            "p_repair_flawed_survivors_upper": float(np.sum(to_repair & alive_upper) / flawed_upper)
            if flawed_upper
            else float("nan"),
            "survivors_lower": int(np.sum(alive)),
            "survivors_upper": int(np.sum(alive_upper)),
            "denominators": "'_all': every sample; '_flawed_survivors': flawed samples alive under the lower pairing; "
            "'_flawed_survivors_upper': the same under the upper pairing",
            "retained": retained,
            "weights": weights,
            "repaired_now": to_repair,
        }

    # ---- deterministic growth --------------------------------------------------------------

    def grow(self, load: float | None = None) -> GrowthHistory:
        """Deterministic growth (no random variable in the problem): sizes and events at the checkpoint
        times, and the driving variable under a fixed fracture-check load ``load`` (Pa) if given. The
        life follows the headline treatment of ``solve``: the first checkpoint (plain, or either side of
        a flaw-type change) that fails or lies outside the solutions' validity, the end of validity of
        the growth solution, and a leak when leaks count. Inspection policies are not simulated here."""
        if self.variables:
            raise ValueError("grow() is for deterministic problems; use solve() with random variables")
        if self.inspections:
            raise ValueError("grow() does not simulate inspections; use solve(), or an assessment without them")
        state = _start_state(self.problem, 1, self.spectrum.membrane.size)
        grower = Grower(
            self.law, self.spectrum, 1, float(self.grid[-1]), self.eta, self.tol, cylinder=_cylinder_spec(self.problem)
        )
        grower.bind(state)
        times, a, c, kind, g = [0.0], [float(state.a[0])], [float(state.c[0])], [int(state.kind[0])], []
        m = np.array([load if load is not None else np.nan])
        fail_times: list[float] = []

        def check(snap: tuple[FloatArray, FloatArray, IntArray], t: float) -> float:
            """The driving variable at a state; records t if the state fails or is outside the
            assessment's validity (the headline treatment of ``solve``)."""
            g_s, v_s, _ = self._driving(self.problem, grower, snap, m)
            if (v_s[0] and g_s[0] <= 0.0) or (not v_s[0] and snap[2][0] != ENDED):
                fail_times.append(t)
            return float(g_s[0])

        def record(t: float) -> None:
            g.append(np.nan if load is None else check(state.snapshot(), t))

        record(0.0)
        for k in range(self.n_steps):
            t_start = float(self.grid[k - 1]) if k > 0 else 0.0
            t_end = float(self.grid[k])
            state.new_segment()
            for frac in (*INTERIOR, 1.0):
                t_k = t_start + frac * (t_end - t_start)
                grower.advance(state, t_k)
                times.append(t_k)
                a.append(float(state.a[0]))
                c.append(float(state.c[0]))
                kind.append(int(state.kind[0]))
                record(t_k)
            if load is not None:  # both sides of every flaw-type change in the step
                for e in range(int(state.event_count[0])):
                    for side in (state.event_pre, state.event_post):
                        snap = (side[0:1, e, 0].copy(), side[0:1, e, 1].copy(), side[0:1, e, 2].astype(np.intp))
                        check(snap, float(state.event_time[0, e]))
        g_arr = np.asarray(g, dtype=float)
        candidates = fail_times + [float(state.t_ended[0])]  # end of the growth solution's validity: failed
        if self.leak_is_failure:
            candidates += [float(state.t_leak[0]), float(state.t_leak_assumed[0])]
        finite = [x for x in candidates if np.isfinite(x)]
        first = min(finite) if finite else float("nan")
        life = first if np.isfinite(first) else float(self.grid[-1])
        cycles = self.spectrum.cycles_per_year * life
        if self.spectrum.kind == "ordered":
            ends = self.spectrum.block_end_times()
            starts = np.concatenate([[0.0], ends[:-1]])
            cycles = float(np.sum(np.clip(life - starts, 0.0, ends - starts) * self.spectrum.cycles_per_year))
        return GrowthHistory(
            np.asarray(times),
            np.asarray(a),
            np.asarray(c),
            np.asarray(kind, dtype=np.intp),
            g_arr,
            float(state.t_leak[0]),
            float(state.t_leak_assumed[0]),
            float(state.t_ended[0]),
            int(state.leak_trigger[0]),
            first,
            cycles,
        )

    def compare_orders(self, load: float) -> dict[str, GrowthHistory]:
        """Ordered spectra, deterministic: the life under each of the three declared orders and the cycles
        applied before failure. The shortest is the worst of these three orders, not of all permutations."""
        if self.spectrum.kind != "ordered":
            raise ValueError("compare_orders applies to an ordered spectrum")
        if self.inspections:
            raise ValueError("compare_orders does not simulate inspections; use an assessment without them")
        blocks = np.column_stack([self.spectrum.membrane, self.spectrum.bending, self.spectrum.cycles])
        out: dict[str, GrowthHistory] = {}
        for order in ("as_given", "largest_first", "smallest_first"):
            sp = Spectrum.ordered(
                blocks,
                self.spectrum.cycles_per_year,
                order,
                ktm=self.spectrum.ktm,
                ktb=self.spectrum.ktb,
                km=self.spectrum.km,
                load_scale=self.spectrum.load_scale,
                life_factor=self.spectrum.life_factor,
            )
            run = TimeDependentAssessment(
                self.problem,
                self.law,
                sp,
                self.extreme,
                list(self.times),
                self.leak_is_failure,
                (),
                self.delta,
                None,
                self.eta,
                self.tol,
            )
            out[order] = run.grow(load)
        return out

    # ---- Monte Carlo channel ------------------------------------------------------------------

    def solve(
        self,
        seed: int = 0,
        samples: int = 20_000,
        subset: bool = False,
        refine: bool = False,
        replicates: int = 8,
        samples_per_level: int = 2000,
        stage_sensitivity: bool = True,
        curve_times: Sequence[float] = (),
    ) -> TimeDependentResult:
        if self.spectrum.life_factor != 1.0:
            raise ValueError("life_factor is for deterministic runs only (grow / compare_orders)")
        rng = np.random.default_rng(seed)
        u = rng.standard_normal((samples, self.dimension))
        grid: FloatArray = self.grid
        run = self.run(u, grid)
        result = self._assemble(run, grid, 0)
        refinements = 0
        while refine and self._needs_refinement(result) and refinements < MAX_REFINEMENTS:
            u, grid = self._refined(u, grid, rng)
            refinements += 1
            run = self.run(u, grid)
            result = self._assemble(run, grid, refinements)
        if stage_sensitivity and self.law.is_random and self.law.two_stage:
            result = dataclasses.replace(
                result, sensitivity=self._sensitivity(seed, samples, grid), knee_shift_log10=self._knee_shift()
            )
        if subset:
            if self.leak_is_failure:
                raise ValueError("subset simulation in time is not offered when leak is failure")
            extra = self._subset(seed, samples, grid, replicates, samples_per_level, curve_times)
            result = dataclasses.replace(result, **extra)
        return result

    @staticmethod
    def _needs_refinement(result: TimeDependentResult) -> bool:
        at_conditioning = bool(result.conditioning and result.conditioning.get("refine_at_conditioning_time"))
        return bool(np.any(result.refine)) or at_conditioning

    def _refined(self, u: FloatArray, grid: FloatArray, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
        """Halve the load-process step with the coupled draw: the maximum of the two sub-step maxima equals
        the coarse maximum (scores Phi^-1(sqrt p) and Phi^-1(V sqrt p), V uniform, in random order)."""
        k = grid.size
        u_inv, u_steps, u_insp = (
            u[:, : self.n_invariant],
            u[:, self.n_invariant : self.n_invariant + k],
            u[:, self.n_invariant + k :],
        )
        log_root = 0.5 * special.log_ndtr(u_steps)  # log sqrt(p), stable in the tails
        v = rng.uniform(size=u_steps.shape)
        hold = special.ndtri(np.exp(log_root))
        other = special.ndtri(np.exp(log_root + np.log(v)))
        first_is_max = rng.uniform(size=u_steps.shape) < 0.5
        fine = np.empty((u.shape[0], 2 * k))
        fine[:, 0::2] = np.where(first_is_max, hold, other)
        fine[:, 1::2] = np.where(first_is_max, other, hold)
        new_grid = np.asarray(np.arange(1, 2 * k + 1) * (grid[0] / 2.0), dtype=float)
        return np.column_stack([u_inv, fine, u_insp]), new_grid

    def _knee_shift(self) -> float:
        """Expected shift of log10 rate at the knee from taking the minimum of two random stage lines:
        -sqrt((s_A^2 + s_B^2 - 2 rho s_A s_B) / (2 pi)) (two normals with equal mean at the knee)."""
        s_a = float(self.law.a_coefficient.sigma_ln) / np.log(10.0)
        s_b = float(self.law.a_coefficient_b.sigma_ln) / np.log(10.0)  # type: ignore[union-attr]
        rho = float(self.law.stage_correlation or 0.0)
        return float(-np.sqrt(max(s_a**2 + s_b**2 - 2 * rho * s_a * s_b, 0.0) / (2 * np.pi)))

    def _sensitivity(self, seed: int, samples: int, grid: FloatArray) -> dict[str, Curve]:
        out = {}
        for rho in (0.0, 1.0):
            law = self.law.with_(stage_correlation=rho)
            variant = TimeDependentAssessment(
                self.problem,
                law,
                self.spectrum,
                self.extreme,
                list(self.times),
                self.leak_is_failure,
                self.inspections,
                self.delta,
                self.shared_scale if self._random_scale else None,
                self.eta,
                self.tol,
            )
            if self.vector is not None and variant.vector is not None:  # keep the user's other correlations
                latent = self.vector.latent_correlation.copy()
                i = next(p for p, v in enumerate(self.variables) if v is self.law.a_coefficient)
                j = next(p for p, v in enumerate(self.variables) if v is self.law.a_coefficient_b)
                latent[i, j] = latent[j, i] = float(np.clip(rho, -(1 - 1e-9), 1 - 1e-9))
                variant.vector._latent = latent
                variant.vector._cholesky = None
            dim = self.n_invariant + grid.size + self.n_inspection_draws
            u = np.random.default_rng(seed).standard_normal((samples, dim))
            res = variant._assemble(variant.run(u, grid), grid, 0)
            out[f"stage_correlation={rho:g}"] = res.cumulative_upper
        return out

    def _subset(
        self, seed: int, samples: int, grid: FloatArray, replicates: int, per_level: int, curve_times: Sequence[float]
    ) -> dict[str, Any]:
        """Subset simulation on the lifetime minimum (upper pairing, out-of-validity as failed), the
        curve of shares from the final-level failure samples with a resolution floor, and separate runs
        at up to five requested times."""
        K = grid.size

        def limit_state_to(k_last: int) -> Callable[[FloatArray], Evaluation]:
            def fn(uu: FloatArray) -> Evaluation:
                r = self.run(uu, grid)
                g = np.where(r["any_invalid"], -np.inf, r["g_min"])
                y = np.min(g[:, : k_last + 1], axis=1)
                fail = y <= 0.0
                invalid = np.any(r["any_invalid"][:, : k_last + 1], axis=1)
                return Evaluation(np.where(np.isfinite(y), y, -1.0), fail, invalid)

            return fn

        dim = self.n_invariant + K + self.n_inspection_draws
        rel = solve(limit_state_to(K - 1), dim, seed, samples, replicates, per_level)
        shares: dict[str, Any] = {"label": "conditional; shares of the final-level failure samples by first passage"}
        if rel.subset is not None:
            runs = rel.subset.runs
            per_run = []
            for r in runs:
                pts = r.failure_points
                if pts.shape[0] == 0:
                    continue
                rr = self.run(pts, grid)
                g = np.where(rr["any_invalid"], -np.inf, rr["g_min"])
                failed_by = np.cumsum(g <= 0.0, axis=1) > 0
                per_run.append(np.mean(failed_by, axis=0))
            if per_run:
                arr = np.array(per_run)
                gamma = float(np.mean([np.max(r.gammas) if len(r.gammas) else 0.0 for r in runs]))
                n_final = float(np.mean([r.failure_points.shape[0] for r in runs]))
                floor = (1.0 + gamma) / max(n_final, 1.0)
                mean_share = np.mean(arr, axis=0)
                shares.update(
                    {
                        "share_mean": mean_share,
                        "share_spread": np.std(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(K),
                        "resolution_floor": floor,
                        "below_resolution": mean_share < floor,
                        "curve": np.where(mean_share < floor, np.nan, rel.pf * mean_share),
                        "grid": grid,
                    }
                )
        separate = {}
        for t in list(curve_times)[:5]:
            k = int(np.argmin(np.abs(grid - t)))
            separate[float(grid[k])] = solve(limit_state_to(k), dim, seed + 1 + k, samples, replicates, per_level)
        shares["separate_runs"] = separate
        return {"subset": rel, "subset_curve": shares}

    def _assemble(self, run: dict[str, Any], grid: FloatArray, refinements: int) -> TimeDependentResult:
        K, n = grid.size, run["n"]
        g_min, g_max, any_invalid = run["g_min"], run["g_max"], run["any_invalid"]
        leak_step = run["leak_step"]
        report = np.array([int(np.argmin(np.abs(grid - t))) for t in self.times])

        def first_passage(pairing: str, invalid_failed: bool) -> IntArray:
            if pairing == "upper":
                fail = (g_min <= 0.0) | (any_invalid & invalid_failed)
            else:
                fail = ((g_max <= 0.0) & ~any_invalid) | (any_invalid & invalid_failed)
            if self.leak_is_failure:
                hit = leak_step >= 0
                fail[np.nonzero(hit)[0], leak_step[hit]] = True
            cum = np.cumsum(fail, axis=1) > 0
            return np.asarray(np.where(np.any(cum, axis=1), np.argmax(cum, axis=1), K), dtype=np.intp)

        fp_u, fp_l, fp_v = first_passage("upper", True), first_passage("lower", True), first_passage("upper", False)

        def counts(fp: IntArray) -> tuple[IntArray, IntArray]:
            nk = np.bincount(fp[fp < K], minlength=K)
            s = n - np.concatenate([[0], np.cumsum(nk)[:-1]])
            return nk.astype(np.intp), s.astype(np.intp)

        nu, su = counts(fp_u)
        nl, sl = counts(fp_l)
        nv, _ = counts(fp_v)
        starts = [0] + [int(r) + 1 for r in report[:-1]]

        def cumulative_curve(nk: IntArray, kind: str) -> Curve:
            v, lo, hi = [], [], []
            for k in report:
                f = int(np.sum(nk[: k + 1]))
                ci = clopper_pearson(f, n)
                v.append(f / n)
                lo.append(ci.lower)
                hi.append(ci.upper)
            return Curve(self.times, np.array(v), np.array(lo), np.array(hi), kind)

        def per_interval_curve(nk: IntArray, s: IntArray, kind: str, hazard: bool) -> Curve:
            v, lo, hi = [], [], []
            for k0, k in zip(starts, report, strict=True):
                f = int(np.sum(nk[k0 : k + 1]))  # counts aggregated over the steps of the reporting interval
                denom = int(s[k0]) if hazard else n
                if denom == 0:
                    v.append(np.nan)
                    lo.append(np.nan)
                    hi.append(np.nan)
                    continue
                ci = clopper_pearson(f, denom)
                v.append(f / denom)
                lo.append(ci.lower)
                hi.append(ci.upper)
            return Curve(self.times, np.array(v), np.array(lo), np.array(hi), kind)

        ub, lb = np.zeros(report.size), np.zeros(report.size)
        discordant = np.zeros(report.size, dtype=np.intp)
        refine = np.zeros(report.size, dtype=bool)
        for j, (k0, k) in enumerate(zip(starts, report, strict=True)):
            ub[j], lb[j] = self._conservative_pair(fp_u, fp_l, k0 - 1, k, np.ones(n, dtype=bool))
            discordant[j] = int(np.sum((fp_u <= k) & (fp_l > k)))
            up_c, lo_c = float(np.sum(nu[: k + 1]) / n), float(np.sum(nl[: k + 1]) / n)
            refine[j] = discordant[j] >= 1 and up_c > 1.2 * lo_c
        failed = fp_u < K
        rows = np.arange(n)
        step_of = np.minimum(fp_u, K - 1)
        structural = failed & (g_min[rows, step_of] <= 0.0)
        leak_first = failed & (leak_step == fp_u) & self.leak_is_failure
        ended_first = failed & (run["ended_step"] == fp_u) & ~structural
        invalid_first = failed & any_invalid[rows, step_of] & ~structural & ~leak_first & ~ended_first
        exit_at = run["mode"][rows, step_of]
        fracture = structural & np.isin(exit_at, ("curve", "step"))
        collapse = structural & np.isin(exit_at, ("cutoff", "domain_end"))
        state = run["state"]
        assumed = np.isfinite(state.t_leak_assumed) & ~(
            np.isfinite(state.t_leak) & (state.t_leak <= state.t_leak_assumed)
        )
        modes = {
            "fracture": float(np.sum(fracture & ~leak_first) / n),
            "collapse": float(np.sum(collapse & ~leak_first) / n),
            "leak": float(np.sum(leak_first & ~structural & ~assumed) / n),
            "leak_assumed": float(np.sum(leak_first & ~structural & assumed) / n),
            "leak_and_structural_same_step_order_unresolved": float(np.sum(leak_first & structural) / n),
            "out_of_validity": float(np.sum((ended_first | invalid_first) & ~leak_first) / n),
        }
        state = run["state"]
        seen = run.get("codes_seen")
        if seen is None:  # a record without the per-event codes (e.g. a direct test caller): from the event record,
            # which holds the last segment's events only (run() accumulates every segment's codes)
            seen = np.zeros((n, 32), dtype=bool)
            for e in range(state.event_trigger.shape[1]):
                has_e = state.event_count > e
                seen[np.nonzero(has_e)[0], state.event_trigger[has_e, e]] = True
        by_trigger = {int(k): float(np.mean(seen[:, k])) for k in range(1, seen.shape[1]) if np.any(seen[:, k])}

        def event_share(t_event: FloatArray, only_before: bool) -> float:
            happened = np.isfinite(t_event)
            if only_before:
                step = np.searchsorted(grid, t_event * (1 - 1e-12), side="left")
                happened &= step <= fp_u
            return float(np.mean(happened))

        events = {
            "leak": event_share(state.t_leak, True),
            "leak_assumed": event_share(state.t_leak_assumed, True),
            "surface_breaking": event_share(state.t_surface_breaking, True),
            "ended_outside_validity": event_share(state.t_ended, True),
            "repaired": event_share(run["t_repair"], True),
        }
        events_any = {
            "leak": event_share(state.t_leak, False),
            "leak_assumed": event_share(state.t_leak_assumed, False),
            "surface_breaking": event_share(state.t_surface_breaking, False),
            "ended_outside_validity": event_share(state.t_ended, False),
            "repaired": event_share(run["t_repair"], False),
            "removed_flaw_had_leaked": float(np.mean(run["removed_leaked"])),
            "removed_flaw_had_ended": float(np.mean(run["removed_ended"])),
            "by_trigger": by_trigger,  # share of samples whose leak / leak_assumed carries each boundary code
        }
        order_note = (
            f", order {self.spectrum.order} (the worst of three declared orders is not the most damaging of all "
            "permutations)"
            if self.spectrum.order
            else ""
        )
        table_edge_note = (
            (
                "a re-characterisation at a validity limit (leak_assumed, incl. table edges) is counted as a failure "
                "because leaks count; it is not a breakthrough",
            )
            if self.leak_is_failure and bool(np.any(np.isfinite(state.t_leak_assumed)))
            else ()
        )
        assumptions = (
            *table_edge_note,
            "assumption A: severity monotone between checkpoints (checked at the checkpoints, not proved); "
            f"{run['violations']} sample-steps in {run['violating_samples']} samples violated it",
            f"independent step maxima of length {float(grid[0]):g} years (stationary, no seasonality)",
            f"spectrum: {self.spectrum.kind}{order_note}; sequence effects on the rate not modelled",
            f"growth law: {self.law.name}; validity: {self.law.validity or 'not stated'}",
            f"stage correlation: {self.law.stage_correlation}; shared load scale: {self.shared_scale}",
            "successive inspections independent given the flaw size; no false calls",
            "intervals are pointwise (99% each); interval probabilities from the upper pairing are approximations; "
            "within-pairing hazards are exact for their discretised model (a sensitivity, not a bracket); the "
            "conservative pair brackets the hazard under assumption A",
            f"leak counted as failure: {self.leak_is_failure}",
            f"initial flaws outside the surface solution's validity: {run['invalid_start']:.3g} (counted as failed "
            "in the headline, as safe in the valid-only curve)",
        )
        insp_out: list[dict[str, Any]] = []
        for i in run["inspections"]:
            info = {k: v for k, v in i.items() if k not in ("retained", "weights")}
            k_i = int(i["step"])
            nxt = k_i + 1 < K
            info["cumulative_upper_before"] = float(np.sum(nu[: k_i + 1]) / n)
            info["cumulative_lower_before"] = float(np.sum(nl[: k_i + 1]) / n)
            info["cumulative_upper_after_next_step"] = float(np.sum(nu[: k_i + 2]) / n) if nxt else float("nan")
            info["cumulative_lower_after_next_step"] = float(np.sum(nl[: k_i + 2]) / n) if nxt else float("nan")
            info["hazard_upper_next_step"] = float(nu[k_i + 1] / su[k_i + 1]) if nxt and su[k_i + 1] else float("nan")
            info["hazard_lower_next_step"] = float(nl[k_i + 1] / sl[k_i + 1]) if nxt and sl[k_i + 1] else float("nan")
            info["interval_probability_next_step"] = float(nu[k_i + 1] / n) if nxt else float("nan")
            info["hazard_conservative_next_step"] = (
                self._conservative_pair(fp_u, fp_l, k_i, k_i + 1, np.ones(n, dtype=bool))[::-1]
                if nxt
                else (float("nan"), float("nan"))
            )  # (lower, upper) bracket of the next step's hazard under assumption A
            insp_out.append(info)
        conditioning = None
        weighted = False
        observed = [i for i in run["inspections"] if i["observed"] is not None]
        if observed:
            conditioning, weighted = self._condition(run, observed, fp_u, fp_l, nu, nl, n, K)
        return TimeDependentResult(
            self.times,
            self.leak_is_failure,
            cumulative_curve(nu, "upper pairing; out-of-validity as failed"),
            cumulative_curve(nl, "lower pairing; out-of-validity as failed"),
            cumulative_curve(nv, "upper pairing; out-of-validity as safe"),
            per_interval_curve(nu, su, "interval probability, upper pairing (approximation)", False),
            per_interval_curve(nu, su, "hazard within the upper pairing (exact for its discretised model)", True),
            per_interval_curve(nl, sl, "hazard within the lower pairing (exact for its discretised model)", True),
            ub,
            lb,
            discordant,
            refine,
            modes,
            events,
            events_any,
            int(run["violations"]),
            assumptions,
            tuple(insp_out),
            n,
            float(grid[0]),
            refinements,
            conditioning=conditioning,
            weighted=weighted,
            invalid_start=run["invalid_start"],
        )

    @staticmethod
    def _conservative_pair(fp_u: IntArray, fp_l: IntArray, k_j: int, k: int, keep: BoolArray) -> tuple[float, float]:
        """Conservative pair for the conditional probability of failure by step k given survival to
        step k_j (k_j = -1: no conditioning): UB = n_x / (n_x + S_up), LB = n_y / (n_y + S_lo), with
        n_x = lower-pairing survivors at k_j that have failed by k under the upper pairing (whenever),
        n_y the mirror, S the survivors to k under each pairing."""
        n_x = int(np.sum(keep & (fp_l > k_j) & (fp_u <= k)))
        n_y = int(np.sum(keep & (fp_u > k_j) & (fp_l <= k)))
        s_up = int(np.sum(keep & (fp_u > k)))
        s_lo = int(np.sum(keep & (fp_l > k)))
        ub = n_x / (n_x + s_up) if n_x + s_up > 0 else float("nan")
        lb = n_y / (n_y + s_lo) if n_y + s_lo > 0 else float("nan")
        return ub, lb

    def _condition(
        self,
        run: dict[str, Any],
        observed: list[dict[str, Any]],
        fp_u: IntArray,
        fp_l: IntArray,
        nu: IntArray,
        nl: IntArray,
        n: int,
        K: int,
    ) -> tuple[dict[str, Any], bool]:
        """Probabilities after the last observed inspection, conditional on survival to it and the
        observed outcomes; by rejection (not detected / detected) or by weighting (sized). Once weighted,
        the population stays weighted."""
        last = observed[-1]
        k_j = int(last["step"])
        keep = np.ones(n, dtype=bool)
        weights = np.ones(n)
        weighted = False
        sizes: list[tuple[float, float]] = []
        later = fp_u < K
        for info in observed:
            if info["weights"] is not None:
                weights = weights * info["weights"]
                weighted = True
            else:
                keep &= info["retained"]
            w = weights * keep
            ess_all = float(np.sum(w) ** 2 / np.sum(w**2)) if np.any(w > 0) else 0.0
            ess_fail = float(np.sum(w[later]) ** 2 / np.sum(w[later] ** 2)) if np.any(w[later] > 0) else 0.0
            sizes.append((ess_all, ess_fail))
        out: dict[str, Any] = {
            "conditioning": "survival to the inspection and the observed outcome(s)",
            "weighted": weighted,
            "effective_sizes_after_each_outcome": sizes,
            "exact_for": None if weighted else "the discretised model of each pairing (rejection)",
        }
        up_c, lo_c = float(np.sum(nu[: k_j + 1]) / n), float(np.sum(nl[: k_j + 1]) / n)
        disc = int(np.sum((fp_u <= k_j) & (fp_l > k_j)))
        out["refine_at_conditioning_time"] = disc >= 1 and up_c > 1.2 * lo_c
        for name, fp in (("upper", fp_u), ("lower", fp_l)):
            sel = keep & (fp > k_j)
            w = weights * sel
            total = float(np.sum(w))
            fails = fp < K
            if weighted:
                ess_fail = float(np.sum(w[fails]) ** 2 / np.sum(w[fails] ** 2)) if np.any(w[fails] > 0) else 0.0
                value = float(np.sum(w[fails]) / total) if total > 0 else float("nan")
                out[f"pf_after_{name}"] = value if ess_fail >= KISH_GATE else None
                out[f"withheld_{name}"] = ess_fail < KISH_GATE
                out[f"samples_needed_{name}"] = (
                    int(np.ceil(n * KISH_GATE / ess_fail)) if 0 < ess_fail < KISH_GATE else n
                )
                out[f"label_{name}"] = "conditional; weighted; no guaranteed interval"
            else:
                m = int(np.sum(sel))
                f = int(np.sum(sel & fails))
                out[f"p_conditioning_event_{name}"] = m / n
                out[f"retained_{name}"] = m
                out[f"pf_after_{name}"] = f / m if m else float("nan")
                out[f"interval_{name}"] = clopper_pearson(f, m) if m else None
                out[f"label_{name}"] = f"exact for the {name} pairing's discretised model"
        if not weighted:
            ub, lb = self._conservative_pair(fp_u, fp_l, k_j, K - 1, keep)
            out["conservative_pair"] = (lb, ub)
        return out, weighted
