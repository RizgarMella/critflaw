"""Fatigue crack growth of a surface flaw in a flat plate, through breakthrough to a through-thickness
flaw (theory manual, section 5.3). Vectorised over samples; each sample advances with its own step.

Second-order (Heun) steps in time with: a bound on the relative size increment, an error controller
with a carried proposal, a frozen active set (which blocks are above threshold and on which stage,
and the branch of the K solution) with the first change located by bisection and the step cut there,
terminal steps that land on geometry boundaries (after any rate event that precedes them), a sliding
state at a threshold, and an error-controlled quadrature finish for a through flaw approaching the
plate width (block by block under an ordered spectrum).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from critflaw.fatigue.laws import GrowthLaw
from critflaw.fatigue.spectrum import Spectrum
from critflaw.solutions import cylinder as cyl
from critflaw.solutions import cylinder_tables as tables
from critflaw.solutions import embedded, plate

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]
BoolArray = NDArray[np.bool_]

SURFACE, THROUGH, NONE, ENDED, EMBEDDED, EXTENDED = 0, 1, 2, 3, 4, 5  # flaw kinds
GROWING = (SURFACE, THROUGH, EMBEDDED, EXTENDED)  # has a growing flaw
DEPTH_GROWING = (SURFACE, EMBEDDED, EXTENDED)
LENGTH_GROWING = (SURFACE, THROUGH, EMBEDDED)
EVENT_CAPACITY = 4  # flaw-type changes recorded per segment (theory manual, section 5.3.3)
# trigger codes: 1-4 plate surface and through flaws; 5 embedded a/B' limit, 6 embedded a/2c = 1, 7 embedded 2c/W limit,
# 8 cylinder table edge a/B, 9 table edge a/c, 10 outside the through-wall range, 11 destination outside the
# surface solution's range at an embedded -> surface change, 12 extended a/B = 0.8, 15 cylinder table edge 2c/W
EMBEDDED_WIDTH_LIMIT = 0.5
SINGULAR_ZONE = 0.9  # a/B at which a kind with a K singular at a = B is ended (trigger code 17)
ETA = 0.02  # bound on the relative size increment per step
TOL = 2e-4  # local error tolerance (|Heun - Euler| / size)
MIN_STEP_FRACTION = 1e-12  # of the requested life, for controller-chosen steps
TERMINAL_ZONE = 0.9  # through flaw: 2c/W at or beyond which the quadrature finish is used
SURFACE_WIDTH_LIMIT = plate.FW_SURFACE_MAX_2C_OVER_W  # 0.8
EVENT_TOLERANCE = 1e-6  # of the step, for the location of an active-set change
SLIDING_FRACTION = 1e-3  # a reverse change this close to the start of a step means a sliding mode
PASS_FRACTION = 0.25  # growth per pass of a repeating spectrum must stay below this fraction of eta
INSIDE = 1.0 - 1e-9  # factor that puts a boundary state just inside an open validity limit
FINISH_TOLERANCE = 1e-10  # relative, for the quadrature finish (two grid levels)
_GL = np.polynomial.legendre.leggauss(8)
GL_NODES: FloatArray = np.asarray(_GL[0], dtype=float)
GL_WEIGHTS: FloatArray = np.asarray(_GL[1], dtype=float)


@dataclass
class GrowthState:
    """Mutable per-sample state of the integration (arrays of length n)."""

    a: FloatArray  # depth (surface flaw) or half-height (embedded); = thickness for a through flaw
    c: FloatArray  # half-length
    b_eff: FloatArray  # B' = 2a + 2p of an embedded flaw (invariant under symmetric growth); NaN otherwise
    kind: IntArray
    t: FloatArray  # years
    thickness: FloatArray
    width: FloatArray
    t_leak: FloatArray  # breakthrough at a = B (NaN if none)
    t_leak_assumed: FloatArray  # re-characterisation at a validity limit
    leak_trigger: IntArray  # 0 none, 1 a/B, 2 depth limit of long flaws, 3 aspect ratio, 4 2c/W
    depth_growing_at_recharacterisation: BoolArray
    t_ended: FloatArray  # through-flaw width limit reached
    t_surface_breaking: FloatArray  # embedded flaw broke through to the nearer surface (NaN if none)
    # flaw-type changes in the current segment: count, and both sides of each (a, c, kind), up to two
    event_count: IntArray
    event_pre: FloatArray  # (n, EVENT_CAPACITY, 3)
    event_post: FloatArray  # (n, EVENT_CAPACITY, 3)
    event_time: FloatArray  # (n, EVENT_CAPACITY)
    event_trigger: IntArray  # (n, EVENT_CAPACITY): the boundary code of each change (0 for a plain breakthrough)
    clamped: BoolArray  # a non-positive combined range was clamped to zero
    slid: BoolArray
    sliding: BoolArray  # (n, nb, 2): pair in the sliding state
    last_change: IntArray  # (n, nb, 2): +1 / -1 direction of the last located threshold change, 0 none
    dormant: BoolArray
    steps: IntArray
    has_override: BoolArray  # the next step starts with the set below (far side of a located change)
    override_above: BoolArray
    override_stage: BoolArray
    force: BoolArray  # the next step is short (the change was at the start of the previous step)
    h_next: FloatArray  # carried step proposal (years), NaN if none
    invalid_start: BoolArray  # the initial flaw was outside the surface solution's validity
    zero_cuts: IntArray  # consecutive changes located at the start of a step
    ignore_branch: BoolArray  # the a/c = 1 branch change is no longer an event (the state hovers on it)

    @classmethod
    def start(
        cls,
        a: FloatArray,
        c: FloatArray,
        thickness: FloatArray,
        width: FloatArray,
        blocks: int,
        ligament: FloatArray | None = None,
    ) -> GrowthState:
        """Surface flaws (``ligament`` None) or embedded flaws of half-height ``a`` with ligament ``p``."""
        n = a.size
        nan = np.full(n, np.nan)
        thick = np.broadcast_to(np.asarray(thickness, dtype=float), (n,)).copy()
        wide = np.broadcast_to(np.asarray(width, dtype=float), (n,)).copy()
        a_, c_ = a.astype(float).copy(), c.astype(float).copy()
        if ligament is None:
            valid = plate.surface_flaw_valid(a_, c_, thick, wide)
            kind0 = SURFACE
            b_eff = nan.copy()
        else:
            p_ = np.broadcast_to(np.asarray(ligament, dtype=float), (n,)).copy()
            valid = embedded.embedded_flaw_valid(a_, c_, p_, thick, wide)
            kind0 = EMBEDDED
            b_eff = 2.0 * a_ + 2.0 * p_
        return cls(
            a=a_,
            c=c_,
            b_eff=b_eff,
            kind=np.where(valid, kind0, ENDED).astype(np.intp),
            t=np.zeros(n),
            thickness=thick,
            width=wide,
            t_leak=nan.copy(),
            t_leak_assumed=nan.copy(),
            leak_trigger=np.zeros(n, dtype=np.intp),
            depth_growing_at_recharacterisation=np.zeros(n, dtype=bool),
            t_ended=np.where(valid, np.nan, 0.0),
            t_surface_breaking=nan.copy(),
            event_count=np.zeros(n, dtype=np.intp),
            event_pre=np.full((n, EVENT_CAPACITY, 3), np.nan),
            event_post=np.full((n, EVENT_CAPACITY, 3), np.nan),
            event_time=np.full((n, EVENT_CAPACITY), np.nan),
            event_trigger=np.zeros((n, EVENT_CAPACITY), dtype=np.intp),
            clamped=np.zeros(n, dtype=bool),
            slid=np.zeros(n, dtype=bool),
            sliding=np.zeros((n, blocks, 2), dtype=bool),
            last_change=np.zeros((n, blocks, 2), dtype=np.intp),
            dormant=np.zeros(n, dtype=bool),
            steps=np.zeros(n, dtype=np.intp),
            has_override=np.zeros(n, dtype=bool),
            override_above=np.zeros((n, blocks, 2), dtype=bool),
            override_stage=np.zeros((n, blocks, 2), dtype=bool),
            force=np.zeros(n, dtype=bool),
            h_next=nan.copy(),
            invalid_start=~valid,
            zero_cuts=np.zeros(n, dtype=np.intp),
            ignore_branch=np.zeros(n, dtype=bool),
        )

    def snapshot(self) -> tuple[FloatArray, FloatArray, IntArray]:
        return self.a.copy(), self.c.copy(), self.kind.copy()

    def new_segment(self) -> None:
        self.event_count[:] = 0
        self.event_pre[:] = np.nan
        self.event_post[:] = np.nan
        self.event_time[:] = np.nan
        self.event_trigger[:] = 0

    def record_event(
        self, i: int, pre: tuple[float, float, int], post: tuple[float, float, int], t: float, trigger: int = 0
    ) -> None:
        k = int(self.event_count[i])
        if k >= EVENT_CAPACITY:
            raise RuntimeError(
                f"more than {EVENT_CAPACITY} flaw-type changes in one segment (sample {i}); none is dropped"
            )
        self.event_pre[i, k] = pre
        self.event_post[i, k] = post
        self.event_time[i, k] = t
        self.event_trigger[i, k] = trigger
        self.event_count[i] = k + 1

    def ligament(self) -> FloatArray:
        """p = (B' - 2a)/2 of embedded flaws (NaN otherwise)."""
        return np.asarray(0.5 * (self.b_eff - 2.0 * self.a), dtype=float)


@dataclass(frozen=True)
class CylinderSpec:
    """The cylinder and the flaw family grown in it (one per Grower): the route decides the K model."""

    r_i: float
    orientation: str  # "axial" | "circumferential"
    position: str  # "internal" | "external"
    k_route: str  # "pressure_only" | "tabulated"
    bending_tensile_surface: str = "outside"

    @property
    def external(self) -> bool:
        return self.position == "external"


class Grower:
    """Growth of n samples under one law and one spectrum (coefficients and scale realised per sample).
    ``cylinder`` (a :class:`CylinderSpec`) switches the K solutions and boundaries to a cylinder."""

    def __init__(
        self,
        law: GrowthLaw,
        spectrum: Spectrum,
        n: int,
        life_years: float,
        eta: float = ETA,
        tol: float = TOL,
        cylinder: CylinderSpec | None = None,
    ) -> None:
        self.cylinder = cylinder
        if law.effective_range:
            raise NotImplementedError(
                "TODO(verify): effective-range growth near the threshold (no constants in the source)"
            )
        self.law, self.spectrum, self.n = law, spectrum, n
        self.eta, self.tol = eta, tol
        self.min_step = MIN_STEP_FRACTION * life_years
        self.a1 = np.broadcast_to(np.asarray(law.a_coefficient, dtype=float), (n,))
        self.a2 = (
            None if law.a_coefficient_b is None else np.broadcast_to(np.asarray(law.a_coefficient_b, dtype=float), (n,))
        )
        self.thr = np.broadcast_to(np.asarray(law.threshold, dtype=float), (n,))
        self.scale = np.broadcast_to(np.asarray(spectrum.load_scale, dtype=float), (n,))
        self.nb = spectrum.membrane.size
        self.nu = spectrum.cycles_per_year * spectrum.life_factor
        self.block_ends = (
            spectrum.block_end_times() * 1.0 / spectrum.life_factor if spectrum.kind == "ordered" else None
        )
        self.pass_years = spectrum.pass_cycles / self.nu if spectrum.kind == "repeating" else None
        self.b_eff: FloatArray = np.full(n, np.nan)  # B' per sample; bound to the state's array by advance()
        self.thickness: FloatArray = np.ones(n)  # per sample; bound by advance() (the finish of cylinder flaws)

    # ---- ranges and rates -------------------------------------------------------------------

    def threshold_at(self, idx: IntArray, a: FloatArray, kind: IntArray) -> FloatArray:
        thr = self.thr[idx]
        if self.law.shallow_cap is None:
            return np.asarray(thr + 0.0, dtype=float)
        shallow = ((kind == SURFACE) | (kind == EXTENDED)) & (a < self.law.shallow_cap[0])
        return np.asarray(np.where(shallow, np.minimum(thr, self.law.shallow_cap[1]), thr), dtype=float)

    def ranges(
        self, idx: IntArray, a: FloatArray, c: FloatArray, kind: IntArray, thickness: FloatArray, width: FloatArray
    ) -> FloatArray:
        """Stress intensity factor ranges, shape (n, nb, 2): position 0 the deepest point (growth of a),
        position 1 the surface point (growth of c). Through flaw: position 0 is zero, position 1 the tip.
        A non-positive combined range is clamped to zero."""
        sp = self.spectrum
        m = sp.ktm * sp.membrane[None, :] + sp.ktb * (sp.km - 1.0) * sp.membrane[None, :]
        b = sp.ktb * sp.bending[None, :]
        if self.cylinder is not None:
            return self._cylinder_ranges(idx, a, c, kind, thickness, width, m, b)
        out = np.zeros((a.size, self.nb, 2))
        surf = kind == SURFACE
        if np.any(surf):
            aa, cc, bb, ww = a[surf], c[surf], thickness[surf], width[surf]
            fw = plate.surface_flaw_fw(aa, cc, bb, ww)
            root = np.sqrt(np.pi * aa)
            for p, theta in ((0, np.pi / 2), (1, 0.0)):
                mm, mb = plate.surface_flaw_mm_mb(aa, cc, bb, theta)
                ys = fw[:, None] * (mm[:, None] * m + mb[:, None] * b)
                out[surf, :, p] = np.maximum(ys, 0.0) * root[:, None]
        thr = kind == THROUGH
        if np.any(thr):
            cc, ww = c[thr], width[thr]
            fw = plate.through_flaw_fw(np.minimum(cc, 0.5 * ww * (1 - 1e-12)), ww)
            out[thr, :, 1] = fw[:, None] * (m + b) * np.sqrt(np.pi * cc)[:, None]
        emb = kind == EMBEDDED
        if np.any(emb):
            # membrane only (bending-loaded embedded flaws are refused by the model layer): position 0 the
            # near minor-axis end (grows a), position 1 the major-axis ends (grow c)
            aa, cc, ww = a[emb], c[emb], width[emb]
            pp = 0.5 * (self.b_eff[idx][emb] - 2.0 * aa)
            fw = embedded.embedded_flaw_fw(aa, cc, np.maximum(pp, 0.0), ww)
            root = np.sqrt(np.pi * aa)
            for p_, theta in ((0, embedded.THETA_NEAR_END), (1, embedded.THETA_MAJOR_AXIS)):
                mm = embedded.embedded_flaw_mm(aa, cc, np.maximum(pp, 0.0), theta)
                out[emb, :, p_] = np.maximum(fw[:, None] * mm[:, None] * m, 0.0) * root[:, None]
        return np.asarray(out * self.scale[idx][:, None, None], dtype=float)

    def active_set(
        self, idx: IntArray, dk: FloatArray, a: FloatArray, c: FloatArray, kind: IntArray
    ) -> tuple[BoolArray, BoolArray, BoolArray]:
        """(above threshold (n, nb, 2), on stage B (n, nb, 2), a/c > 1 branch (n,)). A range exactly at
        the threshold counts as growing."""
        thr = self.threshold_at(idx, a, kind)[:, None, None]
        above = (dk >= thr) & (dk > 0.0)
        if self.a2 is None:
            stage_b = np.zeros_like(above)
        else:
            ra = self.a1[idx][:, None, None] * dk**self.law.m
            rb = self.a2[idx][:, None, None] * dk ** float(self.law.m_b)  # type: ignore[arg-type]
            stage_b = rb < ra
        branch = ((kind == SURFACE) | (kind == EMBEDDED)) & (a > c)
        return above, stage_b, branch

    def stage_rate(self, idx: IntArray, dk: FloatArray, above: BoolArray, stage_b: BoolArray) -> FloatArray:
        """Rate per cycle for the frozen set (n, nb, 2)."""
        ra = self.a1[idx][:, None, None] * dk**self.law.m
        if self.a2 is None:
            r = ra
        else:
            rb = self.a2[idx][:, None, None] * dk ** float(self.law.m_b)  # type: ignore[arg-type]
            r = np.where(stage_b, rb, ra)
        return np.asarray(np.where(above, r, 0.0), dtype=float)

    def block_weights(self, t: FloatArray) -> FloatArray:
        """Cycles per year of each block at time t, shape (n, nb)."""
        if self.block_ends is None:
            return np.broadcast_to(self.nu * self.spectrum.weights[None, :], (t.size, self.nb)).copy()
        starts = np.concatenate([[0.0], self.block_ends[:-1]])
        current = (t[:, None] >= starts[None, :] * (1 - 1e-12)) & (t[:, None] < self.block_ends[None, :] * (1 - 1e-12))
        return np.asarray(np.where(current, self.nu, 0.0), dtype=float)

    def velocities(
        self,
        idx: IntArray,
        a: FloatArray,
        c: FloatArray,
        kind: IntArray,
        t: FloatArray,
        thickness: FloatArray,
        width: FloatArray,
        above: BoolArray,
        stage_b: BoolArray,
        sliding: BoolArray | None = None,
        b_eff: FloatArray | None = None,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """da/dt, dc/dt (m/year) with the frozen set; also returns dk (n, nb, 2). ``sliding`` is updated
        in place where a pair leaves the sliding state."""
        dk = self.ranges(idx, a, c, kind, thickness, width)
        rate = self.stage_rate(idx, dk, above, stage_b)  # per cycle
        w = self.block_weights(t)
        per_block = rate * w[:, :, None]  # per year
        if sliding is not None and np.any(sliding):
            per_block = self._apply_sliding(idx, a, c, kind, t, thickness, width, stage_b, sliding, per_block, b_eff)
        va = np.sum(per_block[:, :, 0], axis=1)
        vc = np.sum(per_block[:, :, 1], axis=1)
        va = np.where(np.isin(kind, DEPTH_GROWING), va, 0.0)
        vc = np.where(np.isin(kind, LENGTH_GROWING), vc, 0.0)
        # a range the solution does not give (NaN: outside a tabulated domain) is an undefined rate, not zero
        undefined = np.any(np.isnan(dk), axis=(1, 2))
        va = np.where(undefined, np.nan, va)
        vc = np.where(undefined, np.nan, vc)
        return np.asarray(va, dtype=float), np.asarray(vc, dtype=float), dk

    def _derivatives(
        self,
        gi: IntArray,
        a: FloatArray,
        c: FloatArray,
        kind: IntArray,
        thickness: FloatArray,
        width: FloatArray,
        b_eff: FloatArray | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """dK/da and dK/dc (n, nb, 2) by central differences, one-sided (on the side of the current branch
        of the K solution, and inside the whole boundary set) within 1e-6 B of a branch change or boundary."""
        h_a = np.maximum(1e-6 * a, 1e-9)
        h_c = np.maximum(1e-6 * c, 1e-9)
        guard = 1e-6 * thickness
        shallow = a <= c  # the branch at the base state

        def inside(aa: FloatArray, cc: FloatArray) -> BoolArray:
            return np.asarray(np.all(self.boundary_excess(aa, cc, kind, thickness, width, b_eff) <= 0.0, axis=1))

        a_up_ok = np.where(shallow, a + h_a <= c - guard, True) & inside(a + h_a, c)
        a_dn_ok = np.where(shallow, True, a - h_a > c + guard) & inside(a - h_a, c)
        a_up = np.where(a_up_ok, a + h_a, a)
        a_dn = np.where(a_dn_ok, a - h_a, a)
        dk_a = (self.ranges(gi, a_up, c, kind, thickness, width) - self.ranges(gi, a_dn, c, kind, thickness, width)) / (
            a_up - a_dn
        )[:, None, None]
        c_up_ok = np.where(shallow, True, c + h_c < a - guard) & inside(a, c + h_c)
        c_dn_ok = np.where(shallow, c - h_c >= a + guard, True) & inside(a, c - h_c)
        c_up = np.where(c_up_ok, c + h_c, c)
        c_dn = np.where(c_dn_ok, c - h_c, c)
        dk_c = (self.ranges(gi, a, c_up, kind, thickness, width) - self.ranges(gi, a, c_dn, kind, thickness, width)) / (
            c_up - c_dn
        )[:, None, None]
        return dk_a, dk_c

    def _apply_sliding(
        self,
        idx_all: IntArray,
        a: FloatArray,
        c: FloatArray,
        kind: IntArray,
        t: FloatArray,
        thickness: FloatArray,
        width: FloatArray,
        stage_b: BoolArray,
        sliding: BoolArray,
        per_block: FloatArray,
        b_eff: FloatArray | None = None,
    ) -> FloatArray:
        """Hold the sliding pairs' ranges at the threshold. The total rate of the pair's own dimension that
        keeps dK constant is X = -(dK_other v_other) / dK_own; the sliding blocks share the residual
        X - v_others in order, each up to its own rate at the threshold; the pairs leave the state when the
        residual is outside [0, their rates at the threshold]."""
        rows = np.nonzero(np.any(sliding, axis=(1, 2)))[0]
        if rows.size == 0:
            return per_block
        gi = idx_all[rows]
        b_rows = None if b_eff is None else b_eff[rows]
        dk_a, dk_c = self._derivatives(gi, a[rows], c[rows], kind[rows], thickness[rows], width[rows], b_rows)
        pb = per_block[rows].copy()
        thr = self.threshold_at(gi, a[rows], kind[rows])
        w = self.block_weights(t[rows])
        shape = pb.shape
        rate_thr = self.stage_rate(
            gi, np.broadcast_to(thr[:, None, None], shape), np.ones(shape, dtype=bool), stage_b[rows]
        )
        rate_thr = rate_thr * w[:, :, None]
        sl = sliding[rows]
        for p, own, other in ((0, dk_a, dk_c), (1, dk_c, dk_a)):
            s_p = sl[:, :, p].copy()
            if not np.any(s_p):
                continue
            v_other_total = np.sum(pb[:, :, 1 - p], axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                x = -(other[:, :, p] * v_other_total[:, None]) / own[:, :, p]
            x = np.where(np.isfinite(x), x, 0.0)
            others = np.sum(np.where(s_p, 0.0, pb[:, :, p]), axis=1)
            residual = np.max(np.where(s_p, x, -np.inf), axis=1) - others
            capacity = np.sum(np.where(s_p, rate_thr[:, :, p], 0.0), axis=1)
            leave = (residual < 0.0) | (residual > capacity)
            for j in range(self.nb):  # the sliding blocks share the holding rate in order
                take = np.where(s_p[:, j], np.clip(residual, 0.0, rate_thr[:, j, p]), pb[:, j, p])
                pb[:, j, p] = take
                residual = np.where(s_p[:, j], residual - take, residual)
            s_p &= ~leave[:, None]
            sl[:, :, p] = s_p
        sliding[rows] = sl
        per_block = per_block.copy()
        per_block[rows] = pb
        return per_block

    # ---- boundaries -----------------------------------------------------------------------

    def boundary_excess(
        self,
        a: FloatArray,
        c: FloatArray,
        kind: IntArray,
        thickness: FloatArray,
        width: FloatArray,
        b_eff: FloatArray | None = None,
    ) -> FloatArray:
        """(n, 9) signed distances past each boundary (positive = outside). Surface: 0 a/B = 1; 1 long-flaw
        depth limit a/B = 1.25 (a/c + 0.6) for a/2c <= 0.1; 2 a/2c = 1; 3 2c/W = 0.8. Through: 4 2c/W = 0.9.
        Embedded: 5 breakthrough 2a = B'; 6 a/B' = 0.625 (a/c + 0.6) for a/2c <= 0.1; 7 a/2c = 1; 8 2c/W = 0.5."""
        out = np.full((a.size, 15), -np.inf)
        s = kind == SURFACE
        ab = a / thickness
        if self.cylinder is not None:
            return self._cylinder_boundaries(out, a, c, kind, thickness, width)
        out[:, 0] = np.where(s, ab - 1.0, -np.inf)
        long = s & (a / (2.0 * c) <= 0.1)
        out[:, 1] = np.where(long, ab - 1.25 * (a / c + 0.6), -np.inf)
        out[:, 2] = np.where(s, a / (2.0 * c) - 1.0, -np.inf)
        out[:, 3] = np.where(s, 2.0 * c / width - SURFACE_WIDTH_LIMIT, -np.inf)
        out[:, 4] = np.where(kind == THROUGH, 2.0 * c / width - TERMINAL_ZONE, -np.inf)
        e = kind == EMBEDDED
        if np.any(e):
            b_eff = self._b_eff_of(a, kind, b_eff)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[:, 5] = np.where(e, 2.0 * a / b_eff - 1.0, -np.inf)
                long_e = e & (a / (2.0 * c) <= 0.1)
                out[:, 6] = np.where(long_e, a / b_eff - 0.625 * (a / c + 0.6), -np.inf)
                out[:, 7] = np.where(e, a / (2.0 * c) - 1.0, -np.inf)
                out[:, 8] = np.where(e, 2.0 * c / width - EMBEDDED_WIDTH_LIMIT, -np.inf)
        return np.asarray(out, dtype=float)

    @staticmethod
    def _b_eff_of(a: FloatArray, kind: IntArray, b_eff: FloatArray | None) -> FloatArray:
        if b_eff is None:
            if np.any(kind == EMBEDDED):
                raise ValueError("b_eff (B') is required for embedded flaws")
            return np.full(a.shape, np.inf)
        return np.broadcast_to(np.asarray(b_eff, dtype=float), a.shape)

    def _embedded_long_limit(self, c: FloatArray, b_eff: FloatArray) -> FloatArray:
        """Half-height at the embedded long-flaw limit a/B' = 0.625 (a/c + 0.6): a = 0.375 B' / (1 - 0.625 B'/c)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray(np.where(c > 0.625 * b_eff, 0.375 * b_eff / (1.0 - 0.625 * b_eff / c), np.inf))

    def _long_limit(self, c: FloatArray, thickness: FloatArray) -> FloatArray:
        """Depth at the long-flaw limit a/B = 1.25 (a/c + 0.6): a = 0.75 B / (1 - 1.25 B / c), or infinite."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray(np.where(c > 1.25 * thickness, 0.75 * thickness / (1.0 - 1.25 * thickness / c), np.inf))

    def clip_inside(
        self,
        a: FloatArray,
        c: FloatArray,
        kind: IntArray,
        thickness: FloatArray,
        width: FloatArray,
        b_eff: FloatArray | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """Project a state onto the closed boundary set (for the rate evaluation at a predictor)."""
        s = kind == SURFACE
        e = kind == EMBEDDED
        if self.cylinder is not None and self._tabulated_family():
            # the tabulated solutions end at their printed ranges; the projection keeps the state inside them
            x = kind == EXTENDED
            ac_min = self._table_ac_min()
            a2 = np.where(s | x, np.minimum(a, cyl.TABLE_A_OVER_B_MAX * thickness), a)
            c2 = np.where(s, np.minimum(c, a2 / ac_min), c)  # a/c >= its table minimum
            if self.cylinder.orientation == "axial":
                c2 = np.where(s, np.minimum(c2, 0.075 * width), c2)
            c2 = np.where(s, np.maximum(c2, a2), c2)  # a/c <= 1 last: the tables have no width axis, so a width
            # excess at a corner keeps the factors finite while the boundary set lands on the limit
            if np.any(kind == THROUGH):
                lam_end = self._lambda_end(thickness)
                c_max = (
                    lam_end
                    * np.sqrt((self.cylinder.r_i + 0.5 * thickness) * thickness)
                    / (12.0 * (1.0 - 0.3**2)) ** 0.25
                )
                c2 = np.where(kind == THROUGH, np.minimum(c, c_max), c2)
            return np.asarray(a2, dtype=float), np.asarray(c2, dtype=float)
        a2 = np.where(s, np.minimum(a, thickness), a)
        c2 = np.where(s, np.minimum(c, SURFACE_WIDTH_LIMIT * 0.5 * width), c)
        c2 = np.where(kind == THROUGH, np.minimum(c, TERMINAL_ZONE * 0.5 * width), c2)
        if self.cylinder is not None:  # pressure-only route: the plate extended solution ends at a/B = 0.6
            a2 = np.where(kind == EXTENDED, np.minimum(a, cyl.EXTENDED_PLATE_A_OVER_B_MAX * thickness), a2)
            if self.cylinder.k_route == "tabulated" and np.any(kind == THROUGH):
                # the plate-directed family's through-wall flaw is tabulated: project at the lambda end as above
                lam_end = self._lambda_end(thickness)
                c_max = (
                    lam_end
                    * np.sqrt((self.cylinder.r_i + 0.5 * thickness) * thickness)
                    / (12.0 * (1.0 - 0.3**2)) ** 0.25
                )
                c2 = np.where(kind == THROUGH, np.minimum(c, c_max), c2)
            if self.cylinder.k_route == "pressure_only" and self.cylinder.orientation == "axial":
                a2 = np.where(s, np.minimum(a2, SINGULAR_ZONE * thickness), a2)
            if self.cylinder.orientation == "circumferential" and self.cylinder.external:
                a2 = np.where(kind == EXTENDED, np.minimum(a, SINGULAR_ZONE * thickness), a2)
        c2 = np.where(s, np.maximum(c2, 0.5 * a2), c2)  # a/2c <= 1
        long = s & (a2 / (2.0 * c2) <= 0.1)
        a2 = np.where(long, np.minimum(a2, self._long_limit(c2, thickness)), a2)
        if np.any(e):
            b_eff = self._b_eff_of(a, kind, b_eff)
            a2 = np.where(e, np.minimum(a, 0.5 * b_eff), a2)
            c2 = np.where(e, np.minimum(c, EMBEDDED_WIDTH_LIMIT * 0.5 * width), c2)
            c2 = np.where(e, np.maximum(c2, 0.5 * a2), c2)
            long_e = e & (a2 / (2.0 * c2) <= 0.1)
            a2 = np.where(long_e, np.minimum(a2, self._embedded_long_limit(c2, b_eff)), a2)
        return np.asarray(a2, dtype=float), np.asarray(c2, dtype=float)

    def inside_state(
        self,
        a: FloatArray,
        c: FloatArray,
        kind: IntArray,
        thickness: FloatArray,
        width: FloatArray,
        b_eff: FloatArray | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """A state moved just inside the engine's open validity limits (for its assessment). Unlike
        ``clip_inside`` (the predictor projection), a through flaw keeps its actual size up to W/2."""
        s = kind == SURFACE
        e = kind == EMBEDDED
        if self.cylinder is not None and self._tabulated_family():
            x = kind == EXTENDED
            ac_min = self._table_ac_min()
            a2 = np.where(s | x, np.minimum(a, cyl.TABLE_A_OVER_B_MAX * thickness * INSIDE), a)
            c2 = np.where(s, np.minimum(c, a2 / ac_min * INSIDE), c)
            if self.cylinder.orientation == "axial":
                c2 = np.where(s, np.minimum(c2, 0.075 * width * INSIDE), c2)
            c2 = np.where(s, np.maximum(c2, a2 / INSIDE), c2)
            return np.asarray(a2, dtype=float), np.asarray(c2, dtype=float)
        a2 = np.where(s, np.minimum(a, thickness * INSIDE), a)
        c2 = np.where(s, np.minimum(c, SURFACE_WIDTH_LIMIT * 0.5 * width), c)
        if self.cylinder is not None:
            a2 = np.where(kind == EXTENDED, np.minimum(a, cyl.EXTENDED_PLATE_A_OVER_B_MAX * thickness * INSIDE), a2)
            if self.cylinder.k_route == "pressure_only" and self.cylinder.orientation == "axial":
                a2 = np.where(s, np.minimum(a2, SINGULAR_ZONE * thickness * INSIDE), a2)
            if self.cylinder.orientation == "circumferential" and self.cylinder.external:
                a2 = np.where(kind == EXTENDED, np.minimum(a, SINGULAR_ZONE * thickness * INSIDE), a2)
        long = s & (a2 / (2.0 * c2) <= 0.1)
        a2 = np.where(long, np.minimum(a2, self._long_limit(c2, thickness) * INSIDE), a2)
        c2 = np.where(s, np.maximum(c2, 0.5 * a2 / INSIDE), c2)
        c2 = np.where(kind == THROUGH, np.minimum(c2, 0.5 * width * INSIDE), c2)
        if np.any(e):
            b_eff = self._b_eff_of(a, kind, b_eff)
            a2 = np.where(e, np.minimum(a, 0.5 * b_eff * INSIDE), a2)
            c2 = np.where(e, np.minimum(c, EMBEDDED_WIDTH_LIMIT * 0.5 * width * INSIDE), c2)
            long_e = e & (a2 / (2.0 * c2) <= 0.1)
            a2 = np.where(long_e, np.minimum(a2, self._embedded_long_limit(c2, b_eff) * INSIDE), a2)
            c2 = np.where(e, np.maximum(c2, 0.5 * a2 / INSIDE), c2)
        return np.asarray(a2, dtype=float), np.asarray(c2, dtype=float)

    # ---- one Heun step for a subset ----------------------------------------------------------

    def heun(
        self,
        st: GrowthState,
        idx: IntArray,
        h: FloatArray,
        above: BoolArray,
        stage_b: BoolArray,
        v0: tuple[FloatArray, FloatArray],
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Predictor (a1, c1) and corrector (a2, c2) for the samples idx with steps h (years). The
        corrector rate is evaluated at the predictor projected into the boundary set, at a time just
        inside the current block of an ordered spectrum."""
        a0, c0, kind, t0 = st.a[idx], st.c[idx], st.kind[idx], st.t[idx]
        thick, width = st.thickness[idx], st.width[idx]
        va0, vc0 = v0
        a1, c1 = a0 + h * va0, c0 + h * vc0
        ap, cp = self.clip_inside(a1, c1, kind, thick, width, st.b_eff[idx])
        sl = st.sliding[idx]
        hold = sl.copy() if np.any(sl) else None
        va1, vc1, _ = self.velocities(
            idx, ap, cp, kind, t0 + h * INSIDE, thick, width, above, stage_b, hold, st.b_eff[idx]
        )
        a2, c2 = a0 + 0.5 * h * (va0 + va1), c0 + 0.5 * h * (vc0 + vc1)
        return a1, c1, a2, c2

    # ---- advance to a time ----------------------------------------------------------------

    def advance(self, st: GrowthState, t_end: float, max_iterations: int = 200_000) -> None:
        """Advance every sample to ``t_end`` (years), handling events on the way."""
        self.bind(st)
        for _ in range(max_iterations):
            active = (st.t < t_end * (1 - 1e-15)) & np.isin(st.kind, GROWING) & ~st.dormant
            idx = np.nonzero(active)[0]
            if idx.size == 0:
                break
            self._step(st, idx, t_end)
        else:
            raise RuntimeError("the growth integration did not reach the requested time (iteration cap)")
        move = (st.t < t_end) & (~np.isin(st.kind, GROWING) | st.dormant)
        st.t[move] = t_end

    def bind(self, st: GrowthState) -> None:
        """Share the state's B' array (embedded flaws) and thickness with the range evaluation."""
        self.b_eff = st.b_eff
        self.thickness = st.thickness

    def _step(self, st: GrowthState, idx: IntArray, t_end: float) -> None:
        a0, c0, kind, t0 = st.a[idx], st.c[idx], st.kind[idx], st.t[idx]
        thick, width = st.thickness[idx], st.width[idx]
        dk0 = self.ranges(idx, a0, c0, kind, thick, width)
        above, stage_b, branch0 = self.active_set(idx, dk0, a0, c0, kind)
        ov = st.has_override[idx]
        if np.any(ov):
            above = np.where(ov[:, None, None], st.override_above[idx], above)
            stage_b = np.where(ov[:, None, None], st.override_stage[idx], stage_b)
            st.has_override[idx] = False
        sl = st.sliding[idx]
        above = above | sl
        va0, vc0, _ = self.velocities(
            idx, a0, c0, kind, t0, thick, width, above, stage_b, sl if np.any(sl) else None, st.b_eff[idx]
        )
        st.sliding[idx] = sl
        st.clamped[idx] |= np.any((dk0 <= 0.0) & np.isin(kind, GROWING)[:, None, None], axis=(1, 2))
        # growth per pass of a repeating spectrum must stay below eta / 4 (checked before any other path)
        if self.pass_years is not None:
            per_pass = np.maximum(va0 * self.pass_years / a0, vc0 * self.pass_years / c0)
            in_zone = (
                (kind == THROUGH) & (2.0 * c0 / width >= TERMINAL_ZONE * (1 - 1e-12)) & self._finish_allowed()
            )  # finish: accuracy is one pass
            if self.cylinder is not None and self.cylinder.k_route == "tabulated":
                # the last tenth of the through-wall lambda range: the terminal step lands on the range end
                # within one pass, as the plate terminal-zone exemption does (choice, listed)
                lam0 = cyl.shell_lambda(c0, self.cylinder.r_i + 0.5 * thick, thick)
                in_zone |= (kind == THROUGH) & (lam0 >= TERMINAL_ZONE * self._lambda_end(thick))
            if np.any((per_pass > PASS_FRACTION * self.eta) & ~in_zone):
                raise ValueError(
                    "the growth within one pass of the repeating spectrum exceeds eta / 4 for some sample; the "
                    "expectation over the pass is not accurate there: use Spectrum.ordered or shorter passes"
                )
        keep = np.ones(idx.size, dtype=bool)
        # ordered spectra: exhausted blocks, and samples sitting exactly on a block end
        if self.block_ends is not None:
            exhausted = t0 >= self.block_ends[-1] * (1 - 1e-12)
            st.t[idx[exhausted]] = t_end
            keep &= ~exhausted
            to_block = self._next_block_end(t0) - t0
            at_boundary = keep & (to_block < 1e-9 * np.maximum(t0, 1e-300))
            st.t[idx[at_boundary]] = np.minimum(self._next_block_end(t0[at_boundary]), t_end)
            keep &= ~at_boundary
        # dormant: nothing grows and (repeating) nothing can change; ordered: wait for the next block
        # a rate the solution does not define at the step start (a cylinder state outside its tables) is an end
        # of validity (ENDED, trigger 18), never dormancy
        undefined = keep & (~np.isfinite(va0) | ~np.isfinite(vc0))
        if np.any(undefined):
            for i in idx[undefined]:
                k_i = int(st.kind[i])
                st.record_event(
                    int(i),
                    (float(st.a[i]), float(st.c[i]), k_i),
                    (float(st.a[i]), float(st.c[i]), ENDED),
                    float(st.t[i]),
                    18,
                )
                st.kind[i], st.t_ended[i] = ENDED, st.t[i]
            keep &= ~undefined
        still = keep & (va0 <= 0.0) & (vc0 <= 0.0)
        if self.block_ends is None:
            st.dormant[idx[still]] = True
            st.t[idx[still]] = t_end
        else:
            st.t[idx[still]] = np.minimum(self._next_block_end(t0[still]), t_end)
        keep &= ~still
        if not np.any(keep):
            return
        idx, a0, c0, kind, t0, thick, width = (
            idx[keep],
            a0[keep],
            c0[keep],
            kind[keep],
            t0[keep],
            thick[keep],
            width[keep],
        )
        above, stage_b, branch0, va0, vc0 = above[keep], stage_b[keep], branch0[keep], va0[keep], vc0[keep]
        remaining = t_end - t0
        # step size: relative growth bound, carried proposal, next time event
        with np.errstate(divide="ignore"):
            h_a = np.where(va0 > 0.0, self.eta * a0 / va0, np.inf)
            h_c = np.where(vc0 > 0.0, self.eta * c0 / vc0, np.inf)
        h = np.minimum(np.minimum(h_a, h_c), remaining)
        carried = st.h_next[idx]
        h = np.where(np.isfinite(carried), np.minimum(h, carried), h)
        if self.block_ends is not None:
            h = np.minimum(h, self._next_block_end(t0) - t0)
        forced = st.force[idx]
        h = np.where(forced, np.minimum(h, np.maximum(EVENT_TOLERANCE * h, 10.0 * self.min_step)), h)
        st.force[idx] = False
        h = np.maximum(h, 0.0)
        # terminal zone: quadrature finish for through flaws (the small-step trigger follows the controller)
        zone = (kind == THROUGH) & (2.0 * c0 / width >= TERMINAL_ZONE * (1 - 1e-12)) & self._finish_allowed()
        for i in idx[zone]:
            self._finish(st, int(i), t_end)
        keep = ~zone
        if not np.any(keep):
            return
        idx, a0, c0, kind, t0, thick, width = (
            idx[keep],
            a0[keep],
            c0[keep],
            kind[keep],
            t0[keep],
            thick[keep],
            width[keep],
        )
        above, stage_b, branch0, va0, vc0, h = above[keep], stage_b[keep], branch0[keep], va0[keep], vc0[keep], h[keep]
        for _ in range(60):  # controller
            a1, c1, a2, c2 = self.heun(st, idx, h, above, stage_b, (va0, vc0))
            e_a = np.where(np.isin(kind, DEPTH_GROWING), np.abs(a2 - a1) / np.maximum(a0, 1e-300), 0.0)
            e = np.maximum(e_a, np.abs(c2 - c1) / c0)
            finite = np.isfinite(a2) & np.isfinite(c2) & np.isfinite(e)
            e = np.where(finite, e, np.inf)  # an invalid trial (the solution undefined at the trial) is halved
            bad = e > self.tol
            if not np.any(bad):
                break
            factor = np.where(finite, 0.9 * np.sqrt(self.tol / np.maximum(e, 1e-3 * self.tol)), 0.5)
            h = np.where(bad, h * np.minimum(factor, 1.0), h)
            if np.any(bad & (kind != THROUGH) & (h < self.min_step)):
                raise RuntimeError(
                    "the growth integration reached the minimum step (controller); the rate is not resolvable"
                )
            stuck = bad & (kind == THROUGH) & (h < 100.0 * self.min_step) & self._finish_allowed()
            if np.any(stuck):
                # a through flaw: the quadrature finish takes over (its rate is unbounded towards W/2)
                for i in idx[stuck]:
                    self._finish(st, int(i), t_end)
                keep = ~stuck
                if not np.any(keep):
                    return
                idx, a0, c0, kind, t0, thick, width = (
                    idx[keep],
                    a0[keep],
                    c0[keep],
                    kind[keep],
                    t0[keep],
                    thick[keep],
                    width[keep],
                )
                above, stage_b, branch0 = above[keep], stage_b[keep], branch0[keep]
                va0, vc0, h = va0[keep], vc0[keep], h[keep]
        with np.errstate(divide="ignore", invalid="ignore"):
            proposal = np.where(e > 0.0, 0.9 * h * np.sqrt(self.tol / e), 2.0 * h)
        st.h_next[idx] = np.minimum(proposal, 2.0 * h)
        # a through flaw whose controller step has become tiny: quadrature finish instead
        tiny = (kind == THROUGH) & (h < 100.0 * self.min_step) & self._finish_allowed()
        if np.any(tiny):
            for i in idx[tiny]:
                self._finish(st, int(i), t_end)
            keep = ~tiny
            if not np.any(keep):
                return
            idx, kind, thick, width, h = idx[keep], kind[keep], thick[keep], width[keep], h[keep]
            above, stage_b, branch0, va0, vc0 = above[keep], stage_b[keep], branch0[keep], va0[keep], vc0[keep]
            a2, c2 = a2[keep], c2[keep]
        # a corrector outside the boundary set -> terminal step (after any active-set change before it)
        out2 = np.any(self.boundary_excess(a2, c2, kind, thick, width, st.b_eff[idx]) > 0.0, axis=1)
        if np.any(out2):
            self._terminal(st, idx[out2], h[out2], above[out2], stage_b[out2], branch0[out2], (va0[out2], vc0[out2]))
        rest = ~out2
        if not np.any(rest):
            return
        idx_r, h_r = idx[rest], h[rest]
        a2_r, c2_r = a2[rest], c2[rest]
        dk2 = self.ranges(idx_r, a2_r, c2_r, kind[rest], thick[rest], width[rest])
        above2, stage2, branch2 = self.active_set(idx_r, dk2, a2_r, c2_r, kind[rest])
        ab_r, sb_r = above[rest], stage_b[rest]
        sl_r = st.sliding[idx_r]
        branch_change = (branch2 != branch0[rest]) & ~st.ignore_branch[idx_r]
        known = ~np.isnan(dk2)  # outside a tabulated solution the range is unknown, not a change of set
        changed = np.any(((above2 != ab_r) | (stage2 != sb_r)) & ~sl_r & known, axis=(1, 2)) | branch_change
        if np.any(changed):
            self._cut(
                st,
                idx_r[changed],
                h_r[changed],
                ab_r[changed],
                sb_r[changed],
                branch0[rest][changed],
                (va0[rest][changed], vc0[rest][changed]),
            )
        plain = ~changed
        if np.any(plain):
            i = idx_r[plain]
            st.a[i], st.c[i] = a2_r[plain], c2_r[plain]
            st.t[i] = np.minimum(st.t[i] + h_r[plain], t_end)
            st.steps[i] += 1
            st.zero_cuts[i] = 0

    def _next_block_end(self, t: FloatArray) -> FloatArray:
        assert self.block_ends is not None
        pos = np.searchsorted(self.block_ends, t * (1 + 1e-12), side="right")
        pos = np.minimum(pos, self.block_ends.size - 1)
        return np.asarray(self.block_ends[pos], dtype=float)

    def _terminal(
        self,
        st: GrowthState,
        idx: IntArray,
        h: FloatArray,
        above: BoolArray,
        stage_b: BoolArray,
        branch0: BoolArray,
        v0: tuple[FloatArray, FloatArray],
    ) -> None:
        """Land on the first boundary reached: bisection on the step so that the corrector state has
        boundary excess 0 (to 1e-12 relative). If the active set changes before the boundary, the change
        is located instead and the boundary is reached on a later step."""
        kind, thick, width = st.kind[idx], st.thickness[idx], st.width[idx]
        lo, hi = np.zeros(idx.size), h.copy()
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            _, _, a2, c2 = self.heun(st, idx, mid, above, stage_b, v0)
            outside = np.any(self.boundary_excess(a2, c2, kind, thick, width, st.b_eff[idx]) > 0.0, axis=1)
            hi, lo = np.where(outside, mid, hi), np.where(outside, lo, mid)
        _, _, a2, c2 = self.heun(st, idx, lo, above, stage_b, v0)
        a2, c2 = self.clip_inside(a2, c2, kind, thick, width, st.b_eff[idx])
        excess = self.boundary_excess(a2, c2, kind, thick, width, st.b_eff[idx])
        if np.any(np.max(excess, axis=1) < -1e-6):
            raise RuntimeError("terminal step did not land on a boundary")
        dk = self.ranges(idx, a2, c2, kind, thick, width)
        ab, sb, br = self.active_set(idx, dk, a2, c2, kind)
        branch_change = (br != branch0) & ~st.ignore_branch[idx]
        known = ~np.isnan(dk)  # a range the solution does not give (outside a table) is no active-set change
        changed = np.any(((ab != above) | (sb != stage_b)) & ~st.sliding[idx] & known, axis=(1, 2)) | branch_change
        if np.any(changed):  # a rate event before the boundary takes precedence
            self._cut(
                st,
                idx[changed],
                lo[changed],
                above[changed],
                stage_b[changed],
                branch0[changed],
                (v0[0][changed], v0[1][changed]),
            )
            keep = ~changed
            idx, lo, a2, c2, excess, kind = idx[keep], lo[keep], a2[keep], c2[keep], excess[keep], kind[keep]
            v0 = (v0[0][keep], v0[1][keep])
            if idx.size == 0:
                return
        which = np.argmax(excess, axis=1)
        t_new = st.t[idx] + lo
        st.steps[idx] += 1
        st.h_next[idx] = np.nan
        for j, i in enumerate(idx):
            a_i, c_i, k_i = float(a2[j]), float(c2[j]), int(kind[j])
            tabulated_surface = self._tabulated_family() and k_i == SURFACE and int(which[j]) in (2, 8, 9, 11)
            if self.cylinder is not None and (int(which[j]) >= 9 or tabulated_surface):
                self._cylinder_destination(st, int(i), int(which[j]), a_i, c_i, float(t_new[j]), bool(v0[0][j] > 0.0))
            elif k_i == SURFACE:
                b = int(which[j])
                if b == 0:
                    st.t_leak[i] = t_new[j]
                else:
                    if np.isnan(st.t_leak_assumed[i]):  # the first re-characterisation is the event's time
                        st.t_leak_assumed[i] = t_new[j]
                    st.depth_growing_at_recharacterisation[i] = bool(v0[0][j] > 0.0)
                if st.leak_trigger[i] == 0:  # the first leak / leak_assumed origin is the one reported
                    st.leak_trigger[i] = b + 1
                st.kind[i] = THROUGH
                st.a[i], st.c[i], st.t[i] = st.thickness[i], c_i, t_new[j]
                st.record_event(
                    int(i), (a_i, c_i, SURFACE), (float(st.thickness[i]), c_i, THROUGH), float(t_new[j]), b + 1
                )
                if self.cylinder is not None and self.cylinder.k_route == "tabulated":
                    self._through_wall_range_check(st, int(i), c_i, float(t_new[j]))
            elif k_i == EMBEDDED:
                self._embedded_destination(st, int(i), int(which[j]), a_i, c_i, float(t_new[j]), bool(v0[0][j] > 0.0))
            else:  # through flaw entering the terminal zone: no flaw-type change, the records are kept
                st.a[i], st.c[i], st.t[i] = a_i, c_i, t_new[j]

    def _embedded_destination(
        self, st: GrowthState, i: int, which: int, a_i: float, c_i: float, t: float, growing: bool
    ) -> None:
        """Embedded flaw at boundary ``which`` (5 breakthrough, 6 long-flaw limit, 7 aspect, 8 width): becomes a
        surface flaw of depth 2a + p and length 2c (theory manual, section 5.3.3), checked at once against the surface
        boundary set; an invalid destination cascades to the through flaw (every side a checkpoint)."""
        p = max(0.5 * (float(st.b_eff[i]) - 2.0 * a_i), 0.0)
        depth = 2.0 * a_i + p
        thick, width = float(st.thickness[i]), float(st.width[i])
        st.t[i] = t
        code = {5: 0, 6: 5, 7: 6, 8: 7}[which]  # a plain breakthrough carries no trigger code
        st.record_event(i, (a_i, c_i, EMBEDDED), (depth, c_i, SURFACE), t, code)
        if which == 5:
            st.t_surface_breaking[i] = t
        else:
            if np.isnan(st.t_leak_assumed[i]):
                st.t_leak_assumed[i] = t
                st.leak_trigger[i] = code
            st.depth_growing_at_recharacterisation[i] = growing
        st.kind[i] = SURFACE
        st.a[i], st.c[i] = depth, c_i  # B' is kept so that the pre-change embedded state can be assessed
        # destination check: a/B = 1 is a leak (the far surface is reached); other failures cascade to the
        # through flaw with leak_assumed (trigger 11)
        if depth >= thick * (1 - 1e-12):
            st.t_leak[i] = t
            if st.leak_trigger[i] == 0:
                st.leak_trigger[i] = 1
            st.record_event(i, (depth, c_i, SURFACE), (thick, c_i, THROUGH), t, 1)
            st.kind[i] = THROUGH
            st.a[i] = thick
            return
        excess = self.boundary_excess(
            np.array([depth]), np.array([c_i]), np.array([SURFACE]), np.array([thick]), np.array([width])
        )
        if np.any(excess[0, :4] > 0.0):
            if np.isnan(st.t_leak_assumed[i]):
                st.t_leak_assumed[i] = t
                st.leak_trigger[i] = 11
            st.record_event(i, (depth, c_i, SURFACE), (thick, c_i, THROUGH), t, 11)
            st.kind[i] = THROUGH
            st.a[i] = thick

    def _cut(
        self,
        st: GrowthState,
        idx: IntArray,
        h: FloatArray,
        above: BoolArray,
        stage_b: BoolArray,
        branch0: BoolArray,
        v0: tuple[FloatArray, FloatArray],
    ) -> None:
        """Locate the first active-set change on the step (bisection on the fraction of the step) and
        accept the step up to it; the next step starts on the far side with the set seen there. A reverse
        threshold change located at the start of a step means a sliding mode for that pair."""
        kind, thick, width = st.kind[idx], st.thickness[idx], st.width[idx]
        lo, hi = np.zeros(idx.size), np.ones(idx.size)
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            _, _, a2, c2 = self.heun(st, idx, h * mid, above, stage_b, v0)
            dk = self.ranges(idx, a2, c2, kind, thick, width)
            ab, sb, br = self.active_set(idx, dk, a2, c2, kind)
            differs = np.any(((ab != above) | (sb != stage_b)) & ~st.sliding[idx] & ~np.isnan(dk), axis=(1, 2)) | (
                (br != branch0) & ~st.ignore_branch[idx]
            )
            hi, lo = np.where(differs, mid, hi), np.where(differs, lo, mid)
        frac = lo
        _, _, a2, c2 = self.heun(st, idx, h * frac, above, stage_b, v0)
        a_far, c_far = self.heun(st, idx, h * hi, above, stage_b, v0)[2:]
        dk_end = self.ranges(idx, a_far, c_far, kind, thick, width)
        ab_end, sb_end, _ = self.active_set(idx, dk_end, a_far, c_far, kind)
        direction = np.where(ab_end & ~above, 1, np.where(~ab_end & above, -1, 0))
        reverse = (direction != 0) & (direction == -st.last_change[idx]) & (frac <= SLIDING_FRACTION)[:, None, None]
        st.sliding[idx] |= reverse
        st.slid[idx] |= np.any(reverse, axis=(1, 2))
        st.last_change[idx] = np.where(direction != 0, direction, st.last_change[idx])
        st.override_above[idx], st.override_stage[idx] = ab_end | st.sliding[idx], sb_end
        st.has_override[idx] = True
        at_start = frac <= EVENT_TOLERANCE
        st.force[idx] = at_start
        st.zero_cuts[idx] = np.where(at_start, st.zero_cuts[idx] + 1, 0)
        # a state hovering on the a/c = 1 branch change (equal growth of a and c) would be cut at the start
        # of every step: after three such cuts the branch change stops being an event for that sample
        st.ignore_branch[idx] |= (st.zero_cuts[idx] >= 3) & ~np.any(direction != 0, axis=(1, 2))
        st.a[idx], st.c[idx] = a2, c2
        st.t[idx] = st.t[idx] + h * frac
        st.steps[idx] += 1
        st.h_next[idx] = np.nan

    # ---- cylinders -------------------------------------------------------------------------------------

    def _finish_allowed(self) -> bool:
        """The quadrature finish exists for the plate width singularity: plate through flaws and the
        pressure-only cylinder route (plate f_w with W the cylinder length); never at a table edge."""
        return self.cylinder is None or self.cylinder.k_route == "pressure_only"

    def _cylinder_ranges(
        self,
        idx: IntArray,
        a: FloatArray,
        c: FloatArray,
        kind: IntArray,
        thickness: FloatArray,
        width: FloatArray,
        m: FloatArray,
        b: FloatArray,
    ) -> FloatArray:
        spec = self.cylinder
        assert spec is not None
        out = np.zeros((a.size, self.nb, 2))
        r_i = spec.r_i
        r_m = r_i + 0.5 * thickness
        axial = spec.orientation == "axial"
        surf = kind == SURFACE
        if np.any(surf):
            aa, cc, bb, ww = a[surf], c[surf], thickness[surf], width[surf]
            root = np.sqrt(np.pi * aa)
            if spec.k_route == "pressure_only":
                fw = plate.surface_flaw_fw(aa, cc, bb, ww)
                bulge = cyl.bulging_surface_axial(aa, cc, r_m[surf], bb) if axial else np.ones_like(aa)
                for p, theta in ((0, plate.THETA_DEEPEST), (1, plate.THETA_SURFACE)):
                    mm, mb = plate.surface_flaw_mm_mb(aa, cc, bb, theta)
                    ys = (bulge * fw)[:, None] * (mm[:, None] * m + mb[:, None] * b)
                    out[surf, :, p] = np.maximum(ys, 0.0) * root[:, None]
            else:
                table = tables.TABLES[
                    {("axial", False): "M.2", ("axial", True): "M.4", ("circumferential", False): "M.7"}.get(
                        (spec.orientation, spec.external), "M.2"
                    )
                ]
                if spec.orientation == "circumferential" and spec.external:
                    fw = plate.surface_flaw_fw(aa, cc, bb, ww)  # the plate solution carries its f_w (W = length)
                    for p, theta in ((0, plate.THETA_DEEPEST), (1, plate.THETA_SURFACE)):
                        mm, mb = plate.surface_flaw_mm_mb(aa, cc, bb, theta)
                        out[surf, :, p] = (
                            np.maximum(fw[:, None] * (mm[:, None] * m + mb[:, None] * b), 0.0) * root[:, None]
                        )
                else:
                    # no clamping: a state outside the table is invalid (NaN); the boundary set lands on the edge
                    vals = [cyl.interpolate(table, q, aa / cc, aa / bb, bb / r_i).value for q in range(4)]
                    for p, (mm, mb) in enumerate(((vals[0], vals[1]), (vals[2], vals[3]))):
                        out[surf, :, p] = np.maximum(mm[:, None] * m + mb[:, None] * b, 0.0) * root[:, None]
        thr = kind == THROUGH
        if np.any(thr):
            cc, bb, ww = c[thr], thickness[thr], width[thr]
            root = np.sqrt(np.pi * cc)
            if spec.k_route == "pressure_only":
                fw = plate.through_flaw_fw(np.minimum(cc, 0.5 * ww * (1 - 1e-12)), ww)
                bulge = cyl.bulging_through_axial(cc, r_m[thr], bb) if axial else np.ones_like(cc)
                out[thr, :, 1] = (bulge * fw)[:, None] * np.maximum(m + b, 0.0) * root[:, None]
            else:
                lam = cyl.shell_lambda(cc, r_m[thr], bb)
                names = ("M.1a", "M.1b", "M.1c", "M.1d") if axial else ("M.6a", "M.6b", "M.6c", "M.6d")
                m1, m2, m3, m4 = (
                    cyl.interpolate_columns(tables.TABLES[n], 0, lam, r_m[thr] / bb).value  # no clamp: NaN past the end
                    for n in names
                )
                sign = 1.0 if spec.bending_tensile_surface == "outside" else -1.0  # linear superposition
                beta = cyl.shell_beta(cc, r_m[thr]) if not axial else np.ones_like(cc)
                outside = (m1 + m2)[:, None] * m * beta[:, None] + sign * (m3 + m4)[:, None] * b
                inside = (m1 - m2)[:, None] * m * beta[:, None] + sign * (m3 - m4)[:, None] * b
                out[thr, :, 1] = np.maximum(np.maximum(outside, inside), 0.0) * root[:, None]  # the larger tip
        ext = kind == EXTENDED
        if np.any(ext):
            aa, bb = a[ext], thickness[ext]
            root = np.sqrt(np.pi * aa)
            if spec.k_route == "pressure_only" or (not axial and spec.external):
                if not axial and spec.external:
                    mm = cyl.fully_circumferential_external_mm(aa, r_i, r_i + bb)
                    mb = mm
                else:
                    # no clamping: beyond the polynomial's range the rate is undefined (NaN); the boundary set
                    # lands on 0.6 B (within rounding, as the table edges)
                    mm, mb = cyl.extended_plate_mm_mb(aa, bb)
                    inside = aa / bb <= cyl.EXTENDED_PLATE_A_OVER_B_MAX * (1.0 + 1e-12)
                    mm, mb = np.where(inside, mm, np.nan), np.where(inside, mb, np.nan)
            else:
                name = {("axial", False): "M.3", ("axial", True): "M.5", ("circumferential", False): "M.8"}[
                    (spec.orientation, spec.external)
                ]
                table = tables.TABLES[name]
                ab = aa / bb  # no clamping: outside the table the rate is undefined (NaN)
                if name == "M.5":
                    mm = cyl.interpolate(table, 0, bb / r_i, ab).value
                    mb = cyl.interpolate(table, 1, bb / r_i, ab).value
                else:
                    mm = cyl.interpolate(table, 0, ab, bb / r_i).value
                    mb = cyl.interpolate(table, 1, ab, bb / r_i).value
            out[ext, :, 0] = np.maximum(mm[:, None] * m + mb[:, None] * b, 0.0) * root[:, None]
        return np.asarray(out * self.scale[idx][:, None, None], dtype=float)

    def _tabulated_family(self) -> bool:
        """True when the flaw family of this cylinder uses the tabulated surface/extended factors (the
        circumferential external family uses the plate and (M.26) solutions under both routes)."""
        spec = self.cylinder
        if spec is None:
            return False
        return spec.k_route == "tabulated" and not (spec.orientation == "circumferential" and spec.external)

    def _table_ac_min(self) -> float:
        spec = self.cylinder
        assert spec is not None
        return 0.05 if spec.orientation == "axial" else 0.1

    def _lambda_end(self, thickness: FloatArray) -> FloatArray:
        """The through-wall table's interpolable lambda end at this cylinder's r_m/B (column dependent)."""
        spec = self.cylinder
        assert spec is not None
        name = "M.1a" if spec.orientation == "axial" else "M.6a"
        return cyl.lambda_range_end(tables.TABLES[name], (spec.r_i + 0.5 * thickness) / thickness)

    def _cylinder_boundaries(
        self, out: FloatArray, a: FloatArray, c: FloatArray, kind: IntArray, thickness: FloatArray, width: FloatArray
    ) -> FloatArray:
        """Boundary excess for cylinder kinds: 0-4 the plate entries where the plate solutions are in use;
        9 table edge a/B = 0.8 (surface, tabulated); 10 through-wall lambda range end (tabulated);
        11 table edge 2c/W = 0.15 (surface, tabulated); 12 extended a/B limit (0.8 tabulated, 0.6 plate);
        13 extended a/B = 1 (leak)."""
        spec = self.cylinder
        assert spec is not None
        s, t, x = kind == SURFACE, kind == THROUGH, kind == EXTENDED
        ab = a / thickness
        tabulated = self._tabulated_family()
        if tabulated:
            out[:, 9] = np.where(s, ab - cyl.TABLE_A_OVER_B_MAX, -np.inf)
            out[:, 2] = np.where(s, a / (2.0 * c) - 0.5, -np.inf)  # the tables end at a/c = 1
            out[:, 8] = np.where(s, self._table_ac_min() - a / c, -np.inf)  # and start at their a/c minimum
            if spec.orientation == "axial":
                out[:, 11] = np.where(s, 2.0 * c / width - 0.15, -np.inf)  # the axial tables' width limit only
        else:
            out[:, 0] = np.where(s, ab - 1.0, -np.inf)
            long = s & (a / (2.0 * c) <= 0.1)
            out[:, 1] = np.where(long, ab - 1.25 * (a / c + 0.6), -np.inf)
            out[:, 2] = np.where(s, a / (2.0 * c) - 1.0, -np.inf)
            out[:, 3] = np.where(s, 2.0 * c / width - SURFACE_WIDTH_LIMIT, -np.inf)
        if spec.k_route == "tabulated":
            r_m = spec.r_i + 0.5 * thickness
            lam_end_t = self._lambda_end(thickness)
            out[:, 10] = np.where(t, cyl.shell_lambda(c, r_m, thickness) - lam_end_t, -np.inf)
            out[:, 12] = np.where(x, ab - cyl.TABLE_A_OVER_B_MAX, -np.inf)
        else:
            out[:, 4] = np.where(t, 2.0 * c / width - TERMINAL_ZONE, -np.inf)
            out[:, 12] = np.where(x, ab - cyl.EXTENDED_PLATE_A_OVER_B_MAX, -np.inf)
        if spec.orientation == "circumferential" and spec.external:
            out[:, 12] = -np.inf
        out[:, 13] = np.where(x, ab - 1.0, -np.inf)
        # kinds whose K is singular at a = B ((M.22) axial surface flaw under pressure-only; the (M.26) fully
        # circumferential external flaw): a declared end at a/B = 0.9 (trigger code 17)
        singular = np.zeros(a.shape, dtype=bool)
        if spec.k_route == "pressure_only" and spec.orientation == "axial":
            singular |= s
        if spec.orientation == "circumferential" and spec.external:
            singular |= x
        out[:, 14] = np.where(singular, ab - SINGULAR_ZONE, -np.inf)
        out[:, 0] = np.where(singular & s, -np.inf, out[:, 0])
        out[:, 13] = np.where(singular & x, -np.inf, out[:, 13])
        return np.asarray(out, dtype=float)

    def _through_wall_range_check(self, st: GrowthState, i: int, c_i: float, t: float) -> None:
        """A through-wall destination outside the tabulated through-wall range (lambda or r_m/B) is ENDED at once
        (trigger 10); every side is a checkpoint."""
        spec = self.cylinder
        assert spec is not None
        thick = float(st.thickness[i])
        r_m = spec.r_i + 0.5 * thick
        lam = float(cyl.shell_lambda(np.array([c_i]), np.array([r_m]), np.array([thick]))[0])
        lam_end = float(self._lambda_end(np.array([thick]))[0])
        if not (lam <= lam_end) or not (cyl.RM_OVER_B_MIN <= r_m / thick <= cyl.RM_OVER_B_MAX):
            st.record_event(i, (thick, c_i, THROUGH), (thick, c_i, ENDED), t, 10)
            st.kind[i], st.t_ended[i] = ENDED, t

    def _cylinder_destination(
        self, st: GrowthState, i: int, which: int, a_i: float, c_i: float, t: float, growing: bool
    ) -> None:
        """Cylinder boundaries: a surface flaw at a table edge becomes the through-wall flaw of length 2c with
        leak_assumed (trigger 8, 9 or 15), checked at once against the through-wall range (ENDED, trigger 10,
        if outside); the through-wall range end is ENDED (trigger 10); an extended flaw at a/B = 1 leaks and is
        ENDED; at its solution's depth limit it is leak_assumed and ENDED (trigger 12)."""
        spec = self.cylinder
        assert spec is not None
        thick = float(st.thickness[i])
        st.t[i] = t
        st.a[i], st.c[i] = a_i, c_i  # the landed boundary state
        if which in (2, 8, 9, 11):
            code = {2: 9, 8: 9, 9: 8, 11: 15}[which]
            if np.isnan(st.t_leak_assumed[i]):
                st.t_leak_assumed[i] = t
                st.leak_trigger[i] = code
            st.depth_growing_at_recharacterisation[i] = growing
            st.record_event(i, (a_i, c_i, SURFACE), (thick, c_i, THROUGH), t, code)
            st.kind[i], st.a[i], st.c[i] = THROUGH, thick, c_i
            self._through_wall_range_check(st, i, c_i, t)
            return
        if which == 10:
            st.record_event(i, (a_i, c_i, THROUGH), (a_i, c_i, ENDED), t, 10)
            st.kind[i], st.t_ended[i] = ENDED, t
            return
        if which == 14:  # the singular-zone end of a bulging or (M.26) solution (trigger code 17)
            if np.isnan(st.t_leak_assumed[i]):
                st.t_leak_assumed[i] = t
                st.leak_trigger[i] = 17
            kind_i = int(st.kind[i])
            st.record_event(i, (a_i, c_i, kind_i), (a_i, c_i, ENDED), t, 17)
            st.kind[i], st.a[i], st.c[i], st.t_ended[i] = ENDED, a_i, c_i, t
            return
        if which == 13:
            st.t_leak[i] = t
            if st.leak_trigger[i] == 0:
                st.leak_trigger[i] = 1
            code = 1
        else:  # 12: the extended solution's depth limit (16 for the plate polynomial's 0.6 B under pressure-only)
            code = 16 if spec.k_route == "pressure_only" else 12
            if np.isnan(st.t_leak_assumed[i]):
                st.t_leak_assumed[i] = t
                st.leak_trigger[i] = code
        st.record_event(i, (a_i, c_i, EXTENDED), (a_i, c_i, ENDED), t, code)
        st.kind[i], st.a[i], st.t_ended[i] = ENDED, a_i, t

    # ---- quadrature finish ------------------------------------------------------------------

    def _through_velocity(self, i: int, c: FloatArray, w: float, t: float) -> FloatArray:
        """dc/dt of sample i as a through flaw at half-lengths c, under the block(s) active at time t."""
        n = c.size
        idx = np.full(n, i)
        kind = np.full(n, THROUGH)
        thick = np.full(n, float(self.thickness[i]))
        dk = self.ranges(idx, thick, c, kind, thick, np.full(n, w))
        above, stage_b, _ = self.active_set(idx, dk, thick, c, kind)
        rate = self.stage_rate(idx, dk, above, stage_b)
        w_blocks = self.block_weights(np.full(n, t))
        return np.asarray(np.sum(rate[:, :, 1] * w_blocks, axis=1), dtype=float)

    def _kinks(self, i: int, c0: float, w: float, t: float) -> list[float]:
        """Half-lengths between c0 and W/2 at which a block's range crosses the threshold or the stage
        transition (dK increases with c for a through flaw): kinks of the rate, made grid points."""
        thick1 = np.array([float(self.thickness[i])])
        levels = [float(self.thr[i])]
        tr = self.law.transition()
        if tr is not None:
            levels.append(float(np.broadcast_to(tr, (self.n,))[i]))
        out: list[float] = []
        c_hi = 0.5 * w * (1 - 1e-12)
        for lvl in levels:
            if lvl <= 0.0:
                continue
            for ib in range(self.nb):

                def dk_of(cc: float, ib: int = ib) -> float:
                    one = np.array([i])
                    return float(
                        self.ranges(one, thick1, np.array([cc]), np.array([THROUGH]), thick1, np.array([w]))[0, ib, 1]
                    )

                lo_c, hi_c = c0, c_hi
                if (dk_of(lo_c) - lvl) * (dk_of(hi_c) - lvl) < 0.0:
                    for _ in range(60):
                        mid = 0.5 * (lo_c + hi_c)
                        if (dk_of(mid) - lvl) * (dk_of(lo_c) - lvl) <= 0.0:
                            hi_c = mid
                        else:
                            lo_c = mid
                    out.append(0.5 * (lo_c + hi_c))
        return out

    def _panel_integrals(self, i: int, s: FloatArray, w: float, t: float) -> FloatArray:
        """Integral of 2 s / v(W/2 - s^2) over each panel of the decreasing grid ``s`` by Gauss-Legendre
        quadrature. The nodes are interior to the panels, so a jump of the rate at a panel end (a block
        crossing its threshold) never enters a panel: the integrand is smooth inside every panel."""
        c_hi = 0.5 * w
        mid = 0.5 * (s[:-1] + s[1:])
        half = 0.5 * (s[:-1] - s[1:])
        nodes = mid[:, None] - half[:, None] * GL_NODES[None, :]  # (panels, nodes)
        c = np.minimum(c_hi - nodes.ravel() ** 2, c_hi * (1 - 1e-15))
        v = self._through_velocity(i, c, w, t).reshape(nodes.shape)
        with np.errstate(divide="ignore"):
            f = np.where(v > 0.0, 2.0 * nodes / v, np.inf)
        return np.asarray(np.sum(f * GL_WEIGHTS[None, :], axis=1) * half, dtype=float)

    def _time_table(self, i: int, c0: float, w: float, t: float) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Cumulative time to reach each size of a grid from c0 to W/2 (the integral of 1/v), with the
        end-point singularity removed by c = W/2 - s^2, the kinks of the rate as grid points and panel-wise
        Gauss-Legendre quadrature, the grid doubled until two levels agree to FINISH_TOLERANCE on the total.
        Returns (s, sizes, times): the grid in s (decreasing), in c (increasing) and the cumulative times."""
        c_hi = 0.5 * w
        s0 = np.sqrt(max(c_hi - c0, 0.0))
        s_kinks = np.sqrt(np.clip(c_hi - np.asarray(self._kinks(i, c0, w, t)), 0.0, s0**2))
        points = 1001
        prev_total: float | None = None
        for _ in range(10):
            s = np.unique(np.concatenate([np.linspace(s0, 0.0, points), s_kinks]))[::-1]  # decreasing s
            c = c_hi - s**2
            panels = self._panel_integrals(i, s, w, t)
            if not np.all(np.isfinite(panels)):
                return s, np.asarray(c), np.full(c.size, np.inf)
            cum = np.concatenate([[0.0], np.cumsum(panels)])
            total = float(cum[-1])
            if prev_total is not None and abs(total - prev_total) <= FINISH_TOLERANCE * max(abs(total), 1e-300):
                return s, np.asarray(c), np.asarray(cum)
            prev_total = total
            points = 2 * points - 1
        raise RuntimeError("the quadrature finish did not converge to the required tolerance")

    def _size_at(
        self, i: int, table: tuple[FloatArray, FloatArray, FloatArray], w: float, t: float, tau: float
    ) -> float:
        """The half-length reached after a time ``tau`` (within the table's range), by solving for the
        position inside the bracketing panel with the same quadrature (bisection then Newton on s)."""
        s, sizes, times = table
        j = int(np.searchsorted(times, tau, side="right") - 1)
        j = min(max(j, 0), s.size - 2)
        target = tau - float(times[j])
        s_top = float(s[j])  # s decreases with c: the solution lies in [s_{j+1}, s_j]
        lo, hi = float(s[j + 1]), s_top
        if target <= 0.0:
            return float(sizes[j])

        def elapsed(x: float) -> float:
            return float(self._panel_integrals(i, np.array([s_top, x]), w, t)[0])

        x = 0.5 * (lo + hi)
        for _ in range(100):
            r = elapsed(x) - target
            if r > 0.0:
                lo = x
            else:
                hi = x
            c_x = min(0.5 * w - x * x, 0.5 * w * (1 - 1e-15))
            v = float(self._through_velocity(i, np.array([c_x]), w, t)[0])
            step = r * v / (2.0 * x) if x > 0.0 and v > 0.0 else 0.0  # d(elapsed)/ds = -2 s / v
            x_new = x + step
            if not (lo < x_new < hi):
                x_new = 0.5 * (lo + hi)
            if abs(x_new - x) <= 1e-15 * max(abs(x), 1e-300) or hi - lo <= 1e-15 * max(hi, 1e-300):
                x = x_new
                break
            x = x_new
        return float(0.5 * w - x * x)

    def _finish(self, st: GrowthState, i: int, t_end: float) -> None:
        """Through flaw in the terminal zone: time to the width limit by size quadrature of 1/v, block by
        block under an ordered spectrum; the state at ``t_end`` is read from the size-time table."""
        for _ in range(2 + (0 if self.block_ends is None else self.block_ends.size)):
            c0, w, t0 = float(st.c[i]), float(st.width[i]), float(st.t[i])
            if t0 >= t_end:
                return
            if self.block_ends is not None and t0 >= self.block_ends[-1] * (1 - 1e-12):
                st.t[i] = t_end  # no cycles left
                return
            block_end = float(self._next_block_end(np.array([t0]))[0]) if self.block_ends is not None else np.inf
            table = self._time_table(i, c0, w, t0)
            times = table[2]
            if not np.all(np.isfinite(times)):
                if self.block_ends is None:
                    st.dormant[i] = True
                    st.t[i] = t_end
                    return
                st.t[i] = min(block_end, t_end)
                continue
            t_limit = t0 + float(times[-1])
            stop = min(t_end, block_end)
            if t_limit <= stop:
                st.t_ended[i] = t_limit
                st.kind[i] = ENDED
                st.c[i] = 0.5 * w
                st.t[i] = t_end
                return
            st.c[i] = self._size_at(i, table, w, t0, stop - t0)
            st.t[i] = stop
            if stop >= t_end:
                return
