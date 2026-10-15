"""``FECase`` / ``FEResult`` and the ``JSolver`` (theory manual, sections 6.1.2 and 6.1.7): the immutable case record
with the declared parameter box, its canonical JSON and case id, and the WARP3D solver that runs the elastic
admissibility case (signed ``K_I > 0`` at every front position), the plastic case with its elastic companion,
collects every QA check with its value and band, and labels the result.

Verification status: ``fe-failed`` when any QA check fails (the failing check named); ``fe-verified`` only when
every check passes **and** the family verification is complete: D4, D5, D6, D7 and D8 passed and the D3
deep/bending band resolved (theory manual, section 6.2; D4 and D5 are pending their primary references and D7 is
unresolved, so every passing case is ``unverified`` until then — the flag ``FAMILY_VERIFIED`` records it).

Units: the case is SI; the deck is written in N, mm, MPa (``critflaw.fe.units``)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from critflaw.fe import crack_mesh as CM
from critflaw.fe import mesh as M
from critflaw.fe.elastic_case import ElasticCase, ElasticResult, SolveFailed
from critflaw.fe.elastic_case import run_case as run_elastic
from critflaw.fe.plastic_case import LOCAL_STRAIN_CAP, PlasticCase, PlasticResult
from critflaw.fe.plastic_case import run_case as run_plastic
from critflaw.fe.runner import PARSER_VERSION, RunRecord, SolveCache, WarpRunner
from critflaw.fe.warp_law import WarpDeformationLaw

FloatArray = NDArray[np.float64]

FAMILY = "plate_surface_flaw"
LEVEL_1 = (6, 12, 12, 3, 3)  # the production mesh level (D7, theory manual section 6.2)
LEVELS = (LEVEL_1, (8, 16, 16, 4, 4), (10, 20, 20, 5, 5), (12, 24, 24, 6, 6))
FAMILY_VERIFIED = False  # D4, D5 pending; D7 unresolved for the family; D3 deep/bending band unresolved
PATH_INDEPENDENCE_BAND = 0.02  # theory manual section 6.1.7: J over rings 2..n at every position and step
REMOTE_PROFILE_BAND = 0.005  # D8: the remote stress profile at the declared distance (Saint-Venant check of L_far)
EQUILIBRIUM_BAND = 1e-4  # D8: reactions balance the applied force and moment (five printed digits per row)
J1_ELASTIC_BAND = 0.005  # condition (b) of the J_e provenance (theory manual, section 6.1.4)
ZONE_STRESS_RATIO = 0.95  # condition (a): sigma_vm / sigma_0 outside ring 1 after step 1
PARAMETER_BOX: dict[str, tuple[float, float]] = {
    "n": (3.0, 20.0),
    "sigma_0_over_E": (1e-3, 5e-3),
    "nu": (0.25, 0.35),
    "beta": (0.0, 1.0),
    "a_over_B": (0.2, 0.6),  # the range covered by the current verification evidence
    "a_over_c": (0.4, 1.0),
    "c_over_W": (0.0, 0.1),
}


def _in(name: str, value: float) -> None:
    lo, hi = PARAMETER_BOX[name]
    if not (lo <= value <= hi):
        raise ValueError(f"{name} = {value:g} outside the declared box [{lo:g}, {hi:g}]")


@dataclass(frozen=True)
class FECase:
    """Immutable case record (SI). Geometry ``a, c, B, W, L_far`` (m), material ``E, sigma_0`` (Pa), ``nu, n``,
    the loading mix ``beta = Pb / (Pm + Pb)``, the mesh level vector, the parametric evaluation angle ``phi_s``,
    the load schedule controls and the solver controls that enter the deck."""

    a: float
    c: float
    B: float
    W: float
    L_far: float
    E: float
    nu: float
    sigma_0: float
    n: float
    beta: float
    level: tuple[int, int, int, int, int] = LEVEL_1
    phi_s_deg: float = 10.0
    n_steps: int = 12
    lambda_1_lr: float = 0.05
    norm_res_tol: float = 0.002
    threads: int = 4
    family: str = FAMILY

    def __post_init__(self) -> None:
        if self.family != FAMILY:
            raise NotImplementedError(f"TODO(verify): geometry family {self.family!r} (only {FAMILY} exists)")
        for name, v in (("a", self.a), ("c", self.c), ("B", self.B), ("W", self.W), ("L_far", self.L_far)):
            if not v > 0.0:
                raise ValueError(f"{name} must be positive")
        _in("n", self.n)
        _in("sigma_0_over_E", self.sigma_0 / self.E)
        _in("nu", self.nu)
        _in("beta", self.beta)
        _in("a_over_B", self.a / self.B)
        _in("a_over_c", self.a / self.c)
        _in("c_over_W", self.c / self.W)
        if self.L_far < CM.remote_length(self.B, self.W) * (1.0 - 1e-9):
            raise ValueError(
                f"L_far = {self.L_far:g} m is below the converged remote length max(20 B, 2 W) = "
                f"{CM.remote_length(self.B, self.W):g} m (the D7b far-field check)"
            )
        if self.level not in LEVELS:
            raise ValueError(f"level {self.level} is not one of the declared levels {LEVELS}")
        if not 0.0 < self.phi_s_deg < 90.0:
            raise ValueError("phi_s must lie strictly between 0 and 90 degrees")

    def with_(self, **changes: Any) -> FECase:
        return dataclasses.replace(self, **changes)

    @property
    def law(self) -> WarpDeformationLaw:
        return WarpDeformationLaw(self.E, self.nu, self.sigma_0, self.n)

    @property
    def phi_s_physical_deg(self) -> float:
        """The physical angle from the free surface of the parametric angle ``phi_s``."""
        return float(np.degrees(np.arctan(self.a / self.c * np.tan(np.radians(self.phi_s_deg)))))

    def mesh_params(self) -> CM.CrackBlockParams:
        return self.plastic_case().params()

    def plastic_case(self) -> PlasticCase:
        return PlasticCase(
            a=self.a, c=self.c, B=self.B, W=self.W, L_far=self.L_far, E=self.E, nu=self.nu, sigma_0=self.sigma_0,
            n=self.n, beta=self.beta, level=self.level, n_steps=self.n_steps, lambda_1_lr=self.lambda_1_lr,
            phi_s_deg=self.phi_s_deg, norm_res_tol=self.norm_res_tol, far_strain_output=True,
        )  # fmt: skip

    def elastic_case(self) -> ElasticCase:
        """The admissibility case: the unit load pattern (1 MPa at the declared mix) on the tied quarter-point mesh
        of the same level."""
        nr, ns, nf, nfs, nb = self.level
        pm, pb = self.plastic_case().unit_stresses()
        return ElasticCase(
            a=self.a, c=self.c, B=self.B, W=self.W, L_far=self.L_far, E=self.E, nu=self.nu, pm=pm, pb=pb,
            n_rings=nr, n_sectors=ns, n_front=nf, n_front_s=nfs, phi_s_deg=self.phi_s_deg, n_blend=nb,
        )  # fmt: skip

    def canonical_json(self) -> str:
        d = dataclasses.asdict(self)
        p = self.mesh_params()
        d["tube_rule"] = {"k1": p.k1, "k2": p.k2, "k3": p.k3}  # the constants of the mesh actually built
        d["topology_id"] = p.topology_id
        return json.dumps(d, sort_keys=True, separators=(",", ":"))

    @property
    def case_id(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class QACheck:
    name: str
    value: float
    band: float
    passed: bool
    note: str = ""


@dataclass(frozen=True)
class FEResult:
    case: FECase
    status: str  # complete | truncated | failed | timeout (of the plastic run)
    verification_status: str  # unverified | fe-verified | fe-failed
    failing_check: str | None
    topology_id: str
    topology_hash: str
    lam: FloatArray  # cumulative load factors read back from the output
    lr: FloatArray
    phi: FloatArray  # parametric angles of the front stations
    phi_physical: FloatArray
    j: FloatArray  # (steps, stations), J/m^2, mean over rings 2..n
    j_spread: FloatArray
    j_e: FloatArray
    kr: FloatArray
    k_elastic: FloatArray  # signed K_I per station (Pa sqrt(m)) of the admissibility case per unit factor
    outside_small_strain: tuple[bool, ...]
    max_far_eff_strain: FloatArray
    qa: tuple[QACheck, ...]
    lr_end: float
    proof_stress_0p2: float
    considere_stress: float
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def usable_steps(self) -> FloatArray:
        """Mask of the steps that may enter an Option 3 curve: inside the small-strain cap."""
        return np.asarray([not o for o in self.outside_small_strain], dtype=bool)


MIN_SCALED_JACOBIAN = 0.2  # ordinary bricks outside the tube (achieved on every box corner)


def mesh_checks(cb: CM.CrackBlockMesh, p: CM.CrackBlockParams) -> tuple[QACheck, ...]:
    """Mesh-layer QA (theory manual, section 6.1.7): positive Jacobian at every 2x2x2 Gauss point of every element
    (the collapsed tip elements included), the scaled-Jacobian band for the ordinary bricks outside the tube,
    the tip-element count and the coincident-node pattern confined to the front, and the aspect-ratio extremes
    reported (not banded: the thin footprint and far-field layers)."""
    m = cb.mesh
    j = M.jacobians(m, 2)
    sj = M.scaled_jacobian(m)
    n_el = m.n_elements
    tube = np.zeros(n_el, bool)
    tube[cb.block_elements[: p.n_rings * p.n_sectors * p.n_front] - 1] = True
    tip = np.zeros(n_el, bool)
    tip[np.asarray(cb.tip_elements) - 1] = True
    expected_tips = p.n_sectors * p.n_front
    ar = np.full(n_el, np.nan)
    ordinary = np.flatnonzero(~tip)
    ar[ordinary] = M.aspect_ratio(M.Hex20Mesh(m.coords, m.incid[ordinary]))  # collapsed bricks have a zero edge
    pairs = M.coincident_nodes(m)
    front = {int(n) for nodes in cb.front_nodes.values() for n in nodes}
    off_front = sum(1 for a, b in pairs if a not in front or b not in front)
    n_corner = 2 * p.n_sectors + 1 if p.tip == "untied" else 1
    n_mid = p.n_sectors + 1 if p.tip == "untied" else 1
    lig = set(int(n) for n in cb.ligament_nodes)
    pattern_ok = all(
        len(nodes) == (n_corner if j % 2 == 0 else n_mid)
        and len(set(nodes)) == len(nodes)
        and nodes[0] in lig
        and not any(n in lig for n in nodes[1:])
        for j, nodes in cb.front_nodes.items()
    )
    repeated = sum(1 for e in cb.tip_elements if len(set(m.incid[e - 1].tolist())) < 20)
    sz = CM.template_2d(p)
    quarter = abs(float(np.hypot(*sz[1, 0]) / np.hypot(*sz[2, 0])) - 0.25)
    return (
        QACheck("jacobian_positive", float(j.min()), 0.0, bool(np.all(j > 0.0)),
                "min det J over every 2x2x2 Gauss point, tip elements included (mm^3)"),
        QACheck("scaled_jacobian_outside_tube", float(sj[~tube].min()), MIN_SCALED_JACOBIAN,
                bool(sj[~tube].min() >= MIN_SCALED_JACOBIAN), "ordinary bricks outside the tube"),
        QACheck("scaled_jacobian_rings_2plus", float(sj[tube & ~tip].min()), MIN_SCALED_JACOBIAN,
                bool(sj[tube & ~tip].min() >= MIN_SCALED_JACOBIAN), "tube rings 2..n (ring 1 is collapsed)"),
        QACheck("tip_element_count", float(tip.sum()), float(expected_tips), int(tip.sum()) == expected_tips,
                "n_sectors x n_front collapsed bricks"),
        QACheck("coincident_nodes_on_front_only", float(off_front), 0.0, off_front == 0,
                f"{len(pairs)} coincident pairs, all on the crack front (untied tip)"),
        QACheck("tip_node_pattern", float(pattern_ok), 1.0, pattern_ok,
                "one front node per angular position (untied) or one (tied); the crack-plane constraint on the "
                "ligament-side node only"),
        QACheck("tip_repeated_incidences", float(repeated), float(p.tip == "tied") * float(len(cb.tip_elements)),
                (repeated == 0) if p.tip == "untied" else (repeated == len(cb.tip_elements)),
                "untied: no repeated node number in a tip element; tied: every tip element collapsed"),
        QACheck("quarter_point_midside", quarter, 1e-12, (quarter <= 1e-12) if p.tip == "tied" else True,
                "tied mesh: the ring-1 radial mid-side at a quarter of the ring-1 radius"),
        QACheck("aspect_ratio_max_outside_tube", float(ar[~tube].max()), float("inf"), True,
                "reported, not banded (thin footprint and far-field layers)"),
        QACheck("aspect_ratio_max_tube", float(np.nanmax(ar[tube])), float("inf"), True,
                "rings 2..n, reported, not banded"),
    )  # fmt: skip


def qa_checks(case: FECase, elastic: ElasticResult, plastic: PlasticResult) -> tuple[QACheck, ...]:
    """Every mechanics-layer check with its value and band (the run-layer checks are enforced in the case
    runners and raise)."""
    interior = slice(1, None)  # the free-surface vertex is excluded from every criterion
    k_min = float(np.min(elastic.k_fe[interior]))
    spread = float(np.max(plastic.j_spread[:, interior]))
    j1 = float(np.nanmax(np.abs(plastic.j1_vs_elastic[interior])))
    far = plastic.max_far_eff_strain
    far_max = float(np.nanmax(far)) if np.any(~np.isnan(far)) else float("nan")
    first = plastic.assessed_steps[0] if plastic.assessed_steps else None
    first_assessed_strain = float(far[first]) if first is not None and first < far.size else float("nan")
    first_assessed_ok = bool(
        first is not None
        and first < len(plastic.outside_small_strain)
        and not plastic.outside_small_strain[first]
        and np.isfinite(first_assessed_strain)
    )
    eq_e = elastic.equilibrium
    eq_p = plastic.equilibrium
    eq_c = plastic.equilibrium_elastic  # the companion on the untied mesh
    eq_all = eq_e + eq_p + eq_c
    eq_worst = max([q.force_residual for q in eq_all] + [q.moment_residual for q in eq_all] + [0.0])
    eq_ok = bool(
        eq_e and eq_p and eq_c and all(q.passed for q in eq_all) and len(eq_p) == plastic.lam.size and len(eq_c) == 1
    )
    return (
        QACheck("elastic_K_I_positive", k_min, 0.0, k_min > 0.0, "min signed K_I over interior stations, Pa sqrt(m)"),
        QACheck("remote_profile", elastic.remote_profile_dev, REMOTE_PROFILE_BAND,
                bool(elastic.remote_profile_dev <= REMOTE_PROFILE_BAND),
                f"D8: worst |sigma_zz - profile| / (|Pm| + |Pb|) at z/B = {elastic.remote_profile_z_over_B:.2f}"),
        QACheck("equilibrium", eq_worst, EQUILIBRIUM_BAND, eq_ok,
                "D8: worst force/moment residual of the reactions against the applied resultants, every step of "
                "all three runs (admissibility, plastic, companion), every constrained node present"),
        QACheck("J_path_independence", spread, PATH_INDEPENDENCE_BAND, spread <= PATH_INDEPENDENCE_BAND,
                "max relative deviation over rings 2..n, every step and interior station"),
        QACheck("plastic_zone_in_ring_1", plastic.max_vm_outside_ring_1_over_sigma_0, ZONE_STRESS_RATIO,
                plastic.plastic_zone_in_ring_1, "max sigma_vm / sigma_0 outside ring 1 after step 1"),
        QACheck("J_1_vs_elastic_same_mesh", j1, J1_ELASTIC_BAND, j1 <= J1_ELASTIC_BAND,
                "max |J_1 / J_elastic(lambda_1) - 1| over interior stations"),
        QACheck("far_region_strain_evaluated", 0.0 if np.isnan(far_max) else 1.0, 1.0, not np.isnan(far_max),
                "the far-region effective strain was output at the steps with Lr >= 0.8"),
        QACheck("small_strain_cap_first_assessed_step", first_assessed_strain, LOCAL_STRAIN_CAP,
                first_assessed_ok,
                "the far-region effective strain at the first step with Lr >= 0.8; later steps beyond the cap are "
                "stored, marked outside_small_strain and never fe-verified"),
    )  # fmt: skip


class WarpJSolver:
    """``run(case) -> FEResult`` through the WARP3D runner; job directories under ``root/<case id>``, the
    solve cache under ``root/cache`` (keyed by the generated inputs and the execution identity, so a changed
    case or image is a miss). The case's ``threads`` is the thread count of the solve. A solve that does not
    complete is returned as an ``FEResult`` with its status and ``fe-failed``, never raised."""

    def __init__(self, root: Path, runner: WarpRunner | None = None, cache: SolveCache | None = None) -> None:
        self.root = Path(root)
        self.runner = runner or WarpRunner(threads=4, memory="8g", timeout_seconds=5400)
        self.cache = cache if cache is not None else SolveCache(self.root / "cache")

    def run(self, case: FECase) -> FEResult:
        work = self.root / case.case_id
        work.mkdir(parents=True, exist_ok=True)
        (work / "case.json").write_text(case.canonical_json(), encoding="utf-8")
        runner = dataclasses.replace(self.runner, threads=case.threads)
        try:
            elastic = run_elastic(case.elastic_case(), work / "elastic", runner, self.cache)
            if np.any(elastic.k_fe[1:] <= 0.0):  # crack-face contact: the nonlinear run is refused (FE-03)
                refused = dataclasses.replace(elastic.record, status="failed")
                return failed_result(case, refused, "elastic_K_I_positive: crack-face contact", runner)
            plastic = run_plastic(case.plastic_case(), work / "plastic", runner, self.cache)
        except SolveFailed as exc:
            return failed_result(case, exc.record, str(exc), runner)
        return assemble(case, elastic, plastic, runner)


def failed_result(case: FECase, record: RunRecord, reason: str, runner: WarpRunner | None) -> FEResult:
    """The result of a solve that did not complete: its status, ``fe-failed``, the reason as the failing
    check, empty arrays."""
    status = record.status if record.status != "complete" else "failed"
    empty = np.zeros(0)
    law = case.law
    return FEResult(
        case=case, status=status, verification_status="fe-failed", failing_check=f"solve: {reason}",
        topology_id=case.mesh_params().topology_id, topology_hash="", lam=empty, lr=empty, phi=empty,
        phi_physical=empty, j=np.zeros((0, 0)), j_spread=np.zeros((0, 0)), j_e=np.zeros((0, 0)),
        kr=np.zeros((0, 0)), k_elastic=empty, outside_small_strain=(), max_far_eff_strain=empty, qa=(),
        lr_end=case.plastic_case().lr_end(), proof_stress_0p2=float(law.proof_stress_0p2()),
        considere_stress=float(law.considere_stress()), provenance=provenance_of(case, runner),
    )  # fmt: skip


def provenance_of(case: FECase, runner: WarpRunner | None) -> dict[str, Any]:
    return {
        "parser_version": PARSER_VERSION,
        "image": runner.image if runner is not None else None,
        "image_id": runner.image_id if runner is not None else None,
        "warp3d_exe": runner.warp3d_exe if runner is not None else None,
        "threads": runner.threads if runner is not None else case.threads,
        "deck_units": "N-mm-MPa",
        "family_verification": "D6, D8 passed; D4, D5 pending" if not FAMILY_VERIFIED else "complete",
    }


def assemble(case: FECase, elastic: ElasticResult, plastic: PlasticResult, runner: WarpRunner | None) -> FEResult:
    p = case.mesh_params()
    cb = CM.build(p)
    p_tied = case.elastic_case().params()
    cb_tied = CM.build(p_tied)
    if elastic.phi.shape != plastic.phi.shape or not np.allclose(elastic.phi, plastic.phi, atol=1e-12):
        raise ValueError("the elastic and plastic results are at different front stations")
    checks = (
        tuple(dataclasses.replace(c, name=f"tied_{c.name}") for c in mesh_checks(cb_tied, p_tied))
        + mesh_checks(cb, p)
        + qa_checks(case, elastic, plastic)
    )
    failing = next((c.name for c in checks if not c.passed), None)
    if plastic.record.status != "complete":
        status, failing = "fe-failed", failing or f"run status {plastic.record.status}"
    elif failing is not None:
        status = "fe-failed"
    else:
        status = "fe-verified" if FAMILY_VERIFIED else "unverified"
    phi = plastic.phi
    law = case.law
    provenance = provenance_of(case, runner)
    return FEResult(
        case=case, status=plastic.record.status, verification_status=status, failing_check=failing,
        topology_id=case.mesh_params().topology_id, topology_hash=cb.topology_hash,
        lam=plastic.lam, lr=plastic.lr, phi=phi,
        phi_physical=np.arctan(case.a / case.c * np.tan(phi)),
        j=plastic.j, j_spread=plastic.j_spread, j_e=plastic.j_e, kr=plastic.kr, k_elastic=elastic.k_fe,
        outside_small_strain=plastic.outside_small_strain, max_far_eff_strain=plastic.max_far_eff_strain,
        qa=checks, lr_end=case.plastic_case().lr_end(),
        proof_stress_0p2=float(law.proof_stress_0p2()), considere_stress=float(law.considere_stress()),
        provenance=provenance,
    )  # fmt: skip


def result_summary(res: FEResult) -> dict[str, Any]:
    """JSON-serialisable record of a result (arrays as lists), for the per-case record file."""
    return {
        "case": json.loads(res.case.canonical_json()),
        "case_id": res.case.case_id,
        "status": res.status,
        "verification_status": res.verification_status,
        "failing_check": res.failing_check,
        "topology_id": res.topology_id,
        "topology_hash": res.topology_hash,
        "lam": res.lam.tolist(),
        "lr": res.lr.tolist(),
        "phi_deg": np.degrees(res.phi).tolist(),
        "phi_physical_deg": np.degrees(res.phi_physical).tolist(),
        "j": res.j.tolist(),
        "j_spread": res.j_spread.tolist(),
        "kr": res.kr.tolist(),
        "k_elastic_per_unit": res.k_elastic.tolist(),
        "outside_small_strain": list(res.outside_small_strain),
        "max_far_eff_strain": res.max_far_eff_strain.tolist(),
        "qa": [dataclasses.asdict(c) for c in res.qa],
        "lr_end": res.lr_end,
        "proof_stress_0p2": res.proof_stress_0p2,
        "considere_stress": res.considere_stress,
        "j_unit": "J/m^2",
        "k_unit": "Pa sqrt(m)",
        "provenance": res.provenance,
    }


def write_summary(res: FEResult, path: Path) -> None:
    path.write_text(json.dumps(result_summary(res), indent=1, sort_keys=True), encoding="utf-8")


__all__ = [
    "FAMILY_VERIFIED",
    "LEVEL_1",
    "LEVELS",
    "PARAMETER_BOX",
    "FECase",
    "FEResult",
    "QACheck",
    "WarpJSolver",
    "assemble",
    "qa_checks",
    "result_summary",
    "write_summary",
]
