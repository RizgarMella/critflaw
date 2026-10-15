"""Elastic-plastic surface-flaw case (theory manual, sections 6.1.4, 6.1.5 and 6.2): the untied
mesh, the WARP3D ``deformation`` material (``WarpDeformationLaw``), proportional loading under a declared
cumulative schedule capped by the small-strain rule, J domains at every corner front station after every step,
Gauss-point stresses of the tube rings after step 1 (the J_e provenance condition (a)), and the extraction of
Kr(Lr) with J_e = J_1 (lambda/lambda_1)^2 and Lr from the declared limit load (the library's plate surface-flaw
reference stress).

Units: SI in; the deck in N, mm, MPa."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from critflaw.fe import crack_mesh as CM
from critflaw.fe import deck as D
from critflaw.fe import mesh as M
from critflaw.fe import units as U
from critflaw.fe.elastic_case import ELASTIC_RES_TOL, SolveFailed
from critflaw.fe.parser import (
    PRINT_HALF_ULP,
    EquilibriumQA,
    JPacket,
    ParsedOutput,
    ParseError,
    equilibrium_qa,
    gauss_rows,
    parse_gauss_point_tables,
    parse_output,
    read_j_packets,
    run_qa,
)
from critflaw.fe.runner import PARSER_VERSION, RunRecord, SolveCache, WarpRunner
from critflaw.fe.warp_law import WarpDeformationLaw
from critflaw.solutions import plate

FloatArray = NDArray[np.float64]

STRAIN_CAP_REFERENCE = 0.02  # eps_0 Lr^n at the schedule end (theory manual, section 6.1.5)
LR_CEILING = 1.5
STRAIN_CHECK_LR = 0.8  # the small-strain cap is evaluated at the steps with Lr >= 0.8 (strains are monotone in load)
EXCLUSION_RADIUS_FACTOR = 5.0  # elements with centroid at distance >= 5 r_t from the front are the far region
LOCAL_STRAIN_CAP = 0.05  # effective-strain cap of the far region (5 % local maximum)
PACKETS_FILE = "packets.bpf"
GAUSS_POINTS = 8  # 2x2x2 integration: rows per element in a Gauss-point table
PRINT_VS_PACKET_BAND = 1e-3  # the printed J (4 significant digits) must agree with the packet value to rounding


def constrained_nodes(cb: CM.CrackBlockMesh, anchor: int) -> NDArray[np.int64]:
    """Every node with a constraint (symmetry plane, ligament, anchor): the reaction output list."""
    return np.unique(np.concatenate([cb.symmetry_x_nodes, cb.ligament_nodes, np.array([anchor])])).astype(np.int64)


def unit_loads(case: PlasticCase, cb: CM.CrackBlockMesh) -> dict[int, FloatArray]:
    pm, pb = case.unit_stresses()
    pm_mpa, pb_mpa, b_mm = U.stress_to_deck(pm), U.stress_to_deck(pb), cb.params.B
    return M.face_traction_nodal_loads(
        cb.mesh, cb.loaded_faces, lambda x: np.array([0.0, 0.0, pm_mpa + pb_mpa * (1.0 - 2.0 * x[1] / b_mm)])
    )


def anchor_node(cb: CM.CrackBlockMesh) -> int:
    p = cb.params
    return int(np.argmin(np.linalg.norm(cb.mesh.coords - np.array([0.0, p.B, p.L_far]), axis=1))) + 1


_gauss_rows = gauss_rows  # moved to the parser (shared with the D8 check)


def far_elements(cb: CM.CrackBlockMesh, p: CM.CrackBlockParams) -> tuple[int, ...]:
    """Element numbers whose centroid lies at a distance >= 5 r_t from the nearest crack-front point (the
    small-strain exclusion region, theory manual section 6.1.5), from a 721-point sampling of the elliptical front."""
    m = cb.mesh
    centroids = m.coords[m.incid[:, :8] - 1].mean(axis=1)
    phi = np.linspace(0.0, np.pi / 2, 721)
    front = np.stack([CM.front_point(p, float(f)) for f in phi])
    d = np.sqrt(((centroids[:, None, :] - front[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    return tuple(int(e) for e in np.flatnonzero(d >= EXCLUSION_RADIUS_FACTOR * p.r_t) + 1)


@dataclass(frozen=True)
class PlasticCase:
    a: float
    c: float
    B: float
    W: float
    L_far: float
    E: float
    nu: float
    sigma_0: float
    n: float
    beta: float  # Pb / (Pm + Pb), the loading mix
    level: tuple[int, int, int, int, int] = (6, 12, 12, 3, 3)  # (n_rings, n_sectors, n_front, n_front_s, n_blend)
    n_steps: int = 12
    lambda_1_lr: float = 0.05  # the first step's Lr (the plastic zone must stay inside ring 1: checked)
    phi_s_deg: float = 10.0  # the parametric evaluation angle (a node sits there on every level)
    norm_res_tol: float = 0.002  # Newton residual-norm tolerance; D9 tightens it 10x on the same mesh
    far_strain_output: bool = False  # Gauss-point strains of the elements beyond 5 r_t at the steps with Lr >= 0.8
    packets: bool = True  # full-precision J through the binary packets file (the printed tables have 4 digits)
    n_far: int = 6  # z layers above the block to L_far (D7b doubles it)
    n_outer: int = 10  # divisions from the outer offset curve to the plate edges (D7b doubles it)
    far_first: float = 1.0  # first far-field layer as a multiple of the blend thickness (0: legacy n_far layers)

    def params(self) -> CM.CrackBlockParams:
        nr, ns, nf, nfs, nb = self.level
        return CM.CrackBlockParams(
            a=U.length_to_deck(self.a), c=U.length_to_deck(self.c), B=U.length_to_deck(self.B),
            W=U.length_to_deck(self.W), L_far=U.length_to_deck(self.L_far),
            n_rings=nr, n_sectors=ns, n_front=nf, n_front_s=nfs, n_blend=nb, n_far=self.n_far, n_outer=self.n_outer,
            far_first=self.far_first, phi_s=float(np.deg2rad(self.phi_s_deg)), tip="untied",
        )  # fmt: skip

    @property
    def law(self) -> WarpDeformationLaw:
        return WarpDeformationLaw(self.E, self.nu, self.sigma_0, self.n)

    def unit_stresses(self) -> tuple[float, float]:
        """(Pm, Pb) in Pa of the unit load pattern: Pm + Pb = 1 MPa at the declared mix."""
        unit = 1.0e6
        return (1.0 - self.beta) * unit, self.beta * unit

    def lr_per_unit(self) -> float:
        """Lr per unit load factor from the declared limit load (the library's surface-flaw reference stress)."""
        pm, pb = self.unit_stresses()
        sref = float(plate.surface_flaw_reference_stress(pm, pb, self.a, self.c, self.B, self.W)[()])
        return sref / self.sigma_0

    def lr_end(self) -> float:
        """The schedule end: the smaller of Lr = 1.5 and the Lr at which eps_0 Lr^n reaches the strain cap."""
        eps0 = self.sigma_0 / self.E
        return float(min(LR_CEILING, (STRAIN_CAP_REFERENCE / eps0) ** (1.0 / self.n)))

    def strain_output_steps(self) -> tuple[int, ...]:
        """0-based steps whose far-region strains are output: the first assessed step (Lr >= 0.8), the last step
        with Lr <= 1.2 (the emulator's support end) and the last step. Under proportional loading of a
        deformation-plasticity solid the strains are monotone in the load factor, so the cap decision at the
        steps in between follows conservatively from these three (evaluate)."""
        lr = np.asarray(self.schedule()) * self.lr_per_unit()
        assessed = self.assessed_steps()
        if not assessed:
            return (lr.size - 1,)
        support = [k for k in assessed if lr[k] <= 1.2 + 1e-12]
        return tuple(sorted({assessed[0], *(support[-1:]), lr.size - 1}))

    def assessed_steps(self) -> tuple[int, ...]:
        """0-based indices of the steps at which the far-region strain cap is assessed (Lr >= 0.8 by the
        computed schedule; the deck outputs the strains there and the D7 driver assesses the same steps)."""
        lr = np.asarray(self.schedule()) * self.lr_per_unit()
        return tuple(int(k) for k in range(1, lr.size) if lr[k] >= STRAIN_CHECK_LR * (1.0 - 1e-12))

    def schedule(self) -> tuple[float, ...]:
        """Cumulative load factors: lambda_1 at Lr = lambda_1_lr, geometric to Lr = 0.8, then evenly spaced
        to the end; strictly increasing."""
        lr_unit = self.lr_per_unit()
        lr_end = self.lr_end()
        lr_1 = min(self.lambda_1_lr, 0.5 * lr_end)
        n_lo = max(2, self.n_steps // 2)
        n_hi = max(2, self.n_steps - n_lo)
        lo = np.geomspace(lr_1, min(0.8, 0.9 * lr_end), n_lo)
        hi = np.linspace(lo[-1], lr_end, n_hi + 1)[1:]
        lrs = np.concatenate([lo, hi])
        return tuple(float(v / lr_unit) for v in lrs)


def write_inputs(case: PlasticCase, work: Path) -> tuple[CM.CrackBlockMesh, dict[str, Path], tuple[D.DomainSpec, ...]]:
    work.mkdir(parents=True, exist_ok=True)
    p = case.params()
    cb = CM.build(p)
    m = cb.mesh
    (work / "coords.inp").write_text(M.write_coordinates(m), encoding="utf-8")
    (work / "incid.inp").write_text(M.write_incidences(m), encoding="utf-8")
    anchor = anchor_node(cb)
    fixed = {"u": cb.symmetry_x_nodes, "w": cb.ligament_nodes, "v": np.array([anchor])}
    (work / "constraints.inp").write_text(M.write_constraints(fixed), encoding="utf-8")
    constrained = constrained_nodes(cb, anchor)
    loads = unit_loads(case, cb)
    (work / "unit_load.inp").write_text(M.write_nodal_load_pattern("unit_load", loads), encoding="utf-8")
    domains = []
    for station in range(len(cb.phi_corner_stations)):
        front = cb.domain_front_nodes(station)
        end = station in (0, len(cb.phi_corner_stations) - 1)
        domains.append(
            D.DomainSpec(
                f"S{station}",
                front,
                (1, p.n_rings),
                True,
                tangent=(0.0, -1.0, 0.0) if station == 0 else ((1.0, 0.0, 0.0) if end else None),
                function_type="c" if station == 0 else ("a" if end else "b"),  # see elastic_case
                kind="J",
                packets=case.packets,
                comment=f"phi = {np.degrees(cb.phi_corner_stations[station]):.3f} deg",
            )  # fmt: skip
        )
    nr, ns, nf = p.n_rings, p.n_sectors, p.n_front
    ring_elements = {
        r: [int(cb.block_elements[(r * ns + k) * nf + j]) for k in range(ns) for j in range(nf)] for r in range(nr)
    }
    tube = [e for r in range(nr) for e in ring_elements[r]]
    far = far_elements(cb, p) if case.far_strain_output else ()
    checked = [k + 1 for k in case.strain_output_steps()]
    output_steps: tuple[int, ...] = (1, *checked) if far else (1,)  # stresses at step 1, strains at the checked steps
    material = D.DeformationMaterial(U.stress_to_deck(case.E), case.nu, U.stress_to_deck(case.sigma_0), case.n)
    solver = D.SolverControls(
        solution_technique="sparse direct",
        norm_res_tol=case.norm_res_tol,
        maximum_iterations=20,
        minimum_iterations=2,
        packets_file=PACKETS_FILE if case.packets else "",
    )
    spec = D.DeckSpec(
        "plate_surface_flaw", m.n_nodes, m.n_elements, material,
        ("coords.inp", "incid.inp", "constraints.inp", "unit_load.inp"), "load", "unit_load", case.schedule(),
        tuple(domains), solver=solver, element_output="gausspts short",
        stress_elements=" ".join(str(e) for e in tube), output_after_steps=output_steps,
        strain_elements=" ".join(str(e) for e in far), stress_after_steps=(1,), strain_after_steps=output_steps,
        reaction_nodes=" ".join(str(n) for n in constrained), reactions_every_step=True,
    )  # fmt: skip
    (work / "control.inp").write_text(D.write_deck(spec), encoding="utf-8")
    # the elastic companion on the same untied mesh at lambda_1 (condition (b) of the J_e provenance)
    elastic = D.DeckSpec(
        "plate_surface_flaw", m.n_nodes, m.n_elements,
        D.ElasticMaterial(U.stress_to_deck(case.E), case.nu, max_expected_stress=U.stress_to_deck(case.sigma_0) * 100),
        ("coords.inp", "incid.inp", "constraints.inp", "unit_load.inp"), "load", "unit_load", (case.schedule()[0],),
        tuple(domains),
        solver=D.SolverControls(
            solution_technique="sparse direct", norm_res_tol=ELASTIC_RES_TOL, maximum_iterations=5, minimum_iterations=1
        ),
        element_output="gausspts short",
        reaction_nodes=" ".join(str(n) for n in constrained), reactions_every_step=True,
    )  # fmt: skip
    (work / "elastic.inp").write_text(D.write_deck(elastic), encoding="utf-8")
    inputs = {
        k: work / k
        for k in ("coords.inp", "incid.inp", "constraints.inp", "unit_load.inp", "control.inp", "elastic.inp")
    }
    return cb, inputs, tuple(domains)


@dataclass(frozen=True)
class PlasticResult:
    record: RunRecord
    lam: FloatArray  # cumulative factors read back
    lr: FloatArray
    phi: FloatArray
    j: FloatArray  # (steps, stations) J (J/m^2), mean over rings 2..n
    j_spread: FloatArray  # (steps, stations)
    j_e: FloatArray  # (steps, stations) J_1 (lambda/lambda_1)^2
    kr: FloatArray  # sqrt(J_e / J)
    plastic_zone_in_ring_1: bool  # step-1 condition (a)
    max_vm_outside_ring_1_over_sigma_0: float
    j1_vs_elastic: FloatArray  # condition (b): J_1 / J_elastic(lambda_1) - 1 per station (same untied mesh)
    max_far_eff_strain: FloatArray = field(default_factory=lambda: np.zeros(0))  # per step (NaN: not output)
    outside_small_strain: tuple[bool, ...] = ()  # per step: the cap exceeded at this or an earlier step
    equilibrium: tuple[EquilibriumQA, ...] = ()  # D8 per step: reactions balance the applied resultants
    assessed_steps: tuple[int, ...] = ()  # 0-based steps at which the strain cap was assessed (Lr >= 0.8)
    iterations: tuple[int, ...] = ()  # Newton iterations per step (the convergence trace)
    equilibrium_elastic: tuple[EquilibriumQA, ...] = ()  # the same for the elastic companion's one step
    record_elastic: RunRecord | None = None  # the companion's run record


def run_case(case: PlasticCase, work: Path, runner: WarpRunner, cache: SolveCache | None = None) -> PlasticResult:
    """Solve (or read back from ``cache`` by the full solve key: generated inputs, deck, image digest,
    executable, threads, memory, timeout) the plastic deck and its elastic companion; the run QA is applied on
    cache hits as on fresh solves. A solve that is not ``complete`` raises ``SolveFailed`` with its record."""
    cb, inputs, domains = write_inputs(case, work)
    record, text, packets = _solve(runner, cache, work / "job", inputs, "control.inp")
    if record.status != "complete":
        raise SolveFailed(f"plastic case did not complete: {record.status}", record)
    parsed = parse_output(text)
    collapsed = frozenset(int(e) for e in cb.tip_elements)
    qa = run_qa(parsed, "unit_load", case.schedule(), collapsed_elements=collapsed)
    if not qa.passed:
        raise SolveFailed(f"run QA failed: {qa}", record)
    rec_e, text_e, _ = _solve(runner, cache, work / "job_elastic", inputs, "elastic.inp")
    if rec_e.status != "complete":
        raise SolveFailed(f"elastic companion did not complete: {rec_e.status}", rec_e)
    parsed_e = parse_output(text_e)
    qa_e = run_qa(parsed_e, "unit_load", (case.schedule()[0],), collapsed_elements=collapsed)
    if not qa_e.passed:
        raise SolveFailed(f"elastic companion run QA failed: {qa_e}", rec_e)
    try:
        return evaluate(case, cb, parsed, text, domains, record, parsed_e, packets=packets, record_elastic=rec_e)
    except (ParseError, ValueError, IndexError, KeyError) as exc:  # an unevaluable product is a failed solve
        raise SolveFailed(f"evaluation failed: {exc}", replace(record, status="failed", diagnostic=str(exc))) from exc


def _solve(
    runner: WarpRunner, cache: SolveCache | None, job: Path, inputs: dict[str, Path], deck: str
) -> tuple[RunRecord, str, bytes | None]:
    key = runner.key(inputs, deck) if cache is not None else None
    if cache is not None and key is not None and cache.has(key):
        stored = cache.record(key)
        if stored.status == "complete" or not cache.retry_incomplete:
            return stored, cache.raw_output(key), cache.packets(key)  # the stored attempt, complete or not
        # explicit retry policy of the cache: a non-complete attempt is superseded
    record, out = runner.run(job, inputs, deck)
    if cache is not None and key is not None:
        cache.store(key, inputs, out, record, {"deck": deck, "parser_version": PARSER_VERSION})
    return record, out.read_text(encoding="utf-8", errors="replace"), packets_of(out)


def packets_of(out: Path) -> bytes | None:
    """The binary packets file beside an output file, if the run produced one."""
    pk = out.with_name(PACKETS_FILE)
    return pk.read_bytes() if pk.exists() else None


def evaluate(
    case: PlasticCase,
    cb: CM.CrackBlockMesh,
    parsed: ParsedOutput,
    text: str,
    domains: tuple[D.DomainSpec, ...],
    record: RunRecord,
    parsed_elastic: ParsedOutput | None = None,
    packets: bytes | None = None,
    record_elastic: RunRecord | None = None,
) -> PlasticResult:
    """With ``packets`` (the binary packets file) J comes from the full-precision packet values, the printed
    four-digit tables being checked against them to rounding; without, from the printed tables."""
    sched = case.schedule()
    n_steps = len(sched)
    read_back = np.asarray([s.load_factors["unit_loa"] for s in parsed.steps[:n_steps]])
    lam = np.asarray(sched, dtype=float)  # the exact declared factors (12 digits); the read-back is checked
    if read_back.shape != lam.shape or np.any(np.abs(read_back - lam) > PRINT_HALF_ULP * np.abs(lam)):
        raise ParseError("the cumulative load factors read back do not round to the declared schedule")
    lr = lam * case.lr_per_unit()
    j = np.zeros((n_steps, len(domains)))
    spread = np.zeros_like(j)
    by_key: dict[tuple[str, int], JPacket] = {}
    if case.packets and packets is None:
        raise ParseError("the case requests full-precision J packets and the run produced no packets file")
    if packets is not None:
        for q in read_j_packets(packets):
            if (q.domain_id, q.step) in by_key:
                raise ParseError(f"packets: domain {q.domain_id} step {q.step} appears twice")
            by_key[(q.domain_id, q.step)] = q
    for k in range(1, n_steps + 1):
        for i, d in enumerate(domains):
            rec = parsed.domain(d.domain_id, k, "J")
            total = rec.total
            if packets is not None:
                if (d.domain_id, k) not in by_key:
                    raise ParseError(f"packets: no J packet for domain {d.domain_id} step {k}")
                q = by_key[(d.domain_id, k)]
                if not np.array_equal(q.rings, rec.rings):
                    raise ParseError(f"packets: ring set of domain {d.domain_id} step {k} differs from the print")
                if not np.all(np.isfinite(q.total)) or np.max(np.abs(rec.total / q.total - 1.0)) > PRINT_VS_PACKET_BAND:
                    raise ParseError(f"packets: printed J of domain {d.domain_id} step {k} disagrees beyond rounding")
                total = q.total
            vals = total[rec.rings >= 2]
            j[k - 1, i] = U.j_from_deck(float(np.mean(vals)))
            spread[k - 1, i] = float(np.max(np.abs(vals - np.mean(vals))) / abs(np.mean(vals)))
    j_e = j[0][None, :] * (lam / lam[0])[:, None] ** 2
    kr = np.sqrt(j_e / j)
    # condition (a): after step 1 every integration point above 0.95 sigma_0 lies in ring 1; the check demands
    # the full Gauss-point coverage of rings 2..n (a missing table never passes as an elastic region)
    p = case.params()
    nr, ns, nf = p.n_rings, p.n_sectors, p.n_front
    ring_of = {
        int(cb.block_elements[(r * ns + k) * nf + jj]): r for r in range(nr) for k in range(ns) for jj in range(nf)
    }
    gauss = parse_gauss_point_tables(text)
    tables = [t for t in gauss if t.quantity == "stresses" and t.step == 1]
    sigma_0_mpa = U.stress_to_deck(case.sigma_0)
    worst = 0.0
    outer = [e for e, r in ring_of.items() if r >= 1]
    for row in _gauss_rows(tables, outer, 7, "step-1 stress tables").values():
        worst = max(worst, float(row[7]) / sigma_0_mpa)  # mises is the 8th printed quantity
    equilibrium: tuple[EquilibriumQA, ...] = ()
    if parsed.reactions:
        origin = np.array([0.0, 0.5 * p.B, p.L_far])
        f_app, m_app = M.resultants(cb.mesh, unit_loads(case, cb), origin)
        equilibrium = equilibrium_qa(
            parsed, cb.mesh.coords, constrained_nodes(cb, anchor_node(cb)), f_app, m_app, tuple(lam), origin
        )
    j1_vs_e = np.full(len(domains), np.nan)
    equilibrium_e: tuple[EquilibriumQA, ...] = ()
    if parsed_elastic is not None:
        for i, d in enumerate(domains):
            rec = parsed_elastic.domain(d.domain_id, 1, "J")
            j_el = U.j_from_deck(float(np.mean(rec.total[rec.rings >= 2])))
            j1_vs_e[i] = j[0, i] / j_el - 1.0
        if parsed_elastic.reactions:  # the companion's own equilibrium against its one-step load
            origin = np.array([0.0, 0.5 * p.B, p.L_far])
            f_app, m_app = M.resultants(cb.mesh, unit_loads(case, cb), origin)
            lam_e = (float(sched[0]),)
            equilibrium_e = equilibrium_qa(
                parsed_elastic, cb.mesh.coords, constrained_nodes(cb, anchor_node(cb)), f_app, m_app, lam_e, origin
            )
    far_strain = np.full(n_steps, np.nan)
    if case.far_strain_output:
        far = far_elements(cb, p)
        checked = [k + 1 for k in case.strain_output_steps()]
        for step in checked:
            strain_tables = [t for t in gauss if t.quantity == "strains" and t.step == step]
            rows = _gauss_rows(strain_tables, far, 6, f"strain tables step {step}")
            far_strain[step - 1] = max(float(row[6]) for row in rows.values())  # eff_eps is the 7th quantity
    # the cap decision from the checked steps (monotone strains): every step up to the last checked step inside
    # the cap is inside; from the step after it onwards, once a later checked step exceeds the cap, every step is
    # marked outside (the steps in between are not observed and are taken as outside — conservative)
    outside = [False] * n_steps
    if case.far_strain_output:
        prev = 0  # 1-based: the last checked step found inside the cap (0 before any)
        for step in checked:
            if far_strain[step - 1] > LOCAL_STRAIN_CAP:
                outside = [k + 1 > prev for k in range(n_steps)]
                break
            prev = step
    return PlasticResult(
        record, lam, lr, cb.phi_corner_stations, j, spread, j_e, kr, worst < 0.95, worst, j1_vs_e,
        far_strain, tuple(outside), equilibrium, case.assessed_steps(),
        tuple(int(s.iterations) for s in parsed.steps[:n_steps]), equilibrium_e, record_elastic,
    )  # fmt: skip
