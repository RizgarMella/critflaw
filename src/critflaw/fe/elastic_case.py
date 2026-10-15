"""Elastic surface-flaw case: mesh (``crack_mesh``), constraints, consistent remote loads (membrane
plus bending traction on the far face), interaction-integral domains at every corner front station with the
plane-strain auxiliary field, the deck, and the comparison of the signed ``K_I`` with the library's plate solution
(Newman–Raju as cited there) at the same parametric angles (D3, theory manual section 6.2).

Units: the library is SI; the deck is written in N, mm, MPa (``critflaw.fe.units``)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from critflaw.fe import crack_mesh as CM
from critflaw.fe import deck as D
from critflaw.fe import mesh as M
from critflaw.fe import units as U
from critflaw.fe.parser import (
    EquilibriumQA,
    ParsedOutput,
    ParseError,
    equilibrium_qa,
    gauss_rows,
    parse_gauss_point_tables,
    parse_output,
    run_qa,
)
from critflaw.fe.runner import PARSER_VERSION, RunRecord, SolveCache, WarpRunner
from critflaw.solutions import plate

FloatArray = NDArray[np.float64]
# WARP3D's 'norm res tol' is a percentage of the total load norm (the trace prints 'ratio*100'); 1e-6 % = 1e-8
# relative, the direct solve's roundoff floor for a self-equilibrating (bending) load vector on 1e5 equations
ELASTIC_RES_TOL = 1e-6


class SolveFailed(RuntimeError):
    """A solve that did not complete (or failed its run QA); carries the run record."""

    def __init__(self, message: str, record: RunRecord) -> None:
        super().__init__(message)
        self.record = record


@dataclass(frozen=True)
class ElasticCase:
    """SI inputs: ``a, c, B, W, L_far`` in m, ``E`` and stresses in Pa."""

    a: float
    c: float
    B: float
    W: float
    L_far: float
    E: float
    nu: float
    pm: float
    pb: float
    n_rings: int = 6  # defaults: the production level_1 = (6, 12, 12, 3, 3)
    n_sectors: int = 12
    n_front: int = 12
    n_front_s: int = 3
    phi_s_deg: float = 10.0
    n_blend: int = 3
    n_far: int = 6
    n_outer: int = 10
    far_first: float = 1.0
    remote_check_fraction: float = 0.5  # D8: the far-field layer nearest z = fraction x L_far carries the profile check

    def params(self) -> CM.CrackBlockParams:
        return CM.CrackBlockParams(
            a=U.length_to_deck(self.a), c=U.length_to_deck(self.c), B=U.length_to_deck(self.B),
            W=U.length_to_deck(self.W), L_far=U.length_to_deck(self.L_far),
            n_rings=self.n_rings, n_sectors=self.n_sectors, n_front=self.n_front, n_front_s=self.n_front_s,
            phi_s=float(np.deg2rad(self.phi_s_deg)), n_blend=self.n_blend, n_far=self.n_far, n_outer=self.n_outer,
            far_first=self.far_first, tip="tied",
        )  # fmt: skip


def write_inputs(case: ElasticCase, work: Path) -> tuple[CM.CrackBlockMesh, dict[str, Path], tuple[D.DomainSpec, ...]]:
    """Mesh files, constraints, the unit remote-load pattern (traction Pm + Pb (1 - 2y/B) on z = L_far in MPa per
    unit factor), and the control deck; returns the mesh record, the input map and the domain specs."""
    work.mkdir(parents=True, exist_ok=True)
    p = case.params()
    cb = CM.build(p)
    m = cb.mesh
    (work / "coords.inp").write_text(M.write_coordinates(m), encoding="utf-8")
    (work / "incid.inp").write_text(M.write_incidences(m), encoding="utf-8")
    # constraints: symmetry u = 0 on x = 0, ligament w = 0 on z = 0 outside the ellipse, one node v = 0 against the
    # y translation (the remote load has no y resultant)
    origin_node = int(np.argmin(np.linalg.norm(m.coords - np.array([0.0, p.B, p.L_far]), axis=1))) + 1
    fixed = {"u": cb.symmetry_x_nodes, "w": cb.ligament_nodes, "v": np.array([origin_node])}
    (work / "constraints.inp").write_text(M.write_constraints(fixed), encoding="utf-8")
    constrained = np.unique(np.concatenate([cb.symmetry_x_nodes, cb.ligament_nodes, np.array([origin_node])]))
    pm_mpa, pb_mpa = U.stress_to_deck(case.pm), U.stress_to_deck(case.pb)
    b_mm = p.B
    loads = M.face_traction_nodal_loads(
        m, cb.loaded_faces, lambda x: np.array([0.0, 0.0, pm_mpa + pb_mpa * (1.0 - 2.0 * x[1] / b_mm)])
    )
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
                normal=(0.0, 0.0, 1.0),
                # the front tangent in the listed (decreasing phi) sense: -y at the surface point, +x at the deepest
                tangent=(0.0, -1.0, 0.0) if station == 0 else ((1.0, 0.0, 0.0) if end else None),
                # the lists run with decreasing phi: the deepest station is first in its own list (type a:
                # q = 1 at the first node), the surface vertex is last in its list (type c: q = 1 at the last
                # node), interior stations are centred (type b) - manual 4.4.8
                function_type="c" if station == 0 else ("a" if end else "b"),
                kind="II",
                plane_stress=False,
                comment=f"phi = {np.degrees(cb.phi_corner_stations[station]):.3f} deg",
            )  # fmt: skip
        )
    material = D.ElasticMaterial(
        U.stress_to_deck(case.E), case.nu, max_expected_stress=max(abs(pm_mpa) + abs(pb_mpa), 1.0) * 20.0
    )
    solver = D.SolverControls(
        solution_technique="sparse direct", norm_res_tol=ELASTIC_RES_TOL, maximum_iterations=5, minimum_iterations=1
    )
    remote, _z, _y = remote_layer(cb, case.remote_check_fraction * p.L_far)
    spec = D.DeckSpec(
        "plate_surface_flaw", m.n_nodes, m.n_elements, material,
        ("coords.inp", "incid.inp", "constraints.inp", "unit_load.inp"), "load", "unit_load", (1.0,), tuple(domains),
        solver=solver,
        element_output="gausspts short",
        stress_elements=" ".join(str(e) for e in remote),
        reaction_nodes=" ".join(str(n) for n in constrained),
    )  # fmt: skip
    (work / "control.inp").write_text(D.write_deck(spec), encoding="utf-8")
    inputs = {k: work / k for k in ("coords.inp", "incid.inp", "constraints.inp", "unit_load.inp", "control.inp")}
    return cb, inputs, tuple(domains)


@dataclass(frozen=True)
class ElasticResult:
    record: RunRecord
    phi: FloatArray  # parametric angles of the stations
    k_fe: FloatArray  # signed K_I (Pa sqrt(m)) per station, mean over rings 2..n
    k_spread: FloatArray  # max relative deviation over rings 2..n
    k_ref: FloatArray  # the library's plate solution at the same angles
    j_mixed: FloatArray  # J from K (plane strain), ring average, J/m^2
    remote_profile_dev: float = float("nan")  # D8: worst |sigma_zz - profile| / (|Pm| + |Pb|) on the checked layer
    remote_profile_z_over_B: float = float("nan")  # the checked layer's distance from the crack plane
    equilibrium: tuple[EquilibriumQA, ...] = ()  # D8: the reactions balance the applied resultants


def remote_layer(cb: CM.CrackBlockMesh, z_target: float) -> tuple[tuple[int, ...], float, FloatArray]:
    """The far-field element layer (centroids above 0.3 L_far share one z per layer) nearest ``z_target``: the
    element numbers, the layer's centroid z and the centroid y of each element (D8, remote profile)."""
    m = cb.mesh
    centroids = m.coords[m.incid[:, :8] - 1].mean(axis=1)
    z_far = centroids[:, 2] > 0.3 * cb.params.L_far
    levels = np.unique(np.round(centroids[z_far, 2], 6))
    z_layer = float(levels[np.argmin(np.abs(levels - z_target))])
    sel = np.flatnonzero(z_far & (np.abs(centroids[:, 2] - z_layer) < 1e-6 * cb.params.L_far))
    return tuple(int(e) + 1 for e in sel), z_layer, centroids[sel, 1]


def remote_profile_check(
    case: ElasticCase, cb: CM.CrackBlockMesh, text: str
) -> tuple[float, float, FloatArray, FloatArray]:
    """D8: the element-mean sigma_zz (over the 2x2x2 Gauss points; exact for the linear profile on the affine
    far-field bricks) of the checked layer against ``Pm + Pb (1 - 2 y_c / B)`` per unit factor. Returns the worst
    relative deviation on the scale max(|Pm| + |Pb|), the layer distance z / B, and the (fe, expected) arrays
    (MPa)."""
    p = cb.params
    elements, z_layer, y_c = remote_layer(cb, case.remote_check_fraction * p.L_far)
    tables = [t for t in parse_gauss_point_tables(text) if t.quantity == "stresses" and t.step == 1]
    rows = gauss_rows(tables, elements, 2, "D8 remote layer")  # sigma_zz is the third printed quantity
    fe = np.array([np.mean([float(rows[(e, g)][2]) for g in range(1, 9)]) for e in elements])
    pm, pb = U.stress_to_deck(case.pm), U.stress_to_deck(case.pb)
    expected = pm + pb * (1.0 - 2.0 * y_c / p.B)
    scale = max(abs(pm) + abs(pb), 1e-12)
    return float(np.max(np.abs(fe - expected)) / scale), z_layer / p.B, fe, expected


def run_case(case: ElasticCase, work: Path, runner: WarpRunner, cache: SolveCache | None = None) -> ElasticResult:
    """Solve (or read back from ``cache`` by the full solve key) the elastic deck; the run QA is applied on
    cache hits as on fresh solves. A solve that is not ``complete`` raises ``SolveFailed`` with its record."""
    cb, inputs, domains = write_inputs(case, work)
    key = runner.key(inputs, "control.inp") if cache is not None else None
    if (
        cache is not None
        and key is not None
        and cache.has(key)
        and (cache.record(key).status == "complete" or not cache.retry_incomplete)
    ):
        record, text = cache.record(key), cache.raw_output(key)
    else:
        record, out = runner.run(work / "job", inputs, "control.inp")
        if cache is not None and key is not None:
            cache.store(key, inputs, out, record, {"deck": "control.inp", "parser_version": PARSER_VERSION})
        text = out.read_text(encoding="utf-8", errors="replace")
    if record.status != "complete":
        raise SolveFailed(f"elastic case did not complete: {record.status}", record)
    parsed = parse_output(text)
    qa = run_qa(parsed, "unit_load", (1.0,), collapsed_elements=frozenset(int(e) for e in cb.tip_elements))
    if not qa.passed:
        raise SolveFailed(f"run QA failed: {qa}", record)
    try:
        return evaluate(case, cb, parsed, domains, record, text)
    except (ParseError, ValueError, IndexError, KeyError) as exc:  # an unevaluable analysis product is a failed solve
        failed = replace(record, status="failed", diagnostic=str(exc))
        raise SolveFailed(f"evaluation failed: {exc}", failed) from exc


def evaluate(
    case: ElasticCase,
    cb: CM.CrackBlockMesh,
    parsed: ParsedOutput,
    domains: tuple[D.DomainSpec, ...],
    record: RunRecord,
    text: str | None = None,
) -> ElasticResult:
    phi = cb.phi_corner_stations
    if len(parsed.mixed_mode_j) != len(domains):  # one mixed-mode J block per interaction domain, in order
        raise ParseError(f"{len(parsed.mixed_mode_j)} mixed-mode J blocks for {len(domains)} interaction domains")
    k_fe = np.zeros(len(domains))
    spread = np.zeros(len(domains))
    j_mixed = np.zeros(len(domains))
    for i, d in enumerate(domains):
        rec = parsed.domain(d.domain_id, 1, "II", mode="KI")
        if rec.k is None or rec.aux_field != "plane strain":
            raise ParseError(f"domain {d.domain_id}: no plane-strain K_I block")
        sel = rec.rings >= 2
        vals = rec.k[sel]
        k_fe[i] = U.k_from_deck(float(np.mean(vals)))
        spread[i] = float(np.max(np.abs(vals - np.mean(vals))) / abs(np.mean(vals)))
        aux, j_rings = parsed.mixed_mode_j[i]  # one mixed-mode J block per interaction domain, in order
        if aux != "plane strain":
            raise ParseError(f"domain {d.domain_id}: mixed-mode J block is not the plane-strain one")
        j_mixed[i] = U.j_from_deck(float(np.mean(j_rings[1:])))
    mm, mb = plate.surface_flaw_mm_mb(case.a, case.c, case.B, phi)
    fw = plate.surface_flaw_fw(case.a, case.c, case.B, case.W)
    k_ref = np.asarray((mm * case.pm + mb * case.pb) * fw * np.sqrt(np.pi * case.a), dtype=float)
    remote_dev, remote_z = (float("nan"), float("nan"))
    equilibrium: tuple[EquilibriumQA, ...] = ()
    if text is not None:
        remote_dev, remote_z, _fe, _exp = remote_profile_check(case, cb, text)
    if parsed.reactions:
        p = cb.params
        origin = np.array([0.0, 0.5 * p.B, p.L_far])
        origin_node = int(np.argmin(np.linalg.norm(cb.mesh.coords - np.array([0.0, p.B, p.L_far]), axis=1))) + 1
        constrained = np.unique(np.concatenate([cb.symmetry_x_nodes, cb.ligament_nodes, np.array([origin_node])]))
        pm_mpa, pb_mpa, b_mm = U.stress_to_deck(case.pm), U.stress_to_deck(case.pb), p.B
        loads = M.face_traction_nodal_loads(
            cb.mesh, cb.loaded_faces, lambda x: np.array([0.0, 0.0, pm_mpa + pb_mpa * (1.0 - 2.0 * x[1] / b_mm)])
        )
        f_app, m_app = M.resultants(cb.mesh, loads, origin)
        equilibrium = equilibrium_qa(parsed, cb.mesh.coords, constrained, f_app, m_app, (1.0,), origin)
    return ElasticResult(record, phi, k_fe, spread, k_ref, j_mixed, remote_dev, remote_z, equilibrium)
