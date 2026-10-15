"""Phase 4: the ``FECase`` record (parameter box, immutability, canonical id, derived cases) and the ``FEResult``
assembly with every QA check carrying its value and band (pure: assembled from the stored level-1 outputs)."""

from __future__ import annotations

import dataclasses
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from critflaw.fe import crack_mesh as CM
from critflaw.fe import fecase as F
from critflaw.fe.elastic_case import ElasticResult
from critflaw.fe.parser import parse_output
from critflaw.fe.plastic_case import LOCAL_STRAIN_CAP, PlasticCase, PlasticResult, evaluate, far_elements, write_inputs
from critflaw.fe.runner import RunRecord

DATA = Path(__file__).parent / "data" / "fe"
BASE = dict(a=8e-3, c=20e-3, B=20e-3, W=200e-3, L_far=400e-3, E=207e9, nu=0.3, sigma_0=355e6, n=10.0, beta=0.0)


def test_parameter_box_is_enforced_at_construction() -> None:
    case = F.FECase(**BASE)
    for name, value in (
        ("n", 2.9), ("n", 20.1), ("nu", 0.2), ("beta", 1.1), ("sigma_0", 150e6), ("sigma_0", 1.1e9),
        ("a", 3e-3), ("a", 13e-3), ("c", 7e-3), ("W", 150e-3),
    ):  # fmt: skip
        with pytest.raises(ValueError):
            case.with_(**{name: value})
    with pytest.raises(ValueError):
        case.with_(level=(8, 12, 16, 4, 4))  # not a declared level
    with pytest.raises(ValueError, match="remote length"):
        case.with_(L_far=200e-3)  # 10 B: below the converged remote length max(20 B, 2 W)
    case.with_(L_far=450e-3)  # longer is allowed
    with pytest.raises(NotImplementedError, match="TODO\\(verify\\)"):
        case.with_(family="cylinder_surface_flaw")
    with pytest.raises(dataclasses.FrozenInstanceError):
        case.a = 1.0  # type: ignore[misc]


def test_canonical_id_is_deterministic_and_parameter_sensitive() -> None:
    case = F.FECase(**BASE)
    assert case.case_id == F.FECase(**BASE).case_id
    assert case.case_id != case.with_(n=10.5).case_id
    assert case.case_id != case.with_(level=F.LEVELS[1]).case_id
    d = json.loads(case.canonical_json())
    assert d["topology_id"].startswith("plate_surface_flaw_v1:6-12-12-3-3") and d["tube_rule"] == {
        "k1": 0.5, "k2": 0.5, "k3": 0.1
    }  # fmt: skip
    assert case.phi_s_physical_deg == pytest.approx(np.degrees(np.arctan(0.4 * np.tan(np.radians(10.0)))))


def test_derived_cases_share_geometry_level_and_unit_load() -> None:
    case = F.FECase(**{**BASE, "beta": 0.25})
    pc, ec = case.plastic_case(), case.elastic_case()
    assert pc.level == case.level and (ec.n_rings, ec.n_sectors, ec.n_front, ec.n_front_s, ec.n_blend) == case.level
    assert (ec.pm, ec.pb) == pc.unit_stresses() == (0.75e6, 0.25e6)
    assert pc.far_strain_output and pc.norm_res_tol == case.norm_res_tol
    assert ec.params().topology_id.replace("tied", "untied") == pc.params().topology_id


def test_far_elements_exclude_the_tube_and_its_neighbourhood() -> None:
    pc = PlasticCase(**BASE, level=F.LEVEL_1)
    p = pc.params()
    cb = CM.build(p)
    far = np.asarray(far_elements(cb, p))
    tube = set(int(e) for e in cb.block_elements[: p.n_rings * p.n_sectors * p.n_front])
    assert not tube & set(far.tolist()) and 0 < far.size < cb.mesh.n_elements
    m = cb.mesh
    centroids = m.coords[m.incid[far - 1, :8] - 1].mean(axis=1)
    phi = np.linspace(0.0, np.pi / 2, 2001)
    front = np.stack([p.c * np.cos(phi), p.a * np.sin(phi), np.zeros_like(phi)], axis=1)
    d = np.sqrt(((centroids[:, None, :] - front[None]) ** 2).sum(2)).min(1)
    assert np.all(d >= 5.0 * p.r_t * (1 - 1e-6))
    assert (
        far.size + len(np.flatnonzero(_dist_all(cb, p) < 5.0 * p.r_t)) == cb.mesh.n_elements
    )  # the whole plate beyond 5 r_t


def _dist_all(cb: CM.CrackBlockMesh, p: CM.CrackBlockParams) -> np.ndarray:
    m = cb.mesh
    centroids = m.coords[m.incid[:, :8] - 1].mean(axis=1)
    phi = np.linspace(0.0, np.pi / 2, 721)
    front = np.stack([p.c * np.cos(phi), p.a * np.sin(phi), np.zeros_like(phi)], axis=1)
    return np.sqrt(((centroids[:, None, :] - front[None]) ** 2).sum(2)).min(1)


def _stored_level1(work: Path) -> tuple[PlasticCase, PlasticResult]:
    pc = PlasticCase(  # the stored run: first level and far field, printed J (no packets file)
        **{**BASE, "L_far": 100e-3}, level=(8, 12, 16, 4, 4), n_outer=5, far_first=0.0, packets=False
    )
    cb, _inputs, domains = write_inputs(pc, work)
    text = gzip.open(DATA / "plastic_L1_out.txt.gz", "rt", encoding="utf-8", errors="replace").read()
    parsed_e = parse_output(gzip.open(DATA / "plastic_L1_elastic_out.txt.gz", "rt", encoding="utf-8").read())
    res = evaluate(
        pc, cb, parse_output(text), text, domains, RunRecord("complete", 0, 61.0, 12, 12, True, (), (), ()), parsed_e
    )
    return pc, res


def test_result_assembly_labels_and_qa_from_the_stored_run(tmp_path: Path) -> None:
    pc, plastic = _stored_level1(tmp_path / "w")
    case = F.FECase(**BASE)
    n = plastic.phi.size
    k = np.full(n, 5.0e5)
    elastic = ElasticResult(plastic.record, plastic.phi, k, np.zeros(n), k, np.zeros(n), 0.001, 2.5)
    res = F.assemble(case, elastic, plastic, None)  # type: ignore[arg-type]
    names = [c.name for c in res.qa]
    n_mesh = (
        10  # per mesh: jacobian, two scaled-Jacobian bands, tip count, coincident, pattern, repeated, quarter, 2 aspect
    )
    tied, untied = res.qa[:n_mesh], res.qa[n_mesh : 2 * n_mesh]
    assert [c.name for c in tied] == ["tied_" + c.name for c in untied]
    assert all(c.passed for c in tied) and all(c.passed for c in untied), [
        c for c in res.qa[: 2 * n_mesh] if not c.passed
    ]
    by_m = {c.name: c for c in res.qa[: 2 * n_mesh]}
    assert by_m["tip_element_count"].value == 12 * 12 and by_m["coincident_nodes_on_front_only"].value == 0
    assert by_m["tied_tip_repeated_incidences"].value == 144 and by_m["tip_repeated_incidences"].value == 0
    assert by_m["tied_quarter_point_midside"].value <= 1e-12 and by_m["tip_node_pattern"].passed
    names = names[2 * n_mesh :]
    assert names[:6] == [
        "elastic_K_I_positive",
        "remote_profile",
        "equilibrium",
        "J_path_independence",
        "plastic_zone_in_ring_1",
        "J_1_vs_elastic_same_mesh",
    ]
    by = {c.name: c for c in res.qa}
    assert by["J_path_independence"].band == 0.02 and by["J_1_vs_elastic_same_mesh"].band == 0.005
    # the stored run has no far-region strain output (it predates the cap check): evaluated = False -> fe-failed,
    # with the failing check named; every other check holds except the 2 % band at the last step
    assert res.verification_status == "fe-failed" and res.failing_check in (
        "equilibrium",  # the stored run predates the reaction output
        "J_path_independence",
        "far_region_strain_evaluated",
    )
    assert (
        by["elastic_K_I_positive"].passed
        and by["plastic_zone_in_ring_1"].passed
        and by["J_1_vs_elastic_same_mesh"].passed
    )
    assert res.topology_hash and res.topology_id.startswith("plate_surface_flaw_v1:6-12-12-3-3")
    assert res.lr_end == pytest.approx(pc.lr_end()) and 1.0 < res.proof_stress_0p2 / case.sigma_0 < 1.6
    assert np.allclose(res.phi_physical, np.arctan(0.4 * np.tan(res.phi)))
    # a negative K_I anywhere is a failed check, named first
    bad = F.assemble(
        case, ElasticResult(plastic.record, plastic.phi, -k, np.zeros(n), k, np.zeros(n), 0.001, 2.5), plastic, None
    )  # type: ignore[arg-type]
    assert bad.failing_check == "elastic_K_I_positive"
    # the summary round-trips through JSON
    F.write_summary(res, tmp_path / "r.json")
    d = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert d["case_id"] == case.case_id and d["qa"][23]["band"] == 0.02 and len(d["kr"]) == 12
    assert not F.FAMILY_VERIFIED  # D4, D5 pending: no case may be fe-verified yet


def test_small_strain_cap_marks_this_and_every_later_step(tmp_path: Path) -> None:
    _pc, plastic = _stored_level1(tmp_path)

    far = np.full(12, np.nan)
    far[6:] = [0.01, 0.02, LOCAL_STRAIN_CAP * 1.01, 0.03, 0.08, 0.1]
    exceeded, outside = False, []
    for k in range(12):
        exceeded = exceeded or (not np.isnan(far[k]) and far[k] > LOCAL_STRAIN_CAP)
        outside.append(exceeded)
    assessed = tuple(range(6, 12))
    marked = dataclasses.replace(
        plastic, max_far_eff_strain=far, outside_small_strain=tuple(outside), assessed_steps=assessed
    )
    assert marked.outside_small_strain == (False,) * 8 + (True,) * 4
    case = F.FECase(**BASE)
    n = plastic.phi.size
    k = np.full(n, 5.0e5)
    res = F.assemble(
        case, ElasticResult(plastic.record, plastic.phi, k, np.zeros(n), k, np.zeros(n), 0.001, 2.5), marked, None
    )  # type: ignore[arg-type]
    assert res.usable_steps.tolist() == [True] * 8 + [False] * 4
    by = {c.name: c for c in res.qa}
    assert by["far_region_strain_evaluated"].passed and by["small_strain_cap_first_assessed_step"].passed
    assert by["small_strain_cap_first_assessed_step"].value == 0.01  # the first assessed step's strain, not step 1
    # the cap exceeded already at the first assessed step: the check fails (it is the first assessed step that
    # decides, never step 1, which is always inside the cap)
    far_bad = far.copy()
    far_bad[6:] = [0.06, 0.07, 0.08, 0.09, 0.1, 0.12]
    bad = dataclasses.replace(marked, max_far_eff_strain=far_bad, outside_small_strain=(False,) * 6 + (True,) * 6)
    res_bad = F.assemble(
        case, ElasticResult(plastic.record, plastic.phi, k, np.zeros(n), k, np.zeros(n), 0.001, 2.5), bad, None
    )  # type: ignore[arg-type]
    chk = {c.name: c for c in res_bad.qa}["small_strain_cap_first_assessed_step"]
    assert not chk.passed and chk.value == 0.06 and not res_bad.usable_steps[6:].any()


class _KeyOnlyRunner:
    """A runner stand-in for cache-hit paths: it keys but never solves (the solve would be a failure here)."""

    threads = 4

    def key(self, inputs: object, deck: str) -> str:
        return f"k-{deck}"

    def run(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("no solve expected: the stored output must be read back")


def test_public_solver_returns_fe_failed_when_the_elastic_product_is_incomplete(tmp_path: Path) -> None:
    """R2-02 / G2-C-R2-01: an elastic output whose mixed-mode J blocks are missing is a failed solve through the
    public path (run_case -> SolveFailed -> failed_result), never an IndexError."""
    import re

    from critflaw.fe import runner as R
    from critflaw.fe.elastic_case import ElasticCase, SolveFailed
    from critflaw.fe.elastic_case import run_case as run_elastic

    b = 20e-3
    a = 0.4 * b
    c = a / 0.4
    w = c / 0.05
    case = ElasticCase(a=a, c=c, B=b, W=w, L_far=CM.remote_length(b, w), E=207e9, nu=0.3, pm=100e6, pb=0.0)
    text = gzip.open(DATA / "elastic_aB040_ac040_out.txt.gz", "rt", encoding="utf-8").read()
    broken = re.sub(r"domain\s+J plane strain", "domain   J-removed", text)  # the mixed-mode blocks unreadable
    cache = R.SolveCache(tmp_path / "cache")
    job = tmp_path / "job"
    job.mkdir()
    (job / "out.txt").write_text(broken, encoding="utf-8")
    cache.store("k-control.inp", {"x": job / "out.txt"}, job / "out.txt", R.RunRecord("complete", 0, 5.0, 1, 1, True, (), (), ()),
                {"parser_version": R.PARSER_VERSION})  # fmt: skip
    with pytest.raises(SolveFailed) as exc:
        run_elastic(case, tmp_path / "w", _KeyOnlyRunner(), cache)  # type: ignore[arg-type]
    assert exc.value.record.status == "failed" and "mixed-mode" in exc.value.record.diagnostic
    res = F.failed_result(F.FECase(**BASE), exc.value.record, str(exc.value), None)
    assert res.verification_status == "fe-failed" and res.status == "failed" and res.failing_check.startswith("solve:")
    assert res.usable_steps.size == 0
    # the intact output evaluates (the same path, no failure)
    (job / "out.txt").write_text(text, encoding="utf-8")
    cache2 = R.SolveCache(tmp_path / "cache2")
    cache2.store("k-control.inp", {"x": job / "out.txt"}, job / "out.txt", R.RunRecord("complete", 0, 5.0, 1, 1, True, (), (), ()),
                 {"parser_version": R.PARSER_VERSION})  # fmt: skip
    ok = run_elastic(case, tmp_path / "w2", _KeyOnlyRunner(), cache2)  # type: ignore[arg-type]
    assert ok.record.status == "complete" and np.all(ok.k_fe[1:] > 0)
