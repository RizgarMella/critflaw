"""Phase 4 gate (i), the live verification decks D6 (one-element constitutive response) and D8 (uncracked plate
loads and boundary conditions), run in the pinned image; skipped without it."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from critflaw.fe import WarpDeformationLaw
from critflaw.fe import deck as D
from critflaw.fe import mesh as M
from critflaw.fe import runner as R
from critflaw.fe import units as U
from critflaw.fe.parser import equilibrium_qa, parse_output, run_qa

IMAGE = "critflaw-warp3d:spike"
ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    shutil.which("docker") is None
    or subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode != 0,
    reason="WARP3D image absent",
)

E, NU, S0, N = 207e9, 0.3, 355e6, 10.0


def _write_mesh_files(work: Path, m: M.Hex20Mesh) -> None:
    work.mkdir(parents=True, exist_ok=True)
    (work / "coords.inp").write_text(M.write_coordinates(m), encoding="utf-8")
    (work / "incid.inp").write_text(M.write_incidences(m), encoding="utf-8")


def _strains_and_stresses(text: str) -> list[tuple[float, float, float]]:
    """(sigma_xx, eps_xx, eps_yy) at Gauss point 1 of element 1 for every step (deck units)."""
    eps = re.findall(r"eff_eps\s*\n\s*1\s+1\s+([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)", text)
    sig = re.findall(r"c2\s+c3\s*\n\s*1\s+1\s+([-\d.E+]+)\s+", text)
    return [(float(s), float(e[0]), float(e[1])) for s, e in zip(sig, eps, strict=True)]


def test_d6_one_element_deformation_response_matches_the_library_law(tmp_path: Path) -> None:
    # SI inputs, deck in N-mm-MPa: a 1 mm cube, symmetry planes at x = y = z = 0, face pressure at x = 1
    m = M.structured_block(1.0, 1.0, 1.0, 1, 1, 1)
    _write_mesh_files(tmp_path, m)
    fixed = {"u": M.nodes_on_plane(m, 0, 0.0), "v": M.nodes_on_plane(m, 1, 0.0), "w": M.nodes_on_plane(m, 2, 0.0)}
    (tmp_path / "constraints.inp").write_text(M.write_constraints(fixed), encoding="utf-8")
    faces = M.elements_with_face_on_plane(m, 0, 1.0)
    (tmp_path / "unit_tension.inp").write_text(
        M.write_face_pressure_pattern("unit_tension", faces, -1.0), encoding="utf-8"
    )
    levels = np.array([0.5, 0.9, 0.95, 0.97, 0.99, 1.0, 1.01, 1.02, 1.05, 1.1, 1.2]) * U.stress_to_deck(S0)
    spec = D.DeckSpec(
        "one_element",
        m.n_nodes,
        m.n_elements,
        D.DeformationMaterial(U.stress_to_deck(E), NU, U.stress_to_deck(S0), N),
        ("coords.inp", "incid.inp", "constraints.inp", "unit_tension.inp"),
        "pull",
        "unit_tension",
        tuple(float(v) for v in levels),
        (),
        solver=D.SolverControls(solution_technique="sparse direct", norm_res_tol=1e-6, maximum_iterations=30),
        element_output="gausspts short",
        strain_elements="1",
        stress_elements="1",
        output_after_steps=tuple(range(1, len(levels) + 1)),
    )
    (tmp_path / "control.inp").write_text(D.write_deck(spec), encoding="utf-8")
    names = ("coords.inp", "incid.inp", "constraints.inp", "unit_tension.inp", "control.inp")
    inputs = {k: tmp_path / k for k in names}
    runner = R.WarpRunner(IMAGE, threads=1, memory="2g", timeout_seconds=300)
    rec, out = runner.run(tmp_path / "job", inputs, "control.inp")
    assert rec.status == "complete", rec
    rows = _strains_and_stresses(out.read_text(encoding="utf-8", errors="replace"))
    assert len(rows) == len(levels)
    law = WarpDeformationLaw(E, NU, S0, N)
    k2 = 1.0069  # the arc's end for n = 10 (transition_arc)
    worst_outside, worst_arc = 0.0, 0.0
    for sig_mpa, eps_xx, eps_yy in rows:
        sigma = U.stress_from_deck(sig_mpa)
        s = sigma / S0
        mine = float(law.strain(sigma))
        rel = abs(mine - eps_xx) / eps_xx
        if s <= 0.95 or s >= k2:
            worst_outside = max(worst_outside, rel)
        else:
            worst_arc = max(worst_arc, rel)
        # lateral strain: the elastic volumetric part plus half the deviatoric part (incompressible plastic flow)
        vol = (1 - 2 * NU) * sigma / (3 * E)
        assert eps_yy == pytest.approx(vol - 0.5 * (eps_xx - vol), abs=2e-6)
    # the strain print has six decimals (1e-6 absolute, about 6e-4 relative at the smallest strain), so the
    # design's 1e-6 band is the print resolution; the achieved values are recorded for the gate report
    assert worst_outside < 7e-4, worst_outside
    assert worst_arc < 5e-3, worst_arc  # the arc bound of D6 (achieved 0.14 % at s = 1.0)
    (ROOT / "fe" / "runs_gate1" / "d6_worst_rel_diff.txt").write_text(
        f"outside_arc {worst_outside:.3e}\narc {worst_arc:.3e}\n", encoding="utf-8"
    )


def _reactions(text: str) -> list[tuple[int, float, float, float]]:
    rows: list[tuple[int, float, float, float]] = []
    for m in re.finditer(r"nodal reaction forces.*?Totals:", text, re.S):
        block = m.group(0)
        rows = [
            (int(a), float(b), float(c), float(d))
            for a, b, c, d in re.findall(r"^\s*(\d+)\s+([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)\s*$", block, re.M)
        ]
    return rows


def test_d8_uncracked_plate_resultants_reactions_and_linearity(tmp_path: Path) -> None:
    # a plate L x B x W (y through the thickness), deck units mm: 60 x 20 x 40, loaded at x = L by the linear
    # traction Pm + Pb (1 - 2y/B) as consistent nodal loads (quad8 face integration, exact for a linear field)
    b, w, length = 20.0, 40.0, 60.0
    nx, ny, nz = 3, 4, 2
    m = M.structured_block(length, b, w, nx, ny, nz)
    pm, pb = 100.0, 50.0  # MPa
    _write_mesh_files(tmp_path, m)
    fixed = {"u": M.nodes_on_plane(m, 0, 0.0), "v": np.array([1]), "w": M.nodes_on_plane(m, 2, 0.0)}
    (tmp_path / "constraints.inp").write_text(M.write_constraints(fixed), encoding="utf-8")
    faces = M.elements_with_face_on_plane(m, 0, length)
    loads = M.face_traction_nodal_loads(m, faces, lambda x: np.array([pm + pb * (1 - 2 * x[1] / b), 0.0, 0.0]))
    (tmp_path / "unit_load.inp").write_text(M.write_nodal_load_pattern("unit_load", loads), encoding="utf-8")
    # (a) the applied resultants, exactly: force Pm B W and moment Pb B^2 W / 6 about the mid-plane
    origin = np.array([0.0, b / 2, w / 2])
    f_app, m_app = M.resultants(m, loads, origin)
    assert f_app[0] == pytest.approx(pm * b * w, rel=1e-12) and m_app[2] == pytest.approx(
        pb * b * b * w / 6.0, rel=1e-12
    )
    react_nodes = " ".join(str(int(k)) for k in M.nodes_on_plane(m, 0, 0.0))
    levels = (0.5, 1.0, 2.0)
    spec = D.DeckSpec(
        "plate",
        m.n_nodes,
        m.n_elements,
        D.ElasticMaterial(U.stress_to_deck(E), NU, max_expected_stress=200.0),
        ("coords.inp", "incid.inp", "constraints.inp", "unit_load.inp"),
        "load",
        "unit_load",
        levels,
        (),
        solver=D.SolverControls(
            solution_technique="sparse direct", norm_res_tol=1e-8, maximum_iterations=5, minimum_iterations=1
        ),
        element_output="gausspts short",
        reaction_nodes=react_nodes,
        output_after_steps=(1, 2, 3),
    )
    (tmp_path / "control.inp").write_text(D.write_deck(spec), encoding="utf-8")
    names = ("coords.inp", "incid.inp", "constraints.inp", "unit_load.inp", "control.inp")
    inputs = {k: tmp_path / k for k in names}
    runner = R.WarpRunner(IMAGE, threads=2, memory="2g", timeout_seconds=600)
    rec, out = runner.run(tmp_path / "job", inputs, "control.inp")
    assert rec.status == "complete", rec
    parsed = parse_output(out.read_text(encoding="utf-8", errors="replace"))
    qa = run_qa(parsed, "unit_load", levels)
    assert qa.passed, qa
    # (b) the parsed reactions at x = 0 balance the applied force and moment at every step (linearity), the
    # nodal rows to five printed digits, the totals line to eight
    assert len(parsed.reactions) == 3
    eq = equilibrium_qa(parsed, m.coords, M.nodes_on_plane(m, 0, 0.0), f_app, m_app, levels, origin)
    assert all(q.passed and q.nodes_complete for q in eq), eq
    for factor, rx in zip(levels, parsed.reactions, strict=True):
        react = {int(n): rx.forces[i] for i, n in enumerate(rx.nodes)}
        f_r, m_r = M.resultants(m, react, origin)
        assert f_r[0] == pytest.approx(-factor * f_app[0], rel=1e-4)
        assert m_r[2] == pytest.approx(-factor * m_app[2], rel=1e-3)
        assert rx.totals[0] == pytest.approx(-factor * f_app[0], rel=1e-7)
