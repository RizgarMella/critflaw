"""Phase 4 gate (i), the deck writer: every emitted command is in docs/manual/warp3d_commands.md; the spike's control deck
regenerated from a DeckSpec around its mesh files (declared cumulative factors read from the stored output
instead of the deck's J-adaptive stepping) reproduces the shipped run's J at every step and domain."""

from __future__ import annotations

import gzip
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from critflaw.fe import deck as D
from critflaw.fe import runner as R
from critflaw.fe.parser import parse_output

ROOT = Path(__file__).resolve().parents[1]
SPIKE = ROOT / "fe" / "external" / "warp3d" / "fracture_models" / "corner_at_hole"
COMMANDS = ROOT / "docs" / "manual" / "warp3d_commands.md"
IMAGE = "critflaw-warp3d:spike"


def _listed_keywords() -> set[str]:
    out: set[str] = set()
    for line in COMMANDS.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*(`[^|]+)\|", line)
        if m:
            for kw in re.findall(r"`([^`]+)`", m.group(1)):
                out.add(kw.lower())
    return out


def _spec(cumulative: tuple[float, ...], domains: tuple[D.DomainSpec, ...]) -> D.DeckSpec:
    return D.DeckSpec(
        structure="plate",
        n_nodes=34852,
        n_elements=7344,
        material=D.DeformationMaterial(30000.0, 0.3, 60.0, 10.0),
        input_files=("coords.inp", "incid.inp", "constraints.inp", "unit_load.inp"),
        loading_name="tension",
        pattern_name="unit_remote",
        cumulative_factors=cumulative,
        domains=domains,
        material_name="steel",
        solver=D.SolverControls(),
    )


def _domains_from_get_j_values(text: str) -> tuple[D.DomainSpec, ...]:
    out = []
    for block in re.split(r"(?m)^domain ", text)[1:]:
        lines = [ln.split("$")[0].strip() for ln in block.splitlines()]
        name = lines[0].split()[0]
        front = tuple(int(t) for t in re.search(r"front nodes\s+([\d\s]+)q3disop", block).group(1).split())  # type: ignore[union-attr]
        tangent = None
        tm = re.search(r"tangent vector tx\s*([-\d.]+)\s*ty\s*([-\d.]+)\s*tz\s*([-\d.]+)", block)
        if tm:
            tangent = (float(tm.group(1)), float(tm.group(2)), float(tm.group(3)))
        ftype = re.search(r"function type (\w)", block).group(1)  # type: ignore[union-attr]
        out.append(D.DomainSpec(name, front, (1, 8), True, tangent=tangent, function_type=ftype, print_extents=False))
    return tuple(out)


def test_every_emitted_command_is_listed_in_commands_md() -> None:
    doms = (
        D.DomainSpec("J1", (1, 2, 3), (1, 8), True, tangent=(0.0, -1.0, 0.0), function_type="a"),
        D.DomainSpec("K1", (1, 2, 3), (1, 8), False, kind="II"),
        D.DomainSpec("K2", (1, 2, 3), (1, 8), True, kind="II", plane_stress=True),
    )
    for material in (
        D.ElasticMaterial(207e9, 0.3, max_expected_stress=300e6),
        D.DeformationMaterial(207e9, 0.3, 355e6, 10),
    ):
        spec = D.DeckSpec(
            "p", 10, 1, material, ("c.inp", "i.inp"), "L", "pat", (1.0, 2.0, 3.5), doms,
            reaction_nodes="1-4", stress_elements="1", strain_elements="1", output_after_steps=(1,),
        )  # fmt: skip
        text = D.write_deck(spec)
        listed = _listed_keywords()
        missing = {kw for kw in D.command_keywords(text) if kw not in listed}  # exact match, no prefixes
        assert not missing, missing
        assert "compute displacements" in D.command_keywords(text) and "convergence test" in D.command_keywords(text)
    # every generated include file is audited too (coordinates, incidences, constraints, face and nodal loads)
    from critflaw.fe import mesh as M

    m = M.structured_block(1.0, 1.0, 1.0, 1, 1, 1)
    faces = M.elements_with_face_on_plane(m, 0, 1.0)
    includes = (
        M.write_coordinates(m), M.write_incidences(m), M.write_constraints({"u": M.nodes_on_plane(m, 0, 0.0)}),
        M.write_face_pressure_pattern("p", faces, -1.0),
        M.write_nodal_load_pattern("n", M.face_traction_nodal_loads(m, faces, lambda x: np.array([1.0, 1.0, 1.0]))),
    )  # fmt: skip
    for inc in includes:
        missing = {kw for kw in D.command_keywords(inc) if kw not in _listed_keywords()}
        assert not missing, missing
        assert max(len(ln) for ln in inc.splitlines()) <= 80
    assert D.declared_steps(text) == 3
    assert D.declared_steps(" compute displacements for loading test step 100") == 100
    assert D.declared_steps("compute displacements loading x for steps 10-20") == 20
    with pytest.raises(ValueError, match="compute"):
        D.declared_steps("loading x")
    assert max(len(ln) for ln in text.splitlines()) <= D.MAX_COLUMNS
    assert "step 1 pat 1.0" in text and "step 3 pat 1.5" in text  # increments from the cumulative factors
    assert "plane strain output for interaction integral" in text and "plane stress output" in text
    with pytest.raises(ValueError):
        D.DeckSpec("p", 1, 1, D.ElasticMaterial(1.0, 0.3), (), "L", "pat", (1.0, 0.5), ())
    # full-precision sampled properties stay within 80 columns (one property per continuation line)
    mat = D.DeformationMaterial(206999.999999, 0.299999999999, 354.999999999, 10.0000000001)
    text = D.write_deck(D.DeckSpec("p", 1, 1, mat, (), "L", "pat", (1.0,), ()))
    assert "n_power 10.0000000001" in text and max(len(ln) for ln in text.splitlines()) <= 80
    with pytest.raises(ValueError, match="columns"):
        D.write_deck(D.DeckSpec("p" * 90, 1, 1, mat, (), "L", "pat", (1.0,), ()))
    with pytest.raises(ValueError, match="need rings"):
        D.check_domain_rings((D.DomainSpec("J1", (1, 2, 3), (1, 5), True),))
    D.check_domain_rings((D.DomainSpec("J1", (1, 2, 3), (1, 8), True),))


@pytest.mark.skipif(
    shutil.which("docker") is None
    or subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode != 0
    or not SPIKE.exists(),
    reason="WARP3D image or checkout absent",
)
def test_regenerated_spike_deck_reproduces_the_shipped_run(tmp_path: Path) -> None:
    stored = parse_output(
        gzip.open(
            ROOT / "tests" / "data" / "fe" / "corner_at_hole_t4_out.txt.gz", "rt", encoding="utf-8", errors="replace"
        ).read()
    )
    cumulative = tuple(s.load_factors["unit_rem"] for s in stored.steps)
    n = 12  # the first twelve of the 96 steps: the comparison, not the whole 284 s run
    domains = _domains_from_get_j_values((SPIKE / "get_j_values").read_text(encoding="utf-8"))
    assert len(domains) == 17
    spec = _spec(cumulative[:n], domains)
    text = D.write_deck(spec)
    deck = tmp_path / "control.inp"
    deck.write_text(text, encoding="utf-8")
    inputs = {name: SPIKE / name for name in spec.input_files}
    inputs["control.inp"] = deck
    record, out = R.WarpRunner(IMAGE, threads=4, memory="4g", timeout_seconds=900).run(
        tmp_path / "job", inputs, "control.inp"
    )
    assert record.status == "complete", record
    mine = parse_output(out.read_text(encoding="utf-8", errors="replace"))
    from critflaw.fe.parser import check_domain_coverage

    for step in range(1, n + 1):
        check_domain_coverage(mine, {d.domain_id: d.rings for d in domains}, step)
    worst = 0.0
    for step in range(1, n + 1):
        assert mine.steps[step - 1].load_factors["unit_rem"] == pytest.approx(cumulative[step - 1], rel=1e-9)
        for d in domains:
            a, b = mine.domain(d.domain_id, step), stored.domain(d.domain_id, step)
            worst = max(worst, float(np.max(np.abs(a.total - b.total) / np.abs(b.total))))
    # the declared schedule reproduces the J-adaptive run at the same cumulative loads; the printed J has four
    # significant digits (the recorded worst relative difference is reported in the gate report)
    assert worst < 1.5e-3, worst
    (ROOT / "fe" / "runs_gate1" / "regenerated_worst_rel_diff.txt").write_text(
        f"{worst:.3e}" + chr(10), encoding="utf-8"
    )
