"""Phase 4 gate (ii), live: one elastic surface-flaw case (a/B = 0.4, a/c = 0.4, the geometry of the shipped
test87 model) run in the pinned image; D3 against the Phase 1 plate solution at the same parametric angles with
the design's decision rule, D3b against test87's plane-strain interaction-integral K_I (its domains extended by
the crack-face traction term, see fe/runs_gate2/test87_domain_defs_face_traction.inp; the stored output is the
fixture). Skipped without the image."""

from __future__ import annotations

import gzip
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from critflaw.fe.elastic_case import ElasticCase, run_case
from critflaw.fe.parser import parse_output
from critflaw.fe.runner import WarpRunner

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "critflaw-warp3d:spike"
pytestmark = pytest.mark.skipif(
    shutil.which("docker") is None
    or subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode != 0,
    reason="WARP3D image absent",
)


def _test87_reference() -> tuple[np.ndarray, np.ndarray]:
    """(phi, F = K_I / (sigma sqrt(pi a))) at test87's 21 stations from the stored output with the face-traction
    term (plane strain, mean over rings 2+; the vertex station kept but not used)."""
    text = gzip.open(
        ROOT / "tests" / "data" / "fe" / "test87_face_traction_out.txt.gz", "rt", encoding="utf-8", errors="replace"
    ).read()
    p = parse_output(text)
    defs = (ROOT / "fe" / "external" / "warp3d" / "verification" / "test87" / "domain_defs.inp").read_text(
        encoding="utf-8"
    )
    phis = np.array([float(m) for m in re.findall(r"domain J\d+\s+[$] phi =\s+([\d.]+)", defs)])
    f = []
    for k in range(1, 22):
        # test87 prints both auxiliary fields; the plane-strain block is the second of the two
        cands = [d for d in p.domains if d.domain_id == f"J{k}" and d.mode == "KI" and d.aux_field == "plane strain"]
        assert len(cands) == 1
        d = cands[0]
        assert d.k is not None
        f.append(float(np.mean(d.k[1:])) / np.sqrt(np.pi * 0.4))
        if k > 1:
            assert float(np.max(np.abs(d.k[1:] - np.mean(d.k[1:]))) / abs(np.mean(d.k[1:]))) < 0.01  # path independent
    return phis, np.asarray(f)


def test_d3_and_d3b_elastic_surface_flaw(tmp_path: Path) -> None:
    case = ElasticCase(a=8e-3, c=20e-3, B=20e-3, W=200e-3, L_far=400e-3, E=207e9, nu=0.3, pm=100e6, pb=0.0)
    res = run_case(case, tmp_path / "case", WarpRunner(IMAGE, threads=4, memory="6g", timeout_seconds=1800))
    assert res.record.status == "complete"
    # D8: the remote stress profile Pm + Pb (1 - 2y/B) at the far-field layer nearest z = L_far / 2 (10 B from the
    # crack plane at L_far = 20 B), element means of sigma_zz over the Gauss points, within 0.5 % of |Pm| + |Pb|
    assert 8.0 < res.remote_profile_z_over_B < 12.0 and res.remote_profile_dev < 0.005, res.remote_profile_dev
    interior = slice(1, None)  # the free-surface vertex is excluded from every criterion (design 2.1)
    assert np.all(res.k_spread[interior] < 0.005)  # ring spread (rings 2..n) well inside the 2 % band
    # D3 decision rule: within 3 % of the fit passes; outside 3 % but within the fit's documented accuracy is
    # unresolved (listed), never a fail; the achieved differences are recorded
    diff = res.k_fe[interior] / res.k_ref[interior] - 1.0
    unresolved = [float(np.degrees(ph)) for ph, d in zip(res.phi[interior], diff, strict=True) if abs(d) > 0.03]
    assert np.all(np.abs(diff) < 0.05)
    assert abs(diff[-1]) < 0.03  # the deepest point passes
    # D3b: the independent FE (test87, 8-node B-bar, FEACrack mesh, crack-face pressure = remote tension) at the
    # same parametric angles, PCHIP in phi; expected sign: ours above test87 (its header: 1-3 % below 20-node)
    phi87, f87 = _test87_reference()
    pchip = PchipInterpolator(phi87[1:], f87[1:])
    f_fe = res.k_fe / (case.pm * np.sqrt(np.pi * case.a))
    signed = f_fe[interior] / pchip(res.phi[interior]) - 1.0
    # design D3b rule: pass within 3 %; a positive difference up to 3 % + test87's stated 1-3 % is recorded as
    # 'attributed to the reference's discretisation' (listed), never a pass or a fail
    assert np.all(signed > -0.005)  # never below the 8-node reference by more than rounding
    assert np.all(signed < 0.06), signed
    attributed = [float(np.degrees(ph)) for ph, d in zip(res.phi[interior], signed, strict=True) if d > 0.03]
    assert abs(signed[case.n_front_s - 1]) < 0.03 and abs(signed[-1]) < 0.03  # phi_s and the deepest point pass
    (ROOT / "fe" / "runs_gate2" / "d3_d3b_record.txt").write_text(
        "phi_deg fe_vs_NR_pct fe_vs_test87_pct spread_pct"
        + chr(10)
        + "".join(
            f"{np.degrees(ph):.2f} {d * 100:+.2f} {s87 * 100:+.2f} {sp * 100:.2f}" + chr(10)
            for ph, d, s87, sp in zip(res.phi[interior], diff, signed, res.k_spread[interior], strict=True)
        )
        + f"unresolved_vs_fit_deg {unresolved}"
        + chr(10)
        + f"attributed_to_test87_discretisation_deg {attributed}"
        + chr(10),
        encoding="utf-8",
    )
    # the K-J consistency of the interaction integral: J from K (plane strain) equals K^2 (1 - nu^2) / E
    j_from_k = res.k_fe[interior] ** 2 * (1 - case.nu**2) / case.E
    assert np.allclose(res.j_mixed[interior], j_from_k, rtol=0.01)


def test_jsolver_runs_the_mid_point_case_and_labels_it(tmp_path: Path) -> None:
    """The FECase pipeline end to end at the production level: admissibility run, plastic run with its elastic
    companion, packets, far-region strain cap, every QA check with its value and band; the label is
    'unverified' (family verification incomplete), never 'fe-verified'."""
    from critflaw.fe import fecase as F

    case = F.FECase(a=8e-3, c=20e-3, B=20e-3, W=200e-3, L_far=400e-3, E=207e9, nu=0.3, sigma_0=355e6, n=10.0, beta=0.0)
    solver = F.WarpJSolver(tmp_path, WarpRunner(IMAGE, threads=4, memory="8g", timeout_seconds=1800))
    res = solver.run(case)
    assert res.status == "complete" and res.verification_status == "unverified" and res.failing_check is None, res.qa
    assert all(c.passed for c in res.qa), [c for c in res.qa if not c.passed]
    assert (tmp_path / case.case_id / "plastic" / "job" / "packets.bpf").exists()
    assert np.all(~np.isnan(res.max_far_eff_strain[res.lr >= 0.8])) and np.isnan(res.max_far_eff_strain[1])
    assert res.max_far_eff_strain[-1] < 0.05 and not any(res.outside_small_strain)
    assert res.kr[0, -1] == pytest.approx(1.0) and 0.5 < res.kr[-1, -1] < 0.6
    assert np.all(res.k_elastic[1:] > 0.0)
    F.write_summary(res, tmp_path / "summary.json")
    d = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert d["verification_status"] == "unverified" and d["provenance"]["image_id"].startswith("sha256:")
