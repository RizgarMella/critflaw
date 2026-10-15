"""Phase 4 gate (ii), the crack-block mesher: determinism, counts, Jacobian positivity and the quality bands per
region over the gate-(ii) box corners; the live elastic case (D3 against the Phase 1 plate solution, D3b
against the shipped test87 model with the crack-face traction term) is in the live module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from critflaw.fe import crack_mesh as CM
from critflaw.fe import mesh as M


def _quality(p: CM.CrackBlockParams) -> dict[str, float]:
    cb = CM.build(p)
    m = cb.mesh
    j = M.jacobians(m)
    tube = np.zeros(m.n_elements, bool)
    tube[cb.block_elements[: p.n_rings * p.n_sectors * p.n_front] - 1] = True
    far = m.coords[m.incid[:, :8] - 1][:, :, 2].min(axis=1) >= p.R_b - 1e-9
    sj = M.scaled_jacobian(m)
    with np.errstate(divide="ignore"):
        ar = M.aspect_ratio(m)
    sel = ~tube & ~far
    return {
        "nonpositive": float((j.min(axis=1) <= 0).sum()),
        "min_sj_block_footprint": float(sj[sel].min()),
        "max_ar_block_footprint": float(ar[sel].max()),
        "min_sj_far": float(sj[far].min()),
        "max_ar_far": float(ar[far].max()),
        "min_sj_rings_2plus": float(sj[tube & ~np.isin(np.arange(1, m.n_elements + 1), cb.tip_elements)].min()),
        "n_nodes": float(m.n_nodes),
        "n_elements": float(m.n_elements),
    }


@pytest.mark.parametrize("a_c, a_b", [(0.4, 0.2), (0.4, 0.6), (1.0, 0.2), (1.0, 0.6), (0.6, 0.4)])
def test_mesh_is_valid_over_the_gate_ii_box(a_c: float, a_b: float) -> None:
    b = 20.0
    a = a_b * b
    p = CM.CrackBlockParams(a=a, c=a / a_c, B=b, W=max(200.0, 20 * a / a_c), L_far=100.0)
    q = _quality(p)
    assert q["nonpositive"] == 0  # positive Jacobian at every Gauss point of every element, tip wedges included
    assert q["min_sj_block_footprint"] >= 0.2  # the design band for ordinary bricks (blend, footprint, slab)
    assert q["min_sj_rings_2plus"] > 0.9  # the ring elements are near-orthogonal
    assert q["min_sj_far"] >= 0.2
    # the thin footprint layers inside the block height and the far-field columns above the swept strip are
    # long: reported, not banded (design [impl]: the aspect band applies to the crack block)
    assert q["max_ar_far"] < 1000.0


def test_determinism_counts_front_nodes_and_refusals() -> None:
    p = CM.CrackBlockParams(a=8.0, c=20.0, B=20.0, W=200.0, L_far=100.0)
    cb1, cb2 = CM.build(p), CM.build(p)
    assert cb1.topology_hash == cb2.topology_hash and np.array_equal(cb1.mesh.incid, cb2.mesh.incid)
    m = cb1.mesh
    n_block = (p.n_rings + p.n_blend) * p.n_sectors * p.n_front
    assert len(cb1.block_elements) == n_block and len(cb1.tip_elements) == p.n_sectors * p.n_front
    assert m.n_elements > n_block and cb1.mesh.n_nodes > 10000
    # the front: one node per doubled station (tied), its coordinates on the ellipse, a node exactly at phi_s
    for nodes in cb1.front_nodes.values():
        assert len(nodes) == 1
        x, y, z = m.coords[nodes[0] - 1]
        assert (x / p.c) ** 2 + (y / p.a) ** 2 == pytest.approx(1.0, abs=1e-9) and z == 0.0
    assert np.isclose(cb1.phi_corner_stations[p.n_front_s], p.phi_s)
    nodes_s = cb1.domain_front_nodes(p.n_front_s)
    assert len(nodes_s) == 5 and nodes_s[2] == cb1.front_nodes[2 * p.n_front_s][0]  # centred, decreasing phi
    # quarter-point mid-sides on the ring-1 radial edges (tied mesh)
    sz = CM.template_2d(p)
    r1 = np.hypot(*sz[2, 0])
    assert np.hypot(*sz[1, 0]) == pytest.approx(0.25 * r1)
    # node sets: ligament nodes lie on z = 0 outside the ellipse; symmetry nodes on x = 0; loaded faces on L_far
    lig = m.coords[cb1.ligament_nodes - 1]
    assert np.all(np.abs(lig[:, 2]) < 1e-9) and np.all((lig[:, 0] / p.c) ** 2 + (lig[:, 1] / p.a) ** 2 >= 1.0 - 1e-6)
    assert np.all(np.abs(m.coords[cb1.symmetry_x_nodes - 1, 0]) < 1e-9)
    assert len(cb1.loaded_faces) > 0 and all(f == 2 for _, f in cb1.loaded_faces)  # the +z face of the top layer
    # the untied mesh (keyhole collapsed to a point, manual 4.4): one distinct coincident front node per angular
    # position (2 n_sectors + 1 per station), no repeated node number in any tip element, and the crack-plane
    # constraint on the ligament-side tip node only
    cbu = CM.build(CM.CrackBlockParams(a=8.0, c=20.0, B=20.0, W=200.0, L_far=100.0, tip="untied"))
    n_pos = 2 * p.n_sectors + 1
    assert len(cbu.front_nodes[0]) == n_pos and len(set(cbu.front_nodes[0])) == n_pos
    assert len(cbu.front_nodes[1]) == p.n_sectors + 1  # mid station: the corner angular positions only
    assert max(n for v in cbu.front_nodes.values() for n in v) <= cbu.mesh.n_nodes
    assert all(len(set(cbu.mesh.incid[e - 1].tolist())) == 20 for e in cbu.tip_elements)
    assert len(M.coincident_nodes(cbu.mesh)) > 0 and M.coincident_nodes(cb1.mesh) == []
    lig_u = set(cbu.ligament_nodes.tolist())
    for station, nodes in cbu.front_nodes.items():
        assert nodes[0] in lig_u and not any(n in lig_u for n in nodes[1:]), station
        assert np.allclose(cbu.mesh.coords[np.array(nodes) - 1], cbu.mesh.coords[nodes[0] - 1])
    # every tip element's collapsed edge: three distinct nodes (corner, mid-side, corner) per station, coincident
    for e in cbu.tip_elements[:5]:
        xyz = cbu.mesh.coords[cbu.mesh.incid[e - 1] - 1]
        groups = {tuple(np.round(row, 9)) for row in xyz}
        assert len(groups) < 20  # coincident nodes exist in the element
    # refusals: a block that does not fit, a self-intersecting sweep, bad parameters
    with pytest.raises(ValueError, match="fit"):
        CM.build(CM.CrackBlockParams(a=19.0, c=20.0, B=20.0, W=200.0, L_far=100.0, k2=5.0))
    with pytest.raises(ValueError, match="curvature"):
        CM.build(CM.CrackBlockParams(a=12.0, c=20.0, B=20.0, W=200.0, L_far=100.0, k1=5.0, k2=5.0, k3=5.0))
    with pytest.raises(ValueError):
        CM.CrackBlockParams(a=8.0, c=20.0, B=20.0, W=30.0, L_far=100.0)
    with pytest.raises(ValueError):
        CM.CrackBlockParams(a=8.0, c=20.0, B=20.0, W=200.0, L_far=100.0, n_sectors=10)


def test_rendered_views_are_written_and_cover_the_crack_plane_and_symmetry_plane(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from critflaw.fe import render as R

    p = CM.CrackBlockParams(a=8.0, c=20.0, B=20.0, W=200.0, L_far=100.0)
    cb = CM.build(p)
    crack, sym = R.render_views(cb, tmp_path, "case")
    assert crack.name == "case_crack_plane.png" and sym.name == "case_symmetry.png"
    assert crack.stat().st_size > 10_000 and sym.stat().st_size > 10_000
    edges = R.plane_edges(cb.mesh, 2, 0.0)
    assert edges and all(abs(a[2]) < 1e-9 and abs(b[2]) < 1e-9 for a, b in edges)
    xs = np.array([[a[0], b[0]] for a, b in edges])
    assert xs.min() >= -1e-9 and abs(xs.max() - p.W / 2) < 1e-9  # the whole half-width is in the view
    assert not R.plane_edges(cb.mesh, 2, 0.5 * p.L_far)  # no element face lies in a plane between layers


def test_far_field_grading_rule_starts_at_the_blend_thickness_and_ends_at_L_far() -> None:
    p = CM.CrackBlockParams(a=8.0, c=20.0, B=20.0, W=200.0, L_far=200.0)
    w = CM.far_layer_weights(p)
    h = w / w.sum() * (p.L_far - p.R_b)
    assert h[0] == pytest.approx(p.R_b - p.r_t) and np.all(np.diff(h) > 0.0)
    ratios = h[1:] / h[:-1]
    assert np.allclose(ratios, ratios[0]) and 1.0 < ratios[0] <= p.far_grading
    assert h.sum() == pytest.approx(p.L_far - p.R_b)
    assert len(w) == 16
    # halving the first layer adds layers; the legacy rule keeps n_far
    assert len(CM.far_layer_weights(CM.CrackBlockParams(a=8.0, c=20.0, B=20.0, W=200.0, L_far=200.0, far_first=0.5))) > 16
    legacy = CM.CrackBlockParams(a=8.0, c=20.0, B=20.0, W=200.0, L_far=200.0, far_first=0.0, n_far=6)
    assert len(CM.far_layer_weights(legacy)) == 6 and CM.remote_length(20.0, 200.0) == 400.0
    assert CM.remote_length(20.0, 100.0) == 400.0 and CM.remote_length(20.0, 300.0) == 600.0
    with pytest.raises(ValueError):
        CM.CrackBlockParams(a=8.0, c=20.0, B=20.0, W=200.0, L_far=200.0, far_grading=1.0)
