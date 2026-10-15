"""Phase 4 gate (i), the hex20 mesh tools: shape-function identities, the structured block, the WARP3D file
round trip, and the mesh-layer QA on the spike's shipped mesh."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from critflaw.fe import mesh as M

ROOT = Path(__file__).resolve().parents[1]
SPIKE = ROOT / "fe" / "external" / "warp3d" / "fracture_models" / "corner_at_hole"


def test_shape_function_gradients_partition_of_unity_and_unit_cube_jacobian() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        xi, eta, zeta = rng.uniform(-1, 1, 3)
        g = M.shape_function_gradients(xi, eta, zeta)
        assert np.allclose(g.sum(axis=0), 0.0, atol=1e-13)  # sum N_i = 1
    cube = M.structured_block(2.0, 2.0, 2.0, 1, 1, 1)
    assert cube.n_nodes == 20 and cube.n_elements == 1
    # the element [0,2]^3 has det J = 1 at every point (x = xi + 1): the gradient reproduces linear fields
    assert np.allclose(M.jacobians(cube), 1.0)
    for d in range(3):
        g = M.shape_function_gradients(0.3, -0.2, 0.7)
        assert np.allclose(g.T @ cube.coords[cube.incid[0] - 1][:, d], np.eye(3)[d])  # d x_d / d xi = I
    assert M.scaled_jacobian(cube)[0] == pytest.approx(1.0) and M.aspect_ratio(cube)[0] == pytest.approx(1.0)
    assert M.coincident_nodes(cube) == []
    assert M.jacobians(cube, 3).shape == (1, 27) and np.allclose(M.jacobians(cube, 1), 1.0)
    with pytest.raises(ValueError):
        M.jacobians(cube, 4)


def test_structured_block_counts_faces_and_file_round_trip() -> None:
    nx, ny, nz = 3, 2, 2
    m = M.structured_block(3.0, 1.0, 0.5, nx, ny, nz)
    expected_nodes = (
        (nx + 1) * (ny + 1) * (nz + 1) + nx * (ny + 1) * (nz + 1) + ny * (nx + 1) * (nz + 1) + nz * (nx + 1) * (ny + 1)
    )
    assert m.n_nodes == expected_nodes and m.n_elements == nx * ny * nz
    assert np.all(M.jacobians(m) > 0.0) and np.all(M.scaled_jacobian(m) > 0.99)
    assert np.allclose(M.aspect_ratio(m), 4.0)  # element 1 x 0.5 x 0.25
    # the loaded end x = 3: ny*nz faces, all face 5 (nodes 2-6-7-3 are the +xi corners)
    faces = M.elements_with_face_on_plane(m, 0, 3.0)
    assert len(faces) == ny * nz and all(f == 5 for _, f in faces)
    assert (
        len(M.elements_with_face_on_plane(m, 0, 0.0)) == ny * nz
        and len(M.nodes_on_plane(m, 1, 0.0)) == (2 * nx + 1) * (2 * nz + 1) - nx * nz
    )
    back = M.read_mesh(M.write_coordinates(m), M.write_incidences(m))
    assert np.allclose(back.coords, m.coords) and np.array_equal(back.incid, m.incid)
    text = M.write_face_pressure_pattern("unit", faces, -1.0)
    assert text.count("face 5 pressure") == ny * nz and "loading unit" in text
    assert "   1  u 0.0" in M.write_constraints({"u": np.array([1])})
    with pytest.raises(ValueError, match="20 expected"):
        M.read_mesh(M.write_coordinates(m), M.write_incidences(m).replace("      20", "", 1))


def test_consistent_nodal_loads_reproduce_the_tractions_resultants_exactly() -> None:
    # a linear traction over a curved-free rectangular face: the quad8 integration is exact
    b, w, length = 20.0, 40.0, 60.0
    m = M.structured_block(length, b, w, 3, 4, 2)
    faces = M.elements_with_face_on_plane(m, 0, length)
    pm, pb = 100.0, 50.0
    loads = M.face_traction_nodal_loads(m, faces, lambda x: np.array([pm + pb * (1 - 2 * x[1] / b), 0.0, 0.0]))
    force, moment = M.resultants(m, loads, np.array([length, b / 2, w / 2]))
    assert force == pytest.approx([pm * b * w, 0.0, 0.0], rel=1e-12, abs=1e-9)
    assert moment == pytest.approx([0.0, 0.0, pb * b * b * w / 6.0], rel=1e-12, abs=1e-9)
    # corner nodes of a quad8 face carry a negative share of a uniform traction (serendipity), mid-sides positive
    uniform = M.face_traction_nodal_loads(m, faces[:1], lambda x: np.array([1.0, 0.0, 0.0]))
    e, face = faces[0]
    corners = [int(m.incid[e - 1, k - 1]) for k in M.FACE_NODES[face]]
    assert all(uniform[n][0] < 0 for n in corners) and sum(v[0] for v in uniform.values()) == pytest.approx(
        (b / 4) * (w / 2)
    )
    text = M.write_nodal_load_pattern("unit", loads)
    assert text.startswith("*echo off") and "nodal loads" in text and "force_x" in text and "force_y" not in text


@pytest.mark.skipif(not SPIKE.exists(), reason="WARP3D checkout absent")
def test_spike_mesh_reads_and_passes_the_mesh_layer() -> None:
    m = M.read_mesh(
        (SPIKE / "coords.inp").read_text(encoding="utf-8"), (SPIKE / "incid.inp").read_text(encoding="utf-8")
    )
    assert m.n_nodes == 34852 and m.n_elements == 7344
    j = M.jacobians(m)
    assert np.all(j > 0.0)  # positive at every Gauss point, collapsed tip elements included
    sj = M.scaled_jacobian(m)
    collapsed = sj < 1e-9
    assert 0 < int(collapsed.sum()) < m.n_elements  # the crack-tip wedges
    pairs = M.coincident_nodes(m)
    assert len(pairs) > 0  # untied collapsed-node sets along the front
    assert np.all(sj[~collapsed] > 0.05)
