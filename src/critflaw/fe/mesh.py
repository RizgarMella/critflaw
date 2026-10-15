"""Hex20 (WARP3D ``q3disop``) meshes: a structured block generator, the WARP3D coordinate/incidence/constraint/
load-pattern writers, readers for shipped meshes, and the mesh-layer QA (theory manual, sections 6.1.3 and
6.1.7: scaled Jacobian and aspect ratio per ordinary brick, element count, coincident nodes).

Node ordering of ``q3disop`` (manual Fig. 3.1.1, identical to Abaqus C3D20): corners 1-4 on the face
``zeta = +1`` ... in WARP3D's own parametric labels; here the usual Abaqus description is used: corners 1-4
on the bottom face, 5-8 on the top face (same circulation), 9-12 the bottom edge mid-sides (1-2, 2-3, 3-4,
4-1), 13-16 the top edge mid-sides (5-6, 6-7, 7-8, 8-5), 17-20 the vertical mid-sides (1-5, 2-6, 3-7, 4-8).
Face numbers (manual Fig. 3.1.4): 1 = 1-2-3-4, 2 = 5-8-7-6, 3 = 1-5-6-2, 4 = 4-8-7-3, 5 = 2-6-7-3, 6 = 1-5-8-4;
a positive pressure loads the face in compression.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

FACE_NODES: dict[int, tuple[int, int, int, int]] = {
    1: (1, 2, 3, 4),
    2: (5, 8, 7, 6),
    3: (1, 5, 6, 2),
    4: (4, 8, 7, 3),
    5: (2, 6, 7, 3),
    6: (1, 5, 8, 4),
}

# corner natural coordinates (xi, eta, zeta) in the Abaqus convention used for the shape functions here
_CORNERS = np.array(
    [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float
)
_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


@dataclass(frozen=True)
class Hex20Mesh:
    coords: FloatArray  # (n_nodes, 3), node i has number i + 1
    incid: IntArray  # (n_elements, 20), 1-based node numbers

    @property
    def n_nodes(self) -> int:
        return int(self.coords.shape[0])

    @property
    def n_elements(self) -> int:
        return int(self.incid.shape[0])


def shape_function_gradients(xi: float, eta: float, zeta: float) -> FloatArray:
    """(20, 3) derivatives of the serendipity hex20 shape functions at a natural point."""
    g = np.zeros((20, 3))
    # corner derivatives: N = 1/8 (1+a)(1+b)(1+c)(a+b+c-2)
    for i in range(8):
        xi0, eta0, zeta0 = _CORNERS[i]
        a, b, c = xi * xi0, eta * eta0, zeta * zeta0
        s = a + b + c - 2.0
        g[i, 0] = 0.125 * xi0 * (1 + b) * (1 + c) * (s + (1 + a))
        g[i, 1] = 0.125 * eta0 * (1 + a) * (1 + c) * (s + (1 + b))
        g[i, 2] = 0.125 * zeta0 * (1 + a) * (1 + b) * (s + (1 + c))
    # mid-side nodes: edge (p, q) with the zero natural coordinate along the edge
    for k, (p, q) in enumerate(_EDGES):
        i = 8 + k
        cp, cq = _CORNERS[p], _CORNERS[q]
        mid = 0.5 * (cp + cq)
        d = int(np.argmin(np.abs(mid)))  # the coordinate that varies along the edge (mid = 0 there)
        others = [j for j in range(3) if j != d]
        nat = np.array([xi, eta, zeta])
        # N = 1/4 (1 - t_d^2) (1 + t_o1 m_o1) (1 + t_o2 m_o2)
        o1, o2 = others
        f1, f2 = 1 + nat[o1] * mid[o1], 1 + nat[o2] * mid[o2]
        g[i, d] = 0.25 * (-2.0 * nat[d]) * f1 * f2
        g[i, o1] = 0.25 * (1 - nat[d] ** 2) * mid[o1] * f2
        g[i, o2] = 0.25 * (1 - nat[d] ** 2) * f1 * mid[o2]
    return g


_GAUSS: dict[int, FloatArray] = {
    1: np.array([0.0]),
    2: np.array([-1.0, 1.0]) / np.sqrt(3.0),
    3: np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)]),
}


def jacobians(mesh: Hex20Mesh, order: int = 2) -> FloatArray:
    """Jacobian determinant at every Gauss point of every element, (n_elements, order^3), for the Gauss-Legendre
    orders 1, 2, 3 (WARP3D's 2x2x2 is the production order; others raise)."""
    if order not in _GAUSS:
        raise ValueError(f"integration order {order} is not supported (1, 2 or 3 points per direction)")
    pts = _GAUSS[order]
    grads = [shape_function_gradients(x, y, z) for x in pts for y in pts for z in pts]
    out = np.zeros((mesh.n_elements, len(grads)))
    for e in range(mesh.n_elements):
        x = mesh.coords[mesh.incid[e] - 1]  # (20, 3)
        for k, g in enumerate(grads):
            out[e, k] = np.linalg.det(g.T @ x)
    return out


def scaled_jacobian(mesh: Hex20Mesh) -> FloatArray:
    """Per element: the minimum over the 8 corners of det(J) scaled by the product of the three corner edge
    lengths (1 for a cube, 0 for a degenerate corner), sign kept."""
    out = np.zeros(mesh.n_elements)
    for e in range(mesh.n_elements):
        x = mesh.coords[mesh.incid[e] - 1]
        vals = []
        for i in range(8):
            xi0, eta0, zeta0 = _CORNERS[i]
            g = shape_function_gradients(xi0, eta0, zeta0)
            j = g.T @ x
            lengths = np.linalg.norm(j, axis=1)
            vals.append(np.linalg.det(j) / np.prod(np.where(lengths > 0, lengths, 1.0)))
        out[e] = min(vals)
    return out


def aspect_ratio(mesh: Hex20Mesh) -> FloatArray:
    """Per element: longest / shortest corner edge."""
    out = np.zeros(mesh.n_elements)
    for e in range(mesh.n_elements):
        x = mesh.coords[mesh.incid[e] - 1]
        lengths = [np.linalg.norm(x[p] - x[q]) for p, q in _EDGES]
        out[e] = max(lengths) / min(lengths)
    return out


def coincident_nodes(mesh: Hex20Mesh, tol: float = 1e-12) -> list[tuple[int, int]]:
    """Pairs of distinct node numbers at the same place (collapsed tip nodes are expected; elsewhere a defect)."""
    if not np.all(np.isfinite(mesh.coords)):
        raise ValueError("the mesh has unnumbered nodes (non-finite coordinates)")
    scale = max(float(np.ptp(mesh.coords)), 1.0)
    rounded = np.round(mesh.coords / (tol * scale)).astype(np.int64)
    seen: dict[tuple[int, int, int], int] = {}
    pairs = []
    for i, key in enumerate(map(tuple, rounded)):
        if key in seen:
            pairs.append((seen[key] + 1, i + 1))
        else:
            seen[key] = i
    return pairs


def structured_block(lx: float, ly: float, lz: float, nx: int, ny: int, nz: int) -> Hex20Mesh:
    """A box [0, lx] x [0, ly] x [0, lz] of nx x ny x nz hex20 elements, deterministic numbering (a mesh is a
    pure function of its parameters). Nodes on a (2nx+1) x (2ny+1) x (2nz+1) lattice without the centre nodes
    of faces and elements (serendipity)."""
    if min(nx, ny, nz) < 1:
        raise ValueError("at least one element in each direction")
    numbering: dict[tuple[int, int, int], int] = {}
    coords: list[tuple[float, float, float]] = []

    def node(i: int, j: int, k: int) -> int:
        key = (i, j, k)
        if key not in numbering:
            numbering[key] = len(coords) + 1
            coords.append((lx * i / (2 * nx), ly * j / (2 * ny), lz * k / (2 * nz)))
        return numbering[key]

    incid = []
    for ez in range(nz):
        for ey in range(ny):
            for ex in range(nx):
                i0, j0, k0 = 2 * ex, 2 * ey, 2 * ez
                n = node
                c = [n(i0, j0, k0), n(i0 + 2, j0, k0), n(i0 + 2, j0 + 2, k0), n(i0, j0 + 2, k0)]
                c += [n(i0, j0, k0 + 2), n(i0 + 2, j0, k0 + 2), n(i0 + 2, j0 + 2, k0 + 2), n(i0, j0 + 2, k0 + 2)]
                m = [n(i0 + 1, j0, k0), n(i0 + 2, j0 + 1, k0), n(i0 + 1, j0 + 2, k0), n(i0, j0 + 1, k0)]
                m += [
                    n(i0 + 1, j0, k0 + 2),
                    n(i0 + 2, j0 + 1, k0 + 2),
                    n(i0 + 1, j0 + 2, k0 + 2),
                    n(i0, j0 + 1, k0 + 2),
                ]
                m += [n(i0, j0, k0 + 1), n(i0 + 2, j0, k0 + 1), n(i0 + 2, j0 + 2, k0 + 1), n(i0, j0 + 2, k0 + 1)]
                incid.append(c + m)
    return Hex20Mesh(np.asarray(coords, dtype=float), np.asarray(incid, dtype=np.int64))


def nodes_on_plane(mesh: Hex20Mesh, axis: int, value: float, tol: float = 1e-9) -> IntArray:
    """1-based numbers of the nodes with ``coords[:, axis] == value``."""
    scale = max(float(np.ptp(mesh.coords[:, axis])), 1.0)
    return np.nonzero(np.abs(mesh.coords[:, axis] - value) <= tol * scale)[0] + 1


def elements_with_face_on_plane(mesh: Hex20Mesh, axis: int, value: float, tol: float = 1e-9) -> list[tuple[int, int]]:
    """``(element number, face number)`` for every element face lying in the plane."""
    on = set(int(n) for n in nodes_on_plane(mesh, axis, value, tol))
    out = []
    for e in range(mesh.n_elements):
        for face, corners in FACE_NODES.items():
            if all(int(mesh.incid[e, c - 1]) in on for c in corners):
                out.append((e + 1, face))
    return out


# ---- WARP3D file writers and readers -----------------------------------------------------------------------


def write_coordinates(mesh: Hex20Mesh) -> str:
    lines = ["*echo off", "coordinates"]
    for i, (x, y, z) in enumerate(mesh.coords, start=1):
        lines.append(f"{i:8d}  {x: .14E}  {y: .14E}  {z: .14E}")
    return _guarded(lines)


def write_incidences(mesh: Hex20Mesh) -> str:
    lines = ["*echo off", "incidences"]
    for e, nodes in enumerate(mesh.incid, start=1):
        # three lines of 8, 8 and 4 numbers (the shipped meshes' layout; a long continuation line is misread)
        first = " ".join(f"{int(n):7d}" for n in nodes[:8])
        second = " ".join(f"{int(n):7d}" for n in nodes[8:16])
        third = " ".join(f"{int(n):7d}" for n in nodes[16:])
        lines.append(f"{e:8d} {first},")
        lines.append(f"         {second},")
        lines.append(f"         {third}")
    return _guarded(lines)


def write_constraints(fixed: dict[str, IntArray]) -> str:
    """``{'u' | 'v' | 'w': node numbers}`` fixed to zero (manual 2.7.1)."""
    lines = ["*echo off", "constraints"]
    for dof, nodes in fixed.items():
        for n in nodes:
            lines.append(f"{int(n):8d}  {dof} 0.0")
    return _guarded(lines)


def write_face_pressure_pattern(name: str, faces: list[tuple[int, int]], pressure: float) -> str:
    """A loading pattern of uniform normal pressure on element faces (positive = into the face); a remote
    tension of 1 is ``pressure = -1`` on the loaded end (manual 3.1.5)."""
    lines = ["*echo off", f"loading {name}", "  element loads"]
    for e, face in faces:
        lines.append(f"    {e:7d} face {face} pressure {pressure: .10E}")
    return _guarded(lines)


def read_mesh(coords_text: str, incid_text: str) -> Hex20Mesh:
    """Read WARP3D coordinate and incidence files (continuation lines end with a comma)."""
    coords: dict[int, tuple[float, float, float]] = {}
    for line in coords_text.splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) >= 4 and parts[0].isdigit():
            coords[int(parts[0])] = (float(parts[1]), float(parts[2]), float(parts[3]))
    n = max(coords)
    arr = np.full((n, 3), np.nan)
    for k, xyz in coords.items():
        arr[k - 1] = xyz
    joined = re.sub(r",\s*\r?\n", " ", incid_text)
    rows = []
    for line in joined.splitlines():
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        if len(parts) != 21:
            raise ValueError(f"incidence row with {len(parts) - 1} node numbers (20 expected): {line.strip()[:60]!r}")
        rows.append([int(p) for p in parts[1:21]])
    if not np.all(np.isfinite(arr)):
        raise ValueError("coordinate numbering has gaps")
    return Hex20Mesh(arr, np.asarray(rows, dtype=np.int64))


# ---- consistent nodal loads from a face traction and their resultants --------------------------------------

_QUAD_CORNERS = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
_QUAD_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]
_FACE_MIDSIDES: dict[int, tuple[int, int, int, int]] = {}  # face -> the four mid-side local node numbers


def _face_midsides(face: int) -> tuple[int, int, int, int]:
    """The mid-side local nodes (1-based) of a face's four edges, in the face's corner circulation."""
    if face not in _FACE_MIDSIDES:
        corners = FACE_NODES[face]
        out = []
        for p, q in _QUAD_EDGES:
            cp, cq = corners[p], corners[q]
            for k, (ep, eq) in enumerate(_EDGES):
                if {ep + 1, eq + 1} == {cp, cq}:
                    out.append(9 + k)
                    break
            else:
                raise RuntimeError("face edge not found")
        _FACE_MIDSIDES[face] = (out[0], out[1], out[2], out[3])
    return _FACE_MIDSIDES[face]


def _quad8_shape(xi: float, eta: float) -> tuple[FloatArray, FloatArray]:
    """Serendipity 8-node quad: (N (8,), dN/d(xi, eta) (8, 2)), corners then mid-sides in circulation order."""
    n = np.zeros(8)
    g = np.zeros((8, 2))
    for i in range(4):
        x0, e0 = _QUAD_CORNERS[i]
        a, b = xi * x0, eta * e0
        n[i] = 0.25 * (1 + a) * (1 + b) * (a + b - 1)
        g[i, 0] = 0.25 * x0 * (1 + b) * (2 * a + b)
        g[i, 1] = 0.25 * e0 * (1 + a) * (a + 2 * b)
    mids = [(0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)]  # edge midpoints 1-2, 2-3, 3-4, 4-1
    for k, (mx, me) in enumerate(mids):
        i = 4 + k
        if mx == 0.0:
            n[i] = 0.5 * (1 - xi * xi) * (1 + eta * me)
            g[i, 0] = -xi * (1 + eta * me)
            g[i, 1] = 0.5 * (1 - xi * xi) * me
        else:
            n[i] = 0.5 * (1 - eta * eta) * (1 + xi * mx)
            g[i, 0] = 0.5 * (1 - eta * eta) * mx
            g[i, 1] = -eta * (1 + xi * mx)
    return n, g


def face_traction_nodal_loads(
    mesh: Hex20Mesh,
    faces: list[tuple[int, int]],
    traction: Callable[[FloatArray], FloatArray],
    order: int = 3,
) -> dict[int, FloatArray]:
    """Consistent nodal loads ``{node: (fx, fy, fz)}`` of the traction field ``traction(xyz) -> (3,)`` (force per
    area, global components) applied on the given element faces, integrated with the quad8 face shape functions
    by Gauss-Legendre quadrature of the given order. Exact for a traction polynomial of the face's degree."""
    if order not in _GAUSS:
        raise ValueError("integration order must be 1, 2 or 3")
    pts, wts = _GAUSS[order], {1: np.array([2.0]), 2: np.array([1.0, 1.0]), 3: np.array([5 / 9, 8 / 9, 5 / 9])}[order]
    loads: dict[int, FloatArray] = {}
    for e, face in faces:
        local = list(FACE_NODES[face]) + list(_face_midsides(face))
        nodes = [int(mesh.incid[e - 1, k - 1]) for k in local]
        x = mesh.coords[np.asarray(nodes) - 1]  # (8, 3)
        for i, xi in enumerate(pts):
            for j, eta in enumerate(pts):
                n, g = _quad8_shape(float(xi), float(eta))
                t1, t2 = g[:, 0] @ x, g[:, 1] @ x
                da = float(np.linalg.norm(np.cross(t1, t2)))
                t = np.asarray(traction(n @ x), dtype=float)
                for k, node in enumerate(nodes):
                    loads[node] = loads.get(node, np.zeros(3)) + wts[i] * wts[j] * n[k] * t * da
    return loads


def resultants(mesh: Hex20Mesh, loads: dict[int, FloatArray], origin: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Force sum and moment sum about ``origin`` of nodal loads (global components)."""
    force = np.zeros(3)
    moment = np.zeros(3)
    for node, f in loads.items():
        force += f
        moment += np.cross(mesh.coords[node - 1] - origin, f)
    return force, moment


MAX_COLUMNS = 80  # manual 1.3: nothing beyond column 80 is read


def _guarded(lines: list[str]) -> str:
    too_long = [ln for ln in lines if len(ln) > MAX_COLUMNS]
    if too_long:
        raise ValueError(f"generated input line exceeds {MAX_COLUMNS} columns: {too_long[0]!r}")
    return "\n".join(lines) + "\n"


def write_nodal_load_pattern(name: str, loads: dict[int, FloatArray], tol: float = 0.0) -> str:
    """A loading pattern of nodal forces (manual 2.8.2): one line per node and component (a node may repeat;
    nodal loads are additive), so no line approaches the column limit."""
    lines = ["*echo off", f"loading {name}", "  nodal loads"]
    for node in sorted(loads):
        for c, v in zip("xyz", loads[node], strict=True):
            if abs(v) > tol:
                lines.append(f"    {node:7d} force_{c} {v: .10E}")
    return _guarded(lines)
