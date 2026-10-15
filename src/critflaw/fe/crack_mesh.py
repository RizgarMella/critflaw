"""Parametric crack-block mesher for a semi-elliptical surface flaw in a flat plate (theory manual,
section 6.1.3). A 2-D focused template in the local (s, z) plane normal to the crack front — ``n_rings`` rings of
collapsed-wedge hex20 elements around the front over the half-plane z >= 0 (theta = 0 on the ligament ray
s > 0, theta = pi on the crack-face ray s < 0), then ``n_blend`` radial layers out to a square of half-side
``R_b`` — swept along the elliptical front at fixed parametric angles (``n_front_s`` intervals on [0, phi_s],
``n_front - n_front_s`` on [phi_s, pi/2], a node exactly at ``phi_s``). The crack-plane footprint inside the
swept block (the core) is an O-grid (a central rectangle and two mapped blocks to the inner offset curve), the
footprint outside it a mapped block to the plate edges; both are extruded over the block's own z stations up to
``R_b``; the whole footprint at z = ``R_b`` is then extruded in graded layers to ``L_far``. Nodes are merged by
coordinates, so every region ties to its neighbours without bookkeeping; the collapsed front keeps one node per
station (``tied``, the elastic mesh, quarter-point mid-sides on the ring-1 radial edges) or one per sector
(``untied``, the plastic mesh).

Quarter model. x along the width (symmetry plane x = 0 through the flaw centre), y through the thickness (free
surface y = 0, back face y = B), z normal to the crack plane (crack plane z = 0: the crack face inside the
ellipse x^2/c^2 + y^2/a^2 < 1 is free, the ligament outside it has w = 0; the far end z = L_far is loaded).
Front: (c cos phi, a sin phi), phi = 0 at the surface point, pi/2 at the deepest point.

Tube radius rule: r_t = min(k1 a^2/c, k2 (B - a), k3 a); R_b = blend_factor r_t. A mesh is a pure function of
its parameters (deterministic numbering).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from critflaw.fe.mesh import Hex20Mesh, elements_with_face_on_plane

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class CrackBlockParams:
    a: float
    c: float
    B: float
    W: float  # full plate width (the model spans x in [0, W/2])
    L_far: float
    n_rings: int = 6  # defaults: the production level_1 = (6, 12, 12, 3, 3) of the D7 mesh levels
    n_sectors: int = 12  # over the half plane theta in [0, pi]; a multiple of 4 (square corners on rays pi/4, 3pi/4)
    n_front: int = 12  # front intervals in total
    n_front_s: int = 3  # of which on [0, phi_s]
    phi_s: float = float(np.deg2rad(10.0))  # parametric near-surface angle
    n_blend: int = 3
    n_core: int = 3  # divisions from the inner offset curve to the core rectangle
    n_outer: int = 10  # divisions from the outer offset curve to the plate edges (D7b: within 0.43 % of 20)
    n_far: int = 6  # z layers above the block to L_far
    k1: float = 0.5
    k2: float = 0.5
    k3: float = 0.1
    blend_factor: float = 1.5
    ring_grading: float = 1.6  # ring radii r_i = r_t (i/n)^p
    far_grading: float = 1.4  # geometric growth of the z layers above the block
    far_first: float = 1.0  # > 0: first far layer = far_first x the blend thickness (R_b - r_t), growth <= far_grading,
    # the layer count derived to reach L_far (n_far ignored); 0: n_far layers graded by far_grading (legacy)
    tip: Literal["tied", "untied"] = "tied"

    def __post_init__(self) -> None:
        if not (0.0 < self.a < self.B and self.c > 0.0 and 2.0 * self.c < self.W and self.L_far > 0.0):
            raise ValueError("need 0 < a < B, c > 0, 2c < W, L_far > 0")
        if self.n_rings < 6 or self.n_sectors < 8 or self.n_sectors % 4 or self.n_front_s < 1:
            raise ValueError("need n_rings >= 6, n_sectors a multiple of 4 and >= 8, n_front_s >= 1")
        if self.n_front <= self.n_front_s + 1:
            raise ValueError("need n_front > n_front_s + 1")
        if not (0.0 < self.phi_s < np.pi / 2):
            raise ValueError("phi_s must lie in (0, pi/2)")
        if self.far_first < 0.0 or self.far_grading <= 1.0:
            raise ValueError("need far_first >= 0 and far_grading > 1")

    @property
    def r_t(self) -> float:
        return min(self.k1 * self.a**2 / self.c, self.k2 * (self.B - self.a), self.k3 * self.a)

    @property
    def R_b(self) -> float:
        return self.blend_factor * self.r_t

    @property
    def phi_stations(self) -> FloatArray:
        """Corner-node parametric angles (``n_front + 1`` values), a node exactly at ``phi_s``."""
        first = np.linspace(0.0, self.phi_s, self.n_front_s + 1)
        second = np.linspace(self.phi_s, np.pi / 2, self.n_front - self.n_front_s + 1)[1:]
        return np.concatenate([first, second])

    @property
    def topology_id(self) -> str:
        key = (
            f"{self.n_rings}-{self.n_sectors}-{self.n_front}-{self.n_front_s}-{self.n_blend}-{self.n_core}-"
            f"{self.n_outer}-{self.n_far}-{self.tip}" + (f"-ff{self.far_first:g}" if self.far_first > 0.0 else "")
        )
        return "plate_surface_flaw_v1:" + key


# ---------------------------------------------------------------- front geometry


def remote_length(B: float, W: float) -> float:
    """The remote length L_far: the larger of 20 B and 2 W. D7b at the mid-box point: 5 B was short of 10 B by
    0.9 % (elastic) and 8 % (strain-capped last step); 10 B of 20 B by 0.06 % / 0.6 %, the last figure outside
    the 0.5 % band, hence 20 B (the far layers grow geometrically, so the extra length costs three layers)."""
    return max(20.0 * B, 2.0 * W)


def far_layer_weights(p: CrackBlockParams) -> FloatArray:
    """Relative thicknesses of the z layers above the block. Legacy (``far_first == 0``): ``n_far`` layers in
    geometric progression ``far_grading``. Otherwise the first layer is ``far_first (R_b - r_t)`` (the blend's
    radial thickness, so the z size is continuous across the block top), the count is the smallest that reaches
    ``L_far`` with growth <= ``far_grading`` and the growth is then solved (bisection) so the layers end exactly
    at ``L_far``."""
    if p.far_first <= 0.0:
        return np.array([p.far_grading**m for m in range(p.n_far)])
    h0 = p.far_first * (p.R_b - p.r_t)
    height = p.L_far - p.R_b
    if height <= h0:
        return np.array([1.0])
    g = p.far_grading
    n = int(np.ceil(np.log(1.0 + height * (g - 1.0) / h0) / np.log(g)))
    n = max(n, 2)

    def total(growth: float) -> float:
        return float(h0 * (growth**n - 1.0) / (growth - 1.0)) if growth > 1.0 + 1e-12 else float(h0 * n)

    lo, hi = 1.0, g
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if total(mid) < height:
            lo = mid
        else:
            hi = mid
    growth = 0.5 * (lo + hi)
    return np.array([growth**m for m in range(n)])


def front_point(p: CrackBlockParams, phi: float) -> FloatArray:
    return np.array([p.c * np.cos(phi), p.a * np.sin(phi), 0.0])


def front_normal(p: CrackBlockParams, phi: float) -> FloatArray:
    """Outward in-plane unit normal of the ellipse at phi (points into the ligament)."""
    n = np.array([np.cos(phi) / p.c, np.sin(phi) / p.a, 0.0])
    return np.asarray(n / np.linalg.norm(n), dtype=float)


def curvature_radius(p: CrackBlockParams, phi: float) -> float:
    """Radius of curvature of the ellipse at phi (a^2/c at the surface point, c^2/a at the deepest point)."""
    num = (p.c**2 * np.sin(phi) ** 2 + p.a**2 * np.cos(phi) ** 2) ** 1.5
    return float(num / (p.a * p.c))


def validate_geometry(p: CrackBlockParams) -> None:
    """The swept block must not self-intersect on its inner side (R_b below the smallest curvature radius) and
    must fit between the free surface, the back face, the symmetry plane and the plate edge."""
    rho_min = min(curvature_radius(p, float(phi)) for phi in p.phi_stations)
    if p.R_b >= 0.9 * rho_min:
        raise ValueError(f"R_b = {p.R_b:.4g} too large for the front's curvature radius {rho_min:.4g}")
    if p.a + p.R_b >= p.B or p.R_b >= p.a or p.c + p.R_b >= p.W / 2 or p.R_b >= p.c:
        raise ValueError("the crack block does not fit inside the quarter plate")


# ---------------------------------------------------------------- 2-D template in the (s, z) plane


def _square_perimeter(theta: float, half: float) -> tuple[float, float]:
    """Point where the ray at angle theta from the front meets the square of half-side ``half`` (right side for
    theta <= pi/4, top for pi/4..3pi/4, left side beyond): the blend's radial lines stay radial, so they never
    cross or fan; the square's corners fall on the rays theta = pi/4 and 3pi/4."""
    if theta <= np.pi / 4:
        return half, half * np.tan(theta)
    if theta <= 3 * np.pi / 4:
        return half / np.tan(theta) if abs(theta - np.pi / 2) > 1e-12 else 0.0, half
    return -half, -half * np.tan(theta)


def template_2d(p: CrackBlockParams) -> FloatArray:
    """Template nodes on the doubled grid: ``sz[i, k]`` = (s, z) at radial station i (0 = front ...
    2 n_rings = outer ring ... 2 (n_rings + n_blend) = square perimeter) and angular station k (0 = ligament
    ray theta = 0 ... 2 n_sectors = crack-face ray theta = pi). Station i = 0 is the front for every k."""
    nr, nb, ns = p.n_rings, p.n_blend, p.n_sectors
    n_rad, n_ang = 2 * (nr + nb) + 1, 2 * ns + 1
    sz = np.zeros((n_rad, n_ang, 2))
    thetas = np.linspace(0.0, np.pi, n_ang)
    radii = np.zeros(2 * nr + 1)
    for j in range(nr + 1):
        radii[2 * j] = p.r_t * (j / nr) ** p.ring_grading
    for j in range(nr):
        radii[2 * j + 1] = 0.5 * (radii[2 * j] + radii[2 * j + 2])
    if p.tip == "tied":
        radii[1] = 0.25 * radii[2]  # quarter-point mid-side nodes of the ring-1 radial edges
    for i in range(2 * nr + 1):
        for k, th in enumerate(thetas):
            sz[i, k] = (radii[i] * np.cos(th), radii[i] * np.sin(th))
    for k, th in enumerate(thetas):
        start = sz[2 * nr, k]
        end = np.array(_square_perimeter(th, p.R_b))
        for m in range(1, 2 * nb + 1):
            sz[2 * nr + m, k] = start + (m / (2 * nb)) * (end - start)
    return sz


# ---------------------------------------------------------------- assembly by coordinates


class _Nodes:
    """Nodes keyed by rounded coordinates and a tag (0 for ordinary nodes; the sector index for untied front
    nodes, which must stay distinct although coincident)."""

    def __init__(self, scale: float) -> None:
        self.coords: list[FloatArray] = []
        self.index: dict[tuple[int, int, int, int], int] = {}
        self.scale = scale

    def node(self, xyz: FloatArray, tag: int = 0) -> int:
        q = [int(round(float(v) / (1e-10 * self.scale))) for v in xyz]
        key = (q[0], q[1], q[2], tag)
        if key not in self.index:
            self.index[key] = len(self.coords) + 1
            self.coords.append(np.asarray(xyz, dtype=float))
        return self.index[key]

    def lookup(self, xyz: FloatArray, tag: int = 0) -> int:
        """An existing node only (never creates one)."""
        q = [int(round(float(v) / (1e-10 * self.scale))) for v in xyz]
        key = (q[0], q[1], q[2], tag)
        if key not in self.index:
            raise KeyError(f"no node at {np.asarray(xyz)} with tag {tag}")
        return self.index[key]


_HEX20_INDEX = [
    (0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0), (0, 0, 2), (2, 0, 2), (2, 2, 2), (0, 2, 2),
    (1, 0, 0), (2, 1, 0), (1, 2, 0), (0, 1, 0), (1, 0, 2), (2, 1, 2), (1, 2, 2), (0, 1, 2),
    (0, 0, 1), (2, 0, 1), (2, 2, 1), (0, 2, 1),
]  # fmt: skip


def _element(nodes: _Nodes, grid: FloatArray, tags: NDArray[np.int64] | None = None) -> list[int]:
    """One hex20 from a (3, 3, 3, 3) block of doubled-grid coordinates (indices i, j, k), q3disop order."""
    return [nodes.node(grid[i, j, k], 0 if tags is None else int(tags[i, j, k])) for i, j, k in _HEX20_INDEX]


def _extrude_elements(nodes: _Nodes, xy: FloatArray, z: FloatArray) -> list[list[int]]:
    """Elements of a footprint given on a doubled (u, v) grid ``xy`` (n_u, n_v, 2) extruded over doubled z
    stations ``z`` (n_z,): the local axes are (u, v, z)."""
    out = []
    n_u, n_v = xy.shape[0], xy.shape[1]
    for ue in range((n_u - 1) // 2):
        for ve in range((n_v - 1) // 2):
            for ze in range((len(z) - 1) // 2):
                grid = np.zeros((3, 3, 3, 3))
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            grid[i, j, k, :2] = xy[2 * ue + i, 2 * ve + j]
                            grid[i, j, k, 2] = z[2 * ze + k]
                out.append(_element(nodes, grid))
    return out


def _arc_fraction(curve: FloatArray) -> FloatArray:
    """Cumulative arc-length fraction (0 .. 1) of the stations of a polyline."""
    seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return np.asarray(cum / cum[-1], dtype=float)


def _mapped(curve: FloatArray, target: FloatArray, n_div: int) -> FloatArray:
    """A doubled (u, v) footprint grid between a curve (n_u, 2) and a target polyline sampled at the same
    stations (n_u, 2): straight v-lines with 2 n_div + 1 stations."""
    n_u = curve.shape[0]
    out = np.zeros((n_u, 2 * n_div + 1, 2))
    for v in range(2 * n_div + 1):
        f = v / (2 * n_div)
        out[:, v] = (1.0 - f) * curve + f * target
    return out


@dataclass(frozen=True)
class CrackBlockMesh:
    mesh: Hex20Mesh
    params: CrackBlockParams
    front_nodes: dict[int, list[int]]  # doubled front station -> front node(s): one (tied) or one per angular position
    # (untied; the first is the ligament-side node theta = 0, the last the crack-face node theta = pi)
    tip_elements: IntArray  # the collapsed ring-1 elements
    block_elements: IntArray
    ligament_nodes: IntArray  # z = 0 outside the ellipse (w = 0)
    symmetry_x_nodes: IntArray  # x = 0 (u = 0)
    loaded_faces: list[tuple[int, int]]  # (element, face) on z = L_far
    phi_corner_stations: FloatArray
    topology_hash: str

    def domain_front_nodes(self, station: int) -> tuple[int, ...]:
        """WARP3D front-node list (corner, mid-side, corner, ...) for the domain centred on corner station
        ``station`` (0-based corner index): its two neighbouring corner stations and the mid-side nodes between
        (the end stations use the one-sided pair)."""
        j = 2 * station
        n = len(self.front_nodes) - 1
        lo, hi = max(j - 2, 0), min(j + 2, n)
        # listed with decreasing phi (deepest point towards the surface): WARP3D derives the crack-advance
        # direction from the node order and the normal plane; the increasing-phi order gives it pointing into the
        # crack (J comes out negative); the end tangents in elastic_case follow the same sense
        return tuple(self.front_nodes[k][0] for k in range(hi, lo - 1, -1))


def build(p: CrackBlockParams) -> CrackBlockMesh:
    validate_geometry(p)
    sz = template_2d(p)
    nodes = _Nodes(max(p.W, p.L_far, p.B))
    elements: list[list[int]] = []
    tip_elements: list[int] = []
    block_elements: list[int] = []

    phi_c = p.phi_stations
    n_phi = 2 * p.n_front + 1
    phis = np.zeros(n_phi)
    phis[::2] = phi_c
    phis[1::2] = 0.5 * (phi_c[:-1] + phi_c[1:])
    n_rad, n_ang = sz.shape[0], sz.shape[1]

    # ---- the swept block
    origins = np.asarray([front_point(p, float(phi)) for phi in phis])
    normals = np.asarray([front_normal(p, float(phi)) for phi in phis])
    block_xyz = np.zeros((n_rad, n_ang, n_phi, 3))
    for j in range(n_phi):
        for i in range(n_rad):
            for k in range(n_ang):
                s, z = sz[i, k]
                block_xyz[i, k, j] = origins[j] + s * normals[j] + np.array([0.0, 0.0, z])
    tags = np.zeros((n_rad, n_ang, n_phi), dtype=np.int64)
    if p.tip == "untied":
        for k in range(n_ang):
            # one distinct front node per angular position (corner and mid-side), shared by the two elements
            # meeting there: the keyhole-collapsed-to-a-point design (manual 4.4): no repeated incidences, every
            # tip element has its own three collapsed-edge nodes per station; k = 0 is the ligament-side node
            tags[0, k, :] = k + 1
    for ie in range(p.n_rings + p.n_blend):
        for ke in range(p.n_sectors):
            for je in range(p.n_front):
                # right-handed local axes (angular, radial, front): e_theta x e_r = the front tangent
                grid = block_xyz[2 * ie : 2 * ie + 3, 2 * ke : 2 * ke + 3, 2 * je : 2 * je + 3].transpose(1, 0, 2, 3)
                tg = tags[2 * ie : 2 * ie + 3, 2 * ke : 2 * ke + 3, 2 * je : 2 * je + 3].transpose(1, 0, 2)
                elements.append(_element(nodes, grid, tg))
                block_elements.append(len(elements))
                if ie == 0:
                    tip_elements.append(len(elements))

    # ---- z stations of the block's sides (the extrusion levels of the footprints inside the block height)
    side_k = [k for k in range(n_ang) if sz[n_rad - 1, k, 0] >= p.R_b - 1e-12]
    z_side = np.asarray([sz[n_rad - 1, k, 1] for k in side_k])
    if len(z_side) % 2 == 0 or z_side[0] != 0.0 or abs(z_side[-1] - p.R_b) > 1e-12:
        raise RuntimeError("the side station layout is not corner-mid-corner from 0 to R_b")

    # ---- core footprint: O-grid. Inner offset curve at the stations; central rectangle [0, x_r] x [0, y_r];
    # block A maps the curve on [0, phi_mid] to the rectangle's right edge (x_r, 0) -> (x_r, y_r); block B maps
    # the curve on [phi_mid, pi/2] to the top edge (x_r, y_r) -> (0, y_r); the rectangle is structured.
    inner = origins - p.R_b * normals  # (n_phi, 3)
    x_r, y_r = 0.5 * (p.c - p.R_b), 0.5 * (p.a - p.R_b)
    # the O-grid diagonal: the corner station nearest the ellipse's phi = 45 degrees (the point on the ray
    # through the rectangle corner (x_r, y_r)), so that the diagonal from the curve to the corner lies on that ray
    j_mid = 2 * int(np.argmin(np.abs(phi_c - np.pi / 4)))
    if j_mid == 0 or j_mid == n_phi - 1:
        raise ValueError("no interior corner station near phi = 45 degrees")
    n_a, n_b = j_mid + 1, n_phi - j_mid
    # targets placed by the curve's arc-length fraction (the phi stations are dense near the surface point)
    frac_a = _arc_fraction(inner[: j_mid + 1, :2])
    frac_b = _arc_fraction(inner[j_mid:, :2])
    right_edge = np.asarray([[x_r, y_r * f] for f in frac_a])
    top_edge = np.asarray([[x_r - x_r * f, y_r] for f in frac_b])
    core_a = _mapped(inner[: j_mid + 1, :2], right_edge, p.n_core)
    core_b = _mapped(inner[j_mid:, :2], top_edge, p.n_core)
    rect = np.zeros((n_b, n_a, 2))  # (u along x from x_r down to 0, v along y from 0 to y_r)
    for u in range(n_b):
        for v in range(n_a):
            rect[u, v] = (x_r - x_r * frac_b[u], y_r * frac_a[v])
    # orientation: (u along the front tangent, v inward, z) is right-handed for the two mapped core blocks; the
    # rectangle (u along -x, v along +y) and the outer block (v outward) are transposed to (v, u, z)
    for xy in (core_a, core_b, rect.transpose(1, 0, 2)):
        elements += _extrude_elements(nodes, xy, z_side)

    # ---- outer footprint: the outer offset curve mapped to the plate edges (right edge x = W/2 from y = 0,
    # then the back face y = B to x = 0), the corner (W/2, B) at the station nearest its direction
    outer = origins + p.R_b * normals
    # targets on rays from the flaw centre through the outer-curve stations (v-lines never cross); the corner
    # station whose ray is nearest the plate corner's ray gets the corner itself, the stations before it are
    # spread over the right edge and those after it over the back face by their ray angles
    # the mapped outer block ends at an intermediate box x = X_box (about 2c, never past the plate edge), so a
    # small curve is not mapped onto a very wide plate; the slab beyond it is a plain structured block
    x_box = min(p.W / 2, max(2.0 * p.c, p.c + 2.0 * (p.a + p.R_b)))
    ang = np.arctan2(outer[:, 1], outer[:, 0])
    ang_corner = np.arctan2(p.B, x_box)
    j_star = 2 * int(np.argmin(np.abs(ang[::2] - ang_corner)))
    j_star = min(max(j_star, 2), n_phi - 3)
    target = np.zeros((n_phi, 2))
    for j, th in enumerate(ang):
        if j <= j_star:
            target[j] = (x_box, p.B * th / ang[j_star])
        else:
            f = (th - ang[j_star]) / (np.pi / 2 - ang[j_star])
            target[j] = (x_box * (1.0 - f), p.B)
    outer_xy = _mapped(outer[:, :2], target, p.n_outer).transpose(1, 0, 2)
    elements += _extrude_elements(nodes, outer_xy, z_side)
    # the slab x in [X_box, W/2]: y stations = the box's right-edge targets (stations 0..j_star), x stations graded
    slab_xy: FloatArray | None = None
    if x_box < p.W / 2 - 1e-9:
        y_st = target[: j_star + 1, 1]
        n_sx = max(2, int(round((p.W / 2 - x_box) / (x_box / max(j_star // 2, 1)))))
        n_sx = min(n_sx, 6)
        x_st = np.linspace(x_box, p.W / 2, 2 * n_sx + 1)
        slab_xy = np.zeros((len(x_st), len(y_st), 2))
        for u, x in enumerate(x_st):
            for v, y in enumerate(y_st):
                slab_xy[u, v] = (x, y)
        elements += _extrude_elements(nodes, slab_xy, z_side)

    # ---- the far field: every footprint quad at z = R_b (block top strip, core, outer) extruded to L_far
    weights = far_layer_weights(p)
    z_far_c = p.R_b + np.concatenate([[0.0], np.cumsum(weights)]) / weights.sum() * (p.L_far - p.R_b)
    z_far = np.zeros(2 * len(weights) + 1)
    z_far[::2] = z_far_c
    z_far[1::2] = 0.5 * (z_far_c[:-1] + z_far_c[1:])
    top_k = sorted(
        (k for k in range(n_ang) if abs(sz[n_rad - 1, k, 1] - p.R_b) < 1e-12), key=lambda k: -sz[n_rad - 1, k, 0]
    )
    top_xy = np.zeros((n_phi, len(top_k), 2))
    for j in range(n_phi):
        for m, k in enumerate(top_k):
            top_xy[j, m] = block_xyz[n_rad - 1, k, j, :2]
    for xy in (top_xy, core_a, core_b, rect.transpose(1, 0, 2), outer_xy) + ((slab_xy,) if slab_xy is not None else ()):
        elements += _extrude_elements(nodes, xy, z_far)

    incid = np.asarray(elements, dtype=np.int64)
    coords = np.asarray(nodes.coords, dtype=float)
    mesh = Hex20Mesh(coords, incid)

    tol = 1e-9 * max(p.W, p.L_far)
    on_plane = np.abs(coords[:, 2]) <= tol
    inside = (coords[:, 0] / p.c) ** 2 + (coords[:, 1] / p.a) ** 2 < 1.0 - 1e-9
    front: dict[int, list[int]] = {}
    for j in range(n_phi):
        if p.tip == "tied":
            front[j] = [nodes.lookup(block_xyz[0, 0, j])]
        else:
            # corner stations carry every angular position; mid stations only the corner positions (the point
            # (r = 0, mid angle, mid station) is the centre of the collapsed face, not a hex20 node)
            positions = range(n_ang) if j % 2 == 0 else range(0, n_ang, 2)
            front[j] = [nodes.lookup(block_xyz[0, 0, j], k + 1) for k in positions]
    # the crack-plane constraint w = 0 on the ligament (z = 0 outside the ellipse): of the coincident untied tip
    # nodes only the ligament-side one (theta = 0) carries it (manual 4.4: the constraint is not imposed on the
    # other nodes of a collapsed keyhole, which must be free to blunt)
    free_tip = {n for j in range(n_phi) for n in front[j][1:]}
    ligament = np.asarray([n for n in np.nonzero(on_plane & ~inside)[0] + 1 if int(n) not in free_tip], dtype=np.int64)
    sym_x = np.nonzero(np.abs(coords[:, 0]) <= tol)[0] + 1
    loaded = elements_with_face_on_plane(mesh, 2, p.L_far)
    h = hashlib.sha256(incid.tobytes()).hexdigest()[:16]
    return CrackBlockMesh(
        mesh, p, front, np.asarray(tip_elements, dtype=np.int64), np.asarray(block_elements, dtype=np.int64),
        ligament, sym_x, loaded, phi_c, p.topology_id + ":" + h,
    )  # fmt: skip
