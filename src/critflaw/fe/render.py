"""Rendered mesh views (theory manual, section 6.1.7, mesh layer): the crack plane ``z = 0`` (front, tube, blend,
core and outer block) and the symmetry plane ``x = 0`` (through-thickness section at the deepest point), as
PNG files. Requires the ``plot`` extra (matplotlib); import is deferred so the solver path never needs it."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from critflaw.fe import crack_mesh as CM
from critflaw.fe import mesh as M

# the 12 corner edges of a hex (corner-node index pairs in the q3disop ordering, nodes 1..8)
_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def plane_edges(mesh: M.Hex20Mesh, axis: int, value: float, tol: float = 1e-9) -> list[tuple[np.ndarray, np.ndarray]]:
    """Corner edges of every element face lying in the plane ``coords[axis] == value`` (both ends in the plane),
    as pairs of 3-vectors; shared edges appear once."""
    on = np.abs(mesh.coords[:, axis] - value) <= tol * max(float(np.ptp(mesh.coords)), 1.0)
    seen: set[tuple[int, int]] = set()
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for e in range(mesh.n_elements):
        corners = mesh.incid[e, :8] - 1
        for i, j in _EDGES:
            a, b = int(corners[i]), int(corners[j])
            if a == b or not (on[a] and on[b]):
                continue
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            out.append((mesh.coords[a], mesh.coords[b]))
    return out


def render_views(cb: CM.CrackBlockMesh, directory: Path, stem: str = "mesh") -> tuple[Path, Path]:
    """Write ``<stem>_crack_plane.png`` (z = 0, the x–y plane with the front drawn) and ``<stem>_symmetry.png``
    (x = 0, the y–z plane) and return their paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    directory.mkdir(parents=True, exist_ok=True)
    p = cb.params
    out = []
    views = ((2, (0, 1), "crack_plane", "x (mm)", "y (mm)"), (0, (1, 2), "symmetry", "y (mm)", "z (mm)"))
    for axis, keep, name, xlabel, ylabel in views:
        segs = [(a[list(keep)], b[list(keep)]) for a, b in plane_edges(cb.mesh, axis, 0.0)]
        fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
        ax.add_collection(LineCollection(segs, colors="0.25", linewidths=0.4))
        if name == "crack_plane":
            phi = np.linspace(0.0, np.pi / 2, 200)
            ax.plot(p.c * np.cos(phi), p.a * np.sin(phi), color="crimson", linewidth=1.2, label="crack front")
            ax.legend(loc="upper right")
        pts = np.concatenate([np.stack(s) for s in segs]) if segs else np.zeros((1, 2))
        ax.set_xlim(pts[:, 0].min(), pts[:, 0].max())
        ax.set_ylim(pts[:, 1].min(), pts[:, 1].max())
        ax.set_aspect("equal")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{p.topology_id}  a={p.a:g} c={p.c:g} B={p.B:g} W={p.W:g} (mm)", fontsize=8)
        path = directory / f"{stem}_{name}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        out.append(path)
    return out[0], out[1]
