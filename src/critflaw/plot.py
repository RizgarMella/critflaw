"""Failure assessment diagram (FAD) graphics: assessment lines, assessment points and load rays, load or growth
paths, probability-density fields of uncertain assessment points with highest-density-region contours, and the
exceedance field of an uncertain assessment line.

Requires matplotlib (the ``plot`` extra). Every function draws on a given ``Axes`` (or a new one) and returns it.
Invalid realisations (NaN Lr or Kr, outside a solution's validity) are never silently dropped: density and
scatter functions exclude them from what they draw and report how many in the legend.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from critflaw.assessment import Result

FloatArray = NDArray[np.float64]

SAFE = "#2a9d8f"
FAIL = "#e76f51"
LINE = "#264653"
POINT = "#e9c46a"
DENSITY_CMAP = "magma_r"
DEFAULT_MASSES = (0.5, 0.9, 0.99)


class AssessmentLine(Protocol):
    """Any assessment line: callable on Lr, with an effective plastic-collapse cut-off ``lr_max``."""

    @property
    def lr_max(self) -> FloatArray: ...

    def __call__(self, lr: ArrayLike) -> FloatArray: ...


def _plt() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "critflaw.plot needs matplotlib: install the 'plot' extra (pip install critflaw[plot])"
        ) from exc
    return plt


def _axes(ax: Axes | None, figsize: tuple[float, float] = (7.0, 5.0)) -> Axes:
    if ax is not None:
        return ax
    _, new = _plt().subplots(figsize=figsize)
    return new  # type: ignore[no-any-return]


def _scalar(v: ArrayLike) -> float:
    a = np.asarray(v, dtype=float)
    if a.size != 1:
        raise ValueError("an assessment line with array properties cannot be drawn as one line; draw it per sample")
    return float(a.reshape(()))


def line_samples(line: AssessmentLine, lr: ArrayLike) -> FloatArray:
    """The line's ``f(Lr)`` on a grid, with ``f = 0`` at and beyond the cut-off (where nothing is acceptable)."""
    grid = np.asarray(lr, dtype=float)
    f = np.asarray(line(np.minimum(grid, _scalar(line.lr_max))), dtype=float)
    return np.where(grid < _scalar(line.lr_max), f, 0.0)


def fad(
    lines: AssessmentLine | Mapping[str, AssessmentLine],
    ax: Axes | None = None,
    *,
    lr_extent: float | None = None,
    kr_extent: float = 1.25,
    shade: bool = True,
    title: str | None = "Failure assessment diagram",
) -> Axes:
    """Draw one or more assessment lines with their cut-offs; shade the acceptable and unacceptable regions of the
    first line. ``lines`` is a line or a mapping {label: line}."""
    ax = _axes(ax)
    named = dict(lines) if isinstance(lines, Mapping) else {"assessment line": lines}
    first = next(iter(named.values()))
    lr_max = _scalar(first.lr_max)
    extent = lr_extent if lr_extent is not None else max(1.35, 1.12 * lr_max)
    styles = ("-", "-.", "--", ":")
    for i, (label, line) in enumerate(named.items()):
        cut = _scalar(line.lr_max)
        grid = np.linspace(0.0, cut, 500)
        f = np.asarray(line(grid), dtype=float)
        x = np.append(grid, cut)
        y = np.append(f, 0.0)  # the vertical drop at the cut-off
        ax.plot(x, y, color=LINE, lw=2.2 if i == 0 else 1.5, ls=styles[i % len(styles)], label=label, zorder=4)
    if shade:
        grid = np.linspace(0.0, extent, 600)
        f = line_samples(first, grid)
        ax.fill_between(grid, 0.0, f, color=SAFE, alpha=0.07, lw=0, zorder=0)
        ax.fill_between(grid, f, kr_extent, color=FAIL, alpha=0.07, lw=0, zorder=0)
    ax.set_xlim(0.0, extent)
    ax.set_ylim(0.0, kr_extent)
    ax.set_xlabel("Lr  (load / plastic collapse)")
    ax.set_ylabel("Kr  (crack driving force / toughness)")
    if title:
        ax.set_title(title)
    return ax


def point(
    result: Result,
    ax: Axes | None = None,
    *,
    ray: bool = True,
    label: str = "assessment point",
    color: str = POINT,
) -> Axes:
    """Draw a deterministic assessment point (governing crack-front position) and, with ``ray``, the proportional
    load ray to the line at the radial margin. Acceptance is the result's direct test, shown in the legend."""
    ax = _axes(ax)
    lr = np.atleast_1d(np.asarray(result.lr, dtype=float))
    kr = np.atleast_1d(np.asarray(result.kr, dtype=float))
    ok = np.atleast_1d(np.asarray(result.acceptable, dtype=bool))
    for i in range(lr.size):
        verdict = "acceptable" if ok[i] else "not acceptable"
        ax.plot(
            lr[i], kr[i], "o", ms=9, color=color, mec=LINE, zorder=6, label=f"{label} ({verdict})" if i == 0 else None
        )
        if ray:
            m = float(np.atleast_1d(np.asarray(result.radial_margin, dtype=float))[i])
            if np.isfinite(m):
                ax.plot([0.0, lr[i] * m], [0.0, kr[i] * m], color="0.45", ls="--", lw=1, zorder=3)
                ax.plot(lr[i] * m, kr[i] * m, "x", ms=9, mew=2, color=FAIL, zorder=6,
                        label=f"on the line (load x{m:.2f})" if i == 0 else None)  # fmt: skip
    return ax


def path(
    lr: ArrayLike,
    kr: ArrayLike,
    ax: Axes | None = None,
    *,
    values: ArrayLike | None = None,
    cmap: str = "viridis",
    label: str | None = None,
    colorbar_label: str | None = None,
    lw: float = 2.0,
) -> Axes:
    """Draw a path of assessment points (a load sweep, a fatigue-growth history), optionally coloured by a
    parameter such as time. NaN points (outside validity) break the path."""
    from matplotlib.collections import LineCollection

    ax = _axes(ax)
    x = np.asarray(lr, dtype=float).ravel()
    y = np.asarray(kr, dtype=float).ravel()
    if values is None:
        ax.plot(x, y, color=LINE, lw=lw, label=label, zorder=5)
        return ax
    v = np.asarray(values, dtype=float).ravel()
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    keep = np.all(np.isfinite(segs), axis=(1, 2))
    lc = LineCollection(list(segs[keep]), cmap=cmap, lw=lw, zorder=5)
    lc.set_array(0.5 * (v[:-1] + v[1:])[keep])
    ax.add_collection(lc)
    if colorbar_label is not None:
        ax.figure.colorbar(lc, ax=ax, label=colorbar_label)
    if label is not None:
        ax.plot([], [], color=_plt().get_cmap(cmap)(0.7), lw=lw, label=label)
    return ax


def _density_grid(
    lr: FloatArray,
    kr: FloatArray,
    weights: FloatArray | None,
    extent: tuple[float, float, float, float],
    bins: int,
    smooth: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    from scipy.ndimage import gaussian_filter

    x_edges = np.linspace(extent[0], extent[1], bins + 1)
    y_edges = np.linspace(extent[2], extent[3], bins + 1)
    h, _, _ = np.histogram2d(lr, kr, bins=(x_edges, y_edges), weights=weights)
    h = gaussian_filter(h, smooth, mode="constant")
    cell = (x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0])
    total = h.sum()
    dens = h / (total * cell) if total > 0 else h
    return 0.5 * (x_edges[:-1] + x_edges[1:]), 0.5 * (y_edges[:-1] + y_edges[1:]), dens.T


def hdr_levels(density: ArrayLike, masses: Sequence[float] = DEFAULT_MASSES) -> FloatArray:
    """Density thresholds of the highest-density regions holding the given probability masses (on a uniform grid):
    the region ``density >= level`` contains that share of the probability."""
    d = np.sort(np.asarray(density, dtype=float).ravel())[::-1]
    cum = np.cumsum(d)
    if cum[-1] <= 0:
        return np.full(len(masses), np.nan)
    cum /= cum[-1]
    return np.array([d[min(int(np.searchsorted(cum, m)), d.size - 1)] for m in masses])


def density(
    lr: ArrayLike,
    kr: ArrayLike,
    ax: Axes | None = None,
    *,
    weights: ArrayLike | None = None,
    bins: int = 160,
    smooth: float = 1.5,
    masses: Sequence[float] = DEFAULT_MASSES,
    log: bool = True,
    cmap: str = DENSITY_CMAP,
    colorbar: bool = True,
    fill: bool = True,
    contour_color: str = "white",
    floor: float = 1e-3,
    label: str = "realisations",
) -> Axes:
    """Draw the probability density of uncertain assessment points (Lr, Kr) as a field, with contours of the
    highest-density regions holding ``masses`` of the probability. ``weights`` (e.g. importance-sampling weights)
    are respected. ``fill=False`` draws the contours only; cells below ``floor`` times the peak density are left
    blank (the thin tail; it stays inside the contours' complement). NaN points are excluded from the field and
    counted in the legend."""
    from matplotlib.colors import LogNorm, Normalize

    ax = _axes(ax)
    x = np.asarray(lr, dtype=float).ravel()
    y = np.asarray(kr, dtype=float).ravel()
    w = None if weights is None else np.asarray(weights, dtype=float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    n_bad = int((~ok).sum())
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    gx, gy, dens = _density_grid(x[ok], y[ok], None if w is None else w[ok], (x0, x1, y0, y1), bins, smooth)
    positive = dens[dens > 0]
    if positive.size == 0:
        return ax
    if log:
        lo = max(float(positive.max()) * floor, float(positive.min()))
        norm: Any = LogNorm(vmin=lo, vmax=float(positive.max()))
        shown = np.ma.masked_less(dens, lo)
    else:
        norm = Normalize(vmin=0.0, vmax=float(positive.max()))
        shown = np.ma.masked_less_equal(dens, 0.0)
    if fill:
        mesh = ax.pcolormesh(gx, gy, shown, cmap=cmap, norm=norm, shading="auto", zorder=1, alpha=0.9)
        if colorbar:
            ax.figure.colorbar(mesh, ax=ax, label="probability density of (Lr, Kr)")
    levels = hdr_levels(dens, masses)
    order = np.argsort(levels)
    if np.all(np.isfinite(levels)) and len(set(levels.tolist())) == len(levels):
        cs = ax.contour(gx, gy, dens, levels=levels[order], colors=contour_color, linewidths=1.1, zorder=2)
        ax.clabel(cs, fmt={levels[i]: f"{masses[i]:.0%}" for i in order}, fontsize=7)
    text = f"{label}: {int(ok.sum()):,}" + (f" ({n_bad:,} outside validity, not drawn)" if n_bad else "")
    marker = ("s", _plt().get_cmap(cmap)(0.6)) if fill else ("-", contour_color)
    ax.plot([], [], marker[0], color=marker[1], label=text + ("" if fill else " (50/90/99 % contours)"))
    return ax


def outcomes(
    lr: ArrayLike, kr: ArrayLike, failed: ArrayLike, ax: Axes | None = None, *, max_safe: int = 4000, seed: int = 0
) -> Axes:
    """Scatter safe and failed realisations (every failure is drawn; safe points are thinned to ``max_safe``)."""
    ax = _axes(ax)
    x = np.asarray(lr, dtype=float).ravel()
    y = np.asarray(kr, dtype=float).ravel()
    f = np.asarray(failed, dtype=bool).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    safe = np.flatnonzero(ok & ~f)
    if safe.size > max_safe:
        safe = np.random.default_rng(seed).choice(safe, max_safe, replace=False)
    ax.scatter(x[safe], y[safe], s=3, color=SAFE, alpha=0.35, lw=0, zorder=3, label="safe (thinned)")
    ax.scatter(x[ok & f], y[ok & f], s=14, color=FAIL, edgecolor="white", lw=0.4, zorder=5,
               label=f"failed ({int((ok & f).sum()):,})")  # fmt: skip
    return ax


def exceedance_field(
    lines: Sequence[AssessmentLine],
    ax: Axes | None = None,
    *,
    bins: int = 200,
    cmap: str = "RdYlGn_r",
    colorbar: bool = True,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
) -> Axes:
    """For an uncertain assessment line (samples of the line, e.g. from scattered tensile properties), draw the
    probability that a point at (Lr, Kr) is not acceptable, ``P(Kr >= f(Lr) or Lr >= Lr,max)``, as a field, with
    quantile lines of ``f``."""
    ax = _axes(ax)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    gx = np.linspace(x0, x1, bins)
    gy = np.linspace(y0, y1, bins)
    f = np.vstack([line_samples(line, gx) for line in lines])  # (n_lines, n_lr)
    p = (gy[None, :, None] >= f[:, None, :]).mean(axis=0)  # (n_kr, n_lr)
    mesh = ax.pcolormesh(gx, gy, p, cmap=cmap, vmin=0.0, vmax=1.0, shading="auto", zorder=0, alpha=0.75)
    if colorbar:
        ax.figure.colorbar(mesh, ax=ax, label="P(point not acceptable | line uncertainty)")
    for q in quantiles:
        ax.plot(gx, np.quantile(f, q, axis=0), color=LINE, lw=1.0 if q != 0.5 else 2.0,
                ls="--" if q != 0.5 else "-", zorder=4, label=f"line, {q:.0%} quantile")  # fmt: skip
    return ax


def small_multiples(
    panels: Sequence[tuple[str, Callable[[Axes], object]]],
    ncols: int = 3,
    size: tuple[float, float] = (4.2, 3.6),
) -> Any:
    """A figure of FAD panels: ``panels`` is a list of (title, draw(ax)) pairs. Returns the figure."""
    plt = _plt()
    n = len(panels)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size[0] * ncols, size[1] * nrows), squeeze=False)
    for ax, (title, draw) in zip(axes.ravel(), panels, strict=False):
        draw(ax)
        ax.set_title(title)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


__all__ = [
    "AssessmentLine",
    "density",
    "exceedance_field",
    "fad",
    "hdr_levels",
    "line_samples",
    "outcomes",
    "path",
    "point",
    "small_multiples",
]
