"""Stress intensity factor solutions for flaws in cylinders (theory manual, section 2.4)
[provenance: k.cylinder.bulging_*, k.cylinder.tables.*, k.plate.extended, k.cylinder.extended_external].

Two routes, chosen explicitly by the user (``k_route``):
``"pressure_only"``: the flat-plate factors with a bulging factor (axial through-wall and axial surface
flaws; 1 otherwise) and the plate finite-width factor with W the cylinder length;
``"tabulated"``: the tabulated factors (``cylinder_tables``) with multilinear interpolation on the
printed axes, M = f_w = 1, a corner envelope per evaluation and a per-sample "suspect" status.

Symbols: a flaw depth (part-wall) or half-length (through-wall), c half-length, B wall, r_i, r_m = r_i + B/2,
nu Poisson's ratio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from critflaw.solutions import cylinder_tables as tables
from critflaw.solutions.plate import BoolArray, FloatArray, _arrays

Orientation = Literal["axial", "circumferential"]
Position = Literal["internal", "external", "through"]
Route = Literal["pressure_only", "tabulated"]

LAMBDA_MAX_AXIAL = 12.211
LAMBDA_MAX_CIRCUMFERENTIAL = 15.143
RM_OVER_B_MIN, RM_OVER_B_MAX = 5.0, 100.0
TABLE_A_OVER_B_MAX = 0.8
EXTENDED_PLATE_A_OVER_B_MAX = 0.6


# ---- pressure-only route: bulging factors and the extended plate solution ---------------------------


def bulging_through_axial(a: ArrayLike, r_m: ArrayLike, thickness: ArrayLike) -> FloatArray:
    """M = [1 + 3.2 a^2 / (2 r_m B)]^0.5 for an axial through-wall flaw of half-length a."""
    a_, r_, b_ = _arrays(a, r_m, thickness)
    return np.asarray(np.sqrt(1.0 + 3.2 * a_**2 / (2.0 * r_ * b_)), dtype=float)


def bulging_surface_axial(a: ArrayLike, c: ArrayLike, r_m: ArrayLike, thickness: ArrayLike) -> FloatArray:
    """M = (1 - a / (B M_T)) / (1 - a/B), M_T = [1 + 3.2 c^2 / (2 r_m B)]^0.5 for an axial surface flaw."""
    a_, c_, r_, b_ = _arrays(a, c, r_m, thickness)
    m_t = np.sqrt(1.0 + 3.2 * c_**2 / (2.0 * r_ * b_))
    return np.asarray((1.0 - a_ / (b_ * m_t)) / (1.0 - a_ / b_), dtype=float)


def shell_lambda(a: ArrayLike, r_m: ArrayLike, thickness: ArrayLike, nu: ArrayLike = 0.3) -> FloatArray:
    """lambda = [12 (1 - nu^2)]^0.25 a / sqrt(r_m B) for a through-wall flaw of half-length a."""
    a_, r_, b_, nu_ = _arrays(a, r_m, thickness, nu)
    return np.asarray((12.0 * (1.0 - nu_**2)) ** 0.25 * a_ / np.sqrt(r_ * b_), dtype=float)


def shell_beta(a: ArrayLike, r_m: ArrayLike) -> FloatArray:
    """beta = [(2 r_m / a) tan(a / (2 r_m))]^0.5, the membrane multiplier of a circumferential through-wall flaw."""
    a_, r_ = _arrays(a, r_m)
    x = a_ / (2.0 * r_)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.sqrt(np.tan(x) / x)
    return np.asarray(np.where(x == 0.0, 1.0, beta), dtype=float)


def moment_membrane_stress(m_global: ArrayLike, r_i: ArrayLike, a: ArrayLike, thickness: ArrayLike) -> FloatArray:
    """P_mb = 4 M (r_i + a) / (pi [(r_i + B)^4 - r_i^4]): the membrane stress a global moment adds at a
    circumferential internal surface flaw of depth a (at the position farthest from the bending axis)."""
    m_, r_, a_, b_ = _arrays(m_global, r_i, a, thickness)
    return np.asarray(4.0 * m_ * (r_ + a_) / (np.pi * ((r_ + b_) ** 4 - r_**4)), dtype=float)


def extended_plate_mm_mb(a: ArrayLike, thickness: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Extended (full-width) surface flaw in a plate, a/B <= 0.6, f_w = 1:
    Mm = 1.12 - 0.23 x + 10.6 x^2 - 21.7 x^3 + 30.4 x^4,  Mb = 1.12 - 1.39 x + 7.32 x^2 - 13.1 x^3 + 14 x^4."""
    a_, b_ = _arrays(a, thickness)
    x = a_ / b_
    mm = 1.12 - 0.23 * x + 10.6 * x**2 - 21.7 * x**3 + 30.4 * x**4
    mb = 1.12 - 1.39 * x + 7.32 * x**2 - 13.1 * x**3 + 14.0 * x**4
    return np.asarray(mm, dtype=float), np.asarray(mb, dtype=float)


def extended_plate_valid(a: ArrayLike, thickness: ArrayLike) -> BoolArray:
    a_, b_ = _arrays(a, thickness)
    return np.asarray((a_ > 0.0) & (a_ / b_ <= EXTENDED_PLATE_A_OVER_B_MAX), dtype=bool)


def fully_circumferential_external_mm(a: ArrayLike, r_i: ArrayLike, r_o: ArrayLike) -> FloatArray:
    """Mm = Mb for a fully circumferential external surface flaw (membrane loading only; bending stresses are
    added to the membrane ones by the source's note), M = f_w = 1:
    Mm = (1 - l^2) / {[1 - (1 - l) mu]^2 - l^2}
         x {0.8 + ((1 - l) mu / (1 - (1 - l) mu)) [4 + 1.08 l / ((1 - l)(1 - mu))]}^-0.5,
    l = r_i / r_o, mu = a/B with B = r_o - r_i."""
    a_, ri, ro = _arrays(a, r_i, r_o)
    lam = ri / ro
    mu = a_ / (ro - ri)
    y = (1.0 - lam) * mu
    first = (1.0 - lam**2) / ((1.0 - y) ** 2 - lam**2)
    second = 0.8 + (y / (1.0 - y)) * (4.0 + 1.08 * lam / ((1.0 - lam) * (1.0 - mu)))
    return np.asarray(first * second**-0.5, dtype=float)


# ---- tabulated route: multilinear interpolation with corner envelope and suspect status --------------


@dataclass(frozen=True)
class TableValue:
    value: FloatArray  # multilinear interpolant
    lower: FloatArray  # corner envelope: bound under within-cell monotonicity (checked at the nodes, not proved)
    upper: FloatArray
    valid: BoolArray  # inside the printed grid (no extrapolation)
    suspect: BoolArray  # the containing cell touches a cell the transcription or the diff flags


def _cell(axis: FloatArray, x: FloatArray) -> tuple[np.ndarray, FloatArray, BoolArray]:
    """Index of the lower node of the containing cell (in the stored, possibly descending, order), the
    fraction within it, and the in-range flag."""
    descending = axis[0] > axis[-1]
    ax = axis[::-1] if descending else axis
    tol = 1e-12 * max(abs(float(ax[0])), abs(float(ax[-1])), 1e-300)  # rounding at an edge is not extrapolation
    inside = (x >= ax[0] - tol) & (x <= ax[-1] + tol)
    xi = np.clip(x, ax[0], ax[-1])
    i = np.clip(np.searchsorted(ax, xi, side="right") - 1, 0, ax.size - 2)
    frac = (xi - ax[i]) / (ax[i + 1] - ax[i])
    if descending:  # map back to stored indices: stored j = n - 1 - ascending j
        i = ax.size - 2 - i
        frac = 1.0 - frac
    return i, np.asarray(frac, dtype=float), np.asarray(inside, dtype=bool)


def _suspect_cells(table: tables.Table, quantity: int, idx: list[np.ndarray], shape: tuple[int, ...]) -> BoolArray:
    suspect = np.zeros(shape, dtype=bool)
    for table_name, quantity_name, cell in tables.SUSPECT:
        if table_name != table.name or quantity_name != table.quantities[quantity]:
            continue
        hit = np.ones(shape, dtype=bool)
        for j in range(len(idx)):
            hit &= (idx[j] == cell[j]) | (idx[j] + 1 == cell[j])
        suspect |= hit
    return suspect


def interpolate(table: tables.Table, quantity: int, *coords: ArrayLike) -> TableValue:
    """Multilinear interpolation of one quantity of a full-grid table (M.2 to M.5, M.7, M.8) at the given
    coordinates (axis order of the table). Outside the grid: ``valid`` False and NaN (never extrapolated).
    """
    axes = [np.asarray(ax[1], dtype=float) for ax in table.axes]
    if len(coords) != len(axes):
        raise ValueError(f"{table.name} takes {len(axes)} coordinates ({', '.join(ax[0] for ax in table.axes)})")
    values = np.asarray(table.values[quantity], dtype=float)
    if np.any(np.isnan(values)):
        raise ValueError(f"{table.name} is not a full grid; use interpolate_columns")
    xs = _arrays(*coords)
    shape = xs[0].shape
    idx, frac, inside = [], [], np.ones(shape, dtype=bool)
    for axis, x in zip(axes, xs, strict=True):
        i, f, ok = _cell(axis, x)
        idx.append(i)
        frac.append(f)
        inside &= ok
    n_ax = len(axes)
    corners = np.empty((2**n_ax, *shape))
    weights = np.empty((2**n_ax, *shape))
    for k in range(2**n_ax):
        bits = [(k >> j) & 1 for j in range(n_ax)]
        corners[k] = values[tuple(idx[j] + bits[j] for j in range(n_ax))]
        w = np.ones(shape)
        for j in range(n_ax):
            w = w * (frac[j] if bits[j] else 1.0 - frac[j])
        weights[k] = w
    value = np.sum(corners * weights, axis=0)
    nan = np.full(shape, np.nan)
    return TableValue(
        value=np.asarray(np.where(inside, value, nan), dtype=float),
        lower=np.asarray(np.where(inside, np.min(corners, axis=0), nan), dtype=float),
        upper=np.asarray(np.where(inside, np.max(corners, axis=0), nan), dtype=float),
        valid=np.asarray(inside, dtype=bool),
        suspect=np.asarray(_suspect_cells(table, quantity, idx, shape) & inside, dtype=bool),
    )


def interpolate_columns(table: tables.Table, quantity: int, lam: ArrayLike, rm_over_b: ArrayLike) -> TableValue:
    """Interpolation of the sparse-column tables M.1a-d and M.6a-d: each r_m/B column is printed at its own
    lambda values, so the value is interpolated linearly in lambda within each of the two bracketing columns
    (over that column's printed points) and then linearly in r_m/B. A query beyond a column's last printed
    lambda is outside validity; a query inside the printed gap of a column (M.1, r_m/B = 10) is interpolated
    across the gap and flagged ``suspect`` (the printed value after the gap is a flagged cell). The envelope
    is the min/max of the four bracketing printed values.
    """
    lam_axis = np.asarray(table.axes[0][1], dtype=float)
    col_axis = np.asarray(table.axes[1][1], dtype=float)
    values = np.asarray(table.values[quantity], dtype=float)
    x, r = _arrays(lam, rm_over_b)
    shape = x.shape
    flagged = {cell for name, q, cell in tables.SUSPECT if name == table.name and q == table.quantities[quantity]}
    j, fr, inside_r = _cell(col_axis, r)
    value_cols, lo_cols, hi_cols, ok_cols, sus_cols = [], [], [], [], []
    for side in (0, 1):
        col_index = j + side
        out_v = np.full(shape, np.nan)
        out_lo = np.full(shape, np.nan)
        out_hi = np.full(shape, np.nan)
        out_ok = np.zeros(shape, dtype=bool)
        out_sus = np.zeros(shape, dtype=bool)
        for jc in range(col_axis.size):
            sel = col_index == jc
            if not np.any(sel):
                continue
            rows = np.nonzero(~np.isnan(values[:, jc]))[0]
            lam_c, val_c = lam_axis[rows], values[rows, jc]
            xi = x[sel]
            tol = 1e-12 * max(abs(float(lam_c[-1])), 1e-300)
            ok = (xi >= lam_c[0] - tol) & (xi <= lam_c[-1] + tol)
            xc = np.clip(xi, lam_c[0], lam_c[-1])
            i = np.clip(np.searchsorted(lam_c, xc, side="right") - 1, 0, lam_c.size - 2)
            f = (xc - lam_c[i]) / (lam_c[i + 1] - lam_c[i])
            v0, v1 = val_c[i], val_c[i + 1]
            out_v[sel] = v0 + f * (v1 - v0)
            # on a printed node (f = 0 or 1) the value is the printed one: envelope and flag of that node only
            on0, on1 = f <= 0.0, f >= 1.0
            out_lo[sel] = np.where(on0, v0, np.where(on1, v1, np.minimum(v0, v1)))
            out_hi[sel] = np.where(on0, v0, np.where(on1, v1, np.maximum(v0, v1)))
            out_ok[sel] = ok
            # a bracketing printed value that the transcription or the diff flags (incl. the cell after a gap)
            flagged_rows = np.array([int((int(row), jc) in flagged) for row in rows], dtype=bool)
            out_sus[sel] = (flagged_rows[i] & ~on1) | (flagged_rows[i + 1] & ~on0)
        value_cols.append(out_v)
        lo_cols.append(out_lo)
        hi_cols.append(out_hi)
        ok_cols.append(out_ok)
        sus_cols.append(out_sus)
    # a bracketing column with zero weight (query exactly on the other column) does not constrain
    use0, use1 = fr < 1.0, fr > 0.0
    inside = inside_r & (ok_cols[0] | ~use0) & (ok_cols[1] | ~use1)
    v0, v1 = np.where(use0, value_cols[0], 0.0), np.where(use1, value_cols[1], 0.0)
    value = np.where(use0, v0, 0.0) * (1.0 - fr) + np.where(use1, v1, 0.0) * fr
    lower = np.minimum(np.where(use0, lo_cols[0], np.inf), np.where(use1, lo_cols[1], np.inf))
    upper = np.maximum(np.where(use0, hi_cols[0], -np.inf), np.where(use1, hi_cols[1], -np.inf))
    suspect = (sus_cols[0] & (fr < 1.0)) | (sus_cols[1] & (fr > 0.0))
    nan = np.full(shape, np.nan)
    return TableValue(
        value=np.asarray(np.where(inside, value, nan), dtype=float),
        lower=np.asarray(np.where(inside, lower, nan), dtype=float),
        upper=np.asarray(np.where(inside, upper, nan), dtype=float),
        valid=np.asarray(inside, dtype=bool),
        suspect=np.asarray(suspect & inside, dtype=bool),
    )


def lambda_range_end(table: tables.Table, rm_over_b: ArrayLike) -> FloatArray:
    """The largest lambda at which a sparse-column table (M.1, M.6) can be interpolated at r_m/B: the smaller of
    the last printed lambdas of the two bracketing columns (one column when the query sits on it)."""
    lam_axis = np.asarray(table.axes[0][1], dtype=float)
    col_axis = np.asarray(table.axes[1][1], dtype=float)
    values = np.asarray(table.values[0], dtype=float)
    last = np.array([lam_axis[np.nonzero(~np.isnan(values[:, j]))[0][-1]] for j in range(col_axis.size)])
    (r,) = _arrays(rm_over_b)
    j, fr, inside = _cell(col_axis, r)
    end0 = last[j]
    end1 = last[np.minimum(j + 1, col_axis.size - 1)]
    end = np.minimum(np.where(fr < 1.0, end0, np.inf), np.where(fr > 0.0, end1, np.inf))
    return np.asarray(np.where(inside, end, np.nan), dtype=float)


def table_monotonicity(table: tables.Table, quantity: int, axis: int) -> tuple[int, int, int]:
    """(non-decreasing steps, non-increasing steps, total steps) along one axis, NaN steps skipped."""
    v = np.moveaxis(np.asarray(table.values[quantity], dtype=float), axis, 0)
    d = np.diff(v, axis=0)
    finite = np.isfinite(d)
    return int(np.sum((d >= 0.0) & finite)), int(np.sum((d <= 0.0) & finite)), int(np.sum(finite))
