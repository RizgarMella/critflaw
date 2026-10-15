"""Tabulated stress-intensity magnification factors for flaws in cylinders.

Plain data module: the eight tables of magnification factors (Tables M.1 a)-d), M.2, M.3, M.4,
M.5, M.6 a)-d), M.7 and M.8) for through-thickness, surface and extended flaws in cylindrical
shells under combined internal pressure and mechanical loads. Every number was entered from an
independent transcription of the printed pages and re-read against it a second time; nothing here is
interpolated, smoothed or corrected. Cells that the transcription flags as odd are listed in ``SUSPECT``
and left exactly as printed.

Primary references, by bibliography number as the transcription's headings cite them:

* Tables M.1 a)-d) and M.6 a)-d): [176]
* Tables M.2, M.3, M.4: [278], [279], [280]
* Table M.5: [281]
* Table M.7: [278], [279], [282], [283]
* Table M.8: [282], [284], [285]

Symbols: a flaw depth (surface flaws) or half-length (through-thickness flaws), c surface-flaw
half-length, B wall thickness, r_i inner radius, r_m mean radius, W cylinder length, lambda the
shell parameter of the through-thickness solutions. Through-thickness tables (M.1, M.6) give
M_1..M_4; surface-flaw tables (M.2, M.4, M.7) give M_m and M_b at the deepest point (d) and the
surface point (s); extended-flaw tables (M.3, M.5, M.8) give M_m and M_b.

Layout conventions
------------------
* ``Table.values`` holds one array per entry of ``Table.quantities``; each array is indexed in the
  order of ``Table.axes``.
* Tables M.1 and M.6 are stored on the full printed grid (all printed lambda rows by the five
  r_m/B columns); cells that are blank in print are NaN. The raw per-column rows are kept as
  typed (``_M1_COLUMNS``, ``_M6_COLUMNS``) so the grid can be audited against the transcription.
* Tables M.2, M.4 and M.7 are indexed (a/c, a/B, B/r_i); M.3 and M.8 (a/B, B/r_i); M.5 (B/r_i, a/B),
  i.e. in each case the reading order of the printed table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

Axis = tuple[str, tuple[float, ...]]
Row = tuple[float, ...]


@dataclass(frozen=True, eq=False)
class Table:
    """One tabulated magnification-factor table.

    ``axes`` names each index of the value arrays and gives its printed grid values;
    ``quantities`` names the arrays in ``values`` (same length, same order); ``reference`` lists the
    bibliography numbers the transcription's heading cites; ``notes`` records printing details.
    """

    name: str
    axes: tuple[Axis, ...]
    quantities: tuple[str, ...]
    values: tuple[FloatArray, ...]
    reference: tuple[int, ...]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        shape = tuple(len(grid) for _, grid in self.axes)
        if len(self.values) != len(self.quantities):
            raise ValueError(f"{self.name}: {len(self.values)} arrays for {len(self.quantities)} quantities")
        for name, arr in zip(self.quantities, self.values, strict=True):
            if arr.shape != shape:
                raise ValueError(f"{self.name} {name}: shape {arr.shape} does not match axes {shape}")

    def array(self, quantity: str) -> FloatArray:
        """Return the value array for ``quantity``."""
        return self.values[self.quantities.index(quantity)]


def _arr(rows: tuple[Row, ...] | tuple[tuple[Row, ...], ...]) -> FloatArray:
    out = np.array(rows, dtype=np.float64)
    out.setflags(write=False)
    return out


# ======================================================================================
# Through-thickness tables: M.1 a)-d) (axial) and M.6 a)-d) (circumferential)
# Each printed column (one r_m/B value) is typed as rows (lambda, M_1, M_2, M_3, M_4).
# ======================================================================================

RM_OVER_B: Final[tuple[float, ...]] = (5.0, 10.0, 20.0, 50.0, 100.0)

_M1_LAMBDA: Final[tuple[float, ...]] = (
    0.000, 0.862, 0.910, 1.016, 1.285, 1.818, 1.928, 2.012, 2.032, 3.636, 3.856, 4.024, 4.065, 5.784, 6.036,
    6.097, 6.362, 7.712, 7.926, 8.048, 8.130, 8.186, 9.959, 9.998, 10.162, 10.283, 11.816, 11.991, 12.072,
    12.194, 12.211,
)  # fmt: skip

_M1_COLUMNS: Final[dict[float, tuple[Row, ...]]] = {
    5.0: (
        (0.000, 1.000, 0.000, 0.000, 1.000),
        (1.016, 1.433, 0.098, 0.053, 0.701),
        (2.032, 1.912, 0.143, 0.083, 0.604),
        (4.065, 3.133, -0.030, 0.121, 0.493),
        (6.097, 4.116, -0.419, 0.139, 0.417),
        (7.926, 4.980, -0.851, 0.150, 0.364),
        (9.959, 5.873, -1.358, 0.161, 0.314),
        (11.991, 6.687, -1.829, 0.171, 0.276),
    ),
    10.0: (
        (0.000, 1.000, 0.000, 0.000, 1.000),
        (0.862, 1.158, 0.093, 0.040, 0.694),
        (2.012, 1.636, 0.156, 0.075, 0.608),
        (4.024, 2.604, 0.041, 0.109, 0.517),
        (6.036, 3.527, -0.264, 0.128, 0.453),
        (8.048, 4.377, -0.684, 0.138, 0.403),
        (12.072, 5.874, -1.718, 0.150, 0.328),
    ),
    20.0: (
        (0.000, 1.000, 0.000, 0.000, 1.000),
        (1.016, 1.249, 0.125, 0.040, 0.659),
        (2.032, 1.691, 0.182, 0.068, 0.602),
        (4.065, 2.709, 0.089, 0.103, 0.524),
        (6.097, 3.650, -0.200, 0.123, 0.467),
        (8.130, 4.605, -0.622, 0.135, 0.421),
        (10.162, 5.463, -1.122, 0.143, 0.382),
        (12.194, 6.257, -1.700, 0.146, 0.355),
    ),
    50.0: (
        (0.000, 1.000, 0.000, 0.000, 1.000),
        (1.285, 1.383, 0.165, 0.042, 0.629),
        (1.928, 1.663, 0.205, 0.060, 0.600),
        (3.856, 2.642, 0.161, 0.097, 0.529),
        (5.784, 3.613, -0.077, 0.119, 0.474),
        (7.712, 4.534, -0.436, 0.134, 0.430),
        (10.283, 5.688, -1.034, 0.145, 0.381),
        (12.211, 6.503, -1.543, 0.150, 0.353),
    ),
    100.0: (
        (0.000, 1.000, 0.000, 0.000, 1.000),
        (0.910, 1.264, 0.143, 0.025, 0.637),
        (1.818, 1.609, 0.229, 0.055, 0.598),
        (3.636, 2.543, 0.218, 0.095, 0.527),
        (6.362, 3.927, -0.126, 0.127, 0.448),
        (8.186, 4.799, -0.475, 0.139, 0.407),
        (9.998, 5.628, -0.884, 0.147, 0.374),
        (11.816, 6.416, -1.339, 0.151, 0.348),
    ),
}

_M6_LAMBDA: Final[tuple[float, ...]] = (
    0.000, 0.177, 0.251, 0.355, 0.502, 0.561, 0.709, 0.793, 1.064, 1.122, 1.505, 1.586, 1.596, 2.128, 2.257,
    2.306, 3.193, 3.261, 3.365, 3.902, 4.515, 4.612, 4.759, 4.789, 5.498, 5.518, 6.385, 6.772, 7.139, 7.776,
    7.804, 9.032, 9.578, 10.096, 10.997, 12.770, 15.143,
)  # fmt: skip

_M6_COLUMNS: Final[dict[float, tuple[Row, ...]]] = {
    5.0: (
        (0.000, 1.000, 0.000, 0.000, 1.000),
        (0.177, 1.248, 0.069, 0.023, 0.918),
        (0.355, 1.290, 0.077, 0.037, 0.816),
        (1.064, 1.406, 0.140, 0.064, 0.624),
        (1.596, 1.522, 0.153, 0.079, 0.533),
        (2.306, 1.723, 0.112, 0.092, 0.441),
        (3.193, 2.044, -0.014, 0.106, 0.361),
        (3.902, 2.367, -0.158, 0.117, 0.315),
        (4.789, 2.883, -0.385, 0.135, 0.270),
        (5.498, 3.414, -0.622, 0.156, 0.239),
        (6.385, 4.301, -1.015, 0.191, 0.203),
    ),
    10.0: (
        (0.000, 1.000, 0.000, 0.000, 1.000),
        (0.251, 1.032, 0.035, 0.021, 0.828),
        (0.502, 1.066, 0.057, 0.028, 0.733),
        (1.505, 1.192, 0.121, 0.054, 0.544),
        (2.257, 1.324, 0.108, 0.063, 0.450),
        (3.261, 1.545, 0.019, 0.069, 0.364),
        (4.515, 1.864, -0.153, 0.074, 0.299),
        (5.518, 2.164, -0.328, 0.079, 0.264),
        (6.772, 2.641, -0.528, 0.088, 0.230),
        (7.776, 3.117, -0.747, 0.100, 0.205),
        (9.032, 3.917, -1.071, 0.119, 0.179),
    ),
    20.0: (
        (0.000, 1.000, 0.000, 0.000, 1.000),
        (0.355, 1.050, 0.034, 0.015, 0.750),
        (0.709, 1.085, 0.069, 0.025, 0.666),
        (2.128, 1.276, 0.116, 0.048, 0.465),
        (3.193, 1.469, 0.041, 0.052, 0.373),
        (4.612, 1.752, -0.119, 0.052, 0.301),
        (6.385, 2.140, -0.318, 0.054, 0.249),
        (7.804, 2.495, -0.485, 0.059, 0.218),
        (9.578, 3.040, -0.762, 0.065, 0.187),
        (10.997, 3.580, -0.944, 0.068, 0.182),
        (12.770, 4.502, -1.281, 0.078, 0.161),
    ),
    50.0: (
        (0.000, 1.000, 0.000, 0.000, 1.000),
        (0.561, 1.061, 0.045, 0.013, 0.673),
        (1.122, 1.096, 0.087, 0.026, 0.587),
        (3.365, 1.378, 0.043, 0.043, 0.364),
        (10.096, 2.588, -0.585, 0.046, 0.184),
        (15.143, 3.623, -1.126, 0.029, 0.205),
    ),
    100.0: (
        (0.000, 1.000, 0.000, 0.000, 1.000),
        (0.793, 1.088, 0.063, 0.012, 0.633),
        (1.586, 1.139, 0.102, 0.025, 0.544),
        (4.759, 1.425, -0.082, 0.032, 0.293),
        (7.139, 1.732, -0.277, 0.033, 0.228),
    ),
}


def _sparse_grid(lambdas: tuple[float, ...], columns: dict[float, tuple[Row, ...]], field: int) -> FloatArray:
    """Place one field (1..4 -> M_1..M_4) of the per-column rows on the (lambda, r_m/B) grid; blanks are NaN."""
    grid = np.full((len(lambdas), len(RM_OVER_B)), np.nan, dtype=np.float64)
    for j, rm_b in enumerate(RM_OVER_B):
        for row in columns[rm_b]:
            i = lambdas.index(row[0])
            if not np.isnan(grid[i, j]):
                raise ValueError(f"duplicate cell at lambda={row[0]}, r_m/B={rm_b}")
            grid[i, j] = row[field]
    grid.setflags(write=False)
    return grid


_M1_AXES: Final[tuple[Axis, ...]] = (("lambda", _M1_LAMBDA), ("rm_over_B", RM_OVER_B))
_M1_NOTES: Final[tuple[str, ...]] = (
    "Full printed grid of 31 lambda rows by 5 r_m/B columns; cells blank in print are NaN.",
    "Each r_m/B column is populated only at its own lambda values (lambda = 0.000 is populated in all five "
    "columns; lambda = 1.016, 2.032, 4.065 and 6.097 appear in both the r_m/B = 5 and r_m/B = 20 columns).",
    "The r_m/B = 10 column has only 7 entries: nothing is printed between lambda = 8.048 and 12.072, "
    "where every other column has an eighth entry.",
    "Columns end at different lambda (11.991, 12.072, 12.194, 12.211, 11.816) although the printed range "
    "of application is 0 <= lambda <= 12.211 for all r_m/B.",
    "No footnotes are printed under any of the four sub-tables.",
)

M1A: Final = Table(
    name="M.1a",
    axes=_M1_AXES,
    quantities=("M_1",),
    values=(_sparse_grid(_M1_LAMBDA, _M1_COLUMNS, 1),),
    reference=(176,),
    notes=_M1_NOTES,
)
M1B: Final = Table(
    name="M.1b",
    axes=_M1_AXES,
    quantities=("M_2",),
    values=(_sparse_grid(_M1_LAMBDA, _M1_COLUMNS, 2),),
    reference=(176,),
    notes=_M1_NOTES,
)
M1C: Final = Table(
    name="M.1c",
    axes=_M1_AXES,
    quantities=("M_3",),
    values=(_sparse_grid(_M1_LAMBDA, _M1_COLUMNS, 3),),
    reference=(176,),
    notes=_M1_NOTES,
)
M1D: Final = Table(
    name="M.1d",
    axes=_M1_AXES,
    quantities=("M_4",),
    values=(_sparse_grid(_M1_LAMBDA, _M1_COLUMNS, 4),),
    reference=(176,),
    notes=_M1_NOTES,
)

_M6_AXES: Final[tuple[Axis, ...]] = (("lambda", _M6_LAMBDA), ("rm_over_B", RM_OVER_B))
_M6_NOTES: Final[tuple[str, ...]] = (
    "Full printed grid of 37 lambda rows by 5 r_m/B columns; cells blank in print are NaN.",
    "lambda = 0.000 is populated in all five columns; lambda = 0.355, 3.193 and 6.385 appear in both the "
    "r_m/B = 5 and r_m/B = 20 columns.",
    "Column coverage is uneven: r_m/B = 5 ends at lambda = 6.385, 10 at 9.032, 20 at 12.770, 100 at 7.139; "
    "only r_m/B = 50 reaches the printed upper limit 15.143, and that column has 6 entries with a gap "
    "between 3.365 and 10.096.",
    "No footnotes are printed under any of the four sub-tables.",
)

M6A: Final = Table(
    name="M.6a",
    axes=_M6_AXES,
    quantities=("M_1",),
    values=(_sparse_grid(_M6_LAMBDA, _M6_COLUMNS, 1),),
    reference=(176,),
    notes=_M6_NOTES,
)
M6B: Final = Table(
    name="M.6b",
    axes=_M6_AXES,
    quantities=("M_2",),
    values=(_sparse_grid(_M6_LAMBDA, _M6_COLUMNS, 2),),
    reference=(176,),
    notes=_M6_NOTES,
)
M6C: Final = Table(
    name="M.6c",
    axes=_M6_AXES,
    quantities=("M_3",),
    values=(_sparse_grid(_M6_LAMBDA, _M6_COLUMNS, 3),),
    reference=(176,),
    notes=_M6_NOTES,
)
M6D: Final = Table(
    name="M.6d",
    axes=_M6_AXES,
    quantities=("M_4",),
    values=(_sparse_grid(_M6_LAMBDA, _M6_COLUMNS, 4),),
    reference=(176,),
    notes=_M6_NOTES,
)


# ======================================================================================
# Semi-elliptical surface-flaw tables: M.2, M.4 (axial), M.7 (circumferential)
# Each printed row is typed as (a/c, a/B, then for the first B/r_i: M_m(d), M_b(d), M_m(s), M_b(s),
# then the same four for the second B/r_i), i.e. exactly the printed column order.
# ======================================================================================

SURFACE_QUANTITIES: Final[tuple[str, ...]] = ("M_m(d)", "M_b(d)", "M_m(s)", "M_b(s)")
A_OVER_B: Final[tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8)


def _surface_table(
    name: str,
    a_over_c: tuple[float, ...],
    b_over_ri: tuple[float, float],
    rows: tuple[Row, ...],
    reference: tuple[int, ...],
    notes: tuple[str, ...],
) -> Table:
    """Split printed surface-flaw rows into four (a/c, a/B, B/r_i) arrays."""
    n_ac, n_ab = len(a_over_c), len(A_OVER_B)
    if len(rows) != n_ac * n_ab:
        raise ValueError(f"{name}: {len(rows)} rows, expected {n_ac * n_ab}")
    arrays = [np.full((n_ac, n_ab, 2), np.nan, dtype=np.float64) for _ in SURFACE_QUANTITIES]
    for r, row in enumerate(rows):
        if len(row) != 10:
            raise ValueError(f"{name}: row {r} has {len(row)} entries, expected 10")
        i, j = divmod(r, n_ab)
        if row[0] != a_over_c[i] or row[1] != A_OVER_B[j]:
            raise ValueError(f"{name}: row {r} labelled ({row[0]}, {row[1]}), expected ({a_over_c[i]}, {A_OVER_B[j]})")
        for q in range(4):
            arrays[q][i, j, 0] = row[2 + q]
            arrays[q][i, j, 1] = row[6 + q]
    for arr in arrays:
        arr.setflags(write=False)
    return Table(
        name=name,
        axes=(("a_over_c", a_over_c), ("a_over_B", A_OVER_B), ("B_over_r_i", b_over_ri)),
        quantities=SURFACE_QUANTITIES,
        values=tuple(arrays),
        reference=reference,
        notes=notes,
    )


_M2_A_OVER_C: Final[tuple[float, ...]] = (1.0, 0.4, 0.2, 0.1, 0.05)
_M2_B_OVER_RI: Final[tuple[float, float]] = (0.1, 0.25)

_M2_ROWS: Final[tuple[Row, ...]] = (
    (1.0, 0.0, 0.663, 0.663, 0.729, 0.729, 0.663, 0.663, 0.729, 0.729),
    (1.0, 0.2, 0.647, 0.464, 0.726, 0.676, 0.643, 0.461, 0.719, 0.669),
    (1.0, 0.4, 0.661, 0.291, 0.760, 0.649, 0.656, 0.288, 0.745, 0.638),
    (1.0, 0.6, 0.677, 0.110, 0.804, 0.623, 0.677, 0.107, 0.785, 0.610),
    (1.0, 0.8, 0.694, -0.080, 0.859, 0.599, 0.704, -0.079, 0.838, 0.585),
    (0.4, 0.0, 0.951, 0.951, 0.662, 0.662, 0.951, 0.951, 0.662, 0.662),
    (0.4, 0.2, 0.932, 0.698, 0.676, 0.632, 0.919, 0.688, 0.669, 0.627),
    (0.4, 0.4, 1.016, 0.519, 0.768, 0.651, 0.998, 0.506, 0.759, 0.644),
    (0.4, 0.6, 1.109, 0.316, 0.896, 0.674, 1.110, 0.311, 0.889, 0.666),
    (0.4, 0.8, 1.211, 0.090, 1.060, 0.700, 1.255, 0.103, 1.060, 0.694),
    (0.2, 0.0, 1.059, 1.059, 0.521, 0.521, 1.059, 1.059, 0.521, 0.521),
    (0.2, 0.2, 1.062, 0.806, 0.578, 0.548, 1.045, 0.791, 0.577, 0.547),
    (0.2, 0.4, 1.260, 0.677, 0.695, 0.597, 1.240, 0.663, 0.698, 0.599),
    (0.2, 0.6, 1.500, 0.515, 0.876, 0.660, 1.514, 0.515, 0.887, 0.665),
    (0.2, 0.8, 1.783, 0.320, 1.123, 0.737, 1.865, 0.348, 1.144, 0.745),
    (0.1, 0.0, 1.103, 1.103, 0.384, 0.384, 1.103, 1.103, 0.384, 0.384),
    (0.1, 0.2, 1.172, 0.897, 0.451, 0.429, 1.153, 0.881, 0.451, 0.428),
    (0.1, 0.4, 1.494, 0.834, 0.582, 0.503, 1.470, 0.816, 0.585, 0.504),
    (0.1, 0.6, 1.985, 0.765, 0.820, 0.623, 2.003, 0.765, 0.830, 0.627),
    (0.1, 0.8, 2.737, 0.689, 1.219, 0.810, 2.864, 0.749, 1.242, 0.819),
    (0.05, 0.0, 1.120, 1.120, 0.275, 0.275, 1.120, 1.120, 0.275, 0.275),
    (0.05, 0.2, 1.231, 0.946, 0.335, 0.318, 1.211, 0.929, 0.334, 0.318),
    (0.05, 0.4, 1.701, 0.971, 0.469, 0.406, 1.674, 0.950, 0.471, 0.407),
    (0.05, 0.6, 2.619, 1.080, 0.765, 0.584, 2.285, 1.079, 0.774, 0.587),
    (0.05, 0.8, 4.364, 1.301, 1.374, 0.919, 3.163, 1.081, 1.400, 0.928),
)

M2: Final = _surface_table(
    "M.2",
    _M2_A_OVER_C,
    _M2_B_OVER_RI,
    _M2_ROWS,
    reference=(278, 279, 280),
    notes=(
        "Axial internal semi-elliptical surface flaw; printed as two side-by-side halves "
        "(B/r_i = 0.1 and 0.25), each with columns M_m(d), M_b(d), M_m(s), M_b(s).",
        "Tabulated a/c values are 1.0, 0.4, 0.2, 0.1, 0.05 only; B/r_i values 0.1 and 0.25 only.",
        "No footnotes.",
    ),
)

_M4_ROWS: Final[tuple[Row, ...]] = (
    (1.0, 0.0, 0.663, 0.663, 0.729, 0.729, 0.663, 0.663, 0.729, 0.729),
    (1.0, 0.2, 0.653, 0.470, 0.736, 0.685, 0.656, 0.473, 0.741, 0.689),
    (1.0, 0.4, 0.675, 0.301, 0.783, 0.666, 0.683, 0.307, 0.793, 0.673),
    (1.0, 0.6, 0.695, 0.122, 0.846, 0.649, 0.710, 0.131, 0.864, 0.659),
    (1.0, 0.8, 0.712, -0.068, 0.926, 0.634, 0.736, -0.055, 0.954, 0.647),
    (0.4, 0.0, 0.951, 0.951, 0.662, 0.662, 0.951, 0.951, 0.662, 0.662),
    (0.4, 0.2, 0.953, 0.716, 0.685, 0.641, 0.964, 0.726, 0.689, 0.644),
    (0.4, 0.4, 1.077, 0.561, 0.799, 0.673, 1.110, 0.582, 0.806, 0.678),
    (0.4, 0.6, 1.213, 0.377, 0.970, 0.715, 1.289, 0.417, 0.982, 0.721),
    (0.4, 0.8, 1.361, 0.167, 1.198, 0.769, 1.502, 0.230, 1.217, 0.775),
    (0.2, 0.0, 1.059, 1.059, 0.521, 0.521, 1.059, 1.059, 0.521, 0.521),
    (0.2, 0.2, 1.092, 0.831, 0.583, 0.552, 1.106, 0.844, 0.583, 0.552),
    (0.2, 0.4, 1.370, 0.750, 0.706, 0.606, 1.410, 0.776, 0.693, 0.598),
    (0.2, 0.6, 1.735, 0.644, 0.912, 0.681, 1.838, 0.693, 0.867, 0.659),
    (0.2, 0.8, 2.188, 0.514, 1.202, 0.780, 2.390, 0.595, 1.105, 0.736),
    (0.1, 0.0, 1.103, 1.103, 0.384, 0.384, 1.103, 1.103, 0.384, 0.384),
    (0.1, 0.2, 1.206, 0.926, 0.455, 0.432, 1.222, 0.939, 0.455, 0.432),
    (0.1, 0.4, 1.624, 0.923, 0.592, 0.510, 1.672, 0.955, 0.581, 0.504),
    (0.1, 0.6, 2.295, 0.957, 0.853, 0.643, 2.432, 1.029, 0.811, 0.622),
    (0.1, 0.8, 3.360, 1.108, 1.305, 0.857, 3.670, 1.128, 1.199, 0.809),
    (0.05, 0.0, 1.120, 1.120, 0.275, 0.275, 1.120, 1.120, 0.275, 0.275),
    (0.05, 0.2, 1.266, 0.976, 0.338, 0.321, 1.282, 0.991, 0.338, 0.321),
    (0.05, 0.4, 1.849, 1.075, 0.477, 0.412, 1.753, 1.011, 0.468, 0.407),
    (0.05, 0.6, 2.628, 1.349, 0.796, 0.602, 2.581, 1.107, 0.757, 0.583),
    (0.05, 0.8, 4.090, 1.549, 1.471, 0.972, 3.839, 1.153, 1.352, 0.918),
)

M4: Final = _surface_table(
    "M.4",
    _M2_A_OVER_C,
    _M2_B_OVER_RI,
    _M4_ROWS,
    reference=(278, 279, 280),
    notes=(
        "Axial external semi-elliptical surface flaw; same layout and grid as Table M.2.",
        "The a/B = 0.0 rows are identical to those of Table M.2 for the same a/c, with M_m = M_b there.",
        "No footnotes.",
    ),
)

_M7_A_OVER_C: Final[tuple[float, ...]] = (1.0, 0.5, 0.2, 0.1)
_M7_B_OVER_RI: Final[tuple[float, float]] = (0.1, 0.2)

_M7_ROWS: Final[tuple[Row, ...]] = (
    (1.0, 0.0, 0.663, 0.663, 0.729, 0.729, 0.663, 0.663, 0.729, 0.729),
    (1.0, 0.2, 0.667, 0.574, 0.681, 0.623, 0.667, 0.582, 0.681, 0.623),
    (1.0, 0.4, 0.670, 0.327, 0.706, 0.528, 0.670, 0.334, 0.706, 0.528),
    (1.0, 0.6, 0.686, 0.140, 0.733, 0.431, 0.686, 0.117, 0.733, 0.431),
    (1.0, 0.8, 0.702, -0.105, 0.764, 0.332, 0.702, -0.099, 0.764, 0.332),
    (0.5, 0.0, 0.896, 0.896, 0.697, 0.697, 0.896, 0.896, 0.697, 0.697),
    (0.5, 0.2, 0.999, 0.731, 0.731, 0.628, 1.004, 0.735, 0.731, 0.628),
    (0.5, 0.4, 1.031, 0.504, 0.801, 0.563, 1.030, 0.503, 0.801, 0.563),
    (0.5, 0.6, 1.121, 0.306, 0.889, 0.502, 1.124, 0.305, 0.889, 0.502),
    (0.5, 0.8, 1.148, 0.014, 0.993, 0.445, 1.192, 0.027, 0.993, 0.445),
    (0.2, 0.0, 1.059, 1.059, 0.521, 0.521, 1.059, 1.059, 0.521, 0.521),
    (0.2, 0.2, 1.168, 0.870, 0.617, 0.623, 1.144, 0.851, 0.617, 0.623),
    (0.2, 0.4, 1.375, 0.736, 0.835, 0.591, 1.318, 0.698, 0.835, 0.591),
    (0.2, 0.6, 1.599, 0.561, 1.048, 0.556, 1.517, 0.515, 1.048, 0.556),
    (0.2, 0.8, 1.803, 0.269, 1.255, 0.519, 1.782, 0.253, 1.255, 0.519),
    (0.1, 0.0, 1.103, 1.103, 0.384, 0.384, 1.103, 1.103, 0.384, 0.384),
    (0.1, 0.2, 1.219, 0.921, 0.482, 0.487, 1.214, 0.903, 0.482, 0.487),
    (0.1, 0.4, 1.529, 0.829, 0.700, 0.498, 1.382, 0.776, 0.700, 0.498),
    (0.1, 0.6, 1.939, 0.677, 0.981, 0.525, 1.661, 0.624, 0.981, 0.525),
    (0.1, 0.8, 2.411, 0.479, 1.363, 0.570, 2.031, 0.386, 1.363, 0.570),
)

M7: Final = _surface_table(
    "M.7",
    _M7_A_OVER_C,
    _M7_B_OVER_RI,
    _M7_ROWS,
    reference=(278, 279, 282, 283),
    notes=(
        "Circumferential internal semi-elliptical surface flaw; layout as Table M.2 but with halves "
        "B/r_i = 0.1 and 0.2.",
        "Tabulated a/c values are 1.0, 0.5, 0.2, 0.1 only (no 0.4 and no 0.05 as in M.2/M.4).",
        "The surface-point columns M_m(s) and M_b(s) are identical in the B/r_i = 0.1 and 0.2 halves for "
        "every row, as printed.",
        "No footnotes.",
    ),
)


# ======================================================================================
# Extended (full-length / fully circumferential) flaw tables: M.3, M.5, M.8
# ======================================================================================

EXTENDED_QUANTITIES: Final[tuple[str, ...]] = ("M_m", "M_b")


def _extended_table(
    name: str,
    b_over_ri: tuple[float, ...],
    rows: tuple[Row, ...],
    reference: tuple[int, ...],
    notes: tuple[str, ...],
) -> Table:
    """Split printed rows (a/B, M_m, M_b for first B/r_i, M_m, M_b for second B/r_i) into two (a/B, B/r_i) arrays."""
    if len(b_over_ri) != 2 or len(rows) != len(A_OVER_B):
        raise ValueError(f"{name}: unexpected grid")
    mm = np.full((len(A_OVER_B), 2), np.nan, dtype=np.float64)
    mb = np.full((len(A_OVER_B), 2), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        if len(row) != 5 or row[0] != A_OVER_B[i]:
            raise ValueError(f"{name}: row {i} malformed")
        mm[i, 0], mb[i, 0], mm[i, 1], mb[i, 1] = row[1], row[2], row[3], row[4]
    mm.setflags(write=False)
    mb.setflags(write=False)
    return Table(
        name=name,
        axes=(("a_over_B", A_OVER_B), ("B_over_r_i", b_over_ri)),
        quantities=EXTENDED_QUANTITIES,
        values=(mm, mb),
        reference=reference,
        notes=notes,
    )


_M3_ROWS: Final[tuple[Row, ...]] = (
    (0.0, 1.122, 1.122, 1.122, 1.122),
    (0.2, 1.380, 1.018, 1.304, 1.002),
    (0.4, 1.930, 1.143, 1.784, 1.033),
    (0.6, 2.960, 1.484, 2.566, 1.094),
    (0.8, 4.820, 1.990, 3.461, 0.949),
)

M3: Final = _extended_table(
    "M.3",
    (0.1, 0.25),
    _M3_ROWS,
    reference=(278, 279, 280),
    notes=(
        "Extended axial internal surface flaw; columns M_m, M_b for B/r_i = 0.1 then B/r_i = 0.25.",
        "The text places both M_m and M_b at the deepest point of the flaw.",
        "No footnotes.",
    ),
)

_M8_ROWS: Final[tuple[Row, ...]] = (
    (0.0, 1.122, 1.122, 1.122, 1.122),
    (0.2, 1.261, 0.954, 1.215, 0.933),
    (0.4, 1.582, 0.909, 1.446, 0.810),
    (0.6, 2.091, 0.810, 1.804, 0.650),
    (0.8, 2.599, 0.600, 2.280, 0.411),
)

M8: Final = _extended_table(
    "M.8",
    (0.1, 0.2),
    _M8_ROWS,
    reference=(282, 284, 285),
    notes=(
        "Fully circumferential (extended) internal surface flaw; columns M_m, M_b for B/r_i = 0.1 then B/r_i = 0.2.",
        "No footnotes.",
    ),
)

# Table M.5: nine B/r_i blocks of five a/B rows each; typed as (B/r_i, a/B, M_m, M_b) in printed order.
_M5_B_OVER_RI: Final[tuple[float, ...]] = (0.001, 0.00333, 0.01, 0.01667, 0.025, 0.05, 0.1, 0.2, 0.333333)

_M5_ROWS: Final[tuple[Row, ...]] = (
    (0.001, 0.0, 1.120000, 1.120000),
    (0.001, 0.2, 1.362492, 1.052320),
    (0.001, 0.4, 2.106159, 1.258945),
    (0.001, 0.6, 4.023909, 1.912231),
    (0.001, 0.8, 11.909190, 4.657704),
    (0.00333, 0.0, 1.120000, 1.120000),
    (0.00333, 0.2, 1.357654, 1.048520),
    (0.00333, 0.4, 2.098131, 1.253795),
    (0.00333, 0.6, 3.988986, 1.893041),
    (0.00333, 0.8, 11.418040, 4.448136),
    (0.01, 0.0, 1.120000, 1.120000),
    (0.01, 0.2, 1.355914, 1.047032),
    (0.01, 0.4, 2.086534, 1.245785),
    (0.01, 0.6, 3.923337, 1.856235),
    (0.01, 0.8, 10.534910, 4.060550),
    (0.01667, 0.0, 1.120000, 1.120000),
    (0.01667, 0.2, 1.354559, 1.045929),
    (0.01667, 0.4, 2.075495, 1.238148),
    (0.01667, 0.6, 3.858610, 1.820100),
    (0.01667, 0.8, 9.814990, 3.742856),
    (0.025, 0.0, 1.120000, 1.120000),
    (0.025, 0.2, 1.352815, 1.044438),
    (0.025, 0.4, 2.058880, 1.227570),
    (0.025, 0.6, 3.783314, 1.777812),
    (0.025, 0.8, 9.072502, 3.416980),
    (0.05, 0.0, 1.120000, 1.120000),
    (0.05, 0.2, 1.348153, 1.040533),
    (0.05, 0.4, 2.028188, 1.205201),
    (0.05, 0.6, 3.584289, 1.665773),
    (0.05, 0.8, 7.522466, 2.733754),
    (0.1, 0.0, 1.120000, 1.120000),
    (0.1, 0.2, 1.338976, 1.032891),
    (0.1, 0.4, 1.964321, 1.160635),
    (0.1, 0.6, 3.270363, 1.543946),
    (0.1, 0.8, 5.839919, 1.993885),
    (0.2, 0.0, 1.120000, 1.120000),
    (0.2, 0.2, 1.324199, 1.020313),
    (0.2, 0.4, 1.861734, 1.089804),
    (0.2, 0.6, 2.864663, 1.261675),
    (0.2, 0.8, 4.412961, 1.367035),
    (0.333333, 0.0, 1.120000, 1.120000),
    (0.333333, 0.2, 1.310620, 1.008616),
    (0.333333, 0.4, 1.768778, 1.024620),
    (0.333333, 0.6, 2.576840, 1.098873),
    (0.333333, 0.8, 3.652018, 1.033655),
)


def _m5_table() -> Table:
    n_b, n_a = len(_M5_B_OVER_RI), len(A_OVER_B)
    if len(_M5_ROWS) != n_b * n_a:
        raise ValueError("M.5: unexpected row count")
    mm = np.full((n_b, n_a), np.nan, dtype=np.float64)
    mb = np.full((n_b, n_a), np.nan, dtype=np.float64)
    for r, row in enumerate(_M5_ROWS):
        i, j = divmod(r, n_a)
        if len(row) != 4 or row[0] != _M5_B_OVER_RI[i] or row[1] != A_OVER_B[j]:
            raise ValueError(f"M.5: row {r} malformed")
        mm[i, j], mb[i, j] = row[2], row[3]
    mm.setflags(write=False)
    mb.setflags(write=False)
    return Table(
        name="M.5",
        axes=(("B_over_r_i", _M5_B_OVER_RI), ("a_over_B", A_OVER_B)),
        quantities=EXTENDED_QUANTITIES,
        values=(mm, mb),
        reference=(281,),
        notes=(
            "Extended axial external surface flaw; values printed to six decimals (a thin space after the "
            "third decimal in print is dropped here).",
            "The whole table is enclosed in printed corrigendum tags (C1), not the A1 amendment tags used elsewhere.",
            "The table does not state at which point M_m and M_b apply.",
            "B/r_i labels are stored exactly as printed (0.001, 0.00333, 0.01, 0.01667, 0.025, 0.05, 0.1, 0.2, "
            "0.333333); the printed range of application uses 0.333 333 for the upper limit.",
            "At a/B = 0 the value is 1.120000 here, versus 1.122 in Tables M.3 and M.8.",
            "No footnotes.",
        ),
    )


M5: Final = _m5_table()


# ======================================================================================
# Collections, ranges of application, suspect cells, cell counts
# ======================================================================================

TABLES: Final[dict[str, Table]] = {t.name: t for t in (M1A, M1B, M1C, M1D, M2, M3, M4, M5, M6A, M6B, M6C, M6D, M7, M8)}

#: Printed range of application for each table, as the transcription prints it.
RANGES: Final[dict[str, tuple[str, ...]]] = {
    "M.1": ("0 <= lambda <= 12.211", "5 <= r_m/B <= 100"),
    "M.2": ("0 <= a/B <= 0.8", "0.05 <= a/c <= 1", "0.1 <= B/r_i <= 0.25", "2c/W <= 0.15"),
    "M.3": ("0 <= a/B <= 0.8", "0.1 <= B/r_i <= 0.25"),
    "M.4": ("0 <= a/B <= 0.8", "0.05 <= a/c <= 1", "0.1 <= B/r_i <= 0.25", "2c/W <= 0.15"),
    "M.5": ("0 <= a/B <= 0.8", "0.001 <= B/r_i <= 0.333 333"),
    "M.6": ("0 <= lambda <= 15.143", "5 <= r_m/B <= 100"),
    "M.7": ("0 <= a/B <= 0.8", "0.1 <= a/c <= 1", "0.1 <= B/r_i <= 0.2"),
    "M.8": ("0 <= a/B <= 0.8", "0.1 <= B/r_i <= 0.2"),
}

#: Cells the transcription flags as odd, keyed (table name, quantity, index into that quantity's array).
#: Values are stored exactly as printed; the reason paraphrases the transcriber's note.
SUSPECT: Final[dict[tuple[str, str, tuple[int, ...]], str]] = {
    # Table M.1: r_m/B = 10 column (index 1) has only 7 entries.
    ("M.1a", "M_1", (28, 1)): (
        "r_m/B = 10 column: nothing is printed between lambda = 8.048 (row 19) and 12.072 (row 28); the "
        "other four columns each have an eighth entry. Printed gap, not a transcription loss."
    ),
    ("M.1b", "M_2", (28, 1)): "Same printed gap in the r_m/B = 10 column as noted for M.1a.",
    ("M.1c", "M_3", (28, 1)): "Same printed gap in the r_m/B = 10 column as noted for M.1a.",
    ("M.1d", "M_4", (28, 1)): "Same printed gap in the r_m/B = 10 column as noted for M.1a.",
    # Table M.2: a/c = 0.05 block (index 4), B/r_i = 0.1 vs 0.25 discrepancy.
    ("M.2", "M_m(d)", (4, 3, 0)): (
        "a/c = 0.05, a/B = 0.6: B/r_i = 0.1 value 2.619 against 2.285 at B/r_i = 0.25; elsewhere the two "
        "halves differ by a few per cent. Legible in print; possible source misprint."
    ),
    ("M.2", "M_m(d)", (4, 4, 0)): (
        "a/c = 0.05, a/B = 0.8: B/r_i = 0.1 value 4.364 against 3.163 at B/r_i = 0.25. Legible in print; "
        "possible source misprint."
    ),
    ("M.2", "M_b(d)", (4, 4, 0)): (
        "a/c = 0.05, a/B = 0.8: B/r_i = 0.1 value 1.301 against 1.081 at B/r_i = 0.25. Legible in print; "
        "possible source misprint."
    ),
    # Table M.3: large B/r_i effect at a/B = 0.8, same kind as in the a/c = 0.05 block of M.2.
    ("M.3", "M_m", (4, 0)): "a/B = 0.8: 4.820 at B/r_i = 0.1 against 3.461 at 0.25; transcriber observation.",
    ("M.3", "M_b", (4, 0)): "a/B = 0.8: 1.990 at B/r_i = 0.1 against 0.949 at 0.25; transcriber observation.",
    # Table M.6: r_m/B = 5 column (index 0), first points of M_1.
    ("M.6a", "M_1", (1, 0)): (
        "r_m/B = 5, lambda = 0.177: M_1 jumps from 1.000 to 1.248, whereas the other columns rise slowly "
        "from 1.000 at their first points (1.032, 1.050, 1.061, 1.088). As printed."
    ),
    ("M.6a", "M_1", (3, 0)): (
        "r_m/B = 5, lambda = 0.355: M_1 = 1.290 continues from the flagged jump at lambda = 0.177 "
        "(the r_m/B = 20 column has 1.050 at the same lambda). As printed."
    ),
    # Table M.6: r_m/B = 50 column (index 3), last row lambda = 15.143 (index 36).
    ("M.6c", "M_3", (36, 3)): (
        "r_m/B = 50, lambda = 15.143: M_3 falls to 0.029 from 0.046, against the monotonic trend of every "
        "other column. Clearly legible; possible source misprint."
    ),
    ("M.6d", "M_4", (36, 3)): (
        "r_m/B = 50, lambda = 15.143: M_4 rises to 0.205 from 0.184, against the monotonic trend of every "
        "other column. Clearly legible; possible source misprint."
    ),
    # Table M.7: oddities flagged by the transcriber (indices: a/c, a/B, B/r_i).
    ("M.7", "M_b(s)", (2, 1, 0)): (
        "a/c = 0.2, a/B = 0.2: M_b(s) = 0.623 exceeds M_m(s) = 0.617 in the same row, then falls below it "
        "at larger a/B. As printed (both halves identical)."
    ),
    ("M.7", "M_b(s)", (2, 1, 1)): "Same cell as the B/r_i = 0.1 half (the (s) columns are identical in both halves).",
    ("M.7", "M_b(s)", (3, 1, 0)): (
        "a/c = 0.1, a/B = 0.2: M_b(s) = 0.487 exceeds M_m(s) = 0.482 in the same row, then falls below it "
        "at larger a/B. As printed (both halves identical)."
    ),
    ("M.7", "M_b(s)", (3, 1, 1)): "Same cell as the B/r_i = 0.1 half (the (s) columns are identical in both halves).",
    ("M.7", "M_b(d)", (1, 1, 0)): (
        "a/c = 0.5, a/B = 0.2: M_b(d) = 0.731 equals M_m(s) = 0.731 in the same row. As printed."
    ),
    ("M.7", "M_b(d)", (0, 3, 1)): (
        "a/c = 1.0, a/B = 0.6: M_b(d) = 0.117 at B/r_i = 0.2 against 0.140 at B/r_i = 0.1, while the "
        "neighbouring rows differ between halves by 0.006-0.008. As printed."
    ),
    ("M.7", "M_m(s)", (0, 0, 1)): (
        "Whole-column note: every M_m(s) and M_b(s) value in the B/r_i = 0.2 half is identical to the "
        "B/r_i = 0.1 half, for all 20 rows. As printed."
    ),
}

#: Number of printed (non-blank) value cells per table, excluding the axis labels. For M.1 and M.6 the
#: stored grid is larger (31 x 5 and 37 x 5 per sub-table) and the remaining cells are NaN.
CELL_COUNTS: Final[dict[str, int]] = {
    "M.1a": 39,
    "M.1b": 39,
    "M.1c": 39,
    "M.1d": 39,
    "M.2": 200,
    "M.3": 20,
    "M.4": 200,
    "M.5": 90,
    "M.6a": 44,
    "M.6b": 44,
    "M.6c": 44,
    "M.6d": 44,
    "M.7": 160,
    "M.8": 20,
}

#: Printed grid sizes (rows x columns of the printed table) for the sparse through-thickness tables.
GRID_SHAPES: Final[dict[str, tuple[int, int]]] = {
    "M.1": (31, 5),
    "M.6": (37, 5),
}


def _check_counts() -> None:
    for name, table in TABLES.items():
        n = sum(int(np.count_nonzero(~np.isnan(arr))) for arr in table.values)
        if n != CELL_COUNTS[name]:
            raise ValueError(f"{name}: {n} non-blank cells stored, CELL_COUNTS says {CELL_COUNTS[name]}")
    for key in SUSPECT:
        table_name, quantity, index = key
        _ = TABLES[table_name].array(quantity)[index]


_check_counts()
