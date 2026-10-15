"""Option 2 (material-specific) assessment line (theory manual, section 1.4).

With s = stress / yield strength and normalised true strain e = E * strain / yield strength,

    f(Lr) = [ e(Lr)/Lr + Lr^3 / (2 e(Lr)) ]^(-1/2)   for Lr < Lr,max,   0 otherwise.

A material law supplies the plastic true strain eps_p(s) >= 0, so that e(s) = s (1 + kappa q(s))
with kappa = E / yield strength and q(s) = eps_p(s) / s (the secant plastic compliance).

The line is made non-increasing by replacing q with its running maximum q_hat. This is
exact, never raises the line, and is valid when e(s) >= s^2 / sqrt(2), which holds for
every s <= sqrt(2) and, beyond that, for any material that has strained plastically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from critflaw.fad.curves import lr_cutoff

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

_SQRT2 = float(np.sqrt(2.0))
MAX_TABULATED_CORRECTION = 0.05
ELASTIC_NOISE_FRACTION = 0.02  # of the yield strain


class MaterialLaw(Protocol):
    step_locations: tuple[float, ...]
    s_end: float  # end of the law's domain in s (inf for an analytic law)

    def q(self, s: FloatArray) -> FloatArray:
        """Secant plastic compliance eps_p(s)/s as given by the law (not made monotone)."""
        ...

    def q_hat(self, s: FloatArray) -> FloatArray:
        """Running maximum over t <= s of q(t)."""
        ...

    def hardened_enough(self, kappa: FloatArray, lr_end: FloatArray) -> BoolArray:
        """True where e(s) >= s^2/sqrt(2) on (sqrt(2), lr_end)."""
        ...


# --------------------------------------------------------------------------------------
# Analytic power law
# --------------------------------------------------------------------------------------


def hardening_exponent_lower_bound(yield_strength: ArrayLike, uts: ArrayLike) -> FloatArray:
    """n = 1 / (0.3 (1 - sy/su)): the source's lower-bound estimate of strain hardening
    [provenance: material.power_law]."""
    ratio = np.asarray(yield_strength, dtype=float) / np.asarray(uts, dtype=float)
    return np.asarray(1.0 / (0.3 * (1.0 - ratio)), dtype=float)


@dataclass(frozen=True)
class PowerLaw:
    """eps_p(s) = offset * s^n, plus ``luders_strain`` for s >= 1  [provenance: material.power_law].

    ``offset`` is the plastic strain at the yield strength and must be chosen explicitly:
    0.002 when the yield strength is a 0.2% proof strength, 0 when it is a lower yield point.
    """

    offset: ArrayLike
    n: ArrayLike
    luders_strain: ArrayLike | None = None

    def __post_init__(self) -> None:
        offset, n = np.asarray(self.offset, dtype=float), np.asarray(self.n, dtype=float)
        if not (np.all(np.isfinite(offset)) and np.all(offset >= 0.0)):
            raise ValueError("offset must be finite and non-negative")
        if not (np.all(np.isfinite(n)) and np.all(n >= 1.0)):
            raise ValueError("hardening exponent n must be finite and at least 1")
        if self.luders_strain is not None:
            strain = np.asarray(self.luders_strain, dtype=float)
            if not (np.all(np.isfinite(strain)) and np.all(strain > 0.0)):
                raise ValueError("luders_strain must be finite and positive")

    @property
    def step_locations(self) -> tuple[float, ...]:
        return () if self.luders_strain is None else (1.0,)

    @property
    def declared_plateau(self) -> bool:
        return self.luders_strain is not None

    @property
    def s_end(self) -> float:
        return float("inf")

    def _base(self, s: FloatArray) -> FloatArray:
        offset, n = np.asarray(self.offset, dtype=float), np.asarray(self.n, dtype=float)
        # a zero offset is exactly elastic: the power is not evaluated (it can overflow for large n)
        s_b, off_b, n_b = np.broadcast_arrays(np.asarray(s, dtype=float), offset, n)
        out = np.zeros(s_b.shape)
        active = off_b > 0.0
        with np.errstate(over="ignore"):
            out[active] = off_b[active] * s_b[active] ** (n_b[active] - 1.0)
        return np.asarray(out, dtype=float)

    @property
    def parameter_shape(self) -> tuple[int, ...]:
        """Broadcast shape of the law parameters (one law per sample when they are arrays)."""
        values = [self.offset, self.n] + ([] if self.luders_strain is None else [self.luders_strain])
        return tuple(np.broadcast_shapes(*(np.shape(v) for v in values)))

    def q(self, s: FloatArray) -> FloatArray:
        base = self._base(s)
        if self.luders_strain is None:
            return base
        jump = np.asarray(self.luders_strain, dtype=float) / np.maximum(s, 1.0)
        return np.asarray(base + np.where(s >= 1.0, jump, 0.0), dtype=float)

    def q_hat(self, s: FloatArray) -> FloatArray:
        q = self.q(s)
        if self.luders_strain is None:
            return q  # offset * s^(n-1) is non-decreasing for n >= 1
        at_step = np.asarray(self.offset, dtype=float) + np.asarray(self.luders_strain, dtype=float)
        return np.asarray(np.where(s >= 1.0, np.maximum(q, at_step), q), dtype=float)

    def hardened_enough(self, kappa: FloatArray, lr_end: FloatArray) -> BoolArray:
        kappa_, end = np.broadcast_arrays(np.asarray(kappa, dtype=float), np.asarray(lr_end, dtype=float))
        ok = np.ones(kappa_.shape, dtype=bool)
        beyond = end > _SQRT2
        if not np.any(beyond):
            return ok
        # 200 points over [sqrt(2), lr_end] with a 0.1% margin (the law is not piecewise linear)
        t = np.linspace(0.0, 1.0, 200)
        s = _SQRT2 + (np.where(beyond, end, _SQRT2)[..., None] - _SQRT2) * t
        def column(value: ArrayLike | None) -> ArrayLike | None:
            # parameters given per sample get a trailing axis to broadcast against the 200 points
            return np.asarray(value, dtype=float)[..., None] if value is not None and np.ndim(value) else value

        offset, n = column(self.offset), column(self.n)
        assert offset is not None and n is not None
        law = PowerLaw(offset, n, column(self.luders_strain))
        e = s * (1.0 + kappa_[..., None] * law.q(s))
        return np.asarray(np.where(beyond, np.all(e >= 1.001 * s**2 / _SQRT2, axis=-1), True), dtype=bool)


# --------------------------------------------------------------------------------------
# Tabulated law from a measured curve
# --------------------------------------------------------------------------------------


class StressStrainDataError(ValueError):
    pass


def _isotonic(values: FloatArray) -> FloatArray:
    """Least-squares non-decreasing fit (pool adjacent violators)."""
    means: list[float] = []
    counts: list[int] = []
    for v in values:
        means.append(float(v))
        counts.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            total = counts[-2] + counts[-1]
            means[-2] = (means[-2] * counts[-2] + means[-1] * counts[-1]) / total
            counts[-2] = total
            means.pop()
            counts.pop()
    return np.repeat(np.asarray(means, dtype=float), counts)


class TabulatedLaw:
    """Piecewise-linear plastic strain eps_p(s) from a measured stress-strain curve (theory manual, section 3.6.2).

    Build with :meth:`from_engineering` or :meth:`from_true`. ``notes`` records every
    cleaning step applied to the data.
    """

    def __init__(
        self,
        s: FloatArray,
        eps_p: FloatArray,
        step: float | None,
        notes: tuple[str, ...],
        declared_plateau: bool | None = None,
    ) -> None:
        self.knots = np.asarray(s, dtype=float)
        self.eps_p = np.asarray(eps_p, dtype=float)
        self.declared_plateau = (step is not None) if declared_plateau is None else declared_plateau
        # the line steps wherever the plastic strain jumps at constant stress (a repeated knot)
        repeated = np.unique(self.knots[:-1][np.diff(self.knots) == 0.0])
        locations = set(float(v) for v in repeated) | ({float(step)} if step is not None else set())
        self.step_locations: tuple[float, ...] = tuple(sorted(locations))
        self.notes = notes
        if not (self.knots.size >= 2 and self.knots[0] == 0.0 and self.eps_p[0] == 0.0 and self.knots[1] > 0.0):
            raise StressStrainDataError("a tabulated law starts at the origin and has at least two points")
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(self.knots > 0.0, self.eps_p / np.where(self.knots > 0.0, self.knots, 1.0), 0.0)
        q[0] = q[1]
        self._q_knots = q
        self._qhat_knots = np.maximum.accumulate(q)
        # smallest kappa for which e >= s^2/sqrt(2) at each knot beyond sqrt(2), accumulated
        with np.errstate(divide="ignore"):
            need = np.where(
                self.knots > _SQRT2,
                (self.knots**2 / _SQRT2 - self.knots) / np.where(self.eps_p > 0.0, self.eps_p, np.nan),
                0.0,
            )
        need = np.where(np.isnan(need), np.inf, need)
        self._kappa_needed = np.maximum.accumulate(need)

    @property
    def s_end(self) -> float:
        return float(self.knots[-1])

    @property
    def parameter_shape(self) -> tuple[int, ...]:
        return ()

    # ---- construction ------------------------------------------------------------------

    @classmethod
    def from_engineering(
        cls,
        stress: ArrayLike,
        strain: ArrayLike,
        yield_strength: float,
        modulus: float,
        yielding: Literal["continuous", "plateau"],
        luders_strain: float | None = None,
    ) -> TabulatedLaw:
        return cls._build(stress, strain, yield_strength, modulus, yielding, luders_strain, engineering=True)

    @classmethod
    def from_true(
        cls,
        stress: ArrayLike,
        strain: ArrayLike,
        yield_strength: float,
        modulus: float,
        yielding: Literal["continuous", "plateau"],
        luders_strain: float | None = None,
    ) -> TabulatedLaw:
        return cls._build(stress, strain, yield_strength, modulus, yielding, luders_strain, engineering=False)

    @classmethod
    def _build(
        cls,
        stress: ArrayLike,
        strain: ArrayLike,
        yield_strength: float,
        modulus: float,
        yielding: str,
        luders_strain: float | None,
        engineering: bool,
    ) -> TabulatedLaw:
        if yielding not in ("continuous", "plateau"):
            raise ValueError("yielding must be 'continuous' or 'plateau' (it is a required declaration)")
        if not (np.isfinite(yield_strength) and yield_strength > 0.0 and np.isfinite(modulus) and modulus > 0.0):
            raise ValueError("yield_strength and modulus must be finite and positive")
        sig, eps = np.asarray(stress, dtype=float).ravel(), np.asarray(strain, dtype=float).ravel()
        if sig.size != eps.size or sig.size < 3:
            raise StressStrainDataError("stress and strain must have the same length, at least 3 points")
        if not (np.all(np.isfinite(sig)) and np.all(np.isfinite(eps)) and np.all(sig >= 0.0) and np.all(eps >= 0.0)):
            raise StressStrainDataError("stress and strain must be finite and non-negative")
        notes: list[str] = [f"declared yielding: {yielding}"]
        eps_y = yield_strength / modulus

        # 1. sort by strain
        order = np.argsort(eps, kind="stable")
        sig, eps = sig[order], eps[order]

        # 2. declared plateau: remove its points (identified by strain, never detected from stress)
        jump, s_step = 0.0, 1.0
        if yielding == "plateau":
            if luders_strain is None or not (np.isfinite(luders_strain) and luders_strain > 0.0):
                raise ValueError("yielding='plateau' needs luders_strain (a positive number)")
            # the plateau is defined in engineering coordinates, whichever kind of data was supplied
            eng_eps = eps if engineering else np.expm1(eps)
            eng_sig = sig if engineering else sig / (1.0 + eng_eps)
            on_plateau = (eng_eps > eps_y * (1.0 + 1e-9)) & (eng_eps <= eps_y + luders_strain)
            if np.any(np.abs(eng_sig[on_plateau] - yield_strength) > 0.05 * yield_strength):
                raise StressStrainDataError(
                    "points inside the declared plateau differ from the yield strength by more than 5%: "
                    "the declared plateau is not flat, or the yield strength or Lueders strain is wrong"
                )
            notes.append(f"plateau points removed: {int(np.sum(on_plateau))}")
            if np.any(on_plateau):
                # The step is put at the measured plateau stress when that is below the declared yield
                # strength (e.g. an upper yield strength was entered), never above it: the line then
                # cannot stay elastic above the stress at which the material actually yields.
                s_step = min(1.0, float(np.mean(eng_sig[on_plateau])) / yield_strength)
                if s_step < 1.0:
                    notes.append(f"plateau stress is {s_step:.4f} of the declared yield strength; step placed there")
            sig, eps = sig[~on_plateau], eps[~on_plateau]
            jump = float(np.log1p(eps_y + luders_strain) - np.log1p(eps_y))  # Lueders strain as a true strain
        elif luders_strain is not None:
            raise ValueError("luders_strain is only meaningful with yielding='plateau'")

        # 3. stress made non-decreasing in strain by isotonic (least-squares monotone) regression,
        #    which averages out measurement noise; the largest adjustment is limited to 5% of yield
        running = _isotonic(sig)
        largest = float(np.max(np.abs(running - sig)))
        if largest > 0.05 * yield_strength:
            raise StressStrainDataError(
                f"stress had to be adjusted by {largest:.3g} (more than 5% of yield strength) to be monotonic; "
                "the data are not a monotonic stress-strain record"
            )
        if largest > 0.0:
            notes.append(f"stress made non-decreasing; largest adjustment {largest / yield_strength:.2%} of yield")
        # Of a run of points at the same stress keep both ends: the first is where the material was
        # still at the smaller strain (often the elastic limit) and the last is the largest strain
        # reached at that stress. Keeping only the first would discard strain and raise the line;
        # keeping only the last would smear that strain over the stress interval below it.
        rises = np.diff(running) > 0.0
        keep = np.concatenate([[True], rises]) | np.concatenate([rises, [True]])
        sig, eps = running[keep], eps[keep]

        # 4. true stress and strain; plastic strain
        if engineering:
            sig, eps = sig * (1.0 + eps), np.log1p(eps)
        eps_p = eps - sig / modulus
        s = sig / yield_strength

        # 5. elastic range. Plastic strain below 2% of the yield strain is noise-level and is set to
        #    zero up to the last such point, so the line starts exactly elastic. (q = eps_p / s is
        #    what the monotonicity rule accumulates, so noise at low stress would otherwise lower the
        #    whole line.) The cost is real plastic strain below that level being ignored, which raises
        #    the line by up to about 0.02 well below yield for clean data.
        if np.any(eps_p < -0.25 * eps_y):
            raise StressStrainDataError(
                "plastic strain is negative by more than a quarter of the yield strain: the modulus is too low "
                "for these data"
            )
        most_negative = float(np.min(eps_p))
        if most_negative < -ELASTIC_NOISE_FRACTION * eps_y:
            notes.append(
                f"negative plastic strain set to zero (down to {most_negative / eps_y:.2f} of the yield strain): "
                "check the modulus"
            )
        eps_p = np.where(eps_p < ELASTIC_NOISE_FRACTION * eps_y, 0.0, eps_p)
        if s[0] > 0.0:
            s, eps_p = np.concatenate([[0.0], s]), np.concatenate([[0.0], eps_p])
        if not np.any((s > 0.0) & (eps_p == 0.0)):
            raise StressStrainDataError("the data must include the elastic range (a point with no plastic strain)")
        last_elastic = int(np.max(np.nonzero(eps_p == 0.0)[0]))
        eps_p[: last_elastic + 1] = 0.0
        eps_p = np.maximum.accumulate(eps_p)
        # 6. step at the plateau stress for a declared plateau. Data after the step are kept as
        #    measured (their strain already contains the Lueders strain); they are not shifted.
        step = yielding == "plateau"
        if step:
            left, right = s < s_step, s > s_step
            ep_left = float(eps_p[left][-1]) if np.any(left) else 0.0
            ep_right = ep_left + jump
            s = np.concatenate([s[left], [s_step, s_step], s[right]])
            eps_p = np.concatenate([eps_p[left], [ep_left, ep_right], np.maximum(eps_p[right], ep_right)])
            eps_p = np.maximum.accumulate(eps_p)

        # 7. coverage of the source's minimum evaluation set
        required = np.array([0.7, 0.9, 0.98, 1.0, 1.02, 1.1])
        counts = np.histogram(s, bins=required)[0]
        if np.any(counts == 0) and not step:
            notes.append("sparse data near yield: no point in some interval of Lr = 0.7, 0.9, 0.98, 1.0, 1.02, 1.1")
        if s[-1] <= 1.0:
            raise StressStrainDataError("the data must extend beyond the yield strength")
        flats = np.unique(s[:-1][np.diff(s) == 0.0])
        undeclared = [float(v) for v in flats if not (step and v == s_step)]
        if undeclared:
            notes.append(
                "the data are flat (strain increases at constant stress) at s = "
                + ", ".join(f"{v:.4f}" for v in undeclared)
                + "; the line has a step there"
            )
        return cls(s, eps_p, s_step if step else None, tuple(notes), declared_plateau=step)

    # ---- evaluation --------------------------------------------------------------------

    def _interval(self, s: FloatArray) -> NDArray[np.intp]:
        idx = np.searchsorted(self.knots, s, side="right") - 1
        return np.clip(idx, 0, self.knots.size - 2)

    def eps_p_at(self, s: FloatArray) -> FloatArray:
        i = self._interval(s)
        s0, s1 = self.knots[i], self.knots[i + 1]
        width = np.where(s1 > s0, s1 - s0, 1.0)
        t = np.clip((s - s0) / width, 0.0, 1.0)
        return np.asarray(self.eps_p[i] + t * (self.eps_p[i + 1] - self.eps_p[i]), dtype=float)

    def q(self, s: FloatArray) -> FloatArray:
        s_ = np.asarray(s, dtype=float)
        safe = np.where(s_ > 0.0, s_, 1.0)
        # eps_p is linear from the origin on the first interval, so q is constant there: its value at s = 0
        at_origin = self.eps_p[1] / self.knots[1]
        return np.asarray(np.where(s_ > 0.0, self.eps_p_at(s_) / safe, at_origin), dtype=float)

    def q_hat(self, s: FloatArray) -> FloatArray:
        """Exact running maximum: on each linear piece q is monotone, so the maximum over
        [0, s] is the larger of q(s) and the running maximum at the knot before s."""
        s_ = np.asarray(s, dtype=float)
        return np.asarray(np.maximum(self.q(s_), self._qhat_knots[self._interval(s_)]), dtype=float)

    def hardened_enough(self, kappa: FloatArray, lr_end: FloatArray) -> BoolArray:
        kappa_, end = np.broadcast_arrays(np.asarray(kappa, dtype=float), np.asarray(lr_end, dtype=float))
        # knots strictly inside (sqrt(2), lr_end]: accumulated requirement up to the last such knot
        last = np.searchsorted(self.knots, end, side="right") - 1
        knots_ok = kappa_ >= self._kappa_needed[np.clip(last, 0, self.knots.size - 1)]
        e_end = end * (1.0 + kappa_ * self.q(np.minimum(end, self.s_end)))
        end_ok = (end <= _SQRT2) | (e_end >= end**2 / _SQRT2)
        return np.asarray(knots_ok & end_ok, dtype=bool)


# --------------------------------------------------------------------------------------
# The curve
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Option2Curve:
    """Option 2 assessment line for a material law; one curve per sample when the tensile
    properties are arrays. ``lr_max`` is the effective cut-off: the plastic collapse cut-off,
    or the end of a tabulated law if that comes first."""

    law: MaterialLaw
    yield_strength: ArrayLike
    uts: ArrayLike
    modulus: ArrayLike

    def __post_init__(self) -> None:
        sy, su, e = (np.asarray(v, dtype=float) for v in (self.yield_strength, self.uts, self.modulus))
        if not (np.all(np.isfinite(sy)) and np.all(np.isfinite(su)) and np.all(np.isfinite(e))):
            raise ValueError("yield strength, tensile strength and modulus must be finite")
        if np.any(sy <= 0.0) or np.any(e <= 0.0) or np.any(su <= sy):
            raise ValueError("need yield strength > 0, modulus > 0 and tensile strength above yield strength")
        # a WARP3D deformation law carries its own E and sigma_0: the curve's modulus and yield strength must be
        # the law's (theory manual, section 3.6.3), so that Option 2 and Option 3 describe one material
        law_e, law_s0 = getattr(self.law, "E", None), getattr(self.law, "sigma_0", None)
        if law_e is not None and law_s0 is not None:
            if not (
                np.allclose(np.asarray(law_e, dtype=float), e, rtol=1e-12, atol=0.0)
                and np.allclose(np.asarray(law_s0, dtype=float), sy, rtol=1e-12, atol=0.0)
            ):
                raise ValueError(
                    "the law's E and sigma_0 differ from the curve's modulus and yield strength; "
                    "use Option2Curve.from_warp_law(law, uts)"
                )

    @classmethod
    def from_warp_law(cls, law: MaterialLaw, uts: ArrayLike) -> Option2Curve:
        """An Option 2 curve for a WARP3D deformation law: modulus and yield strength are the law's own ``E``
        and ``sigma_0`` (its reference stress, not a 0.2 % proof stress); ``uts`` sets ``Lr,max`` only."""
        sigma_0 = getattr(law, "sigma_0", None)
        modulus = getattr(law, "E", None)
        if sigma_0 is None or modulus is None:
            raise TypeError("from_warp_law needs a law carrying E and sigma_0 (a WarpDeformationLaw)")
        return cls(law, sigma_0, uts, modulus)

    @property
    def kappa(self) -> FloatArray:
        modulus, sy = np.asarray(self.modulus, dtype=float), np.asarray(self.yield_strength, dtype=float)
        return np.asarray(modulus / sy, dtype=float)

    @property
    def collapse_cutoff(self) -> FloatArray:
        return lr_cutoff(self.yield_strength, self.uts)

    @property
    def lr_max(self) -> FloatArray:
        return np.asarray(np.minimum(self.collapse_cutoff, self.law.s_end), dtype=float)

    @property
    def ends_at_data(self) -> BoolArray:
        """True where the tabulated data end before the plastic collapse cut-off."""
        return np.asarray(self.collapse_cutoff > self.law.s_end, dtype=bool)

    @property
    def step_locations(self) -> tuple[float, ...]:
        return self.law.step_locations

    @property
    def valid(self) -> BoolArray:
        """False where the law has not strained plastically beyond Lr = sqrt(2) (the monotone line's validity)."""
        return self.law.hardened_enough(self.kappa, self.lr_max)

    def _line(self, x: FloatArray, q: FloatArray) -> FloatArray:
        r = 1.0 + self.kappa * q
        return np.asarray((r + x**2 / (2.0 * r)) ** -0.5, dtype=float)

    def unmodified(self, lr: ArrayLike) -> FloatArray:
        """The source expression without the monotonicity rule (for reporting the correction)."""
        x = np.asarray(lr, dtype=float)
        inside = x < self.lr_max
        xs = np.where(inside, x, 0.0)
        return np.asarray(np.where(inside, self._line(xs, self.law.q(xs)), 0.0), dtype=float)

    def __call__(self, lr: ArrayLike) -> FloatArray:
        x = np.asarray(lr, dtype=float)
        if np.any(np.isnan(x)) or np.any(x < 0.0):
            raise ValueError("Lr must be non-negative and not NaN")
        inside = x < self.lr_max
        xs = np.where(inside, x, 0.0)  # the law is never evaluated outside its domain
        return np.asarray(np.where(inside, self._line(xs, self.law.q_hat(xs)), 0.0), dtype=float)

    def correction(self, points: int = 1500, chunk: int = 20_000) -> FloatArray:
        """Largest lowering of the line by the monotonicity rule, max(unmodified - line), for each
        realisation over **its own** domain [0, lr_max), evaluated at that realisation's own
        kappa = E / yield strength (no interpolation between realisations).

        Only the Lr values where the rule changes anything (q_hat > q) are evaluated, so a law that
        needs no correction costs nothing. NaN for a law with per-sample parameters."""
        kappa = np.asarray(self.kappa, dtype=float)
        cutoff = np.asarray(self.lr_max, dtype=float)
        shape = np.broadcast_shapes(kappa.shape, cutoff.shape)
        if np.ndim(self.law.q(np.asarray(1.0))) > 0:
            return np.full(shape, np.nan)
        kappa_b = np.broadcast_to(kappa, shape).ravel()
        cutoff_b = np.broadcast_to(cutoff, shape).ravel()
        end = float(np.max(cutoff_b))
        regular = np.linspace(0.0, end, points, endpoint=False)
        before_steps = np.nextafter(self.step_locations, 0.0)
        x = np.unique(np.concatenate([regular, before_steps, self.step_locations]))
        x = x[x < end]
        q, q_hat = self.law.q(x), self.law.q_hat(x)
        active = q_hat > q
        out = np.zeros(kappa_b.size)
        if np.any(active):
            xa, qa, qh = x[active], q[active], q_hat[active]
            for start in range(0, kappa_b.size, chunk):
                k = kappa_b[start : start + chunk, None]
                r, r_hat = 1.0 + k * qa, 1.0 + k * qh
                with np.errstate(over="ignore", invalid="ignore"):
                    diff = (r + xa**2 / (2.0 * r)) ** -0.5 - (r_hat + xa**2 / (2.0 * r_hat)) ** -0.5
                diff = np.where(xa < cutoff_b[start : start + chunk, None], np.nan_to_num(diff), 0.0)
                out[start : start + chunk] = np.max(diff, axis=1)
        return np.asarray(out.reshape(shape), dtype=float)
    def largest_correction(self) -> float:
        """Largest correction over all realisations (NaN for per-sample law parameters)."""
        return float(np.max(self.correction()))