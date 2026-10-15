"""The WARP3D ``deformation`` material's uniaxial law as a library stress-strain object (theory manual,
section 3.6.3): linear-elastic to ``K1 = 0.95`` of the reference stress, a circular transition arc in the
normalised plane ``(eps_e/eps_0, sigma_e/sigma_0)``, then the pure power law ``eps/eps_0 = (sigma/sigma_0)^n``.
The arc is the circle tangent to the linear branch at ``(K1, K1)`` and at first contact with the power branch;
``K2``, the centre and the radius are derived from ``n`` (a construction; the WARP3D manual gives
``K1`` and refers the rest to its reference, Wang's 1991 MIT thesis). That reference has **not been obtained**
(its open-access download needs a browser verification step), so D6's measured arc residual against WARP3D's
one-element response (0.14 % at s = 1.0; section 6.2) is the bound on any difference between this construction and
WARP3D's. The object satisfies the Option 2 ``MaterialLaw`` protocol so that the Option 2 and Option 3 curves
describe one material.

Units: SI (Pa) for ``E`` and ``sigma_0``; the law itself is dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

K1 = 0.95
_SQRT2 = float(np.sqrt(2.0))


def _f(x: ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=float)


def _tangency_residual(r: FloatArray, n: FloatArray, iterations: int = 80) -> tuple[FloatArray, FloatArray]:
    """For a circle of radius ``r`` centred on the inward normal at ``(K1, K1)``, the point ``x2`` of the power
    branch whose downward normal passes through the centre at distance ``r`` along ``x``; returns the residual of
    the ``y`` distance condition and ``x2``. Residual zero means the circle is tangent to the power branch."""
    xc = K1 + r / _SQRT2
    yc = K1 - r / _SQRT2
    # x2 solves x2 = xc - r g / sqrt(1 + g^2), g = (1/n) x2^(1/n - 1): bisection on x2 in [K1^n, xc]
    lo = np.full_like(r, K1) ** n
    hi = xc.copy()
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        with np.errstate(over="ignore", invalid="ignore"):
            g = (1.0 / n) * mid ** (1.0 / n - 1.0)
            f = mid - (xc - r * g / np.sqrt(1.0 + g * g))
        lo = np.where(f < 0.0, mid, lo)
        hi = np.where(f < 0.0, hi, mid)
    x2 = 0.5 * (lo + hi)
    with np.errstate(over="ignore", invalid="ignore"):
        g = (1.0 / n) * x2 ** (1.0 / n - 1.0)
        y2 = x2 ** (1.0 / n)
        residual = (yc + r / np.sqrt(1.0 + g * g)) - y2
    return np.asarray(residual, dtype=float), np.asarray(x2, dtype=float)


def transition_arc(n: ArrayLike, iterations: int = 100) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """``(K2, xc, yc, r)`` of the circular transition arc for hardening exponent ``n`` (vectorised). The radius is
    the unique root of the tangency residual (one sign change on ``(0, r_max)``, checked for
    ``n`` in 3..20); raises if no sign change is found."""
    n_ = _f(n)
    if np.any(~np.isfinite(n_)) or np.any(n_ <= 1.0):
        raise ValueError("the hardening exponent n must be finite and greater than 1")
    shape = n_.shape
    n_b = n_.reshape(-1)
    lo = np.full(n_b.shape, 1e-6)
    hi = np.full(n_b.shape, K1 * _SQRT2 * 0.999)  # the centre's y stays positive
    r_lo, _ = _tangency_residual(lo, n_b)
    r_hi, _ = _tangency_residual(hi, n_b)
    if np.any(np.sign(r_lo) == np.sign(r_hi)):
        raise ValueError("TODO(verify): no tangent transition arc found for some n (no sign change in the residual)")
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        r_mid, _ = _tangency_residual(mid, n_b)
        same = np.sign(r_mid) == np.sign(r_lo)
        lo = np.where(same, mid, lo)
        r_lo = np.where(same, r_mid, r_lo)
        hi = np.where(same, hi, mid)
    r = 0.5 * (lo + hi)
    _, x2 = _tangency_residual(r, n_b)
    k2 = x2 ** (1.0 / n_b)
    xc = K1 + r / _SQRT2
    yc = K1 - r / _SQRT2
    return (
        np.asarray(k2.reshape(shape), dtype=float),
        np.asarray(xc.reshape(shape), dtype=float),
        np.asarray(yc.reshape(shape), dtype=float),
        np.asarray(r.reshape(shape), dtype=float),
    )


_ARC_CACHE: dict[float, tuple[float, float, float, float]] = {}
_ARC_CACHE_MAX = 200_000  # exponents; cleared when exceeded (a Monte Carlo sample of n stays far below)


def _arcs(uniq: FloatArray) -> FloatArray:
    """Arc parameters (K2, xc, yc, r) for distinct exponents, memoised: the uncached ones are solved together in
    one vectorised bisection (the law is evaluated many times per exponent, e.g. in the proof-stress
    bisection, and the arc depends on ``n`` alone)."""
    missing = [float(v) for v in uniq if float(v) not in _ARC_CACHE]
    if missing:
        if len(_ARC_CACHE) + len(missing) > _ARC_CACHE_MAX:
            _ARC_CACHE.clear()
        k2, xc, yc, r = transition_arc(np.asarray(missing))
        for i, v in enumerate(missing):
            _ARC_CACHE[v] = (float(k2[i]), float(xc[i]), float(yc[i]), float(r[i]))
    return np.array([_ARC_CACHE[float(v)] for v in uniq]).reshape(-1, 4)


def _arc_broadcast(n_: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """The arc parameters for a broadcast ``n`` array, solved once per distinct ``n`` (the bisection is the
    costly part; a fine stress grid shares a handful of exponents)."""
    uniq, inverse = np.unique(n_, return_inverse=True)
    params = _arcs(uniq)
    k2u, xcu, ycu, ru = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
    shape = n_.shape
    return (
        k2u[inverse].reshape(shape),
        xcu[inverse].reshape(shape),
        ycu[inverse].reshape(shape),
        ru[inverse].reshape(shape),
    )


def normalised_strain(y: ArrayLike, n: ArrayLike) -> FloatArray:
    """Normalised total strain ``x = eps_e/eps_0`` at normalised stress ``y = sigma_e/sigma_0`` (vectorised):
    linear branch, the lower-left branch of the arc, or the power branch."""
    y_, n_ = np.broadcast_arrays(_f(y), _f(n))
    return _strain_given_arc(y_, n_, *_arc_broadcast(n_))


def _strain_given_arc(
    y_: FloatArray, n_: FloatArray, k2: FloatArray, xc: FloatArray, yc: FloatArray, r: FloatArray
) -> FloatArray:
    """``normalised_strain`` with the arc parameters already resolved (for loops over the stress at fixed n)."""
    with np.errstate(invalid="ignore", over="ignore"):
        arc = xc - np.sqrt(np.maximum(r * r - (y_ - yc) ** 2, 0.0))
        power = y_**n_
    out = np.where(y_ <= K1, y_, np.where(y_ >= k2, power, arc))
    return np.asarray(out, dtype=float)


@dataclass(frozen=True)
class WarpDeformationLaw:
    """WARP3D ``deformation`` law with reference stress ``sigma_0`` (the library's yield strength for this
    family: the stress where the power branch meets the elastic line, not a 0.2 % proof stress), modulus ``E``,
    Poisson's ratio ``nu`` and hardening exponent ``n``. Option 2 ``MaterialLaw`` protocol: ``q(s)`` is the
    plastic strain over ``s`` with ``s = sigma/sigma_0``."""

    E: ArrayLike
    nu: ArrayLike
    sigma_0: ArrayLike
    n: ArrayLike

    def __post_init__(self) -> None:
        e, nu, s0, n = (_f(v) for v in (self.E, self.nu, self.sigma_0, self.n))
        if not (np.all(np.isfinite(e)) and np.all(e > 0.0)):
            raise ValueError("E must be finite and positive")
        if not (np.all(np.isfinite(nu)) and np.all(nu > 0.0) and np.all(nu < 0.5)):
            raise ValueError("nu must lie in (0, 0.5)")
        if not (np.all(np.isfinite(s0)) and np.all(s0 > 0.0)):
            raise ValueError("sigma_0 must be finite and positive")
        if not (np.all(np.isfinite(n)) and np.all(n > 1.0)):
            raise ValueError("n must be finite and greater than 1")
        _arcs(np.unique(_f(n)))  # raises if the arc does not exist (memoised per exponent)

    # ---- Option 2 protocol
    @property
    def step_locations(self) -> tuple[float, ...]:
        return ()

    @property
    def declared_plateau(self) -> bool:
        return False

    @property
    def s_end(self) -> float:
        return float("inf")

    @property
    def parameter_shape(self) -> tuple[int, ...]:
        return tuple(np.broadcast_shapes(*(np.shape(v) for v in (self.E, self.nu, self.sigma_0, self.n))))

    @property
    def eps_0(self) -> FloatArray:
        return np.asarray(_f(self.sigma_0) / _f(self.E), dtype=float)

    def q(self, s: FloatArray) -> FloatArray:
        s_, n_, eps0 = np.broadcast_arrays(_f(s), _f(self.n), self.eps_0)
        x = normalised_strain(s_, n_)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(s_ > 0.0, eps0 * (x / s_ - 1.0), 0.0)
        return np.asarray(np.maximum(out, 0.0), dtype=float)

    def q_hat(self, s: FloatArray) -> FloatArray:
        return self.q(s)  # non-decreasing: zero, then the convex arc, then eps_0 (s^(n-1) - 1)

    def hardened_enough(self, kappa: FloatArray, lr_end: FloatArray) -> BoolArray:
        kappa_, end = np.broadcast_arrays(_f(kappa), _f(lr_end))
        ok = np.ones(kappa_.shape, dtype=bool)
        beyond = end > _SQRT2
        if not np.any(beyond):
            return ok
        t = np.linspace(0.0, 1.0, 200)
        s = _SQRT2 + (np.where(beyond, end, _SQRT2)[..., None] - _SQRT2) * t
        n_ = _f(self.n)
        n_col = n_[..., None] if np.ndim(n_) else n_
        eps0 = self.eps_0
        eps0_col = eps0[..., None] if np.ndim(eps0) else eps0
        x = normalised_strain(s, n_col)
        e = s * (1.0 + kappa_[..., None] * eps0_col * (x / s - 1.0))
        return np.asarray(np.where(beyond, np.all(e >= 1.001 * s**2 / _SQRT2, axis=-1), True), dtype=bool)

    # ---- reported quantities
    def strain(self, stress: ArrayLike) -> FloatArray:
        """Total uniaxial strain at ``stress`` (Pa)."""
        s = _f(stress) / _f(self.sigma_0)
        return np.asarray(self.eps_0 * normalised_strain(s, self.n), dtype=float)

    def proof_stress_0p2(self, iterations: int = 100) -> FloatArray:
        """The implied 0.2 % proof stress (Pa): the stress whose plastic strain ``eps_0 (x(s) - s)`` is 0.002."""
        n_, eps0, s0 = np.broadcast_arrays(_f(self.n), self.eps_0, _f(self.sigma_0))
        arc = _arc_broadcast(n_)  # resolved once; the bisection varies the stress only
        lo = np.ones(n_.shape)
        hi = np.full(n_.shape, 10.0)
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            ep = eps0 * (_strain_given_arc(mid, n_, *arc) - mid)
            lo = np.where(ep < 0.002, mid, lo)
            hi = np.where(ep < 0.002, hi, mid)
        return np.asarray(0.5 * (lo + hi) * s0, dtype=float)

    def considere_stress(self) -> FloatArray:
        """The true stress at the Considère condition ``d sigma / d eps = sigma`` on the power branch (Pa), the
        tensile instability the law implies; reported for comparison with a user-supplied ``uts``, never used."""
        # on the power branch eps = eps_0 s^n: d sigma / d eps = sigma_0 / (n eps_0 s^(n-1)); equals sigma = s sigma_0
        # when s^n = 1 / (n eps_0)
        n_, eps0, s0 = np.broadcast_arrays(_f(self.n), self.eps_0, _f(self.sigma_0))
        s = (1.0 / (n_ * eps0)) ** (1.0 / n_)
        return np.asarray(s * s0, dtype=float)
