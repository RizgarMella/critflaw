"""Gaussian-copula dependence between random variables (theory manual, section 4.2).

x_i = F_i^-1(Phi(z_i)),  z = L u,  u independent standard normal,  L L^T = latent correlation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize

from critflaw.prob.distributions import Distribution, LogNormal

FloatArray = NDArray[np.float64]

_NODES = (64, 128, 256)
_TOL = 1e-8


class CorrelationError(ValueError):
    pass


def _gauss_hermite(n: int) -> tuple[FloatArray, FloatArray]:
    nodes, weights = np.polynomial.hermite_e.hermegauss(n)
    return np.asarray(nodes, dtype=float), np.asarray(weights / np.sqrt(2.0 * np.pi), dtype=float)


def _expect_product(dx: Distribution, dy: Distribution, rho0: float, n: int, mx: float = 0.0, my: float = 0.0) -> float:
    """E[(x(z1) - mx)(y(z2) - my)] for standard normal (z1, z2) with correlation rho0.

    The means are subtracted before multiplying, so a large location does not cause cancellation."""
    a, w = _gauss_hermite(n)
    with np.errstate(over="ignore", invalid="ignore"):
        xc = dx.from_normal_score(a) - mx
        if abs(rho0) >= 1.0:
            return float(np.sum(w * xc * (dy.from_normal_score(np.sign(rho0) * a) - my)))
        z2 = rho0 * a[:, None] + np.sqrt(1.0 - rho0**2) * a[None, :]
        inner = (dy.from_normal_score(z2) - my) @ w
        return float(np.sum(w * xc * inner))


def _log_expm1(x: float) -> float:
    """log(exp(x) - 1) for x > 0 without overflow."""
    return float(x + np.log1p(-np.exp(-x))) if x > 30.0 else float(np.log(np.expm1(x)))


def _resolves_moments(d: Distribution, n: int) -> bool:
    """True if an n-point rule reproduces the marginal's mean and variance to 1e-6."""
    a, w = _gauss_hermite(n)
    with np.errstate(over="ignore", invalid="ignore"):
        x = d.from_normal_score(a)
        mean = float(np.sum(w * x))
        var = float(np.sum(w * (x - mean) ** 2))
        scale = d.std()
        ok = abs(mean - d.mean()) <= 1e-6 * scale and abs(var - scale**2) <= 1e-6 * scale**2
    return bool(ok)


def latent_to_pearson(dx: Distribution, dy: Distribution, rho0: float) -> float:
    """Pearson correlation of the physical variables for a copula coefficient ``rho0``.

    Closed form for a lognormal pair; otherwise Gauss-Hermite quadrature refined until the
    result is stable to 1e-8. Raises :class:`CorrelationError` if it cannot be evaluated
    (extremely heavy tails), never returning an inaccurate value.
    """
    if not -1.0 <= rho0 <= 1.0:
        raise CorrelationError("copula coefficient must lie in [-1, 1]")
    if isinstance(dx, LogNormal) and isinstance(dy, LogNormal):
        # closed form, evaluated in logs so that very large sigma_ln does not overflow
        a, b = dx.sigma_ln, dy.sigma_ln
        log_den = 0.5 * (_log_expm1(a * a) + _log_expm1(b * b))
        if rho0 == 0.0:
            return 0.0
        magnitude = float(np.exp(_log_expm1(abs(rho0) * a * b) - log_den))
        if rho0 > 0.0:
            return magnitude
        return float(np.expm1(rho0 * a * b) / np.exp(log_den)) if log_den < 700.0 else 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        sx, sy = dx.std(), dy.std()
        mx, my = dx.mean(), dy.mean()
    if not (sx > 0.0 and sy > 0.0 and np.isfinite(sx) and np.isfinite(sy)):
        raise CorrelationError("correlation needs marginals with finite, non-zero variance")
    previous = None
    for n in _NODES:
        value = _expect_product(dx, dy, rho0, n, mx, my) / (sx * sy)
        # Stability between refinements is not enough: with very heavy tails the rule can agree
        # with itself and still miss the tail. It must also reproduce each marginal's variance.
        resolved = _resolves_moments(dx, n) and _resolves_moments(dy, n)
        if resolved and previous is not None and abs(value - previous) < _TOL and np.isfinite(value):
            return float(np.clip(value, -1.0, 1.0))
        previous = value
    raise CorrelationError(
        "cannot evaluate the correlation of these marginals to 1e-8 by quadrature "
        "(very heavy tails); use kind='latent' or a closed-form pair"
    )


def attainable_range(dx: Distribution, dy: Distribution) -> tuple[float, float]:
    """Smallest and largest Pearson correlation a Gaussian copula can give these marginals."""
    return latent_to_pearson(dx, dy, -1.0), latent_to_pearson(dx, dy, 1.0)


def pearson_to_latent(dx: Distribution, dy: Distribution, rho: float) -> float:
    """Copula coefficient that gives Pearson correlation ``rho``. The map is monotone."""
    if dx.is_discrete or dy.is_discrete:
        raise CorrelationError("discrete marginals can only be correlated with kind='latent'")
    if rho == 0.0:
        return 0.0
    lo, hi = attainable_range(dx, dy)
    if not lo <= rho <= hi:
        raise CorrelationError(
            f"Pearson correlation {rho:g} is not attainable for these marginals; "
            f"the attainable range is [{lo:.4f}, {hi:.4f}]"
        )
    if rho >= hi:
        return 1.0
    if rho <= lo:
        return -1.0
    root = optimize.brentq(lambda r: latent_to_pearson(dx, dy, r) - rho, -1.0, 1.0, xtol=1e-13, rtol=1e-13)
    return float(root)


class RandomVector:
    """Named random variables with a Gaussian copula."""

    def __init__(self, marginals: Sequence[Distribution]) -> None:
        if len({id(m) for m in marginals}) != len(marginals):
            raise ValueError("each distribution object may appear only once")
        self.marginals = list(marginals)
        self._latent = np.eye(len(self.marginals))
        self._cholesky: FloatArray | None = None

    @property
    def size(self) -> int:
        return len(self.marginals)

    def correlate(
        self, x: Distribution, y: Distribution, rho: float, kind: Literal["pearson", "latent"] = "pearson"
    ) -> RandomVector:
        """Declare dependence. ``kind="pearson"``: correlation of the physical variables (the
        copula coefficient is solved); ``kind="latent"``: the copula coefficient itself."""
        if kind not in ("pearson", "latent"):
            raise ValueError("kind must be 'pearson' or 'latent'")
        i, j = self._index(x), self._index(y)
        if i == j:
            raise ValueError("cannot correlate a variable with itself")
        if not -1.0 < rho < 1.0:
            raise CorrelationError("correlation must lie strictly between -1 and 1")
        if x.std() == 0.0 or y.std() == 0.0:
            raise CorrelationError("a zero-variance variable cannot be correlated")
        rho0 = pearson_to_latent(x, y, rho) if kind == "pearson" else float(rho)
        if abs(rho0) >= 1.0:
            raise CorrelationError("the requested correlation needs a degenerate copula (|rho0| = 1)")
        # Positive definiteness is a property of the whole matrix, so it is checked by ``check()``
        # (called on first use) and not after each pair: a valid matrix can pass through
        # indefinite intermediate states while it is being declared.
        self._latent[i, j] = self._latent[j, i] = rho0
        self._cholesky = None
        return self

    def check(self) -> RandomVector:
        """Raise :class:`CorrelationError` unless the latent correlation matrix is positive definite."""
        if self._cholesky is None:
            try:
                self._cholesky = np.asarray(np.linalg.cholesky(self._latent), dtype=float)
            except np.linalg.LinAlgError as exc:
                raise CorrelationError(
                    "the declared correlations are not jointly attainable: the latent correlation matrix "
                    "is not positive definite"
                ) from exc
        return self

    def _index(self, d: Distribution) -> int:
        for i, m in enumerate(self.marginals):
            if m is d:
                return i
        raise ValueError("distribution is not part of this random vector")

    @property
    def latent_correlation(self) -> FloatArray:
        return np.asarray(self._latent.copy(), dtype=float)

    def to_physical(self, u: ArrayLike) -> FloatArray:
        """Map independent standard normal samples (m, n) to physical values (m, n)."""
        u_ = np.atleast_2d(np.asarray(u, dtype=float))
        if u_.shape[1] != self.size:
            raise ValueError(f"expected {self.size} columns")
        self.check()
        assert self._cholesky is not None
        z = u_ @ self._cholesky.T
        return np.column_stack([m.from_normal_score(z[:, i]) for i, m in enumerate(self.marginals)])
