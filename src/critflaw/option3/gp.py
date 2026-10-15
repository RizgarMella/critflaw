"""Exact Gaussian-process regression (theory manual, section 6.4.4): anisotropic Matérn-5/2 kernel with a
constant mean, hyperparameters (log length scales, log signal variance, log noise variance) by maximum
marginal likelihood with multi-start L-BFGS-B; prediction of the mean, the variance and joint draws. Inputs are
scaled to the unit box by the caller-declared bounds; outputs are standardised internally.

The engine is deliberately small (numpy/scipy only); OpenTURNS kriging is the independent cross-check
(theory manual, section 6.5)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

FloatArray = NDArray[np.float64]

SQRT5 = np.sqrt(5.0)
JITTER = 1e-10  # relative to the signal variance, for numerical positive definiteness only


def matern52(x1: FloatArray, x2: FloatArray, length: FloatArray, variance: float) -> FloatArray:
    """k(x, x') = s2 (1 + sqrt5 r + 5 r^2 / 3) exp(-sqrt5 r), r = || (x - x') / l ||."""
    d = (x1[:, None, :] - x2[None, :, :]) / length
    r = np.sqrt(np.maximum((d * d).sum(axis=2), 0.0))
    return np.asarray(variance * (1.0 + SQRT5 * r + 5.0 * r * r / 3.0) * np.exp(-SQRT5 * r), dtype=float)


@dataclass(frozen=True)
class GPFit:
    lower: FloatArray  # input bounds (the unit-box scaling)
    upper: FloatArray
    x: FloatArray  # scaled training inputs (n, d)
    y_mean: float
    y_scale: float
    length: FloatArray  # in scaled units
    variance: float  # of the standardised output
    noise: float  # of the standardised output
    chol: tuple[FloatArray, bool] = field(repr=False)
    alpha: FloatArray = field(repr=False)  # K^-1 y_std
    y_std: FloatArray = field(repr=False)  # standardised training outputs
    log_marginal_likelihood: float = 0.0

    def _scale(self, x: FloatArray) -> FloatArray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return (x - self.lower) / (self.upper - self.lower)

    def predict(self, x: FloatArray, full_cov: bool = False) -> tuple[FloatArray, FloatArray]:
        """Posterior mean and variance (or the full covariance) of the latent function at ``x`` (original
        units); the noise variance is not included."""
        xs = self._scale(x)
        ks = matern52(xs, self.x, self.length, self.variance)
        mean = ks @ self.alpha
        v = cho_solve(self.chol, ks.T)
        if full_cov:
            cov = matern52(xs, xs, self.length, self.variance) - ks @ v
            return self.y_mean + self.y_scale * mean, (self.y_scale**2) * cov
        var = np.maximum(self.variance - np.einsum("ij,ji->i", ks, v), 0.0)
        return self.y_mean + self.y_scale * mean, (self.y_scale**2) * var

    def draw(self, x: FloatArray, n_draws: int, rng: np.random.Generator) -> FloatArray:
        """Joint posterior draws of the latent function at ``x`` (n_draws, len(x))."""
        mean, cov = self.predict(x, full_cov=True)
        jitter = JITTER * max(float(np.max(np.diag(cov))), 1e-300)
        w, u = np.linalg.eigh(cov + jitter * np.eye(cov.shape[0]))
        root = u * np.sqrt(np.maximum(w, 0.0))
        return np.asarray(mean + rng.standard_normal((n_draws, cov.shape[0])) @ root.T, dtype=float)

    def loo(self) -> tuple[FloatArray, FloatArray]:
        """Leave-one-out predictive mean and variance of the observations (closed form, hyperparameters held
        fixed): y_i - mu_-i = alpha_i / [K^-1]_ii, var_-i = 1 / [K^-1]_ii (noise included); original units."""
        kinv = cho_solve(self.chol, np.eye(self.x.shape[0]))
        d = np.diag(kinv)
        mean_std = self.y_std - self.alpha / d
        return self.y_mean + self.y_scale * mean_std, (self.y_scale**2) / d


def _nll(theta: FloatArray, x: FloatArray, y: FloatArray, noise_floor: float) -> tuple[float, FloatArray]:
    """Negative log marginal likelihood and its analytic gradient in theta = (log l_1..l_d, log s2, log noise):
    d(nll) = 0.5 tr((K^-1 - a a^T) dK), with dk/dlog l_j = s2 (5/3) (1 + sqrt5 r) exp(-sqrt5 r) (d_j / l_j)^2."""
    d = x.shape[1]
    n = x.shape[0]
    length = np.exp(theta[:d])
    variance = float(np.exp(theta[d]))
    noise_raw = float(np.exp(theta[d + 1]))
    noise = noise_raw + noise_floor
    diff = (x[:, None, :] - x[None, :, :]) / length
    r = np.sqrt(np.maximum((diff * diff).sum(axis=2), 0.0))
    e = np.exp(-SQRT5 * r)
    kern = variance * (1.0 + SQRT5 * r + 5.0 * r * r / 3.0) * e
    k = kern + (noise + JITTER * variance) * np.eye(n)
    try:
        c, low = cho_factor(k, lower=True)
    except np.linalg.LinAlgError:
        return 1e25, np.zeros_like(theta)
    a = cho_solve((c, low), y)
    f = float(0.5 * y @ a + np.log(np.diag(c)).sum() + 0.5 * n * np.log(2.0 * np.pi))
    w = cho_solve((c, low), np.eye(n)) - np.outer(a, a)  # K^-1 - a a^T
    g = np.zeros_like(theta)
    common = variance * (5.0 / 3.0) * (1.0 + SQRT5 * r) * e
    for j in range(d):
        g[j] = 0.5 * float(np.sum(w * (common * diff[:, :, j] ** 2)))
    g[d] = 0.5 * float(np.sum(w * kern) + JITTER * variance * np.trace(w))
    g[d + 1] = 0.5 * noise_raw * float(np.trace(w))
    return f, g


def fit(
    x: FloatArray,
    y: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    n_starts: int = 8,
    noise_floor: float = 1e-8,
    seed: int = 0,
) -> GPFit:
    """Fit by maximum marginal likelihood. Bounds: length scales in [0.02, 20] (scaled units), signal variance
    in [1e-3, 1e2], noise variance in [1e-10, 1] (standardised output); ``noise_floor`` is added to the noise
    (the FE outputs are deterministic up to the solver tolerance)."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if x.shape[0] != y.size or x.shape[0] < 2:
        raise ValueError("need at least two training points with one output each")
    if np.any(upper <= lower) or np.any((x < lower - 1e-12) | (x > upper + 1e-12)):
        raise ValueError("training inputs must lie inside the declared bounds")
    if not np.all(np.isfinite(y)):
        raise ValueError("non-finite training output")
    xs = (x - lower) / (upper - lower)
    y_mean = float(y.mean())
    y_scale = float(y.std()) or 1.0
    ys = (y - y_mean) / y_scale
    d = x.shape[1]
    bounds = [(np.log(0.02), np.log(20.0))] * d + [(np.log(1e-3), np.log(1e2)), (np.log(1e-10), np.log(1.0))]
    rng = np.random.default_rng(seed)
    best = None
    starts = [np.r_[np.full(d, np.log(0.5)), 0.0, np.log(1e-6)]]
    for _ in range(n_starts - 1):
        starts.append(np.array([rng.uniform(lo, hi) for lo, hi in bounds]))
    for t0 in starts:
        res = minimize(_nll, t0, args=(xs, ys, noise_floor), method="L-BFGS-B", bounds=bounds, jac=True)
        if best is None or res.fun < best.fun:
            best = res
    assert best is not None
    theta = best.x
    length = np.exp(theta[:d])
    variance = float(np.exp(theta[d]))
    noise = float(np.exp(theta[d + 1])) + noise_floor
    k = matern52(xs, xs, length, variance) + (noise + JITTER * variance) * np.eye(xs.shape[0])
    chol = cho_factor(k, lower=True)
    alpha = cho_solve(chol, ys)
    return GPFit(lower, upper, xs, y_mean, y_scale, length, variance, noise, chol, alpha, ys, -float(best.fun))


@dataclass(frozen=True)
class PathwiseDraws:
    """Function draws of a fitted GP evaluable at any points with mutually consistent values (pathwise
    conditioning, Wilson et al. 2020): a random-Fourier-feature sample of the Matérn-5/2 prior plus the exact
    posterior update, f(.) = f_prior(.) + k(., X) K^-1 (y - f_prior(X) - eps)."""

    fit: GPFit
    omega: FloatArray  # (n_draws, F, d) frequencies in scaled units, fresh per draw (unbiased covariance)
    phase: FloatArray  # (n_draws, F)
    weights: FloatArray  # (n_draws, F)
    update: FloatArray  # (n_draws, n) K^-1 (y - f_prior(X) - eps), standardised

    def _features(self, xs: FloatArray) -> FloatArray:
        """(n_draws, N, F) features of every draw at the scaled points ``xs``."""
        f = self.omega.shape[1]
        arg = np.einsum("nd,kfd->knf", xs, self.omega) + self.phase[:, None, :]
        return np.asarray(np.sqrt(2.0 * self.fit.variance / f) * np.cos(arg), dtype=float)

    def _prior(self, xs: FloatArray) -> FloatArray:
        return np.asarray(np.einsum("knf,kf->kn", self._features(xs), self.weights), dtype=float)

    def __call__(self, x: FloatArray) -> FloatArray:
        """Values of every draw at ``x`` (original units): (n_draws, len(x))."""
        return np.vstack([self.one(i, x) for i in range(self.weights.shape[0])])

    def one(self, i: int, x: FloatArray, chunk: int = 4096) -> FloatArray:
        """Values of draw ``i`` at ``x`` (original units), evaluated in point chunks (memory O(chunk x F))."""
        xs = self.fit._scale(x)
        f = self.omega.shape[1]
        out = np.empty(xs.shape[0])
        amp = np.sqrt(2.0 * self.fit.variance / f)
        for a in range(0, xs.shape[0], chunk):
            xc = xs[a : a + chunk]
            prior = amp * np.cos(xc @ self.omega[i].T + self.phase[i]) @ self.weights[i]
            kxs = matern52(xc, self.fit.x, self.fit.length, self.fit.variance)
            out[a : a + chunk] = prior + kxs @ self.update[i]
        return np.asarray(self.fit.y_mean + self.fit.y_scale * out, dtype=float)


def pathwise(fit: GPFit, n_draws: int, rng: np.random.Generator, n_features: int = 2000) -> PathwiseDraws:
    """Draws for ``fit`` (an approximation: the RFF prior's error decays as 1/sqrt(F); at F = 3000 the posterior
    covariance is reproduced to ~6 % of its largest variance on the test function). Matérn-5/2 spectral
    measure: omega = (z / l) / sqrt(g / 5), z ~ N(0, I), g ~ chi2(5) (a multivariate Student t with 2 nu = 5
    degrees of freedom, scale 1/l)."""
    d = fit.x.shape[1]
    z = rng.standard_normal((n_draws, n_features, d))
    g = rng.chisquare(5.0, (n_draws, n_features, 1))
    omega = (z / fit.length) / np.sqrt(g / 5.0)
    phase = rng.uniform(0.0, 2.0 * np.pi, (n_draws, n_features))
    weights = rng.standard_normal((n_draws, n_features))
    amp = np.sqrt(2.0 * fit.variance / n_features)
    prior_at_x = np.vstack(  # one draw at a time: memory O(n x F), not O(n_draws x n x F)
        [amp * np.cos(fit.x @ omega[i].T + phase[i]) @ weights[i] for i in range(n_draws)]
    )
    eps = np.sqrt(fit.noise + JITTER * fit.variance) * rng.standard_normal(prior_at_x.shape)
    update = cho_solve(fit.chol, (fit.y_std - prior_at_x - eps).T).T
    return PathwiseDraws(fit, omega, phase, weights, np.asarray(update, dtype=float))


__all__ = ["GPFit", "PathwiseDraws", "fit", "matern52", "pathwise"]
