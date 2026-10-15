"""Phase 5: the GP engine — exact interpolation, the Matérn-5/2 kernel's closed form, LOO against brute-force
refits, calibrated coverage on a smooth function, joint draws matching the posterior, and the OpenTURNS kriging
cross-check (spec S5.12)."""

from __future__ import annotations

import numpy as np
import pytest

from critflaw.option3 import gp as G


def _f(x: np.ndarray) -> np.ndarray:
    return np.sin(3.0 * x[:, 0]) + 0.5 * np.cos(5.0 * x[:, 1]) + x[:, 0] * x[:, 1]


def _sobol(n: int, d: int, seed: int = 1) -> np.ndarray:
    from scipy.stats import qmc

    return np.asarray(qmc.Sobol(d, scramble=True, seed=seed).random(n))


def test_kernel_closed_form_and_symmetry() -> None:
    x = np.array([[0.0, 0.0], [0.3, 0.4]])
    k = G.matern52(x, x, np.array([1.0, 1.0]), 2.0)
    r = 0.5
    expected = 2.0 * (1 + np.sqrt(5) * r + 5 * r * r / 3) * np.exp(-np.sqrt(5) * r)
    assert k[0, 0] == pytest.approx(2.0) and k[0, 1] == pytest.approx(expected) and k[0, 1] == k[1, 0]


def test_interpolates_training_data_and_recovers_a_smooth_function() -> None:
    x = _sobol(32, 2)
    y = _f(x)
    fit = G.fit(x, y, np.zeros(2), np.ones(2))
    m, v = fit.predict(x)
    assert np.max(np.abs(m - y)) < 1e-3 * np.ptp(y) and np.all(v < 1e-4 * np.var(y))
    xt = _sobol(200, 2, seed=7)
    mt, vt = fit.predict(xt)
    err = mt - _f(xt)
    assert np.sqrt(np.mean(err**2)) < 0.05 * np.std(_f(xt))
    z = np.abs(err) / np.sqrt(vt)
    assert 0.85 <= np.mean(z < 1.96) <= 1.0  # roughly calibrated on a smooth function


def test_loo_matches_brute_force_refits_at_fixed_hyperparameters() -> None:
    x = _sobol(16, 2)
    y = _f(x)
    fit = G.fit(x, y, np.zeros(2), np.ones(2))
    mu, var = fit.loo()
    from scipy.linalg import cho_factor, cho_solve

    xs, ys = fit.x, (y - fit.y_mean) / fit.y_scale
    for i in (0, 5, 11):
        keep = np.arange(16) != i
        k = G.matern52(xs[keep], xs[keep], fit.length, fit.variance) + (fit.noise + G.JITTER * fit.variance) * np.eye(
            15
        )
        c = cho_factor(k, lower=True)
        ks = G.matern52(xs[i : i + 1], xs[keep], fit.length, fit.variance)
        m = float((ks @ cho_solve(c, ys[keep]))[0])
        v = float((fit.variance - ks @ cho_solve(c, ks.T))[0, 0]) + fit.noise + G.JITTER * fit.variance
        assert mu[i] == pytest.approx(fit.y_mean + fit.y_scale * m, rel=1e-6, abs=1e-9)
        assert var[i] == pytest.approx(fit.y_scale**2 * v, rel=1e-5)


def test_draws_reproduce_the_posterior_mean_and_covariance() -> None:
    x = _sobol(12, 2)
    fit = G.fit(x, _f(x), np.zeros(2), np.ones(2))
    xt = _sobol(6, 2, seed=3)
    m, cov = fit.predict(xt, full_cov=True)
    d = fit.draw(xt, 20000, np.random.default_rng(0))
    assert np.allclose(d.mean(axis=0), m, atol=4 * np.sqrt(np.max(np.diag(cov)) / 20000) + 1e-9)
    assert np.allclose(np.cov(d.T), cov, atol=0.05 * np.max(np.diag(cov)) + 1e-12)


def test_refusals() -> None:
    with pytest.raises(ValueError):
        G.fit(np.array([[0.5, 0.5]]), np.array([1.0]), np.zeros(2), np.ones(2))
    with pytest.raises(ValueError):
        G.fit(np.array([[0.5, 1.5], [0.2, 0.2]]), np.array([1.0, 2.0]), np.zeros(2), np.ones(2))
    with pytest.raises(ValueError):
        G.fit(np.array([[0.5, 0.5], [0.2, 0.2]]), np.array([1.0, np.nan]), np.zeros(2), np.ones(2))


def test_openturns_kriging_cross_check() -> None:
    ot = pytest.importorskip("openturns")
    x = _sobol(32, 2)
    y = _f(x)
    fit = G.fit(x, y, np.zeros(2), np.ones(2))
    cov = ot.MaternModel([0.5, 0.5], [1.0], 2.5)
    algo = ot.KrigingAlgorithm(ot.Sample(x), ot.Sample(y[:, None]), cov, ot.ConstantBasisFactory(2).build())
    algo.run()
    krig = algo.getResult()
    xt = _sobol(50, 2, seed=11)
    m_ot = np.array(krig.getMetaModel()(ot.Sample(xt))).ravel()
    m, v = fit.predict(xt)
    # independent implementations (OT estimates its own hyperparameters): agreement inside the GP's intervals
    assert np.mean(np.abs(m - m_ot) <= 1.96 * np.sqrt(v) + 1e-6) >= 0.9
    assert np.sqrt(np.mean((m - m_ot) ** 2)) < 0.05 * np.std(y)


def test_likelihood_gradient_matches_finite_differences() -> None:
    from scipy.optimize import approx_fprime

    rng = np.random.default_rng(0)
    x = rng.random((30, 3))
    y = np.sin(3 * x[:, 0]) + x[:, 1]
    th = np.r_[np.log([0.4, 0.7, 1.2]), 0.1, np.log(1e-4)]
    g = G._nll(th, x, y, 1e-8)[1]
    fd = approx_fprime(th, lambda t: G._nll(t, x, y, 1e-8)[0], 1e-6)
    assert np.linalg.norm(g - fd) <= 1e-5 * np.linalg.norm(g)


def test_pathwise_draws_match_the_exact_posterior() -> None:
    """Phase 6 groundwork: RFF prior sample + exact update reproduces the posterior mean and covariance, and one
    draw is a single consistent function (the same values however the points are batched)."""
    x = _sobol(16, 2)
    fit = G.fit(x, _f(x), np.zeros(2), np.ones(2))
    xt = _sobol(8, 2, seed=5)
    m, cov = fit.predict(xt, full_cov=True)
    pw = G.pathwise(fit, 3000, np.random.default_rng(2), n_features=3000)
    d = pw(xt)
    sd = np.sqrt(np.diag(cov))
    assert np.all(np.abs(d.mean(axis=0) - m) <= 5 * sd / np.sqrt(3000) + 1e-3 * np.std(_f(x)))
    # the RFF approximation (error ~ 1/sqrt(F)) plus the sampling error of 3000 draws
    assert np.max(np.abs(np.cov(d.T) - cov)) <= 0.12 * np.max(np.diag(cov))
    assert np.allclose(pw(xt[:3])[:, 1], d[:, 1])  # consistency: batching does not change a draw's value
    # the RFF kernel matches Matern-5/2 (prior covariance between two points)
    from critflaw.option3.gp import matern52

    pw0 = G.pathwise(fit, 1, np.random.default_rng(3), n_features=200000)
    a, b = fit._scale(xt[:1]), fit._scale(xt[1:2])
    k_rff = float((pw0._features(a)[0] @ pw0._features(b)[0].T)[0, 0])
    assert k_rff == pytest.approx(float(matern52(a, b, fit.length, fit.variance)[0, 0]), abs=0.02 * fit.variance)
