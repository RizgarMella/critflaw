"""Random variables, dependence, intervals and the generic FORM/SORM routines."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate, special, stats

from critflaw import prob
from critflaw.prob.dependence import _expect_product

# ---- distributions ---------------------------------------------------------------------


def test_lognormal_parameters_match_the_closed_forms() -> None:
    mean, sd = 120.0, 30.0
    d = prob.LogNormal.from_mean_sd(mean, sd)
    sigma = np.sqrt(np.log(1 + (sd / mean) ** 2))
    assert d.sigma_ln == pytest.approx(sigma, rel=1e-14)
    assert d.mu_ln == pytest.approx(np.log(mean) - 0.5 * sigma**2, rel=1e-14)
    assert d.mean() == pytest.approx(mean, rel=1e-12) and d.std() == pytest.approx(sd, rel=1e-12)
    assert prob.LogNormal(median=380.0, cov=0.07).ppf(0.5) == pytest.approx(380.0, rel=1e-14)


def test_weibull_moments_match_the_gamma_function_relations() -> None:
    theta, k = 150.0, 4.0
    d = prob.Weibull(scale=theta, shape=k)
    mean = theta / k * special.gamma(1 / k)
    sd = np.sqrt(theta**2 / k * (2 * special.gamma(2 / k) - special.gamma(1 / k) ** 2 / k))
    assert d.mean() == pytest.approx(mean, rel=1e-12) and d.std() == pytest.approx(sd, rel=1e-12)
    back = prob.Weibull.from_mean_sd(mean, sd)
    assert back.shape == pytest.approx(k, rel=1e-10) and back.scale == pytest.approx(theta, rel=1e-10)
    shifted = prob.Weibull.from_mean_sd(100.0, 20.0, loc=20.0)
    assert shifted.mean() == pytest.approx(100.0, rel=1e-10) and shifted.std() == pytest.approx(20.0, rel=1e-10)
    assert shifted.support()[0] == 20.0


def test_gumbel_uniform_exponential_moments() -> None:
    g = prob.Gumbel.from_mean_sd(140.0, 14.0)
    assert g.mean() == pytest.approx(140.0, rel=1e-12) and g.std() == pytest.approx(14.0, rel=1e-12)
    assert prob.Uniform(0.1, 0.5).mean() == pytest.approx(0.3)
    assert prob.Exponential(mean=1.0e-3).std() == pytest.approx(1.0e-3)


def test_normal_score_transform_is_tail_safe_and_round_trips() -> None:
    for d in (prob.Normal(10, 2), prob.LogNormal(100, 2.0), prob.Weibull(1.0, 0.5), prob.Gumbel(5, 1), prob.Uniform(2, 3)):
        z = np.array([-14.9, -8.3, -3.0, 0.0, 3.0, 8.3, 14.9])
        x = d.from_normal_score(z)
        assert np.all(np.isfinite(x)) and np.all(np.diff(x) >= 0)
        inner = np.abs(z) <= 8.0
        np.testing.assert_allclose(d.normal_score(x[inner]), z[inner], atol=1e-7)


def test_truncated_distribution() -> None:
    t = prob.Truncated(prob.Normal(1.0, 1.0), lo=0.0)
    ref = stats.truncnorm(a=-1.0, b=np.inf, loc=1.0, scale=1.0)
    assert t.mean() == pytest.approx(ref.mean(), rel=1e-9) and t.std() == pytest.approx(ref.std(), rel=1e-9)
    x = np.array([0.0, 0.3, 1.0, 4.0])
    np.testing.assert_allclose(t.cdf(x), ref.cdf(x), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(t.ppf(t.cdf(x)), x, atol=1e-10)
    assert t.support() == (0.0, np.inf) and t.cdf(-1.0) == 0.0
    # far upper tail truncation keeps relative accuracy
    far = prob.Truncated(prob.Normal(0.0, 1.0), lo=8.0)
    assert far.ppf(0.5) == pytest.approx(stats.norm.isf(0.5 * stats.norm.sf(8.0)), rel=1e-10)
    with pytest.raises(ValueError):
        prob.Truncated(prob.Uniform(0, 1), lo=2.0, hi=3.0)


def test_empirical_distribution_has_real_masses() -> None:
    e = prob.Empirical([2.0, 1.0, 2.0, 5.0])
    assert list(e.values) == [1.0, 2.0, 5.0] and list(e.weights) == [0.25, 0.5, 0.25]
    np.testing.assert_allclose(e.cdf([0.9, 1.0, 1.5, 2.0, 5.0, 9.0]), [0, 0.25, 0.25, 0.75, 1, 1])
    np.testing.assert_allclose(e.ppf([0.0, 0.25, 0.26, 0.75, 0.76, 1.0]), [1, 1, 2, 2, 5, 5])
    assert e.is_discrete and e.mean() == pytest.approx(2.5)
    scores = e.normal_score([1.0, 2.0, 5.0])  # mid-distribution scores are finite
    np.testing.assert_allclose(scores, special.ndtri([0.125, 0.5, 0.875]))
    rng = np.random.default_rng(0)
    assert set(np.unique(e.sample(rng, 2000))) == {1.0, 2.0, 5.0}
    assert prob.Deterministic(3.0).std() == 0.0


def test_distributions_are_compared_by_identity() -> None:
    assert prob.Normal(0, 1) != prob.Normal(0, 1)


# ---- dependence ------------------------------------------------------------------------


def test_normal_pair_latent_equals_pearson() -> None:
    a, b = prob.Normal(0, 1), prob.Normal(5, 3)
    assert prob.pearson_to_latent(a, b, 0.6) == pytest.approx(0.6, abs=1e-9)


def test_lognormal_closed_form_matches_quadrature() -> None:
    a, b = prob.LogNormal(100, 0.3), prob.LogNormal(50, 0.5)
    for rho0 in (-0.9, -0.3, 0.4, 0.95):
        quad = _expect_product(a, b, rho0, 128, a.mean(), b.mean()) / (a.std() * b.std())
        assert prob.latent_to_pearson(a, b, rho0) == pytest.approx(quad, abs=1e-10)


def test_attainable_range_and_infeasible_target() -> None:
    a, b = prob.LogNormal(1.0, 0.1), prob.LogNormal(1.0, 2.0)
    lo, hi = prob.attainable_range(a, b)
    assert lo == pytest.approx(-0.594, abs=1e-3) and hi == pytest.approx(0.675, abs=1e-3)
    with pytest.raises(prob.CorrelationError, match=r"attainable range is \[-0\.59\d+, 0\.67\d+\]"):
        prob.pearson_to_latent(a, b, 0.8)
    assert -1.0 < prob.pearson_to_latent(a, b, 0.5) < 1.0


def test_pearson_target_is_achieved_for_mixed_marginals() -> None:
    sy, k = prob.LogNormal(380e6, 0.07), prob.Weibull.from_mean_sd(120e6, 30e6)
    load = prob.Gumbel.from_mean_sd(140e6, 14e6)
    vec = prob.RandomVector([sy, k, load]).correlate(sy, k, 0.5).correlate(k, load, -0.3)
    x = vec.to_physical(np.random.default_rng(3).standard_normal((400_000, 3)))
    corr = np.corrcoef(x.T)
    assert corr[0, 1] == pytest.approx(0.5, abs=0.006) and corr[1, 2] == pytest.approx(-0.3, abs=0.006)
    assert corr[0, 2] == pytest.approx(0.0, abs=0.006)
    np.testing.assert_allclose(x.mean(axis=0), [sy.mean(), k.mean(), load.mean()], rtol=3e-3)
    # the latent coefficient differs from the Pearson target for non-normal marginals
    assert vec.latent_correlation[0, 1] != pytest.approx(0.5, abs=1e-4)


def test_dependence_errors() -> None:
    a, b, c = prob.Normal(0, 1), prob.Normal(0, 1), prob.Normal(0, 1)
    vec = prob.RandomVector([a, b, c]).correlate(a, b, 0.6).correlate(a, c, 0.6).correlate(b, c, -0.9)
    with pytest.raises(prob.CorrelationError, match="positive definite"):
        vec.check()
    with pytest.raises(prob.CorrelationError, match="positive definite"):
        vec.to_physical(np.zeros((1, 3)))
    # a valid matrix is accepted whatever the order of declaration (it passes through indefinite states)
    for order in ((0, 1, 2), (2, 1, 0), (1, 0, 2)):
        v3 = [prob.Normal(0, 1) for _ in range(3)]
        full = prob.RandomVector(v3)
        pairs = [(0, 1), (0, 2), (1, 2)]
        for k in order:
            full.correlate(v3[pairs[k][0]], v3[pairs[k][1]], 0.9)
        x = full.check().to_physical(np.random.default_rng(0).standard_normal((200_000, 3)))
        assert np.corrcoef(x.T)[0, 1] == pytest.approx(0.9, abs=0.01)
    with pytest.raises(prob.CorrelationError, match="kind='latent'"):
        prob.pearson_to_latent(a, prob.Empirical([1, 2]), 0.5)
    d = prob.Deterministic(1.0)
    with pytest.raises(prob.CorrelationError):
        prob.RandomVector([a, d]).correlate(a, d, 0.5, kind="latent")
    with pytest.raises(prob.CorrelationError, match="cannot evaluate"):
        heavy = prob.LogNormal(1.0, 1e40)  # sigma_ln = 13.6: the rule cannot reproduce its variance
        prob.latent_to_pearson(heavy, prob.Normal(0, 1), 0.5)
    # ... while a lognormal pair of the same kind uses the closed form
    # the reviewer's case: sigma_ln = 20, Pearson 0.5 is attained at a latent coefficient of 0.998267
    big = np.sqrt(np.expm1(400.0))
    r0 = prob.pearson_to_latent(prob.LogNormal(1.0, big), prob.LogNormal(1.0, big), 0.5)
    assert r0 == pytest.approx(0.998267, abs=1e-6)
    with pytest.raises(ValueError):
        prob.RandomVector([a, a])


# ---- intervals -------------------------------------------------------------------------


def test_clopper_pearson_against_the_binomial_distribution() -> None:
    iv = prob.clopper_pearson(7, 1000)
    assert stats.binom.sf(6, 1000, iv.lower) == pytest.approx(0.005, rel=1e-8)
    assert stats.binom.cdf(7, 1000, iv.upper) == pytest.approx(0.005, rel=1e-8)
    zero = prob.clopper_pearson(0, 1_000_000)
    assert zero.lower == 0.0 and zero.upper == pytest.approx(1 - 0.005 ** (1 / 1e6), rel=1e-12)
    assert zero.upper == pytest.approx(5.298e-6, rel=1e-3)
    full = prob.clopper_pearson(50, 50)
    assert full.upper == 1.0 and full.lower == pytest.approx(0.005 ** (1 / 50), rel=1e-12)
    one = prob.clopper_pearson(1, 1)
    assert one.lower == pytest.approx(0.005) and one.upper == 1.0
    for bad in ((0, 0), (5, 3), (-1, 10)):
        with pytest.raises(ValueError):
            prob.clopper_pearson(*bad)


def test_clopper_pearson_coverage_is_at_least_nominal() -> None:
    # includes the case that defeats a one-sided zero-failure bound: N p = 4.61
    for n, p in ((400, 0.03), (1_000_000, 4.61e-6), (1_000_000, 5.2e-6), (50, 0.5)):
        k = np.arange(0, min(80, n) + 1)
        covered = np.array([prob.clopper_pearson(int(i), n).contains(p) for i in k])
        coverage = float(np.sum(stats.binom.pmf(k, n, p) * covered))
        assert coverage >= 0.99 - 1e-9, (n, p, coverage)


def test_empirical_bernstein_half_width() -> None:
    iv = prob.empirical_bernstein(1e-3, 4e-6, 100_000, bound=5.0)
    log_term = np.log(300.0)
    half = np.sqrt(2 * 4e-6 * log_term / 1e5) + 3 * 5.0 * log_term / 1e5
    assert iv.upper - 1e-3 == pytest.approx(half, rel=1e-12) and iv.lower == pytest.approx(max(1e-3 - half, 0.0))
    assert 3 * log_term / 0.2 == pytest.approx(85.6, abs=0.1)  # the "about 86/N" of the design note


# ---- generic FORM / SORM ---------------------------------------------------------------


def test_form_linear_limit_state() -> None:
    a = np.array([3.0, -1.0, 2.0, 0.5])
    beta = 4.2

    def g(u: np.ndarray) -> np.ndarray:
        return beta - u @ a / np.linalg.norm(a)

    r = prob.form(g, 4)
    assert r.beta == pytest.approx(beta, abs=1e-6) and r.pf == pytest.approx(stats.norm.cdf(-beta), rel=1e-5)
    np.testing.assert_allclose(r.design_point, beta * a / np.linalg.norm(a), atol=1e-6)
    np.testing.assert_allclose(r.alpha, -a / np.linalg.norm(a), atol=1e-6)  # u* = -beta alpha


def test_form_correlated_normals_design_point() -> None:
    r_var, s_var = prob.Normal(130.0, 10.0), prob.Normal(100.0, 10.0)
    vec = prob.RandomVector([r_var, s_var]).correlate(r_var, s_var, 0.8)

    def g(u: np.ndarray) -> np.ndarray:
        x = vec.to_physical(u)
        return x[:, 0] - x[:, 1]

    res = prob.form(g, 2)
    assert res.beta == pytest.approx(30 / np.sqrt(40), abs=1e-6)
    np.testing.assert_allclose(vec.to_physical(res.design_point)[0], [115.0, 115.0], atol=1e-5)
    # independent case: resistance has alpha > 0, load alpha < 0 (spec S3.7)
    ind = prob.RandomVector([r_var, s_var])
    res_i = prob.form(lambda u: ind.to_physical(u)[:, 0] - ind.to_physical(u)[:, 1], 2)
    assert res_i.alpha[0] > 0 > res_i.alpha[1]


def test_form_correlated_lognormals_closed_form() -> None:
    r_var, s_var = prob.LogNormal(200.0, 0.15), prob.LogNormal(100.0, 0.25)
    vec = prob.RandomVector([r_var, s_var]).correlate(r_var, s_var, 0.4)
    rho0 = vec.latent_correlation[0, 1]
    assert prob.latent_to_pearson(r_var, s_var, rho0) == pytest.approx(0.4, abs=1e-9)
    sd = np.sqrt(r_var.sigma_ln**2 + s_var.sigma_ln**2 - 2 * rho0 * r_var.sigma_ln * s_var.sigma_ln)
    beta_exact = np.log(2.0) / sd

    def g(u: np.ndarray) -> np.ndarray:
        x = vec.to_physical(u)
        return np.log(x[:, 0] / x[:, 1])

    assert prob.form(g, 2).beta == pytest.approx(beta_exact, abs=1e-6)


def test_form_refuses_a_failed_origin() -> None:
    with pytest.raises(prob.FormNotApplicable, match="origin is in the failure region"):
        prob.form(lambda u: u[:, 0] - 2.0, 1)


@pytest.mark.parametrize("kappa", [-0.1, 0.05, 0.1])
def test_sorm_on_a_quadratic_limit_state(kappa: float) -> None:
    beta = 3.0

    def g(u: np.ndarray) -> np.ndarray:
        return beta - u[:, 0] + 0.5 * kappa * u[:, 1] ** 2

    exact = integrate.quad(lambda v: stats.norm.pdf(v) * stats.norm.sf(beta + 0.5 * kappa * v**2), -12, 12, epsabs=0, epsrel=1e-12)[0]
    res = prob.form(g, 2)
    assert res.beta == pytest.approx(beta, abs=1e-6)
    sorm = prob.sorm_breitung(g, res)
    assert sorm == pytest.approx(stats.norm.cdf(-beta) / np.sqrt(1 + beta * kappa), rel=1e-5)
    assert sorm == pytest.approx(exact, rel=0.02)
    assert abs(sorm - exact) < abs(res.pf - exact)  # the second-order term improves on FORM


def test_sorm_is_refused_when_the_curvature_term_is_not_positive() -> None:
    def g(u: np.ndarray) -> np.ndarray:
        return 3.0 - u[:, 0] - 0.5 * 0.4 * u[:, 1] ** 2

    res = prob.FormResult(3.0, np.array([3.0, 0.0]), np.array([-1.0, 0.0]), float(stats.norm.cdf(-3)), 1, 1)
    with pytest.raises(prob.FormNotApplicable):
        prob.sorm_breitung(g, res)


# ---- added after the phase-2 code reviews ---------------------------------------------------


def test_truncated_far_tails_against_scipy() -> None:
    base = stats.norm(0.0, 1.0)
    for lo, hi in ((6.0, np.inf), (-np.inf, -6.0), (6.0, 7.0), (-7.0, -6.0), (-1.0, 8.0)):
        t = prob.Truncated(prob.Normal(0.0, 1.0), lo=lo, hi=hi)
        ref = stats.truncnorm(lo, hi)
        x = np.linspace(max(lo, -9.0), min(hi, 9.0), 9)[1:-1]
        np.testing.assert_allclose(t.cdf(x), ref.cdf(x), rtol=1e-9, atol=1e-300)
        np.testing.assert_allclose(t.sf(x), ref.sf(x), rtol=1e-9, atol=1e-300)
        q = np.array([1e-12, 1e-6, 0.3])
        # quantiles are checked by their defining identities (scipy's own far-tail isf is only good to 1e-5)
        # Round trips: a probability measured from the far side of the truncated mass is resolved to
        # about 1e-14 absolute (a difference of two nearly equal numbers); from the near side, relatively.
        np.testing.assert_allclose(t.sf(t.isf(q)), q, rtol=1e-9, atol=1e-14)
        np.testing.assert_allclose(t.cdf(t.ppf(q)), q, rtol=1e-9, atol=1e-14)
        if lo >= 6.0 and np.isinf(hi):  # one-sided upper-tail truncation: fully relative accuracy
            np.testing.assert_allclose(t.sf(t.isf(q)), q, rtol=1e-9)
        assert t.mean() == pytest.approx(ref.mean(), rel=1e-8) and t.std() == pytest.approx(ref.std(), rel=1e-7)
    assert base.sf(6.0) < 1e-8  # these really are far-tail cases


def test_moments_and_correlation_do_not_depend_on_a_large_location() -> None:
    shifted = prob.Truncated(prob.Normal(1e9, 1.0), lo=0.0)
    assert shifted.std() == pytest.approx(1.0, rel=1e-7) and shifted.mean() == pytest.approx(1e9, rel=1e-12)
    a, b = prob.Normal(1e9, 1.0), prob.Truncated(prob.Normal(1e9, 2.0), lo=0.0)
    assert prob.latent_to_pearson(a, b, 0.5) == pytest.approx(0.5, abs=1e-7)
    assert prob.pearson_to_latent(a, b, -0.3) == pytest.approx(-0.3, abs=1e-7)


def test_empirical_keeps_tiny_upper_masses() -> None:
    e = prob.Empirical([0.0, 1.0], weights=[1.0, 1e-20])
    assert e.sf(0.0) == pytest.approx(1e-20, rel=1e-12) and e.sf(1.0) == 0.0
    assert e.from_normal_score(10.0) == 1.0  # the normal upper tail at 10 (7.6e-24) is inside the top atom
    assert e.from_normal_score(9.0) == 0.0  # 1.1e-19 is not
    assert e.isf(1e-21) == 1.0 and e.isf(1e-19) == 0.0
    assert np.isfinite(e.normal_score(1.0)) and e.normal_score(1.0) > 9.0
    np.testing.assert_allclose(prob.Empirical([1, 2, 5]).isf(np.array([0.9, 0.5, 0.1, 0.0])), [1, 2, 5, 5])


def test_weibull_with_a_small_coefficient_of_variation() -> None:
    w = prob.Weibull.from_mean_sd(1.0, 0.001)
    assert w.shape == pytest.approx(1282.0, rel=2e-3)
    assert w.mean() == pytest.approx(1.0, rel=1e-9) and w.std() == pytest.approx(0.001, rel=1e-6)
    assert prob.Exponential(mean=3.0, loc=1.0).mean() == pytest.approx(3.0)
    assert prob.Exponential(mean=3.0, loc=1.0).std() == pytest.approx(2.0)


def test_upper_tail_normal_scores_use_the_survival_function() -> None:
    d = prob.LogNormal(100.0, 0.3)
    z = np.array([8.5, 12.0, 20.0, 37.0])
    x = d.from_normal_score(z)
    assert np.all(np.isfinite(x)) and np.all(np.diff(x) > 0)
    np.testing.assert_allclose(d.normal_score(x), z, rtol=1e-9)
    np.testing.assert_allclose(d.normal_score(d.from_normal_score(-z)), -z, rtol=1e-9)


def test_form_returns_the_nearest_design_point() -> None:
    # two branches at distances 3 and 4: the nearer one must be reported
    def g(u: np.ndarray) -> np.ndarray:
        return np.minimum(3.0 - u[:, 0], 4.0 + u[:, 1] * 0.0 + (4.0 - u[:, 1]))

    res = prob.form(lambda u: np.minimum(3.0 - u[:, 0], 4.0 - u[:, 1]), 2, starts=12, seed=3)
    assert res.beta == pytest.approx(3.0, abs=1e-5)
    assert g(res.design_point[None, :])[0] == pytest.approx(0.0, abs=1e-6)
    # g(0) exactly zero: the origin is on the boundary, which is the failure side
    with pytest.raises(prob.FormNotApplicable):
        prob.form(lambda u: u[:, 0], 1)