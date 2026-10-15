"""Cross-check of the probability transform and of generic FORM against OpenTURNS (design note
phase 2, benchmark 19). Skipped when OpenTURNS is not installed (extra ``crosscheck``)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import special

from critflaw import prob
from critflaw.prob.form import form

ot = pytest.importorskip("openturns")


def six_variables() -> tuple[list[prob.Distribution], list[object]]:
    """The marginals of the six-variable plate problem (benchmark 18), in both libraries, each
    parameterised from its own definition."""
    ours: list[prob.Distribution] = [
        prob.LogNormal(5.0e-3, 0.2),  # flaw height: median, coefficient of variation
        prob.Weibull(100e6, 4.0),  # toughness: scale, shape
        prob.Truncated(prob.Normal(355e6, 25e6), lo=0.0),  # yield strength
        prob.Gumbel(150e6, 12e6),  # primary stress: mode, scale
        prob.Exponential(mean=20e6),  # bending stress
        prob.Uniform(0.4, 0.9),  # residual stress as a fraction
    ]
    theirs = [
        ot.LogNormal(float(np.log(5.0e-3)), float(np.sqrt(np.log(1.0 + 0.2**2))), 0.0),
        ot.WeibullMin(100e6, 4.0, 0.0),
        ot.TruncatedDistribution(ot.Normal(355e6, 25e6), 0.0, ot.TruncatedDistribution.LOWER),
        ot.Gumbel(12e6, 150e6),
        ot.Exponential(1.0 / 20e6),
        ot.Uniform(0.4, 0.9),
    ]
    return ours, theirs


LATENT = {(0, 1): 0.3, (1, 2): 0.5, (3, 4): -0.4, (0, 5): 0.2, (2, 3): 0.1}


def test_marginal_moments_agree() -> None:
    ours, theirs = six_variables()
    for a, b in zip(ours, theirs, strict=True):
        assert a.mean() == pytest.approx(b.getMean()[0], rel=1e-9)
        assert a.std() == pytest.approx(b.getStandardDeviation()[0], rel=1e-8)


def test_physical_samples_from_identical_standard_normal_points_agree() -> None:
    ours, theirs = six_variables()
    vector = prob.RandomVector(ours)
    matrix = ot.CorrelationMatrix(6)
    for (i, j), rho in LATENT.items():
        vector.correlate(ours[i], ours[j], rho, kind="latent")
        matrix[i, j] = rho
    joint = ot.JointDistribution(theirs, ot.NormalCopula(matrix))
    inverse = joint.getInverseIsoProbabilisticTransformation()
    rng = np.random.default_rng(0)
    u = rng.standard_normal((4000, 6))
    expected = np.asarray(inverse(ot.Sample(u.tolist())))
    np.testing.assert_allclose(vector.to_physical(u), expected, rtol=1e-10, atol=0.0)
    # Far tails (scores to +-8). The two codes agree except for the Weibull variable deep in its
    # lower tail, where OpenTURNS returns values that are too small, down to exactly 0 (it forms
    # 1 - p). There this library is compared with the closed form x = scale (-ln(1 - Phi(z)))^(1/shape).
    far = np.clip(6.0 * rng.standard_normal((400, 6)), -8.0, 8.0)
    mine, other = vector.to_physical(far), np.asarray(inverse(ot.Sample(far.tolist())))
    columns = [0, 2, 3, 4, 5]
    np.testing.assert_allclose(mine[:, columns], other[:, columns], rtol=1e-9, atol=0.0)
    z = far @ np.linalg.cholesky(vector.latent_correlation).T
    closed = 100e6 * (-special.log1p(-special.ndtr(z[:, 1]))) ** 0.25
    low = z[:, 1] < 0.0
    closed[low] = 100e6 * (-special.log1p(-special.ndtr(z[low, 1]))) ** 0.25
    upper = ~low  # in the upper tail 1 - Phi(z) is formed from the other side: -ln(Phi(-z))
    closed[upper] = 100e6 * (-special.log_ndtr(-z[upper, 1])) ** 0.25
    np.testing.assert_allclose(mine[:, 1], closed, rtol=1e-10, atol=0.0)
    assert bool(np.all(z[~np.isclose(mine[:, 1], other[:, 1], rtol=1e-6, atol=0.0), 1] < -5.0))


def test_pearson_correlation_from_a_copula_coefficient() -> None:
    # OpenTURNS' getCorrelation() for a joint distribution is a coarse numerical integral: it
    # differs from a converged double integral by 1 to 2% here (0.2919 against 0.29657 for the
    # first pair, checked with scipy's dblquad to 1e-15 of this library's value). It is therefore
    # used only as a sanity bound; the accurate reference is a Gauss-Hermite rule written out here.
    ours, theirs = six_variables()
    nodes, weights = np.polynomial.hermite_e.hermegauss(120)
    weights = weights / np.sum(weights)
    z1, z2 = np.meshgrid(nodes, nodes, indexing="ij")
    w = weights[:, None] * weights[None, :]
    for (i, j), rho in LATENT.items():
        x = ours[i].from_normal_score(z1.ravel()).reshape(z1.shape)
        y = ours[j].from_normal_score((rho * z1 + np.sqrt(1.0 - rho**2) * z2).ravel()).reshape(z1.shape)
        reference = float(np.sum(w * (x - ours[i].mean()) * (y - ours[j].mean()))) / (ours[i].std() * ours[j].std())
        mine = prob.latent_to_pearson(ours[i], ours[j], rho)
        assert mine == pytest.approx(reference, abs=1e-7)
        matrix = ot.CorrelationMatrix(2)
        matrix[0, 1] = rho
        joint = ot.JointDistribution([theirs[i], theirs[j]], ot.NormalCopula(matrix))
        assert mine == pytest.approx(joint.getCorrelation()[0, 1], abs=0.01)


def test_generic_form_on_a_smooth_limit_state() -> None:
    # resistance minus load: lognormal resistance, Gumbel load, copula coefficient 0.3
    resistance, load = prob.LogNormal(300.0, 0.1), prob.Gumbel(150.0, 15.0)
    vector = prob.RandomVector([resistance, load]).correlate(resistance, load, 0.3, kind="latent")

    def g(u: np.ndarray) -> np.ndarray:
        x = vector.to_physical(u)
        return np.asarray(x[:, 0] - x[:, 1])

    mine = form(g, 2)

    matrix = ot.CorrelationMatrix(2)
    matrix[0, 1] = 0.3
    marginals = [ot.LogNormal(float(np.log(300.0)), float(np.sqrt(np.log(1.01))), 0.0), ot.Gumbel(15.0, 150.0)]
    joint = ot.JointDistribution(marginals, ot.NormalCopula(matrix))
    function = ot.SymbolicFunction(["r", "s"], ["r - s"])
    event = ot.ThresholdEvent(ot.CompositeRandomVector(function, ot.RandomVector(joint)), ot.Less(), 0.0)
    solver = ot.AbdoRackwitz()
    solver.setMaximumAbsoluteError(1e-12)
    solver.setMaximumRelativeError(1e-12)
    solver.setMaximumResidualError(1e-12)
    solver.setMaximumConstraintError(1e-12)
    solver.setMaximumIterationNumber(1000)
    solver.setStartingPoint(joint.getMean())
    algorithm = ot.FORM(solver, event)
    algorithm.run()
    result = algorithm.getResult()
    assert mine.beta == pytest.approx(result.getHasoferReliabilityIndex(), abs=1e-6)
    np.testing.assert_allclose(vector.to_physical(mine.design_point[None, :])[0],
                               np.asarray(result.getPhysicalSpaceDesignPoint()), rtol=1e-5)
