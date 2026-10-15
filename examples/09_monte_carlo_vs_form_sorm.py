# %% [markdown]
# # Monte Carlo versus FORM and SORM
#
# `critflaw.prob.form` and `sorm_breitung` are generic first- and second-order reliability methods for a
# **smooth, user-supplied** limit state in standard normal space. They are deliberately **not** applied to
# failure-assessment-diagram problems: the FAD failure set is a union of guarded, non-smooth events (two
# crack-front positions, the cut-off, steps, validity limits), for which no applicability rule proved reliable.
# FAD problems are solved by Monte Carlo with a guaranteed interval and, for rare events, subset simulation
# (examples 10-12).
#
# This example uses the textbook linear-elastic limit state of a centre crack in a wide plate (Irwin),
# `g = Kmat - S sqrt(pi a)`, and compares:
#
# * a lognormal case, where the exact reliability index is known in closed form (FORM must reproduce it);
# * a Weibull toughness / Gumbel stress case, where the limit-state surface is curved in standard normal space,
#   so FORM is approximate and SORM (Breitung) corrects it; plain Monte Carlo is the reference.
#
# Theory: `docs/manual/theory/04_probabilistic.md` (FORM, SORM, Monte Carlo intervals). All numbers are
# **illustrative**.

# %%
import numpy as np
from scipy import stats

from critflaw import prob
from critflaw.prob.solvers import Evaluation, monte_carlo

MPA = 1e6


def make_limit_state(vector):
    """g(u) in MPa m^0.5 for an (m, 3) array of standard normals: columns Kmat, S, a."""

    def g(u):
        x = vector.to_physical(u)
        return (x[:, 0] - x[:, 1] * np.sqrt(np.pi * x[:, 2])) / MPA

    def for_monte_carlo(u):
        value = g(u)
        return Evaluation(value, value <= 0.0)

    return g, for_monte_carlo


# %% [markdown]
# ## 1. Lognormal variables: FORM against the closed form
#
# With lognormal `Kmat`, `S` and `a`, `ln Kmat - ln S - 0.5 ln(pi a)` is normal, so
# `beta = (mu_K - mu_S - 0.5 ln(pi) - 0.5 mu_a) / sqrt(s_K^2 + s_S^2 + s_a^2 / 4)`. The failure surface
# `g = 0` is the same set as for the logarithmic form, so FORM on `g` must give this beta.

# %%
kmat = prob.LogNormal(median=100e6, cov=0.20, name="Kmat")
stress = prob.LogNormal(median=200e6, cov=0.10, name="S")
half_length = prob.LogNormal(median=0.012, cov=0.30, name="a")
vec_ln = prob.RandomVector([kmat, stress, half_length])
g_ln, mc_ln = make_limit_state(vec_ln)

num = kmat.mu_ln - stress.mu_ln - 0.5 * np.log(np.pi) - 0.5 * half_length.mu_ln
den = np.sqrt(kmat.sigma_ln**2 + stress.sigma_ln**2 + 0.25 * half_length.sigma_ln**2)
beta_exact = num / den
f_ln = prob.form(g_ln, 3)
mc = monte_carlo(mc_ln, 3, 200_000, seed=7)
print(f"exact beta = {beta_exact:.6f}, FORM beta = {f_ln.beta:.6f}, Pf = {stats.norm.sf(beta_exact):.4g}")
print(
    f"Monte Carlo: {mc.failures} failures in {mc.samples}, Pf = {mc.pf:.4g}, "
    f"99% Clopper-Pearson [{mc.interval.lower:.3g}, {mc.interval.upper:.3g}]"
)
print(
    "design point (physical):",
    np.round(vec_ln.to_physical(f_ln.design_point)[0] / [MPA, MPA, 1e-3], 3),
    "(MPa m^0.5, MPa, mm)",
)
print("alpha (sign: + resistance, - load):", np.round(f_ln.alpha, 3))
assert abs(f_ln.beta - beta_exact) < 1e-4
assert f_ln.alpha[0] > 0 > f_ln.alpha[1]
assert mc.interval.contains(float(stats.norm.sf(beta_exact)))

# %% [markdown]
# ## 2. Weibull toughness and Gumbel stress: FORM, SORM and Monte Carlo
#
# A two-parameter Weibull toughness (mean 100, standard deviation 20 MPa m^0.5) and a Gumbel annual-maximum stress
# truncated at zero. The transformation to standard normal space is nonlinear, the surface is curved, and the
# first-order answer is an approximation without an error bound.

# %%
kmat_w = prob.Weibull.from_mean_sd(100e6, 20e6, name="Kmat")
stress_g = prob.Truncated(prob.Gumbel.from_mean_sd(200e6, 20e6), lo=0.0, name="S")
vec_w = prob.RandomVector([kmat_w, stress_g, half_length])
g_w, mc_w = make_limit_state(vec_w)
f_w = prob.form(g_w, 3, starts=5, seed=0)
try:
    pf_sorm = prob.sorm_breitung(g_w, f_w)
except prob.FormNotApplicable as exc:
    pf_sorm = None
    print("SORM not applicable:", exc)
mc2 = monte_carlo(mc_w, 3, 100_000, seed=11)
print(
    f"FORM: beta = {f_w.beta:.4f}, Pf = {f_w.pf:.4g} (starts converged: {f_w.starts_converged}, "
    f"competing design points: {f_w.competing_design_points})"
)
if pf_sorm is not None:
    print(f"SORM (Breitung): Pf = {pf_sorm:.4g}")
print(
    f"Monte Carlo: Pf = {mc2.pf:.4g}, 99% interval [{mc2.interval.lower:.3g}, {mc2.interval.upper:.3g}] "
    f"({mc2.failures} failures)"
)
assert mc2.failures >= 50
assert 0.5 < f_w.pf / mc2.pf < 2.0, "FORM should be within a factor of two here"
if pf_sorm is not None:
    assert 0.5 < pf_sorm / mc2.pf < 2.0

# %% [markdown]
# ## 3. FORM refuses a failed origin
#
# If the median point already fails, FORM is not used (there is no design point in the usual sense).

# %%
try:
    prob.form(lambda u: -1.0 - u[:, 0], 1)
except prob.FormNotApplicable as exc:
    print("refused:", exc)

# %% [markdown]
# ## Interpretation
#
# * For the lognormal case the limit state is linear in standard normal space and FORM is exact; Monte Carlo
#   agrees within its guaranteed interval.
# * With Weibull and Gumbel marginals FORM is an approximation whose error is unknown in general; SORM corrects
#   for curvature at the design point but not for distant failure regions. Only the Monte Carlo interval is a
#   statement with guaranteed coverage.
# * For FAD problems the library does not offer FORM at all: use `ProbabilisticAssessment.solve`, which gives
#   the guaranteed interval and, when failures are rare, a clearly labelled conditional estimate.
