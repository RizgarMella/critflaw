# %% [markdown]
# # Partial safety factors: published lookup, design check and implied factors
#
# `critflaw.prob.partial_factors.published_partial_factors` is a discrete lookup of published partial factors
# by target failure probability and coefficients of variation: no interpolation, and combinations marked "not
# permitted" raise `NotTabulated`. The factors apply to stated reference values: stress and flaw height at the
# mean, toughness at mean minus one standard deviation, yield strength at mean minus two standard deviations.
# They were derived by first-order analysis for the continuous-yield Option 1 line with independent normal
# stress and flaw size, lognormal or Weibull toughness and lognormal yield strength.
#
# The example:
#
# 1. looks up the factors for a target of 1e-3;
# 2. finds the mean stress at which the factored (design-value) assessment is just acceptable;
# 3. computes the failure probability of the same problem with the corresponding distributions, to see how
#    close the partial-factor design comes to its target;
# 4. derives the partial factors implied by the most likely failure point.
#
# Theory: `docs/manual/theory/04_probabilistic.md` (partial factors). All inputs are **illustrative**.

# %%
import numpy as np

import critflaw as cf
from critflaw import prob
from critflaw.prob.partial_factors import NotTabulated, published_partial_factors

# %% [markdown]
# ## 1. Look up the factors

# %%
TARGET, COV_S, COV_A, COV_K, COV_Y = 1e-3, 0.1, 0.2, 0.2, 0.1
pf_ = published_partial_factors(TARGET, cov_stress=COV_S, cov_flaw_size=COV_A, cov_toughness=COV_K, cov_yield=COV_Y)
print(pf_)
try:
    published_partial_factors(TARGET, cov_stress=0.15, cov_flaw_size=COV_A, cov_toughness=COV_K)
except NotTabulated as exc:
    print("not tabulated:", exc)

# %% [markdown]
# ## 2. Design values and the design-limiting mean stress
#
# Mean values: flaw depth 4 mm (2c = 40 mm fixed), toughness 100 MPa m^0.5, yield strength 380 MPa. The design
# assessment uses factored values; the load reserve factor of the design assessment scales a trial mean stress to
# the one at which the design check is just met.

# %%
A_MEAN, K_MEAN, SY_MEAN = 4e-3, 100e6, 380e6
UTS_RATIO, E = 1.35, 207e9
plate = cf.FlatPlate(thickness=25e-3, width=500e-3)
k_ref = K_MEAN * (1 - COV_K)  # mean - 1 SD
sy_ref = SY_MEAN * (1 - 2 * COV_Y)  # mean - 2 SD
trial_stress = 150e6
design = cf.Assessment(
    plate,
    cf.SurfaceFlaw(a=A_MEAN * pf_.flaw_size, two_c=40e-3),
    cf.Material(
        sy_ref / pf_.yield_strength,
        sy_ref / pf_.yield_strength * UTS_RATIO,
        E,
        k_ref / pf_.toughness,
        yielding="continuous",
    ),
    cf.StressState(pm=trial_stress * pf_.stress),
)
rf = cf.reserve_factor_on_load(design)
assert rf.bounded and rf.unique and not rf.unacceptable_at_zero
S_MEAN = trial_stress * rf.value
print(f"design-limiting mean stress: {S_MEAN / 1e6:.1f} MPa (design check just met)")

# %% [markdown]
# ## 3. The failure probability of the design
#
# The same problem with distributions of the stated families and coefficients of variation. The tensile strength
# is given as a fixed ratio of the (random) yield strength, so the ordering holds in every realisation.

# %%
stress = prob.Truncated(prob.Normal(S_MEAN, COV_S * S_MEAN), lo=0.0, name="stress")
depth = prob.Truncated(prob.Normal(A_MEAN, COV_A * A_MEAN), lo=0.0, hi=20e-3, name="a")
kmat = prob.LogNormal.from_mean_sd(K_MEAN, COV_K * K_MEAN, name="kmat")
sy = prob.LogNormal.from_mean_sd(SY_MEAN, COV_Y * SY_MEAN, name="sy")
problem = cf.Assessment(
    plate,
    cf.SurfaceFlaw(a=depth, two_c=40e-3),
    cf.Material(sy, cf.Ratio(UTS_RATIO), E, kmat, yielding="continuous"),
    cf.StressState(pm=stress),
)
pa = cf.ProbabilisticAssessment(problem)
res = pa.solve(seed=2, n_explore=200_000, replicates=4, samples_per_level=2000)
print(res.summary())
g = res.reliability.guaranteed
print(f"target {TARGET:g}; achieved Pf {res.reliability.pf:.3g} (guaranteed 99% [{g.lower:.3g}, {g.upper:.3g}])")
assert 1e-6 < g.upper and g.lower < 0.1

# %% [markdown]
# ## 4. Implied partial factors at the most likely failure point
#
# At the most likely failure point, each variable's value divided by (load) or into (resistance) its reference
# value is the partial factor implied by this problem. Approximate, as the point is.

# %%
points = pa.most_likely_failure_points(res)
if points:
    refs = {
        "stress": (S_MEAN, "load"),
        "a": (A_MEAN, "load"),
        "kmat": (k_ref, "resistance"),
        "sy": (sy_ref, "resistance"),
    }
    implied = pa.implied_partial_factors(points[0], refs)
    published = {"stress": pf_.stress, "a": pf_.flaw_size, "kmat": pf_.toughness, "sy": pf_.yield_strength}
    print(f"most likely failure point: distance {points[0].distance:.3f} ({points[0].label})")
    for name in refs:
        print(f"  {name:>6}: implied {implied[name]:.3f}   published {published[name]:.3f}")
    assert all(np.isfinite(v) and v > 0 for v in implied.values())

# %% [markdown]
# ## Interpretation
#
# * The partial factors are a calibrated shortcut: a design that just meets the factored check has a failure
#   probability of the order of the target, not exactly the target, because the calibration was made for
#   a generic set of problems (first-order, independent variables, one line).
# * The implied factors show where this particular problem differs from the calibration: a variable whose implied
#   factor is far below the published one contributes little to failure here, and vice versa.
# * The table is used only as tabulated: a coefficient of variation between the tabulated values is refused.
