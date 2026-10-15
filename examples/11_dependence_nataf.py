# %% [markdown]
# # Dependence between random variables (Gaussian copula / Nataf)
#
# Random variables are made dependent through a Gaussian copula (the Nataf transformation):
# `x_i = F_i^-1(Phi(z_i))`, `z = L u`. A correlation can be declared as
#
# * `kind="pearson"`: the correlation of the **physical** variables; the copula coefficient is solved for it
#   (and refused if the pair of marginals cannot attain it);
# * `kind="latent"`: the copula coefficient itself.
#
# Joint attainability (positive definiteness of the latent matrix) is checked when the problem is solved.
# The example shows the attainable range, the Pearson/latent mapping, a check of the realised correlation, and
# the effect of the sign of a toughness-load correlation on the failure probability.
#
# Theory: `docs/manual/theory/04_probabilistic.md` (dependence). All inputs are **illustrative**.

# %%
import numpy as np

import critflaw as cf
from critflaw import prob

# %% [markdown]
# ## 1. Attainable correlations for a pair of marginals

# %%
sy = prob.LogNormal(380e6, 0.07, name="sy")
kmat_w = prob.Weibull.from_mean_sd(110e6, 30e6, name="kmat")
lo, hi = prob.attainable_range(sy, kmat_w)
print(f"attainable Pearson correlation of (sy, kmat): [{lo:.4f}, {hi:.4f}]")
for rho in (-0.5, 0.3, 0.8):
    rho0 = prob.pearson_to_latent(sy, kmat_w, rho)
    back = prob.latent_to_pearson(sy, kmat_w, rho0)
    print(f"Pearson {rho:+.2f} -> latent {rho0:+.5f} -> Pearson {back:+.5f}")
    assert abs(back - rho) < 1e-7
try:
    prob.pearson_to_latent(prob.LogNormal(1.0, 2.0), prob.LogNormal(1.0, 0.1), -0.9)
except prob.CorrelationError as exc:
    print("not attainable:", exc)

# %% [markdown]
# ## 2. A problem with correlated toughness and load
#
# The toughness and the annual-maximum stress are made dependent through a latent coefficient (for instance a
# common temperature effect). We solve the same problem for a negative, zero and positive coefficient.


# %%
def build(rho):
    kmat = prob.LogNormal.from_mean_sd(100e6, 25e6, name="kmat")
    load = prob.Truncated(prob.Gumbel.from_mean_sd(190e6, 19e6), lo=0.0, name="pm")
    steel = cf.Material(355e6, 490e6, 207e9, kmat, yielding="continuous")
    problem = cf.Assessment(cf.FlatPlate(25e-3, 0.5), cf.SurfaceFlaw(6e-3, 60e-3), steel, cf.StressState(pm=load))
    pa = cf.ProbabilisticAssessment(problem)
    if rho != 0.0:
        pa.correlate(kmat, load, rho, kind="latent")
    return pa


results = {}
for rho in (-0.5, 0.0, 0.5):
    out = build(rho).solve(seed=4, n_explore=200_000, replicates=4, samples_per_level=2000)
    results[rho] = out.reliability
    g = out.reliability.guaranteed
    print(
        f"latent rho(kmat, pm) = {rho:+.1f}: Pf = {out.reliability.pf:.3g} ({out.reliability.status}), "
        f"guaranteed [{g.lower:.3g}, {g.upper:.3g}]"
    )
# high load together with high toughness (positive rho) protects; negative rho makes things worse
assert results[0.5].pf < results[0.0].pf < results[-0.5].pf

# %% [markdown]
# ## 3. Check the realised correlation of a declared Pearson coefficient

# %%
pa = build(0.0)
kmat_v, load_v = pa.variables[pa.names.index("kmat")], pa.variables[pa.names.index("pm")]
pa.correlate(kmat_v, load_v, -0.4, kind="pearson")
u = np.random.default_rng(0).standard_normal((100_000, pa.vector.size))
realised = pa.realise(u)
r = np.corrcoef(np.asarray(realised.material.toughness), np.asarray(realised.loads.pm))[0, 1]
print(f"declared Pearson -0.4, realised {r:.4f}; latent coefficient {pa.vector.latent_correlation[0, 1]:+.4f}")
assert abs(r + 0.4) < 0.02

# %% [markdown]
# ## Interpretation
#
# * A Pearson correlation is a property of the physical pair: not every value is attainable for skewed
#   marginals, and the library refuses those rather than silently capping them.
# * The sign and size of a dependence between resistance and load change the failure probability by a large
#   factor; independence is an assumption to be stated, not a default to be forgotten.
# * The copula is Gaussian: tail dependence beyond what it implies is not represented.
