# %% [markdown]
# # Rare failure probabilities: subset simulation and importance sampling
#
# `ProbabilisticAssessment.solve` always starts with plain Monte Carlo, which gives a **guaranteed** 99 %
# Clopper-Pearson interval whatever the limit state. If Monte Carlo sees fewer than 10 failures, subset
# simulation (independent replicates) gives a **conditional** estimate: valid only if the driving variable led
# the sampler to every failure region that matters, which cannot be verified from the samples. The headline
# status says which statement you have.
#
# Importance sampling (`critflaw.prob.importance.importance_sampling`) is an optional refinement: a defensive
# mixture centred on the most likely failure point. It reports an asymptotic interval (conditional on the
# proposal covering every failure region) and an empirical-Bernstein interval that holds regardless.
#
# Theory: `docs/manual/theory/04_probabilistic.md`. All inputs are **illustrative**.

# %%
import numpy as np
from scipy import stats

import critflaw as cf
from critflaw import prob
from critflaw.prob.importance import importance_sampling

# %% [markdown]
# ## 1. A problem with a small failure probability
#
# A tough steel and a moderate annual-maximum stress: the failure probability is below what 20 000 Monte Carlo
# samples can resolve.

# %%
kmat = prob.LogNormal.from_mean_sd(100e6, 15e6, name="kmat")
load = prob.Truncated(prob.Gumbel.from_mean_sd(150e6, 10e6), lo=0.0, name="pm")
steel = cf.Material(355e6, 490e6, 207e9, kmat, yielding="continuous")
problem = cf.Assessment(cf.FlatPlate(25e-3, 0.5), cf.SurfaceFlaw(4e-3, 40e-3), steel, cf.StressState(pm=load))
pa = cf.ProbabilisticAssessment(problem)
print("random variables:", pa.names)

# %% [markdown]
# ## 2. Monte Carlo first, subset simulation when failures are rare

# %%
res = pa.solve(seed=0, n_explore=20_000, replicates=4, samples_per_level=2000)
print(res.summary())
rel = res.reliability
assert rel.guaranteed.lower <= rel.guaranteed.upper < 1e-2
if rel.subset is not None:
    print(
        "replicate estimates:",
        [f"{v:.3g}" for v in rel.subset.replicate_estimates],
        f"evaluations: {rel.subset.evaluations}",
    )

# %% [markdown]
# ## 3. The most likely failure point and importance sampling around it
#
# The failure samples (from the channel that gave the headline) are clustered by failure region; each region
# yields an approximate most likely failure point. We centre the importance-sampling proposal there.

# %%
points = pa.most_likely_failure_points(res)
for p in points:
    print(
        f"region share {p.share:.2f}: distance {p.distance:.3f}, physical {dict(zip(pa.names, np.round(p.physical, -3), strict=True))}"
        f" ({p.label})"
    )
if points:
    centres = np.vstack([p.u for p in points])
    imp = importance_sampling(pa.limit_state("failed"), pa.vector.size, centres, samples=20_000, seed=3)
    ci = imp.conditional_interval
    print(
        f"importance sampling: Pf = {imp.pf:.3g}, CoV = {imp.cov:.3f}, effective failures = {imp.effective_failures:.0f}"
    )
    if ci is not None:
        print(f"  conditional (asymptotic) 99% interval [{ci.lower:.3g}, {ci.upper:.3g}]: {ci.kind}")
    g = imp.guaranteed_interval
    print(f"  guaranteed 99% interval [{g.lower:.3g}, {g.upper:.3g}]: {g.kind}")
    print(f"  share from the defensive (nominal) component: {imp.defensive_share:.3f}")
    print(f"  FORM-like index from the point: Phi(-distance) = {stats.norm.sf(points[0].distance):.3g}")
    # consistency checks between independent channels
    assert imp.guaranteed_interval.lower <= rel.guaranteed.upper
    if rel.status == "conditional" and imp.pf > 0:
        assert 1 / 3 < imp.pf / rel.pf < 3

# %% [markdown]
# ## 4. Which variables are associated with failure
#
# `association` ranks variables by how far their normal scores among the failure samples depart from the
# standard normal (a divergence). It is association, not causation.

# %%
for a in pa.association(res):
    print(f"{a.name:>6}: mean shift {a.mean_shift:+.2f}, spread {a.spread:.2f}, divergence {a.divergence:.2f}")

# %% [markdown]
# ## Interpretation
#
# * Plain Monte Carlo with 20 000 samples cannot resolve a probability this small: it gives only an upper
#   bound. Subset simulation gives an estimate, labelled *conditional*, that must be read together with
#   that guaranteed bound.
# * Importance sampling centred on the most likely failure point agrees with subset simulation here; its
#   guaranteed (Bernstein) interval is honest but wide at small probabilities, its asymptotic interval narrow but
#   conditional on the proposal covering every failure region.
# * Low toughness and high load are both associated with failure, as expected.
