# %% [markdown]
# # End to end: one flaw from a deterministic check to a time-dependent failure probability
#
# One welded plate with a surface flaw, taken through the whole library in one place:
#
# 1. **Deterministic FAD check** (Option 1): is the flaw acceptable today, and by how much?
# 2. **Reserve factors and limiting flaw size**: how far are we from the boundary?
# 3. **Probabilistic assessment**: toughness and load are random. What is the failure probability?
# 4. **Fatigue**: how does the flaw grow under cyclic service loading?
# 5. **Time-dependent Pf with an inspection**: the probability of failure over the life, with and without an
#    inspection and repair at year 5.
#
# Each step has its own detailed example (01, 04, 10, 13, 14). Theory: `docs/manual/theory/` chapters 01, 04 and 05.
# All numbers are **illustrative** SI inputs (Pa, m, Pa m^0.5), not taken from any standard's worked example.

# %%
import numpy as np

import critflaw as cf
from critflaw import prob
from critflaw.fatigue import ExtremeLoad, Inspection, Spectrum, TimeDependentAssessment, steel_in_air
from critflaw.prob.inspection import LogLogisticPoD

# %% [markdown]
# ## 1. Deterministic check
#
# A 25 mm plate, 500 mm wide, of a C-Mn steel; a 3 mm deep, 20 mm long surface flaw; 150 MPa membrane stress.

# %%
plate = cf.FlatPlate(thickness=25e-3, width=0.5)
flaw = cf.SurfaceFlaw(a=3e-3, two_c=20e-3)
steel = cf.Material(yield_strength=355e6, uts=490e6, modulus=207e9, toughness=90e6, yielding="continuous")
problem = cf.Assessment(plate, flaw, steel, cf.StressState(pm=150e6))

r = problem.run()
print(f"Lr = {float(r.lr):.3f}, Kr = {float(r.kr):.3f}, acceptable: {bool(r.acceptable)}")
print(f"radial margin (load factor to the boundary, proportional loading): {float(r.radial_margin):.2f}")
assert bool(r.acceptable)

# %% [markdown]
# ## 2. Reserve factors and limiting flaw size

# %%
on_load = cf.reserve_factor_on_load(problem)
on_k = cf.reserve_factor_on_toughness(problem)
limit = cf.limiting_flaw_size(problem)
print(f"reserve factor on load: {on_load.value:.2f}")
print(f"reserve factor on toughness: {on_k.value:.2f}")
print(f"limiting flaw depth at this shape: {limit.value * 1e3:.2f} mm (limited by validity: {limit.limited_by_validity})")
assert on_load.value > 1.0 and limit.value > 3e-3

# %% [markdown]
# ## 3. Probabilistic assessment
#
# Toughness lognormal (mean 90, sd 15 MPa m^0.5) and the extreme membrane stress Gumbel (mean 150, sd 15 MPa).
# `solve` reports a guaranteed Monte Carlo interval and, for small probabilities, a subset-simulation estimate
# labelled *conditional*.

# %%
kmat = prob.LogNormal.from_mean_sd(90e6, 15e6, name="kmat")
pm = prob.Truncated(prob.Gumbel.from_mean_sd(150e6, 15e6), lo=0.0, name="pm")
random_problem = problem.with_(
    material=steel.with_(toughness=kmat), loads=cf.StressState(pm=pm)
)
pa = cf.ProbabilisticAssessment(random_problem)
res = pa.solve(seed=0, n_explore=20_000, replicates=4, samples_per_level=2000)
print(res.summary())
rel = res.reliability
assert 0.0 <= rel.guaranteed.lower <= rel.guaranteed.upper < 0.1

# %% [markdown]
# ## 4. Fatigue crack growth
#
# Constant-amplitude membrane range 70 MPa at 10^5 cycles per year with the two-stage upper-bound law for steel
# in air. `grow` follows the flaw through the life and checks it against a fixed fracture load.

# %%
law = steel_in_air()
spectrum = Spectrum.repeating([(70e6, 0.0, 1.0)], cycles_per_year=1e5)
annual_max = ExtremeLoad(pm, reference_period_years=1.0)
years = range(1, 21)
history = TimeDependentAssessment(problem, law, spectrum, annual_max, times=years, leak_is_failure=False).grow(
    load=150e6
)
for year in (0, 5, 10, 15, 20):
    i = int(np.argmin(np.abs(history.times - year)))
    print(f"t = {history.times[i]:4.1f} y: a = {history.a[i] * 1e3:5.2f} mm, 2c = {2 * history.c[i] * 1e3:6.2f} mm")
print(f"first failing checkpoint: {history.first_failure:.2f} y (NaN: none within {max(years)} years)")
assert history.a[-1] >= history.a[0]

# %% [markdown]
# ## 5. Time-dependent failure probability, with and without an inspection
#
# The initial depth is now random (lognormal, mean 1.5 mm), the toughness random and the annual maximum load
# Gumbel. An inspection at year 5 with a log-logistic probability of detection (50 % at 3 mm) repairs what it finds.
# Sample sizes are small to keep the example quick, so the intervals are wide.

# %%
a0 = prob.Truncated(prob.LogNormal.from_mean_sd(1.5e-3, 0.8e-3), hi=10e-3, name="a0")
population = problem.with_(flaw=cf.SurfaceFlaw(a=a0, two_c=15e-3), material=steel.with_(toughness=kmat))
common = dict(times=range(1, 11), leak_is_failure=False)
no_insp = TimeDependentAssessment(population, law, spectrum, annual_max, **common).solve(seed=0, samples=2000)
insp = TimeDependentAssessment(
    population,
    law,
    spectrum,
    annual_max,
    inspections=[Inspection(time=5.0, pod=LogLogisticPoD(a50=3e-3, beta=4.0), action="repair")],
    **common,
).solve(seed=0, samples=2000)
for i, t in enumerate(no_insp.times):
    print(
        f"t = {t:4.0f} y: Pf no inspection [{no_insp.cumulative_lower.value[i]:.4f}, {no_insp.cumulative_upper.value[i]:.4f}]"
        f"   with inspection [{insp.cumulative_lower.value[i]:.4f}, {insp.cumulative_upper.value[i]:.4f}]"
    )
assert np.all(np.diff(no_insp.cumulative_upper.value) >= -1e-12)
assert insp.cumulative_upper.lower[-1] <= no_insp.cumulative_upper.upper[-1]

# %% [markdown]
# ## Reading the results
#
# * Today the flaw is acceptable with a reserve on load above 1; the limiting depth says how much growth can be
#   tolerated at this shape.
# * With random toughness and load the failure probability is small but not zero; the guaranteed interval is the
#   statement to rely on, the subset estimate is conditional.
# * Fatigue grows the flaw over the life, so the failure probability rises with time; an inspection with repair at
#   year 5 removes the largest flaws and flattens the curve.
# * Every result carries a trace (`r.trace.report()`) listing the equations used, their source references and
#   their verification status, so an assessment can be audited.
