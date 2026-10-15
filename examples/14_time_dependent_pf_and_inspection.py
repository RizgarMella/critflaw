# %% [markdown]
# # Time-dependent failure probability, inspection and repair
#
# `TimeDependentAssessment.solve` grows a random population of flaws through the life and checks each against the
# annual-maximum fracture load, step by step. It reports, per reporting time:
#
# * the cumulative failure probability as a **bracket** `[lower, upper pairing]` (the two ways of pairing the
#   growing flaw with the step maximum load within a step), each with a pointwise 99 % interval;
# * the valid-only curve (realisations outside a solution's validity counted as safe instead of failed);
# * hazards, failure modes and path events (leak, re-characterisation, repair).
#
# An `Inspection` at a step boundary with a probability-of-detection model and a repair policy changes the
# population from then on. `critflaw.prob.inspection.undetected` gives the flaw-size distribution after an
# inspection that found nothing.
#
# Theory: `docs/manual/theory/05_fatigue.md` (time-dependent reliability) and
# `docs/manual/theory/04_probabilistic.md` (inspection updating). All inputs are **illustrative**; sample sizes
# are kept small so the example stays cheap (expect wide intervals).

# %%
from pathlib import Path

import numpy as np

import critflaw as cf
from critflaw import prob
from critflaw.fatigue import ExtremeLoad, Inspection, Spectrum, TimeDependentAssessment, steel_in_air
from critflaw.prob.inspection import LogLogisticPoD, per_component_upper_bound, undetected

SCRIPT = "__file__" in globals()
HERE = (
    Path(__file__).resolve().parent
    if SCRIPT
    else (Path.cwd() / "examples" if (Path.cwd() / "examples").is_dir() else Path.cwd())
)
OUT = HERE / "_output"
OUT.mkdir(exist_ok=True)
try:
    import matplotlib

    if SCRIPT:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # plotting is optional
    plt = None


def save(fig, name):
    fig.savefig(OUT / name, dpi=120, bbox_inches="tight")
    if not SCRIPT:
        plt.show()
    plt.close(fig)


# %% [markdown]
# ## 1. The population: random initial depth, deterministic law, annual-maximum load

# %%
B, W = 25e-3, 0.5
a0 = prob.Truncated(prob.LogNormal.from_mean_sd(1.5e-3, 0.8e-3), hi=10e-3, name="a0")
steel = cf.Material(355e6, 490e6, 207e9, 90e6, yielding="continuous")
problem = cf.Assessment(cf.FlatPlate(B, W), cf.SurfaceFlaw(a=a0, two_c=15e-3), steel, cf.StressState(pm=150e6))
law = steel_in_air()
spectrum = Spectrum.repeating([(70e6, 10e6, 1.0)], cycles_per_year=1e5)
annual_max = ExtremeLoad(prob.Truncated(prob.Gumbel.from_mean_sd(200e6, 20e6), lo=0.0), reference_period_years=1.0)
TIMES = range(1, 11)
SAMPLES = 2000

plain = TimeDependentAssessment(problem, law, spectrum, annual_max, times=TIMES, leak_is_failure=False)
print("dimension of the standard-normal space:", plain.dimension, "(1 time-invariant + 10 annual load draws)")
res = plain.solve(seed=0, samples=SAMPLES)
print(res.summary())

# %% [markdown]
# ## 2. The same population with an inspection at year 5
#
# Log-logistic PoD of the depth (50 % at 3 mm); detected flaws are repaired (removed).

# %%
pod = LogLogisticPoD(a50=3e-3, beta=4.0)
inspected_run = TimeDependentAssessment(
    problem,
    law,
    spectrum,
    annual_max,
    times=TIMES,
    leak_is_failure=False,
    inspections=[Inspection(time=5.0, pod=pod, action="repair")],
)
res_i = inspected_run.solve(seed=0, samples=SAMPLES)
info = res_i.inspections[0]
print(f"P(detect) at year 5 over all samples: {info['p_detect_all']:.3f}; P(repair): {info['p_repair_all']:.3f}")
print(f"events: {res_i.events}")
for i, t in enumerate(res.times):
    print(
        f"t = {t:4.0f}: Pf no inspection [{res.cumulative_lower.value[i]:.3f}, {res.cumulative_upper.value[i]:.3f}]"
        f"   with inspection [{res_i.cumulative_lower.value[i]:.3f}, {res_i.cumulative_upper.value[i]:.3f}]"
    )
for r_ in (res, res_i):
    assert np.all(np.diff(r_.cumulative_upper.value) >= -1e-12)  # cumulative: never decreases
    assert np.all(r_.cumulative_lower.value <= r_.cumulative_upper.value + 1e-12)
# the inspection cannot make things worse (compared through the intervals: the random draws differ)
assert res_i.cumulative_upper.lower[-1] <= res.cumulative_upper.upper[-1]

# %% [markdown]
# ## 3. Updating the flaw-size distribution after a clean inspection
#
# Before any growth: the depth distribution given that the inspection did not detect the flaw, and the
# probability of non-detection. `per_component_upper_bound` turns a per-flaw probability into an upper bound per
# component for a Poisson number of flaws.

# %%
post, p_nd = undetected(a0, pod)
print(
    f"P(not detected) = {p_nd:.3f}; mean depth {a0.mean() * 1e3:.3f} mm -> {post.mean() * 1e3:.3f} mm after a clean inspection"
)
print(f"P(a > 4 mm): prior {float(a0.sf(4e-3)):.4f} -> posterior {float(post.sf(4e-3)):.4f}")
assert post.mean() < a0.mean() and float(post.sf(4e-3)) < float(a0.sf(4e-3))
pf_flaw = float(res.cumulative_upper.value[-1])
print(
    f"per component (2 flaws expected before inspection, P(ND) {p_nd:.3f}, Pf|flaw {pf_flaw:.3g}): "
    f"<= {per_component_upper_bound(2.0, p_nd, pf_flaw):.3g}"
)

if plt is not None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    t = np.asarray(res.times)
    ax.fill_between(
        t, res.cumulative_lower.value, res.cumulative_upper.value, alpha=0.3, label="no inspection: bracket"
    )
    ax.plot(t, res.cumulative_upper.upper, "C0:", lw=1, label="no inspection: 99% upper")
    ax.fill_between(
        t, res_i.cumulative_lower.value, res_i.cumulative_upper.value, alpha=0.3, label="inspection at year 5"
    )
    ax.plot(t, res_i.cumulative_upper.upper, "C1:", lw=1, label="inspection: 99% upper")
    ax.set(xlabel="time (years)", ylabel="cumulative Pf per flaw", title="Time-dependent Pf (illustrative)")
    ax.legend(fontsize=8)
    save(fig, "14_time_dependent.png")

# %% [markdown]
# ## Interpretation
#
# * The cumulative failure probability rises with time as flaws grow; the bracket between the two pairings is the
#   discretisation uncertainty of the annual load steps (refine with `delta` or `solve(refine=True)`).
# * An inspection with repair removes the large flaws that would fail later: the curve flattens after year 5.
# * A clean inspection is information: it shifts the flaw-size distribution towards smaller flaws.
# * With 2000 samples the pointwise intervals are wide; production runs need more samples (and, for rare
#   events, `solve(subset=True)`).
