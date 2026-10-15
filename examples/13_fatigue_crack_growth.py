# %% [markdown]
# # Fatigue crack growth of a surface flaw (deterministic)
#
# A surface flaw grows under a cyclic stress spectrum according to a growth law (rate against stress intensity
# factor range, with a threshold), through breakthrough to a through-thickness flaw. `TimeDependentAssessment.grow`
# integrates one deterministic path and checks the fracture assessment at every checkpoint under a fixed
# fracture-check load. The example shows:
#
# * the catalogue laws (`steel_in_air`, `simplified_law`) and their rate curves, in SI units;
# * a constant-amplitude `Spectrum` with membrane and bending ranges;
# * the growth history `a(t)`, `c(t)`, the flaw kind, the breakthrough (leak) time and the first failing
#   checkpoint.
#
# Theory: `docs/manual/theory/05_fatigue.md`. All inputs are **illustrative**.

# %%
from pathlib import Path

import numpy as np

import critflaw as cf
from critflaw import prob
from critflaw.fatigue import ExtremeLoad, Spectrum, TimeDependentAssessment, simplified_law, steel_in_air
from critflaw.fatigue.growth import ENDED, SURFACE, THROUGH
from critflaw.fatigue.laws import dk_to_si

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
# ## 1. Growth laws
#
# Rates are in m/cycle and ranges in Pa m^0.5 internally (the catalogue's N/mm^(3/2) coefficients are converted
# once). `steel_in_air()` is the two-stage upper-bound law for welded joints at high stress ratio, with the
# welded-joint threshold.

# %%
upper = steel_in_air()  # stress_ratio="high", curve="upper"
simple = simplified_law("air")
print(
    f"{upper.name}: threshold {float(upper.threshold) / 1e6:.2f} MPa m^0.5, "
    f"knee at {float(upper.transition()) / 1e6:.2f} MPa m^0.5"
)
print(f"{simple.name}: m = {simple.m}, threshold {float(simple.threshold) / 1e6:.2f} MPa m^0.5")
print("validity:", upper.validity)
dk = np.geomspace(1e6, 100e6, 300)  # Pa m^0.5
rate_upper, rate_simple = upper.rate(dk), simple.rate(dk)
assert np.all(rate_upper[dk < float(upper.threshold)] == 0.0) and np.all(np.diff(rate_upper[rate_upper > 0]) >= 0)
print(f"1 N/mm^1.5 = {float(dk_to_si(1.0)):.4g} Pa m^0.5")

# %% [markdown]
# ## 2. Spectrum, problem and fracture-check load
#
# Constant amplitude: membrane range 60 MPa, bending range 10 MPa, 10^5 cycles per year. The fracture check uses
# a fixed membrane stress of 250 MPa (`grow(load=...)`). The constructor also needs an extreme-load model (used by
# `solve`); a near-deterministic one is given.

# %%
B, W = 25e-3, 0.5
steel = cf.Material(355e6, 490e6, 207e9, 100e6, yielding="continuous")
problem = cf.Assessment(cf.FlatPlate(B, W), cf.SurfaceFlaw(a=2e-3, two_c=10e-3), steel, cf.StressState(pm=250e6))
spectrum = Spectrum.repeating([(60e6, 10e6, 1.0)], cycles_per_year=1e5)
extreme = ExtremeLoad(prob.Uniform(250e6, 250.001e6), reference_period_years=1.0)
YEARS = 30
run = TimeDependentAssessment(problem, upper, spectrum, extreme, times=range(1, YEARS + 1), leak_is_failure=False)
history = run.grow(load=250e6)

# %% [markdown]
# ## 3. The growth history

# %%
kinds = {SURFACE: "surface", THROUGH: "through", ENDED: "ended"}
for year in (0, 5, 10, 15, 20, 25, 30):
    i = int(np.argmin(np.abs(history.times - year)))
    print(
        f"t = {history.times[i]:5.1f} y: a = {history.a[i] * 1e3:6.2f} mm, 2c = {2 * history.c[i] * 1e3:7.2f} mm, "
        f"kind = {kinds.get(int(history.kind[i]), int(history.kind[i]))}, g = {history.g[i]:+.3f}"
    )
print(
    f"breakthrough (leak) at t = {history.t_leak:.2f} y; re-characterised leak at {history.t_leak_assumed:.2f} y; "
    f"through flaw ended at {history.t_ended:.2f} y"
)
print(
    f"first failing checkpoint (or end of validity): {history.first_failure:.2f} y; "
    f"cycles applied: {history.cycles_applied:.3g}"
)
assert np.all(np.diff(history.c) >= -1e-12)
assert history.a[0] == 2e-3 and history.a[-1] > history.a[0]
assert np.isnan(history.first_failure) or history.first_failure > 0.0

# %% [markdown]
# ## 4. Law comparison: the same flaw with the simplified law

# %%
h_simple = TimeDependentAssessment(
    problem, simple, spectrum, extreme, times=range(1, YEARS + 1), leak_is_failure=False
).grow(load=250e6)
print(f"simplified law: first failure {h_simple.first_failure:.2f} y, leak {h_simple.t_leak:.2f} y")

if plt is not None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.loglog(dk / 1e6, rate_upper, label=upper.name)
    ax1.loglog(dk / 1e6, rate_simple, "--", label=simple.name)
    ax1.set(xlabel="stress intensity factor range (MPa m^0.5)", ylabel="da/dN (m/cycle)", title="Growth laws")
    ax1.legend(fontsize=8)
    ax2.plot(history.times, history.a * 1e3, label="depth a (upper law)")
    ax2.plot(history.times, 2 * history.c * 1e3, label="length 2c (upper law)")
    ax2.plot(h_simple.times, h_simple.a * 1e3, ":", label="depth a (simplified law)")
    if np.isfinite(history.first_failure):
        ax2.axvline(history.first_failure, color="tab:red", ls="--", label="first failure (upper law)")
    ax2.set(xlabel="time (years)", ylabel="size (mm)", title="Growth history", yscale="log")
    ax2.legend(fontsize=8)
    save(fig, "13_fatigue.png")

# %% [markdown]
# ## Interpretation
#
# * Growth accelerates as the flaw deepens; the surface flaw breaks through the wall (leak) and continues as a
#   through-thickness flaw until a fracture check fails or the solution's validity ends (both count as the end of
#   life in `first_failure`; whether a leak counts is the declared `leak_is_failure`).
# * The choice of law (two-stage upper bound or simplified) changes the life; the catalogue's statement of
#   applicability travels with each law (`validity`).
# * For random initial flaws, scatter in the law and annual-maximum loads, see example 14.
