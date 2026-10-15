# %% [markdown]
# # Fracture toughness: conversions, Master Curve and Charpy correlations
#
# `critflaw.toughness` takes and returns SI units (Pa, m, Pa m^0.5, J/m^2, joules, degrees C). This example:
#
# * converts J and CTOD to `Kmat`;
# * evaluates the Master Curve at several probabilities and crack-front lengths;
# * estimates `T0` and a toughness from Charpy data, with the lower-bound and upper-limit correlations;
# * derives a characteristic value from a small data set (minimum-of-three equivalent, tolerance bound);
# * uses a `MasterCurve` with a random `T0` and a uniform probability inside a probabilistic assessment, where
#   the toughness of each realisation is computed from the crack-front length of the flaw being assessed.
#
# Theory: `docs/manual/theory/03_materials_toughness.md`. All inputs are **illustrative**.

# %%
from pathlib import Path

import numpy as np

import critflaw as cf
from critflaw import prob
from critflaw import toughness as T

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


MPA = 1e6

# %% [markdown]
# ## 1. Conversions from J and CTOD

# %%
E = 207e9
k_j = float(T.k_from_j(j_mat=60e3, modulus=E))  # J = 60 kJ/m^2
k_d = float(T.k_from_ctod(ctod=0.15e-3, test_yield=420e6, test_uts=560e6, modulus=E))  # CTOD = 0.15 mm
print(f"K from J = 60 kJ/m^2: {k_j / MPA:.1f} MPa m^0.5")
print(
    f"K from CTOD = 0.15 mm (constraint factor {float(T.ctod_constraint_factor(420e6, 560e6)):.3f}): {k_d / MPA:.1f} MPa m^0.5"
)
assert abs(k_j - np.sqrt(E * 60e3 / (1 - 0.3**2))) < 1e-6 * k_j

# %% [markdown]
# ## 2. The Master Curve
#
# `master_curve(probability, temperature, t0, crack_front_length, thickness)`: the front length is capped at twice
# the thickness, and the distribution has a lower bound (`T.K_MIN_MASTER_CURVE`).

# %%
temps = np.linspace(-120.0, 20.0, 141)
T0 = -60.0
front, thick = 25e-3, 25e-3
curves = {p: T.master_curve(p, temps, T0, front, thick) for p in (0.05, 0.5, 0.95)}
print(f"at T = T0, 25 mm front: median {float(T.master_curve(0.5, T0, T0, front, thick)) / MPA:.1f} MPa m^0.5")
long_front = float(T.master_curve(0.5, -40.0, T0, 200e-3, 100e-3))
short_front = float(T.master_curve(0.5, -40.0, T0, 25e-3, 100e-3))
print(
    f"median at -40 C: 25 mm front {short_front / MPA:.1f}, 200 mm front {long_front / MPA:.1f} MPa m^0.5 (size effect)"
)
assert long_front < short_front
assert np.all(curves[0.05] >= T.K_MIN_MASTER_CURVE) and np.all(np.diff(curves[0.5]) > 0)

# %% [markdown]
# ## 3. Charpy correlations

# %%
t27 = -30.0  # 27 J transition temperature, degrees C
t0_est = float(T.t0_from_charpy(t27, energy=27))
print(f"T0 from T27J = {t27} C: {t0_est:.1f} C (standard deviation {T.T0_CHARPY_STANDARD_DEVIATION} C)")
k_charpy_mc = float(T.charpy_master_curve(temperature=-20.0, t0=t0_est, thickness=30e-3))
k_lower = float(T.charpy_lower_bound(charpy_energy=40.0, thickness=30e-3))
k_cap = float(T.charpy_upper_limit(charpy_energy=40.0, yield_strength=355e6))
k_us = float(T.charpy_upper_shelf(upper_shelf_energy=150.0, modulus=E))
shift = float(T.subsize_charpy_temperature_shift(specimen_thickness=5e-3))
print(f"Charpy Master Curve estimate at -20 C, B = 30 mm: {k_charpy_mc / MPA:.1f} MPa m^0.5")
print(
    f"lower-bound correlation, 40 J, B = 30 mm: {k_lower / MPA:.1f}; upper limit for 40 J: {k_cap / MPA:.1f} MPa m^0.5"
)
print(f"upper-shelf initiation toughness from 150 J: {k_us / MPA:.1f} MPa m^0.5")
print(f"transition temperature shift of a 5 mm sub-size specimen: {shift:.1f} C")
assert k_lower > 0 and shift > 0

# %% [markdown]
# ## 4. A characteristic value from a small set of results

# %%
results = np.array([92.0, 118.0, 105.0, 131.0, 99.0, 87.0, 124.0]) * MPA  # seven tests (illustrative)
mote = T.minimum_of_three_equivalent(results)
lb = T.lower_bound_toughness(results, lognormal=True)
print(f"minimum-of-three equivalent: {mote / MPA:.1f} MPa m^0.5 (rank {int(np.searchsorted(np.sort(results), mote)) + 1} of 7)")
print(f"lower tolerance bound, default quantile and confidence (lognormal): {lb / MPA:.1f} MPa m^0.5, k = {T.tolerance_factor(7):.3f}")
assert mote in results and mote <= np.median(results) and lb < results.mean()

# %% [markdown]
# ## 5. A Master Curve toughness in a probabilistic assessment
#
# `MasterCurve(t0, temperature, probability)` puts the toughness model in the problem; the engine computes `Kmat`
# per realisation from the flaw's own crack-front length (here 2c = 30 mm). `T0` is uncertain (a Charpy-based
# estimate, with the correlation's standard deviation `T.T0_CHARPY_STANDARD_DEVIATION`) and the probability is
# `Uniform(0, 1)`. We compare three service temperatures. `n_explore` is kept small (the example must stay cheap).

# %%
plate = cf.FlatPlate(thickness=30e-3, width=600e-3)
flaw = cf.SurfaceFlaw(a=5e-3, two_c=30e-3)
pf = {}
for temperature in (-50.0, -30.0, -10.0):
    t0 = prob.Normal(t0_est, T.T0_CHARPY_STANDARD_DEVIATION, name="T0")
    p_mc = prob.Uniform(0.0, 1.0, name="p_mc")
    mat = cf.Material(
        380e6, 520e6, 205e9, cf.MasterCurve(t0=t0, temperature=temperature, probability=p_mc), yielding="continuous"
    )
    problem = cf.Assessment(plate, flaw, mat, cf.StressState(pm=200e6, residual=cf.Residual.as_welded(relaxation=True)))
    res = cf.ProbabilisticAssessment(problem).solve(seed=1, n_explore=100_000, replicates=4, samples_per_level=2000)
    g = res.reliability.guaranteed
    pf[temperature] = res.reliability.pf
    print(
        f"T = {temperature:5.1f} C: Pf = {res.reliability.pf:.3g} ({res.reliability.status}); "
        f"guaranteed 99% interval [{g.lower:.3g}, {g.upper:.3g}]"
    )
assert pf[-50.0] >= pf[-10.0]

if plt is not None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for p, k in curves.items():
        ax.plot(temps, k / MPA, label=f"Master Curve, P = {p}")
    ax.axhline(T.K_MIN_MASTER_CURVE / MPA, color="0.5", lw=0.8, ls=":")
    ax.set(xlabel="temperature (C)", ylabel="K (MPa m^0.5)", title=f"Master Curve, T0 = {T0} C, 25 mm front")
    ax.legend(fontsize=8)
    save(fig, "08_master_curve.png")

# %% [markdown]
# ## Interpretation
#
# * The Master Curve gives a distribution, not a single value: the 5 % and 95 % curves differ by a factor of
#   about two in the transition region, and longer crack fronts have lower toughness (weakest-link size effect).
# * Charpy-based estimates carry the scatter of the correlation (here through a random `T0`); they are
#   estimates for screening, not substitutes for fracture tests.
# * In the probabilistic assessment the failure probability falls quickly as the service temperature rises
#   through the transition region.
