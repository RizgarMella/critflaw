# %% [markdown]
# # Reserve factors and the limiting flaw size
#
# Three scalar questions are answered by scanning, not by assuming that acceptability is monotone:
#
# * `reserve_factor_on_load`: the smallest factor on the primary stresses (secondary held) that is not
#   acceptable;
# * `reserve_factor_on_toughness`: the smallest divisor on the fracture toughness that is not acceptable;
# * `limiting_flaw_size`: the smallest flaw (at fixed shape) that is not acceptable.
#
# Each returns a `Crossing` that also says what it could not resolve: whether the crossing is unique, whether
# the answer is below the scanned range, and whether it is a validity limit rather than a predicted failure.
#
# Theory: `docs/manual/theory/01_fad.md` (margins and reserve factors). All inputs are **illustrative**.

# %%
from pathlib import Path

import numpy as np

import critflaw as cf

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


def show(name, c):
    print(
        f"{name}: value = {c.value:.6g}, bounded = {c.bounded}, unique = {c.unique}, "
        f"crossings = {tuple(round(x, 6) for x in c.crossings)}, below_range = {c.below_range}, "
        f"limited_by_validity = {c.limited_by_validity}, scanned {c.range} on {c.points} points"
    )


# %% [markdown]
# ## 1. The problem: a weld toe flaw with an as-welded, relaxing residual stress

# %%
KMAT = 110e6
plate = cf.FlatPlate(thickness=30e-3, width=600e-3)
flaw = cf.SurfaceFlaw(a=5e-3, two_c=30e-3)
steel = cf.Material(380e6, 520e6, 205e9, KMAT, yielding="continuous")
loads = cf.StressState(pm=180e6, pb=30e6, residual=cf.Residual.as_welded(relaxation=True))
problem = cf.Assessment(plate, flaw, steel, loads)
base = problem.run()
print(f"Lr = {float(base.lr):.4f}, Kr = {float(base.kr):.4f}, acceptable = {bool(base.acceptable)}")
assert bool(base.acceptable)

# %% [markdown]
# ## 2. Reserve factor on load
#
# The value is the **not-acceptable** side of the crossing. Just below it (if the crossing is unique) the
# assessment is acceptable.

# %%
rf_load = cf.reserve_factor_on_load(problem)
show("reserve on load", rf_load)
assert rf_load.bounded and not rf_load.unacceptable_at_zero
at = problem.with_(loads=loads.with_(pm=180e6 * rf_load.value, pb=30e6 * rf_load.value)).run()
assert not bool(at.acceptable)
if rf_load.unique:
    below = problem.with_(loads=loads.with_(pm=180e6 * rf_load.value * 0.999, pb=30e6 * rf_load.value * 0.999))
    assert bool(below.run().acceptable)

# %% [markdown]
# With **no secondary stress** the load reserve factor equals the radial margin (a consistency check that holds
# for any curve and any K solution). With secondary stress it does not, because the secondary stress is held.

# %%
primary_only = problem.with_(loads=cf.StressState(pm=180e6, pb=30e6))
rf_p = cf.reserve_factor_on_load(primary_only)
margin_p = float(primary_only.run().radial_margin)
print(f"primary only: load reserve {rf_p.value:.6f} vs radial margin {margin_p:.6f}")
print(f"with residual stress: load reserve {rf_load.value:.4f} vs radial margin {float(base.radial_margin):.4f}")
assert abs(rf_p.value / margin_p - 1.0) < 1e-6

# %% [markdown]
# ## 3. Reserve factor on toughness

# %%
rf_k = cf.reserve_factor_on_toughness(problem)
show("reserve on toughness", rf_k)
if not rf_k.below_range:
    edge = problem.with_(material=steel.with_(toughness=KMAT / rf_k.value)).run()
    print(f"at Kmat / {rf_k.value:.4f}: radial margin = {float(edge.radial_margin):.8f}")
    assert abs(float(edge.radial_margin) - 1.0) < 1e-5

# %% [markdown]
# ## 4. Limiting flaw size at fixed shape (2c / a = 6)

# %%
lim = cf.limiting_flaw_size(problem)
show("limiting depth (m)", lim)
edge_flaw = problem.with_(flaw=cf.SurfaceFlaw(a=lim.value, two_c=6.0 * lim.value)).run(on_invalid="mask")
if lim.limited_by_validity:
    print("the limit is the end of the solution's validity range, not a predicted failure")
    assert not bool(edge_flaw.valid)
else:
    assert bool(edge_flaw.valid) and not bool(edge_flaw.acceptable)
    smaller = lim.value * (1 - 1e-6)
    assert bool(problem.with_(flaw=cf.SurfaceFlaw(a=smaller, two_c=6.0 * smaller)).run().acceptable) or not lim.unique
print(f"limiting depth = {lim.value * 1e3:.3f} mm (current flaw 5 mm)")

# %% [markdown]
# ## 5. Picture: radial margin against flaw depth

# %%
depths = np.linspace(1e-3, 29e-3, 200)
scan = problem.with_(flaw=cf.SurfaceFlaw(a=depths, two_c=6.0 * depths)).run(on_invalid="mask")
if plt is not None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(depths * 1e3, scan.radial_margin, label="radial margin")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.axvline(lim.value * 1e3, color="tab:red", ls="--", label="limiting depth (scan)")
    ax.axvline(5.0, color="tab:green", ls=":", label="current flaw")
    ax.set(xlabel="flaw depth a (mm), 2c = 6a", ylabel="radial margin", ylim=(0, 4), title="Limiting flaw size")
    ax.legend(fontsize=8)
    save(fig, "04_limiting_flaw.png")

# %% [markdown]
# ## Interpretation
#
# * With secondary stress present the load reserve factor and the radial margin differ: the reserve factor
#   holds the residual stress fixed while scaling the primary load (and relaxation changes with the load).
# * The toughness reserve answers "by how much could Kmat be lower"; if plastic collapse governed, no toughness
#   reserve would exist and `below_range` would be True.
# * Always read `unique` and `limited_by_validity`: a limiting size that is the end of the validity range is
#   not a predicted failure, and more than one crossing means the acceptable region is not an interval.
