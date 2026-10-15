# %% [markdown]
# # Option 2: a material-specific assessment line from a stress-strain curve
#
# Option 2 builds the assessment line from the material's own stress-strain behaviour. The library offers two
# material laws:
#
# * `PowerLaw(offset, n)`: plastic strain `offset * (stress/yield)^n` (the offset must be chosen explicitly:
#   0.002 for a 0.2 % proof strength, 0 for a lower yield point), with `hardening_exponent_lower_bound` giving a
#   lower-bound `n` from the yield-to-tensile ratio;
# * `TabulatedLaw.from_engineering` / `.from_true`: a measured curve, cleaned (sorted, made monotone, elastic
#   range detected) with every cleaning step recorded in `notes`.
#
# The example compares Option 1 and the two Option 2 lines and runs the same flaw with each.
#
# Theory: `docs/manual/theory/01_fad.md` (Options 1 and 2) and `docs/manual/theory/03_materials_toughness.md`
# (material laws).
#
# The "measured" curve below is **synthetic and illustrative** (generated from a Ramberg-Osgood-type formula),
# not test data.

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


# %% [markdown]
# ## 1. Material laws

# %%
SY, SU, E = 380e6, 520e6, 205e9  # Pa (illustrative)
n_lb = float(cf.hardening_exponent_lower_bound(SY, SU))
power = cf.PowerLaw(offset=0.002, n=n_lb)  # yield strength taken as a 0.2 % proof strength
print(f"lower-bound hardening exponent n = {n_lb:.2f}")

# Synthetic "measured" curve: true strain = s/E + 0.002 (s/SY)^9, converted to engineering values as a
# tensile test would report them. Dense around yield, extending well past the collapse cut-off.
s_true = SY * np.concatenate(
    [np.linspace(0.0, 0.9, 10, endpoint=False), np.linspace(0.9, 1.1, 21, endpoint=False), np.linspace(1.1, 1.5, 21)]
)
eps_true = s_true / E + 0.002 * (s_true / SY) ** 9
eps_eng = np.expm1(eps_true)
s_eng = s_true / (1.0 + eps_eng)
tabulated = cf.TabulatedLaw.from_engineering(s_eng, eps_eng, yield_strength=SY, modulus=E, yielding="continuous")
print("cleaning notes:")
for note in tabulated.notes:
    print("  -", note)

# %% [markdown]
# ## 2. The three assessment lines
#
# `Option2Curve(law, yield, uts, modulus)` is the line for a law; the plastic collapse cut-off is the same as
# for Option 1. Every line starts at `f(0) = 1` and is non-increasing.

# %%
opt1 = cf.Option1Curve(SY, SU, E, "continuous")
opt2_power = cf.Option2Curve(power, SY, SU, E)
opt2_tab = cf.Option2Curve(tabulated, SY, SU, E)
lr_grid = np.linspace(0.0, 1.25, 500)
for name, c in (("Option 1", opt1), ("Option 2 power law", opt2_power), ("Option 2 tabulated", opt2_tab)):
    values = c(lr_grid)
    print(f"{name:>20}: f(0.5) = {float(c(0.5)):.4f}, f(1.0) = {float(c(1.0)):.4f}, Lr,max = {float(c.lr_max):.4f}")
    assert abs(float(c(0.0)) - 1.0) < 1e-12
    assert np.all(np.diff(values) <= 1e-12) and np.all(values[lr_grid >= float(c.lr_max)] == 0.0)
print(f"monotonicity correction of the tabulated line: {opt2_tab.largest_correction():.4f}")
print(f"tabulated data end before the cut-off: {bool(opt2_tab.ends_at_data)}")

if plt is not None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(lr_grid, opt1(lr_grid), label="Option 1 (continuous yielding)")
    ax.plot(lr_grid, opt2_power(lr_grid), label=f"Option 2, power law n = {n_lb:.1f}")
    ax.plot(lr_grid, opt2_tab(lr_grid), "--", label="Option 2, tabulated (synthetic data)")
    ax.set(xlabel="Lr", ylabel="Kr", ylim=(0, 1.05), title="Option 1 and Option 2 lines (illustrative)")
    ax.legend(fontsize=8)
    save(fig, "02_option2_lines.png")

# %% [markdown]
# ## 3. Assess one flaw with each line
#
# For Option 2 the material is declared with `fad_option=2` and a `law`; the declared `yielding` must agree with
# the law (a law without a Lueders plateau is "continuous").

# %%
plate = cf.FlatPlate(thickness=30e-3, width=600e-3)
flaw = cf.SurfaceFlaw(a=6e-3, two_c=40e-3)
loads = cf.StressState(pm=260e6, pb=30e6)  # a high load, so that the point sits in the plastic part of the FAD
materials = {
    "Option 1": cf.Material(SY, SU, E, 110e6, yielding="continuous"),
    "Option 2, power law": cf.Material(SY, SU, E, 110e6, yielding="continuous", fad_option=2, law=power),
    "Option 2, tabulated": cf.Material(SY, SU, E, 110e6, yielding="continuous", fad_option=2, law=tabulated),
}
results = {}
for name, mat in materials.items():
    res = cf.Assessment(plate, flaw, mat, loads).run()
    results[name] = res
    print(
        f"{name:>20}: Lr = {float(res.lr):.4f}, Kr = {float(res.kr):.4f}, margin = {float(res.radial_margin):.3f}, "
        f"acceptable = {bool(res.acceptable)}, law = {res.trace.options['material_law']}"
    )
# Lr and Kr do not depend on the line (no secondary stress here); only the line and so the margin differ
lrs = [float(r.lr) for r in results.values()]
krs = [float(r.kr) for r in results.values()]
assert np.allclose(lrs, lrs[0]) and np.allclose(krs, krs[0])
assert "fad.option2" in results["Option 2, tabulated"].trace.equations

# %% [markdown]
# ## 4. A law that does not fit the declaration is refused

# %%
try:
    cf.Material(SY, SU, E, 110e6, yielding="discontinuous", fad_option=2, law=power)
except ValueError as exc:
    print("refused:", exc)
else:
    raise AssertionError("a discontinuous declaration with a law without a plateau must be refused")

# %% [markdown]
# ## Interpretation
#
# * The Option 2 lines use more of the material's hardening than Option 1, so for this (synthetic) material the
#   radial margin at the same assessment point differs between the options. Which option is less conservative
#   depends on the material: Option 1 is not a bound on Option 2 in general.
# * The lower-bound power law (`n` from the yield-to-tensile ratio) is a convenient Option 2 surrogate when no
#   curve is available; a measured curve, cleaned by `TabulatedLaw`, reflects the real hardening.
# * The cleaning notes are part of the audit trail and appear in the result's trace options.
