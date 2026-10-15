# %% [markdown]
# # Option 1 assessment of a surface flaw in a flat plate
#
# This example runs a deterministic failure assessment diagram (FAD) check of a semi-elliptical surface flaw
# in a flat plate under membrane and bending primary stress, with the Option 1 (material-independent)
# assessment line. It shows:
#
# * how a problem is built from immutable objects (`FlatPlate`, `SurfaceFlaw`, `Material`, `StressState`);
# * the result fields (`lr`, `kr`, the crack-front positions, the acceptance flag and the radial margin);
# * that acceptance is decided by the **direct test** (`Kr < f(Lr)` and `Lr < Lr,max`), with the radial
#   margin as a by-product;
# * vectorised evaluation over many load levels and flaw sizes in one call.
#
# Theory: `docs/manual/theory/01_fad.md` (FAD and Option 1) and `docs/manual/theory/02_solutions.md`
# (stress intensity factor and reference stress solutions).
#
# All numbers are **illustrative** SI inputs (Pa, m, Pa m^0.5), not taken from any standard's worked example.

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
# ## 1. The problem
#
# A 30 mm thick, 600 mm wide plate of a C-Mn steel with a 5 mm deep, 30 mm long surface flaw. The yield
# behaviour must be declared explicitly (`yielding` has no default, because the assumption changes the
# assessment line by a large factor around `Lr = 1`).

# %%
plate = cf.FlatPlate(thickness=30e-3, width=600e-3)  # m
flaw = cf.SurfaceFlaw(a=5e-3, two_c=30e-3)  # depth a, full surface length 2c (m)
steel = cf.Material(
    yield_strength=380e6,  # Pa (illustrative)
    uts=520e6,  # Pa
    modulus=205e9,  # Pa
    toughness=110e6,  # Kmat in Pa m^0.5 (110 MPa m^0.5)
    yielding="continuous",
)
loads = cf.StressState(pm=180e6, pb=30e6)  # primary membrane and bending stress, Pa
problem = cf.Assessment(plate, flaw, steel, loads)

# %% [markdown]
# ## 2. Run and read the result
#
# A surface flaw is assessed at two crack-front positions (deepest point and surface point). `kr` is the value
# at the governing position, `kr_by_position` has both.

# %%
r = problem.run()
lr, kr = float(r.lr), float(r.kr)
print(f"Lr = {lr:.4f}   Kr = {kr:.4f}   Lr,max = {float(r.lr_max):.4f}")
print(f"positions: {r.positions}; governing: {r.positions[int(r.governing_position)]}")
print("Kr by position:", np.round(r.kr_by_position, 4))
print("primary K by position (MPa m^0.5):", np.round(r.k_primary / 1e6, 3))
print(f"reference stress = {float(r.reference_stress) / 1e6:.1f} MPa")
print(f"acceptable: {bool(r.acceptable)}; radial margin = {float(r.radial_margin):.3f} (exit: {r.exit.item()})")

# %% [markdown]
# ## 3. The direct acceptance test
#
# Acceptance is the direct test at every crack-front position, never a comparison of the radial margin with 1.
# We reproduce it by hand from the assessment line returned by `problem.curve()`.

# %%
curve = problem.curve()
f_at_lr = float(curve(lr))
by_hand = bool(np.all(r.kr_by_position < f_at_lr) and lr < float(curve.lr_max))
print(f"f(Lr) = {f_at_lr:.4f}; direct test by hand: {by_hand}")
assert by_hand == bool(r.acceptable)
assert 0.0 < lr < float(r.lr_max) and 0.0 < kr < 1.0

# %% [markdown]
# ## 4. Proportional loading: the radial margin is the load factor to the boundary
#
# With primary stress only, `Lr` and `Kr` are both proportional to the load, so scaling the load moves the point
# along a ray from the origin. All load levels are evaluated in **one vectorised call**. The first unacceptable
# factor brackets the radial margin.

# %%
factors = np.linspace(0.05, 3.0, 300)
sweep = problem.with_(loads=cf.StressState(pm=180e6 * factors, pb=30e6 * factors)).run(on_invalid="mask")
ratio = sweep.kr / sweep.lr
assert np.allclose(ratio, ratio[0], rtol=1e-9), "primary-only loading must stay on one ray"
first_bad = int(np.argmax(~sweep.acceptable))
assert not sweep.acceptable[first_bad] and sweep.acceptable[first_bad - 1]
print(
    f"boundary between load factors {factors[first_bad - 1]:.3f} and {factors[first_bad]:.3f}; "
    f"radial margin {float(r.radial_margin):.3f}"
)
assert factors[first_bad - 1] <= float(r.radial_margin) <= factors[first_bad] + 1e-12

# %% [markdown]
# ## 5. Flaw depth sweep (vectorised, with masking of invalid sizes)
#
# The flaw shape is held at `a/2c = 0.2`. Sizes outside the validity range of the stress intensity factor
# solution are **not** extrapolated: with `on_invalid="mask"` they come back with `valid = False` and NaN fields.

# %%
depths = np.linspace(0.5e-3, 28e-3, 120)
locus = problem.with_(flaw=cf.SurfaceFlaw(a=depths, two_c=5.0 * depths)).run(on_invalid="mask")
print(f"valid sizes: {int(locus.valid.sum())} of {depths.size}; acceptable: {int(locus.acceptable.sum())}")
i_bad = int(np.argmax(~locus.acceptable))
reason = "outside validity" if not locus.valid[i_bad] else "assessment point on or beyond the line"
print(f"smallest depth on this grid that is not acceptable: {depths[i_bad] * 1e3:.2f} mm ({reason})")
assert locus.valid[0] and locus.acceptable[0]

# %% [markdown]
# ## 6. The FAD, and the effect of the declared yield behaviour

# %%
lr_grid = np.linspace(0.0, 1.3, 600)
lines = {
    kind: cf.Option1Curve(380e6, 520e6, 205e9, kind)  # type: ignore[arg-type]
    for kind in ("continuous", "discontinuous", "unknown")
}
for kind, c in lines.items():
    print(f"{kind:>13}: f(1) = {float(c.f_at_one):.4f}, Lr,max = {float(c.lr_max):.4f}")
assert float(lines["unknown"].f_at_one) <= min(
    float(lines["continuous"].f_at_one), float(lines["discontinuous"].f_at_one)
)

if plt is not None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for kind, c in lines.items():
        ax.plot(lr_grid, c(lr_grid), label=f"Option 1, {kind}")
    ok = sweep.acceptable
    ax.plot(sweep.lr[ok], sweep.kr[ok], ".", ms=3, color="tab:green", label="load sweep, acceptable")
    ax.plot(sweep.lr[~ok], sweep.kr[~ok], ".", ms=3, color="tab:red", label="load sweep, not acceptable")
    ax.plot(locus.lr, locus.kr, "-", color="0.4", lw=1, label="depth sweep (a/2c = 0.2)")
    ax.plot([lr], [kr], "ko", label="assessment point")
    ax.set(xlabel="Lr", ylabel="Kr", xlim=(0, 1.3), ylim=(0, 1.1), title="Option 1 FAD (illustrative)")
    ax.legend(fontsize=8)
    save(fig, "01_fad.png")

# %% [markdown]
# ## Interpretation
#
# * The point lies inside the assessment line with a radial margin above 1, so the flaw is acceptable under
#   these (illustrative) inputs; the load-factor sweep confirms that the radial margin is the factor on the
#   primary load that takes the point to the boundary.
# * The deepest point governs for this shallow, fairly long flaw.
# * The declared yield behaviour matters: the discontinuous and "unknown" lines drop at `Lr = 1`, so a point
#   near `Lr = 1` can change from acceptable to not acceptable on that declaration alone.
# * Flaw sizes outside the solution's validity are reported as invalid, never extrapolated.
