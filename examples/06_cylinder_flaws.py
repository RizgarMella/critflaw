# %% [markdown]
# # Flaws in cylinders: routes, pressure records and positions
#
# A `Cylinder` needs two explicit model choices:
#
# * `k_route`: `"pressure_only"` (plate factors with a bulging correction; refuses mechanical primary stress)
#   or `"tabulated"` (the tabulated cylinder factors, interpolated, never extrapolated);
# * `sref_route`: `"thin"` (thin-wall reference stresses), `"thick_pressure"` (limit pressure; needs a
#   `PressureLoad`) or `"thick"` (thick-wall limit loads).
#
# Pressure is declared once, as a `PressureLoad` with the user's pressure-to-stress factors, so `K` and `Lr`
# always describe the same loading. The example covers an axial internal surface flaw under pressure (both K
# routes), a circumferential through-wall flaw under combined loading, and a fully circumferential external
# flaw.
#
# Theory: `docs/manual/theory/02_solutions.md` (cylinder solutions). All inputs are **illustrative**.

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
# ## 1. Geometry, material and the pressure record
#
# Pipe: inner radius 100 mm, wall 20 mm (mean radius over wall 5.5). The pressure-to-stress factors are the
# user's choice; here the thin-shell membrane values `r_m / B` (hoop) and `r_m / (2B)` (axial).

# %%
R_I, B = 0.10, 0.02
r_m_over_b = (R_I + 0.5 * B) / B
steel = cf.Material(355e6, 490e6, 207e9, 100e6, yielding="continuous")
pressure = cf.PressureLoad(pressure=5e6, hoop_per_pressure=r_m_over_b, axial_per_pressure=0.5 * r_m_over_b)


def pipe(k_route, sref_route="thin"):
    return cf.Cylinder(r_i=R_I, thickness=B, length=2.0, k_route=k_route, sref_route=sref_route)


axial = cf.CylinderSurfaceFlaw(a=5e-3, two_c=40e-3, orientation="axial", position="internal")

# %% [markdown]
# ## 2. Axial internal surface flaw under pressure: the two K routes
#
# The two routes are different models of the same shell: their difference is a model uncertainty, recorded in
# the trace, not an error.

# %%
by_route = {}
for route in ("pressure_only", "tabulated"):
    res = cf.Assessment(pipe(route), axial, steel, cf.StressState(pressure=pressure)).run()
    by_route[route] = res
    print(
        f"{route:>13}: Lr = {float(res.lr):.4f}, Kr = {float(res.kr):.4f}, positions = {res.positions}, "
        f"K equations = {[e for e in res.trace.equations if e.startswith('k.cylinder')]}"
    )
assert by_route["pressure_only"].trace.options["k_route"] == "pressure_only"
assert np.isclose(float(by_route["pressure_only"].lr), float(by_route["tabulated"].lr))  # same thin-wall Lr

# The pressure-only route refuses mechanical primary stress (it has no factors for it)
try:
    cf.Assessment(pipe("pressure_only"), axial, steel, cf.StressState(pm=1e6, pressure=pressure)).run()
except cf.OutOfValidityError as exc:
    print("pressure_only with mechanical stress refused:", exc)

# %% [markdown]
# ## 3. Thick-wall limit pressure as the reference stress
#
# With `sref_route="thick_pressure"`, `Lr` is the ratio of the pressure to the flawed cylinder's limit pressure.

# %%
thick = cf.Assessment(pipe("tabulated", "thick_pressure"), axial, steel, cf.StressState(pressure=pressure)).run()
print(f"thick_pressure: Lr = {float(thick.lr):.4f} (thin-wall: {float(by_route['tabulated'].lr):.4f})")
print("options:", {k: v for k, v in thick.trace.options.items() if "sref" in k or "route" in k})

# %% [markdown]
# ## 4. Pressure sweep (vectorised)

# %%
p = np.linspace(1e6, 30e6, 60)
sweep = cf.Assessment(
    pipe("tabulated"), axial, steel, cf.StressState(pressure=cf.PressureLoad(p, r_m_over_b, 0.5 * r_m_over_b))
).run(on_invalid="mask")
i_bad = int(np.argmax(~sweep.acceptable))
print(f"first pressure on the grid that is not acceptable: {p[i_bad] / 1e6:.1f} MPa")
assert sweep.valid.all() and np.all(np.diff(sweep.kr) > 0)  # Kr grows with pressure

# %% [markdown]
# ## 5. Circumferential through-wall flaw under membrane, bending, global moment and pressure
#
# Through-wall flaws are assessed at the outer and inner surfaces. `pb` is through-wall bending, tensile on the
# outside by default (`bending_tensile_surface`). Interpolation through table cells flagged as suspect is
# recorded as a flag and downgrades the knowledge-base status of that table in the trace.

# %%
through = cf.CylinderThroughFlaw(two_a=0.05, orientation="circumferential")
combined = cf.StressState(pm=30e6, pb=5e6, pm_moment=20e6, pressure=pressure)
rt = cf.Assessment(pipe("tabulated"), through, steel, combined).run()
print(
    f"positions {rt.positions}: Kr = {np.round(rt.kr_by_position, 4)}, governing = {rt.positions[int(rt.governing_position)]}"
)
print("flags:", rt.trace.flags)
inside = cf.Assessment(pipe("tabulated"), through, steel, combined.with_(bending_tensile_surface="inside")).run()
print(f"bending tensile inside: Kr = {np.round(inside.kr_by_position, 4)}")

# %% [markdown]
# ## 6. Fully circumferential external flaw

# %%
ext = cf.ExtendedFlaw(a=5e-3, orientation="circumferential", position="external")
re_ = cf.Assessment(pipe("tabulated"), ext, steel, cf.StressState(pm=50e6, pb=20e6)).run()
print(
    f"extended external: Lr = {float(re_.lr):.4f}, Kr = {float(re_.kr):.4f}, equations: "
    f"{[e for e in re_.trace.equations if e.startswith(('k.', 'sref.'))]}"
)

if plt is not None:
    lr_grid = np.linspace(0, 1.3, 400)
    curve = cf.Assessment(pipe("tabulated"), axial, steel, cf.StressState(pressure=pressure)).curve()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(lr_grid, curve(lr_grid), "k", label="Option 1 line")
    ax.plot(sweep.lr, sweep.kr, label="axial flaw, pressure sweep (tabulated K)")
    for route, res in by_route.items():
        ax.plot(float(res.lr), float(res.kr), "o", label=f"5 MPa, {route}")
    ax.plot(float(rt.lr), float(rt.kr), "s", label="circumferential through flaw")
    ax.plot(float(re_.lr), float(re_.kr), "^", label="fully circumferential external")
    ax.set(xlabel="Lr", ylabel="Kr", ylim=(0, 1.2), title="Cylinder flaws (illustrative)")
    ax.legend(fontsize=8)
    save(fig, "06_cylinders.png")

# %% [markdown]
# ## Interpretation
#
# * The K route is a declared model choice; the two routes give different `Kr` for the same shell and the
#   difference is a model uncertainty to be carried (for example through `k_model`).
# * The pressure record keeps `K` and `Lr` consistent; doubling the pressure doubles both on the thin-wall route.
# * Geometries outside the tables are refused, never extrapolated; reads through suspect table cells are
#   flagged and the trace reports the affected table as unverified.
