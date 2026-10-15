# %% [markdown]
# # Secondary and residual stresses: the plasticity interaction (rho or V)
#
# Secondary stresses (thermal, welding residual) contribute to `Kr` but not to `Lr`. Their interaction with
# plasticity from the primary load is represented by an additive term (`interaction="rho"`, the default) or a
# multiplying factor (`interaction="V"`). This example shows:
#
# * user-given secondary stresses `qm`, `qb` and welding residual stress (`Residual.as_welded`,
#   `Residual.pwht`), with and without relaxation by the primary load;
# * the membrane secondary stress actually used (`result.qm`);
# * the two interaction routes side by side over a load sweep;
# * the flags the engine raises (secondary stress acting alone, chi above the simplified range, relaxation).
#
# Theory: `docs/manual/theory/01_fad.md` (secondary stresses and plasticity interaction).
# All inputs are **illustrative**.

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
# ## 1. One flaw, several secondary-stress cases

# %%
SY = 380e6
plate = cf.FlatPlate(thickness=30e-3, width=600e-3)
flaw = cf.SurfaceFlaw(a=5e-3, two_c=30e-3)
steel = cf.Material(SY, 520e6, 205e9, 110e6, yielding="continuous")
PM = 150e6
cases = {
    "primary only": cf.StressState(pm=PM),
    "thermal qm=60, qb=40 MPa": cf.StressState(pm=PM, qm=60e6, qb=40e6),
    "as welded": cf.StressState(pm=PM, residual=cf.Residual.as_welded()),
    "as welded, relaxed": cf.StressState(pm=PM, residual=cf.Residual.as_welded(relaxation=True)),
    "PWHT, flaw parallel to weld": cf.StressState(pm=PM, residual=cf.Residual.pwht("parallel")),
}
out = {}
for name, loads in cases.items():
    for method in ("rho", "V"):
        r = cf.Assessment(plate, flaw, steel, loads, interaction=method).run()
        out[(name, method)] = r
    r = out[(name, "rho")]
    chi = float(r.k_secondary[0]) * float(r.lr) / float(r.k_primary[0])  # deepest point
    print(
        f"{name:>28}: qm used = {float(r.qm) / 1e6:6.1f} MPa, chi = {chi:5.3f}, "
        f"Kr(rho) = {float(r.kr):.4f}, Kr(V) = {float(out[(name, 'V')].kr):.4f}, Lr = {float(r.lr):.4f}"
    )

primary = out[("primary only", "rho")]
welded = out[("as welded", "rho")]
relaxed = out[("as welded, relaxed", "rho")]
pwht = out[("PWHT, flaw parallel to weld", "rho")]
assert float(welded.kr) > float(primary.kr) and float(welded.lr) == float(primary.lr)  # Lr is primary only
assert float(relaxed.qm) <= float(welded.qm)
pwht_fraction = cf.Residual.pwht("parallel").fraction  # the library's PWHT level (register residual.pwht)
assert abs(float(pwht.qm) - pwht_fraction * SY) < 1e-6 * SY
print("flags (as welded, relaxed):", relaxed.trace.flags)

# %% [markdown]
# ## 2. Secondary stress acting alone
#
# With exactly no primary stress, the engine uses a plastic-zone-corrected secondary `K` and flags the case.
# This is a documented discontinuity: a vanishing but non-zero primary stress takes the interaction route.

# %%
alone = cf.Assessment(plate, flaw, steel, cf.StressState(qm=200e6)).run()
tiny = cf.Assessment(plate, flaw, steel, cf.StressState(pm=1e-6, qm=200e6)).run()
print(f"qm alone: Lr = {float(alone.lr)}, Kr = {float(alone.kr):.4f}, flags = {alone.trace.flags}")
print(f"pm = 1 micro-Pa: Kr = {float(tiny.kr):.4f}, flags = {tiny.trace.flags}")
assert "secondary_stress_acting_alone" in alone.trace.flags
assert "secondary_stress_acting_alone" not in tiny.trace.flags

# %% [markdown]
# ## 3. Load sweep with a relaxed as-welded residual stress
#
# Relaxation lowers the residual stress once the reference stress exceeds a fraction of the flow strength, so
# the secondary contribution falls as the primary load rises. Both interaction routes are evaluated in one
# vectorised call each.

# %%
pm = np.linspace(10e6, 420e6, 120)
sweep = {
    m: cf.Assessment(
        plate, flaw, steel, cf.StressState(pm=pm, residual=cf.Residual.as_welded(relaxation=True)), interaction=m
    ).run(on_invalid="mask")
    for m in ("rho", "V")
}
no_res = cf.Assessment(plate, flaw, steel, cf.StressState(pm=pm)).run(on_invalid="mask")
qm_used = sweep["rho"].qm
assert np.all(np.diff(qm_used[np.isfinite(qm_used)]) <= 1e-6)  # relaxation only ever lowers qm with load
for m, s in sweep.items():
    i = int(np.argmax(~s.acceptable))
    print(f"interaction {m:>3}: first unacceptable pm = {pm[i] / 1e6:.0f} MPa")
print(f"primary only:    first unacceptable pm = {pm[int(np.argmax(~no_res.acceptable))] / 1e6:.0f} MPa")

if plt is not None:
    curve = cf.Assessment(plate, flaw, steel, cf.StressState(pm=PM)).curve()
    lr_grid = np.linspace(0, 1.3, 400)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(lr_grid, curve(lr_grid), "k", label="Option 1 line")
    ax1.plot(no_res.lr, no_res.kr, label="primary only")
    ax1.plot(sweep["rho"].lr, sweep["rho"].kr, label="as welded, relaxed, rho")
    ax1.plot(sweep["V"].lr, sweep["V"].kr, "--", label="as welded, relaxed, V")
    ax1.set(xlabel="Lr", ylabel="Kr", ylim=(0, 1.3), title="Load sweep")
    ax1.legend(fontsize=8)
    ax2.plot(pm / 1e6, qm_used / 1e6)
    ax2.set(xlabel="primary membrane stress (MPa)", ylabel="residual qm used (MPa)", title="Relaxation")
    save(fig, "03_secondary.png")

# %% [markdown]
# ## Interpretation
#
# * A residual stress at yield magnitude raises `Kr` substantially at low primary load; post-weld heat treatment
#   (a fraction of yield set by `Residual.pwht`, register `residual.pwht`) recovers most of the margin.
# * Relaxation reduces the residual stress as the primary load rises, so its benefit appears at high `Lr`.
# * The rho and V routes give slightly different `Kr`; the choice is declared (`interaction`) and recorded in
#   the trace, never made silently.
# * The secondary-only case uses a different `K` (plastic-zone corrected); the jump between "no primary load"
#   and "almost no primary load" is a property of the procedure and is flagged.
