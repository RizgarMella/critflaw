# %% [markdown]
# # FAD gallery: the failure assessment diagram, deterministic and probabilistic
#
# The FAD is a graphical method, so this example is all pictures, drawn with `critflaw.plot`:
#
# 1. assessment lines (Option 1 for each yield behaviour, Option 2) with a point and its load ray;
# 2. load and flaw-size sweeps as paths across the diagram (vectorised);
# 3. **probability-density field** of uncertain assessment points, with highest-density-region contours;
# 4. the **exceedance field** of an uncertain assessment line, with the point density overlaid;
# 5. **fatigue trajectories**: flaws growing across the diagram until they reach the line;
# 6. a density field **marching towards the line** as a flaw population grows.
#
# Theory: `docs/manual/theory/01_fad.md` (the diagram), `04_probabilistic.md`, `05_fatigue.md`.
# All inputs are **illustrative** SI values, not taken from any standard's worked example.

# %%
from pathlib import Path

import numpy as np

import critflaw as cf
from critflaw import plot as fadplot
from critflaw import prob
from critflaw.fatigue import ExtremeLoad, Spectrum, TimeDependentAssessment, steel_in_air
from critflaw.fatigue.growth import SURFACE

SCRIPT = "__file__" in globals()
HERE = Path(__file__).resolve().parent if SCRIPT else Path.cwd()
OUT = HERE / "_output"
OUT.mkdir(exist_ok=True)
import matplotlib  # noqa: E402

if SCRIPT:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def show(fig, name):
    fig.savefig(OUT / name, dpi=130, bbox_inches="tight")
    if not SCRIPT:
        plt.show()
    plt.close(fig)


SY, SU, E = 355e6, 490e6, 207e9
plate = cf.FlatPlate(thickness=25e-3, width=0.5)
steel = cf.Material(SY, SU, E, toughness=90e6, yielding="continuous")
problem = cf.Assessment(plate, cf.SurfaceFlaw(a=3e-3, two_c=20e-3), steel, cf.StressState(pm=150e6, pb=30e6))
r = problem.run()

# %% [markdown]
# ## 1. Assessment lines, a point and its load ray
#
# The declared yield behaviour changes the Option 1 line near `Lr = 1`; an Option 2 line comes from a
# stress–strain law. The dashed ray is proportional loading: the point meets the line at the radial margin
# (acceptance itself is the direct test).

# %%
n_lb = float(cf.hardening_exponent_lower_bound(SY, SU))
lines = {
    "Option 1, continuous yielding": cf.Option1Curve(SY, SU, E, "continuous"),
    "Option 1, discontinuous yielding": cf.Option1Curve(SY, SU, E, "discontinuous"),
    f"Option 2, power law (n = {n_lb:.1f})": cf.Option2Curve(cf.PowerLaw(offset=0.002, n=n_lb), SY, SU, E),
}
fig, ax = plt.subplots(figsize=(7, 5))
fadplot.fad(lines, ax)
fadplot.point(r, ax)
ax.legend(fontsize=8, loc="upper right")
show(fig, "19_lines.png")
assert bool(r.acceptable)

# %% [markdown]
# ## 2. Paths: load sweep and flaw-size sweep (one vectorised call each)
#
# Under primary load only the point moves along a ray; growing the flaw at fixed load bends the path because the
# reference stress (Lr) and the crack driving force (Kr) grow at different rates.

# %%
factors = np.linspace(0.0, 2.6, 200)
sweep = problem.with_(loads=cf.StressState(pm=150e6 * factors, pb=30e6 * factors)).run(on_invalid="mask")
depths = np.linspace(1e-3, 22e-3, 200)
locus = problem.with_(flaw=cf.SurfaceFlaw(a=depths, two_c=depths * 20 / 3)).run(on_invalid="mask")
fig, ax = plt.subplots(figsize=(7.6, 5))
fadplot.fad(lines["Option 1, continuous yielding"], ax, title="Paths across the FAD")
fadplot.path(sweep.lr, sweep.kr, ax, values=factors, cmap="Blues", label="load factor 0 to 2.6")
fadplot.path(locus.lr, locus.kr, ax, values=depths * 1e3, cmap="Oranges", colorbar_label="flaw depth (mm)",
             label="flaw depth 1 to 22 mm (2c/a fixed)")  # fmt: skip
fadplot.point(r, ax, ray=False)
ax.legend(fontsize=8, loc="upper right")
show(fig, "19_paths.png")

# %% [markdown]
# ## 3. Probability-density field of uncertain assessment points
#
# Flaw depth, toughness and the extreme membrane stress are random. 100 000 realisations are assessed in one
# vectorised call; the field is their joint density on the diagram, the white contours enclose 50 %, 90 % and
# 99 % of the probability, and every failed realisation is drawn on top.

# %%
kmat = prob.LogNormal.from_mean_sd(80e6, 20e6, name="Kmat")
pm = prob.Truncated(prob.Gumbel.from_mean_sd(180e6, 25e6), lo=0.0, name="pm")
a_rand = prob.Truncated(prob.LogNormal.from_mean_sd(3e-3, 1.5e-3), hi=12e-3, name="a")
pa = cf.ProbabilisticAssessment(
    cf.Assessment(plate, cf.SurfaceFlaw(a=a_rand, two_c=20e-3), steel.with_(toughness=kmat),
                  cf.StressState(pm=pm, pb=30e6))  # fmt: skip
)
u = np.random.default_rng(7).standard_normal((100_000, pa.vector.size))
cloud = pa.evaluate(u)
failed = ~cloud.acceptable
print(f"{u.shape[0]:,} realisations: {int(failed.sum())} failed (crude estimate {failed.mean():.1e})")

fig, ax = plt.subplots(figsize=(7.6, 5))
fadplot.fad(lines["Option 1, continuous yielding"], ax, title="Probability density of the assessment point")
fadplot.density(cloud.lr, cloud.kr, ax)
ax.scatter(cloud.lr[failed], cloud.kr[failed], s=14, color=fadplot.FAIL, edgecolor="white", lw=0.4, zorder=5,
           label=f"failed realisations ({int(failed.sum())})")  # fmt: skip
ax.legend(fontsize=8, loc="upper right")
show(fig, "19_density.png")

# %% [markdown]
# ## 4. An uncertain assessment line: the exceedance field
#
# Yield and tensile strength scatter, and with them the Option 2 line (here from a power law whose hardening
# exponent follows the sampled strengths). The background is the probability that a point at each (Lr, Kr) is
# not acceptable over that scatter, with the 5 %, 50 % and 95 % lines; the black contours enclose 50/90/99 % of
# the point density from section 3. Failure probability lives where the two fields overlap.
# (Simplification for the picture: the points are referred to the nominal yield strength.)

# %%
rng = np.random.default_rng(3)
sy_s = SY * np.exp(rng.normal(0.0, 0.07, 300))
su_s = np.maximum(sy_s * 1.12, SU * np.exp(rng.normal(0.0, 0.05, 300)))
n_s = cf.hardening_exponent_lower_bound(sy_s, su_s)
line_samples = [
    cf.Option2Curve(cf.PowerLaw(offset=0.002, n=float(n)), float(sy), float(su), E)
    for sy, su, n in zip(sy_s, su_s, n_s, strict=True)
]
fig, ax = plt.subplots(figsize=(7.6, 5))
fadplot.fad(line_samples[0], ax, shade=False, title="Uncertain line and uncertain point")
ax.lines[0].remove()  # the field and its quantile lines replace the single line
fadplot.exceedance_field(line_samples, ax)
fadplot.density(cloud.lr, cloud.kr, ax, fill=False, contour_color="black", label="assessment point")
ax.legend(fontsize=7, loc="upper right")
show(fig, "19_exceedance.png")

# %% [markdown]
# ## 5. Fatigue trajectories across the diagram
#
# One flaw grows under cyclic loading; at every checkpoint it is assessed against four candidate fracture
# loads. Each path is coloured by time and the cross marks where it reaches the line: the fatigue life to
# fracture at that load. Higher loads start nearer the line and get there sooner.

# %%
law = steel_in_air()
spectrum = Spectrum.repeating([(70e6, 10e6, 1.0)], cycles_per_year=1e5)
fixed = ExtremeLoad(prob.Uniform(150e6, 150.001e6), reference_period_years=0.25)  # fixed load, fine time steps
h = TimeDependentAssessment(problem, law, spectrum, fixed, times=np.arange(0.25, 30.01, 0.25),
                            leak_is_failure=False).grow()  # fmt: skip
surface = h.kind == SURFACE
t_s = np.append(0.0, h.times[surface])
a_s = np.append(3e-3, h.a[surface])
c_s = np.append(10e-3, h.c[surface])
fig, ax = plt.subplots(figsize=(7.6, 5))
fadplot.fad(lines["Option 1, continuous yielding"], ax, title="Fatigue growth trajectories (one flaw, four loads)")
for i, s_check in enumerate((100e6, 150e6, 200e6, 250e6)):
    pts = problem.with_(flaw=cf.SurfaceFlaw(a=a_s, two_c=2 * c_s),
                        loads=cf.StressState(pm=s_check, pb=30e6)).run(on_invalid="mask")  # fmt: skip
    fadplot.path(pts.lr, pts.kr, ax, values=t_s, cmap="plasma", colorbar_label="time (years)" if i == 0 else None)
    bad = np.flatnonzero(~pts.acceptable)
    life = f"fails at {t_s[bad[0]]:.1f} y" if bad.size else "no failure before breakthrough"
    ax.plot(pts.lr[0], pts.kr[0], "o", ms=5, color=fadplot.LINE, zorder=7)
    if bad.size:
        ax.plot(pts.lr[bad[0]], pts.kr[bad[0]], "X", ms=10, color=fadplot.FAIL, mec="white", zorder=7)
    ax.plot([], [], "X" if bad.size else "o", color=fadplot.FAIL if bad.size else fadplot.LINE,
            label=f"pm = {s_check / 1e6:.0f} MPa: {life}")  # fmt: skip
ax.legend(fontsize=8, loc="upper right")
show(fig, "19_trajectories.png")

# %% [markdown]
# ## 6. A population marching towards the line
#
# The nominal flaw grows; toughness and the extreme load stay random. Snapshots of the density of the
# assessment point show the whole population drifting towards the line, and the failed share growing.

# %%
pa_t = cf.ProbabilisticAssessment(
    cf.Assessment(plate, cf.SurfaceFlaw(a=3e-3, two_c=20e-3), steel.with_(toughness=kmat), cf.StressState(pm=pm))
)
u_t = np.random.default_rng(11).standard_normal((40_000, pa_t.vector.size))
panels = []
for year in (0.0, 2.0, 4.0, 5.0, 5.5, 6.0):
    i = int(np.argmin(np.abs(np.append(0.0, h.times) - year)))
    a_t = float(np.append(3e-3, h.a)[i])
    c_t = float(np.append(10e-3, h.c)[i])
    real = pa_t.realise(u_t).with_(flaw=cf.SurfaceFlaw(a=a_t, two_c=2 * c_t)).run(on_invalid="mask")
    share = float((~real.acceptable).mean())

    def draw(ax, real=real):
        fadplot.fad(lines["Option 1, continuous yielding"], ax, title=None, kr_extent=1.4)
        fadplot.density(real.lr, real.kr, ax, colorbar=False, bins=110)
        ax.set_xlabel("Lr")
        ax.set_ylabel("Kr")

    panels.append((f"t = {year:.1f} y, a = {a_t * 1e3:.1f} mm, failed {share:.1%}", draw))
show(fadplot.small_multiples(panels, ncols=3), "19_marching.png")

# %% [markdown]
# ## Reading the pictures
#
# * The FAD separates acceptable (inside the line, left of the cut-off) from not acceptable.
# * A density field shows not one point but where the assessment point is likely to be; the contours give its
#   central 50/90/99 % regions, and the failure probability is the mass beyond the line.
# * An uncertain line adds a second field; failure lives where the point's density meets the line's exceedance.
# * Fatigue moves every point towards the line over time; the trajectory's crossing is the life at that load.
