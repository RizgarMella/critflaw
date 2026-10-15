# %% [markdown]
# # Failure probability with the Option 3 emulator (EXPERIMENTAL, synthetic family)
#
# With an emulated curve the failure probability has two sources of uncertainty: sampling (the Monte Carlo or
# importance-sampling population) and the surrogate (which curve is the true one). `critflaw.option3.reliability`
# handles both:
#
# * **coherent function draws** of the emulator (`function_draws`): each draw is one curve-valued function of `x`;
# * each realisation is classified in a fixed order: collapse first (`Lr >= Lr,max`, failed without querying the
#   emulator), then `Kr >= 1` (failed for every draw), then against the draw's between-step **bounds**;
#   realisations the bounds cannot decide, or that the emulator does not support, are *undecided*;
# * the result is a **bound pair** `Pf_low` (undecided counted safe) and `Pf_high` (undecided counted failed) per
#   draw and population, with a crossed two-way ANOVA separating the surrogate and sampling components.
#
# `pf_crossed` refuses to run unless `experimental=True` is passed. The synthetic family of example 16 stands in
# for the FE, so the true failure probability is known by brute force.
#
# Theory: `docs/manual/theory/07_pf_emulator.md`.

# %%
import numpy as np
from scipy import stats

from critflaw.option3 import curve as C
from critflaw.option3 import emulator as EM
from critflaw.option3 import reliability as R
from critflaw.option3.design import LOWER, UPPER, sobol_points


def schedule(x):
    lr_n = min(1.5, (0.02 / x[3]) ** (1.0 / x[2]))
    lo = np.geomspace(0.05, min(0.8, 0.9 * lr_n), 6)
    return np.concatenate([lo, np.linspace(lo[-1], lr_n, 7)[1:]])


def true_ratio(x, lr):
    m = 2.0 + 0.4 * x[2]
    amp = 0.4 + 0.8 * x[0] + 0.2 * x[4]
    return 1.0 + amp * (lr / 1.2) ** m


def synth_curve(x, level=(6, 12, 12, 3, 3)):
    lr = schedule(x)
    ratio = true_ratio(x, lr) / true_ratio(x, lr)[0]
    keep = (np.arange(lr.size) < C.MAX_STEP) & (lr <= C.LR_SUPPORT)
    k = int(np.argmin(keep)) if not keep.all() else lr.size
    return C.Option3Curve(x, level, "synthetic", lr[:k], ratio[:k])


# %% [markdown]
# ## 1. The probabilistic problem
#
# Seven standard normals: the first five are mapped into a central sub-box of the emulator's inputs (geometry,
# hardening and load mix are uncertain), the sixth sets `Lr` (load) and the seventh `Kr` (applied K over
# toughness). `Lr,max` is 1.15. The problem is a function `u -> (x, lr, kr, lr_max)`.

# %%
DIM = 7


def to_x(u):
    p = stats.norm.cdf(u[:, :5])
    return LOWER + (0.15 + 0.7 * p) * (UPPER - LOWER)


def problem(u):
    x = to_x(u)
    lr = 0.75 * np.exp(0.12 * u[:, 5])
    kr = 0.55 * np.exp(0.15 * u[:, 6])
    return x, lr, kr, np.full(lr.size, 1.15)


def true_failed(u):
    """Brute-force truth on the synthetic family (beyond the support counts as failed, as in the headline)."""
    x, lr, kr, lr_max = problem(u)
    out = lr >= lr_max
    for j in np.flatnonzero(~out):
        c = synth_curve(x[j])
        if lr[j] > c.lr[-1]:
            out[j] = True
            continue
        out[j] = kr[j] >= np.interp(lr[j], c.lr, c.j_over_je) ** -0.5
    return out


# %% [markdown]
# ## 2. Emulator and coherent function draws

# %%
emulator = EM.train([synth_curve(x) for x in sobol_points(48, 1)], LOWER, UPPER, schedule, n_starts=3)
draws = EM.function_draws(emulator, 12, np.random.default_rng(1), n_features=1000)
print(f"{draws.n_draws} function draws (pathwise GP draws with random Fourier features)")

# %% [markdown]
# ## 3. Truth, populations and the crossed Pf

# %%
rng = np.random.default_rng(3)
u_ref = rng.standard_normal((40_000, DIM))
pf_true = float(true_failed(u_ref).mean())
N_POP = 2000  # samples per population (kept small so the example runs in about a minute)
pops = [R.population(problem, DIM, N_POP, np.zeros(DIM), rng, alpha=1.0) for _ in range(3)]  # plain MC populations
try:
    R.pf_crossed(draws, pops)
except PermissionError as exc:
    print("refused without the flag:", exc)
res = R.pf_crossed(draws, pops, experimental=True)
print("EXPERIMENTAL result:")
print(f"  Pf_high (headline) = {res.pf:.4f}; Pf_low mean = {res.pf_low.mean():.4f}; brute-force truth = {pf_true:.4f}")
print(f"  90% surrogate-only interval of Pf_high: [{res.interval[0]:.4f}, {res.interval[1]:.4f}]")
print(
    f"  variance components: draws {res.draw_component:.3g}, populations {res.population_component:.3g}, "
    f"residual {res.residual:.3g}; sampling SE {res.sampling_se:.4f}"
)
print("  undecided mass by class:", {k: round(v, 5) for k, v in res.undecided_mass.items()})
for label in res.labels:
    print("  label:", label)
se = np.sqrt(pf_true * (1 - pf_true) / N_POP)  # one population (conservative: three are averaged)
assert res.pf_low.mean() - 4 * se <= pf_true <= res.pf_high.mean() + 4 * se
assert any("experimental" in lab for lab in res.labels)

# %% [markdown]
# ## 4. The rigorous (monotone) bounds
#
# The default between-step bounds use the stated assumption that `J/J_e` is convex in `Lr` above a small floor
# (supported by every accepted FE curve, checked locally per draw). `convex=False` gives the rigorous monotone
# pair, which is wider. One population and six draws are used here to keep the example cheap.

# %%
few = EM.function_draws(emulator, 4, np.random.default_rng(2), n_features=1000)
mono = R.pf_crossed(few, pops[:1], experimental=True, convex=False)
conv = R.pf_crossed(few, pops[:1], experimental=True)
print(
    f"one population: convex pair [{conv.pf_low.mean():.4f}, {conv.pf_high.mean():.4f}], "
    f"monotone pair [{mono.pf_low.mean():.4f}, {mono.pf_high.mean():.4f}]"
)
assert np.all(mono.pf_low <= conv.pf_low + 1e-15) and np.all(mono.pf_high >= conv.pf_high - 1e-15)

# %% [markdown]
# ## Interpretation
#
# * The bound pair brackets the brute-force truth within sampling error; the gap between `Pf_low` and `Pf_high` is
#   the mass the emulator could not decide (between steps, beyond support).
# * The ANOVA separates how much of the spread comes from not knowing the curve (draws) and how much from finite
#   sampling (populations); more FE runs reduce the first, more samples the second.
# * Defensive importance sampling (`alpha < 1` with a design point from a cheap proxy such as the Option 2 line)
#   is available through `population(..., design_point, alpha)` for rarer events.
# * Everything here is experimental and labelled so; the surrogate-only interval excludes model uncertainty,
#   distribution-parameter uncertainty and FE-versus-reality differences.
