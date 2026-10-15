# %% [markdown]
# # The Option 3 curve emulator, trained on a synthetic curve family (EXPERIMENTAL)
#
# FE runs are expensive, so the Option 3 curve is emulated across the five-dimensional parameter box
# `x = (a/B, a/c, n, sigma_0/E, beta)` by one Gaussian process over `(x, load step)` for the log increments of
# `J/J_e`. Every draw is a valid curve by construction (`Kr <= 1`, non-increasing in `Lr`), and every prediction
# carries a status (`supported`, `outside_box`, `beyond_support`, `cap_uncertain`, `invalid`) instead of raising.
#
# This example replaces the FE "truth" by a closed-form, smooth, monotone family (the one used by the library's
# own tests), so it runs in seconds: train, predict, validate against held-out curves, and pass the guard that
# refuses experimental curves unless declared.
#
# **Every emulator prediction is `verification_status = "unverified"`, `assessment_grade = False`: experimental.**
#
# Theory: `docs/manual/theory/06_fe_option3.md` (Option 3 curves and the emulator).

# %%
from pathlib import Path

import numpy as np

from critflaw.option3 import curve as C
from critflaw.option3 import emulator as EM
from critflaw.option3 import validation as V
from critflaw.option3.design import LOWER, NAMES, UPPER, sobol_points
from critflaw.option3.interface import NotAssessmentGrade, assessment_curve

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
# ## 1. The synthetic "FE" family
#
# `schedule(x)` mimics the FE load schedule (geometric then linear steps up to a strain cap); `true_ratio(x, lr)`
# is a smooth monotone `J/J_e(Lr)` that grows faster for small hardening exponents and deep flaws. Curves are
# cut to the emulator's support (step and `Lr` limits), exactly like FE curves.


# %%
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
    ratio = true_ratio(x, lr)
    ratio = ratio / ratio[0]  # J_e is defined by the first step
    keep = (np.arange(lr.size) < C.MAX_STEP) & (lr <= C.LR_SUPPORT)
    k = int(np.argmin(keep)) if not keep.all() else lr.size
    return C.Option3Curve(x, level, "synthetic", lr[:k], ratio[:k])


print("inputs:", dict(zip(NAMES, zip(LOWER, UPPER, strict=True), strict=True)))

# %% [markdown]
# ## 2. Train on 48 Sobol' points

# %%
train_x = sobol_points(48, 1)
emulator = EM.train([synth_curve(x) for x in train_x], LOWER, UPPER, schedule, n_starts=3)
print("trained; cap strain:", emulator.cap_strain)

# %% [markdown]
# ## 3. Predict one curve, with draws

# %%
x0 = np.array([0.4, 0.7, 10.0, 2e-3, 0.5])
p = emulator.predict(x0, n_draws=400, rng=np.random.default_rng(5))
truth = synth_curve(x0)
print(
    f"status {p.status}; {p.lr.size} supported steps up to Lr = {p.lr[-1]:.3f}; "
    f"verification {p.verification_status}, assessment grade {p.assessment_grade}  -> EXPERIMENTAL"
)
lo95, hi95 = np.percentile(p.kr_draws, [2.5, 97.5], axis=0)
for k in range(p.lr.size):
    print(f"  Lr = {p.lr[k]:.3f}: truth {truth.kr[k]:.4f}, mean {p.kr_mean[k]:.4f}, 95% [{lo95[k]:.4f}, {hi95[k]:.4f}]")
assert p.status == "supported" and np.all(p.kr_draws <= 1.0 + 1e-12)
assert np.all(np.diff(p.kr_draws, axis=1) <= 1e-12)  # every draw is non-increasing
kr_q, status_q = p.kr_at(np.array([0.5, p.lr[-1] + 0.05]))
print("query beyond the support:", list(status_q), "(NaN, never clipped)")
assert status_q[1] in ("beyond_support", "cap_uncertain") and np.all(np.isnan(kr_q[:, 1]))
print("outside the box:", emulator.predict(np.array([0.1, 0.5, 10.0, 2e-3, 0.0]), 10).status)

# %% [markdown]
# ## 4. Validate against 12 held-out curves
#
# Criteria fixed in advance: RMSE of `Kr` at most 0.02 over supported steps, 95 % interval coverage, and the
# energy score against a mean-only forecast.

# %%
heldout = [synth_curve(x) for x in sobol_points(12, 99)]
rep = V.validate(emulator, heldout, n_draws=400)
print(
    f"RMSE {rep.rmse:.4f} (pass: {rep.rmse_pass}), coverage {rep.coverage:.3f} +- {rep.coverage_se_curves:.3f}, "
    f"energy {rep.energy:.4f} vs mean-only {rep.energy_baseline:.4f}, support overruns {rep.support_overruns}"
)
assert rep.rmse_pass and rep.coverage >= 0.8

# %% [markdown]
# ## 5. The guard: experimental curves must be declared

# %%
try:
    assessment_curve(p, lr_max=1.1)
except NotAssessmentGrade as exc:
    print("refused:", exc)
else:
    raise AssertionError("an experimental curve must be refused without experimental=True")
cur = assessment_curve(p, lr_max=1.1, experimental=True)
print("accepted with experimental=True; labels:", cur.labels)
assert cur.experimental and "experimental" in cur.labels

if plt is not None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for d in p.kr_draws[:40]:
        ax.plot(p.lr, d, color="0.7", lw=0.6)
    ax.plot(p.lr, p.kr_mean, "C0", label="emulator mean")
    ax.fill_between(p.lr, lo95, hi95, alpha=0.25, label="95% band")
    ax.plot(truth.lr, truth.kr, "k--", label="synthetic truth")
    ax.set(xlabel="Lr", ylabel="Kr", ylim=(0, 1.05), title="Option 3 emulator (EXPERIMENTAL, synthetic family)")
    ax.legend(fontsize=8)
    save(fig, "16_emulator.png")

# %% [markdown]
# ## Interpretation
#
# * The emulator reproduces the held-out synthetic curves within the RMSE target, and its 95 % bands cover the
#   truth at the required rate. This is a synthetic family and is not evidence of performance on real FE data.
#   On the real FE data (`benchmarks/option3_v9.py`) the RMSE target was not met (leave-one-curve-out 0.0241
#   against 0.02), leave-one-curve-out coverage was below nominal (0.935 against 0.95), and the held-out checks
#   that passed rest on two curves; the emulator remains experimental.
# * Structure is built in: every draw is a monotone curve below 1, and the support is never exceeded (queries
#   beyond it are NaN with a status).
# * The result is experimental. The guard (`assessment_curve`) refuses it unless `experimental=True` is passed, and
#   the label travels with the curve.
