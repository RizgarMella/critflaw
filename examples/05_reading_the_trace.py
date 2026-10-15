# %% [markdown]
# # Reading the trace: what a result depended on and what it does not account for
#
# Every `Result` carries a `Trace`: the equations used (with honest citations and their verification and
# knowledge-base status), the options in force, the flags raised during the evaluation, and the uncertainty
# sources a deterministic result does **not** account for. This example walks through:
#
# * `result.report()` and the individual trace fields;
# * flags for out-of-validity samples, negative primary K, step exits and secondary-only loading;
# * masking versus raising for samples outside a solution's validity range;
# * model-uncertainty factors (`k_model`, `sref_model`, `fad_model`).
#
# Theory: `docs/manual/theory/01_fad.md` (result fields) and `docs/manual/theory/08_knowledge_base.md`
# (verification status). All inputs are **illustrative**.

# %%
import numpy as np

import critflaw as cf

plate = cf.FlatPlate(thickness=25e-3, width=500e-3)
steel = cf.Material(380e6, 520e6, 205e9, 100e6, yielding="continuous")
problem = cf.Assessment(
    plate, cf.SurfaceFlaw(a=4e-3, two_c=24e-3), steel, cf.StressState(pm=170e6, residual=cf.Residual.as_welded())
)
r = problem.run()

# %% [markdown]
# ## 1. The human-readable report

# %%
text = r.report()
print(text)
assert "Not accounted for:" in text and "Equations used:" in text

# %% [markdown]
# ## 2. The trace fields one by one

# %%
t = r.trace
print("edition of the source:", t.edition)
print("options:", t.options)
print("flags:", t.flags or "none")
print("equations:", t.equations)
print("excluded uncertainty:")
for s in t.excluded_uncertainty:
    print("  -", s)
print("\nper-equation status:")
for dep in t.dependencies():
    print(
        f"  {dep['id']:<28} kb={dep['kb_status']:<16} printed-source-checked={dep['checked_against_printed_source']}"
        f"  primary-checked={dep['checked_against_primary_reference']}  self-consistent-only={dep['self_consistent_only']}"
    )
print("\nbelow 'cross-checked' in the knowledge base:", t.below_cross_checked())
print("self-consistent only:", t.self_consistent_only())
assert set(t.below_cross_checked()) <= set(t.equations)
assert "residual.as_welded" in t.equations

# %% [markdown]
# ## 3. Out-of-validity samples: raise or mask
#
# By default a sample outside a solution's validity raises `OutOfValidityError`. With `on_invalid="mask"` it
# is returned blanked (NaN fields, `valid = False`, `exit = "invalid"`), for the caller to account for
# explicitly. Nothing is extrapolated, clipped or dropped.

# %%
flaws = cf.SurfaceFlaw(a=np.array([4e-3, 12e-3, 30e-3]), two_c=np.array([24e-3, 60e-3, 80e-3]))  # last: a > B
vector = problem.with_(flaw=flaws, loads=cf.StressState(pm=170e6))
try:
    vector.run()
except cf.OutOfValidityError as exc:
    print("raised:", exc)
masked = vector.run(on_invalid="mask")
print("valid:", masked.valid, " exit:", masked.exit, " Lr:", np.round(masked.lr, 4))
print("flags:", masked.trace.flags)
assert list(masked.valid) == [True, True, False] and masked.exit[2] == "invalid" and np.isnan(masked.lr[2])
assert "out_of_validity_samples_present" in masked.trace.flags

# %% [markdown]
# ## 4. Other flags you may meet

# %%
# (a) bending on a deep flaw: the primary K at the deepest point is negative and is set to zero (conservative)
deep = cf.Assessment(plate, cf.SurfaceFlaw(22e-3, 44e-3), steel, cf.StressState(pb=100e6, qm=100e6)).run()
print("deep flaw under bending:", deep.trace.flags)
assert "negative_primary_k_set_to_zero" in deep.trace.flags

# (b) a discontinuous-yield line: the point can leave the acceptable region through the vertical step at Lr = 1
disc = cf.Assessment(
    plate, cf.SurfaceFlaw(3e-3, 15e-3), steel.with_(yielding="discontinuous"), cf.StressState(pm=200e6)
)
rd = disc.run()
print(f"discontinuous yielding: exit = {rd.exit.item()}, margin = {float(rd.radial_margin):.4f}")

# (c) secondary stress only
alone = cf.Assessment(plate, cf.SurfaceFlaw(4e-3, 24e-3), steel, cf.StressState(qm=150e6)).run()
print("secondary only:", alone.trace.flags)
assert "secondary_stress_acting_alone" in alone.trace.flags

# %% [markdown]
# ## 5. Model-uncertainty factors
#
# `k_model`, `sref_model` and `fad_model` multiply the stress intensity factors, the reference stress and the
# assessment line. A fixed value is a deterministic sensitivity; a distribution (see the probabilistic examples)
# turns the corresponding entry of "Not accounted for" into an accounted-for random variable.

# %%
for k_model in (1.0, 1.1, 1.2):
    rk = problem.with_(k_model=k_model).run()
    print(f"k_model = {k_model}: Kr = {float(rk.kr):.4f}, margin = {float(rk.radial_margin):.4f}")
assert float(problem.with_(k_model=1.2).run().kr) > float(r.kr)

# %% [markdown]
# ## Interpretation
#
# * The trace is the audit record of a result: which equations (and so which verification status) it rests on,
#   which options were in force and which special cases occurred.
# * Equations listed below `cross-checked` have no linked check against primary-reference values or a
#   definition-pinning limiting case; `strict=True`
#   in `Assessment.run` refuses such results (see example 18).
# * "Not accounted for" lists the uncertainty a single deterministic evaluation ignores; the probabilistic
#   examples (09 onwards) show how to account for it.
