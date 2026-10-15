# %% [markdown]
# # Embedded and through-thickness flaws in plates
#
# * An `EmbeddedFlaw` is an elliptical flaw of height `two_a`, length `two_c`, with ligament `p` to the
#   **nearer** surface. It is assessed at the end of the minor axis nearer the surface and at the ends of the
#   major axis. A bending-loaded embedded flaw is refused (`NotImplementedError`): the source gives no bending
#   factor at the major-axis ends, and none is invented.
# * A `ThroughFlaw` of total length `two_a` is assessed at its tip; `K` uses the half-length `a`.
#
# The example also shows how a random ligament that crosses the validity limit gives a reported
# out-of-validity probability (nothing is dropped or reflected).
#
# Theory: `docs/manual/theory/02_solutions.md` (plate solutions). All inputs are **illustrative**.

# %%
import numpy as np

import critflaw as cf
from critflaw import prob

B, W = 25e-3, 0.5
plate = cf.FlatPlate(thickness=B, width=W)
steel = cf.Material(355e6, 490e6, 207e9, 90e6, yielding="continuous")
membrane = cf.StressState(pm=150e6)

# %% [markdown]
# ## 1. An embedded flaw and the surface flaw of the same depth

# %%
emb = cf.EmbeddedFlaw(two_a=4e-3, two_c=20e-3, p=5e-3)  # 4 mm high, 20 mm long, 5 mm below the surface
re_ = cf.Assessment(plate, emb, steel, membrane).run()
print(
    f"embedded: positions {re_.positions}, K = {np.round(re_.k_primary[:, ...] / 1e6, 3)} MPa m^0.5, "
    f"Lr = {float(re_.lr):.4f}, Kr = {float(re_.kr):.4f}"
)
print("flags:", re_.trace.flags)
surf = cf.Assessment(plate, cf.SurfaceFlaw(a=2e-3, two_c=20e-3), steel, membrane).run()
print(f"surface flaw a = 2 mm: K(deepest) = {float(surf.k_primary[0]) / 1e6:.3f} MPa m^0.5, Lr = {float(surf.lr):.4f}")
# the free surface raises K: the surface flaw of the same crack size has the larger K
assert float(surf.k_primary[0]) > float(re_.k_primary[0])

# %% [markdown]
# ## 2. Ligaments: give the nearer one
#
# `from_ligaments` takes both ligaments and keeps the smaller. Declaring the far ligament as `p` puts the flaw
# outside the validity range (it is not silently reflected).

# %%
near = cf.EmbeddedFlaw.from_ligaments(two_a=4e-3, two_c=20e-3, p_1=12e-3, p_2=9e-3)
print("p from ligaments 12 mm and 9 mm:", float(np.asarray(near.p)))
assert bool(cf.Assessment(plate, near, steel, membrane).run().valid)
far = cf.Assessment(plate, cf.EmbeddedFlaw(4e-3, 20e-3, 12e-3), steel, membrane)
try:
    far.run()
except cf.OutOfValidityError as exc:
    print("far ligament declared as p:", exc)

# %% [markdown]
# ## 3. Bending on an embedded flaw is refused

# %%
try:
    cf.Assessment(plate, emb, steel, cf.StressState(pm=150e6, pb=20e6)).run()
except NotImplementedError as exc:
    print("refused:", exc)
else:
    raise AssertionError("a bending-loaded embedded flaw must be refused")

# %% [markdown]
# ## 4. Through-thickness flaw: the half-length definition
#
# In a very wide plate, `K` tends to `sigma * sqrt(pi * a)` with `a` the **half** length.

# %%
wide = cf.FlatPlate(thickness=B, width=1e3)
rt = cf.Assessment(wide, cf.ThroughFlaw(two_a=40e-3), steel, cf.StressState(pm=100e6)).run()
k_closed = 100e6 * np.sqrt(np.pi * 0.020)
print(
    f"through flaw 2a = 40 mm: K = {float(rt.k_primary[0]) / 1e6:.4f} vs sigma sqrt(pi a) = {k_closed / 1e6:.4f} MPa m^0.5"
)
assert abs(float(rt.k_primary[0]) / k_closed - 1.0) < 1e-6
narrow = cf.Assessment(plate, cf.ThroughFlaw(two_a=40e-3), steel, cf.StressState(pm=100e6)).run()
print(f"in the 500 mm plate the finite-width factor raises K to {float(narrow.k_primary[0]) / 1e6:.4f} MPa m^0.5")

# %% [markdown]
# ## 5. A random ligament crossing the validity limit
#
# With `p ~ Uniform(6, 10) mm` and an 8 mm high flaw in a 25 mm plate, realisations with `2a + 2p > B` are
# outside validity. They are counted and reported, never dropped. The expected share is (10 - 8.5) / 4 = 0.375.

# %%
random_p = cf.Assessment(
    plate,
    cf.EmbeddedFlaw(two_a=8e-3, two_c=40e-3, p=prob.Uniform(6e-3, 10e-3, name="p")),
    steel,
    cf.StressState(pm=prob.Truncated(prob.Normal(150e6, 10e6), lo=0.0, name="pm")),
)
pa = cf.ProbabilisticAssessment(random_p)
u = np.random.default_rng(0).standard_normal((4000, pa.vector.size))
ev = pa.evaluate(u)
share = float(np.mean(~ev.valid))
print(f"share outside validity: {share:.3f} (expected 0.375)")
assert abs(share - 0.375) < 0.03
assert np.all(np.isnan(ev.radial_margin[~ev.valid]))

# %% [markdown]
# ## Interpretation
#
# * An embedded flaw has a lower `K` than a surface flaw of the same crack size but a higher reference stress
#   (its height is twice the crack size `a`).
# * The declared ligament must be the nearer one; anything else is refused as out of validity.
# * Bending on embedded flaws is a documented gap (`TODO(verify)`), refused rather than assessed with an
#   invented factor.
# * In probabilistic work, the out-of-validity share is part of the answer: `ProbabilisticAssessment.solve`
#   counts it as failed in the headline and also reports the bracket with it counted as safe.
