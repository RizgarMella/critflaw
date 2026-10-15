# %% [markdown]
# # The knowledge base: verification status from linked evidence
#
# Every equation the engine uses is a provenance record. Its **status** is computed, never declared, from the
# oracle evidence that named tests actually carry, on a ladder
# `unverified < self-consistent < cross-checked < fe-verified`:
#
# * evidence that cannot catch an error in the source text (a second transcription, an independent
#   re-implementation, dimensional or regression checks) gives at most `self-consistent`;
# * published values of the primary reference, or analytic limiting cases that pin the variable definitions,
#   give `cross-checked`;
# * FE evidence can promote a solution to `fe-verified` only inside the FE evidence's box, only when the FE family
#   verification is complete, and never for a solution the FE itself was validated against.
#
# The example reads the status of a result's equations, uses `strict=True`, prints the coverage map of the built-in
# solutions, loads a user-supplied solution under the mandatory contract and runs the release audit.
#
# Theory: `docs/manual/theory/08_knowledge_base.md`.

# %%
import json
from collections import Counter
from pathlib import Path

import critflaw as cf
from critflaw.kb import core as K
from critflaw.kb import evidence as ev
from critflaw.provenance import EQUATIONS

SCRIPT = "__file__" in globals()
HERE = (
    Path(__file__).resolve().parent
    if SCRIPT
    else (Path.cwd() / "examples" if (Path.cwd() / "examples").is_dir() else Path.cwd())
)
OUT = HERE / "_output"
OUT.mkdir(exist_ok=True)

# %% [markdown]
# ## 1. The status of what a result used

# %%
problem = cf.Assessment(
    cf.FlatPlate(25e-3, 0.5),
    cf.SurfaceFlaw(4e-3, 24e-3),
    cf.Material(355e6, 490e6, 207e9, 100e6, yielding="continuous"),
    cf.StressState(pm=150e6),
)
res = problem.run()
for dep in res.trace.dependencies():
    print(f"{dep['id']:<26} {dep['kb_status']:<16} {dep['citation']}")
below = res.trace.below_cross_checked()
print("\nbelow cross-checked:", below)

# %% [markdown]
# ## 2. `strict=True` refuses a result that rests on anything below cross-checked

# %%
try:
    problem.run(strict=True)
    print("strict run accepted: every equation is at least cross-checked")
except ValueError as exc:
    print("strict run refused:", exc)
    assert below

# %% [markdown]
# ## 3. The ladder for one record, from evidence

# %%
rid = "k.plate.surface"
print(
    f"{rid}: claimed oracle kinds {EQUATIONS[rid].oracles}, linked {sorted(ev.ORACLE_TESTS.get(rid, {}))}, "
    f"status {K.record_status(rid, ev.ORACLE_TESTS)}"
)
print("with no evidence at all:", K.record_status(rid, {}))
assert K.record_status(rid, {}) == "unverified"

# %% [markdown]
# ## 4. Coverage map of the built-in solutions

# %%
rows = K.coverage(ev.ORACLE_TESTS)
for row in rows:
    print(f"{row['id']:<38} {row['status']:<16} {row['family']}")
print("\nstatus counts:", dict(Counter(r["status"] for r in rows)))
assert {r["id"] for r in rows} == set(K.REGISTRY)

# %% [markdown]
# ## 5. FE evidence is region-scoped and never promotes what validated the FE
#
# The stored FE comparison for the plate surface solution covers a box of `a/B` and `a/c`, but the FE was itself
# validated against that solution, so it can never promote it.

# %%
sol = K.REGISTRY["k.plate.surface"]
inside = {"a_over_B": 0.4, "a_over_c": 0.7}
print("D3 evidence:", K.D3_EVIDENCE.note)
print("covers the point:", K.D3_EVIDENCE.covers(inside), "| may promote:", K.D3_EVIDENCE.promotes("k.plate.surface"))
print("status with the FE evidence:", sol.status(ev.ORACLE_TESTS, inside, (K.D3_EVIDENCE,)))
assert sol.status(ev.ORACLE_TESTS, inside, (K.D3_EVIDENCE,)) != "fe-verified"

# %% [markdown]
# ## 6. A user-supplied solution under the contract
#
# A user solution must declare its units, normalisation, dimensions, front position, loading, interpolation and
# validity box. It enters as `unverified` and `handbook`; it cannot declare its own status or reuse a built-in id.
# (The file below only describes a hypothetical solution; no coefficients are involved.)

# %%
user_file = OUT / "18_user_solution.json"
user_file.write_text(
    json.dumps(
        {
            "id": "user.k.example",
            "quantity": "K",
            "family": "plate / surface flaw (user)",
            "units": "SI",
            "normalisation": "K / (sigma sqrt(pi a))",
            "dimensions": "a depth, c half-length, B thickness",
            "front_position": "parametric angle from the free surface",
            "loading": "membrane",
            "interpolation": "linear in a/c and a/B",
            "validity": {"a_over_B": [0.1, 0.5], "a_over_c": [0.2, 1.0]},
        }
    ),
    encoding="utf-8",
)
user = K.load_user(user_file)
print(f"loaded {user.id}: source class {user.source_class}, status {user.status({})}, box {user.outer_box}")
bad = json.loads(user_file.read_text(encoding="utf-8")) | {"status": "cross-checked"}
user_file.write_text(json.dumps(bad), encoding="utf-8")
try:
    K.load_user(user_file)
except ValueError as exc:
    print("refused:", exc)

# %% [markdown]
# ## 7. Release audit
#
# A record may ship only if it was cleared by the owner **and** checked against an open primary source;
# user-loaded (handbook) material never ships.

# %%
blocked = K.release_audit(user=(user,))
print(
    f"blocked: {len(blocked)} of {len(EQUATIONS)} provenance records, plus user solutions: "
    f"{[b for b in blocked if b.startswith('user.')]}"
)
assert "user.k.example" in blocked

# %% [markdown]
# ## Interpretation
#
# * The status of every equation in a result is in its trace; `strict=True` turns "below cross-checked" into a
#   refusal instead of a note.
# * Status is a function of linked evidence: adding a linked test of kind a (primary-reference values) or b (a
#   definition-pinning limiting case) changes it;
#   claiming a kind without a test does not.
# * FE evidence is never circular: the FE validated against a solution cannot promote that solution.
# * The release audit is conservative by design: today every standard-derived record is blocked until it has been
#   cleared and checked against an open primary source.
