# 9. The knowledge base

The knowledge base (`critflaw.kb`) records **how well each solution is supported by evidence**. It also records
which coefficient sets may be released. It does not change any computed number. It labels results and gates
strict mode.

Theory: [the knowledge base](../theory/08_knowledge_base.md). Examples: *reading the trace and the knowledge
base status*, *coverage map*.

## Statuses

Every equation or table the library implements has a provenance record. Each record has a status on a ladder:

| Status | Meaning |
|---|---|
| `unverified` | no linked test evidence |
| `self-consistent` | evidence only of kinds that cannot catch an error in the source text itself: an independent second transcription, an independent reference implementation written from the same equations, or dimensional consistency |
| `cross-checked` | evidence that can catch such an error: values from the primary reference, or analytic limiting cases and invariants that pin the definitions |
| `fe-verified` | cross-checked **and** covered by FE evidence over an applicability box. The FE pipeline must have completed its own family verification and must not have been validated against that same solution. |

Rules:

- **Only linked evidence counts.** A kind of evidence counts only if a named test carries it. A claim
  without a test does not raise the status.
- **A composite solution takes its weakest record.** A route that combines several records has the status of
  the weakest one. The generic K expression is part of every K route.
- **Suspect cells downgrade.** A tabulated factor read through a table cell flagged as suspect is
  `unverified` in that result.
- **FE evidence is region-scoped.** It promotes a solution only inside its box, and never promotes a solution
  that the FE was itself validated against. The stored FE comparison for the plate surface solution
  (`critflaw.kb.core.D3_EVIDENCE`) therefore never promotes it.

At present no built-in record has linked evidence from a primary reference's own published values. The
source standard's own printed values are kept for information and do not count as such evidence. Many records
are therefore `self-consistent` or `unverified`, and no assessment passes [strict mode](03_deterministic.md#strict-mode).

## The status in results

Every `Result.trace` carries the status of each equation it used:

```python
import critflaw as cf
r = cf.Assessment(cf.FlatPlate(25e-3, 0.5), cf.ThroughFlaw(10e-3),
                  cf.Material(355e6, 490e6, 207e9, 120e6, yielding="continuous"),
                  cf.StressState(pm=150e6)).run()
for dep in r.trace.dependencies():
    print(dep["id"], dep["kb_status"], dep["self_consistent_only"], dep["checked_against_primary_reference"])
print(r.trace.below_cross_checked())     # the equations strict mode would refuse
```

`r.trace.report()` prints `knowledge base: <status>` beside each equation. To look up a single record outside
a result, use `critflaw.kb.status.kb_status("k.plate.surface")`.

## Coverage

```python
from critflaw.kb import core as K
from critflaw.kb.evidence import ORACLE_TESTS

for row in K.coverage(ORACLE_TESTS):
    print(row["id"], row["status"], row["source_class"], row["outer_box"], row["below_cross_checked_records"])
```

`coverage(evidence)` returns one row per built-in solution (`K.BUILTIN`, also indexed as `K.REGISTRY`). Each
row has:

| Key | Content |
|---|---|
| `id`, `family`, `quantity` | solution identifier, geometry/flaw/loading family, and `"K"`, `"reference_stress"`, `"curve"` or `"factor"` |
| `outer_box` | an outer applicability box for the map only. The implementing module's own validity checks are authoritative. |
| `status` | status from the evidence |
| `source_class` | `standard-derived` for every built-in solution |
| `below_cross_checked_records` | the records that keep the solution below `cross-checked` |

`Solution.status(evidence, params=None, fe=(), suspect_cell=False)` gives the composite status at a particular
evaluation. It includes FE evidence that covers `params`.

## Source classes

| Class | Meaning | May ship |
|---|---|---|
| `standard-derived` | transcribed from the licensed standard | only after the owner clears it **and** it has been checked against an open primary source |
| `open` | from an open source | subject to the same audit |
| `handbook` | user-loaded material | never shipped with the library |

## User-loaded solutions

`K.load_user(path)` reads a solution description from JSON, under a mandatory contract:

```json
{
  "id": "user.k.my_surface_flaw",
  "quantity": "K",
  "family": "plate / surface flaw",
  "units": "SI: K in Pa m^0.5, lengths in m",
  "normalisation": "K / (sigma sqrt(pi a))",
  "dimensions": "a depth, c half-length, B thickness",
  "front_position": "parametric angle phi from the free surface",
  "loading": "uniform membrane stress",
  "interpolation": "linear in a/c and a/B",
  "validity": {"a_over_B": [0.1, 0.5], "a_over_c": [0.2, 1.0]}
}
```

```python
from pathlib import Path
sol = K.load_user(Path("my_solution.json"))
sol.source_class, sol.status({})          # "handbook", "unverified"
```

The contract:

- **Required fields.** `id`, `quantity`, `family`, `units`, `normalisation`, `dimensions`, `front_position`,
  `loading`, `interpolation` and `validity`. The descriptive fields must be non-empty strings.
- **`quantity`** is one of `"K"`, `"reference_stress"`, `"curve"` or `"factor"`.
- **`validity`** names at least one parameter range `[low, high]`, with finite numbers and `low < high`.
- **`id`** must not collide with a built-in solution.
- **`status` and `source_class` cannot be declared.** A user solution always enters as `handbook` and
  `unverified`.

Any violation raises `ValueError` naming the field. `load_user` registers the **description** of a solution as
a `Solution` record for the coverage map and the release audit. It does not plug numbers into the assessment
engine.

## The release audit

```python
blocked = K.release_audit()                                   # every record that may not ship
blocked = K.release_audit(cleared=frozenset({"fad.cutoff"}), user=(sol,))
```

`release_audit(cleared=frozenset(), user=())` runs over **every implemented provenance record**, not just the
registry's routes. A record ships only if its id is in `cleared` **and** it has been checked against an open
primary source. A user-loaded (`handbook`) solution never ships. The return value is the list of blocked
record ids, plus any user solution ids.

Today the audit blocks every implemented record, because the built-in coefficient sets and tables are
`standard-derived` and not yet cleared (see the
[index](index.md#release-and-legal-status-of-the-built-in-coefficients)). Run it before redistributing the
library or results that embed its coefficients.

Next: [Troubleshooting](10_troubleshooting.md).
