# 8. The K-solution knowledge base

Every equation, coefficient set and table that the library implements has a record in the equation register
(`src/critflaw/provenance.py`). The knowledge base (`critflaw.kb`) builds on that register to answer three questions
for every result: *where did this solution come from and may it be shipped*, *how far has it been checked*, and
*does any FE evidence apply to it here*. This chapter defines the vocabulary and the rules. Implementation:
`critflaw.provenance`, `critflaw.kb.core`, `critflaw.kb.evidence`, `critflaw.kb.status`.

---

## 8.1 The equation register

A record (`EquationRecord`) carries:

| Field | Meaning |
|---|---|
| `id` | a stable register id (e.g. `k.plate.surface`) |
| `description`, `implemented_in` | what and where |
| `primary_reference` | the primary reference **as cited by the source standard** (a key into the bibliography), or `None` when the standard cites none; never invented |
| `checked_against_primary` | `yes` / `no` / `unobtainable`: whether the implemented coefficients have been compared with the primary publication itself. Every record is currently `no`; a citation then means "as cited by the source standard" and results say so |
| `pdf_checked` | the record was transcribed a second time from the printed page and diffed (true for every record) |
| `oracles` | the oracle kinds (a)–(e) **claimed** for the record, declared explicitly (there is no default claim) |
| `applicability` | the validity statement |

`self_consistent_only` is true when the record claims neither kind (a) nor kind (b) — no oracle independent of the
source text. The clause of the standard each item came from is recorded only in a private, git-ignored map; it is
never part of the code, the documentation or a result. Items with `primary_reference=None` are listed in
`docs/unsourced_items.md`. The register currently holds 51 records.

Oracle kinds ([index.md](index.md) §4): (a) published values of the primary reference; (b) analytic limiting cases and
invariants that pin variable definitions; (c) an independent second transcription diffed against the first;
(d) an independent re-implementation from the transcribed equations; (e) dimensional consistency. The source
standard's own worked examples and printed tables do not count as kind (a): reproducing them is recorded as
information only (`w`).

## 8.2 Source classes

| Class | Meaning | Ships in a release? |
|---|---|---|
| `standard-derived` | transcribed from the licensed standard, whatever primary reference the standard cites. The default class of every current built-in coefficient set and table | not until the item has been checked against an open source (`checked_against_primary == "yes"` against the open paper) **and** the owner has cleared it |
| `open` | implemented from, and checked against, an open-literature source | yes |
| `handbook` | user-loaded from any compiled source; lives outside the package | never |

## 8.3 The status ladder

Four rungs, aligned with the oracle kinds:

| Status | Rule |
|---|---|
| `unverified` | no linked oracle evidence |
| `self-consistent` | only kinds (c), (d), (e) are linked. These cannot catch an error in the source text: a misprinted coefficient transcribed twice and re-implemented independently is still wrong |
| `cross-checked` | kind (a) or kind (b) is linked |
| `fe-verified` | `cross-checked`, **and** FE evidence that covers the evaluation parameters and may promote this solution (§8.5) |

### 8.3.1 Status from linked evidence

A kind counts only when it is both **claimed** in the register and **linked** to named tests that carry it
(`kb.evidence.ORACLE_TESTS`, keyed by record, kind and test id; a test checks that every linked test exists). A claim
without a linked test does not count; evidence for an unclaimed kind does not count either until the record claims it.
Formally, for record $r$ with claimed kinds $C_r$ and linked kinds $L_r$,

$$ \text{status}(r) = \begin{cases} \texttt{cross-checked}, & (C_r \cap L_r) \cap \{a, b\} \ne \emptyset \\ \texttt{self-consistent}, & (C_r \cap L_r) \cap \{c, d, e\} \ne \emptyset \\ \texttt{unverified}, & \text{otherwise.} \end{cases} $$

**Region scope.** Evidence may carry an applicability box; a kind whose evidence has a box counts only at evaluation
parameters inside it. Evidence without a box counts over the record's declared applicability, which the implementing
module enforces by refusal. No partial-range evidence box is declared at present (every linked test exercises its
record over its declared range); establishing per-test regions for every linked test is an open item.

### 8.3.2 Composite status of a solution

A `Solution` is a callable route (for example, the cylinder axial surface flaw under pressure: the general K form, the
plate surface-flaw factors and the surface bulging factor). Its status at an evaluation is the **minimum** over the
records it uses. An evaluation that interpolated through a table cell flagged `suspect`
([02_solutions.md](02_solutions.md) §2.4.3) is `unverified` for that evaluation, and the flag stays in the trace.

### 8.3.3 Where the library stands

The audit of all 51 records against the tests linked kinds b 17 times, c 40, d 23, e 0 and a 0 (no test compares a
primary reference's own published values yet; six links that reproduce the source standard's own worked values or
tables are recorded as information and do not count). Twenty-two claimed letters on 16 records have no linked test
(mostly the cylinder bulging factors and tables, and some toughness relations). Because every K route also uses the
generic K form `k.general`, which has kind (c) and (d) evidence only, **every K route is currently below
`cross-checked`**. That is the honest state; the audit lists what would lift it (kind (a) comparisons with the primary
references, or kind (b) evidence linked to `k.general`).

## 8.4 The status in results

Every assessment trace records, for each equation used: its citation, whether it was checked against the printed
source and against the primary reference, whether it is self-consistent only, and its knowledge-base status
([01_fad.md](01_fad.md) §1.9). Nothing is silent. The existing deterministic APIs keep working whatever the status;
`Assessment.run(strict=True)` refuses a result that used any equation below `cross-checked` (today that refuses every
K route). User-loaded solutions enter as `unverified` (§8.6) and so never satisfy strict use.

## 8.5 FE evidence

An FE evidence record (`FEEvidence`) names the solution it compares, its applicability box, whether the FE family
verification is complete, and **`validated_against`**: the solutions the FE pipeline was itself validated against.
Evidence **never promotes** a solution in its own `validated_against` set: an FE pipeline checked against a K solution
cannot then be used to verify that K solution. It promotes a solution to `fe-verified` only when

1. the solution is already `cross-checked` from non-FE evidence;
2. the evidence box covers the evaluation parameters (an empty box covers nothing);
3. the solution is not in `validated_against`;
4. the FE family verification is complete (`FAMILY_VERIFIED`; [06_fe_option3.md](06_fe_option3.md) §6.1.7).

The stored D3 comparison of the plate surface-flaw solution (34 elastic cases, $a/B \in [0.2, 0.6]$,
$a/c \in [0.4, 1.0]$: 25 pass, 8 unresolved, 1 fail; [06_fe_option3.md](06_fe_option3.md) §6.2) is evidence with
`validated_against = {k.plate.surface}` and `family_verified = False`; a test confirms that it never promotes
`k.plate.surface`. No solution is currently `fe-verified`.

## 8.6 User-loaded solutions

`kb.core.load_user(path)` reads a JSON solution under a mandatory contract: units; the K normalisation (for example
$K/(\sigma\sqrt{\pi a})$); full versus half dimensions; the crack-front position definition; the loading convention;
interpolation coordinates and rule; a validity box with at least one finite range $[low, high]$, $low < high$. Missing
or empty fields are refused. A user file cannot reuse a built-in id, nor declare its own status or class: it always
enters as `unverified` and `handbook`.

## 8.7 The release audit

`kb.core.release_audit(cleared)` runs over **every implemented register record** (the package's coefficient sets, tables and
formulae, not only the registry routes) plus any user-loaded solution. A record may ship only if the owner has cleared
it **and** it has been checked against an open primary source; `handbook` material never ships. The function returns
the blocked ids; today every built-in record is blocked (all are `standard-derived` and none is checked against its
primary reference). Whether each transcribed coefficient set may ship in a public release, or must be loaded from a
user-supplied file, is the owner's decision (`docs/unsourced_items.md` §3).

An inventory test supports the audit: it scans the package (excluding the knowledge base, FE and emulator modules) for
numeric literal containers of six or more numbers and requires each module that holds one to be mapped to the register
records it implements; a new module with tables fails the test until mapped. The inventory works at module
granularity: a new coefficient set added to an already-mapped module is not flagged by it (a per-constant inventory is
an open item).

## 8.8 The coverage map

`kb.core.coverage(evidence)` lists, for every built-in solution: its family and quantity, an outer validity box (labelled as
an outer box — the modules' own coupled refusal rules stay authoritative), its status from the evidence, its source
class, and the records that keep it below `cross-checked`. Gaps in the map are where Option 3 FE work would add value
([06_fe_option3.md](06_fe_option3.md)).

## 8.9 Limits

- The ladder measures checks, not truth: `cross-checked` by a limiting case pins definitions but cannot prove every
  interior coefficient; kinds (c)–(e) cannot detect a misprint in the source.
- Region scoping is in place but no per-test regions are declared; non-FE evidence counts over the record's declared
  applicability.
- The inventory is per module, not per constant.
- `checked_against_primary` is `no` for every record; comparisons with the primary publications are the main route to
  `cross-checked` K solutions and to shippable `open` items.
