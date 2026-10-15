# Development roadmap

**Status: development stopped in October 2026.** The repository was published as it stood. The sections below
record how it was built and what was left open, for anyone who picks it up.

This is the one place in the repository that describes how `critflaw` was developed and what remains. Everything
else (the source, the manuals, the examples, the tests) describes the software as it is.

## 1. How it was built

The library was built in phases against a written specification, each phase gated by evidence. Every design note
and code change was reviewed adversarially by two independent reviewers: a model from a different lab (through
`tools/adversary/partner.py`) and a second independent reviewer that saw only the artefact and the specification. Every
finding was answered by a fix or a rebuttal with executable evidence; disagreements left after two rebuttal rounds
were logged for the owner. Equations were transcribed twice from the source document, independently, and the
partner model derived independent reference implementations from the transcriptions; those are kept unchanged
in `tests/oracles/`.

The working record (proposal, specification, design notes, review rounds and responses, partner logs and the FE
verification run records) is not published; this document summarises what it established.

| phase | delivered |
|---|---|
| 0 | specification from the adversarially reviewed proposal; tooling |
| 1 | deterministic core: problem objects, plate K and reference stress, Option 1 FAD, secondary stress and ρ, margins, reserve factors, limiting flaw size, trace |
| 2 | Option 2 and stress–strain handling, toughness (Master Curve, Charpy), random variables, Nataf, MC, FORM/SORM, importance sampling, subset simulation, partial factors |
| 3 | fatigue growth, time-dependent Pf, inspection updating; cylinders, embedded and through-thickness flaws |
| 4 | WARP3D backend: mesher, decks, runner, parser, cache, QA layers, elastic and elastic–plastic surface-flaw cases |
| 5 | experimental deepest-point Option 3 emulator (GP on log J increments with an Option 2 prior) |
| 6 | Pf with the emulator: function draws, bound pair, crossed variance components, acquisition and stopping |
| 7 | knowledge base: status from linked evidence, FE evidence scope, user-loaded solutions, release audit |
| clean-up | repository restructured, theory and user manuals, worked examples as scripts and notebooks |

## 2. Verification state

| item | state |
|---|---|
| deterministic, probabilistic, fatigue modules | tested against independent oracles; statuses per equation in the knowledge base (`critflaw.kb`) |
| FE pipeline | verified for the code (path independence, elastic K, equilibrium, mesh convergence at the mid point: deepest-point production error 0.34–0.42 % for Lr 0.8–1.2); family-level verification incomplete (D4/D5 references unavailable, D7 not resolved across the box, deep-flaw/bending band of D3 unresolved) |
| Option 3 emulator (V9) | leave-one-curve-out RMSE 0.0241 (target 0.02, not met), coverage 0.935; experimental |
| Pf with the emulator (V10) | not passed: 6 of 20 seeds run, coverage 4/6, 2 premature stops, feature-count check failed in 2 seeds (theory manual, chapter 07) |

## 3. Open decisions for the owner

- **D-1 Embedded-flaw re-characterisation.** The static procedure assesses an embedded flaw directly; the optional
  ligament re-characterisation after an unacceptable result is not implemented (a conservative omission). The
  owner confirms this reading of the source's annex, or the rule is added.
- **D-2 Training on FE data whose family verification is incomplete.** The emulator is trained on such data and is
  therefore experimental (`experimental=True` required, never assessment grade). Needs the owner's explicit
  approval; reversible.
- **D-3 Support of the small-strain cap for a new input.** The emulator's support is set by a rule on the reference
  strain, capped at step 11 and Lr 1.2; no rule can prove in advance that a new input stays inside its FE strain cap.
- **D-4 Knowledge-base granularity.** The content inventory works per module, not per constant; non-FE evidence
  counts over a record's declared applicability unless a region is declared. Refinements, listed below.
- **Unsourced items.** Records without an open primary reference are listed in `docs/unsourced_items.md`.
- **Release.** Coefficient sets derived from the source standard are blocked by `release_audit` until cleared.

## 4. Next steps on current resources

1. **V10b** (pre-registered): the stopping interval combines the draw component with the sampling error and an
   in-run random-feature error estimate (F against 2F); default F = 2000; same criteria as V10. One seed at a
   time, about 10 minutes each.
2. Owed minor items: per-constant content inventory and per-test evidence regions (D-4); mutable mappings inside
   frozen dataclasses in the knowledge base; rank-one GP updates in the acquisition loop (the loop refits today);
   embedded flaws in cylinders; `ExtremeLoad(component="pressure")`.
3. References to obtain: EPRI NP-1931 and Lei (2004) for D4/D5; NASA TM 85793 for the deep-flaw/bending band;
   Wang (1991) for the transition arc; the source of the 6 % band.
4. API reference generated from the docstrings; `CITATION.cff`; release notes (nothing is published without the
   owner asking).

## 5. Refinements with more compute

More compute refines the same pipeline (more levels, more data, wider coverage); it replaces nothing. Results enter
the database and solve cache under new keys; labels change only when the corresponding comparison passes.

| refinement | today (15 GB workstation) | with more compute | how it plugs in | gain |
|---|---|---|---|---|
| mesh convergence, four levels | three levels (mid, corner) | run level 4 (and 5), *est.* 20–40 min each without paging | `benchmarks/fe_d7_convergence.py <point> 4`; `fecase.LEVELS` | four-level clauses evaluable; convergence for the family |
| near-surface datum | mesh-sensitive, unresolved | front-direction refinement at levels 1–3 | `CrackBlockParams`; same driver | surface-point curve; complete-flaw Option 3 |
| convergence over the box | mid and corner points | the four box corners incl. n = 3, σ0/E = 1e-3, a/B 0.6, bending | `POINTS` in the drivers | family-wide convergence statement |
| production mesh | level 1 | train on level 2; level-3 pairs for the discrepancy | `FECase(level=LEVELS[1])` | smaller discretisation term |
| held-out validation | leave-one-curve-out + 2 cases | 12 independent held-out cases | `benchmarks/option3_campaign.py heldout` | V9 on an independent design |
| training data | 22 cases + 2 pairs, 5 inputs | 200–500 cases; 50+ pairs (GP discrepancy) | `training_design(n_train, n_pairs)` | tighter intervals |
| inputs | ν = 0.3, c/W ≤ 0.05 fixed | ν and c/W as inputs (7-D) | `design.NAMES/LOWER/UPPER` | two validity bounds removed |
| geometry range | a/B 0.2–0.6, a/c 0.4–1.0 | a/B 0.05–0.8, a/c 0.2–1.0 | mesher parameters + topology id | the full proposed box |
| parallel FE | one container at a time | N concurrent containers | `WarpRunner` per job | wall time ÷ N |
| acquisition | sequential, small batches | batch acquisition with parallel FE | already batch-capable | fewer wall-hours per Pf target |
| direct-FE Pf check | none run | direct-FE MC on a reduced problem | `WarpJSolver` | independent check of the surrogate interval |
| elastic K database | 34 cases | dense design over the box, three levels each | `benchmarks/fe_d3_elastic_k.py` | FE-verified K entries |
