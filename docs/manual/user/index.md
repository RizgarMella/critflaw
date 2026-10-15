# critflaw user manual

> **⚠️ Research software in development: not verified, not validated, not for real-world use.** See
> [`DISCLAIMER.md`](../../../DISCLAIMER.md).

`critflaw` is an open Python library for the **engineering critical assessment (ECA) of flaws in metallic
structures** using failure assessment diagram (FAD) methods, with deterministic and probabilistic routes. You
describe a component, a flaw, a material and the stresses. The library evaluates the assessment point
(`Lr`, `Kr`) against the failure assessment line, and from there it can give reserve factors, limiting flaw
sizes, failure probabilities, fatigue crack growth and time-dependent reliability.

This manual covers how to use the library. The equations, their assumptions and their limits are in the
[theory manual](../theory/01_fad.md). Worked examples, as Python scripts and notebooks, are in `examples/`. Each
chapter below names the examples that go with it by topic.

## What the library is

- A **vectorised assessment engine**. Every numeric input can be a scalar or a NumPy array (one value per
  sample), so one call can evaluate a single case or a million Monte Carlo realisations.
- A **probabilistic layer**. Any input can be a random variable (`critflaw.prob`), with Gaussian-copula
  dependence. Solvers report a *guaranteed* Monte Carlo interval and, for rare events, a *conditional*
  subset-simulation estimate. Each result is labelled for what it actually claims.
- A **fatigue and time-dependent layer**: crack growth under a load spectrum, with inspection and repair
  policies and failure probability against time.
- **Explicit about its limits.**
  - A solution used outside its validity range is refused (`OutOfValidityError`), never extrapolated.
  - Invalid realisations are counted and reported, never silently dropped.
  - Equations the source does not give raise `NotImplementedError("TODO(verify): ...")`.
  - Every result carries a `Trace` that lists the equations used, their verification status and the
    uncertainties that are *not* accounted for.
- Optionally, a **finite-element back end** (WARP3D, run in Docker) and an **experimental Option 3 emulator**
  that builds material- and geometry-specific FAD curves from FE results.

## What the library is not

- **It is not certified** for any regulated application. It supports, and does not replace, the judgement of a
  competent engineer. Results are not a substitute for an assessment carried out and signed off under the
  applicable code.
- **It does not decide anything for you.** It refuses to supply defaults where a wrong assumption would matter.
  For example, you must declare the yielding behaviour, the cylinder K route, and whether a leak counts as
  failure.
- **It does not give a per-component or per-year probability by default.** A probabilistic result is the
  failure probability *per flaw, given that the flaw exists* (see [chapter 4](04_probabilistic.md)).
- **The Option 3 emulator is not assessment-grade.** It is experimental, and the library will not use it in an
  assessment unless you say `experimental=True` (see [chapter 8](08_option3_emulator.md)).

## Release and legal status of the built-in coefficients

The equations follow BS 7910:2013+A1:2015 ("the source standard"). The edition is recorded in every trace as
`2013+A1`.

Every built-in coefficient set and table is classed as **`standard-derived`**. This covers the
stress-intensity and reference-stress solutions, the cylinder tables, the fatigue crack growth catalogue and the
published partial factors. These values were transcribed from the licensed standard. They are used locally and
are **not cleared for public release** until each one has been checked against an open primary source and
cleared by the owner. `critflaw.kb.core.release_audit()` lists everything that may not ship. Today that is
every implemented record (see [chapter 9](09_knowledge_base.md)). If you redistribute the library or its
outputs, check that list first.

Where the source standard cites a primary reference for an item, the library cites that reference, not the
standard. Items for which the standard gives no reference are listed in `docs/unsourced_items.md`.

## Conventions used throughout

| Convention | Rule |
|---|---|
| Units | SI floats everywhere: m, Pa, Pa·m^0.5, J/m^2. Temperatures for the Master Curve are in °C. Fatigue: m/cycle and Pa·m^0.5. |
| Problem objects | Immutable (frozen dataclasses). Make variants with `.with_(field=value)`. |
| Arrays | Every numeric field broadcasts. Results have the common broadcast shape. |
| Refusals | Outside validity: `OutOfValidityError` (or masked samples with `on_invalid="mask"`). Missing equation: `NotImplementedError("TODO(verify): ...")`. |
| Acceptance | Decided by the direct test `Kr < f(Lr)` and `Lr < Lr,max`, never from the radial margin. |

## Contents

1. [Installation](01_install.md): from source with `uv` or `pip`, optional extras, optional WARP3D image.
2. [Problem objects](02_problem_objects.md): geometry, flaws, materials, toughness, stresses, residual
   stress, units, immutability, vectorisation.
3. [Deterministic assessment](03_deterministic.md): `Assessment`, `Result`, Option 1 and 2 curves, secondary
   stress, margins, reserve factors, limiting flaw size, the `Trace` and its report, strict mode.
4. [Probabilistic assessment](04_probabilistic.md): `ProbabilisticAssessment`, distributions, dependence,
   Monte Carlo, subset simulation, importance sampling, FORM and SORM, intervals, diagnostics, partial factors,
   inspection-conditioned flaw sizes.
5. [Fatigue and time-dependent reliability](05_fatigue.md): growth laws, `Spectrum`, `ExtremeLoad`,
   `TimeDependentAssessment`, `Inspection`.
6. [Cylinders and embedded flaws](06_cylinders_embedded.md).
7. [The FE back end (WARP3D)](07_fe_backend.md): `FECase`, `WarpJSolver`, `PlasticCase` and `run_case`,
   the cache, QA checks, resource use.
8. [The Option 3 emulator (experimental)](08_option3_emulator.md): design, curves, training, prediction
   statuses, validation, `assessment_curve`, failure probability with the emulator, active learning.
9. [The knowledge base](09_knowledge_base.md): verification statuses, coverage, user-loaded solutions, the
   release audit.
10. [Troubleshooting](10_troubleshooting.md): the errors you are likely to meet and what they mean.
11. [FAD graphics](11_fad_graphics.md): drawing assessment lines, points, paths and probability-density fields.
