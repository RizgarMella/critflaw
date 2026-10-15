# 3. Deterministic assessment

Theory: [the FAD method](../theory/01_fad.md), [solutions](../theory/02_solutions.md). Examples: *plate surface
flaw basics*, *Option 1 versus Option 2*, *reserve factors and limiting flaw size*, *reading the trace*.

```python
import numpy as np
import critflaw as cf
```

## Running an assessment

```python
plate = cf.FlatPlate(thickness=25e-3, width=500e-3)
flaw = cf.SurfaceFlaw(a=3e-3, two_c=15e-3)
steel = cf.Material(yield_strength=355e6, uts=490e6, modulus=207e9, toughness=120e6, yielding="continuous")
loads = cf.StressState(pm=150e6, pb=40e6, residual=cf.Residual.as_welded())

problem = cf.Assessment(plate, flaw, steel, loads)
r = problem.run()
print(bool(r.acceptable), float(r.lr), float(r.kr), float(r.radial_margin), r.exit.item())
```

### `Assessment` fields

| Field | Default | Meaning |
|---|---|---|
| `component`, `flaw`, `material`, `loads` | required | see [chapter 2](02_problem_objects.md) |
| `bending_restraint` | `"normal"` | `"pin_jointed"` uses the pin-jointed form of the surface-flaw reference stress (never lower) |
| `interaction` | `"rho"` | plasticity interaction of primary and secondary stress: `"rho"` or `"V"` |
| `plane_stress` | `True` | plastic-zone correction used when secondary stress acts alone (`False`: plane strain) |
| `k_model`, `sref_model`, `fad_model` | `1.0` | model-uncertainty factors ([chapter 2](02_problem_objects.md#model-uncertainty-factors)) |

### `run(on_invalid="raise", strict=False)`

- `on_invalid="raise"` (the default) raises `OutOfValidityError` if **any** sample is outside a solution's
  validity range. Solutions are never extrapolated.
- `on_invalid="mask"` returns those samples blanked instead. `valid` and `acceptable` are `False`, floats are
  NaN, `governing_position` is -1 and `exit` is `"invalid"`. The flag `out_of_validity_samples_present` is set.
  It is then up to you to account for them. They are not failures and not passes.
- `strict=True` refuses a result that used any equation whose knowledge-base status is below
  `cross-checked`. See [strict mode](#strict-mode) below.

## The `Result`

Every array has the common broadcast shape of the inputs. The exceptions are the `*_by_position` and `k_*`
arrays, which have an extra leading axis over crack-front positions.

| Field | Meaning |
|---|---|
| `lr` | `Lr = sref / yield strength` |
| `kr` | `Kr` at the governing crack-front position |
| `acceptable` | the **direct test** at every position: `Kr < f(Lr)` and `Lr < Lr,max` |
| `valid` | inside every solution's validity range |
| `radial_margin` | factor F along the ray from the origin through (`Lr`, `Kr`) to the line (minimum over positions). It is a margin, not the acceptance decision. |
| `exit` | where that ray leaves the diagram: `"curve"`, `"step"` (a vertical drop of the line), `"cutoff"` (`Lr,max`), `"domain_end"` (a tabulated Option 2 law ends first), `"none"` (point at the origin) or `"invalid"` |
| `positions` | names of the crack-front positions, for example `("deepest", "surface")` for a surface flaw, `("tip",)` for a through flaw |
| `governing_position` | index into `positions`: the smallest margin. Where margins tie, the larger `Kr` governs. |
| `kr_by_position` | `Kr` at each position |
| `k_primary`, `k_secondary` | stress intensity factors at each position (Pa·m^0.5) |
| `reference_stress` | Pa |
| `qm` | membrane secondary stress actually used, including residual stress after any relaxation |
| `lr_max` | the plastic-collapse cut-off (the effective cut-off for Option 2) |
| `trace` | a `Trace`: what the result depended on and what it does not account for |

`r.report()` prints a summary. For a single valid evaluation it gives `Lr`, `Kr`, the radial margin, the exit
and the governing position, followed by the trace report.

> **Acceptance versus margin.** The library never decides acceptance from the radial margin. Near a step in the
> line, or when secondary stress makes `Kr` depend on the load, the margin and the direct test can disagree.
> Use `acceptable`.

## Failure assessment lines

### Option 1 (generic line)

The Option 1 line is chosen by `fad_option=1` (the default) and uses only yield strength, tensile strength,
modulus and the declared yielding behaviour. You can evaluate it directly:

```python
curve = cf.Option1Curve(355e6, 490e6, 207e9, "continuous")
lr = np.linspace(0.0, 1.3, 200)
f = curve(lr)                   # 0 at and beyond curve.lr_max
curve.lr_max, curve.f_at_one, curve.step_locations
```

`yielding="discontinuous"` gives the line with a plateau step at `Lr = 1`. `"unknown"` takes the pointwise
minimum of the two forms, so the result is never above either. `problem.curve()` returns the curve an
assessment uses.

### Option 2 (material-specific line)

Option 2 needs a stress-strain law: either an analytic `PowerLaw` or a `TabulatedLaw` built from a measured
curve.

```python
n = float(cf.hardening_exponent_lower_bound(355e6, 490e6))   # the source's lower-bound estimate of n
law = cf.PowerLaw(offset=0.002, n=n)        # offset: plastic strain at the yield strength (0.002 for Rp0.2)
steel2 = cf.Material(355e6, 490e6, 207e9, 120e6, yielding="continuous", fad_option=2, law=law)
r2 = cf.Assessment(plate, flaw, steel2, loads).run()
```

- `PowerLaw(offset, n, luders_strain=None)`: `offset` has no default. Use 0.002 when the yield strength is a
  0.2 % proof strength, and 0 when it is a lower yield point. With `luders_strain` the law has a plateau, and
  the material must then say `yielding="discontinuous"`.
- `TabulatedLaw.from_engineering(stress, strain, yield_strength, modulus, yielding, luders_strain=None)` or
  `TabulatedLaw.from_true(...)`. Here `yielding` is `"continuous"` or `"plateau"`, and a plateau needs
  `luders_strain`. The data are cleaned: sorted, the declared plateau removed, made monotone (an adjustment of
  more than 5 % of yield is refused with `StressStrainDataError`, a `ValueError` subclass imported with
  `from critflaw.fad.option2 import StressStrainDataError`; it is not in the top-level `cf` namespace), and the
  noise-level plastic strain zeroed.
  Every cleaning step is recorded in `law.notes`, and the notes appear in the trace options as
  `material_law_notes`.

```python
# stress_pa, strain: your measured engineering stress (Pa) and strain (-) arrays, including the elastic range
law_t = cf.TabulatedLaw.from_engineering(stress_pa, strain, yield_strength=355e6, modulus=207e9, yielding="continuous")
steel_t = cf.Material(355e6, 490e6, 207e9, 120e6, yielding="continuous", fad_option=2, law=law_t)
```

The Option 2 line is made non-increasing by a monotonicity rule that can only lower it. The largest lowering
is reported in the trace option `option2_monotone_correction`. For a tabulated law, a lowering above 0.05 means
the data are suspect, and the realisation is out of validity. A law that has not strained plastically beyond
`Lr = sqrt(2)` is also out of validity, because the line is not defined for it. When the tabulated data end
before the plastic-collapse cut-off, the line ends there: `exit` is `"domain_end"` and the flag
`stress_strain_data_end_before_collapse_cut_off` is set.

## Secondary and residual stress

Secondary stresses (`qm`, `qb`, and the residual stress) enter `Kr` through the plasticity-interaction term of
the chosen `interaction` method, not `Lr`. The trace shows which branch was used:

- `fad.rho.simplified` or `fad.v.simplified` is listed when primary and secondary stresses act together.
- `fad.kjs.plastic_zone` and the flag `secondary_stress_acting_alone` appear when there is no primary stress.
  A plastic-zone-corrected K is then used, with `plane_stress` choosing the correction.
- The flag `chi_above_4_simplified_interaction_over_conservative` means the simplified interaction is outside
  the range where it is accurate, and it errs on the conservative side.
- The flag `negative_primary_k_set_to_zero` means that for a deep flaw under bending the primary K came out
  negative at some position. It is assessed as zero, which is conservative.

## Reserve factors and limiting flaw size

These functions take a **scalar** assessment (one sample). Acceptability is not assumed to be monotone in the
scanned parameter. Each function scans a stated range, polishes every change of acceptability by bisection,
and reports what it could not resolve.

```python
rf_load = cf.reserve_factor_on_load(problem)          # factor on pm and pb; secondary stress held
rf_tough = cf.reserve_factor_on_toughness(problem)    # divisor on Kmat
lim = cf.limiting_flaw_size(problem)                  # surface flaw: depth at fixed a/c; through flaw: 2a
print(rf_load.value, rf_load.unique, rf_load.crossings)
```

| Function | Scans | Defaults |
|---|---|---|
| `reserve_factor_on_load(problem, min_factor=1e-3, max_factor=100.0, points=400)` | factor on `pm`, `pb` (log grid) | |
| `reserve_factor_on_toughness(problem, min_factor=1e-3, max_factor=1000.0, points=400)` | divisor on a numeric `toughness` (log grid) | not for `MasterCurve` toughness |
| `limiting_flaw_size(problem, points=400)` | `SurfaceFlaw` depth up to B at fixed shape, or `ThroughFlaw` length up to W (linear grid) | plate surface and through flaws only |

Each returns a `Crossing`:

| Field | Meaning |
|---|---|
| `value` | smallest scanned value at which the assessment is **not** acceptable (the not-acceptable side of the crossing); `inf` if none was found |
| `bounded` | `False` if no such value was found in range |
| `unique` | `False` if acceptability changes more than once. Look at `crossings`. |
| `crossings` | every change of acceptability found, in increasing order |
| `below_range` | not acceptable even at the smallest value probed (the range is extended downward first). `value` is then only an upper bound. For toughness this means no reserve exists, for example because plastic collapse governs. |
| `limited_by_validity` | `value` is where a solution leaves its validity range, **not** a predicted failure |
| `unacceptable_at_zero` | (load only) not acceptable with exactly zero primary load: `value = 0` |
| `range`, `points`, `log_spaced` | the scan actually performed. Regions narrower than the grid spacing can be missed. |

Always check `limited_by_validity` before reading `limiting_flaw_size(...).value` as a critical size.

## Reading the trace

`r.trace` records what the result depended on:

| Attribute / method | Content |
|---|---|
| `edition` | edition of the source standard (`"2013+A1"`) |
| `equations` | identifiers of the equations used, for example `k.plate.surface`, `sref.plate.surface`, `fad.option1.continuous`, `fad.cutoff` |
| `options` | the choices in effect: `fad_option`, `material_law`, `yielding`, `interaction`, `bending_restraint`, `plastic_zone_correction`; for cylinders also `k_route`, `sref_route` and route notes; for Option 2 the monotone correction and law notes |
| `flags` | conditions met during the run (see [troubleshooting](10_troubleshooting.md#trace-flags)) |
| `excluded_uncertainty` | what the result does **not** account for: the model uncertainty of K, of the reference stress and of the line (unless the corresponding factor was an array), and the scatter of every input |
| `citations()` | a citation per equation: the primary reference the source gives, marked "(as cited by the source standard)" until it has been checked against that reference, or "[no primary reference given by the source standard]" |
| `dependencies()` | per equation: `id`, `citation`, `checked_against_printed_source`, `checked_against_primary_reference`, `self_consistent_only`, `kb_status` |
| `kb_status(eq)` | the knowledge-base status of one equation in this result ([chapter 9](09_knowledge_base.md)) |
| `below_cross_checked()` | equations whose status is `unverified` or `self-consistent` |
| `self_consistent_only()`, `not_checked_against_primary()` | the equations in those classes |
| `report()` | all of the above as text |

An excerpt of `r.report()` for the example at the top of the chapter (the exact wording and statuses depend on
the installed version):

```text
Evaluations: 1; acceptable: 1; outside validity: 0
Lr = ..., Kr = ..., radial margin = ... (exit: curve), governing position: deepest
Source edition: 2013+A1
Options: fad_option=1, material_law=none, yielding=continuous, interaction=rho, ...
Flags: none
Equations used:
  - ...  [knowledge base: self-consistent; not checked against a primary reference]
  - ...  [knowledge base: unverified; SELF-CONSISTENT ONLY (no oracle independent of the source text); ...]
Not accounted for:
  - model uncertainty of the stress intensity factor solution
  ...
```

How to read the labels on each equation:

- **knowledge base: ...** is the status from the linked test evidence: `unverified`, `self-consistent`,
  `cross-checked` or `fe-verified`.
- **SELF-CONSISTENT ONLY** means no test independent of the source text checks this equation, so an error in
  the source itself would not be caught.
- **not checked against a primary reference** means the primary reference's own published values have not
  been compared.
- **NOT checked against the printed source** would mean the transcription itself is unchecked.

## Strict mode

```python
r = problem.run(strict=True)
```

`strict=True` raises `ValueError("strict assessment refused: equations below 'cross-checked' in the knowledge
base: ...")` if any equation the result used has a knowledge-base status below `cross-checked`. By default the
status of every equation is only recorded in the trace.

At present every assessment uses at least one equation below `cross-checked`, for example the generic K
expression shared by all K routes. **Strict mode therefore refuses every assessment today.** It exists so that
a workflow that requires cross-checked equations fails loudly instead of silently relying on weaker evidence,
and so that it starts passing as evidence is added. A tabulated cylinder factor read through a table cell
flagged as suspect is always `unverified` in that result ([chapter 6](06_cylinders_embedded.md)).

Next: [Probabilistic assessment](04_probabilistic.md).
