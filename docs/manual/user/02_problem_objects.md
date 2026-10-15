# 2. Problem objects

An assessment is built from four objects: a **component** (geometry), a **flaw**, a **material** and a
**stress state**. They are combined in an `Assessment` ([chapter 3](03_deterministic.md)). This chapter
describes each object, the units, and the three rules that apply to all of them: SI floats, immutability, and
vectorisation.

Theory: [the FAD method](../theory/01_fad.md), [solutions](../theory/02_solutions.md),
[materials and toughness](../theory/03_materials_toughness.md). Examples: *plate surface flaw basics*, *building
problems from arrays*.

```python
import numpy as np
import critflaw as cf
```

## Units: SI floats

Every number is an SI float. Internally there is no unit system and no conversion, so a value in MPa or mm
gives a wrong answer without any error.

| Quantity | Unit | Example |
|---|---|---|
| Length (thickness, flaw size, radius) | m | `25e-3` for 25 mm |
| Stress, strength, modulus | Pa | `355e6` for 355 MPa, `207e9` for 207 GPa |
| Fracture toughness `Kmat` | Pa·m^0.5 | `120e6` for 120 MPa·m^0.5 |
| J-integral | J/m^2 (N/m) | |
| CTOD | m | |
| Temperature (Master Curve, Charpy correlations) | °C | `-20.0` |
| Charpy energy | J | |
| Pressure | Pa | |

## Immutability and `.with_()`

Problem objects are frozen dataclasses. You cannot change a field in place. Use `.with_(...)`, which returns a
modified copy and runs every validation again:

```python
steel = cf.Material(355e6, 490e6, 207e9, 120e6, yielding="continuous")
tougher = steel.with_(toughness=150e6)          # steel itself is unchanged
loads = cf.StressState(pm=150e6)
heavier = loads.with_(pm=1.2 * 150e6, pb=20e6)
```

`Assessment` has the same method, so `problem.with_(flaw=cf.SurfaceFlaw(4e-3, 20e-3))` is a variant of a whole
problem.

## Vectorisation

Any numeric field can be a scalar, a NumPy array (one value per sample), or a random variable from
`critflaw.prob` ([chapter 4](04_probabilistic.md)). Arrays broadcast against each other. The result has the
common broadcast shape:

```python
depths = np.linspace(1e-3, 10e-3, 50)
problem = cf.Assessment(
    cf.FlatPlate(25e-3, 500e-3),
    cf.SurfaceFlaw(a=depths, two_c=5 * depths),       # 50 flaws
    steel,
    cf.StressState(pm=150e6),
)
r = problem.run(on_invalid="mask")                    # r.lr, r.kr, r.acceptable have shape (50,)
```

`Option1Curve`, `Option2Curve` and the toughness functions are also vectorised. A problem that contains a
random variable cannot be run deterministically. `Assessment.run()` raises a `TypeError` that names the random
fields, and you assess the problem with `ProbabilisticAssessment` instead.

## Components

### `FlatPlate(thickness, width)`

| Field | Meaning | Unit |
|---|---|---|
| `thickness` | section thickness B | m |
| `width` | plate width W, used by the finite-width correction and the reference stress | m |

Both must be finite and positive.

### `Cylinder(r_i, thickness, length, k_route, sref_route="thin")`

Described in [chapter 6](06_cylinders_embedded.md). `k_route` has **no default**: `"pressure_only"` and
`"tabulated"` are different models, and you must choose one.

## Flaws

| Class | Fields (m) | Component |
|---|---|---|
| `SurfaceFlaw(a, two_c)` | depth `a`, surface length `2c` | `FlatPlate` |
| `ThroughFlaw(two_a)` | total length `2a` | `FlatPlate` |
| `EmbeddedFlaw(two_a, two_c, p, near_surface=None)` | height `2a`, length `2c`, ligament `p` to the **nearer** surface | `FlatPlate` |
| `CylinderSurfaceFlaw(a, two_c, orientation, position)` | `orientation` `"axial"` or `"circumferential"`, `position` `"internal"` or `"external"` | `Cylinder` |
| `CylinderThroughFlaw(two_a, orientation)` | circumferential length measured at mid-thickness | `Cylinder` |
| `ExtendedFlaw(a, orientation, position)` | uniform depth `a`: fully circumferential, or extended axial | `Cylinder` |

Convenience constructors:

```python
cf.SurfaceFlaw.from_a_over_c(a=3e-3, a_over_c=0.4)              # two_c = 2a / (a/c)
cf.EmbeddedFlaw.from_ligaments(4e-3, 20e-3, p_1=12e-3, p_2=9e-3)  # p = the smaller ligament
```

Pairing a plate flaw with a `Cylinder`, or a cylinder flaw with a `FlatPlate`, raises a `TypeError` when the
assessment runs. The exception is an `EmbeddedFlaw` with a `Cylinder`, which raises
`NotImplementedError("TODO(verify): embedded flaws in cylinders ...")` first: embedded flaws in cylinders are not
implemented.

## Material

```python
cf.Material(yield_strength, uts, modulus, toughness, yielding, fad_option=1, luders_strain=None, law=None)
```

| Field | Meaning | Unit / values |
|---|---|---|
| `yield_strength` | yield or 0.2 % proof strength | Pa |
| `uts` | tensile strength, or `Ratio(r)` meaning `yield_strength * r` with `r > 1` | Pa or `Ratio` |
| `modulus` | Young's modulus E | Pa |
| `toughness` | `Kmat`, or a `MasterCurve` | Pa·m^0.5 |
| `yielding` | **required**: `"continuous"`, `"discontinuous"` (yield plateau) or `"unknown"` | |
| `fad_option` | `1` (generic Option 1 line) or `2` (material-specific Option 2 line) | |
| `luders_strain` | Lüders plateau strain (Option 1, discontinuous or unknown yielding only) | strain (-) |
| `law` | material law for Option 2: `PowerLaw` or `TabulatedLaw` | |

Rules that are checked when the object is built:

- `uts` must exceed `yield_strength`. When either is random, the ordering must be guaranteed by the
  distributions' supports. Otherwise the error asks you to give the tensile strength as `Ratio(...)`, which
  keeps the ordering by construction.
- `yielding` has no default. A wrong assumption changes the line near `Lr = 1` considerably.
- For discontinuous or unknown yielding without an explicit `luders_strain`, the library estimates the
  Lüders strain from the yield strength. The estimate is refused above a yield-strength limit, which the error
  message states. Supply `luders_strain` there.
- `fad_option=2` needs a `law`, and `yielding` must agree with it: `"discontinuous"` if the law declares a
  plateau, `"continuous"` otherwise. For Option 2 the Lüders strain belongs to the law, not to the material.
  [Chapter 3](03_deterministic.md#option-2-material-specific-line) shows how to build the laws.

### Toughness from the Master Curve

```python
mc = cf.MasterCurve(t0=-60.0, temperature=-10.0, probability=0.05)   # degrees C; probability in [0, 1)
steel_mc = cf.Material(355e6, 490e6, 207e9, mc, yielding="continuous")
```

The engine computes the toughness from the **crack-front length of the flaw being assessed**, capped at twice
the section thickness. That length depends on the flaw type, so the same `MasterCurve` gives different `Kmat` for
different flaws. In a probabilistic assessment, `probability` is usually `prob.Uniform(0, 1)` and `t0` can be
random too.

### Toughness helpers (`critflaw.toughness`)

These functions convert test data to `Kmat`. All of them take and return SI values.

| Function | Gives |
|---|---|
| `k_from_j(j_mat, modulus, poisson=0.3)` | `Kmat` from J |
| `k_from_ctod(ctod, test_yield, test_uts, modulus, poisson=0.3)` | `Kmat` from CTOD, using the tensile properties **of the test**, not the sampled assessment properties |
| `master_curve(probability, temperature, t0, crack_front_length, thickness)` | Master Curve `K` at a cumulative probability |
| `t0_from_charpy(transition_temperature, energy=27)` | T0 from a Charpy transition temperature (27 J or 40 J) |
| `charpy_master_curve(temperature, t0, thickness, probability=0.05, temperature_margin=25.0)` | Charpy-based Master Curve estimate |
| `charpy_lower_bound(charpy_energy, thickness)` | lower-shelf and transition estimate |
| `charpy_upper_limit(charpy_energy, yield_strength)` | cap on Charpy-based estimates (restricted yield-strength range; refused outside it) |
| `charpy_upper_shelf(upper_shelf_energy, modulus, poisson=0.3)` | upper-shelf initiation toughness |
| `subsize_correction(k_small, specimen_thickness, component_thickness)` | thickness adjustment of a cleavage result |
| `k_jc_limit(ligament, test_yield, modulus, poisson=0.3)` | small-scale-yielding limit for that adjustment |
| `subsize_charpy_temperature_shift(specimen_thickness)` | transition-temperature shift for sub-size Charpy specimens |
| `minimum_of_three_equivalent(values)` | `Kmat` from 3 to 15 results |
| `lower_bound_toughness(values, percentile=0.2, confidence=0.9, lognormal=False)` | statistical lower bound, from mean and standard deviation (use `lognormal=True` for cleavage data) |
| `tolerance_factor(count, percentile=0.2, confidence=0.9)` | the one-sided tolerance factor used by the previous function |

```python
from critflaw import toughness
kmat = toughness.k_from_j(j_mat=60e3, modulus=207e9)          # J/m^2 -> Pa m^0.5
kmat = toughness.lower_bound_toughness([95e6, 110e6, 120e6, 101e6, 130e6], lognormal=True)
```

## Stress state

```python
cf.StressState(pm=0.0, pb=0.0, qm=0.0, qb=0.0, residual=None,
               pressure=None, pm_moment=0.0, bending_tensile_surface="outside")
```

| Field | Meaning | Unit |
|---|---|---|
| `pm`, `pb` | primary membrane and bending stress (must be non-negative, that is tensile) | Pa |
| `qm`, `qb` | secondary membrane and bending stress | Pa |
| `residual` | welding residual stress (`Residual`), added to the secondary membrane stress | |
| `pressure` | cylinders only: a `PressureLoad` ([chapter 6](06_cylinders_embedded.md)) | |
| `pm_moment` | cylinders only: membrane stress from a global bending moment, at the flaw position farthest from the bending axis | Pa |
| `bending_tensile_surface` | cylinders: the surface on which `pb` is tensile, `"outside"` (the source's convention) or `"inside"` | |

A random primary stress must have support at or above zero. Truncate an unbounded distribution explicitly,
for example `prob.Truncated(prob.Normal(150e6, 30e6), lo=0.0)`. Nothing is clipped for you.

### Residual stress

```python
cf.Residual.as_welded()                           # uniform residual stress at the appropriate yield strength
cf.Residual.as_welded(relaxation=True)            # with relaxation by primary load
cf.Residual.pwht("parallel")                      # post-weld heat treated, flaw parallel to the weld
cf.Residual.pwht("transverse", furnace_range="550-580C")
cf.Residual(fraction=0.5)                         # a user fraction of the appropriate yield strength
```

| Field | Meaning |
|---|---|
| `fraction` | fraction of the appropriate yield strength, in (0, 1] |
| `appropriate_yield` | yield strength the residual stress refers to (default: the material's) |
| `appropriate_uts` | tensile strength for the relaxation term (only with `relaxation=True`) |
| `relaxation` | relax the as-welded value with the primary load. Defined only for the as-welded condition (`fraction == 1`). |
| `room_temperature_yield` | replaces the yield strength in the unrelaxed term, for assessments below ambient temperature |

The residual stress is added to `qm`. `Result.qm` reports the membrane secondary stress actually used, after
any relaxation. The trace records `residual.as_welded` or `residual.pwht`, and the flag
`residual_stress_relaxation_applied` when relaxation was used.

## Model-uncertainty factors

`Assessment` also takes three multiplicative factors, which default to 1.0, meaning "not accounted for":

| Field | Multiplies |
|---|---|
| `k_model` | the primary and secondary stress intensity factors |
| `sref_model` | the reference stress |
| `fad_model` | the assessment line (applied as an equivalent test on `Kr`; the cut-off is unaffected) |

They are usually random variables in a probabilistic assessment. If a factor is left at a scalar value, the
corresponding source of uncertainty is listed under "Not accounted for" in the trace.

Next: [Deterministic assessment](03_deterministic.md).
