# 6. Cylinders and embedded flaws

Theory: [solutions](../theory/02_solutions.md). Examples: *axial surface flaw in a pressurised pipe*,
*circumferential flaws under bending*, *embedded flaw in a plate*.

```python
import critflaw as cf
```

## Cylinders

```python
cf.Cylinder(r_i, thickness, length, k_route, sref_route="thin")
```

| Field | Meaning | Unit / values |
|---|---|---|
| `r_i` | inner radius | m |
| `thickness` | wall thickness B | m |
| `length` | the width W used by the finite-width rule | m |
| `k_route` | **required**: `"pressure_only"` or `"tabulated"` | |
| `sref_route` | `"thin"`, `"thick_pressure"` or `"thick"` | |

The two K routes are **different models** of the same shell, so there is no default:

- `"pressure_only"` uses plate solutions with a bulging factor. It is intended for pressure loading and
  **refuses mechanical primary stresses** (`pm`, `pb`, `pm_moment`). An affected realisation is out of
  validity: `OutOfValidityError`, or masked with `on_invalid="mask"`. The flag
  `pressure_only_route_refuses_mechanical_primary_stress` is set.
- `"tabulated"` interpolates the tabulated cylinder factors. Queries outside the tables are out of validity,
  never extrapolated. A few configurations use the plate solution even under this route, and the trace option
  `tabulated_route_uses_plate_solution` says so.

The two routes give different `Kr` for the same case. This is recorded in the trace (`k_route`), not hidden.

Reference-stress routes:

- `"thin"`: thin-wall reference stresses.
- `"thick_pressure"`: thick-wall limit pressure for **axial surface flaws under a `PressureLoad`**. Here
  `Lr = p / p_L`. Mechanical primary stress is refused, and another flaw type raises `ValueError`.
- `"thick"`: thick-wall limit loads, for **external** axial flaws under membrane and bending stress, and for
  circumferential flaws under tension and global bending. Through-wall bending is refused for the
  circumferential case.

When the reference stress comes from a limit load, the trace option `sref_from_limit_load` states how.

### Pressure

```python
press = cf.PressureLoad(pressure=5e6, hoop_per_pressure=5.5, axial_per_pressure=2.75)
loads = cf.StressState(pressure=press)
```

| Field | Meaning |
|---|---|
| `pressure` | internal pressure (Pa) |
| `hoop_per_pressure` | the primary membrane stress per unit pressure for **axial** flaws (your own conversion, for example r/B or a thick-wall expression) |
| `axial_per_pressure` | the same for **circumferential** flaws (default 0) |

The source gives no pressure-to-stress conversion, so you supply it. The same `pressure` feeds the thick-wall
limit pressure, so K and `Lr` always describe one loading. The K solutions exclude crack-face pressure, and the
trace option `crack_face_pressure` records this.

### Other cylinder loads

- `pm`, `pb`: membrane and through-wall bending stress. `bending_tensile_surface` (`"outside"`, the default,
  or `"inside"`) states which surface `pb` is tensile on.
- `pm_moment`: the membrane stress from a global bending moment, at the flaw position farthest from the
  bending axis. Circumferential flaws only.

### Flaws and positions

```python
cyl = cf.Cylinder(r_i=0.1, thickness=0.02, length=2.0, k_route="tabulated")
axial = cf.Assessment(cyl, cf.CylinderSurfaceFlaw(5e-3, 40e-3, "axial", "internal"),
                      cf.Material(355e6, 490e6, 207e9, 90e6, yielding="continuous"),
                      cf.StressState(pressure=press)).run()

through = cf.Assessment(cyl, cf.CylinderThroughFlaw(0.05, "circumferential"),
                        cf.Material(355e6, 490e6, 207e9, 90e6, yielding="continuous"),
                        cf.StressState(pm=30e6, pb=5e6, pm_moment=20e6, pressure=press)).run()
print(through.positions)        # ("outside", "inside")

extended = cf.Assessment(cyl, cf.ExtendedFlaw(5e-3, "circumferential", "external"),
                         cf.Material(355e6, 490e6, 207e9, 90e6, yielding="continuous"),
                         cf.StressState(pm=50e6, pb=20e6)).run()
```

`Result.positions` names the crack-front positions each route evaluates, for example `("outside", "inside")`
for a circumferential through-wall flaw, or `("deepest",)` for an internal circumferential extended flaw under
the tabulated route. The governing position is chosen exactly as
for plates.

### Suspect table cells

Some cells of the cylinder tables are flagged as **suspect**. When any realisation was interpolated through
such a cell, the flag `tabulated_factor_interpolated_through_a_suspect_cell_samples_present` is set. The
knowledge-base status of that tabulated factor is then reported as `unverified` in that result's trace, and in
`trace.dependencies()`. Treat such a result with corresponding caution, or move the query away from the
flagged region.

### Cylinders in the other layers

- **Probabilistic.** Every cylinder field can be random. Realisations that a route refuses are counted as out
  of validity, never dropped.
- **Fatigue.** `TimeDependentAssessment` grows cylinder surface, through and extended flaws. A random inner
  radius is refused in fatigue growth.
- **Embedded flaws in cylinders** are not implemented (`NotImplementedError("TODO(verify): ...")`).

## Embedded flaws in plates

```python
flaw = cf.EmbeddedFlaw(two_a=4e-3, two_c=20e-3, p=5e-3)                  # p: ligament to the NEARER surface
flaw = cf.EmbeddedFlaw.from_ligaments(4e-3, 20e-3, p_1=12e-3, p_2=9e-3)   # p = min(p_1, p_2)
r = cf.Assessment(cf.FlatPlate(25e-3, 0.5), flaw,
                  cf.Material(355e6, 490e6, 207e9, 90e6, yielding="continuous"),
                  cf.StressState(pm=150e6)).run()
print(r.positions)              # ("near_end", "major_axis")
```

- `p` is the ligament to the **nearer** surface. Declaring the far ligament as `p` makes the geometry invalid
  (`OutOfValidityError`). It is never mirrored silently. Use `from_ligaments` when you know both ligaments.
  The geometric condition (flaw height plus twice the smaller ligament within the thickness) is checked per
  realisation, so a random ligament that crosses it gives out-of-validity mass, which is reported.
- **Bending is refused.** The source gives the bending magnification only at the minor-axis ends, so any
  non-zero `pb` or `qb` raises `NotImplementedError("TODO(verify): M_b at the major-axis ends ...")` instead of
  assuming zero.
- `near_surface` (`"tension"`, `"compression"` or `None`) names the nearer surface by the sign of the declared
  bending stress there. It is reserved for when bending is supported.
- Every embedded result carries the flag `embedded_recharacterisation_relaxation_not_applied`.
- An embedded flaw grows in `TimeDependentAssessment` (no bending range allowed). Inspections of embedded
  flaws need `pod_embedded`.

Next: [The FE back end](07_fe_backend.md).
