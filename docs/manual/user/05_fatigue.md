# 5. Fatigue crack growth and time-dependent reliability

The fatigue layer grows the flaw of an `Assessment` under a cyclic stress spectrum. At every step it checks the
grown flaw against the largest fracture-check load of that step. The result is the failure probability against
time, with optional inspections and repairs.

Theory: [fatigue and time-dependent reliability](../theory/05_fatigue.md). Examples: *deterministic crack
growth*, *time-dependent failure probability*, *inspection and repair*, *updating from an inspection result*.

```python
import numpy as np
import critflaw as cf
from critflaw import prob
from critflaw.fatigue import (ExtremeLoad, GrowthLaw, Inspection, Spectrum, TimeDependentAssessment,
                              recommended_law, simplified_law, steel_in_air, steel_in_seawater)
```

`ExtremeLoad`, `Inspection`, `Spectrum` and `TimeDependentAssessment` are also available as `cf.<name>`.

## Growth laws

`GrowthLaw` works in SI units: the rate is in m/cycle and the range ΔK is in Pa·m^0.5.

```python
GrowthLaw(a_coefficient, m, threshold=0.0, a_coefficient_b=None, m_b=None,
          stage_correlation=None, shallow_cap=None, name="growth law", validity="")
```

| Field | Meaning |
|---|---|
| `a_coefficient`, `m` | `da/dN = A ΔK^m` above the threshold. `A` may be a distribution. |
| `threshold` | ΔK below which there is no growth (Pa·m^0.5). May be a distribution. |
| `a_coefficient_b`, `m_b` | second stage. The rate is the minimum of the two stage lines, and `m > m_b` is required. |
| `stage_correlation` | correlation of log A between the stages. **Required** (no default) when a two-stage law has random coefficients. |
| `shallow_cap` | `(depth, threshold)`: for surface flaws shallower than `depth`, the smaller of the two thresholds is used |
| `name`, `validity` | text carried into the result's assumptions |

The built-in catalogue (the coefficient values are `standard-derived`, see the
[index](index.md#release-and-legal-status-of-the-built-in-coefficients)):

```python
law = steel_in_air()                                         # two-stage, R >= 0.5, "upper" curve (fixed)
law = steel_in_air(stress_ratio="low", curve="mean", stage_correlation=0.5)   # mean law, log A random
law = steel_in_seawater(protection="cp850")                  # "free", "cp850" or "cp1100"
law = recommended_law("air", stress_ratio="high", curve="mean-fixed", welded=True)
law = simplified_law("air")                                   # single-stage upper bound (m = 3)
```

`curve="upper"` uses fixed upper-bound coefficients. `"mean-fixed"` uses the mean law with fixed coefficients.
`"mean"` makes log A random with the published scatter, and requires `stage_correlation`. Each law carries its
statement of applicability in `law.validity`, and the result lists it among its assumptions.

The effective-range form of growth near the threshold is not implemented. Requesting it raises
`NotImplementedError("TODO(verify): ...")`.

## Spectrum

A spectrum is a set of blocks of stress range `(membrane, bending, cycles)` in Pa. The membrane and bending
ranges of a block are in phase.

```python
sp = Spectrum.repeating([(80e6, 10e6, 0.7), (40e6, 0.0, 0.3)], cycles_per_year=3e4)
sp_o = Spectrum.ordered([(120e6, 0.0, 6e4), (40e6, 0.0, 6e4)], cycles_per_year=6e4, order="as_given")
```

| Field / argument | Meaning |
|---|---|
| `repeating(blocks, cycles_per_year, **factors)` | the blocks are one pass of a pattern that repeats. Here `cycles` are relative weights within the pass, and the growth rate is the expectation over the pass. |
| `ordered(blocks, cycles_per_year, order, **factors)` | the blocks are applied once, in order: a lifetime histogram. Their cycles must sum to the last requested time × `cycles_per_year`. `order` (**required**) is `"as_given"`, `"largest_first"` or `"smallest_first"`. |
| `load_scale` | multiplies every range. May be a distribution, in which case `TimeDependentAssessment` requires `shared_scale`. |
| `ktm`, `ktb`, `km` | stress-concentration and misalignment factors applied to the ranges |
| `life_factor` | deterministic runs only (≥ 1; 2 halves the calculated life). Refused in probabilistic runs. |

If the growth within one pass of a repeating spectrum is too large for the expectation over the pass to be
accurate, the integrator refuses with a `ValueError` and suggests `Spectrum.ordered` or shorter passes.

## The extreme (fracture-check) load

```python
ext = ExtremeLoad(prob.Truncated(prob.Gumbel.from_mean_sd(150e6, 15e6), lo=0.0), reference_period_years=1.0,
                  component="pm")
```

`distribution` is the distribution of the maximum of the stress component `component` (`"pm"` or `"pb"`) over
`reference_period_years`. The library maximises over a load-process step of length δ under the assumption of
independent sub-period maxima. The distribution must be non-negative, so truncate it explicitly. In each step,
the problem's stress component is replaced by that step's maximum.

## Time-dependent assessment

```python
problem = cf.Assessment(cf.FlatPlate(25e-3, 0.5), cf.SurfaceFlaw(3e-3, 30e-3),
                        cf.Material(355e6, 490e6, 207e9, 90e6, yielding="continuous"),
                        cf.StressState(pm=100e6))
run = TimeDependentAssessment(problem, steel_in_air(), Spectrum.repeating([(80e6, 0.0, 1.0)], 3e4), ext,
                              times=range(1, 6), leak_is_failure=False)
res = run.solve(seed=0, samples=20_000)
print(res.summary())
```

### Constructor

| Argument | Meaning |
|---|---|
| `problem` | an `Assessment` whose flaw is the initial flaw: `SurfaceFlaw`, `EmbeddedFlaw`, or a cylinder flaw. Its fields may be random. |
| `law`, `spectrum`, `extreme` | as above |
| `times` | reporting times in years. They must be positive, increasing, and multiples of the step δ. |
| `leak_is_failure` | **required**, `True` or `False`: whether breaking through the wall counts as failure |
| `inspections` | a sequence of `Inspection` (see below), at step boundaries |
| `delta` | the load-process step in years (default: the extreme load's reference period) |
| `shared_scale` | **required** when `spectrum.load_scale` is random: one scale for the life (`True`) or a separate draw per step |
| `eta`, `tol` | integrator controls |

`correlate(x, y, rho, kind="pearson")` declares dependence between the time-invariant random variables, as in
[chapter 4](04_probabilistic.md#dependence). The total dimension of the problem is
`run.dimension = n_invariant + n_steps + inspection draws`.

Refusals: a plate `ThroughFlaw` as the initial flaw, a random inner radius of a cylinder, an embedded flaw in a
cylinder, and an embedded flaw under any bending range or bending extreme load all raise
`NotImplementedError("TODO(verify): ...")`.

### `solve(seed=0, samples=20_000, subset=False, refine=False, replicates=8, samples_per_level=2000, stage_sensitivity=True, curve_times=())`

- Each step is evaluated at interior checkpoints and on both sides of every change of flaw type
  (re-characterisation, breakthrough). The result is **bracketed** between two pairings of the step's extreme
  load with the flaw configurations: the *upper* pairing (the headline) and the *lower* pairing.
- `refine=True` halves the step, up to four times, where the bracket is wide. The coarse maximum is kept
  exactly by a coupled draw.
- `subset=True` adds subset simulation in time for rare lifetime failures. It is not offered when
  `leak_is_failure=True`. `curve_times` asks for separate runs at up to five times.
- `stage_sensitivity` reruns a random two-stage law at stage correlation 0 and 1.

### Reading `TimeDependentResult`

| Field | Meaning |
|---|---|
| `cumulative_upper` | headline cumulative Pf(0, t): upper pairing, out-of-validity counted as failed. A `Curve(times, value, lower, upper, kind)` with pointwise 99 % Clopper-Pearson intervals. |
| `cumulative_lower` | the same under the lower pairing. With the upper curve it brackets Pf under assumption A (severity monotone between checkpoints). |
| `cumulative_valid_only` | upper pairing with out-of-validity counted as safe |
| `interval_probability` | probability of failing in each reporting interval (approximation) |
| `hazard_upper`, `hazard_lower` | hazard per interval within each pairing (exact for that discretised model, a sensitivity rather than a bracket) |
| `hazard_conservative_upper`, `hazard_conservative_lower` | the conservative pair, which brackets the hazard under assumption A |
| `refine` | per reporting time: the bracket is wide, so refine the step |
| `modes` | first failure mode: `fracture`, `collapse`, `leak`, `leak_assumed`, `out_of_validity`, and leak with structural failure in the same step |
| `events`, `events_any` | shares of samples with a leak, surface breaking, end of validity or repair, before the first failure and at any time |
| `assumption_violations`, `assumptions` | the checks and every assumption of the run as text, including the growth law's validity statement |
| `inspections` | per inspection: detection and repair probabilities (with their denominators), Pf before and after, and the next-step hazard |
| `conditioning`, `weighted` | present when an inspection outcome was observed (below) |
| `invalid_start` | share of initial flaws already outside the solution's validity |
| `summary()` | all of the above as text |

## Deterministic growth

For a problem with no random variable:

```python
det = TimeDependentAssessment(problem, steel_in_air(), sp_o, ExtremeLoad(prob.Uniform(200e6, 200.001e6), 1.0),
                              times=[1, 2], leak_is_failure=False)
hist = det.grow(load=200e6)                    # GrowthHistory
hist.times, hist.a, hist.c, hist.life, hist.cycles_applied
orders = det.compare_orders(load=200e6)        # ordered spectra: life under each of the three declared orders
```

`grow(load=None)` returns the sizes, the flaw kind and the driving variable at every checkpoint, the leak and
end-of-validity times, and `first_failure` (also `life`). Failure means the first failing or out-of-validity
checkpoint, the end of the growth solution's validity, or a leak when leaks count. The `ExtremeLoad` is
required by the constructor even here. `grow` and `compare_orders` do not simulate inspections.

## Inspections

```python
from critflaw.prob.inspection import LogLogisticPoD, StepPoD
insp = Inspection(time=3.0, pod=LogLogisticPoD(a50=2e-3, beta=3.0), action="repair")
run_i = TimeDependentAssessment(problem, steel_in_air(), sp, ext, times=range(1, 11), leak_is_failure=False,
                                inspections=[insp])
```

| Field | Meaning |
|---|---|
| `time` | inspection time (years), at a step boundary |
| `pod` | PoD of the flaw **depth** (`a`) |
| `pod_through` | PoD of a through flaw's length (default: always detected) |
| `pod_embedded` | PoD of an embedded flaw's height `2a`. Required when embedded flaws are present. |
| `sizing_error`, `accept_below` | measured minus true size, and the measured depth below which a detected flaw is accepted, not repaired |
| `action` | `"repair"` (remove the flaw, or replace it with `repair_flaw=(a, two_c)`) or `"none"` (record only) |
| `observed` | condition on this component's own outcome: `"not_detected"`, `"detected"` or `("sized", a_measured)` (the last needs `sizing_error`) |

With `observed` set, `res.conditioning` holds the probabilities after the inspection, conditional on survival
to it and on the observed outcome. "not detected" and "detected" are handled by rejection: you get
`pf_after_upper`/`_lower`, Clopper-Pearson `interval_*`, and the `conservative_pair`. A sized outcome is
handled by weighting (`res.weighted` is `True`): `pf_after_*` is withheld (`None`) when the effective sample
size is too small, with `samples_needed_*` stating how many samples would be needed. Sizing of a through flaw
is not defined and raises `NotImplementedError`.

Next: [Cylinders and embedded flaws](06_cylinders_embedded.md).
