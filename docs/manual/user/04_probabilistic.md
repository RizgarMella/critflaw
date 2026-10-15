# 4. Probabilistic assessment

Any numeric input of a problem can be a random variable. `ProbabilisticAssessment` turns the problem into a
limit state. **The deterministic engine decides each realisation**: a realisation fails if it is not
acceptable at any crack-front position. The solvers estimate the probability of that event and say what the
estimate claims.

Theory: [probabilistic methods](../theory/04_probabilistic.md). Examples: *probabilistic plate assessment*,
*correlated inputs*, *rare events and subset simulation*, *diagnostics and partial factors*,
*inspection-conditioned flaw size*.

```python
import numpy as np
import critflaw as cf
from critflaw import prob
```

## Random variables (`critflaw.prob`)

| Constructor | Parameters | Notes |
|---|---|---|
| `Normal(mean, sd, name=None)` | | unbounded: truncate it before using it for a positive quantity |
| `LogNormal(median, cov, name=None)` | **median** and coefficient of variation | `LogNormal.from_mean_sd(mean, sd)` for mean and standard deviation |
| `Weibull(scale, shape, loc=0.0, name=None)` | | `Weibull.from_mean_sd(mean, sd, loc=0.0)` |
| `Gumbel(mode, scale, name=None)` | distribution of maxima | `Gumbel.from_mean_sd(mean, sd)` |
| `Uniform(lo, hi, name=None)` | | |
| `Exponential(mean, loc=0.0, name=None)` | | |
| `Truncated(dist, lo=-inf, hi=inf, name=None)` | a continuous `dist` restricted to [lo, hi] | numerically safe in both tails |
| `Empirical(values, weights=None, name=None)` | discrete masses | a mass exactly on the assessment line counts as failure |
| `Deterministic(value, name=None)` | a single mass | |

Rules:

- **Identity defines a variable.** One distribution object used in two places is one random variable. Two
  separately constructed objects are two independent variables.
- **Name your variables** (`name="kmat"`). The names label results, diagnostics and implied partial factors,
  and `implied_partial_factors` refuses duplicate names.
- **Supports are checked when the object is built.** A positive quantity (dimensions, toughness, strengths,
  primary stress) must not have support below zero. The error tells you to write the truncation explicitly.
  Nothing is clipped, dropped or resampled.
- **Tensile and yield strength.** When either is random, give the tensile strength as
  `cf.Ratio(dist)` with the support of `dist` above 1, so the two cannot cross.

## A probabilistic problem

```python
sy = prob.LogNormal(380e6, 0.07, name="sy")
ratio = prob.Truncated(prob.LogNormal(1.35, 0.04), lo=1.02, name="uts/sy")
kmat = prob.Weibull.from_mean_sd(110e6, 25e6, name="kmat")
a = prob.Truncated(prob.LogNormal(4e-3, 0.4), lo=1e-4, hi=20e-3, name="a")
pm = prob.Truncated(prob.Gumbel.from_mean_sd(170e6, 20e6), lo=0.0, name="pm")
k_model = prob.LogNormal(1.0, 0.1, name="k_model")

problem = cf.Assessment(
    cf.FlatPlate(25e-3, 500e-3),
    cf.SurfaceFlaw(a=a, two_c=60e-3),
    cf.Material(sy, cf.Ratio(ratio), 207e9, kmat, yielding="continuous"),
    cf.StressState(pm=pm, residual=cf.Residual.as_welded()),
    k_model=k_model,
)

pa = cf.ProbabilisticAssessment(problem).correlate(sy, kmat, -0.5)
res = pa.solve(seed=1, n_explore=200_000)
print(res.summary())
```

`ProbabilisticAssessment(problem)` raises `ValueError` if the problem has no random variable. Use
`Assessment.run()` for that case.

### Dependence

`correlate(x, y, rho, kind="pearson")` declares dependence through a Gaussian copula, and returns the
assessment so calls can be chained.

- `kind="pearson"`: `rho` is the correlation of the physical variables. The copula coefficient is solved for
  it. A value that these marginals cannot attain raises `CorrelationError` and states the attainable range,
  which `prob.attainable_range(x, y)` also gives.
- `kind="latent"`: `rho` is the copula coefficient itself. Discrete marginals can only be correlated this way.
- Positive definiteness is a property of the whole matrix, so it is checked when the solver starts. A set of
  correlations that cannot hold jointly raises `CorrelationError` then.

### `solve(seed=0, n_explore=1_000_000, replicates=8, samples_per_level=5000)`

1. Plain Monte Carlo with `n_explore` samples gives the **guaranteed** statement: a two-sided 99 %
   Clopper-Pearson interval, valid whatever the limit state looks like.
2. If Monte Carlo sees fewer than 10 failures, `replicates` independent subset-simulation runs follow, giving
   a **conditional** rare-event estimate.

Samples are evaluated in batches of 200,000. `n_explore` sets the cost. The default of one million is
appropriate for a final run. For exploration, `100_000` to `200_000` is usually enough.

## Reading the result

`ProbabilisticResult`:

| Field | Meaning |
|---|---|
| `reliability` | the headline `ReliabilityResult`, with realisations outside validity **counted as failed** |
| `reliability_valid_only` | the same with them counted as safe (`None` if none occurred). The two bracket the treatment of invalid realisations. |
| `p_out_of_validity`, `p_out_of_validity_interval` | share of the guaranteed channel's samples outside validity, with its 99 % interval |
| `modes` | `ModeSplit(fracture, collapse, out_of_validity, samples)`: shares of the failure samples by exit (line or step; cut-off or end of data; invalid) |
| `variables` | the variable names |
| `per` | always `"flaw"`: the probability is **per flaw, given that the flaw exists** |
| `summary()` | text summary of all of the above |

`ReliabilityResult`:

| Field | Meaning |
|---|---|
| `status` | see the table below |
| `pf` | the headline value |
| `guaranteed` | the Clopper-Pearson `Interval` from Monte Carlo (`lower`, `upper`, `confidence`, `kind`) |
| `monte_carlo` | `MonteCarloResult` (`failures`, `samples`, `pf`, `interval`, `invalid`) |
| `subset` | `SubsetResult` or `None` (`status`, `pf`, `interval`, `replicate_estimates`, `runs`) |
| `conditional_estimate` | the subset estimate when it is usable, else `None` |

| `status` | Meaning | What to use |
|---|---|---|
| `resolved` | Monte Carlo saw at least 10 failures | `pf` and `guaranteed` |
| `conditional` | rare event. The subset estimate lies inside the guaranteed interval. | `pf` is valid **only if no failure region was missed**, which cannot be verified from the samples. Values up to `guaranteed.upper` cannot be excluded. |
| `contradicted` | the subset estimate lies outside the guaranteed interval and is rejected | `guaranteed` only |
| `replicates inconsistent` | the subset replicates disagree too much to give an estimate | `guaranteed` only |
| `plateau` | the driving variable carried no information for subset simulation | `guaranteed` only |
| `level cap reached; unresolved` | subset simulation did not reach the failure domain | `guaranteed` only |

With zero Monte Carlo failures, the guaranteed upper bound is set by the sample size. Report that bound, not
zero.

## Lower-level solvers

The building blocks are public. They work on any limit state, a function `u (m, n) -> Evaluation(y, fail,
invalid)` in independent standard-normal space, where `y` is negative for failed samples.

```python
g = pa.limit_state("failed")          # or "safe": how out-of-validity realisations are counted
mc = prob.solvers.monte_carlo(g, dimension=pa.vector.size, samples=200_000, seed=3)
sub = prob.solvers.subset_replicates(g, pa.vector.size, seed=4, replicates=8, samples_per_level=2000)
rel = prob.solvers.solve(g, pa.vector.size, seed=5, n_explore=100_000)
```

- `subset_simulation(..., samples_per_level=5000, p0=0.1, max_levels=20)`: `samples_per_level * p0` must be an
  integer of at least 2 that divides `samples_per_level`.
- `prob.importance.importance_sampling(limit_state, dimension, centres, samples, seed, defensive_weight=0.2,
  confidence=0.99)` is a defensive-mixture importance sampler, offered as an optional refinement. It returns
  `ImportanceResult` with a **guaranteed** (empirical Bernstein) interval that holds whatever the centres
  miss, and an asymptotic `conditional_interval`. The conditional interval is valid only if the centres cover
  every failure region, and it is given only when the effective number of failures is at least 100.
- `prob.form(g, dimension, starts=5, seed=0)` and `prob.sorm_breitung(g, form_result)` are generic FORM and
  SORM for **user-supplied smooth** limit states `g(u) -> (m,)` values. They are **not applied to FAD
  problems**, whose failure set is a union of non-smooth guarded events. Both raise `FormNotApplicable` when
  their assumptions fail, for example when the origin is in the failure region or the curvature is too strong.
  `FormResult.pf` is a first-order approximation without an error bound.

### Intervals

```python
prob.clopper_pearson(failures=3, samples=100_000, confidence=0.99)   # exact two-sided binomial interval
prob.empirical_bernstein(values_mean, values_variance, samples, bound, confidence=0.99)
```

Each returns an `Interval(lower, upper, confidence, kind)`. `kind` states how it was obtained and what it
assumes.

## Diagnostics

None of these is a probability estimate.

```python
points = pa.most_likely_failure_points(res)     # list[FailurePoint], sorted by share of failure samples
p0 = min(points, key=lambda p: p.distance)       # the governing point is the smallest distance, not always the first
print(dict(zip(pa.names, p0.physical)), p0.distance, p0.share, p0.label)

for assoc in pa.association(res):               # association with failure, NOT causation
    print(assoc.name, assoc.mean_shift, assoc.spread, assoc.divergence)

rows = pa.family_sensitivity(seed=0, n_explore=200_000)
```

- `FailurePoint`: `u` (standard-normal coordinates), `distance`, `share`, `physical` (values of the named
  variables), `dimension`, and `label`. The label says how far the point can be trusted. With more than about
  ten variables its direction is only indicative.
- `association(res)` ranks variables by how the failure samples shift their distribution. A variable that is
  merely correlated with an important one ranks high too. Nothing is returned with fewer than 30 failure
  samples.
- `family_sensitivity(...)` re-solves with each normal, lognormal or Weibull variable replaced by the other two
  families, matched in mean, standard deviation and support. It returns `FamilyVariant(variable, family,
  status, pf, guaranteed)`. Variables of other kinds are listed as `"no alternative family defined"`. This
  reruns the solver many times, so it is expensive.

## Partial factors

**Implied partial factors** from a failure point, against reference values that you supply:

```python
factors = pa.implied_partial_factors(p0, {"kmat": (75e6, "resistance"), "pm": (170e6, "load")})
```

The kind `"load"` gives design / reference, and `"resistance"` gives reference / design. These factors are as
approximate as the point.

**Published partial factors** are a discrete lookup of the source standard's table:

```python
from critflaw.prob.partial_factors import published_partial_factors, NotTabulated
pf_set = published_partial_factors(target_pf=7e-5, cov_stress=0.2, cov_flaw_size=0.1, cov_toughness=0.2)
pf_set.stress, pf_set.flaw_size, pf_set.toughness, pf_set.yield_strength, pf_set.beta
```

There is no interpolation. A target or coefficient of variation that is not tabulated, or an entry marked "not
permitted", raises `NotTabulated`. The factors apply to specific reference values: stress and flaw height at
the mean, toughness at the mean minus one standard deviation, and yield strength at the mean minus two
standard deviations. They were derived for a specific model, described in the theory manual. The table is
`standard-derived` and is not cleared for release (see the [index](index.md#release-and-legal-status-of-the-built-in-coefficients)).

## Flaw size after an inspection

`critflaw.prob.inspection` conditions a prior flaw-size distribution on an inspection outcome. The result is a
distribution that you use directly as the flaw size.

```python
from critflaw.prob.inspection import LogLogisticPoD, undetected, detected, per_component_upper_bound

pod = LogLogisticPoD(a50=3.0e-3, beta=4.0)
depth, p_nd = undetected(prob.Exponential(mean=2.0e-3), pod, name="a")   # flaw missed by the inspection
problem_i = problem.with_(flaw=cf.SurfaceFlaw(a=depth, two_c=80e-3))
res_i = cf.ProbabilisticAssessment(problem_i).solve(seed=0, n_explore=200_000)
bound = per_component_upper_bound(3.0, p_nd, res_i.reliability.guaranteed.upper)
```

| Function / class | Meaning |
|---|---|
| `undetected(prior, pod, name=None)` | `(distribution, P(not detected))`: density proportional to prior × (1 − PoD) |
| `detected(a_measured, sizing_error, prior, pod, name=None)` | `(posterior, normalising constant)`. `sizing_error` is the distribution of measured minus true size. |
| `LogLogisticPoD(a50, beta)`, `StepPoD(threshold, level=1.0)`, `ConstantPoD(level)` | PoD models. Any callable `pod(a)` with values in [0, 1] works, and it can also declare `non_detection(a)` and `discontinuities`. |
| `per_component_upper_bound(flaws_per_component, p_not_detected, pf_per_flaw)` | an **upper bound** on the per-component probability for a Poisson number of flaws (exact only if the flaws share no random variable) |

An observation so far into a tail of the prior that it cannot be integrated in double precision is refused
with an error. The library does not return a cut-off distribution in that case.

Next: [Fatigue and time-dependent reliability](05_fatigue.md).
