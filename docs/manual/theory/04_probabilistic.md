# 4. The probabilistic layer

This chapter describes how a problem with random inputs is turned into a failure probability: random variables and
their dependence, the authoritative failure event, the solvers and the statements each can make, invalid
realisations, diagnostics, distribution-family sensitivity, partial safety factors and inspection-conditioned flaw
sizes. Time-dependent reliability under fatigue is in [05_fatigue.md](05_fatigue.md); failure probability with the
Option 3 emulator is in [07_pf_emulator.md](07_pf_emulator.md). Implementation: `critflaw.prob` (`distributions`,
`dependence`, `solvers`, `importance`, `form`, `intervals`, `diagnostics`, `families`, `partial_factors`,
`inspection`) and `critflaw.probabilistic`.

---

## 4.0 What can and cannot be claimed about a small failure probability

FORM on the FAD, importance sampling and subset simulation can each report a failure probability that is orders of
magnitude too low with a tight interval, when a failure region is missed or the driving variable is uninformative.
Counter-examples exist for each. No rare-event method can verify from its own samples
that it found every failure region. Every probabilistic result therefore separates two statements:

- **Guaranteed:** a 99 % two-sided Clopper–Pearson interval from plain Monte Carlo samples of the nominal
  distribution. It assumes nothing about the limit state. With $n$ samples and no failures its upper bound is about
  $5.3/n$.
- **Conditional:** a rare-event estimate (subset simulation, optionally refined by importance sampling) with an
  approximate interval, valid only if the driving variable led the sampler to every failure region that matters.

and carries a status (§4.4) that says which one is the headline and, in words, when values between the two cannot be
excluded.

## 4.1 Random variables

Any numeric field of a problem may hold a `Distribution`. A distribution object is one random variable: the same
object used in two places of a problem is the same variable (identity, not equality). Families:

| Class | Parameterisation |
|---|---|
| `Normal(mean, sd)` | |
| `LogNormal` | median and coefficient of variation, or `from_mean_sd` |
| `Weibull(scale, shape, loc=0)` | or `from_mean_sd` |
| `Gumbel` | (used for extreme loads) |
| `Uniform`, `Exponential` | |
| `Truncated(base, lo, hi)` | explicit truncation of any continuous family |
| `Empirical(values, weights)`, `Deterministic(value)` | discrete masses |

**Positivity by declaration.** A quantity that must be positive may not have a distribution with support at or below
zero; the user writes the truncation explicitly (`Truncated(Normal(...), lo=0)`). Reproducing published assumptions
that use normal distributions for positive quantities therefore uses truncated normals, and says so. Primary stresses
must be non-negative. The tensile strength may be given as `Ratio(r)` of the yield strength with $r > 1$ by
construction.

**Realisation.** A realisation substitutes sampled arrays for the distributions and **re-runs every validation** of
the problem objects on the realised values. A realisation that violates a physical constraint raises with advice to
constrain the distribution; realisations are never dropped or clipped.

**Transform to standard normal space.** Each marginal is mapped as $x = F^{-1}(\Phi(z))$, evaluated through the
survival function in the upper tail so that $\Phi(z)$ rounding to 1 does not give an infinite value; discrete
variables use the mid-distribution score $\Phi^{-1}(F(x^-) + \tfrac12 P(X = x))$ where a score is needed.

## 4.2 Dependence: the Gaussian copula (Nataf transformation)

Dependence is a Gaussian copula: with $u$ independent standard normal,

$$ z = L\,u,\quad L L^T = R_0,\qquad x_i = F_i^{-1}(\Phi(z_i)), $$

where $R_0$ is the latent (copula) correlation matrix. `correlate(x, y, rho, kind="pearson")` declares the Pearson
correlation of the **physical** variables; the latent coefficient $\rho_0$ is solved numerically from

$$ \rho = \frac{E\left[(x(z_1) - \mu_x)(y(z_2) - \mu_y)\right]}{\sigma_x\sigma_y},\qquad \operatorname{corr}(z_1, z_2) = \rho_0, $$

by Gauss–Hermite quadrature refined (64, 128, 256 nodes) until stable to $10^{-8}$ **and** until the rule reproduces
each marginal's mean and variance to $10^{-6}$ (agreement between refinements alone can miss heavy tails). A
lognormal pair uses the closed form. Marginals whose tails the quadrature cannot resolve raise ("cannot evaluate"),
never returning an inaccurate value. `kind="latent"` sets $\rho_0$ directly (the only option for discrete marginals).

The attainable Pearson range is $[\rho(\rho_0 = -1),\ \rho(\rho_0 = +1)]$; a target outside it is an error stating
the range (for example a lognormal of CoV 0.1 with one of CoV 2 cannot exceed 0.675). Zero-variance variables cannot
be correlated, a correlation needing $|\rho_0| = 1$ is refused, and the full latent matrix must be positive definite
(checked by Cholesky on first use, so that a valid matrix may pass through indefinite intermediate states while it
is declared). Achieved correlations are tested by sampling.

**Under correlation**, the u-space direction cosines belong to the uncorrelated components, depend on variable
ordering and are not per-variable quantities; importance is reported by the association measure of §4.7.

## 4.3 The failure event and the driving variable

A realisation **fails** iff it is not acceptable under the direct test at any crack-front position
([01_fad.md](01_fad.md) §1.6); equality with the line or the cut-off is failure. This is the one authoritative
indicator; every solver, every reported split and the time-dependent first-passage use it.

Samplers that need a continuous variable use the **driving variable** $y = F - 1$ (radial margin minus one). Its sign
is forced to agree with the indicator where $|y| < 10^{-9}$ (to $\mp10^{-12}$); a larger disagreement raises an
internal error, so the clamp cannot hide a defect. $F = +\infty$ (zero load) only orders samples. Out-of-validity
realisations get $y = -\infty$ or $+\infty$ according to the bound being computed (§4.6).

## 4.4 Solvers and statuses

`ProbabilisticAssessment(problem).solve(seed, n_explore=10⁶, replicates=8, samples_per_level=5000)` runs:

1. **Monte Carlo (the guaranteed channel)** with `n_explore` samples. Estimator $k/N$; 99 % two-sided
   Clopper–Pearson interval for every $k$: the interior from beta quantiles at $\alpha/2$; for $k = 0$,
   $[0,\ 1 - 0.005^{1/N}]$; for $k = N$, $[0.005^{1/N},\ 1]$. (A one-sided bound at zero failures mixed with
   two-sided intervals elsewhere has coverage 98.7 % near $Np = 4.6$; it is not used.) Flagged when $k < 10$.
2. If Monte Carlo saw at least 10 failures, the status is **`resolved`** and its estimate is the headline.
3. Otherwise **subset simulation** with replicates (§4.4.1). The status is
   - **`contradicted`** if the replicate mean lies outside the guaranteed interval: the guaranteed interval is the
     headline and nothing derived from the rare-event estimate is reported;
   - **`conditional`** otherwise: the headline is the rare-event estimate **together with** the guaranteed upper
     bound, with the statement that values up to it cannot be excluded;
   - **`plateau`**, **`level cap reached; unresolved`** or **`replicates inconsistent`** when subset simulation gives
     no estimate; only the guaranteed interval stands.

Every result is labelled **per flaw** (conditional on the flaw existing). The failure-mode split (fracture through the
line or a step; collapse through the cut-off or the end of tabulated data; out of validity) uses the failure samples
of the channel that gave the headline.

### 4.4.1 Subset simulation

Levels with $p_0 = 0.1$ and $N = 5000$ samples per level by default:

- the level threshold is the order statistic of $y$ at rank $p_0 N$, never interpolated; the level probability is
  the **counted** fraction of samples with $y \le$ threshold (so atoms in $y$ are handled); seeds are all samples at or
  below the threshold, and chain lengths are allocated so that the next level has exactly $N$ samples;
- conditional sampling is adaptive (component-wise proposal spread scaled toward an acceptance rate of 0.44);
- **plateau detection**: if the counted fraction exceeds 0.5, more than half of the population shares the threshold
  value; the driving variable carries no information and the run stops with status `plateau`;
- stopping: when at least $p_0 N$ samples of a level fail, $P_f$ is the product of the level fractions and the final
  counted failure fraction; after 20 levels the status is `level cap reached; unresolved` with no bound.

The estimator has a small positive bias, measured at +0.8 % at $P_f = 10^{-3}$ and +1.3 % to +1.6 % at $10^{-6}$ for
$N = 5000$ (+5 % at $N = 1000$).

**Replicates and interval.** The point estimate is the arithmetic mean $m$ of 8 independent replicate estimates. The
interval targets that mean,

$$ m\,\exp\!\left(\pm\,t_{n-1}\,\frac{s}{m\sqrt n}\right), $$

with $s$ the replicate standard deviation: a delta-method interval that stays positive (it is not an interval for
$E[\log P_f]$). It is labelled **approximate**; its measured coverage on smooth cases is 97–99 % at nominal 99 %. It
is **withheld** when any replicate is zero, when any replicate ended in `plateau` or at the level cap, or when the
replicate CoV exceeds 1 (`replicates inconsistent`). A single-run CoV is reported only as an approximation, with
each level's CoV including the chain-correlation factor $(1 + \gamma_i)$ estimated from the chains, under both the
independent-levels and the fully-correlated combination, both labelled as approximations.

### 4.4.2 Importance sampling (optional refinement)

A defensive mixture in standard normal space,

$$ q(u) = w_0\,\varphi(u) + \sum_{j=1}^{k} w_j\,\varphi(u - c_j),\qquad w_0 = 0.2,\ w_j = \frac{1-w_0}{k}, $$

with centres $c_j$ supplied by the caller — normally the most likely failure points of §4.7, which come from
clusters of failure samples, not from FORM. The weights $\varphi/q \le 1/w_0$ are bounded, so the estimator
$\hat P_f = \frac1N\sum \mathbf 1_{fail}\,\varphi/q$ has finite variance whatever the centres miss. Two intervals are
reported:

- an **asymptotic** lognormal interval from the sample CoV, labelled *conditional on the proposal covering every
  failure region*, given only when the effective number of failing samples $(\sum w)^2/\sum w^2$ is at least 100 (a
  gate against noise only: it does not detect a missed region);
- the **empirical-Bernstein** interval, which holds whatever was missed [Audibert, Munos and Szepesvári 2009,
  Theorem 1]: half-width $\sqrt{2\hat V\ln(3/\delta)/N} + 3\ln(3/\delta)/(w_0 N)$ with $\delta = 0.01$. Its second term
  is about $86/N$, far wider than a small $P_f$.

Importance sampling is never the headline and never replaces the guaranteed channel. Per-centre shares and the
defensive share of the estimate are reported.

### 4.4.3 FORM and SORM

`prob.form` and `prob.sorm_breitung` are **generic routines for a user-supplied smooth limit-state function** in
standard normal space:

- FORM: improved HL-RF (Newton direction with an Armijo line search on a merit function), central-difference
  gradients, several starting points; the design point is the converged point nearest the origin; convention
  $\alpha = \nabla g/|\nabla g|$ and $u^* = -\beta\alpha$; distinct converged points within 0.5 in $\beta$ are counted
  and reported; FORM refuses when the origin is in the failure region ($g(0) \le 0$) or no start converges;
  $P_f \approx \Phi(-\beta)$ is a first-order approximation, not an estimate with an error bound;
- SORM (Breitung): $P_f \approx \Phi(-\beta)\prod_i(1 + \beta\kappa_i)^{-1/2}$ with principal curvatures from a
  finite-difference Hessian in the tangent plane; refused when any $1 + \beta\kappa_i < 0.5$.

**They are not applied to FAD problems.** The FAD failure set is a union of guarded, non-smooth events
([01_fad.md](01_fad.md) §1.6). Every applicability rule tried (design point strictly inside a segment, a gap in
$\beta$ to competing events, finite-difference smoothness, consistency with sampling) was shown to pass on some case
while FORM was wrong by orders of magnitude, or to give a verdict that depended on sample size. For engine problems
the design point is replaced by the most likely failure point found from samples (§4.7), labelled approximate.

The generic routines are verified on a linear-normal case ($\beta$ to $10^{-6}$), correlated normals (design point to
$10^{-6}$), correlated lognormals with a Pearson target ($\beta$ to $10^{-3}$), a quadratic limit state (SORM within
2 % for $|\beta\kappa| < 0.5$; refused when $1 + \beta\kappa \le 0$) and the origin-in-failure case (refused), and
against OpenTURNS on a smooth limit state ($\beta$ to $10^{-6}$).

## 4.5 Verification of the solvers

Each benchmark has a criterion that fails if the implementation is wrong in the way named; coverage is measured
over 1000 seeds unless stated.

| Case | Exact | Criterion |
|---|---|---|
| linear, independent normals, $P_f = 10^{-3}$ and $10^{-6}$ | $\Phi(-\beta)$ | MC interval covers in ≥ 975/1000 at $10^{-3}$; subset interval ≥ 950/1000 at both; mean subset estimate within 4 % |
| two-branch series, $\beta = 3.0, 3.2$ | $1 - \Phi(3.0)\Phi(3.2)$ | as above; two failure regions with shares within 0.05 |
| two-sided $\lvert U\rvert > 5$ | $2\Phi(-5)$ | subset covers ≥ 950/1000; two regions; association ranks $U$ above an inert variable in ≥ 990/1000 |
| missed region (a failure set the driving variable does not lead to; $P_f = 1.03\times10^{-5}$) | closed form | with $10^7$ MC samples `resolved` and covered in ≥ 195/200; with $10^6$, any rare-event estimate outside the guaranteed interval is `contradicted`, no `conditional` headline lies below the guaranteed lower bound, the guaranteed interval covers ≥ 975/1000; with $10^4$, the guaranteed upper bound exceeds the truth in ≥ 990/1000. A test also records that the conditional interval alone misses the truth in most runs — the hazard the status exists for |
| discontinuous-yield FAD and high-hardening continuous FAD with normal $L_r$, $K_r$ | 2-D quadrature | MC ≥ 975/1000; subset ≥ 950/1000 |
| step line dominated by a guard boundary | closed form | as above |
| discrete input with mass exactly on the line, the cut-off and the step | exact sum | boundary masses counted as failures |
| plateau cases | closed form | status `plateau` in every run; result equals the MC channel |
| 100 failure events, one at $\beta = 3$, others at 4 | closed form | as the linear case |
| subset with proposal correlation fixed at 0.98 | $\Phi(-\beta)$ | estimated $\gamma > 5$; replicate interval covers ≥ 950/1000 |
| replicate-interval function on synthetic replicate sets | exact enumeration | two-point set: coverage 0.9609 ± 0.001; any set with a zero or CoV above 1 withheld |
| plate surface flaw, 6 variables, $P_f \approx 10^{-4}$ | stored $10^8$-sample MC | MC and subset cover in ≥ 95/100 (tests the solvers, not the engine) |

Clopper–Pearson coverage is at least 0.99 when computed exactly at $Np = 4.61$ and $5.2$.

## 4.6 Invalid realisations: the bound pair

Support and joint constraints are validated before sampling. A realisation for which some solution is out of validity
is never dropped, clipped or resampled. Its probability is estimated from the same nominal samples as the guaranteed
interval (with its own Clopper–Pearson interval), and $P_f$ is reported as a pair:

- **headline**: out-of-validity realisations counted as **failed** ($y = -\infty$);
- **valid-only bound**: the same counted as **safe** ($y = +\infty$), computed by a second solve whenever any
  out-of-validity realisation was seen (in the MC channel, among failure samples, or in any subset run).

Fatigue growth past a solution's validity limit is handled by re-characterisation instead ([05_fatigue.md](05_fatigue.md)
§5.3.3).

## 4.7 Diagnostics from failure samples

**Most likely failure points.** From the failure samples of the channel that gave the headline. If the origin (every
variable at its median) fails, it is returned alone with distance 0. Otherwise the samples are split into up to 64
compact groups (deterministic k-means: first centre nearest the origin, further centres farthest-first, 25 rounds).
For each group two straight lines from the origin are scanned (200 points) and the first change from acceptable to
failing is polished by bisection: the line to the group's innermost sample and the line along its mean direction; the
nearer crossing is kept. Groups are joined into **regions** from the origin outward: a group joins the nearest
(in direction) inner group to which it is *linked* — every tested point fails on the radial line from the inner
crossing out to the larger radius and on the arc between the two directions at that radius (17 points each). A region
is thus a set of groups reachable without a barrier; two limit states that intersect further out stay two regions.
More than six regions are merged into the nearest kept ones and labelled. Each region's point is reported in
physical variables with $|u|$, the number of variables and its share of the failure samples, labelled
**approximate** and, above ten variables, **indicative in direction**.

Measured behaviour: one half-space gives one region within 0.4 % of the reliability index for 2–50 variables; a
second region with 5 % of the samples at 90° or 180° is found up to 12–30 variables; regions closer than about 45°
merge from about 20 variables at an index of 3, and at an index of 2 regions 60° apart merge in one run in five to
ten from 15 variables; a merged pair is reported up to 15 % too far. A failing band thinner than 1/200 of a scanned
line can be stepped over. No probability is derived from these points.

**Association with failure.** For each random variable, with $m$ and $v$ the mean and variance of its normal score
among the failure samples,

$$ A = \tfrac12\left(m^2 + v - 1 - \ln v\right), $$

the Kullback–Leibler divergence from the standard normal of a normal with those moments. It responds to shifts and to
changes of spread (a two-sided cause is not ranked zero). It measures **association, not causation** — a variable
correlated with an important one ranks high, and an inert discrete variable gets a non-zero value from noise — and
the output says so. Nothing is reported from fewer than 30 failure samples; a variable with no spread among the
failure samples is listed last without a number.

**Zero sensitivity to toughness in a collapse-governed region** is a correct result, not an error.

## 4.8 Distribution-family sensitivity

`family_sensitivity()` re-runs the assessment with **each** variable declared `Normal`, `LogNormal` or `Weibull`
(truncated or not) replaced in turn by the other two families, with parameters solved so that mean and standard
deviation **after truncation** match and the support is the same. Where no moment-matched alternative exists (a normal
truncated at zero cannot reach a CoV of 1; the search also stops within about 1 % of a family's reach) the output says
so. The latent copula coefficients are kept, so a correlation declared as a Pearson coefficient changes slightly in a
variant (0.6 became 0.594–0.596 in one case). Other variable kinds are listed as having no alternative family.

## 4.9 Partial safety factors

**Published factors** [register `partial_factors.table`, primary reference: AEA Technology [AEA2002]]. A discrete
lookup of published partial factors on stress, flaw size, toughness and yield strength by target failure probability
and coefficient of variation: values exactly as tabulated, **no interpolation**, and "not permitted" entries raise.
The factors apply to reference values: the mean stress and flaw height, mean minus one standard deviation of
toughness, mean minus two standard deviations of yield strength. They were derived by first-order second-moment
analysis for the continuous-yield Option 1 line, independent variables, normal stress and flaw size, Weibull or
lognormal toughness and lognormal yield strength; any statement that they are conservative for a problem is made only
when a run reproduces those assumptions, and the output says so.

**Implied factors.** From a most likely failure point and user-supplied reference values per variable,

$$ \gamma_{load} = \frac{x^*}{x_{ref}},\qquad \gamma_{resistance} = \frac{x_{ref}}{x^*}, $$

approximate as the point is. (The closed form $x^* \approx \mu(1 - \alpha\beta V)$ applies to an independent normal
only and is not used.) Mean minus two standard deviations is not positive at a CoV of 0.5 or more; no factor is then
defined.

## 4.10 Inspection-conditioned flaw sizes (static)

[`prob.inspection`] With a prior flaw-size distribution and a probability-of-detection model:

- `undetected(prior, pod)`: density $\propto \text{prior}(a)\,(1 - \text{PoD}(a))$, returned with the probability of
  non-detection $P(ND)$;
- `detected(a_measured, sizing_error, prior, pod)`: density
  $\propto \text{prior}(a)\,\text{PoD}(a)\,p_{err}(a_{measured} - a)$, returned with its normalising constant.

Both are built with respect to the prior measure: discrete masses are reweighted; a continuous prior is integrated in
its normal score from both ends (so tail probabilities keep relative accuracy), with cell boundaries at declared PoD
discontinuities and, for a sized outcome, at the measured size less the sizing error's quantiles evenly spaced in its
normal score (resolving errors far narrower than the prior, skewed, or unbounded at zero). A PoD model may supply
$1 - \text{PoD}$ directly (`non_detection`), which the built-in log-logistic, step and constant models do, so
non-detection probabilities far below $10^{-16}$ are kept. A conditioned distribution more than about 35 normal scores
into a prior tail is refused, never returned cut off. Flaw sizes are positive by construction. Checked against
conjugate results down to a ratio of $10^{-6}$ and against independent integrals.

**Per component.** With a Poisson flaw-occurrence rate $\nu$ (before inspection),
$P_{f,comp} \le 1 - \exp(-\nu\,P(ND)\,P_{f|flaw})$: exact only if flaws share no random variable; with a shared load
or material property the exact value is smaller (Jensen's inequality). Every $P_f$ is labelled per flaw or per
component.

Inspection *policies* during service life are part of the time-dependent layer ([05_fatigue.md](05_fatigue.md) §5.6).

## 4.11 Uncertainty that is and is not represented

Model-uncertainty factors on K, the reference stress and the line are available and off by default
([01_fad.md](01_fad.md) §1.1). Every result lists the uncertainty sources it excludes. Distribution-parameter
(statistical) uncertainty is not propagated. The guaranteed interval is a sampling statement about the declared model,
not about reality.
