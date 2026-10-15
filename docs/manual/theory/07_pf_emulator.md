# 7. Failure probability with the Option 3 emulator

This chapter describes how a failure probability is computed when the assessment line is the experimental Option 3
emulator of [06_fe_option3.md](06_fe_option3.md): how one realisation is classified, how the curve is bounded between
FE load steps, how function draws of the emulator are formed, the bound pair that carries undecided mass, the crossed
design that separates surrogate from sampling uncertainty, and the acquisition and stopping rules of the adaptive
loop. Implementation: `critflaw.option3.emulator` (`FunctionDraws`, `step_bounds`), `critflaw.option3.reliability`
(`population`, `pf_crossed`), `critflaw.option3.active` (`run`).

**Status.** With the current emulator every Pf is **experimental**: `pf_crossed` and `active.run` raise unless called
with `experimental=True`, and every result carries the labels "surrogate-only interval: excludes model uncertainty,
distribution-parameter uncertainty, FE-versus-reality, GP hyperparameter uncertainty", "experimental: unverified FE
data, deepest point only" and the between-step bound mode.

---

## 7.1 The problem and the failure event of one realisation

A problem maps standard normal samples $u$ to $(x, L_r, K_r, L_{r,max})$: the emulator inputs $x$ (the geometry and
material part of $u$), the assessment point (with $L_r$ referred to $\sigma_0$, the yield strength of the curve; a
user's proof stress is converted with `sigma0_from_proof_stress`) and the collapse cut-off from the sampled tensile
properties. The failure event is the library's single indicator ([01_fad.md](01_fad.md) §1.6): failed iff
$L_r \ge L_{r,max}$ or $K_r \ge f(L_r)$. Each realisation is classified in this order:

1. **Collapse**, $L_r \ge L_{r,max}$: **failed**; the emulator is not queried.
2. **Non-finite** $L_r$, $K_r$ or $L_{r,max}$ (not already failed by a valid collapse): **`invalid`** — undecided,
   counted in $P_{f,high}$ and not in $P_{f,low}$; never silently safe.
3. **$K_r \ge 1$**: **failed for every draw**, since every emulated curve has $f \le 1$ by construction.
4. $x$ outside the trained box or non-finite: **`outside_box`** / **`invalid`** (undecided).
5. $L_r$ beyond the curve's support: **`beyond_support`** (schedule rule) or **`cap_uncertain`** (small-strain cap
   rule) (undecided).
6. Otherwise the draw's between-step bounds decide (§7.2): **safe**, **failed**, or **`between_steps`** (undecided).

Nothing is dropped, clipped or resampled; the mass of each undecided class is reported.

## 7.2 Between-step bounds

The emulator gives the curve at the FE load steps $L_{r,k}$ only. The library claims no interpolation for decisions;
it uses bounds.

**Monotone (rigorous).** The true curve is non-increasing, so for $L_r \in [L_{r,k}, L_{r,k+1}]$

$$ K_{r,k+1} \le f(L_r) \le K_{r,k}. $$

A realisation is **safe** if $K_r < K_{r,k+1}$, **failed** if $K_r \ge K_{r,k}$, and **`between_steps`** otherwise.
At a step the bound is the step value; below the first step $f = 1$ ($J = J_e$).

**Convex (tightened, under an assumption).** Write $R = J/J_e = K_r^{-2}$. If $R$ is convex in $L_r$ between the
steps, the chord bounds $R$ from above (so $K_r$ from below) and the extended neighbouring secants bound it from below
(so $K_r$ from above):

$$ R(L_r) \le R_a + s_{a}\,(L_r - L_{r,a}),\qquad R(L_r) \ge \max\left\{R_a,\ R_a + s_{a-1}(L_r - L_{r,a}),\ R_b - s_{b}(L_{r,b} - L_r)\right\}, $$

with $s$ the secant slopes of the interval $[a, b]$ and of its neighbours. The tightening is used only where the left
step has $R - 1 \ge 10^{-3}$ and the secant slopes are non-decreasing across the interval and its neighbours (the
convexity visible at the steps); elsewhere the monotone bounds apply. The assumption — $J/J_e$ convex in $L_r$
between steps above $J/J_e - 1 = 10^{-3}$ — is supported by every accepted FE curve (20 of 20,
`benchmarks/pf_convexity_evidence.py`) and checked per draw at the steps; the check catches violations visible at
the steps, not between them. **The headline uses the convex mode and carries the assumption label; the monotone pair
is always available (`convex=False`).**

## 7.3 Function draws

A Pf over a population of realisations needs emulator draws that are **coherent functions of $x$**: the same draw
evaluated at two realisations must be one curve-valued function. A function draw fixes, once for the whole draw:

1. **the GP path**, by pathwise conditioning [Wilson et al. 2020]:
   $$ f(\cdot) = f_{prior}(\cdot) + k(\cdot, Z)\,K^{-1}\left(y - f_{prior}(Z) - \varepsilon\right),\qquad \varepsilon \sim N(0, \sigma_n^2 + \text{jitter}), $$
   where $f_{prior}$ is a random-Fourier-feature sample of the Matérn-5/2 prior,
   $$ f_{prior}(z) = \sqrt{\frac{2s^2}{F}}\sum_{j=1}^{F} w_j\cos(\omega_j\cdot z + b_j),\quad w_j \sim N(0,1),\ b_j \sim U(0, 2\pi), $$
   with frequencies $\omega_j$ from the Matérn-5/2 spectral measure (a multivariate Student t with 5 degrees of
   freedom and scale $1/\ell$: $\omega = (g/5)^{-1/2}\,z/\ell$, $z \sim N(0, I)$, $g \sim \chi^2_5$), drawn fresh for
   each draw, and $K$ the stored training covariance with noise and jitter on its diagonal;
2. **the Richardson factor** $R$ (through the sampled order $p$);
3. **the discrepancy's common standard normal**;
4. **the far-field factor**.

At any single $x$ the distribution of the draws equals that of `Emulator.predict` (tested). The random-feature
approximation error decays as $1/\sqrt F$; at $F = 3000$ the posterior covariance was reproduced to about 6 % of its
largest variance on the test function. `function_draws` defaults to $F = 2000$; the adaptive loop uses 1000, and the
V10 criteria include a check that $F = 1000$ and $2000$ give the same Pf within 5 %.

## 7.4 Populations and the bound pair

**Population.** One fixed weighted population per replicate, shared by all draws (common random numbers): defensive
importance sampling with a stratified mixture

$$ q(u) = \alpha\,\varphi(u) + (1-\alpha)\,\varphi(u - u^*),\qquad \alpha = 0.3\ (\text{realised share } n_0/N), $$

and exact weights $w = \varphi(u)/q(u)$. The centre $u^*$ is the design point of a cheap proxy limit state supplied by
the caller — in an application, the Option 2 curve of the same `WarpDeformationLaw`. The centre only places the
proposal: no probability is taken from it, the weights are exact, and the defensive share bounds them by $1/\alpha$, so
a poor centre costs efficiency, not correctness (a missed region is the hazard addressed by exploration, §7.6).

**Bound pair.** For draw $i$ and population $s$ of size $N$:

$$ P_{f,high}^{(i,s)} = \frac1N\left[\sum_{\text{always failed}} w + \sum_{\text{invalid}} w + \sum_{\text{band: failed or undecided}} w\right],\qquad
P_{f,low}^{(i,s)} = \frac1N\left[\sum_{\text{always failed}} w + \sum_{\text{band: failed}} w\right], $$

where "always failed" is collapse or $K_r \ge 1$ and the band is every other finite realisation, evaluated per draw.
$P_{f,high}$ (undecided counted as failed) is the **headline**; $P_{f,low}$ (undecided counted as safe) is always
reported, with the undecided mass by class (`outside_box`, `invalid`, `beyond_support`, `cap_uncertain`,
`between_steps`), averaged over draws and populations. These validity bounds are distinct from the surrogate-only
interval of §7.5.

## 7.5 Crossed design and variance decomposition

$D$ function draws are crossed with $S$ independent populations, giving a $D \times S$ table of $P_{f,high}$. A two-way
analysis of variance without replication separates the components:

$$ MS_{draw} = \frac{S\sum_i(\bar P_{i\cdot} - \bar P)^2}{D-1},\quad MS_{pop} = \frac{D\sum_s(\bar P_{\cdot s} - \bar P)^2}{S-1},\quad MS_{res} = \frac{\sum_{i,s}(P_{is} - \bar P_{i\cdot} - \bar P_{\cdot s} + \bar P)^2}{(D-1)(S-1)}, $$

$$ \hat\sigma^2_{draw} = \max\left(\frac{MS_{draw} - MS_{res}}{S},\,0\right),\qquad \hat\sigma^2_{pop} = \max\left(\frac{MS_{pop} - MS_{res}}{D},\,0\right). $$

$\hat\sigma^2_{draw}$ is the **surrogate** component net of the interaction (used by the stopping rule);
$\hat\sigma^2_{pop}$ the **sampling** component; the residual holds the interaction and within-cell noise. The
reported Pf is the grand mean of $P_{f,high}$.

- The **surrogate-only interval** is the 5th–95th percentile range of the draw means $\bar P_{i\cdot}$ (each averaged
  over populations). It is printed with what it excludes: model uncertainty, distribution-parameter uncertainty,
  FE-versus-reality, GP hyperparameter uncertainty, and the experimental status of the FE data. It is not "the
  uncertainty of Pf".
- The **sampling standard error** of the population-averaged $P_{f,high}$ comes from the within-population weighted
  variance of each cell (not from the spread of a few population means), averaged over draws.

## 7.6 Acquisition, exploration and stopping

`active.run(problem, dim, oracle, curves, ..., experimental=True)` iterates: train the emulator on the curves so far;
form $D$ coherent function draws; compute the crossed Pf over the fixed populations; test the stopping rule; otherwise
call the oracle (an FE case, or the synthetic function of V10) at one new input and add the whole curve.

**Acquisition (a ranking proxy).** Over the band realisations of the first population, with $m$ the fraction of draws
that classify a realisation as failed ($K_r$ at or above the midpoint of that draw's bounds),

$$ \text{score} = m\,(1 - m)\,w, $$

and the next input is the $x$ of the highest-scoring realisation. The criterion is computed from the coherent draws, not
from a Gaussian closed form, and is declared a ranking proxy. The emulator is refitted at each iteration (no rank-one
update).

**Exploration.** A declared share of the oracle calls (default 15 %) goes to exploration independent of the predicted
boundary: among 500 realisations sampled from the nominal density, the input at which the GP posterior variance (at the
middle supported step) is the largest fraction of its prior variance. This guards against a surrogate that is
confidently wrong in a failure region it has never been shown.

**Explored mass.** The result reports the **unexplored mass**: the (weighted) physical probability of the
failure-relevant band realisations — those that at least one of the first eight draws classifies failed or undecided —
at which the posterior variance still exceeds 0.9 times the prior variance. A calibrated-Pf claim is not made while the
unexplored mass exceeds the reported Pf.

**Stopping** (fixed before V10). Converged when, in order,

1. the sampling CoV of $P_{f,high}$ is at most 0.10 (a zero standard error never counts);
2. $P_{f,high}$ exceeds an absolute tolerance of $10^{-9}$;
3. the surrogate width $2 \times 1.645\,\hat\sigma_{draw}$ is positive and at most $0.5\,P_{f,high}$ (zero estimated
   spread never counts as convergence).

Otherwise the loop ends at the declared oracle budget (default 20 calls, status `budget`), or as **`draw-independent`**
when no population has a band realisation (every realisation is decided without the emulator, so Pf does not depend
on the draws). The reported uncertainty is the total; convergence of the surrogate component does not reduce the
sampling component.

## 7.7 V10: the synthetic end-to-end test

The primary end-to-end test of emulator, draws, acquisition and stopping is a synthetic "FE" function with a known
failure probability near $10^{-5}$ (`benchmarks/pf_v10.py`):

- **oracle**: a closed-form family of monotone curves with the emulator's structure — curves at load steps,
  input-dependent support, a cap region — used in place of FE;
- **problem**: seven standard normal inputs; five map through $\Phi$ uniformly onto the emulator box with $n \le 12$ (so
  every support reaches $L_r \ge 1.12$); $L_r = 0.96\,e^{0.025u_6}$ and $K_r = 0.535\,e^{0.11u_7}$, which places the
  failure region near $L_r \approx 0.97$, $K_r \approx 0.91$ where the curve decides; $L_{r,max} = 1.5$;
- **truth**: $P_{f,high}$ of the true synthetic curves (linear in $J/J_e$ between steps; beyond the support counted as
  failed, as in the headline) by a large importance-sampling run with the same mixture;
- **design point**: found by multi-start optimisation over all seven inputs, since the dominant failures combine tails
  in $x$ (weak curves) with a high load-to-toughness ratio; a centre at the box median would miss them;
- **criteria over 20 independent seeds**: the 90 % surrogate-only interval of $P_{f,high}$ covers the truth in at least
  15 of 20 seeds (binomial at 90 %: $P(\ge 15) \approx 0.99$); premature stops (converged and not covering) at most 2 of
  20; the random-feature check ($F = 1000$ against $2000$) changes Pf by at most 5 % in every seed;
- **isolated-region variant**: a ball of radius 0.14 in unit-box coordinates of $x$, away from the main failure region,
  inside which $J/J_e - 1$ grows six times faster; its exact extra contribution (about $6.3\times10^{-6}$) doubles the
  true Pf. Pass if the region is found (a training curve inside it) or its coverage is reported unresolved (unexplored
  mass at least Pf).

**V10 results (incomplete run, 6 of 20 main seeds; isolated-region variant not run).** The 90 % interval of
`Pf_high` covered the true Pf in 4 of 6 seeds; 2 seeds stopped as converged with an interval that missed it
(the criterion allows at most 2 in 20); the relative change in Pf from 1000 to 2000 random Fourier features was
0.5–7.6 %, above 5 % in 2 seeds. V10 is therefore **not passed**. The likely cause is that the reported interval
spans only the function-draw component: the sampling error and the random-feature approximation error are not in
it, and in the two failing seeds the feature error is as large as the interval's half-width. Until a revised test
passes, read the surrogate-only interval as a lower bound on the uncertainty of `Pf_high`, and confirm stopping
decisions at a higher feature count.

A small direct-FE comparison is the secondary mechanics check, to be run when the machine allows.

## 7.8 Assumptions and limits

- Everything in [06_fe_option3.md](06_fe_option3.md) §6.6 applies: deepest point only, unverified FE data,
  experimental status.
- The headline between-step bounds assume convexity of $J/J_e$ in $L_r$ between steps (checked at the steps, not
  proved); the monotone pair is rigorous.
- The surrogate-only interval conditions on the GP hyperparameters and on the discrepancy model; it excludes model,
  parameter and FE-versus-reality uncertainty.
- The importance-sampling population depends on the proxy's design point; the defensive share bounds the weights but a
  failure region far from both the nominal and the proxy centres is found only through the exploration share and is
  otherwise visible only as unexplored mass.
- Defaults in the adaptive loop ($D = 24$ draws, $S = 3$ populations of 4000, $F = 1000$ features) are light for this
  workstation; the stopping rule, not the defaults, decides convergence.
