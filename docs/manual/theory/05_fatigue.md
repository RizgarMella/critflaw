# 5. Fatigue crack growth and time-dependent reliability

This chapter describes how flaws grow under cyclic loading, how the growth is integrated with its events
(thresholds, stage changes, breakthrough, validity limits), how the fracture check is paired with an extreme load in
time, the three time-dependent probabilities, and inspection and repair in service. Implementation:
`critflaw.fatigue` (`laws`, `spectrum`, `growth`, `timedependent`).

Not modelled: load-sequence effects on the rate (overload retardation), crack closure, effective-range growth near the
threshold (the source gives no constants: `TODO(verify)`), creep, stress corrosion, wall loss, flaw interaction,
initiation life.

---

## 5.1 Growth laws

[register `fatigue.paris_law`, primary reference: Paris and Erdogan [ParisErdogan1963]]

One law serves the depth and length directions. The rate depends on the stress intensity factor range only:

$$
\frac{da}{dN} = \begin{cases} 0, & \Delta K < \Delta K_0 \\ A\,\Delta K^m, & \text{single stage} \\ \min\left(A_A\,\Delta K^{m_A},\ A_B\,\Delta K^{m_B}\right), & \text{two stage} \end{cases}
$$

A range exactly at the threshold counts as growing. The two-stage form is the transition by intersection only when
$m_A > m_B$; a law with $m_A \le m_B$ is refused. The transition point is

$$ \Delta K_{tr} = \left(\frac{A_B}{A_A}\right)^{1/(m_A - m_B)}, $$

computed, not read from print (the intersection reproduces the printed transition points within rounding, a test for
all sixteen catalogue pairs). Rates are SI (m/cycle, Pa√m). Published coefficients in mm/cycle and N/mm^{3/2} are
converted once, $A_{SI} = A\cdot10^{-3}\cdot(10^{-6}\sqrt{1000})^m$, with a test that both unit systems give the same
rate to $10^{-12}$.

**Catalogue** [registers `fatigue.recommended_laws`, `fatigue.thresholds`, primary reference: King [King1998]]. The
recommended two-stage laws for steels in air and in marine environments (free corrosion, and with cathodic
protection), mean and mean + 2 SD, for stress ratio below and at or above 0.5; the simplified $m = 3$ laws; the
standard deviation of $\log_{10}A$ per stage; thresholds for welded joints and, for unwelded steel, as a function of
stress ratio; a cap on the threshold for surface flaws less than 1 mm deep (which makes the threshold depth dependent
and an event of the integrator). Each entry carries its validity statement (strength, temperature, environment) as
text printed in the trace; the library cannot check temperature or environment. The cap is applied to plate and
cylinder surface flaws and to extended flaws, **not** to embedded flaws (they are not surface breaking; this is the
less conservative side for a shallow embedded flaw).

**Statistics.** Deterministic assessments use the upper-bound (mean + 2 SD) laws, by default for stress ratio at or
above 0.5. Probabilistic assessments use the **mean** law with $\log_{10}A$ of each stage normal with the published
standard deviation and fixed exponents; random coefficients are never layered on an upper-bound law (attempting it
raises). The correlation $\rho$ between the two stages' $\log A$ is a **required argument without default**. Two
consequences are stated in every result:

- the transition point is random, with $\ln\Delta K_{tr}$ normal of standard deviation
  $\ln 10\,\sqrt{s_A^2 + s_B^2 - 2\rho s_A s_B}/(m_A - m_B)$;
- the minimum of two random lines lowers the expected log rate near the knee below the published mean law (more at
  low $\rho$; an unconservative shift); results that depend on a two-stage law are also reported at $\rho = 0$ and
  $\rho = 1$ as a sensitivity.

## 5.2 Cyclic loading and spectra

**Range.** A block is $(\Delta\sigma_m, \Delta\sigma_b, \text{cycles})$; membrane and bending ranges of a block are in
phase and are combined first:

$$ \Delta K = \max\left(Y_m\Delta\sigma_m + Y_b\Delta\sigma_b,\ 0\right)\sqrt{\pi a}, $$

with the geometry factors of [02_solutions.md](02_solutions.md) (including finite width and the stress-concentration
and misalignment factors) at the current flaw size and at each growing position. A non-positive combined range (the
bending factor is negative at the deepest point of deep flaws) is treated as zero and flagged. Residual stress is not
part of the range; no plasticity correction is applied. A random `load_scale` multiplies every range; whether it also
multiplies the extreme load is a required explicit argument `shared_scale` whenever it is random.

**Spectra.** The order of cycles is part of the model, not of the numerical error (4 % to 44 % differences in life
between orders were found when a threshold is present). Two forms:

- `Spectrum.repeating(blocks, cycles_per_year)`: the blocks are one pass of a repeating pattern; the rate is the
  expectation over the pass,
  $$ \frac{da}{dt} = \nu \sum_i p_i\,\text{rate}(\Delta K_i), $$
  with the threshold and the two-stage law applied inside the expectation (no equivalent-range shortcut). The growth of
  either dimension within one pass must stay below $\eta/4$ for every sample (an error otherwise; at that level the
  within-pass order effect is below 0.1 % of life). The accuracy claim is 0.5 % of cycles **or one pass, whichever is
  larger**.
- `Spectrum.ordered(blocks, cycles_per_year, order)`: the blocks are applied once in the declared order (a lifetime
  histogram whose cycles must sum to the life); `order` is required (`as_given`, `largest_first`, `smallest_first`).
  `compare_orders()` runs the three and reports the shortest life. This is the worst of three declared orders, not of
  all permutations, and the result says so.

A deterministic run may halve a variable-amplitude life by `life_factor`; a probabilistic run refuses it.

## 5.3 Integration and events

### 5.3.1 State

Per sample: flaw kind (surface, through, embedded, extended, none, ended), $a$, $c$, the embedded flaw's $B'$, time,
status and flags. A flaw is **dormant** when every block at every position is below threshold; it is retired from
integration and still assessed for fracture. A flaw is not dormant while any position grows (growth in length can wake
the deepest point and vice versa).

### 5.3.2 Scheme

Explicit second-order (Heun) steps in time, per sample, with events located and never stepped over:

1. **Frozen active set.** At the start of a step each (block, position) pair is above or below threshold and on stage
   A or B; with the K solution's branch, the set is frozen for the step.
2. **Step size**: the smallest of the time for the relative growth of $a$ or $c$ to reach $\eta = 0.02$, the
   controller's proposal, and the time to the next checkpoint or block end.
3. **Error control**: $e = |\text{Heun} - \text{Euler}|/\text{size}$ per dimension; a step with $e > 2\times10^{-4}$
   is repeated with $h \leftarrow 0.9h\sqrt{\text{tol}/e}$, and the next proposal grows by at most a factor 2. A
   controller step below $10^{-12}$ of the requested life raises (never a silent stall).
4. **Rate discontinuities** (threshold crossings, stage transitions, the 1 mm cap, a K-solution branch change): the
   first change on a step is located by bisection in time to $10^{-6}$ of the step and the step is cut there.
5. **Geometry boundaries**: if the predictor or the accepted state lies outside the closed boundary set, the step is
   replaced by a **terminal step** onto the earliest boundary, $h = 2d/(v_0 + v_1)$ with one secant correction, so the
   accepted state lies on the boundary to $10^{-12}$ relative. Rates are never evaluated outside the boundary set.
6. **Sliding at a threshold**: under bending-dominated ranges the deepest-point $\Delta K$ falls as $a$ grows and rises
   as $c$ grows, so a state can be unable to stay on either side of the threshold. When a located crossing is followed
   by the reverse crossing within $10^{-3}$ of the step the pair enters a sliding state in which the depth rate holds
   $\Delta K$ at the threshold: $X = -(\partial\Delta K/\partial c\cdot v_c)/(\partial\Delta K/\partial a)$, the sliding
   block contributing $\operatorname{clip}(X - v_{others}, 0, v_{a,i})$. Such samples are flagged.
7. **Through flaw approaching the width.** At $2c \to W$ the finite-width factor and the rate are unbounded, but the
   remaining time is finite. Once the flaw enters the terminal zone $2c/W \ge 0.9$ (or the controller step falls below
   100 minimum steps) the remaining time is computed as the size integral of $1/v$: the end-point singularity is
   removed by the substitution $c = W/2 - s^2$ and the integral is taken panel by panel with 8-point Gauss–Legendre on
   a grid that includes every kink of the rate, doubled until two levels agree to $10^{-10}$. Sizes at interior times
   are solved inside their panel, not interpolated. The sample then ends with status `ended: outside validity`.

On smooth growth the global error over-predicts life (the unconservative sign) by $5\times10^{-5}$ to
$1.5\times10^{-4}$ in cycles at the defaults for $m$ = 2 to 8.16; the error is proportional to `tol` with the
controller on and to $\eta^2$ with it off. Cost is about 60–250 steps per sample per decade of growth.

### 5.3.3 Breakthrough, re-characterisation and the end of validity

Breakthrough is not failure. Events are `fracture`, `collapse` and `leak`, and the failure definition is the single
required argument **`leak_is_failure`** (fracture and collapse are always failure), printed with every result.

| Situation | Treatment | Event |
|---|---|---|
| surface flaw reaches $a = B$ | becomes a through-thickness flaw of half-length $c$ | `leak` |
| surface flaw reaches any validity limit of its solution before breakthrough ($a/B$, aspect ratio, $2c/W$) | re-characterised to a through flaw of half-length $c$; the destination is checked at once | `leak_assumed`, with the trigger and whether the depth was growing; mass reported separately |
| embedded flaw: ligament $p$ reaches 0 | surface flaw of depth $2a$ and length $2c$, checked against the surface solution at once and cascaded to a through flaw if invalid; a mid-thickness flaw reaches both surfaces at once (`surface_breaking` and `leak`) | `surface_breaking` (never a failure) |
| embedded flaw reaches a validity limit first | surface flaw of depth $2a + p$ and length $2c$ (the ligament deemed broken), cascaded as above | `leak_assumed` (embedded trigger codes) |
| cylinder surface flaw (tabulated route) reaches a table edge ($a/B = 0.8$, an $a/c$ end, $2c/W = 0.15$) | through-wall cylinder flaw of length $2c$, checked against the through-wall range ($r_m/B \ge 5$, $\lambda$ in range); outside it — the common case — the sample ends | `leak_assumed`, then possibly `ended` |
| through flaw reaches its range end (plate $2c \to W$; tabulated cylinder $\lambda$ end, column dependent) | no further re-characterisation; terminal step or quadrature finish | `ended: outside validity` |
| kinds whose K is singular at $a = B$ (axial surface flaw under the pressure-only route; fully circumferential external flaw) | ended at $a/B = 0.9$ (a declared end, not a source rule) | `leak_assumed`, then `ended` |
| extended flaws | end at their depth limit (0.8 B tabulated, 0.6 B plate polynomial, 0.9 B singular kind); no continuation solution | `leak_assumed`, then `ended` |
| a rate undefined at a step start (state not covered by the tables) | the sample ends, never dormancy | `ended` |

Re-characterisation at the aspect-ratio limit is a conservative device so that no growth is computed outside a
solution's range, not a physical statement (a long shallow flaw whose depth is arrested does not leak). After
`ended: outside validity` a sample follows the bound-pair rule: counted failed in the headline and safe in the
valid-only bound ([04_probabilistic.md](04_probabilistic.md) §4.6). Every event carries a trigger code; results report
the share per code and separate `leak` from `leak_assumed`. Up to four flaw-type changes per step are recorded (a
fifth raises).

### 5.3.4 Accuracy evidence

The maximum over samples of the relative error in cycles to a given size is at most 0.5 % at default settings
against closed forms (single stage, constant geometry factor, $m$ = 2, 3, 4), against a cycle-by-cycle reference with
$a$–$c$ coupling, threshold, both stages and breakthrough (on the samples nearest failure), and the integrator agrees
to $10^{-8}$ with an independent fixed-step reference implementation at identical steps on event-free cases (plate
surface flaw, embedded flaw, tabulated cylinder surface and extended flaws). Definition-pinning limits: after
breakthrough the remaining life equals the through-flaw closed form with half-length $c$ (not $2c$); a shallow
semicircular flaw grows at both positions in the ratio of its geometry factors to the power $m$. Landing on every
reachable boundary is checked to $10^{-12}$ with no evaluation outside the boundary set.

## 5.4 The extreme load and its pairing with the flaw

The fracture check uses an extreme load declared as a maximum distribution **with its reference period**,
`ExtremeLoad(distribution, reference_period_years, component)`, replacing one named primary stress component. The
load process is discretised in steps $\Delta$ (default: the reference period), separate from the reporting times; the
maximum over a step has distribution function

$$ F_\Delta(x) = F(x)^{\Delta/T_{ref}}, $$

sampled as $x = F^{-1}(\Phi(u)^{T_{ref}/\Delta})$ through the upper tail for precision. This assumes independent,
identically distributed sub-period maxima (a stationary process with no seasonality or serial dependence), stated in
every result; refining the reporting grid does not change the lifetime result (tested). One standard normal per step.

**Pairing.** The time within a step at which its maximum occurs is unknown, so each step is bracketed. The checkpoints
of a step are its start, the states immediately before and after each flaw-type change in it, its end, and four
interior states at fractions 0.2, 0.4, 0.6, 0.8 of the step. With the step maximum $M$:

- **upper pairing** (headline for the cumulative probability): the step fails if $M$ fails the acceptance test at the
  most severe checkpoint;
- **lower pairing**: the step fails if $M$ fails at the least severe checkpoint.

These bound the step's failure probability under **assumption A: the driving variable at fixed load is monotone in
time between consecutive checkpoints.** Assumption A cannot be proved from checkpoint values (a severity excursion
between two checkpoints is invisible to them), so every pairing result is labelled *bracket under assumption A
(monotone severity between checkpoints; checked at the checkpoints, not proved)*. Violations visible at the
checkpoints are reported and the step then uses the most and least severe of all its checkpoints. Both sides of a
flaw-type change are checkpoints because a change can make the flaw less severe (reference stress 1.998 $P_m$ just
before breakthrough against 1.111 $P_m$ just after in one case).

**Refinement.** When the discordant count (samples failing under the upper pairing and surviving under the lower one)
is at least 1 and the upper estimate exceeds the lower by more than 20 %, the result says "bracket under assumption A
is wide; refine the load-process step"; `solve(refine=True)` halves $\Delta$ up to four times, drawing sub-step maxima
so that the maximum of the two sub-steps equals the coarse step's maximum (sample-wise coupling).

## 5.5 The three time-dependent probabilities

The failure indicator at a step is the direct acceptance test at the governing checkpoint (with `leak_is_failure`, a
step in which a leak occurs is failing); first passage uses $y_k = \min_{j\le k} g_j$. From $N$ Monte Carlo samples,
with $n_k$ first failures in step $k$ and $S_k$ survivors at its start, three probabilities are reported separately
and never interchanged:

| Quantity | Estimator | Interval (99 %, pointwise) |
|---|---|---|
| cumulative first-passage $P_f(0, t_k)$ | $\sum_{j\le k} n_j/N$ | Clopper–Pearson |
| interval (e.g. annual) probability | $n_k/N$ | Clopper–Pearson |
| hazard (conditional on survival) | $n_k/S_k$ | Clopper–Pearson on $n_k$ of $S_k$; undefined when $S_k = 0$ |

The cumulative probability is given as the pair (lower pairing, upper pairing), each with its interval, from the same
samples. Only annual values are compared with per-year targets; when $\Delta$ is not a year, annual values aggregate
counts over the year's steps, which keeps the intervals exact. A repaired sample is a survivor at risk only from a
repair flaw, so the hazard is a population average and the result says so. A step in which both a leak and a
load-induced failure occur is reported in a separate class "order unresolved" (its size falls as $\Delta$ is refined).

**Survival-conditioned quantities** (the hazard, and the inspection update of §5.6.2) are **not** bracketed by the two
pairings: each within-pairing value carries two biases of opposite sign. Three things are reported:

- the two **within-pairing** values, each exact for that pairing's discretised model — a sensitivity, not a bracket;
- the **conservative pair**, which does bracket the truth under assumption A:
  $$ UB = \frac{n_x}{n_x + S_{up}(t_k)},\qquad LB = \frac{n_y}{n_y + S_{lo}(t_k)}, $$
  where $n_x$ counts samples that survive to $t_j$ under the lower pairing and have failed by $t_k$ under the upper
  pairing (whenever that failure occurred), $S_{up}(t_k)$ the samples surviving to $t_k$ under the upper pairing, and
  $n_y$, $S_{lo}$ the mirror. (Under assumption A, true survival to $t_j$ implies lower-pairing survival and true
  failure by $t_k$ implies upper-pairing failure, and $x/(x+y)$ is monotone in each argument.) For this every sample is
  carried until it fails under the lower pairing. A variant using "first passage in $(t_j, t_k]$" is not a bound and is
  not used;
- a mixed ratio of one pairing's failures over the other's survivors is not computed (it can exceed 1).

**Subset simulation in time** is offered only when `leak_is_failure=False` (the driving variable is then continuous):
one run on the lifetime minimum gives the lifetime probability with the statuses of [04_probabilistic.md](04_probabilistic.md)
§4.4; earlier times come from separate runs, or from the lifetime run as $P_f(0,T)$ times the share of final-level
failure samples with first passage by $t_k$, with a resolution floor below which the value is printed as "below
resolution", never zero. Hazards are not derived from subset results.

## 5.6 Inspection and repair in service

### 5.6.1 Policy simulation (population view)

`Inspection(time, pod, sizing_error=None, accept_below=None, action="repair", repair_flaw=None, ...)` at step
boundaries. The order at a boundary is: fracture check of the ending step, inspection, repair. Detection uses one
standard normal per inspection (`detected` iff $\Phi(u) < \text{PoD}(a)$); with `accept_below` and a sizing error (one
more variable) a detected flaw is repaired only if its measured size is at or above the acceptance size. The random
vector is augmented with these variables, so $g$ remains a fixed function of one random vector. Surface flaws are
inspected by depth, embedded flaws by height $2a$ (`pod_embedded`, required when embedded flaws are grown), through
flaws by length (`pod_through`; detected with probability 1 if none is given). Successive inspections of the same flaw
are independent given its size (the favourable assumption; real inspections are positively dependent); no false calls
are modelled. Repair removes the flaw; a `repair_flaw` then grows from the repair time, and the removed flaw's event
record is cleared (its history is reported separately). Reported per inspection: probabilities of detection and of
repair with two denominators (all samples; survivors carrying a flaw), and the probabilities of §5.5 before and after.

### 5.6.2 Updating one component from its own inspection outcome

`Inspection(..., observed="not_detected" | "detected" | ("sized", a_m))`, Monte Carlo channel only. The conditioning
event is stated: *survival to the inspection time and the observed outcome*.

- *Not detected* and *detected without a size* have positive probability and are handled by **rejection**: the retained
  samples are exact draws from the conditional distribution of each pairing's discretised model; later probabilities are
  counts among them with Clopper–Pearson intervals, and the bracket is the conservative pair. Flaw-free samples (after an
  earlier repair) are "not detected" with probability 1 and have zero likelihood for a detection.
- *Detected and sized* at $a_m$ has probability zero, so surviving samples are **weighted** by
  $\text{PoD}(a_i)\,p_{err}(a_m - a_i)$; the result is conditional, with Kish effective sizes among all weighted samples
  and among those that fail later, and is withheld when the latter is below 100.
- Several outcomes are applied in time order; once a population is weighted it stays weighted (plain counts after a
  later non-detection were shown to give 0.5 for a true 0.1 in a two-inspection case).

The update is gated on the discordant-count test of §5.4 at the inspection time.

### 5.6.3 Verification

Exponential flaw size with a step PoD and a two-point distribution against closed forms; component updates against the
static `undetected` result, two-size cases and a policy case; the conservative pair against closed-form toys where the
within-pairing values do not bracket (for example a true hazard of 0.208 with within-pairing values 0.127 and 0.099);
no-growth cases against $1 - (1-p)^k$ and its expectation over a random resistance; a deterministic decline of the
critical load (lower ≤ limit ≤ upper, width first order in $\Delta$); a severity dip that the interior checkpoints see
(the violation label appears) and one they miss (both pairings give 0 and the assumption label is present — the limit
of what checkpoints can do, shown).

## 5.7 Assumptions printed with every time-dependent result

Assumption A (monotone severity between checkpoints; checked, not proved) on every pairing result; independent step
maxima (stationary, no seasonality); spectrum form and order; sequence effects on the rate not modelled; the stage
correlation; whether the load scale is shared; independence of successive inspections; pointwise intervals; which
pairing each number uses; the leak definition.
