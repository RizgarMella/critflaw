# critflaw theory manual

> **⚠️ Research software in development: not verified, not validated, not for real-world use.** See
> [`DISCLAIMER.md`](../../../DISCLAIMER.md).

`critflaw` is an open Python library for the engineering critical assessment of flaws in metallic structures by
failure assessment diagram (FAD) methods, deterministic and probabilistic. The procedures follow the structure of
BS 7910-style FAD assessments. This manual describes the methods the library implements, the assumptions behind
them, the evidence that supports them and the limits beyond which they refuse to give an answer.

The manual does not reproduce the text, tables, coefficient sets or standard-specific closed forms of any standard.
Methods are described by their structure and role, with the implementing function and its register id; only
generic symbolic definitions (such as $L_r$, $K_r$, the cut-off and the Option 2 line from which the library's own
rules are derived), textbook limiting values and the validity ranges a user needs are stated. Methods are attributed to the
primary references recorded in the library's equation register (`src/critflaw/provenance.py`). Where the register
has no primary reference for an item, the manual says *"no open primary reference identified; see
`docs/unsourced_items.md`"*. The full bibliography is in [references.md](references.md).

The source code is the authority on what is implemented; this manual describes the code.

## Contents

| Chapter | Subject |
|---|---|
| [01_fad.md](01_fad.md) | The FAD framework: `Kr`, `Lr`, the Option 1 and Option 2 assessment lines, the cut-off `Lr,max`, secondary stress and the plasticity interaction term `ρ` (or `V`), the direct acceptance test, the radial margin, reserve factors and limiting flaw size |
| [02_solutions.md](02_solutions.md) | Stress intensity factor and reference stress solutions: plates (surface, through-thickness, embedded, extended flaws) and cylinders (the two K routes, bulging, tabulated factors, reference stress branches); validity ranges and refusal |
| [03_materials_toughness.md](03_materials_toughness.md) | Fracture toughness (Master Curve, Charpy correlations, J and CTOD conversions, lower-bound statistics) and the tensile laws used by Option 2 and by the FE pipeline |
| [04_probabilistic.md](04_probabilistic.md) | Random variables, Gaussian-copula (Nataf) dependence, Monte Carlo, subset simulation, importance sampling, FORM/SORM, intervals and result statuses, failure-region diagnostics, family sensitivity, partial factors |
| [05_fatigue.md](05_fatigue.md) | Fatigue crack growth laws and thresholds, the integrator and its events, spectra, extreme loads, the three time-dependent probabilities, inspection and repair, updating from an inspection outcome |
| [06_fe_option3.md](06_fe_option3.md) | The WARP3D finite element pipeline, its QA layers and verification evidence; extraction of the Option 3 curve; the experimental Gaussian-process emulator of the deepest-point curve and its validation |
| [07_pf_emulator.md](07_pf_emulator.md) | Failure probability with the emulator: classification of a realisation, between-step bounds, function draws, the `Pf` bound pair, the crossed variance decomposition, acquisition and stopping, the V10 synthetic test |
| [08_knowledge_base.md](08_knowledge_base.md) | The K-solution knowledge base: source classes, the status ladder, status from linked evidence, FE evidence, the release audit |
| [references.md](references.md) | Bibliography |

## 1. Conventions

### 1.1 Units

All internal quantities are SI floats: lengths in m, stresses and moduli in Pa, stress intensity factors in
Pa√m, the J-integral in J/m² (= N/m), energies in J, time in years for time-dependent results, temperatures in °C
(the Master Curve and the Charpy correlations are written in °C differences). Correlations whose sources are in
other units (MPa√m, mm, J, N/mm^{3/2}, mm/cycle) are converted inside the function that implements them; no
conversion is left to the user. The FE deck is written in N, mm and MPa and every parsed quantity is converted back
([06_fe_option3.md](06_fe_option3.md) §6.1.6).

### 1.2 Arrays and immutability

Every public numerical function accepts NumPy arrays and broadcasts its arguments; one array element is one
realisation (sample). Problem objects (`FlatPlate`, `Cylinder`, flaws, `Material`, `StressState`, `Assessment`,
`FECase`) are frozen dataclasses; a variant is made with `.with_(...)`, never by mutation. A field may hold a scalar,
an array, or a random variable (a `Distribution`); a problem holding random variables is assessed through
`ProbabilisticAssessment` ([04_probabilistic.md](04_probabilistic.md)).

### 1.3 Three rules that run through the whole library

1. **No extrapolation.** A solution is evaluated only inside its stated validity range. Outside it the realisation
   is marked out of validity; it is never clipped, dropped or resampled ([02_solutions.md](02_solutions.md) §2.1).
2. **Invalid mass is reported, not hidden.** In probabilistic results the probability of out-of-validity
   realisations is reported and the failure probability is given as a bound pair (that mass counted as failed, and
   as safe); the headline counts it as failed ([04_probabilistic.md](04_probabilistic.md) §4.6).
3. **Acceptance is the direct test.** A point is acceptable iff `Kr < f(Lr)` and `Lr < Lr,max`. The radial margin
   is reported but never decides acceptance ([01_fad.md](01_fad.md) §1.6).

Unknown values are never invented: a missing equation or coefficient raises
`NotImplementedError("TODO(verify): ...")`.

### 1.4 Mathematical notation

Equations are written in LaTeX notation. `log` without a base is the natural logarithm; `Φ` is the standard
normal distribution function and `φ` its density.

## 2. Notation

| Symbol | Meaning | Unit |
|---|---|---|
| $B$ | plate or wall thickness | m |
| $W$ | plate width; for cylinders the length used in the finite-width rule | m |
| $a$ | surface-flaw depth; half-length of a through-thickness flaw; half-height of an embedded flaw | m |
| $2c$ | surface or embedded flaw length (so $c$ is the half-length) | m |
| $p$ | ligament of an embedded flaw to the nearer surface | m |
| $B'$ | effective thickness of an embedded flaw, $2a + 2p$ | m |
| $r_i, r_o, r_m$ | inner, outer and mean radius of a cylinder, $r_m = r_i + B/2$ | m |
| $\theta$ | parametric angle on an elliptical crack front ($\pi/2$ deepest point, $0$ surface) | rad |
| $P_m, P_b$ | primary membrane and bending stress | Pa |
| $Q_m, Q_b$ | secondary membrane and bending stress | Pa |
| $\sigma_Y, \sigma_U$ | yield (or 0.2 % proof) strength, tensile strength | Pa |
| $\sigma_f$ | flow strength $(\sigma_Y + \sigma_U)/2$ | Pa |
| $E, \nu$ | Young's modulus, Poisson's ratio | Pa, – |
| $\sigma_{ref}$ | reference stress (from the limit load) | Pa |
| $K_I^p, K_I^s$ | primary and secondary mode-I stress intensity factors | Pa√m |
| $K_{mat}$ | fracture toughness | Pa√m |
| $L_r$ | load ratio $\sigma_{ref}/\sigma_Y$ | – |
| $K_r$ | fracture ratio | – |
| $f(L_r)$ | assessment line (FAD curve) | – |
| $L_{r,max}$ | plastic collapse cut-off | – |
| $\rho, V$ | additive / multiplicative plasticity interaction for secondary stress | – |
| $\chi$ | $K_I^s L_r / K_I^p$ | – |
| $F$ | radial margin (first exit along the ray from the origin) | – |
| $M_m, M_b$ | membrane and bending magnification factors | – |
| $f_w$ | finite-width correction | – |
| $M$ | bulging factor (cylinders) | – |
| $u$ | independent standard normal vector | – |
| $P_f$ | failure probability | – |
| $\beta$ | reliability index (FORM); in the FE chapters, the loading mix $P_b/(P_m+P_b)$ | – |
| $\Delta K, \Delta K_0$ | stress intensity factor range, growth threshold | Pa√m |
| $A, m$ | growth-law coefficient and exponent | SI |
| $J, J_e$ | J-integral; its elastic value | J/m² |
| $\sigma_0, \varepsilon_0, n$ | reference stress, reference strain $\sigma_0/E$, hardening exponent of the WARP3D deformation law | Pa, –, – |
| $x$ | emulator input $(a/B,\ a/c,\ n,\ \sigma_0/E,\ \beta)$ | – |
| $y_k$ | emulated log increment of $J/J_e$ at FE load step $k$ | – |

## 3. Label and status glossary

Labels are part of every result. They say what a number is, what it assumes, and how far it has been checked.

### 3.1 Deterministic assessment ([01_fad.md](01_fad.md))

| Label | Meaning |
|---|---|
| `valid` (per sample) | every solution used was inside its validity range |
| out of validity (`valid = False`) | some solution was outside its range; all outputs of that sample are blanked (NaN) and `acceptable` is False; `run(on_invalid="raise")` raises instead |
| exit `curve` / `step` / `cutoff` | where the ray from the origin leaves the acceptable region: through the assessment line, a vertical step of it, or `Lr = Lr,max` |
| exit `domain_end` | a tabulated Option 2 law ends before the collapse cut-off; its last point acts as the cut-off |
| exit `none` | the point is the origin ($F = +\infty$) |
| exit `invalid` | the sample is out of validity |
| governing position | the crack-front position with the smallest radial margin (ties go to the larger `Kr`); acceptance itself requires every position to pass |
| flags | named conditions such as `negative_primary_k_set_to_zero`, `secondary_stress_acting_alone`, `chi_above_4_simplified_interaction_over_conservative`, `residual_stress_relaxation_applied`, `embedded_recharacterisation_relaxation_not_applied`, `tabulated_factor_interpolated_through_a_suspect_cell_samples_present` |
| excluded uncertainty | the trace lists the uncertainty sources the result does not account for (model uncertainty of K, `σref` and the line unless model factors were sampled; input scatter in a deterministic run) |

### 3.2 Probabilistic results ([04_probabilistic.md](04_probabilistic.md))

| Label | Meaning |
|---|---|
| guaranteed interval | two-sided 99 % Clopper–Pearson interval from plain Monte Carlo of the nominal distribution; assumes nothing about the limit state |
| `resolved` | the Monte Carlo channel saw at least 10 failures; its estimate is the headline |
| `conditional` | a subset-simulation estimate inside the guaranteed interval; valid only if no failure region was missed; shown with the guaranteed upper bound |
| `contradicted` | the rare-event estimate lies outside the guaranteed interval and is rejected |
| `plateau` | more than half of a subset level shares the threshold value; no subset estimate |
| `level cap reached; unresolved` | subset simulation reached 20 levels without terminating; no estimate |
| `replicates inconsistent` | a replicate was zero or the replicate coefficient of variation exceeded 1; no subset estimate |
| approximate | the subset-simulation replicate interval (finite-replicate, delta-method) |
| per flaw | the probability is conditional on the flaw existing |
| association | the variable-ranking measure; association with failure, not causation |

### 3.3 Fatigue and time-dependent results ([05_fatigue.md](05_fatigue.md))

| Label | Meaning |
|---|---|
| `growing`, `dormant` | a flaw grows, or every block at every position is below threshold (it is still assessed for fracture) |
| `leak` | a part-wall flaw reached the far surface |
| `leak_assumed` | a growing flaw reached its solution's validity limit before breakthrough and was re-characterised; trigger-coded; its mass is reported separately |
| `surface_breaking` | an embedded flaw broke through to the nearer surface (never a failure) |
| `ended: outside validity` | no continuation solution exists; from then on the sample is counted failed in the headline and safe in the valid-only bound |
| upper / lower pairing | the step's extreme load paired with the most / least severe checkpoint of the step |
| bracket under assumption A | the pairings bound the step failure probability only if severity is monotone between checkpoints (checked at checkpoints, not proved) |
| conservative pair | the bracket for survival-conditioned quantities (hazard, inspection update) |
| exact for the [upper / lower] pairing's discretised model | within-pairing values; a sensitivity, not a bracket |

### 3.4 FE results ([06_fe_option3.md](06_fe_option3.md))

| Label | Meaning |
|---|---|
| run `status` | `complete`, `truncated` (a step did not converge), `failed` (unparsable output), `timeout` |
| `adaptive` | a step completed through WARP3D's adaptive sub-increments; converged |
| `verification_status` | `fe-failed` (a QA check failed, named), `unverified` (all checks passed but the family verification is incomplete), `fe-verified` (never issued at present: `FAMILY_VERIFIED = False`) |
| `outside_small_strain` | the step and all later steps exceeded the local strain cap; stored, never used for a curve |
| "unresolved: mesh-sensitive" | the near-surface evaluation angle `phi_s`: its convergence could not be demonstrated; excluded from verified or training use |

### 3.5 Option 3 emulator and Pf ([06_fe_option3.md](06_fe_option3.md), [07_pf_emulator.md](07_pf_emulator.md))

| Label | Meaning |
|---|---|
| `assessment_grade = False`, `unverified` | every emulator record and prediction; assessment and Pf use require `experimental=True` |
| experimental | the label carried by any curve or Pf obtained with `experimental=True` |
| `supported` | the input is in the box and `Lr` lies inside the curve's support |
| `outside_box`, `invalid` | the input lies outside the trained box / is non-finite |
| `beyond_support` | `Lr` exceeds the support given by the schedule rule |
| `cap_uncertain` | `Lr` exceeds the support because the small-strain cap rule cut it |
| `between_steps` | the realisation lies between the curve's step bounds and is not decided |
| `Pf_high`, `Pf_low` | undecided mass counted as failed / as safe; `Pf_high` is the headline |
| surrogate-only interval | the 90 % interval of `Pf_high` over function draws; excludes model uncertainty, distribution-parameter uncertainty, FE-versus-reality and GP hyperparameter uncertainty |
| between-step bounds `monotone` / `convex` | rigorous bracket / bracket tightened under the stated convexity assumption |

### 3.6 Knowledge base ([08_knowledge_base.md](08_knowledge_base.md))

| Label | Meaning |
|---|---|
| `unverified` | no test-linked oracle evidence |
| `self-consistent` | only checks that cannot catch an error in the source text (independent re-transcription, independent re-implementation, dimensions) |
| `cross-checked` | a definition-pinning limiting case or published values of the primary reference, linked to named tests |
| `fe-verified` | region-scoped FE evidence from a verified FE family that was not itself validated against the solution |
| `standard-derived`, `open`, `handbook` | source class: transcribed from the licensed standard (used locally, not shipped); from and checked against open literature; user-loaded |
| `self_consistent_only` | register flag: no kind (a) or kind (b) oracle is claimed |
| `checked_against_primary` | `yes` / `no` / `unobtainable`: whether coefficients were compared with the primary publication; `no` means "as cited by the source standard" |
| `suspect` (table cell) | a tabulated value flagged as a printed anomaly; an evaluation that interpolates through it is `unverified` for that evaluation |

## 4. Oracle kinds

The verification vocabulary used throughout ([08_knowledge_base.md](08_knowledge_base.md) §8.1):

- (a) published values of the primary reference (the source standard's own worked examples do not count);
- (b) analytic limiting cases and invariants, including at least one definition-pinning check per geometric
  argument (so that `c` versus `2c`, `a` versus `2a`, `W` versus `W/2` errors cannot pass);
- (c) a second, independent transcription of every equation and coefficient from the printed page, diffed against
  the first;
- (d) an independent reference implementation written from the transcribed equations without sight of the code;
- (e) dimensional consistency.

Kinds (c), (d) and (e) cannot detect a coefficient that is wrong in the printed source itself. No end-to-end
deterministic oracle exists for the FAD engine.

## 5. What the library does not model

Load-sequence effects on the growth rate (overload retardation), crack closure, creep, stress corrosion, wall
loss, flaw interaction, initiation life, arbitrary through-wall stress profiles (there is no input for them; only
membrane plus bending), embedded flaws in cylinders (raise `TODO(verify)`), bending-loaded embedded flaws
(refused), the optional ligament re-characterisation relaxation of a failed embedded-flaw assessment, and Option 3
curves for anything other than the deepest point of a semi-elliptical surface flaw in a plate under proportional
tension and bending. Each is stated where it matters in the chapters.
