# 10. Troubleshooting

The library refuses rather than guessing. Most errors you meet mean "this input is outside what is defined" or
"you must declare this explicitly". The tables below are grouped by where the error appears. Message texts are
quoted from the library, abbreviated where marked with "...".

## Building problem objects

| Error | Cause | What to do |
|---|---|---|
| `ValueError: <field> must be finite and positive` | a length, strength, modulus or toughness is zero, negative or NaN | Check units (all SI) and signs. |
| `TypeError: <field> must be a number, an array of numbers or a distribution` | a string or a boolean was passed | |
| `ValueError: <field> must be positive but its distribution has support at or below zero; write the truncation explicitly, ...` | an unbounded distribution (for example `Normal`) on a positive quantity | Use `prob.Truncated(dist, lo=0.0)`, or a positive family. Nothing is truncated for you. |
| `ValueError: primary stresses must be non-negative (tensile); ...` | negative `pm` or `pb`, or a primary-stress distribution with support below zero | Truncate at zero. Compressive primary stress is not assessed. |
| `ValueError: uts must exceed yield_strength` | fixed values in the wrong order | |
| `ValueError: these yield and tensile strength distributions can give tensile <= yield; ...` | random strengths whose supports overlap | Give the tensile strength as `cf.Ratio(dist)` with the support of `dist` above 1. |
| `ValueError: the support of a tensile-to-yield ratio must lie above 1` | the `Ratio` distribution reaches 1 | Truncate it, for example `Truncated(..., lo=1.02)`. |
| `ValueError: yielding must be 'continuous', 'discontinuous' or 'unknown'` | `yielding` missing or misspelt | It is required, with no default. |
| `ValueError: the Lueders strain estimate applies only below ... MPa yield strength; supply luders_strain` | discontinuous or unknown yielding at high strength | Pass `luders_strain=...`. |
| `ValueError: luders_strain has no meaning for continuous yielding` | | Remove it, or change `yielding`. |
| `ValueError: fad_option=2 needs a material law (PowerLaw or TabulatedLaw)` / `for Option 2 'yielding' must be 'discontinuous' if the law has a yield plateau ...` | Option 2 set up inconsistently | Give `law=`, and match `yielding` to whether the law has a plateau. |
| `ValueError: a material law is only used with fad_option=2` | `law` given with Option 1 | |
| `NotImplementedError: FAD Options 1 and 2 are implemented` | `fad_option=3` | Option 3 curves come from the experimental emulator ([chapter 8](08_option3_emulator.md)). |
| `ValueError: k_route must be 'pressure_only' or 'tabulated' (no default: ...)` | `Cylinder` without a K route | Choose one ([chapter 6](06_cylinders_embedded.md)). |
| `ValueError: relaxation with primary load is defined for the as-welded condition only` | `Residual(fraction<1, relaxation=True)` | |
| `StressStrainDataError: ...` | measured stress-strain data that cannot be cleaned within the limits: not monotone within 5 % of yield, plateau not flat, modulus too low, no elastic range, or not extending beyond yield | Check the data, the units, the declared plateau and the modulus. `law.notes` lists what was cleaned. The class is not exported at the top level: `from critflaw.fad.option2 import StressStrainDataError` (it subclasses `ValueError`, so `except ValueError` also catches it). |

## Running an assessment

| Error | Cause | What to do |
|---|---|---|
| `OutOfValidityError: N of M evaluation(s) are outside the validity range of the stress intensity factor solution; solutions are not extrapolated` | some sample is outside a solution's range: a flaw too deep or too long, a far ligament declared as `p`, a cylinder outside the tables, or a cylinder route that refuses the load | Fix the input. Or use `run(on_invalid="mask")` and account for the masked samples (`valid` is `False`). |
| `OutOfValidityError: the stress-strain law has not strained plastically beyond Lr = 1.41 ...` | Option 2 law too elastic for the line to be defined | Supply data or a law that extends into plasticity. |
| `OutOfValidityError: the monotonicity rule lowers this tabulated Option 2 line by ..., more than 0.05: the stress-strain data are suspect` | | Check the measured curve. |
| `TypeError: the problem contains random variables (...); assess it with critflaw.ProbabilisticAssessment` | `Assessment.run()` on a problem with distributions | Use `ProbabilisticAssessment` ([chapter 4](04_probabilistic.md)). |
| `TypeError: cylinder flaws need a Cylinder component and plate flaws a FlatPlate` | mismatched component and flaw | |
| `NotImplementedError: TODO(verify): M_b at the major-axis ends of an embedded flaw ...` | an embedded flaw with non-zero `pb` or `qb` | Not available: the source gives no factor. |
| `NotImplementedError: TODO(verify): embedded flaws in cylinders ...` | | Not available. |
| `ValueError: sref_route 'thick_pressure' applies to axial surface flaws with a PressureLoad` | wrong flaw or no `PressureLoad` | |
| `ValueError: the thick-wall membrane-and-bending limit is given for external axial flaws only` | `sref_route="thick"` with an internal axial flaw | |
| `ValueError: strict assessment refused: equations below 'cross-checked' in the knowledge base: ...` | `run(strict=True)` | Expected today for every assessment ([chapter 3](03_deterministic.md#strict-mode)). Run without `strict` and read the trace. |
| `ValueError: on_invalid must be 'raise' or 'mask'` | | |
| `ValueError: reserve factors and limiting flaw size take a scalar assessment (one sample)` | arrays passed to a reserve function | Loop over cases, or index one sample. |
| `ValueError: Lr must be non-negative and not NaN` | evaluating a curve at a bad `Lr` | |

Remember that `NotImplementedError("TODO(verify): ...")` always means the equation or coefficient is not
available from the source. It is never a temporary bug to work around with a guess.

## Trace flags

Flags in `result.trace.flags` record conditions met during a run. They are information, not errors.

| Flag | Meaning |
|---|---|
| `out_of_validity_samples_present` | masked samples exist (`on_invalid="mask"`) |
| `negative_primary_k_set_to_zero` | the primary K was negative at a position (deep flaw in bending) and was assessed as zero (conservative) |
| `secondary_stress_acting_alone` | no primary stress: the plastic-zone-corrected secondary K was used |
| `chi_above_4_simplified_interaction_over_conservative` | the simplified plasticity interaction is outside its accurate range (conservative) |
| `residual_stress_relaxation_applied` | the as-welded residual stress was relaxed by the primary load |
| `option2_law_too_elastic_beyond_root_two_samples_present`, `option2_tabulated_correction_above_limit_samples_present` | masked Option 2 samples, for the reasons above |
| `stress_strain_data_end_before_collapse_cut_off` | a tabulated law ends before the collapse cut-off: the line ends there (`exit = "domain_end"`) |
| `embedded_recharacterisation_relaxation_not_applied` | present on every embedded-flaw result |
| `tabulated_factor_interpolated_through_a_suspect_cell_samples_present` | a cylinder table cell flagged as suspect was used. That factor is `unverified` in this result. |
| `pressure_only_route_refuses_mechanical_primary_stress` | some realisations had `pm`, `pb` or `pm_moment` under the pressure-only route. They are out of validity. |
| `thick_pressure_reference_stress_refuses_mechanical_primary_stress` | the same, for the thick-wall limit-pressure route |
| `thick_circumferential_limit_load_refuses_through_wall_bending` | `pb` with the thick-wall circumferential limit load. Those realisations are out of validity. |

## Probabilistic assessment

| Error | Cause | What to do |
|---|---|---|
| `ValueError: the problem has no random variables; use Assessment.run()` | | |
| `ValueError: a sampled realisation violates a physical constraint: ... realisations are never dropped or clipped.` | a sample broke a rule such as `uts > yield` | Constrain the distributions: truncation, or `Ratio` for the tensile strength. |
| `CorrelationError: Pearson correlation ... is not attainable for these marginals; the attainable range is [...]` | | Use a value in range, or `kind="latent"`. |
| `CorrelationError: the declared correlations are not jointly attainable: ...` | the latent correlation matrix is not positive definite | Revise the set of correlations. |
| `CorrelationError: discrete marginals can only be correlated with kind='latent'` | | |
| `CorrelationError: cannot evaluate the correlation of these marginals to 1e-8 by quadrature ...` | very heavy tails | Use `kind="latent"`. |
| `ValueError: variable names are not unique (...); name the distributions` | `implied_partial_factors` with duplicate or default names | Pass `name=` to each distribution. |
| `ValueError: samples_per_level * p0 must be an integer of at least 2 that divides samples_per_level` | subset-simulation settings | |
| `FormNotApplicable: ...` | FORM or SORM assumptions fail (origin in the failure region, no convergence, strong curvature) | Use Monte Carlo or subset simulation. FORM is not applied to FAD problems anyway. |
| `NotTabulated: ...` | a published partial factor that is not in the table (no interpolation), or marked not permitted | Use a tabulated target and coefficient of variation. |
| `ValueError: the conditioned distribution lies about 35.5 or more standard normal scores into a tail of the prior ...` | an inspection observation far outside the prior | The prior does not describe this flaw. Revise it. |
| `RuntimeError: driving variable and failure indicator disagree beyond rounding` | an internal consistency check | Report it as a bug, with the inputs. |

A result `status` other than `resolved` or `conditional` is not an error. It means only the guaranteed
interval is available ([chapter 4](04_probabilistic.md#reading-the-result)).

## Fatigue and time-dependent runs

| Error | Cause | What to do |
|---|---|---|
| `ValueError: leak_is_failure must be given explicitly (True or False)` | | It is required. |
| `ValueError: every reporting time must be a multiple of the load-process step` / `inspections must be at load-process step boundaries` | | Align `times` and inspections with `delta`. |
| `ValueError: the blocks of an ordered spectrum must sum to the last requested time` | | |
| `ValueError: shared_scale (True or False) is required when the load scale is random` | | |
| `ValueError: an ordered spectrum needs order='as_given' \| 'largest_first' \| 'smallest_first' (no default)` | | |
| `ValueError: a two-stage law with random coefficients needs stage_correlation (no default)` | | |
| `ValueError: life_factor is for deterministic runs only ...` | | Use `life_factor` only with `grow()` and `compare_orders()`. |
| `ValueError: subset simulation in time is not offered when leak is failure ...` | | |
| `ValueError: the growth within one pass of the repeating spectrum exceeds eta / 4 ...` | a repeating pass is too long for the averaged rate | Use `Spectrum.ordered` or shorter passes. |
| `ValueError: an inspection of embedded flaws needs pod_embedded ...` | | |
| `ValueError: grow() is for deterministic problems; ...` | | Use `solve()`. |
| `NotImplementedError: TODO(verify): fatigue growth is implemented for plate and cylinder flaws` | a plate `ThroughFlaw` as the initial flaw | |
| `NotImplementedError: TODO(verify): effective-range growth near the threshold ...` | | Not available. |
| `RuntimeError: the growth integration reached the minimum step ...` / `did not reach the requested time (iteration cap)` / `more than ... flaw-type changes in one segment ...` | the integrator could not resolve a sample | Check for extreme inputs (very high ranges, tiny thresholds). Report it if the inputs are reasonable. |

## FE back end

| Symptom | Cause | What to do |
|---|---|---|
| `FEResult.status == "failed"`, diagnostic `image resolution failed: docker image 'critflaw-warp3d:spike' not found` | image not built, or a different tag | Build it ([chapter 1](01_install.md#building-the-image)), or pass `WarpRunner(image=...)`. |
| diagnostic `docker launch failed: ...` | Docker not installed or not running | Start Docker. |
| `status == "timeout"` | wall-clock limit reached | Raise `timeout_seconds`, or reduce load on the host. |
| `status == "failed"` after a memory kill | the container or host ran out of memory | Lower the thread count, keep the memory ceiling within the host, and run one case at a time. |
| `status == "truncated"` | a load step did not converge | Usually high `Lr` with low hardening. The steps before it are recorded. |
| `verification_status == "fe-failed"` with `failing_check` | a QA check failed | See [QA checks](07_fe_backend.md#qa-checks-and-what-a-failure-means). Do not use the case. |
| `verification_status == "unverified"` | every check passed | Expected: the FE family verification is incomplete. |
| `ValueError: <name> = ... outside the declared box [...]` | `FECase` input outside the parameter box | |
| `ValueError: L_far = ... m is below the converged remote length ...` | | Use `critflaw.fe.crack_mesh.remote_length(B, W)`. |
| `FileExistsError: ... exists and is not a runner job directory` | the runner will not overwrite a directory it did not create | Choose a new job directory. |
| `SolveFailed: ...` | from `plastic_case.run_case` or `elastic_case.run_case`: the solve did not complete, or the run QA or evaluation failed | `exc.record` holds the run record and its diagnostic. |

## Option 3 emulator

| Error | Cause | What to do |
|---|---|---|
| `NotAssessmentGrade: the Option 3 emulator is experimental ...` | `assessment_curve` without `experimental=True` | Pass `experimental=True` and report the result as experimental. |
| `PermissionError: the Pf uses the experimental Option 3 emulator: pass experimental=True` | `reliability.pf_crossed` without the flag | Pass `experimental=True` and report the result as experimental. |
| `PermissionError: active learning uses the experimental Option 3 emulator: pass experimental=True` | `active.run` without the flag | |
| `ValueError: no Option 3 curve: the realisation is outside_box` (or another status) | the prediction is not `supported` | Report the realisation as undecided. |
| `CurveQAError: ...` | an FE summary that is not complete and QA-passing, has no `"design"` inputs, or has a decreasing `J/J_e` | Exclude the case, and record why. |
| prediction `status` `outside_box`, `cap_uncertain` or `beyond_support` | not errors: the emulator does not support that input or `Lr` | These count as undecided in Pf (`undecided_mass`). |
