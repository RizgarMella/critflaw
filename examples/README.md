# critflaw worked examples

> **⚠️ Research software in development: not verified, not validated, not for real-world use** (`../DISCLAIMER.md`).

**Start here:** `showcase.ipynb` is a one-notebook tour told through the failure assessment diagram (FAD, a
14 400-point acceptability map, margins, probability-density fields, an uncertain line, fatigue trajectories, a
population marching to the line, life-time Pf with inspection, the audit trace), stored with its outputs and
figures. Rebuild it with `uv run --with nbclient --with ipykernel python tools/build_showcase.py`.

Each example is one Python file in [jupytext](https://jupytext.readthedocs.io) "percent" format: it runs as a
script and converts to a notebook. All inputs are illustrative SI values (Pa, m, Pa m^0.5, J/m^2), chosen for the
examples and not taken from any standard's worked examples. Every example ends with assertions on sane outcomes,
so running them in CI catches API drift.

```bash
# run one example as a script (headless; figures go to examples/_output/)
uv run python examples/01_option1_plate_surface_flaw.py

# run them all, one at a time, with pass/fail and wall time
uv run python tools/run_examples.py

# regenerate the notebooks in examples/notebooks/ (no extra dependency; --check verifies they are current)
uv run python tools/make_notebooks.py
```

Plots need the optional `plot` extra (`matplotlib`); without it the examples still run and skip the figures.
Figures are saved to `examples/_output/` (git-ignored) and also shown inline in notebooks.

## Index

Runtime classes as measured on a 15 GB Windows workstation: **fast** < 5 s, **moderate** 5-30 s, **slow** > 30 s.

| File | Topic | Runtime | Features used | Theory chapter |
|---|---|---|---|---|
| `00_end_to_end.py` | One flaw through the whole chain: FAD check, reserve factors, limiting size, Pf, fatigue growth, time-dependent Pf with inspection | slow | `Assessment`, reserve factors, `ProbabilisticAssessment.solve`, `TimeDependentAssessment.grow/solve`, `Inspection` | `01_fad`, `04_probabilistic`, `05_fatigue` |
| `01_option1_plate_surface_flaw.py` | Option 1 FAD check of a plate surface flaw; direct acceptance test; vectorised load and size sweeps | fast | `Assessment`, `Option1Curve`, `Result` fields, `on_invalid="mask"` | `01_fad` |
| `02_option2_with_stress_strain_curve.py` | Option 2 lines from a power law and a (synthetic) measured curve | fast | `PowerLaw`, `TabulatedLaw.from_engineering`, `Option2Curve`, `fad_option=2` | `01_fad`, `03_materials_toughness` |
| `03_secondary_stress_and_rho.py` | Secondary and residual stress, relaxation, rho vs V interaction | fast | `Residual.as_welded/pwht`, `interaction`, flags | `01_fad` |
| `04_reserve_factors_and_limiting_flaw.py` | Reserve factors on load and toughness, limiting flaw size | fast | `reserve_factor_on_load`, `reserve_factor_on_toughness`, `limiting_flaw_size`, `Crossing` | `01_fad` |
| `05_reading_the_trace.py` | The result trace: equations, status, options, flags, exclusions; masking | fast | `Trace.report/dependencies`, `OutOfValidityError`, model factors | `01_fad`, `08_knowledge_base` |
| `06_cylinder_flaws.py` | Cylinder flaws: K routes, pressure record, thick-wall reference stress, through and extended flaws | fast | `Cylinder`, `PressureLoad`, `CylinderSurfaceFlaw`, `CylinderThroughFlaw`, `ExtendedFlaw` | `02_solutions` |
| `07_embedded_and_through_flaws.py` | Embedded and through-thickness plate flaws; refusals; random ligament validity share | fast | `EmbeddedFlaw`, `from_ligaments`, `ThroughFlaw`, `ProbabilisticAssessment.evaluate` | `02_solutions` |
| `08_toughness_master_curve_and_charpy.py` | Toughness conversions, Master Curve, Charpy correlations, characteristic values; Master Curve in a probabilistic run | moderate | `critflaw.toughness`, `MasterCurve`, `ProbabilisticAssessment.solve` | `03_materials_toughness` |
| `09_monte_carlo_vs_form_sorm.py` | FORM and SORM against Monte Carlo on a smooth LEFM limit state | fast | `prob.form`, `prob.sorm_breitung`, `prob.solvers.monte_carlo`, `RandomVector` | `04_probabilistic` |
| `10_importance_and_subset_simulation.py` | Rare events: guaranteed MC interval, conditional subset estimate, importance sampling, failure points, association | moderate | `solve`, `most_likely_failure_points`, `importance_sampling`, `association` | `04_probabilistic` |
| `11_dependence_nataf.py` | Gaussian-copula (Nataf) dependence: attainable range, Pearson vs latent, effect on Pf | moderate | `attainable_range`, `pearson_to_latent`, `correlate` | `04_probabilistic` |
| `12_partial_factors.py` | Published partial factors, design check, achieved Pf, implied factors | moderate | `published_partial_factors`, `implied_partial_factors` | `04_probabilistic` |
| `13_fatigue_crack_growth.py` | Deterministic fatigue growth of a surface flaw through breakthrough | fast | `steel_in_air`, `simplified_law`, `Spectrum`, `TimeDependentAssessment.grow` | `05_fatigue` |
| `14_time_dependent_pf_and_inspection.py` | Time-dependent Pf bracket, inspection with repair, updated flaw-size distribution | slow | `TimeDependentAssessment.solve`, `Inspection`, `LogLogisticPoD`, `undetected` | `05_fatigue`, `04_probabilistic` |
| `15_fe_warp3d_case.py` | One WARP3D surface-flaw case and its Option 3 curve (**guarded**, **experimental**) | fast (stored) / minutes (live, opt-in) | `critflaw.fe`, `WarpJSolver`, `option3.curve.from_summary`, `Option2Curve.from_warp_law` | `06_fe_option3` |
| `16_option3_emulator_synthetic.py` | The Option 3 curve emulator on a synthetic family: train, predict, validate, guard (**experimental**) | moderate | `option3.emulator.train/predict`, `option3.validation.validate`, `assessment_curve(experimental=True)` | `06_fe_option3` |
| `17_pf_with_emulator_synthetic.py` | Pf bound pair with emulator function draws and crossed ANOVA (**experimental**) | slow | `function_draws`, `option3.reliability.population`, `pf_crossed(experimental=True)` | `07_pf_emulator` |
| `18_knowledge_base_and_status.py` | Verification status from linked evidence, `strict=True`, coverage map, user solutions, release audit | fast | `critflaw.kb.core`, `Trace.below_cross_checked`, `load_user`, `release_audit` | `08_knowledge_base` |
| `19_fad_gallery.py` | FAD graphics: lines and load ray, paths, probability-density field with HDR contours, exceedance field of an uncertain line, fatigue trajectories, a population marching to the line | moderate | `critflaw.plot` (`fad`, `point`, `path`, `density`, `exceedance_field`, `small_multiples`) | `01_fad`, `04_probabilistic`, `05_fatigue` |

Theory chapters are `docs/manual/theory/<chapter>.md`.

## Notes on scope (what the examples deliberately do not do)

* **FORM/SORM are not applied to FAD problems.** The library offers them only for smooth, user-supplied limit
  states (the FAD failure set is a union of guarded, non-smooth events). Example 09 therefore uses a textbook
  linear-elastic limit state; FAD problems use Monte Carlo and subset simulation (examples 10-12).
* **WARP3D is not run by default.** Example 15 runs a case only when Docker with the image
  `critflaw-warp3d:spike` is available **and** `CRITFLAW_RUN_WARP3D=1` is set (a case takes minutes and several
  GB). Otherwise it loads a stored result of the FE training campaign from `fe/runs_p5/results/`; if none is present it prints how to
  produce one and stops cleanly.
* **Experimental features are labelled.** The Option 3 emulator and its failure probability (examples 15-17) are
  unverified and not assessment-grade; the API refuses them unless `experimental=True` is passed, and the examples
  print the labels.
* **Embedded flaws under bending** are refused by the library (`NotImplementedError`, a documented gap);
  example 07 shows the refusal rather than working around it.
* Monte Carlo sample sizes are kept small (at most a few hundred thousand cheap evaluations) so that each example
  stays within about a minute; production analyses need more samples, and the printed intervals say how much the
  small samples cost in precision.
