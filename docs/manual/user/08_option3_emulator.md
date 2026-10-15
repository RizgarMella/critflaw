# 8. The Option 3 emulator (experimental)

> **Everything in this chapter is experimental.**
>
> - The emulator is trained on FE results that are `unverified` ([chapter 7](07_fe_backend.md#verification-status)).
> - It models the **deepest point only**. The surface position is unresolved because it is mesh-sensitive.
> - Every curve and prediction carries `verification_status = "unverified"` and `assessment_grade = False`.
>
> The library refuses to use the emulator in an assessment or a failure probability unless you pass
> `experimental=True`. Report any result obtained this way as experimental.

`critflaw.option3` builds Option 3 failure assessment curves (`Kr` against `Lr` from J analyses) for a plate
surface flaw. It learns them with a Gaussian-process emulator over five inputs, so that a curve can be
predicted, with its uncertainty, for inputs where no FE case was run.

| Index | Input | Box |
|---|---|---|
| `x[0]` | `a/B` | [0.2, 0.6] |
| `x[1]` | `a/c` | [0.4, 1.0] |
| `x[2]` | `n` (hardening exponent of the `deformation` law) | [3, 20] |
| `x[3]` | `sigma_0 / E` | [1e-3, 5e-3] |
| `x[4]` | `beta = Pb / (Pm + Pb)` | [0, 1] |

The box is `critflaw.option3.design.LOWER` / `UPPER`. `Lr` is referred to the law's reference stress
`sigma_0`, **not** to a 0.2 % proof stress.

Theory: [FE and Option 3](../theory/06_fe_option3.md), [Pf with the emulator](../theory/07_pf_emulator.md).
Examples: *training the Option 3 emulator*, *Option 3 curve for an assessment*, *Pf with the emulator*.

```python
import json
from pathlib import Path
import numpy as np
from critflaw.option3 import curve as C, emulator as EM, reliability as R, active
from critflaw.option3.design import LOWER, UPPER, schedule_lr
from critflaw.fe.fecase import LEVELS
```

## From a proof stress to `sigma_0`

```python
from critflaw.option3.interface import sigma0_from_proof_stress
s0 = sigma0_from_proof_stress(rp02=400e6, modulus=207e9, n=10.0)     # vectorised
x = np.array([0.4, 0.7, 10.0, float(s0) / 207e9, 0.3])
```

If you use `Rp0.2` in place of `sigma_0`, the `Lr` axis shifts by several to tens of per cent over the box.

## Design

`critflaw.option3.design` generates the FE designs: scrambled Sobol' points in the box, with fixed seeds.

| Function | Gives |
|---|---|
| `training_design(n_train=48, n_pairs=12)` | `DesignPoint`s at mesh level 1, plus `n_pairs` of them repeated at level 2. The level pairs estimate the discretisation error. |
| `heldout_design(n=12)` | independent level-1 points for validation |
| `DesignPoint.case(c_over_w=0.05)` | the `FECase` for a point (B = 20 mm, E = 207 GPa, ν = 0.3, converged remote length) |
| `schedule_lr(x)` | `Lr` of the 12 FE load steps for `x` (the closed-form schedule) |

Each design point is run with `WarpJSolver` ([chapter 7](07_fe_backend.md)), one at a time. Save
`result_summary(res)` with the design inputs added under `"design"`, because the curve extraction needs them:

```python
from critflaw.fe.fecase import WarpJSolver, result_summary
from critflaw.option3.design import training_design

for p in training_design():
    res = solver.run(p.case())                          # solver: a WarpJSolver (chapter 7)
    summary = result_summary(res)
    summary["design"] = {"role": p.role, "x": p.x.tolist()}
    Path(f"results/{p.role}_L{p.level[0]}_{res.case.case_id}.json").write_text(json.dumps(summary))
```

The repository script `benchmarks/option3_campaign.py` does this for the full training and held-out designs. It
takes a lock and retries infrastructure failures. It is resumable and takes many hours.

## Curves

`C.from_summary(summary)` extracts the deepest-point curve of one case as an `Option3Curve`:

| Field | Meaning |
|---|---|
| `x`, `level`, `case_id` | inputs, mesh level, case id |
| `lr`, `j_over_je` | the **supported** steps: at most step 11, `Lr ≤ 1.2`, and inside the small-strain cap (a prefix of the steps) |
| `kr` | `(J/J_e)^-1/2` |
| `increments()` | `log` of the increments of `J/J_e`, which is the quantity the emulator models |

`from_summary` raises `CurveQAError` for a case that is not complete and QA-passing, has no `"design"` inputs,
has non-finite values, has fewer than two supported steps, or whose `J/J_e` decreases by more than the
tolerance. Do not silently drop such a case: record why it was excluded.

## Training

```python
curves, excluded = [], []
for f in sorted(Path("results").glob("train_*.json")):
    try:
        curves.append(C.from_summary(json.loads(f.read_text())))
    except C.CurveQAError as exc:
        excluded.append((f.name, str(exc)))

level1 = [c for c in curves if c.level == LEVELS[0]]
level2 = {tuple(c.x): c for c in curves if c.level == LEVELS[1]}
pairs = [(c, level2[tuple(c.x)]) for c in level1 if tuple(c.x) in level2]

from critflaw.option3.prior import option2_log_increments
em = EM.train(level1, LOWER, UPPER, schedule_lr, pairs=pairs, prior=option2_log_increments)
```

`EM.train(curves, lower, upper, schedule_lr, pairs=(), n_starts=6, prior=None, log_inputs=())` fits one joint
GP over the inputs and the load step:

- `pairs` (level 1, level 2) add a **discretisation discrepancy** (Richardson-scaled, with its regression
  uncertainty). Without pairs, no discretisation error is represented.
- `prior=option2_log_increments` uses the Option 2 curve of the same `deformation` law as the GP mean, which
  leaves geometry and constraint effects to the GP.
- If the small-strain cap cut a training curve short, the emulator records that strain level (`em.cap_strain`)
  and does not advertise steps beyond it for new inputs.

Every draw has `Kr ≤ 1` and is non-increasing in `Lr` by construction.

## Prediction and statuses

```python
p = em.predict(x, n_draws=200, rng=np.random.default_rng(0))
p.status, p.lr, p.kr_mean, p.kr_draws.shape          # (n_draws, n_steps)
kr, status = p.kr_at(np.array([0.5, 0.9, 1.15]))      # draws at arbitrary Lr, and a status per query
```

`predict` **never raises** for a realisation. Each prediction carries a status:

| Status | Meaning |
|---|---|
| `supported` | inside the box. `lr`, `kr_mean` and `kr_draws` are given at the supported steps. |
| `outside_box` | some input is outside the training box. No curve is given. |
| `invalid` | wrong length or non-finite input |
| `cap_uncertain` | the small-strain cap leaves fewer than two steps (whole prediction), or a query `Lr` lies beyond the cap-cut support but within the schedule (in `kr_at`) |
| `beyond_support` | (in `kr_at`) the query `Lr` lies beyond the supported range. The draw is NaN, never clipped. |

## Validation

```python
from critflaw.option3 import validation as V
heldout = [C.from_summary(json.loads(f.read_text())) for f in sorted(Path("results").glob("heldout_*.json"))]
report = V.validate(em, heldout, n_draws=1000)
report.rmse, report.rmse_pass, report.coverage, report.coverage_se_curves, report.support_overruns
```

`ValidationReport` contains:

- `rmse`, and `rmse_pass`, which requires `rmse ≤ 0.02` **and** no support overrun. An overrun is a held-out
  curve shorter than the support the emulator advertised.
- `coverage`, the mean per-curve coverage of the 95 % band, with its standard error `coverage_se_curves`, and
  the point-weighted `coverage_points`.
- The energy score against a mean-only baseline (`energy`, `energy_baseline`), and the `variogram` score.

`V.openturns_agreement(em, train, heldout)` (needs the `crosscheck` extra) compares the GP with an independent
OpenTURNS kriging fit.

## Using a curve in an assessment

```python
from critflaw.option3.interface import assessment_curve, NotAssessmentGrade
curve = assessment_curve(p, lr_max=1.15, experimental=True)
curve.lr, curve.kr_draws, curve.lr_max, curve.experimental, curve.labels
```

`assessment_curve(prediction, lr_max, experimental=False)`:

- raises **`NotAssessmentGrade`** if the prediction is not assessment-grade (always, today) and
  `experimental=True` was not passed;
- raises `ValueError` if the prediction's status is not `supported`. An outside-box realisation is an
  undecided case for you to report, not a curve;
- cuts the curve at `lr_max` and keeps the labels: the verification status, "deepest point only", and
  "experimental".

## Failure probability with the emulator

The deterministic engine does not use the emulator. Pf with the emulator has its own machinery in
`critflaw.option3.reliability`. You provide a **problem**: a function from standard-normal samples `u`
(N × dim) to `(x, lr, kr, lr_max)`, which are the emulator inputs, the assessment point (`Lr` referred to
`sigma_0`) and the collapse limit for each realisation.

```python
from scipy import stats
DIM = 7

def problem(u):
    # illustrative only: map your random variables to the emulator inputs and the assessment point
    x = LOWER + (0.15 + 0.7 * stats.norm.cdf(u[:, :5])) * (UPPER - LOWER)
    lr = 0.75 * np.exp(0.12 * u[:, 5])
    kr = 0.55 * np.exp(0.15 * u[:, 6])
    return x, lr, kr, np.full(lr.size, 1.15)

rng = np.random.default_rng(1)
fd = EM.function_draws(em, n_draws=24, rng=rng, n_features=1000)     # coherent function draws
pops = [R.population(problem, DIM, 4000, np.zeros(DIM), rng, alpha=1.0) for _ in range(3)]
res = R.pf_crossed(fd, pops, experimental=True)
print(res.pf, res.pf_low.mean(), res.interval, res.undecided_mass, res.labels)
```

- `function_draws(em, n_draws, rng, n_features=2000)` gives `FunctionDraws`. Each draw is one curve-valued
  function of `x`, with the GP path, the discretisation terms and the far-field allowance fixed for that draw.
  `fd.curve(i, x)` and `fd.bounds(i, x, lr)` evaluate draw `i`.
- `population(problem, dim, n, design_point, rng, alpha=0.3)` is defensive importance sampling: a share
  `alpha` from N(0, I) and the rest centred on `design_point`, with exact weights. A good `design_point` is
  the design point of a cheap proxy, such as the Option 2 curve of the same law. `alpha=1.0` with a zero
  design point is plain Monte Carlo.
- `pf_crossed(fd, pops, experimental=False, convex=True)` raises **`PermissionError`** unless
  `experimental=True`.

Each realisation is classified in a fixed order:

1. **Collapse** (`Lr ≥ Lr,max`): failed, and the emulator is not queried.
2. `Kr ≥ 1`: failed for every draw.
3. Non-finite values: `invalid`.
4. Otherwise, by the draw's bounds between FE steps: safe below the lower bound, failed at or above the upper
   bound, `between_steps` in between. A status other than `supported` makes the realisation undecided.

`convex=True` tightens the bounds between steps under the assumption, stated in the labels, that `J/J_e` is
convex in `Lr` above a small floor. `convex=False` gives the rigorous monotone bounds, which are wider.

### Reading `PfResult`

| Field | Meaning |
|---|---|
| `pf_high`, `pf_low` | (draws × populations) with undecided realisations counted as failed (high) or safe (low) |
| `pf` | mean of `pf_high`: the headline |
| `undecided_mass` | weighted probability of each undecided class: `outside_box`, `invalid`, `beyond_support`, `cap_uncertain`, `between_steps` |
| `interval` | 90 % **surrogate-only** interval of `pf_high` across draws |
| `draw_component`, `population_component`, `residual` | two-way ANOVA: the surrogate (draw) and sampling (population) variance components |
| `sampling_se` | standard error of the population-averaged `pf_high` |
| `labels` | what the interval excludes (model uncertainty, distribution-parameter uncertainty, FE versus reality, GP hyperparameter uncertainty), "experimental", and the bound mode |

A large undecided mass means the emulator's support does not cover the realisations that matter. The gap
between `pf_low` and `pf_high` is then the honest answer.

### Active learning

```python
def oracle(x):                                   # one new FE case at x, returned as an Option3Curve
    from critflaw.option3.design import DesignPoint
    res = solver.run(DesignPoint(np.asarray(x), "train", LEVELS[0]).case())
    summary = result_summary(res)
    summary["design"] = {"x": [float(v) for v in x]}
    return C.from_summary(summary)                # raises CurveQAError for a failed case

out = active.run(problem, DIM, oracle, level1, LOWER, UPPER, schedule_lr, np.zeros(DIM),
                 np.random.default_rng(2), experimental=True)
out.converged, out.final.pf, [it.added for it in out.history]
```

`active.run(...)` raises `PermissionError` unless `experimental=True`. Each iteration does four things:

1. retrains the emulator on the curves so far;
2. estimates Pf over fixed populations;
3. stops when the sampling coefficient of variation of `pf_high` is below its bound **and** the surrogate
   interval is narrow relative to Pf (`StoppingRule(sampling_cov=0.10, width_ratio=0.5, abs_tol=1e-9,
   budget=20)`);
4. otherwise calls the oracle at the most informative input, with a declared share of calls (default 15 %)
   spent on exploration.

`ActiveResult` holds the `history` of `Iteration` records, `converged`, the `final` `PfResult` and the training
`curves`. Each `Iteration` has `pf`, `pf_low`, `interval`, `sampling_cov`, `draw_component`,
`unexplored_mass`, `undecided_mass` and `added`. `added` is `"acquisition"`, `"exploration"`, `"stopped"`,
`"budget"`, or `"draw-independent"` (every realisation was decided without the emulator).

Every oracle call is an FE solve of several minutes ([chapter 7](07_fe_backend.md#resource-use)). Run it with
the FE resource rules in mind: one solve at a time.

Next: [The knowledge base](09_knowledge_base.md).
