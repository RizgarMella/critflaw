# 7. The FE back end (WARP3D)

`critflaw.fe` writes and runs WARP3D finite-element models of a **semi-elliptical surface flaw in a plate**
under combined membrane and bending load. It reads back J along the crack front and extracts the Option 3
quantities `Lr` and `Kr = sqrt(J_e / J)`. The material is WARP3D's `deformation` (power-law) model,
`WarpDeformationLaw(E, nu, sigma_0, n)`, which is the same law the Option 2 line can use
(`Option2Curve.from_warp_law(law, uts)`).

You need the Docker image described in [chapter 1](01_install.md#optional-the-warp3d-finite-element-back-end).
The FE back end is used for research and to train the [Option 3 emulator](08_option3_emulator.md). A result is
labelled `unverified` at best (see [verification status](#verification-status)), and it does not feed the
deterministic engine.

Theory: [FE and Option 3](../theory/06_fe_option3.md). Examples: *one WARP3D case end to end* (requires
Docker).

## One case, end to end

```python
from pathlib import Path
from critflaw.fe import FECase, WarpJSolver
from critflaw.fe.fecase import write_summary
from critflaw.fe.runner import WarpRunner

case = FECase(a=8e-3, c=20e-3, B=20e-3, W=200e-3, L_far=400e-3,
              E=207e9, nu=0.3, sigma_0=355e6, n=10.0, beta=0.0)
solver = WarpJSolver(Path("fe_runs"), WarpRunner(threads=4, memory="8g", timeout_seconds=1800))
res = solver.run(case)                       # never raises for a failed solve: see res.status

print(res.status, res.verification_status, res.failing_check)
write_summary(res, Path("fe_runs") / f"{case.case_id}.json")
```

`WarpJSolver.run` runs three WARP3D jobs, one after another:

1. an elastic **admissibility** case on a tied quarter-point mesh, which requires `K_I > 0` at every interior
   front station (crack-face contact refuses the nonlinear run);
2. the **plastic** case under proportional loading;
3. its **elastic companion** on the same mesh.

It then collects every QA check. Job directories go under `root/<case_id>/`, and the solve cache under
`root/cache/`.

## `FECase`: the case record

`FECase` is immutable (use `.with_()`) and SI. All of its inputs enter the case id
(`case.case_id`, a hash of `case.canonical_json()`).

| Field | Meaning | Unit / declared range |
|---|---|---|
| `a`, `c` | flaw depth and half-length | m; `a/B` in [0.2, 0.6], `a/c` in [0.4, 1.0] |
| `B`, `W` | plate thickness and width | m; `c/W` in [0, 0.1] |
| `L_far` | remote length | m; at least `max(20 B, 2 W)` (`critflaw.fe.crack_mesh.remote_length(B, W)`) |
| `E`, `nu` | elastic constants | Pa, -; `nu` in [0.25, 0.35] |
| `sigma_0` | the law's reference stress (**not** a 0.2 % proof stress) | Pa; `sigma_0/E` in [1e-3, 5e-3] |
| `n` | hardening exponent | [3, 20] |
| `beta` | loading mix `Pb / (Pm + Pb)` | [0, 1] |
| `level` | mesh level `(n_rings, n_sectors, n_front, n_front_s, n_blend)` | one of `critflaw.fe.fecase.LEVELS`; default `LEVEL_1 = (6, 12, 12, 3, 3)` |
| `phi_s_deg` | parametric evaluation angle near the free surface | degrees, in (0, 90) |
| `n_steps`, `lambda_1_lr`, `norm_res_tol` | load schedule and solver controls | defaults 12, 0.05, 0.002 |
| `threads` | thread count of the solve | default 4 |

A value outside the declared box raises `ValueError("... outside the declared box [...]")`. Only the
`plate_surface_flaw` family exists. Any other family raises `NotImplementedError("TODO(verify): ...")`.

The 0.2 % proof stress implied by a law is `case.law.proof_stress_0p2()`. To start from a measured `Rp0.2`, use
`critflaw.option3.interface.sigma0_from_proof_stress` ([chapter 8](08_option3_emulator.md)).

## `FEResult`

| Field | Meaning |
|---|---|
| `status` | run status of the plastic solve: `complete`, `truncated`, `failed` or `timeout` |
| `verification_status` | `fe-failed` (a QA check or the solve failed), `unverified` (every check passed) or `fe-verified` (see below) |
| `failing_check` | the first failing check's name, or `solve: <reason>` |
| `lam`, `lr` | cumulative load factors read back, and `Lr` at each step (from the plate reference stress, referred to `sigma_0`) |
| `phi`, `phi_physical` | parametric and physical angles of the front stations; the deepest point is the last station |
| `j`, `j_spread`, `j_e`, `kr` | (steps × stations): J (J/m^2, mean over rings 2..n), ring spread, elastic J, and `Kr` |
| `k_elastic` | signed `K_I` of the admissibility case per unit load factor (Pa·m^0.5) |
| `outside_small_strain`, `usable_steps` | steps beyond the far-region small-strain cap. They are stored but must not be used. |
| `qa` | every `QACheck(name, value, band, passed, note)` |
| `lr_end`, `proof_stress_0p2`, `considere_stress` | schedule end and law properties |
| `provenance` | parser version, image tag and digest, executable, threads, deck units, family-verification state |

`critflaw.fe.fecase.result_summary(res)` gives a JSON-serialisable dictionary of all of this, and
`write_summary(res, path)` writes it. This summary is the record that the Option 3 training reads.

## Lower level: `PlasticCase` and `run_case`

To run only the plastic case (with its elastic companion), outside the full solver:

```python
from critflaw.fe.plastic_case import run_case
from critflaw.fe.elastic_case import SolveFailed
from critflaw.fe.runner import SolveCache

try:
    plastic = run_case(case.plastic_case(), Path("fe_runs/p1"), WarpRunner(threads=4, memory="8g"),
                       SolveCache(Path("fe_runs/cache")))
    print(plastic.lr, plastic.kr[:, -1])        # deepest-point Kr at each step
except SolveFailed as exc:
    print("not complete:", exc, exc.record.status, exc.record.diagnostic)
```

Unlike `WarpJSolver.run`, `run_case` **raises** `SolveFailed` (carrying the `RunRecord`) when the solve is not
complete, when the run QA fails, or when the output cannot be evaluated. `critflaw.fe.elastic_case` has a
matching `ElasticCase(a, c, B, W, L_far, E, nu, pm, pb, ...)` and `run_case` for elastic K along the front.

`WarpRunner` fields: `image="critflaw-warp3d:spike"`, `threads=4`, `memory="6g"` (the container's memory
ceiling), `timeout_seconds=3600.0`, `warp3d_exe`. `WarpJSolver` without an explicit runner uses
`threads=4, memory="8g", timeout_seconds=5400`.

## Run statuses

| Status | Meaning |
|---|---|
| `complete` | exit code 0, normal termination, and exactly the declared steps converged |
| `truncated` | a step did not converge, or fewer steps converged than were declared |
| `failed` | non-zero exit, no normal termination, more steps than declared, an output the parser cannot read, or Docker or the image not available. The diagnostic is kept in `RunRecord.diagnostic`. |
| `timeout` | the wall-clock limit was reached. The container is killed before the output is read. |

Every outcome leaves a durable record with the raw output. Only `complete` runs can become `fe-verified`.

## QA checks and what a failure means

`res.qa` lists every check with its value and band. `failing_check` names the first failure. Mesh checks appear
twice, once with a `tied_` prefix for the admissibility mesh and once without for the plastic mesh.

| Check | What a failure means |
|---|---|
| `jacobian_positive`, `scaled_jacobian_outside_tube`, `scaled_jacobian_rings_2plus` | the generated mesh has inverted or badly distorted elements for this geometry |
| `tip_element_count`, `coincident_nodes_on_front_only`, `tip_node_pattern`, `tip_repeated_incidences`, `quarter_point_midside` | the crack-tip mesh topology is not as designed |
| `aspect_ratio_max_*` | reported only, never failing |
| `elastic_K_I_positive` | crack-face contact at some interior station: the nonlinear run is refused |
| `remote_profile` | the remote stress profile at the checked layer deviates from the applied membrane-plus-bending profile by more than the band (the far field is too short or the loading is wrong) |
| `equilibrium` | the reactions do not balance the applied force and moment in some run or step |
| `J_path_independence` | J varies over rings 2..n by more than 2 % at some step and interior station (the mesh does not resolve the field) |
| `plastic_zone_in_ring_1` | after step 1 the plastic zone extends beyond ring 1, so the elastic reference `J_e` is not valid |
| `J_1_vs_elastic_same_mesh` | the first-step J does not match the elastic companion on the same mesh |
| `far_region_strain_evaluated` | the far-region strains were not output at the steps with `Lr ≥ 0.8` |
| `small_strain_cap_first_assessed_step` | the far-region effective strain already exceeds the small-strain cap at the first assessed step. Later steps beyond the cap are always marked `outside_small_strain`. |

A failing check gives `verification_status = "fe-failed"`. The arrays are still returned for inspection, but
the case must not be used, and the Option 3 curve extraction refuses it.

## Verification status

`fe-verified` requires every QA check to pass **and** the verification of the FE family against independent
references to be complete. That family verification is not complete (`critflaw.fe.fecase.FAMILY_VERIFIED` is
`False`). Every passing case is therefore labelled **`unverified`** today, and `provenance["family_verification"]`
says what is pending.

## The solve cache

`SolveCache(root, retry_incomplete=False)` is an append-only store.

- **Solve key.** Each entry is keyed by the SHA-256 of the generated solver inputs, the deck name, the
  executable path, the thread count, the memory ceiling, the timeout, and the **image digest**. A changed case,
  a rebuilt image or a larger budget is a new key, so it is solved again.
- **Contents.** An entry holds the inputs, the gzipped raw output and stderr, `run.json` (the run record) and
  `meta.json`. Parsed products are recomputed from the stored raw output when the parser version changes,
  without solving again.
- **Retries.** A stored attempt that did not complete is returned as it is, unless `retry_incomplete=True`. In
  that case it is solved again, and the earlier attempt moves to `attempts/<n>/` and stays on record. A
  complete entry is never overwritten.
- **Location.** Cache hits go through the same run QA as fresh solves. Keep the cache on a disk with room: the
  raw outputs of production cases are large.

## Resource use

- **One case at a time.** Each WARP3D job runs in its own container with the memory ceiling `memory`.
  `WarpJSolver.run` runs its three jobs one after another. Do not run two solvers in parallel on a
  workstation: two production solves at once can exhaust a 16 GB host and destabilise Docker. When launching a
  batch, hold a lock so that a second instance cannot start.
- **Memory.** The default ceiling of 8 GB suits the production mesh level. On a host with about 16 GB, a
  ceiling of 4 GB to 6 GB with 4 threads leaves room for the operating system. A job killed for memory ends as
  `failed` (or `timeout`), and its record is kept.
- **Time.** A production-level case (the three jobs) takes a few minutes on 4 threads. The finer `LEVELS`
  take much longer and need more memory.
- **Docker restarts.** Under memory pressure the Docker daemon can restart, and image lookup then fails for a
  while. The runner turns that into a `failed` record ("image resolution failed: ..."), not an exception.
  Retry later.
- **Infrastructure failures are not properties of the case.** A run killed by the host is not evidence about
  the geometry. When batching, retry such failures instead of recording them as case results.

Next: [The Option 3 emulator](08_option3_emulator.md).
