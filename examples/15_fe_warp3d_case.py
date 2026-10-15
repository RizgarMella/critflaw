# %% [markdown]
# # An FE (WARP3D) surface-flaw case and its Option 3 curve
#
# The FE backend (`critflaw.fe`) writes WARP3D decks for a plate with a semi-elliptical surface flaw, runs them in
# a pinned Docker image, parses J along the crack front, runs the QA checks and returns an `FEResult`. The
# deepest-point Option 3 curve `Kr = (J / J_e)^(-1/2)` against `Lr` is extracted by
# `critflaw.option3.curve.from_summary`.
#
# **Guarded example.** Running a case needs Docker, the image `critflaw-warp3d:spike` and several minutes and GB
# of memory, so it is done only when the backend is available **and** `CRITFLAW_RUN_WARP3D=1` is set. Otherwise
# the example loads a stored result of the FE training campaign from `fe/runs_p5/results/`. If neither is available
# it prints how to run it and does nothing else.
#
# **Status.** Every FE result here is `verification_status = "unverified"` (the FE family verification is
# incomplete) and every Option 3 curve is `assessment_grade = False`: **experimental**.
#
# Theory: `docs/manual/theory/06_fe_option3.md`.

# %%
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

import critflaw as cf

SCRIPT = "__file__" in globals()
HERE = (
    Path(__file__).resolve().parent
    if SCRIPT
    else (Path.cwd() / "examples" if (Path.cwd() / "examples").is_dir() else Path.cwd())
)
REPO = HERE.parent
OUT = HERE / "_output"
OUT.mkdir(exist_ok=True)
try:
    import matplotlib

    if SCRIPT:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # plotting is optional
    plt = None


def save(fig, name):
    fig.savefig(OUT / name, dpi=120, bbox_inches="tight")
    if not SCRIPT:
        plt.show()
    plt.close(fig)


IMAGE = "critflaw-warp3d:spike"


def warp3d_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


# %% [markdown]
# ## 1. Get an FE result: run one (opt-in) or load a stored one

# %%
summary = None
source = ""
if warp3d_available() and os.environ.get("CRITFLAW_RUN_WARP3D") == "1":
    from critflaw.fe.fecase import WarpJSolver, result_summary
    from critflaw.fe.runner import WarpRunner
    from critflaw.option3.design import C_OVER_W, heldout_design

    point = heldout_design(1)[0]  # the first held-out design point (inside the declared parameter box)
    case = point.case(C_OVER_W)
    print(f"running WARP3D case {case.case_id} (this takes minutes) ...")
    solver = WarpJSolver(OUT / "fe_runs", WarpRunner(IMAGE, threads=4, memory="6g", timeout_seconds=3600))
    result = solver.run(case)
    summary = result_summary(result)
    summary["design"] = {"role": "example", "x": point.x.tolist(), "c_over_W": C_OVER_W}
    source = f"live WARP3D run ({result.status}, {result.verification_status})"
else:
    stored = sorted((REPO / "fe" / "runs_p5" / "results").glob("*.json"))
    if stored:
        path = stored[0]
        summary = json.loads(path.read_text(encoding="utf-8"))
        source = f"stored result {path.relative_to(REPO).as_posix()}"
HAVE_RESULT = summary is not None
if not HAVE_RESULT:
    print(
        "No WARP3D backend and no stored results found.\n"
        "To run this example live: build the image 'critflaw-warp3d:spike' (see docker/warp3d), start Docker, then\n"
        "  CRITFLAW_RUN_WARP3D=1 uv run python examples/15_fe_warp3d_case.py\n"
        "Stored results are written by: uv run python benchmarks/option3_campaign.py train"
    )
else:
    print("source:", source)

# %% [markdown]
# ## 2. The case and its QA record

# %%
if HAVE_RESULT:
    case_d = summary["case"]
    print(
        f"case {summary['case_id']}: a = {case_d['a'] * 1e3:.2f} mm, c = {case_d['c'] * 1e3:.2f} mm, "
        f"B = {case_d['B'] * 1e3:.0f} mm, W = {case_d['W'] * 1e3:.0f} mm, n = {case_d['n']:.2f}, "
        f"sigma_0 = {case_d['sigma_0'] / 1e6:.0f} MPa, beta = {case_d['beta']:.2f}"
    )
    print(
        f"status {summary['status']}, verification {summary['verification_status']}, "
        f"failing check {summary['failing_check']}"
    )
    for check in summary.get("qa", [])[:8]:
        print(f"  QA {check['name']}: value {check['value']:.4g}, band {check['band']:.4g}, passed {check['passed']}")

# %% [markdown]
# ## 3. The deepest-point Option 3 curve, against Option 1 and Option 2 of the same law
#
# `Lr` in the FE is referred to the law's reference stress `sigma_0` (not a 0.2 % proof stress). The Option 2 line
# of the same WARP3D deformation law uses the same reference (`Option2Curve.from_warp_law`). The tensile strength,
# which only sets the collapse cut-off, is an illustrative assumption (1.3 sigma_0).

# %%
if HAVE_RESULT:
    from critflaw.fe.warp_law import WarpDeformationLaw
    from critflaw.option3.curve import CurveQAError, from_summary

    try:
        curve3 = from_summary(summary)
    except CurveQAError as exc:
        curve3 = None
        print("the FE result does not give a usable Option 3 curve:", exc)
    if curve3 is not None:
        law = WarpDeformationLaw(case_d["E"], case_d["nu"], case_d["sigma_0"], case_d["n"])
        uts = 1.3 * case_d["sigma_0"]  # illustrative; sets Lr,max only
        opt2 = cf.Option2Curve.from_warp_law(law, uts)
        opt1 = cf.Option1Curve(case_d["sigma_0"], uts, case_d["E"], "continuous")
        print(
            "EXPERIMENTAL Option 3 curve (verification_status =",
            curve3.verification_status,
            ", assessment_grade =",
            curve3.assessment_grade,
            ")",
        )
        for lr_k, kr_k in zip(curve3.lr, curve3.kr, strict=True):
            print(
                f"  Lr = {lr_k:.3f}: Kr(Option 3, FE) = {kr_k:.4f}, Kr(Option 2) = {float(opt2(lr_k)):.4f}, "
                f"Kr(Option 1) = {float(opt1(lr_k)):.4f}"
            )
        assert curve3.verification_status == "unverified" and not curve3.assessment_grade
        assert np.all(curve3.kr <= 1.0 + 1e-9) and np.all(np.diff(curve3.kr) <= 1e-12)

        if plt is not None:
            lr_grid = np.linspace(0.0, max(1.25, float(curve3.lr[-1])), 400)
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            ax.plot(lr_grid, opt1(lr_grid), label="Option 1 (continuous)")
            ax.plot(lr_grid, opt2(lr_grid), label=f"Option 2, WARP3D law n = {case_d['n']:.1f}")
            ax.plot(curve3.lr, curve3.kr, "ko-", ms=4, label="Option 3, FE deepest point (EXPERIMENTAL)")
            ax.set(
                xlabel="Lr (referred to sigma_0)",
                ylabel="Kr",
                ylim=(0, 1.05),
                title=f"FE case {summary['case_id']} (unverified)",
            )
            ax.legend(fontsize=8)
            save(fig, "15_fe_option3.png")

# %% [markdown]
# ## Interpretation
#
# * The FE curve is case-specific: it reflects this geometry, hardening exponent and load mix (`beta`). Where it
#   lies above the Option 2 line the generic line is conservative for this case, and vice versa.
# * Only the deepest point is extracted (the surface position is mesh-sensitive), the support ends at the
#   last accepted load step, and the family verification is incomplete, so the curve is **experimental** and is
#   refused by `critflaw.option3.interface.assessment_curve` unless `experimental=True` is declared.
# * Examples 16 and 17 show how such curves are emulated across the parameter box and used in a failure
#   probability, on a synthetic family that is cheap to evaluate.
