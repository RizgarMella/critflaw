# 1. Installation

`critflaw` is installed from source. It needs **Python 3.11 or later**. The core depends only on NumPy (1.26 or
later) and SciPy (1.11 or later).

## With uv (recommended)

From the repository root:

```bash
uv sync                      # the package plus the development tools (pytest, ruff, mypy, ...)
uv sync --extra plot         # add an optional extra (repeat --extra for several)
uv run python -c "import critflaw; print(critflaw.__version__)"
```

Run your own scripts through the project environment with `uv run python my_script.py`.

## With pip

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; on Linux/macOS: source .venv/bin/activate
pip install -e .              # core only
pip install -e ".[plot]"      # with an extra
pip install -e ".[units,plot,crosscheck]"
```

## Optional extras

| Extra | Installs | What it is for |
|---|---|---|
| `units` | `pint` | Your own unit handling before you call the library. `critflaw` itself takes **plain SI floats or arrays**, does not accept `pint` quantities, and does not import `pint`. Convert to SI first (for example `q.to("Pa").magnitude`). |
| `plot` | `matplotlib` | Plotting in the examples, and the FE mesh renderer `critflaw.fe.render`, which imports matplotlib only when called. |
| `crosscheck` | `openturns` | Independent cross-checks, for example `critflaw.option3.validation.openturns_agreement` (an independent kriging fit compared with the emulator). |

## Checking the installation

```python
import critflaw as cf

plate = cf.FlatPlate(thickness=25e-3, width=500e-3)
flaw = cf.SurfaceFlaw(a=3e-3, two_c=15e-3)
steel = cf.Material(yield_strength=355e6, uts=490e6, modulus=207e9, toughness=120e6, yielding="continuous")
result = cf.Assessment(plate, flaw, steel, cf.StressState(pm=150e6, pb=40e6)).run()
print(result.report())
```

If this prints `Evaluations: 1; acceptable: 1; outside validity: 0`, followed by the trace, the core works.
[Chapter 3](03_deterministic.md) explains the output.

The test suite (`uv run pytest -q`) is long and memory-hungry. On a small machine, run single test files
(`uv run pytest -q tests/test_fad.py`) rather than the whole suite. The tests that need WARP3D skip themselves
when the Docker image is absent.

## Optional: the WARP3D finite-element back end

You need this only for [chapter 7](07_fe_backend.md) (FE cases) and for training the
[Option 3 emulator](08_option3_emulator.md) on new FE data. The deterministic, probabilistic and fatigue
features do not need it.

### What you need

- **Docker** (Docker Desktop on Windows or macOS, or Docker Engine on Linux), with the `docker` command on
  your `PATH`. The runner calls `docker run`, `docker image inspect` and `docker kill` as subprocesses.
- **A local WARP3D image.** The runner expects an image tagged **`critflaw-warp3d:spike`** by default
  (`critflaw.fe.runner.WarpRunner.image`). That image contains the executable at
  `/opt/warp3d/run_linux/warp3d.exe` (`WarpRunner.warp3d_exe`). No pre-built image is published. You build it
  yourself from the Dockerfile in the repository.
- **Memory.** The production cases run with a container memory ceiling of several GB (the solver default is
  8 GB). A host with about 16 GB of RAM can run **one case at a time**. See
  [chapter 7](07_fe_backend.md#resource-use).

### Building the image

The Dockerfile is `docker/warp3d/Dockerfile`. It builds WARP3D from its public source repository at a pinned
commit, using the Intel Fortran compiler and MKL from Intel's apt repository. The result is a plain Ubuntu
runtime image.

```bash
cd docker/warp3d
docker build -t critflaw-warp3d:spike .
```

The build downloads the Intel oneAPI compiler packages and the WARP3D source, so it needs network access and
takes a while. It is done once. The Dockerfile targets AVX2 with a vendor-neutral `-march` flag, so it also
runs on AMD processors.

### Checking the image

```bash
docker image inspect critflaw-warp3d:spike --format "{{.Id}}"
```

This prints the image digest (`sha256:...`). The runner records that digest as the identity of the
executable. Results computed by a rebuilt image are therefore cached separately from older ones (see
[the cache](07_fe_backend.md#the-solve-cache)).

If you tag the image differently, pass the tag explicitly: `WarpRunner(image="my-tag")`.

Next: [Problem objects](02_problem_objects.md).
