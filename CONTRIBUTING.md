# Contributing to critflaw

Thank you for looking. critflaw is research software in development (see [`DISCLAIMER.md`](DISCLAIMER.md)).
Contributions that make it more correct, better evidenced or easier to use are very welcome.

## Set up

```bash
git clone https://github.com/RizgarMella/critflaw.git
cd critflaw
uv sync --extra plot            # runtime + dev tools (pytest, ruff, mypy, hypothesis, openturns)
```

## Checks (run before every pull request)

```bash
uv run ruff check src tests tools examples
uv run ruff format --check src tests tools
uv run mypy                                  # strict, on src/critflaw
uv run pytest -q                             # the full suite takes tens of minutes; pick files while iterating
uv run python tools/run_examples.py          # every worked example, one at a time
uv run python tools/make_notebooks.py        # regenerate examples/notebooks/ after editing an example
```

The live WARP3D tests (`tests/test_fe_live_*.py`, `tests/test_fe_runner.py`) skip unless Docker has the
`critflaw-warp3d:spike` image (build it from `docker/warp3d/Dockerfile`). Rebuild the showcase notebook with
`uv run --with nbclient --with ipykernel python tools/build_showcase.py`.

## Rules the code keeps

1. **Never invent equations, coefficients or tables.** If a value is unknown, raise
   `NotImplementedError("TODO(verify): <what is missing>")`.
2. **Cite the primary reference** for every implemented equation in `src/critflaw/provenance.py`, never a
   standard. If no primary reference is known, set `primary_reference=None` and add the item to
   `docs/unsourced_items.md` (a test enforces this). Do not paste text from licensed standards anywhere.
3. **Do not weaken or delete oracle tests** in `tests/oracles/`; reference implementations there are kept as
   received, and any correction is applied in the test with the reason stated.
4. Public numerical functions are **vectorised** (NumPy arrays in, arrays out) and **typed** (`mypy --strict`).
5. Problem objects are **immutable**; use `.with_()`.
6. Internal units are **SI floats**.
7. **No extrapolation** outside a solution's validity; invalid realisations are never silently dropped, clipped or
   resampled.
8. Acceptance is the **direct test** (`Kr < f(Lr)` and `Lr < Lr,max`), never the radial margin.

## Most valuable contributions

- **Checks against primary references**: comparing an implemented solution with the values published in the
  paper it cites (and setting `checked_against_primary`). This is what moves a record from `self-consistent` to
  `cross-checked` (theory manual, chapter 08).
- Independent verification and validation cases with published results.
- Bug reports with a minimal reproducing script.
- New worked examples and clearer documentation.

## How changes are reviewed

Every change of substance is reviewed adversarially before it is merged: an independent reviewer that sees only
the change and the theory manual, with every finding answered by a fix or a rebuttal backed by a runnable test.
Disagreements that remain are recorded as open decisions in `docs/development/roadmap.md`. Evidence beats
argument: a disagreement is settled by a test, a calculation or a cited source.
