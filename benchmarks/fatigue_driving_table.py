"""B16 (design note phase 3 Part B): the driving variable across the embedded -> surface change (G4(c)).

Grows one embedded flaw in a flat plate deterministically with `grow()` under a fixed extreme load and, at
each checkpoint, assesses the current flaw statically with the Phase 1/Part B engine: embedded flaws directly
(the source's re-characterisation annex is not applied, G4(c)); after breakthrough the surface flaw `2a + p`
by `2c`. The table records, per checkpoint: time, kind, a, c, p (ligament), Lr, Kr, radial margin, driving
variable g = margin - 1, validity and exit. It is information for the owner (dispute D-1), not a test.

    uv run python benchmarks/fatigue_driving_table.py --out benchmarks/out/fatigue_driving_table.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import critflaw as cf
from critflaw import prob
from critflaw.fatigue import ExtremeLoad, GrowthLaw, Spectrum, TimeDependentAssessment
from critflaw.fatigue import growth as G
from critflaw.fatigue.laws import coefficient_to_si

B, W = 25e-3, 0.5
LOAD = 200e6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("benchmarks/out/fatigue_driving_table.md"))
    args = parser.parse_args()

    material = cf.Material(355e6, 490e6, 207e9, 90e6, yielding="continuous")
    # mid-thickness-ish embedded flaw, 2a = 6 mm, 2c = 24 mm, ligament p = 6 mm: breaks through to the nearer surface
    flaw = cf.EmbeddedFlaw(6e-3, 24e-3, 6e-3)
    problem = cf.Assessment(cf.FlatPlate(B, W), flaw, material, cf.StressState(pm=100e6))
    law = GrowthLaw(float(coefficient_to_si(5.86e-13, 2.88)), 2.88, 0.0)
    spectrum = Spectrum.repeating([(120e6, 0.0, 1.0)], 2e5)
    times = np.arange(1, 49) * 0.25  # the load-process step is one year: ExtremeLoad(..., 0.25)
    run = TimeDependentAssessment(
        problem,
        law,
        spectrum,
        ExtremeLoad(prob.Uniform(LOAD, LOAD * (1 + 1e-6)), 0.25),
        times=times,
        leak_is_failure=False,
    )
    hist = run.grow(load=LOAD)

    rows = []
    for k, t in enumerate(hist.times):
        a, c, kind = float(hist.a[k]), float(hist.c[k]), int(hist.kind[k])
        if kind == G.EMBEDDED:
            res = cf.Assessment(
                cf.FlatPlate(B, W),
                cf.EmbeddedFlaw(2.0 * a, 2.0 * c, _ligament(flaw, a)),
                material,
                cf.StressState(pm=LOAD),
            ).run()
            label = "embedded"
        elif kind == G.SURFACE:
            res = cf.Assessment(cf.FlatPlate(B, W), cf.SurfaceFlaw(a, 2.0 * c), material, cf.StressState(pm=LOAD)).run()
            label = "surface"
        elif kind == G.THROUGH:
            res = cf.Assessment(cf.FlatPlate(B, W), cf.ThroughFlaw(2.0 * c), material, cf.StressState(pm=LOAD)).run()
            label = "through"
        else:
            rows.append(
                (t, G_NAMES.get(kind, str(kind)), a, c, float("nan"), *(float("nan"),) * 3, float(hist.g[k]), "-", "-")
            )
            continue
        p = _ligament(flaw, a) if kind == G.EMBEDDED else float("nan")
        rows.append(
            (
                t,
                label,
                a,
                c,
                p,
                float(res.lr),
                float(res.kr),
                float(res.radial_margin),
                float(hist.g[k]),
                str(bool(res.valid)),
                str(res.exit.item()),
            )
        )

    geometry = (
        f"Flat plate B = {B * 1e3:.0f} mm, W = {W * 1e3:.0f} mm; embedded flaw 2a = {flaw.two_a * 1e3:.0f} mm, "
        f"2c = {flaw.two_c * 1e3:.0f} mm, p = {flaw.p * 1e3:.0f} mm; Paris law C = 5.86e-13 (N, mm), m = 2.88; "
        f"spectrum 120 MPa membrane range, 2e5 cycles/year; extreme load Pm = {LOAD / 1e6:.0f} MPa. The embedded flaw "
        "is assessed directly (G4(c)); after breakthrough the surface flaw has depth 2a + p (the destination rule) "
        "and the same length."
    )
    events = (
        f"Events: surface breaking at t = {_fmt(getattr(hist, 't_surface_breaking', float('nan')))}, "
        f"leak at t = {_fmt(hist.t_leak)}, leak_assumed at t = {_fmt(hist.t_leak_assumed)}, "
        f"ended at t = {_fmt(hist.t_ended)}, first failure at t = {_fmt(hist.first_failure)}."
    )
    lines = [
        "# B16: driving variable across the embedded -> surface change",
        "",
        geometry,
        "",
        events,
        "",
        "| t (yr) | kind | a (mm) | c (mm) | p (mm) | Lr | Kr | radial margin | g (grow) | valid | exit |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t, label, a, c, p, lr, kr, rm, g, valid, exit_ in rows:
        cells = [f"{t:.2f}", label, f"{a * 1e3:.3f}", f"{c * 1e3:.3f}", _fmt(p * 1e3), _fmt(lr), _fmt(kr), _fmt(rm)]
        cells += [_fmt(g), valid, exit_]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Reading: the severity (radial margin) just before breakthrough is that of the embedded flaw assessed "
        "directly; the first surface checkpoint carries the destination depth 2a + p. A jump in Kr across the "
        "change is the re-characterisation effect the owner is asked to judge in dispute D-1; the engine's "
        "checkpoint fracture check evaluates both sides of the event.",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(rows)} rows)")


G_NAMES = {
    G.SURFACE: "surface",
    G.THROUGH: "through",
    G.NONE: "none",
    G.ENDED: "ended",
    G.EMBEDDED: "embedded",
    G.EXTENDED: "extended",
}


def _ligament(flaw: cf.EmbeddedFlaw, a: float) -> float:
    """Symmetric growth keeps B' = 2a + 2p invariant (G4): p = (B' - 2a) / 2."""
    b_prime = float(flaw.two_a) + 2.0 * float(flaw.p)
    return 0.5 * (b_prime - 2.0 * a)


def _fmt(x: float) -> str:
    return "-" if not np.isfinite(x) else f"{x:.4g}"


if __name__ == "__main__":
    main()
