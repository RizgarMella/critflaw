# 11. FAD graphics

`critflaw.plot` draws failure assessment diagrams with matplotlib (install the `plot` extra). Every function
draws on a given `Axes` (or creates one) and returns it, so the pieces compose. Worked examples:
`examples/19_fad_gallery.py` and `examples/showcase.ipynb`.

| function | draws |
|---|---|
| `fad(lines, ax=None, *, lr_extent, kr_extent, shade, title)` | one line or a `{label: line}` mapping, with the cut-off; shades the first line's acceptable and unacceptable regions |
| `point(result, ax, *, ray=True)` | the assessment point(s) of a `Result` and the proportional load ray to the line at the radial margin; the legend states the direct-test verdict |
| `path(lr, kr, ax, *, values, cmap, colorbar_label)` | a path of points (load sweep, flaw-size sweep, fatigue history), optionally coloured by a parameter such as time; NaN points break the path |
| `density(lr, kr, ax, *, weights, bins, smooth, masses, fill, floor)` | the probability density of uncertain points as a field, with contours of the highest-density regions holding `masses` (default 50/90/99 %); importance-sampling weights respected |
| `outcomes(lr, kr, failed, ax)` | safe (thinned) and failed (all) realisations as a scatter |
| `exceedance_field(lines, ax, *, quantiles)` | for samples of an uncertain line, the probability that a point at each (Lr, Kr) is not acceptable, with quantile lines |
| `small_multiples(panels, ncols)` | a figure of FAD panels from `(title, draw(ax))` pairs |
| `hdr_levels(density, masses)`, `line_samples(line, lr)` | the helpers behind the above |

```python
import numpy as np
import critflaw as cf
from critflaw import plot as fadplot, prob

problem = cf.Assessment(cf.FlatPlate(25e-3, 0.5), cf.SurfaceFlaw(3e-3, 20e-3),
                        cf.Material(355e6, 490e6, 207e9, 90e6, yielding="continuous"), cf.StressState(pm=150e6))
ax = fadplot.fad(problem.curve())
fadplot.point(problem.run(), ax)

kmat = prob.LogNormal.from_mean_sd(80e6, 20e6, name="Kmat")
pa = cf.ProbabilisticAssessment(problem.with_(material=problem.material.with_(toughness=kmat)))
cloud = pa.evaluate(np.random.default_rng(0).standard_normal((50_000, pa.vector.size)))
fadplot.density(cloud.lr, cloud.kr, ax)
```

Notes:

- An assessment line with array properties (one line per sample) cannot be drawn as one line: pass a list of
  lines to `exceedance_field`, or draw them one by one.
- Realisations outside a solution's validity (NaN `Lr` or `Kr`) are excluded from what is drawn and **counted in
  the legend**; they are never silently dropped.
- The density field shows where the assessment point is likely to be; the failure probability is its mass
  beyond the line. Compute it with `ProbabilisticAssessment.solve` (chapter 4), not from the picture.
