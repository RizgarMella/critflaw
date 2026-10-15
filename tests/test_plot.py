"""FAD graphics (critflaw.plot): highest-density-region levels, line samples beyond the cut-off, and that every
drawing function runs headless and reports the realisations it could not draw."""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import critflaw as cf  # noqa: E402
from critflaw import plot as fp  # noqa: E402

LINE = cf.Option1Curve(355e6, 490e6, 207e9, "continuous")


def test_hdr_levels_enclose_the_stated_mass() -> None:
    rng = np.random.default_rng(0)
    dens = rng.random((50, 50)) ** 4
    for mass, level in zip((0.5, 0.9, 0.99), fp.hdr_levels(dens, (0.5, 0.9, 0.99)), strict=True):
        enclosed = dens[dens >= level].sum() / dens.sum()
        assert mass <= enclosed < mass + 0.01
    assert np.all(np.diff(fp.hdr_levels(dens, (0.5, 0.9, 0.99))) < 0)  # larger mass, lower threshold


def test_line_samples_are_zero_at_and_beyond_the_cut_off() -> None:
    lr = np.array([0.0, 0.5, float(LINE.lr_max), float(LINE.lr_max) + 0.1])
    f = fp.line_samples(LINE, lr)
    assert f[0] == pytest.approx(1.0) and f[1] == pytest.approx(float(LINE(0.5)))
    assert f[2] == 0.0 and f[3] == 0.0


def test_drawing_functions_run_and_count_invalid_realisations() -> None:
    problem = cf.Assessment(cf.FlatPlate(25e-3, 0.5), cf.SurfaceFlaw(3e-3, 20e-3),
                            cf.Material(355e6, 490e6, 207e9, 90e6, yielding="continuous"),
                            cf.StressState(pm=150e6))  # fmt: skip
    r = problem.run()
    fig, ax = plt.subplots()
    fp.fad({"Option 1": LINE}, ax)
    fp.point(r, ax)
    fp.path([0.1, 0.2, np.nan, 0.4], [0.1, 0.2, 0.3, 0.4], ax, values=[0, 1, 2, 3], colorbar_label="t")
    rng = np.random.default_rng(1)
    lr = np.append(rng.normal(0.5, 0.1, 5000), [np.nan, np.nan])
    kr = np.append(rng.normal(0.3, 0.05, 5000), [0.2, np.nan])
    fp.density(lr, kr, ax)
    fp.outcomes(lr, kr, kr > 0.4, ax)
    labels = [t.get_label() for t in ax.get_lines()]
    assert any("2 outside validity, not drawn" in s for s in labels)  # never silently dropped
    fp.exceedance_field([LINE, cf.Option1Curve(330e6, 470e6, 207e9, "continuous")], ax)
    fig2 = fp.small_multiples([("a", lambda a: fp.fad(LINE, a)), ("b", lambda a: fp.fad(LINE, a))], ncols=3)
    assert len([a for a in fig2.axes if a.get_visible()]) == 2
    plt.close("all")


def test_array_line_is_refused_as_one_line() -> None:
    with pytest.raises(ValueError, match="per sample"):
        fp.fad(cf.Option1Curve(np.array([355e6, 400e6]), np.array([490e6, 520e6]), 207e9, "continuous"))
