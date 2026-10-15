"""Phase 5: the curve extraction and the emulator, first on a closed-form synthetic family of monotone curves
with input-dependent support (design note phase 5, section 6), then the constraint, status and label properties."""

from __future__ import annotations

import numpy as np
import pytest

from critflaw.option3 import curve as C
from critflaw.option3 import emulator as EM
from critflaw.option3.design import LOWER, UPPER, sobol_points


def schedule(x: np.ndarray) -> np.ndarray:
    """The FE schedule's shape: geometric 0.05 -> 0.8 over six steps, then linear to the strain-cap end."""
    lr_n = min(1.5, (0.02 / x[3]) ** (1.0 / x[2]))
    lo = np.geomspace(0.05, min(0.8, 0.9 * lr_n), 6)
    return np.concatenate([lo, np.linspace(lo[-1], lr_n, 7)[1:]])


def true_ratio(x: np.ndarray, lr: np.ndarray) -> np.ndarray:
    """A smooth monotone J/J_e(Lr): 1 at small Lr, growing faster for small n and deep flaws."""
    m = 2.0 + 0.4 * x[2]
    amp = 0.4 + 0.8 * x[0] + 0.2 * x[4]
    return 1.0 + amp * (lr / 1.2) ** m


def synth_curve(x: np.ndarray, level: tuple[int, ...] = (6, 12, 12, 3, 3)) -> C.Option3Curve:
    lr = schedule(x)
    ratio = true_ratio(x, lr)
    ratio = ratio / ratio[0]  # J_e is defined by step 1
    keep = (np.arange(lr.size) < C.MAX_STEP) & (lr <= C.LR_SUPPORT)
    k = int(np.argmin(keep)) if not keep.all() else lr.size
    return C.Option3Curve(x, level, "synthetic", lr[:k], ratio[:k])


@pytest.fixture(scope="module")
def trained() -> EM.Emulator:
    xs = sobol_points(48, 1)
    return EM.train([synth_curve(x) for x in xs], LOWER, UPPER, schedule, n_starts=3)


def test_heldout_accuracy_coverage_and_constraints(trained: EM.Emulator) -> None:
    xs = sobol_points(12, 99)
    errs, inside, n_pts = [], 0, 0
    rng = np.random.default_rng(5)
    for x in xs:
        p = trained.predict(x, 400, rng)
        truth = synth_curve(x)
        assert p.status == "supported" and np.allclose(p.lr, truth.lr)
        assert np.all(p.kr_draws <= 1.0 + 1e-12) and np.all(np.diff(p.kr_draws, axis=1) <= 1e-12)  # by construction
        lo, hi = np.percentile(p.kr_draws, [2.5, 97.5], axis=0)
        tol = EM.COVERAGE_RESOLUTION  # the increment floor's resolution in Kr (design note phase 5, section 6)
        inside += int(np.sum((truth.kr >= lo - tol) & (truth.kr <= hi + tol)))
        n_pts += truth.kr.size
        errs.append(p.kr_mean - truth.kr)
    rmse = float(np.sqrt(np.mean(np.concatenate(errs) ** 2)))
    assert rmse <= 0.02, rmse  # the design's V9 target, met on the synthetic family
    assert inside / n_pts >= 0.8


def test_status_never_raises_and_labels(trained: EM.Emulator) -> None:
    out = trained.predict(np.array([0.1, 0.5, 10.0, 2e-3, 0.0]), 10)
    assert out.status == "outside_box" and out.kr_draws.shape == (10, 0)
    p = trained.predict(np.array([0.4, 0.7, 10.0, 2e-3, 0.5]), 50)
    kr, status = p.kr_at(np.array([0.5, p.lr[-1], p.lr[-1] + 0.05]))
    assert list(status) == ["supported", "supported", "beyond_support"] and np.all(np.isnan(kr[:, 2]))
    assert p.verification_status == "unverified" and p.assessment_grade is False


def test_discrepancy_enters_the_draws() -> None:
    xs = sobol_points(24, 3)
    l1 = [synth_curve(x) for x in xs]
    # a synthetic level 2 whose increments are 1 % larger: d = y_L1 - y_L2 = -log(1.01)
    l2 = [C.Option3Curve(c.x, (8, 16, 16, 4, 4), "s", c.lr, 1.0 + np.r_[0.0, np.cumsum(np.diff(c.j_over_je) * 1.01)])
          for c in l1[:6]]  # fmt: skip
    em = EM.train(l1, LOWER, UPPER, schedule, pairs=list(zip(l1[:6], l2, strict=True)), n_starts=2)
    y = l1[0].increments()
    live = np.flatnonzero(y > np.log(C.INCREMENT_FLOOR) + 1e-9)  # floored steps carry no level difference
    assert live.size >= 4
    assert np.allclose(em.discrepancy.coef[live[-4:], 0], -np.log(1.01), atol=1e-12)  # intercept (no x dependence)
    x = xs[0]
    base = EM.Emulator(em.lower, em.upper, schedule, em.gp, EM.Discrepancy.none(), 0.0).predict(x, 2000)
    corr = em.predict(x, 2000)
    # the continuum correction (R ~ 3.5 times the level-1 -> level-2 change) raises J/J_e: Kr lower
    assert np.mean(corr.kr_mean[1:] - base.kr_mean[1:]) < 0.0


def test_curve_from_summary_support_and_qa() -> None:
    lr = np.array([0.05, 0.09, 0.16, 0.28, 0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
    ratio = 1.0 + 0.3 * lr**4
    ratio /= ratio[0]
    summary = {
        "status": "complete", "verification_status": "unverified", "case_id": "abc", "lr": lr.tolist(),
        "kr": np.column_stack([ratio**-0.5] * 3).tolist(), "outside_small_strain": [False] * 12,
        "case": {"level": [6, 12, 12, 3, 3]}, "design": {"x": [0.4, 0.7, 10.0, 2e-3, 0.0]},
    }  # fmt: skip
    c = C.from_summary(summary)
    with pytest.raises(C.CurveQAError, match="design"):  # CC-05
        C.from_summary({k: v for k, v in summary.items() if k != "design"})
    assert c.n_steps == 10 and c.lr[-1] == 1.2 and c.verification_status == "unverified" and not c.assessment_grade
    summary["outside_small_strain"] = [False] * 8 + [True] * 4  # the cap ends the support at step 8
    assert C.from_summary(summary).n_steps == 8
    bad = dict(
        summary, kr=np.column_stack([(ratio * np.r_[1.0] * np.r_[[1.0] * 6, 0.85, [1.0] * 5]) ** -0.5] * 3).tolist()
    )
    with pytest.raises(C.CurveQAError, match="decreases"):
        C.from_summary(bad)
    with pytest.raises(C.CurveQAError):
        C.from_summary(dict(summary, verification_status="fe-failed"))


def test_sigma0_from_proof_stress_inverts_the_law() -> None:
    """P5-C-01: Lr is referred to sigma_0; a user's Rp0.2 is converted, the ratio 1.04-1.52 over the box."""
    from critflaw.fe.warp_law import WarpDeformationLaw
    from critflaw.option3.interface import sigma0_from_proof_stress

    ns = np.array([3.0, 10.0, 20.0])
    s0 = np.array([1e-3, 1.7e-3, 5e-3]) * 207e9
    rp = np.asarray(WarpDeformationLaw(207e9, 0.3, s0, ns).proof_stress_0p2())
    assert np.allclose(sigma0_from_proof_stress(rp, 207e9, ns), s0, rtol=1e-9)  # vectorised (CC-06)
    s0 = 1e-3 * 207e9
    assert float(WarpDeformationLaw(207e9, 0.3, s0, 3.0).proof_stress_0p2()) / s0 == pytest.approx(1.52, abs=0.01)


def test_assessment_guard_refuses_experimental_curves_unless_declared(trained: EM.Emulator) -> None:
    from critflaw.option3.interface import NotAssessmentGrade, assessment_curve

    p = trained.predict(np.array([0.4, 0.7, 10.0, 2e-3, 0.5]), 20)
    with pytest.raises(NotAssessmentGrade):
        assessment_curve(p, 1.1)
    cur = assessment_curve(p, 1.1, experimental=True)
    assert cur.experimental and cur.lr[-1] <= 1.1 and "experimental" in cur.labels
    assert any("unverified" in lab for lab in cur.labels) and cur.kr_draws.shape[1] == cur.lr.size
    with pytest.raises(ValueError, match="outside_box"):
        assessment_curve(trained.predict(np.array([0.1, 0.5, 10.0, 2e-3, 0.0]), 5), 1.1, experimental=True)


def test_scores_and_validation_on_the_synthetic_family(trained: EM.Emulator) -> None:
    from critflaw.option3 import validation as V

    rng = np.random.default_rng(0)
    obs = np.array([1.0, 0.9, 0.8])
    good = obs + 0.01 * rng.standard_normal((2000, 3))
    bad = obs + 0.2 + 0.01 * rng.standard_normal((2000, 3))
    assert V.energy_score(good, obs) < V.energy_score(bad, obs)
    # the variogram score sees a wrong dependence that the marginals do not: same marginals, shuffled columns
    corr = np.repeat(rng.standard_normal((2000, 1)), 3, axis=1) * 0.05 + obs
    indep = np.column_stack([rng.permutation(corr[:, i]) for i in range(3)])
    tied = obs + np.full(3, 0.05 * rng.standard_normal())
    assert V.variogram_score(corr, tied) < V.variogram_score(indep, tied)
    rep = V.validate(trained, [synth_curve(x) for x in sobol_points(12, 99)], n_draws=400)
    assert rep.rmse_pass and rep.coverage >= 0.8 and rep.energy < rep.energy_baseline + 1e-12
    assert rep.n_curves == 12 and np.isfinite(rep.coverage_se_curves)


def test_round1_fixes_cap_regression_status_and_curve_qa(trained: EM.Emulator) -> None:
    """Phase 5 code round 1 (red team A): P-01 cap-cut training curves limit the advertised support; P-02 the
    discrepancy is a per-step linear regression in x; P-03 outside-box/invalid predictions never raise; P-04
    non-finite Kr is a curve QA failure."""
    # P-01
    xs = sobol_points(24, 3)
    curves = [synth_curve(x) for x in xs]
    cut = C.Option3Curve(curves[0].x, curves[0].level, "cut", curves[0].lr[:5], curves[0].j_over_je[:5])
    em = EM.train([cut, *curves[1:]], LOWER, UPPER, schedule, n_starts=2)
    lr0 = em.support(curves[0].x)
    assert em.cap_strain == pytest.approx(EM.reference_strain(curves[0].x, lr0)[5])  # the strain where the cap cut
    # a new input: its steps at or beyond that reference strain are not advertised (case-specific, R2 P-01)
    hit = next(x for x in xs[1:] if np.any(EM.reference_strain(x, em.support(x)) >= em.cap_strain))
    p = em.predict(hit, 20)
    ref = EM.reference_strain(hit, p.lr)
    assert p.status == "supported" and np.all(ref[2:] < em.cap_strain) and p.schedule_end_lr > p.lr[-1]
    kr, st = p.kr_at(np.array([p.lr[1], p.lr[-1] + 1e-3, p.schedule_end_lr + 0.01]))
    assert list(st) == ["supported", "cap_uncertain", "beyond_support"] and np.all(np.isfinite(kr[:, 0]))
    # validation counts a support overrun: a held-out curve shorter than the advertised support fails the gate
    from critflaw.option3 import validation as V

    short = C.Option3Curve(xs[5], curves[5].level, "short", curves[5].lr[:3], curves[5].j_over_je[:3])
    rep = V.validate(trained, [short], n_draws=50)
    assert rep.support_overruns == 1 and not rep.rmse_pass
    # P-02: a discrepancy linear in a/B is recovered at the live steps
    l1 = curves[:12]
    l2 = []
    for c in l1:
        y = c.increments()
        d = 0.002 + 0.01 * (c.x[0] - LOWER[0]) / (UPPER[0] - LOWER[0])  # y_L1 - y_L2
        l2.append(C.Option3Curve(c.x, (8, 16, 16, 4, 4), "s", c.lr, 1.0 + np.r_[0.0, np.cumsum(np.exp(y - d))]))
    disc = EM.Discrepancy.from_pairs(list(zip(l1, l2, strict=True)), LOWER, UPPER)
    i = 8  # step 10
    assert disc.coef[i, 0] == pytest.approx(0.002, abs=1e-9) and disc.coef[i, 1] == pytest.approx(0.01, abs=1e-9)
    assert disc.sd[i] < 1e-9 and disc.mean_at(LOWER + 0.5 * (UPPER - LOWER), np.array([10]))[0] == pytest.approx(0.007)
    # CC-02: the regression's own uncertainty: leverage grows towards the box corners
    lev_mid = disc.leverage_at(LOWER + 0.5 * (UPPER - LOWER), np.array([10]))[0]
    lev_corner = disc.leverage_at(UPPER.copy(), np.array([10]))[0]
    assert 0.0 < lev_mid < lev_corner
    # P-03
    for bad in (np.array([0.1, 0.5, 10.0, 2e-3, 0.0]), np.array([0.4, 0.7, np.nan, 2e-3, 0.0]), np.array([0.4, 0.7])):
        q = trained.predict(bad, 5)
        kr, st = q.kr_at(np.array([0.5, 1.0]))
        assert q.status in ("outside_box", "invalid") and np.all(np.isnan(kr)) and set(st) == {q.status}
    # P-04
    lr = np.linspace(0.05, 1.4, 12)
    summary = {"status": "complete", "verification_status": "unverified", "case_id": "x", "lr": lr.tolist(),
               "kr": [[1.0, 1.0, k] for k in np.r_[1.0, 0.99, np.nan, [0.9] * 9]], "outside_small_strain": [False] * 12,
               "case": {"level": [6, 12, 12, 3, 3]}, "design": {"x": [0.4, 0.7, 10.0, 2e-3, 0.0]}}  # fmt: skip
    with pytest.raises(C.CurveQAError, match="non-finite"):
        C.from_summary(summary)


def test_law_evaluation_scales_to_monte_carlo_samples_of_n() -> None:
    """P5-CC-07: the arc is solved for all uncached exponents in one vectorised call."""
    import time

    from critflaw.fe.warp_law import WarpDeformationLaw
    from critflaw.option3.interface import sigma0_from_proof_stress

    rng = np.random.default_rng(1)
    n = rng.uniform(3.0, 20.0, 2000)
    s0 = rng.uniform(1e-3, 5e-3, 2000) * 207e9
    t = time.perf_counter()
    rp = np.asarray(WarpDeformationLaw(207e9, 0.3, s0, n).proof_stress_0p2())
    back = sigma0_from_proof_stress(rp, 207e9, n)
    assert time.perf_counter() - t < 30.0 and np.allclose(back, s0, rtol=1e-8)


def test_function_draws_match_predict_at_a_point_and_are_coherent(trained: EM.Emulator) -> None:
    """Phase 6 §2: a function draw fixes the GP path and the discretisation terms once; at a single x its
    distribution equals predict's; the between-step bounds bracket the curve monotonically."""
    x = np.array([0.4, 0.7, 10.0, 2e-3, 0.5])
    fd = EM.function_draws(trained, 1500, np.random.default_rng(4), n_features=2000)
    curves = np.vstack([fd.curve(i, x)[1] for i in range(fd.n_draws)])
    p = trained.predict(x, 1500, np.random.default_rng(5))
    assert curves.shape == p.kr_draws.shape
    sd = p.kr_draws.std(axis=0)
    assert np.all(np.abs(curves.mean(axis=0) - p.kr_mean) <= 4 * sd / np.sqrt(1500) + 2e-4)
    assert np.allclose(curves.std(axis=0), sd, rtol=0.15, atol=2e-4)
    # coherence: the same draw evaluated again gives the same curve; bounds bracket and order correctly
    lr, kr, st = fd.curve(3, x)
    assert st == "supported" and np.array_equal(kr, fd.curve(3, x)[1])
    mid = 0.5 * (lr[5] + lr[6])
    lo, hi, st2 = fd.bounds(3, x, mid)
    assert st2 == "supported" and kr[6] <= lo <= hi <= kr[5]  # inside the monotone bracket (convexity may tighten)
    assert fd.bounds(3, x, lr[-1] + 0.01)[2] == "beyond_support"
    assert fd.bounds(3, np.array([0.1, 0.5, 10.0, 2e-3, 0.0]), 0.5)[2] == "outside_box"


def test_step_bounds_bracket_a_convex_curve_tighter_than_monotone() -> None:
    from critflaw.option3.emulator import step_bounds

    lr = np.array([0.05, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1])
    r = 1.0 + 0.5 * lr**4  # convex J/J_e
    kr = r**-0.5
    for q in (0.7, 0.85, 0.95, 1.05):
        lo, hi = step_bounds(lr, kr, q)
        truth = (1.0 + 0.5 * q**4) ** -0.5
        p = int(np.searchsorted(lr, q))
        assert kr[p] <= lo <= truth <= hi <= kr[p - 1]
        if q > 0.8:
            assert hi - lo < 0.5 * (kr[p - 1] - kr[p])  # convexity tightens the bracket
    # a non-convex kink falls back to the monotone bracket
    r2 = r.copy()
    r2[5] += 0.05
    kr2 = r2**-0.5
    lo, hi = step_bounds(lr, kr2, 0.95)
    assert (lo, hi) == (kr2[6], kr2[5])
    # in the elastic range (J/J_e - 1 < floor) only the monotone bracket is used
    lo, hi = step_bounds(lr, kr, 0.3)
    assert (lo, hi) == (kr[2], kr[1])
