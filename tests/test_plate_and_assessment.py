"""Plate solutions and the assessment engine: limiting cases and invariants."""

from __future__ import annotations

import numpy as np
import pytest

import critflaw as cf
from critflaw.reserve import _scan
from critflaw.solutions import plate

DEEP, SURF = plate.THETA_DEEPEST, plate.THETA_SURFACE


def test_shallow_long_flaw_tends_to_edge_crack_value() -> None:
    mm, mb = plate.surface_flaw_mm_mb(1e-6, 1.0, 1.0, DEEP)
    assert mm == pytest.approx(1.13, abs=2e-3)
    assert mb == pytest.approx(float(mm), rel=1e-5)  # bending equals membrane for a vanishing depth


def test_semicircular_shallow_flaw_matches_penny_shape_factor() -> None:
    # a = c, a/B -> 0, deepest point: 1.04 * 2/pi from the exact elliptic integral, fit within 0.1%.
    mm, _ = plate.surface_flaw_mm_mb(1e-6, 1e-6, 1.0, DEEP)
    assert mm == pytest.approx(1.04 * 2 / np.pi, rel=1.5e-3)


@pytest.mark.parametrize("theta", [DEEP, SURF, 0.7])
@pytest.mark.parametrize("a_over_b", [0.05, 0.4, 0.8])
def test_aspect_ratio_branches_meet_at_a_equals_c(theta: float, a_over_b: float) -> None:
    b = 0.02
    a = a_over_b * b
    below = plate.surface_flaw_mm_mb(a, a * (1 + 1e-9), b, theta)
    above = plate.surface_flaw_mm_mb(a, a * (1 - 1e-9), b, theta)
    assert below[0] == pytest.approx(float(above[0]), rel=0.02)
    assert below[1] == pytest.approx(float(above[1]), rel=0.03)


def test_angle_simplifications() -> None:
    a, c, b = 4e-3, 10e-3, 20e-3
    ac, ab = a / c, a / b
    phi = np.sqrt(1 + 1.464 * ac**1.65)
    poly = (1.13 - 0.09 * ac) + (0.89 / (0.2 + ac) - 0.54) * ab**2 + (0.5 - 1 / (0.65 + ac) + 14 * (1 - ac) ** 24) * ab**4
    mm_d, mb_d = plate.surface_flaw_mm_mb(a, c, b, DEEP)
    mm_s, mb_s = plate.surface_flaw_mm_mb(a, c, b, SURF)
    assert mm_d == pytest.approx(poly / phi, rel=1e-13)  # g = 1, f_theta = 1
    assert mm_s == pytest.approx(poly * (1.1 + 0.35 * ab**2) * ac**0.5 / phi, rel=1e-13)
    h1 = 1 - 0.34 * ab - 0.11 * ac * ab
    h2 = 1 + (-1.22 - 0.12 * ac) * ab + (0.55 - 1.05 * ac**0.75 + 0.47 * ac**1.5) * ab**2
    assert mb_d == pytest.approx(h2 * float(mm_d), rel=1e-13)
    assert mb_s == pytest.approx(h1 * float(mm_s), rel=1e-13)


def test_finite_width_factors() -> None:
    assert plate.surface_flaw_fw(1e-3, 5e-3, 20e-3, 1e3) == pytest.approx(1.0, abs=1e-9)
    assert plate.surface_flaw_fw(5e-3, 20e-3, 20e-3, 0.1) == pytest.approx(
        np.cos(np.pi * 20e-3 / 0.1 * np.sqrt(0.25)) ** -0.5, rel=1e-14
    )
    assert plate.through_flaw_fw(0.01, 0.1) == pytest.approx(np.cos(np.pi * 0.1) ** -0.5, rel=1e-14)


def test_validity_limits() -> None:
    b, w = 20e-3, 1.0
    assert plate.surface_flaw_valid(5e-3, 10e-3, b, w)
    assert not plate.surface_flaw_valid(20e-3, 30e-3, b, w)  # a/B = 1
    assert not plate.surface_flaw_valid(12e-3, 5e-3, b, w)  # a/2c > 1
    # long shallow-aspect flaw: a/B limited to 1.25 (a/c + 0.6)
    a, c = 16e-3, 160e-3  # a/2c = 0.05, limit = 1.25 * 0.7 = 0.875 > 0.8 -> valid
    assert plate.surface_flaw_valid(a, c, b, w)
    a, c = 16e-3, 400e-3  # a/c = 0.04, limit 0.8 -> a/B = 0.8 not < 0.8
    assert not plate.surface_flaw_valid(a, c, b, 10.0)
    assert not plate.surface_flaw_valid(5e-3, 45e-3, b, 0.1)  # 2c/W = 0.9
    assert not plate.through_flaw_valid(0.05, 0.1)


def test_reference_stress_limits() -> None:
    # vanishing flaw: membrane alone gives Pm; bending alone gives 2/3 Pb (plastic shape factor 1.5)
    assert plate.surface_flaw_reference_stress(100e6, 0.0, 1e-9, 1e-3, 0.02, 1.0) == pytest.approx(100e6, rel=1e-6)
    assert plate.surface_flaw_reference_stress(0.0, 90e6, 1e-9, 1e-3, 0.02, 1.0) == pytest.approx(60e6, rel=1e-6)
    # pin-jointed is never less than normal restraint
    args = (100e6, 30e6, 8e-3, 20e-3, 0.02, 1.0)
    assert plate.surface_flaw_reference_stress(*args, pin_jointed=True) > plate.surface_flaw_reference_stress(*args)
    # narrow-plate branch of alpha
    al = (8e-3 / 0.02) * (40e-3 / 0.05)
    assert plate.surface_flaw_reference_stress(100e6, 0.0, 8e-3, 20e-3, 0.02, 0.05) == pytest.approx(
        100e6 / (1 - al), rel=1e-12
    )
    # through flaw under membrane stress: net section stress
    assert plate.through_flaw_reference_stress(100e6, 0.0, 0.01, 0.1) == pytest.approx(100e6 / 0.8, rel=1e-14)


def test_secondary_stress_has_no_width_factor() -> None:
    kp, ks = plate.stress_intensity(5e-3, 1.1, 0.9, 1.3, 100e6, 20e6, 50e6, 10e6)
    root = np.sqrt(np.pi * 5e-3)
    assert kp == pytest.approx(1.3 * (1.1 * 100e6 + 0.9 * 20e6) * root, rel=1e-14)
    assert ks == pytest.approx((1.1 * 50e6 + 0.9 * 10e6) * root, rel=1e-14)


# ---- assessment ---------------------------------------------------------------------

PLATE = cf.FlatPlate(thickness=25e-3, width=500e-3)
FLAW = cf.SurfaceFlaw(a=3e-3, two_c=15e-3)
STEEL = cf.Material(yield_strength=355e6, uts=490e6, modulus=207e9, toughness=120e6, yielding="continuous")


def test_primary_only_assessment_by_hand() -> None:
    p = cf.Assessment(PLATE, FLAW, STEEL, cf.StressState(pm=150e6, pb=40e6))
    r = p.run()
    a, c, b, w = 3e-3, 7.5e-3, 25e-3, 0.5
    sref = plate.surface_flaw_reference_stress(150e6, 40e6, a, c, b, w)
    assert r.lr == pytest.approx(float(sref) / 355e6, rel=1e-14)
    fw = plate.surface_flaw_fw(a, c, b, w)
    for i, theta in enumerate((DEEP, SURF)):
        mm, mb = plate.surface_flaw_mm_mb(a, c, b, theta)
        k = fw * (mm * 150e6 + mb * 40e6) * np.sqrt(np.pi * a)
        assert r.kr_by_position[i] == pytest.approx(float(k) / 120e6, rel=1e-13)
    assert r.kr == pytest.approx(float(np.max(r.kr_by_position)), rel=1e-14)
    assert r.acceptable and r.radial_margin > 1.0
    assert "k.plate.surface" in r.trace.equations and r.trace.edition == "2013+A1"
    assert any("no primary reference" in s for s in r.trace.citations())


def test_residual_stress_and_relaxation() -> None:
    base = cf.Assessment(PLATE, FLAW, STEEL, cf.StressState(pm=150e6, residual=cf.Residual.as_welded()))
    r = base.run()
    assert r.qm == pytest.approx(355e6)
    assert np.all(r.k_secondary > 0) and r.kr > base.with_(loads=cf.StressState(pm=150e6)).run().kr
    # relaxation: sref/flow below 0.4 leaves sy; above it reduces Qm, never below 0.4 sy
    relaxed = cf.Residual.as_welded(relaxation=True)
    low = base.with_(loads=cf.StressState(pm=100e6, residual=relaxed)).run()
    assert low.qm == pytest.approx(355e6)
    high = base.with_(loads=cf.StressState(pm=300e6, residual=relaxed)).run()
    flow = 0.5 * (355e6 + 490e6)
    assert high.qm == pytest.approx((1.4 - float(high.reference_stress) / flow) * 355e6, rel=1e-13)
    assert "residual_stress_relaxation_applied" in high.trace.flags
    very_high = base.with_(loads=cf.StressState(pm=450e6, residual=relaxed)).run()
    assert very_high.qm == pytest.approx(0.4 * 355e6, rel=1e-13)
    # PWHT levels
    assert cf.Residual.pwht("parallel").fraction == 0.2
    assert cf.Residual.pwht("transverse").fraction == 0.3
    assert cf.Residual.pwht("transverse", "550-580C").fraction == 0.4


def test_secondary_only_is_flagged() -> None:
    r = cf.Assessment(PLATE, FLAW, STEEL, cf.StressState(qm=200e6)).run()
    assert r.lr == 0.0 and "secondary_stress_acting_alone" in r.trace.flags
    assert r.radial_margin == pytest.approx(1.0 / float(r.kr), rel=1e-12)


def test_out_of_validity_is_never_extrapolated() -> None:
    bad = cf.Assessment(PLATE, cf.SurfaceFlaw(a=30e-3, two_c=80e-3), STEEL, cf.StressState(pm=100e6))
    with pytest.raises(cf.OutOfValidityError):
        bad.run()
    r = bad.run(on_invalid="mask")
    assert not r.valid and not r.acceptable and np.isnan(r.radial_margin)


def test_vectorised_run_matches_scalar_runs() -> None:
    rng = np.random.default_rng(1)
    n = 50
    a = rng.uniform(1e-3, 20e-3, n)
    two_c = rng.uniform(5e-3, 120e-3, n)
    sy = rng.uniform(300e6, 500e6, n)
    kmat = rng.uniform(40e6, 200e6, n)
    pm = rng.uniform(0.0, 300e6, n)
    disc = cf.Material(yield_strength=sy, uts=sy * 1.3, modulus=207e9, toughness=kmat, yielding="discontinuous")
    loads = cf.StressState(pm=pm, pb=30e6, residual=cf.Residual.as_welded(relaxation=True))
    vec = cf.Assessment(PLATE, cf.SurfaceFlaw(a, two_c), disc, loads, interaction="V").run(on_invalid="mask")
    assert 3 <= n - int(vec.valid.sum()) <= n - 5  # the sample must contain both valid and invalid flaws
    for i in range(n):
        one = cf.Assessment(
            PLATE,
            cf.SurfaceFlaw(a[i], two_c[i]),
            disc.with_(yield_strength=sy[i], uts=sy[i] * 1.3, toughness=kmat[i]),
            loads.with_(pm=pm[i]),
            interaction="V",
        ).run(on_invalid="mask")
        assert bool(one.valid) == bool(vec.valid[i])
        if one.valid:
            assert vec.radial_margin[i] == pytest.approx(float(one.radial_margin), rel=1e-12)
            assert bool(vec.acceptable[i]) == bool(one.acceptable)
            assert vec.kr[i] == pytest.approx(float(one.kr), rel=1e-13)


def test_through_flaw_assessment() -> None:
    p = cf.Assessment(PLATE, cf.ThroughFlaw(two_a=40e-3), STEEL, cf.StressState(pm=100e6))
    r = p.run()
    k = np.cos(np.pi * 0.02 / 0.5) ** -0.5 * 100e6 * np.sqrt(np.pi * 0.02)
    assert r.kr == pytest.approx(k / 120e6, rel=1e-13)
    assert r.lr == pytest.approx(100e6 / (1 - 0.08) / 355e6, rel=1e-13)
    assert r.positions == ("tip",)


# ---- reserve factors ----------------------------------------------------------------


def test_load_reserve_equals_radial_margin_without_secondary_stress() -> None:
    # Holds for any curve and any K solution, so it is a consistency check only (spec S2.7).
    p = cf.Assessment(PLATE, FLAW, STEEL, cf.StressState(pm=150e6, pb=40e6))
    rf = cf.reserve_factor_on_load(p)
    assert rf.unique and rf.value == pytest.approx(float(p.run().radial_margin), rel=1e-9)


def test_reserve_on_toughness_and_limiting_flaw_are_on_the_boundary() -> None:
    p = cf.Assessment(PLATE, FLAW, STEEL, cf.StressState(pm=150e6, residual=cf.Residual.as_welded()))
    rf = cf.reserve_factor_on_toughness(p)
    at = p.with_(material=STEEL.with_(toughness=120e6 / rf.value)).run()
    assert at.radial_margin == pytest.approx(1.0, abs=1e-6)
    lim = cf.limiting_flaw_size(p)
    at = p.with_(flaw=cf.SurfaceFlaw(a=lim.value, two_c=5.0 * lim.value)).run()
    assert at.radial_margin == pytest.approx(1.0, abs=1e-6)
    assert lim.value > 3e-3


def test_scan_reports_every_crossing_and_the_smallest_unacceptable_value() -> None:
    # acceptable on [0, 2), unacceptable on [2, 3), acceptable on [3, 7), unacceptable beyond
    def ok(x: float) -> bool:
        return x < 2.0 or 3.0 <= x < 7.0

    out = _scan(lambda x: 0 if ok(x) else 1, 0.1, 10.0, 200, log=False)
    assert not out.unique and len(out.crossings) == 3
    assert out.value == pytest.approx(2.0, abs=1e-9)
    assert not ok(out.value) and ok(float(np.nextafter(out.value, -np.inf)))  # the not-acceptable side
    assert out.range == (0.1, 10.0) and out.points == 200 and not out.log_spaced
    never = _scan(lambda x: 0, 0.1, 10.0, 50, log=False)
    assert not never.bounded and np.isinf(never.value) and never.unique
    always = _scan(lambda x: 1, 0.1, 10.0, 50, log=False)
    assert always.below_range and always.value < 0.1 and not always.limited_by_validity
    # not acceptable at the low end of the range, but acceptable further down: found by extending downward
    low = _scan(lambda x: 0 if x < 0.004 else 1, 0.1, 10.0, 50, log=True)
    assert not low.below_range and low.value == pytest.approx(0.004, rel=1e-12)
    assert low.crossings == (low.value,) and low.unique
    two = _scan(lambda x: 0 if (x < 2.0 or x >= 3.0) else 1, 0.1, 10.0, 200, log=False)
    assert not two.unique and len(two.crossings) == 2


# ---- added after round-2/3 sign-off findings ------------------------------------------


def test_through_flaw_in_wide_plate_pins_the_half_length_definition() -> None:
    # K = sigma sqrt(pi a) with a the HALF length: two_a = 40 mm must give sqrt(pi * 0.020).
    wide = cf.FlatPlate(thickness=25e-3, width=1e3)
    r = cf.Assessment(wide, cf.ThroughFlaw(two_a=40e-3), STEEL, cf.StressState(pm=100e6)).run()
    assert r.k_primary[0] == pytest.approx(100e6 * np.sqrt(np.pi * 0.020), rel=1e-9)


def test_surface_flaw_pins_depth_and_half_length_definitions() -> None:
    # Shallow semicircular flaw in a very wide, thick plate: deepest-point K -> 1.04 (2/pi) sigma sqrt(pi a),
    # which holds only if a is the depth and two_c = 2a is the full surface length.
    big = cf.FlatPlate(thickness=10.0, width=1e3)
    r = cf.Assessment(big, cf.SurfaceFlaw(a=5e-3, two_c=10e-3), STEEL, cf.StressState(pm=100e6)).run()
    assert r.k_primary[0] == pytest.approx(1.04 * (2 / np.pi) * 100e6 * np.sqrt(np.pi * 5e-3), rel=2e-3)


def test_reference_stress_tends_to_net_section_as_flaw_fills_a_narrow_plate() -> None:
    # W < 2(c + B): alpha'' = (a/B)(2c/W) is the flawed fraction of the cross-section (rectangular idealisation).
    assert plate.surface_flaw_reference_stress(100e6, 0.0, 0.019, 0.02, 0.02, 0.05) == pytest.approx(
        100e6 / (1 - 0.95 * 0.8), rel=1e-12
    )


def test_vanishing_primary_load_is_not_the_secondary_only_case() -> None:
    # Spec S2.5: chi is load independent, so Kr at 1e-12 of nominal load uses the rho route,
    # while exactly zero primary load uses the plastic-zone-corrected K. The jump is documented.
    nominal = cf.Assessment(PLATE, FLAW, STEEL, cf.StressState(pm=150e6, qm=200e6)).run()
    tiny = cf.Assessment(PLATE, FLAW, STEEL, cf.StressState(pm=150e6 * 1e-12, qm=200e6)).run()
    none = cf.Assessment(PLATE, FLAW, STEEL, cf.StressState(qm=200e6)).run()
    chi_nominal = nominal.k_secondary[0] * nominal.lr / nominal.k_primary[0]
    chi_tiny = tiny.k_secondary[0] * tiny.lr / tiny.k_primary[0]
    assert chi_tiny == pytest.approx(float(chi_nominal), rel=1e-9)
    assert "secondary_stress_acting_alone" not in tiny.trace.flags
    assert "secondary_stress_acting_alone" in none.trace.flags
    assert tiny.kr != pytest.approx(float(none.kr), rel=1e-3)


def test_negative_primary_k_from_bending_on_a_deep_flaw_is_zeroed_and_flagged() -> None:
    deep = cf.SurfaceFlaw(a=22e-3, two_c=44e-3)  # a/c = 1, a/B = 0.88: Mb < 0 at the deepest point
    r = cf.Assessment(PLATE, deep, STEEL, cf.StressState(pb=100e6, qm=100e6)).run()
    assert "negative_primary_k_set_to_zero" in r.trace.flags
    assert r.k_primary[0] == 0.0 and r.lr > 0.0
    # primary load is present, so this is not the secondary-only row; rho takes its cap
    assert "secondary_stress_acting_alone" not in r.trace.flags
    assert r.kr_by_position[0] == pytest.approx(float(r.k_secondary[0]) / 120e6 + 0.25, rel=1e-12)


def test_unknown_yielding_is_never_less_conservative_than_either_form() -> None:
    for sy, strain in ((355e6, None), (355e6, 0.002), (930e6, None)):
        kinds = {
            k: cf.Option1Curve(sy, sy * 1.2, 207e9, k, luders_strain=strain)  # type: ignore[arg-type]
            for k in ("continuous", "discontinuous", "unknown")
        }
        lr = np.linspace(0.0, 1.3, 1301)
        assert np.all(kinds["unknown"](lr) <= kinds["continuous"](lr) + 1e-15)
        assert np.all(kinds["unknown"](lr[lr >= 1.0]) <= kinds["discontinuous"](lr[lr >= 1.0]) + 1e-15)
        assert np.all(np.diff(kinds["unknown"](lr)) <= 1e-13)


def test_room_temperature_yield_applies_to_the_unrelaxed_term_only() -> None:
    res = cf.Residual.as_welded(relaxation=True, room_temperature_yield=330e6)
    low = cf.Assessment(PLATE, FLAW, STEEL, cf.StressState(pm=50e6, residual=res)).run()
    assert low.qm == pytest.approx(330e6)
    with pytest.raises(ValueError):
        cf.Residual(fraction=0.2, relaxation=True)


def test_yielding_is_required() -> None:
    with pytest.raises(TypeError):
        cf.Material(yield_strength=355e6, uts=490e6, modulus=207e9, toughness=120e6)  # type: ignore[call-arg]


# ---- added after the phase-1 code reviews (mutation survivors and reviewer findings) -----


def _run(loads: cf.StressState, flaw: cf.SurfaceFlaw | cf.ThroughFlaw = FLAW, **kw: object) -> cf.Result:
    return cf.Assessment(PLATE, flaw, STEEL, loads, **kw).run()  # type: ignore[arg-type]


def test_residual_stress_options_reach_the_engine() -> None:
    assert _run(cf.StressState(pm=50e6, residual=cf.Residual.pwht("parallel"))).qm == pytest.approx(0.2 * 355e6)
    assert _run(cf.StressState(pm=50e6, residual=cf.Residual.pwht("transverse"))).qm == pytest.approx(0.3 * 355e6)
    assert _run(cf.StressState(pm=50e6, residual=cf.Residual.pwht("transverse", "550-580C"))).qm == pytest.approx(
        0.4 * 355e6
    )
    assert _run(cf.StressState(pm=50e6, residual=cf.Residual.as_welded(appropriate_yield=300e6))).qm == pytest.approx(300e6)
    # a user secondary stress and a residual stress add
    assert _run(cf.StressState(pm=50e6, qm=40e6, residual=cf.Residual.as_welded())).qm == pytest.approx(395e6)


def test_room_temperature_yield_is_not_used_in_the_relaxed_term() -> None:
    res = cf.Residual.as_welded(relaxation=True, room_temperature_yield=400e6)
    r = _run(cf.StressState(pm=300e6, residual=res))  # reference stress above 0.4 of flow strength
    flow = 0.5 * (355e6 + 490e6)
    assert float(r.reference_stress) / flow > 0.4
    assert r.qm == pytest.approx((1.4 - float(r.reference_stress) / flow) * 355e6, rel=1e-13)
    # appropriate yield and tensile strength both enter the relaxed term
    res2 = cf.Residual.as_welded(appropriate_yield=300e6, appropriate_uts=500e6, relaxation=True)
    r2 = _run(cf.StressState(pm=300e6, residual=res2))
    assert r2.qm == pytest.approx((1.4 - float(r2.reference_stress) / 400e6) * 300e6, rel=1e-13)


def test_plane_strain_option_reaches_the_engine() -> None:
    stress = _run(cf.StressState(qm=200e6))
    strain = _run(cf.StressState(qm=200e6), plane_stress=False)
    ks, a = float(stress.k_secondary[0]), 3e-3
    for r, beta in ((stress, 1.0), (strain, 3.0)):
        expected = np.sqrt((a + (ks / 355e6) ** 2 / (2 * np.pi * beta)) / a) * ks / 120e6
        assert r.kr_by_position[0] == pytest.approx(expected, rel=1e-13)
    assert strain.trace.options["plastic_zone_correction"] == "plane strain"
    assert stress.trace.options["plastic_zone_correction"] == "plane stress"


def test_through_flaw_with_bending_and_secondary_by_hand() -> None:
    r = _run(cf.StressState(pm=100e6, pb=40e6, qm=30e6, qb=10e6), cf.ThroughFlaw(two_a=40e-3), interaction="V")
    fw, root = np.cos(np.pi * 0.02 / 0.5) ** -0.5, np.sqrt(np.pi * 0.02)
    assert r.k_primary[0] == pytest.approx(fw * 140e6 * root, rel=1e-13)  # Mm = Mb = 1
    assert r.k_secondary[0] == pytest.approx(40e6 * root, rel=1e-13)  # no width factor on secondary
    assert r.reference_stress == pytest.approx((40e6 + np.sqrt(40e6**2 + 9 * 100e6**2)) / (3 * (1 - 0.08)), rel=1e-13)
    assert r.trace.options["interaction"] == "V" and "fad.v.simplified" in r.trace.equations


@pytest.mark.parametrize(
    ("a_over_2c", "a_over_b", "expected"),
    [
        (0.08, 0.97, False),  # a/2c <= 0.1: limit 1.25 (0.16 + 0.6) = 0.95
        (0.08, 0.94, True),
        (0.11, 0.97, True),  # a/2c > 0.1: limit a/B < 1
        (0.11, 1.0, False),
        (0.75, 0.5, True),  # deep-aspect branch is inside validity
        (1.0, 0.5, True),
        (1.01, 0.5, False),
    ],
)
def test_validity_boundaries(a_over_2c: float, a_over_b: float, expected: bool) -> None:
    b = 0.02
    a = a_over_b * b
    assert bool(plate.surface_flaw_valid(a, a / (2 * a_over_2c), b, 100.0)) is expected


@pytest.mark.parametrize("method", ["rho", "V"])
def test_chi_flag_is_raised_by_the_engine(method: str) -> None:
    # chi is close to qm / yield for a small flaw, so chi > 4 needs an (artificially) large secondary stress
    high = _run(cf.StressState(pm=5e6, qm=1600e6), interaction=method)
    assert "chi_above_4_simplified_interaction_over_conservative" in high.trace.flags
    low = _run(cf.StressState(pm=150e6, qm=30e6), interaction=method)
    assert "chi_above_4_simplified_interaction_over_conservative" not in low.trace.flags


def test_kp_zero_with_primary_load_takes_the_limiting_interaction() -> None:
    from critflaw.fad import fracture_ratio

    rho = fracture_ratio(0.0, 30e6, 0.5, 100e6, 5e-3, 355e6, method="rho")
    v = fracture_ratio(0.0, 30e6, 0.5, 100e6, 5e-3, 355e6, method="V")
    assert not rho.secondary_only and np.isinf(rho.chi) and rho.chi_above_limit
    assert rho.kr == pytest.approx(0.3 + 0.25, rel=1e-14)
    assert v.kr == pytest.approx((3.1 - 2 * 0.5) * 0.3, rel=1e-14)
    # self-balancing term is added in the secondary-only row too, and must be non-negative
    alone = fracture_ratio(0.0, 30e6, 0.0, 100e6, 5e-3, 355e6)
    with_sb = fracture_ratio(0.0, 30e6, 0.0, 100e6, 5e-3, 355e6, k_self_balancing=7e6)
    assert with_sb.kr == pytest.approx(float(alone.kr) + 0.07, rel=1e-14)
    with pytest.raises(ValueError):
        fracture_ratio(10e6, 0.0, 0.1, 100e6, 5e-3, 355e6, k_self_balancing=-1e6)
    with pytest.raises(ValueError):
        fracture_ratio(40e6, 30e6, np.nan, 100e6, 5e-3, 355e6)
    with pytest.raises(ValueError):
        fracture_ratio(40e6, 30e6, 0.5, np.nan, 5e-3, 355e6)


def test_invalid_samples_are_blanked_in_every_field() -> None:
    flaw = cf.SurfaceFlaw(a=np.array([3e-3, 30e-3]), two_c=np.array([15e-3, 80e-3]))  # second: a/B = 1.2
    loads = cf.StressState(pm=np.array([1e8, 4e8]), residual=cf.Residual.as_welded(relaxation=True))
    r = cf.Assessment(PLATE, flaw, STEEL, loads).run(on_invalid="mask")
    assert list(r.valid) == [True, False] and list(r.acceptable)[1] is np.False_
    for name in ("lr", "kr", "radial_margin", "reference_stress", "qm", "lr_max"):
        value = getattr(r, name)
        assert np.isfinite(value[0]) and np.isnan(value[1]), name
    for name in ("kr_by_position", "k_primary", "k_secondary"):
        value = getattr(r, name)
        assert np.all(np.isfinite(value[:, 0])) and np.all(np.isnan(value[:, 1])), name
    assert r.governing_position[1] == -1 and r.exit[1] == "invalid" and r.exit[0] in ("curve", "cutoff")
    # the trace is built from valid samples only: the only valid sample here has no secondary stress
    r2 = cf.Assessment(PLATE, flaw, STEEL, cf.StressState(pm=np.array([1e8, 0.0]), qm=np.array([0.0, 2e8]))).run(
        on_invalid="mask"
    )
    assert "fad.rho.simplified" not in r2.trace.equations
    assert "secondary_stress_acting_alone" not in r2.trace.flags
    with pytest.raises(ValueError):
        cf.Assessment(PLATE, flaw, STEEL, loads).run(on_invalid="msk")  # type: ignore[arg-type]


def test_shapes_and_broadcasting() -> None:
    # array-valued Lueders strain or residual parameter with otherwise scalar inputs
    disc = STEEL.with_(yielding="discontinuous", luders_strain=np.array([0.01, 0.02]))
    assert cf.Assessment(PLATE, FLAW, disc, cf.StressState(pm=100e6)).run().radial_margin.shape == (2,)
    res = cf.Residual.as_welded(appropriate_yield=np.array([300e6, 355e6]))
    r = cf.Assessment(PLATE, FLAW, STEEL, cf.StressState(pm=100e6, residual=res)).run()
    assert r.qm.shape == (2,) and list(r.qm) == [300e6, 355e6]
    # (2, 1) against (3,) gives (2, 3) in every field, and matches scalar runs
    a = np.array([[2e-3], [6e-3]])
    pm = np.array([50e6, 150e6, 250e6])
    grid = cf.Assessment(PLATE, cf.SurfaceFlaw(a, 20e-3), STEEL, cf.StressState(pm=pm, qm=100e6)).run()
    for name in ("lr", "kr", "radial_margin", "acceptable", "valid", "governing_position", "exit", "qm", "lr_max"):
        assert getattr(grid, name).shape == (2, 3), name
    assert grid.kr_by_position.shape == (2, 2, 3)
    for i in range(2):
        for j in range(3):
            one = cf.Assessment(
                PLATE, cf.SurfaceFlaw(float(a[i, 0]), 20e-3), STEEL, cf.StressState(pm=float(pm[j]), qm=100e6)
            ).run()
            assert grid.radial_margin[i, j] == pytest.approx(float(one.radial_margin), rel=1e-13)
            assert grid.exit[i, j] == one.exit.item() and grid.governing_position[i, j] == int(one.governing_position)
    # length-1 arrays behave as arrays
    assert cf.Assessment(PLATE, cf.SurfaceFlaw(np.array([3e-3]), 15e-3), STEEL, cf.StressState(pm=1e8)).run().kr.shape == (1,)


def test_step_exit_is_reported_by_the_engine() -> None:
    disc = STEEL.with_(yielding="discontinuous")
    r = cf.Assessment(PLATE, FLAW, disc, cf.StressState(pm=150e6)).run()
    assert r.exit.item() == "step" and float(r.radial_margin) * float(r.lr) == pytest.approx(1.0, rel=1e-9)


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        STEEL.with_(yielding="discontinuous", luders_strain=np.nan)
    with pytest.raises(ValueError):
        STEEL.with_(yielding="discontinuous", luders_strain=-1.0)
    with pytest.raises(ValueError):
        STEEL.with_(luders_strain=0.01)  # meaningless for continuous yielding
    with pytest.raises(ValueError):
        cf.Material(1000e6, 1100e6, 207e9, 1e8, yielding="unknown")  # estimate not applicable
    with pytest.raises(TypeError):
        cf.Material("355e6", 490e6, 207e9, 1e8, yielding="continuous")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        cf.Residual(relaxation="no")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        cf.Assessment(PLATE, FLAW, STEEL, cf.StressState(pm=1e8), plane_stress="no")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        cf.Residual.as_welded(appropriate_yield=400e6, appropriate_uts=100e6, relaxation=True)
    with pytest.raises(ValueError):
        cf.Residual.as_welded(appropriate_uts=500e6)  # unused without relaxation


def test_reserve_functions_report_what_they_could_not_resolve() -> None:
    # not acceptable even with no primary load: no reserve on load exists
    brittle = STEEL.with_(toughness=5e6)
    none = cf.reserve_factor_on_load(cf.Assessment(PLATE, FLAW, brittle, cf.StressState(pm=100e6, qm=300e6)))
    assert none.unacceptable_at_zero and none.value == 0.0 and none.crossings == ()
    # collapse governs: no reserve on toughness
    collapse = cf.reserve_factor_on_toughness(cf.Assessment(PLATE, FLAW, STEEL, cf.StressState(pm=600e6)))
    assert collapse.below_range
    # a true answer far below the default range is found by extending downward
    tiny = cf.Assessment(PLATE, cf.ThroughFlaw(40e-3), STEEL.with_(toughness=1e20), cf.StressState(pm=355e12))
    rf = cf.reserve_factor_on_load(tiny)
    assert not rf.below_range and rf.value == pytest.approx((355 + 490) / (2 * 355) * 0.92 / 1e6, rel=1e-9)
    assert rf.range == (1e-3, 100.0) and rf.points == 400 and rf.log_spaced


def test_limiting_flaw_size_distinguishes_validity_limit_from_failure() -> None:
    # tough material, low load: nothing fails before the solution leaves validity
    tough = STEEL.with_(toughness=1e9)
    lim = cf.limiting_flaw_size(cf.Assessment(PLATE, FLAW, tough, cf.StressState(pm=20e6)))
    assert lim.limited_by_validity and lim.value == pytest.approx(25e-3, rel=1e-9)
    # ordinary case: a real crossing, on the not-acceptable side
    p = cf.Assessment(PLATE, FLAW, STEEL, cf.StressState(pm=150e6, residual=cf.Residual.as_welded()))
    real = cf.limiting_flaw_size(p)
    assert not real.limited_by_validity
    assert not p.with_(flaw=cf.SurfaceFlaw(real.value, 5 * real.value)).run().acceptable
    below = float(np.nextafter(real.value, 0.0))
    assert p.with_(flaw=cf.SurfaceFlaw(below, 5 * below)).run().acceptable
    # through-thickness flaw: the scanned size is the total length 2a
    t = cf.Assessment(PLATE, cf.ThroughFlaw(10e-3), STEEL, cf.StressState(pm=150e6))
    lim_t = cf.limiting_flaw_size(t)
    assert not t.with_(flaw=cf.ThroughFlaw(lim_t.value)).run().acceptable
    assert t.with_(flaw=cf.ThroughFlaw(float(np.nextafter(lim_t.value, 0.0)))).run().acceptable


def test_report_lists_verification_status_and_exclusions() -> None:
    r = _run(cf.StressState(pm=150e6, residual=cf.Residual.as_welded()))
    text = r.report()
    assert "SELF-CONSISTENT ONLY" in text and "not checked against a primary reference" in text
    assert "Not accounted for:" in text and "model uncertainty of the stress intensity factor solution" in text
    assert set(r.trace.not_checked_against_primary()) == set(r.trace.equations)
    assert "k.plate.surface" not in r.trace.self_consistent_only()
    assert "fad.option1.continuous" in r.trace.self_consistent_only()


def test_load_reserve_checks_exactly_zero_primary_load() -> None:
    # Secondary stress alone fails (plastic-zone-corrected K), but a vanishing primary load is acceptable (rho route).
    qm = 100e6 / np.sqrt(np.pi * 0.02)
    p = cf.Assessment(PLATE, cf.ThroughFlaw(two_a=40e-3), STEEL, cf.StressState(pm=100e6, qm=qm))
    assert not p.with_(loads=cf.StressState(qm=qm)).run().acceptable
    assert p.with_(loads=cf.StressState(pm=100e6 * 1e-3, qm=qm)).run().acceptable
    rf = cf.reserve_factor_on_load(p)
    assert rf.unacceptable_at_zero and rf.value == 0.0 and len(rf.crossings) >= 1
    # the earlier "brittle" case is unacceptable at zero as well
    ok = cf.reserve_factor_on_load(cf.Assessment(PLATE, FLAW, STEEL, cf.StressState(pm=150e6, pb=40e6)))
    assert not ok.unacceptable_at_zero and ok.value > 1.0


# ---- round-2 sign-off: tie-breaking of the governing position, report, labels ------------


def test_governing_position_on_tied_margins_is_the_larger_kr() -> None:
    disc = cf.Material(355e6, 490e6, 207e9, 60e6, yielding="discontinuous")
    r = cf.Assessment(PLATE, cf.SurfaceFlaw(3e-3, 6e-3), disc, cf.StressState(pm=150e6)).run()
    assert r.exit.item() == "step"
    assert r.kr_by_position[1] > r.kr_by_position[0]
    assert r.positions[int(r.governing_position)] == "surface"
    assert r.kr == pytest.approx(float(r.kr_by_position[1]), rel=1e-14)


def test_report_for_single_sample_arrays_and_its_content() -> None:
    for a in (3e-3, np.array([3e-3]), np.array([[3e-3]])):
        r = cf.Assessment(PLATE, cf.SurfaceFlaw(a, 15e-3), STEEL, cf.StressState(pm=150e6)).run()
        text = r.report()
        assert "Evaluations: 1; acceptable: 1; outside validity: 0" in text
        assert f"Lr = {r.lr.item():.6g}, Kr = {r.kr.item():.6g}" in text and "governing position: deepest" in text
    # self-consistent-only label is attached to exactly the right equations
    lines = {line.strip() for line in text.splitlines()}
    surface = next(line for line in lines if "Semi-elliptical surface flaw" in line)
    line_fad = next(line for line in lines if "Option 1 assessment line, continuous" in line)
    assert "SELF-CONSISTENT ONLY" not in surface and "SELF-CONSISTENT ONLY" in line_fad
    # an invalid sample prints counts only
    bad = cf.Assessment(PLATE, cf.SurfaceFlaw(30e-3, 80e-3), STEEL, cf.StressState(pm=1e8)).run(on_invalid="mask")
    assert "acceptable: 0; outside validity: 1" in bad.report() and "Lr =" not in bad.report()
    assert "out_of_validity_samples_present" in bad.trace.flags


def test_trace_equations_and_flags_come_from_valid_samples_only() -> None:
    flaw = cf.SurfaceFlaw(a=np.array([3e-3, 30e-3]), two_c=np.array([15e-3, 80e-3]))  # second is invalid
    # only the invalid sample has a large secondary stress / bending on a deep flaw
    r = cf.Assessment(PLATE, flaw, STEEL, cf.StressState(pm=np.array([1e8, 1e6]), qm=np.array([0.0, 3e9]))).run(
        on_invalid="mask"
    )
    assert "fad.rho.simplified" not in r.trace.equations
    assert "chi_above_4_simplified_interaction_over_conservative" not in r.trace.flags
    assert "negative_primary_k_set_to_zero" not in r.trace.flags
    assert _run(cf.StressState(pm=50e6, residual=cf.Residual.pwht("parallel"))).trace.equations.count("residual.pwht") == 1
    assert "residual.as_welded" in _run(cf.StressState(pm=50e6, residual=cf.Residual.as_welded())).trace.equations
    unknown = cf.Assessment(PLATE, FLAW, STEEL.with_(yielding="unknown"), cf.StressState(pm=1e8)).run()
    assert {"fad.option1.continuous", "fad.option1.discontinuous", "material.luders_strain"} <= set(unknown.trace.equations)


def test_exit_labels_near_the_step_and_the_cut_off() -> None:
    from critflaw.fad import radial_margin
    from critflaw.fad.margins import EXIT_CURVE, EXIT_CUTOFF

    disc = cf.Option1Curve(355e6, 490e6, 207e9, "discontinuous")
    lr_max = float(disc.lr_max)
    # exits through the curve at Lr = 1.01 (just after the step) and at 0.99 Lr,max (just before the cut-off)
    for lr_exit in (1.01, 0.99 * lr_max):
        kr_exit = float(disc(lr_exit))
        f, code = radial_margin(0.5 * lr_exit, 0.5 * kr_exit, disc)
        assert f == pytest.approx(2.0, rel=1e-9) and code == EXIT_CURVE
    assert radial_margin(0.5, 1e-6, disc)[1] == EXIT_CUTOFF


def test_more_validation_and_reported_chi() -> None:
    from critflaw.fad import fracture_ratio, radial_margin

    cont = cf.Option1Curve(355e6, 490e6, 207e9, "continuous")
    with pytest.raises(ValueError):
        radial_margin(0.5, -0.1, cont)
    with pytest.raises(TypeError):
        cf.FlatPlate(thickness=True, width=0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        cf.Option1Curve(355e6, 490e6, 207e9, "discontinuous", luders_strain=0.0)
    with pytest.raises(ValueError):
        cf.Option1Curve(np.inf, 490e6, 207e9, "continuous")
    r = fracture_ratio(0.0, 0.0, 0.5, 100e6, 5e-3, 355e6)
    assert r.kr == 0.0 and r.chi == 0.0 and not r.chi_above_limit and not r.secondary_only
    assert not fracture_ratio(10e6, 80e6, 0.5, 100e6, 5e-3, 355e6).chi_above_limit  # chi exactly 4
    assert fracture_ratio(10e6, 80.001e6, 0.5, 100e6, 5e-3, 355e6).chi_above_limit
    res = cf.Residual.as_welded(room_temperature_yield=np.array([330e6, 360e6]))
    assert list(cf.Assessment(PLATE, FLAW, STEEL, cf.StressState(pm=1e8, residual=res)).run().qm) == [330e6, 360e6]


def test_reserve_functions_require_a_scalar_problem() -> None:
    vec = cf.Assessment(PLATE, cf.SurfaceFlaw(np.array([2e-3, 3e-3]), 15e-3), STEEL, cf.StressState(pm=1e8))
    with pytest.raises(ValueError):
        cf.reserve_factor_on_load(vec)
    one = cf.Assessment(PLATE, cf.SurfaceFlaw(np.array([3e-3]), np.array([15e-3])), STEEL, cf.StressState(pm=150e6))
    assert cf.limiting_flaw_size(one).value == pytest.approx(
        cf.limiting_flaw_size(cf.Assessment(PLATE, FLAW, STEEL, cf.StressState(pm=150e6))).value, rel=1e-12
    )


@pytest.mark.parametrize("flaw_kind", ["surface", "through"])
@pytest.mark.parametrize("yielding", ["continuous", "discontinuous", "unknown"])
@pytest.mark.parametrize("interaction", ["rho", "V"])
@pytest.mark.parametrize("relaxation", [False, True])
@pytest.mark.parametrize("restraint", ["normal", "pin_jointed"])
def test_vectorised_equals_scalar_in_every_field(
    flaw_kind: str, yielding: str, interaction: str, relaxation: bool, restraint: str
) -> None:
    rng = np.random.default_rng(7)
    n = 40
    sy = rng.uniform(300e6, 500e6, n)
    kmat = rng.uniform(40e6, 200e6, n)
    pm = np.where(rng.random(n) < 0.15, 0.0, rng.uniform(0.0, 300e6, n))
    pb = np.where(pm == 0.0, 0.0, rng.uniform(0.0, 60e6, n))
    qm = rng.uniform(-20e6, 150e6, n)
    size = rng.uniform(1e-3, 30e-3, n)
    two_c = rng.uniform(5e-3, 120e-3, n)

    def problem(i: int | None) -> cf.Assessment:
        pick = (lambda v: v) if i is None else (lambda v: float(v[i]))
        flaw = cf.SurfaceFlaw(pick(size), pick(two_c)) if flaw_kind == "surface" else cf.ThroughFlaw(pick(size) * 30)
        material = cf.Material(pick(sy), pick(sy * 1.3), 207e9, pick(kmat), yielding=yielding)  # type: ignore[arg-type]
        loads = cf.StressState(pick(pm), pick(pb), pick(qm), 5e6, cf.Residual.as_welded(relaxation=relaxation))
        return cf.Assessment(PLATE, flaw, material, loads, bending_restraint=restraint, interaction=interaction)  # type: ignore[arg-type]

    vec = problem(None).run(on_invalid="mask")
    assert vec.radial_margin.shape == (n,)
    if flaw_kind == "surface":
        assert 3 <= int(np.sum(~vec.valid)) <= n - 10
    for i in range(n):
        one = problem(i).run(on_invalid="mask")
        assert bool(one.valid) == bool(vec.valid[i]) and bool(one.acceptable) == bool(vec.acceptable[i])
        assert one.exit.item() == vec.exit[i] and int(one.governing_position) == int(vec.governing_position[i])
        for name in ("lr", "kr", "radial_margin", "reference_stress", "qm", "lr_max"):
            np.testing.assert_allclose(getattr(vec, name)[i], getattr(one, name), rtol=1e-12, equal_nan=True, err_msg=name)
        for name in ("kr_by_position", "k_primary", "k_secondary"):
            np.testing.assert_allclose(getattr(vec, name)[:, i], getattr(one, name), rtol=1e-12, atol=1e-6, equal_nan=True)


def test_reserve_functions_reject_vector_problems_with_invalid_samples() -> None:
    mixed = cf.SurfaceFlaw(a=np.array([0.003, 0.030]), two_c=np.array([0.015, 0.080]))  # second is invalid
    invalid = cf.SurfaceFlaw(a=np.array([0.030, 0.031]), two_c=np.array([0.080, 0.080]))
    for flaw in (mixed, invalid):
        problem = cf.Assessment(PLATE, flaw, STEEL, cf.StressState(pm=100e6))
        for fn in (cf.reserve_factor_on_load, cf.reserve_factor_on_toughness, cf.limiting_flaw_size):
            with pytest.raises(ValueError, match="scalar assessment"):
                fn(problem)


def test_exit_labels_very_close_to_the_step_and_the_cut_off() -> None:
    from critflaw.fad import radial_margin
    from critflaw.fad.margins import EXIT_CURVE

    disc = cf.Option1Curve(355e6, 490e6, 207e9, "discontinuous")
    for lr_exit in (1.0 + 1e-6, float(disc.lr_max) * (1.0 - 1e-6)):
        f, code = radial_margin(0.5 * lr_exit, 0.5 * float(disc(lr_exit)), disc)
        assert f == pytest.approx(2.0, rel=1e-9) and code == EXIT_CURVE, lr_exit


def test_report_numbers_are_the_right_numbers() -> None:
    disc = cf.Material(355e6, 490e6, 207e9, 60e6, yielding="discontinuous")
    r = cf.Assessment(PLATE, cf.SurfaceFlaw(3e-3, 6e-3), disc, cf.StressState(pm=150e6)).run()
    text = r.report()
    assert f"radial margin = {r.radial_margin.item():.6g} (exit: step)" in text
    assert "governing position: surface" in text and f"Kr = {r.kr_by_position[1]:.6g}" in text
    assert r.radial_margin.item() != pytest.approx(r.kr.item())
    failing = cf.Assessment(PLATE, FLAW, STEEL, cf.StressState(pm=500e6)).run()
    assert "Evaluations: 1; acceptable: 0; outside validity: 0" in failing.report()
    many = cf.Assessment(PLATE, FLAW, STEEL, cf.StressState(pm=np.array([100e6, 200e6, 500e6]))).run()
    assert "Evaluations: 3; acceptable: 2; outside validity: 0" in many.report()
    for bad in (dict(uts=np.inf), dict(modulus=np.nan)):
        with pytest.raises(ValueError):
            cf.Option1Curve(**{"yield_strength": 355e6, "uts": 490e6, "modulus": 207e9, "yielding": "continuous", **bad})
