"""Embedded flaws, cylinder tables and cylinder reference stresses (theory manual, chapter 02)."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import critflaw as cf
from critflaw import prob
from critflaw.fatigue import ExtremeLoad, GrowthLaw, Inspection, Spectrum, TimeDependentAssessment
from critflaw.fatigue import growth as G
from critflaw.fatigue.growth import Grower, GrowthState
from critflaw.fatigue.laws import coefficient_to_si, dk_to_si
from critflaw.prob.inspection import ConstantPoD, StepPoD
from critflaw.solutions import cylinder_tables as T
from critflaw.solutions import embedded as E
from critflaw.solutions import plate

HERE = Path(__file__).parent


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # reviewed oracle / data file
    return module


ORACLE_E = _load("partner_reference_partB_embedded", HERE / "oracles" / "partner_reference_partB_embedded.py")
SECOND = _load("cylinder_tables_second_copy", HERE / "data" / "cylinder_tables_second_copy.py")

B, W = 25e-3, 0.5
NMM = float(dk_to_si(1.0))


def _grid() -> list[tuple[float, float, float]]:
    """Valid embedded states (a, c, p) spanning both branches, the thresholds and the width limit."""
    out = []
    for a_2c in (0.02, 0.1, 0.1 + 1e-9, 0.3, 0.5, 0.5 + 1e-9, 0.8, 1.0):
        for a_b in (0.02, 0.1, 0.125, 0.125 + 1e-9, 0.2, 0.3):
            a = a_b * B
            c = a / (2.0 * a_2c)
            for p_b in (0.05, 0.1841, 0.1841 + 1e-9, 0.3, (1 - 2 * a_b) / 2 * 0.999):
                p = p_b * B
                if E.embedded_flaw_valid(a, c, p, B, W)[()]:
                    out.append((a, c, p))
    return out


# ---- oracle comparison (partner reference, kind (c) on the transcribed equations) -------------


def test_embedded_mm_mb_fw_against_the_partner_oracle() -> None:
    states = _grid()
    assert len(states) > 100
    for a, c, p in states:
        for theta in (E.THETA_NEAR_END, E.THETA_MAJOR_AXIS, 0.3, -1.1):
            mine = float(E.embedded_flaw_mm(a, c, p, theta)[()])
            ref = ORACLE_E.membrane(a, c, p, B, W, theta)
            assert mine == pytest.approx(ref, rel=1e-13)
        assert float(E.embedded_flaw_fw(a, c, p, W)[()]) == pytest.approx(ORACLE_E.finite_width(a, c, p, W), rel=1e-14)
        if a / (2 * c) <= 0.5:
            assert float(E.embedded_flaw_mb(a, c, p, B)[()]) == pytest.approx(ORACLE_E.bending(a, c, p, B), rel=1e-13)
        for pm, pb in ((100e6, 0.0), (100e6, 50e6), (0.0, 80e6)):
            mine_s = float(E.embedded_flaw_reference_stress(pm, pb, a, c, p, B, W)[()])
            assert mine_s == pytest.approx(ORACLE_E.embedded_reference(a, c, p, B, W, pm, pb), rel=1e-13)
    # the extended embedded reference stress on its own range
    for a_b, p_b, pm, pb in ((0.05, 0.1, 100e6, 20e6), (0.1, 0.05, 100e6, 0.0), (0.02, 0.3, 50e6, 50e6)):
        a, p = a_b * B, p_b * B
        mine_x = float(E.extended_embedded_reference_stress(pm, pb, a, p, B)[()])
        assert mine_x == pytest.approx(ORACLE_E.extended_reference(a, p, B, pm, pb), rel=1e-13)
    assert np.isnan(E.extended_embedded_reference_stress(100e6, 0.0, 0.2 * B, 0.05 * B, B))  # beyond alpha_2


# ---- kind (b) limits of design note G3 -----------------------------------------------------------


def test_embedded_infinite_body_limits_and_branch_continuity() -> None:
    p = 10.0  # 2a/B' -> 0 at fixed p (2a/B' = 1e-7: the M2, M3 terms are below 1e-13)
    for ac in (0.1, 0.5, 1.0, 1.5, 2.0):
        a = 1e-6
        c = a / ac
        phi = math.sqrt(1 + 1.464 * min(ac, 1 / ac) ** 1.65)
        near = float(E.embedded_flaw_mm(a, c, p, E.THETA_NEAR_END)[()])
        major = float(E.embedded_flaw_mm(a, c, p, E.THETA_MAJOR_AXIS)[()])
        if ac <= 1.0:
            assert near == pytest.approx(1 / phi, rel=1e-12) and major == pytest.approx(math.sqrt(ac) / phi, rel=1e-12)
        else:
            q = 1 / ac
            assert near == pytest.approx(q / phi, rel=1e-12) and major == pytest.approx(math.sqrt(q) / phi, rel=1e-12)
    # penny-shaped crack: 2/pi within the Phi fit's error
    penny = float(E.embedded_flaw_mm(1e-6, 1e-6, p, E.THETA_NEAR_END)[()])
    assert penny == pytest.approx(2 / math.pi, rel=1e-3) and abs(penny - 2 / math.pi) / (2 / math.pi) < 8e-4
    # Griffith: a/c -> 0 at the minor-axis ends
    assert float(E.embedded_flaw_mm(1e-5, 10.0, p, E.THETA_NEAR_END)[()]) == pytest.approx(1.0, rel=1e-6)
    # continuity of M_m across the branch at a/c = 1 (finite 2a/B')
    for theta in (E.THETA_NEAR_END, E.THETA_MAJOR_AXIS, 0.7):
        below = float(E.embedded_flaw_mm(5e-3, 5e-3 * (1 + 1e-10), 5e-3, theta)[()])
        above = float(E.embedded_flaw_mm(5e-3, 5e-3 * (1 - 1e-10), 5e-3, theta)[()])
        assert below == pytest.approx(above, rel=1e-8)
    # M_m is even in theta
    for theta in (0.4, 1.2):
        assert E.embedded_flaw_mm(3e-3, 10e-3, 4e-3, theta) == E.embedded_flaw_mm(3e-3, 10e-3, 4e-3, -theta)


def test_embedded_fw_limits() -> None:
    for c in (0.05, 0.1, 0.2):
        assert float(E.embedded_flaw_fw(0.01, c, 0.0, W)[()]) == pytest.approx(
            float(plate.through_flaw_fw(c, W)[()]), rel=1e-14
        )
    assert float(E.embedded_flaw_fw(1e-9, 0.1, 10.0, W)[()]) == pytest.approx(1.0, abs=1e-9)


def test_embedded_mb_limits_and_coefficient_set_jumps() -> None:
    # M_b/M_m -> 1 - 2p/B as a -> 0: the first coefficient set reproduces it, the second only within 6 %
    a, c = 1e-7, 1e-6
    for p_b, tol in ((0.1, 2e-3), (0.3, 0.06)):
        ratio = float((E.embedded_flaw_mb(a, c, p_b * B, B) / E.embedded_flaw_mm(a, c, p_b * B, E.THETA_NEAR_END))[()])
        assert ratio == pytest.approx(1 - 2 * p_b, rel=tol)
    # 0 <= M_b <= M_m on the valid grid (bending-valid part)
    for a, c, p in _grid():
        if a / (2 * c) <= 0.5:
            mb = float(E.embedded_flaw_mb(a, c, p, B)[()])
            mm = float(E.embedded_flaw_mm(a, c, p, E.THETA_NEAR_END)[()])
            assert -1e-12 <= mb <= mm * (1 + 1e-12)
    # B2: the coefficient sets are discontinuous at both thresholds (jumps, not kinks): recorded sizes
    eps = 1e-9
    jumps_p = [
        abs(
            float(
                (
                    E.embedded_flaw_mb(a_b * B, 4 * a_b * B, (0.1841 + eps) * B, B)
                    - E.embedded_flaw_mb(a_b * B, 4 * a_b * B, (0.1841 - eps) * B, B)
                )[()]
            )
        )
        for a_b in (0.02, 0.1, 0.125, 0.2)
    ]
    jumps_a = [
        abs(
            float(
                (
                    E.embedded_flaw_mb((0.125 + eps) * B, 0.5 * B, p_b * B, B)
                    - E.embedded_flaw_mb((0.125 - eps) * B, 0.5 * B, p_b * B, B)
                )[()]
            )
        )
        for p_b in (0.25, 0.3, 0.35)
    ]
    assert max(jumps_p) > 1e-3 and max(jumps_a) > 1e-3  # events are needed when the bending path opens
    assert max(jumps_p) < 0.05 and max(jumps_a) < 0.05
    # a -> 0 at the p/B threshold: the first two sets agree to 2e-5 (the fits meet there to rounding)
    lo = float(E.embedded_flaw_mb(1e-7, 1e-6, (0.1841 - eps) * B, B)[()])
    hi = float(E.embedded_flaw_mb(1e-7, 1e-6, (0.1841 + eps) * B, B)[()])
    assert lo == pytest.approx(hi, abs=3e-5)


def test_embedded_below_surface_deep_point_and_reference_stress_limits() -> None:
    # G3(5) as actually observed: a mid-thickness embedded flaw (height 2a in B) against the surface flaw of
    # depth a in the half plate B/2 (the free-surface counterpart): the embedded value is the smaller for
    # a/2c <= 0.5 and a/(B/2) <= 0.6 (27 states tried; the three exceptions, a/2c >= 0.8 with
    # a/(B/2) >= 0.6, are the two fits near their limits and are recorded, not asserted)
    for a_2c in (0.02, 0.1, 0.3, 0.5):
        for a_b in (0.02, 0.1, 0.2, 0.3):
            a = a_b * B
            c = a / (2 * a_2c)
            p = (B - 2 * a) / 2
            if E.embedded_flaw_valid(a, c, p, B, W)[()] and plate.surface_flaw_valid(a, c, B / 2, W)[()]:
                emb = float(E.embedded_flaw_mm(a, c, p, E.THETA_NEAR_END)[()])
                surf = float(plate.surface_flaw_mm_mb(a, c, B / 2, plate.THETA_DEEPEST)[0][()])
                assert emb < surf
    # a -> 0: the embedded reference stress tends to the net-section value Pm + Pb/... of the form (al -> 0)
    s0 = float(E.embedded_flaw_reference_stress(100e6, 30e6, 1e-9, 1e-8, 5e-3, B, W)[()])
    assert s0 == pytest.approx((30e6 + math.sqrt(30e6**2 + 9 * 100e6**2)) / 3.0, rel=1e-6)
    # p -> 0 against the surface-flaw pin-jointed form with depth 2a (both have the 3 Pm al term; the
    # embedded denominator lacks the ligament term at p = 0): the two agree exactly at p = 0
    a, c = 2e-3, 10e-3
    emb0 = float(E.embedded_flaw_reference_stress(100e6, 20e6, a, c, 0.0, B, W)[()])
    surf0 = float(plate.surface_flaw_reference_stress(100e6, 20e6, 2 * a, c, B, W, pin_jointed=True)[()])
    assert emb0 == pytest.approx(surf0, rel=1e-12)


def test_embedded_validity_pins_the_nearer_ligament_and_the_limits() -> None:
    valid = E.embedded_flaw_valid
    assert not valid(3e-3, 10e-3, 9e-3, 20e-3, W)[()]  # 2a + 2p = 24 > B = 20: p is the far ligament (near one 5 mm)
    assert valid(3e-3, 10e-3, 7e-3, 20e-3, W)[()]  # 2a + 2p = 20 = B: mid-thickness, equality allowed
    assert not valid(3e-3, 10e-3, 7.001e-3, 20e-3, W)[()]
    assert not valid(3e-3, 1e-3, 2e-3, B, W)[()]  # a/2c = 1.5 > 1
    assert valid(3e-3, 1.5e-3, 2e-3, B, W)[()]  # a/2c = 1 exactly
    # long flaw a/2c = 0.05 (a/c = 0.1): the depth limit a/B' < 0.625 (0.1 + 0.6) = 0.4375 applies
    assert valid(1e-3, 10e-3, 0.15e-3, B, W)[()]  # a/B' = 1/2.3 = 0.435
    assert not valid(1e-3, 10e-3, 0.1e-3, B, W)[()]  # a/B' = 1/2.2 = 0.4545
    # a/2c = 0.2: no depth limit printed; only B' <= B bounds it
    assert valid(1e-3, 2.5e-3, 0.05e-3, B, W)[()]  # a/B' = 0.476
    assert not valid(2e-3, 130e-3, 5e-3, B, W)[()]  # 2c/W = 0.52 >= 0.5
    assert valid(2e-3, 120e-3, 5e-3, B, W)[()]  # 2c/W = 0.48
    assert not valid(2e-3, 10e-3, 0.0, B, W)[()]  # p must be positive


# ---- cylinder tables: two independent typings agree cell by cell -------------------------------


def test_cylinder_tables_match_the_second_copy_cell_by_cell() -> None:
    sc = SECOND.SECOND_COPY
    compared = 0
    for key in ("1a", "1b", "1c", "1d", "6a", "6b", "6c", "6d"):
        t, s = T.TABLES["M." + key], sc["M" + key]
        assert tuple(t.axes[0][1]) == tuple(s["axes"][0][1]) and tuple(t.axes[1][1]) == tuple(s["axes"][1][1])
        mine, theirs = np.asarray(t.values[0], float), np.asarray(s["values"], float)
        same = (mine == theirs) | (np.isnan(mine) & np.isnan(theirs))
        assert same.all()
        compared += int(np.sum(~np.isnan(theirs)))
    names4 = ("Mm_d", "Mb_d", "Mm_s", "Mb_s")
    for key, names in (("2", names4), ("4", names4), ("7", names4), ("3", ("Mm", "Mb")), ("8", ("Mm", "Mb"))):
        t, s = T.TABLES["M." + key], sc["M" + key]
        for ax_t, ax_s in zip(t.axes, s["axes"], strict=True):
            assert tuple(ax_t[1]) == tuple(ax_s[1])
        for q, n in enumerate(names):
            mine, theirs = np.asarray(t.values[q], float), np.asarray(s["values"][n], float)
            assert np.array_equal(mine, theirs)
            compared += theirs.size
    t, s = T.TABLES["M.5"], sc["M5"]  # stored with the axes swapped
    for q, n in enumerate(("Mm", "Mb")):
        assert np.array_equal(np.asarray(t.values[q], float).T, np.asarray(s["values"][n], float))
        compared += np.asarray(s["values"][n]).size
    assert compared == 1022 == sum(T.CELL_COUNTS.values()) == sum(SECOND.CELL_COUNTS.values())


def test_cylinder_tables_reproduce_the_plate_at_zero_depth() -> None:
    # G8: every a/B = 0 row of Tables M.2, M.4, M.7 equals the plate M_m to the printed three decimals
    for key in ("M.2", "M.4", "M.7"):
        t = T.TABLES[key]
        a_c = np.asarray(t.axes[0][1])
        a_b = np.asarray(t.axes[1][1])
        assert a_b[0] == 0.0
        for i, ac in enumerate(a_c):
            a = 1e-9
            c = a / ac
            mm_d = float(plate.surface_flaw_mm_mb(a, c, B, plate.THETA_DEEPEST)[0][()])
            mm_s = float(plate.surface_flaw_mm_mb(a, c, B, plate.THETA_SURFACE)[0][()])
            for j in range(2):  # both curvature columns
                assert float(np.asarray(t.values[0])[i, 0, j]) == pytest.approx(mm_d, abs=5.1e-4)
                assert float(np.asarray(t.values[2])[i, 0, j]) == pytest.approx(mm_s, abs=5.1e-4)


# ---- static assessment of embedded flaws (design note G1, G2, G4(c)) -----------------------------


def _steel() -> cf.Material:
    return cf.Material(355e6, 490e6, 207e9, 90e6, yielding="continuous")


def test_embedded_static_assessment_positions_refusal_and_validity() -> None:
    plate_ = cf.FlatPlate(B, W)
    res = cf.Assessment(plate_, cf.EmbeddedFlaw(4e-3, 20e-3, 5e-3), _steel(), cf.StressState(pm=150e6)).run()
    assert res.positions == ("near_end", "major_axis") and int(res.governing_position) == 0
    assert "embedded_recharacterisation_relaxation_not_applied" in res.trace.flags
    assert "k.plate.embedded" in res.trace.equations and "sref.plate.embedded" in res.trace.equations
    # the K at the near end equals the solution module's value (crack size a = half-height)
    a, c, p = 2e-3, 10e-3, 5e-3
    k_expected = (
        E.embedded_flaw_mm(a, c, p, E.THETA_NEAR_END) * E.embedded_flaw_fw(a, c, p, W) * 150e6 * math.sqrt(math.pi * a)
    )
    assert float(res.k_primary[0]) == pytest.approx(float(k_expected[()]), rel=1e-12)
    # a surface flaw of the same depth a (= half-height) under the same stress has the larger K at its
    # deepest point (the free surface raises K); the embedded flaw's reference stress is the larger (twice
    # the height), so only K is compared
    surf = cf.Assessment(plate_, cf.SurfaceFlaw(2e-3, 20e-3), _steel(), cf.StressState(pm=150e6)).run()
    assert float(surf.k_primary[0]) > float(res.k_primary[0]) and float(surf.reference_stress) < float(
        res.reference_stress
    )
    # bending is refused (no M_b at the major-axis ends), primary or secondary
    for loads in (cf.StressState(pm=150e6, pb=1e6), cf.StressState(pm=150e6, qb=1e6)):
        with pytest.raises(NotImplementedError, match=r"TODO\(verify\): M_b"):
            cf.Assessment(plate_, cf.EmbeddedFlaw(4e-3, 20e-3, 5e-3), _steel(), loads).run()
    # the far ligament declared as p: outside validity, never reflected
    far = cf.Assessment(plate_, cf.EmbeddedFlaw(4e-3, 20e-3, 12e-3), _steel(), cf.StressState(pm=150e6))
    with pytest.raises(cf.OutOfValidityError):
        far.run()
    assert not bool(far.run(on_invalid="mask").valid)
    near = cf.EmbeddedFlaw.from_ligaments(4e-3, 20e-3, 12e-3, 9e-3)
    assert float(np.asarray(near.p)) == 9e-3
    assert bool(cf.Assessment(plate_, near, _steel(), cf.StressState(pm=150e6)).run().valid)
    with pytest.raises(ValueError):
        cf.EmbeddedFlaw(4e-3, 20e-3, 5e-3, near_surface="top")


def test_embedded_random_ligament_crossing_the_limit_is_out_of_validity_mass() -> None:
    # B18: p ~ Uniform(6, 10) mm with 2a = 8 mm in B = 25 mm: the limit is p <= 8.5 mm, exceeded with
    # probability 0.375; the mass is reported, nothing dropped or reflected
    from critflaw import prob

    problem = cf.Assessment(
        cf.FlatPlate(B, W),
        cf.EmbeddedFlaw(8e-3, 40e-3, prob.Uniform(6e-3, 10e-3)),
        _steel(),
        cf.StressState(pm=prob.Truncated(prob.Normal(150e6, 10e6), lo=0.0)),
    )
    pa = cf.ProbabilisticAssessment(problem)
    u = np.random.default_rng(0).standard_normal((4000, pa.vector.size))
    res = pa.evaluate(u)
    share = float(np.mean(~res.valid))
    assert share == pytest.approx(0.375, abs=0.03)
    assert np.all(np.isnan(res.radial_margin[~res.valid])) and np.all(np.isfinite(res.radial_margin[res.valid]))


# ---- cylinders: K factors and reference stresses against the partner oracle --------------------------

ORACLE_C = _load("partner_reference_partB_cylinders", HERE / "oracles" / "partner_reference_partB_cylinders.py")


def test_cylinder_shell_parameters_and_bulging_against_the_oracle() -> None:
    from critflaw.solutions import cylinder as C

    for a, r_m, b in ((0.05, 0.5, 0.02), (0.2, 1.0, 0.05), (0.001, 0.3, 0.01)):
        assert float(C.bulging_through_axial(a, r_m, b)[()]) == pytest.approx(ORACLE_E.bulging(a, r_m, b), rel=1e-14)
        assert float(C.shell_lambda(a, r_m, b, 0.3)[()]) == pytest.approx(
            ORACLE_E.shell_lambda(a, r_m, b, 0.3), rel=1e-14
        )
        assert float(C.shell_beta(a, r_m)[()]) == pytest.approx(ORACLE_E.shell_beta(a, r_m), rel=1e-14)
        if a > 0.0:
            assert float(C.bulging_surface_axial(0.5 * b, a, r_m, b)[()]) == pytest.approx(
                ORACLE_E.surface_axial(0.5 * b, a, r_m, b), rel=1e-14
            )
    assert float(C.moment_membrane_stress(1e5, 0.5, 0.01, 0.02)[()]) == pytest.approx(
        ORACLE_E.moment_membrane(1e5, 0.5, 0.01, 0.02), rel=1e-14
    )
    # (M.26) edge-crack limit and the extended plate solution at a = 0
    assert float(C.fully_circumferential_external_mm(1e-12, 0.9, 1.0)[()]) == pytest.approx(1.118034, rel=1e-6)
    mm, mb = C.extended_plate_mm_mb(0.0, 1.0)
    assert float(mm[()]) == 1.12 and float(mb[()]) == 1.12


def test_cylinder_reference_stresses_against_the_oracle() -> None:
    from critflaw.solutions import cylinder_sref as S

    b, r_i, w = 0.02, 0.5, 2.0
    for a, c, pm, pb in ((0.005, 0.05, 100.0, 20.0), (0.01, 0.1, 150.0, 0.0), (0.015, 0.02, 80.0, 60.0)):
        assert float(S.axial_through_thin(pm, pb, a, r_i, b, w)[()]) == pytest.approx(
            ORACLE_C.axial_thin("through", a, b, r_i, w, pm, pb)["sigma_ref"], rel=1e-14
        )
        assert float(S.axial_surface_thin(pm, pb, a, c, r_i, b, w)[()]) == pytest.approx(
            ORACLE_C.axial_thin("surface", a, b, r_i, w, pm, pb, c=c)["sigma_ref"], rel=1e-14
        )
        for ext in (False, True):
            kind = "extended_external" if ext else "extended_internal"
            assert float(S.axial_extended_thin(pm, pb, a, r_i, b, w, ext)[()]) == pytest.approx(
                ORACLE_C.axial_thin(kind, a, b, r_i, w, pm, pb)["sigma_ref"], rel=1e-14
            )
        for ext in (False, True):
            r_in = ORACLE_C.axial_pressure("internal", a, b, r_i, 300.0, c=c, extended=ext)["p_L"]
            r_ex = ORACLE_C.axial_pressure("external", a, b, r_i, 300.0, c=c, extended=ext)["p_L"]
            assert float(S.axial_internal_limit_pressure(a, c, r_i, b, 300.0, ext)[()]) == pytest.approx(
                r_in, rel=1e-12
            )
            assert float(S.axial_external_limit_pressure(a, c, r_i, b, 300.0, ext)[()]) == pytest.approx(
                r_ex, rel=1e-12
            )
    # P.9.8 normalised limits over both branches, pure tension and pure bending
    for alpha in (0.1, 0.4, 0.7, 0.9):
        for psi in (0.0, 0.5, 2.0):
            for lam in (0.0, 0.2, 1.0, 5.0, None):
                c = psi * b
                width = 1e6 if psi == 0 else 2 * c / (psi / (1 + psi)) * 1.01 + 1
                pm, pb = (0.0, 100.0) if lam is None else (100.0, 6 * lam * 100.0)
                l_n, l_b, ok = S.axial_external_thick_limits(pm, pb, alpha * b, c, b, width)
                ref = ORACLE_C.axial_external_limits(alpha, psi, lam, beta=2 * c / width)
                assert bool(ok) and float(l_n[()]) == pytest.approx(ref["L_N"], abs=1e-14)
                assert float(l_b[()]) == pytest.approx(ref["L_b"], abs=1e-14)
    # circumferential, thin wall
    assert float(S.circumferential_through_thin(50, 80, 30, 20, 0.05, r_i, b)[()]) == pytest.approx(
        ORACLE_C.circumferential_thin("through", 0.05, b, r_i, P_ma=50, P_mp=80, P_mb=30, P_bl=20)["sigma_ref"],
        rel=1e-14,
    )
    for c in (0.05, 1.2):  # both alpha'' branches (pi r_i >= c + B and <)
        assert float(S.circumferential_surface_thin(100, 20, 0.01, c, r_i, b)[()]) == pytest.approx(
            ORACLE_C.circumferential_thin("surface", 0.01, b, r_i, P_m=100, P_b=20, c=c)["sigma_ref"], rel=1e-14
        )
    assert float(S.circumferential_fully_thin(100, 20, 0.01, r_i, b)[()]) == pytest.approx(
        ORACLE_C.circumferential_thin("fully", 0.01, b, r_i, P_m=100, P_b=20)["sigma_ref"], rel=1e-14
    )


def test_circumferential_thick_limit_load_against_the_oracle() -> None:
    from critflaw.solutions import cylinder_sref as S

    cases = 0
    for alpha in (0.2, 0.5, 0.8, 1.0):
        for eta in (0.0, 0.1, 0.3):
            for theta in (0.3, 1.0, 2.0, 2.356194490192345, 3.0, math.pi):
                for lam in (0.0, 0.3, 1.0, 3.0, None):
                    for side in ("internal", "external"):
                        if alpha == 1.0 and theta == math.pi:
                            continue
                        try:
                            ref = ORACLE_C.circumferential_net(alpha, eta, theta, lam, side)
                        except ValueError:
                            continue
                        r_m = 1.0
                        b = eta * r_m if eta > 0 else 1e-9
                        l_n, l_b, beta, _ = S.circumferential_thick_limit(
                            alpha * b, theta, r_m, b, np.inf if lam is None else lam, external=(side == "external")
                        )
                        assert float(l_n[()]) == pytest.approx(ref["L_N"], abs=1e-9)
                        assert float(l_b[()]) == pytest.approx(ref["L_b"], abs=1e-9)
                        assert float(beta[()]) == pytest.approx(ref["beta"], abs=1e-9)
                        cases += 1
    assert cases > 600


# ---- cylinder assessments (design note G5, G6, B10) ----------------------------------------------------


def _cyl(route: str, sref: str = "thin") -> cf.Cylinder:
    return cf.Cylinder(r_i=0.1, thickness=0.02, length=2.0, k_route=route, sref_route=sref)  # type: ignore[arg-type]


PRESS = cf.PressureLoad(5e6, hoop_per_pressure=5.5, axial_per_pressure=2.75)


def test_cylinder_route_is_explicit_and_the_pressure_only_route_refuses_mechanical_stress() -> None:
    with pytest.raises(ValueError, match="k_route"):
        cf.Cylinder(r_i=0.1, thickness=0.02, length=2.0, k_route="auto")  # type: ignore[arg-type]
    flaw = cf.CylinderSurfaceFlaw(5e-3, 40e-3, "axial", "internal")
    res_p = cf.Assessment(_cyl("pressure_only"), flaw, _steel(), cf.StressState(pressure=PRESS)).run()
    res_t = cf.Assessment(_cyl("tabulated"), flaw, _steel(), cf.StressState(pressure=PRESS)).run()
    assert res_p.trace.options["k_route"] == "pressure_only" and res_t.trace.options["k_route"] == "tabulated"
    assert (
        "k.cylinder.bulging_surface" in res_p.trace.equations
        and "k.cylinder.tables.axial_surface" in res_t.trace.equations
    )
    # the two routes are different models of the same shell: the difference is recorded, not zero (B10)
    assert float(res_p.kr) != pytest.approx(float(res_t.kr), rel=1e-3)
    assert float(res_p.lr) == pytest.approx(float(res_t.lr))  # the same thin-wall reference stress
    with pytest.raises(cf.OutOfValidityError):
        cf.Assessment(_cyl("pressure_only"), flaw, _steel(), cf.StressState(pm=1e6, pressure=PRESS)).run()
    with pytest.raises(TypeError):
        cf.Assessment(cf.FlatPlate(0.02, 2.0), flaw, _steel(), cf.StressState(pm=1e6)).run()
    with pytest.raises(TypeError):
        cf.Assessment(_cyl("tabulated"), cf.SurfaceFlaw(5e-3, 40e-3), _steel(), cf.StressState(pm=1e6)).run()


def test_cylinder_pressure_record_scales_k_and_lr_together() -> None:
    # B21: doubling the pressure doubles the pressure-derived membrane stress in K and in the thin-wall
    # reference stress; the thick-wall limit pressure uses the same pressure (Lr = p / p_L)
    flaw = cf.CylinderSurfaceFlaw(5e-3, 40e-3, "axial", "internal")
    one = cf.Assessment(_cyl("tabulated"), flaw, _steel(), cf.StressState(pressure=cf.PressureLoad(10e6, 5.5))).run()
    two = cf.Assessment(_cyl("tabulated"), flaw, _steel(), cf.StressState(pressure=cf.PressureLoad(20e6, 5.5))).run()
    assert float(two.kr) == pytest.approx(2 * float(one.kr), rel=1e-12)
    assert float(two.reference_stress) == pytest.approx(2 * float(one.reference_stress), rel=1e-12)
    thick = cf.Assessment(
        _cyl("tabulated", "thick_pressure"), flaw, _steel(), cf.StressState(pressure=cf.PressureLoad(20e6, 5.5))
    ).run()
    from critflaw.solutions import cylinder_sref as S

    p_l = float(S.axial_internal_limit_pressure(5e-3, 20e-3, 0.1, 0.02, 355e6)[()])
    assert float(thick.lr) == pytest.approx(20e6 / p_l, rel=1e-12)
    assert "sref_from_limit_load" in thick.trace.options
    with pytest.raises(ValueError, match="thick_pressure"):
        cf.Assessment(_cyl("tabulated", "thick_pressure"), flaw, _steel(), cf.StressState(pm=50e6)).run()


def test_cylinder_tabulated_positions_suspect_flag_and_through_wall_sign_convention() -> None:
    res = cf.Assessment(
        _cyl("tabulated"),
        cf.CylinderThroughFlaw(0.05, "circumferential"),
        _steel(),
        cf.StressState(pm=30e6, pb=5e6, pm_moment=20e6, pressure=PRESS),
    ).run()
    assert res.positions == ("outside", "inside")
    assert float(res.kr_by_position[0]) > float(
        res.kr_by_position[1]
    )  # bending tensile outside (the source's convention)
    # with the bending declared tensile inside, the bending contribution moves to the inside surface:
    # under a bending-dominated load the inside governs
    bend = cf.StressState(pm=2e6, pb=60e6)
    out_t = cf.Assessment(_cyl("tabulated"), cf.CylinderThroughFlaw(0.05, "circumferential"), _steel(), bend).run()
    in_t = cf.Assessment(
        _cyl("tabulated"),
        cf.CylinderThroughFlaw(0.05, "circumferential"),
        _steel(),
        bend.with_(bending_tensile_surface="inside"),
    ).run()
    assert float(out_t.kr_by_position[0]) > float(out_t.kr_by_position[1])
    assert float(in_t.kr_by_position[1]) > float(in_t.kr_by_position[0])
    # a query through the flagged M.6 cells (r_m/B = 5.5 between the suspect r_m/B = 5 entries): flagged
    assert "tabulated_factor_interpolated_through_a_suspect_cell_samples_present" in res.trace.flags
    # outside the tables (B/r_i = 0.04): out of validity, never extrapolated
    thin_wall = cf.Cylinder(r_i=0.5, thickness=0.02, length=4.0, k_route="tabulated")
    with pytest.raises(cf.OutOfValidityError):
        cf.Assessment(
            thin_wall, cf.CylinderSurfaceFlaw(5e-3, 40e-3, "axial", "internal"), _steel(), cf.StressState(pm=50e6)
        ).run()


def test_cylinder_extended_and_fully_circumferential_flaws() -> None:
    from critflaw.solutions import cylinder as C

    loads = cf.StressState(pm=50e6, pb=20e6)
    ext = cf.Assessment(_cyl("tabulated"), cf.ExtendedFlaw(5e-3, "circumferential", "external"), _steel(), loads).run()
    assert "k.cylinder.extended_external" in ext.trace.equations
    mm = float(C.fully_circumferential_external_mm(5e-3, 0.1, 0.12)[()])
    assert float(ext.k_primary[0]) == pytest.approx(mm * (50e6 + 20e6) * math.sqrt(math.pi * 5e-3), rel=1e-12)
    tab = cf.Assessment(_cyl("tabulated"), cf.ExtendedFlaw(5e-3, "circumferential", "internal"), _steel(), loads).run()
    assert "k.cylinder.tables.circumferential_extended" in tab.trace.equations and tab.positions == ("deepest",)
    thick = cf.Assessment(
        _cyl("tabulated", "thick"),
        cf.ExtendedFlaw(5e-3, "circumferential", "internal"),
        _steel(),
        cf.StressState(pm=50e6, pm_moment=30e6),
    ).run()
    assert 0.0 < float(thick.lr) < 1.0 and "sref.cylinder.circumferential_thick" in thick.trace.equations
    plate_ext = cf.Assessment(
        _cyl("pressure_only"), cf.ExtendedFlaw(5e-3, "axial", "internal"), _steel(), cf.StressState(pressure=PRESS)
    ).run()
    assert "k.plate.extended" in plate_ext.trace.equations


# ---- embedded flaw growth (design note G4, benchmarks B3, B4, B14, B20) --------------------------------


def _embedded_reference(
    law: GrowthLaw, a: float, c: float, p: float, ds: float, cycles: int, thickness: float, width: float
) -> tuple[float, float, float, int]:
    """Cycle-by-cycle growth of an embedded flaw (near end grows a, major-axis ends grow c, B' constant) until
    breakthrough or an embedded validity limit; returns (a, c, cycles used, boundary code)."""
    b_eff = 2 * a + 2 * p
    for n in range(cycles):
        p_now = 0.5 * (b_eff - 2 * a)
        if p_now <= 0.0:
            return a, c, n, 5
        if a / (2 * c) <= 0.1 and a / b_eff >= 0.625 * (a / c + 0.6):
            return a, c, n, 6
        if a / (2 * c) >= 1.0:
            return a, c, n, 7
        if 2 * c / width >= 0.5:
            return a, c, n, 8
        fw = float(E.embedded_flaw_fw(a, c, p_now, width)[()])
        dk_a = float(E.embedded_flaw_mm(a, c, p_now, E.THETA_NEAR_END)[()]) * fw * ds * math.sqrt(math.pi * a)
        dk_c = float(E.embedded_flaw_mm(a, c, p_now, E.THETA_MAJOR_AXIS)[()]) * fw * ds * math.sqrt(math.pi * a)
        a += float(law.rate(dk_a))
        c += float(law.rate(dk_c))
    return a, c, cycles, 0


def test_embedded_growth_cycle_by_cycle_breakthrough_cascade_mid_thickness_and_long_flaw() -> None:
    # B4: four embedded flaws against the cycle-by-cycle reference: a/c < 1 (surface destination valid),
    # a/c > 1 (cascade to the through flaw), mid-thickness (surface_breaking and leak together), and a long
    # flaw (a/2c <= 0.1, a/c < 0.2) that leaves by the depth limit (leak_assumed, trigger 5), never p = 0
    law = GrowthLaw(float(coefficient_to_si(5.86e-13, 2.88)), 2.88, 0.0)
    nu = 1e5
    sp = Spectrum.repeating([(100e6, 0.0, 1.0)], nu)
    a0 = np.array([2e-3, 3e-3, 4e-3, 1e-3])
    c0 = np.array([8e-3, 2e-3, 8e-3, 12e-3])
    p0 = np.array([5e-3, 5e-3, 8.5e-3, 0.5e-3])
    st = GrowthState.start(a0, c0, np.full(4, B), np.full(4, W), 1, ligament=p0)
    assert np.all(st.kind == G.EMBEDDED) and not st.invalid_start.any()
    g = Grower(law, sp, 4, 20.0)
    for t in (1, 2, 5, 10, 20):
        st.new_segment()
        g.advance(st, float(t))
    expected = {}
    for i in range(4):
        a, c, n, code = _embedded_reference(law, a0[i], c0[i], p0[i], 100e6, 3_000_000, B, W)
        expected[i] = (a, c, n / nu, code)
    # case 0: breakthrough; the surface flaw then leaks (Part A path)
    assert np.isfinite(st.t_surface_breaking[0]) and st.t_surface_breaking[0] == pytest.approx(expected[0][2], rel=5e-3)
    assert expected[0][3] == 5 and np.isfinite(st.t_leak[0]) and st.t_leak[0] > st.t_surface_breaking[0]
    # case 1: a/c > 1 at breakthrough: the destination has a/2c > 1 and cascades to the through flaw (trigger 11)
    assert expected[1][3] == 5 and st.t_surface_breaking[1] == pytest.approx(expected[1][2], rel=5e-3)
    assert st.leak_trigger[1] == 11 and st.t_leak_assumed[1] == st.t_surface_breaking[1]
    # case 2: mid-thickness: surface_breaking and leak at the same time
    assert expected[2][3] == 5 and st.t_surface_breaking[2] == pytest.approx(expected[2][2], rel=5e-3)
    assert st.t_leak[2] == st.t_surface_breaking[2]
    # case 3: long flaw: the depth limit (trigger 5 = embedded a/B' limit) before breakthrough; never p = 0
    assert expected[3][3] == 6 and np.isnan(st.t_surface_breaking[3])
    assert st.leak_trigger[3] in (5, 11, 1) and st.t_leak_assumed[3] == pytest.approx(expected[3][2], rel=5e-3)


def test_embedded_growth_closed_form_and_time_dependent_run() -> None:
    # B3: a near-constant-Y embedded flaw far from every limit; the Paris closed form within 0.5 %
    law = GrowthLaw(float(coefficient_to_si(1e-13, 3.0)), 3.0, 0.0)
    nu, ds = 1e5, 100e6
    sp = Spectrum.repeating([(ds, 0.0, 1.0)], nu)
    a0, c0, p0, thick = 1e-3, 100e-3, 200e-3, 1.0  # a/c = 0.01, 2a/B' ~ 0.005: Y ~ 1/Phi(0.01) at the near end
    st = GrowthState.start(
        np.array([a0]), np.array([c0]), np.array([thick]), np.array([10.0]), 1, ligament=np.array([p0])
    )
    g = Grower(law, sp, 1, 5.0)
    g.advance(st, 5.0)
    y = float(E.embedded_flaw_mm(a0, c0, p0, E.THETA_NEAR_END)[()])
    a_closed = (a0**-0.5 - 0.5 * float(law.a_coefficient) * (y * ds) ** 3 * math.pi**1.5 * nu * 5.0) ** -2.0
    assert float(st.a[0]) == pytest.approx(a_closed, rel=5e-3)
    # the time-dependent run through solve(): the embedded flaw breaks through, becomes a surface flaw and fails
    law2 = GrowthLaw(float(coefficient_to_si(5.86e-13, 2.88)), 2.88, 0.0)
    sp2 = Spectrum.repeating([(100e6, 0.0, 1.0)], 1e5)
    problem = cf.Assessment(cf.FlatPlate(B, W), cf.EmbeddedFlaw(4e-3, 16e-3, 5e-3), _steel(), cf.StressState(pm=100e6))
    ext = ExtremeLoad(prob.Uniform(200e6, 200.001e6), 1.0)
    run = TimeDependentAssessment(problem, law2, sp2, ext, times=range(1, 11), leak_is_failure=False)
    h = run.grow(load=200e6)
    assert h.kind[0] == G.EMBEDDED and G.SURFACE in h.kind and np.isfinite(h.first_failure)
    res = run.solve(seed=0, samples=20, stage_sensitivity=False)
    assert res.events_any["surface_breaking"] == 1.0 and res.cumulative_upper.value[-1] == 1.0
    assert res.cumulative_upper.value[0] == 0.0
    # bending anywhere is refused for embedded growth
    with pytest.raises(NotImplementedError, match="M_b"):
        TimeDependentAssessment(
            problem, law2, Spectrum.repeating([(100e6, 10e6, 1.0)], 1e5), ext, times=[1], leak_is_failure=False
        )
    # inspection of an embedded flaw needs pod_embedded (of the height 2a)
    with pytest.raises(ValueError, match="pod_embedded"):
        TimeDependentAssessment(
            problem,
            law2,
            sp2,
            ext,
            times=[1, 2],
            leak_is_failure=False,
            inspections=[Inspection(1.0, ConstantPoD(1.0))],
        ).solve(seed=0, samples=4, stage_sensitivity=False)
    seen = TimeDependentAssessment(
        problem,
        law2,
        sp2,
        ext,
        times=[1, 2],
        leak_is_failure=False,
        inspections=[Inspection(1.0, ConstantPoD(0.0), pod_embedded=StepPoD(4.5e-3))],
    ).solve(seed=0, samples=4, stage_sensitivity=False)
    assert seen.inspections[0]["p_detect_all"] == 1.0  # the height 2a ~ 5 mm after a year is above 4.5 mm
    unseen = TimeDependentAssessment(
        problem,
        law2,
        sp2,
        ext,
        times=[1, 2],
        leak_is_failure=False,
        inspections=[Inspection(1.0, ConstantPoD(0.0), pod_embedded=StepPoD(7e-3))],
    ).solve(seed=0, samples=4, stage_sensitivity=False)
    assert unseen.inspections[0]["p_detect_all"] == 0.0  # the height, not the half-height, is inspected


def test_embedded_boundary_landing_from_just_below_each_limit() -> None:
    # B20: starts just below the breakthrough, aspect and width boundaries land on them to 1e-12 with the
    # right event, and no evaluation outside the boundary set
    law = GrowthLaw(float(coefficient_to_si(5.86e-13, 2.88)), 2.88, 0.0)
    sp = Spectrum.repeating([(100e6, 0.0, 1.0)], 1e5)
    # (the aspect limit a/2c = 1 is not reachable by membrane-loaded growth: the major-axis ends grow faster
    # than the near end for a/c > 1, so a/2c decreases; checked below)
    cases = [
        (4e-3, 8e-3, 4e-3 * 1e-6, 5),  # p ~ 0: breakthrough
        (2e-3, 0.5 * W * 0.5 * (1 - 1e-6), 5e-3, 8),  # 2c/W ~ 0.5: width limit
    ]
    st_a = GrowthState.start(
        np.array([3.999e-3]), np.array([2e-3]), np.array([B]), np.array([W]), 1, ligament=np.array([5e-3])
    )
    Grower(law, sp, 1, 1.0).advance(st_a, 1.0)
    assert st_a.kind[0] == G.EMBEDDED and st_a.event_count[0] == 0 and st_a.a[0] / (2 * st_a.c[0]) < 0.9997
    for a0, c0, p0, code in cases:
        st = GrowthState.start(np.array([a0]), np.array([c0]), np.array([B]), np.array([W]), 1, ligament=np.array([p0]))
        assert st.kind[0] == G.EMBEDDED
        g = Grower(law, sp, 1, 1.0)
        g.advance(st, 1.0)
        assert st.event_count[0] >= 1 and st.event_pre[0, 0, 2] == G.EMBEDDED
        a_b, c_b = st.event_pre[0, 0, 0], st.event_pre[0, 0, 1]
        excess = g.boundary_excess(
            np.array([a_b]),
            np.array([c_b]),
            np.array([G.EMBEDDED]),
            np.array([B]),
            np.array([W]),
            np.array([2 * a0 + 2 * p0]),
        )
        assert abs(float(np.max(excess))) < 1e-9 and float(np.max(excess)) <= 1e-12
        if code == 5:
            assert np.isfinite(st.t_surface_breaking[0])
        else:
            assert st.t_leak_assumed[0] == st.event_time[0, 0]  # the first re-characterisation is the event


# ---- cylinder growth (design note G9, benchmarks B7, B8, B9) -------------------------------------------


def _cyl_law_spectrum() -> tuple[GrowthLaw, Spectrum]:
    return GrowthLaw(float(coefficient_to_si(5.86e-13, 2.88)), 2.88, 0.0), Spectrum.repeating([(30e6, 0.0, 1.0)], 1e5)


def test_tabulated_surface_flaw_grows_to_the_table_edge_then_the_through_wall_range_end() -> None:
    # B7: cycle-by-cycle reference with the tabulated factors; the table edge a/B = 0.8 (trigger 8, leak_assumed)
    # becomes a through-wall flaw which reaches the lambda range end (ENDED, trigger 10); both sides recorded
    from critflaw.fatigue.growth import CylinderSpec
    from critflaw.solutions import cylinder as C

    law, sp = _cyl_law_spectrum()
    b, r_i, w = 0.02, 0.1, 2.0
    spec = CylinderSpec(r_i, "axial", "internal", "tabulated")
    a0, c0 = 4e-3, 60e-3
    st = GrowthState.start(np.array([a0]), np.array([c0]), np.array([b]), np.array([w]), 1)
    g = Grower(law, sp, 1, 50.0, cylinder=spec)
    for t in (10, 20, 30, 40, 41, 42, 43, 50):
        st.new_segment()
        g.advance(st, float(t))
        if st.kind[0] == G.ENDED:
            break
    assert st.kind[0] == G.ENDED and st.leak_trigger[0] == 8  # the first leak_assumed origin: the table edge
    assert np.isfinite(st.t_leak_assumed[0]) and st.t_ended[0] > st.t_leak_assumed[0]
    # reference: the surface phase cycle by cycle (tabulated factors at the deepest and surface points)
    table = T.TABLES["M.2"]
    a, c, n = a0, c0, 0
    nu, ds, block = 1e5, 30e6, 50  # Euler in blocks of 50 cycles (error well below the 5e-3 tolerance)
    while a / b < 0.8:
        vals = [float(C.interpolate(table, q, a / c, a / b, b / r_i).value[()]) for q in (0, 2)]
        root = math.sqrt(math.pi * a)
        a += block * float(law.rate(vals[0] * ds * root))
        c += block * float(law.rate(vals[1] * ds * root))
        n += block
    assert st.t_leak_assumed[0] == pytest.approx(n / nu, rel=5e-3)
    # the through-wall phase: lambda at the end equals the column-dependent range end
    lam_end = float(C.lambda_range_end(T.TABLES["M.1a"], (r_i + 0.5 * b) / b)[()])
    lam_final = float(C.shell_lambda(st.c[0], r_i + 0.5 * b, b)[()])
    assert lam_final == pytest.approx(lam_end, rel=1e-9) and lam_end < C.LAMBDA_MAX_AXIAL
    # the boundary state before -> ENDED is a checkpoint (the pre-side of the last event)
    k = int(st.event_count[0]) - 1
    assert st.event_pre[0, k, 2] == G.THROUGH and st.event_post[0, k, 2] == G.ENDED and st.event_trigger[0, k] == 10


def test_through_wall_cylinder_flaw_ends_at_the_lambda_range_end_without_the_finish() -> None:
    # B8: the tabulated range end is a plain validity limit (finite rate): terminal step, ENDED, no finish
    from critflaw.fatigue.growth import CylinderSpec
    from critflaw.solutions import cylinder as C

    law, sp = _cyl_law_spectrum()
    b, r_i, w = 0.02, 0.1, 2.0
    st = GrowthState.start(np.array([b]), np.array([0.05]), np.array([b]), np.array([w]), 1)
    st.kind[:] = G.THROUGH
    st.t_ended[:] = np.nan
    g = Grower(law, sp, 1, 10.0, cylinder=CylinderSpec(r_i, "axial", "internal", "tabulated"))
    g.advance(st, 10.0)
    assert st.kind[0] == G.ENDED and st.event_trigger[0, 0] == 10 and 0.0 < st.t_ended[0] < 10.0
    assert st.leak_trigger[0] == 0  # no leak or leak_assumed: the range end is an ended state
    lam_end = float(C.lambda_range_end(T.TABLES["M.1a"], (r_i + 0.5 * b) / b)[()])
    assert float(C.shell_lambda(st.c[0], r_i + 0.5 * b, b)[()]) == pytest.approx(lam_end, rel=1e-9)
    # cycle-by-cycle reference for the whole through-wall phase (the larger of the two tip ranges drives c)
    c, n, block = 0.05, 0, 20
    names = ("M.1a", "M.1b")
    while float(C.shell_lambda(c, r_i + 0.5 * b, b)[()]) < lam_end:
        lam = float(C.shell_lambda(c, r_i + 0.5 * b, b)[()])
        m1, m2 = (float(C.interpolate_columns(T.TABLES[nm], 0, lam, (r_i + 0.5 * b) / b).value[()]) for nm in names)
        dk = max(m1 + m2, m1 - m2) * 30e6 * math.sqrt(math.pi * c)
        c += block * float(law.rate(dk))
        n += block
    assert st.t_ended[0] == pytest.approx(n / 1e5, rel=1e-2)
    # the pressure-only route of a plate-like through flaw keeps the Part A finish (W = the cylinder length)
    st2 = GrowthState.start(np.array([b]), np.array([0.13]), np.array([b]), np.array([0.3]), 1)
    st2.kind[:] = G.THROUGH
    st2.t_ended[:] = np.nan
    g2 = Grower(law, sp, 1, 100.0, cylinder=CylinderSpec(r_i, "axial", "internal", "pressure_only"))
    g2.advance(st2, 100.0)
    assert st2.kind[0] == G.ENDED and st2.c[0] == 0.15  # the width limit by the quadrature finish


def test_extended_flaws_leak_or_end_at_their_depth_limits() -> None:
    # B9: an extended flaw grows in depth only; tabulated: leak_assumed and ENDED at 0.8 B (trigger 12);
    # pressure-only: the plate extended solution ends at 0.6 B (trigger 12); a closed-form check of the
    # depth phase with the plate polynomial
    from scipy import integrate

    from critflaw.fatigue.growth import CylinderSpec
    from critflaw.solutions import cylinder as C

    law, sp = _cyl_law_spectrum()
    b, r_i, w = 0.02, 0.1, 1.0
    for route, limit in (("tabulated", 0.8), ("pressure_only", 0.6)):
        st = GrowthState.start(np.array([4e-3]), np.array([w]), np.array([b]), np.array([w]), 1)
        st.kind[:] = G.EXTENDED
        st.t_ended[:] = np.nan
        g = Grower(law, sp, 1, 50.0, cylinder=CylinderSpec(r_i, "axial", "internal", route))
        g.advance(st, 50.0)
        assert st.kind[0] == G.ENDED and st.leak_trigger[0] == (12 if route == "tabulated" else 16) and st.c[0] == w
        assert st.a[0] == pytest.approx(limit * b, rel=1e-9) and st.t_leak_assumed[0] == st.t_ended[0]
        if route == "pressure_only":

            def inv_v(a: float, nu: float = 1e5, ds: float = 30e6) -> float:
                mm = float(C.extended_plate_mm_mb(a, b)[0][()])
                return 1.0 / (float(law.rate(mm * ds * math.sqrt(math.pi * a))) * nu)

            t_ref = integrate.quad(inv_v, 4e-3, 0.6 * b, epsabs=0, epsrel=1e-10)[0]
            assert st.t_ended[0] == pytest.approx(t_ref, rel=2e-3)
    # through the API: headline failed, valid-only safe, leak_is_failure decides the mode only
    cyl_ = cf.Cylinder(r_i=r_i, thickness=b, length=w, k_route="tabulated")
    problem = cf.Assessment(
        cyl_, cf.ExtendedFlaw(4e-3, "circumferential", "internal"), _steel(), cf.StressState(pm=10e6)
    )
    ext = ExtremeLoad(prob.Uniform(20e6, 20.001e6), 1.0, component="pm")  # never fails structurally
    for leak_counts in (False, True):
        run = TimeDependentAssessment(problem, law, sp, ext, times=range(10, 51, 10), leak_is_failure=leak_counts)
        res = run.solve(seed=0, samples=4, stage_sensitivity=False)
        assert res.cumulative_upper.value[-1] == 1.0
        assert res.cumulative_valid_only.value[-1] == (1.0 if leak_counts else 0.0)  # leak_assumed counts when leaks do
        assert res.events_any["leak_assumed"] == 1.0 and res.events_any["ended_outside_validity"] == 1.0
        assert (res.modes["leak_assumed"] == 1.0) == leak_counts and res.modes["leak"] == 0.0
        assert (res.modes["out_of_validity"] == 1.0) != leak_counts
        if leak_counts:
            assert any("not a breakthrough" in s for s in res.assumptions)


def test_cylinder_time_dependent_runs_and_the_route_is_explicit() -> None:
    law, sp = _cyl_law_spectrum()
    press = cf.PressureLoad(5e6, 5.5, 2.75)
    cyl_ = cf.Cylinder(r_i=0.1, thickness=0.02, length=2.0, k_route="tabulated")
    ext = ExtremeLoad(prob.Uniform(60e6, 60.001e6), 1.0, component="pm")
    problem = cf.Assessment(
        cyl_,
        cf.CylinderSurfaceFlaw(4e-3, 120e-3, "axial", "internal"),
        _steel(),
        cf.StressState(pm=10e6, pressure=press),
    )
    run = TimeDependentAssessment(problem, law, sp, ext, times=range(5, 51, 5), leak_is_failure=False)
    h = run.grow(load=60e6)
    assert np.isfinite(h.t_leak_assumed) and h.first_failure == pytest.approx(
        h.t_leak_assumed
    )  # fails at the table edge
    res = run.solve(seed=0, samples=6, stage_sensitivity=False)
    assert res.cumulative_upper.value[-1] == 1.0 and res.modes["fracture"] == 1.0
    with pytest.raises(TypeError):
        TimeDependentAssessment(
            cf.Assessment(
                cf.FlatPlate(0.02, 2.0),
                cf.CylinderSurfaceFlaw(4e-3, 120e-3, "axial", "internal"),
                _steel(),
                cf.StressState(pm=10e6),
            ),
            law,
            sp,
            ext,
            times=[1],
            leak_is_failure=False,
        )


# ---- tables: monotonicity, ordering, envelope, suspect status (G5, G8, B11, B12, B13) ------------------


def test_table_monotonicity_ordering_and_the_bulging_check() -> None:
    from critflaw.solutions import cylinder as C

    # a/B axis, direction from physics: the membrane factors rise with depth (finite-thickness magnification)
    # from a/B = 0.2 on for every a/c (the first step from a/B = 0 dips by at most 0.04 for the rounder
    # flaws, as the plate solution's deep point does); the extended-flaw membrane factor rises everywhere; the
    # bending factors fall with depth at first and have no single direction, so they are checked for the
    # absence of isolated spikes (bounded second differences) only
    for name in ("M.2", "M.4", "M.7"):
        table = T.TABLES[name]
        for q in (0, 2):  # M_m(d), M_m(s): rising from a/B = 0.2 on for every a/c, and from 0 for a/c <= 0.2
            v = np.asarray(table.values[q])
            assert np.all(np.diff(v[:, 1:], axis=1) >= 0.0), (name, table.quantities[q])
            if not (name == "M.7" and q == 2):  # Table M.7's surface-point columns are a recorded suspect
                assert np.all(np.diff(v[:, :2], axis=1) >= -0.04), (name, table.quantities[q])  # first-step dip
    for name in ("M.3", "M.5", "M.8"):
        table = T.TABLES[name]
        axis = [ax[0] for ax in table.axes].index("a_over_B")
        up, _, total = C.table_monotonicity(table, 0, axis)
        assert up == total, name
    for name in ("M.2", "M.3", "M.4", "M.5", "M.7", "M.8"):
        table = T.TABLES[name]
        axis = [ax[0] for ax in table.axes].index("a_over_B")
        for q in range(len(table.quantities)):
            v = np.moveaxis(np.asarray(table.values[q]), axis, 0)
            d2 = np.abs(np.diff(v, n=2, axis=0))
            scale = np.maximum(np.abs(v[1:-1]), 0.5)
            assert np.all(d2 <= 2.0 * scale), (name, table.quantities[q])  # no isolated spike (relative bound)
    # ordering: finite-flaw M_m at the deepest point <= the extended-flaw M_m at the same a/B and B/r_i
    for finite, extended in (("M.2", "M.3"), ("M.4", "M.5")):
        tf, te = T.TABLES[finite], T.TABLES[extended]
        ab_f = np.asarray(tf.axes[1][1])
        for j, br in enumerate(tf.axes[2][1]):
            for i, ab in enumerate(ab_f):
                ext_v = float(C.interpolate(te, 0, *((br, ab) if extended == "M.5" else (ab, br))).value[()])
                assert np.max(np.asarray(tf.values[0])[:, i, j]) <= ext_v * (1 + 1e-12), (finite, ab, br)
    # the bulging factor of (M.21) against Table M.1a over r_m/B = 5..100 (its only independent check): the
    # difference is recorded within a stated band, not asserted equal
    worst = 0.0
    for rm_b in (5.0, 10.0, 20.0, 50.0, 100.0):
        for lam in (0.5, 1.0, 2.0, 4.0):
            b = 0.02
            r_m = rm_b * b
            a = lam * math.sqrt(r_m * b) / (12 * (1 - 0.09)) ** 0.25
            m21 = float(C.bulging_through_axial(a, r_m, b)[()])
            m1 = float(C.interpolate_columns(T.TABLES["M.1a"], 0, lam, rm_b).value[()])
            worst = max(worst, abs(m21 - m1) / m1)
    assert worst < 0.35  # the two models differ by up to a third on this range (recorded band)


def test_interpolation_envelope_and_suspect_status() -> None:
    from critflaw.solutions import cylinder as C

    table = T.TABLES["M.2"]
    rng = np.random.default_rng(0)
    for _ in range(200):
        ac = rng.uniform(0.05, 1.0)
        ab = rng.uniform(0.0, 0.8)
        br = rng.uniform(0.1, 0.25)
        for q in range(4):
            v = C.interpolate(table, q, ac, ab, br)
            assert bool(v.valid) and float(v.lower) <= float(v.value) <= float(v.upper)
    # the corners are reproduced exactly and queries outside the grid are invalid (never extrapolated)
    assert float(C.interpolate(table, 0, 0.4, 0.6, 0.25).value[()]) == float(np.asarray(table.values[0])[1, 3, 1])
    assert not bool(C.interpolate(table, 0, 0.4, 0.81, 0.25).valid)
    # suspect: through the M.2 a/c = 0.05 block at a/B >= 0.6 only
    assert bool(C.interpolate(table, 0, 0.06, 0.7, 0.1).suspect) and not bool(
        C.interpolate(table, 0, 0.3, 0.3, 0.1).suspect
    )
    # sparse-column tables: inside the printed gap of Table M.1 (r_m/B = 10) the value is interpolated across
    # the gap and flagged; a query exactly on another column is not affected by that column's gap
    v = C.interpolate_columns(T.TABLES["M.1a"], 0, 10.0, 10.0)
    assert bool(v.valid) and bool(v.suspect)
    assert not bool(C.interpolate_columns(T.TABLES["M.1a"], 0, 10.0, 20.0).suspect)
    # the sign-aware combination of a through-wall factor's envelope: [min M1 - max M2, max M1 + max M2]
    m1 = C.interpolate_columns(T.TABLES["M.1a"], 0, 3.0, 7.5)
    m2 = C.interpolate_columns(T.TABLES["M.1b"], 0, 3.0, 7.5)
    lo, hi = float(m1.lower - m2.upper), float(m1.upper + m2.upper)
    assert lo <= float(m1.value - m2.value) <= float(m1.value + m2.value) <= hi


# ---- B5: the partner's fixed-step integrator for the new kinds ------------------------------------------

ORACLE_I = _load("partner_reference_partB_integrator", HERE / "oracles" / "partner_reference_partB_integrator.py")


def _fixed_heun_kind(
    kind: int,
    a0: float,
    c0: float,
    thickness: float,
    width: float,
    law: GrowthLaw,
    sp: Spectrum,
    h: float,
    n: int,
    ligament: float | None = None,
    spec: Any = None,
) -> tuple[float, float]:
    """The production Heun step at a fixed step with the active set frozen (event-free case)."""
    lig = None if ligament is None else np.array([ligament])
    st = GrowthState.start(np.array([a0]), np.array([c0]), np.array([thickness]), np.array([width]), 1, ligament=lig)
    st.kind[:] = kind
    st.t_ended[:] = np.nan
    g = Grower(law, sp, 1, 1.0, cylinder=spec)
    g.bind(st)
    idx = np.array([0])
    for _ in range(n):
        dk = g.ranges(idx, st.a, st.c, st.kind, st.thickness, st.width)
        above, stage_b, _ = g.active_set(idx, dk, st.a, st.c, st.kind)
        va0, vc0, _ = g.velocities(
            idx, st.a, st.c, st.kind, st.t, st.thickness, st.width, above, stage_b, None, st.b_eff
        )
        _, _, a2, c2 = g.heun(st, idx, np.array([h]), above, stage_b, (va0, vc0))
        st.a[:], st.c[:] = a2, c2
        st.t += h
    return float(st.a[0]), float(st.c[0])


def test_partner_fixed_step_integrator_for_the_new_kinds_agrees_to_1e_8() -> None:
    from critflaw.fatigue.growth import CylinderSpec

    cases = ORACLE_I.tabulate()
    assert len(cases) == 5
    for case in cases:
        args = case["inputs"]["args"]
        expected = case["expected"]["value"]
        if case["inputs"]["function"] == "integrate_embedded_heun":
            a0, c0, p0, b, w, ds, a_si, m, nu, h, n = args
            law = GrowthLaw(a_si, m, 0.0)
            sp = Spectrum.repeating([(ds, 0.0, 1.0)], nu)
            a, c = _fixed_heun_kind(G.EMBEDDED, a0, c0, b, w, law, sp, h, n, ligament=p0)
            assert a == pytest.approx(expected[0], rel=1e-8) and c == pytest.approx(expected[1], rel=1e-8)
            assert 0.5 * (2 * a0 + 2 * p0 - 2 * a) == pytest.approx(expected[2], rel=1e-8)  # p from the constant B'
        else:
            a0, c0, b, r_i, ds, a_si, m, nu, h, n = args
            law = GrowthLaw(a_si, m, 0.0)
            sp = Spectrum.repeating([(ds, 0.0, 1.0)], nu)
            spec = CylinderSpec(r_i, "axial", "internal", "tabulated")
            a, c = _fixed_heun_kind(G.SURFACE, a0, c0, b, 10.0, law, sp, h, n, spec=spec)
            assert a == pytest.approx(expected[0], rel=1e-8) and c == pytest.approx(expected[1], rel=1e-8)


# ---- Part B code review round 1 (partner PB-01..PB-14) ---------------------------------------------------


def test_pb_moment_enters_k_and_ineligible_realisations_are_masked_not_dropped() -> None:
    # PB-01: a pure global moment gives a non-zero K for a circumferential flaw; PB-02/03/08: ineligible
    # realisations of a route are out of validity per sample, never a batch abort
    flaw = cf.CylinderSurfaceFlaw(5e-3, 40e-3, "circumferential", "internal")
    res = cf.Assessment(_cyl("tabulated"), flaw, _steel(), cf.StressState(pm_moment=50e6)).run()
    assert float(res.kr) > 0.0 and float(res.k_primary[0]) > 0.0
    # pressure-only with a mechanical stress in half of the realisations: that half is masked
    ax = cf.CylinderSurfaceFlaw(5e-3, 40e-3, "axial", "internal")
    loads = cf.StressState(pm=np.array([0.0, 1e6]), pressure=PRESS)
    masked = cf.Assessment(_cyl("pressure_only"), ax, _steel(), loads).run(on_invalid="mask")
    assert masked.valid.tolist() == [True, False] and np.isfinite(masked.radial_margin[0])
    with pytest.raises(cf.OutOfValidityError):
        cf.Assessment(_cyl("pressure_only"), ax, _steel(), loads).run()
    # thick_pressure refuses mechanical components per sample; the thick circumferential limit refuses pb
    tp = cf.Assessment(
        _cyl("tabulated", "thick_pressure"),
        ax,
        _steel(),
        cf.StressState(pm=np.array([0.0, 2e6]), pressure=cf.PressureLoad(20e6, 5.5)),
    )
    assert tp.run(on_invalid="mask").valid.tolist() == [True, False]
    circ = cf.ExtendedFlaw(5e-3, "circumferential", "internal")
    th = cf.Assessment(
        _cyl("tabulated", "thick"), circ, _steel(), cf.StressState(pm=50e6, pb=np.array([0.0, 5e6]), pm_moment=30e6)
    )
    assert th.run(on_invalid="mask").valid.tolist() == [True, False]


def test_pb_pure_bending_thick_wall_conversion_and_extended_embedded_stability() -> None:
    from critflaw.solutions import cylinder_sref as S

    # PB-04: sref = (2/3) Pb / L_b under pure bending, continuous with the vanishing-membrane limit
    flaw = cf.CylinderSurfaceFlaw(5e-3, 40e-3, "axial", "external")
    pure = cf.Assessment(_cyl("tabulated", "thick"), flaw, _steel(), cf.StressState(pb=60e6)).run()
    _, l_b, _ = S.axial_external_thick_limits(0.0, 60e6, 5e-3, 20e-3, 0.02, 2.0)
    assert float(pure.reference_stress) == pytest.approx((2.0 / 3.0) * 60e6 / float(l_b[()]), rel=1e-12)
    almost = cf.Assessment(_cyl("tabulated", "thick"), flaw, _steel(), cf.StressState(pm=1.0, pb=60e6)).run()
    assert float(almost.reference_stress) == pytest.approx(float(pure.reference_stress), rel=1e-6)
    # PB-13: the extended embedded reference stress at Pm = 0 and at zero load
    assert float(E.extended_embedded_reference_stress(0.0, 30e6, 1e-3, 5e-3, B)[()]) == pytest.approx(
        float(E.extended_embedded_reference_stress(1e-6, 30e6, 1e-3, 5e-3, B)[()]), rel=1e-6
    )
    assert float(E.extended_embedded_reference_stress(0.0, 0.0, 1e-3, 5e-3, B)[()]) == 0.0


def test_pb_p10_2_both_branches_and_the_equality_case() -> None:
    # PB-14: the second branch (pi r_i < c + B) needs a thick wall to stay within c/(pi r_i) <= 0.8
    from critflaw.solutions import cylinder_sref as S

    r_i, b = 0.1, 0.08
    boundary = math.pi * r_i - b  # c at which the two forms meet
    for c in (0.9 * boundary, boundary, 1.05 * boundary):
        mine = float(S.circumferential_surface_thin(100.0, 20.0, 0.01, c, r_i, b)[()])
        ref = ORACLE_C.circumferential_thin("surface", 0.01, b, r_i, P_m=100.0, P_b=20.0, c=c)["sigma_ref"]
        assert mine == pytest.approx(ref, rel=1e-14)
        assert S.circumferential_surface_thin_valid(c, r_i)[()]
    # the two forms differ away from the boundary: swapping them is detected
    c = 1.05 * boundary
    ab = 0.01 / b
    alpha_first = ab / (1.0 + b / c)
    alpha_second = ab * (c / (math.pi * r_i))
    assert alpha_first != pytest.approx(alpha_second, rel=1e-3)
    mine = float(S.circumferential_surface_thin(100.0, 20.0, 0.01, c, r_i, b)[()])
    wrong = 100.0 * (math.pi * (1 - ab) + 2 * ab * math.sin(c / r_i)) / (
        (1 - ab) * (math.pi - (c / r_i) * ab)
    ) + 2 * 20.0 / (3 * (1 - alpha_first) ** 2)
    assert mine != pytest.approx(wrong, rel=1e-6)


def test_pb_extended_flaws_are_inspected_by_depth_and_random_pressure_is_discovered() -> None:
    # PB-09: an extended flaw with unit PoD is detected and repaired
    law, sp = _cyl_law_spectrum()
    cyl_ = cf.Cylinder(r_i=0.1, thickness=0.02, length=1.0, k_route="tabulated")
    problem = cf.Assessment(
        cyl_, cf.ExtendedFlaw(4e-3, "circumferential", "internal"), _steel(), cf.StressState(pm=10e6)
    )
    ext = ExtremeLoad(prob.Uniform(20e6, 20.001e6), 1.0, component="pm")
    run = TimeDependentAssessment(
        problem,
        law,
        sp,
        ext,
        times=range(10, 51, 10),
        leak_is_failure=False,
        inspections=[Inspection(10.0, ConstantPoD(1.0))],
    )
    res = run.solve(seed=0, samples=4, stage_sensitivity=False)
    assert res.inspections[0]["p_detect_all"] == 1.0 and res.events_any["repaired"] == 1.0
    assert res.cumulative_upper.value[-1] == 0.0  # repaired: no flaw left to end
    # PB-10: random fields inside a PressureLoad are discovered and realised jointly
    rp = cf.StressState(pressure=cf.PressureLoad(prob.Uniform(4e6, 6e6), 5.5))
    problem_r = cf.Assessment(_cyl("tabulated"), cf.CylinderSurfaceFlaw(5e-3, 40e-3, "axial", "internal"), _steel(), rp)
    assert problem_r.random_fields() == ["loads.pressure"]
    with pytest.raises(TypeError):
        problem_r.run()
    pa = cf.ProbabilisticAssessment(problem_r)
    u = np.random.default_rng(0).standard_normal((8, pa.vector.size))
    r = pa.realise(u)
    assert r.loads.pressure is not None
    p = np.asarray(r.loads.pressure.pressure)
    assert p.shape == (8,) and np.all((p >= 4e6) & (p <= 6e6))
    out = pa.evaluate(u)
    assert np.all(out.valid) and np.all(np.diff(out.lr[np.argsort(p)]) > 0)  # Lr rises with the realised pressure
    # a random inner radius is refused in growth until vectorised
    cyl_r = cf.Cylinder(r_i=prob.Uniform(0.1, 0.11), thickness=0.02, length=1.0, k_route="tabulated")
    with pytest.raises(NotImplementedError, match="radius"):
        TimeDependentAssessment(
            cf.Assessment(
                cyl_r, cf.CylinderSurfaceFlaw(5e-3, 40e-3, "axial", "internal"), _steel(), cf.StressState(pm=10e6)
            ),
            law,
            sp,
            ext,
            times=[1],
            leak_is_failure=False,
        )


def test_pb_tabulated_projection_and_boundaries_follow_the_solution_family() -> None:
    # PB-05/07: a circumferential external flaw under the tabulated route uses the plate solution: no table
    # projection, no table edge, no axial width limit; PB-06: the a/c table minimum is a boundary (trigger 9)
    from critflaw.fatigue.growth import CylinderSpec

    law, sp = _cyl_law_spectrum()
    b, r_i, w = 0.02, 0.1, 0.5
    ext_spec = CylinderSpec(r_i, "circumferential", "external", "tabulated")
    g = Grower(law, sp, 1, 1.0, cylinder=ext_spec)
    a = np.array([0.9 * b])
    c = np.array([0.05])  # 2c/W = 0.2 > 0.15: fine for the plate family (a/2c = 0.18, no long-flaw limit)
    kind = np.array([G.SURFACE])
    a2, c2 = g.clip_inside(a, c, kind, np.array([b]), np.array([w]))
    assert float(a2[0]) == 0.9 * b and float(c2[0]) == 0.05
    excess = g.boundary_excess(a, c, kind, np.array([b]), np.array([w]))
    assert np.all(excess[0, [8, 9, 11]] == -np.inf) and excess[0, 0] < 0.0
    # the a/c minimum of the axial tables (0.05): a long flaw starting just above it leaves by trigger 9
    ax = CylinderSpec(r_i, "axial", "internal", "tabulated")
    st = GrowthState.start(np.array([2e-3]), np.array([2e-3 / 0.0505]), np.array([b]), np.array([w]), 1)
    g2 = Grower(law, Spectrum.repeating([(30e6, 0.0, 1.0)], 1e5), 1, 5.0, cylinder=ax)
    excess0 = g2.boundary_excess(st.a, st.c, st.kind, st.thickness, st.width)
    assert -0.01 < excess0[0, 8] < 0.0  # just inside the a/c minimum
    # the rate evaluation does not clamp: a state below the minimum is NaN (invalid), not evaluated at the edge
    dk = g2.ranges(np.array([0]), np.array([2e-3]), np.array([2e-3 / 0.04]), st.kind, st.thickness, st.width)
    assert np.all(np.isnan(dk))


# ---- Part B code review round 1 (findings PB-C-04/05, 06, 07, 12, 13, 23, 24) ----------------------------


def test_pbc_destination_depth_rule_triggers_and_mid_thickness_record() -> None:
    # PB-C-13: the re-characterisation depth is 2a + p (pinned on the width case of B20); PB-C-07: triggers
    # are per event and a plain breakthrough carries none; PB-C-24: the mid-thickness case by its event record
    law = GrowthLaw(float(coefficient_to_si(5.86e-13, 2.88)), 2.88, 0.0)
    sp = Spectrum.repeating([(100e6, 0.0, 1.0)], 1e5)
    a0, c0, p0 = 2e-3, 0.5 * W * 0.5 * (1 - 1e-6), 5e-3
    st = GrowthState.start(np.array([a0]), np.array([c0]), np.array([B]), np.array([W]), 1, ligament=np.array([p0]))
    Grower(law, sp, 1, 1.0).advance(st, 1.0)
    a_b, c_b = st.event_pre[0, 0, 0], st.event_pre[0, 0, 1]
    p_b = 0.5 * (2 * a0 + 2 * p0) - a_b
    assert st.event_post[0, 0, 0] == pytest.approx(2 * a_b + p_b, rel=1e-12) and st.event_post[0, 0, 1] == c_b
    assert st.event_trigger[0, 0] == 7 and st.leak_trigger[0] == 7  # the embedded width limit
    # a plain breakthrough: no trigger on the event, leak_trigger stays 0 until a leak or re-characterisation
    st2 = GrowthState.start(
        np.array([2e-3]), np.array([8e-3]), np.array([B]), np.array([W]), 1, ligament=np.array([5e-3])
    )
    g2 = Grower(law, sp, 1, 6.0)
    g2.advance(st2, 5.0)
    assert st2.kind[0] == G.SURFACE and st2.event_trigger[0, 0] == 0 and st2.leak_trigger[0] == 0
    assert st2.event_post[0, 0, 0] == pytest.approx(2 * st2.event_pre[0, 0, 0], rel=1e-9)  # p = 0: depth 2a
    # mid-thickness: two events at the same time, surface_breaking then leak (trigger 1), by the record
    st3 = GrowthState.start(
        np.array([4e-3]), np.array([8e-3]), np.array([B]), np.array([W]), 1, ligament=np.array([8.5e-3])
    )
    Grower(law, sp, 1, 10.0).advance(st3, 10.0)
    assert st3.event_count[0] == 2 and st3.event_post[0, 1, 2] == G.THROUGH
    assert st3.event_time[0, 0] == st3.event_time[0, 1] and st3.event_trigger[0, 1] == 1 and st3.leak_trigger[0] == 1


def test_pbc_hovering_at_the_embedded_branch() -> None:
    # B6: an embedded flaw at a/c = 1 (the branch of the solution) grows without stalling and matches the
    # cycle-by-cycle reference (which crosses the branch freely)
    law = GrowthLaw(float(coefficient_to_si(5.86e-13, 2.88)), 2.88, 0.0)
    sp = Spectrum.repeating([(100e6, 0.0, 1.0)], 1e5)
    a0, c0, p0 = 3e-3, 3e-3, 6e-3
    st = GrowthState.start(np.array([a0]), np.array([c0]), np.array([B]), np.array([W]), 1, ligament=np.array([p0]))
    g = Grower(law, sp, 1, 2.0)
    g.advance(st, 2.0)
    assert st.kind[0] == G.EMBEDDED and st.steps[0] < 400
    a, c, n, code = _embedded_reference(law, a0, c0, p0, 100e6, 200_000, B, W)
    assert code == 0 and float(st.a[0]) == pytest.approx(a, rel=2e-3) and float(st.c[0]) == pytest.approx(c, rel=2e-3)


def test_pbc_table_edge_destination_outside_the_through_wall_range_ends_at_once() -> None:
    # B7 (the design's common case): B/r_i = 0.25 gives r_m/B = 4.5 < 5, so the through-wall destination of a
    # surface flaw at the table edge is outside the through-wall range: leak_assumed (8) then ENDED (10) at once
    from critflaw.fatigue.growth import CylinderSpec

    law = GrowthLaw(float(coefficient_to_si(5.86e-13, 2.88)), 2.88, 0.0)
    sp = Spectrum.repeating([(30e6, 0.0, 1.0)], 1e5)
    b, r_i, w = 0.025, 0.1, 2.0
    st = GrowthState.start(np.array([4e-3]), np.array([40e-3]), np.array([b]), np.array([w]), 1)
    g = Grower(law, sp, 1, 100.0, cylinder=CylinderSpec(r_i, "axial", "internal", "tabulated"))
    for t in (20, 40, 60, 80, 100):
        st.new_segment()
        g.advance(st, float(t))
        if st.kind[0] == G.ENDED:
            break
    assert st.kind[0] == G.ENDED and st.event_count[0] == 2
    assert st.event_trigger[0, 0] == 8 and st.event_trigger[0, 1] == 10 and st.event_time[0, 0] == st.event_time[0, 1]
    assert st.t_ended[0] == st.t_leak_assumed[0] and st.event_pre[0, 0, 0] == pytest.approx(0.8 * b, rel=1e-9)
    assert st.a[0] == b  # the ended through-wall destination (a = B)


def test_pbc_singular_kinds_end_at_the_declared_zone() -> None:
    # PB-C-04/05: the (M.22) axial surface flaw under pressure-only and the (M.26) fully circumferential external
    # flaw have a K singular at a = B; they end at a/B = 0.9 with leak_assumed (trigger 17), never raising
    from critflaw.fatigue.growth import CylinderSpec

    law = GrowthLaw(float(coefficient_to_si(5.86e-13, 2.88)), 2.88, 0.0)
    sp = Spectrum.repeating([(30e6, 0.0, 1.0)], 1e5)
    b, r_i, w = 0.02, 0.1, 2.0
    st = GrowthState.start(np.array([4e-3]), np.array([40e-3]), np.array([b]), np.array([w]), 1)
    g = Grower(law, sp, 1, 200.0, cylinder=CylinderSpec(r_i, "axial", "internal", "pressure_only"))
    g.advance(st, 200.0)
    assert st.kind[0] == G.ENDED and st.leak_trigger[0] == 17 and st.a[0] == pytest.approx(0.9 * b, rel=1e-9)
    st2 = GrowthState.start(np.array([4e-3]), np.array([w]), np.array([b]), np.array([w]), 1)
    st2.kind[:] = G.EXTENDED
    st2.t_ended[:] = np.nan
    g2 = Grower(law, sp, 1, 200.0, cylinder=CylinderSpec(r_i, "circumferential", "external", "tabulated"))
    g2.advance(st2, 200.0)
    assert st2.kind[0] == G.ENDED and st2.leak_trigger[0] == 17 and st2.a[0] == pytest.approx(0.9 * b, rel=1e-9)


def test_pbc_inside_tensile_bending_is_linear_superposition_and_the_half_angle_reading() -> None:
    # PB-C-06: with the bending tensile inside the bending term changes sign at both surfaces
    flaw = cf.CylinderThroughFlaw(0.05, "circumferential")
    out = cf.Assessment(_cyl("tabulated"), flaw, _steel(), cf.StressState(pm=50e6, pb=20e6)).run()
    inn = cf.Assessment(
        _cyl("tabulated"), flaw, _steel(), cf.StressState(pm=50e6, pb=20e6, bending_tensile_surface="inside")
    ).run()
    mem = cf.Assessment(_cyl("tabulated"), flaw, _steel(), cf.StressState(pm=50e6)).run()
    k_b_out = out.k_primary - mem.k_primary  # the bending contribution at (outside, inside)
    k_b_in = inn.k_primary - mem.k_primary
    np.testing.assert_allclose(k_b_in, -k_b_out, rtol=1e-12)
    assert "bending_tensile_inside" in inn.trace.options
    # PB-C-23: the half-angle readings c / r_m and c / r_i differ by B / (2 r_m) in theta and measurably in Lr
    # (round 3: the conservative reading of each pair is implemented, test_r3_half_angle_...)
    from critflaw.solutions import cylinder_dispatch as D

    r_m = 0.1 + 0.01
    thick = D._net_section(
        np.array([5e-3]), 0.04 / r_m, np.array([r_m]), np.array([0.02]), np.array([50e6]), np.array([30e6]), False
    )
    alt = D._net_section(
        np.array([5e-3]), 0.04 / 0.1, np.array([r_m]), np.array([0.02]), np.array([50e6]), np.array([30e6]), False
    )
    assert float(thick[0]) != pytest.approx(float(alt[0]), rel=1e-4)


# ---- Part B code review round 2 (partner R2-01..04 and the still-open items) ----------------------------


def test_r2_global_moment_enters_the_reference_stress_once() -> None:
    # R2-01: a pure global moment on a circumferential surface flaw: the thin form takes the total P_m once;
    # the thick net-section form takes zero uniform force and the moment only
    from critflaw.solutions import cylinder_dispatch as D
    from critflaw.solutions import cylinder_sref as S

    flaw = cf.CylinderSurfaceFlaw(5e-3, 40e-3, "circumferential", "internal")
    thin = cf.Assessment(_cyl("tabulated"), flaw, _steel(), cf.StressState(pm_moment=50e6)).run()
    expected = float(S.circumferential_surface_thin(50e6, 0.0, 5e-3, 20e-3, 0.1, 0.02)[()])
    assert float(thin.reference_stress) == pytest.approx(expected, rel=1e-12)
    mixed = cf.Assessment(_cyl("tabulated"), flaw, _steel(), cf.StressState(pm=30e6, pm_moment=50e6)).run()
    assert float(mixed.reference_stress) == pytest.approx(
        float(S.circumferential_surface_thin(80e6, 0.0, 5e-3, 20e-3, 0.1, 0.02)[()]), rel=1e-12
    )
    thick = cf.Assessment(_cyl("tabulated", "thick"), flaw, _steel(), cf.StressState(pm_moment=50e6)).run()
    r_m = 0.1 + 0.01
    ref = D._net_section(  # round 3: the half-angle of an internal flaw is read at its own surface, c / r_i
        np.array([5e-3]), 20e-3 / 0.1, np.array([r_m]), np.array([0.02]), np.array([0.0]), np.array([50e6]), False
    )
    assert float(thick.reference_stress) == pytest.approx(float(ref[0]), rel=1e-12)
    # K still sees the moment (R2-01 keeps PB-01)
    assert float(thin.k_primary[0]) > 0.0


def test_r2_pressure_only_cylinder_finish_uses_the_real_thickness() -> None:
    # R2-02: the quadrature finish of a pressure-only axial through-wall flaw integrates 1/v with the bulging
    # factor at the sample's thickness; the end time matches scipy, with a threshold crossing inside the window
    from scipy import integrate

    from critflaw.fatigue.growth import CylinderSpec
    from critflaw.solutions import cylinder as C

    b, r_i, w, nu = 0.02, 0.1, 0.3, 1e5
    thr = 80e6  # a threshold crossed inside the finish window at this stress range
    law = GrowthLaw(float(coefficient_to_si(5.86e-13, 2.88)), 2.88, thr)
    sp = Spectrum.repeating([(30e6, 0.0, 1.0), (10e6, 0.0, 1.0)], nu)

    def inv_v(c: float) -> float:
        bulge = float(C.bulging_through_axial(c, r_i + 0.5 * b, b)[()])
        fw = float(plate.through_flaw_fw(c, w)[()])
        v = sum(float(law.rate(bulge * fw * ds * math.sqrt(math.pi * c))) * nu * 0.5 for ds in (30e6, 10e6))
        return 1.0 / v

    c0 = 0.136  # inside the terminal zone (2c/W >= 0.9) from the start: the finish alone
    t_total = integrate.quad(inv_v, c0, 0.15, epsabs=0, epsrel=1e-12, limit=400)[0]
    st = GrowthState.start(np.array([b]), np.array([c0]), np.array([b]), np.array([w]), 2)
    st.kind[:] = G.THROUGH
    st.t_ended[:] = np.nan
    g = Grower(law, sp, 1, 1.0, cylinder=CylinderSpec(r_i, "axial", "internal", "pressure_only"))
    g.advance(st, 1.0)
    assert st.kind[0] == G.ENDED and st.t_ended[0] == pytest.approx(t_total, rel=1e-8)
    # the 10 MPa block crosses its threshold inside the window (a kink the finish must split at)
    dk_lo = (
        float(C.bulging_through_axial(c0, r_i + 0.5 * b, b)[()])
        * float(plate.through_flaw_fw(c0, w)[()])
        * 10e6
        * math.sqrt(math.pi * c0)
    )
    dk_hi = (
        float(C.bulging_through_axial(0.149, r_i + 0.5 * b, b)[()])
        * float(plate.through_flaw_fw(0.149, w)[()])
        * 10e6
        * math.sqrt(math.pi * 0.149)
    )
    assert dk_lo < thr < dk_hi


def test_r2_invalid_trial_is_shortened_not_accepted() -> None:
    # R2-03: a predictor at a corner of the table domain (width and aspect limits at once) gives a trial the
    # solution does not define; the step is shortened and the state never holds a NaN
    from critflaw.fatigue.growth import CylinderSpec

    law = GrowthLaw(float(coefficient_to_si(5.86e-13, 2.88)), 2.88, 0.0)
    sp = Spectrum.repeating([(30e6, 0.0, 1.0)], 1e5)
    b, r_i, w = 0.02, 0.1, 0.2
    g = Grower(law, sp, 1, 1.0, cylinder=CylinderSpec(r_i, "axial", "internal", "tabulated"))
    a2, c2 = g.clip_inside(np.array([0.0151]), np.array([0.0152]), np.array([G.SURFACE]), np.array([b]), np.array([w]))
    assert float(c2[0]) >= float(a2[0])  # a/c <= 1 always holds after the projection
    st = GrowthState.start(np.array([0.0149]), np.array([0.0151]), np.array([b]), np.array([w]), 1)
    assert st.kind[0] == G.SURFACE
    g.advance(st, 0.5)
    assert np.all(np.isfinite(st.a)) and np.all(np.isfinite(st.c))


def test_r2_growth_sign_convention_fw_and_by_trigger() -> None:
    # PB-C-06 in growth: the inside-tensile range is the outside-tensile one with the bending term negated;
    # R2-04: the plate-directed circumferential external surface flaw carries f_w in growth as in the assessment
    from critflaw.fatigue.growth import CylinderSpec

    law = GrowthLaw(float(coefficient_to_si(5.86e-13, 2.88)), 2.88, 0.0)
    b, r_i, w = 0.02, 0.1, 2.0
    idx = np.array([0])
    for sp_b in (Spectrum.repeating([(30e6, 20e6, 1.0)], 1e5),):
        out = Grower(
            law, sp_b, 1, 1.0, cylinder=CylinderSpec(r_i, "circumferential", "internal", "tabulated", "outside")
        )
        inn = Grower(
            law, sp_b, 1, 1.0, cylinder=CylinderSpec(r_i, "circumferential", "internal", "tabulated", "inside")
        )
        mem = Grower(
            law,
            Spectrum.repeating([(30e6, 0.0, 1.0)], 1e5),
            1,
            1.0,
            cylinder=CylinderSpec(r_i, "circumferential", "internal", "tabulated"),
        )
        args = (idx, np.array([b]), np.array([0.05]), np.array([G.THROUGH]), np.array([b]), np.array([w]))
        # the range is the larger of the two tips: with the bending negated the inside tip governs instead
        dk_out, dk_in, dk_m = out.ranges(*args)[0, 0, 1], inn.ranges(*args)[0, 0, 1], mem.ranges(*args)[0, 0, 1]
        assert dk_out != dk_in and dk_out > dk_m
        # PB-C3-01: under both declarations the growth range equals the static K range (the maximum over the two
        # surfaces of the dispatch's factors), so the growth convention is the assessment's, not merely different
        from critflaw.solutions import cylinder_dispatch as D

        for g, surface in ((out, "outside"), (inn, "inside")):
            cylr = cf.Cylinder(r_i=r_i, thickness=b, length=w, k_route="tabulated")
            f = D.cylinder_factors(
                cylr,
                cf.CylinderThroughFlaw(0.1, "circumferential"),
                cf.StressState(pm=30e6, pb=20e6, bending_tensile_surface=surface),
                np.asarray(30e6),
                np.asarray(20e6),
            )
            k_static = max(
                float((mm_ * 30e6 * f.membrane_multiplier + mb_ * 20e6) * f.bulging * f.fw * math.sqrt(math.pi * 0.05))
                for mm_, mb_ in zip(f.mm, f.mb, strict=True)
            )
            assert float(g.ranges(*args)[0, 0, 1]) == pytest.approx(k_static, rel=1e-12), surface
    ext = Grower(
        law,
        Spectrum.repeating([(30e6, 0.0, 1.0)], 1e5),
        1,
        1.0,
        cylinder=CylinderSpec(r_i, "circumferential", "external", "tabulated"),
    )
    a, c = np.array([5e-3]), np.array([0.2])
    dk = ext.ranges(idx, a, c, np.array([G.SURFACE]), np.array([b]), np.array([w]))[0, 0, 0]
    mm = float(plate.surface_flaw_mm_mb(a, c, np.array([b]), plate.THETA_DEEPEST)[0][0])
    fw = float(plate.surface_flaw_fw(a, c, np.array([b]), np.array([w]))[0])
    assert float(dk) == pytest.approx(fw * mm * 30e6 * math.sqrt(math.pi * 5e-3), rel=1e-12) and fw > 1.0
    # PB-C-07: by_trigger reports every event code of the run (8 then 10 at once in B7's common case)
    cyl_ = cf.Cylinder(r_i=0.1, thickness=0.025, length=2.0, k_route="tabulated")
    problem = cf.Assessment(
        cyl_, cf.CylinderSurfaceFlaw(4e-3, 40e-3, "axial", "internal"), _steel(), cf.StressState(pm=10e6)
    )
    run = TimeDependentAssessment(
        problem,
        law,
        Spectrum.repeating([(60e6, 0.0, 1.0)], 1e5),
        ExtremeLoad(prob.Uniform(20e6, 20.001e6), 1.0, component="pm"),
        times=range(5, 21, 5),
        leak_is_failure=False,
    )
    res = run.solve(seed=0, samples=3, stage_sensitivity=False)
    assert res.events_any["by_trigger"] == {8: 1.0, 10: 1.0}


def test_r2_convergence_cap_overflow_and_pass_guard_boundary() -> None:
    # PB-C-12: (i) the tabulated-route controller error falls with tol (B7 convergence); (ii) the threshold cap
    # does not apply to embedded flaws; (iii) a fifth event in one segment raises; (iv) the pass guard trips just
    # below the exempt tenth of the lambda range and not inside it
    from critflaw.fatigue.growth import CylinderSpec
    from critflaw.solutions import cylinder as C

    law = GrowthLaw(float(coefficient_to_si(5.86e-13, 2.88)), 2.88, 0.0)
    sp = Spectrum.repeating([(30e6, 0.0, 1.0)], 1e5)
    b, r_i, w = 0.02, 0.1, 2.0
    sizes = {}
    for tol in (2e-4, 5e-5):
        st = GrowthState.start(np.array([4e-3]), np.array([10e-3]), np.array([b]), np.array([w]), 1)
        Grower(law, sp, 1, 40.0, tol=tol, cylinder=CylinderSpec(r_i, "axial", "internal", "tabulated")).advance(
            st, 40.0
        )
        sizes[tol] = float(st.a[0])
    st = GrowthState.start(np.array([4e-3]), np.array([10e-3]), np.array([b]), np.array([w]), 1)
    Grower(law, sp, 1, 40.0, tol=1e-6, cylinder=CylinderSpec(r_i, "axial", "internal", "tabulated")).advance(st, 40.0)
    fine = float(st.a[0])
    assert abs(sizes[5e-5] - fine) < abs(sizes[2e-4] - fine)  # the error falls with the tolerance
    # (ii) threshold cap
    capped = GrowthLaw(1e-12, 3.0, 100.0 * NMM, shallow_cap=(1e-3, 63.0 * NMM))
    gc = Grower(capped, sp, 1, 1.0)
    assert gc.threshold_at(np.array([0]), np.array([0.5e-3]), np.array([G.EMBEDDED]))[0] == pytest.approx(100.0 * NMM)
    assert gc.threshold_at(np.array([0]), np.array([0.5e-3]), np.array([G.SURFACE]))[0] == pytest.approx(63.0 * NMM)
    # (iii) event capacity
    st = GrowthState.start(np.array([1e-3]), np.array([5e-3]), np.array([b]), np.array([w]), 1)
    for k in range(G.EVENT_CAPACITY):
        st.record_event(0, (1e-3, 5e-3, G.SURFACE), (1e-3, 5e-3, G.SURFACE), 0.1 * k)
    with pytest.raises(RuntimeError, match="none is dropped"):
        st.record_event(0, (1e-3, 5e-3, G.SURFACE), (1e-3, 5e-3, G.SURFACE), 0.9)
    # (iv) pass guard: a fast through-wall crack below 0.9 of the lambda end trips the guard, inside the tenth it does not
    fast = GrowthLaw(float(coefficient_to_si(5.86e-13, 2.88)) * 300, 2.88, 0.0)
    lam_end = float(C.lambda_range_end(T.TABLES["M.1a"], (r_i + 0.5 * b) / b)[()])
    c_of = lambda lam: lam * math.sqrt((r_i + 0.5 * b) * b) / (12 * (1 - 0.09)) ** 0.25  # noqa: E731
    for frac, trips in ((0.85, True), (0.95, False)):
        st = GrowthState.start(np.array([b]), np.array([c_of(frac * lam_end)]), np.array([b]), np.array([w]), 1)
        st.kind[:] = G.THROUGH
        st.t_ended[:] = np.nan
        g = Grower(fast, sp, 1, 1.0, cylinder=CylinderSpec(r_i, "axial", "internal", "tabulated"))
        if trips:
            with pytest.raises(ValueError, match="one pass"):
                g.advance(st, 1.0)
        else:
            g.advance(st, 1.0)
            assert st.kind[0] == G.ENDED


# ---- Part B code review round 2 (findings PB-C2-01..07 and the surviving mutants) --------------------------


def test_r2c_embedded_sizing_acts_on_the_height() -> None:
    # PB-C-02 pinned: accept_below between a and 2a repairs (the height is inspected), below a does not change
    law = GrowthLaw(float(coefficient_to_si(5.86e-13, 2.88)), 2.88, 0.0)
    sp = Spectrum.repeating([(100e6, 0.0, 1.0)], 1e5)
    problem = cf.Assessment(cf.FlatPlate(B, W), cf.EmbeddedFlaw(4e-3, 16e-3, 5e-3), _steel(), cf.StressState(pm=100e6))
    ext = ExtremeLoad(prob.Uniform(200e6, 200.001e6), 1.0)
    state = _start_state_for(problem)
    half = float(state.a[0])
    run = TimeDependentAssessment(problem, law, sp, ext, times=[1, 2], leak_is_failure=False)
    for accept, repaired in ((1.5 * half, 1.0), (2.5 * half, 0.0)):
        insp = Inspection(
            1.0,
            ConstantPoD(0.0),
            pod_embedded=ConstantPoD(1.0),
            sizing_error=prob.Normal(0.0, 1e-9),
            accept_below=accept,
        )
        res = TimeDependentAssessment(
            problem, law, sp, ext, times=[1, 2], leak_is_failure=False, inspections=[insp]
        ).solve(seed=0, samples=4, stage_sensitivity=False)
        assert res.events_any["repaired"] == repaired, accept
    assert run.n_steps == 2


def _start_state_for(problem: cf.Assessment) -> GrowthState:
    from critflaw.fatigue.timedependent import _start_state

    return _start_state(problem, 1, 1)


def test_r2c_dispatch_envelope_width_zone_gating_and_landings() -> None:
    from critflaw.fatigue.growth import CylinderSpec
    from critflaw.solutions import cylinder as C
    from critflaw.solutions import cylinder_dispatch as D

    # PB-11 pinned on the dispatch result: the envelope contains M_1 +/- M_2 and M_3 +/- M_4 at both surfaces
    f = D.cylinder_factors(
        _cyl("tabulated"),
        cf.CylinderThroughFlaw(0.05, "axial"),
        cf.StressState(pm=50e6, pb=20e6),
        np.asarray(50e6),
        np.asarray(20e6),
    )
    assert f.envelope is not None
    lam = float(C.shell_lambda(0.025, 0.11, 0.02)[()])
    vals = [C.interpolate_columns(T.TABLES[n], 0, lam, 5.5) for n in ("M.1a", "M.1b", "M.1c", "M.1d")]
    for combo in (
        vals[0].value + vals[1].value,
        vals[0].value - vals[1].value,
        vals[2].value + vals[3].value,
        vals[2].value - vals[3].value,
    ):
        assert float(f.envelope[0][()]) <= float(combo[()]) <= float(f.envelope[1][()])
    assert float(f.envelope[0][()]) <= float(vals[0].lower - vals[1].upper)
    # PB-12 pinned: a tabulated through flaw in the plate width zone (2c/W >= 0.9) is NOT exempt from the pass guard
    fast = GrowthLaw(float(coefficient_to_si(5.86e-13, 2.88)) * 300, 2.88, 0.0)
    sp = Spectrum.repeating([(30e6, 0.0, 1.0)], 1e5)
    b, r_i = 0.02, 0.1
    st = GrowthState.start(np.array([b]), np.array([0.095]), np.array([b]), np.array([0.2]), 1)  # 2c/W = 0.95
    st.kind[:] = G.THROUGH
    st.t_ended[:] = np.nan
    with pytest.raises(ValueError, match="one pass"):
        Grower(fast, sp, 1, 1.0, cylinder=CylinderSpec(r_i, "axial", "internal", "tabulated")).advance(st, 1.0)
    # landing on the axial width edge 2c/W = 0.15 (trigger 15). The a/c table ends are not reachable by
    # membrane-loaded growth (a/c moves towards 1 from either side, checked here), so no landing test exists for them
    law = GrowthLaw(float(coefficient_to_si(5.86e-13, 2.88)), 2.88, 0.0)
    spec = CylinderSpec(r_i, "axial", "internal", "tabulated")
    st9 = GrowthState.start(np.array([5e-3]), np.array([5e-3 * (1 + 1e-6)]), np.array([b]), np.array([2.0]), 1)
    Grower(law, sp, 1, 2.0, cylinder=spec).advance(st9, 2.0)
    assert st9.event_count[0] == 0 and st9.a[0] / st9.c[0] < 1.0 - 1e-6
    st15 = GrowthState.start(np.array([10e-3]), np.array([0.075 * 2.0 * (1 - 1e-6)]), np.array([b]), np.array([2.0]), 1)
    Grower(law, sp, 1, 50.0, cylinder=spec).advance(st15, 50.0)
    assert st15.event_count[0] >= 1 and st15.event_trigger[0, 0] == 15 and st15.leak_trigger[0] == 15
    # the embedded branch flag: a > c flags the c/a branch for the embedded kind as for the surface flaw
    g = Grower(law, sp, 1, 1.0)
    dk = np.ones((2, 1, 2))
    _, _, branch = g.active_set(
        np.array([0, 0]), dk, np.array([2e-3, 1e-3]), np.array([1e-3, 2e-3]), np.array([G.EMBEDDED, G.EMBEDDED])
    )
    assert branch.tolist() == [True, False]


def test_r2c_side_loop_to_capacity_first_origin_and_undefined_rate_end() -> None:
    from critflaw.fatigue.growth import CylinderSpec

    # PB-C-14 pinned: a segment with two events whose second pre-side is the most severe checkpoint (the
    # mid-thickness embedded case: surface_breaking and leak at once; the pre-leak surface state at a = B is the
    # most severe configuration of the step)
    law = GrowthLaw(float(coefficient_to_si(5.86e-13, 2.88)), 2.88, 0.0)
    sp = Spectrum.repeating([(100e6, 0.0, 1.0)], 1e5)
    problem = cf.Assessment(
        cf.FlatPlate(B, W), cf.EmbeddedFlaw(8e-3, 16e-3, 8.5e-3), _steel(), cf.StressState(pm=100e6)
    )
    run = TimeDependentAssessment(
        problem, law, sp, ExtremeLoad(prob.Uniform(150e6, 150.001e6), 1.0), times=range(1, 8), leak_is_failure=False
    )
    r = run.run(np.zeros((1, run.dimension)))
    k = int(r["leak_step"][0])
    assert k >= 0 and r["state"].event_count[0] == 0  # the events were in step k's segment (cleared since)
    assert r["g_min"][0, k] < r["g_max"][0, k]
    # the second event's pre-side (the surface flaw at a = B) is more severe than every plain checkpoint
    hist = run.grow(load=150.001e6)
    plain_min = np.nanmin(hist.g)
    assert r["g_min"][0, k] < plain_min
    # N16: a leak_assumed followed by a leak keeps the first origin
    st = GrowthState.start(
        np.array([1e-3]), np.array([12e-3]), np.array([B]), np.array([W]), 1, ligament=np.array([0.5e-3])
    )
    Grower(law, sp, 1, 20.0).advance(st, 20.0)
    assert st.kind[0] == G.ENDED and np.isfinite(st.t_leak[0]) and st.leak_trigger[0] == 5
    # PB-C2-05: an undefined rate at a step start is an end of validity (trigger 18), never dormancy
    b, r_i, w = 0.02, 0.1, 2.0
    st2 = GrowthState.start(np.array([0.85 * b]), np.array([0.05]), np.array([b]), np.array([w]), 1)  # beyond the table
    g2 = Grower(
        law,
        Spectrum.repeating([(30e6, 0.0, 1.0)], 1e5),
        1,
        1.0,
        cylinder=CylinderSpec(r_i, "axial", "internal", "tabulated"),
    )
    g2.advance(st2, 1.0)
    assert st2.kind[0] == G.ENDED and st2.event_trigger[0, 0] == 18 and not st2.dormant[0]
    # the plate-directed family's start validity includes r_m/B (its through-wall destination is tabulated)
    thin = cf.Cylinder(r_i=3.0, thickness=0.02, length=4.0, k_route="tabulated")  # r_m/B = 150.5 > 100
    with pytest.raises(cf.OutOfValidityError):
        cf.Assessment(
            thin, cf.CylinderSurfaceFlaw(5e-3, 40e-3, "circumferential", "external"), _steel(), cf.StressState(pm=50e6)
        ).run()


# ---- Part B code review round 3 (partner PB-C-12, PB-C-16, PB-C-23, PB-C2-07) ------------------------------


def test_r3_convergence_error_is_proportional_to_tol() -> None:
    # PB-C-12 / B7: the controller error of the tabulated route falls in proportion to tol where the step is
    # tolerance-limited (tol <= 2e-4 here; above that the maximum step limits the error): the error ratio
    # between tol levels 4:1 apart is 4 within 15 %
    from critflaw.fatigue.growth import CylinderSpec

    law = GrowthLaw(float(coefficient_to_si(5.86e-13, 2.88)), 2.88, 0.0)
    sp = Spectrum.repeating([(30e6, 0.0, 1.0)], 1e5)
    b, r_i, w = 0.02, 0.1, 2.0

    def final_depth(tol: float) -> float:
        st = GrowthState.start(np.array([4e-3]), np.array([10e-3]), np.array([b]), np.array([w]), 1)
        Grower(law, sp, 1, 40.0, tol=tol, cylinder=CylinderSpec(r_i, "axial", "internal", "tabulated")).advance(
            st, 40.0
        )
        return float(st.a[0])

    fine = final_depth(1e-7)
    errors = {tol: final_depth(tol) - fine for tol in (2e-4, 1e-4, 5e-5, 2.5e-5)}
    assert all(abs(e) > 1e-9 for e in errors.values())  # the errors are resolved, not rounding
    for coarse, finer in ((2e-4, 5e-5), (1e-4, 2.5e-5)):
        ratio = errors[coarse] / errors[finer]
        assert 3.4 < ratio < 4.6, (coarse, finer, ratio)


def test_r3_sparse_column_m6_and_the_flagged_last_interval() -> None:
    # PB-C-16: Table M.6's r_m/B = 50 column is sparse by design (printed at six lambdas against eleven in
    # the dense columns), queried without a gap flag below its flagged last entry; only the interval between
    # the last two printed lambdas (10.096 .. 15.143, the flagged cell at its end) is suspect, and the column
    # ends at 15.143 (queries beyond are invalid, never extrapolated)
    from critflaw.solutions import cylinder as C

    for name in ("M.6c", "M.6d"):
        table = T.TABLES[name]
        lam = np.asarray(table.axes[0][1], dtype=float)
        cols = list(np.asarray(table.axes[1][1], dtype=float))
        values = np.asarray(table.values[0], dtype=float)
        counts = [int(np.isfinite(values[:, k]).sum()) for k in range(len(cols))]
        assert counts == [11, 11, 11, 6, 5]
        j = cols.index(50.0)
        printed = lam[np.isfinite(values[:, j])]
        assert printed[-2:].tolist() == [10.096, 15.143]
        for q, suspect in ((8.0, False), (10.0, False), (10.096, False), (12.0, True), (14.0, True), (15.143, True)):
            v = C.interpolate_columns(table, 0, q, 50.0)
            assert bool(v.valid) and bool(v.suspect) == suspect, (name, q)
        assert not bool(C.interpolate_columns(table, 0, 15.2, 50.0).valid)
        # the dense neighbour column (r_m/B = 20) is unaffected by the flag at 12 (ends at 12.77)
        assert not bool(C.interpolate_columns(table, 0, 12.0, 20.0).suspect)
        assert float(C.lambda_range_end(table, 50.0)[()]) == 15.143
        assert float(C.lambda_range_end(table, 20.0)[()]) == pytest.approx(12.77)


def test_r3_half_angle_reading_is_the_conservative_one_of_each_pair() -> None:
    # PB-C-23 / G7: over a declared grid (P.11 prints no range) the readings differ by far more than the
    # Phase 1 tolerance, so the conservative reading of each pair is implemented: c / r_i for internal and
    # c / r_m for external flaws. The sampled maxima are the recorded ones (docs/unsourced_items.md).
    from critflaw.solutions import cylinder_dispatch as D

    r_i = 0.1
    worst_thin = {False: 0.0, True: 0.0}
    worst_all = {False: 0.0, True: 0.0}
    n_cases = 0
    for b_ri in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
        b = b_ri * r_i
        r_m, r_o = r_i + 0.5 * b, r_i + b
        for ab in (0.05, 0.2, 0.4, 0.6, 0.8):
            for pm, mom in ((50e6, 0.0), (50e6, 12.5e6), (50e6, 50e6), (50e6, 200e6), (0.0, 50e6)):
                for frac in (0.02, 0.1, 0.3, 0.5, 0.7, 0.9):
                    c = frac * math.pi * r_i
                    for external in (False, True):
                        lr = {}
                        for r in (r_i, r_m, r_o):
                            lr[r] = float(
                                D._net_section(
                                    np.array([ab * b]),
                                    c / r,
                                    np.array([r_m]),
                                    np.array([b]),
                                    np.array([pm]),
                                    np.array([mom]),
                                    external,
                                )[0]
                            )
                        assert all(np.isfinite(v) and v > 0.0 for v in lr.values())
                        n_cases += 1
                        # the implemented reading is the conservative member of its pair (the larger Lr), up to
                        # 2e-4 relative where the source's branch (ii) (closure ignored) makes L_N rise with theta
                        impl, alt = (r_i, r_m) if not external else (r_m, r_o)
                        assert lr[alt] <= lr[impl] * (1.0 + 2e-4), (b_ri, ab, pm, mom, frac, external)
                        diff = abs(lr[alt] - lr[impl]) / lr[impl]
                        worst_all[external] = max(worst_all[external], diff)
                        if b_ri <= 0.2:
                            worst_thin[external] = max(worst_thin[external], diff)
    assert n_cases == 1800
    # the recorded sampled maxima (the earlier "< 5 %" claim was wrong: it sampled one depth and one load ratio)
    assert 0.09 < worst_thin[False] < 0.12 and 0.09 < worst_thin[True] < 0.12
    assert 0.25 < worst_all[False] < 0.40 and 0.30 < worst_all[True] < 0.45
    # PB-C3-04: the force/moment conversion of _net_section from the limit-load definitions, by hand: with
    # F = 2 pi r_m B Pm and M = pi r_m^2 B Pm,b the load ratio is M / (r_m F) = Pm,b / (2 Pm); Lr sigma_Y = Pm / L_N
    # from F / F_limit and = pi Pm,b / (4 L_b) from M / M_limit (the same number at that ratio)
    from critflaw.solutions import cylinder_sref as S

    a_, th, r_m_, b_ = np.array([6e-3]), 0.4, np.array([0.11]), np.array([0.02])
    pm_, mom_ = 40e6, 30e6
    l_n, l_b, _, _ = S.circumferential_thick_limit(a_, th, r_m_, b_, mom_ / (2.0 * pm_), external=False)
    got = float(D._net_section(a_, th, r_m_, b_, np.array([pm_]), np.array([mom_]), False)[0])
    assert got == pytest.approx(pm_ / float(l_n[0]), rel=1e-12)
    assert got == pytest.approx(math.pi * mom_ / (4.0 * float(l_b[0])), rel=1e-9)
    l_n0, _, _, _ = S.circumferential_thick_limit(a_, th, r_m_, b_, 0.0, external=False)
    assert float(D._net_section(a_, th, r_m_, b_, np.array([pm_]), np.array([0.0]), False)[0]) == pytest.approx(
        pm_ / float(l_n0[0]), rel=1e-12
    )
    _, l_b_inf, _, _ = S.circumferential_thick_limit(a_, th, r_m_, b_, np.inf, external=False)
    assert float(D._net_section(a_, th, r_m_, b_, np.array([0.0]), np.array([mom_]), False)[0]) == pytest.approx(
        math.pi * mom_ / (4.0 * float(l_b_inf[0])), rel=1e-12
    )
    # the dispatch uses these readings
    cyl = cf.Cylinder(r_i=r_i, thickness=0.02, length=4.0, k_route="tabulated", sref_route="thick")
    for position, r in (("internal", r_i), ("external", r_i + 0.01)):
        f = D.cylinder_factors(
            cyl,
            cf.CylinderSurfaceFlaw(8e-3, 60e-3, "circumferential", position),
            cf.StressState(pm=50e6, pm_moment=20e6),
            np.asarray(50e6),
            np.asarray(0.0),
        )
        expected = float(
            D._net_section(
                np.array([8e-3]),
                0.03 / r,
                np.array([r_i + 0.01]),
                np.array([0.02]),
                np.array([50e6]),
                np.array([20e6]),
                position == "external",
            )[0]
        )
        assert float(f.sref[()]) == pytest.approx(expected, rel=1e-12)


def test_r3_aspect_ends_under_bending_and_the_extended_rate_is_not_clamped() -> None:
    # PB-C2-06 / B20 / R4-01: at both a/c table ends, under membrane, pure-bending and mixed spectra, the engine's
    # outcome follows the direction of r = a/c from the velocities, dr/dt = (v_a - r v_c) / c (not the K ratio:
    # v scales with dK^m): away -> no event; towards -> trigger 9 landing. In the tables r moves away from both
    # ends in every case (the ends are reachable only by a start on or outside them), so no landing from inside
    # exists; the direction is asserted case by case
    from critflaw.fatigue.growth import CylinderSpec
    from critflaw.solutions import cylinder as C

    law = GrowthLaw(float(coefficient_to_si(5.86e-13, 2.88)), 2.88, 0.0)
    b, r_i, w = 0.02, 0.1, 8.0  # the length keeps the longest start (c = 20 a at a = 0.75 B) inside 2c/W = 0.15
    idx = np.array([0])
    seen_landing = 0
    for blocks in ([(30e6, 0.0, 1.0)], [(0.0, 60e6, 1.0)], [(20e6, 60e6, 1.0)]):
        sp = Spectrum.repeating(blocks, 1e5)
        for orientation, ac_min in (("axial", 0.05), ("circumferential", 0.1)):
            spec = CylinderSpec(r_i, orientation, "internal", "tabulated")
            for ab in (0.05, 0.25, 0.5, 0.75):
                a0 = ab * b
                for end, c0 in ((ac_min, a0 / (ac_min * (1 + 1e-6))), (1.0, a0 * (1 + 1e-6))):
                    st = GrowthState.start(np.array([a0]), np.array([c0]), np.array([b]), np.array([w]), 1)
                    g = Grower(law, sp, 1, 0.02, cylinder=spec)
                    g.bind(st)
                    dk = g.ranges(idx, st.a, st.c, st.kind, st.thickness, st.width)
                    above, stage_b, _ = g.active_set(idx, dk, st.a, st.c, st.kind)
                    va, vc, _ = g.velocities(
                        idx, st.a, st.c, st.kind, np.zeros(1), st.thickness, st.width, above, stage_b
                    )
                    r = a0 / c0
                    drdt = (float(va[0]) - r * float(vc[0])) / c0
                    assert np.isfinite(drdt) and drdt != 0.0
                    towards = (drdt > 0.0) == (end == 1.0)
                    g.advance(st, 0.02)  # a week: no other boundary is reached from these starts
                    if towards:
                        seen_landing += 1
                        assert st.event_count[0] >= 1 and st.event_trigger[0, 0] == 9, (blocks, orientation, ab, end)
                    else:
                        assert st.event_count[0] == 0 and st.kind[0] == G.SURFACE, (blocks, orientation, ab, end)
                        assert (float(st.a[0] / st.c[0]) > end) == (end != 1.0)
    assert seen_landing == 0  # no a/c end is approached from inside in any of the 48 cases
    # the extended branch evaluates no rate beyond its depth end: NaN (undefined), never a clamped value
    for spec_, a_over_b in (
        (CylinderSpec(r_i, "axial", "internal", "tabulated"), C.TABLE_A_OVER_B_MAX),
        (CylinderSpec(r_i, "axial", "internal", "pressure_only"), C.EXTENDED_PLATE_A_OVER_B_MAX),
    ):
        st = GrowthState.start(np.array([a_over_b * b * (1 + 1e-6)]), np.array([w]), np.array([b]), np.array([w]), 1)
        st.kind[:] = G.EXTENDED
        g = Grower(law, Spectrum.repeating([(30e6, 0.0, 1.0)], 1e5), 1, 1.0, cylinder=spec_)
        g.bind(st)
        dk = g.ranges(np.array([0]), st.a, st.c, st.kind, st.thickness, st.width)
        assert np.isnan(dk[0, 0, 0])
        st.a[:] = a_over_b * b  # on the edge (within rounding) the rate is defined
        assert np.isfinite(g.ranges(np.array([0]), st.a, st.c, st.kind, st.thickness, st.width)[0, 0, 0])


def test_r3_invalid_trial_is_halved_not_refused() -> None:
    # PB-C3-02: a trial whose corrector is not finite halves the step (a defensive guard behind the projection; it
    # would otherwise be unreachable), it does not drive the step to the minimum and raise
    law = GrowthLaw(float(coefficient_to_si(5.86e-13, 2.88)), 2.88, 0.0)
    sp = Spectrum.repeating([(100e6, 0.0, 1.0)], 1e5)
    calls: list[float] = []

    class Spoiled(Grower):
        def heun(self, st: Any, idx: Any, h: Any, above: Any, stage_b: Any, v0: Any) -> Any:
            calls.append(float(h[0]))
            a1, c1, a2, c2 = super().heun(st, idx, h, above, stage_b, v0)
            if h[0] > 0.05:
                a2 = np.full_like(a2, np.nan)
            return a1, c1, a2, c2

    st = GrowthState.start(np.array([2e-3]), np.array([10e-3]), np.array([B]), np.array([W]), 1)
    Spoiled(law, sp, 1, 1.0).advance(st, 1.0)
    assert st.kind[0] == G.SURFACE and np.isfinite(st.a[0]) and st.a[0] > 2e-3
    assert calls[0] > 0.05 and calls[1] == pytest.approx(0.5 * calls[0])  # halved, not shortened to the minimum


def test_r4_plate_directed_through_flaw_is_projected_at_the_lambda_end() -> None:
    # PB-C4-01: the circumferential external family's through-wall flaw is tabulated, so the predictor is projected
    # at the lambda end like the tabulated families' (no NaN corrector on the way); it lands on the end (trigger 10)
    from critflaw.fatigue.growth import CylinderSpec
    from critflaw.solutions import cylinder as C

    law = GrowthLaw(float(coefficient_to_si(5.86e-13, 2.88)) * 30, 2.88, 0.0)
    sp = Spectrum.repeating([(30e6, 0.0, 1.0)], 1e5)
    b, r_i, w = 0.02, 0.1, 8.0
    lam_end = float(C.lambda_range_end(T.TABLES["M.6a"], (r_i + 0.5 * b) / b)[()])
    c_end = lam_end * math.sqrt((r_i + 0.5 * b) * b) / (12 * (1 - 0.09)) ** 0.25
    nonfinite: list[float] = []

    class Counting(Grower):
        def heun(self, st: Any, idx: Any, h: Any, above: Any, stage_b: Any, v0: Any) -> Any:
            a1, c1, a2, c2 = super().heun(st, idx, h, above, stage_b, v0)
            if not (np.all(np.isfinite(a2)) and np.all(np.isfinite(c2))):
                nonfinite.append(float(h[0]))
            return a1, c1, a2, c2

    st = GrowthState.start(np.array([b]), np.array([0.6 * c_end]), np.array([b]), np.array([w]), 1)
    st.kind[:] = G.THROUGH
    st.t_ended[:] = np.nan
    Counting(law, sp, 1, 50.0, cylinder=CylinderSpec(r_i, "circumferential", "external", "tabulated")).advance(st, 50.0)
    assert st.kind[0] == G.ENDED and st.event_trigger[0, 0] == 10
    assert float(st.event_pre[0, 0, 1]) == pytest.approx(c_end, rel=1e-9)
    assert nonfinite == []


def test_b15_toughness_length_per_part_b_kind() -> None:
    # B15 (owed, now pinned): the Master Curve toughness of each Part B kind is computed from the crack-front
    # length of spec S2.11 (2c surface, 4c embedded, 2B through-thickness, extended = 2B; all capped at 2B).
    # With no secondary stress Kr = Kp / Kmat, so Kmat is read back as Kp / Kr at the governing position.
    from critflaw import toughness as TM

    mc = cf.Material(
        355e6, 490e6, 207e9, cf.MasterCurve(t0=-60.0, temperature=-20.0, probability=0.4), yielding="continuous"
    )
    b_c, r_i = 0.02, 0.1
    cyl = cf.Cylinder(r_i=r_i, thickness=b_c, length=4.0, k_route="tabulated")
    cases = [
        (cf.FlatPlate(B, W), cf.EmbeddedFlaw(4e-3, 20e-3, 5e-3), cf.StressState(pm=150e6), 4.0 * 10e-3, B),
        (cf.FlatPlate(B, W), cf.EmbeddedFlaw(4e-3, 80e-3, 5e-3), cf.StressState(pm=150e6), 2.0 * B, B),  # 4c > 2B: cap
        (cyl, cf.CylinderSurfaceFlaw(5e-3, 24e-3, "axial", "internal"), cf.StressState(pm=150e6), 24e-3, b_c),
        (
            cyl,
            cf.CylinderSurfaceFlaw(5e-3, 60e-3, "axial", "internal"),
            cf.StressState(pm=150e6),
            2.0 * b_c,
            b_c,
        ),  # cap
        (cyl, cf.CylinderThroughFlaw(0.05, "circumferential"), cf.StressState(pm=150e6), 2.0 * b_c, b_c),
        (cyl, cf.ExtendedFlaw(5e-3, "axial", "internal"), cf.StressState(pm=150e6), 2.0 * b_c, b_c),
    ]
    for component, flaw, loads, front, thickness in cases:
        res = cf.Assessment(component, flaw, mc, loads).run()
        assert bool(res.valid), flaw
        gov = int(res.governing_position)
        k_mat = float(res.k_primary[gov]) / float(res.kr)
        expected = float(TM.master_curve(0.4, -20.0, -60.0, front, thickness)[()])
        assert k_mat == pytest.approx(expected, rel=1e-9), flaw
        # the pinned length is not the other candidates' (the cap and the plain length differ unless capped)
        if front < 2.0 * thickness:
            assert k_mat != pytest.approx(
                float(TM.master_curve(0.4, -20.0, -60.0, 2.0 * thickness, thickness)[()]), rel=1e-3
            )


ORACLE_X = _load("partner_reference_partB_extended", HERE / "oracles" / "partner_reference_partB_extended.py")


def test_partner_fixed_step_integrator_for_the_tabulated_extended_flaw_agrees_to_1e_8() -> None:
    # B5 (owed item closed): the partner's extended-flaw Heun integrator (oracle 4, received 2026-10-07) against the
    # production step with the active set frozen. The partner's cases 1-3 leave the grid at steps 146, 148 and 85
    # (its records say so); the comparison therefore stops short of the edge with the step counts below, which is a
    # test decision, not a change to the oracle.
    from critflaw.fatigue.growth import CylinderSpec

    cases = [
        ((8.2e-3, 20e-3, 0.1, 30e6, 0.0, 1.4097e-28, 2.88, 1e5, 1e-2), 140),
        ((8.2e-3, 20e-3, 0.1, 20e6, 20e6, 1.4097e-28, 2.88, 1e5, 1e-2), 140),
        ((9e-3, 20e-3, 0.15, 25e6, 10e6, 1.4097e-28, 2.88, 2e5, 5e-3), 80),
        ((12.5e-3, 25e-3, 0.125, 0.0, 40e6, 1.64754666094773e-29, 3.0, 5e4, 2e-2), 100),
    ]
    for (a0, b, r_i, dsm, dsb, a_si, m, cycles, h), n in cases:
        expected = float(ORACLE_X.integrate_extended_heun(a0, b, r_i, dsm, dsb, a_si, m, cycles, h, n))
        law = GrowthLaw(a_si, m, 0.0)
        sp = Spectrum.repeating([(dsm, dsb, 1.0)], cycles)
        spec = CylinderSpec(r_i, "axial", "internal", "tabulated")
        a, _ = _fixed_heun_kind(G.EXTENDED, a0, 1.0, b, 10.0, law, sp, h, n, spec=spec)
        assert a == pytest.approx(expected, rel=1e-8), (a0, dsm, dsb)
        assert a > a0 * 1.05  # the comparison covers real growth, not the start
    # the partner's own record of case 4 (the only one it could complete on its grid)
    rec = [c for c in ORACLE_X.tabulate() if c["name"] == "extended_4"][0]
    assert rec["inputs"]["args"][-1] == 100
