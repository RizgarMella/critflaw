"""Compare critflaw with the partner model's independent reference implementation (spec S1.5 kind d).

``partner_reference_phase1.py`` and ``partner_cases_phase1.json`` were written by a
different model from the transcribed equations, without sight of critflaw. They are
kept exactly as received.

The request sent to the partner carried one OCR error from the markdown conversion of
the source: M3 = 0.5 - 1/(0.65 + (a/c)^2) + ... where the printed page has
1/(0.65 + (a/c)). The partner implemented it as sent. ``test_ocr_error_is_detected``
shows that the unmodified reference disagrees with critflaw for exactly that reason;
all other comparisons use the reference with that single term corrected.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import critflaw as cf
from critflaw.fad import (
    Option1Curve,
    fracture_ratio,
    kj_secondary,
    lr_cutoff,
    radial_margin,
    rho_simplified,
    v_simplified,
)
from critflaw.solutions import plate

HERE = Path(__file__).parent
SOURCE = (HERE / "partner_reference_phase1.py").read_text(encoding="utf-8")
OCR_TERM = "(0.65 + r ** 2)"
PRINTED_TERM = "(0.65 + r)"

REL_CLOSED = 1e-10
REL_ITERATIVE = 1e-9
N = 3000


def _load(source: str, name: str) -> ModuleType:
    module = ModuleType(name)
    exec(compile(source, name, "exec"), module.__dict__)  # reviewed: math/numpy only, no I/O
    return module


assert SOURCE.count(OCR_TERM) == 1
REF_AS_RECEIVED = _load(SOURCE, "partner_reference_as_received")
REF = _load(SOURCE.replace(OCR_TERM, PRINTED_TERM), "partner_reference_corrected")


def _curve(sy: float, su: float, e: float, discontinuous: bool, luders: float | None = None) -> Option1Curve:
    return Option1Curve(sy, su, e, yielding="discontinuous" if discontinuous else "continuous", luders_strain=luders)


def _close(mine: Any, theirs: Any, rel: float) -> bool:
    if isinstance(theirs, float) and math.isinf(theirs):
        return bool(np.isinf(mine))
    return bool(abs(float(mine) - float(theirs)) <= 1e-12 + rel * abs(float(theirs)))


@pytest.fixture()
def rng(request: pytest.FixtureRequest) -> np.random.Generator:
    # One independent stream per test, so that a change in one test cannot shift another's inputs.
    seed = int.from_bytes(request.node.name.encode(), "little") % (2**32)
    return np.random.default_rng([20261004, seed])


def _geometry(rng: np.random.Generator) -> tuple[float, float, float]:
    """Random valid surface flaw (a, c, B) covering both aspect-ratio branches."""
    b = rng.uniform(5e-3, 80e-3)
    a_over_c = rng.uniform(0.05, 2.0)
    a = rng.uniform(0.02, 0.8) * b
    return a, a / a_over_c, b


def test_ocr_error_is_detected(rng: np.random.Generator) -> None:
    """The reference built from the uncorrected transcription must disagree for a/c <= 1 and only there."""
    worst_shallow, worst_deep = 0.0, 0.0
    for _ in range(500):
        a, c, b = _geometry(rng)
        mine = float(plate.surface_flaw_mm_mb(a, c, b, plate.THETA_DEEPEST)[0])
        theirs = REF_AS_RECEIVED.surface_mm_mb(a, c, b, plate.THETA_DEEPEST)[0]
        diff = abs(mine - theirs) / abs(theirs)
        if a / c <= 1.0:
            worst_shallow = max(worst_shallow, diff)
        else:
            worst_deep = max(worst_deep, diff)
    assert worst_shallow > 1e-3
    assert worst_deep < REL_CLOSED


def test_curve(rng: np.random.Generator) -> None:
    for i in range(N):
        sy = rng.uniform(200e6, 900e6)
        su = sy * rng.uniform(1.02, 2.0)
        e = rng.uniform(60e9, 220e9)
        disc = bool(i % 2)
        lr = [0.0, 1.0, float(np.nextafter(1.0, 0)), float(np.nextafter(1.0, 2))][i % 4] if i % 5 == 0 else rng.uniform(0, 2.2)
        assert _close(lr_cutoff(sy, su), REF.lr_max(sy, su), REL_CLOSED)
        assert _close(_curve(sy, su, e, disc)(lr), REF.option1_curve(lr, sy, su, e, disc), REL_CLOSED), (lr, sy, su, e, disc)
    assert _close(
        _curve(355e6, 490e6, 207e9, True, 0.01)(1.1), REF.option1_curve(1.1, 355e6, 490e6, 207e9, True, 0.01), REL_CLOSED
    )


def test_interaction(rng: np.random.Generator) -> None:
    for _ in range(N):
        chi = rng.uniform(0, 9.0)
        lr = rng.uniform(0, 1.4)
        assert _close(rho_simplified(chi, lr), REF.rho(chi, lr), REL_CLOSED), (chi, lr)
        assert _close(v_simplified(chi, lr), REF.v_factor(chi, lr), REL_CLOSED), (chi, lr)
        ks, a, sy = rng.uniform(1e6, 200e6), rng.uniform(1e-3, 50e-3), rng.uniform(200e6, 900e6)
        for plane_stress, beta in ((True, 1.0), (False, 3.0)):
            assert _close(kj_secondary(ks, a, sy, plane_stress=plane_stress), REF.kj_secondary(ks, a, sy, beta), REL_CLOSED)


def test_fracture_ratio(rng: np.random.Generator) -> None:
    for i in range(N):
        kp = 0.0 if i % 7 == 0 else rng.uniform(1e6, 150e6)
        ks = [0.0, -rng.uniform(1e6, 50e6)][i % 2] if i % 5 == 0 else rng.uniform(1e6, 200e6)
        lr = 0.0 if kp == 0.0 else rng.uniform(0.01, 1.4)
        kmat, a, sy = rng.uniform(30e6, 300e6), rng.uniform(1e-3, 50e-3), rng.uniform(200e6, 900e6)
        for method in ("rho", "V"):
            for plane_stress, beta in ((True, 1.0), (False, 3.0)):
                mine = fracture_ratio(kp, ks, lr, kmat, a, sy, method=method, plane_stress=plane_stress).kr  # type: ignore[arg-type]
                assert _close(mine, REF.kr(kp, ks, lr, kmat, a, sy, method, beta), REL_CLOSED), (kp, ks, lr, method)


def test_surface_flaw_solutions(rng: np.random.Generator) -> None:
    for i in range(N):
        a, c, b = _geometry(rng)
        theta = [0.0, math.pi / 2][i % 2] if i % 3 == 0 else rng.uniform(0, math.pi / 2)
        mm, mb = plate.surface_flaw_mm_mb(a, c, b, theta)
        ref_mm, ref_mb = REF.surface_mm_mb(a, c, b, theta)
        assert _close(mm, ref_mm, REL_CLOSED) and _close(mb, ref_mb, REL_CLOSED), (a, c, b, theta)
        w = 2 * c / rng.uniform(0.01, 0.8)
        assert _close(plate.surface_flaw_fw(a, c, b, w), REF.surface_fw(a, c, b, w), REL_CLOSED)
        pm, pb = rng.uniform(0, 400e6), rng.uniform(0, 200e6)
        for pin in (False, True):
            assert _close(
                plate.surface_flaw_reference_stress(pm, pb, a, c, b, w, pin_jointed=pin),
                REF.surface_sref(pm, pb, a, c, b, w, pin),
                REL_CLOSED,
            ), (pm, pb, a, c, b, w, pin)


def test_through_flaw_and_general_form(rng: np.random.Generator) -> None:
    for _ in range(N):
        w = rng.uniform(0.05, 2.0)
        a = rng.uniform(0.001, 0.45) * w
        pm, pb = rng.uniform(0, 400e6), rng.uniform(0, 200e6)
        assert _close(plate.through_flaw_fw(a, w), REF.through_fw(a, w), REL_CLOSED)
        assert _close(plate.through_flaw_reference_stress(pm, pb, a, w), REF.through_sref(pm, pb, a, w), REL_CLOSED)
        mm, mb, fw = rng.uniform(0.3, 2), rng.uniform(0.1, 2), rng.uniform(1, 2)
        qm, qb = rng.uniform(-100e6, 400e6), rng.uniform(-100e6, 100e6)
        kp, ks = plate.stress_intensity(a, mm, mb, fw, pm, pb, qm, qb)
        ref_kp, ref_ks = REF.k_primary_secondary(a, mm, mb, fw, pm, pb, qm, qb)
        assert _close(kp, ref_kp, REL_CLOSED) and _close(ks, ref_ks, REL_CLOSED)


def test_radial_margin(rng: np.random.Generator) -> None:
    for i in range(N):
        sy = rng.uniform(200e6, 900e6)
        su = sy * rng.uniform(1.02, 2.0)
        e = rng.uniform(60e9, 220e9)
        disc = bool(i % 2)
        lr = 0.0 if i % 11 == 0 else rng.uniform(0.0, 2.0)
        kr = 0.0 if i % 13 == 0 else rng.uniform(0.0, 2.0)
        mine, _ = radial_margin(lr, kr, _curve(sy, su, e, disc))
        assert _close(mine, REF.radial_margin(lr, kr, sy, su, e, disc), REL_ITERATIVE), (lr, kr, sy, su, e, disc)


def test_full_surface_flaw_assessment(rng: np.random.Generator) -> None:
    compared = negative_kp = 0
    for i in range(1500):
        a, c, b = _geometry(rng)
        w = 2 * c / rng.uniform(0.01, 0.8)
        sy = rng.uniform(250e6, 700e6)
        su = sy * rng.uniform(1.05, 1.8)
        kmat = rng.uniform(40e6, 300e6)
        pm = 0.0 if i % 9 == 0 else rng.uniform(1e6, 0.9 * sy)
        pb = 0.0 if i % 9 == 0 else rng.uniform(0, 0.3 * sy)
        qm = 0.0 if i % 4 == 0 else rng.uniform(0, sy)
        qb = rng.uniform(0, 0.2 * sy) if i % 3 == 0 else 0.0
        method = ("rho", "V")[i % 2]
        pin = bool(i % 5 == 0)
        problem = cf.Assessment(
            cf.FlatPlate(b, w),
            cf.SurfaceFlaw(a, 2 * c),
            cf.Material(sy, su, 207e9, kmat, yielding="continuous"),
            cf.StressState(pm, pb, qm, qb),
            bending_restraint="pin_jointed" if pin else "normal",
            interaction=method,  # type: ignore[arg-type]
        )
        if not plate.surface_flaw_valid(a, c, b, w):
            continue
        result = problem.run()
        if "negative_primary_k_set_to_zero" in result.trace.flags:
            # The source has no rule for a negative primary K and the partner reference rejects it;
            # critflaw's rule (spec S2.5) is tested separately in tests/test_plate_and_assessment.py.
            negative_kp += 1
            with pytest.raises(ValueError):
                REF.assess_surface(b, w, a, 2 * c, sy, su, 207e9, kmat, pm, pb, qm, qb, method, pin)
            continue
        compared += 1
        ref = REF.assess_surface(b, w, a, 2 * c, sy, su, 207e9, kmat, pm, pb, qm, qb, method, pin)
        assert _close(result.lr, ref["lr"], REL_CLOSED)
        assert _close(result.kr_by_position[0], ref["kr_deepest"], REL_CLOSED)
        assert _close(result.kr_by_position[1], ref["kr_surface"], REL_CLOSED)
        assert _close(result.radial_margin, ref["F"], REL_ITERATIVE), (a, c, b, w, sy, su, kmat, pm, pb, qm, qb)
    assert compared >= 1000, (compared, negative_kp)


def test_partner_hand_cases_agree_with_partner_reference() -> None:
    """The partner's hand-derived expected values must match its own module as received."""
    cases = json.loads((HERE / "partner_cases_phase1.json").read_text(encoding="utf-8-sig"))
    assert len(cases) >= 40
    failures = []
    for case in cases:
        got = getattr(REF_AS_RECEIVED, case["function"])(**case["inputs"])
        expected = case["expected"]
        if isinstance(expected, dict):
            pairs = [(got[k], v) for k, v in expected.items()]
        elif isinstance(expected, list):
            pairs = list(zip(got, expected, strict=True))
        else:
            pairs = [(got, expected)]
        if not all(_close(g, float(x), 1e-7) for g, x in pairs):
            failures.append(case["name"])
    assert not failures, failures


def _critflaw_call(function: str, kw: dict[str, Any]) -> Any:
    """Evaluate one partner hand case with critflaw; raises if critflaw rejects the inputs."""
    k = dict(kw)
    if function == "lr_max":
        return float(lr_cutoff(k["sy"], k["su"]))
    if function == "option1_curve":
        return float(_curve(k["sy"], k["su"], k["E"], k.get("discontinuous", False), k.get("luders_strain"))(k["lr"]))
    if function == "rho":
        return float(rho_simplified(k["chi"], k["lr"]))
    if function == "v_factor":
        return float(v_simplified(k["chi"], k["lr"]))
    if function == "kj_secondary":
        return float(kj_secondary(k["ks"], k["a"], k["sy"], plane_stress=k["beta"] == 1.0))
    if function == "kr":
        if k["kp"] == 0.0 and k["lr"] > 0.0 and k["ks"] > 0.0:
            raise ValueError("partner treats Kp = 0 with Lr > 0 as secondary-only; critflaw does not (spec S2.5)")
        return float(
            fracture_ratio(k["kp"], k["ks"], k["lr"], k["kmat"], k["a"], k["sy"], method=k.get("method", "rho"),
                           plane_stress=k.get("beta", 1.0) == 1.0).kr
        )
    if function == "surface_mm_mb":
        if k["a"] / k["c"] <= 1.0 and k["a"] != k["c"]:
            raise ValueError("hand value computed with the OCR form of M3")
        return [float(v) for v in plate.surface_flaw_mm_mb(k["a"], k["c"], k["B"], k["theta"])]
    if function == "surface_fw":
        return float(plate.surface_flaw_fw(k["a"], k["c"], k["B"], k["W"]))
    if function == "surface_sref":
        return float(plate.surface_flaw_reference_stress(k["pm"], k["pb"], k["a"], k["c"], k["B"], k["W"],
                                                         pin_jointed=k.get("pin_jointed", False)))
    if function == "through_fw":
        return float(plate.through_flaw_fw(k["a"], k["W"]))
    if function == "through_sref":
        return float(plate.through_flaw_reference_stress(k["pm"], k["pb"], k["a"], k["W"]))
    if function == "k_primary_secondary":
        return [float(v) for v in plate.stress_intensity(k["a"], k["mm"], k["mb"], k["fw"], k["pm"], k["pb"], k["qm"], k["qb"])]
    if function == "radial_margin":
        return float(radial_margin(k["lr"], k["kr_value"], _curve(k["sy"], k["su"], k["E"], k.get("discontinuous", False)))[0])
    raise ValueError(f"no adapter for {function}")


def test_partner_hand_cases_agree_with_critflaw() -> None:
    """Every partner hand case that critflaw accepts must agree; the rest are listed, not hidden."""
    cases = json.loads((HERE / "partner_cases_phase1.json").read_text(encoding="utf-8-sig"))
    agreed, skipped, failures = 0, [], []
    for case in cases:
        try:
            got = _critflaw_call(case["function"], case["inputs"])
        except (ValueError, TypeError, KeyError) as exc:
            skipped.append((case["name"], str(exc)[:60]))
            continue
        expected = case["expected"]
        pairs = list(zip(got, expected, strict=True)) if isinstance(expected, list) else [(got, expected)]
        if all(_close(g, float(x), 1e-7) for g, x in pairs):
            agreed += 1
        else:
            failures.append((case["name"], got, expected))
    assert not failures, failures
    assert agreed >= 50, (agreed, skipped)

# ---- addendum: residual stress rules and the zero-primary-K rule (round 2) ---------------

ADDENDUM = _load((HERE / "partner_reference_phase1_addendum.py").read_text(encoding="utf-8"), "partner_addendum")


def test_addendum_fracture_ratio_rule(rng: np.random.Generator) -> None:
    for i in range(N):
        kp = [0.0, -rng.uniform(1e6, 50e6)][i % 2] if i % 4 == 0 else rng.uniform(1e6, 150e6)
        ks = [0.0, -rng.uniform(1e6, 50e6)][i % 2] if i % 5 == 0 else rng.uniform(1e6, 200e6)
        lr = 0.0 if (kp == 0.0 and i % 8 == 0) else rng.uniform(0.01, 1.4)
        if kp > 0.0 and lr == 0.0:
            continue
        kmat, a, sy = rng.uniform(30e6, 300e6), rng.uniform(1e-3, 50e-3), rng.uniform(200e6, 900e6)
        for method in ("rho", "V"):
            for plane_stress, beta in ((True, 1.0), (False, 3.0)):
                # the engine zeroes a negative primary K before calling fracture_ratio (spec S2.5)
                mine = fracture_ratio(max(kp, 0.0), ks, lr, kmat, a, sy, method=method, plane_stress=plane_stress).kr  # type: ignore[arg-type]
                assert _close(mine, ADDENDUM.kr_rule(kp, ks, lr, kmat, a, sy, method, beta), REL_CLOSED), (kp, ks, lr, method)


def test_addendum_residual_stress_through_the_engine(rng: np.random.Generator) -> None:
    plate_ = cf.FlatPlate(25e-3, 0.5)
    flaw = cf.SurfaceFlaw(3e-3, 15e-3)
    for i in range(600):
        sy = rng.uniform(250e6, 700e6)
        su = sy * rng.uniform(1.05, 1.8)
        sy_app = rng.uniform(250e6, 700e6)
        su_app = sy_app * rng.uniform(1.05, 1.8)
        pm = rng.uniform(1e6, 1.3 * sy)
        relaxation = bool(i % 2)
        room = sy_app * rng.uniform(1.0, 1.2) if i % 3 == 0 else None
        if i % 5 == 0:
            res = cf.Residual.pwht(("parallel", "transverse")[i % 2], ("580-620C", "550-580C")[(i // 2) % 2], sy_app)  # type: ignore[arg-type]
            relaxation, room, su_arg = False, None, su_app
        else:
            res = cf.Residual.as_welded(sy_app, su_app if relaxation else None, relaxation, room)
            su_arg = su_app
        material = cf.Material(sy, su, 207e9, 150e6, yielding="continuous")
        r = cf.Assessment(plate_, flaw, material, cf.StressState(pm=pm, residual=res)).run()
        expected = ADDENDUM.residual_qm(sy_app, su_arg, float(r.reference_stress), res.fraction, relaxation, room)
        assert _close(r.qm, expected, REL_CLOSED), (i, sy_app, su_app, pm, relaxation, room)


def test_addendum_hand_cases() -> None:
    cases = json.loads((HERE / "partner_cases_phase1_addendum.json").read_text(encoding="utf-8-sig"))
    checked = 0
    for case in cases:
        k, expected = case["inputs"], case["expected"]
        if "raises" in expected:
            with pytest.raises(ValueError):
                getattr(ADDENDUM, case["function"])(**k)
            continue
        expected = expected["value"]
        assert _close(getattr(ADDENDUM, case["function"])(**k), float(expected), 1e-7), case["name"]
        if case["function"] != "kr_rule":
            continue
        if not (k["kp"] > 0.0 and k["lr"] == 0.0) and k.get("beta", 1.0) in (1.0, 3.0):
            mine = fracture_ratio(max(k["kp"], 0.0), k["ks"], k["lr"], k["kmat"], k["a"], k["sy"],
                                  method=k.get("method", "rho"), plane_stress=k.get("beta", 1.0) == 1.0).kr
            assert _close(mine, float(expected), 1e-7), case["name"]
            checked += 1
    assert checked >= 15, checked
