"""Compare critflaw with the partner model's independent Phase 2 reference (spec S1.5 kind d).

``partner_reference_phase2.py`` and ``partner_cases_phase2.json`` are kept exactly as received.
The reference works in the units of the source equations (MPa m^0.5, mm); critflaw in SI.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from critflaw import prob
from critflaw import toughness as tg
from critflaw.fad.option2 import Option2Curve, PowerLaw

HERE = Path(__file__).parent
MPA = 1.0e6


def _load() -> ModuleType:
    module = ModuleType("partner_reference_phase2")
    source = (HERE / "partner_reference_phase2.py").read_text(encoding="utf-8")
    exec(compile(source, "partner_reference_phase2", "exec"), module.__dict__)  # reviewed: math only
    return module


REF = _load()
REL = 1e-10


def close(mine: float, theirs: float, rel: float = REL) -> bool:
    return abs(float(mine) - float(theirs)) <= 1e-12 + rel * abs(float(theirs))


@pytest.fixture()
def rng(request: pytest.FixtureRequest) -> np.random.Generator:
    return np.random.default_rng([20261004, int.from_bytes(request.node.name.encode(), "little") % (2**32)])


def test_option2_power_law_line(rng: np.random.Generator) -> None:
    for i in range(3000):
        sy = rng.uniform(200e6, 900e6)
        su = sy * rng.uniform(1.02, 1.8)
        e = rng.uniform(60e9, 220e9)
        offset = [0.0, 0.002, rng.uniform(0.0, 0.004)][i % 3]
        n = [1.0, rng.uniform(1.0, 40.0)][i % 2]
        luders = None if i % 4 else rng.uniform(0.001, 0.04)
        lr = [0.0, 1.0, float(np.nextafter(1.0, 0)), rng.uniform(0, 2.0)][i % 4] if i % 5 == 0 else rng.uniform(0, 2.0)
        mine = Option2Curve(PowerLaw(offset, n, luders), sy, su, e)(lr)
        assert close(mine, REF.option2_power_law(lr, sy, su, e, offset, n, luders)), (lr, sy, su, e, offset, n, luders)
        s = max(lr, 1e-9)
        law = PowerLaw(offset, n, luders)
        assert close(law.q(np.asarray(s)), REF.power_law_q(s, offset, n, luders))
        assert close(law.q_hat(np.asarray(s)), REF.power_law_q_hat(s, offset, n, luders))


def test_toughness_relations(rng: np.random.Generator) -> None:
    for _ in range(2000):
        sy, e = rng.uniform(200, 900), rng.uniform(60e3, 220e3)
        su = sy * rng.uniform(1.0, 3.5)
        j, ctod = rng.uniform(1, 500), rng.uniform(1e-5, 2e-3)
        assert close(tg.k_from_j(j * 1e3, e * MPA) / MPA, REF.k_from_j(j * 1e-3, e))  # J in kJ/m^2 -> MPa m
        assert close(tg.ctod_constraint_factor(sy * MPA, su * MPA), REF.ctod_m(sy, su))
        assert close(tg.k_from_ctod(ctod, sy * MPA, su * MPA, e * MPA) / MPA, REF.k_from_ctod(ctod, sy, su, e))
        kb, b, big = rng.uniform(25, 300), rng.uniform(0.005, 0.05), rng.uniform(0.05, 0.2)
        assert close(tg.subsize_correction(kb * MPA, b, big) / MPA, REF.subsize(kb, b, big))
        assert close(tg.k_jc_limit(b, sy * MPA, e * MPA) / MPA, REF.k_jc_limit(b, sy, e))
        p, t, t0 = rng.uniform(0.001, 0.999), rng.uniform(-100, 50), rng.uniform(-120, 20)
        l_mm, b_mm = rng.uniform(5, 300), rng.uniform(10, 100)
        assert close(tg.master_curve(p, t, t0, l_mm / 1e3, b_mm / 1e3) / MPA, REF.master_curve(p, t, t0, l_mm, b_mm))
        pf, tk = rng.uniform(0.01, 0.5), rng.uniform(0, 30)
        assert close(tg.charpy_master_curve(t, t0, b_mm / 1e3, pf, tk) / MPA, REF.charpy_master_curve(t, t0, b_mm, pf, tk))
        cv = rng.uniform(10, 300)
        assert close(tg.charpy_lower_bound(cv, b_mm / 1e3) / MPA, REF.charpy_lower_bound(cv, b_mm))
        assert close(tg.charpy_upper_limit(cv, 300e6) / MPA, REF.charpy_upper_limit(cv))
        assert close(tg.charpy_upper_shelf(cv, e * MPA) / MPA, REF.charpy_upper_shelf(cv, e))
        bss = rng.uniform(1.0, 10.0)
        assert close(tg.subsize_charpy_temperature_shift(bss / 1e3), REF.subsize_charpy_shift(bss), 1e-9)


def test_distribution_relations_and_interval_end_points(rng: np.random.Generator) -> None:
    for _ in range(500):
        mean, sd = rng.uniform(1, 1e3), rng.uniform(0.05, 0.9) * 100
        d = prob.LogNormal.from_mean_sd(mean, sd)
        mu, sigma = REF.lognormal_params(mean, sd)
        assert close(d.mu_ln + 0.5 * d.sigma_ln**2 - 0.5 * sigma**2, mu) and close(d.sigma_ln, sigma)
        theta, k = rng.uniform(1, 500), rng.uniform(0.7, 30)
        w = prob.Weibull(theta, k)
        ref_mean, ref_sd = REF.weibull_mean_sd(theta, k)
        assert close(w.mean(), ref_mean, 1e-9) and close(w.std(), ref_sd, 1e-8)
        n = int(rng.integers(1, 10_000_000))
        assert close(prob.clopper_pearson(0, n).upper, REF.clopper_pearson_zero_upper(n), 1e-9)
        assert close(prob.clopper_pearson(n, n).lower, REF.clopper_pearson_all_lower(n), 1e-9)


def test_partner_hand_cases_agree_with_its_reference_and_with_the_source_examples() -> None:
    cases = json.loads((HERE / "partner_cases_phase2.json").read_text(encoding="utf-8"))
    assert len(cases) >= 50
    failures = []
    for case in cases:
        got = getattr(REF, case["function"])(**case["inputs"])
        expected = case["expected"]["value"] if isinstance(case["expected"], dict) else case["expected"]
        pairs = list(zip(got, expected, strict=True)) if isinstance(expected, list) else [(got, expected)]
        for g, x in pairs:
            ok = (math.isinf(x) and math.isinf(g)) if isinstance(x, float) and math.isinf(x) else close(g, x, 1e-6)
            if not ok:
                failures.append((case["name"], g, x))
    # Three of the partner's hand-computed values differ from its own module in the sixth or seventh
    # digit. They are arithmetic slips in the hand values: the module (and critflaw) give the exact
    # results, checked here from first principles.
    assert sorted(name for name, _, _ in failures) == ["charpy_shift_log_half", "charpy_shift_log_two", "ctod_m_inside_interval"]
    assert REF.subsize_charpy_shift(10.0 * 0.75**4) == pytest.approx(-51.4 * math.log(0.5), rel=1e-12)
    assert REF.subsize_charpy_shift(10.0 * 1.5**4) == pytest.approx(-51.4 * math.log(2.0), rel=1e-12)
    slip = next(c for c in cases if c["name"] == "ctod_m_inside_interval")
    ratio = slip["inputs"]["sy"] / slip["inputs"]["su"]
    assert REF.ctod_m(**slip["inputs"]) == pytest.approx(1.517 * ratio**-0.3188, rel=1e-13)
    # the source's own worked values, through the partner's implementation
    assert abs(REF.charpy_master_curve(0.0, -68.0, 60.0) - 90.85) < 0.01
    assert abs(REF.charpy_upper_limit(60.0) - 87.4) < 1e-9
    assert abs(REF.subsize_charpy_shift(5.0) - 19.69) < 0.01
