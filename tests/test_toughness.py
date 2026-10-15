"""Toughness conversions and correlations, checked against the source's own worked values."""

from __future__ import annotations

import numpy as np
import pytest

from critflaw import toughness as tg

MPA = 1e6


def test_k_from_j_and_ctod() -> None:
    assert tg.k_from_j(100e3, 207e9) == pytest.approx(np.sqrt(207e9 * 100e3 / 0.91), rel=1e-14)
    m = 1.517 * (355 / 490) ** -0.3188
    assert tg.ctod_constraint_factor(355e6, 490e6) == pytest.approx(m, rel=1e-14)
    assert tg.ctod_constraint_factor(99e6, 100e6) == 1.5 and tg.ctod_constraint_factor(29e6, 100e6) == 1.5
    k = tg.k_from_ctod(0.2e-3, 355e6, 490e6, 207e9)
    assert k == pytest.approx(np.sqrt(m * 355e6 * 0.2e-3 * 207e9 / 0.91), rel=1e-14)
    assert 100 * MPA < k < 200 * MPA  # dimensional sanity: 0.2 mm CTOD in a C-Mn steel


def test_subsize_correction_and_limit() -> None:
    assert tg.subsize_correction(100 * MPA, 0.010, 0.040) == pytest.approx((20 + 80 * 0.25**0.25) * MPA, rel=1e-14)
    assert tg.subsize_correction(100 * MPA, 0.025, 0.025) == pytest.approx(100 * MPA)
    # units: E and yield in MPa, ligament in m gives MPa m^0.5 in the source; same in SI
    assert tg.k_jc_limit(0.025, 400e6, 207e9) == pytest.approx(np.sqrt(207e3 * 0.025 * 400 / (30 * 0.91)) * MPA, rel=1e-12)


def test_master_curve_median_and_crack_front_length() -> None:
    # median at a 25 mm crack front: 30 + 70 exp(0.019 (T - T0))
    for dt in (-50.0, 0.0, 40.0):
        k = tg.master_curve(0.5, dt, 0.0, 0.025, 1.0)
        assert k == pytest.approx((30 + 70 * np.exp(0.019 * dt)) * MPA, rel=5e-3)  # the rounded median form
    assert tg.master_curve(0.5, 0.0, 0.0, 0.025, 1.0) == pytest.approx(100 * MPA, rel=5e-3)  # T0 definition (20 + 88 ln(2)^0.25 = 100.3)
    # crack-front-length adjustment and the cap at twice the thickness
    base = tg.master_curve(0.3, 10.0, -40.0, 0.025, 1.0)
    short = tg.master_curve(0.3, 10.0, -40.0, 0.015, 1.0)
    assert (short - 20 * MPA) / (base - 20 * MPA) == pytest.approx((25 / 15) ** 0.25, rel=1e-12)
    assert tg.master_curve(0.3, 10.0, -40.0, 0.5, 0.025) == pytest.approx(
        float(tg.master_curve(0.3, 10.0, -40.0, 0.050, 0.025)), rel=1e-14
    )
    assert tg.master_curve(0.0, 10.0, -40.0, 0.025, 1.0) == tg.K_MIN_MASTER_CURVE  # lower bound
    with pytest.raises(ValueError):
        tg.master_curve(1.0, 0.0, 0.0, 0.025, 1.0)


def test_source_worked_example_for_the_charpy_correlations() -> None:
    # 60 mm ferritic steel, T27J = -50 C, 60 J at the service temperature of 0 C
    t0 = tg.t0_from_charpy(-50.0)
    assert t0 == -68.0
    k = tg.charpy_master_curve(0.0, t0, 0.060)
    assert k / MPA == pytest.approx(90.8, abs=0.06)
    assert tg.charpy_upper_limit(60.0, 400e6) / MPA == pytest.approx(87.4, abs=1e-9)
    # with the exponent coefficient printed in the probabilistic annex (0.01055) the example is not reproduced
    wrong = 20 + (11 + 77 * np.exp(0.01055 * (0 + 68 - 25))) * (25 / 60) ** 0.25 * np.log(1 / 0.95) ** 0.25
    assert abs(wrong - 90.8) > 15
    assert tg.t0_from_charpy(-50.0, 40) == -74.0
    with pytest.raises(ValueError):
        tg.charpy_upper_limit(60.0, 480e6)


def test_other_charpy_relations() -> None:
    assert tg.charpy_lower_bound(27.0, 0.025) == pytest.approx((12 * np.sqrt(27) - 20 + 20) * MPA, rel=1e-14)
    assert tg.charpy_lower_bound(27.0, 0.100) < tg.charpy_lower_bound(27.0, 0.025)
    # source example: a 5 mm sub-size specimen is shifted upward by 20 C
    assert tg.subsize_charpy_temperature_shift(0.005) == pytest.approx(19.7, abs=0.05)
    assert tg.subsize_charpy_temperature_shift(0.010) == pytest.approx(0.0, abs=1e-12)
    cv = 200.0
    j = 0.53 * cv**1.28 * 0.2 ** (0.133 * cv**0.256)  # kJ/m^2
    assert tg.charpy_upper_shelf(cv, 207e9) == pytest.approx(np.sqrt(207e9 * j * 1e3 / 0.91), rel=1e-12)
    # consistency of the two routes: K from that J-integral
    assert tg.charpy_upper_shelf(cv, 207e9) == pytest.approx(float(tg.k_from_j(j * 1e3, 207e9)), rel=1e-12)


def test_minimum_of_three_equivalent() -> None:
    data = np.arange(1.0, 16.0)[::-1]
    assert tg.minimum_of_three_equivalent(data[:3]) == 13.0
    assert tg.minimum_of_three_equivalent(data[:5]) == 11.0
    assert tg.minimum_of_three_equivalent(data[:6]) == 11.0  # second lowest of 15..10
    assert tg.minimum_of_three_equivalent(data[:10]) == 7.0
    assert tg.minimum_of_three_equivalent(data[:11]) == 7.0  # third lowest of 15..5
    assert tg.minimum_of_three_equivalent(data) == 3.0
    for n in (2, 16):
        with pytest.raises(ValueError):
            tg.minimum_of_three_equivalent(np.ones(n))


SOURCE_TABLE = {3: 3.039, 4: 2.295, 5: 1.976, 6: 1.794, 7: 1.675, 8: 1.590, 9: 1.525, 10: 1.474, 11: 1.432,
                12: 1.397, 13: 1.368, 14: 1.342, 15: 1.320, 16: 1.307, 17: 1.283, 18: 1.268, 19: 1.253, 20: 1.240}


def test_tolerance_factor_reproduces_the_source_table_except_one_entry() -> None:
    for m, printed in SOURCE_TABLE.items():
        exact = tg.tolerance_factor(m)
        if m == 16:
            # the printed 1.307 breaks the sequence; the exact factor is 1.301 (a misprint in the source)
            assert exact == pytest.approx(1.301, abs=1e-3) and abs(exact - printed) > 5e-3
        else:
            assert exact == pytest.approx(printed, abs=1.5e-3), m


def test_lower_bound_toughness() -> None:
    values = np.array([110.0, 95.0, 130.0, 120.0, 105.0]) * MPA
    k = tg.lower_bound_toughness(values)
    assert k == pytest.approx(np.mean(values) - tg.tolerance_factor(5) * np.std(values, ddof=1), rel=1e-14)
    log_k = tg.lower_bound_toughness(values, lognormal=True)
    assert 0 < k < values.min() and 0 < log_k < values.min() and log_k != pytest.approx(k)
    # coverage check: the bound is below the true 20th percentile in about 90% of samples
    rng = np.random.default_rng(1)
    samples = rng.normal(100.0, 15.0, size=(20000, 6))
    bounds = samples.mean(axis=1) - tg.tolerance_factor(6) * samples.std(axis=1, ddof=1)
    assert np.mean(bounds < 100.0 - 0.8416212 * 15.0) == pytest.approx(0.90, abs=0.01)
