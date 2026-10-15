"""Phase 4: the deck unit conversions (pure, no image needed)."""

from __future__ import annotations

import math

import pytest

from critflaw.fe import units as U


def test_units_round_trip() -> None:
    assert U.stress_from_deck(U.stress_to_deck(355e6)) == 355e6
    assert U.length_from_deck(U.length_to_deck(0.02)) == 0.02
    assert U.k_from_deck(1.0) == pytest.approx(1e6 * math.sqrt(1e-3))  # 1 MPa sqrt(mm) in Pa sqrt(m)
    assert U.j_from_deck(1.0) == 1e3  # 1 N/mm = 1000 N/m
