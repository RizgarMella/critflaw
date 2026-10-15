"""Deck units (theory manual, section 6.1.6): the deck is written in N, mm, MPa (so that
WARP3D's fixed-width printed output does not overflow and its number scanner reads every value: a deck in Pa
with 12-digit values is mis-scanned); the library's SI values are converted on the way in and
every parsed quantity on the way out. J in N/mm = 1e3 J/m^2; K in MPa sqrt(mm) = sqrt(1e-3) MPa sqrt(m)."""

from __future__ import annotations

import math

MM_PER_M = 1.0e3
MPA_PER_PA = 1.0e-6


def length_to_deck(metres: float) -> float:
    return metres * MM_PER_M


def length_from_deck(mm: float) -> float:
    return mm / MM_PER_M


def stress_to_deck(pa: float) -> float:
    return pa * MPA_PER_PA


def stress_from_deck(mpa: float) -> float:
    return mpa / MPA_PER_PA


def j_from_deck(n_per_mm: float) -> float:
    """J in N/mm -> J/m^2 (N/m)."""
    return n_per_mm * MM_PER_M


def k_from_deck(mpa_sqrt_mm: float) -> float:
    """K in MPa sqrt(mm) -> Pa sqrt(m)."""
    return mpa_sqrt_mm / MPA_PER_PA * math.sqrt(1.0 / MM_PER_M)


def force_from_deck(newton: float) -> float:
    return newton
