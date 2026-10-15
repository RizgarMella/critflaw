"""Training and held-out designs over the emulator's input box (theory manual, sections 6.4.1 and 6.5): scrambled Sobol'
points in the 5-D input box with fixed seeds, mapped to ``FECase`` records with the declared validity bounds
(nu = 0.3, c/W = 0.05, B = 20 mm, E = 207 GPa) and the converged remote length."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc

from critflaw.fe import crack_mesh as CM
from critflaw.fe.fecase import LEVELS, FECase

FloatArray = NDArray[np.float64]

NAMES = ("a_over_B", "a_over_c", "n", "sigma0_over_E", "beta")
LOWER = np.array([0.2, 0.4, 3.0, 1e-3, 0.0])
UPPER = np.array([0.6, 1.0, 20.0, 5e-3, 1.0])
B = 20e-3
E = 207e9
NU = 0.3
C_OVER_W = 0.05
TRAIN_SEED = 20261010
HELDOUT_SEED = 20261011


@dataclass(frozen=True)
class DesignPoint:
    x: FloatArray  # (a/B, a/c, n, sigma0/E, beta)
    role: str  # "train" | "heldout" | "check"
    level: tuple[int, int, int, int, int]

    def case(self, c_over_w: float = C_OVER_W) -> FECase:
        a_b, a_c, n, s_e, beta = (float(v) for v in self.x)
        a = a_b * B
        c = a / a_c
        w = c / c_over_w
        return FECase(
            a=a, c=c, B=B, W=w, L_far=CM.remote_length(B, w), E=E, nu=NU, sigma_0=s_e * E, n=n, beta=beta,
            level=self.level,
        )  # fmt: skip


def schedule_lr(x: FloatArray) -> FloatArray:
    """Lr of the 12 FE load steps of the case at ``x`` (the closed-form schedule of ``PlasticCase``)."""
    pc = DesignPoint(np.asarray(x, dtype=float), "train", LEVELS[0]).case().plastic_case()
    return np.asarray(pc.schedule(), dtype=float) * pc.lr_per_unit()


def sobol_points(n: int, seed: int) -> FloatArray:
    m = int(np.ceil(np.log2(max(n, 2))))  # the first n points of a 2^m scrambled Sobol' block
    u = qmc.Sobol(len(NAMES), scramble=True, seed=seed).random_base2(m)[:n]
    return np.asarray(LOWER + u * (UPPER - LOWER), dtype=float)


def training_design(n_train: int = 48, n_pairs: int = 12) -> list[DesignPoint]:
    """``n_train`` level-1 points and, for every ``n_train // n_pairs``-th point, the same point at level 2."""
    xs = sobol_points(n_train, TRAIN_SEED)
    out = [DesignPoint(x, "train", LEVELS[0]) for x in xs]
    step = n_train // n_pairs
    out += [DesignPoint(xs[i], "train", LEVELS[1]) for i in range(0, n_train, step)][:n_pairs]
    return out


def heldout_design(n: int = 12) -> list[DesignPoint]:
    return [DesignPoint(x, "heldout", LEVELS[0]) for x in sobol_points(n, HELDOUT_SEED)]


__all__ = ["schedule_lr", "NAMES", "LOWER", "UPPER", "DesignPoint", "sobol_points", "training_design", "heldout_design"]
