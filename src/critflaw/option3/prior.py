"""Physics prior for the Option 3 emulator: the Option 2 curve of the same WARP3D ``deformation`` law.

Option 2 (``Option2Curve.from_warp_law``) gives a material-only ``Kr(Lr)``; ``R0 = Kr^-2`` is its
analogue of ``J/J_e``. The emulator models the FE increments relative to the prior's increments,
``y_k - log(dR0_k)``, which removes the dominant material dependence (the hardening exponent and ``sigma_0/E``
set most of the curve) and leaves the geometry and constraint effects to the GP — a smoother quantity, so fewer
FE runs reach the same accuracy. The prior is deterministic; it changes the GP's mean, not the constraints
(``Kr <= 1`` and monotonicity still hold by construction for every draw)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from critflaw.fad.option2 import Option2Curve
from critflaw.fe.warp_law import WarpDeformationLaw
from critflaw.option3.curve import INCREMENT_FLOOR

FloatArray = NDArray[np.float64]

E_REF = 207e9  # the FE training modulus (theory manual, section 6.4.1); the prior depends on sigma0/E and n only
NU_REF = 0.3
UTS_FACTOR = 50.0  # a nominal tensile strength far above sigma_0: Lr,max plays no part below Lr = 1.2


def option2_log_increments(x: FloatArray, lr: FloatArray) -> FloatArray:
    """``log(max(dR0_k, floor))`` for k = 2..K at the curve's step ``lr`` values, ``R0 = Kr_Option2^-2``."""
    s0 = float(x[3]) * E_REF
    law = WarpDeformationLaw(E_REF, NU_REF, s0, float(x[2]))
    # s_end is a read-only property of the law, which the protocol declares as a variable
    curve = Option2Curve.from_warp_law(law, UTS_FACTOR * s0)  # type: ignore[arg-type]
    kr = np.asarray(curve(np.asarray(lr, dtype=float)), dtype=float)
    r0 = kr**-2.0
    r0 = r0 / r0[0]  # the same normalisation as the FE curve (J/J_e = 1 at step 1)
    return np.asarray(np.log(np.maximum(np.diff(r0), INCREMENT_FLOOR)), dtype=float)


__all__ = ["option2_log_increments"]
