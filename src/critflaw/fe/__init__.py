"""FE backend: WARP3D deck writer, runner, parser, cache and QA, and the ``deformation``
material law shared with Option 2 (theory manual, chapter 06)."""

from critflaw.fe.fecase import FECase, FEResult, QACheck, WarpJSolver
from critflaw.fe.warp_law import WarpDeformationLaw, normalised_strain, transition_arc

__all__ = ["FECase", "FEResult", "QACheck", "WarpDeformationLaw", "WarpJSolver", "normalised_strain", "transition_arc"]
