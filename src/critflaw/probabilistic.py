"""Probabilistic assessment: a problem with random variables as a limit state (theory manual, chapter 04)."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from critflaw.assessment import Assessment, Result
from critflaw.model import Ratio
from critflaw.prob.dependence import RandomVector
from critflaw.prob.diagnostics import Association, FailurePoint, association_with_failure, most_likely_failure_points
from critflaw.prob.distributions import Distribution
from critflaw.prob.families import FAMILIES, NoMatch, family_of, moment_matched
from critflaw.prob.intervals import Interval, clopper_pearson
from critflaw.prob.partial_factors import implied_partial_factor
from critflaw.prob.solvers import Evaluation, ReliabilityResult, consistent_driving_variable, solve

FloatArray = NDArray[np.float64]


def collect(obj: Any) -> list[Distribution]:
    """Distinct distribution objects found in a problem, in order of appearance (by identity)."""
    found: list[Distribution] = []

    def walk(value: Any) -> None:
        if isinstance(value, Distribution):
            if not any(value is d for d in found):
                found.append(value)
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            for f in dataclasses.fields(value):
                walk(getattr(value, f.name))

    walk(obj)
    return found


def substitute(obj: Any, replacements: dict[int, Distribution]) -> Any:
    """Copy of ``obj`` with the listed distribution objects replaced by other distributions."""
    if isinstance(obj, Distribution):
        return replacements.get(id(obj), obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        changes = {}
        for f in dataclasses.fields(obj):
            old = getattr(obj, f.name)
            new = substitute(old, replacements)
            if new is not old:
                changes[f.name] = new
        if not changes:
            return obj
        if isinstance(obj, Ratio):
            return Ratio(changes["value"])
        return dataclasses.replace(obj, **changes)
    return obj


@dataclass(frozen=True)
class FamilyVariant:
    variable: str
    family: str  # the alternative family (or the variable's type if it has none)
    status: str  # solver status, or "no moment-matched alternative" / "no alternative family defined"
    pf: float | None
    guaranteed: Interval | None


def realise(obj: Any, values: dict[int, FloatArray]) -> Any:
    """Copy of ``obj`` with every distribution replaced by its sampled array. Construction of the
    copy re-runs every validation on the realised values; a violation raises (nothing is dropped)."""
    if isinstance(obj, Distribution):
        return values[id(obj)]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        changes = {}
        for f in dataclasses.fields(obj):
            old = getattr(obj, f.name)
            new = realise(old, values)
            if new is not old:
                changes[f.name] = new
        if not changes:
            return obj
        if isinstance(obj, Ratio):
            return Ratio(changes["value"])
        return dataclasses.replace(obj, **changes)
    return obj


@dataclass(frozen=True)
class ModeSplit:
    """Shares of the failure probability by mode, from the failure samples of the result's channel."""

    fracture: float  # exit through the assessment line or a step
    collapse: float  # exit through the cut-off, or the end of tabulated stress-strain data
    out_of_validity: float
    samples: int


@dataclass(frozen=True)
class ProbabilisticResult:
    reliability: ReliabilityResult  # headline, with outside-validity realisations counted as failed
    reliability_valid_only: ReliabilityResult | None  # the same with them counted as safe (None if none occur)
    p_out_of_validity: float  # share of the guaranteed channel's samples that were outside validity
    p_out_of_validity_interval: Interval
    variables: tuple[str, ...]
    modes: ModeSplit | None = field(default=None)
    per: Literal["flaw"] = "flaw"  # the probability is conditional on this one flaw existing

    def summary(self) -> str:
        lines = [f"Failure probability per {self.per} (given that the flaw exists)", self.reliability.summary()]
        if self.reliability_valid_only is not None:
            g = self.reliability_valid_only.guaranteed
            lines.append(
                f"Realisations outside the validity of a solution: probability {self.p_out_of_validity:.3g} "
                f"(99% interval [{self.p_out_of_validity_interval.lower:.3g}, "
                f"{self.p_out_of_validity_interval.upper:.3g}]). "
                f"They are counted as failed above. Counted as safe, the guaranteed interval is "
                f"[{g.lower:.3g}, {g.upper:.3g}]."
            )
        if self.modes is not None and self.modes.samples > 0:
            lines.append(
                f"Failure modes ({self.modes.samples} failure samples): fracture {self.modes.fracture:.0%}, "
                f"collapse {self.modes.collapse:.0%}, outside validity {self.modes.out_of_validity:.0%}"
            )
        return "\n".join(lines)


class ProbabilisticAssessment:
    """An :class:`Assessment` whose fields may hold distributions.

    The failure event is the authoritative indicator of the deterministic engine (theory manual, section 4.3):
    a realisation fails if it is not acceptable at any crack-front position.
    """

    def __init__(self, problem: Assessment) -> None:
        self.problem = problem
        self.variables = collect(problem)
        if not self.variables:
            raise ValueError("the problem has no random variables; use Assessment.run()")
        self.vector = RandomVector(self.variables)

    def correlate(
        self, x: Distribution, y: Distribution, rho: float, kind: Literal["pearson", "latent"] = "pearson"
    ) -> ProbabilisticAssessment:
        self.vector.correlate(x, y, rho, kind)
        return self

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(v.name or f"x{i}" for i, v in enumerate(self.variables))

    def realise(self, u: FloatArray) -> Assessment:
        x = self.vector.to_physical(u)
        values = {id(v): x[:, i] for i, v in enumerate(self.variables)}
        try:
            realised: Assessment = realise(self.problem, values)
        except ValueError as exc:
            raise ValueError(
                f"a sampled realisation violates a physical constraint: {exc}. "
                "Constrain the distributions (truncation, or Ratio for the tensile strength); "
                "realisations are never dropped or clipped."
            ) from exc
        return realised

    def evaluate(self, u: FloatArray) -> Result:
        return self.realise(np.atleast_2d(u)).run(on_invalid="mask")

    def limit_state(self, invalid: Literal["failed", "safe"] = "failed") -> Any:
        """Limit state for the solvers. Realisations outside validity are counted as failed
        (driving variable -inf) or safe (+inf) (theory manual, section 4.6)."""
        if invalid not in ("failed", "safe"):
            raise ValueError("invalid must be 'failed' or 'safe'")

        def fn(u: FloatArray) -> Evaluation:
            r = self.evaluate(u)
            fail = ~r.acceptable & r.valid
            y = consistent_driving_variable(np.where(r.valid, r.radial_margin - 1.0, 0.0), fail)
            if invalid == "failed":
                return Evaluation(np.where(r.valid, y, -np.inf), fail | ~r.valid, ~r.valid)
            return Evaluation(np.where(r.valid, y, np.inf), fail, ~r.valid)

        return fn

    def solve(
        self, seed: int = 0, n_explore: int = 1_000_000, replicates: int = 8, samples_per_level: int = 5000
    ) -> ProbabilisticResult:
        self.vector.check()
        dim = self.vector.size
        failed = solve(self.limit_state("failed"), dim, seed, n_explore, replicates, samples_per_level)
        mc = failed.monte_carlo
        points = mc.failure_points
        if failed.subset is not None and failed.status == "conditional":
            # the same number of failure samples from every replicate, so that none dominates the split
            share = max(1, 20_000 // len(failed.subset.runs))
            points = np.vstack([r.failure_points[:share] for r in failed.subset.runs])
        modes = self._modes(points)
        # Out-of-validity mass from the same nominal samples as the guaranteed interval (theory manual, section 4.6).
        # The valid-only bound is computed whenever any out-of-validity realisation was seen anywhere.
        # (in the Monte Carlo channel, among failure samples, or in any subset run, accepted or not).
        in_subset = failed.subset is not None and any(r.invalid_seen > 0 for r in failed.subset.runs)
        seen_invalid = mc.invalid > 0 or in_subset or (modes is not None and modes.out_of_validity > 0.0)
        valid_only = None
        if seen_invalid:
            valid_only = solve(self.limit_state("safe"), dim, seed, n_explore, replicates, samples_per_level)
        return ProbabilisticResult(
            failed, valid_only, mc.invalid / mc.samples, clopper_pearson(mc.invalid, mc.samples), self.names, modes
        )

    def _modes(self, failure_points: FloatArray) -> ModeSplit | None:
        if failure_points.shape[0] == 0:
            return None
        r = self.evaluate(failure_points[:20_000])
        n = r.valid.size
        invalid = float(np.mean(~r.valid))
        collapse = float(np.mean(r.valid & np.isin(r.exit, ("cutoff", "domain_end"))))
        return ModeSplit(1.0 - invalid - collapse, collapse, invalid, n)

    # ---- diagnostics (theory manual, sections 4.7 to 4.9) -----------------------------------

    def failure_samples(self, result: ProbabilisticResult, limit: int = 20_000) -> FloatArray:
        """Failure samples of the channel that gave the headline (standard normal space)."""
        rel = result.reliability
        if rel.subset is not None and rel.status == "conditional":
            share = max(1, limit // len(rel.subset.runs))
            return np.vstack([r.failure_points[:share] for r in rel.subset.runs])
        return rel.monte_carlo.failure_points[:limit]

    def most_likely_failure_points(self, result: ProbabilisticResult) -> list[FailurePoint]:
        """Indicative failure point(s) in physical variables, one per failure region found among
        the failure samples. Approximate; with more than about ten variables the direction is
        only indicative (each point's label says so)."""
        points = self.failure_samples(result)
        return most_likely_failure_points(self.limit_state("failed"), points, self.vector)

    def implied_partial_factors(
        self, point: FailurePoint, references: Mapping[str, tuple[float, str]]
    ) -> dict[str, float]:
        """Partial factors implied by a most likely failure point against reference values given
        per variable name as ``(reference value, kind)``, kind being "load" or "resistance" (theory
        manual, section 4.9). Approximate, as the point is. The points are listed by share of the failure samples;
        the governing one in the usual sense is the point of smallest ``distance``, which is
        normally but not always the first."""
        if point.physical is None:
            raise ValueError("the failure point has no physical coordinates")
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"variable names are not unique ({self.names}); name the distributions")
        values = dict(zip(self.names, point.physical, strict=True))
        unknown = sorted(set(references) - set(values))
        if unknown:
            raise ValueError(f"not random variables of this problem: {unknown}")
        return {
            name: implied_partial_factor(float(values[name]), reference, kind)
            for name, (reference, kind) in references.items()
        }

    def association(self, result: ProbabilisticResult) -> list[Association]:
        """Association of each variable with failure (not causation; see the function's notes)."""
        return association_with_failure(self.vector, self.failure_samples(result), self.names)

    def family_sensitivity(
        self, seed: int = 0, n_explore: int = 1_000_000, replicates: int = 8, samples_per_level: int = 5000
    ) -> list[FamilyVariant]:
        """Re-run the assessment with each normal, lognormal or Weibull variable replaced by the other
        two families, matched in mean, standard deviation and support. Variables of other kinds are
        listed as having no alternative family. The copula (latent) coefficients are kept: a
        correlation declared as a Pearson coefficient is not solved again for the new marginal,
        so its Pearson value changes slightly in a variant. The baseline is :meth:`solve` with the
        same arguments; it is not repeated in the list."""
        out: list[FamilyVariant] = []
        for i, variable in enumerate(self.variables):
            declared = family_of(variable)
            if declared is None:
                kind = type(variable).__name__
                out.append(FamilyVariant(self.names[i], kind, "no alternative family defined", None, None))
                continue
            for family in FAMILIES:
                if family == declared:
                    continue
                try:
                    alternative = moment_matched(variable, family)
                except NoMatch:
                    out.append(FamilyVariant(self.names[i], family, "no moment-matched alternative", None, None))
                    continue
                variant = ProbabilisticAssessment(substitute(self.problem, {id(variable): alternative}))
                variant.vector._latent = self.vector.latent_correlation  # same copula coefficients
                rel = variant.solve(seed, n_explore, replicates, samples_per_level).reliability
                out.append(FamilyVariant(self.names[i], family, rel.status, rel.pf, rel.guaranteed))
        return out