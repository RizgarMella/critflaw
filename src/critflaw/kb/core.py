"""K-solution knowledge base (theory manual, chapter 08): source classes, the status ladder computed from
test-linked evidence, composite and region-scoped status, FE evidence with its exclusion of the solutions the FE
was validated against, the user loader contract, the release audit and the coverage map.

Every current built-in coefficient set and table is ``standard-derived`` (transcribed from the licensed standard):
used locally, never shipped until checked against an open source and cleared by the owner."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from critflaw.provenance import EQUATIONS

SourceClass = Literal["standard-derived", "open", "handbook"]
Status = Literal["unverified", "self-consistent", "cross-checked", "fe-verified"]
LADDER: tuple[Status, ...] = ("unverified", "self-consistent", "cross-checked", "fe-verified")
RANK: dict[str, int] = {s: i for i, s in enumerate(LADDER)}
ROOT = Path(__file__).resolve().parents[3]
# applicability boxes of oracle evidence that covers only part of a record's range: {record: {kind: box}}; none
# today (every linked test exercises the record over its declared range); add one when a partial test appears
EVIDENCE_BOXES: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}


def record_status(
    record_id: str,
    evidence: Mapping[str, Mapping[str, tuple[str, ...]]],
    params: Mapping[str, float] | None = None,
    boxes: Mapping[str, Mapping[str, Mapping[str, tuple[float, float]]]] | None = None,
) -> Status:
    """A record's status from its **linked** oracle kinds (theory manual, section 8.3.1): kind a or b ->
    ``cross-checked``; only c, d, e -> ``self-consistent`` (they cannot catch an error in the source text); none ->
    ``unverified``. A kind claimed in the provenance record but without a linked test does not count."""
    if record_id not in EQUATIONS:
        raise KeyError(f"unknown record {record_id!r}")
    claimed = set(EQUATIONS[record_id].oracles)
    linked = {k for k, tests in evidence.get(record_id, {}).items() if tests and k in claimed}
    # region scope: a kind whose evidence declares an applicability box counts only at
    # parameters inside it (outside, or without parameters, it does not count); a kind without a box counts
    # over the record's declared applicability, which the implementing module enforces by refusal
    rbox = (boxes or EVIDENCE_BOXES).get(record_id, {})
    for k in list(linked):
        if k in rbox:
            box = rbox[k]
            inside = (
                params is not None
                and bool(box)
                and all(n in params and lo <= params[n] <= hi for n, (lo, hi) in box.items())
            )
            if not inside:
                linked.discard(k)
    if linked & {"a", "b"}:
        return "cross-checked"
    if linked & {"c", "d", "e"}:
        return "self-consistent"
    return "unverified"


@dataclass(frozen=True)
class FEEvidence:
    """FE comparison evidence for a solution over an applicability box. ``validated_against`` names the solutions
    the FE pipeline was itself validated against: this evidence never promotes any of them (section 8.5)."""

    solution_id: str
    box: dict[str, tuple[float, float]]
    validated_against: frozenset[str]
    family_verified: bool  # the FE family verification complete (D4-D8 passed, D3 deep/bending band resolved)
    note: str = ""

    def covers(self, params: Mapping[str, float]) -> bool:
        if not self.box:  # an empty box covers nothing (never every parameter set)
            return False
        return all(k in params and lo <= params[k] <= hi for k, (lo, hi) in self.box.items())

    def promotes(self, solution_id: str) -> bool:
        return self.family_verified and solution_id == self.solution_id and solution_id not in self.validated_against


@dataclass(frozen=True)
class Solution:
    id: str
    quantity: Literal["K", "reference_stress", "curve", "factor"]
    family: str  # geometry / flaw / loading family
    route: str  # the implementing callable(s)
    records: tuple[str, ...]  # the provenance records the route uses
    source_class: SourceClass = "standard-derived"
    outer_box: dict[str, tuple[float, float]] = field(default_factory=dict)  # coverage map only; modules refuse

    def status(
        self,
        evidence: Mapping[str, Mapping[str, tuple[str, ...]]],
        params: Mapping[str, float] | None = None,
        fe: tuple[FEEvidence, ...] = (),
        suspect_cell: bool = False,
    ) -> Status:
        """Composite status at an evaluation: the minimum over the records used; ``unverified`` when the evaluation
        read a table cell flagged suspect; ``fe-verified`` only from FE evidence that covers ``params`` and may
        promote this solution (not validated against it, and the family verification complete)."""
        if suspect_cell:
            return "unverified"
        ranks = [RANK[record_status(r, evidence)] for r in self.records] or [0]
        base = min(ranks)
        if base >= RANK["cross-checked"] and params is not None:
            if any(e.promotes(self.id) and e.covers(params) for e in fe):
                return "fe-verified"
        return LADDER[base]


BUILTIN: tuple[Solution, ...] = (
    # plate (the generic K form k.general is used by every K route)
    Solution("k.plate.surface", "K", "plate / surface flaw / membrane + bending", "solutions.plate.stress_intensity",
             ("k.general", "k.plate.surface"), outer_box={"a_over_B": (0.0, 0.8), "a_over_c": (0.0, 2.0)}),
    Solution("k.plate.through", "K", "plate / through flaw", "solutions.plate.stress_intensity",
             ("k.general", "k.plate.through")),
    Solution("k.plate.embedded", "K", "plate / embedded flaw", "solutions.embedded",
             ("k.general", "k.plate.embedded")),
    Solution("k.plate.extended", "K", "plate / extended surface flaw", "solutions.cylinder.extended_plate_mm_mb",
             ("k.general", "k.plate.extended")),
    Solution("sref.plate.surface", "reference_stress", "plate / surface flaw",
             "solutions.plate.surface_flaw_reference_stress", ("sref.plate.surface",)),
    Solution("sref.plate.through", "reference_stress", "plate / through flaw",
             "solutions.plate.through_flaw_reference_stress", ("sref.plate.through",)),
    Solution("sref.plate.embedded", "reference_stress", "plate / embedded flaw",
             "solutions.embedded.embedded_flaw_reference_stress", ("sref.plate.embedded",)),
    Solution("sref.plate.embedded_extended", "reference_stress", "plate / extended embedded flaw",
             "solutions.embedded.extended_embedded_reference_stress", ("sref.plate.embedded_extended",)),
    # cylinders: tables
    Solution("k.cylinder.axial_surface", "K", "cylinder / axial surface flaw (tables)",
             "solutions.cylinder.interpolate", ("k.general", "k.cylinder.tables.axial_surface")),
    Solution("k.cylinder.axial_extended_external", "K", "cylinder / axial extended external flaw (tables)",
             "solutions.cylinder.interpolate", ("k.general", "k.cylinder.tables.axial_extended_external")),
    Solution("k.cylinder.circumferential_surface", "K", "cylinder / circumferential surface flaw (tables)",
             "solutions.cylinder.interpolate", ("k.general", "k.cylinder.tables.circumferential_surface")),
    Solution("k.cylinder.circumferential_extended", "K", "cylinder / circumferential extended flaw (tables)",
             "solutions.cylinder.interpolate", ("k.general", "k.cylinder.tables.circumferential_extended")),
    Solution("k.cylinder.extended_external", "K", "cylinder / fully circumferential external flaw",
             "solutions.cylinder.fully_circumferential_external_mm", ("k.general", "k.cylinder.extended_external")),
    Solution("k.cylinder.through", "K", "cylinder / through flaw (tables)", "solutions.cylinder.interpolate_columns",
             ("k.general", "k.cylinder.tables.through")),
    # cylinders: plate solutions with bulging (the routes the cylinder dispatch uses under pressure)
    Solution("k.cylinder.axial_surface_bulging", "K", "cylinder / axial surface flaw under pressure (plate x bulging)",
             "solutions.cylinder_dispatch.cylinder_factors",
             ("k.general", "k.plate.surface", "k.cylinder.bulging_surface")),
    Solution("k.cylinder.axial_through_bulging", "K", "cylinder / axial through flaw under pressure (plate x bulging)",
             "solutions.cylinder_dispatch.cylinder_factors",
             ("k.general", "k.plate.through", "k.cylinder.bulging_through")),
    # cylinders: reference stress, one entry per branch (branches are alternatives, never combined)
    Solution("sref.cylinder.axial_thin", "reference_stress", "cylinder / axial flaws, thin-wall",
             "solutions.cylinder_sref", ("sref.cylinder.axial_thin",)),
    Solution("sref.cylinder.axial_pressure", "reference_stress", "cylinder / axial flaws, limit pressure",
             "solutions.cylinder_sref", ("sref.cylinder.axial_pressure",)),
    Solution("sref.cylinder.axial_thick_bending", "reference_stress", "cylinder / axial flaws, thick-wall bending",
             "solutions.cylinder_sref.axial_external_thick_limits", ("sref.cylinder.axial_thick_bending",)),
    Solution("sref.cylinder.circumferential_thin", "reference_stress", "cylinder / circumferential flaws, thin-wall",
             "solutions.cylinder_sref", ("sref.cylinder.circumferential_thin",)),
    Solution("sref.cylinder.circumferential_thick", "reference_stress", "cylinder / circumferential flaws, thick-wall",
             "solutions.cylinder_sref.circumferential_thick_limit", ("sref.cylinder.circumferential_thick",)),
    # FAD curves (with the cut-off they always carry)
    Solution("curve.option1", "curve", "FAD Option 1", "fad.curves.Option1Curve",
             ("fad.option1.continuous", "fad.option1.discontinuous", "material.luders_strain", "fad.cutoff")),
    Solution("curve.option2", "curve", "FAD Option 2", "fad.option2.Option2Curve", ("fad.option2", "fad.cutoff")),
)  # fmt: skip

REGISTRY: dict[str, Solution] = {s.id: s for s in BUILTIN}

# the stored D3 comparison (theory manual, sections 6.2 and 8.5): evidence for the plate surface solution, but the
# FE was itself validated against it (D3) and the family verification is incomplete — it never promotes it
D3_EVIDENCE = FEEvidence(
    "k.plate.surface", {"a_over_B": (0.2, 0.6), "a_over_c": (0.4, 1.0)}, frozenset({"k.plate.surface"}),
    family_verified=False, note="34-case D3 design: 25 pass, 8 unresolved, 1 FAIL",
)  # fmt: skip


# Package contents that carry coefficient sets or tables (numeric literal containers of >= 6 numbers, found by
# `content_inventory`), each mapped to the provenance records it implements; anything found outside this map
# fails the release audit's inventory test. Closed-form coefficients written as expressions
# are covered by the module inventory (every solution module implements records).
CONTENT_RECORDS: dict[str, tuple[str, ...]] = {
    "solutions/cylinder_tables.py": (
        "k.cylinder.tables.through", "k.cylinder.tables.axial_surface", "k.cylinder.tables.axial_extended_external",
        "k.cylinder.tables.circumferential_surface", "k.cylinder.tables.circumferential_extended",
    ),
    "fad/option2.py": (),  # a check grid of Lr points (no coefficient set)
    "fatigue/growth.py": (),  # integer flaw-kind codes (no coefficient set)
}  # fmt: skip


def content_inventory(package: Path | None = None) -> dict[str, int]:
    """Modules of the package (excluding kb, fe, option3) holding numeric literal containers of >= 6 numbers:
    {relative path: count}."""
    import ast

    root = package or ROOT / "src" / "critflaw"
    out: dict[str, int] = {}
    for f in sorted(root.rglob("*.py")):
        rel = f.relative_to(root).as_posix()
        if rel.split("/")[0] in ("kb", "fe", "option3"):
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.List | ast.Tuple):
                n = sum(
                    isinstance(e, ast.Constant) and isinstance(e.value, int | float) and not isinstance(e.value, bool)
                    for e in node.elts
                )
                if n >= 6:
                    out[rel] = out.get(rel, 0) + 1
    return out


REQUIRED_CONTRACT = ("units", "normalisation", "dimensions", "front_position", "loading", "interpolation", "validity")


def load_user(path: Path) -> Solution:
    """A user-supplied solution file (JSON) under the mandatory contract (theory manual, section 8.6). It enters
    as ``unverified`` and ``handbook``; it cannot reuse a built-in id nor declare its own status or class."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in ("id", "quantity", "family", *REQUIRED_CONTRACT) if k not in data]
    if missing:
        raise ValueError(f"user solution {path}: missing contract fields {missing}")
    if data["id"] in REGISTRY:
        raise ValueError(f"user solution id {data['id']!r} collides with a built-in solution")
    if data["quantity"] not in ("K", "reference_stress", "curve", "factor"):
        raise ValueError(f"user solution {path}: quantity {data['quantity']!r} is not supported")
    for k in ("id", "family", "units", "normalisation", "dimensions", "front_position", "loading", "interpolation"):
        if not isinstance(data[k], str) or not data[k].strip():
            raise ValueError(f"user solution {path}: contract field {k!r} must be a non-empty description")
    if not isinstance(data["validity"], dict) or not data["validity"]:
        raise ValueError(f"user solution {path}: the validity box must name at least one parameter range")
    for k, v in data["validity"].items():
        if not (
            isinstance(v, list) and len(v) == 2
            and all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t) for t in v)
            and v[0] < v[1]
        ):  # fmt: skip
            raise ValueError(f"user solution {path}: validity range {k!r} must be [low, high] with low < high")
    for forbidden in ("status", "source_class"):
        if forbidden in data:
            raise ValueError(f"user solution {path}: {forbidden!r} cannot be declared (computed / always 'handbook')")
    box = {k: (float(v[0]), float(v[1])) for k, v in data["validity"].items()}
    return Solution(str(data["id"]), data["quantity"], str(data["family"]), f"user:{path}", (), "handbook", box)


def release_audit(cleared: frozenset[str] = frozenset(), user: tuple[Solution, ...] = ()) -> list[str]:
    """Everything that may not ship (theory manual, section 8.7), over **every implemented provenance record**
    (the package contents' coefficient sets, tables and formulae — not only the registry's routes) plus any
    user-loaded solution: a record ships only if the owner cleared it **and** it was checked against an open
    primary source (``checked_against_primary == 'yes'``); ``handbook`` (user-loaded) material never ships.
    Returns the blocked record ids (and user solution ids)."""
    blocked = [
        rid for rid, rec in EQUATIONS.items()
        if not (rid in cleared and rec.checked_against_primary == "yes")
    ]  # fmt: skip
    return blocked + [s.id for s in user if s.source_class == "handbook"]


def coverage(evidence: Mapping[str, Mapping[str, tuple[str, ...]]]) -> list[dict[str, object]]:
    """For each built-in solution: family, outer box (an outer box — the modules' coupled refusals are
    authoritative), status from the evidence, source class, and the records whose claimed kinds are unlinked."""
    rows: list[dict[str, object]] = []
    for s in BUILTIN:
        unlinked = [r for r in s.records if record_status(r, evidence) != "cross-checked"]
        rows.append({
            "id": s.id, "family": s.family, "quantity": s.quantity, "outer_box": s.outer_box,
            "status": s.status(evidence), "source_class": s.source_class, "below_cross_checked_records": unlinked,
        })  # fmt: skip
    return rows


__all__ = [
    "Solution", "FEEvidence", "REGISTRY", "BUILTIN", "D3_EVIDENCE", "record_status", "load_user", "release_audit",
    "coverage", "RANK",
]  # fmt: skip
