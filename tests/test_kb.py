"""Phase 7: the knowledge base — status from linked evidence (spec S1.5 ladder), composite and suspect-cell status,
region-scoped FE evidence with the S6.6 exclusion, the user loader contract, the release audit and the coverage
map."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from critflaw.kb import core as K
from critflaw.provenance import EQUATIONS


def test_status_ladder_counts_only_linked_kinds() -> None:
    rid = "k.plate.surface"
    claimed = set(EQUATIONS[rid].oracles)
    assert K.record_status(rid, {}) == "unverified"
    cd = {k: ("tests/x.py::t",) for k in claimed & {"c", "d", "e"}}
    if cd:
        assert K.record_status(rid, {rid: cd}) == "self-consistent"  # c/d/e cannot catch a source-text error
    ab = sorted(claimed & {"a", "b"})
    if ab:
        assert K.record_status(rid, {rid: {ab[0]: ("tests/x.py::t",)}}) == "cross-checked"
    # a kind linked but not claimed in provenance does not count; an empty link does not count
    assert K.record_status(rid, {rid: {"z": ("t",), **{k: () for k in claimed}}}) == "unverified"
    with pytest.raises(KeyError):
        K.record_status("no.such.record", {})


def test_composite_status_is_the_minimum_and_suspect_cells_degrade() -> None:
    sol = K.REGISTRY["k.plate.surface"]
    full = {r: {k: ("tests/x.py::t",) for k in EQUATIONS[r].oracles} for r in sol.records}
    best = sol.status(full)
    weakest = min(K.RANK[K.record_status(r, full)] for r in sol.records)
    assert K.RANK[best] == weakest
    assert sol.status(full, suspect_cell=True) == "unverified"


def test_fe_evidence_is_region_scoped_and_never_promotes_what_validated_the_fe() -> None:
    sol = K.REGISTRY["k.plate.surface"]
    full = {r: {k: ("tests/x.py::t",) for k in ("a", "b", "c", "d")} for r in sol.records}
    inside = {"a_over_B": 0.4, "a_over_c": 0.7}
    # the stored D3 evidence: the FE was validated against this very solution -> no promotion, ever (S6.6)
    assert not K.D3_EVIDENCE.promotes("k.plate.surface")
    assert sol.status(full, inside, (K.D3_EVIDENCE,)) != "fe-verified"
    family_done = K.FEEvidence("k.plate.surface", K.D3_EVIDENCE.box, frozenset({"k.plate.surface"}), True)
    assert sol.status(full, inside, (family_done,)) != "fe-verified"  # still excluded once D4/D5 pass
    # a fixture that does reach cross-checked (a solution on one record with linked kind b), so the promotion and
    # the out-of-region case are exercised (P7-CC-02 / P-06)
    fixture = K.Solution("k.plate.surface", "K", "fixture", "x", ("k.plate.surface",))
    ev = {"k.plate.surface": {"b": ("tests/x.py::t",)}}
    assert "b" in EQUATIONS["k.plate.surface"].oracles and fixture.status(ev) == "cross-checked"
    independent = K.FEEvidence("k.plate.surface", K.D3_EVIDENCE.box, frozenset({"k.other"}), True)
    assert fixture.status(ev, inside, (independent,)) == "fe-verified"
    assert fixture.status(ev, {"a_over_B": 0.7, "a_over_c": 0.7}, (independent,)) == "cross-checked"  # outside box
    assert fixture.status(ev, inside, (K.D3_EVIDENCE,)) == "cross-checked"  # S6.6: never promoted by D3
    empty = K.FEEvidence("k.plate.surface", {}, frozenset(), True)
    assert not empty.covers(inside)  # an empty box covers nothing


def test_user_loader_contract(tmp_path: Path) -> None:
    good = {
        "id": "user.k.custom", "quantity": "K", "family": "plate / surface flaw", "units": "SI",
        "normalisation": "K / (sigma sqrt(pi a))", "dimensions": "a depth, c half-length", "front_position": "phi from surface",
        "loading": "membrane", "interpolation": "linear in a/c", "validity": {"a_over_B": [0.1, 0.5]},
    }  # fmt: skip
    p = tmp_path / "s.json"
    p.write_text(json.dumps(good), encoding="utf-8")
    sol = K.load_user(p)
    assert sol.source_class == "handbook" and sol.records == () and sol.status({}) == "unverified"
    for bad, msg in ((dict(good, quantity="stress"), "not supported"), (dict(good, units=""), "non-empty"),
                     (dict(good, validity={"a_over_B": [0.5, 0.1]}), "low < high"), (dict(good, validity={}), "at least"),
                     (dict(good, id="k.plate.surface"), "collides"), ({k: v for k, v in good.items() if k != "units"},
                     "missing"), (dict(good, status="cross-checked"), "cannot be declared"),
                     (dict(good, source_class="open"), "cannot be declared")):  # fmt: skip
        p.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError, match=msg):
            K.load_user(p)


def test_release_audit_covers_every_record_and_requires_clearance_and_an_open_check(tmp_path: Path) -> None:
    blocked = K.release_audit()
    assert set(blocked) == set(EQUATIONS)  # every implemented record is blocked today (standard-derived)
    # clearance alone does not release a record not checked against an open primary source
    assert set(K.release_audit(cleared=frozenset(EQUATIONS))) == {
        r for r, rec in EQUATIONS.items() if rec.checked_against_primary != "yes"
    }
    # handbook material never ships
    user = K.Solution("user.x", "K", "f", "user", (), "handbook")
    assert "user.x" in K.release_audit(cleared=frozenset(EQUATIONS) | {"user.x"}, user=(user,))
    # inventory: every module of src/critflaw/solutions implements at least one provenance record
    sol_dir = Path(__file__).resolve().parents[1] / "src" / "critflaw" / "solutions"
    implemented = " ".join(rec.implemented_in for rec in EQUATIONS.values())
    for mod in sol_dir.glob("*.py"):
        if mod.stem in ("__init__", "cylinder_dispatch"):
            continue
        assert f"solutions.{mod.stem}" in implemented, mod.stem
    rows = K.coverage({})
    assert {r["id"] for r in rows} == set(K.REGISTRY) and all(r["source_class"] == "standard-derived" for r in rows)
    assert all(set(s.records) <= set(EQUATIONS) for s in K.BUILTIN)  # every route maps to provenance records


def test_linked_evidence_names_existing_tests() -> None:
    """Every test id in the oracle evidence exists (a linked kind must be carried by a real test)."""
    import ast

    from critflaw.kb import evidence as ev

    root = Path(__file__).resolve().parents[1]
    names: dict[str, set[str]] = {}
    for rec, kinds in ev.ORACLE_TESTS.items():
        assert rec in EQUATIONS, rec
        for tests in kinds.values():
            for t in tests:
                path, _, fn = t.partition("::")
                if path not in names:
                    tree = ast.parse((root / path).read_text(encoding="utf-8"))
                    names[path] = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
                assert fn in names[path], t


def test_worked_values_of_the_standard_do_not_count_as_kind_a() -> None:
    """P7-CC-03: spec S1.5 removes the source's own worked examples as an acceptance basis."""
    from critflaw.kb import evidence as ev

    assert not any("a" in kinds for kinds in ev.ORACLE_TESTS.values())
    assert any("w" in kinds for kinds in ev.ORACLE_TESTS.values())
    for rid in ("toughness.charpy_t0", "fatigue.paris_law"):
        assert K.record_status(rid, ev.ORACLE_TESTS) != "cross-checked" or any(
            k in ev.ORACLE_TESTS[rid] for k in ("b",)
        )


def test_trace_records_the_kb_status_and_strict_refuses() -> None:
    """P-01 / P7-CC-01: every result trace records the knowledge-base status; strict=True refuses below
    cross-checked."""
    import critflaw as cf
    from test_plate_and_assessment import PLATE, STEEL  # the Phase 1 reference problem

    a = cf.Assessment(PLATE, cf.ThroughFlaw(10e-3), STEEL, cf.StressState(pm=150e6))
    res = a.run()
    deps = res.trace.dependencies()
    assert deps and all("kb_status" in d for d in deps)
    assert res.trace.below_cross_checked()  # e.g. k.general has only c/d evidence: below cross-checked
    with pytest.raises(ValueError, match="strict assessment refused"):
        a.run(strict=True)
    assert "knowledge base:" in res.trace.report()


def test_round2_inventory_boxes_bounds_and_suspect_trace(tmp_path: Path) -> None:
    """Phase 7 sign-off round 1 (red team A): P-02 every module with coefficient containers is inventoried and mapped
    to existing records; P-04 an evidence box restricts where a kind counts; P-05 non-finite bounds are refused;
    P7-CC-01 a suspect-cell read downgrades the tabulated record in the trace."""
    found = K.content_inventory()
    assert set(found) <= set(K.CONTENT_RECORDS), set(found) - set(K.CONTENT_RECORDS)
    assert all(r in EQUATIONS for recs in K.CONTENT_RECORDS.values() for r in recs)
    # a planted, unmapped table is detected
    pkg = tmp_path / "pkg"
    (pkg / "solutions").mkdir(parents=True)
    (pkg / "solutions" / "new.py").write_text("T = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)\n", encoding="utf-8")
    assert "solutions/new.py" in K.content_inventory(pkg) and "solutions/new.py" not in K.CONTENT_RECORDS
    # P-04
    ev = {"k.plate.surface": {"b": ("tests/x.py::t",)}}
    boxes = {"k.plate.surface": {"b": {"a_over_B": (0.2, 0.4)}}}
    assert K.record_status("k.plate.surface", ev, {"a_over_B": 0.3}, boxes) == "cross-checked"
    assert K.record_status("k.plate.surface", ev, {"a_over_B": 0.5}, boxes) != "cross-checked"
    assert K.record_status("k.plate.surface", ev, None, boxes) != "cross-checked"
    # P-05
    good = {
        "id": "user.k.b", "quantity": "K", "family": "f", "units": "SI", "normalisation": "n", "dimensions": "d",
        "front_position": "p", "loading": "l", "interpolation": "i", "validity": {"a": [0.0, float("inf")]},
    }  # fmt: skip
    p = tmp_path / "u.json"
    p.write_text(json.dumps(good).replace("Infinity", "1e999"), encoding="utf-8")
    with pytest.raises(ValueError, match="low < high"):
        K.load_user(p)
    # P7-CC-01
    from critflaw.assessment import Trace

    t = Trace(
        "e", ("k.cylinder.tables.axial_surface",), {}, ("tabulated_factor_interpolated_through_a_suspect_cell_x",)
    )
    assert t.kb_status("k.cylinder.tables.axial_surface") == "unverified"
    assert t.dependencies()[0]["kb_status"] == "unverified"


def test_unlinked_is_exactly_claimed_minus_linked() -> None:
    """N-01: the audit list matches the provenance claims and the linked evidence."""
    from critflaw.kb import evidence as ev

    for rid, rec in EQUATIONS.items():
        linked = {k for k, t in ev.ORACLE_TESTS.get(rid, {}).items() if t}
        assert tuple(sorted(set(rec.oracles) - linked)) == ev.UNLINKED.get(rid, ()), rid
    assert not any("a" in rec.oracles for rec in EQUATIONS.values())  # no primary-reference values compared yet


def test_every_unreferenced_record_is_listed_for_the_owner() -> None:
    """Project rule: an item with no primary reference is listed in docs/unsourced_items.md."""
    text = (Path(__file__).resolve().parents[1] / "docs" / "unsourced_items.md").read_text(encoding="utf-8")
    missing = [r for r, e in EQUATIONS.items() if e.primary_reference is None and f"`{r}`" not in text]
    assert not missing, missing
