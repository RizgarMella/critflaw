"""Phase 4 gate (i), the parser: pinned on three stored WARP3D outputs (tests/data/fe): the SE(B) regression deck
testI.3 (D1, stored reference J 0.8370E-01 in the shipped driver.inp), the interaction-integral case test61, and
the spike's corner-crack run (D2, 96 steps x 17 domains)."""

from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import pytest

from critflaw.fe.parser import ParsedOutput, parse_output

DATA = Path(__file__).parent / "data" / "fe"


def _read(name: str) -> str:
    path = DATA / name
    if name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace").read()
    return path.read_text(encoding="utf-8", errors="replace")


def test_d1_se_b_regression_deck_reproduces_the_stored_reference_to_four_digits() -> None:
    p = parse_output(_read("testI_3_out.txt"))
    assert p.terminated_normally and p.steps_completed == 100 and not p.nonconverged_steps
    d = p.domain("one", 100)
    assert f"{d.average:.4E}" == "8.3700E-02"  # the shipped reference 0.8370E-01, 4 significant digits
    assert d.doubled_for_symmetry and d.rings.tolist() == list(range(1, len(d.rings) + 1))
    assert d.minimum <= d.total.min() and d.maximum >= d.total.max()
    assert d.path_independence() < 0.002
    # the cumulative load factors are read back from the output (displacement control: the constraints pattern)
    assert p.steps[-1].load_factors == {"constraints": pytest.approx(0.005)}
    assert p.fatal == () and p.warnings == () and len(p.benign) == 1  # the tied-node-set note is benign
    assert all(s.converged for s in p.steps) and p.steps[0].iterations >= 1


def test_interaction_integrals_give_signed_k_per_ring_with_the_auxiliary_field_label() -> None:
    p = parse_output(_read("test61_out.txt"))
    assert p.terminated_normally and p.steps_completed == 1
    kinds = {(d.domain_id, d.kind, d.mode) for d in p.domains}
    assert ("right", "J", None) in kinds and ("right", "II", "KI") in kinds and ("right", "II", "KII") in kinds
    ki = [d for d in p.domains if d.domain_id == "right" and d.mode == "KI"][0]
    assert ki.aux_field == "plane stress" and ki.k is not None and ki.k.shape == ki.total.shape
    assert ki.k[0] == pytest.approx(1.030) and ki.average == pytest.approx(1.034)
    # every K block and the T-stress blocks are parsed; the constraints multiplier is read back
    kiii = p.domain("right", 1, "II", mode="KIII")
    assert kiii.k is not None and kiii.k[0] == pytest.approx(-0.5125e-6) and kiii.aux_field is None
    t = p.domain("right", 1, "II", mode="T11")
    assert (
        set(t.values) == {"T11", "T33"}
        and t.values["T11"][0] == pytest.approx(-0.3882)
        and t.aux_field == "plane stress"
    )
    assert p.domain("right", 1, "II", mode="T13").values["T13"].shape == (13,)
    assert p.steps[0].load_factors == {"constraints": 1.0}
    kii = [d for d in p.domains if d.domain_id == "right" and d.mode == "KII"][0]
    assert kii.k is not None and kii.k[0] == pytest.approx(-0.6208)  # signed
    # the mixed-mode J from K is parsed too, and is consistent with the K values (plane stress: J = (KI^2 + KII^2)/E)
    assert len(p.mixed_mode_j) == 2 and p.mixed_mode_j[0][0] == "plane stress"
    j = p.mixed_mode_j[0][1]
    assert j[0] == pytest.approx(0.9655)
    # WARP3D's own warnings are collected, echoed input is not
    assert (
        len(p.warnings) == 3 and p.benign == () and p.fatal == ()
    )  # collapsed-element notes are warnings a caller declares
    from critflaw.fe.parser import run_qa

    # the warnings name their elements (continuation lines joined); only declared collapsed elements are expected
    assert all("element" in w for w in p.warnings)
    dup = {int(v) for v in re.findall(r"[0-9]+", p.warnings[0].split("elements:")[1])}
    assert run_qa(p, "constraints", (1.0,), collapsed_elements=frozenset(dup | {807})).passed
    assert not run_qa(p, "constraints", (1.0,)).passed  # undeclared, the Jacobian warnings are unexpected
    assert not run_qa(p, "constraints", (1.0,), collapsed_elements=frozenset({807})).passed  # others undeclared
    # M10: a Jacobian warning about an undeclared element is unexpected even when every other element is declared
    extra = _read("test61_out.txt").replace("of element    807 is non-positive", "of element    999 is non-positive", 1)
    assert not run_qa(parse_output(extra), "constraints", (1.0,), collapsed_elements=frozenset(dup | {807})).passed


def test_d2_spike_corner_crack_all_steps_and_domains() -> None:
    p = parse_output(_read("corner_at_hole_t4_out.txt.gz"))
    assert p.terminated_normally and p.steps_completed == 96 and len(p.domains) == 96 * 17
    assert p.boundary_notes == 96 * 17 and p.warnings == () and p.fatal == ()
    assert p.steps[0].load_factors == {"unit_rem": 2.0, "constraints": 1.0}
    # D2 (rings 2-8, max |J - mean| / mean): the 0.5 % band holds for the interior domains J5..J27 at every
    # step (achieved 0.25 %); the two domains at each end of the front (J1, J3 at the free surface; J29, J31,
    # J33 at the hole) reach up to 1.6 % late in the run and are reported, not banded (design D2 [impl])
    from critflaw.fe.parser import ParseError, check_domain_coverage

    # the expected domain set comes from the shipped deck (get_j_values: J1, J3, ..., J33 with rings 1-8), never
    # from the output being checked; a domain missing from every step must be caught
    ids = [f"J{k}" for k in range(1, 34, 2)]
    requested = {did: (1, 8) for did in ids}
    for step in range(1, 97):  # every requested domain present with rings 1..8 before any band is read
        check_domain_coverage(p, requested, step)
    without = ParsedOutput(
        p.steps, tuple(d for d in p.domains if d.domain_id != "J33"), p.mixed_mode_j, True, (), (), ()
    )
    with pytest.raises(ParseError, match="J33"):
        check_domain_coverage(without, requested, 1)
    worst = {did: max(d.path_independence() for d in p.domains if d.domain_id == did) for did in ids}
    interior = [f"J{k}" for k in range(5, 30, 2)]  # thirteen domains J5..J29
    assert max(worst[d] for d in interior) < 0.005
    assert max(worst.values()) < 0.017 and worst["J33"] > 0.01
    stored = dict(
        ln.split()
        for ln in (DATA / "d2_path_independence.txt")
        .read_text()
        .split(chr(10))
        if ln
    )
    assert {k: round(v, 5) for k, v in worst.items()} == {k: float(v) for k, v in stored.items()}  # the recorded table
    d1 = p.domain("J1", 1)
    assert d1.average == pytest.approx(2.996e-4) and d1.total[0] == pytest.approx(2.889e-4) and len(d1.rings) == 8
    assert np.all(d1.killed == 0) and d1.components.shape == (8, 8)
    # cumulative load factors per step are read back: the remote pattern grows monotonically
    lam = [s.load_factors["unit_rem"] for s in p.steps]
    assert lam[0] == 2.0 and lam[-1] == pytest.approx(66.038) and np.all(np.diff(lam) > 0)


def test_truncated_and_failed_outputs_are_distinguished() -> None:
    text = _read("testI_3_out.txt")
    cut = text[: text.index(">> solution for step:      50")]
    p = parse_output(cut)
    assert not p.terminated_normally and p.steps_completed == 49
    bad = text.replace(">> solution for step:      50 converged:", ">> solution for step:      50 did not converge:")
    p2 = parse_output(bad)
    assert p2.nonconverged_steps == (50,)  # a synthetic message; the real one is pinned below


def test_a_non_convergent_step_is_a_truncated_run(tmp_path: Path) -> None:
    # one element driven to 17 sigma_0 with two Newton iterations and no adaptivity: WARP3D stops at step 3
    from critflaw.fe.runner import classify

    p = parse_output(_read("one_element_nonconvergent_out.txt"))
    assert p.nonconverged_steps == (3,) and p.steps_completed == 2 and not p.terminated_normally
    assert any("iteration limit exceeded" in w for w in p.fatal)
    assert classify(0, False, p, steps_declared=5) == "truncated"


def test_strictness_extents_and_run_qa() -> None:
    from critflaw.fe.parser import ParseError, run_qa

    text = _read("spike_step1_extents_out.txt")
    p = parse_output(text)
    d = p.domain("J1", 1)
    assert d.extents is not None and d.extents.shape == (8, 6) and np.all(d.extents[:, 3:] >= d.extents[:, :3])
    assert np.all(np.diff(d.extents[:, 0]) < 0)  # rings grow outward: x-min decreases ring by ring
    qa = run_qa(p, "unit_rem", (2.0,))
    assert qa.passed and qa.read_back_matches_declared
    assert not run_qa(p, "unit_rem", (2.5,)).read_back_matches_declared
    assert not run_qa(p, "unit_rem", (2.0, 4.0)).all_declared_converged
    # the read-back comparison allows the print's five significant digits (a declared 1.61051 prints 1.6105)
    assert run_qa(
        parse_output(text.replace("unit_rem       2.0000", "unit_rem       1.6105")), "unit_rem", (1.61051,)
    ).read_back_matches_declared
    assert not run_qa(
        parse_output(text.replace("unit_rem       2.0000", "unit_rem       1.6105")), "unit_rem", (1.62,)
    ).read_back_matches_declared
    # an overflowed table field, a non-contiguous ring, a missing loading line: never silently dropped
    row = re.search(r"^\s*3\s+0\.\d+E[-+]\d+", text, re.M)
    assert row is not None
    with pytest.raises(ParseError, match="unreadable"):
        parse_output(text[: row.start()] + "     3    ************" + text[row.end() :])
    with pytest.raises(ParseError, match="contiguous"):
        parse_output(text.replace("     3     0.", "     9     0.", 1))
    with pytest.raises(ParseError, match="loading"):
        parse_output(text.replace("loading: tension      step:     1", "", 1))
    # steps_completed is the contiguous count from 1; a gap ends it
    gap = _read("testI_3_out.txt").replace(
        ">> solution for step:      50 converged:", ">> solution for step:     50x converged:"
    )
    assert parse_output(gap).steps_completed == 49


def test_reactions_are_parsed_per_step() -> None:
    text = (
        "        step no.       2                        loading load"
        + chr(10)
        + "         nodal reaction forces"
        + chr(10)
        + chr(10)
        + "        node               Force X             Force Y             Force Z"
        + chr(10)
        + "            1         0.54394E+03         0.32490E-08         0.96251E-01"
        + chr(10)
        + "            4        -0.10268E+04         0.37808E-11         0.89567E-01"
        + chr(10)
        + chr(10)
        + "     Totals:     -0.48286000E+03      0.32528E-08      0.18582E+00"
        + chr(10)
    )
    p = parse_output(text)
    assert len(p.reactions) == 1 and p.reactions[0].step == 2 and p.reactions[0].nodes.tolist() == [1, 4]
    assert p.reactions[0].forces[1, 0] == pytest.approx(-1026.8) and p.reactions[0].totals[0] == pytest.approx(-482.86)
    from critflaw.fe.parser import ParseError

    with pytest.raises(ParseError, match="overflowed"):
        parse_output(text.replace("0.54394E+03", "************"))


def test_reaction_blocks_are_anchored_to_their_own_step_header() -> None:
    # NF-01: a stress block with its own "step no." header between two reaction blocks must not relabel the second
    def block(step: int, fx: float) -> str:
        return (
            f"        step no.       {step}                        loading load"
            + chr(10)
            + "         nodal reaction forces"
            + chr(10)
            + chr(10)
            + "        node               Force X             Force Y             Force Z"
            + chr(10)
            + f"            1         {fx:.5E}         0.00000E+00         0.00000E+00"
            + chr(10)
            + chr(10)
            + f"     Totals:     {fx:.8E}      0.00000000E+00      0.00000000E+00"
            + chr(10)
        )

    stresses = (
        "        step no.       1                        loading load"
        + chr(10)
        + "        stresses"
        + chr(10)
        + chr(10)
        + "      elem  g.p.         sigma_xx            sigma_yy            sigma_zz"
        + chr(10)
        + "          1      1       1.000000            0.000000            0.000000"
        + chr(10)
        + chr(10)
    )
    p = parse_output(block(1, 10.0) + stresses + block(2, 20.0))
    assert [r.step for r in p.reactions] == [1, 2] and p.reactions[1].totals[0] == pytest.approx(20.0)
    from critflaw.fe.parser import ParseError

    with pytest.raises(ParseError, match="totals"):
        parse_output(block(1, 10.0).replace("1.00000000E+01", "************"))


def test_equilibrium_qa_detects_missing_and_unbalanced_reactions() -> None:
    from critflaw.fe import mesh as M
    from critflaw.fe.parser import ReactionRecord, equilibrium_qa

    m = M.structured_block(2.0, 1.0, 1.0, 1, 1, 1)
    nodes = M.nodes_on_plane(m, 0, 0.0)
    f_app, m_app = np.array([100.0, 0.0, 0.0]), np.array([0.0, 0.0, 5.0])
    origin = np.array([0.0, 0.5, 0.5])  # the face centroid: the uniform part of the reactions has no moment about it

    def record(step: int, scale: float, drop: int = 0) -> ReactionRecord:
        # reactions that balance scale x the applied resultants exactly: a force -scale*f/n at each node plus a
        # couple distributed over two nodes for the moment
        n = len(nodes) - drop
        forces = np.tile(-scale * f_app / n, (n, 1))
        # add a y-force couple at two nodes with x = 0: moment_z = x fy - y fx; use fx on nodes at y = 0 and y = 1
        y = m.coords[nodes[:n] - 1, 1]
        top, bottom = int(np.argmax(y)), int(np.argmin(y))
        dy = y[top] - y[bottom]
        forces[top, 0] += scale * m_app[2] / dy  # moment_z = -sum(y fx): the reactions must sum to -scale m_app
        forces[bottom, 0] -= scale * m_app[2] / dy
        return ReactionRecord(step, nodes[:n].copy(), forces, forces.sum(axis=0))

    class P:  # a minimal parsed-output stand-in
        def __init__(self, reactions: tuple[ReactionRecord, ...]) -> None:
            self.reactions = reactions

    good = equilibrium_qa(P((record(1, 0.5), record(2, 1.0))), m.coords, nodes, f_app, m_app, (0.5, 1.0), origin)  # type: ignore[arg-type]
    assert all(q.passed for q in good) and good[1].force_residual < 1e-12
    missing = equilibrium_qa(P((record(1, 0.5, drop=1),)), m.coords, nodes, f_app, m_app, (0.5,), origin)  # type: ignore[arg-type]
    assert not missing[0].nodes_complete and not missing[0].passed
    unbalanced = equilibrium_qa(P((record(1, 0.9),)), m.coords, nodes, f_app, m_app, (1.0,), origin)  # type: ignore[arg-type]
    assert unbalanced[0].force_residual == pytest.approx(0.1) and not unbalanced[0].passed
    # a wrong moment alone fails (M6): the couple halved
    r = record(1, 1.0)
    y = m.coords[r.nodes - 1, 1]
    top, bottom = int(np.argmax(y)), int(np.argmin(y))
    f = r.forces.copy()
    f[top, 0] -= 2.5
    f[bottom, 0] += 2.5
    half = equilibrium_qa(
        P((ReactionRecord(1, r.nodes, f, f.sum(axis=0)),)), m.coords, nodes, f_app, m_app, (1.0,), origin
    )  # type: ignore[arg-type]
    assert half[0].force_residual < 1e-12 and half[0].moment_residual > 0.01 and not half[0].passed
    # records are matched by step id, never by position; pure tension about the centroid has a zero applied moment
    from critflaw.fe.parser import ParseError

    with pytest.raises(ParseError, match="do not match"):
        equilibrium_qa(P((record(2, 1.0),)), m.coords, nodes, f_app, m_app, (1.0,), origin)  # type: ignore[arg-type]
    with pytest.raises(ParseError, match="two reaction blocks"):
        equilibrium_qa(P((record(1, 1.0), record(1, 1.0))), m.coords, nodes, f_app, m_app, (1.0, 1.0), origin)  # type: ignore[arg-type]
    swapped = equilibrium_qa(P((record(2, 1.0), record(1, 0.5))), m.coords, nodes, f_app, m_app, (0.5, 1.0), origin)  # type: ignore[arg-type]
    assert all(q.passed for q in swapped)
    n = len(nodes)
    tension = np.tile(-f_app / n, (n, 1)) + np.random.default_rng(0).normal(0.0, 1e-9, (n, 3))
    pure = equilibrium_qa(
        P((ReactionRecord(1, nodes.copy(), tension, tension.sum(axis=0)),)),
        m.coords,
        nodes,
        f_app,
        np.zeros(3),
        (1.0,),
        origin,
    )  # type: ignore[arg-type]
    assert pure[0].passed and pure[0].moment_residual < 1e-9


def test_gauss_point_tables_and_domain_coverage() -> None:
    from critflaw.fe.parser import ParseError, check_domain_coverage, parse_gauss_point_tables

    text = _read("one_element_d6_out.txt")
    tables = parse_gauss_point_tables(text)
    strains = [t for t in tables if t.quantity == "strains"]
    stresses = [t for t in tables if t.quantity == "stresses"]
    assert len(strains) == 11 and len(stresses) == 11 and strains[0].step == 1
    assert strains[0].values[(1, 1)][0] == pytest.approx(0.000857) and len(strains[0].values) == 8
    assert [t.step for t in strains] == list(range(1, 12)) and [t.step for t in stresses] == list(range(1, 12))
    # each step's sigma_xx equals that step's load factor (the tables are attributed to their own step)
    p6 = parse_output(text)
    for t in stresses:
        assert t.values[(1, 1)][0] == pytest.approx(p6.steps[t.step - 1].load_factors["unit_ten"], rel=1e-6)
    assert stresses[-1].values[(1, 8)][0] == pytest.approx(426.0, rel=1e-6)  # sigma_xx at the last step
    with pytest.raises(ParseError, match="overflowed"):
        parse_gauss_point_tables(text.replace("0.000857", "********", 1))
    p = parse_output(_read("spike_step1_extents_out.txt"))
    check_domain_coverage(p, {"J1": (1, 8), "J3": (1, 8), "J5": (1, 8)}, 1)
    with pytest.raises(ParseError, match="differ from"):
        check_domain_coverage(p, {"J1": (1, 10)}, 1)
    with pytest.raises(ParseError):
        check_domain_coverage(p, {"J7": (1, 8)}, 1)


def test_pure_pins_of_the_name_truncation_and_the_exact_step_set() -> None:
    # NF-10 / M1, M2: synthetic outputs, no image needed
    from critflaw.fe.parser import ParseError, run_qa
    from critflaw.fe.runner import classify

    def step(k: int, factor: float) -> str:
        return (
            f" >> solution for step:       {k} converged:  2 iters.   model time:  0.100000E+11"
            + chr(10)
            + f" >> total applied load pattern factors through step:       {k}"
            + chr(10)
            + f"      > unit_loa      {factor:.5f}      constraints    {float(k):.4f}"
            + chr(10)
            + chr(10)
        )

    tail = ">> total job wall time (secs):        1.00" + chr(10)
    three = parse_output(step(1, 0.5) + step(2, 1.0) + step(3, 2.0) + tail)
    assert three.steps_completed == 3 and three.steps[1].load_factors == {"unit_loa": 1.0, "constraints": 2.0}
    assert run_qa(three, "unit_load_pattern", (0.5, 1.0, 2.0)).read_back_matches_declared  # eight-character truncation
    assert not run_qa(three, "other_pattern", (0.5, 1.0, 2.0)).read_back_matches_declared
    assert classify(0, False, three, 3) == "complete"
    assert classify(0, False, three, 2) == "failed"  # a superset of the declared steps is never complete
    assert classify(0, False, three, 4) == "truncated"
    gap = parse_output(step(1, 0.5) + step(3, 2.0) + tail)
    assert gap.steps_completed == 1 and classify(0, False, gap, 3) == "truncated"
    with pytest.raises(ParseError, match="no applied-load-factor block"):
        parse_output(" >> solution for step:       1 converged:  2 iters.   model time:  0.1E+11" + chr(10) + tail)


def test_truncated_warning_lists_are_never_exempted() -> None:
    from critflaw.fe.parser import MAX_MESSAGE_LINES, expected_collapsed_warning

    head = (
        " >> warning: there are duplicate entries in the incidences for"
        + chr(10)
        + "            the following elements:"
        + chr(10)
    )
    body = "".join(f"      {k}" + chr(10) for k in range(1, MAX_MESSAGE_LINES + 5)) + "      999" + chr(10)
    p = parse_output(head + body + ">> total job wall time (secs):        1.00" + chr(10))
    assert len(p.warnings) == 1 and "incomplete" in p.warnings[0]
    assert not expected_collapsed_warning(p.warnings[0], frozenset(range(1, MAX_MESSAGE_LINES + 5)))
    short = parse_output(head + "      1      2" + chr(10) + ">> total job wall time (secs):        1.00" + chr(10))
    assert expected_collapsed_warning(short.warnings[0], frozenset({1, 2}))
    assert not expected_collapsed_warning(short.warnings[0], frozenset({1}))


def test_image_resolution_failure_yields_a_failed_record(tmp_path: Path) -> None:
    from critflaw.fe import runner as R

    deck = tmp_path / "d.inp"
    deck.write_text(" compute displacements for loading x step 1" + chr(10), encoding="utf-8")
    runner = R.WarpRunner(image="critflaw-no-such-image:none")
    record, out = runner.run(tmp_path / "job", {"d.inp": deck}, "d.inp")  # the job directory does not exist yet
    assert record.status == "failed" and "image resolution failed" in record.diagnostic and out.exists()
    (tmp_path / "plain").mkdir()
    with pytest.raises(FileExistsError):
        runner.run(tmp_path / "plain", {"d.inp": deck}, "d.inp")  # an unmarked directory is never touched
    assert list((tmp_path / "plain").iterdir()) == []


# ----------------------------------------------------------------------------------------------------------------
# binary packets (manual 2.12 / Appendix F.19): full-precision J


def _rec(payload: bytes) -> bytes:
    n = len(payload).to_bytes(4, "little", signed=True)
    return n + payload + n


def _j_packet(domain: str, step: int, totals: list[float], comps: list[list[float]] | None = None) -> bytes:
    import struct

    n = len(totals)
    head = _rec(struct.pack("<4i", 17, 1 + n + (1 if n > 1 else 0) + n, step, 0))
    first = _rec(struct.pack("<i", n) + domain.ljust(24).encode() + b"plate_su" + b"load    " + struct.pack("<i", step))
    rings = b""
    for r in range(n):
        c = comps[r] if comps is not None else [0.0] * 8
        rings += _rec(struct.pack("<i", r + 1) + struct.pack("<9d", *c, totals[r]) + struct.pack("<i", 0))
    stats = _rec(struct.pack("<3d", sum(totals) / n, min(totals), max(totals))) if n > 1 else b""
    ks = b"".join(_rec(struct.pack("<i", r + 1) + struct.pack("<5d", 1.0, 2.0, 3.0, 4.0, 5.0)) for r in range(n))
    return head + first + rings + stats + ks


def test_j_packets_are_read_at_full_precision_and_other_packet_types_skipped() -> None:
    import struct

    from critflaw.fe.parser import ParseError, read_j_packets

    other = _rec(struct.pack("<4i", 24, 2, 1, 0)) + _rec(b"unit_loa" + struct.pack("<d", 14.2)) + _rec(b"x" * 20)
    totals = [0.031468465570044271, 0.05391416, 0.10212889123456789]
    data = other + _j_packet("S0", 1, totals, [[0.1 * k] * 8 for k in range(3)]) + _j_packet("S1", 2, [0.5])
    pk = read_j_packets(data)
    assert [(q.domain_id, q.step, q.loading) for q in pk] == [("S0", 1, "load"), ("S1", 2, "load")]
    assert pk[0].total.tolist() == totals and pk[0].rings.tolist() == [1, 2, 3]  # exact doubles, no rounding
    assert pk[0].components[2].tolist() == [0.2] * 8 and pk[0].minimum == totals[0] and pk[0].maximum == totals[2]
    assert pk[1].average == pk[1].minimum == pk[1].maximum == 0.5
    with pytest.raises(ParseError, match="truncated"):
        read_j_packets(data[:-7])
    bad = data[: len(other)] + _rec(struct.pack("<4i", 17, 9, 1, 0))  # declares records the file lacks
    with pytest.raises(ParseError, match="declares"):
        read_j_packets(bad)
    with pytest.raises(ParseError, match="disagree"):
        read_j_packets(data[:-4] + b"\x07\x00\x00\x00")


def test_stored_packets_match_the_printed_tables_to_four_digits() -> None:
    import gzip

    from critflaw.fe.parser import read_j_packets

    pk = read_j_packets(gzip.open(DATA / "plastic_L1p_packets.bpf.gz").read())
    parsed = parse_output(gzip.open(DATA / "plastic_L1p_out.txt.gz", "rt", encoding="utf-8").read())
    assert len(pk) == 13 * 12  # thirteen domains, twelve steps, one packet each
    worst = 0.0
    for q in pk:
        d = parsed.domain(q.domain_id, q.step, "J")
        assert np.array_equal(d.rings, q.rings)
        worst = max(worst, float(np.max(np.abs(d.total / q.total - 1.0))))
        assert abs(d.average / q.average - 1.0) < 6e-4
    assert 1e-5 < worst < 5e-4  # the print carries four significant digits; the packets more


def test_an_adaptively_recovered_step_is_converged_not_truncated() -> None:
    """WARP3D's adaptive driver (manual 2.10.6): a step whose Newton iterations hit the limit is subdivided; each
    sub-increment reports 'step: N.250 converged' and the whole step then 'step: N converged'. Such a step is
    complete (recorded as adaptive, its iterations the total), its limit message a recorded warning, never fatal;
    a step without the final whole-step convergence stays nonconverged."""
    from critflaw.fe.parser import run_qa
    from critflaw.fe.runner import classify

    def block(step: int, iters: int, factor: float) -> str:
        return (
            f" >> solution for step:       {step} converged:  {iters} iters.   model time:  0.1E+10\n"
            f" >> total applied load pattern factors through step:       {step}\n"
            f"      > unit_ten       {factor:.2f}      constraints    1.0000    \n\n"
        )

    recovered = (
        block(1, 2, 1.0)
        + " >> iteration limit exceeded for current step:      2\n"
        + "    (or adaptive sub-increment) or the solution appears to be diverging....\n\n"
        + "      adaptive solution driver for step:       2\n"
        + " >> solution for step:       2.250 converged: 12 iters.   model time:  0.1E+10\n"
        + " >> solution for step:       2.500 converged:  2 iters.   model time:  0.1E+10\n"
        + " >> solution for step:       2.750 converged:  3 iters.   model time:  0.1E+10\n"
        + block(2, 4, 2.0)
        + block(3, 3, 3.0)
        + ">> total job wall time (secs):  10.0\n"
    )
    p = parse_output(recovered)
    assert p.adaptive_steps == (2,) and p.nonconverged_steps == () and p.steps_completed == 3
    s2 = next(s for s in p.steps if s.step == 2)
    assert s2.converged and s2.adaptive and s2.iterations == 4 + 12 + 2 + 3
    assert p.fatal == () and any("iteration limit exceeded" in w for w in p.warnings)
    assert classify(0, False, p, steps_declared=3) == "complete"
    qa = run_qa(p, "unit_ten", (1.0, 2.0, 3.0))
    assert qa.passed and qa.unexpected_warnings == ()
    # the same limit hit without the whole-step convergence: nonconverged, fatal, truncated
    failed = block(1, 2, 1.0) + " >> iteration limit exceeded for current step:      2\n" + ">>>>> analysis terminated.\n"
    q = parse_output(failed)
    assert q.nonconverged_steps == (2,) and q.adaptive_steps == () and any("iteration limit" in f for f in q.fatal)
    assert classify(0, False, q, steps_declared=3) == "truncated"
