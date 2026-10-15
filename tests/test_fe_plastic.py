"""Phase 4 gate (ii), the elastic-plastic path (pure: pinned on the stored level-1 outputs; the solve itself is
the live benchmark): the schedule's strain cap, the J_e provenance conditions (a) and (b), path independence
per step, the Option 3 curve Kr(Lr) monotone and bounded, and Lr from the declared limit load."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

from critflaw.fe.parser import parse_output
from critflaw.fe.plastic_case import LR_CEILING, STRAIN_CAP_REFERENCE, PlasticCase, evaluate, write_inputs
from critflaw.fe.runner import RunRecord

DATA = Path(__file__).parent / "data" / "fe"
CASE = PlasticCase(
    a=8e-3, c=20e-3, B=20e-3, W=200e-3, L_far=100e-3, E=207e9, nu=0.3, sigma_0=355e6, n=10.0, beta=0.0,
    level=(8, 12, 16, 4, 4), n_outer=5, far_first=0.0, packets=False,  # the stored run: first level, far field, printed J
)  # fmt: skip


def _read(name: str) -> str:
    return gzip.open(DATA / name, "rt", encoding="utf-8", errors="replace").read()


def test_schedule_is_capped_by_the_reference_strain_and_refined_above_lr_0_8() -> None:
    lr = np.asarray(CASE.schedule()) * CASE.lr_per_unit()
    eps0 = CASE.sigma_0 / CASE.E
    assert lr[-1] == pytest.approx(min(LR_CEILING, (STRAIN_CAP_REFERENCE / eps0) ** (1 / CASE.n)))
    assert eps0 * lr[-1] ** CASE.n == pytest.approx(STRAIN_CAP_REFERENCE, rel=1e-9)
    assert lr[0] == pytest.approx(0.05) and np.all(np.diff(lr) > 0)
    assert np.sum(lr > 0.8) >= 5  # the finer spacing above Lr = 0.8
    low = PlasticCase(**{**CASE.__dict__, "n": 3.0})
    assert low.lr_end() == LR_CEILING  # a low exponent reaches the Lr ceiling first


def test_option3_curve_from_the_stored_level1_run(tmp_path: Path) -> None:
    cb, _inputs, domains = write_inputs(CASE, tmp_path)
    text = _read("plastic_L1_out.txt.gz")
    parsed = parse_output(text)
    parsed_e = parse_output(_read("plastic_L1_elastic_out.txt.gz"))
    res = evaluate(CASE, cb, parsed, text, domains, RunRecord("complete", 0, 61.0, 12, 12, True, (), (), ()), parsed_e)
    assert res.plastic_zone_in_ring_1 and res.max_vm_outside_ring_1_over_sigma_0 < 0.95  # condition (a)
    assert np.nanmax(np.abs(res.j1_vs_elastic[1:])) < 5e-3  # condition (b): same untied mesh, 0.5 % band
    assert np.allclose(res.lr, np.asarray(CASE.schedule()) * CASE.lr_per_unit(), rtol=1e-4)  # read back
    inner = res.j_spread[:, 1:]
    assert inner[:-1].max() < 0.02 and inner.max() < 0.025  # the 2 % band holds to the penultimate step
    kr_deep = res.kr[:, -1]
    # non-increasing to the printed precision of J (four digits: 3e-5 at the first step)
    assert kr_deep[0] == pytest.approx(1.0) and np.all(np.diff(kr_deep) <= 1e-4) and 0.5 < kr_deep[-1] < 0.6
    # from phi_s inward Kr never exceeds 1 (print precision at step 1); the stations nearer the surface than
    # phi_s show Kr up to 1.04 at mid loads (J there grows slower than lambda^2 as the deeper front yields:
    # recorded, outside every criterion, as the design excludes the vertex region)
    s = CASE.level[3]
    assert np.all(res.kr[:, s:] <= 1.0 + 1e-4) and np.all(res.kr[:, 1:] > 0.4) and np.all(res.kr[:, 1:s] < 1.1)
    # the Option 3 curve lies below the Phase 2 Option 2 curve of the same material at the same Lr (the FE
    # includes the geometry's own constraint; the design compares, never equates)
    from critflaw.fad.option2 import Option2Curve

    curve = Option2Curve.from_warp_law(CASE.law, 490e6)
    kr_o2 = curve(res.lr[:-1])
    inside = res.lr[:-1] < float(np.asarray(curve.lr_max))  # beyond Lr,max the Option 2 line is the cut-off (0)
    assert np.all(np.isfinite(kr_o2)) and np.all(kr_o2[inside] > 0.0)
    # the FE curve of this geometry lies above the Option 2 line (Option 2 is the conservative general curve)
    assert np.all(res.kr[:-1, -1][inside] >= kr_o2[inside] - 1e-3)


def test_gauss_point_coverage_demands_exact_identities() -> None:
    """R2-01: eight rows are not enough — the eight distinct Gauss points of every expected element are, once each;
    a duplicate across tables, an unexpected point number, a missing point or a non-finite value fails."""
    from critflaw.fe.parser import GaussPointTable, ParseError
    from critflaw.fe.plastic_case import _gauss_rows

    def table(step: int, points: range, elements: tuple[int, ...], value: float = 1.0) -> GaussPointTable:
        return GaussPointTable(step, "strains", {(e, g): np.full(8, value) for e in elements for g in points})

    full = [table(1, range(1, 9), (10, 11))]
    assert len(_gauss_rows(full, (10, 11), 6, "t")) == 16
    halves_twice = [table(1, range(1, 5), (10, 11)), table(1, range(1, 5), (10, 11))]  # 8 rows per element, 4 points
    with pytest.raises(ParseError, match="appears twice"):
        _gauss_rows(halves_twice, (10, 11), 6, "t")
    with pytest.raises(ParseError, match="missing"):
        _gauss_rows([table(1, range(1, 5), (10, 11))], (10, 11), 6, "t")
    with pytest.raises(ParseError, match="unexpected Gauss point"):
        _gauss_rows([table(1, range(1, 10), (10,))], (10,), 6, "t")
    with pytest.raises(ParseError, match="non-finite"):
        _gauss_rows([table(1, range(1, 9), (10,), float("nan"))], (10,), 6, "t")
    assert len(_gauss_rows(full + [table(1, range(1, 9), (99,))], (10, 11), 6, "t")) == 16  # other elements ignored


def test_requested_packets_must_exist(tmp_path: Path) -> None:
    """R2-04: a case with packets on is never evaluated from the four-digit print."""
    from critflaw.fe.parser import ParseError

    case = PlasticCase(**{**CASE.__dict__, "packets": True})
    cb, _inputs, domains = write_inputs(case, tmp_path)
    text = _read("plastic_L1_out.txt.gz")
    with pytest.raises(ParseError, match="packets"):
        evaluate(case, cb, parse_output(text), text, domains, RunRecord("complete", 0, 61.0, 12, 12, True, (), (), ()))


CASE_Q = PlasticCase(  # the D7 level-1 run at the mid point on the current code (packets, reactions, untied tip)
    a=8e-3, c=20e-3, B=20e-3, W=200e-3, L_far=400e-3, E=207e9, nu=0.3, sigma_0=355e6, n=10.0, beta=0.0,
)  # fmt: skip


def _evaluate_q(tmp_path: Path, text: str | None = None, packets: bytes | None = None):  # type: ignore[no-untyped-def]
    cb, _inputs, domains = write_inputs(CASE_Q, tmp_path)
    text = text if text is not None else _read("plastic_L1q_out.txt.gz")
    packets = packets if packets is not None else gzip.open(DATA / "plastic_L1q_packets.bpf.gz").read()
    rec = RunRecord("complete", 0, 94.0, 12, 12, True, (), (), ())
    return evaluate(
        CASE_Q, cb, parse_output(text), text, domains, rec, parse_output(_read("plastic_L1q_elastic_out.txt.gz")),
        packets=packets,
    )  # fmt: skip


def test_current_code_level1_run_pins_packets_equilibrium_and_conditions(tmp_path: Path) -> None:
    """G2-C-R2-06: the pure pins on an output of the current mesh and decks (the D7 level-1 run)."""
    res = _evaluate_q(tmp_path)
    assert res.plastic_zone_in_ring_1 and res.max_vm_outside_ring_1_over_sigma_0 < 0.4  # condition (a)
    assert np.nanmax(np.abs(res.j1_vs_elastic[1:])) < 5e-4  # condition (b), same untied mesh
    assert np.all(res.j_spread[:, 1:] < 0.006)  # corrected tip: ring spread <= 0.6 % at every step
    assert len(res.equilibrium) == 12 and all(q.passed and q.nodes_complete for q in res.equilibrium)
    assert len(res.equilibrium_elastic) == 1 and res.equilibrium_elastic[0].passed
    assert np.allclose(res.lam, CASE_Q.schedule(), rtol=0, atol=0)  # the exact declared factors (G2-C-R2-05)
    assert res.kr[0, -1] == pytest.approx(1.0) and 0.5 < res.kr[-1, -1] < 0.6
    assert res.j[5, -1] == pytest.approx(6768.18, rel=1e-6)  # the D7 record's J_1 at Lr 0.8, deepest point


def test_failure_modes_of_the_evaluation_are_parse_errors(tmp_path: Path) -> None:
    """G2-C-R2-01 / -09: a missing domain is a ParseError (MissingRecord), a NaN packet fails the print check."""
    import struct

    from critflaw.fe.parser import MissingRecord, ParseError

    text = _read("plastic_L1q_out.txt.gz")
    last = text.rfind("domain-integral post-processing")  # remove the last domain block: its record is missing
    nxt = text.find(">>", last + 40)
    with pytest.raises(MissingRecord):
        _evaluate_q(tmp_path / "a", text=text[:last] + text[nxt:])
    assert issubclass(MissingRecord, ParseError) and issubclass(MissingRecord, KeyError)
    pk = bytearray(gzip.open(DATA / "plastic_L1q_packets.bpf.gz").read())
    # walk the Fortran records to the first type-17 header; its first ring record follows the domain record
    pos, injected = 0, False
    while pos < len(pk) and not injected:
        n = struct.unpack_from("<i", pk, pos)[0]
        if n == 16 and struct.unpack_from("<i", pk, pos + 4)[0] == 17:
            ring = pos + 4 + 16 + 4 + (4 + 48 + 4)  # past the header and the domain record
            total = ring + 4 + 4 + 8 * 8  # the ring record's marker, ring number, dm1..dm8, then the total
            pk[total : total + 8] = struct.pack("<d", float("nan"))
            injected = True
        pos += 4 + n + 4
    assert injected  # the fixture holds J packets (its first record is another packet type)
    with pytest.raises(ParseError, match="disagrees|non-finite"):
        _evaluate_q(tmp_path / "b", packets=bytes(pk))


def test_equilibrium_moment_scale_is_the_sum_of_moment_magnitudes() -> None:
    """G2-C-R2-04: the moment residual is scaled by sum |r_i x R_i|, not sum |r_i| |R_i| (8x looser here)."""
    from critflaw.fe.parser import ParsedOutput, ReactionRecord, StepRecord, equilibrium_qa

    coords = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
    forces = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])  # node 3's force along its long arm
    rx = ReactionRecord(1, np.array([1, 2, 3]), forces, forces.sum(axis=0))
    parsed = ParsedOutput((StepRecord(1, True, 1, {"u": 1.0}),), (), (), True, (), (), (), (rx,))
    q = equilibrium_qa(parsed, coords, np.array([1, 2, 3]), np.array([0.0, 0.0, 3.0]), np.array([0.0, 2e-3, 0.0]),
                       (1.0,), np.zeros(3))  # fmt: skip
    # sum |r_i x R_i| = 1 + 1 + 0 = 2 -> residual 1e-3; the old scale sum |r_i| |R_i| = 12 gave 1.7e-4 (fail-open)
    assert q[0].moment_residual == pytest.approx(1e-3) and not q[0].passed
