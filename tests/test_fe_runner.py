"""Phase 4 gate (i), runner and cache, on the D1 regression deck (11 s in the container). Skipped when the
pinned image is not present (the parser tests cover the stored output)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from critflaw.fe import runner as R

ROOT = Path(__file__).resolve().parents[1]
TESTI = ROOT / "fe" / "external" / "warp3d" / "verification" / "testI"
IMAGE = "critflaw-warp3d:spike"


def _image_present() -> bool:
    if shutil.which("docker") is None:
        return False
    out = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, text=True)
    return out.returncode == 0


pytestmark = pytest.mark.skipif(not (_image_present() and TESTI.exists()), reason="WARP3D image or checkout absent")


def _inputs() -> dict[str, Path]:
    return {name: TESTI / name for name in ("warp3d_3.inp", "coords.inp", "incid.inp", "constraints.inp")}


def test_run_cache_hit_miss_and_truncation(tmp_path: Path) -> None:
    runner = R.WarpRunner(IMAGE, threads=4, memory="4g", timeout_seconds=600)
    digest = R.image_digest(IMAGE)
    inputs = _inputs()
    key = runner.key(inputs, "warp3d_3.inp")
    assert key == R.solve_key(R.inputs_digest(inputs), "warp3d_3.inp", 4, digest, runner.warp3d_exe, "4g", 600)
    cache = R.SolveCache(tmp_path / "cache")
    assert not cache.has(key)
    record, out = runner.run(tmp_path / "job", inputs, "warp3d_3.inp")  # the declaration is read from the deck
    assert record.status == "complete" and record.steps_completed == 100 and record.exit_code == 0
    cache.store(key, inputs, out, record, {"case": "testI.3", "parser_version": R.PARSER_VERSION})
    assert cache.has(key) and cache.record(key).status == "complete"
    parsed = cache.parsed(key)
    assert f"{parsed.domain('one', 100).average:.4E}" == "8.3700E-02"  # D1 reproduced from the stored raw output
    assert (cache.entry(key) / "parsed" / f"{R.PARSER_VERSION}-{R.REFERENCE_SET_VERSION}.json").exists()
    assert cache.packets(key) is None  # the shipped deck writes no packets file
    # append-only
    with pytest.raises(FileExistsError):
        cache.store(key, inputs, out, record, {})
    # any change of a generated input is a miss; a different thread count or image is a miss
    changed = dict(inputs)
    deck = tmp_path / "warp3d_3.inp"
    deck.write_text(
        inputs["warp3d_3.inp"].read_text(encoding="utf-8").replace("step 100", "step 100 "), encoding="utf-8"
    )
    changed["warp3d_3.inp"] = deck
    kw = dict(
        deck="warp3d_3.inp",
        threads=4,
        image_digest=digest,
        warp3d_exe=runner.warp3d_exe,
        memory="4g",
        timeout_seconds=600,
    )
    assert R.solve_key(R.inputs_digest(changed), **kw) != key
    for change in (
        dict(threads=2),
        dict(image_digest="sha256:other"),
        dict(deck="coords.inp"),
        dict(memory="8g"),
        dict(timeout_seconds=900),
    ):
        assert R.solve_key(R.inputs_digest(inputs), **{**kw, **change}) != key
    assert (cache.entry(key) / "err.txt.gz").exists()
    # a declared step count above what the deck computes is a truncated run, never complete
    assert R.classify(0, False, parsed, steps_declared=120) == "truncated"
    assert R.classify(0, False, parsed, steps_declared=50) == "failed"  # more steps than declared: a mismatch
    assert R.classify(1, False, parsed, 100) == "failed" and R.classify(None, True, parsed, 100) == "timeout"
    assert R.classify(0, False, None, 100) == "failed"  # an unparsable output still has a record
    with pytest.raises(FileExistsError):
        runner.run(tmp_path, inputs, "warp3d_3.inp")  # not a runner job directory


def test_timeout_kills_the_container_and_keeps_a_record(tmp_path: Path) -> None:
    runner = R.WarpRunner(IMAGE, threads=4, memory="4g", timeout_seconds=3.0)
    record, out = runner.run(tmp_path / "job", _inputs(), "warp3d_3.inp")
    assert record.status == "timeout" and "killed" in record.diagnostic and out.exists()
    name = record.diagnostic.split("container ")[1].split()[0]
    assert not R.container_exists(name)


def test_cache_stores_the_packets_file_beside_the_output(tmp_path: Path) -> None:
    import gzip

    job = tmp_path / "job"
    job.mkdir()
    (job / "out.txt").write_text("x", encoding="utf-8")
    (job / "packets.bpf").write_bytes(gzip.open(ROOT / "tests" / "data" / "fe" / "plastic_L1p_packets.bpf.gz").read())
    inputs = {"control.inp": job / "out.txt"}
    cache = R.SolveCache(tmp_path / "cache")
    cache.store("k", inputs, job / "out.txt", R.RunRecord("complete", 0, 1.0, 1, 1, True, (), (), ()), {})
    pk = cache.packets("k")
    assert pk is not None and pk == (job / "packets.bpf").read_bytes()


def test_cache_supersedes_a_non_complete_attempt_and_keeps_it_on_record(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    (job / "out.txt").write_text("x", encoding="utf-8")
    inputs = {"control.inp": job / "out.txt"}
    cache = R.SolveCache(tmp_path / "cache")
    failed = R.RunRecord("failed", 137, 26.0, 12, 0, False, (), (), (), "killed")
    cache.store("k", inputs, job / "out.txt", failed, {"parser_version": R.PARSER_VERSION})
    assert cache.record("k").status == "failed"
    done = R.RunRecord("complete", 0, 240.0, 12, 12, True, (), (), ())
    cache.store("k", inputs, job / "out.txt", done, {"parser_version": R.PARSER_VERSION})  # supersedes
    assert cache.record("k").status == "complete" and (cache.entry("k") / "attempts" / "1" / "run.json").exists()
    with pytest.raises(FileExistsError):  # a complete entry is append-only
        cache.store("k", inputs, job / "out.txt", done, {})


def test_cache_returns_a_stored_non_complete_attempt_unless_retry_is_explicit(tmp_path: Path) -> None:
    """R2-06: the design's no-silent-retry contract — a stored timeout/failure is returned as stored; a retry is
    the cache's explicit policy (``retry_incomplete``), the earlier attempt kept under attempts/."""
    from critflaw.fe.elastic_case import run_case as run_elastic  # noqa: F401 - the policy lives in the runners

    cache = R.SolveCache(tmp_path / "cache")
    assert cache.retry_incomplete is False
    job = tmp_path / "job"
    job.mkdir()
    (job / "out.txt").write_text("x", encoding="utf-8")
    inputs = {"control.inp": job / "out.txt"}
    cache.store("k", inputs, job / "out.txt", R.RunRecord("timeout", None, 3600.0, 12, 5, False, (), (), ()), {})
    assert cache.has("k") and cache.record("k").status == "timeout"  # returned as stored by _solve when not retrying
    retrying = R.SolveCache(tmp_path / "cache", retry_incomplete=True)
    assert retrying.retry_incomplete and retrying.record("k").status == "timeout"
