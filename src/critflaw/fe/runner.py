"""WARP3D runner and the two-layer cache (theory manual, section 6.1.6).

Runner: ``docker run`` of the pinned image with the generated inputs mounted, one job at a time on a fixed
thread count, a memory ceiling and a wall-clock timeout; the container is named so that a timeout kills it
(``docker kill``) before the output is read. Status: ``complete`` (exit 0, normal termination, the converged
steps are exactly 1..N declared, no non-converged step), ``truncated`` (a non-converged step, or fewer
contiguous converged steps than declared), ``failed`` (non-zero exit, no normal termination, more steps than
declared, or an output the parser cannot read — the diagnostic is kept), ``timeout``. Every outcome yields a
durable record with the raw stdout and stderr. Only ``complete`` runs may become ``fe-verified``.

Cache: the **solve key** is the SHA-256 of (the digests of every generated solver input, the entry deck name,
the executable path, the thread count, the memory ceiling, the timeout, the executable identity = image
digest): a stored run and its status are a pure function of the key; a rebuilt image or a larger budget is a
new key. Metadata (the case record, versions, template hash) is stored with the entry, not hashed into the
key. The **parsed layer** is keyed by (solve key, parser version, reference-set version) and is recomputed
from the stored raw output without re-solving. Append-only.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from critflaw.fe.deck import declared_steps
from critflaw.fe.parser import ParsedOutput, ParseError, parse_output

PARSER_VERSION = "3"  # 3: adaptive-driver recovered steps are complete
RUNNER_VERSION = "2"
REFERENCE_SET_VERSION = "1"
JOB_MARKER = ".critflaw_job"


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inputs_digest(inputs: dict[str, Path]) -> dict[str, str]:
    """``{published name: sha256}`` of every generated solver input (deck, coordinates, incidences, ...)."""
    return {name: file_digest(path) for name, path in sorted(inputs.items())}


def solve_key(
    input_digests: dict[str, str],
    deck: str,
    threads: int,
    image_digest: str,
    warp3d_exe: str = "/opt/warp3d/run_linux/warp3d.exe",
    memory: str = "6g",
    timeout_seconds: float = 3600.0,
) -> str:
    payload = json.dumps(
        {
            "inputs": input_digests,
            "deck": deck,
            "exe": warp3d_exe,
            "threads": threads,
            "memory": memory,
            "timeout": timeout_seconds,
            "image": image_digest,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def image_digest(image: str) -> str:
    """The Docker image id (content digest of the local image), the executable identity."""
    out = subprocess.run(["docker", "image", "inspect", image, "--format", "{{.Id}}"], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"docker image {image!r} not found: {out.stderr.strip()}")
    return out.stdout.strip()


@dataclass(frozen=True)
class RunRecord:
    status: str  # complete | truncated | failed | timeout
    exit_code: int | None
    wall_seconds: float
    steps_declared: int
    steps_completed: int
    terminated_normally: bool
    nonconverged_steps: tuple[int, ...]
    fatal: tuple[str, ...]
    warnings: tuple[str, ...]
    diagnostic: str = ""  # parser or docker failure text
    adaptive_steps: tuple[int, ...] = ()  # steps recovered by WARP3D's adaptive driver (converged, recorded)


def classify(exit_code: int | None, timed_out: bool, parsed: ParsedOutput | None, steps_declared: int) -> str:
    if timed_out:
        return "timeout"
    if parsed is None:
        return "failed"
    if parsed.nonconverged_steps:
        return "truncated"  # "nonconvergent solutions stop": the job ends at the failed step without the marker
    if exit_code != 0 or not parsed.terminated_normally or parsed.fatal:
        return "failed"
    converged = set(parsed.converged_steps)
    declared = set(range(1, steps_declared + 1))
    if converged == declared:
        return "complete"
    if converged < declared:
        return "truncated"
    return "failed"  # steps beyond the declared count, or a gap: the record does not match the declaration


def _docker_path(path: Path) -> str:
    """A host path in the form Docker Desktop on Windows accepts (forward slashes, drive letter)."""
    s = str(path)
    if len(s) > 1 and s[1] == ":":
        return s[0].upper() + ":" + s[2:].replace("\\", "/")
    return s


@dataclass
class WarpRunner:
    """Runs one deck in the pinned image. ``inputs`` are copied into ``workdir`` (created by the runner, or an
    earlier job directory of the runner, never an arbitrary existing directory), bind-mounted at ``/job``;
    WARP3D reads the deck from stdin and writes ``out.txt`` and ``err.txt`` there."""

    image: str = "critflaw-warp3d:spike"
    threads: int = 4
    memory: str = "6g"
    timeout_seconds: float = 3600.0
    warp3d_exe: str = "/opt/warp3d/run_linux/warp3d.exe"
    image_id: str = ""  # the resolved image digest; resolved once (``resolve``) and used by key() and run() alike

    def resolve(self) -> str:
        """Resolve the image tag to its digest once; key() and run() both use this identity, so a tag that moves
        between keying and running cannot store one image's result under another's key."""
        if not self.image_id:
            self.image_id = image_digest(self.image)
        return self.image_id

    def key(self, inputs: dict[str, Path], deck: str) -> str:
        return solve_key(
            inputs_digest(inputs), deck, self.threads, self.resolve(), self.warp3d_exe, self.memory,
            self.timeout_seconds,
        )  # fmt: skip

    def run(self, workdir: Path, inputs: dict[str, Path], deck: str) -> tuple[RunRecord, Path]:
        """The declared step count is read from the entry deck (so that the record is a function of the key);
        the image is run by its resolved digest (the tag may move between keying and running)."""
        workdir = Path(workdir)
        if deck not in inputs:
            raise ValueError(f"the entry deck {deck!r} is not among the inputs")
        steps_declared = declared_steps(Path(inputs[deck]).read_text(encoding="utf-8"))
        # the job directory first: validated (runner-owned) and created before any write or image resolution
        if workdir.exists():
            if not (workdir / JOB_MARKER).exists():
                raise FileExistsError(f"{workdir} exists and is not a runner job directory")
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True)
        (workdir / JOB_MARKER).write_text("critflaw.fe runner job directory\n", encoding="utf-8")
        for name, src in inputs.items():
            shutil.copyfile(src, workdir / name)
        try:
            image_id = self.resolve()
        except (RuntimeError, OSError) as exc:  # no image / no docker: a failed record, never an escaped exception
            out_path = workdir / "out.txt"
            out_path.write_text("", encoding="utf-8")
            record = RunRecord(
                "failed", None, 0.0, steps_declared, 0, False, (), (), (), f"image resolution failed: {exc}"
            )
            return record, out_path
        name = f"critflaw-{uuid.uuid4().hex[:12]}"
        mount = _docker_path(workdir.resolve())
        cmd = [
            "docker", "run", "--rm", "--name", name, f"--memory={self.memory}", f"--cpus={self.threads}",
            "-v", f"{mount}:/job", "-w", "/job",
            "-e", f"OMP_NUM_THREADS={self.threads}", "-e", f"MKL_NUM_THREADS={self.threads}",
            image_id, "bash", "-c",
            f"ulimit -s unlimited; {self.warp3d_exe} < {deck} > out.txt 2> err.txt",
        ]  # fmt: skip
        start = time.time()
        timed_out = False
        diagnostic = ""
        exit_code: int | None = None
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_seconds)
            exit_code = proc.returncode
            if proc.returncode != 0:
                diagnostic = proc.stderr.strip()[:2000]
        except subprocess.TimeoutExpired:
            timed_out = True
            diagnostic = f"timeout after {self.timeout_seconds} s; container {name} killed"
            try:
                subprocess.run(["docker", "kill", name], capture_output=True, text=True, timeout=60)
                subprocess.run(["docker", "wait", name], capture_output=True, text=True, timeout=60)
            except (OSError, subprocess.SubprocessError) as exc:  # the record still exists; the kill is retried below
                diagnostic += f"; cleanup error: {exc}"
        except OSError as exc:  # docker not launchable: a failed record, never an escaped exception
            diagnostic = f"docker launch failed: {exc}"
        wall = time.time() - start
        out_path = workdir / "out.txt"
        if not out_path.exists():
            out_path.write_text("", encoding="utf-8")
        text = out_path.read_text(encoding="utf-8", errors="replace")
        parsed: ParsedOutput | None
        try:
            parsed = parse_output(text)
        except (ParseError, ValueError) as exc:  # any malformed output is a failed record with its diagnostic
            parsed = None
            diagnostic = (diagnostic + "; " if diagnostic else "") + f"parse error: {exc}"
        status = classify(exit_code, timed_out, parsed, steps_declared)
        record = RunRecord(
            status,
            exit_code,
            wall,
            steps_declared,
            parsed.steps_completed if parsed else 0,
            parsed.terminated_normally if parsed else False,
            parsed.nonconverged_steps if parsed else (),
            parsed.fatal if parsed else (),
            parsed.warnings if parsed else (),
            diagnostic,
            parsed.adaptive_steps if parsed else (),
        )
        return record, out_path


def container_exists(name: str) -> bool:
    out = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.Names}}"], capture_output=True, text=True
    )
    return out.stdout.strip() == name


@dataclass
class SolveCache:
    """Append-only store under ``root``: one directory per solve key holding the inputs, the gzipped raw output
    and stderr, ``run.json`` (the run record) and ``meta.json`` (case record and versions); parsed products
    under ``parsed/<parser version>-<reference set version>.json``."""

    root: Path
    entries_touched: list[str] = field(default_factory=list)
    retry_incomplete: bool = False  # explicit policy: re-solve a stored non-complete attempt (kept under attempts/)

    def entry(self, key: str) -> Path:
        return Path(self.root) / key

    def has(self, key: str) -> bool:
        return (self.entry(key) / "run.json").exists()

    def store(self, key: str, inputs: dict[str, Path], out_path: Path, record: RunRecord, meta: dict[str, Any]) -> Path:
        """Append-only for complete solves. A stored attempt that did not complete (killed, timed out, failed) is
        superseded by a later attempt under the same key: the earlier record and output move to
        ``attempts/<n>/`` and stay on record."""
        e = self.entry(key)
        if e.exists() and not (e / "run.json").exists():  # a write interrupted before completion
            shutil.move(str(e), str(e.with_name(f"{e.name}.interrupted-{uuid.uuid4().hex[:8]}")))
        if (e / "run.json").exists():
            if self.record(key).status == "complete":
                raise FileExistsError(f"cache entry {key} exists (append-only)")
            attempts = e / "attempts"
            attempts.mkdir(exist_ok=True)
            n = len([d for d in attempts.iterdir() if d.is_dir()]) + 1
            dest = attempts / str(n)
            dest.mkdir()
            for name in ("run.json", "meta.json", "out.txt.gz", "err.txt.gz", "packets.bpf.gz", "inputs", "parsed"):
                if (e / name).exists():
                    shutil.move(str(e / name), str(dest / name))
        (e / "inputs").mkdir(parents=True)
        for name, src in inputs.items():
            shutil.copyfile(src, e / "inputs" / name)
        with out_path.open("rb") as f_in, gzip.open(e / "out.txt.gz", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        for extra in ("err.txt", "packets.bpf"):  # stderr and the binary packets file (full-precision J)
            src_extra = out_path.with_name(extra)
            if src_extra.exists():
                with src_extra.open("rb") as f_extra, gzip.open(e / f"{extra}.gz", "wb") as g_extra:
                    shutil.copyfileobj(f_extra, g_extra)
        (e / "meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True), encoding="utf-8")
        tmp = e / "run.json.tmp"  # run.json last and atomically: its presence marks a complete entry
        tmp.write_text(json.dumps(asdict(record), indent=1), encoding="utf-8")
        tmp.replace(e / "run.json")
        self.entries_touched.append(key)
        return e

    def record(self, key: str) -> RunRecord:
        """The stored run record. The raw output is the durable fact and the classification a parser product:
        when the entry was stored by an older parser, the parsed fields and the status are re-derived from the
        stored raw output (the exit code, timeout flag, wall time and declaration are kept as stored)."""
        d = json.loads((self.entry(key) / "run.json").read_text(encoding="utf-8"))
        d.setdefault("adaptive_steps", [])
        for k in ("nonconverged_steps", "fatal", "warnings", "adaptive_steps"):
            d[k] = tuple(d[k])
        stored = RunRecord(**d)
        meta_path = self.entry(key) / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if meta.get("parser_version") == PARSER_VERSION or stored.status == "timeout":
            return stored
        try:
            parsed = parse_output(self.raw_output(key))
        except (ParseError, ValueError) as exc:
            return replace(stored, status="failed", diagnostic=f"parse error: {exc}")
        return replace(
            stored,
            status=classify(stored.exit_code, False, parsed, stored.steps_declared),
            steps_completed=parsed.steps_completed,
            terminated_normally=parsed.terminated_normally,
            nonconverged_steps=parsed.nonconverged_steps,
            fatal=parsed.fatal,
            warnings=parsed.warnings,
            adaptive_steps=parsed.adaptive_steps,
        )

    def packets(self, key: str) -> bytes | None:
        pk = self.entry(key) / "packets.bpf.gz"
        if not pk.exists():
            return None
        with gzip.open(pk, "rb") as f:
            return f.read()

    def raw_output(self, key: str) -> str:
        with gzip.open(self.entry(key) / "out.txt.gz", "rt", encoding="utf-8", errors="replace") as f:
            return f.read()

    def parsed(
        self, key: str, parser_version: str = PARSER_VERSION, reference_set_version: str = REFERENCE_SET_VERSION
    ) -> ParsedOutput:
        """The parsed layer: recomputed from the stored raw output when the (parser, reference set) version
        pair has no entry; the marker records what was computed."""
        marker = self.entry(key) / "parsed" / f"{parser_version}-{reference_set_version}.json"
        parsed = parse_output(self.raw_output(key))
        if not marker.exists():
            marker.parent.mkdir(exist_ok=True)
            summary = {
                "parser_version": parser_version,
                "reference_set_version": reference_set_version,
                "steps_completed": parsed.steps_completed,
                "domains": len(parsed.domains),
                "terminated_normally": parsed.terminated_normally,
            }
            marker.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        return parsed
