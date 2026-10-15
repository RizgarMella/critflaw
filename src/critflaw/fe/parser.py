"""Parser of WARP3D text output (theory manual, sections 6.1.4 and 6.1.7): per step the convergence record and
the cumulative applied load-pattern factors (every pattern and the constraints multiplier, read back, never
trusted from the declared list); per domain and step the J-integral table (one row per ring with ``total J``),
the average/min/max line, the symmetry-doubling note, the domain extents; interaction-integral blocks with the
per-ring value columns (``KI``/``KII``/``KIII`` with the auxiliary-field label plane stress / plane strain, the
T-stress columns) and the mixed-mode J; per-step nodal reactions with their totals; the normal-termination
marker; WARP3D's own fatal and warning messages against a documented list. Formats are those of the manual's
illustrative problems and the shipped verification decks, pinned by tests on stored outputs (``tests/data/fe``).

Strictness: a data row that cannot be read (an overflowed field, a malformed line) raises;
ring numbers must be contiguous from 1; a domain block without its loading/step line raises; nothing is
guessed. A caller that needs a durable record of an unparsable output catches ``ParseError``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class ParseError(ValueError):
    """The output could not be read as WARP3D output of the expected form."""


_STEP_RE = re.compile(r">> solution for step:\s+(\d+)\s+converged:\s+(\d+)\s+iters\.")
# the adaptive solution driver (manual 2.10.6) subdivides a step whose Newton iterations hit the limit and reports
# each sub-increment as '>> solution for step: N.250 converged' before the whole step 'N converged'
_SUBSTEP_RE = re.compile(r">> solution for step:\s+(\d+)\.(\d+)\s+converged:\s+(\d+)\s+iters\.")
_ADAPTIVE_RE = re.compile(r"adaptive solution driver for step:\s+(\d+)")
# WARP3D's message when a step cannot converge (as observed): ">> iteration limit exceeded for current
# step: N" followed by ">>>>> analysis terminated." (no normal-termination marker)
_NONCONV_RE = re.compile(
    r">> iteration limit exceeded for current step:\s+(\d+)"
    r"|>> solution for step:\s+(\d+)\s+(?:did not|failed to) converge",
    re.I,
)
_LOAD_HDR_RE = re.compile(r">> total applied load pattern factors through step:\s+(\d+)")
_PAIR_RE = re.compile(r"(\S+)\s+([-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+\.?)")
_LOADING_RE = re.compile(r"loading:\s+(\S+)\s+step:\s+(\d+)")
_DOMAIN_ID_RE = re.compile(r"domain id:\s+(\S+)")
_J_HDR_RE = re.compile(r"^\s*domain\s+dm1\s+.*total J\s+killed ele\s*$", re.M)
_II_HDR_RE = re.compile(r"^\s*domain\s+II1\s+.*total I\s+(.*?)\s+killed ele\s*$", re.M)
_AUX_RE = re.compile(r"--\s*(KI = \d, KII = \d, KIII = \d(?:, plane (?:stress|strain))?|T-stress \([^)]*\))\s*--")
_AVG_RE = re.compile(
    r"average:\s+([-+]?\d\.\d+E[-+]\d+)\s+minimum:\s+([-+]?\d\.\d+E[-+]\d+)\s+maximum:\s*([-+]?\d\.\d+E[-+]\d+)\s+\((\S+)\)"
)
_NUM = r"[-+]?\d\.\d+E[-+]\d+"
# the T-stress blocks print the labels on one line and one numeric (average, minimum, maximum) line per value column
_AVG_MULTI_RE = re.compile(
    r"average:\s+minimum:\s+maximum:\s+\((\S+)\)\s*\n((?:\s*" + _NUM + r"\s+" + _NUM + r"\s+" + _NUM + r"\s*\n)+)"
)
_ROW_RE = re.compile(rf"^\s*(\d+)\s+((?:{_NUM}\s+)+)\(\s*(\d+)\)\s*$")
_MIXED_J_RE = re.compile(r"domain\s+J plane (stress|strain)\s*\n((?:\s*\d+\s+[-+]?\d\.\d+E[-+]\d+\s*\n)+)", re.S)
# a clean finish: the "stop" command's timing summary; the J-adaptive stepping of some decks ends earlier with
# "Job terminated normally" (the spike deck). Either marks a normal termination.
_TERMINATED_RE = re.compile(r">>>> Job terminated normally|>> total job wall time \(secs\):")
_DOUBLED_RE = re.compile(r"\* note: J-values doubled for symmetry \*")
_BOUNDARY_NOTE_RE = re.compile(r"automatic domain builder cannot detect")
_EXTENTS_HDR_RE = re.compile(r"Physical Extents of Domains in Global Coordinates\s*\n\s*Domain\s+X-min.*?\n", re.S)
_EXTENTS_ROW_RE = re.compile(r"^\s*(\d+)\s+(" + r"[-+]?\d+\.\d+\s+" * 5 + r"[-+]?\d+\.\d+)\s*$", re.M)
_REACTION_BLOCK_RE = re.compile(r"step no\.\s+(\d+)(?:(?!step no\.).)*?nodal reaction forces(.*?)Totals:([^\n]*)", re.S)
_REAL = r"[-+]?\d*\.\d+(?:E[-+]\d+)?"
_REACTION_ROW_RE = re.compile(rf"^\s*(\d+)\s+({_REAL})\s+({_REAL})\s+({_REAL})\s*$", re.M)

# WARP3D's own fatal-condition messages (a run with any of these is never complete) and the benign messages
# seen in the reference outputs (the shipped verification decks), which are reported, not fatal.
FATAL_PATTERNS: tuple[str, ...] = (
    ">>>>> error",
    ">> error",
    "fatal error",
    "errors prevent analysis",
    "iteration limit exceeded",
    "analysis terminated",
    "diverging iterations",
    "nonconvergent solution detected",
    "local iterative algorithm",
    "failed to converge in material",
)
BENIGN_PATTERNS: tuple[str, ...] = (
    "node sets contain different numbers of nodes",  # testI (D1): tied node sets of unequal size
    "'symmetric' option detected",  # a symmetric domain: KII, KIII, T13 set to zero (mode I), as declared
)
_ELEMENT_LIST_FOLLOWS = re.compile(r"^[\d\s]+$")  # a continuation line that is itself a bare number list
# warnings a caller may declare expected for a run (collapsed crack-tip elements in a shipped mesh); never benign by
# default: a Jacobian warning on an ordinary element is a defect
COLLAPSED_ELEMENT_WARNINGS: tuple[str, ...] = (
    "duplicate entries in the incidences",
    "determinant of the jacobian matrix for gauss point",
)
WARNING_PATTERNS: tuple[str, ...] = (">>>>> warning", ">> warning")
PATTERN_NAME_WIDTH = 8  # the load-factor print truncates pattern names to eight characters (as observed)
PRINT_HALF_ULP = 5.1e-5  # five significant digits in the load-factor print: half a unit in the fifth digit
MAX_MESSAGE_LINES = 200  # a message longer than this is kept truncated and marked incomplete (never exempted)


@dataclass(frozen=True)
class StepRecord:
    step: int
    converged: bool
    iterations: int
    load_factors: dict[str, float]  # every cumulative applied pattern factor through this step, incl. constraints
    adaptive: bool = False  # converged through the adaptive driver's sub-increments after an iteration-limit hit


@dataclass(frozen=True)
class DomainRecord:
    """One domain-integral evaluation. ``components`` holds the component columns per ring; ``total`` the
    ``total J`` (J domains) or ``total I`` column; ``values`` the named value columns of an interaction
    integral (``{'KI': ..., 'KII': ..., 'KIII': ..., 'T11': ..., 'T33': ..., 'T13': ...}`` per ring),
    ``aux_field`` the auxiliary-field label of the block."""

    loading: str
    step: int
    domain_id: str
    kind: str  # "J" | "II"
    rings: NDArray[np.int64]
    components: FloatArray
    total: FloatArray
    average: float
    minimum: float
    maximum: float
    killed: NDArray[np.int64]
    doubled_for_symmetry: bool
    aux_field: str | None = None
    values: dict[str, FloatArray] = field(default_factory=dict)
    extents: FloatArray | None = None  # (rings, 6): xmin ymin zmin xmax ymax zmax

    @property
    def mode(self) -> str | None:
        keys = list(self.values)
        return keys[0] if len(keys) == 1 else None

    @property
    def k(self) -> FloatArray | None:
        return self.values[self.mode] if self.mode in ("KI", "KII", "KIII") else None

    def path_independence(self, first_ring: int = 2) -> float:
        """Largest relative deviation of ``total`` over rings >= ``first_ring`` from their mean (the
        'rings 2..n' path-independence measure); NaN when fewer than two rings qualify."""
        sel = self.rings >= first_ring
        if int(np.sum(sel)) < 2:
            return float("nan")
        vals = self.total[sel]
        mean = float(np.mean(vals))
        return float(np.max(np.abs(vals - mean)) / abs(mean)) if mean != 0.0 else float("inf")


@dataclass(frozen=True)
class ReactionRecord:
    step: int
    nodes: NDArray[np.int64]
    forces: FloatArray  # (n, 3)
    totals: FloatArray  # (3,), as printed (eight digits)


@dataclass(frozen=True)
class ParsedOutput:
    steps: tuple[StepRecord, ...]
    domains: tuple[DomainRecord, ...]
    mixed_mode_j: tuple[tuple[str, FloatArray], ...]  # (aux field, per-ring J from K) per interaction block
    terminated_normally: bool
    fatal: tuple[str, ...]  # lines matching FATAL_PATTERNS
    warnings: tuple[str, ...]  # WARP3D warning lines not in the benign list
    benign: tuple[str, ...]  # warning lines in the benign list
    reactions: tuple[ReactionRecord, ...] = field(default_factory=tuple)
    boundary_notes: int = 0
    nonconverged_steps: tuple[int, ...] = field(default_factory=tuple)
    adaptive_steps: tuple[int, ...] = field(default_factory=tuple)  # recovered by the adaptive driver

    @property
    def converged_steps(self) -> tuple[int, ...]:
        return tuple(s.step for s in self.steps if s.converged)

    @property
    def steps_completed(self) -> int:
        """The number of steps converged contiguously from step 1 (a gap ends the count)."""
        done = set(self.converged_steps)
        n = 0
        while n + 1 in done:
            n += 1
        return n

    def domain(self, domain_id: str, step: int, kind: str = "J", mode: str | None = None) -> DomainRecord:
        for d in self.domains:
            if d.domain_id == domain_id and d.step == step and d.kind == kind and (mode is None or mode in d.values):
                return d
        raise MissingRecord(f"no {kind} domain {domain_id!r} (mode {mode}) at step {step}")


class MissingRecord(ParseError, KeyError):
    """A requested record (domain, step) is absent from the output: a parse failure (and a KeyError for
    callers that catch that)."""


def _parse_table(block: str) -> tuple[NDArray[np.int64], FloatArray, NDArray[np.int64]]:
    """Every data row of a domain table, line by line: a row that is not readable (an overflowed field, a
    malformed line) raises; ring numbers must be 1..n contiguous. Returns (rings, numeric columns, killed)."""
    rings, rows, killed = [], [], []
    for raw in block.splitlines():
        line = raw.strip()
        if not line or not line[0].isdigit():
            continue
        m = _ROW_RE.match(raw)
        if m is None:
            raise ParseError(f"unreadable domain table row: {line!r}")
        rings.append(int(m.group(1)))
        rows.append([float(v) for v in m.group(2).split()])
        killed.append(int(m.group(3)))
    if not rings:
        raise ParseError("no table rows in the domain block")
    if rings != list(range(1, len(rings) + 1)):
        raise ParseError(f"domain rings are not contiguous from 1: {rings}")
    if len({len(r) for r in rows}) != 1:
        raise ParseError("domain table rows have different widths")
    return np.asarray(rings, dtype=np.int64), np.asarray(rows, dtype=float), np.asarray(killed, dtype=np.int64)


def _header_columns(header_tail: str) -> tuple[list[str], str | None]:
    """The value column names after ``total I`` of an interaction-integral header and the label they carry:
    'KI (pstrs)' -> (['KI'], 'plane stress'); 'T11 (pstrs)      T33' -> (['T11', 'T33'], 'plane stress')."""
    names: list[str] = []
    label: str | None = None
    for tok in header_tail.split():
        if tok.startswith("("):
            label = {"(pstrs)": "plane stress", "(pstrn)": "plane strain"}.get(tok, label)
        else:
            names.append(tok)
    return names, label


def _messages(text: str) -> list[str]:
    """WARP3D messages with their continuation lines joined: a message starts at a line holding a fatal or
    warning marker and continues over the following non-blank lines that carry no marker of their own."""
    out: list[str] = []
    current: list[str] | None = None
    for raw in text.splitlines():
        low = raw.lower()
        marker = any(p in low for p in FATAL_PATTERNS) or any(p in low for p in WARNING_PATTERNS)
        if marker:
            if current:
                out.append(" ".join(current))
            current = [raw.strip()]
        elif current is not None:
            # continuation: the following non-blank lines without a ">>" marker; a blank line ends the message
            # unless the next non-blank line is a bare number list (a duplicate-incidence warning lists its
            # elements after a blank line); at most MAX_MESSAGE_LINES (then marked incomplete)
            stripped = raw.strip()
            if ">>" in raw:
                out.append(" ".join(current))
                current = None
            elif not stripped:
                if not _ELEMENT_LIST_FOLLOWS.match("".join(current[-1:])) and not current[-1].endswith(":"):
                    out.append(" ".join(current))
                    current = None
            elif len(current) >= MAX_MESSAGE_LINES:
                current = current[:MAX_MESSAGE_LINES] + ["[message truncated: incomplete]"]
            elif current[-1].endswith(":") or _ELEMENT_LIST_FOLLOWS.match(current[-1]):
                if re.fullmatch(r"[\d\s]+", stripped):
                    current.append(stripped)
                else:
                    out.append(" ".join(current))
                    current = None
            else:
                current.append(stripped)
    if current:
        out.append(" ".join(current))
    return out


def _classify_messages(text: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    fatal: list[str] = []
    warnings: list[str] = []
    benign: list[str] = []
    for msg in _messages(text):
        low = msg.lower()
        if any(p in low for p in FATAL_PATTERNS):
            fatal.append(msg)
        elif any(b in low for b in BENIGN_PATTERNS):
            benign.append(msg)
        else:
            warnings.append(msg)
    return tuple(fatal), tuple(warnings), tuple(benign)


_JACOBIAN_WARNING_RE = re.compile(r"determinant of the jacobian matrix for gauss point\s+\d+\s+of element\s+(\d+)")
_DUPLICATE_WARNING_RE = re.compile(r"duplicate entries in the incidences for the following elements:\s*((?:\d+\s*)+)")


def expected_collapsed_warning(message: str, collapsed_elements: frozenset[int]) -> bool:
    """True when ``message`` is a Jacobian or duplicate-incidence warning about elements that the caller declares
    collapsed (crack-tip wedges); a warning about any other element is never expected."""
    low = message.lower()
    if "[message truncated: incomplete]" in low:
        return False  # a partial element list cannot be checked
    m = _JACOBIAN_WARNING_RE.search(low)
    if m:
        return int(m.group(1)) in collapsed_elements
    m = _DUPLICATE_WARNING_RE.search(low)
    if m:
        listed = {int(v) for v in m.group(1).split()}
        return bool(listed) and listed <= collapsed_elements
    return False


def _load_factors(tail: str, step: int) -> dict[str, float]:
    factors: dict[str, float] = {}
    h = _LOAD_HDR_RE.search(tail)
    if h and int(h.group(1)) == step:
        for line in tail[h.end() :].splitlines()[1:]:
            stripped = line.strip()
            if not stripped:
                if factors:
                    break
                continue
            if not stripped.startswith(">"):
                break
            for name, value in _PAIR_RE.findall(stripped[1:]):
                factors[name] = float(value)
    return factors


def _interaction_records(
    block: str, loading: str, step: int, doubled: bool, extents: FloatArray | None
) -> list[DomainRecord]:
    headers = list(_II_HDR_RE.finditer(block))
    aux_lines = list(_AUX_RE.finditer(block))
    if headers and len(aux_lines) != len(headers):
        raise ParseError(f"{len(headers)} interaction-integral headers but {len(aux_lines)} auxiliary-field lines")
    out: list[DomainRecord] = []
    for aux, hdr in zip(aux_lines, headers, strict=True):
        if aux.start() > hdr.start():
            raise ParseError("auxiliary-field line after its header")
        avg = _AVG_RE.search(block, hdr.end())
        multi = _AVG_MULTI_RE.search(block, hdr.end())
        if avg is None and multi is None:
            raise ParseError("interaction-integral block without an average line")
        if avg is not None and (multi is None or avg.start() < multi.start()):
            table_end, domain_id = avg.start(), avg.group(4)
            stats = (float(avg.group(1)), float(avg.group(2)), float(avg.group(3)))
        else:
            assert multi is not None
            table_end, domain_id = multi.start(), multi.group(1)
            first = multi.group(2).strip().splitlines()[0].split()
            stats = (float(first[0]), float(first[1]), float(first[2]))
        rings, cols, killed = _parse_table(block[hdr.end() : table_end])
        names, label = _header_columns(hdr.group(1))
        n_val = len(names)
        if n_val < 1 or cols.shape[1] < n_val + 1:
            raise ParseError(f"interaction-integral header {hdr.group(0)!r} does not match its table")
        aux_text = aux.group(1)
        aux_label = (
            "plane stress" if "plane stress" in aux_text else ("plane strain" if "plane strain" in aux_text else None)
        )
        if label is not None and aux_label is not None and label != aux_label:
            raise ParseError(f"auxiliary field {aux_label!r} disagrees with the header label {label!r}")
        values = {name: cols[:, cols.shape[1] - n_val + i] for i, name in enumerate(names)}
        out.append(
            DomainRecord(
                loading,
                step,
                domain_id,
                "II",
                rings,
                cols[:, : -n_val - 1],
                cols[:, -n_val - 1],
                stats[0],
                stats[1],
                stats[2],
                killed,
                doubled,
                aux_field=aux_label or label,
                values=values,
                extents=extents,
            )  # fmt: skip
        )
    return out


def parse_output(text: str) -> ParsedOutput:
    """Parse one WARP3D text output (raises ``ParseError`` on an unreadable table or block)."""
    steps: dict[int, StepRecord] = {}
    for m in _STEP_RE.finditer(text):
        n = int(m.group(1))
        if n in steps:
            raise ParseError(f"step {n} converged twice in the output")
        factors = _load_factors(text[m.end() : m.end() + 4000], n)
        if not factors:
            raise ParseError(f"converged step {n} has no applied-load-factor block")
        steps[n] = StepRecord(n, True, int(m.group(2)), factors)
    limit_hits = {int(m.group(1) or m.group(2)) for m in _NONCONV_RE.finditer(text)}
    adaptive_driver = {int(m.group(1)) for m in _ADAPTIVE_RE.finditer(text)}
    # a step that hit the iteration limit, was taken over by the adaptive driver and finally converged as a
    # whole step is recovered (recorded as adaptive, its iterations the total over the sub-increments); one
    # without the final whole-step convergence line is nonconverged
    sub_iters: dict[int, int] = {}
    for m in _SUBSTEP_RE.finditer(text):
        sub_iters[int(m.group(1))] = sub_iters.get(int(m.group(1)), 0) + int(m.group(3))
    adaptive = tuple(sorted(n for n in limit_hits | adaptive_driver if n in steps))
    for n in adaptive:
        rec = steps[n]
        steps[n] = StepRecord(n, True, rec.iterations + sub_iters.get(n, 0), rec.load_factors, True)
    nonconverged = tuple(sorted(n for n in limit_hits | adaptive_driver if n not in steps))
    for n in nonconverged:
        steps.setdefault(n, StepRecord(n, False, 0, {}))

    domains: list[DomainRecord] = []
    mixed: list[tuple[str, FloatArray]] = []
    starts = [m.start() for m in re.finditer(r"domain-integral post-processing", text)]
    starts.append(len(text))
    for a, b in zip(starts[:-1], starts[1:], strict=True):
        block = text[a:b]
        lm = _LOADING_RE.search(block)
        if lm is None:
            raise ParseError("domain block without a 'loading: ... step:' line")
        loading, step = lm.group(1), int(lm.group(2))
        dm = _DOMAIN_ID_RE.search(block)
        doubled = bool(_DOUBLED_RE.search(block))
        extents: FloatArray | None = None
        em = _EXTENTS_HDR_RE.search(block)
        if em:
            rows = _EXTENTS_ROW_RE.findall(block[em.end() : em.end() + 20000])
            extents = np.asarray([[float(v) for v in r[1].split()] for r in rows], dtype=float)
        jh = _J_HDR_RE.search(block)
        if jh:
            avg = _AVG_RE.search(block, jh.end())
            if avg is None:
                raise ParseError("J block without an average line")
            rings, cols, killed = _parse_table(block[jh.end() : avg.start()])
            domain_id = dm.group(1) if dm else avg.group(4)
            domains.append(
                DomainRecord(
                    loading,
                    step,
                    domain_id,
                    "J",
                    rings,
                    cols[:, :-1],
                    cols[:, -1],
                    float(avg.group(1)),
                    float(avg.group(2)),
                    float(avg.group(3)),
                    killed,
                    doubled,
                    extents=extents,
                )  # fmt: skip
            )
        domains += _interaction_records(block, loading, step, doubled, extents)
        for mm in _MIXED_J_RE.finditer(block):
            vals = [float(line.split()[1]) for line in mm.group(2).strip().splitlines() if line.strip()]
            mixed.append((f"plane {mm.group(1)}", np.asarray(vals, dtype=float)))

    reactions: list[ReactionRecord] = []
    for m in _REACTION_BLOCK_RE.finditer(text):
        for raw in m.group(2).splitlines():
            s = raw.strip()
            if s and s[0].isdigit() and "*" in s:
                raise ParseError(f"overflowed reaction row: {s!r} (use 'output wide eformat reactions')")
        rows = _REACTION_ROW_RE.findall(m.group(2))
        if not rows:
            raise ParseError("reaction block without readable rows")
        totals = re.findall(_REAL, m.group(3))
        if len(totals) != 3 or "*" in m.group(3):
            raise ParseError(f"unreadable reaction totals: {m.group(3).strip()!r}")
        reactions.append(
            ReactionRecord(
                int(m.group(1)),
                np.asarray([int(r[0]) for r in rows], dtype=np.int64),
                np.asarray([[float(r[1]), float(r[2]), float(r[3])] for r in rows], dtype=float),
                np.asarray([float(v) for v in totals], dtype=float),
            )
        )

    fatal, warnings, benign = _classify_messages(text)
    if adaptive:  # the iteration-limit message of a recovered step is not fatal; it is kept as a warning
        recovered = [f for f in fatal if _recovered_limit(f, adaptive)]
        fatal = tuple(f for f in fatal if f not in recovered)
        warnings = warnings + tuple(recovered)
    return ParsedOutput(
        tuple(steps[k] for k in sorted(steps)),
        tuple(domains),
        tuple(mixed),
        bool(_TERMINATED_RE.search(text)),
        fatal,
        warnings,
        benign,
        tuple(reactions),
        len(_BOUNDARY_NOTE_RE.findall(text)),
        nonconverged,
        adaptive,
    )


def _recovered_limit(message: str, adaptive: tuple[int, ...]) -> bool:
    m = re.search(r"iteration limit exceeded for current step:\s+(\d+)", message)
    return m is not None and int(m.group(1)) in adaptive


@dataclass(frozen=True)
class RunQA:
    """The run-layer QA record (theory manual, section 6.1.7): every check a named value with its verdict."""

    steps_declared: int
    steps_completed: int
    all_declared_converged: bool
    read_back_matches_declared: bool  # cumulative factors read back equal the declared list (1e-9 relative)
    fatal_messages: tuple[str, ...]
    unexpected_warnings: tuple[str, ...]
    terminated_normally: bool

    @property
    def passed(self) -> bool:
        return (
            self.all_declared_converged
            and self.read_back_matches_declared
            and not self.fatal_messages
            and not self.unexpected_warnings
            and self.terminated_normally
        )


def run_qa(
    parsed: ParsedOutput,
    pattern: str,
    declared: tuple[float, ...],
    collapsed_elements: frozenset[int] = frozenset(),
) -> RunQA:
    """``collapsed_elements``: the elements the caller declares collapsed (crack-tip wedges); a Jacobian or
    duplicate-incidence warning about one of them is expected, every other warning is unexpected."""
    n = len(declared)
    all_ok = set(parsed.converged_steps) == set(range(1, n + 1)) and not parsed.nonconverged_steps
    match = True
    for k, lam in enumerate(declared, start=1):
        rec = next((s for s in parsed.steps if s.step == k), None)
        got = None if rec is None else rec.load_factors.get(pattern, rec.load_factors.get(pattern[:PATTERN_NAME_WIDTH]))
        # the factors are printed to five significant digits: the declared value must round to the printed one
        if got is None or abs(got - lam) > PRINT_HALF_ULP * max(abs(lam), 1e-300):
            match = False
    unexpected = tuple(
        w
        for w in parsed.warnings
        if not expected_collapsed_warning(w, collapsed_elements) and not _recovered_limit(w, parsed.adaptive_steps)
    )  # an iteration-limit message of a step the adaptive driver recovered is recorded, not unexpected
    return RunQA(n, parsed.steps_completed, all_ok, match, parsed.fatal, unexpected, parsed.terminated_normally)


def check_domain_coverage(
    parsed: ParsedOutput, requested: dict[str, tuple[int, int]], step: int, kind: str = "J"
) -> None:
    """Mechanics-layer QA: every requested domain is present at ``step`` with exactly the requested rings (a
    truncated table — fewer rings than requested — is a defect, not a smaller domain set)."""
    for domain_id, (first, last) in requested.items():
        try:
            d = parsed.domain(domain_id, step, kind)
        except KeyError as exc:
            raise ParseError(str(exc)) from exc
        want = list(range(first, last + 1))
        if d.rings.tolist() != want:
            raise ParseError(f"domain {domain_id} at step {step}: rings {d.rings.tolist()} differ from {want}")


@dataclass(frozen=True)
class EquilibriumQA:
    step: int
    nodes_complete: bool  # every declared constrained node has a reaction row
    force_residual: float  # |sum R + factor * F_applied| / |factor * F_applied|
    moment_residual: float  # same for the moment about the origin
    passed: bool


def equilibrium_qa(
    parsed: ParsedOutput,
    coords: FloatArray,
    constrained_nodes: NDArray[np.int64],
    applied_force: FloatArray,
    applied_moment: FloatArray,
    factors: tuple[float, ...],
    origin: FloatArray,
    tolerance: float = 1e-4,
) -> tuple[EquilibriumQA, ...]:
    """Reaction completeness and force/moment balance per step (run-layer QA): the parsed reactions at
    the declared constrained nodes must balance ``factor x`` the applied resultants (nodal rows carry five
    printed digits, hence the 1e-4 default). ``coords`` is the (n_nodes, 3) array, node i at row i - 1."""
    by_step: dict[int, ReactionRecord] = {}
    for rx in parsed.reactions:
        if rx.step in by_step:
            raise ParseError(f"two reaction blocks for step {rx.step}")
        by_step[rx.step] = rx
    missing = [k for k in range(1, len(factors) + 1) if k not in by_step]
    if missing or set(by_step) - set(range(1, len(factors) + 1)):
        raise ParseError(
            f"reaction blocks for steps {sorted(by_step)} do not match the declared steps 1..{len(factors)}"
        )
    out = []
    wanted = set(int(n) for n in constrained_nodes)
    for k, factor in enumerate(factors, start=1):
        rx = by_step[k]
        present = set(int(n) for n in rx.nodes)
        complete = wanted <= present
        arms = coords[rx.nodes - 1] - origin
        force = rx.forces.sum(axis=0)
        moment = np.cross(arms, rx.forces).sum(axis=0)
        f_ref, m_ref = factor * applied_force, factor * applied_moment
        # dimensional scales: the larger of the reference resultant and the sum of the reaction magnitudes (a
        # zero applied moment, pure tension about the load centroid, must not demand exact cancellation)
        f_scale = max(float(np.linalg.norm(f_ref)), float(np.linalg.norm(rx.forces, axis=1).sum()), 1e-300)
        m_scale = max(  # the magnitudes of the individual moment contributions |r_i x R_i|
            float(np.linalg.norm(m_ref)),
            float(np.linalg.norm(np.cross(arms, rx.forces), axis=1).sum()),
            1e-300,
        )
        f_res = float(np.linalg.norm(force + f_ref) / f_scale)
        m_res = float(np.linalg.norm(moment + m_ref) / m_scale)
        ok = complete and f_res <= tolerance and m_res <= tolerance
        out.append(EquilibriumQA(k, complete, f_res, m_res, ok))
    return tuple(out)


_GP_TABLE_RE = re.compile(
    r"step no\.\s+(\d+)(?:(?!step no\.).)*?\n\s+(strains|stresses)\s*\n(.*?)"
    r"(?=\n\s+(?:strains|stresses|nodal displacements|nodal reaction forces)\s*\n"
    r"|\n\s+(?:step no\.|structure |domain-integral post-processing)|>> |\Z)",
    re.S,
)


@dataclass(frozen=True)
class GaussPointTable:
    """Element Gauss-point output of one step: ``values[(element, gauss point)] = row of the printed quantities
    in print order`` (strains: eps_xx eps_yy eps_zz gam_xy gam_yz gam_xz eff_eps; stresses: sigma_xx sigma_yy
    sigma_zz sigma_xy sigma_yz sigma_xz eng_dens mises c1 c2 c3)."""

    step: int
    quantity: str
    values: dict[tuple[int, int], FloatArray]


def _gauss_point_rows(body: str, step: int) -> dict[tuple[int, int], FloatArray]:
    values: dict[tuple[int, int], FloatArray] = {}
    element: int | None = None
    key: tuple[int, int] | None = None
    current: list[float] = []
    first_width: int | None = None  # values on a row's first line (6 wide eformat, 3 fixed format)
    for raw in body.splitlines():
        parts = raw.split()
        if not parts or not re.match(r"[-+]?\d", parts[0]):
            continue
        if "*" in raw:
            raise ParseError(f"overflowed Gauss-point row at step {step}: {raw.strip()!r}")
        starts_element = len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit()
        if starts_element and first_width is None:
            first_width = len(parts) - 2
        # a line with one leading integer is a new Gauss point when it carries a full first line; at a page break
        # WARP3D repeats the element number before a row's continuation values (fewer than a first line)
        n_values = len(parts) - 1
        starts_point = (
            parts[0].isdigit() and element is not None and not starts_element and first_width is not None
            and n_values >= first_width
        )  # fmt: skip
        if parts[0].isdigit() and not starts_element and not starts_point and element is not None:
            parts = parts[1:]  # the repeated element number of a split row
        if starts_element or starts_point:
            if key is not None:
                values[key] = np.asarray(current, dtype=float)
            if starts_element:
                element = int(parts[0])
                key, current = (element, int(parts[1])), [float(v) for v in parts[2:]]
            else:
                assert element is not None
                key, current = (element, int(parts[0])), [float(v) for v in parts[1:]]
        else:
            current += [float(v) for v in parts]
    if key is not None:
        values[key] = np.asarray(current, dtype=float)
    return values


def parse_gauss_point_tables(text: str) -> tuple[GaussPointTable, ...]:
    """The ``output strains/stresses for elements`` tables (Gauss-point location) of every step; an overflowed
    field raises."""
    # WARP3D paginates a long table, repeating the step header on every page (a row's two lines may be split
    # by a page break): consecutive pages of the same (step, quantity) are joined before the rows are read
    groups: list[tuple[int, str, list[str]]] = []
    for m in _GP_TABLE_RE.finditer(text):
        step, quantity, body = int(m.group(1)), m.group(2), m.group(3)
        if groups and groups[-1][0] == step and groups[-1][1] == quantity:
            groups[-1][2].append(body)
        else:
            groups.append((step, quantity, [body]))
    return tuple(GaussPointTable(step, q, _gauss_point_rows(chr(10).join(bodies), step)) for step, q, bodies in groups)


# ----------------------------------------------------------------------------------------------------------------
# Binary packets file (manual 2.12 "Binary Packets File", Appendix F): Fortran sequential unformatted records,
# each framed by a 4-byte little-endian length before and after. Every packet starts with a header record of
# four integers (type, number of following records, load step, Newton iteration). Packet type 17 (F.19, J
# domain integral): record 1 = rings (int), domain id (char 24), structure (char 8), loading (char 8), step
# (int); one record per ring = ring (int), dm1..dm8 and total J (9 doubles), killed elements (int); then
# average/min/max (3 doubles) when several rings; then one record per ring of K from J (ring, 5 doubles).
# The printed tables carry four significant digits; the packets carry the full double-precision values.


@dataclass(frozen=True)
class JPacket:
    domain_id: str
    loading: str
    step: int
    rings: NDArray[np.int64]
    components: FloatArray  # (rings, 8)
    total: FloatArray
    killed: NDArray[np.int64]
    average: float
    minimum: float
    maximum: float


def _fortran_records(data: bytes) -> list[bytes]:
    out: list[bytes] = []
    pos = 0
    n_bytes = len(data)
    while pos < n_bytes:
        if pos + 4 > n_bytes:
            raise ParseError("packets file: truncated record marker")
        n = int(np.frombuffer(data, dtype="<i4", count=1, offset=pos)[0])
        pos += 4
        if n < 0 or pos + n + 4 > n_bytes:
            raise ParseError("packets file: truncated record")
        out.append(data[pos : pos + n])
        pos += n
        n2 = int(np.frombuffer(data, dtype="<i4", count=1, offset=pos)[0])
        pos += 4
        if n2 != n:
            raise ParseError(f"packets file: record markers disagree ({n} != {n2})")
    return out


def read_j_packets(data: bytes) -> list[JPacket]:
    """Every type-17 packet of a binary packets file, in file order; other packet types are skipped by their
    declared record counts."""
    recs = _fortran_records(data)
    out: list[JPacket] = []
    i = 0
    while i < len(recs):
        head = recs[i]
        if len(head) != 16:
            raise ParseError(f"packets file: header record of {len(head)} bytes at record {i}")
        ptype, n_more, step, _iteration = (int(v) for v in np.frombuffer(head, dtype="<i4"))
        body = recs[i + 1 : i + 1 + n_more]
        if len(body) != n_more:
            raise ParseError(f"packets file: packet {ptype} at record {i} declares {n_more} records, file ends")
        i += 1 + n_more
        if ptype != 17:
            continue
        if n_more < 1:
            raise ParseError(f"packets file: J packet at record {i} declares no records")
        first = body[0]
        if len(first) != 48:
            raise ParseError(f"packets file: domain record of {len(first)} bytes at record {i}")
        n_rings = int(np.frombuffer(first, dtype="<i4", count=1)[0])
        try:
            domain_id = first[4:28].decode("ascii").strip()
            loading = first[36:44].decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise ParseError(f"packets file: non-ASCII name in a J packet at record {i}") from exc
        expected = 1 + n_rings + (1 if n_rings > 1 else 0) + n_rings  # F.19: rings, stats, K from J
        if n_more != expected:
            raise ParseError(f"packets file: J packet of {n_rings} rings declares {n_more} records, not {expected}")
        step_in = int(np.frombuffer(first, dtype="<i4", count=1, offset=44)[0])
        if step_in != step:
            raise ParseError(f"packets file: domain {domain_id} step {step_in} in a packet headed step {step}")
        rings = np.zeros(n_rings, dtype=np.int64)
        comps = np.zeros((n_rings, 8))
        total = np.zeros(n_rings)
        killed = np.zeros(n_rings, dtype=np.int64)
        for r in range(n_rings):
            rec = body[1 + r]
            if len(rec) != 80:
                raise ParseError(f"packets file: ring record of {len(rec)} bytes in domain {domain_id}")
            rings[r] = int(np.frombuffer(rec, dtype="<i4", count=1)[0])
            vals = np.frombuffer(rec, dtype="<f8", count=9, offset=4)
            comps[r] = vals[:8]
            total[r] = vals[8]
            killed[r] = int(np.frombuffer(rec, dtype="<i4", count=1, offset=76)[0])
        if n_rings > 1:
            avg, lo, hi = (float(v) for v in np.frombuffer(body[1 + n_rings], dtype="<f8", count=3))
        else:
            avg = lo = hi = float(total[0])
        out.append(JPacket(domain_id, loading, step, rings, comps, total, killed, avg, lo, hi))
    return out


def gauss_rows(
    tables: list[GaussPointTable], elements: Iterable[int], column: int, what: str, points: int = 8
) -> dict[tuple[int, int], FloatArray]:
    """The rows of exactly the expected (element, Gauss point 1..points) identities of ``elements`` over
    ``tables``: a duplicate identity across tables, an unexpected Gauss-point number, a missing identity or a
    non-finite value in ``column`` is a ParseError (nothing is ever counted as observed by default)."""
    wanted_elements = {int(e) for e in elements}
    wanted = {(e, g) for e in wanted_elements for g in range(1, points + 1)}
    out: dict[tuple[int, int], FloatArray] = {}
    for t in tables:
        for key, row in t.values.items():
            if key not in wanted and key[0] in wanted_elements:
                raise ParseError(f"{what}: unexpected Gauss point {key[1]} of element {key[0]}")
            if key in out:
                raise ParseError(f"{what}: Gauss point {key[1]} of element {key[0]} appears twice")
            if key in wanted:
                if not np.isfinite(row[column]):
                    raise ParseError(f"{what}: non-finite value at Gauss point {key[1]} of element {key[0]}")
                out[key] = row
    missing = wanted - set(out)
    if missing:
        raise ParseError(f"{what}: {len(missing)} of {len(wanted)} expected Gauss-point rows missing")
    return out
