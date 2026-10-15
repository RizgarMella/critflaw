"""WARP3D deck writer (theory manual, chapter 06, section 6.1). Every command emitted here is listed in
``docs/manual/warp3d_commands.md`` with the manual section it was checked against; a test tokenises the emitted decks against
that list. The writer takes a ``DeckSpec`` (what the mesher or a hand-built case supplies) and returns the
control deck text; coordinates, incidences, constraints and load patterns are separate files named in
``DeckSpec.input_files`` and included with ``*input from``.

Step loads: WARP3D step loads are increments of the pattern factors (manual 2.8.4); the writer takes the
declared **cumulative** factors ``lambda_1 < ... < lambda_N`` and emits the increments; the parser reads the
cumulative factors back from the output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

MAX_COLUMNS = 80  # manual 1.3: nothing beyond column 80 is read
OUTPUT_CHUNK = 1500  # elements per Gauss-point output command


def num(x: float) -> str:
    """A real number in a form WARP3D's scanner accepts: a decimal point always present, no '+' in the exponent
    (``1e+10`` is rejected: 'expecting real number'), 12 significant digits."""
    s = f"{float(x):.12g}"
    if "e" in s:
        mant, exp = s.split("e")
        if "." not in mant:
            mant += ".0"
        return f"{mant}e{int(exp)}"
    return s if "." in s else s + ".0"


@dataclass(frozen=True)
class ElasticMaterial:
    """Linear-elastic run: the ``mises`` model with a yield stress far above any stress reached (there is no
    linear-elastic model type); the QA run layer checks that no integration point yields."""

    E: float
    nu: float
    yld_pt_multiple: float = 1e4
    max_expected_stress: float = 1.0

    def lines(self, name: str) -> list[str]:
        yld = self.yld_pt_multiple * self.max_expected_stress
        return [
            f" material {name}",
            f"  properties mises e {num(self.E)},",
            f"     nu {num(self.nu)},",
            f"     yld_pt {num(yld)}",
        ]


@dataclass(frozen=True)
class DeformationMaterial:
    """WARP3D ``deformation`` model (manual 3.4): ``e``, ``nu``, ``yld_pt`` = sigma_0, ``n_power`` = n."""

    E: float
    nu: float
    sigma_0: float
    n: float

    def lines(self, name: str) -> list[str]:
        return [
            f" material {name}",
            f"  properties deformation e {num(self.E)},",
            f"     nu {num(self.nu)},",
            f"     yld_pt {num(self.sigma_0)},",
            f"     n_power {num(self.n)}",
        ]


@dataclass(frozen=True)
class SolverControls:
    """Every numerical control in the deck, explicit (no template defaults); b-bar is not applicable to the
    20-node element and is not a field."""

    solution_technique: str = "sparse iterative"
    time_step: float = 1.0e10
    maximum_iterations: int = 10
    minimum_iterations: int = 2
    extrapolate: bool = True
    nonconvergent_stop: bool = True
    norm_res_tol: float = 0.02
    adaptive: bool = True
    line_search: bool = True
    divergence_check: bool = True
    trace_solution: bool = True
    batch_messages: bool = False
    packets_file: str = ""  # binary packets file (manual 2.12, Appendix F): full-precision J (packet 17); "" = off

    def lines(self) -> list[str]:
        packets = [f"  binary packets on file '{self.packets_file}'"] if self.packets_file else []
        return [
            " nonlinear analysis parameters",
            f"  solution technique {self.solution_technique}",
            *packets,
            f"  time step {num(self.time_step)}",
            f"  maximum iterations {self.maximum_iterations}",
            f"  minimum iterations {self.minimum_iterations}",
            f"  extrapolate {'on' if self.extrapolate else 'off'}",
            f"  nonconvergent solutions {'stop' if self.nonconvergent_stop else 'continue'}",
            f"  convergence test norm res tol {num(self.norm_res_tol)}",
            f"  trace solution {'on' if self.trace_solution else 'off'}",
            f"  adaptive solution {'on' if self.adaptive else 'off'}",
            f"  line search {'on' if self.line_search else 'off'}",
            f"  divergence check {'on' if self.divergence_check else 'off'}",
            f"  batch messages {'on' if self.batch_messages else 'off'}",
        ]


@dataclass(frozen=True)
class DomainSpec:
    """One crack-front domain: ``front_nodes`` in WARP3D's order (corner, mid-side, corner, ...);
    ``kind`` 'J' (domain integral) or 'II' (interaction integrals, elastic runs, plane-strain auxiliary field
    unless ``plane_stress``)."""

    domain_id: str
    front_nodes: tuple[int, ...]
    rings: tuple[int, int]
    symmetric: bool  # required: J is doubled for a symmetric domain (checked against the declared planes)
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    tangent: tuple[float, float, float] | None = None
    function_type: str = "b"
    element_type: str = "q3disop"
    kind: Literal["J", "II"] = "J"
    plane_stress: bool = False
    print_extents: bool = True
    packets: bool = False  # 'output packets J' (manual 4.4.13): J per ring at full precision in the packets file
    comment: str = ""

    def lines(self) -> list[str]:
        out = [f"domain {self.domain_id}" + (f"   $ {self.comment}" if self.comment else "")]
        nx, ny, nz = self.normal
        out.append(f"  normal plane nx {nx:.6f} ny {ny:.6f} nz {nz:.6f}")
        if self.tangent is not None:
            tx, ty, tz = self.tangent
            out.append(f"  tangent vector tx {tx:.6f} ty {ty:.6f} tz {tz:.6f}")
        if self.symmetric:
            out.append("  symmetric")
        out += wrap_list("  front nodes", " ".join(str(n) for n in self.front_nodes) + f" {self.element_type} verify")
        out.append(f"  q-values automatic rings {self.rings[0]}-{self.rings[1]}")
        out.append(f"  function type {self.function_type}")
        if self.kind == "II":
            if self.plane_stress:
                out.append("  plane stress output for interaction integral")
            else:
                out.append("  plane strain output for interaction integral")
        if self.print_extents:
            out.append("  print domain extents")
        if self.packets:
            if self.kind != "J":
                raise NotImplementedError("TODO(verify): packet output of interaction integrals (manual 4.5)")
            out.append("  output packets J")
        out.append("  compute domain integral" if self.kind == "J" else "  compute domain interaction integrals")
        return out


@dataclass(frozen=True)
class DeckSpec:
    structure: str
    n_nodes: int
    n_elements: int
    material: ElasticMaterial | DeformationMaterial
    input_files: tuple[str, ...]  # coordinates, incidences, constraints, load patterns (in this order of use)
    loading_name: str
    pattern_name: str
    cumulative_factors: tuple[float, ...]
    domains: tuple[DomainSpec, ...]
    solver: SolverControls = SolverControls()
    element_type: str = "q3disop"
    integration_order: str = "2x2x2"
    element_output: str = "nodpts short"
    material_name: str = "mat"
    reaction_nodes: str = ""  # a node list expression for output reactions, "" for none
    stress_elements: str = ""  # an element list expression for output stresses, "" for none
    strain_elements: str = ""
    displacement_nodes: str = ""  # a node list expression for output displacements, "" for none
    output_after_steps: tuple[int, ...] = field(default_factory=tuple)  # steps at which the element output is requested
    reactions_every_step: bool = False  # reactions after every step (equilibrium QA), not only the output steps
    stress_after_steps: tuple[int, ...] | None = None  # None: the output steps; else the steps for the stress output
    strain_after_steps: tuple[int, ...] | None = None  # None: the output steps; else the steps for the strain output
    echo: bool = False

    def __post_init__(self) -> None:
        lam = np.asarray(self.cumulative_factors, dtype=float)
        if lam.size == 0 or np.any(np.diff(lam) <= 0.0) or lam[0] <= 0.0:
            raise ValueError("cumulative load factors must be positive and strictly increasing")

    @property
    def increments(self) -> tuple[float, ...]:
        lam = np.asarray(self.cumulative_factors, dtype=float)
        return tuple(float(v) for v in np.diff(np.concatenate([[0.0], lam])))


def write_deck(spec: DeckSpec) -> str:
    """The control deck text."""
    lines: list[str] = [f" structure {spec.structure}", f" number of nodes {spec.n_nodes} elements {spec.n_elements}"]
    lines += spec.material.lines(spec.material_name)
    lines += [
        " elements",
        f"    all type {spec.element_type} linear material {spec.material_name},",
        f"       order {spec.integration_order} {spec.element_output}",
    ]
    lines += [f"*input from '{name}'" for name in spec.input_files]
    lines.append(" blocking automatic")
    lines += [f" loading {spec.loading_name}", "  nonlinear"]
    for k, inc in enumerate(spec.increments, start=1):
        lines.append(f"   step {k} {spec.pattern_name} {num(inc)}")
    lines += spec.solver.lines()
    n_steps = len(spec.cumulative_factors)
    outputs = set(spec.output_after_steps) or {n_steps}
    stress_steps = outputs if spec.stress_after_steps is None else set(spec.stress_after_steps)
    strain_steps = outputs if spec.strain_after_steps is None else set(spec.strain_after_steps)
    for k in range(1, n_steps + 1):
        lines.append(f" compute displacements for loading {spec.loading_name} step {k}")
        if spec.reaction_nodes and (spec.reactions_every_step or k in outputs):
            lines += wrap_list(" output wide eformat reactions", spec.reaction_nodes)
        if spec.stress_elements and k in stress_steps:
            lines += wrap_list(" output wide eformat stresses for elements", spec.stress_elements)
        if spec.strain_elements and k in strain_steps:
            # in element chunks: one output command for the whole far region drove the solver's memory over the
            # container limit (exit 137 at the strain-output steps); WARP3D buffers per output command
            tokens = spec.strain_elements.split()
            for i in range(0, len(tokens), OUTPUT_CHUNK):
                lines += wrap_list(" output wide eformat strains for elements", " ".join(tokens[i : i + OUTPUT_CHUNK]))
        if spec.displacement_nodes and k in outputs:
            lines += wrap_list(" output wide eformat displacements for nodes", spec.displacement_nodes)
        for d in spec.domains:
            lines += d.lines()
    lines.append(" stop")
    check_domain_rings(spec.domains)
    too_long = [ln for ln in lines if len(ln) > MAX_COLUMNS]
    if too_long:
        raise ValueError(f"deck line exceeds {MAX_COLUMNS} columns (WARP3D ignores the rest silently): {too_long[0]!r}")
    return "\n".join(lines) + "\n"


_COMPUTE_RE = re.compile(
    r"^\s*compute\s+displacements\s+(?:for\s+)?loading\s+\S+\s+(?:for\s+)?steps?\s+(\d+)(?:\s*-\s*(\d+))?", re.M | re.I
)


def declared_steps(deck_text: str) -> int:
    """The number of load steps a control deck computes: the largest step of its ``compute displacements ...
    step(s) N`` commands (a shipped deck may define step loads for more steps than it computes). Read from the
    deck itself, never passed beside it, so that a cached record is a function of the solve key."""
    found = [int(m.group(2) or m.group(1)) for m in _COMPUTE_RE.finditer(deck_text)]
    if not found:
        raise ValueError("the deck has no 'compute displacements ... step' command")
    return max(found)


def check_domain_rings(domains: tuple[DomainSpec, ...], min_rings: int = 6) -> None:
    """Mesh-layer QA: every domain requests at least ``min_rings`` rings from ring 1 (n_rings >= 6 so
    that the path-independence measure over rings 2..n has >= 5 domains)."""
    for d in domains:
        if d.rings[0] != 1 or d.rings[1] - d.rings[0] + 1 < min_rings:
            raise ValueError(f"domain {d.domain_id}: rings {d.rings} (need rings 1..n with n >= {min_rings})")


def wrap_list(prefix: str, items: str, width: int = 72) -> list[str]:
    """``prefix`` followed by a long integer list, split over continuation lines (a trailing comma continues a
    WARP3D command; a single long line is read only up to the scanner's column limit)."""
    tokens = items.split()
    lines: list[str] = []
    current = prefix
    for tok in tokens:
        if len(current) + 1 + len(tok) > width:
            lines.append(current + ",")
            current = "    " + tok
        else:
            current = current + " " + tok
    lines.append(current)
    return lines


NAME_COMMANDS = ("structure", "material", "loading", "domain", "step")  # commands followed by a name or number
CONTINUATION_PREFIXES = ("nu ", "yld_pt ", "n_power ")  # property continuation lines of the material commands


def command_keywords(deck_text: str) -> set[str]:
    """The command keyword of every command line of a deck (first two words, lower case; the name-bearing
    commands reduced to their first word), for the exact COMMANDS.md audit. A comment line is one whose first
    token is exactly 'c' or 'C' or that starts with '!' (so 'compute ...' and 'convergence ...' are audited);
    data lines (a leading integer, the continuation of a wrapped list, a property continuation) are not."""
    out: set[str] = set()
    continuing = False
    for raw in deck_text.splitlines():
        line = raw.split("$")[0].rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        if stripped.startswith("*echo"):
            out.add("*echo")
            continue
        words = stripped.replace(",", " ").split()
        if words[0] in ("c", "C"):
            continue
        was_continuing, continuing = continuing, line.endswith(",")
        if was_continuing or words[0].isdigit() or stripped.startswith(CONTINUATION_PREFIXES):
            continue
        if stripped.startswith("*input"):
            out.add("*input")
            continue
        if words[0].lower() in NAME_COMMANDS:
            out.add(words[0].lower())
            continue
        out.add(" ".join(words[:2]).lower() if len(words) > 1 else words[0].lower())
    return out
