"""Red-team partner CLI: sends artefacts to a different model via OpenRouter.

The partner only ever sees the artefact, the spec/context files passed on the
command line and (for cross-examination / sign-off) other parties' findings.
It never sees the builder's reasoning.

Usage examples:
    python tools/adversary/partner.py review-doc --phase phase0 --round 1 \
        --artefact docs/proposal.md --context reviews/phase0/context.md
    python tools/adversary/partner.py review-diff --phase phase1 --round 1 \
        --artefact reviews/phase1/r1.diff --context docs/spec.md
    python tools/adversary/partner.py cross-examine --phase phase0 --round 1 \
        --artefact docs/proposal.md --findings reviews/phase0/r1-reviewer-b.json
    python tools/adversary/partner.py derive-oracle --phase phase1 --name newman_raju \
        --artefact reviews/phase1/oracle-request-newman_raju.md
    python tools/adversary/partner.py signoff --phase phase0 --round 2 \
        --artefact docs/spec.md --findings reviews/phase0/responses.json
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVIEWS = ROOT / "reviews"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-6.1-sol-pro"

FINDING_SCHEMA = """Return a single JSON object:
{
  "summary": "<3-6 sentences: overall verdict>",
  "findings": [
    {
      "id": "<short unique id, e.g. P-01>",
      "severity": "blocker | major | minor",
      "location": "<section / file:line / equation the finding is about>",
      "claim": "<what is wrong, stated so it can be checked>",
      "evidence": "<derivation, counter-example, numbers or citation supporting the claim>",
      "proposed_fix": "<concrete change>"
    }
  ]
}
Severity: blocker = produces wrong or unsafe results or makes the work infeasible;
major = materially weakens correctness, verification or feasibility; minor = the rest.
Do not pad the list. Do not report style. Every finding must be checkable."""

SYSTEM = {
    "review-doc": (
        "You are an adversarial technical reviewer of a plan/specification for an engineering "
        "software library (fracture mechanics, structural reliability, FE, Gaussian-process "
        "surrogates). Your job is to find what is wrong: incorrect or imprecise equations, "
        "algorithms that fail on edge cases, statistical errors, infeasible steps, verification "
        "gaps, wrong ordering, missing risks. Work the mathematics; give counter-examples with "
        "numbers. Assume the authors are competent and look for the errors competent authors "
        "make. " + FINDING_SCHEMA
    ),
    "review-diff": (
        "You are an adversarial code reviewer for a numerical engineering library. You are given "
        "a spec/context and code (a diff or whole files). Find correctness bugs: wrong equations "
        "versus the supplied reference equations, transcription errors in coefficients, unit "
        "errors, broadcasting/vectorisation bugs, edge cases (zeros, bounds, non-convergence), "
        "statistical errors, tests that cannot fail or that test the implementation against "
        "itself. Give a concrete failing input for each finding where possible. "
        + FINDING_SCHEMA
    ),
    "cross-examine": (
        "You are an adversarial reviewer cross-examining ANOTHER reviewer's findings about an "
        "artefact. For each finding decide whether it is correct. Try to refute it; confirm it "
        "only if it survives. Then add any findings both of you missed. Return a single JSON "
        'object: {"verdicts": [{"id": "<their id>", "verdict": "confirmed | refuted | '
        'partially-correct", "reasoning": "<checkable argument>", "severity": "blocker | major '
        '| minor"}], "new_findings": [<findings in the schema below>]}. ' + FINDING_SCHEMA
    ),
    "derive-oracle": (
        "You are an independent implementer. From the supplied equations and cited primary "
        "reference ONLY, produce test oracles. You have not seen and must not assume anything "
        "about any other implementation. Return a single JSON object: "
        '{"reference_implementation": "<complete, self-contained Python (numpy only) module '
        'source>", "cases": [{"name": "...", "inputs": {...}, "expected": {...}, '
        '"how_obtained": "<hand calculation steps or published value with source>"}], '
        '"doubts": ["<anything in the supplied equations that looks wrong, ambiguous or '
        'mistranscribed, with reasons>"]}. Compute expected values carefully to at least 8 '
        "significant figures; the cases will be executed against your reference implementation "
        "and against a second independent implementation."
    ),
    "signoff": (
        "You are an adversarial reviewer deciding whether your earlier findings have been "
        "adequately addressed. You are given the revised artefact and the builder's responses "
        "(accept+fix or rebuttal with evidence). Do not accept a rebuttal without checkable "
        'evidence. Return a single JSON object: {"decisions": [{"id": "...", "status": "closed '
        '| still-open", "reasoning": "..."}], "new_findings": [<schema below>], "signoff": '
        "true | false}. signoff is true only if no blocker or major item is still open. "
        + FINDING_SCHEMA
    ),
}


def load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def read_labelled(paths: list[str], label: str) -> str:
    parts = []
    for p in paths:
        text = Path(p).read_text(encoding="utf-8")
        parts.append(f"===== {label}: {p} =====\n{text}\n===== END {p} =====")
    return "\n\n".join(parts)


def call(model: str, system: str, user: str, max_tokens: int) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens,
            "usage": {"include": True},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        sys.exit(f"OpenRouter HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:2000]}")


def log_spend(phase: str, command: str, model: str, usage: dict) -> None:
    REVIEWS.mkdir(exist_ok=True)
    path = REVIEWS / "spend.csv"
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(
                ["timestamp", "phase", "command", "model", "prompt_tokens", "completion_tokens", "cost_usd"]
            )
        writer.writerow(
            [
                dt.datetime.now().isoformat(timespec="seconds"),
                phase,
                command,
                model,
                usage.get("prompt_tokens", ""),
                usage.get("completion_tokens", ""),
                usage.get("cost", ""),
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=sorted(SYSTEM))
    parser.add_argument("--phase", required=True)
    parser.add_argument("--round", default="1")
    parser.add_argument("--name", default="", help="suffix for the output file name")
    parser.add_argument("--artefact", nargs="+", required=True, help="file(s) under review")
    parser.add_argument("--context", nargs="*", default=[], help="spec / reference files")
    parser.add_argument("--findings", nargs="*", default=[], help="findings or responses files")
    parser.add_argument("--focus", default="", help="extra instruction for this call")
    parser.add_argument("--max-tokens", type=int, default=32000)
    args = parser.parse_args()

    load_env()
    model = os.environ.get("PARTNER_MODEL", DEFAULT_MODEL)

    sections = []
    if args.context:
        sections.append(read_labelled(args.context, "CONTEXT"))
    sections.append(read_labelled(args.artefact, "ARTEFACT"))
    if args.findings:
        sections.append(read_labelled(args.findings, "FINDINGS/RESPONSES"))
    if args.focus:
        sections.append(f"===== FOCUS =====\n{args.focus}")
    user = "\n\n".join(sections)

    response = call(model, SYSTEM[args.command], user, args.max_tokens)
    if "choices" not in response:
        sys.exit(f"Unexpected response: {json.dumps(response)[:2000]}")
    content = response["choices"][0]["message"]["content"] or ""
    log_spend(args.phase, args.command, model, response.get("usage", {}))

    out_dir = REVIEWS / args.phase
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"r{args.round}-partner-{args.command}" + (f"-{args.name}" if args.name else "")
    try:
        parsed = json.loads(content)
        out = out_dir / f"{stem}.json"
        out.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")
    except json.JSONDecodeError:
        out = out_dir / f"{stem}.raw.txt"
        out.write_text(content, encoding="utf-8")
        print("WARNING: response was not valid JSON; saved raw text", file=sys.stderr)
    usage = response.get("usage", {})
    print(f"{out.relative_to(ROOT)}  model={response.get('model', model)}  "
          f"in={usage.get('prompt_tokens')} out={usage.get('completion_tokens')} cost=${usage.get('cost')}")


if __name__ == "__main__":
    main()
