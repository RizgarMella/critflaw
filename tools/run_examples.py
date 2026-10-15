"""Run the worked examples one at a time (never in parallel) and report pass/fail and wall time.

Usage: ``python tools/run_examples.py [pattern ...]`` (default: every ``examples/[0-9]*.py``). Exits non-zero if any
example fails. Each example runs headless in its own process with a time limit.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIME_LIMIT_S = 600


def main(patterns: list[str]) -> int:
    files = sorted({f for p in (patterns or ["[0-9]*.py"]) for f in (ROOT / "examples").glob(p)})
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    failed = []
    for f in files:
        t0 = time.perf_counter()
        try:
            r = subprocess.run(
                [sys.executable, str(f)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=TIME_LIMIT_S
            )
            ok, tail = r.returncode == 0, (r.stdout + r.stderr).strip().splitlines()[-15:]
        except subprocess.TimeoutExpired:
            ok, tail = False, [f"timed out after {TIME_LIMIT_S} s"]
        dt = time.perf_counter() - t0
        print(f"{'PASS' if ok else 'FAIL'} {dt:7.1f} s  {f.name}", flush=True)
        if not ok:
            failed.append(f.name)
            print("    " + "\n    ".join(tail), flush=True)
    print(f"{len(files) - len(failed)}/{len(files)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
