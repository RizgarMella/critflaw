"""Convert the percent-format example scripts (``examples/*.py``) into Jupyter notebooks (``examples/notebooks/``).

Cells are separated by ``# %%``; ``# %% [markdown]`` starts a markdown cell whose lines are ``# ``-prefixed comments.
No third-party dependency: the notebook is written as nbformat 4 JSON. Usage: ``python tools/make_notebooks.py``
(``--check`` exits non-zero if any notebook is missing or out of date).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "examples"
OUT = SRC / "notebooks"


def _cell(kind: str, lines: list[str]) -> dict[str, object] | None:
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return None
    if kind == "markdown":
        lines = [ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in lines]
    source = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
    cell: dict[str, object] = {"cell_type": kind, "metadata": {}, "source": source}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def convert(text: str) -> dict[str, object]:
    """The nbformat 4.5 notebook for one percent-format script (cell ids are their index, so output is stable)."""
    cells: list[dict[str, object]] = []
    kind, buf = "code", []
    for line in text.splitlines():
        if line.startswith("# %%"):
            if (c := _cell(kind, buf)) is not None:
                cells.append(c)
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    if (c := _cell(kind, buf)) is not None:
        cells.append(c)
    for i, c in enumerate(cells):
        c["id"] = f"cell-{i}"
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main(check: bool = False) -> int:
    OUT.mkdir(exist_ok=True)
    stale = []
    for src in sorted(SRC.glob("[0-9]*.py")):
        target = OUT / (src.stem + ".ipynb")
        new = json.dumps(convert(src.read_text(encoding="utf-8")), indent=1, ensure_ascii=False) + "\n"
        if check:
            if not target.exists() or target.read_text(encoding="utf-8") != new:
                stale.append(target.name)
        else:
            target.write_text(new, encoding="utf-8", newline="\n")
    if stale:
        print("out of date:", ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv))
