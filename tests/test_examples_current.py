"""The notebooks in examples/notebooks/ are generated from the example scripts and must be current."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_notebooks_match_the_example_scripts() -> None:
    spec = importlib.util.spec_from_file_location("make_notebooks", ROOT / "tools" / "make_notebooks.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(check=True) == 0, "run: python tools/make_notebooks.py"
