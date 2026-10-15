"""Build the executed showcase notebook ``examples/showcase.ipynb`` (with its outputs and figures) from
``examples/showcase.py``.

Usage: ``uv run --with nbclient --with ipykernel python tools/build_showcase.py``
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location("make_notebooks", ROOT / "tools" / "make_notebooks.py")
assert spec is not None and spec.loader is not None
mk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mk)

nb = nbformat.reads(json.dumps(mk.convert((ROOT / "examples" / "showcase.py").read_text(encoding="utf-8"))), 4)
run_in = {"metadata": {"path": str(ROOT / "examples")}}
NotebookClient(nb, timeout=600, kernel_name="python3", resources=run_in).execute()
nbformat.write(nb, ROOT / "examples" / "showcase.ipynb")
print("wrote examples/showcase.ipynb")
