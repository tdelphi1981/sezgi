"""Execution gate for the four executed Jupyter notebook tutorials under
`docs/site/notebooks/`.

Each notebook ships with COMMITTED outputs (the mkdocs-jupyter plugin
renders them with `execute: false` -- see `mkdocs.yml` -- so the docs
build never re-runs them). This test re-executes each notebook headless,
against the SAME `py-sezgi/.venv` interpreter the rest of this test suite
runs under, and asserts every cell completes with no error -- catching a
notebook that has silently drifted out of sync with the library (an API
rename, a signature change, ...) the way the docs build itself cannot,
since the docs build only renders already-committed outputs and never
imports `sezgi` to check them.

This does not assert the re-executed outputs are byte-identical to the
committed ones (unlike `test_docs_figures.py`'s sidecar gate): several
cells here print floating-point results from real `sezgi` runs, which are
bit-exact under a fixed seed on a given build but not asserted equal
across arbitrary environments in this test. The notebooks' OWN
determinism-check cells (comparing two `.run()` calls within the same
process) already assert bit-for-bit equality directly.
"""
import pathlib

import nbformat as nbf
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = pathlib.Path(__file__).resolve().parents[2]
NOTEBOOKS_DIR = ROOT / "docs" / "site" / "notebooks"

NOTEBOOK_NAMES = [
    "01-interactive-quickstart.ipynb",
    "02-your-own-algorithm.ipynb",
    "03-benchmarking-and-statistics.ipynb",
    "04-metaheuristics-101.ipynb",
]

# The kernelspec these notebooks are committed with (see
# `docs/site/notebooks/*.ipynb`'s own `metadata.kernelspec.name`) --
# registered against this project's own `py-sezgi/.venv` interpreter by
# `py-sezgi/.venv/bin/python -m ipykernel install --user --name
# sezgi-docs-venv`, the same interpreter this test itself runs under.
KERNEL_NAME = "sezgi-docs-venv"

TIMEOUT_SECONDS = 300


@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_notebook_executes_without_error(name):
    path = NOTEBOOKS_DIR / name
    assert path.exists(), f"missing notebook: {path}"

    nb = nbf.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=TIMEOUT_SECONDS,
        kernel_name=KERNEL_NAME,
        resources={"metadata": {"path": str(NOTEBOOKS_DIR)}},
    )
    try:
        client.execute()
    except CellExecutionError as exc:
        pytest.fail(f"notebook {name!r} raised a cell execution error:\n{exc}")

    for index, cell in enumerate(nb.cells):
        if cell.get("cell_type") != "code":
            continue
        errors = [
            output for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        assert not errors, (
            f"notebook {name!r} cell {index} produced error output: {errors}")
