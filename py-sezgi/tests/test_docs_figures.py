"""M4-3 Task 7: sidecar-data gate for the 8 docs figure generators under
`docs/scripts/fig_*.py`.

Per the docs-and-example-migration research (§E4 "Figures: deterministic
matplotlib"), this gate is DATA-gated, not PIXEL-gated: matplotlib's own
PNG/SVG output shifts across patch releases and font availability, so
byte-comparing rendered images would produce false failures unrelated to
any real regression. Every generator script instead separates
`compute_data()` (pure, seeded, sezgi-only -- no matplotlib import) from
`render(data)` (matplotlib-only); this file re-runs ONLY `compute_data()`
for each figure and asserts it reproduces the exact committed JSON
sidecar under `docs/site/assets/figures/<name>.json` byte-for-byte-equal
as Python values (every number here comes from sezgi's own deterministic
engine under a fixed seed, so exact equality -- not an approximate
tolerance -- is the right assertion, matching this project's existing
anchored-example convention, e.g. `test_examples_engine.py`).

A second, separate check confirms each figure's own committed PNG exists
and is non-empty -- catching an accidentally un-committed or emptied
image file, which the data-only check above cannot see.
"""
import importlib.util
import json
import sys

import pytest

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "docs" / "scripts"
ASSETS_DIR = ROOT / "docs" / "site" / "assets" / "figures"

FIGURE_NAMES = [
    "convergence_curve",
    "multiseed_band",
    "explore_exploit",
    "encoding_space",
    "feature_mask_recovery",
    "nsga2_pareto_front",
    "stats_comparison",
    "budget_anytime_curve",
]


def _load_script_module(name):
    """Imports `docs/scripts/fig_<name>.py` as a fresh module. The script
    itself does `from _figutil import ...` (a same-directory import,
    exactly how it is meant to be run standalone:
    `py-sezgi/.venv/bin/python docs/scripts/fig_<name>.py`) -- so
    `SCRIPTS_DIR` is put on `sys.path` first, matching that same
    directory-relative resolution."""
    script_path = SCRIPTS_DIR / f"fig_{name}.py"
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(f"docs_fig_{name}", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", FIGURE_NAMES)
def test_figure_data_matches_committed_sidecar(name):
    """Re-running `fig_<name>.py`'s own `compute_data()` reproduces the
    committed `<name>.json` sidecar exactly -- the figure's data is
    regenerated, not the PNG (see module docstring)."""
    module = _load_script_module(name)
    recomputed = module.compute_data()

    sidecar_path = ASSETS_DIR / f"{name}.json"
    assert sidecar_path.exists(), (
        f"missing committed sidecar for figure {name!r}: {sidecar_path}")
    with open(sidecar_path) as f:
        committed = json.load(f)

    # Round-trip the recomputed data through JSON too, so both sides are
    # compared as plain JSON-native types (e.g. tuples -> lists), exactly
    # as `write_sidecar` itself would serialize them.
    recomputed_via_json = json.loads(json.dumps(recomputed))
    assert recomputed_via_json == committed, (
        f"figure {name!r}: recomputed data no longer matches the "
        f"committed sidecar -- regenerate it with "
        f"`py-sezgi/.venv/bin/python docs/scripts/fig_{name}.py` and "
        f"commit the new sidecar (and PNG) if the change is intentional")


@pytest.mark.parametrize("name", FIGURE_NAMES)
def test_figure_png_exists(name):
    """Each figure's own committed PNG exists and is non-empty -- catches
    an accidentally un-committed or emptied image file, which the
    data-only sidecar check above cannot see."""
    png_path = ASSETS_DIR / f"{name}.png"
    assert png_path.exists(), f"missing committed PNG for figure {name!r}: {png_path}"
    assert png_path.stat().st_size > 0, f"figure {name!r}'s PNG is empty: {png_path}"
