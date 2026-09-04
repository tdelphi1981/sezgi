"""Shared plumbing for the docs figure generators (M4-3 Task 7).

Every `docs/scripts/fig_*.py` module is a standalone, seeded script: run
directly (`py-sezgi/.venv/bin/python docs/scripts/fig_NAME.py`) it writes a
PNG under `docs/site/assets/figures/` and a JSON "sidecar" of the exact
plotted series next to it. The sidecar -- not the PNG -- is what
`py-sezgi/tests/test_docs_figures.py` gates: matplotlib's own pixel output
shifts across patch releases and font availability,
so re-running a generator and byte-comparing the PNG would produce false
failures. The JSON sidecar is pinned instead -- every number in it comes
from a fixed seed through sezgi's own deterministic engine, so it is
reproduced exactly on every re-run.
"""
import json
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent / "site" / "assets" / "figures"


def write_sidecar(name, data, out_dir=ASSETS_DIR):
    """Writes `data` (must be JSON-serializable) to `<out_dir>/<name>.json`,
    sorted keys, stable formatting -- so a re-run's diff (if any) is a real
    content diff, not key-order noise."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def png_path(name, out_dir=ASSETS_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{name}.png"
