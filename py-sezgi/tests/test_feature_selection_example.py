"""M4-1 Task 6: determinism/anchored-output gate for
examples/python/oop/feature_selection.py -- the FeatureSelection data
recipe's own worked example.

Follows the exact same "run the script as a subprocess, parse its printed
metrics line" pattern test_tsp_two_opt_example.py / test_onemax_ga_example.py
already use.
"""
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "python" / "oop" / "feature_selection.py"

FIELDS = re.compile(
    r"evals_used=(\S+) best_f=(\S+) popcount=(\S+) mask=(\S+) recovered=(\S+)")


def run_fields():
    out = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True,
                         text=True, check=True, cwd=ROOT).stdout
    m = FIELDS.search(out)
    assert m, f"no metrics line in {SCRIPT}: {out!r}"
    return m.groups()


def test_feature_selection_example_runs_and_reports_valid_metrics():
    evals_used, best_f, popcount, mask, recovered = run_fields()
    assert int(evals_used) == 200
    assert int(popcount) == mask.count("1")
    assert recovered == "True"


def test_feature_selection_example_is_deterministic_across_runs():
    a = run_fields()
    b = run_fields()
    assert a == b, "same seed/dataset/budget must reproduce identical output"


def test_feature_selection_example_anchored_output():
    """ANCHORED (per this project's convention): measured with this
    script's own constants (SEED=98, N_SAMPLES=20, N_FEATURES=8,
    INFORMATIVE=(1, 3, 6), PENALTY=0.3, POP_SIZE=20, BUDGET=200,
    GA_SEED=42). evals_used lands exactly on budget; best_f is the OLS
    residual-sum-of-squares (plus the popcount penalty) at the recovered
    mask; mask=01010010 is exactly the informative columns (indices 1, 3,
    6, 0-indexed, left to right) and no others -- recovered=True. Also
    cross-checked in py-sezgi/tests/test_oop_recipes.py by a full 2**8
    brute-force enumeration confirming this mask is the UNIQUE global
    minimum under this penalty, not merely what the GA happened to find.
    """
    evals_used, best_f, popcount, mask, recovered = run_fields()
    assert evals_used == "200"
    assert best_f == "0.3607495483"
    assert popcount == "3"
    assert mask == "01010010"
    assert recovered == "True"
