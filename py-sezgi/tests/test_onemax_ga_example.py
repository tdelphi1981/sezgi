"""M3-8 Task 11: determinism/anchored-output gate for
examples/python/onemax_ga.py -- the ga_bin/OneMax matched example pair for
the typed-operator milestone.

Follows the exact same "run the script as a subprocess, parse its printed
metrics line" pattern test_tsp_two_opt_example.py (M3-8 Task 7) and
test_examples_oop_parity.py's own `run_fields` helper already use, so the R
twin (examples/r/onemax_ga.R, r-sezgi/tests/testthat/test-onemax-ga.R) can
anchor to the SAME regex-extracted fields.
"""
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "python" / "onemax_ga.py"

FIELDS = re.compile(
    r"evals_used=(\S+) best_f=(\S+) gap=(\S+) ones=(\S+)")


def run_fields():
    out = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True,
                         text=True, check=True, cwd=ROOT).stdout
    m = FIELDS.search(out)
    assert m, f"no metrics line in {SCRIPT}: {out!r}"
    return m.groups()


def test_onemax_ga_example_runs_and_reports_valid_metrics():
    evals_used, best_f, gap, ones = run_fields()
    assert int(evals_used) == 400
    assert best_f == gap, "OneMax's known optimum is 0.0, so gap must equal best_f exactly"
    assert 0 <= int(ones) <= 100


def test_onemax_ga_example_is_deterministic_across_runs():
    a = run_fields()
    b = run_fields()
    assert a == b, "same seed/n_bits/budget must reproduce identical output"


def test_onemax_ga_example_anchored_output():
    """ANCHORED (per this project's convention): measured with N_BITS=100,
    POP_SIZE=20, BUDGET=400, SEED=7 (onemax_ga.py's own constants).
    evals_used lands exactly on budget (20 generations past init); best_f=16
    (16 of 100 bits still zero) is a genuine partial result, deliberately
    NOT the trivial fully-converged optimum (see the script's own module
    doc for why a non-converging parameter set was chosen -- the same
    reasoning M3-8 Task 10's own cross-language anchor report gives).
    examples/r/onemax_ga.R is meant to reproduce this SAME best_f/gap/ones
    output (numbers bit-identical across languages, formatting divergence
    acceptable, per the M3-7 example-pair "N2" clarification).
    """
    evals_used, best_f, gap, ones = run_fields()
    assert evals_used == "400"
    assert best_f == "16"
    assert gap == "16"
    assert ones == "84"
