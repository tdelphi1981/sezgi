"""M3-8 Task 7: determinism/anchored-output gate for
examples/python/oop/tsp_two_opt.py -- the first worked example over a
permutation-typed problem.

This example sits OUTSIDE test_examples_oop_parity.py's 17-pair PAIRS
twin gate (see tsp_two_opt.py's own module docstring for why: it has no
pure-script counterpart to reproduce -- it IS the example, not a port of
one). It gets its own gate here instead, following the exact same
"run the script as a subprocess, parse its printed metrics line" pattern
test_examples_oop_parity.py's own `run_fields` helper uses, so a future
Task 8 R twin can anchor to the SAME regex-extracted fields.
"""
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "python" / "oop" / "tsp_two_opt.py"

FIELDS = re.compile(
    r"evals_used=(\S+) best_f=(\S+) gap=(\S+) tour_length=(\S+)")


def run_fields():
    out = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True,
                         text=True, check=True, cwd=ROOT).stdout
    m = FIELDS.search(out)
    assert m, f"no metrics line in {SCRIPT}: {out!r}"
    return m.groups()


def test_tsp_two_opt_example_runs_and_reports_valid_metrics():
    evals_used, best_f, gap, tour_length = run_fields()
    assert int(evals_used) == 2000
    assert best_f == tour_length, "best_f and the recomputed tour_length must agree"
    assert float(gap) >= 0.0, "cannot beat the published berlin52 optimum"


def test_tsp_two_opt_example_is_deterministic_across_runs():
    a = run_fields()
    b = run_fields()
    assert a == b, "same seed/instance/budget must reproduce identical output"


def test_tsp_two_opt_example_anchored_output():
    """ANCHORED (per this project's convention): measured with
    INSTANCE="berlin52", BUDGET=2000, SEED=42 (tsp_two_opt.py's own
    constants). evals_used lands exactly on budget (one setup() eval plus
    budget-1 single-eval steps, per the script's own docstring); best_f is
    the first-improvement 2-opt sweep's final tour length from that one
    random start, well above the published optimum of 7542.0 (a short,
    single-restart local search is not expected to reach it) but strictly
    better than the identity-order tour's own random start would be before
    any improving move is found. Task 8's R twin (examples/r/oop/
    tsp_two_opt.R) is meant to reproduce this SAME best_f/tour_length pair
    (language-neutral numbers, see tsp_two_opt.py's module doc) despite its
    own 1-based tour indexing.
    """
    evals_used, best_f, gap, tour_length = run_fields()
    assert evals_used == "2000"
    assert best_f == "9077"
    assert gap == "1535"
    assert tour_length == "9077"
