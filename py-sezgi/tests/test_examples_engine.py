"""M4-3 Task 2: determinism/anchored-output gate for the five NEW
engine-hosted examples under `examples/python/oop/engine/` -- the
class-first front door's flagship authoring examples (an
`Algorithm`/`PopulationAlgorithm`/`LocalSearch` subclass, or a `Problem`
subclass driven by a built-in wrapper class). Same "run the script as a
subprocess, parse its printed metrics line" convention as
`test_tsp_two_opt_example.py`/`test_oop_families.py`'s own example-runner
tests (mechanics copied from there): each script already asserts its own
anchors internally (a non-zero exit means either an uncaught exception or
one of the script's own failed `assert`s), so this file's own job is (a)
confirm a clean exit, (b) confirm determinism across two independent
subprocess runs, and (c) re-assert at least one pinned value per script
from Python (not just trusting the subprocess's own internal assert),
against the SAME anchors each script's own module-level comment records
(measured by running the script once and pinning the printed value, this
project's own anchored-example convention).
"""
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
ENGINE_DIR = ROOT / "examples" / "python" / "oop" / "engine"

FIELDS = re.compile(r"evals_used=(\S+) best_f=(\S+) gap=(\S+)")
TSP_FIELDS = re.compile(r"evals_used=(\S+) best_f=(\S+) tour_length=(\S+)")


def _run(script):
    """Runs `script` as a subprocess, returns (returncode, stdout)."""
    proc = subprocess.run([sys.executable, str(script)], capture_output=True,
                          text=True, cwd=ROOT)
    return proc.returncode, proc.stdout


def _run_ok(script):
    """Runs `script`, asserts a clean exit (the script's own internal
    asserts did not fire), returns its stdout."""
    code, out = _run(script)
    assert code == 0, f"{script.name} exited {code}, stdout={out!r}"
    return out


# ---------------------------------------------------------------------
# custom_de.py: PopulationAlgorithm, DE/rand/2/bin vary(), tournament
# select() (inherited, not overridden).
# ---------------------------------------------------------------------

def test_custom_de_example_runs_and_is_deterministic():
    script = ENGINE_DIR / "custom_de.py"
    out_a = _run_ok(script)
    out_b = _run_ok(script)
    assert out_a == out_b, "same seed/problem/budget must reproduce identical output"


def test_custom_de_example_anchored_output():
    script = ENGINE_DIR / "custom_de.py"
    out = _run_ok(script)
    m = FIELDS.search(out)
    assert m, f"no metrics line in {script}: {out!r}"
    evals_used, best_f, gap = m.groups()
    assert evals_used == "2000"
    assert best_f == "-84.3855"
    assert gap == "0.0184872"


# ---------------------------------------------------------------------
# custom_pso_variant.py: Algorithm, full generate() override, PSO-flavored
# velocity update with per-instance state.
# ---------------------------------------------------------------------

def test_custom_pso_variant_example_runs_and_is_deterministic():
    script = ENGINE_DIR / "custom_pso_variant.py"
    out_a = _run_ok(script)
    out_b = _run_ok(script)
    assert out_a == out_b


def test_custom_pso_variant_example_anchored_output():
    script = ENGINE_DIR / "custom_pso_variant.py"
    out = _run_ok(script)
    m = FIELDS.search(out)
    assert m, f"no metrics line in {script}: {out!r}"
    evals_used, best_f, gap = m.groups()
    assert evals_used == "2000"
    assert best_f == "-76.9567"
    assert gap == "7.4472"


# ---------------------------------------------------------------------
# local_search_2opt.py: LocalSearch, random-2-opt neighbor() over a
# hand-authored permutation-space Problem (no known optimum -> tour_length
# instead of gap, same field convention as ../tsp_two_opt.py's own gate).
# ---------------------------------------------------------------------

def test_local_search_2opt_example_runs_and_is_deterministic():
    script = ENGINE_DIR / "local_search_2opt.py"
    out_a = _run_ok(script)
    out_b = _run_ok(script)
    assert out_a == out_b


def test_local_search_2opt_example_anchored_output():
    script = ENGINE_DIR / "local_search_2opt.py"
    out = _run_ok(script)
    m = TSP_FIELDS.search(out)
    assert m, f"no metrics line in {script}: {out!r}"
    evals_used, best_f, tour_length = m.groups()
    assert evals_used == "1500"
    assert best_f == "33.3313"
    assert best_f == tour_length, "best_f and the recomputed tour_length must agree"


# ---------------------------------------------------------------------
# custom_problem_rastrigin.py: Problem authoring (Rastrigin, pure stdlib
# math) solved by a built-in wrapper class (GreyWolfOptimizer).
# ---------------------------------------------------------------------

def test_custom_problem_rastrigin_example_runs_and_is_deterministic():
    script = ENGINE_DIR / "custom_problem_rastrigin.py"
    out_a = _run_ok(script)
    out_b = _run_ok(script)
    assert out_a == out_b


def test_custom_problem_rastrigin_example_anchored_output():
    script = ENGINE_DIR / "custom_problem_rastrigin.py"
    out = _run_ok(script)
    m = FIELDS.search(out)
    assert m, f"no metrics line in {script}: {out!r}"
    evals_used, best_f, gap = m.groups()
    assert evals_used == "3000"
    assert best_f == "1.42109e-14"
    assert gap == best_f, "gap must equal best_f when the known optimum is 0.0"


# ---------------------------------------------------------------------
# mixed_space_tuning.py: mixed-space (Float + Categorical + Int) Problem +
# Algorithm handling the per-block tuple payload; also demonstrates
# validate_space's build-time veto.
# ---------------------------------------------------------------------

def test_mixed_space_tuning_example_runs_and_is_deterministic():
    script = ENGINE_DIR / "mixed_space_tuning.py"
    out_a = _run_ok(script)
    out_b = _run_ok(script)
    assert out_a == out_b


def test_mixed_space_tuning_example_anchored_output():
    script = ENGINE_DIR / "mixed_space_tuning.py"
    out = _run_ok(script)
    m = FIELDS.search(out)
    assert m, f"no metrics line in {script}: {out!r}"
    evals_used, best_f, gap = m.groups()
    assert evals_used == "3000"
    assert best_f == "0.00151122"
    assert gap == best_f


def test_mixed_space_tuning_example_demonstrates_validate_space_veto():
    script = ENGINE_DIR / "mixed_space_tuning.py"
    out = _run_ok(script)
    assert "validate_space_veto=True" in out, (
        "the script's own validate_space() demo must have vetoed the "
        f"Float-only space it points its algorithm at: {out!r}")


# ---------------------------------------------------------------------
# All five scripts, gathered: every file exits 0 and prints its own label.
# ---------------------------------------------------------------------

SCRIPTS = ["custom_de.py", "custom_pso_variant.py", "local_search_2opt.py",
           "custom_problem_rastrigin.py", "mixed_space_tuning.py"]


def test_every_engine_example_exits_cleanly_and_prints_its_own_label():
    for name in SCRIPTS:
        script = ENGINE_DIR / name
        out = _run_ok(script)
        label = script.stem
        assert label in out, f"{script} did not print its own label {label!r}: {out!r}"
