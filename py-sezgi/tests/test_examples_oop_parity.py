"""Bit-exact parity: each OOP twin reproduces its pure script's output
fields at the shared scenario (same seed => same RNG draw sequence =>
identical evals_used/best_f/gap, compared as printed strings)."""
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PAIRS = ["gwo", "woa", "hs", "cs", "goa", "sca", "jaya", "mfo", "ssa",
         "fa", "ba", "fpa", "tlbo"]
PAIRS += ["hho", "alo", "abc", "gsa"]

FIELDS = re.compile(r"evals_used=(\S+) best_f=(\S+) gap=(\S+)")


def run_fields(script):
    out = subprocess.run([sys.executable, str(script)], capture_output=True,
                         text=True, check=True, cwd=ROOT).stdout
    m = FIELDS.search(out)
    assert m, f"no metrics line in {script}: {out!r}"
    return m.groups()


@pytest.mark.parametrize("algo", PAIRS)
def test_oop_twin_matches_pure_script(algo):
    pure = run_fields(ROOT / "examples" / "python" / f"{algo}.py")
    oop = run_fields(ROOT / "examples" / "python" / "oop" / f"{algo}.py")
    assert oop == pure
