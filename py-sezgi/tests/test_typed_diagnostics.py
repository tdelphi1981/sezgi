"""M3-8 Task 9: Python bindings for the typed operator families and the
diagnostic problem trio.

Exposes:
  - sezgi.problems.onemax/int_quadratic/cat_match -- Problem handles for
    the diagnostic trio (crates/problems/src/diagnostics.rs, Task 5):
    minimal, hand-verifiable, single-global-optimum landscapes that exist
    to make ga_bin/ga_int/ga_cat reachable and testable from Python, NOT
    benchmark targets (that module's own "diagnostic, not benchmark"
    framing).
  - sezgi.presets.ga_bin/ga_int/ga_cat (Tasks 2-4), pairing 1:1 with the
    diagnostics above.
  - solve() end to end against each pairing, at the SAME (n/lo/hi/k,
    pop_size, budget, master_seed) anchors
    crates/problems/tests/solve_typed_presets.rs uses (Task 5's own
    "each typed preset solves its diagnostic" Rust test) -- solve() runs
    the IDENTICAL Engine::from_spec/Engine::run core those Rust tests do,
    so the same inputs must reach the SAME exact best_f == 0.0.
  - sezgi.problems.mixed_diagnostic(...) -- a Task 9 addition (NOT one of
    Task 5's brief-pinned diagnostics) proving gen/compound (Task 5) is
    reachable end to end from a Python-authored, mixed-space TOML
    AlgorithmSpec, parsed via the stdlib's own tomllib into the dict
    solve() already accepts (no new solve-side API).

Typed-result-genotype design decision (implemented + documented in
py-sezgi/src/lib.rs's solve(), restated here): a SINGLE-block genotype's
best_x is a flat list in its block's own natural Python type -- Float stays
list[float] (byte-identical to every result before this task), Perm stays
list[int], Int -> list[int], Cat -> list[int] (category INDICES 0..k, no
label concept), Bin -> list[bool] (Python's native boolean, one per bit). A
MULTI-block genotype (reachable only via mixed_diagnostic(...)) is a list of
per-block lists, one per SearchSpace block in order, each typed as above.
"""
import tomllib

import pytest
import sezgi


# ---------------------------------------------------------------------
# problems.onemax + presets.ga_bin
# ---------------------------------------------------------------------

def test_onemax_space_dim_and_optimum():
    p = sezgi.problems.onemax(20)
    assert p.dim() == 20
    assert p.optimum() == 0.0


def test_onemax_bounds_raises_value_error():
    """A Binary block has no (lo, hi) -- bounds() is Float-only (mirrors
    problems.tsp(...)'s own Permutation-space rejection)."""
    with pytest.raises(ValueError, match="continuous"):
        sezgi.problems.onemax(5).bounds()


def test_ga_bin_solves_onemax_to_exact_optimum():
    """Anchor mirrors crates/problems/tests/solve_typed_presets.rs::
    ga_bin_solves_one_max exactly: n_bits=20, pop_size=40, budget=6000,
    master_seed=1."""
    p = sezgi.problems.onemax(20)
    r = sezgi.solve(sezgi.presets.ga_bin(40, 6000), p, master_seed=1)
    assert r["best_f"] == 0.0
    assert len(r["best_x"]) == 20
    assert all(isinstance(b, bool) for b in r["best_x"])
    assert all(r["best_x"])  # the all-true genotype, at OneMax's optimum


# ---------------------------------------------------------------------
# problems.int_quadratic + presets.ga_int
# ---------------------------------------------------------------------

def test_int_quadratic_space_dim_and_optimum():
    p = sezgi.problems.int_quadratic(-10, 10, 5)
    assert p.dim() == 5
    assert p.optimum() == 0.0


def test_int_quadratic_lo_ge_hi_raises_value_error():
    with pytest.raises(ValueError, match="lo"):
        sezgi.problems.int_quadratic(5, 5, 3)
    with pytest.raises(ValueError, match="lo"):
        sezgi.problems.int_quadratic(5, -5, 3)


def test_ga_int_solves_int_quadratic_to_exact_optimum():
    """Anchor mirrors solve_typed_presets.rs::ga_int_solves_int_quadratic:
    lo=-10, hi=10, n=5, pop_size=40, budget=8000, master_seed=3."""
    p = sezgi.problems.int_quadratic(-10, 10, 5)
    r = sezgi.solve(sezgi.presets.ga_int(40, 8000), p, master_seed=3)
    assert r["best_f"] == 0.0
    assert len(r["best_x"]) == 5
    assert all(isinstance(x, int) for x in r["best_x"])


# ---------------------------------------------------------------------
# problems.cat_match + presets.ga_cat
# ---------------------------------------------------------------------

def test_cat_match_space_dim_and_optimum():
    p = sezgi.problems.cat_match(4, 8, 123)
    assert p.dim() == 8
    assert p.optimum() == 0.0


def test_ga_cat_solves_cat_match_to_exact_optimum():
    """Anchor mirrors solve_typed_presets.rs::ga_cat_solves_cat_match:
    k=4, n=8, seed=123, pop_size=40, budget=8000, master_seed=5."""
    p = sezgi.problems.cat_match(4, 8, 123)
    r = sezgi.solve(sezgi.presets.ga_cat(40, 8000), p, master_seed=5)
    assert r["best_f"] == 0.0
    assert len(r["best_x"]) == 8
    assert all(isinstance(c, int) and 0 <= c < 4 for c in r["best_x"])


def test_cat_match_different_construction_seed_yields_different_target():
    """CatMatch::new's seed IS caller-supplied (unlike int_quadratic's
    fixed internal seed) -- a different construction seed derives a
    different target, so two ga_cat runs each converging to their OWN
    problem's exact optimum (best_f == 0.0) land on different genotypes."""
    r1 = sezgi.solve(sezgi.presets.ga_cat(40, 8000),
                      sezgi.problems.cat_match(4, 8, 123), master_seed=5)
    r2 = sezgi.solve(sezgi.presets.ga_cat(40, 8000),
                      sezgi.problems.cat_match(4, 8, 999), master_seed=5)
    assert r1["best_f"] == 0.0
    assert r2["best_f"] == 0.0
    assert r1["best_x"] != r2["best_x"]


# ---------------------------------------------------------------------
# solve()/for_problem: no IOH identity, no ask/tell session, for the trio
# ---------------------------------------------------------------------

def test_solve_onemax_log_dir_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="log_dir"):
        sezgi.solve(sezgi.presets.ga_bin(8, 40), sezgi.problems.onemax(5),
                    master_seed=0, log_dir=str(tmp_path))


def test_for_problem_onemax_raises_value_error():
    """No ask/tell session type exists yet for Binary/Int/Categorical/
    mixed-typed spaces (out of this task's scope) -- EvalSession.for_problem
    rejects onemax(...)/int_quadratic(...)/cat_match(...)/
    mixed_diagnostic(...) explicitly rather than mis-building a Float
    session over a non-Float space."""
    with pytest.raises(ValueError, match="not currently supported"):
        sezgi.EvalSession.for_problem(sezgi.problems.onemax(5), budget=10)


# ---------------------------------------------------------------------
# gen/compound reachable via a Python-authored, mixed-space TOML
# AlgorithmSpec (Task 9's own reachability scaffold: mixed_diagnostic)
# ---------------------------------------------------------------------

MIXED_COMPOUND_TOML = """
name = "mixed-compound"
pop_size = 12

[init]
kind = "init/uniform"

[boundary]
kind = "boundary/clamp"

[[stages]]

[stages.generator]
kind = "gen/compound"
blocks = [
  { kind = "gen/ga-real", tournament_k = 2, pc = 0.9, eta_c = 15.0, eta_m = 20.0 },
  { kind = "gen/ga-int", tournament_k = 2, p_c = 0.9, eta_c = 15.0, eta_m = 20.0 },
  { kind = "gen/ga-cat", tournament_k = 2, p_c = 0.9 },
  { kind = "gen/ga-bin", tournament_k = 2, p_c = 0.9 },
]

[stages.replacer]
kind = "replace/mu-plus-lambda"

[termination]
budget = 200
"""


def test_gen_compound_mixed_space_toml_spec_solves_end_to_end():
    """Mirrors crates/components/src/compound.rs's own
    spec_validation_accepts_gen_compound_on_matching_mixed_space test's
    block layout (Float(2)+Int(2)+Categorical(k=3,n=2)+Binary(4)) and its
    compound_spec_json()'s exact sub-generator params -- proving the SAME
    mixed-space gen/compound spec is reachable from a Python-authored TOML
    document via the stdlib's own tomllib (no new solve-side API: solve()
    already accepts a dict, and TOML/JSON are just two serializations of
    the same AlgorithmSpec schema)."""
    spec = tomllib.loads(MIXED_COMPOUND_TOML)
    assert spec["stages"][0]["generator"]["kind"] == "gen/compound"

    problem = sezgi.problems.mixed_diagnostic(n_float=2, n_int=2, k_cat=3, n_cat=2, n_bin=4)
    r = sezgi.solve(spec, problem, master_seed=42)

    assert isinstance(r["best_f"], float)
    # Multi-block genotype: best_x is a list of 4 per-block lists, in
    # SearchSpace::blocks() order (Float, Int, Categorical, Binary).
    assert len(r["best_x"]) == 4
    float_block, int_block, cat_block, bin_block = r["best_x"]
    assert len(float_block) == 2 and all(isinstance(x, float) for x in float_block)
    assert len(int_block) == 2 and all(isinstance(x, int) for x in int_block)
    assert len(cat_block) == 2 and all(isinstance(x, int) and 0 <= x < 3 for x in cat_block)
    assert len(bin_block) == 4 and all(isinstance(x, bool) for x in bin_block)


def test_mixed_diagnostic_dim_and_optimum_is_none():
    p = sezgi.problems.mixed_diagnostic(2, 2, 3, 2, 4)
    assert p.dim() == 2 + 2 + 2 + 4
    assert p.optimum() is None
