"""M4-1 Task 1: sezgi.Problem (subclassable Python ABC), sezgi.spaces
builders, and the block-typed callable-problem bridge
(`_sezgi.from_callable_spaced`, `py-sezgi/src/lib.rs`) that underlies
`Problem._to_native()`.

Anchors below (sphere via ga_real, OneMax via ga_bin) were derived by
running the test once, printing the resulting `best_f`, and pinning that
exact value -- the same convention `test_typed_diagnostics.py`/
`test_onemax_ga_example.py` already use for this project's own anchored-
output tests.
"""
import tomllib

import pytest
import sezgi
from sezgi.problem import as_native_problem


# ---------------------------------------------------------------------
# (a) Pure-Python sphere Problem, solved via the compat path
# (sezgi.solve(presets.ga_real(...), prob._to_native(), master_seed=...)),
# anchored at a fixed seed.
# ---------------------------------------------------------------------

class Sphere(sezgi.Problem):
    def __init__(self, n=3, lo=-5.0, hi=5.0):
        self.n = n
        self.lo = lo
        self.hi = hi

    def space(self):
        return sezgi.Float(self.lo, self.hi, self.n)

    def evaluate(self, x):
        assert isinstance(x, list)
        assert all(isinstance(v, float) for v in x)
        return sum(v * v for v in x)


def test_sphere_problem_solves_via_compat_path_anchored():
    """Anchor: Sphere(n=3, lo=-5, hi=5) via presets.ga_real(pop_size=20,
    budget=2000), master_seed=1 -- best_f=1.0462797854373178e-05, measured
    by running this exact test once and pinning the printed value."""
    prob = Sphere()
    spec = sezgi.presets.ga_real(pop_size=20, budget=2000)
    r = sezgi.solve(spec, prob._to_native(), master_seed=1)
    assert r["best_f"] == pytest.approx(1.0462797854373178e-05)
    assert len(r["best_x"]) == 3
    assert all(isinstance(v, float) for v in r["best_x"])


def test_sphere_problem_solve_is_deterministic():
    prob = Sphere()
    spec = sezgi.presets.ga_real(pop_size=20, budget=2000)
    a = sezgi.solve(spec, prob._to_native(), master_seed=1)
    b = sezgi.solve(spec, prob._to_native(), master_seed=1)
    assert a["best_f"] == b["best_f"]
    assert a["best_x"] == b["best_x"]


def test_sphere_native_handle_dim_and_bounds():
    prob = Sphere(n=3, lo=-5.0, hi=5.0)
    native = prob._to_native()
    assert native.dim() == 3
    assert native.bounds() == (-5.0, 5.0)


# ---------------------------------------------------------------------
# (b) Binary-space Python problem (inline mini-OneMax), end-to-end via
# presets.ga_bin.
# ---------------------------------------------------------------------

class MiniOneMax(sezgi.Problem):
    def __init__(self, n=20):
        self.n = n

    def space(self):
        return sezgi.Binary(self.n)

    def evaluate(self, x):
        assert isinstance(x, list)
        assert all(isinstance(v, bool) for v in x)
        return self.n - sum(x)  # minimized at the all-True genotype


def test_mini_onemax_solves_to_exact_optimum_via_ga_bin():
    """Anchor: MiniOneMax(n=20) via presets.ga_bin(pop_size=40,
    budget=6000), master_seed=1 -- reaches the exact optimum (best_f=0.0),
    same anchor shape as test_typed_diagnostics.py's
    test_ga_bin_solves_onemax_to_exact_optimum (a different problem
    instance -- native problems.onemax(...) -- but same preset/budget/seed,
    confirming the pure-Python path reaches the same known-solvable
    result)."""
    prob = MiniOneMax(20)
    spec = sezgi.presets.ga_bin(40, 6000)
    r = sezgi.solve(spec, prob._to_native(), master_seed=1)
    assert r["best_f"] == 0.0
    assert len(r["best_x"]) == 20
    assert all(isinstance(b, bool) for b in r["best_x"])
    assert all(r["best_x"])


# ---------------------------------------------------------------------
# (c) Mixed-space Python problem: evaluate must receive a tuple with the
# EXACT per-block Python types pinned by the conversion table
# (py-sezgi/python/sezgi/problem.py's own module docstring).
# ---------------------------------------------------------------------

_MIXED_COMPOUND_TOML = """
name = "mixed-compound-oop"
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


class MixedDiagnostic(sezgi.Problem):
    """Float(2) + Int(2) + Categorical(k=3, n=2) + Binary(4) -- same block
    layout as `sezgi.problems.mixed_diagnostic(2, 2, 3, 2, 4)`
    (test_typed_diagnostics.py), reused here as a pure-Python Problem."""

    def __init__(self):
        self.calls = []

    def space(self):
        return sezgi.Space(
            sezgi.Float(-5.0, 5.0, 2),
            sezgi.Int(-5, 5, 2),
            sezgi.Categorical(3, 2),
            sezgi.Binary(4),
        )

    def evaluate(self, x):
        self.calls.append(x)
        assert isinstance(x, tuple)
        assert len(x) == 4
        float_block, int_block, cat_block, bin_block = x

        assert isinstance(float_block, list)
        assert len(float_block) == 2
        assert all(type(v) is float for v in float_block)

        assert isinstance(int_block, list)
        assert len(int_block) == 2
        assert all(type(v) is int for v in int_block)

        assert isinstance(cat_block, list)
        assert len(cat_block) == 2
        assert all(type(v) is int and 0 <= v < 3 for v in cat_block)

        assert isinstance(bin_block, list)
        assert len(bin_block) == 4
        assert all(type(v) is bool for v in bin_block)

        return (sum(float_block) + sum(int_block)
                + sum(1 for c in cat_block if c != 0)
                + sum(1 for b in bin_block if not b))


def test_mixed_space_problem_evaluate_receives_pinned_tuple_types():
    prob = MixedDiagnostic()
    spec = tomllib.loads(_MIXED_COMPOUND_TOML)
    r = sezgi.solve(spec, prob._to_native(), master_seed=42)
    assert isinstance(r["best_f"], float)
    assert len(prob.calls) > 0, "evaluate must have been called at least once"


def test_mixed_space_native_handle_bounds_raises_value_error():
    """bounds() is Float-only; a mixed space with non-Float blocks raises,
    mirroring problems.mixed_diagnostic(...)'s own behavior."""
    prob = MixedDiagnostic()
    with pytest.raises(ValueError, match="continuous"):
        prob._to_native().bounds()


# ---------------------------------------------------------------------
# (d) An exception raised inside evaluate surfaces unchanged (same
# exception type/message) to the caller.
# ---------------------------------------------------------------------

class Boom(sezgi.Problem):
    def space(self):
        return sezgi.Float(-1.0, 1.0, 2)

    def evaluate(self, x):
        raise ValueError("intentional-oop-boom")


def test_exception_in_evaluate_propagates_unchanged():
    prob = Boom()
    spec = sezgi.presets.ga_real(10, 200)
    with pytest.raises(ValueError, match="intentional-oop-boom"):
        sezgi.solve(spec, prob._to_native(), master_seed=1)


class BoomKeyError(sezgi.Problem):
    def space(self):
        return sezgi.Float(-1.0, 1.0, 2)

    def evaluate(self, x):
        raise KeyError("wrong-kind-of-exception")


def test_exception_type_is_preserved_not_just_message():
    prob = BoomKeyError()
    spec = sezgi.presets.ga_real(10, 200)
    with pytest.raises(KeyError):
        sezgi.solve(spec, prob._to_native(), master_seed=1)


# ---------------------------------------------------------------------
# (e) optimum() plumbing: None by default; a subclass override is
# reachable on the native handle (Inner::CallableSpaced carries an
# `optimum: Option<f64>` field plumbed straight to PyProblem::optimum()).
# ---------------------------------------------------------------------

class NoOptimum(sezgi.Problem):
    def space(self):
        return sezgi.Float(-1.0, 1.0, 2)

    def evaluate(self, x):
        return 0.0


class WithOptimum(sezgi.Problem):
    def space(self):
        return sezgi.Float(-5.0, 5.0, 3)

    def evaluate(self, x):
        return sum(v * v for v in x)

    def optimum(self):
        return 0.0


def test_optimum_defaults_to_none_python_and_native():
    prob = NoOptimum()
    assert prob.optimum() is None
    assert prob._to_native().optimum() is None


def test_optimum_override_reachable_on_native_handle():
    prob = WithOptimum()
    assert prob.optimum() == 0.0
    assert prob._to_native().optimum() == 0.0


# ---------------------------------------------------------------------
# (f) as_native_problem: native handle passes through; a Problem subclass
# is converted via _to_native(); anything else raises TypeError.
# ---------------------------------------------------------------------

def test_as_native_problem_passes_through_native_handle():
    native = sezgi.bbob(fid=1, dim=5, instance=1)
    assert as_native_problem(native) is native


def test_as_native_problem_converts_problem_subclass():
    prob = Sphere()
    converted = as_native_problem(prob)
    assert converted is not prob
    assert converted.dim() == 3


def test_as_native_problem_raises_type_error_for_anything_else():
    with pytest.raises(TypeError):
        as_native_problem(42)
    with pytest.raises(TypeError):
        as_native_problem("not a problem")
    with pytest.raises(TypeError):
        as_native_problem(object())


# ---------------------------------------------------------------------
# sezgi.spaces builders: bare-block acceptance, Space composition,
# validation.
# ---------------------------------------------------------------------

def test_bare_block_accepted_wherever_a_space_is_expected():
    class BareFloat(sezgi.Problem):
        def space(self):
            return sezgi.Float(-1.0, 1.0, 2)  # bare block, not wrapped in Space

        def evaluate(self, x):
            return sum(v * v for v in x)

    native = BareFloat()._to_native()
    assert native.dim() == 2


def test_space_rejects_empty_and_non_block_arguments():
    with pytest.raises(ValueError):
        sezgi.Space()
    with pytest.raises(TypeError):
        sezgi.Space(42)


def test_space_composes_multiple_blocks_in_order():
    space = sezgi.Space(sezgi.Float(-1.0, 1.0, 2), sezgi.Binary(3))
    assert space.blocks == (sezgi.Float(-1.0, 1.0, 2), sezgi.Binary(3))


# ---------------------------------------------------------------------
# Fix round 1: the new public symbols must be listed in sezgi.__all__ so
# `from sezgi import *` and __all__-driven tooling see them.
# ---------------------------------------------------------------------

def test_new_oop_symbols_are_in_dunder_all():
    for name in ("Float", "Int", "Categorical", "Binary", "Permutation",
                 "Space", "Problem", "as_native_problem"):
        assert name in sezgi.__all__, f"{name} missing from sezgi.__all__"
