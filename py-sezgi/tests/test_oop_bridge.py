"""M4-1 Task 2: the engine-hosted Python callback bridge --
`_sezgi.solve_with_py_generator` (`py-sezgi/src/lib.rs`), which registers a
Python-authored generator (any object with a `generate(pop, ctx)` method,
optionally `validate_space(space)`) as a real `sezgi_core::component::
Generator`, running INSIDE the Rust `Engine` loop.

`_sezgi.solve_with_py_generator`/`_sezgi.PyRng` are INTERNAL (not
re-exported from `sezgi/__init__.py`'s public namespace -- same status as
`_sezgi.from_callable_spaced`, M4-1 Task 1): Task 3's `Algorithm`
front-door class hierarchy is the intended caller, so this test file talks
to `sezgi._sezgi` directly and writes a MINIMAL raw-callback Python class
inline (Task 3's base class does not exist yet).

Every generator below draws from `ctx.rng`, converts `pop.individuals`/
returns offspring via the SAME per-individual convention `sezgi.Problem.
evaluate(x)` already uses (M4-1 Task 1): bare for a single-block space, a
tuple of per-block values for a multi-block one.
"""
import pytest
import sezgi
from sezgi import _sezgi


# ---------------------------------------------------------------------
# Shared fixtures: minimal Problem subclasses and generator callbacks.
# ---------------------------------------------------------------------

class Sphere(sezgi.Problem):
    """Single-block (Float) problem -- anchors (a)/(b)/(c)/(d)."""

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


class Categories(sezgi.Problem):
    """Single-block (Categorical) problem -- for the out-of-range test."""

    def __init__(self, k=3, n=2):
        self.k = k
        self.n = n

    def space(self):
        return sezgi.Categorical(self.k, self.n)

    def evaluate(self, x):
        assert isinstance(x, list)
        return float(sum(x))


class MixedFloatBinary(sezgi.Problem):
    """Multi-block (Float + Binary) problem -- anchors (e)."""

    def __init__(self, n_float=2, n_bin=3, lo=-5.0, hi=5.0):
        self.n_float = n_float
        self.n_bin = n_bin
        self.lo = lo
        self.hi = hi

    def space(self):
        return sezgi.Space(sezgi.Float(self.lo, self.hi, self.n_float), sezgi.Binary(self.n_bin))

    def evaluate(self, x):
        floats, bits = x
        assert isinstance(floats, list) and all(isinstance(v, float) for v in floats)
        assert isinstance(bits, list) and all(isinstance(b, bool) for b in bits)
        return sum(v * v for v in floats) + sum(0.0 if b else 1.0 for b in bits)


class RandomStepGenerator:
    """Draws ONE `ctx.rng.next_f64()` value per PARENT and perturbs every
    coordinate by that same step -- exercises the RNG bridge
    deterministically, offspring count == pop size (not required, but keeps
    this fixture simple)."""

    def generate(self, pop, ctx):
        offspring = []
        for x in pop.individuals:
            step = ctx.rng.next_f64() - 0.5
            offspring.append([v + step for v in x])
        return offspring


class FirstDrawRecorder:
    """Records its FIRST `ctx.rng` draw on every call (and consumes no
    other draws), returning the parents unchanged as offspring -- used by
    the RNG-continuity test (b)."""

    def __init__(self):
        self.draws = []

    def generate(self, pop, ctx):
        self.draws.append(ctx.rng.next_f64())
        return list(pop.individuals)


class WrongTypeGenerator:
    """Returns a non-float element for a Float block -- malformed offspring
    (c), "wrong element type"."""

    def generate(self, pop, ctx):
        return [["not", "a", "float"] for _ in pop.individuals]


class OutOfRangeCategoricalGenerator:
    """Returns a category index >= k -- malformed offspring (c),
    "out-of-space categorical index". `Categories(k=3, n=2)`'s valid range
    is 0..3, so index 5 is out of range."""

    def generate(self, pop, ctx):
        return [[5, 0] for _ in pop.individuals]


class BoomGenerator:
    """Raises a plain Python exception inside generate() -- (c)."""

    def generate(self, pop, ctx):
        raise RuntimeError("boom")


class VetoingGenerator:
    """`validate_space` always raises -- (d). Tracks whether `generate` was
    ever called, to prove the veto happens at build time."""

    def __init__(self):
        self.generate_calls = 0

    def generate(self, pop, ctx):
        self.generate_calls += 1
        return list(pop.individuals)

    def validate_space(self, space):
        raise ValueError("no thanks")


class MixedGenerator:
    """Multi-block (Float + Binary) generator -- (e). Receives/returns
    tuple-form individuals."""

    def generate(self, pop, ctx):
        offspring = []
        for x in pop.individuals:
            floats, bits = x
            assert isinstance(floats, list) and all(isinstance(v, float) for v in floats)
            assert isinstance(bits, list) and all(isinstance(b, bool) for b in bits)
            new_floats = [v + (ctx.rng.next_f64() - 0.5) for v in floats]
            new_bits = [not b if ctx.rng.next_f64() < 0.2 else b for b in bits]
            offspring.append((new_floats, new_bits))
        return offspring


# ---------------------------------------------------------------------
# (a) Determinism: same seed -> byte-identical best_f/best_x; a different
# seed differs.
# ---------------------------------------------------------------------

def test_determinism_same_seed_byte_identical():
    prob = Sphere(n=3)._to_native()
    r1 = _sezgi.solve_with_py_generator(
        RandomStepGenerator(), prob, 2000, master_seed=1, run_id=0, pop_size=20)
    r2 = _sezgi.solve_with_py_generator(
        RandomStepGenerator(), prob, 2000, master_seed=1, run_id=0, pop_size=20)
    assert r1["best_f"] == r2["best_f"]
    assert r1["best_x"] == r2["best_x"]
    assert r1["evals_used"] == r2["evals_used"]
    assert r1["iterations"] == r2["iterations"]


def test_determinism_different_seed_differs():
    prob = Sphere(n=3)._to_native()
    r1 = _sezgi.solve_with_py_generator(
        RandomStepGenerator(), prob, 2000, master_seed=1, run_id=0, pop_size=20)
    r2 = _sezgi.solve_with_py_generator(
        RandomStepGenerator(), prob, 2000, master_seed=2, run_id=0, pop_size=20)
    assert r1["best_f"] != r2["best_f"]


# ---------------------------------------------------------------------
# (b) RNG continuity: the callback's recorded first-draw-per-call sequence
# equals draws from a standalone reconstruction of the stage 0 generator
# stream, RngStream::from_master(master_seed, [run_id, 1 + 2*0]) ==
# [run_id, 1] (engine.rs:857-914's own documented per-stage path, mirrored
# here from Python via PyRng.from_master).
# ---------------------------------------------------------------------

def test_rng_continuity_matches_standalone_stream_reconstruction():
    prob = Sphere(n=2)._to_native()
    rec = FirstDrawRecorder()
    master_seed, run_id, pop_size = 7, 0, 4
    # 5 full generations' worth of budget, exactly divisible -- the engine's
    # own outer loop still attempts one further speculative generate() call
    # after the budget is exhausted (see engine.rs's own
    # stage_rng_streams_use_the_documented_per_stage_indices test comment),
    # so len(rec.draws) may be 6, not 5 -- this test does not hardcode the
    # count, it only compares the ACTUAL recorded sequence against that many
    # draws from the independently-reconstructed stream.
    budget = pop_size + 5 * pop_size
    _sezgi.solve_with_py_generator(
        rec, prob, budget, master_seed=master_seed, run_id=run_id, pop_size=pop_size)

    assert len(rec.draws) >= 5, "at least the 5 completed generations must have called generate()"

    expected_rng = _sezgi.PyRng.from_master(master_seed, [run_id, 1])
    expected = [expected_rng.next_f64() for _ in range(len(rec.draws))]
    assert rec.draws == expected


# ---------------------------------------------------------------------
# (c) Error tests: malformed offspring (honest error naming the offending
# index); a Python exception raised inside generate() is re-raised
# unchanged.
# ---------------------------------------------------------------------

def test_wrong_element_type_offspring_raises_honest_error_naming_index():
    prob = Sphere(n=3)._to_native()
    with pytest.raises(ValueError, match=r"offspring\[0\]"):
        _sezgi.solve_with_py_generator(WrongTypeGenerator(), prob, 100, pop_size=5)


def test_out_of_range_categorical_offspring_raises_honest_error_naming_index():
    prob = Categories(k=3, n=2)._to_native()
    with pytest.raises(ValueError, match=r"offspring\[0\].*category index"):
        _sezgi.solve_with_py_generator(OutOfRangeCategoricalGenerator(), prob, 100, pop_size=5)


def test_exception_inside_generate_is_reraised_unchanged():
    prob = Sphere(n=3)._to_native()
    with pytest.raises(RuntimeError, match="boom"):
        _sezgi.solve_with_py_generator(BoomGenerator(), prob, 100, pop_size=5)


# ---------------------------------------------------------------------
# (d) validate_space veto: a callback whose validate_space raises errors at
# BUILD time, before any generate() call.
# ---------------------------------------------------------------------

def test_validate_space_veto_happens_before_any_generate_call():
    prob = Sphere(n=3)._to_native()
    gen = VetoingGenerator()
    with pytest.raises(ValueError, match="no thanks"):
        _sezgi.solve_with_py_generator(gen, prob, 1000, pop_size=5)
    assert gen.generate_calls == 0


# ---------------------------------------------------------------------
# (e) Multi-block space: the callback receives tuple-form individuals and
# returns tuple-form offspring, end to end on a mixed Float+Binary space.
# ---------------------------------------------------------------------

def test_multi_block_space_tuple_form_end_to_end():
    prob = MixedFloatBinary(n_float=2, n_bin=3)._to_native()
    r = _sezgi.solve_with_py_generator(
        MixedGenerator(), prob, 500, master_seed=3, pop_size=10)

    assert isinstance(r["best_x"], list)
    assert len(r["best_x"]) == 2
    floats, bits = r["best_x"]
    assert len(floats) == 2 and all(isinstance(v, float) for v in floats)
    assert len(bits) == 3 and all(isinstance(b, bool) for b in bits)
    assert isinstance(r["best_f"], float)
