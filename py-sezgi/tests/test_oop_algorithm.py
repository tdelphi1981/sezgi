"""M4-1 Task 3: `sezgi.Algorithm` -- the engine-hosted, class-first
algorithm-authoring base (`py-sezgi/python/sezgi/algorithm.py`), plus the
sanctioned ask/tell rename (`sezgi.Algorithm` -> `sezgi.AskTellAlgorithm`,
`py-sezgi/python/sezgi/algo.py`).

Covers the brief's own pinned test list:
(a) a small custom Algorithm subclass runs deterministically, anchored
    best_f at a fixed seed, two runs identical.
(b) an initialize-overriding subclass actually runs Python init code
    (proved via a counter) and the run stays deterministic.
(c) a NON-overriding subclass is bit-identical to the equivalent raw
    `_sezgi.solve_with_py_generator` call (RULING A proof: the default
    path never calls into Python for initialize()).
(d) AskTellAlgorithm works under both the new name and the
    `sezgi.algo.Algorithm` compat alias.
(e) validate_space veto surfaces before any generate() call.
(f) run() accepts both a Problem subclass instance and a native handle.
"""
import pytest
import sezgi
from sezgi import _sezgi


class Sphere(sezgi.Problem):
    """Single-block (Float) problem, shared by every test below."""

    def __init__(self, n=3, lo=-5.0, hi=5.0):
        self.n, self.lo, self.hi = n, lo, hi

    def space(self):
        return sezgi.Float(self.lo, self.hi, self.n)

    def evaluate(self, x):
        return sum(v * v for v in x)


class TournamentMutation(sezgi.Algorithm):
    """Binary tournament selection (via ctx.rng.next_below) followed by a
    small perturbation mutation (via ctx.rng.next_f64) -- does NOT override
    initialize()."""

    def generate(self, pop, ctx):
        offspring = []
        n = len(pop.individuals)
        for _ in range(n):
            i, j = ctx.rng.next_below(n), ctx.rng.next_below(n)
            winner = pop.individuals[i] if pop.fitness[i] <= pop.fitness[j] else pop.individuals[j]
            offspring.append([v + (ctx.rng.next_f64() - 0.5) * 0.5 for v in winner])
        return offspring


# ---------------------------------------------------------------------
# (a) Determinism + anchored best_f.
# ---------------------------------------------------------------------

def test_custom_algorithm_deterministic_and_anchored():
    r1 = TournamentMutation().run(Sphere(n=3), budget=800, seed=1, pop_size=20)
    r2 = TournamentMutation().run(Sphere(n=3), budget=800, seed=1, pop_size=20)
    assert r1 == r2
    # ANCHORED (per this project's convention): measured once at this exact
    # config (n=3, budget=800, seed=1, pop_size=20), then pinned.
    assert r1.best_f == 0.0005424421612573647
    assert r1.evals_used == 800
    assert r1.algo == "tournamentmutation"
    assert r1.f_opt is None and r1.gap is None  # Sphere() has no known optimum


def test_custom_algorithm_different_seed_differs():
    r1 = TournamentMutation().run(Sphere(n=3), budget=200, seed=1, pop_size=10)
    r2 = TournamentMutation().run(Sphere(n=3), budget=200, seed=2, pop_size=10)
    assert r1.best_f != r2.best_f


# ---------------------------------------------------------------------
# (b) initialize()-overriding subclass: Python init actually runs (proved
# via a counter), run stays deterministic.
# ---------------------------------------------------------------------

class CountingInitAlgorithm(sezgi.Algorithm):
    """Overrides initialize(): draws each coordinate of each individual
    from ctx.rng, uniformly over a DELIBERATELY NARROW [-0.5, 0.5] range
    (distinct from Sphere's own full [-5, 5] domain -- proves the override
    is genuinely live, not coincidentally reproducing the engine's own
    init/uniform sampling formula), and counts how many times initialize()
    itself is called (must be exactly 1 -- the engine calls it once per
    run, before the generation loop starts)."""

    def __init__(self):
        self.init_calls = 0

    def initialize(self, n, ctx):
        self.init_calls += 1
        lo, hi = -0.5, 0.5
        return [[lo + ctx.rng.next_f64() * (hi - lo) for _ in range(3)] for _ in range(n)]

    def generate(self, pop, ctx):
        return [[v + (ctx.rng.next_f64() - 0.5) * 0.5 for v in x] for x in pop.individuals]


def test_initialize_override_runs_and_is_deterministic():
    algo = CountingInitAlgorithm()
    r1 = algo.run(Sphere(n=3), budget=500, seed=5, pop_size=10)
    assert algo.init_calls == 1, "initialize() must be called exactly once per run"

    algo2 = CountingInitAlgorithm()
    r2 = algo2.run(Sphere(n=3), budget=500, seed=5, pop_size=10)
    assert algo2.init_calls == 1
    assert (r1.best_f, r1.best_x, r1.evals_used) == (r2.best_f, r2.best_x, r2.evals_used)


def test_initialize_override_differs_from_default_init():
    """Sanity check that overriding initialize() actually changes the run
    (vs. the same generate() with the default init/uniform path) -- proves
    the override is live, not silently ignored."""
    overridden = CountingInitAlgorithm().run(Sphere(n=3), budget=500, seed=5, pop_size=10)
    default = TournamentMutationLikeCounting().run(Sphere(n=3), budget=500, seed=5, pop_size=10)
    assert overridden.best_x != default.best_x or overridden.best_f != default.best_f


class TournamentMutationLikeCounting(sezgi.Algorithm):
    """Same generate() as CountingInitAlgorithm, but does NOT override
    initialize() -- used only as the default-init comparison point above."""

    def generate(self, pop, ctx):
        return [[v + (ctx.rng.next_f64() - 0.5) * 0.5 for v in x] for x in pop.individuals]


# ---------------------------------------------------------------------
# (c) RULING A proof: a NON-overriding subclass's run() is bit-identical to
# the equivalent raw _sezgi.solve_with_py_generator call (no initializer
# passed, no Python call for initialize() at all -- the pure-Rust
# init/uniform default path).
# ---------------------------------------------------------------------

def test_non_overriding_subclass_bit_identical_to_raw_bridge_call():
    native = Sphere(n=3)._to_native()
    budget, seed, pop_size = 600, 3, 15

    algo = TournamentMutation()
    via_run = algo.run(native, budget=budget, seed=seed, pop_size=pop_size)

    raw = _sezgi.solve_with_py_generator(
        TournamentMutation(), native, budget, master_seed=seed, run_id=0, pop_size=pop_size)

    assert via_run.best_f == raw["best_f"]
    assert via_run.best_x == raw["best_x"]
    assert via_run.evals_used == raw["evals_used"]


def test_non_overriding_subclass_initialize_never_called():
    """A non-overriding subclass's own initialize() (the base class's
    default body) must never be invoked by run() -- proven by it raising
    if it ever were."""
    class NeverInit(sezgi.Algorithm):
        def generate(self, pop, ctx):
            return list(pop.individuals)

    # If run() ever called the inherited default initialize(), this would
    # raise NotImplementedError (see algorithm.py's own default body) and
    # this test would fail with that exception instead of passing cleanly.
    result = NeverInit().run(Sphere(n=2), budget=50, seed=1, pop_size=5)
    assert result.evals_used == 50


# ---------------------------------------------------------------------
# (d) AskTellAlgorithm works under both the new name and the
# sezgi.algo.Algorithm compat alias.
# ---------------------------------------------------------------------

class _RandomSearch(sezgi.AskTellAlgorithm):
    def setup(self, ctx):
        ctx.evaluate([ctx.random_point()])

    def step(self, ctx):
        ctx.evaluate([ctx.random_point()])


class _RandomSearchViaAlias(sezgi.algo.Algorithm):
    def setup(self, ctx):
        ctx.evaluate([ctx.random_point()])

    def step(self, ctx):
        ctx.evaluate([ctx.random_point()])


def test_asktell_algorithm_new_name_works():
    res = _RandomSearch().solve(sezgi.bbob(1, 3, 1), budget=30, seed=1)
    assert res.evals_used == 30
    assert res.algo == "_randomsearch"


def test_asktell_algorithm_compat_alias_works():
    res = _RandomSearchViaAlias().solve(sezgi.bbob(1, 3, 1), budget=30, seed=1)
    assert res.evals_used == 30


def test_compat_alias_is_the_same_class_object():
    assert sezgi.algo.Algorithm is sezgi.AskTellAlgorithm
    assert sezgi.AskTellAlgorithm is not sezgi.Algorithm  # the NEW, unrelated base


# ---------------------------------------------------------------------
# (e) validate_space veto surfaces before any generate() call.
# ---------------------------------------------------------------------

class VetoingAlgorithm(sezgi.Algorithm):
    def __init__(self):
        self.generate_calls = 0

    def validate_space(self, space):
        raise ValueError("this space is not welcome here")

    def generate(self, pop, ctx):
        self.generate_calls += 1
        return list(pop.individuals)


def test_validate_space_veto_surfaces_before_generate():
    algo = VetoingAlgorithm()
    with pytest.raises(ValueError, match="not welcome here"):
        algo.run(Sphere(n=3), budget=100, seed=1, pop_size=5)
    assert algo.generate_calls == 0


# ---------------------------------------------------------------------
# (f) run() accepts both a Problem subclass instance and a native handle.
# ---------------------------------------------------------------------

def test_run_accepts_problem_subclass_instance():
    result = TournamentMutation().run(Sphere(n=3), budget=100, seed=1, pop_size=10)
    assert result.evals_used == 100
    assert len(result.best_x) == 3


def test_run_accepts_native_handle():
    result = TournamentMutation().run(sezgi.bbob(1, 3, 1), budget=100, seed=1, pop_size=10)
    assert result.evals_used == 100
    assert result.f_opt is not None
    assert result.gap == result.best_f - result.f_opt


def test_asktell_solve_problem_subclass_gets_friendly_value_error():
    """Final-review fix 5: AskTellAlgorithm.solve() now routes `problem`
    through as_native_problem before handing it to
    EvalSession.for_problem. A native handle passes through unchanged
    (test_asktell_algorithm_new_name_works, sezgi.bbob(...), already
    covers this, unaffected by this fix). A sezgi.Problem subclass
    instance is converted to its own native CallableSpaced handle, which
    EvalSession.for_problem then rejects -- ask/tell has no session type
    for a CallableSpaced problem at all (a pre-existing, honest, Rust-side
    restriction this fix does NOT touch or attempt to lift, see
    lib.rs:1753-1770) -- with a specific, self-explanatory ValueError
    naming "a sezgi.Problem subclass's native handle" and the
    solve()+presets workaround, rather than pyo3's contradictory raw
    conversion error ("'Sphere' object cannot be converted to 'Problem'")
    a Problem subclass instance used to fail with before routing through
    as_native_problem (that raw TypeError was itself the confusing part:
    post-conversion, the object literally IS a Problem)."""
    with pytest.raises(ValueError, match="sezgi.Problem subclass's native handle"):
        _RandomSearch().solve(Sphere(n=3), budget=30, seed=1)


# ---------------------------------------------------------------------
# RULING B: log_dir wiring (wired end-to-end for BBOB/CEC2022/CEC2014/
# CEC2017, rejected for anything else -- mirroring solve()'s own boundary).
# ---------------------------------------------------------------------

def test_run_log_dir_bbob_smoke(tmp_path):
    result = TournamentMutation().run(sezgi.bbob(1, 3, 1), budget=100, seed=1,
                                      pop_size=10, log_dir=str(tmp_path))
    back = sezgi.read_ioh_records(str(tmp_path), [100])
    assert len(back) == 1
    assert back[0]["best_f"] == result.best_f
    assert back[0]["algo"] == "tournamentmutation"
    assert any(tmp_path.rglob("*.dat"))


def test_run_log_dir_rejected_for_non_loggable_problem(tmp_path):
    with pytest.raises(ValueError, match="log_dir is only supported"):
        TournamentMutation().run(Sphere(n=3), budget=100, seed=1,
                                 pop_size=5, log_dir=str(tmp_path))


# ---------------------------------------------------------------------
# Misc: abc enforcement, distinctness from AskTellAlgorithm.
# ---------------------------------------------------------------------

def test_algorithm_is_abstract_without_generate():
    with pytest.raises(TypeError):
        sezgi.Algorithm()  # abstract: generate() not implemented


def test_algorithm_and_asktell_algorithm_are_unrelated_bases():
    assert not issubclass(sezgi.Algorithm, sezgi.AskTellAlgorithm)
    assert not issubclass(sezgi.AskTellAlgorithm, sezgi.Algorithm)
