"""M4-1 Task 4: the two family bases over `sezgi.Algorithm`
(`py-sezgi/python/sezgi/algorithm.py`) -- `PopulationAlgorithm` (select +
vary) and `LocalSearch` (neighbor + accept), plus their two worked
examples (`examples/python/oop/custom_de_variant.py`,
`examples/python/oop/custom_local_search.py`).

Covers the brief's own pinned test list:
(a) default select() is seeded-deterministic (two engine runs identical)
    and its exact draw pattern matches a hand-trace (small stub pop,
    fixed PyRng seed, next_below draws traced by hand and compared).
(b) overriding ONLY vary() (default select() left in place) yields a
    working algorithm end to end (anchored).
(c) LocalSearch's accept() acceptance logic, hand-fixture tested at the
    Python level with a stub pop/ctx -- no engine involved.
(d) both example files execute deterministically with anchored output.
"""
import pytest
import sezgi
from sezgi import _sezgi


class Sphere(sezgi.Problem):
    """Single-block (Float) problem, shared by every test below (same
    shape as test_oop_algorithm.py's own fixture)."""

    def __init__(self, n=3, lo=-5.0, hi=5.0):
        self.n, self.lo, self.hi = n, lo, hi

    def space(self):
        return sezgi.Float(self.lo, self.hi, self.n)

    def evaluate(self, x):
        return sum(v * v for v in x)


# ---------------------------------------------------------------------
# Stubs for the pure-Python, no-engine tests (default select()'s hand
# trace, LocalSearch's accept()/generate() state machine).
# ---------------------------------------------------------------------

class _StubPop:
    def __init__(self, individuals, fitness):
        self.individuals = individuals
        self.fitness = fitness


class _StubCtx:
    def __init__(self, rng=None):
        self.rng = rng
        self.iteration = 0
        self.space = []


# ---------------------------------------------------------------------
# (a) PopulationAlgorithm.select() default: hand trace + determinism.
# ---------------------------------------------------------------------

class _MinimalPopAlgo(sezgi.PopulationAlgorithm):
    """Only exists to make select() callable (vary() is abstract and
    unused by the hand-trace test below, which calls select() directly)."""

    def vary(self, parents, ctx):
        return parents


def test_default_select_matches_hand_traced_next_below_draws():
    """Small pop (n=5), k=3, fixed PyRng seed (master_seed=7, path=[0]).

    Hand trace (computed once, independently, via the SAME formula
    select()'s own docstring pins -- i = next_below(n); j =
    next_below(n-1), shifted past i if j >= i -- and pinned here as
    literal numbers so this test does not just re-derive the formula it
    is meant to check):

        draw 1: i=1, j=2 -> fitness[1]=1.0 <= fitness[2]=4.0 -> winner 1
        draw 2: i=4, j=2 -> fitness[4]=3.0 <= fitness[2]=4.0 -> winner 4
        draw 3: i=3, j=1 -> fitness[3]=2.0 >  fitness[1]=1.0 -> winner 1

    So the expected draws are [(1, 2), (4, 2), (3, 1)] and the expected
    winning indices are [1, 4, 1].
    """
    individuals = ["a", "b", "c", "d", "e"]
    fitness = [5.0, 1.0, 4.0, 2.0, 3.0]

    rng = _sezgi.PyRng.from_master(7, [0])
    parents = _MinimalPopAlgo().select(_StubPop(individuals, fitness), 3, _StubCtx(rng))

    expected_winners = [1, 4, 1]
    assert parents == [individuals[w] for w in expected_winners]

    # Cross-check: an INDEPENDENT PyRng constructed from the same
    # (master_seed, path) reproduces the identical draw/winner sequence
    # when the exact documented formula is applied by hand.
    trace_rng = _sezgi.PyRng.from_master(7, [0])
    n = len(individuals)
    traced_draws, traced_winners = [], []
    for _ in range(3):
        i = trace_rng.next_below(n)
        j = trace_rng.next_below(n - 1)
        if j >= i:
            j += 1
        traced_draws.append((i, j))
        traced_winners.append(i if fitness[i] <= fitness[j] else j)
    assert traced_draws == [(1, 2), (4, 2), (3, 1)]
    assert traced_winners == expected_winners


def test_default_select_tie_break_favors_first_drawn_index():
    """An exact fitness tie must favor i (the first-drawn contestant),
    never j -- pinned tie-break rule from select()'s own docstring."""
    individuals = ["a", "b"]
    fitness = [1.0, 1.0]  # tie regardless of which index wins the draw
    rng = _sezgi.PyRng.from_master(1, [0])
    parents = _MinimalPopAlgo().select(_StubPop(individuals, fitness), 1, _StubCtx(rng))
    # With n=2, i is drawn from {0, 1}; j is forced to be the OTHER
    # index (n - 1 == 1, next_below(1) is always 0, then shifted past
    # i) -- so {i, j} == {0, 1} always. On the tie, i must win.
    trace_rng = _sezgi.PyRng.from_master(1, [0])
    i = trace_rng.next_below(2)
    j = trace_rng.next_below(1)
    if j >= i:
        j += 1
    assert {i, j} == {0, 1}
    assert parents == [individuals[i]]


def test_default_select_pop_of_one_draws_once_and_wins_trivially():
    """n == 1: only ONE draw (next_below(1), always 0), no second index
    to fight -- the sole individual wins by construction."""
    individuals = ["only"]
    fitness = [42.0]
    rng = _sezgi.PyRng.from_master(2, [0])
    parents = _MinimalPopAlgo().select(_StubPop(individuals, fitness), 4, _StubCtx(rng))
    assert parents == ["only", "only", "only", "only"]


class _TournamentPerturbation(sezgi.PopulationAlgorithm):
    """Uses the DEFAULT select() (not overridden) with a tiny
    perturbation vary() -- the end-to-end determinism vehicle for (a)."""

    def vary(self, parents, ctx):
        return [[v + (ctx.rng.next_f64() - 0.5) * 0.2 for v in p] for p in parents]


def test_default_select_is_seeded_deterministic_end_to_end():
    r1 = _TournamentPerturbation().run(Sphere(n=3), budget=300, seed=11, pop_size=8)
    r2 = _TournamentPerturbation().run(Sphere(n=3), budget=300, seed=11, pop_size=8)
    assert r1 == r2


# ---------------------------------------------------------------------
# (b) Overriding ONLY vary() (default select() left in place) yields a
# working algorithm end to end -- anchored.
# ---------------------------------------------------------------------

class SmallPerturbationVary(sezgi.PopulationAlgorithm):
    """Overrides ONLY vary() -- select() is entirely the inherited
    default (k-fold binary tournament)."""

    def vary(self, parents, ctx):
        return [[v + (ctx.rng.next_f64() - 0.5) * 0.4 for v in p] for p in parents]


def test_only_vary_override_runs_end_to_end_deterministic_and_anchored():
    r1 = SmallPerturbationVary().run(Sphere(n=3), budget=500, seed=3, pop_size=10)
    r2 = SmallPerturbationVary().run(Sphere(n=3), budget=500, seed=3, pop_size=10)
    assert r1 == r2
    # ANCHORED (per this project's convention): measured once at this
    # exact config (n=3, budget=500, seed=3, pop_size=10), then pinned.
    assert r1.best_f == 0.0008086287492790783
    assert r1.evals_used == 500
    assert r1.algo == "smallperturbationvary"


def test_population_algorithm_requires_vary():
    with pytest.raises(TypeError):
        sezgi.PopulationAlgorithm()  # abstract: vary() not implemented

    class NoVary(sezgi.PopulationAlgorithm):
        pass

    with pytest.raises(TypeError):
        NoVary()


# ---------------------------------------------------------------------
# (c) LocalSearch.accept() acceptance logic: hand fixture at the Python
# level with a stub ctx -- no engine needed.
# ---------------------------------------------------------------------

class _StubLocalSearch(sezgi.LocalSearch):
    def neighbor(self, x, ctx):
        return x + 1  # arbitrary, deterministic, never actually run through the engine


def test_default_accept_greedy_rejects_worse_accepts_equal_and_better():
    algo = _StubLocalSearch()
    assert algo.accept(5.0, 3.0, _StubCtx()) is True   # strictly better
    assert algo.accept(5.0, 5.0, _StubCtx()) is True   # exact tie
    assert algo.accept(5.0, 7.0, _StubCtx()) is False  # strictly worse


def test_local_search_generate_advances_anchor_on_default_accept():
    """Walks generate() through two calls by hand, simulating what
    replace/mu-plus-lambda would present on each subsequent call (see
    LocalSearch's own ACCEPT() DESIGN docstring): the population always
    reflects pop_size=1's already-greedy replacement outcome."""
    algo = _StubLocalSearch()
    algo.current_x, algo.current_f = None, None

    # First call: adopts the initial population as the starting anchor.
    offspring = algo.generate(_StubPop([10], [100.0]), _StubCtx())
    assert offspring == [11]  # neighbor(10, ctx) == 10 + 1
    assert (algo.current_x, algo.current_f) == (10, 100.0)

    # Second call: the engine's replacer already picked the (better)
    # neighbor outcome -- default greedy accept() confirms it, ADVANCING
    # the anchor to match.
    offspring = algo.generate(_StubPop([11], [90.0]), _StubCtx())
    assert offspring == [12]  # neighbor(11, ctx) == 11 + 1
    assert (algo.current_x, algo.current_f) == (11, 90.0)


def test_local_search_generate_stays_put_when_accept_rejects():
    """A stricter-than-default accept() (requires an improvement of at
    least 0.5) demonstrates accept()'s REAL power: whether this base's
    own anchor advances to match what the engine's population already
    shows, or stays where it was -- NOT whether the engine's own
    population moves (it already has, unconditionally, before accept()
    is even consulted; see the class docstring)."""

    class PickyAccept(_StubLocalSearch):
        def accept(self, f_old, f_new, ctx):
            return f_old - f_new >= 0.5

    algo = PickyAccept()
    algo.current_x, algo.current_f = None, None

    algo.generate(_StubPop([10], [100.0]), _StubCtx())
    assert (algo.current_x, algo.current_f) == (10, 100.0)

    # The engine's own population already reflects a SMALL improvement
    # (100.0 -> 99.9, replacer kept the marginally-better neighbor) --
    # but PickyAccept requires >= 0.5 improvement, so the base's own
    # anchor must NOT advance; the next neighbor() proposal is still
    # seeded from x=10, not x=11.
    offspring = algo.generate(_StubPop([11], [99.9]), _StubCtx())
    assert offspring == [11]  # neighbor(10, ctx) == 10 + 1, NOT 12
    assert (algo.current_x, algo.current_f) == (10, 100.0)

    # A large-enough improvement DOES advance the anchor -- accept() is
    # evaluated BEFORE this call's own neighbor() proposal, so the
    # advanced anchor (11) is what THIS call's own offspring is seeded
    # from.
    offspring = algo.generate(_StubPop([11], [99.4]), _StubCtx())
    assert offspring == [12]  # neighbor(11, ctx) == 11 + 1
    assert (algo.current_x, algo.current_f) == (11, 99.4)


def test_local_search_requires_neighbor():
    with pytest.raises(TypeError):
        sezgi.LocalSearch()  # abstract: neighbor() not implemented


# ---------------------------------------------------------------------
# LocalSearch end to end (through the real engine): pop_size enforcement,
# determinism, anchored output, per-run state reset.
# ---------------------------------------------------------------------

class SimpleLS(sezgi.LocalSearch):
    """Overrides ONLY neighbor() -- accept() is entirely the inherited
    default (greedy)."""

    def neighbor(self, x, ctx):
        return [v + (ctx.rng.next_f64() - 0.5) * 0.5 for v in x]


def test_local_search_run_rejects_pop_size_other_than_one():
    with pytest.raises(ValueError, match="pop_size=1"):
        SimpleLS().run(Sphere(n=3), budget=10, seed=1, pop_size=2)


def test_local_search_end_to_end_deterministic_and_anchored():
    r1 = SimpleLS().run(Sphere(n=3), budget=300, seed=7)
    r2 = SimpleLS().run(Sphere(n=3), budget=300, seed=7)
    assert r1 == r2
    # ANCHORED (per this project's convention): measured once at this
    # exact config (n=3, budget=300, seed=7, pop_size=1), then pinned.
    assert r1.best_f == 0.001146450487769876
    assert r1.evals_used == 300
    assert r1.algo == "simplels"


def test_local_search_instance_reuse_does_not_leak_state_across_runs():
    """The SAME LocalSearch instance, run() twice, must not carry
    current_x/current_f state over from the first run into the second
    (run() resets both at its own start) -- both calls must match a
    fresh instance's own single run at the same config."""
    reference = SimpleLS().run(Sphere(n=3), budget=300, seed=7)

    reused = SimpleLS()
    first = reused.run(Sphere(n=3), budget=300, seed=7)
    second = reused.run(Sphere(n=3), budget=300, seed=7)
    assert first == reference
    assert second == reference


# ---------------------------------------------------------------------
# (d) Worked examples: deterministic, anchored output (tsp_two_opt.py-
# style "run the script, parse the printed metrics line" gate).
# ---------------------------------------------------------------------

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIELDS = re.compile(r"evals_used=(\S+) best_f=(\S+) gap=(\S+)")


def _run_fields(script):
    out = subprocess.run([sys.executable, str(script)], capture_output=True,
                         text=True, check=True, cwd=ROOT).stdout
    m = FIELDS.search(out)
    assert m, f"no metrics line in {script}: {out!r}"
    return m.groups()


def test_custom_de_variant_example_deterministic_and_anchored():
    script = ROOT / "examples" / "python" / "oop" / "custom_de_variant.py"
    a = _run_fields(script)
    b = _run_fields(script)
    assert a == b, "same seed/problem/budget must reproduce identical output"
    # ANCHORED (per this project's convention): measured once with this
    # script's own pinned constants (sezgi.bbob(1, 10, 1), budget=2000,
    # seed=42, pop_size=20), then pinned.
    evals_used, best_f, gap = a
    assert evals_used == "2000"
    assert best_f == "-59.3946"
    assert gap == "25.0094"


def test_custom_local_search_example_deterministic_and_anchored():
    script = ROOT / "examples" / "python" / "oop" / "custom_local_search.py"
    a = _run_fields(script)
    b = _run_fields(script)
    assert a == b, "same seed/problem/budget must reproduce identical output"
    # ANCHORED (per this project's convention): measured once with this
    # script's own pinned constants (sezgi.bbob(1, 10, 1), budget=2000,
    # seed=42, pop_size=1), then pinned.
    evals_used, best_f, gap = a
    assert evals_used == "2000"
    assert best_f == "-84.3232"
    assert gap == "0.0807033"
