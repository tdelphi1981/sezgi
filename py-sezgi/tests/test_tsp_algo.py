"""M3-8 Task 7: permutation and TSP authoring in the Python Algorithm ABC.

Widens EvalSession.for_problem/sezgi.Algorithm from Float-only to also
cover permutation-typed problems (sezgi.problems.tsp(...)), per the
approved scope ruling's exact minimal surface:
  - EvalSession.kind() -> "float" | "permutation"
  - EvalSession.random_permutation() -> list[int] (0-based, seeded/
    deterministic, only valid on a permutation-typed session)
  - EvalSession.evaluate(tours) accepts a list of 0-based tours for a
    permutation-typed session (previously float rows only)
  - AlgoContext.kind / AlgoContext.n / AlgoContext.random_permutation() /
    AlgoContext.two_opt(tour, i, j)

test_eval_session_generic.py::test_for_problem_tsp_builds_a_permutation_
typed_session covers the one pre-existing test this task's widening
necessarily updates (it used to assert for_problem REJECTED
problems.tsp(...); that rejection is exactly what this task replaces).
Every other pre-existing test is unmodified.
"""
import pytest
import sezgi
from sezgi.algo import AlgoContext, BudgetExhausted, SolveResult


# ---------------------------------------------------------------------
# EvalSession.for_problem over a permutation-typed (TSP) problem: kind,
# dim/n, evaluate (counting + budget honored/exhausted), invalid tours.
# ---------------------------------------------------------------------

def test_tsp_session_kind_and_dim():
    p = sezgi.problems.tsp("berlin52")
    s = sezgi.EvalSession.for_problem(p, budget=100)
    assert s.kind() == "permutation"
    assert p.dim() == 52   # Block::Permutation { n }.dim() == n
    s.finish()


def test_tsp_session_counts_evals_across_batches():
    s = sezgi.EvalSession.for_problem(sezgi.problems.tsp("eil51"), budget=100)
    n = 51
    t1 = list(range(n))
    t2 = list(range(n - 1, -1, -1))
    fs = s.evaluate([t1, t2])
    assert len(fs) == 2 and s.evals_used() == 2
    fs2 = s.evaluate([t1])
    assert s.evals_used() == 3
    s.finish()


def test_tsp_session_budget_honored_all_or_nothing():
    n = 52
    s = sezgi.EvalSession.for_problem(sezgi.problems.tsp("berlin52"), budget=2)
    t = list(range(n))
    with pytest.raises(ValueError):
        s.evaluate([t, t, t])   # 3 requested against budget 2
    assert s.evals_used() == 0, "a rejected batch must not count any rows"
    s.evaluate([t, t])          # fits exactly
    assert s.evals_used() == 2
    s.finish()


def test_tsp_session_budget_exhausted_then_more_raises():
    n = 52
    s = sezgi.EvalSession.for_problem(sezgi.problems.tsp("berlin52"), budget=1)
    t = list(range(n))
    s.evaluate([t])
    assert s.evals_used() == 1
    with pytest.raises(ValueError):
        s.evaluate([t])
    assert s.evals_used() == 1, "the rejected call must not move the counter"
    s.finish()


def test_tsp_session_wrong_length_tour_rejected_with_honest_error():
    s = sezgi.EvalSession.for_problem(sezgi.problems.tsp("berlin52"), budget=10)
    with pytest.raises(ValueError, match="entries"):
        s.evaluate([list(range(51))])   # 51, expected 52
    assert s.evals_used() == 0
    s.finish()


def test_tsp_session_duplicate_city_rejected_with_honest_error():
    n = 52
    bad = list(range(n))
    bad[1] = bad[0]   # duplicate
    s = sezgi.EvalSession.for_problem(sezgi.problems.tsp("berlin52"), budget=10)
    with pytest.raises(ValueError, match="repeated"):
        s.evaluate([bad])
    assert s.evals_used() == 0
    s.finish()


def test_tsp_session_out_of_range_city_rejected_with_honest_error():
    n = 52
    bad = list(range(n))
    bad[0] = n   # out of range: valid indices are 0..n-1
    s = sezgi.EvalSession.for_problem(sezgi.problems.tsp("berlin52"), budget=10)
    with pytest.raises(ValueError, match="out-of-range"):
        s.evaluate([bad])
    assert s.evals_used() == 0
    s.finish()


def test_tsp_session_negative_city_rejected_with_honest_error():
    n = 52
    bad = list(range(n))
    bad[0] = -1
    s = sezgi.EvalSession.for_problem(sezgi.problems.tsp("berlin52"), budget=10)
    with pytest.raises(ValueError, match="out-of-range"):
        s.evaluate([bad])
    assert s.evals_used() == 0
    s.finish()


def test_tsp_session_evaluate_matches_tsp_tour_length():
    tour = list(range(52))
    s = sezgi.EvalSession.for_problem(sezgi.problems.tsp("berlin52"), budget=10)
    fs = s.evaluate([tour])
    assert fs[0] == sezgi.problems.tsp_tour_length("berlin52", tour)
    s.finish()


def test_tsp_session_best_tracks_min_and_returns_int_tour():
    s = sezgi.EvalSession.for_problem(sezgi.problems.tsp("berlin52"), budget=10)
    identity = list(range(52))
    reversed_tour = list(range(51, -1, -1))
    fs = s.evaluate([identity, reversed_tour])
    x, f = s.best()
    assert f == min(fs)
    assert isinstance(x[0], int), "best_x's tour entries must be ints, not floats"
    assert sorted(x) == list(range(52))
    s.finish()


def test_tsp_session_f_opt_is_published_optimum():
    s = sezgi.EvalSession.for_problem(sezgi.problems.tsp("berlin52"), budget=1)
    assert s.f_opt() == 7542.0
    s.finish()


def test_tsp_session_log_dir_rejected(tmp_path):
    with pytest.raises(ValueError):
        sezgi.EvalSession.for_problem(sezgi.problems.tsp("berlin52"), budget=10,
                                      log_dir=str(tmp_path))


def test_tsp_session_finish_then_any_method_raises():
    s = sezgi.EvalSession.for_problem(sezgi.problems.tsp("berlin52"), budget=10)
    s.finish()
    with pytest.raises(ValueError, match="session finished"):
        s.evaluate([list(range(52))])
    with pytest.raises(ValueError, match="session finished"):
        s.finish()


# ---------------------------------------------------------------------
# random_permutation(): validity (a real permutation of 0..n-1),
# determinism under a fixed seed, different draws across successive calls
# and across different seeds. Only valid on a permutation-typed session.
# ---------------------------------------------------------------------

def test_random_permutation_is_a_valid_permutation():
    s = sezgi.EvalSession.for_problem(sezgi.problems.tsp("berlin52"), budget=1,
                                      seed=1)
    tour = s.random_permutation()
    assert len(tour) == 52
    assert sorted(tour) == list(range(52))
    assert all(isinstance(c, int) for c in tour)
    s.finish()


def test_random_permutation_deterministic_under_fixed_seed():
    def draw(seed):
        s = sezgi.EvalSession.for_problem(sezgi.problems.tsp("eil51"), budget=1,
                                          seed=seed)
        t = s.random_permutation()
        s.finish()
        return t

    a = draw(42)
    b = draw(42)
    assert a == b, "same seed must reproduce the same first draw"


def test_random_permutation_different_seeds_diverge():
    def draw(seed):
        s = sezgi.EvalSession.for_problem(sezgi.problems.tsp("eil51"), budget=1,
                                          seed=seed)
        t = s.random_permutation()
        s.finish()
        return t

    assert draw(1) != draw(2)


def test_random_permutation_successive_draws_differ():
    s = sezgi.EvalSession.for_problem(sezgi.problems.tsp("eil51"), budget=1,
                                      seed=7)
    draws = [s.random_permutation() for _ in range(5)]
    s.finish()
    # every draw is individually valid...
    for t in draws:
        assert sorted(t) == list(range(51))
    # ...and the stream actually advances (not the same permutation every
    # call) -- vanishingly unlikely to collide by chance for n=51.
    assert len({tuple(t) for t in draws}) == len(draws)


def test_random_permutation_rejected_on_float_session():
    s = sezgi.EvalSession.for_problem(sezgi.bbob(1, 3, 1), budget=1)
    with pytest.raises(ValueError, match="float"):
        s.random_permutation()
    s.finish()


# ---------------------------------------------------------------------
# two_opt(tour, i, j): pure-Python hand fixtures. Semantics (documented in
# AlgoContext.two_opt's own docstring): reverses tour[i:j+1] -- positions i
# through j, 0-based, INCLUSIVE on both ends. Requires 0 <= i <= j < len.
# ---------------------------------------------------------------------

def _ctx():
    """A bare AlgoContext with no live session -- two_opt touches neither
    the session nor the RNG, so this is safe and avoids a needless session
    per test."""
    return AlgoContext(session=None, dim=0, bounds=None, seed=0, kind="permutation")


def test_two_opt_middle_segment():
    tour = [0, 1, 2, 3, 4, 5]
    # reverse positions 1..3 inclusive: [1, 2, 3] -> [3, 2, 1]
    assert _ctx().two_opt(tour, 1, 3) == [0, 3, 2, 1, 4, 5]
    assert tour == [0, 1, 2, 3, 4, 5], "two_opt must not mutate its argument"


def test_two_opt_adjacent_positions_is_a_plain_swap():
    tour = [0, 1, 2, 3, 4]
    assert _ctx().two_opt(tour, 1, 2) == [0, 2, 1, 3, 4]


def test_two_opt_i_equals_j_is_a_no_op():
    tour = [0, 1, 2, 3, 4]
    for k in range(len(tour)):
        assert _ctx().two_opt(tour, k, k) == tour


def test_two_opt_full_range_reverses_whole_tour():
    tour = [0, 1, 2, 3, 4]
    assert _ctx().two_opt(tour, 0, len(tour) - 1) == [4, 3, 2, 1, 0]


def test_two_opt_leading_segment():
    tour = [0, 1, 2, 3, 4]
    # reverse positions 0..2 inclusive
    assert _ctx().two_opt(tour, 0, 2) == [2, 1, 0, 3, 4]


def test_two_opt_trailing_segment():
    tour = [0, 1, 2, 3, 4]
    # reverse positions 2..4 inclusive
    assert _ctx().two_opt(tour, 2, 4) == [0, 1, 4, 3, 2]


def test_two_opt_rejects_i_greater_than_j():
    with pytest.raises(ValueError):
        _ctx().two_opt([0, 1, 2, 3], 2, 1)


def test_two_opt_rejects_out_of_range_indices():
    ctx = _ctx()
    with pytest.raises(ValueError):
        ctx.two_opt([0, 1, 2, 3], -1, 2)
    with pytest.raises(ValueError):
        ctx.two_opt([0, 1, 2, 3], 0, 4)


def test_two_opt_applied_twice_with_same_bounds_is_identity():
    tour = [0, 1, 2, 3, 4, 5, 6]
    ctx = _ctx()
    once = ctx.two_opt(tour, 2, 5)
    twice = ctx.two_opt(once, 2, 5)
    assert twice == tour


def test_two_opt_result_stays_a_valid_permutation():
    tour = list(range(52))
    ctx = _ctx()
    reversed_mid = ctx.two_opt(tour, 10, 40)
    assert sorted(reversed_mid) == list(range(52))
    assert len(reversed_mid) == len(tour)


# ---------------------------------------------------------------------
# End-to-end: an Algorithm subclass driving a TSP local search through
# AlgoContext (kind/n/bounds/random_permutation/two_opt/evaluate).
# ---------------------------------------------------------------------

class RandomRestartTwoOpt(sezgi.Algorithm):
    """Toy permutation-typed algorithm: one random tour per step, plus one
    first-improvement 2-opt probe from an adjacent segment -- just enough to
    exercise the whole permutation surface end-to-end through solve()."""

    def setup(self, ctx):
        assert ctx.kind == "permutation"
        assert ctx.bounds is None
        self.tour = ctx.random_permutation()
        (self.best_f,) = ctx.evaluate([self.tour])

    def step(self, ctx):
        i = ctx.rng.randrange(0, ctx.n - 1)
        j = ctx.rng.randrange(i, ctx.n)
        candidate = ctx.two_opt(self.tour, i, j)
        (f,) = ctx.evaluate([candidate])
        if f < self.best_f:
            self.tour, self.best_f = candidate, f


def test_algo_context_permutation_surface_via_solve():
    res = RandomRestartTwoOpt().solve(sezgi.problems.tsp("berlin52"),
                                      budget=50, seed=3)
    assert isinstance(res, SolveResult)
    assert res.evals_used == 50
    assert res.f_opt == 7542.0
    assert res.gap == res.best_f - res.f_opt
    assert len(res.best_x) == 52
    assert sorted(res.best_x) == list(range(52))


def test_algo_context_permutation_random_point_raises():
    seen = {}

    class Probe(sezgi.Algorithm):
        def setup(self, ctx):
            seen["kind"] = ctx.kind
            seen["n"] = ctx.n
            seen["bounds"] = ctx.bounds
            with pytest.raises(ValueError):
                ctx.random_point()
            ctx.evaluate([ctx.random_permutation()])

        def step(self, ctx):
            ctx.evaluate([ctx.random_permutation()])

    Probe().solve(sezgi.problems.tsp("st70"), budget=3, seed=1)
    assert seen["kind"] == "permutation"
    assert seen["n"] == 70
    assert seen["bounds"] is None


def test_algo_context_solve_deterministic_same_seed():
    a = RandomRestartTwoOpt().solve(sezgi.problems.tsp("berlin52"), budget=40, seed=9)
    b = RandomRestartTwoOpt().solve(sezgi.problems.tsp("berlin52"), budget=40, seed=9)
    assert (a.best_f, a.best_x, a.evals_used) == (b.best_f, b.best_x, b.evals_used)
