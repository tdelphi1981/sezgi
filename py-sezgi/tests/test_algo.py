"""M3-4 Task 2: sezgi.Algorithm ABC, AlgoContext, and the setup/step driver.

Exercises the pure-Python algorithm-authoring surface over T1's generalized
EvalSession ask/tell core: subclass sezgi.Algorithm, implement setup()/step(),
call solve(). RandomSearch below is the reference toy subclass used
throughout (and by later M3-4 tasks porting real algorithms onto this ABC).
"""
import pytest
import sezgi
from sezgi.algo import AlgoContext, BudgetExhausted, SolveResult


class RandomSearch(sezgi.Algorithm):
    """Uniform random search: one batch of `batch` points per step."""
    def __init__(self, batch=10):
        self.batch = batch

    def setup(self, ctx):
        ctx.evaluate([ctx.random_point() for _ in range(self.batch)])

    def step(self, ctx):
        ctx.evaluate([ctx.random_point() for _ in range(self.batch)])


def test_solve_bbob_exhausts_budget_and_returns_result():
    res = RandomSearch(batch=10).solve(sezgi.bbob(1, 5, 1), budget=200, seed=42)
    assert isinstance(res, SolveResult)
    assert res.evals_used == 200          # 10-per-step fits 200 exactly
    assert res.budget == 200
    assert res.f_opt is not None
    assert res.gap == res.best_f - res.f_opt
    assert len(res.best_x) == 5
    assert res.algo == "randomsearch"     # default: cls.__name__.lower()

def test_solve_is_deterministic():
    a = RandomSearch().solve(sezgi.bbob(1, 5, 1), budget=100, seed=7)
    b = RandomSearch().solve(sezgi.bbob(1, 5, 1), budget=100, seed=7)
    assert (a.best_f, a.best_x, a.evals_used) == (b.best_f, b.best_x, b.evals_used)

def test_solve_cec2022_and_gap_vs_f_star():
    res = RandomSearch().solve(sezgi.problems.cec2022(3, 10), budget=100, seed=1)
    assert res.f_opt == 600.0
    assert res.gap >= 0.0

def test_solve_from_callable_gap_none():
    p = sezgi.from_callable(lambda x: sum(v * v for v in x), -1.0, 1.0, 2, vectorized=False)
    res = RandomSearch().solve(p, budget=50, seed=1)
    assert res.f_opt is None and res.gap is None

def test_partial_final_batch_stops_cleanly():
    # budget 95, batch 10: 9 full batches (90), 10th raises BudgetExhausted
    # inside step -> driver stops; the last 5 evals are never spent.
    res = RandomSearch(batch=10).solve(sezgi.bbob(1, 5, 1), budget=95, seed=3)
    assert res.evals_used == 90

def test_no_progress_step_raises():
    class Lazy(sezgi.Algorithm):
        def setup(self, ctx): ctx.evaluate([ctx.random_point()])
        def step(self, ctx): pass  # consumes nothing
    with pytest.raises(RuntimeError, match="consumed no budget"):
        Lazy().solve(sezgi.bbob(1, 2, 1), budget=10, seed=1)

def test_setup_alone_never_evaluating_raises():
    class Never(sezgi.Algorithm):
        def setup(self, ctx): pass
        def step(self, ctx): pass
    with pytest.raises(RuntimeError):
        Never().solve(sezgi.bbob(1, 2, 1), budget=10, seed=1)

def test_ctx_surface():
    seen = {}
    class Probe(sezgi.Algorithm):
        def setup(self, ctx):
            seen["dim"], seen["bounds"] = ctx.dim, ctx.bounds
            seen["budget"] = ctx.budget
            ctx.evaluate([ctx.random_point()])
            seen["used"], seen["remaining"] = ctx.evals_used, ctx.remaining
            seen["best"] = ctx.best()
        def step(self, ctx):
            ctx.evaluate([ctx.random_point()])
    Probe().solve(sezgi.bbob(1, 3, 1), budget=5, seed=2)
    assert seen["dim"] == 3 and seen["bounds"] == (-5.0, 5.0)
    assert seen["budget"] == 5 and seen["used"] == 1 and seen["remaining"] == 4
    assert seen["best"] is not None and len(seen["best"][0]) == 3

def test_custom_name_and_log_dir(tmp_path):
    class Named(RandomSearch):
        name = "my-rs"
    res = Named().solve(sezgi.bbob(1, 2, 1), budget=20, seed=1,
                        log_dir=str(tmp_path))
    assert res.algo == "my-rs"
    assert any(tmp_path.rglob("*.dat"))   # IOH tree written

def test_abstract_methods_enforced():
    with pytest.raises(TypeError):
        sezgi.Algorithm()  # abstract
