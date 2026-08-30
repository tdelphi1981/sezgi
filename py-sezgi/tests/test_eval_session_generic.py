"""M3-4 Task 1: EvalSession generalized over any continuous problem.

Exercises `EvalSession.for_problem(...)` -- the staticmethod that builds an
ask/tell session over any continuous (float) `Problem` handle
(`sezgi.bbob(...)`, `sezgi.problems.cec2022(...)`, `sezgi.from_callable(...)`),
not just BBOB -- plus the new `Problem` handle accessors `dim()`/`bounds()`/
`optimum()`. Mirrors crates/bench/src/session.rs's own
`generic_session_over_cec2022`/`with_log_requires_known_optimum` tests
through the FFI.
"""
import pytest
import sezgi


def test_for_problem_cec2022_counts_and_f_opt():
    p = sezgi.problems.cec2022(1, 10)
    s = sezgi.EvalSession.for_problem(p, budget=50)
    fs = s.evaluate([[0.0] * 10, [1.0] * 10])
    assert len(fs) == 2 and s.evals_used() == 2
    assert s.f_opt() == sezgi.problems.cec2022_f_star(1)  # 300.0, bit-exact
    s.finish()


def test_for_problem_from_callable_f_opt_none():
    p = sezgi.from_callable(lambda x: sum(v * v for v in x), -5.0, 5.0, 3)
    s = sezgi.EvalSession.for_problem(p, budget=10)
    s.evaluate([[1.0, 2.0, 3.0]])
    assert s.f_opt() is None
    assert s.best()[1] == 14.0


def test_for_problem_tsp_rejected():
    with pytest.raises(ValueError):
        sezgi.EvalSession.for_problem(sezgi.problems.tsp("berlin52"), budget=10)


def test_for_problem_log_dir_requires_known_optimum(tmp_path):
    p = sezgi.from_callable(lambda x: x[0], 0.0, 1.0, 1)
    with pytest.raises(ValueError):
        sezgi.EvalSession.for_problem(p, budget=10, log_dir=str(tmp_path))


def test_problem_accessors():
    b = sezgi.bbob(1, 5, 1)
    assert b.dim() == 5 and b.bounds() == (-5.0, 5.0)
    c = sezgi.problems.cec2022(3, 10)
    assert c.dim() == 10 and c.bounds() == (-100.0, 100.0)
    assert c.optimum() == 600.0
    with pytest.raises(ValueError):
        sezgi.problems.tsp("berlin52").bounds()


def test_bbob_session_budget_all_or_nothing_unchanged():
    s = sezgi.EvalSession(fid=1, dim=2, instance=1, budget=2)
    with pytest.raises(ValueError):
        s.evaluate([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    assert s.evals_used() == 0
