"""M3-4 Task 1: EvalSession generalized over any continuous problem.

Exercises `EvalSession.for_problem(...)` -- the staticmethod that builds an
ask/tell session over any continuous (float) `Problem` handle
(`sezgi.bbob(...)`, `sezgi.problems.cec2022(...)`, `sezgi.from_callable(...)`),
not just BBOB -- plus the new `Problem` handle accessors `dim()`/`bounds()`/
`optimum()`. Mirrors crates/bench/src/session.rs's own
`generic_session_over_cec2022`/`with_log_requires_known_optimum` tests
through the FFI.

Fix round 1 (controller ruling): the callable calling convention
(batched-per-population vs per-point) is carried on the `from_callable(...)`
HANDLE itself via `vectorized` (default `True`, the original frozen
convention), not decided by which consumer (`solve()` vs
`EvalSession.for_problem`) happens to use the handle -- so `for_problem`'s
own scalar-lambda tests below now pass `vectorized=False` explicitly, and
two new tests prove each consumer honors BOTH values of the flag.
"""
import math

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
    p = sezgi.from_callable(lambda x: sum(v * v for v in x), -5.0, 5.0, 3, vectorized=False)
    s = sezgi.EvalSession.for_problem(p, budget=10)
    s.evaluate([[1.0, 2.0, 3.0]])
    assert s.f_opt() is None
    assert s.best()[1] == 14.0


def test_for_problem_tsp_rejected():
    with pytest.raises(ValueError):
        sezgi.EvalSession.for_problem(sezgi.problems.tsp("berlin52"), budget=10)


def test_for_problem_log_dir_requires_known_optimum(tmp_path):
    p = sezgi.from_callable(lambda x: x[0], 0.0, 1.0, 1, vectorized=False)
    with pytest.raises(ValueError):
        sezgi.EvalSession.for_problem(p, budget=10, log_dir=str(tmp_path))


def test_for_problem_cec2022_logging_roundtrip(tmp_path):
    """M3-5 (scope ruling 2): T1's suite-aware record key (a "suite" field
    on RunKey/the record dict) lets read_ioh_records/results_matrix tell a
    CEC 2022 run apart from a BBOB run at the same (fid, dim, instance,
    seed, budget), so the M3-4 final-review's collision concern no longer
    applies -- log_dir on a for_problem(cec2022(...)) session now works,
    widening the BBOB-only restriction to BBOB + CEC 2022."""
    p = sezgi.problems.cec2022(1, 10)
    s = sezgi.EvalSession.for_problem(p, budget=20, log_dir=str(tmp_path),
                                      algo_name="probe", seed=7)
    s.evaluate([[0.0] * 10] * 20)
    s.finish()
    recs = sezgi.read_ioh_records(str(tmp_path), [20])
    assert len(recs) == 1
    assert recs[0]["suite"] == "sezgi-cec2022"
    assert recs[0]["f_opt"] == 300.0


def test_for_problem_callable_and_f0_log_dir_still_rejected(tmp_path):
    """Callable/F0 arms carry no fid/suite identity in their SessionMeta
    (no on-disk IOH record can be built for them), so they keep the
    rejection even after BBOB + CEC 2022 are both allowed to log."""
    p = sezgi.from_callable(lambda x: x[0], 0.0, 1.0, 1, vectorized=False)
    with pytest.raises(ValueError):
        sezgi.EvalSession.for_problem(p, budget=5, log_dir=str(tmp_path))
    with pytest.raises(ValueError):
        sezgi.EvalSession.for_problem(sezgi.bias.f0(2, seed=0), budget=5,
                                      log_dir=str(tmp_path))


def test_for_problem_from_callable_vectorized_default_works_batch_style():
    """`for_problem` honors the handle's DEFAULT (vectorized=True)
    convention too: f is called once per evaluate() batch, with the whole
    batch as a 2-D (n, dim) numpy array, and must return n values."""
    def sphere(xs):
        assert xs.shape[1] == 3
        return (xs ** 2).sum(axis=1)

    p = sezgi.from_callable(sphere, -5.0, 5.0, 3)  # vectorized=True (default)
    s = sezgi.EvalSession.for_problem(p, budget=10)
    fs = s.evaluate([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    assert fs[0] == 14.0 and fs[1] == 0.0
    assert s.evals_used() == 2
    assert s.f_opt() is None


def test_solve_accepts_vectorized_false_handle():
    """solve() honors a `vectorized=False` handle too: f is called once per
    point, with a 1-D length-dim numpy array, and must return a scalar."""
    def sphere_scalar(x):
        return sum(v * v for v in x)

    p = sezgi.from_callable(sphere_scalar, -5.0, 5.0, 3, vectorized=False)
    spec = sezgi.presets.random_search(pop_size=5, budget=50)
    r = sezgi.solve(spec, p, master_seed=1)
    assert math.isfinite(r["best_f"])


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
