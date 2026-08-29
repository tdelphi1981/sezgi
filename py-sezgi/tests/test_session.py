"""M2d-3 Task 3: Python EvalSession (ask/tell) binding.

Mirrors crates/bench/src/session.rs's own unit tests through the FFI
(counting/budget/best), plus a pure-Python random-search proof that an
externally-driven session produces a parseable, bit-identical IOH archive.
"""
import random
import struct

import pytest
import sezgi


def row(xs):
    return list(xs)


def test_counting_across_two_batches():
    s = sezgi.EvalSession(fid=1, dim=3, instance=1, budget=100)
    s.evaluate([row([0.0, 0.0, 0.0]), row([1.0, 1.0, 1.0]), row([2.0, 2.0, 2.0])])
    assert s.evals_used() == 3
    s.evaluate([row([3.0, 3.0, 3.0]), row([4.0, 4.0, 4.0])])
    assert s.evals_used() == 5
    assert s.budget() == 100


def test_batch_crossing_budget_errs_and_does_not_count():
    s = sezgi.EvalSession(fid=1, dim=3, instance=1, budget=2)
    with pytest.raises(ValueError):
        s.evaluate([row([0.0, 0.0, 0.0]), row([1.0, 1.0, 1.0]), row([2.0, 2.0, 2.0])])
    assert s.evals_used() == 0

    # A batch that fits exactly still succeeds and counts fully.
    s.evaluate([row([0.0, 0.0, 0.0]), row([1.0, 1.0, 1.0])])
    assert s.evals_used() == 2


def test_best_tracking_matches_hand_tracked_min():
    s = sezgi.EvalSession(fid=1, dim=2, instance=1, budget=100)
    xs = [row([0.5, 0.5]), row([-1.0, 2.0]), row([3.0, -3.0]), row([0.1, 0.1])]
    fs = s.evaluate(xs)

    hand_best = None
    for i, f in enumerate(fs):
        if hand_best is None or hand_best[1] > f:
            hand_best = (i, f)
    idx, want_f = hand_best

    got_x, got_f = s.best()
    assert got_f == want_f
    assert got_x == xs[idx]


def test_f_opt_is_a_float():
    s = sezgi.EvalSession(fid=1, dim=2, instance=1, budget=10)
    assert isinstance(s.f_opt(), float)
    assert s.best() is None, "best() is None before any evaluation"


def test_dimension_mismatch_raises():
    s = sezgi.EvalSession(fid=1, dim=3, instance=1, budget=100)
    with pytest.raises(ValueError, match="row"):
        s.evaluate([row([0.0, 0.0, 0.0]), row([1.0, 1.0])])
    assert s.evals_used() == 0


def test_nan_row_raises_and_does_not_count():
    s = sezgi.EvalSession(fid=1, dim=2, instance=1, budget=100)
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="row"):
            s.evaluate([row([0.0, 0.0]), row([bad, 0.0])])
        assert s.evals_used() == 0


def test_evaluate_after_finish_raises():
    s = sezgi.EvalSession(fid=1, dim=2, instance=1, budget=10)
    s.evaluate([row([0.0, 0.0])])
    s.finish()
    with pytest.raises(ValueError, match="session finished"):
        s.evaluate([row([1.0, 1.0])])


def test_any_method_after_finish_raises():
    s = sezgi.EvalSession(fid=1, dim=2, instance=1, budget=10)
    s.finish()
    with pytest.raises(ValueError, match="session finished"):
        s.evals_used()
    with pytest.raises(ValueError, match="session finished"):
        s.budget()
    with pytest.raises(ValueError, match="session finished"):
        s.best()
    with pytest.raises(ValueError, match="session finished"):
        s.f_opt()
    with pytest.raises(ValueError, match="session finished"):
        s.finish()


def test_finish_without_log_is_ok():
    s = sezgi.EvalSession(fid=1, dim=2, instance=1, budget=10)
    s.evaluate([row([0.0, 0.0])])
    s.finish()  # must not raise


def test_pure_python_random_search_produces_parseable_ioh_archive(tmp_path):
    """The engine-inside-out proof: a pure stdlib-only random search drives
    the session's ask/tell loop with no Rust algorithm involved, and the
    resulting IOH archive round-trips through read_ioh_records with a
    best_f that is bit-identical (via struct.pack) to the script's own
    independently tracked best."""
    log_dir = str(tmp_path / "log")
    dim = 5
    budget = 200
    seed = 42

    s = sezgi.EvalSession(fid=1, dim=dim, instance=1, budget=budget,
                           log_dir=log_dir, algo_name="pyrs", seed=seed)

    rng = random.Random(seed)
    tracked_best = None
    for _ in range(budget):
        x = [rng.uniform(-5.0, 5.0) for _ in range(dim)]
        f = s.evaluate([x])[0]
        if tracked_best is None or f < tracked_best:
            tracked_best = f
    s.finish()

    records = sezgi.read_ioh_records(log_dir, [200])
    assert len(records) == 1
    rec = records[0]
    assert rec["seed"] == seed
    assert struct.pack("<d", rec["best_f"]) == struct.pack("<d", tracked_best), \
        "read-back best_f must be bit-identical to the script's own tracked best"
