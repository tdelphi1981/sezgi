"""M3-2 Task 9: Python bindings for the multi-objective layer (sezgi.mo).

Binds T6's NSGA-II runner (`sezgi_components::nsga2::nsga2_run`), the
ZDT/DTLZ benchmark suites (`sezgi_problems::{Zdt, Dtlz}`), and the exact
2-objective hypervolume / IGD indicators (`sezgi_stats::{hypervolume_2d,
igd}`).

Budgets/pop sizes are kept tiny throughout (CI speed) -- see each test's own
comment for the actual numbers; none of these runs takes more than a few
milliseconds.

pop_size validation (CORRECTED from the task-9 brief): the merged NSGA-II
runner requires `pop_size >= 4 AND pop_size % 4 == 0` -- a KanGAL-faithful
tightening of the naive "even, >= 4" rule (see
`crates/components/src/nsga2.rs`'s "`popsize % 4 == 0` requirement" doc
section). `test_nsga2_invalid_pop_size_odd` and
`test_nsga2_invalid_pop_size_even_not_multiple_of_4` cover both halves of
that tightening.
"""
import sezgi
import pytest


# ---- sezgi.mo.nsga2: dict shape --------------------------------------------

def test_nsga2_dict_shape():
    r = sezgi.mo.nsga2("zdt1", dim=5, pop_size=8, budget=40, seed=0)
    assert set(r.keys()) == {"individuals", "objectives", "front0", "evals_used"}
    assert len(r["individuals"]) == 8
    assert len(r["objectives"]) == 8
    assert all(len(row) == 5 for row in r["individuals"])
    assert all(len(row) == 2 for row in r["objectives"])
    assert all(isinstance(x, float) for row in r["individuals"] for x in row)
    assert all(isinstance(x, float) for row in r["objectives"] for x in row)
    assert isinstance(r["front0"], list)
    assert all(isinstance(i, int) for i in r["front0"])
    assert all(0 <= i < 8 for i in r["front0"])
    # evals_used: initial pop_size (8) consumed; budget=40 is not an exact
    # multiple of pop_size beyond that, but must be >= pop_size and a
    # multiple of it up to the budget-tail rule -- just check it's in range.
    assert isinstance(r["evals_used"], int)
    assert 8 <= r["evals_used"] <= 40


def test_nsga2_zdt4_two_block_genotype_flattens_correctly():
    """ZDT4 has TWO Float blocks (x1 in [0,1], rest in [-5,5]) -- individuals
    must be flattened to a single flat row of length `dim`, not left as a
    nested/blocked structure."""
    r = sezgi.mo.nsga2("zdt4", dim=4, pop_size=8, budget=32, seed=0)
    assert all(len(row) == 4 for row in r["individuals"])
    for row in r["individuals"]:
        assert 0.0 <= row[0] <= 1.0
        for xi in row[1:]:
            assert -5.0 <= xi <= 5.0


def test_nsga2_dtlz_requires_m():
    r = sezgi.mo.nsga2("dtlz2", dim=5, m=3, pop_size=8, budget=32, seed=0)
    assert set(r.keys()) == {"individuals", "objectives", "front0", "evals_used"}
    assert all(len(row) == 3 for row in r["objectives"])


# ---- sezgi.mo.nsga2: determinism -------------------------------------------

def test_nsga2_deterministic_seeded_zdt1():
    """Tiny budget (200), pop_size=8, zdt1 dim=5, seed=20260830: two runs
    with the same seed must produce bit-identical objectives (and
    individuals)."""
    kwargs = dict(problem="zdt1", dim=5, pop_size=8, budget=200, seed=20260830)
    r1 = sezgi.mo.nsga2(**kwargs)
    r2 = sezgi.mo.nsga2(**kwargs)
    assert r1["objectives"] == r2["objectives"]
    assert r1["individuals"] == r2["individuals"]
    assert r1["front0"] == r2["front0"]
    assert r1["evals_used"] == r2["evals_used"]


def test_nsga2_different_seed_differs():
    kwargs = dict(problem="zdt1", dim=5, pop_size=8, budget=200)
    r1 = sezgi.mo.nsga2(seed=1, **kwargs)
    r2 = sezgi.mo.nsga2(seed=2, **kwargs)
    assert r1["objectives"] != r2["objectives"]


# ---- sezgi.mo.nsga2: errors -------------------------------------------------

def test_nsga2_invalid_problem_string_raises_value_error():
    with pytest.raises(ValueError):
        sezgi.mo.nsga2("not-a-problem", dim=5, pop_size=8, budget=40, seed=0)


def test_nsga2_zdt5_raises_value_error():
    """ZDT5 is a binary-coded problem, explicitly out of scope."""
    with pytest.raises(ValueError):
        sezgi.mo.nsga2("zdt5", dim=5, pop_size=8, budget=40, seed=0)


def test_nsga2_invalid_pop_size_odd():
    with pytest.raises(ValueError):
        sezgi.mo.nsga2("zdt1", dim=5, pop_size=7, budget=40, seed=0)


def test_nsga2_invalid_pop_size_even_not_multiple_of_4():
    """pop_size=6 is even and >= 4, but NOT a multiple of 4 -- the merged
    runner's tightened check (KanGAL's double-permutation tournament
    pairing requires pop_size % 4 == 0) must still reject it."""
    with pytest.raises(ValueError, match="4"):
        sezgi.mo.nsga2("zdt1", dim=5, pop_size=6, budget=40, seed=0)


def test_nsga2_dtlz_without_m_raises_value_error():
    with pytest.raises(ValueError):
        sezgi.mo.nsga2("dtlz2", dim=5, pop_size=8, budget=40, seed=0)


def test_nsga2_m_given_for_zdt_raises_value_error():
    with pytest.raises(ValueError):
        sezgi.mo.nsga2("zdt1", dim=5, m=3, pop_size=8, budget=40, seed=0)


# ---- sezgi.mo.hypervolume_2d: fixtures matching the Rust hand fixtures -----
#
# Copied VALUES from crates/stats/src/moo_indicators.rs's own test fixtures
# (`hypervolume_2d_hand_fixture_exact`, `hypervolume_2d_same_f1_tie_not_double_counted`).

def test_hypervolume_2d_hand_fixture_exact():
    front = [[0.25, 0.75], [0.5, 0.5], [0.75, 0.25]]
    ref_point = [1.0, 1.0]
    assert sezgi.mo.hypervolume_2d(front, ref_point) == 0.375


def test_hypervolume_2d_same_f1_tie_not_double_counted():
    """front = [(1,3), (2,2), (2,1), (3,0.5)], ref_point = (4,4): (2,2) is
    weakly dominated by (2,1) and contributes nothing. Kept points, ascending
    f1: (1,3), (2,1), (3,0.5) -> areas 1 + 3 + 3.5 = 7.5 exactly."""
    front = [[1.0, 3.0], [2.0, 2.0], [2.0, 1.0], [3.0, 0.5]]
    ref_point = [4.0, 4.0]
    assert sezgi.mo.hypervolume_2d(front, ref_point) == 7.5


def test_hypervolume_2d_bad_ref_point_length_raises_value_error():
    with pytest.raises(ValueError):
        sezgi.mo.hypervolume_2d([[0.25, 0.75]], [1.0, 1.0, 1.0])


def test_hypervolume_2d_empty_front_raises_value_error():
    with pytest.raises(ValueError):
        sezgi.mo.hypervolume_2d([], [1.0, 1.0])


# ---- sezgi.mo.igd: fixture matching the Rust hand fixture ------------------

def test_igd_hand_fixture_exact():
    """reference_front = [(0,0), (4,0)], front = [(1,0), (3,0)]: both
    nearest distances are exactly 1.0 -> IGD = 1.0 exactly."""
    front = [[1.0, 0.0], [3.0, 0.0]]
    reference_front = [[0.0, 0.0], [4.0, 0.0]]
    assert sezgi.mo.igd(front, reference_front) == 1.0


def test_igd_empty_front_raises_value_error():
    with pytest.raises(ValueError):
        sezgi.mo.igd([], [[0.0, 0.0]])


# ---- sezgi.mo.pareto_front: shape + spot values matching the Rust source ---

def test_pareto_front_zdt1_shape_and_spot_values():
    front = sezgi.mo.pareto_front("zdt1", dim=30, n=100)
    assert front is not None
    assert len(front) == 100
    assert all(len(row) == 2 for row in front)
    # zdt1 front: f1 = i/(n-1), f2 = 1 - sqrt(f1) -- exact at the endpoints.
    assert front[0] == [0.0, 1.0]
    assert front[-1][0] == 1.0
    assert front[-1][1] == pytest.approx(0.0, abs=1e-12)


def test_pareto_front_dtlz5_m_greater_than_3_is_none():
    """DTLZ5/DTLZ6's degenerate curve is only verified for m <= 3 (see
    crates/problems/src/dtlz.rs's module doc); pareto_front must return None
    for m=4."""
    front = sezgi.mo.pareto_front("dtlz5", dim=13, n=10, m=4)
    assert front is None


def test_pareto_front_dtlz_without_m_raises_value_error():
    with pytest.raises(ValueError):
        sezgi.mo.pareto_front("dtlz2", dim=5, n=10)


def test_pareto_front_invalid_problem_string_raises_value_error():
    with pytest.raises(ValueError):
        sezgi.mo.pareto_front("not-a-problem", dim=5, n=10)
