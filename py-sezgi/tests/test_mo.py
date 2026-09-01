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


def test_pareto_front_wfg1_wfg2_are_none():
    """WFG1/WFG2's Pareto front has no closed dominance-filtered form
    (sezgi_problems::wfg's own module doc, "pareto_front decisions") --
    unlike WFG3-WFG9, which DO have a closed form."""
    assert sezgi.mo.pareto_front("wfg1", dim=None, n=5, m=2) is None
    assert sezgi.mo.pareto_front("wfg2", dim=None, n=5, m=2) is None


def test_pareto_front_wfg4_is_some():
    front = sezgi.mo.pareto_front("wfg4", dim=None, n=5, m=2)
    assert front is not None
    assert len(front) == 5
    assert all(len(row) == 2 for row in front)


# ---- sezgi.mo.evaluate: ZDT5 fixtures (T4/T6 hand fixtures, cross-checked
# against pymoo 0.6.2 -- crates/problems/src/zdt.rs's own test module) -----
#
# All-zeros: u(x1)=0 -> f1=1. Every group u=0<5 -> v=2, g=10*2=20,
# f2=g/f1=20. All-ones: u(x1)=30 -> f1=31. Every group u=5 -> v=1, g=10,
# f2=10/31.

def test_evaluate_zdt5_all_zeros_fixture():
    f = sezgi.mo.evaluate("zdt5", [0.0] * 80)
    assert f == [1.0, 20.0]


def test_evaluate_zdt5_all_ones_fixture():
    f = sezgi.mo.evaluate("zdt5", [1.0] * 80)
    assert f[0] == 31.0
    assert f[1] == pytest.approx(10.0 / 31.0, abs=1e-12)


def test_evaluate_zdt5_mixed_fixture():
    """x1: 7 leading ones (of 30) -> u=7, f1=8. Each of the 10 groups is
    [1,0,1,0,1] -> u=3<5, v=5, g=50, f2=50/8=6.25."""
    x1 = [1.0] * 7 + [0.0] * 23
    group = [1.0, 0.0, 1.0, 0.0, 1.0]
    f = sezgi.mo.evaluate("zdt5", x1 + group * 10)
    assert f == [8.0, 6.25]


# ---- sezgi.mo.evaluate/evaluate_constraints: DTLZ8/DTLZ9 fixtures (T2 hand
# fixtures -- crates/problems/src/dtlz.rs's own test module) -----------------

def test_evaluate_dtlz8_m2_dim4_fixtures():
    f = sezgi.mo.evaluate("dtlz8", [0.0, 0.0, 0.0, 0.0], dim=4, m=2)
    assert f == [0.0, 0.0]
    g = sezgi.mo.evaluate_constraints("dtlz8", [0.0, 0.0, 0.0, 0.0], dim=4, m=2)
    assert g[0] == pytest.approx(-1.0, abs=1e-12)
    assert g[1] == float("inf")

    f = sezgi.mo.evaluate("dtlz8", [1.0, 1.0, 1.0, 1.0], dim=4, m=2)
    assert f == [1.0, 1.0]
    g = sezgi.mo.evaluate_constraints("dtlz8", [1.0, 1.0, 1.0, 1.0], dim=4, m=2)
    assert g[0] == pytest.approx(4.0, abs=1e-12)
    assert g[1] == float("inf")

    f = sezgi.mo.evaluate("dtlz8", [1.0, 0.0, 0.0, 0.0], dim=4, m=2)
    assert f == [0.5, 0.0]
    g = sezgi.mo.evaluate_constraints("dtlz8", [1.0, 0.0, 0.0, 0.0], dim=4, m=2)
    assert g[0] == pytest.approx(1.0, abs=1e-12)


def test_evaluate_dtlz8_m3_dim6_mixed_fixture():
    xs = [1.0, 0.0, 0.5, 0.5, 0.2, 0.2]
    f = sezgi.mo.evaluate("dtlz8", xs, dim=6, m=3)
    assert f[0] == pytest.approx(0.5, abs=1e-12)
    assert f[1] == pytest.approx(0.5, abs=1e-12)
    assert f[2] == pytest.approx(0.2, abs=1e-12)
    g = sezgi.mo.evaluate_constraints("dtlz8", xs, dim=6, m=3)
    assert g[0] == pytest.approx(1.2, abs=1e-9)
    assert g[1] == pytest.approx(1.2, abs=1e-9)
    assert g[2] == pytest.approx(0.4, abs=1e-9)


def test_evaluate_dtlz9_m2_dim4_fixtures():
    f = sezgi.mo.evaluate("dtlz9", [1.0, 1.0, 1.0, 1.0], dim=4, m=2)
    assert f == [2.0, 2.0]
    g = sezgi.mo.evaluate_constraints("dtlz9", [1.0, 1.0, 1.0, 1.0], dim=4, m=2)
    assert len(g) == 1
    assert g[0] == pytest.approx(7.0, abs=1e-12)

    g = sezgi.mo.evaluate_constraints("dtlz9", [0.0, 0.0, 0.0, 0.0], dim=4, m=2)
    assert g[0] == pytest.approx(-1.0, abs=1e-12)


def test_evaluate_constraints_none_for_unconstrained_dtlz():
    assert sezgi.mo.evaluate_constraints("dtlz1", [0.5] * 6, dim=6, m=3) is None
    assert sezgi.mo.evaluate_constraints("zdt1", [0.5] * 5, dim=5) is None


# ---- sezgi.mo.evaluate: WFG fixture (committed
# crates/problems/tests/data/wfg_reference_values.json, which=1, m=2, k=4,
# l=4 -- toolkit-cross-checked, 1e-8 tolerance per that fixture's own
# "tolerance_rationale") ------------------------------------------------------

def test_evaluate_wfg1_zero_and_max_fixture():
    f = sezgi.mo.evaluate("wfg1", [0.0] * 8, m=2, k=4, l=4)
    assert f == pytest.approx([1.0, 5.0], abs=1e-8)

    f = sezgi.mo.evaluate("wfg1", [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0], m=2, k=4, l=4)
    assert f == pytest.approx([3.0, 1.0], abs=1e-8)


def test_evaluate_wfg_default_k_l():
    """k/l default per sezgi_problems::wfg's own module doc: k=4 for m=2,
    k=2*(m-1) for m>=3, l=20 -- n=k+l. Probed via the dimension-mismatch
    error message (no direct dim accessor on an mo problem handle)."""
    with pytest.raises(ValueError, match="24 coordinates"):
        sezgi.mo.evaluate("wfg4", [0.0], m=2)  # k=4, l=20 -> n=24
    with pytest.raises(ValueError, match="24 coordinates"):
        sezgi.mo.evaluate("wfg4", [0.0], m=3)  # k=2*(3-1)=4, l=20 -> n=24
    with pytest.raises(ValueError, match="26 coordinates"):
        sezgi.mo.evaluate("wfg4", [0.0], m=4)  # k=2*(4-1)=6, l=20 -> n=26


# ---- error tests: dim/m/k/l mismatched to a problem family -----------------

def test_evaluate_zdt5_with_m_raises_value_error():
    with pytest.raises(ValueError):
        sezgi.mo.evaluate("zdt5", [0.0] * 80, m=2)


def test_evaluate_wfg_with_dim_raises_value_error():
    with pytest.raises(ValueError, match="dim is not accepted for wfg"):
        sezgi.mo.evaluate("wfg1", [0.0] * 8, dim=8, m=2, k=4, l=4)


def test_evaluate_wfg_bad_k_raises_value_error():
    """k must divide (m-1); k=5 does not divide m-1=2 for m=3."""
    with pytest.raises(ValueError, match="k %"):
        sezgi.mo.evaluate("wfg1", [0.0] * 9, m=3, k=5, l=4)


def test_evaluate_wfg2_odd_l_raises_value_error():
    """WFG2/WFG3's non-separable reduction requires l even."""
    with pytest.raises(ValueError, match="l must be even"):
        sezgi.mo.evaluate("wfg2", [0.0] * 9, m=2, k=4, l=5)


def test_evaluate_zdt_with_k_or_l_raises_value_error():
    with pytest.raises(ValueError, match="wfg-only"):
        sezgi.mo.evaluate("zdt1", [0.0] * 5, dim=5, k=4)


def test_evaluate_dtlz_with_k_or_l_raises_value_error():
    with pytest.raises(ValueError, match="wfg-only"):
        sezgi.mo.evaluate("dtlz2", [0.0] * 5, dim=5, m=3, l=20)


def test_evaluate_dtlz8_dim_equal_m_raises_value_error():
    """DTLZ8/DTLZ9 require dim > m strictly (the constraint-surface
    construction, dtlz.rs's own DtlzError::BadDimConstraintSurface)."""
    with pytest.raises(ValueError, match="dim must be > m"):
        sezgi.mo.evaluate("dtlz8", [0.0, 0.0, 0.0], dim=3, m=3)


def test_hypervolume_missing_ref_point_is_type_error():
    """ref_point has NO default -- calling without it is a plain Python
    TypeError (missing argument), not a ValueError this binding raises."""
    with pytest.raises(TypeError):
        sezgi.mo.hypervolume([[0.25, 0.75]])


# ---- sezgi.mo.nsga2: ZDT5 end-to-end (binary genotype path) ----------------

def test_nsga2_zdt5_dim_none_runs_binary_path():
    r = sezgi.mo.nsga2("zdt5", dim=None, pop_size=8, budget=40, seed=0)
    assert set(r.keys()) == {"individuals", "objectives", "front0", "evals_used"}
    assert len(r["individuals"]) == 8
    assert all(len(row) == 80 for row in r["individuals"])
    # Binary blocks are flattened to 0.0/1.0 (genotype_to_flat_vec's own
    # decision) -- every value is exactly one of those two floats.
    assert all(x in (0.0, 1.0) for row in r["individuals"] for x in row)
    assert all(len(row) == 2 for row in r["objectives"])


def test_nsga2_zdt5_deterministic_seeded():
    kwargs = dict(problem="zdt5", dim=None, pop_size=8, budget=200, seed=20260901)
    r1 = sezgi.mo.nsga2(**kwargs)
    r2 = sezgi.mo.nsga2(**kwargs)
    assert r1["objectives"] == r2["objectives"]
    assert r1["individuals"] == r2["individuals"]


# ---- sezgi.mo.nsga2: constrained (DTLZ8/DTLZ9) violations surfacing -------

def test_nsga2_dtlz8_surfaces_violations():
    r = sezgi.mo.nsga2("dtlz8", dim=6, m=3, pop_size=8, budget=40, seed=0)
    assert "violations" in r
    assert len(r["violations"]) == len(r["individuals"])
    assert all(v <= 0.0 for v in r["violations"])


def test_nsga2_dtlz9_surfaces_violations():
    r = sezgi.mo.nsga2("dtlz9", dim=6, m=3, pop_size=8, budget=40, seed=0)
    assert "violations" in r
    assert len(r["violations"]) == len(r["individuals"])


def test_nsga2_unconstrained_problem_has_no_violations_key():
    """violations is ABSENT (not None) for an unconstrained problem --
    mirrors solve()'s own skipped_empty_runs convention."""
    r = sezgi.mo.nsga2("dtlz1", dim=6, m=3, pop_size=8, budget=40, seed=0)
    assert "violations" not in r
    r = sezgi.mo.nsga2("zdt1", dim=5, pop_size=8, budget=40, seed=0)
    assert "violations" not in r


# ---- sezgi.mo.hypervolume: general-M exact hypervolume (WFG algorithm) ----
#
# Fixture copied from crates/stats/src/moo_indicators.rs's own
# `hypervolume_four_point_constant_dimension_factors_m4` test.

def test_hypervolume_general_m_hand_fixture_exact():
    front = [[1.0, 8.0, 9.0, 5.0], [9.0, 1.0, 8.0, 5.0], [8.0, 9.0, 1.0, 5.0]]
    ref_point = [10.0, 10.0, 10.0, 8.0]
    assert sezgi.mo.hypervolume(front, ref_point) == 147.0


def test_hypervolume_general_m_delegates_to_2d_at_m_equals_2():
    front = [[0.25, 0.75], [0.5, 0.5], [0.75, 0.25]]
    ref_point = [1.0, 1.0]
    assert sezgi.mo.hypervolume(front, ref_point) == sezgi.mo.hypervolume_2d(front, ref_point)


def test_hypervolume_general_m_empty_front_is_zero():
    """Unlike hypervolume_2d, an empty front is NOT an error for the
    general-M hypervolume -- it returns 0.0 (the algorithm's own base
    case)."""
    assert sezgi.mo.hypervolume([], [1.0, 1.0, 1.0]) == 0.0


def test_hypervolume_general_m_bad_dimension_mismatch_raises_value_error():
    with pytest.raises(ValueError):
        sezgi.mo.hypervolume([[0.25, 0.75, 0.5]], [1.0, 1.0])


# ---- sezgi.mo.nsga2(log_dir=...) / sezgi.mo.read_moa: sezgi-moa logging ----

def test_nsga2_log_dir_requires_label():
    with pytest.raises(ValueError, match="label is required"):
        sezgi.mo.nsga2("zdt1", dim=5, pop_size=8, budget=40, seed=0, log_dir="/tmp/does-not-matter")


def test_nsga2_logged_run_end_to_end(tmp_path):
    log_dir = str(tmp_path)
    r = sezgi.mo.nsga2("zdt1", dim=5, pop_size=8, budget=60, seed=123,
                        log_dir=log_dir, label="zdt1run")
    path = tmp_path / "zdt1run-s123.moa"
    assert path.exists()

    d = sezgi.mo.read_moa(str(path))
    assert d["algo"] == "nsga2"
    assert d["problem"] == "zdt1run"
    assert d["m"] == 2
    assert d["seed"] == 123
    assert d["budget"] == 60
    assert d["kind"] == "float"
    assert len(d["records"]) >= 1
    for rec in d["records"]:
        assert set(rec.keys()) == {"eval_index", "objectives", "genotype"}
        assert len(rec["objectives"]) == 2
        assert len(rec["genotype"]) == 5

    # archive is mutually nondominated (plain Pareto dominance, minimization)
    archive = d["archive"]
    assert len(archive) >= 1
    def dominates(a, b):
        return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))
    for i, a in enumerate(archive):
        for j, b in enumerate(archive):
            if i != j:
                assert not dominates(a, b), f"{a} dominates {b}"

    # r's own evals_used matches the archive's own budget bound
    assert r["evals_used"] <= 60


def test_read_moa_zdt5_binary_kind_and_genotype_bools(tmp_path):
    log_dir = str(tmp_path)
    sezgi.mo.nsga2("zdt5", dim=None, pop_size=8, budget=40, seed=5,
                   log_dir=log_dir, label="zdt5run")
    path = tmp_path / "zdt5run-s5.moa"
    d = sezgi.mo.read_moa(str(path))
    assert d["kind"] == "binary"
    for rec in d["records"]:
        assert len(rec["genotype"]) == 80
        assert all(isinstance(b, bool) for b in rec["genotype"])


def test_read_moa_at_parameter_reconstructs_earlier_archive(tmp_path):
    log_dir = str(tmp_path)
    sezgi.mo.nsga2("zdt1", dim=5, pop_size=8, budget=60, seed=9,
                   log_dir=log_dir, label="zdt1at")
    path = tmp_path / "zdt1at-s9.moa"
    d_full = sezgi.mo.read_moa(str(path))
    d_early = sezgi.mo.read_moa(str(path), at=8)  # only the init population
    assert len(d_early["archive"]) <= len(d_full["archive"])


def test_nsga2_logged_run_byte_determinism(tmp_path):
    """Two identical runs (same problem/pop_size/budget/seed/label) must
    produce byte-identical .moa files -- T9's own byte-determinism
    requirement (no timestamp in the header)."""
    log_dir1 = tmp_path / "run1"
    log_dir2 = tmp_path / "run2"
    log_dir1.mkdir()
    log_dir2.mkdir()

    sezgi.mo.nsga2("zdt1", dim=5, pop_size=8, budget=60, seed=77,
                   log_dir=str(log_dir1), label="det")
    sezgi.mo.nsga2("zdt1", dim=5, pop_size=8, budget=60, seed=77,
                   log_dir=str(log_dir2), label="det")

    b1 = (log_dir1 / "det-s77.moa").read_bytes()
    b2 = (log_dir2 / "det-s77.moa").read_bytes()
    assert b1 == b2


def test_read_moa_missing_file_raises_value_error(tmp_path):
    with pytest.raises(ValueError):
        sezgi.mo.read_moa(str(tmp_path / "does-not-exist.moa"))
