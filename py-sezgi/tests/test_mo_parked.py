"""M3-8 Task 9: the four M3-7 parked binding tests (approved scope ruling
6 -- deferred from M3-7's own T9, required to land here).

a) nsga2-on-WFG from Python (WFG4, m=2): seeded, front sanity + a
   determinism anchor.
b) A constrained, logged DTLZ8 run: the sezgi-moa log parses back
   (read_moa), and every logged individual is constraint-feasible --
   documented behavior, not assumed: nsga2_run_logged's own archive
   observer only ever streams a genotype whose summed violation is EXACTLY
   0.0 (crates/bench/src/mo_archive.rs, `let feasible = viol.map(|v|
   v[i] == 0.0).unwrap_or(true)`), so every record in a constrained .moa
   file is feasible BY CONSTRUCTION -- this test verifies that documented
   guarantee independently via mo.evaluate_constraints.
c) Degenerate hypervolume fronts: empty/single-point/non-dominating/
   duplicate inputs, for both hypervolume_2d and the general-M hypervolume
   -- asserting the DOCUMENTED behavior read from
   crates/stats/src/moo_indicators.rs (explicit ref_point always required;
   a point that does not strictly dominate ref_point contributes nothing;
   a duplicate point contributes nothing extra; hypervolume_2d rejects an
   empty front, but the general-M hypervolume treats an empty front as its
   own base case -- 0.0 -- even at M=2, since that check runs BEFORE the
   M==2 shortcut delegates to hypervolume_2d).
d) One nsga2 m>3 run (DTLZ2, m=4): seeded, sane objectives, and the
   general-M hypervolume accepts the resulting front.
"""
import os

import pytest
import sezgi


# ---------------------------------------------------------------------
# a) nsga2 on WFG4 (m=2) from Python
# ---------------------------------------------------------------------

def test_nsga2_wfg4_m2_front_sanity():
    r = sezgi.mo.nsga2("wfg4", dim=None, m=2, pop_size=8, budget=80, seed=20260901)
    assert set(r.keys()) == {"individuals", "objectives", "front0", "evals_used"}
    assert len(r["individuals"]) == 8
    # dim is derived from k/l (default k=4 for m=2, l=20) -> n = 24.
    assert all(len(row) == 24 for row in r["individuals"])
    assert all(len(row) == 2 for row in r["objectives"])
    # WFG toolkit functions are nonnegative and finite by construction.
    assert all(f >= 0.0 for row in r["objectives"] for f in row)
    assert 1 <= len(r["front0"]) <= 8


def test_nsga2_wfg4_m2_deterministic_seeded():
    kwargs = dict(problem="wfg4", dim=None, m=2, pop_size=8, budget=80, seed=20260901)
    r1 = sezgi.mo.nsga2(**kwargs)
    r2 = sezgi.mo.nsga2(**kwargs)
    assert r1["individuals"] == r2["individuals"]
    assert r1["objectives"] == r2["objectives"]
    assert r1["front0"] == r2["front0"]


# ---------------------------------------------------------------------
# b) constrained, logged DTLZ8 run: log parses + archive feasibility
# ---------------------------------------------------------------------

def test_nsga2_dtlz8_logged_run_parses_and_is_feasible(tmp_path):
    log_dir = str(tmp_path)
    r = sezgi.mo.nsga2("dtlz8", dim=6, m=3, pop_size=8, budget=64, seed=777,
                        log_dir=log_dir, label="dtlz8run")
    assert "violations" in r
    assert all(v <= 0.0 for v in r["violations"])

    path = tmp_path / "dtlz8run-s777.moa"
    assert path.exists()
    d = sezgi.mo.read_moa(str(path))
    assert d["algo"] == "nsga2"
    assert d["problem"] == "dtlz8run"
    assert d["m"] == 3
    assert d["seed"] == 777
    assert d["budget"] == 64
    assert d["kind"] == "float"
    assert len(d["records"]) >= 1

    # Documented guarantee: nsga2_run_logged's archive observer only streams
    # a genotype whose summed violation is EXACTLY 0.0 (feasible) --
    # verified independently here, per-record, via mo.evaluate_constraints
    # (g_j >= 0 means SATISFIED, per MoProblem::evaluate_constraints_batch's
    # own sign convention).
    for rec in d["records"]:
        assert set(rec.keys()) == {"eval_index", "objectives", "genotype"}
        assert len(rec["objectives"]) == 3
        g = sezgi.mo.evaluate_constraints("dtlz8", rec["genotype"], dim=6, m=3)
        assert g is not None
        assert all(gj >= 0.0 for gj in g), f"logged record {rec['eval_index']} is infeasible: {g}"

    # The reconstructed final archive (objective rows only) is non-empty
    # and every row has exactly m=3 objectives.
    assert len(d["archive"]) >= 1
    assert all(len(row) == 3 for row in d["archive"])


# ---------------------------------------------------------------------
# c) Degenerate hypervolume fronts
# ---------------------------------------------------------------------

def test_hypervolume_2d_empty_front_raises_but_general_m_at_m2_does_not():
    with pytest.raises(ValueError, match="empty"):
        sezgi.mo.hypervolume_2d([], [1.0, 1.0])
    # Same M=2 ref_point, general-M entry point: 0.0, no error -- the
    # empty-front base case is checked BEFORE `hypervolume` delegates to
    # `hypervolume_2d` for M==2 (moo_indicators.rs's own function body
    # order).
    assert sezgi.mo.hypervolume([], [1.0, 1.0]) == 0.0


def test_hypervolume_general_m_empty_front_is_zero_at_higher_m_too():
    assert sezgi.mo.hypervolume([], [1.0, 1.0, 1.0]) == 0.0
    assert sezgi.mo.hypervolume([], [1.0, 1.0, 1.0, 1.0]) == 0.0


def test_hypervolume_2d_single_dominating_point_exact_area():
    front = [[0.3, 0.4]]
    ref_point = [1.0, 1.0]
    assert sezgi.mo.hypervolume_2d(front, ref_point) == pytest.approx(0.7 * 0.6)


def test_hypervolume_2d_point_not_dominating_ref_contributes_zero():
    """A point on/outside the ref box does not STRICTLY dominate
    ref_point, so it is filtered out before the sweep -- 0.0, not an
    error (moo_indicators.rs's own "Convention": non-dominating points
    contribute an empty/degenerate rectangle)."""
    assert sezgi.mo.hypervolume_2d([[1.0, 0.5]], [1.0, 1.0]) == 0.0
    assert sezgi.mo.hypervolume_2d([[1.5, 0.1]], [1.0, 1.0]) == 0.0


def test_hypervolume_2d_duplicate_points_contribute_nothing_extra():
    front = [[0.3, 0.4], [0.3, 0.4], [0.3, 0.4]]
    ref_point = [1.0, 1.0]
    single = sezgi.mo.hypervolume_2d([[0.3, 0.4]], ref_point)
    assert sezgi.mo.hypervolume_2d(front, ref_point) == single


def test_hypervolume_general_m_point_not_dominating_ref_contributes_zero():
    assert sezgi.mo.hypervolume([[10.0, 5.0, 5.0]], [10.0, 10.0, 10.0]) == 0.0


def test_hypervolume_general_m_duplicate_points_contribute_nothing_extra():
    front = [[1.0, 8.0, 9.0, 5.0], [1.0, 8.0, 9.0, 5.0]]
    ref_point = [10.0, 10.0, 10.0, 8.0]
    single = sezgi.mo.hypervolume([[1.0, 8.0, 9.0, 5.0]], ref_point)
    assert sezgi.mo.hypervolume(front, ref_point) == single


# ---------------------------------------------------------------------
# d) One nsga2 m>3 run (DTLZ2, m=4)
# ---------------------------------------------------------------------

def test_nsga2_dtlz2_m4_sane_objectives_and_general_hypervolume_accepts_front():
    # k=10 (the literature's own canonical DTLZ2 distance-group size) ->
    # dim = m - 1 + k = 13.
    r = sezgi.mo.nsga2("dtlz2", dim=13, m=4, pop_size=8, budget=64, seed=20260901)
    assert len(r["individuals"]) == 8
    assert all(len(row) == 4 for row in r["objectives"])
    assert all(f >= 0.0 and f == f for row in r["objectives"] for f in row)  # finite, nonnegative
    assert 1 <= len(r["front0"]) <= 8

    front = [r["objectives"][i] for i in r["front0"]]
    ref_point = [max(row[j] for row in r["objectives"]) + 1.0 for j in range(4)]
    hv = sezgi.mo.hypervolume(front, ref_point)
    assert hv > 0.0
