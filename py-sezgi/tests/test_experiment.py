"""M2c Task 11: Python bindings for experiments + statistics."""
import sezgi
import pytest


def tiny_experiment_toml():
    return """
        name = "tiny"
        seeds = [1, 2]
        budgets = [400]

        [[algorithms]]
        name = "de"
        preset = { kind = "de_rand_1", pop_size = 8 }

        [[algorithms]]
        name = "rs"
        preset = { kind = "random_search", pop_size = 8 }

        [[problems]]
        suite = "bbob"
        fid = 1
        dim = 5
        instances = [1]
    """


def test_run_experiment_end_to_end():
    spec_toml = tiny_experiment_toml()
    r1 = sezgi.run_experiment(spec_toml, journal=None, parallel=True)
    r2 = sezgi.run_experiment(spec_toml, journal=None, parallel=True)

    # 2 algos x 1 problem x 1 instance x 2 seeds x 1 budget = 4 records
    assert len(r1) == 4

    expected_keys = {"algo", "fid", "dim", "instance", "seed", "budget",
                      "best_f", "f_opt", "gap", "evals_used", "wall_secs"}
    for rec in r1:
        assert expected_keys.issubset(rec.keys())
        assert abs(rec["gap"] - (rec["best_f"] - rec["f_opt"])) < 1e-12
        assert rec["evals_used"] <= 400

    # Deterministic across two calls: compare everything except wall_secs
    # (legitimately varies run to run).
    def strip_wall(records):
        return [{k: v for k, v in r.items() if k != "wall_secs"} for r in records]

    assert strip_wall(r1) == strip_wall(r2)


def test_experiment_resume(tmp_path):
    spec_toml = tiny_experiment_toml()
    journal_path = str(tmp_path / "journal.jsonl")

    r1 = sezgi.run_experiment(spec_toml, journal=journal_path, parallel=False)
    assert len(r1) == 4

    lines_after_first = sum(1 for _ in open(journal_path))

    r2 = sezgi.run_experiment(spec_toml, journal=journal_path, parallel=False)
    assert len(r2) == 4

    lines_after_second = sum(1 for _ in open(journal_path))
    assert lines_after_second == lines_after_first, \
        "resume of a completed experiment must not append new journal lines"

    def strip_wall(records):
        return [{k: v for k, v in r.items() if k != "wall_secs"} for r in records]

    assert strip_wall(r1) == strip_wall(r2)


def test_experiment_resume_tolerates_formatting_only_edit(tmp_path):
    """M2d Task 9: the journal's spec hash is over the canonical TOML form,
    so appending a comment (whitespace to the parser) must not invalidate
    an existing journal on resume."""
    spec_toml = tiny_experiment_toml()
    journal_path = str(tmp_path / "journal.jsonl")

    r1 = sezgi.run_experiment(spec_toml, journal=journal_path, parallel=False)
    assert len(r1) == 4
    lines_after_first = sum(1 for _ in open(journal_path))

    edited_toml = spec_toml + "\n# harmless comment\n"

    r2 = sezgi.run_experiment(edited_toml, journal=journal_path, parallel=False)
    assert len(r2) == 4, "resume with a formatting-only edit must not re-run anything"

    lines_after_second = sum(1 for _ in open(journal_path))
    assert lines_after_second == lines_after_first, \
        "resume with a formatting-only edit must not append new journal lines"

    def strip_wall(records):
        return [{k: v for k, v in r.items() if k != "wall_secs"} for r in records]

    assert strip_wall(r1) == strip_wall(r2)


def test_experiment_resume_rejects_real_field_change(tmp_path):
    """A real field change (budget) must still be caught as a spec change."""
    spec_toml = tiny_experiment_toml()
    journal_path = str(tmp_path / "journal.jsonl")

    r1 = sezgi.run_experiment(spec_toml, journal=journal_path, parallel=False)
    assert len(r1) == 4

    changed_toml = spec_toml.replace("budgets = [400]", "budgets = [500]")
    assert changed_toml != spec_toml

    with pytest.raises(ValueError, match="spec has changed"):
        sezgi.run_experiment(changed_toml, journal=journal_path, parallel=False)


def test_stats_bindings_smoke():
    # friedman on a 3x3 matrix (3 problems x 3 algorithms).
    results = [
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 1.0],
        [3.0, 1.0, 2.0],
    ]
    fr = sezgi.stats.friedman(results)
    assert set(fr.keys()) == {"statistic", "p_value", "mean_ranks"}
    assert isinstance(fr["statistic"], float)
    assert isinstance(fr["p_value"], float)
    assert len(fr["mean_ranks"]) == 3

    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    b = [1.5, 1.8, 3.5, 3.8, 4.8, 6.5]
    wr = sezgi.stats.wilcoxon(a, b)
    assert set(wr.keys()) == {"w_statistic", "z", "p_value", "n_effective", "method"}
    assert isinstance(wr["n_effective"], int)
    assert wr["method"] in {"exact", "normal_approx"}

    delta = sezgi.stats.cliffs_delta(a, b)
    assert isinstance(delta, float)
    mag = sezgi.stats.cliffs_magnitude(delta)
    assert isinstance(mag, str)

    br1 = sezgi.stats.bayesian_signed_rank(a, b, rope=0.0, samples=200, seed=1)
    br2 = sezgi.stats.bayesian_signed_rank(a, b, rope=0.0, samples=200, seed=1)
    assert set(br1.keys()) == {"p_left", "p_rope", "p_right"}
    assert br1 == br2, "same seed must give bit-identical result"

    rankings = [[0, 1, 2], [1, 0, 2], [0, 2, 1]]
    pl = sezgi.stats.plackett_luce(rankings)
    assert set(pl.keys()) == {"worths", "p_best", "iterations"}
    assert len(pl["worths"]) == 3
    assert isinstance(pl["iterations"], int)


def test_stats_wilcoxon_nan_raises():
    """M2d-1 Task 7: non-finite input to any stats entry point raises
    ValueError naming the non-finite policy."""
    a = [1.0, float("nan"), 3.0]
    b = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="non-finite"):
        sezgi.stats.wilcoxon(a, b)


def test_wilcoxon_exact_small_n():
    # Mirrors crates/stats/src/pairwise.rs::exact_small_n_no_ties_uses_exact_distribution.
    # n=8, |d| ranks 1..8 (distinct), only the smallest-|d| pair (|d|=1) has
    # the sign opposite the rest -> W- = 1, W+ = 35, W = 1. Exact two-sided
    # p = min(1, 2 * count(sum <= 1) / 2^8) = 2*2/256 = 0.015625 (empty
    # subset + {1}); see task-6-report.md for the full derivation and the
    # independent SciPy cross-check.
    a = [99.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0]
    b = [100.0] * 8

    wr = sezgi.stats.wilcoxon(a, b)
    assert wr["method"] == "exact"
    assert wr["w_statistic"] == 1.0
    assert wr["p_value"] == 0.015625
    assert wr["n_effective"] == 8


def reporting_grid_toml():
    """2 algos x f1 x 5 instances x 2 seeds x 2 budgets -- the grid used by
    the results_matrix/per_budget_packages tests (M2d-1 Task 8), matching
    the README experiments quickstart."""
    return """
        name = "reporting-grid"
        seeds = [1, 2]
        budgets = [400, 800]

        [[algorithms]]
        name = "de"
        preset = { kind = "de_rand_1", pop_size = 8 }

        [[algorithms]]
        name = "rs"
        preset = { kind = "random_search", pop_size = 8 }

        [[problems]]
        suite = "bbob"
        fid = 1
        dim = 5
        instances = [1, 2, 3, 4, 5]
    """


def test_results_matrix_shape_and_labels():
    records = sezgi.run_experiment(reporting_grid_toml(), parallel=True)
    algo_names, problem_labels, matrix = sezgi.results_matrix(records, budget=400)

    assert algo_names == ["de", "rs"]
    assert problem_labels == ["f1d5i1", "f1d5i2", "f1d5i3", "f1d5i4", "f1d5i5"]
    assert len(matrix) == 5, "one row per problem"
    assert all(len(row) == 2 for row in matrix), "one column per algorithm"


def test_per_budget_packages_ascending_and_no_nan():
    records = sezgi.run_experiment(reporting_grid_toml(), parallel=True)
    packages = sezgi.per_budget_packages(records)

    assert len(packages) == 2, "one package per distinct budget"
    budgets = [b for b, _ in packages]
    assert budgets == sorted(budgets), "ascending budget order"
    assert budgets == [400, 800]

    for _, pkg in packages:
        assert "NaN" not in pkg["latex_summary"]
        assert set(pkg.keys()) == {
            "friedman", "nemenyi_cd", "pairwise_wilcoxon_holm", "cliffs",
            "bayes", "plackett_luce", "latex_summary", "latex_tests",
        }, "package dict must match stats.paper_package's shape exactly"


def test_results_matrix_missing_cell_raises():
    records = sezgi.run_experiment(reporting_grid_toml(), parallel=True)
    incomplete = [r for r in records
                  if not (r["algo"] == "rs" and r["instance"] == 3 and r["budget"] == 400)]

    with pytest.raises(ValueError, match="missing run record"):
        sezgi.results_matrix(incomplete, budget=400)


def test_results_matrix_unknown_aggregate_raises():
    records = sezgi.run_experiment(reporting_grid_toml(), parallel=True)
    with pytest.raises(ValueError, match="unknown aggregate"):
        sezgi.results_matrix(records, budget=400, aggregate="bogus")
    with pytest.raises(ValueError, match="unknown aggregate"):
        sezgi.per_budget_packages(records, aggregate="bogus")


def test_paper_package_latex_nonempty():
    algo_names = ["A", "B", "C"]
    problem_names = [f"P{i}" for i in range(6)]
    results = [
        [1.0, 2.0, 3.0],
        [1.5, 2.2, 2.9],
        [1.2, 1.9, 3.1],
        [0.9, 2.3, 3.2],
        [1.3, 2.1, 2.8],
        [1.1, 2.0, 3.0],
    ]

    pkg1 = sezgi.stats.paper_package(algo_names, problem_names, results,
                                      rope=0.1, samples=1000, seed=42)
    pkg2 = sezgi.stats.paper_package(algo_names, problem_names, results,
                                      rope=0.1, samples=1000, seed=42)

    assert "\\toprule" in pkg1["latex_summary"]
    assert "\\toprule" in pkg1["latex_tests"]
    assert pkg1 == pkg2, "paper_package must be deterministic for a fixed seed"

    assert len(pkg1["friedman"]["mean_ranks"]) == 3
    assert len(pkg1["pairwise_wilcoxon_holm"]) == 3
    assert len(pkg1["cliffs"]) == 3
    assert len(pkg1["bayes"]) == 3
    assert len(pkg1["plackett_luce"]["worths"]) == 3
