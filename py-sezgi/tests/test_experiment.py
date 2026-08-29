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
    assert set(wr.keys()) == {"w_statistic", "z", "p_value", "n_effective"}
    assert isinstance(wr["n_effective"], int)

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
