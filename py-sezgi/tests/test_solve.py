import json
import pathlib
import struct
import sezgi
import pytest


def test_preset_is_dict_with_stages():
    spec = sezgi.presets.de_rand_1(pop_size=20, budget=2000)
    assert spec["pop_size"] == 20
    assert spec["stages"][0]["generator"]["kind"] == "gen/de"


def test_preset_gwo_is_dict_with_stages():
    """M2d-3 Task 5: sezgi.presets.gwo() parses as JSON with the expected shape."""
    spec = sezgi.presets.gwo(pop_size=30, budget=2000)
    assert spec["pop_size"] == 30
    assert spec["stages"][0]["generator"]["kind"] == "gen/gwo"
    assert spec["stages"][0]["replacer"]["kind"] == "replace/generational"


def test_bbob_solve_deterministic():
    p = sezgi.bbob(fid=1, dim=5, instance=1)
    spec = sezgi.presets.de_rand_1(pop_size=20, budget=2000)
    r1 = sezgi.solve(spec, p, master_seed=42)
    r2 = sezgi.solve(spec, p, master_seed=42)
    assert r1["best_f"] == r2["best_f"]
    assert r1["evals_used"] <= 2000
    assert len(r1["best_x"]) == 5


def test_callable_problem():
    calls = []

    def sphere(X):
        assert X.shape == (10, 3)
        calls.append(X.shape[0])
        return ((X - 1.0) ** 2).sum(axis=1)

    p = sezgi.from_callable(sphere, lo=-5.0, hi=5.0, dim=3)
    spec = sezgi.presets.de_rand_1(pop_size=10, budget=500)
    r = sezgi.solve(spec, p, master_seed=1)
    assert r["best_f"] < 0.5
    assert all(c == 10 for c in calls), "evaluation must be a SINGLE call per population"


def test_invalid_spec_raises_clear_error():
    p = sezgi.bbob(fid=1, dim=5, instance=1)
    spec = sezgi.presets.de_rand_1(20, 500)
    spec["stages"][0]["generator"]["kind"] = "yok/boyle"
    try:
        sezgi.solve(spec, p, master_seed=1)
        assert False, "expected an error"
    except ValueError as e:
        assert "yok/boyle" in str(e)


def test_log_dir_writes_ioh(tmp_path):
    p = sezgi.bbob(fid=1, dim=5, instance=1)
    spec = sezgi.presets.de_rand_1(20, 1000)
    sezgi.solve(spec, p, master_seed=42, log_dir=str(tmp_path), algo_name="de")
    dat = tmp_path / "de" / "data_f1_Sphere" / "IOHprofiler_f1_DIM5.dat"
    assert dat.exists()
    assert dat.read_text().splitlines()[0] == '"evaluations" "raw_y"'


def test_callable_wrong_length_raises():
    def bad_sphere(X):
        # Deliberately returns a list of the wrong (too short) length (list — not ndarray).
        return [float(((row - 1.0) ** 2).sum()) for row in X][:-1]

    p = sezgi.from_callable(bad_sphere, lo=-5.0, hi=5.0, dim=3)
    spec = sezgi.presets.de_rand_1(pop_size=10, budget=500)
    with pytest.raises(RuntimeError, match="wrong length"):
        sezgi.solve(spec, p, master_seed=1)


def test_callback_exception_propagates():
    def boom(X):
        raise ValueError("intentional")

    p = sezgi.from_callable(boom, lo=-5.0, hi=5.0, dim=3)
    spec = sezgi.presets.de_rand_1(pop_size=10, budget=500)
    with pytest.raises(ValueError, match="intentional"):
        sezgi.solve(spec, p, master_seed=1)


def test_callable_log_dir_raises(tmp_path):
    def sphere(xs):
        return [sum((v - 1.0) ** 2 for v in x) for x in xs]

    p = sezgi.from_callable(sphere, lo=-5.0, hi=5.0, dim=3)
    spec = sezgi.presets.de_rand_1(pop_size=10, budget=500)
    with pytest.raises(ValueError, match="log_dir is only supported"):
        sezgi.solve(spec, p, master_seed=1, log_dir=str(tmp_path))


def test_cross_language_determinism():
    golden = json.loads(
        (pathlib.Path(__file__).parents[2] / "tests" / "golden"
         / "de_bbob_f1_seed42.json").read_text())
    gp = golden["problem"]
    p = sezgi.bbob(fid=gp["fid"], dim=gp["dim"], instance=gp["instance"])
    spec = sezgi.presets.de_rand_1(20, 2000)
    r = sezgi.solve(spec, p, master_seed=golden["master_seed"],
                    run_id=golden["run_id"])
    got_bits = format(struct.unpack("<Q", struct.pack("<d", r["best_f"]))[0], "016x")
    assert got_bits == golden["best_f_bits"], \
        "Python trajectory diverged from the Rust golden value"


def test_all_presets_solve_smoke():
    """M2c Task 1: every newly-exposed preset bridge solves bbob f1 (dim 5)
    with a small budget without error, and produces a finite, actually-used
    result."""
    dim = 5
    p = sezgi.bbob(fid=1, dim=dim, instance=1)
    budget = 2000

    specs = {
        "de_best_1": sezgi.presets.de_best_1(pop_size=20, budget=budget),
        "gwo": sezgi.presets.gwo(pop_size=30, budget=budget),
        "jde": sezgi.presets.jde(pop_size=20, budget=budget),
        "shade": sezgi.presets.shade(pop_size=20, budget=budget),
        "lshade": sezgi.presets.lshade(dim=dim, budget=budget),
        "cmaes": sezgi.presets.cmaes(pop_size=8, budget=budget),
        "cmaes_ipop": sezgi.presets.cmaes_ipop(dim=dim, budget=budget),
        "sa": sezgi.presets.sa(budget=budget),
        "random_search": sezgi.presets.random_search(pop_size=20, budget=budget),
        "nelder_mead": sezgi.presets.nelder_mead(dim=dim, budget=budget),
        "es_mu_plus_lambda": sezgi.presets.es_mu_plus_lambda(pop_size=20, budget=budget),
    }

    for name, spec in specs.items():
        r = sezgi.solve(spec, p, master_seed=42)
        assert r["evals_used"] > 0, f"{name}: no evaluations were used"
        assert r["best_f"] == r["best_f"], f"{name}: best_f is NaN"  # NaN check
        assert r["best_f"] not in (float("inf"), float("-inf")), \
            f"{name}: best_f is not finite ({r['best_f']})"


def test_log_dir_reports_skipped_runs(tmp_path):
    p = sezgi.bbob(fid=1, dim=5, instance=1)
    spec = sezgi.presets.de_rand_1(20, 1000)

    # With log_dir: should report skipped_empty_runs
    result_with_log = sezgi.solve(spec, p, master_seed=42, log_dir=str(tmp_path), algo_name="de")
    assert "skipped_empty_runs" in result_with_log
    assert result_with_log["skipped_empty_runs"] == 0

    # Without log_dir: should NOT have skipped_empty_runs key
    result_without_log = sezgi.solve(spec, p, master_seed=42)
    assert "skipped_empty_runs" not in result_without_log


def test_es_mu_plus_lambda_default_dist_solves():
    """M2d-1 Task 7: the default (gaussian) es_mu_plus_lambda preset parses
    as JSON and solves bbob f1 without error."""
    spec = sezgi.presets.es_mu_plus_lambda(pop_size=10, budget=500)
    assert spec["name"] == "es/mu+lambda"
    assert spec["stages"][0]["generator"]["kind"] == "gen/step"
    assert spec["stages"][0]["generator"]["dist"]["kind"] == "gaussian"

    p = sezgi.bbob(fid=1, dim=5, instance=1)
    r = sezgi.solve(spec, p, master_seed=1)
    assert r["evals_used"] > 0
    assert r["best_f"] == r["best_f"]  # not NaN


def test_es_mu_plus_lambda_unknown_dist_raises():
    with pytest.raises(ValueError, match="unknown distribution"):
        sezgi.presets.es_mu_plus_lambda(10, 500, dist="banana")
