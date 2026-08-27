import json
import os
import pathlib
import struct
import sezgi


def test_preset_is_dict_with_stages():
    spec = sezgi.presets.de_rand_1(pop_size=20, budget=2000)
    assert spec["pop_size"] == 20
    assert spec["stages"][0]["generator"]["kind"] == "gen/de"


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

    def sphere(xs):
        calls.append(len(xs))
        return [sum((v - 1.0) ** 2 for v in x) for x in xs]

    p = sezgi.from_callable(sphere, lo=-5.0, hi=5.0, dim=3)
    spec = sezgi.presets.de_rand_1(pop_size=10, budget=500)
    r = sezgi.solve(spec, p, master_seed=1)
    assert r["best_f"] < 0.5
    assert all(c == 10 for c in calls), "değerlendirme popülasyon başına TEK çağrı olmalı"


def test_invalid_spec_raises_clear_error():
    p = sezgi.bbob(fid=1, dim=5, instance=1)
    spec = sezgi.presets.de_rand_1(20, 500)
    spec["stages"][0]["generator"]["kind"] = "yok/boyle"
    try:
        sezgi.solve(spec, p, master_seed=1)
        assert False, "hata bekleniyordu"
    except ValueError as e:
        assert "yok/boyle" in str(e)


def test_log_dir_writes_ioh(tmp_path):
    p = sezgi.bbob(fid=1, dim=5, instance=1)
    spec = sezgi.presets.de_rand_1(20, 1000)
    sezgi.solve(spec, p, master_seed=42, log_dir=str(tmp_path), algo_name="de")
    dat = tmp_path / "de" / "data_f1_Sphere" / "IOHprofiler_f1_DIM5.dat"
    assert dat.exists()
    assert dat.read_text().splitlines()[0] == '"evaluations" "raw_y"'


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
        "Python yörüngesi Rust altın değerinden saptı"
