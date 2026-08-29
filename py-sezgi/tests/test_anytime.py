"""M2d-2 Task 7: Python bindings for the analysis cluster (IOH reader,
ECDF/anytime curves, COCO export, Bayesian Plackett-Luce), plus the T1
deferred homework (checkpoint + log_dir resume must not re-log)."""
import os
import struct

import pytest

import sezgi


def tiny_experiment_toml(budgets="[500]"):
    return f"""
        name = "tiny-logged"
        seeds = [1, 2]
        budgets = {budgets}

        [[algorithms]]
        name = "de"
        preset = {{ kind = "de_rand_1", pop_size = 8 }}

        [[problems]]
        suite = "bbob"
        fid = 1
        dim = 5
        instances = [1]
    """


def multi_row_experiment_toml(budgets="[500]"):
    """2 algorithms x 2 instances -- friedman (inside per_budget_packages)
    requires at least 2 problems (rows) and 2 algorithms (columns)."""
    return f"""
        name = "multi-row-logged"
        seeds = [1, 2]
        budgets = {budgets}

        [[algorithms]]
        name = "de"
        preset = {{ kind = "de_rand_1", pop_size = 8 }}

        [[algorithms]]
        name = "rs"
        preset = {{ kind = "random_search", pop_size = 8 }}

        [[problems]]
        suite = "bbob"
        fid = 1
        dim = 5
        instances = [1, 2]
    """


def _record_bits(rec):
    """A record dict with every float field replaced by its exact IEEE-754
    bit pattern (via struct.pack), so equality is a bit-for-bit comparison
    rather than a float `==` (which would also pass here, but the brief
    asks for the bits check explicitly)."""
    float_keys = {"best_f", "f_opt", "gap"}
    out = {}
    for k, v in rec.items():
        if k in float_keys:
            out[k] = struct.pack("<d", v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------
# run_experiment(log_dir=...) -> read_ioh_records round trip
# ---------------------------------------------------------------------

def test_log_dir_round_trip_bit_identical_at_full_budget(tmp_path):
    """CORE test: a SINGLE-budget experiment (per the curtailed-view
    ruling in `sezgi_bench::ioh_records`'s doc comment -- a reconstructed
    view at a budget SMALLER than the run's logged budget is a curtailed
    view of the same trajectory, not guaranteed to equal an independent
    run planned at that smaller budget, so this test only compares at the
    run's one, full, logged budget)."""
    spec_toml = multi_row_experiment_toml(budgets="[500]")
    log_dir = str(tmp_path / "logs")

    in_memory = sezgi.run_experiment(spec_toml, log_dir=log_dir, parallel=False)
    assert len(in_memory) == 8  # 2 algos x 1 problem x 2 instances x 2 seeds

    disk = sezgi.read_ioh_records(log_dir, [500])
    assert len(disk) == 8

    in_memory_by_key = {
        (r["algo"], r["fid"], r["dim"], r["instance"], r["seed"], r["budget"]): r
        for r in in_memory
    }
    disk_by_key = {
        (r["algo"], r["fid"], r["dim"], r["instance"], r["seed"], r["budget"]): r
        for r in disk
    }
    assert set(in_memory_by_key) == set(disk_by_key)

    for key, mem_rec in in_memory_by_key.items():
        disk_rec = disk_by_key[key]
        # wall_secs is not reconstructable from disk (ioh_records always
        # reports 0.0 -- see crates/bench/src/ioh_read.rs), so compare
        # every other field bit-for-bit.
        mem_bits = _record_bits({k: v for k, v in mem_rec.items() if k != "wall_secs"})
        disk_bits = _record_bits({k: v for k, v in disk_rec.items() if k != "wall_secs"})
        assert mem_bits == disk_bits, f"disk round trip must be bit-identical for {key}"
        assert disk_rec["wall_secs"] == 0.0

    # per_budget_packages accepts the disk-reconstructed records unchanged
    # (same record-dict shape as run_experiment) -- this is the "prove
    # per_budget_packages(records) just works" requirement.
    packages = sezgi.per_budget_packages(disk)
    assert len(packages) == 1
    budget, pkg = packages[0]
    assert budget == 500
    assert "NaN" not in pkg["latex_summary"]


def test_read_ioh_records_curtailed_view_present_and_evals_used_capped(tmp_path):
    """A budget SMALLER than the run's logged budget is a valid curtailed
    view (present, no error), with evals_used = min(budget, run.evals) --
    but is NOT asserted equal to anything from an independent smaller-
    budget run (see the doc comment on `ioh_records`)."""
    spec_toml = tiny_experiment_toml(budgets="[500]")
    log_dir = str(tmp_path / "logs")
    sezgi.run_experiment(spec_toml, log_dir=log_dir, parallel=False)

    curtailed = sezgi.read_ioh_records(log_dir, [200])
    assert len(curtailed) == 2
    for r in curtailed:
        assert r["budget"] == 200
        assert r["evals_used"] == 200


# ---------------------------------------------------------------------
# ecdf
# ---------------------------------------------------------------------

def test_ecdf_per_algo_monotone_and_in_unit_interval(tmp_path):
    log_dir = str(tmp_path / "logs")
    sezgi.run_experiment(tiny_experiment_toml(), log_dir=log_dir, parallel=False)

    curves = sezgi.ecdf(log_dir)
    assert isinstance(curves, list)
    assert len(curves) == 1
    algo, curve = curves[0]
    assert algo == "de"
    assert set(curve.keys()) == {"evals", "proportion"}

    evals = curve["evals"]
    proportion = curve["proportion"]
    assert len(evals) == len(proportion)
    assert evals == sorted(evals), "evals must be ascending"
    assert proportion == sorted(proportion), "proportion must be nondecreasing"
    assert all(0.0 <= p <= 1.0 for p in proportion), "proportion must be in [0, 1]"


def test_ecdf_per_algo_false_returns_single_pooled_dict(tmp_path):
    log_dir = str(tmp_path / "logs")
    sezgi.run_experiment(tiny_experiment_toml(), log_dir=log_dir, parallel=False)

    curve = sezgi.ecdf(log_dir, per_algo=False)
    assert set(curve.keys()) == {"evals", "proportion"}
    assert curve["evals"] == sorted(curve["evals"])
    assert curve["proportion"] == sorted(curve["proportion"])


def test_ecdf_custom_targets(tmp_path):
    log_dir = str(tmp_path / "logs")
    sezgi.run_experiment(tiny_experiment_toml(), log_dir=log_dir, parallel=False)

    curve = sezgi.ecdf(log_dir, targets=[10.0, 1.0, 0.1], per_algo=False)
    assert set(curve.keys()) == {"evals", "proportion"}
    assert all(0.0 <= p <= 1.0 for p in curve["proportion"])


# ---------------------------------------------------------------------
# coco_export
# ---------------------------------------------------------------------

def test_coco_export_writes_info_file_with_correct_first_line(tmp_path):
    log_dir = str(tmp_path / "logs")
    sezgi.run_experiment(tiny_experiment_toml(), log_dir=log_dir, parallel=False)

    out_dir = str(tmp_path / "coco")
    written = sezgi.coco_export(log_dir, out_dir)
    assert len(written) > 0
    assert all(os.path.exists(p) for p in written)

    info_path = os.path.join(out_dir, "de", "bbobexp_f1.info")
    assert info_path in written

    with open(info_path) as f:
        first_line = f.readline().rstrip("\n")
    assert first_line == (
        "suite = 'bbob', funcId = 1, DIM = 5, Precision = 1.000e-08, algId = 'de'"
    )

    dat_path = os.path.join(out_dir, "de", "data_f1", "bbobexp_f1_DIM5.dat")
    tdat_path = os.path.join(out_dir, "de", "data_f1", "bbobexp_f1_DIM5.tdat")
    assert dat_path in written
    assert tdat_path in written


# ---------------------------------------------------------------------
# sezgi.stats.bayesian_plackett_luce
# ---------------------------------------------------------------------

def test_bayesian_plackett_luce_shape_and_determinism():
    rankings = [[0, 1, 2], [1, 0, 2], [0, 2, 1], [2, 1, 0]]

    r1 = sezgi.stats.bayesian_plackett_luce(rankings, samples=300, burn_in=100, seed=1)
    r2 = sezgi.stats.bayesian_plackett_luce(rankings, samples=300, burn_in=100, seed=1)

    assert set(r1.keys()) == {"mean_worths", "ci_low", "ci_high", "p_best", "samples"}
    assert len(r1["mean_worths"]) == 3
    assert len(r1["ci_low"]) == 3
    assert len(r1["ci_high"]) == 3
    assert len(r1["p_best"]) == 3
    assert r1["samples"] == 300

    # Same-seed determinism: exact list equality (bit-identical), not
    # merely close -- same convention as the Rust
    # bayes_pl_deterministic_given_seed test.
    assert r1 == r2, "same seed must give an exact list-equal result"

    assert abs(sum(r1["p_best"]) - 1.0) < 1e-9


def test_bayesian_plackett_luce_different_seed_diverges():
    rankings = [[0, 1, 2], [1, 0, 2], [0, 2, 1], [2, 1, 0]]
    r1 = sezgi.stats.bayesian_plackett_luce(rankings, samples=300, burn_in=100, seed=1)
    r2 = sezgi.stats.bayesian_plackett_luce(rankings, samples=300, burn_in=100, seed=2)
    assert r1["mean_worths"] != r2["mean_worths"]


def test_bayesian_plackett_luce_defaults():
    rankings = [[0, 1], [0, 1], [0, 1], [1, 0]]
    r = sezgi.stats.bayesian_plackett_luce(rankings)
    assert r["samples"] == 2000


def test_bayesian_plackett_luce_rejects_invalid_rankings():
    with pytest.raises(ValueError):
        sezgi.stats.bayesian_plackett_luce([[0, 0, 1]])


# ---------------------------------------------------------------------
# T1 deferred homework (MANDATORY): checkpoint + log_dir resume must not
# re-log (the IOH tree must not grow when a run is resumed from the
# journal rather than re-executed).
# ---------------------------------------------------------------------

def _tree_bytes(root):
    """Concatenates every file's bytes under `root`, in a stable
    (sorted-path) order, so it can be compared byte-for-byte across two
    snapshots of the same tree."""
    total = bytearray()
    for dirpath, _dirnames, filenames in sorted(os.walk(root)):
        for name in sorted(filenames):
            with open(os.path.join(dirpath, name), "rb") as f:
                total += f.read()
    return bytes(total)


def test_checkpoint_and_log_dir_resume_does_not_regrow_ioh_tree(tmp_path):
    spec_toml = tiny_experiment_toml(budgets="[500]")
    journal_path = str(tmp_path / "journal.jsonl")
    log_dir = str(tmp_path / "logs")

    r1 = sezgi.run_experiment(spec_toml, journal=journal_path, log_dir=log_dir, parallel=False)
    assert len(r1) == 2
    assert os.path.isdir(log_dir), "first run must have written an IOH tree"

    first_run_bytes = _tree_bytes(log_dir)
    assert len(first_run_bytes) > 0

    lines_after_first = sum(1 for _ in open(journal_path))

    # Resume: same journal, same spec -- every run is already done, so
    # nothing should be re-executed, and hence nothing re-logged.
    r2 = sezgi.run_experiment(spec_toml, journal=journal_path, log_dir=log_dir, parallel=False)
    assert len(r2) == 2

    lines_after_second = sum(1 for _ in open(journal_path))
    assert lines_after_second == lines_after_first, \
        "resume of a completed experiment must not append new journal lines"

    resumed_bytes = _tree_bytes(log_dir)
    assert resumed_bytes == first_run_bytes, \
        "resumed runs must never be re-logged -- the on-disk IOH tree must not grow"

    def strip_wall(records):
        return [{k: v for k, v in r.items() if k != "wall_secs"} for r in records]

    assert strip_wall(r1) == strip_wall(r2)
