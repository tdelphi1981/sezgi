"""M3-6 Task 9: Python bindings for CEC 2014 and CEC 2017 (sezgi.problems),
plus IOH-logging parity for both suites.

Binds `sezgi_problems::Cec2014` (crates/problems/src/cec2014/mod.rs) and
`sezgi_problems::Cec2017` (crates/problems/src/cec2017/mod.rs), mirroring
`sezgi.problems.cec2022(...)`'s trio (`cec2014`/`cec2014_evaluate`/
`cec2014_f_star`, same for 2017) 1:1 -- see test_cec_tsp.py's own CEC 2022
section for the pattern this file mirrors.

Every f64 here is asserted with EXACT equality (`==`), not a tolerance --
these are bit-exact goldens, matching this project's f64-pass-through
mandate. The `x = o` pins below are NOT copied from a Rust `#[test]`'s
printed output (`o` is a private field, not exposed through any public
accessor, and this task does not add one) -- they are instead read directly
from the vendored `shift_data_<fid>.txt` fixtures the SAME way
`crates/problems/src/cec2014/data.rs::shift_vector`/
`crates/problems/src/cec2017/data.rs::shift_vector` do: the first `dim`
whitespace-separated tokens of the fixture file's first line (fid 1-22/1-20
files are already single-line; composition fid files are multi-line, and
`composition_shift_blocks` takes only the FIRST line -- `o` mirrors
`comp_shift[0]`, struct doc). Parsing the same decimal text with Python's
`float()` and Rust's `f64::from_str` produces bit-identical values (IEEE-754
correctly-rounded decimal parsing on both sides) -- cross-checked here
against each crate's own `shift_vector_1_dim10_matches_vendored_file` unit
test, which asserts the identical `o[0]`/`o[1]` tokens.

Rust test provenance for each pin:
- cec2014 fid 1/17/23 dim=10: `fid_1_to_22_x_eq_o_is_exactly_f_star_dim10_and_dim30`
  and `fid_23_to_30_x_eq_o1_is_exactly_f_star_dim10_and_dim30`
  (crates/problems/src/cec2014/mod.rs).
- cec2017 fid 1/11/21 dim=10: `fid_1_3_to_8_and_10_x_eq_o_is_exactly_f_star_dim10_and_dim30`
  and `fid_11_to_20_x_eq_o_is_exactly_f_star_dim10_and_dim30`
  (crates/problems/src/cec2017/mod.rs).
- cec2017 fid 9 dim=10 (Levy, the VERIFIED non-exact exception): the value
  901.4426009870527 is the exact golden pinned by
  `fid_9_levy_x_eq_o_matches_measured_compiled_c_not_f_star`
  (crates/problems/src/cec2017/mod.rs), reproduced here through the Python
  binding.
"""
import sezgi
import pytest


# ---------------------------------------------------------------------
# CEC 2014: f_star -- Cec2014::f_star, F_i* = 100*fid.
# ---------------------------------------------------------------------

def test_cec2014_f_star_matches_report_table():
    for fid in range(1, 31):
        assert sezgi.problems.cec2014_f_star(fid) == 100.0 * fid


def test_cec2014_f_star_invalid_fid_raises_value_error():
    for fid in (0, 31, 100):
        with pytest.raises(ValueError):
            sezgi.problems.cec2014_f_star(fid)


# ---------------------------------------------------------------------
# CEC 2014: evaluate -- golden x=o pins, bit-exact (see module doc for
# provenance/derivation of these vectors).
# ---------------------------------------------------------------------

O1_D10 = [
    5.0355789822908633e+001, 6.4926709932099072e+001, -5.9682109393039028e+001,
    6.6140136982243092e+001, 2.1177479396065522e+001, -6.4393535200094476e+001,
    -3.5440284981272256e+001, 7.5010430727974153e+000, 7.3201093669487619e+001,
    7.4382165631884249e+001,
]

O17_D10 = [
    2.7563237993188579e+001, -1.0958210735697463e+001, 3.1104625549528691e+001,
    -3.8914469876776337e+001, -7.8438616018374475e+001, 5.1652913697421923e+000,
    -3.5297285559525662e+001, 7.1396824563188261e+001, 6.5030922641600341e+001,
    -1.7170467844142152e+001,
]

O23_D10 = [
    2.2195926903730161e+001, -5.8022979077717679e+001, -2.8356594783575240e+001,
    3.1034576162577068e+001, 3.3321483631979191e+001, 1.6682557694960778e+001,
    7.6359921075001978e+001, -8.2295732816317173e+000, 3.7770185136035245e+001,
    -6.5502615532858250e+000,
]


def test_cec2014_evaluate_x_equals_o_pins_f_star_fid1_unimodal_dim10():
    assert sezgi.problems.cec2014_evaluate(1, 10, O1_D10) == 100.0


def test_cec2014_evaluate_x_equals_o_pins_f_star_fid17_hybrid_dim10():
    assert sezgi.problems.cec2014_evaluate(17, 10, O17_D10) == 1700.0


def test_cec2014_evaluate_x_equals_o1_pins_f_star_fid23_composition_dim10():
    assert sezgi.problems.cec2014_evaluate(23, 10, O23_D10) == 2300.0


def test_cec2014_evaluate_invalid_fid_raises_value_error():
    for fid in (0, 31, 100):
        with pytest.raises(ValueError):
            sezgi.problems.cec2014_evaluate(fid, 10, [0.0] * 10)


def test_cec2014_evaluate_invalid_dim_raises_value_error():
    for dim in (2, 5, 20, 50, 100):
        with pytest.raises(ValueError):
            sezgi.problems.cec2014_evaluate(1, dim, [0.0] * dim)


def test_cec2014_evaluate_x_length_mismatch_raises_value_error():
    with pytest.raises(ValueError):
        sezgi.problems.cec2014_evaluate(1, 10, [0.0] * 5)


def test_cec2014_handle_shape():
    p = sezgi.problems.cec2014(1, 10)
    assert p.dim() == 10
    assert p.bounds() == (-100.0, 100.0)
    assert p.optimum() == 100.0


def test_cec2014_construction_smoke_all_fids_both_dims():
    for fid in range(1, 31):
        for dim in (10, 30):
            p = sezgi.problems.cec2014(fid, dim)
            assert p.dim() == dim
            assert p.optimum() == 100.0 * fid


# ---------------------------------------------------------------------
# CEC 2017: f_star -- Cec2017::f_star, F_i* = 100*fid (the fid-gapped C
# dispatch bias, NOT the report's contiguous renumbering -- module doc's
# own numbering-divergence note).
# ---------------------------------------------------------------------

CEC2017_FIDS = [1] + list(range(3, 31))


def test_cec2017_f_star_matches_dispatch_bias():
    for fid in CEC2017_FIDS:
        assert sezgi.problems.cec2017_f_star(fid) == 100.0 * fid


def test_cec2017_f_star_invalid_fid_raises_value_error():
    for fid in (0, 31, 100):
        with pytest.raises(ValueError):
            sezgi.problems.cec2017_f_star(fid)


def test_cec2017_f_star_fid2_withdrawn_raises_dedicated_value_error():
    # Cec2017Error::Withdrawn -- distinct from UnknownFid; the Rust error
    # text is surfaced VERBATIM (this task's own binding requirement).
    with pytest.raises(ValueError) as excinfo:
        sezgi.problems.cec2017_f_star(2)
    msg = str(excinfo.value)
    assert msg == (
        'fid 2 ("Sum of Different Powers") was officially withdrawn from the '
        "CEC 2017 suite; the reference C's own case 2 prints \"Error: This "
        'function (F2) has been deleted" and leaves its result unset -- sezgi '
        "returns this dedicated error instead, never a garbage f64 (module "
        "doc's F2 ruling section has the full quoted C and probe transcript)"
    )


# ---------------------------------------------------------------------
# CEC 2017: evaluate -- golden x=o pins, bit-exact (see module doc for
# provenance/derivation of these vectors).
# ---------------------------------------------------------------------

O1_2017_D10 = [
    -5.5276398498228005e+01, -7.0429559718086182e+01, -2.9610181874414053e+01,
    -5.8326763277094230e+01, 2.2089601877187192e+01, 5.9938749885158018e+01,
    3.0569319851030272e+01, 1.8558736265897153e+01, 7.6680420933608161e+01,
    -3.2165368847625970e+01,
]

O11_2017_D10 = [
    -2.4752590407036521e+01, -1.6786971062788261e+01, -5.8622652282296983e+01,
    9.0830914345956870e+00, 1.9517741249649674e+01, 3.4677182861919135e+01,
    -6.0587639725109398e+01, -5.6460859913593559e+01, 2.2327432233061373e+01,
    5.7153541062564386e+01,
]

O21_2017_D10 = [
    6.4346884556276208e+01, 1.2707352301435243e+01, 7.4495167014419650e+01,
    1.6796387941016064e+01, -7.2646850768777938e+01, -7.2924593266530238e+01,
    5.4639988328628235e+01, 2.3612044948903865e+01, -5.8462829438253692e+01,
    -7.7987176618532828e+01,
]

O9_2017_D10 = [
    -2.4856254283737513e+01, 9.3425322367542663e-01, 2.1993930312802064e+01,
    5.1986767264076860e+01, -1.8402592660115374e+01, -1.2374838559131533e+01,
    1.6088336676400303e+01, -3.3254797399628302e+00, -5.1337989555363535e+01,
    4.2409543005008970e+01,
]


def test_cec2017_evaluate_x_equals_o_pins_f_star_fid1_unimodal_dim10():
    assert sezgi.problems.cec2017_evaluate(1, 10, O1_2017_D10) == 100.0


def test_cec2017_evaluate_x_equals_o_pins_f_star_fid11_hybrid_dim10():
    assert sezgi.problems.cec2017_evaluate(11, 10, O11_2017_D10) == 1100.0


def test_cec2017_evaluate_x_equals_o1_pins_f_star_fid21_composition_dim10():
    assert sezgi.problems.cec2017_evaluate(21, 10, O21_2017_D10) == 2100.0


def test_cec2017_evaluate_fid9_levy_x_eq_o_matches_measured_compiled_c_not_f_star():
    out = sezgi.problems.cec2017_evaluate(9, 10, O9_2017_D10)
    assert out == 901.4426009870527
    assert out != sezgi.problems.cec2017_f_star(9)


def test_cec2017_evaluate_fid2_withdrawn_raises_value_error():
    with pytest.raises(ValueError) as excinfo:
        sezgi.problems.cec2017_evaluate(2, 10, [0.0] * 10)
    assert "officially withdrawn" in str(excinfo.value)


def test_cec2017_evaluate_invalid_fid_raises_value_error():
    for fid in (0, 31, 100):
        with pytest.raises(ValueError):
            sezgi.problems.cec2017_evaluate(fid, 10, [0.0] * 10)


def test_cec2017_evaluate_invalid_dim_raises_value_error():
    for dim in (2, 5, 20, 50, 100):
        with pytest.raises(ValueError):
            sezgi.problems.cec2017_evaluate(1, dim, [0.0] * dim)


def test_cec2017_evaluate_x_length_mismatch_raises_value_error():
    with pytest.raises(ValueError):
        sezgi.problems.cec2017_evaluate(1, 10, [0.0] * 5)


def test_cec2017_handle_shape():
    p = sezgi.problems.cec2017(1, 10)
    assert p.dim() == 10
    assert p.bounds() == (-100.0, 100.0)
    assert p.optimum() == 100.0


def test_cec2017_fid2_withdrawn_rejects_handle_construction():
    with pytest.raises(ValueError):
        sezgi.problems.cec2017(2, 10)


def test_cec2017_construction_smoke_all_fids_both_dims():
    for fid in CEC2017_FIDS:
        for dim in (10, 30):
            p = sezgi.problems.cec2017(fid, dim)
            assert p.dim() == dim
            assert p.optimum() == 100.0 * fid


# ---------------------------------------------------------------------
# for_problem session integration -- mirrors
# test_eval_session_generic.py's own test_for_problem_cec2022_counts_and_f_opt.
# ---------------------------------------------------------------------

def test_for_problem_cec2014_counts_and_f_opt():
    p = sezgi.problems.cec2014(1, 10)
    s = sezgi.EvalSession.for_problem(p, budget=50)
    fs = s.evaluate([[0.0] * 10, [1.0] * 10])
    assert len(fs) == 2 and s.evals_used() == 2
    assert s.f_opt() == sezgi.problems.cec2014_f_star(1)
    s.finish()


def test_for_problem_cec2017_counts_and_f_opt():
    p = sezgi.problems.cec2017(1, 10)
    s = sezgi.EvalSession.for_problem(p, budget=50)
    fs = s.evaluate([[0.0] * 10, [1.0] * 10])
    assert len(fs) == 2 and s.evals_used() == 2
    assert s.f_opt() == sezgi.problems.cec2017_f_star(1)
    s.finish()


# ---------------------------------------------------------------------
# IOH logging end-to-end (M3-5 Task 2's widening ridden by the new suites --
# both entry points, for_problem AND solve(), per the M3-5 T2 asymmetry
# ruling: both or neither). Budgets kept tiny (~1000 evals or less) for
# speed. Mirrors test_eval_session_generic.py's
# test_for_problem_cec2022_logging_roundtrip and test_solve.py's
# test_solve_cec2022_log_dir_writes_ioh /
# test_solve_and_for_problem_cec2022_logging_agree_on_identity.
# ---------------------------------------------------------------------

def test_for_problem_cec2014_logging_roundtrip(tmp_path):
    p = sezgi.problems.cec2014(1, 10)
    s = sezgi.EvalSession.for_problem(p, budget=20, log_dir=str(tmp_path),
                                      algo_name="probe", seed=7)
    s.evaluate([[0.0] * 10] * 20)
    s.finish()
    recs = sezgi.read_ioh_records(str(tmp_path), [20])
    assert len(recs) == 1
    assert recs[0]["suite"] == "sezgi-cec2014"
    assert recs[0]["f_opt"] == 100.0

    # Folder/label carry the new suite string, on disk: fname passed to
    # IohLogger is "cec2014-f{fid}" (crates/bench/src/ioh.rs's
    # `data_f{fid}_{fname}` naming, verified directly here).
    dat = tmp_path / "probe" / "data_f1_cec2014-f1" / "IOHprofiler_f1_DIM10.dat"
    assert dat.exists()
    assert dat.read_text().splitlines()[0] == '"evaluations" "raw_y"'

    # results_matrix's own problem_segment label (M3-5's shared helper,
    # crates/bench/src/experiment.rs): "cec2014-f{fid}d{dim}i{instance}".
    _algos, problems, _matrix = sezgi.results_matrix(recs, budget=20)
    assert problems == ["cec2014-f1d10i1"]


def test_for_problem_cec2017_logging_roundtrip(tmp_path):
    p = sezgi.problems.cec2017(1, 10)
    s = sezgi.EvalSession.for_problem(p, budget=20, log_dir=str(tmp_path),
                                      algo_name="probe", seed=7)
    s.evaluate([[0.0] * 10] * 20)
    s.finish()
    recs = sezgi.read_ioh_records(str(tmp_path), [20])
    assert len(recs) == 1
    assert recs[0]["suite"] == "sezgi-cec2017"
    assert recs[0]["f_opt"] == 100.0

    dat = tmp_path / "probe" / "data_f1_cec2017-f1" / "IOHprofiler_f1_DIM10.dat"
    assert dat.exists()
    assert dat.read_text().splitlines()[0] == '"evaluations" "raw_y"'

    _algos, problems, _matrix = sezgi.results_matrix(recs, budget=20)
    assert problems == ["cec2017-f1d10i1"]


def test_solve_cec2014_log_dir_writes_ioh(tmp_path):
    p = sezgi.problems.cec2014(1, 10)
    spec = sezgi.presets.random_search(pop_size=5, budget=20)
    sezgi.solve(spec, p, master_seed=7, log_dir=str(tmp_path), algo_name="probe")
    recs = sezgi.read_ioh_records(str(tmp_path), [20])
    assert len(recs) == 1
    assert recs[0]["suite"] == "sezgi-cec2014"
    assert recs[0]["f_opt"] == 100.0


def test_solve_cec2017_log_dir_writes_ioh(tmp_path):
    p = sezgi.problems.cec2017(1, 10)
    spec = sezgi.presets.random_search(pop_size=5, budget=20)
    sezgi.solve(spec, p, master_seed=7, log_dir=str(tmp_path), algo_name="probe")
    recs = sezgi.read_ioh_records(str(tmp_path), [20])
    assert len(recs) == 1
    assert recs[0]["suite"] == "sezgi-cec2017"
    assert recs[0]["f_opt"] == 100.0


def test_solve_and_for_problem_cec2014_logging_agree_on_identity(tmp_path):
    """The same (fid, dim, algo_name, seed, budget) CEC 2014 scenario,
    logged once via solve() and once via EvalSession.for_problem into two
    separate trees, must reconstruct with IDENTICAL identity keys -- the two
    entry points must agree (M3-5 T2 asymmetry ruling)."""
    solve_dir = tmp_path / "via_solve"
    fp_dir = tmp_path / "via_for_problem"

    spec = sezgi.presets.random_search(pop_size=5, budget=20)
    sezgi.solve(spec, sezgi.problems.cec2014(1, 10), master_seed=3,
               log_dir=str(solve_dir), algo_name="probe")

    s = sezgi.EvalSession.for_problem(sezgi.problems.cec2014(1, 10), budget=20,
                                      log_dir=str(fp_dir), algo_name="probe", seed=3)
    s.evaluate([[0.0] * 10] * 20)
    s.finish()

    r_solve = sezgi.read_ioh_records(str(solve_dir), [20])[0]
    r_fp = sezgi.read_ioh_records(str(fp_dir), [20])[0]
    identity_keys = ("algo", "suite", "fid", "dim", "instance", "seed", "budget")
    assert {k: r_solve[k] for k in identity_keys} == {k: r_fp[k] for k in identity_keys}


def test_solve_and_for_problem_cec2017_logging_agree_on_identity(tmp_path):
    solve_dir = tmp_path / "via_solve"
    fp_dir = tmp_path / "via_for_problem"

    spec = sezgi.presets.random_search(pop_size=5, budget=20)
    sezgi.solve(spec, sezgi.problems.cec2017(1, 10), master_seed=3,
               log_dir=str(solve_dir), algo_name="probe")

    s = sezgi.EvalSession.for_problem(sezgi.problems.cec2017(1, 10), budget=20,
                                      log_dir=str(fp_dir), algo_name="probe", seed=3)
    s.evaluate([[0.0] * 10] * 20)
    s.finish()

    r_solve = sezgi.read_ioh_records(str(solve_dir), [20])[0]
    r_fp = sezgi.read_ioh_records(str(fp_dir), [20])[0]
    identity_keys = ("algo", "suite", "fid", "dim", "instance", "seed", "budget")
    assert {k: r_solve[k] for k in identity_keys} == {k: r_fp[k] for k in identity_keys}


def test_no_collision_cec2014_vs_cec2017_same_fid(tmp_path):
    """Regression coverage for the M3-4-final-review-style collision
    scenario, extended to the two new suites: one CEC 2014 f1 d10 run and
    one CEC 2017 f1 d10 run, same algo/seed/budget, logged into the same
    tree; read back and build the matrix -- both labels present, distinct."""
    class RS(sezgi.AskTellAlgorithm):
        def setup(self, ctx): ctx.evaluate([ctx.random_point() for _ in range(10)])
        def step(self, ctx): ctx.evaluate([ctx.random_point() for _ in range(10)])
    RS().solve(sezgi.problems.cec2014(1, 10), budget=30, seed=1, log_dir=str(tmp_path))
    RS().solve(sezgi.problems.cec2017(1, 10), budget=30, seed=1, log_dir=str(tmp_path))
    recs = sezgi.read_ioh_records(str(tmp_path), [30])
    assert len(recs) == 2
    _algos, problems, _matrix = sezgi.results_matrix(recs, budget=30)
    assert sorted(problems) == ["cec2014-f1d10i1", "cec2017-f1d10i1"]


# ---------------------------------------------------------------------
# coco_export stays BBOB-only -- one assert per suite that its rejection
# names the offending new suite (permanent boundary, not widened here).
# ---------------------------------------------------------------------

def test_coco_export_rejects_cec2014_and_cec2017(tmp_path):
    RS_dir_2014 = tmp_path / "cec2014_tree"
    p = sezgi.problems.cec2014(1, 10)
    s1 = sezgi.EvalSession.for_problem(p, budget=10, log_dir=str(RS_dir_2014),
                                       algo_name="probe")
    s1.evaluate([[0.0] * 10])
    s1.finish()
    with pytest.raises(ValueError, match="sezgi-cec2014"):
        sezgi.coco_export(str(RS_dir_2014), str(tmp_path / "out2014"))

    RS_dir_2017 = tmp_path / "cec2017_tree"
    p2 = sezgi.problems.cec2017(1, 10)
    s2 = sezgi.EvalSession.for_problem(p2, budget=10, log_dir=str(RS_dir_2017),
                                       algo_name="probe")
    s2.evaluate([[0.0] * 10])
    s2.finish()
    with pytest.raises(ValueError, match="sezgi-cec2017"):
        sezgi.coco_export(str(RS_dir_2017), str(tmp_path / "out2017"))
