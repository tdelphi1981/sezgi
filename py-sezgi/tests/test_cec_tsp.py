"""M3-3 Task 9: Python bindings for CEC 2022 + TSP + ga-perm (sezgi.problems).

Binds `sezgi_problems::Cec2022` (crates/problems/src/cec2022/mod.rs) and
`sezgi_problems::Tsp` (crates/problems/src/tsp.rs), plus exposes the
`gen/ga-perm` preset (`sezgi_components::presets::ga_perm`,
crates/components/src/presets.rs) through the existing `sezgi.presets` /
`sezgi.solve` spec-JSON path -- no new solve mechanism, `solve()` already
dispatches on `PyProblem`'s inner variant.

Every f64 here is asserted with EXACT equality (`==`), not a tolerance --
these are bit-exact goldens copied from the Rust test suite (see each test's
own comment for provenance), matching this project's f64-pass-through
mandate (T10 asserts R/Python bit-equality against these same values).
"""
import json

import sezgi
import pytest


# ---------------------------------------------------------------------
# CEC 2022: f_star -- crates/problems/src/cec2022/mod.rs Cec2022::f_star
# (module doc section 1.2's table, quoted there).
# ---------------------------------------------------------------------

def test_cec2022_f_star_matches_report_table():
    expected = {
        1: 300.0, 2: 400.0, 3: 600.0, 4: 800.0, 5: 900.0,
        6: 1800.0, 7: 2000.0, 8: 2200.0,
        9: 2300.0, 10: 2400.0, 11: 2600.0, 12: 2700.0,
    }
    for fid, f_star in expected.items():
        assert sezgi.problems.cec2022_f_star(fid) == f_star


def test_cec2022_f_star_invalid_fid_raises_value_error():
    for fid in (0, 13, 100):
        with pytest.raises(ValueError):
            sezgi.problems.cec2022_f_star(fid)


# ---------------------------------------------------------------------
# CEC 2022: evaluate -- golden x=o pins, bit-exact.
#
# `o` below is the vendored shift vector's first `dim` entries, read
# directly from crates/problems/data/cec2022/shift_data_{1,5,6}.txt (the
# SAME data `Cec2022::new` embeds via `include_str!` -- see
# crates/problems/src/cec2022/data.rs). This is the exact point the Rust
# tests `x_equals_o_pins_f_star_exactly_dim10` and
# `x_equals_o_pins_f_star_exactly_hybrid_dim10_and_dim20`
# (crates/problems/src/cec2022/mod.rs) pin: F(o) == F* EXACTLY (`assert_eq!`,
# not a tolerance check), reproduced here through the Python binding.
# ---------------------------------------------------------------------

O1_D10 = [
    -55.938326705218444, 4.543065393596464, 35.300070175575115,
    8.279440891127777, -47.43694281551845, 7.302099581482864,
    6.387665982538678, -2.4978314724275066, -61.05355708908921,
    -47.133996322347784,
]

O5_D10 = [
    -24.856254283737513, 0.9342532236754266, 21.993930312802064,
    51.98676726407686, -18.402592660115374, -12.374838559131533,
    16.088336676400303, -3.32547973996283, -51.337989555363535,
    42.40954300500897,
]

O6_D10 = [
    18.154450402667024, 1.8479176246784164, 17.95151797027863,
    -48.837473526987, -11.4175680705428, -55.06656161279723,
    -63.70102628306533, -63.148739479085776, -60.8644249946179,
    -68.39088454301975,
]


def test_cec2022_evaluate_x_equals_o_pins_f_star_fid1_dim10():
    assert sezgi.problems.cec2022_evaluate(1, 10, O1_D10) == 300.0


def test_cec2022_evaluate_x_equals_o_pins_f_star_fid5_dim10():
    assert sezgi.problems.cec2022_evaluate(5, 10, O5_D10) == 900.0


def test_cec2022_evaluate_x_equals_o_pins_f_star_fid6_hybrid_dim10():
    assert sezgi.problems.cec2022_evaluate(6, 10, O6_D10) == 1800.0


def test_cec2022_evaluate_invalid_fid_raises_value_error():
    with pytest.raises(ValueError):
        sezgi.problems.cec2022_evaluate(13, 10, [0.0] * 10)


def test_cec2022_evaluate_invalid_dim_raises_value_error():
    with pytest.raises(ValueError):
        sezgi.problems.cec2022_evaluate(1, 5, [0.0] * 5)


def test_cec2022_evaluate_hybrid_dim2_rejected():
    # Cec2022Error::HybridDim2Unsupported -- fid 6-8 reject dim=2.
    with pytest.raises(ValueError):
        sezgi.problems.cec2022_evaluate(6, 2, [0.0, 0.0])


def test_cec2022_evaluate_x_length_mismatch_raises_value_error():
    with pytest.raises(ValueError):
        sezgi.problems.cec2022_evaluate(1, 10, [0.0] * 5)


# ---------------------------------------------------------------------
# TSP: load -- crates/problems/src/tsp.rs Tsp::vendored / Tsp::from_tsplib.
# ---------------------------------------------------------------------

def test_tsp_load_berlin52_by_vendored_name():
    info = sezgi.problems.tsp_load("berlin52")
    assert info["name"] == "berlin52"
    assert info["n_cities"] == 52
    assert info["known_optimum"] == 7542.0
    assert len(info["coords"]) == 52
    # data/tsplib/berlin52.tsp, first NODE_COORD_SECTION line: "1 565.0 575.0".
    assert info["coords"][0] == (565.0, 575.0)
    # ... last line: "52 1740.0 245.0".
    assert info["coords"][51] == (1740.0, 245.0)


def test_tsp_load_unknown_vendored_name_falls_back_to_raw_text_and_raises_cleanly():
    with pytest.raises(ValueError):
        sezgi.problems.tsp_load("not-a-vendored-instance-and-not-tsplib-text")


def test_tsp_load_raw_tsplib_text():
    text = (
        "NAME: tiny\nTYPE: TSP\nDIMENSION: 2\nEDGE_WEIGHT_TYPE: EUC_2D\n"
        "NODE_COORD_SECTION\n1 0.0 0.0\n2 3.0 4.0\nEOF\n"
    )
    info = sezgi.problems.tsp_load(text)
    assert info["n_cities"] == 2
    assert info["coords"] == [(0.0, 0.0), (3.0, 4.0)]
    assert info["known_optimum"] is None


# ---------------------------------------------------------------------
# TSP: tour_length -- 0-based tour; the published-optimal berlin52 tour
# (data/tsplib/berlin52.opt.tour, 1-based node list, converted to 0-based
# here) must evaluate to EXACTLY 7542.0, mirroring
# `berlin52_published_optimal_tour_evaluates_to_exactly_7542`
# (crates/problems/src/tsp.rs).
# ---------------------------------------------------------------------

BERLIN52_OPT_TOUR_0BASED = [
    0, 48, 31, 44, 18, 40, 7, 8, 9, 42, 32, 50, 10, 51, 13, 12, 46, 25, 26,
    27, 11, 24, 3, 5, 14, 4, 23, 47, 37, 36, 39, 38, 35, 34, 33, 43, 45, 15,
    28, 49, 19, 22, 29, 1, 6, 41, 20, 16, 2, 17, 30, 21,
]


def test_tsp_tour_length_berlin52_optimal_tour_is_exactly_7542():
    length = sezgi.problems.tsp_tour_length("berlin52", BERLIN52_OPT_TOUR_0BASED)
    assert length == 7542.0


def test_tsp_tour_length_wrong_number_of_cities_raises_value_error():
    with pytest.raises(ValueError):
        sezgi.problems.tsp_tour_length("berlin52", list(range(51)))


def test_tsp_tour_length_out_of_range_entry_raises_value_error():
    bad = list(range(52))
    bad[0] = 52  # out of range: valid indices are 0..51
    with pytest.raises(ValueError):
        sezgi.problems.tsp_tour_length("berlin52", bad)


def test_tsp_tour_length_duplicate_entry_raises_value_error():
    bad = list(range(52))
    bad[1] = bad[0]  # duplicate: not a permutation
    with pytest.raises(ValueError):
        sezgi.problems.tsp_tour_length("berlin52", bad)


def test_tsp_tour_length_negative_entry_raises_value_error():
    bad = list(range(52))
    bad[0] = -1
    with pytest.raises(ValueError):
        sezgi.problems.tsp_tour_length("berlin52", bad)


# ---------------------------------------------------------------------
# ga-perm end-to-end via the existing preset/solve path (sezgi.presets.ga_perm
# + sezgi.solve + sezgi.problems.tsp), on the vendored berlin52 instance.
# ---------------------------------------------------------------------

def test_ga_perm_preset_shape():
    spec = sezgi.presets.ga_perm(pop_size=20, budget=2000)
    assert spec["name"] == "ga-perm"
    assert spec["stages"][0]["generator"]["kind"] == "gen/ga-perm"
    assert spec["stages"][0]["replacer"]["kind"] == "replace/mu-plus-lambda"


def test_ga_perm_solves_berlin52_returns_valid_tour():
    problem = sezgi.problems.tsp("berlin52")
    spec = sezgi.presets.ga_perm(pop_size=32, budget=1000)
    result = sezgi.solve(spec, problem, master_seed=42, run_id=0)

    tour = result["best_x"]
    assert len(tour) == 52
    assert sorted(tour) == list(range(52)), "best_x must be a valid permutation of the 52 cities"
    assert result["best_f"] >= 7542.0, "cannot beat the published berlin52 optimum"

    # Cross-check: solve()'s own reported best_f matches tsp_tour_length()
    # recomputing the length of the SAME best_x tour independently.
    assert sezgi.problems.tsp_tour_length("berlin52", tour) == result["best_f"]


def test_ga_perm_deterministic_same_seed_run_twice():
    problem = sezgi.problems.tsp("berlin52")
    spec = sezgi.presets.ga_perm(pop_size=32, budget=1000)
    r1 = sezgi.solve(spec, problem, master_seed=42, run_id=0)
    # A fresh problem handle each call -- solve() consumes engine state, not
    # the problem, but rebuild it anyway to keep both calls fully independent.
    problem2 = sezgi.problems.tsp("berlin52")
    r2 = sezgi.solve(spec, problem2, master_seed=42, run_id=0)

    assert r1["best_f"] == r2["best_f"]
    assert r1["best_x"] == r2["best_x"]
    assert r1["evals_used"] == r2["evals_used"]
    assert r1["iterations"] == r2["iterations"]

    # Same golden the Rust integration test pins
    # (ga_perm_deterministic_golden_berlin52_seed42,
    # crates/components/tests/integration.rs): best_f=16175.0,
    # evals_used=992, iterations=30.
    assert r1["best_f"] == 16175.0
    assert r1["evals_used"] == 992
    assert r1["iterations"] == 30


@pytest.mark.parametrize("name", ["ga_real", "ga_perm", "ga_bin", "ga_int", "ga_cat"])
def test_ga_presets_single_stage_single_generator_no_braces(name):
    # Pins the invariant r-sezgi's textual `.sz_ga_wrap_compound` rewrite
    # depends on: one stage, one flat generator, no braces inside strings.
    spec = getattr(sezgi.presets, name)(pop_size=20, budget=2000)
    assert len(spec["stages"]) == 1
    assert len(json.dumps(spec).split('"generator"')) == 2

    def strings(o):
        if isinstance(o, str):
            yield o
        elif isinstance(o, dict):
            for k, v in o.items():
                yield k
                yield from strings(v)
        elif isinstance(o, list):
            for v in o:
                yield from strings(v)

    for s in strings(spec):
        assert "{" not in s and "}" not in s, s
