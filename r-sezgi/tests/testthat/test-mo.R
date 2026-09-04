# R bindings for the multi-objective module (sz_nsga2 /
# sz_mo_hypervolume_2d / sz_mo_igd / sz_mo_pareto_front), extended with
# zdt5 (binary), dtlz8/dtlz9 (constrained), wfg1-9, general-M
# sz_mo_hypervolume, sz_mo_evaluate / sz_mo_evaluate_constraints, and
# sezgi-moa v1 run logging (sz_nsga2(..., log_dir=, label=) / sz_mo_read_moa).
#
# Mirrors crates/components::nsga2 / crates/problems::{zdt,dtlz,wfg} /
# crates/stats::moo_indicators / crates/bench::mo_archive 1:1 by dict key
# name -- see py-sezgi's `sezgi.mo` module (`py-sezgi/src/lib.rs`) for the
# full field-by-field provenance, and `r-sezgi/src/rust/src/mo.rs`'s module
# doc for this binding's own container-idiom decisions (list-of-vectors for
# `individuals`/`objectives`/`sz_mo_pareto_front`'s return, matrix for
# `sz_mo_hypervolume_2d`/`sz_mo_hypervolume`/`sz_mo_igd`'s `front`/
# `reference_front` inputs, 1-based `front0`, present-only-when-constrained
# `violations`, logical vector for a binary `sz_mo_read_moa` genotype
# record).

f64_bits_hex <- function(x) {
  raw <- writeBin(x, raw(), size = 8L, endian = "little")
  paste0(rev(sprintf("%02x", as.integer(raw))), collapse = "")
}

# ---- sz_nsga2: dict shape ----------------------------------------------

test_that("sz_nsga2 returns the mirrored dict shape (zdt1)", {
  r <- sz_nsga2("zdt1", dim = 5, pop_size = 8, budget = 40, seed = 1)

  expect_named(r, c("individuals", "objectives", "front0", "evals_used"))
  expect_type(r$individuals, "list")
  expect_type(r$objectives, "list")
  expect_length(r$individuals, 8)
  expect_length(r$objectives, 8)
  for (row in r$individuals) {
    expect_true(is.double(row))
    expect_length(row, 5)
  }
  for (row in r$objectives) {
    expect_true(is.double(row))
    expect_length(row, 2) # zdt is always 2-objective
  }
  expect_true(is.double(r$front0))
  expect_true(all(r$front0 >= 1)) # 1-based, R convention
  expect_true(is.double(r$evals_used))
  expect_identical(r$evals_used, 40)
})

test_that("sz_nsga2 flattens a two-Float-block genotype correctly (zdt4)", {
  # ZDT4's space is two Block::Float blocks: x1 (1 var) then x2..xdim
  # (dim - 1 vars) -- individuals must still come back as one flat
  # dim-length row (genotype_to_flat_vec concatenates every Float block).
  r <- sz_nsga2("zdt4", dim = 4, pop_size = 4, budget = 20, seed = 1)
  for (row in r$individuals) {
    expect_length(row, 4)
  }
})

test_that("sz_nsga2 requires m for dtlz problems", {
  expect_error(
    sz_nsga2("dtlz1", dim = 5, pop_size = 8, budget = 40, seed = 1),
    "m \\(number of objectives\\) is required for dtlz"
  )
})

test_that("sz_nsga2 rejects m for zdt problems", {
  expect_error(
    sz_nsga2("zdt1", dim = 5, m = 2, pop_size = 8, budget = 40, seed = 1),
    "m is DTLZ-only"
  )
})

test_that("sz_nsga2 accepts dtlz with m", {
  r <- sz_nsga2("dtlz2", dim = 5, m = 3, pop_size = 8, budget = 40, seed = 1)
  for (row in r$objectives) {
    expect_length(row, 3)
  }
})

# ---- sz_nsga2: determinism ----------------------------------------------

test_that("sz_nsga2 is bit-identical run-twice for the same seed", {
  r1 <- sz_nsga2("zdt1", dim = 5, pop_size = 8, budget = 60, seed = 20260830)
  r2 <- sz_nsga2("zdt1", dim = 5, pop_size = 8, budget = 60, seed = 20260830)
  expect_identical(r1, r2)
})

test_that("sz_nsga2 differs for a different seed", {
  r1 <- sz_nsga2("zdt1", dim = 5, pop_size = 8, budget = 60, seed = 1)
  r2 <- sz_nsga2("zdt1", dim = 5, pop_size = 8, budget = 60, seed = 2)
  expect_false(identical(r1$objectives, r2$objectives))
})

# ---- sz_nsga2: errors -----------------------------------------------------

test_that("sz_nsga2 rejects an unrecognized problem string", {
  expect_error(
    sz_nsga2("nope", dim = 5, pop_size = 8, budget = 40, seed = 1),
    "unknown problem"
  )
})

test_that("sz_nsga2 rejects zdt5 (binary-coded, out of scope)", {
  expect_error(sz_nsga2("zdt5", dim = 5, pop_size = 8, budget = 40, seed = 1))
})

test_that("sz_nsga2 rejects an odd pop_size", {
  expect_error(
    sz_nsga2("zdt1", dim = 5, pop_size = 7, budget = 40, seed = 1),
    "multiple of 4"
  )
})

test_that("sz_nsga2 rejects an even pop_size that is not a multiple of 4", {
  # Post-brief correction (T6): pop_size validation is >= 4 AND % 4 == 0 --
  # NOT merely "even, >= 4". pop_size = 6 is even but 6 %% 4 != 0.
  expect_error(
    sz_nsga2("zdt1", dim = 5, pop_size = 6, budget = 40, seed = 1),
    "multiple of 4"
  )
})

test_that("sz_nsga2 rejects a fractional pop_size (f64_to_usize strictness)", {
  expect_error(
    sz_nsga2("zdt1", dim = 5, pop_size = 8.5, budget = 40, seed = 1),
    "expected a whole number"
  )
})

test_that("sz_nsga2 rejects a fractional dim/budget/seed", {
  expect_error(sz_nsga2("zdt1", dim = 5.5, pop_size = 8, budget = 40, seed = 1), "expected a whole number")
  expect_error(sz_nsga2("zdt1", dim = 5, pop_size = 8, budget = 40.5, seed = 1), "expected a whole number")
  expect_error(sz_nsga2("zdt1", dim = 5, pop_size = 8, budget = 40, seed = 1.5), "expected a whole number")
})

# ---- sz_mo_hypervolume_2d --------------------------------------------------

test_that("sz_mo_hypervolume_2d matches the hand fixture exactly", {
  front <- rbind(c(0.25, 0.75), c(0.5, 0.5), c(0.75, 0.25))
  v <- sz_mo_hypervolume_2d(front, c(1.0, 1.0))
  expect_identical(v, 0.375)
})

test_that("sz_mo_hypervolume_2d does not double-count the same-f1 tie case", {
  front <- rbind(c(1, 3), c(2, 2), c(2, 1), c(3, 0.5))
  v <- sz_mo_hypervolume_2d(front, c(4, 4))
  expect_identical(v, 7.5)
})

test_that("sz_mo_hypervolume_2d rejects a bad ref_point length", {
  front <- rbind(c(0.25, 0.75), c(0.5, 0.5))
  expect_error(sz_mo_hypervolume_2d(front, c(1.0, 1.0, 1.0)), "ref_point")
})

test_that("sz_mo_hypervolume_2d rejects an empty front", {
  front <- matrix(numeric(0), nrow = 0, ncol = 2)
  expect_error(sz_mo_hypervolume_2d(front, c(1.0, 1.0)))
})

test_that("sz_mo_hypervolume_2d rejects a non-matrix front", {
  expect_error(sz_mo_hypervolume_2d(c(0.25, 0.75), c(1.0, 1.0)), "matrix")
})

# ---- sz_mo_igd --------------------------------------------------------------

test_that("sz_mo_igd matches the hand fixture exactly", {
  front <- rbind(c(1, 0), c(3, 0))
  reference_front <- rbind(c(0, 0), c(4, 0))
  v <- sz_mo_igd(front, reference_front)
  expect_identical(v, 1.0)
})

test_that("sz_mo_igd rejects an empty front", {
  front <- matrix(numeric(0), nrow = 0, ncol = 2)
  reference_front <- rbind(c(0, 0), c(4, 0))
  expect_error(sz_mo_igd(front, reference_front))
})

# ---- sz_mo_pareto_front -----------------------------------------------------

test_that("sz_mo_pareto_front shape and spot values (zdt1)", {
  front <- sz_mo_pareto_front("zdt1", dim = 30, n = 100)
  expect_type(front, "list")
  expect_length(front, 100)
  expect_identical(front[[1]], c(0.0, 1.0))
  expect_identical(front[[100]], c(1.0, 0.0))
  expect_identical(front[[51]], c(0.5050505050505051, 0.28933094548129856))
})

test_that("sz_mo_pareto_front is NULL for dtlz5 with m > 3", {
  front <- sz_mo_pareto_front("dtlz5", dim = 4, m = 4, n = 10)
  expect_null(front)
})

test_that("sz_mo_pareto_front requires m for dtlz problems", {
  expect_error(
    sz_mo_pareto_front("dtlz1", dim = 5, n = 10),
    "m \\(number of objectives\\) is required for dtlz"
  )
})

test_that("sz_mo_pareto_front rejects an unrecognized problem string", {
  expect_error(sz_mo_pareto_front("nope", dim = 5, n = 10), "unknown problem")
})

# ---- Cross-language bit-equality anchor ------------------------------------
#
# Anchor invocation ("Deterministic zdt1 invocation + exact output values
# for the cross-language fixture", re-confirmed live via
# `./py-sezgi/.venv/bin/python -c "import sezgi; sezgi.mo.nsga2(problem='zdt1', dim=5, pop_size=8, budget=200, seed=20260830)"`):
#
#   sezgi.mo.nsga2(problem="zdt1", dim=5, pop_size=8, budget=200, seed=20260830)
#
# giving evals_used=200, front0=[0..7] (all non-dominated), and (Python,
# 0-indexed) objectives[0] = [0.9622630618989141, 0.18789353328668065],
# objectives[7] (last) = [0.038343821202107276, 1.3522223452373752].
#
# Compared here via the raw IEEE-754 hex-byte technique from
# test-cross-language.R (the R strtod 1-ULP precedent) -- hex constants,
# not decimal literals, so no R-side parse-precision loss can mask a
# mismatch. Hex values themselves were computed directly from the same live
# Python run via `struct.pack('>d', x).hex()` (big-endian byte order,
# matching f64_bits_hex()'s little-endian-raw-then-reversed output).
test_that("R sz_nsga2 is bit-identical to the Python/Rust zdt1 anchor invocation", {
  r <- sz_nsga2("zdt1", dim = 5, pop_size = 8, budget = 200, seed = 20260830)

  expect_identical(r$evals_used, 200)
  expect_identical(r$front0, as.double(1:8))

  expect_identical(f64_bits_hex(r$objectives[[1]][1]), "3feecadbe7a0262c") # obj[0][0] = 0.9622630618989141
  expect_identical(f64_bits_hex(r$objectives[[1]][2]), "3fc80ce5324c4fa7") # obj[0][1] = 0.18789353328668065
  expect_identical(f64_bits_hex(r$objectives[[8]][2]), "3ff5a2b3e5db706d") # obj[7][1] = 1.3522223452373752
  expect_identical(f64_bits_hex(as.double(r$evals_used)), "4069000000000000") # evals_used = 200
})

# =============================================================================
# M3-7 Task 11: sz_nsga2 new problem families (zdt5, dtlz8/9, wfg1-9),
# sz_mo_hypervolume, sz_mo_evaluate / sz_mo_evaluate_constraints,
# sz_mo_read_moa, cross-language anchors.
# =============================================================================

# ---- sz_nsga2: new-problem smoke -------------------------------------------

test_that("sz_nsga2 runs zdt5 (binary genotype path) end-to-end", {
  r <- sz_nsga2("zdt5", dim = NULL, pop_size = 8, budget = 40, seed = 0)
  expect_named(r, c("individuals", "objectives", "front0", "evals_used"))
  expect_length(r$individuals, 8)
  for (row in r$individuals) {
    expect_length(row, 80) # fixed 80-bit layout
    expect_true(all(row %in% c(0.0, 1.0))) # 0.0/1.0 flattening
  }
  for (row in r$objectives) {
    expect_length(row, 2)
  }
})

test_that("sz_nsga2 zdt5 is bit-identical run-twice for the same seed", {
  kwargs <- list(problem = "zdt5", dim = NULL, pop_size = 8, budget = 40, seed = 20260901)
  r1 <- do.call(sz_nsga2, kwargs)
  r2 <- do.call(sz_nsga2, kwargs)
  expect_identical(r1, r2)
})

test_that("sz_nsga2 runs dtlz8 and surfaces violations", {
  r <- sz_nsga2("dtlz8", dim = 6, m = 3, pop_size = 8, budget = 40, seed = 0)
  expect_true("violations" %in% names(r))
  expect_length(r$violations, length(r$individuals))
  expect_true(all(r$violations <= 0.0))
})

test_that("sz_nsga2 runs dtlz9 and surfaces violations", {
  r <- sz_nsga2("dtlz9", dim = 6, m = 3, pop_size = 8, budget = 40, seed = 0)
  expect_true("violations" %in% names(r))
  expect_length(r$violations, length(r$individuals))
})

test_that("sz_nsga2 has no violations key for an unconstrained problem", {
  r <- sz_nsga2("dtlz1", dim = 6, m = 3, pop_size = 8, budget = 40, seed = 0)
  expect_false("violations" %in% names(r))
  r2 <- sz_nsga2("zdt1", dim = 5, pop_size = 8, budget = 40, seed = 0)
  expect_false("violations" %in% names(r2))
})

test_that("sz_nsga2 runs wfg1..wfg9 end-to-end with default k/l", {
  for (which in 1:9) {
    problem <- paste0("wfg", which)
    r <- sz_nsga2(problem, dim = NULL, m = 2, pop_size = 8, budget = 24, seed = 0)
    expect_named(r, c("individuals", "objectives", "front0", "evals_used"))
    expect_length(r$individuals, 8)
    for (row in r$individuals) {
      expect_length(row, 24) # k=4 (m=2 default) + l=20
    }
    for (row in r$objectives) {
      expect_length(row, 2)
    }
  }
})

test_that("sz_nsga2 wfg accepts explicit k/l", {
  r <- sz_nsga2("wfg4", dim = NULL, m = 3, pop_size = 8, budget = 24, seed = 0, k = 6, l = 4)
  for (row in r$individuals) {
    expect_length(row, 10) # k=6 + l=4
  }
})

test_that("sz_nsga2 accepts p_c_bin/p_m_bin knobs for zdt5", {
  r <- sz_nsga2("zdt5", dim = NULL, pop_size = 8, budget = 40, seed = 0,
                p_c_bin = 0.7, p_m_bin = 0.02)
  expect_length(r$individuals, 8)
})

# ---- sz_nsga2: p_c_cat/p_m_cat (Categorical-genotype knobs) -------------
#
# p_c_cat/p_m_cat (post-M3-8 deferral cleanup) are consulted ONLY when
# `problem` builds a Mixed search space containing a Block::Categorical
# block (Nsga2Config::p_c_cat's own doc). mo_problem_from_str's own catalog
# (zdt1-6, dtlz1-9, wfg1-9) never builds one -- see mo.rs's own module doc
# and py-sezgi's genotype_to_flat_vec doc, "no problem registered here ever
# constructs one". So unlike p_c_bin/p_m_bin (exercised via zdt5's real
# all-Binary space), there is currently no reachable problem through this
# binding whose *outcome* these two knobs can move -- the tests below pin
# down the two properties that ARE observable today: both knobs are
# threaded through and unconditionally range-validated (mirroring
# p_c_bin/p_m_bin's own "validated even when unused" design), and they are
# correctly INERT on every currently reachable problem (an extreme,
# non-default value changes nothing) -- the precise mirror of
# p_c_bin/p_m_bin's own already-shipped behavior on every problem other
# than zdt5.

test_that("sz_nsga2 accepts p_c_cat/p_m_cat knobs and is deterministic on zdt1", {
  r1 <- sz_nsga2("zdt1", dim = 5, pop_size = 8, budget = 200, seed = 20260902,
                 p_c_cat = 0.3, p_m_cat = 0.05)
  r2 <- sz_nsga2("zdt1", dim = 5, pop_size = 8, budget = 200, seed = 20260902,
                 p_c_cat = 0.3, p_m_cat = 0.05)
  expect_length(r1$individuals, 8)
  expect_identical(r1$objectives, r2$objectives)
  expect_identical(r1$individuals, r2$individuals)
  expect_identical(r1$front0, r2$front0)
  expect_identical(r1$evals_used, r2$evals_used)
})

test_that("sz_nsga2 omitting p_c_cat/p_m_cat matches explicit defaults", {
  r_omitted <- sz_nsga2("zdt1", dim = 5, pop_size = 8, budget = 200, seed = 20260903)
  r_explicit <- sz_nsga2("zdt1", dim = 5, pop_size = 8, budget = 200, seed = 20260903,
                          p_c_cat = 0.9, p_m_cat = NULL)
  expect_identical(r_omitted, r_explicit)
})

test_that("sz_nsga2 p_c_cat/p_m_cat are inert on every reachable problem", {
  r_default <- sz_nsga2("zdt1", dim = 5, pop_size = 8, budget = 200, seed = 20260904)
  r_nondefault <- sz_nsga2("zdt1", dim = 5, pop_size = 8, budget = 200, seed = 20260904,
                            p_c_cat = 0.01, p_m_cat = 0.99)
  expect_identical(r_default, r_nondefault)
})

test_that("sz_nsga2 rejects out-of-range p_c_cat", {
  expect_error(
    sz_nsga2("zdt1", dim = 5, pop_size = 8, budget = 40, seed = 0, p_c_cat = 1.5),
    "p_c_cat"
  )
})

test_that("sz_nsga2 rejects out-of-range p_m_cat", {
  expect_error(
    sz_nsga2("zdt1", dim = 5, pop_size = 8, budget = 40, seed = 0, p_m_cat = -0.1),
    "p_m_cat"
  )
})

# ---- sz_nsga2: new-problem errors -------------------------------------------

test_that("sz_nsga2 rejects dim given for zdt5", {
  expect_error(
    sz_nsga2("zdt5", dim = 5, pop_size = 8, budget = 40, seed = 0),
    "dim is not accepted for zdt5"
  )
})

test_that("sz_nsga2 rejects m given for zdt5", {
  expect_error(
    sz_nsga2("zdt5", dim = NULL, m = 2, pop_size = 8, budget = 40, seed = 0),
    "m is DTLZ-only"
  )
})

test_that("sz_nsga2 rejects dim given for a wfg problem", {
  expect_error(
    sz_nsga2("wfg1", dim = 8, m = 2, pop_size = 8, budget = 24, seed = 0),
    "dim is not accepted for wfg problems"
  )
})

test_that("sz_nsga2 requires m for a wfg problem", {
  expect_error(
    sz_nsga2("wfg1", dim = NULL, pop_size = 8, budget = 24, seed = 0),
    "m \\(number of objectives\\) is required for wfg problems"
  )
})

test_that("sz_nsga2 rejects a bad wfg k (does not divide m-1)", {
  expect_error(
    sz_nsga2("wfg1", dim = NULL, m = 3, k = 5, l = 4, pop_size = 8, budget = 24, seed = 0),
    "k %"
  )
})

test_that("sz_nsga2 rejects an odd l for wfg2/wfg3", {
  expect_error(
    sz_nsga2("wfg2", dim = NULL, m = 2, k = 4, l = 5, pop_size = 8, budget = 24, seed = 0),
    "l must be even"
  )
})

# ---- sz_mo_hypervolume: general-M exact hypervolume (WFG algorithm) -------

test_that("sz_mo_hypervolume matches the Rust hand fixture exactly (M=4)", {
  # Copied from crates/stats/src/moo_indicators.rs's own
  # `hypervolume_four_point_constant_dimension_factors_m4` test.
  front <- rbind(c(1.0, 8.0, 9.0, 5.0), c(9.0, 1.0, 8.0, 5.0), c(8.0, 9.0, 1.0, 5.0))
  ref_point <- c(10.0, 10.0, 10.0, 8.0)
  expect_identical(sz_mo_hypervolume(front, ref_point), 147.0)
})

test_that("sz_mo_hypervolume delegates to hypervolume_2d at M=2", {
  front <- rbind(c(0.25, 0.75), c(0.5, 0.5), c(0.75, 0.25))
  ref_point <- c(1.0, 1.0)
  expect_identical(sz_mo_hypervolume(front, ref_point), sz_mo_hypervolume_2d(front, ref_point))
})

test_that("sz_mo_hypervolume treats an empty front as 0.0 (unlike hypervolume_2d)", {
  front <- matrix(numeric(0), nrow = 0, ncol = 3)
  expect_identical(sz_mo_hypervolume(front, c(1.0, 1.0, 1.0)), 0.0)
})

test_that("sz_mo_hypervolume rejects a front/ref_point dimension mismatch", {
  front <- rbind(c(0.25, 0.75, 0.5))
  expect_error(sz_mo_hypervolume(front, c(1.0, 1.0)))
})

test_that("sz_mo_hypervolume without ref_point is a missing-argument error", {
  front <- rbind(c(0.25, 0.75))
  expect_error(sz_mo_hypervolume(front))
})

# ---- sz_mo_pareto_front: wfg cases ------------------------------------------

test_that("sz_mo_pareto_front is NULL for wfg1/wfg2 (no closed dominance-filtered form)", {
  expect_null(sz_mo_pareto_front("wfg1", dim = NULL, n = 5, m = 2))
  expect_null(sz_mo_pareto_front("wfg2", dim = NULL, n = 5, m = 2))
})

test_that("sz_mo_pareto_front is non-NULL for wfg4 (has a closed form)", {
  front <- sz_mo_pareto_front("wfg4", dim = NULL, n = 5, m = 2)
  expect_type(front, "list")
  expect_length(front, 5)
  for (row in front) {
    expect_length(row, 2)
  }
})

# ---- sz_mo_evaluate / sz_mo_evaluate_constraints: fixture pins -------------
#
# zdt5 fixtures (T4/T6 hand fixtures, cross-checked against pymoo 0.6.2 --
# crates/problems/src/zdt.rs's own test module; identical values py-sezgi's
# own test_evaluate_zdt5_*_fixture tests pin).

test_that("sz_mo_evaluate zdt5 all-zeros fixture", {
  f <- sz_mo_evaluate("zdt5", dim = NULL, x = rep(0.0, 80))
  expect_identical(f, c(1.0, 20.0))
})

test_that("sz_mo_evaluate zdt5 all-ones fixture", {
  f <- sz_mo_evaluate("zdt5", dim = NULL, x = rep(1.0, 80))
  expect_identical(f[1], 31.0)
  expect_equal(f[2], 10.0 / 31.0, tolerance = 1e-12)
})

test_that("sz_mo_evaluate zdt5 mixed fixture", {
  # x1: 7 leading ones (of 30) -> u=7, f1=8. Each of the 10 groups is
  # [1,0,1,0,1] -> u=3<5, v=5, g=50, f2=50/8=6.25.
  x1 <- c(rep(1.0, 7), rep(0.0, 23))
  group <- c(1.0, 0.0, 1.0, 0.0, 1.0)
  f <- sz_mo_evaluate("zdt5", dim = NULL, x = c(x1, rep(group, 10)))
  expect_identical(f, c(8.0, 6.25))
})

test_that("sz_mo_evaluate zdt5 rejects m", {
  expect_error(sz_mo_evaluate("zdt5", dim = NULL, x = rep(0.0, 80), m = 2))
})

# dtlz8/dtlz9 fixtures (T2 hand fixtures -- crates/problems/src/dtlz.rs's
# own test module; identical values py-sezgi's own test_evaluate_dtlz8_*/
# test_evaluate_dtlz9_* tests pin).

test_that("sz_mo_evaluate/sz_mo_evaluate_constraints dtlz8 m=2 dim=4 fixtures", {
  f <- sz_mo_evaluate("dtlz8", dim = 4, x = c(0.0, 0.0, 0.0, 0.0), m = 2)
  expect_identical(f, c(0.0, 0.0))
  g <- sz_mo_evaluate_constraints("dtlz8", dim = 4, x = c(0.0, 0.0, 0.0, 0.0), m = 2)
  expect_equal(g[1], -1.0, tolerance = 1e-12)
  expect_identical(g[2], Inf)

  f <- sz_mo_evaluate("dtlz8", dim = 4, x = c(1.0, 1.0, 1.0, 1.0), m = 2)
  expect_identical(f, c(1.0, 1.0))
  g <- sz_mo_evaluate_constraints("dtlz8", dim = 4, x = c(1.0, 1.0, 1.0, 1.0), m = 2)
  expect_equal(g[1], 4.0, tolerance = 1e-12)
  expect_identical(g[2], Inf)

  f <- sz_mo_evaluate("dtlz8", dim = 4, x = c(1.0, 0.0, 0.0, 0.0), m = 2)
  expect_identical(f, c(0.5, 0.0))
  g <- sz_mo_evaluate_constraints("dtlz8", dim = 4, x = c(1.0, 0.0, 0.0, 0.0), m = 2)
  expect_equal(g[1], 1.0, tolerance = 1e-12)
})

test_that("sz_mo_evaluate/sz_mo_evaluate_constraints dtlz8 m=3 dim=6 mixed fixture", {
  xs <- c(1.0, 0.0, 0.5, 0.5, 0.2, 0.2)
  f <- sz_mo_evaluate("dtlz8", dim = 6, x = xs, m = 3)
  expect_equal(f[1], 0.5, tolerance = 1e-12)
  expect_equal(f[2], 0.5, tolerance = 1e-12)
  expect_equal(f[3], 0.2, tolerance = 1e-12)
  g <- sz_mo_evaluate_constraints("dtlz8", dim = 6, x = xs, m = 3)
  expect_equal(g[1], 1.2, tolerance = 1e-9)
  expect_equal(g[2], 1.2, tolerance = 1e-9)
  expect_equal(g[3], 0.4, tolerance = 1e-9)
})

test_that("sz_mo_evaluate/sz_mo_evaluate_constraints dtlz9 m=2 dim=4 fixtures", {
  f <- sz_mo_evaluate("dtlz9", dim = 4, x = c(1.0, 1.0, 1.0, 1.0), m = 2)
  expect_identical(f, c(2.0, 2.0))
  g <- sz_mo_evaluate_constraints("dtlz9", dim = 4, x = c(1.0, 1.0, 1.0, 1.0), m = 2)
  expect_length(g, 1)
  expect_equal(g[1], 7.0, tolerance = 1e-12)

  g0 <- sz_mo_evaluate_constraints("dtlz9", dim = 4, x = c(0.0, 0.0, 0.0, 0.0), m = 2)
  expect_equal(g0[1], -1.0, tolerance = 1e-12)
})

test_that("sz_mo_evaluate_constraints is NULL for an unconstrained problem", {
  expect_null(sz_mo_evaluate_constraints("dtlz1", dim = 6, x = rep(0.5, 6), m = 3))
  expect_null(sz_mo_evaluate_constraints("zdt1", dim = 5, x = rep(0.5, 5)))
})

test_that("sz_mo_evaluate dtlz8 rejects dim == m (constraint-surface requires dim > m)", {
  expect_error(
    sz_mo_evaluate("dtlz8", dim = 3, x = c(0.0, 0.0, 0.0), m = 3),
    "dim must be > m"
  )
})

# wfg1 fixture (committed crates/problems/tests/data/wfg_reference_values.json,
# which=1, m=2, k=4, l=4 -- same points py-sezgi's own
# test_evaluate_wfg1_zero_and_max_fixture pins).

test_that("sz_mo_evaluate wfg1 zero and max fixture", {
  f <- sz_mo_evaluate("wfg1", dim = NULL, x = rep(0.0, 8), m = 2, k = 4, l = 4)
  expect_equal(f, c(1.0, 5.0), tolerance = 1e-8)

  f <- sz_mo_evaluate("wfg1", dim = NULL,
                       x = c(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0), m = 2, k = 4, l = 4)
  expect_equal(f, c(3.0, 1.0), tolerance = 1e-8)
})

test_that("sz_mo_evaluate wfg default k/l (k=4 for m=2, k=2*(m-1) for m>=3, l=20)", {
  expect_error(sz_mo_evaluate("wfg4", dim = NULL, x = c(0.0), m = 2), "24 coordinates")
  expect_error(sz_mo_evaluate("wfg4", dim = NULL, x = c(0.0), m = 3), "24 coordinates")
  expect_error(sz_mo_evaluate("wfg4", dim = NULL, x = c(0.0), m = 4), "26 coordinates")
})

# ---- error tests: dim/m/k/l mismatched to a problem family -----------------

test_that("sz_mo_evaluate rejects dim given for wfg", {
  expect_error(
    sz_mo_evaluate("wfg1", dim = 8, x = rep(0.0, 8), m = 2, k = 4, l = 4),
    "dim is not accepted for wfg"
  )
})

test_that("sz_mo_evaluate rejects a bad wfg k", {
  expect_error(
    sz_mo_evaluate("wfg1", dim = NULL, x = rep(0.0, 9), m = 3, k = 5, l = 4),
    "k %"
  )
})

test_that("sz_mo_evaluate rejects an odd l for wfg2", {
  expect_error(
    sz_mo_evaluate("wfg2", dim = NULL, x = rep(0.0, 9), m = 2, k = 4, l = 5),
    "l must be even"
  )
})

test_that("sz_mo_evaluate rejects k/l given for a zdt problem", {
  expect_error(
    sz_mo_evaluate("zdt1", dim = 5, x = rep(0.0, 5), k = 4),
    "wfg-only"
  )
})

test_that("sz_mo_evaluate rejects k/l given for a dtlz problem", {
  expect_error(
    sz_mo_evaluate("dtlz2", dim = 5, x = rep(0.0, 5), m = 3, l = 20),
    "wfg-only"
  )
})

# ---- sz_nsga2(log_dir=...) / sz_mo_read_moa: sezgi-moa logging -------------

test_that("sz_nsga2 log_dir requires label", {
  expect_error(
    sz_nsga2("zdt1", dim = 5, pop_size = 8, budget = 40, seed = 0, log_dir = tempdir()),
    "label is required"
  )
})

test_that("sz_nsga2 logged run end-to-end + sz_mo_read_moa round-trip", {
  log_dir <- tempfile("sezgi-moa-r-")
  dir.create(log_dir)
  r <- sz_nsga2("zdt1", dim = 5, pop_size = 8, budget = 60, seed = 123,
                log_dir = log_dir, label = "zdt1run")
  path <- file.path(log_dir, "zdt1run-s123.moa")
  expect_true(file.exists(path))

  d <- sz_mo_read_moa(path)
  expect_named(d, c("algo", "problem", "m", "seed", "budget", "kind", "records", "archive"))
  expect_identical(d$algo, "nsga2")
  expect_identical(d$problem, "zdt1run")
  expect_identical(d$m, 2)
  expect_identical(d$seed, 123)
  expect_identical(d$budget, 60)
  expect_identical(d$kind, "float")
  expect_true(length(d$records) >= 1)
  for (rec in d$records) {
    expect_named(rec, c("eval_index", "objectives", "genotype"))
    expect_length(rec$objectives, 2)
    expect_length(rec$genotype, 5)
  }

  # archive is mutually nondominated (plain Pareto dominance, minimization)
  archive <- d$archive
  expect_true(length(archive) >= 1)
  dominates <- function(a, b) all(a <= b) && any(a < b)
  for (i in seq_along(archive)) {
    for (j in seq_along(archive)) {
      if (i != j) {
        expect_false(dominates(archive[[i]], archive[[j]]))
      }
    }
  }

  expect_true(r$evals_used <= 60)
})

test_that("sz_mo_read_moa reports kind=\"binary\" and a logical genotype for zdt5", {
  log_dir <- tempfile("sezgi-moa-r-")
  dir.create(log_dir)
  sz_nsga2("zdt5", dim = NULL, pop_size = 8, budget = 40, seed = 5,
           log_dir = log_dir, label = "zdt5run")
  path <- file.path(log_dir, "zdt5run-s5.moa")
  d <- sz_mo_read_moa(path)
  expect_identical(d$kind, "binary")
  for (rec in d$records) {
    expect_length(rec$genotype, 80)
    expect_true(is.logical(rec$genotype))
  }
})

test_that("sz_mo_read_moa's at parameter reconstructs an earlier archive", {
  log_dir <- tempfile("sezgi-moa-r-")
  dir.create(log_dir)
  sz_nsga2("zdt1", dim = 5, pop_size = 8, budget = 60, seed = 9,
           log_dir = log_dir, label = "zdt1at")
  path <- file.path(log_dir, "zdt1at-s9.moa")
  d_full <- sz_mo_read_moa(path)
  d_early <- sz_mo_read_moa(path, at = 8) # only the init population
  expect_true(length(d_early$archive) <= length(d_full$archive))
})

test_that("sz_nsga2 logged run is byte-deterministic within R (two runs identical bytes)", {
  log_dir1 <- tempfile("sezgi-moa-r1-")
  log_dir2 <- tempfile("sezgi-moa-r2-")
  dir.create(log_dir1)
  dir.create(log_dir2)

  sz_nsga2("zdt1", dim = 5, pop_size = 8, budget = 60, seed = 77,
           log_dir = log_dir1, label = "det")
  sz_nsga2("zdt1", dim = 5, pop_size = 8, budget = 60, seed = 77,
           log_dir = log_dir2, label = "det")

  b1 <- readBin(file.path(log_dir1, "det-s77.moa"), "raw", file.info(file.path(log_dir1, "det-s77.moa"))$size)
  b2 <- readBin(file.path(log_dir2, "det-s77.moa"), "raw", file.info(file.path(log_dir2, "det-s77.moa"))$size)
  expect_identical(b1, b2)
})

test_that("sz_mo_read_moa rejects a missing file", {
  expect_error(sz_mo_read_moa(file.path(tempdir(), "does-not-exist.moa")))
})

# ---- Cross-language bit-equality anchors (M3-7): nsga2-on-wfg / -on-zdt5 --
#
# Both anchors re-confirmed live in this task via
# `./py-sezgi/.venv/bin/python -c "import sezgi; sezgi.mo.nsga2(...)"` --
# hex constants computed directly from that live run via
# `struct.pack('>d', x).hex()` (big-endian, matching f64_bits_hex()'s
# little-endian-raw-then-reversed output), never decimal literals (M3-2
# byte convention).

test_that("R sz_nsga2 is bit-identical to the Python/Rust wfg4 anchor invocation", {
  # Python: sezgi.mo.nsga2(problem="wfg4", dim=None, pop_size=8, budget=200,
  #   seed=20260901, m=2, k=4, l=4) gives evals_used=200, front0=[0..7] (all
  #   non-dominated), objectives[0] = [0.29462747543859824, 4.003752471477419],
  #   objectives[7][1] (last) = 3.9941042459802794.
  r <- sz_nsga2("wfg4", dim = NULL, m = 2, k = 4, l = 4,
                pop_size = 8, budget = 200, seed = 20260901)

  expect_identical(r$evals_used, 200)
  expect_identical(r$front0, as.double(1:8))

  expect_identical(f64_bits_hex(r$objectives[[1]][1]), "3fd2db2d32e0c1b4") # obj[0][0] = 0.29462747543859824
  expect_identical(f64_bits_hex(r$objectives[[1]][2]), "401003d7b0191948") # obj[0][1] = 4.003752471477419
  expect_identical(f64_bits_hex(r$objectives[[8]][2]), "400ff3eced4a667b") # obj[7][1] = 3.9941042459802794
  expect_identical(f64_bits_hex(as.double(r$evals_used)), "4069000000000000") # evals_used = 200
})

test_that("R sz_nsga2 is bit-identical to the Python/Rust zdt5 anchor invocation", {
  # Python: sezgi.mo.nsga2(problem="zdt5", dim=None, pop_size=8, budget=200,
  #   seed=20260901) gives evals_used=200, front0=[0..7] (all non-dominated),
  #   objectives[0] = [8.0, 3.625], objectives[7][1] (last) = 1.588235294117647.
  r <- sz_nsga2("zdt5", dim = NULL, pop_size = 8, budget = 200, seed = 20260901)

  expect_identical(r$evals_used, 200)
  expect_identical(r$front0, as.double(1:8))

  expect_identical(f64_bits_hex(r$objectives[[1]][1]), "4020000000000000") # obj[0][0] = 8.0
  expect_identical(f64_bits_hex(r$objectives[[1]][2]), "400d000000000000") # obj[0][1] = 3.625
  expect_identical(f64_bits_hex(r$objectives[[8]][2]), "3ff9696969696969") # obj[7][1] = 1.588235294117647
  expect_identical(f64_bits_hex(as.double(r$evals_used)), "4069000000000000") # evals_used = 200
})

# ---- Cross-language sezgi-moa file byte-identity (M3-7) --------------------
#
# Same (problem="zdt1", dim=5, pop_size=8, budget=60, seed=777,
# label="anchor") scenario run from both languages produces byte-identical
# .moa files, confirmed live in this task:
#   Python (./py-sezgi/.venv/bin/python): sha256
#     daefdb206c173c3c6352c702083f8856aaedd368ff40a7ff2254f33355586330,
#     md5 5517be3bcb8095ca64a2083872d6e850, file size 2883 bytes.
#   R (this same scenario, below): identical md5 5517be3bcb8095ca64a2083872d6e850
#     (verified via `diff`/`shasum -a 256` against the Python-produced file
#     during this task's own development -- byte-for-byte identical, 0
#     differing bytes).
# This test asserts against the committed md5 of R's OWN run (tools::md5sum,
# base R, no extra dependency) -- the Python match above is the
# cross-language evidence, measured once and quoted here.
test_that("sz_nsga2 log_dir run is byte-identical to the Python-produced .moa file (cross-language md5 anchor)", {
  log_dir <- tempfile("sezgi-moa-r-anchor-")
  dir.create(log_dir)
  r <- sz_nsga2("zdt1", dim = 5, pop_size = 8, budget = 60, seed = 777,
                log_dir = log_dir, label = "anchor")
  expect_identical(r$evals_used, 56)

  path <- file.path(log_dir, "anchor-s777.moa")
  expect_identical(unname(tools::md5sum(path)), "5517be3bcb8095ca64a2083872d6e850")
  expect_equal(file.info(path)$size, 2883)
})

# ---- examples/{python,r}/wfg4_nsga2.R parity anchor (M3-7 Task 12) --------
#
# The example pair's OWN live scenario (problem="wfg4", m=2, k/l omitted ->
# toolkit defaults k=4/l=20/dim=24, pop_size=40, budget=4000, seed=20260830,
# log_dir/label="wfg4demo" -- examples/python/wfg4_nsga2.py and
# examples/r/wfg4_nsga2.R), not the wfg4/zdt5 anchors above (which use
# different pop_size/budget/seed and, for wfg4, an explicit non-default
# l=4). Golden hex from
# `./py-sezgi/.venv/bin/python examples/python/wfg4_nsga2.py`
# (archive size = 217, evals_used = 4000, hypervolume =
# 3.0620256270488344 against ref_point=[2.2, 4.4] -- the WFG4/m=2 analytic
# front's nadir (2.0, 4.0) x 1.1, see that script's own module docstring
# for the derivation), byte-compared, not a decimal-literal eyeball match.
test_that("R wfg4_nsga2 example scenario is bit-identical to the Python/Rust golden (archive size, evals_used, hypervolume)", {
  problem <- "wfg4"
  m <- 2
  pop_size <- 40
  budget <- 4000
  seed <- 20260830
  label <- "wfg4demo"
  ref_point <- c(2.2, 4.4)

  log_dir <- tempfile("sezgi-moa-wfg4-anchor-")
  dir.create(log_dir)
  result <- sz_nsga2(problem, dim = NULL, m = m, pop_size = pop_size, budget = budget, seed = seed,
                      log_dir = log_dir, label = label)
  archive_run <- sz_mo_read_moa(file.path(log_dir, sprintf("%s-s%d.moa", label, seed)))
  archive_mat <- do.call(rbind, archive_run$archive)

  expect_identical(result$evals_used, 4000)
  expect_identical(nrow(archive_mat), 217L)

  hv <- sz_mo_hypervolume(archive_mat, ref_point)
  expect_identical(f64_bits_hex(hv), "40087f074abd8254")
})
