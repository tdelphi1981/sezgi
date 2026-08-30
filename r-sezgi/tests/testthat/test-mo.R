# M3-2 Task 10: R bindings for the multi-objective module (sz_nsga2 /
# sz_mo_hypervolume_2d / sz_mo_igd / sz_mo_pareto_front).
#
# Mirrors crates/components::nsga2 / crates/problems::{zdt,dtlz} /
# crates/stats::moo_indicators 1:1 by dict key name -- see py-sezgi's
# `sezgi.mo` module (M3-2 Task 9, `py-sezgi/src/lib.rs`) and
# `.superpowers/sdd/2026-08-30-sezgi-m3-2/task-9-report.md` for the full
# field-by-field provenance, and `r-sezgi/src/rust/src/mo.rs`'s module doc
# for this binding's own container-idiom decisions (list-of-vectors for
# `individuals`/`objectives`/`sz_mo_pareto_front`'s return, matrix for
# `sz_mo_hypervolume_2d`/`sz_mo_igd`'s `front`/`reference_front` inputs,
# 1-based `front0`).

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
# Anchor invocation (documented in
# .superpowers/sdd/2026-08-30-sezgi-m3-2/task-9-report.md, "Deterministic
# zdt1 invocation + exact output values for T10's cross-language fixture",
# and re-confirmed live in this task via
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
