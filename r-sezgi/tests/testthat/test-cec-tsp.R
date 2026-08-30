# M3-3 Task 10: R bindings for CEC 2022 + TSP (sz_cec2022_evaluate /
# sz_cec2022_f_star / sz_tsp_load / sz_tsp_tour_length / sz_preset_ga_perm /
# sz_solve_tsp).
#
# Mirrors py-sezgi's `sezgi.problems`/`sezgi.presets.ga_perm` module 1:1 by
# key name for direct evaluation (M3-3 Task 9, `py-sezgi/src/lib.rs`, see
# `.superpowers/sdd/2026-08-31-sezgi-m3-3/task-9-report.md` for the full
# provenance) -- EXCEPT for the index convention documented next.
#
# Index-convention decision (r-sezgi/src/rust/src/problems.rs's own module
# doc has the full rationale): r-sezgi uses 1-based city numbering
# throughout -- UNLIKE py-sezgi's 0-based `tour`/`coords`. This mirrors the
# established `sz_stats_plackett_luce`/`sz_bayesian_plackett_luce`
# `rankings` 1-based item-id precedent (`stats.rs`'s `rankings_from_list`),
# not py-sezgi's own 0-based choice. `sz_tsp_tour_length`'s `tour` is a
# permutation of `1:n_cities`; `sz_tsp_load`'s `coords` matrix row `i` is
# city `i`; `sz_solve_tsp`'s `best_x` is a 1-based permutation of
# `1:n_cities` -- consistent with TSPLIB's own 1-based node numbering (so
# `crates/problems/data/tsplib/berlin52.opt.tour`'s TOUR_SECTION can be used
# verbatim as an R tour with no +1 conversion, unlike Python's, which needs
# -1).

f64_bits_hex <- function(x) {
  raw <- writeBin(x, raw(), size = 8L, endian = "little")
  paste0(rev(sprintf("%02x", as.integer(raw))), collapse = "")
}

# ---- sz_cec2022_f_star -----------------------------------------------------

test_that("sz_cec2022_f_star matches the report's F_i* table for all 12 fids", {
  expected <- c(300, 400, 600, 800, 900, 1800, 2000, 2200, 2300, 2400, 2600, 2700)
  actual <- vapply(as.double(1:12), sz_cec2022_f_star, numeric(1))
  expect_identical(actual, as.double(expected))
})

test_that("sz_cec2022_f_star rejects an invalid fid", {
  expect_error(sz_cec2022_f_star(0), "1..=12")
  expect_error(sz_cec2022_f_star(13), "1..=12")
})

# ---- sz_cec2022_evaluate ----------------------------------------------------

test_that("sz_cec2022_evaluate returns a finite value for a basic function", {
  v <- sz_cec2022_evaluate(fid = 1, dim = 10, x = rep(0, 10))
  expect_true(is.finite(v))
})

test_that("sz_cec2022_evaluate returns a finite value for a hybrid function (fid 6-8)", {
  v <- sz_cec2022_evaluate(fid = 6, dim = 10, x = rep(0, 10))
  expect_true(is.finite(v))
})

test_that("sz_cec2022_evaluate rejects an invalid fid", {
  expect_error(sz_cec2022_evaluate(fid = 0, dim = 10, x = rep(0, 10)), "1..=12")
})

test_that("sz_cec2022_evaluate rejects an invalid dim", {
  expect_error(sz_cec2022_evaluate(fid = 1, dim = 5, x = rep(0, 5)), "\\{2,10,20\\}")
})

test_that("sz_cec2022_evaluate rejects dim=2 for a hybrid function", {
  expect_error(sz_cec2022_evaluate(fid = 6, dim = 2, x = rep(0, 2)), "hybrid")
})

test_that("sz_cec2022_evaluate rejects a length mismatch between x and dim", {
  expect_error(sz_cec2022_evaluate(fid = 1, dim = 10, x = rep(0, 5)), "10")
})

# ---- sz_tsp_load ------------------------------------------------------------

test_that("sz_tsp_load loads berlin52 by vendored name", {
  t <- sz_tsp_load("berlin52")
  expect_named(t, c("name", "n_cities", "coords", "known_optimum"))
  expect_identical(t$name, "berlin52")
  expect_identical(t$n_cities, 52)
  expect_true(is.matrix(t$coords))
  expect_identical(dim(t$coords), c(52L, 2L))
  # City 1 (R 1-based) = TSPLIB node 1 = genotype index 0 (Rust); Rust's own
  # `berlin52_parses_with_expected_dimension_and_spot_checked_coords` test
  # pins coords()[0] == (565.0, 575.0).
  expect_identical(t$coords[1, ], c(565.0, 575.0))
  # City 52 = TSPLIB node 52 = genotype index 51; Rust pins (1740.0, 245.0).
  expect_identical(t$coords[52, ], c(1740.0, 245.0))
  expect_identical(t$known_optimum, 7542.0)
})

test_that("sz_tsp_load falls back to raw TSPLIB text for an unrecognized name", {
  text <- "NAME: tiny\nTYPE: TSP\nDIMENSION: 2\nEDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n1 0.0 0.0\n2 3.0 4.0\n"
  t <- sz_tsp_load(text)
  expect_identical(t$n_cities, 2)
  expect_identical(t$coords[1, ], c(0.0, 0.0))
  expect_identical(t$coords[2, ], c(3.0, 4.0))
  expect_null(t$known_optimum) # not a vendored instance -- no published optimum
})

test_that("sz_tsp_load raises a clean error for garbage input (not vendored, not valid TSPLIB)", {
  expect_error(sz_tsp_load("this is not a tsplib file"))
})

# ---- sz_tsp_tour_length -----------------------------------------------------

# berlin52's published-optimal tour, TSPLIB TOUR_SECTION verbatim (already
# 1-based node numbers -- see this file's own header note) --
# `crates/problems/data/tsplib/berlin52.opt.tour`.
berlin52_opt_tour_1based <- c(
  1, 49, 32, 45, 19, 41, 8, 9, 10, 43, 33, 51, 11, 52, 14, 13, 47, 26, 27, 28,
  12, 25, 4, 6, 15, 5, 24, 48, 38, 37, 40, 39, 36, 35, 34, 44, 46, 16, 29, 50,
  20, 23, 30, 2, 7, 42, 21, 17, 3, 18, 31, 22
)

test_that("sz_tsp_tour_length: berlin52 published-optimal tour is exactly 7542", {
  v <- sz_tsp_tour_length("berlin52", berlin52_opt_tour_1based)
  expect_identical(v, 7542.0)
})

test_that("sz_tsp_tour_length rejects the wrong number of entries", {
  expect_error(sz_tsp_tour_length("berlin52", as.double(1:51)), "52")
})

test_that("sz_tsp_tour_length rejects an out-of-range entry (0)", {
  tour <- berlin52_opt_tour_1based
  tour[1] <- 0
  expect_error(sz_tsp_tour_length("berlin52", tour))
})

test_that("sz_tsp_tour_length rejects an out-of-range entry (n + 1)", {
  tour <- berlin52_opt_tour_1based
  tour[1] <- 53
  expect_error(sz_tsp_tour_length("berlin52", tour), "out of range")
})

test_that("sz_tsp_tour_length rejects a duplicated entry", {
  tour <- berlin52_opt_tour_1based
  tour[2] <- tour[1] # duplicate the first city
  expect_error(sz_tsp_tour_length("berlin52", tour), "repeated")
})

test_that("sz_tsp_tour_length rejects a negative entry", {
  tour <- berlin52_opt_tour_1based
  tour[1] <- -1
  expect_error(sz_tsp_tour_length("berlin52", tour))
})

# ---- sz_preset_ga_perm -------------------------------------------------------

test_that("sz_preset_ga_perm has the expected spec shape", {
  skip_if_not_installed("jsonlite")
  spec <- jsonlite::fromJSON(sz_preset_ga_perm(32, 1000))
  expect_identical(spec$name, "ga-perm")
  expect_identical(spec$pop_size, 32L)
  expect_identical(spec$termination$budget, 1000L)
  expect_identical(spec$stages$generator$kind[[1]], "gen/ga-perm")
  expect_identical(spec$stages$replacer$kind[[1]], "replace/mu-plus-lambda")
})

# ---- sz_solve_tsp -------------------------------------------------------------

test_that("sz_solve_tsp solves berlin52 and returns a valid 1-based tour", {
  spec <- sz_preset_ga_perm(32, 1000)
  r <- sz_solve_tsp(spec, "berlin52", master_seed = 1, run_id = 0)
  expect_named(r, c("best_f", "evals", "best_x"))
  expect_true(is.finite(r$best_f))
  expect_length(r$best_x, 52)
  expect_identical(sort(r$best_x), as.double(1:52)) # a permutation of 1:52
})

test_that("sz_solve_tsp is bit-identical run-twice for the same seed", {
  spec <- sz_preset_ga_perm(32, 1000)
  r1 <- sz_solve_tsp(spec, "berlin52", master_seed = 7, run_id = 0)
  r2 <- sz_solve_tsp(spec, "berlin52", master_seed = 7, run_id = 0)
  expect_identical(r1, r2)
})

# ---- Cross-language bit-equality anchor ------------------------------------
#
# Hex constants computed from a live run against the SAME venv T9 built
# (`./py-sezgi/.venv/bin/python`), via `struct.pack('>d', v).hex()`
# (big-endian byte order, matching f64_bits_hex()'s little-endian-raw-then-
# reversed output). Bytes are compared directly -- no decimal literal is
# parsed in R for the compared value itself (the R strtod 1-ULP precedent
# from M3-2) -- `x`/tour inputs below are small integers, exactly
# representable in binary, so there is no parse-precision risk on the INPUT
# side either.
#
# Commands run (in order):
#
#   ./py-sezgi/.venv/bin/python -c "
#   import struct, sezgi
#   def hexof(v): return struct.pack('>d', v).hex()
#   x = [float(i) for i in range(1, 11)]
#   print(hexof(sezgi.problems.cec2022_evaluate(1, 10, x)))   # 42244fbead4c2ae3 (43618621094.08376)
#   print(hexof(sezgi.problems.cec2022_evaluate(6, 10, x)))   # 4206fb26783bc8a1 (12337860359.472963)
#   print(hexof(sezgi.problems.cec2022_f_star(1)))            # 4072c00000000000 (300.0)
#   print(hexof(sezgi.problems.cec2022_f_star(7)))            # 409f400000000000 (2000.0)
#   with open('crates/problems/data/tsplib/berlin52.opt.tour') as f:
#       lines = [l.strip() for l in f]
#   tour_1based = []
#   in_section = False
#   for l in lines:
#       if l == 'TOUR_SECTION': in_section = True; continue
#       if in_section:
#           if l in ('-1', 'EOF', ''): break
#           tour_1based.append(int(l))
#   tour0 = [t - 1 for t in tour_1based]
#   print(hexof(sezgi.problems.tsp_tour_length('berlin52', tour0)))  # 40bd760000000000 (7542.0)
#   "
#
#   ./py-sezgi/.venv/bin/python -c "
#   import struct, sezgi, json
#   def hexof(v): return struct.pack('>d', v).hex()
#   spec = sezgi.presets.ga_perm(pop_size=32, budget=1000)
#   prob = sezgi.problems.tsp('berlin52')
#   r = sezgi.solve(json.dumps(spec), prob, master_seed=42, run_id=0)
#   print(hexof(r['best_f']))   # 40cf978000000000 (16175.0)
#   print(r['evals_used'])      # 992
#   print(r['best_x'])          # [46, 25, 36, 45, 14, 15, 22, 29, 20, 41, 16, 19, 49, 28, 48, 21, 17, 33,
#                                #  44, 42, 10, 32, 50, 11, 27, 8, 40, 30, 35, 18, 0, 43, 31, 38, 34, 37,
#                                #  4, 2, 6, 1, 7, 9, 47, 5, 3, 24, 39, 23, 26, 13, 12, 51]  (0-based)
#   "

test_that("R sz_cec2022_evaluate is bit-identical to the Python anchor (fid=1, dim=10, x=1..10)", {
  v <- sz_cec2022_evaluate(fid = 1, dim = 10, x = as.double(1:10))
  expect_identical(f64_bits_hex(v), "42244fbead4c2ae3")
})

test_that("R sz_cec2022_evaluate is bit-identical to the Python anchor (fid=6 hybrid, dim=10, x=1..10)", {
  v <- sz_cec2022_evaluate(fid = 6, dim = 10, x = as.double(1:10))
  expect_identical(f64_bits_hex(v), "4206fb26783bc8a1")
})

test_that("R sz_cec2022_f_star is bit-identical to the Python anchor (fid 1 and 7)", {
  expect_identical(f64_bits_hex(sz_cec2022_f_star(1)), "4072c00000000000")
  expect_identical(f64_bits_hex(sz_cec2022_f_star(7)), "409f400000000000")
})

test_that("R sz_tsp_tour_length is bit-identical to the Python anchor (berlin52 optimal tour)", {
  v <- sz_tsp_tour_length("berlin52", berlin52_opt_tour_1based)
  expect_identical(f64_bits_hex(v), "40bd760000000000")
})

test_that("R sz_solve_tsp (ga-perm) is bit-identical to the Python/Rust golden (berlin52, seed 42)", {
  spec <- sz_preset_ga_perm(32, 1000)
  r <- sz_solve_tsp(spec, "berlin52", master_seed = 42, run_id = 0)

  expect_identical(f64_bits_hex(r$best_f), "40cf978000000000")
  expect_identical(r$evals, 992)

  # Python's best_x is 0-based; +1 for R's 1-based convention (this file's
  # own header note).
  python_best_x_0based <- c(
    46, 25, 36, 45, 14, 15, 22, 29, 20, 41, 16, 19, 49, 28, 48, 21, 17, 33,
    44, 42, 10, 32, 50, 11, 27, 8, 40, 30, 35, 18, 0, 43, 31, 38, 34, 37,
    4, 2, 6, 1, 7, 9, 47, 5, 3, 24, 39, 23, 26, 13, 12, 51
  )
  expect_identical(r$best_x, python_best_x_0based + 1)
})
