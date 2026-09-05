# M3-6 Task 10: R bindings for CEC 2014 and CEC 2017 (sz_cec2014_evaluate /
# sz_cec2014_f_star / sz_solve_cec2014 / sz_eval_session_cec2014, same trio
# for 2017), plus IOH-logging parity for both suites.
#
# Mirrors py-sezgi's `sezgi.problems` module (M3-6 Task 9, `py-sezgi/src/
# lib.rs`, `py-sezgi/tests/test_cec1417.py`) 1:1 by key name, and mirrors
# this package's own CEC 2022 precedent (`test-cec-tsp.R`'s CEC 2022
# sections + `test-session.R`'s `sz_eval_session_cec2022` sections) 1:1 in
# structure -- see `r-sezgi/src/rust/src/problems.rs`'s module doc for the
# `fid`/`dim` domain difference from CEC 2022 (`dim` in `{10,30}`, not
# `{2,10,20}`; no hybrid/dim=2 special case).
#
# CEC 2017's `fid` domain is `1` union `3..=30` -- `fid = 2` ("Sum of
# Different Powers") was officially withdrawn from the suite, and every
# entry point rejects it with a dedicated error (not an ordinary
# out-of-range `fid` error) -- verified verbatim below against the Rust
# `sezgi_problems::Cec2017Error::Withdrawn` message.

f64_bits_hex <- function(x) {
  raw <- writeBin(x, raw(), size = 8L, endian = "little")
  paste0(rev(sprintf("%02x", as.integer(raw))), collapse = "")
}

withdrawn_msg <- paste0(
  'fid 2 ("Sum of Different Powers") was officially withdrawn from the CEC ',
  "2017 suite; the reference C's own case 2 prints \"Error: This function ",
  '(F2) has been deleted" and leaves its result unset -- sezgi returns this ',
  "dedicated error instead, never a garbage f64 (module doc's F2 ruling ",
  "section has the full quoted C and probe transcript)"
)

# ---- sz_cec2014_f_star -----------------------------------------------------

test_that("sz_cec2014_f_star matches F_i* = 100*fid for all 30 fids", {
  actual <- vapply(as.double(1:30), sz_cec2014_f_star, numeric(1))
  expect_identical(actual, 100.0 * (1:30))
})

test_that("sz_cec2014_f_star rejects an invalid fid", {
  expect_error(sz_cec2014_f_star(0), "1\\.\\.=30")
  expect_error(sz_cec2014_f_star(31), "1\\.\\.=30")
})

# ---- sz_cec2014_evaluate ----------------------------------------------------

test_that("sz_cec2014_evaluate returns a finite value for a basic function", {
  v <- sz_cec2014_evaluate(fid = 1, dim = 10, x = rep(0, 10))
  expect_true(is.finite(v))
})

test_that("sz_cec2014_evaluate returns a finite value for a hybrid function (fid 17-22)", {
  v <- sz_cec2014_evaluate(fid = 17, dim = 10, x = rep(0, 10))
  expect_true(is.finite(v))
})

test_that("sz_cec2014_evaluate returns a finite value for a composition function (fid 23-30)", {
  v <- sz_cec2014_evaluate(fid = 23, dim = 10, x = rep(0, 10))
  expect_true(is.finite(v))
})

test_that("sz_cec2014_evaluate rejects an invalid fid", {
  expect_error(sz_cec2014_evaluate(fid = 0, dim = 10, x = rep(0, 10)), "1\\.\\.=30")
})

test_that("sz_cec2014_evaluate rejects an invalid dim", {
  expect_error(sz_cec2014_evaluate(fid = 1, dim = 5, x = rep(0, 5)), "\\{10,30\\}")
})

test_that("sz_cec2014_evaluate rejects a length mismatch between x and dim", {
  expect_error(sz_cec2014_evaluate(fid = 1, dim = 10, x = rep(0, 5)), "10")
})

# ---- sz_solve_cec2014 -------------------------------------------------------

test_that("sz_solve_cec2014 solves fid 3 dim 10 and returns a finite best_f >= 300.0", {
  spec <- sz_preset_shade(20, 300)
  r <- sz_solve_cec2014(spec, fid = 3, dim = 10, master_seed = 1, run_id = 0)
  expect_named(r, c("best_f", "evals", "best_x"))
  expect_true(is.finite(r$best_f))
  expect_gte(r$best_f, 300.0) # F* (fid 3's pinned optimum) is a lower bound
  expect_identical(r$evals, 300)
  expect_length(r$best_x, 10)
})

test_that("sz_solve_cec2014 rejects an invalid fid", {
  spec <- sz_preset_shade(20, 300)
  expect_error(sz_solve_cec2014(spec, fid = 31, dim = 10, master_seed = 1, run_id = 0), "1\\.\\.=30")
})

test_that("sz_solve_cec2014 rejects an invalid dim", {
  spec <- sz_preset_shade(20, 300)
  expect_error(sz_solve_cec2014(spec, fid = 1, dim = 5, master_seed = 1, run_id = 0), "\\{10,30\\}")
})

test_that("sz_solve_cec2014 is bit-identical run-twice for the same seed", {
  spec <- sz_preset_shade(20, 300)
  r1 <- sz_solve_cec2014(spec, fid = 3, dim = 10, master_seed = 7, run_id = 0)
  r2 <- sz_solve_cec2014(spec, fid = 3, dim = 10, master_seed = 7, run_id = 0)
  expect_identical(r1, r2)
})

# ---- sz_eval_session_cec2014 ------------------------------------------------

test_that("a CEC 2014 session's dim()/bounds() match its construction args", {
  s <- sz_eval_session_cec2014(fid = 1, dim = 10, budget = 10)
  expect_equal(s$dim(), 10)
  expect_equal(s$bounds(), c(-100, 100))
})

test_that("cec2014 session evaluate() matches sz_cec2014_evaluate() for two points", {
  fid <- 3
  dim <- 10
  s <- sz_eval_session_cec2014(fid = fid, dim = dim, budget = 10)
  x1 <- rep(0, dim)
  x2 <- seq_len(dim) * 0.1
  fs <- s$evaluate(matrix(c(x1, x2), nrow = 2, byrow = TRUE))
  expect_equal(fs[1], sz_cec2014_evaluate(fid, dim, x1))
  expect_equal(fs[2], sz_cec2014_evaluate(fid, dim, x2))
})

test_that("cec2014 session f_opt() matches sz_cec2014_f_star()", {
  fid <- 5
  s <- sz_eval_session_cec2014(fid = fid, dim = 10, budget = 10)
  expect_equal(s$f_opt(), sz_cec2014_f_star(fid))
})

test_that("cec2014 session with log_dir writes an IOH .dat file (sezgi-cec2014 suite/label)", {
  log_dir <- file.path(tempfile(), "log")
  s <- sz_eval_session_cec2014(fid = 1, dim = 10, budget = 3,
                                log_dir = log_dir, algo_name = "test-algo", seed = 7)
  s$evaluate(matrix(c(rep(0, 10), rep(1, 10), rep(2, 10)), nrow = 3, byrow = TRUE))
  s$finish()

  dat_files <- list.files(log_dir, pattern = "\\.dat$", recursive = TRUE, full.names = TRUE)
  expect_length(dat_files, 1)

  # Folder/label carry the "cec2014-f{fid}" name (crates/bench/src/ioh.rs's
  # `data_f{fid}_{fname}` naming) -- verified directly on disk, mirroring
  # py-sezgi's `test_for_problem_cec2014_logging_roundtrip`.
  expect_true(dir.exists(file.path(log_dir, "test-algo", "data_f1_cec2014-f1")))

  records <- sz_read_ioh_records(log_dir, 3)
  expect_equal(nrow(records), 1)
  expect_equal(records$seed, 7)
  expect_equal(records$suite, "sezgi-cec2014")
  expect_equal(records$f_opt, 100.0)
})

# ---- sz_cec2017_f_star -----------------------------------------------------

CEC2017_FIDS <- c(1, 3:30)

test_that("sz_cec2017_f_star matches F_i* = 100*fid (dispatch bias) for every valid fid", {
  actual <- vapply(as.double(CEC2017_FIDS), sz_cec2017_f_star, numeric(1))
  expect_identical(actual, 100.0 * CEC2017_FIDS)
})

test_that("sz_cec2017_f_star rejects an invalid fid", {
  expect_error(sz_cec2017_f_star(0), "1 or in 3\\.\\.=30")
  expect_error(sz_cec2017_f_star(31), "1 or in 3\\.\\.=30")
})

test_that("sz_cec2017_f_star: fid=2 raises the dedicated withdrawn error, VERBATIM", {
  expect_error(sz_cec2017_f_star(2), withdrawn_msg, fixed = TRUE)
})

# ---- sz_cec2017_evaluate ----------------------------------------------------

test_that("sz_cec2017_evaluate returns a finite value for a basic function", {
  v <- sz_cec2017_evaluate(fid = 1, dim = 10, x = rep(0, 10))
  expect_true(is.finite(v))
})

test_that("sz_cec2017_evaluate returns a finite value for a hybrid function (fid 11-20)", {
  v <- sz_cec2017_evaluate(fid = 11, dim = 10, x = rep(0, 10))
  expect_true(is.finite(v))
})

test_that("sz_cec2017_evaluate returns a finite value for a composition function (fid 21-30)", {
  v <- sz_cec2017_evaluate(fid = 21, dim = 10, x = rep(0, 10))
  expect_true(is.finite(v))
})

test_that("sz_cec2017_evaluate: fid=2 raises the dedicated withdrawn error, VERBATIM", {
  expect_error(sz_cec2017_evaluate(fid = 2, dim = 10, x = rep(0, 10)), withdrawn_msg, fixed = TRUE)
})

test_that("sz_cec2017_evaluate rejects an invalid fid", {
  expect_error(sz_cec2017_evaluate(fid = 0, dim = 10, x = rep(0, 10)), "1 or in 3\\.\\.=30")
})

test_that("sz_cec2017_evaluate rejects an invalid dim", {
  expect_error(sz_cec2017_evaluate(fid = 1, dim = 5, x = rep(0, 5)), "\\{10,30\\}")
})

test_that("sz_cec2017_evaluate rejects a length mismatch between x and dim", {
  expect_error(sz_cec2017_evaluate(fid = 1, dim = 10, x = rep(0, 5)), "10")
})

# ---- sz_solve_cec2017 -------------------------------------------------------

test_that("sz_solve_cec2017 solves fid 3 dim 10 and returns a finite best_f >= 300.0", {
  spec <- sz_preset_shade(20, 300)
  r <- sz_solve_cec2017(spec, fid = 3, dim = 10, master_seed = 1, run_id = 0)
  expect_named(r, c("best_f", "evals", "best_x"))
  expect_true(is.finite(r$best_f))
  expect_gte(r$best_f, 300.0) # F* (fid 3's pinned optimum) is a lower bound
  expect_identical(r$evals, 300)
  expect_length(r$best_x, 10)
})

test_that("sz_solve_cec2017 rejects an invalid fid", {
  spec <- sz_preset_shade(20, 300)
  expect_error(sz_solve_cec2017(spec, fid = 31, dim = 10, master_seed = 1, run_id = 0), "1 or in 3\\.\\.=30")
})

test_that("sz_solve_cec2017: fid=2 raises the dedicated withdrawn error, VERBATIM", {
  spec <- sz_preset_shade(20, 300)
  expect_error(sz_solve_cec2017(spec, fid = 2, dim = 10, master_seed = 1, run_id = 0),
               withdrawn_msg, fixed = TRUE)
})

test_that("sz_solve_cec2017 rejects an invalid dim", {
  spec <- sz_preset_shade(20, 300)
  expect_error(sz_solve_cec2017(spec, fid = 1, dim = 5, master_seed = 1, run_id = 0), "\\{10,30\\}")
})

test_that("sz_solve_cec2017 is bit-identical run-twice for the same seed", {
  spec <- sz_preset_shade(20, 300)
  r1 <- sz_solve_cec2017(spec, fid = 3, dim = 10, master_seed = 7, run_id = 0)
  r2 <- sz_solve_cec2017(spec, fid = 3, dim = 10, master_seed = 7, run_id = 0)
  expect_identical(r1, r2)
})

# ---- sz_eval_session_cec2017 ------------------------------------------------

test_that("a CEC 2017 session's dim()/bounds() match its construction args", {
  s <- sz_eval_session_cec2017(fid = 1, dim = 10, budget = 10)
  expect_equal(s$dim(), 10)
  expect_equal(s$bounds(), c(-100, 100))
})

test_that("cec2017 session evaluate() matches sz_cec2017_evaluate() for two points", {
  fid <- 3
  dim <- 10
  s <- sz_eval_session_cec2017(fid = fid, dim = dim, budget = 10)
  x1 <- rep(0, dim)
  x2 <- seq_len(dim) * 0.1
  fs <- s$evaluate(matrix(c(x1, x2), nrow = 2, byrow = TRUE))
  expect_equal(fs[1], sz_cec2017_evaluate(fid, dim, x1))
  expect_equal(fs[2], sz_cec2017_evaluate(fid, dim, x2))
})

test_that("cec2017 session f_opt() matches sz_cec2017_f_star()", {
  fid <- 5
  s <- sz_eval_session_cec2017(fid = fid, dim = 10, budget = 10)
  expect_equal(s$f_opt(), sz_cec2017_f_star(fid))
})

test_that("cec2017 session construction rejects fid=2 (withdrawn), VERBATIM", {
  expect_error(sz_eval_session_cec2017(fid = 2, dim = 10, budget = 10), withdrawn_msg, fixed = TRUE)
})

test_that("cec2017 session with log_dir writes an IOH .dat file (sezgi-cec2017 suite/label)", {
  log_dir <- file.path(tempfile(), "log")
  s <- sz_eval_session_cec2017(fid = 1, dim = 10, budget = 3,
                                log_dir = log_dir, algo_name = "test-algo", seed = 7)
  s$evaluate(matrix(c(rep(0, 10), rep(1, 10), rep(2, 10)), nrow = 3, byrow = TRUE))
  s$finish()

  dat_files <- list.files(log_dir, pattern = "\\.dat$", recursive = TRUE, full.names = TRUE)
  expect_length(dat_files, 1)

  expect_true(dir.exists(file.path(log_dir, "test-algo", "data_f1_cec2017-f1")))

  records <- sz_read_ioh_records(log_dir, 3)
  expect_equal(nrow(records), 1)
  expect_equal(records$seed, 7)
  expect_equal(records$suite, "sezgi-cec2017")
  expect_equal(records$f_opt, 100.0)
})

# ---- No collision: cec2014 vs cec2017, same fid ----------------------------
# Mirrors py-sezgi's `test_no_collision_cec2014_vs_cec2017_same_fid`: one
# CEC 2014 f1 d10 run and one CEC 2017 f1 d10 run, same seed/budget, logged
# into the same tree -- both labels present, distinct.

test_that("cec2014 and cec2017 sessions at the same fid do not collide in one log tree", {
  log_dir <- file.path(tempfile(), "log")
  s1 <- sz_eval_session_cec2014(fid = 1, dim = 10, budget = 3,
                                 log_dir = log_dir, algo_name = "probe", seed = 1)
  s1$evaluate(matrix(rep(0, 30), nrow = 3, byrow = TRUE))
  s1$finish()

  s2 <- sz_eval_session_cec2017(fid = 1, dim = 10, budget = 3,
                                 log_dir = log_dir, algo_name = "probe", seed = 1)
  s2$evaluate(matrix(rep(0, 30), nrow = 3, byrow = TRUE))
  s2$finish()

  records <- sz_read_ioh_records(log_dir, 3)
  expect_equal(nrow(records), 2)
  expect_setequal(records$suite, c("sezgi-cec2014", "sezgi-cec2017"))
})

# ---- Cross-language bit-equality anchors ------------------------------------
#
# Hex constants computed from a live run against the SAME venv this task's
# Python bindings were built into (`./py-sezgi/.venv/bin/python`), via
# `struct.pack('>d', v).hex()` (big-endian byte order, matching
# `f64_bits_hex()`'s little-endian-raw-then-reversed output). Bytes are
# compared directly -- no decimal literal is parsed in R for the compared
# value itself (the R strtod 1-ULP precedent from M3-2) -- `x` below is
# small integers, exactly representable in binary, so there is no
# parse-precision risk on the input side either. Same scenario shape as
# `test-cec-tsp.R`'s own `sz_cec2022_evaluate`/`sz_solve_cec2022` anchors
# (fid=1/dim=10/x=1..10 for direct-eval; fid=3/dim=10/seed=99/SHADE(20,1000)
# for solve), reused verbatim here since it is valid for both new suites too
# (`dim=10` and `fid=3` are both in-domain for CEC 2014 and CEC 2017).
#
# Commands run (in order):
#
#   ./py-sezgi/.venv/bin/python -c "
#   import struct, sezgi
#   def hexof(v): return struct.pack('>d', v).hex()
#   x = [float(i) for i in range(1, 11)]
#   print(hexof(sezgi.problems.cec2014_evaluate(1, 10, x)))  # 41ede0fc4e670ae0 (4010271347.2200775)
#   print(hexof(sezgi.problems.cec2017_evaluate(1, 10, x)))  # 421953d501114189 (27195162692.314)
#   "
#
# The `sz_solve_cec2014`/`sz_solve_cec2017` anchors use the SAME
# spec-equivalence method as `test-cec-tsp.R`'s `sz_solve_cec2022` anchor:
# R's `sz_preset_shade(20, 1000)` output was written to a file
# (`Rscript -e 'library(sezgi); cat(sz_preset_shade(20, 1000))' > /tmp/r_spec_1000.json`)
# and that EXACT R-generated JSON string was read and passed verbatim into
# `sezgi.solve()` below -- both sides consume byte-identical spec content:
#
#   ./py-sezgi/.venv/bin/python -c "
#   import struct, sezgi
#   def hexof(v): return struct.pack('>d', v).hex()
#   spec_json = open('/tmp/r_spec_1000.json').read()  # verbatim sz_preset_shade(20, 1000) output
#   p14 = sezgi.problems.cec2014(3, 10)
#   r14 = sezgi.solve(spec_json, p14, master_seed=99, run_id=0)
#   print(hexof(r14['best_f']))  # 409d39e5345a120f (1870.4738325189135)
#   print(r14['evals_used'])     # 1000
#   p17 = sezgi.problems.cec2017(3, 10)
#   r17 = sezgi.solve(spec_json, p17, master_seed=99, run_id=0)
#   print(hexof(r17['best_f']))  # 409a08b80eb77c91 (1666.179743639925)
#   print(r17['evals_used'])     # 1000
#   "

test_that("R sz_cec2014_evaluate is bit-identical to the Python anchor (fid=1, dim=10, x=1..10)", {
  # Float golden anchored on the reference platform; foreign libm rounding diverges by ULPs.
  skip_on_cran()
  v <- sz_cec2014_evaluate(fid = 1, dim = 10, x = as.double(1:10))
  expect_identical(f64_bits_hex(v), "41ede0fc4e670ae0")
})

test_that("R sz_cec2017_evaluate is bit-identical to the Python anchor (fid=1, dim=10, x=1..10)", {
  # Float golden anchored on the reference platform; foreign libm rounding diverges by ULPs.
  skip_on_cran()
  v <- sz_cec2017_evaluate(fid = 1, dim = 10, x = as.double(1:10))
  expect_identical(f64_bits_hex(v), "421953d501114189")
})

test_that("R sz_solve_cec2014 (SHADE) is bit-identical to the Python/Rust golden (fid=3, dim=10, seed=99)", {
  # Float golden anchored on the reference platform; foreign libm rounding diverges by ULPs.
  skip_on_cran()
  spec <- sz_preset_shade(20, 1000)
  r <- sz_solve_cec2014(spec, fid = 3, dim = 10, master_seed = 99, run_id = 0)
  expect_identical(f64_bits_hex(r$best_f), "409d39e5345a120f")
  expect_identical(r$evals, 1000)
})

test_that("R sz_solve_cec2017 (SHADE) is bit-identical to the Python/Rust golden (fid=3, dim=10, seed=99)", {
  # Float golden anchored on the reference platform; foreign libm rounding diverges by ULPs.
  skip_on_cran()
  spec <- sz_preset_shade(20, 1000)
  r <- sz_solve_cec2017(spec, fid = 3, dim = 10, master_seed = 99, run_id = 0)
  expect_identical(f64_bits_hex(r$best_f), "409a08b80eb77c91")
  expect_identical(r$evals, 1000)
})

# ---- examples/{python,r}/cec2014_lshade.R parity anchor ---------------------
#
# The example pair's OWN live scenario (fid=1, dim=10, L-SHADE, budget=9000,
# seed=20260830 -- examples/python/cec2014_lshade.py and
# examples/r/cec2014_lshade.R), not the fid=3/seed=99 anchor above. Golden
# hex from `./py-sezgi/.venv/bin/python examples/python/cec2014_lshade.py`
# (best_f = 180.23000508877507, evals_used = 9000), byte-compared, not a
# decimal-literal eyeball match.

test_that("R sz_solve_cec2014 (L-SHADE) is bit-identical to the Python/Rust golden (fid=1, dim=10, seed=20260830 -- the example pair's own scenario)", {
  # Float golden anchored on the reference platform; foreign libm rounding diverges by ULPs.
  skip_on_cran()
  spec <- sz_preset_lshade(10, 9000)
  r <- sz_solve_cec2014(spec, fid = 1, dim = 10, master_seed = 20260830, run_id = 0)
  expect_identical(f64_bits_hex(r$best_f), "4066875c33a1c67b")
  expect_identical(r$evals, 9000)
})
