# M3-8 Task 8: determinism/anchored-output + cross-language hex-parity gate
# for examples/r/oop/tsp_two_opt.R -- the R twin of
# examples/python/oop/tsp_two_opt.py (M3-8 Task 7).
#
# tsp_two_opt.R has no pure-script counterpart under examples/r/ (see its
# own header comment) -- unlike examples/r/oop/gwo.R, it sits OUTSIDE the
# "R OOP twin matches the pure script" subprocess-diff gate in test-algo.R.
# It gets its OWN gate here instead, mirroring
# py-sezgi/tests/test_tsp_two_opt_example.py's "run the script as a
# subprocess, parse its printed metrics line" pattern, PLUS a cross-language
# hex anchor in the house convention (test-mo.R's wfg4_nsga2 pair anchor,
# test-cross-language.R's golden-hex pattern): the deterministic best_f as
# an exact f64, compared as BYTES via writeBin, never a decimal literal.

#' Locates the repo root by walking up from `start` looking for
#' `examples/r/oop/tsp_two_opt.R` -- same approach as test-algo.R's
#' `.gwo_repo_root()` (not reused directly: each testthat file here is
#' self-contained, matching this file's own local-helper convention already
#' used by `golden_path()` in test-cross-language.R / `f64_bits_hex()` in
#' test-mo.R).
.tsp_two_opt_repo_root <- function() {
  marker <- file.path("examples", "r", "oop", "tsp_two_opt.R")
  start_points <- getwd()
  test_path_dir <- tryCatch(dirname(testthat::test_path()), error = function(e) NA_character_)
  if (!is.na(test_path_dir)) start_points <- c(test_path_dir, start_points)

  for (start in start_points) {
    dir <- normalizePath(start, mustWork = FALSE)
    for (i in 0:6) {
      if (file.exists(file.path(dir, marker))) {
        return(dir)
      }
      parent <- dirname(dir)
      if (identical(parent, dir)) break
      dir <- parent
    }
  }
  stop("could not locate repo root (examples/r/oop/tsp_two_opt.R not found walking up from getwd()/test_path())")
}

f64_bits_hex <- function(x) {
  raw <- writeBin(x, raw(), size = 8L, endian = "little")
  paste0(rev(sprintf("%02x", as.integer(raw))), collapse = "")
}

.tsp_two_opt_run_fields <- function(root) {
  script <- file.path(root, "examples", "r", "oop", "tsp_two_opt.R")
  out <- system2("Rscript", args = script, stdout = TRUE, stderr = TRUE)
  line <- grep("evals_used=", out, value = TRUE)
  expect_length(line, 1)
  m <- regmatches(line, regexec(
    "evals_used=(\\S+) best_f=(\\S+) gap=(\\S+) tour_length=(\\S+)", line
  ))[[1]]
  expect_length(m, 5)
  stats::setNames(as.list(m[2:5]), c("evals_used", "best_f", "gap", "tour_length"))
}

test_that("R tsp_two_opt example runs and reports valid metrics", {
  root <- .tsp_two_opt_repo_root()
  f <- .tsp_two_opt_run_fields(root)
  expect_identical(as.integer(f$evals_used), 2000L)
  expect_identical(f$best_f, f$tour_length)
  expect_gte(as.numeric(f$gap), 0.0)
})

test_that("R tsp_two_opt example is deterministic across runs", {
  root <- .tsp_two_opt_repo_root()
  a <- .tsp_two_opt_run_fields(root)
  b <- .tsp_two_opt_run_fields(root)
  expect_identical(a, b)
})

test_that("R tsp_two_opt example anchored output matches the Python twin", {
  # ANCHORED (per this project's convention): measured with
  # INSTANCE="berlin52", BUDGET=2000, SEED=42 (tsp_two_opt.R's own
  # constants -- identical to tsp_two_opt.py's). The printed
  # evals_used/best_f/gap/tour_length fields are language-neutral (a tour
  # LENGTH is invariant to whether the tour itself is indexed 0-based
  # (Python) or 1-based (R) -- see tsp_two_opt.R's own header comment), so
  # this R twin is expected to reproduce py-sezgi's own anchored values
  # (py-sezgi/tests/test_tsp_two_opt_example.py::
  # test_tsp_two_opt_example_anchored_output) EXACTLY.
  root <- .tsp_two_opt_repo_root()
  f <- .tsp_two_opt_run_fields(root)
  expect_identical(f$evals_used, "2000")
  expect_identical(f$best_f, "9077")
  expect_identical(f$gap, "1535")
  expect_identical(f$tour_length, "9077")
})

test_that("R tsp_two_opt best_f is bit-identical to the Python twin (cross-language hex anchor)", {
  # Cross-language anchor (house precedent: the wfg4_nsga2 pair's hex
  # anchor, test-mo.R's "examples/{python,r}/wfg4_nsga2.R parity anchor").
  # Golden hex derived by running the Python side directly:
  #
  #   ./py-sezgi/.venv/bin/python -c "
  #   import struct, sys
  #   sys.path.insert(0, 'examples/python/oop')
  #   import tsp_two_opt as m
  #   import sezgi
  #   res = m.TspTwoOpt().solve(sezgi.problems.tsp(m.INSTANCE), budget=m.BUDGET, seed=m.SEED)
  #   print(repr(res.best_f), struct.pack('>d', res.best_f).hex())
  #   "
  #   -> best_f = 9077.0 (exact -- TSP tour length is always an nint-rounded
  #   integer sum), big-endian hex 40c1ba8000000000 (struct.pack('>d', ...),
  #   matching f64_bits_hex()'s own little-endian-raw-then-reversed =
  #   big-endian output convention -- see test-mo.R's identical note).
  #
  # `best_f` is read from the SCRIPT's own printed field (not re-run
  # in-process): 9077 has an EXACT double representation (a whole number
  # well within f64's exactly-representable integer range), so
  # `as.numeric("9077")` is bit-identical to the script's own `res$best_f`
  # -- no precision is lost by going through the printed decimal, unlike a
  # value with a fractional part would risk.
  root <- .tsp_two_opt_repo_root()
  f <- .tsp_two_opt_run_fields(root)
  expect_identical(f64_bits_hex(as.numeric(f$best_f)), "40c1ba8000000000")
})
