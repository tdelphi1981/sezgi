# M3-8 Task 11: determinism/anchored-output + cross-language hex-parity gate
# for examples/r/onemax_ga.R -- the R twin of examples/python/onemax_ga.py.
#
# Mirrors py-sezgi/tests/test_onemax_ga_example.py's "run the script as a
# subprocess, parse its printed metrics line" pattern, PLUS a cross-language
# hex anchor in the house convention (test-mo.R's wfg4_nsga2 pair anchor,
# test-tsp-two-opt.R's tsp_two_opt pair anchor, test-cross-language.R's
# golden-hex pattern): the deterministic best_f as an exact f64, compared as
# BYTES via writeBin, never a decimal literal.

#' Locates the repo root by walking up from `start` looking for
#' `examples/r/onemax_ga.R` -- same approach as test-tsp-two-opt.R's
#' `.tsp_two_opt_repo_root()` (not reused directly: each testthat file here
#' is self-contained, matching this file's own local-helper convention).
.onemax_ga_repo_root <- function() {
  marker <- file.path("examples", "r", "onemax_ga.R")
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
  stop("could not locate repo root (examples/r/onemax_ga.R not found walking up from getwd()/test_path())")
}

f64_bits_hex <- function(x) {
  raw <- writeBin(x, raw(), size = 8L, endian = "little")
  paste0(rev(sprintf("%02x", as.integer(raw))), collapse = "")
}

.onemax_ga_run_fields <- function(root) {
  script <- file.path(root, "examples", "r", "onemax_ga.R")
  out <- system2("Rscript", args = script, stdout = TRUE, stderr = TRUE)
  line <- grep("evals_used=", out, value = TRUE)
  expect_length(line, 1)
  m <- regmatches(line, regexec(
    "evals_used=(\\S+) best_f=(\\S+) gap=(\\S+) ones=(\\S+)", line
  ))[[1]]
  expect_length(m, 5)
  stats::setNames(as.list(m[2:5]), c("evals_used", "best_f", "gap", "ones"))
}

test_that("R onemax_ga example runs and reports valid metrics", {
  root <- .onemax_ga_repo_root()
  f <- .onemax_ga_run_fields(root)
  expect_identical(as.integer(f$evals_used), 400L)
  expect_identical(f$best_f, f$gap) # OneMax's known optimum is 0
  expect_true(as.integer(f$ones) >= 0 && as.integer(f$ones) <= 100)
})

test_that("R onemax_ga example is deterministic across runs", {
  root <- .onemax_ga_repo_root()
  a <- .onemax_ga_run_fields(root)
  b <- .onemax_ga_run_fields(root)
  expect_identical(a, b)
})

test_that("R onemax_ga example anchored output matches the Python twin", {
  # ANCHORED (per this project's convention): measured with N_BITS=100,
  # POP_SIZE=20, BUDGET=400, SEED=7 (onemax_ga.R's own constants --
  # identical to onemax_ga.py's). The printed evals_used/best_f/gap/ones
  # fields are expected to reproduce py-sezgi's own anchored values
  # (py-sezgi/tests/test_onemax_ga_example.py::
  # test_onemax_ga_example_anchored_output) EXACTLY.
  root <- .onemax_ga_repo_root()
  f <- .onemax_ga_run_fields(root)
  expect_identical(f$evals_used, "400")
  expect_identical(f$best_f, "16")
  expect_identical(f$gap, "16")
  expect_identical(f$ones, "84")
})

test_that("R onemax_ga best_f is bit-identical to the Python twin (cross-language hex anchor)", {
  # Cross-language anchor (house precedent: the wfg4_nsga2/tsp_two_opt
  # pairs' hex anchors). Golden hex derived by running the Python side
  # directly:
  #
  #   ./py-sezgi/.venv/bin/python -c "
  #   import struct
  #   import sezgi
  #   p = sezgi.problems.onemax(100)
  #   spec = sezgi.presets.ga_bin(20, 400)
  #   r = sezgi.solve(spec, p, master_seed=7, run_id=0)
  #   print(repr(r['best_f']), struct.pack('>d', r['best_f']).hex())
  #   "
  #   -> best_f = 16.0 (exact -- OneMax's fitness is always an integer
  #   zero-bit count, well within f64's exactly-representable integer
  #   range), big-endian hex 4030000000000000 (struct.pack('>d', ...),
  #   matching f64_bits_hex()'s own little-endian-raw-then-reversed =
  #   big-endian output convention -- see test-mo.R's identical note).
  #
  # `best_f` is read from the SCRIPT's own printed field (not re-run
  # in-process): 16 has an EXACT double representation, so
  # `as.numeric("16")` is bit-identical to the script's own `result$best_f`
  # -- no precision is lost by going through the printed decimal.
  root <- .onemax_ga_repo_root()
  f <- .onemax_ga_run_fields(root)
  expect_identical(f64_bits_hex(as.numeric(f$best_f)), "4030000000000000")
})
