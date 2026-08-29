# M2d-3 Task 4: R EvalSession (ask/tell) binding.
#
# Mirrors crates/bench/src/session.rs's own unit tests through the FFI
# (counting/budget/best), plus a pure-R random-search proof that an
# externally-driven session produces a parseable, bit-identical IOH archive
# -- 1:1 mirror of py-sezgi's Task 3 tests (`py-sezgi/tests/test_session.py`).
#
# `sz_eval_session()$evaluate(x)` accepts EITHER a numeric matrix (rows =
# points, `dim` columns) OR a `list` of `dim`-length numeric vectors; both
# shapes are exercised below (`points_matrix()` and `points_list()`).

points_matrix <- function(...) {
  # Each `...` arg is one point (a numeric vector); rbind gives rows = points.
  do.call(rbind, list(...))
}

points_list <- function(...) {
  list(...)
}

test_that("counting across two batches (matrix input)", {
  s <- sz_eval_session(fid = 1, dim = 3, instance = 1, budget = 100)
  s$evaluate(points_matrix(c(0, 0, 0), c(1, 1, 1), c(2, 2, 2)))
  expect_equal(s$evals_used(), 3)
  s$evaluate(points_matrix(c(3, 3, 3), c(4, 4, 4)))
  expect_equal(s$evals_used(), 5)
  expect_equal(s$budget(), 100)
})

test_that("counting across two batches (list input)", {
  s <- sz_eval_session(fid = 1, dim = 3, instance = 1, budget = 100)
  s$evaluate(points_list(c(0, 0, 0), c(1, 1, 1), c(2, 2, 2)))
  expect_equal(s$evals_used(), 3)
  s$evaluate(points_list(c(3, 3, 3), c(4, 4, 4)))
  expect_equal(s$evals_used(), 5)
  expect_equal(s$budget(), 100)
})

test_that("matrix and list input produce identical f values for the same points", {
  pts <- list(c(0.5, 0.5), c(-1.0, 2.0), c(3.0, -3.0))

  s1 <- sz_eval_session(fid = 1, dim = 2, instance = 1, budget = 100)
  fs_matrix <- s1$evaluate(do.call(rbind, pts))

  s2 <- sz_eval_session(fid = 1, dim = 2, instance = 1, budget = 100)
  fs_list <- s2$evaluate(pts)

  expect_identical(fs_matrix, fs_list)
})

test_that("a batch crossing the budget errors and does not count", {
  s <- sz_eval_session(fid = 1, dim = 3, instance = 1, budget = 2)
  expect_error(s$evaluate(points_matrix(c(0, 0, 0), c(1, 1, 1), c(2, 2, 2))))
  expect_equal(s$evals_used(), 0)

  # A batch that fits exactly still succeeds and counts fully.
  s$evaluate(points_matrix(c(0, 0, 0), c(1, 1, 1)))
  expect_equal(s$evals_used(), 2)
})

test_that("best tracking matches a hand-tracked minimum", {
  s <- sz_eval_session(fid = 1, dim = 2, instance = 1, budget = 100)
  xs <- list(c(0.5, 0.5), c(-1.0, 2.0), c(3.0, -3.0), c(0.1, 0.1))
  fs <- s$evaluate(do.call(rbind, xs))

  idx <- which.min(fs)
  best <- s$best()
  expect_equal(best$f, fs[idx])
  expect_equal(best$x, xs[[idx]])
})

test_that("f_opt is a numeric scalar and best() is NULL before any evaluation", {
  s <- sz_eval_session(fid = 1, dim = 2, instance = 1, budget = 10)
  expect_true(is.numeric(s$f_opt()))
  expect_null(s$best())
})

test_that("dimension mismatch raises and does not count", {
  s <- sz_eval_session(fid = 1, dim = 3, instance = 1, budget = 100)
  expect_error(s$evaluate(points_list(c(0, 0, 0), c(1, 1))), "row")
  expect_equal(s$evals_used(), 0)
})

test_that("a data.frame is rejected instead of silently misread column-wise", {
  # A data.frame is ALSO a VECSXP (R's `list` representation), one element
  # per COLUMN -- so it would otherwise slip through the list-of-points
  # branch and be read column-wise, producing plausible-looking but wrong
  # values with no warning (regression fixture for a review finding).
  s <- sz_eval_session(fid = 1, dim = 2, instance = 1, budget = 100)
  df <- data.frame(a = c(0, 10), b = c(5, -5))
  expect_error(s$evaluate(df), "data.frame")
  expect_equal(s$evals_used(), 0)

  # The fix: as.matrix(df) (rows = points) is accepted normally.
  fs <- s$evaluate(as.matrix(df))
  expect_length(fs, 2)
  expect_equal(s$evals_used(), 2)
})

test_that("a non-finite row raises and does not count", {
  s <- sz_eval_session(fid = 1, dim = 2, instance = 1, budget = 100)
  for (bad in c(NaN, Inf, -Inf)) {
    expect_error(s$evaluate(points_list(c(0, 0), c(bad, 0))), "row")
    expect_equal(s$evals_used(), 0)
  }
})

test_that("evaluate after finish raises", {
  s <- sz_eval_session(fid = 1, dim = 2, instance = 1, budget = 10)
  s$evaluate(points_list(c(0, 0)))
  s$finish()
  expect_error(s$evaluate(points_list(c(1, 1))), "session finished")
})

test_that("any method after finish raises", {
  s <- sz_eval_session(fid = 1, dim = 2, instance = 1, budget = 10)
  s$finish()
  expect_error(s$evals_used(), "session finished")
  expect_error(s$budget(), "session finished")
  expect_error(s$best(), "session finished")
  expect_error(s$f_opt(), "session finished")
  expect_error(s$finish(), "session finished")
})

test_that("finish without logging is ok", {
  s <- sz_eval_session(fid = 1, dim = 2, instance = 1, budget = 10)
  s$evaluate(points_list(c(0, 0)))
  expect_no_error(s$finish())
})

test_that("pure-R random search produces a parseable, bit-identical IOH archive", {
  # The engine-inside-out proof: a base-R-only random search drives the
  # session's ask/tell loop with no Rust algorithm involved, and the
  # resulting IOH archive round-trips through sz_read_ioh_records() with a
  # best_f that is bit-identical to the script's own independently tracked
  # best.
  log_dir <- file.path(tempfile(), "log")
  dim <- 5
  budget <- 200
  seed <- 42

  s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget,
                        log_dir = log_dir, algo_name = "rrs", seed = seed)

  set.seed(seed)
  tracked_best <- NA_real_
  for (i in seq_len(budget)) {
    x <- matrix(runif(dim, -5.0, 5.0), nrow = 1)
    f <- s$evaluate(x)[1]
    if (is.na(tracked_best) || f < tracked_best) {
      tracked_best <- f
    }
  }
  s$finish()

  records <- sz_read_ioh_records(log_dir, 200)
  expect_equal(nrow(records), 1)
  expect_equal(records$seed, seed)
  expect_identical(records$best_f, tracked_best)
})
