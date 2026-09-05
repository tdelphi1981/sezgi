# M3-5 Task 6: sz_algorithm/sz_algo_solve -- the pure-R algorithm-authoring
# surface over T5's generalized EvalSession ask/tell core -- and
# sz_bias_structural_positions, the R mirror of py-sezgi's M3-4
# sezgi.Algorithm / sezgi.bias.structural_positions story.
#
# Mirrors py-sezgi/tests/test_algo.py's semantics 1:1 where applicable
# (RandomSearch subclass -> rs_setup/rs_step closures; SolveResult fields;
# BudgetExhausted -> "sz_budget_exhausted" condition class; the
# no-progress-step / zero-evaluations RuntimeError guards; the
# structural_positions uniform-vs-clustered fixtures).

rs_setup <- function(ctx) ctx$evaluate(t(replicate(10, ctx$random_point())))
rs_step  <- function(ctx) ctx$evaluate(t(replicate(10, ctx$random_point())))
rs <- sz_algorithm(rs_setup, rs_step, name = "r-random-search")

test_that("sz_algo_solve exhausts budget and returns the result shape", {
  s <- sz_eval_session(fid = 1, dim = 5, instance = 1, budget = 200)
  res <- sz_algo_solve(rs, s, seed = 42)
  expect_identical(res$evals_used, 200)
  expect_identical(res$algo, "r-random-search")
  expect_identical(res$seed, 42)
  expect_identical(res$budget, 200)
  expect_true(is.numeric(res$f_opt))
  expect_equal(res$gap, res$best_f - res$f_opt)
  expect_length(res$best_x, 5)
})

test_that("sz_algo_solve is deterministic", {
  s1 <- sz_eval_session(fid = 1, dim = 5, instance = 1, budget = 100)
  a <- sz_algo_solve(rs, s1, seed = 7)
  s2 <- sz_eval_session(fid = 1, dim = 5, instance = 1, budget = 100)
  b <- sz_algo_solve(rs, s2, seed = 7)
  expect_identical(a$best_f, b$best_f)
  expect_identical(a$best_x, b$best_x)
  expect_identical(a$evals_used, b$evals_used)
})

test_that("partial final batch stops cleanly", {
  # budget 95, batch 10: 9 full batches (90), the 10th batch would exceed
  # the remaining budget (5) -- driver catches sz_budget_exhausted and
  # stops; the last 5 evals are never spent.
  s <- sz_eval_session(fid = 1, dim = 5, instance = 1, budget = 95)
  res <- sz_algo_solve(rs, s, seed = 3)
  expect_identical(res$evals_used, 90)
})

test_that("no-progress step errors", {
  lazy <- sz_algorithm(
    setup = function(ctx) ctx$evaluate(matrix(ctx$random_point(), nrow = 1)),
    step  = function(ctx) invisible(NULL),  # consumes nothing
    name = "lazy"
  )
  s <- sz_eval_session(fid = 1, dim = 2, instance = 1, budget = 10)
  expect_error(sz_algo_solve(lazy, s, seed = 1), "consumed no budget")
})

test_that("a run that never evaluates anything errors", {
  never <- sz_algorithm(
    setup = function(ctx) invisible(NULL),
    step  = function(ctx) invisible(NULL),
    name = "never"
  )
  s <- sz_eval_session(fid = 1, dim = 2, instance = 1, budget = 10)
  expect_error(sz_algo_solve(never, s, seed = 1))
})

test_that("cec2022 session drives the same algorithm and reports a real gap", {
  s <- sz_eval_session_cec2022(fid = 3, dim = 10, budget = 100)
  res <- sz_algo_solve(rs, s, seed = 1)
  expect_identical(res$f_opt, sz_cec2022_f_star(3))
  expect_true(res$gap >= 0.0)
})

test_that("f0 session drives the same algorithm with f_opt/gap NULL", {
  s <- sz_eval_session_f0(dim = 3, f0_seed = 0, budget = 60)
  res <- sz_algo_solve(rs, s, seed = 1)
  expect_null(res$f_opt)
  expect_null(res$gap)
  expect_length(res$best_x, 3)
})

test_that("the session is finished exactly once, even after sz_algo_solve returns", {
  s <- sz_eval_session(fid = 1, dim = 5, instance = 1, budget = 50)
  sz_algo_solve(rs, s, seed = 1)
  expect_error(s$evals_used(), "session finished")
})

test_that("the session is finished even when solve() errors out", {
  lazy <- sz_algorithm(
    setup = function(ctx) ctx$evaluate(matrix(ctx$random_point(), nrow = 1)),
    step  = function(ctx) invisible(NULL),
    name = "lazy"
  )
  s <- sz_eval_session(fid = 1, dim = 2, instance = 1, budget = 10)
  expect_error(sz_algo_solve(lazy, s, seed = 1), "consumed no budget")
  expect_error(s$evals_used(), "session finished")
})

# ---- sz_bias_structural_positions -------------------------------------

test_that("structural_positions verdicts", {
  set.seed(0)
  u <- matrix(runif(90), nrow = 30)
  out_u <- sz_bias_structural_positions(u)
  expect_named(out_u, c(
    "per_dim_ks", "per_dim_ad", "holm_rejections_ks", "holm_rejections_ad",
    "verdict", "detail", "final_positions"
  ))
  # ANCHORED (per this project's convention): measured at seed = 0. Raw
  # per-dim KS p-values = [0.891, 0.524, 0.516], AD p-values = [0.850,
  # 0.387, 0.473] -- nowhere near alpha = 0.01, not a boundary case:
  # holm_rejections_ks = 0, holm_rejections_ad = 0.
  expect_identical(out_u$verdict, "no_evidence")

  cl <- matrix(0.5 + rnorm(90, 0, 1e-6), nrow = 30)
  out_cl <- sz_bias_structural_positions(cl)
  # ANCHORED: measured at this exact config (continued RNG stream). Raw
  # per-dim KS p-values are all ~2.81e-07, AD p-values ~1.03e-07 -- tight
  # clustering around 0.5 (sigma = 1e-6) is unambiguously non-uniform,
  # nowhere near a boundary.
  expect_identical(out_cl$verdict, "evidence")

  expect_error(sz_bias_structural_positions(u[1:4, ]))
})

test_that("sz_bias_structural_positions rejects ragged list input", {
  ragged <- list(c(0.5, 0.5), c(0.5), c(0.5, 0.5), c(0.5, 0.5), c(0.5, 0.5))
  expect_error(sz_bias_structural_positions(ragged))
})

test_that("sz_bias_structural_positions accepts a list of numeric vectors", {
  set.seed(1)
  pts <- lapply(1:30, function(i) runif(3))
  out <- sz_bias_structural_positions(pts)
  expect_length(out$final_positions, 30)
  expect_true(all(vapply(out$final_positions, length, integer(1)) == 3))
})

test_that("end-to-end bias scan of an R-authored algorithm on f0", {
  positions <- vector("list", 30)
  for (run in 1:30) {
    s <- sz_eval_session_f0(dim = 3, f0_seed = as.double(run), budget = 60)
    res <- sz_algo_solve(rs, s, seed = run)
    positions[[run]] <- res$best_x
  }
  m <- do.call(rbind, positions)
  out <- sz_bias_structural_positions(m)
  # ANCHORED: measured at this exact config (dim=3, budget=60, f0_seed and
  # engine seed both 1..30). Raw per-dim KS p-values = [0.750, 0.384,
  # 0.421], AD p-values = [0.855, 0.415, 0.254] -- nowhere near alpha =
  # 0.01, consistent with rs's uniform per-batch resampling having no
  # directional operator to induce structural bias against f0's own random
  # landscape.
  expect_identical(out$verdict, "no_evidence")
})

# ---- M3-5 Task 7: R OOP twin parity (examples/r/oop/gwo.R) ------------

#' Locates the repo root by walking up from `start` looking for
#' `examples/r/gwo.R` -- avoids depending on exactly which working
#' directory `testthat::test_local()`/`test_dir()`/`R CMD check` leaves the
#' process in (they differ), mirroring what
#' `py-sezgi/tests/test_examples_oop_parity.py` gets for free from
#' `pathlib.Path(__file__).resolve().parents[2]` (R's `testthat::test_path()`
#' is the closest analogue to `__file__`, used as the primary probe here).
.gwo_repo_root <- function() {
  start_points <- getwd()
  test_path_dir <- tryCatch(dirname(testthat::test_path()), error = function(e) NA_character_)
  if (!is.na(test_path_dir)) start_points <- c(test_path_dir, start_points)

  for (start in start_points) {
    dir <- normalizePath(start, mustWork = FALSE)
    for (i in 0:6) {
      if (file.exists(file.path(dir, "examples", "r", "gwo.R"))) {
        return(dir)
      }
      parent <- dirname(dir)
      if (identical(parent, dir)) break
      dir <- parent
    }
  }
  skip("example scripts not available (not in a repo checkout)")
}

test_that("R OOP gwo twin matches the pure gwo.R script (subprocess, string-exact)", {
  root <- .gwo_repo_root()

  run_fields <- function(script) {
    out <- system2("Rscript", args = script, stdout = TRUE, stderr = TRUE)
    line <- grep("evals_used=", out, value = TRUE)
    expect_length(line, 1)
    m <- regmatches(line, regexec("evals_used=(\\S+) best_f=(\\S+) gap=(\\S+)", line))[[1]]
    expect_length(m, 4)
    m[2:4]
  }

  pure <- run_fields(file.path(root, "examples", "r", "gwo.R"))
  oop <- run_fields(file.path(root, "examples", "r", "oop", "gwo.R"))
  expect_identical(oop, pure)
})

# ---- M3-8 Task 8: permutation-typed ctx surface (kind/n/random_permutation/
# two_opt) -- the R mirror of py-sezgi's M3-8 Task 7 AlgoContext widening ---

test_that("ctx$kind()/ctx$n() report float for a BBOB session, permutation for a TSP session", {
  s_float <- sz_eval_session(fid = 1, dim = 5, instance = 1, budget = 10)
  ctx_float <- .sz_algo_context(s_float)
  expect_identical(ctx_float$kind(), "float")
  expect_identical(ctx_float$n(), ctx_float$dim())
  s_float$finish()

  s_perm <- sz_eval_session_tsp("berlin52", budget = 10, seed = 1)
  ctx_perm <- .sz_algo_context(s_perm)
  expect_identical(ctx_perm$kind(), "permutation")
  expect_identical(ctx_perm$n(), 52)
  expect_identical(ctx_perm$n(), ctx_perm$dim())
  s_perm$finish()
})

test_that("ctx$random_point() errors clearly for a permutation-typed context", {
  s <- sz_eval_session_tsp("berlin52", budget = 10, seed = 1)
  ctx <- .sz_algo_context(s)
  expect_error(ctx$random_point(), "permutation")
  s$finish()
})

test_that("ctx$random_permutation() errors clearly for a float-typed context", {
  s <- sz_eval_session(fid = 1, dim = 5, instance = 1, budget = 10)
  ctx <- .sz_algo_context(s)
  expect_error(ctx$random_permutation(), "float")
  s$finish()
})

test_that("ctx$random_permutation() delegates to the session (valid 1-based tour)", {
  s <- sz_eval_session_tsp("berlin52", budget = 10, seed = 1)
  ctx <- .sz_algo_context(s)
  t <- ctx$random_permutation()
  expect_length(t, 52)
  expect_identical(sort(t), as.double(1:52))
  s$finish()
})

# ---- .sz_two_opt() / ctx$two_opt() -----------------------------------------

test_that("two_opt reverses the inclusive [i, j] segment", {
  tour <- 1:6
  expect_identical(.sz_two_opt(tour, 2, 4), c(1L, 4L, 3L, 2L, 5L, 6L))
})

test_that("two_opt: i == j is a no-op (single-element segment)", {
  tour <- 1:6
  expect_identical(.sz_two_opt(tour, 3, 3), tour)
})

test_that("two_opt: i == 1, j == length(tour) reverses the whole tour", {
  tour <- 1:6
  expect_identical(.sz_two_opt(tour, 1, 6), rev(tour))
})

test_that("two_opt: adjacent positions is a plain two-element swap", {
  tour <- 1:6
  expect_identical(.sz_two_opt(tour, 3, 4), c(1L, 2L, 4L, 3L, 5L, 6L))
})

test_that("two_opt: leading segment (i == 1)", {
  tour <- 1:6
  expect_identical(.sz_two_opt(tour, 1, 3), c(3L, 2L, 1L, 4L, 5L, 6L))
})

test_that("two_opt: trailing segment (j == length(tour))", {
  tour <- 1:6
  expect_identical(.sz_two_opt(tour, 4, 6), c(1L, 2L, 3L, 6L, 5L, 4L))
})

test_that("two_opt rejects i > j", {
  tour <- 1:6
  expect_error(.sz_two_opt(tour, 4, 2), "out of range")
})

test_that("two_opt rejects out-of-range indices", {
  tour <- 1:6
  expect_error(.sz_two_opt(tour, 0, 3), "out of range")
  expect_error(.sz_two_opt(tour, 1, 7), "out of range")
})

test_that("two_opt applied twice with the same (i, j) is the identity (reversal is its own inverse)", {
  tour <- 1:6
  once <- .sz_two_opt(tour, 2, 5)
  twice <- .sz_two_opt(once, 2, 5)
  expect_identical(twice, tour)
})

test_that("two_opt does not mutate its argument", {
  tour <- 1:6
  original <- tour
  .sz_two_opt(tour, 2, 4)
  expect_identical(tour, original)
})

# ---- End-to-end: sz_algorithm/sz_algo_solve over a permutation-typed session

test_that("a permutation-typed algorithm runs end-to-end via sz_algo_solve", {
  algo <- sz_algorithm(
    setup = function(ctx) {
      tour <- ctx$random_permutation()
      ctx$evaluate(matrix(tour, nrow = 1))
    },
    step = function(ctx) {
      tour <- ctx$random_permutation()
      ctx$evaluate(matrix(tour, nrow = 1))
    },
    name = "tsp-random-search"
  )
  s <- sz_eval_session_tsp("berlin52", budget = 20, seed = 3)
  res <- sz_algo_solve(algo, s, seed = 3)
  expect_identical(res$evals_used, 20)
  expect_identical(res$f_opt, 7542.0)
  expect_gte(res$gap, 0.0)
  expect_length(res$best_x, 52)
  expect_identical(sort(res$best_x), as.double(1:52))
})
