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
