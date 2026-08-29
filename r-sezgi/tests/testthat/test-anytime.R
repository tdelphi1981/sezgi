# M2d-2 Task 8: R bindings for the analysis cluster (IOH reader, ECDF/
# anytime curves, COCO export, Bayesian Plackett-Luce). 1:1 mirror of
# py-sezgi's Task 7 tests (`py-sezgi/tests/test_anytime.py`), adapted to R's
# data.frame/named-list conventions -- see the M2d-2 Task 8 brief's "MANDATORY
# 1:1 mirror" rule.

tiny_experiment_toml <- function(budgets = "[500]") {
  sprintf('
    name = "tiny-logged"
    seeds = [1, 2]
    budgets = %s

    [[algorithms]]
    name = "de"
    preset = { kind = "de_rand_1", pop_size = 8 }

    [[problems]]
    suite = "bbob"
    fid = 1
    dim = 5
    instances = [1]
  ', budgets)
}

# 2 algorithms x 2 instances -- friedman (inside per_budget_packages) requires
# at least 2 problems (rows) and 2 algorithms (columns).
multi_row_experiment_toml <- function(budgets = "[500]") {
  sprintf('
    name = "multi-row-logged"
    seeds = [1, 2]
    budgets = %s

    [[algorithms]]
    name = "de"
    preset = { kind = "de_rand_1", pop_size = 8 }

    [[algorithms]]
    name = "rs"
    preset = { kind = "random_search", pop_size = 8 }

    [[problems]]
    suite = "bbob"
    fid = 1
    dim = 5
    instances = [1, 2]
  ', budgets)
}

# ---------------------------------------------------------------------
# sz_run_experiment(log_dir = ...) -> sz_read_ioh_records() round trip
# ---------------------------------------------------------------------

test_that("log_dir round trip is bit-identical at the full logged budget", {
  # CORE test: a SINGLE-budget experiment (per the curtailed-view ruling in
  # `sezgi_bench::ioh_records`'s doc comment -- a reconstructed view at a
  # budget SMALLER than the run's logged budget is a curtailed view of the
  # same trajectory, not guaranteed to equal an independent run planned at
  # that smaller budget), so this only compares at the run's one, full,
  # logged budget.
  log_dir <- file.path(tempfile(), "logs")
  spec_toml <- multi_row_experiment_toml(budgets = "[500]")

  in_memory <- sz_run_experiment(spec_toml, log_dir = log_dir, parallel = FALSE)
  expect_equal(nrow(in_memory), 8)  # 2 algos x 1 problem x 2 instances x 2 seeds

  disk <- sz_read_ioh_records(log_dir, 500)
  expect_equal(nrow(disk), 8)

  key <- function(d) paste(d$algo, d$fid, d$dim, d$instance, d$seed, d$budget, sep = "|")
  in_memory <- in_memory[order(key(in_memory)), ]
  disk <- disk[order(key(disk)), ]
  expect_identical(key(in_memory), key(disk))

  expect_identical(in_memory$best_f, disk$best_f)
  expect_identical(in_memory$f_opt, disk$f_opt)
  expect_identical(in_memory$evals, disk$evals)

  # sz_per_budget_packages accepts the disk-reconstructed data.frame
  # unchanged (same column shape as sz_run_experiment) -- the "prove
  # sz_per_budget_packages(records) just works" requirement.
  pkgs <- sz_per_budget_packages(disk)
  expect_equal(names(pkgs), "500")
  expect_false(grepl("NaN", pkgs[["500"]]$latex_summary, fixed = TRUE))
})

test_that("a curtailed-budget view is present with evals capped at the budget", {
  # A budget SMALLER than the run's logged budget is a valid curtailed view
  # (present, no error), with evals = min(budget, run.evals) -- but is NOT
  # asserted equal to anything from an independent smaller-budget run (see
  # the doc comment on `ioh_records`).
  log_dir <- file.path(tempfile(), "logs")
  sz_run_experiment(tiny_experiment_toml(budgets = "[500]"), log_dir = log_dir, parallel = FALSE)

  curtailed <- sz_read_ioh_records(log_dir, 200)
  expect_equal(nrow(curtailed), 2)
  expect_true(all(curtailed$budget == 200))
  expect_true(all(curtailed$evals == 200))
})

# ---------------------------------------------------------------------
# sz_ecdf
# ---------------------------------------------------------------------

test_that("sz_ecdf per_algo curves are monotone and in [0, 1]", {
  log_dir <- file.path(tempfile(), "logs")
  sz_run_experiment(tiny_experiment_toml(), log_dir = log_dir, parallel = FALSE)

  curves <- sz_ecdf(log_dir)
  expect_type(curves, "list")
  expect_named(curves, "de")

  curve <- curves[["de"]]
  expect_named(curve, c("evals", "proportion"))
  expect_equal(length(curve$evals), length(curve$proportion))
  expect_identical(curve$evals, sort(curve$evals))
  expect_identical(curve$proportion, sort(curve$proportion))
  expect_true(all(curve$proportion >= 0.0 & curve$proportion <= 1.0))
})

test_that("sz_ecdf per_algo = FALSE returns a single pooled curve", {
  log_dir <- file.path(tempfile(), "logs")
  sz_run_experiment(tiny_experiment_toml(), log_dir = log_dir, parallel = FALSE)

  curve <- sz_ecdf(log_dir, per_algo = FALSE)
  expect_named(curve, c("evals", "proportion"))
  expect_identical(curve$evals, sort(curve$evals))
  expect_identical(curve$proportion, sort(curve$proportion))
})

test_that("sz_ecdf accepts custom targets", {
  log_dir <- file.path(tempfile(), "logs")
  sz_run_experiment(tiny_experiment_toml(), log_dir = log_dir, parallel = FALSE)

  curve <- sz_ecdf(log_dir, targets = c(10.0, 1.0, 0.1), per_algo = FALSE)
  expect_named(curve, c("evals", "proportion"))
  expect_true(all(curve$proportion >= 0.0 & curve$proportion <= 1.0))
})

# ---------------------------------------------------------------------
# sz_coco_export
# ---------------------------------------------------------------------

test_that("sz_coco_export writes an .info file with the correct first line", {
  log_dir <- file.path(tempfile(), "logs")
  sz_run_experiment(tiny_experiment_toml(), log_dir = log_dir, parallel = FALSE)

  out_dir <- file.path(tempfile(), "coco")
  written <- sz_coco_export(log_dir, out_dir)
  expect_true(length(written) > 0)
  expect_true(all(file.exists(written)))

  info_path <- file.path(out_dir, "de", "bbobexp_f1.info")
  expect_true(info_path %in% written)

  first_line <- readLines(info_path, n = 1)
  expect_identical(
    first_line,
    "suite = 'bbob', funcId = 1, DIM = 5, Precision = 1.000e-08, algId = 'de'"
  )

  dat_path <- file.path(out_dir, "de", "data_f1", "bbobexp_f1_DIM5.dat")
  tdat_path <- file.path(out_dir, "de", "data_f1", "bbobexp_f1_DIM5.tdat")
  expect_true(dat_path %in% written)
  expect_true(tdat_path %in% written)
})

# ---------------------------------------------------------------------
# sz_bayesian_plackett_luce
# ---------------------------------------------------------------------

test_that("sz_bayesian_plackett_luce shape and same-seed determinism", {
  rankings <- list(c(1, 2, 3), c(2, 1, 3), c(1, 3, 2), c(3, 2, 1))

  r1 <- sz_bayesian_plackett_luce(rankings, samples = 300, burn_in = 100, seed = 1)
  r2 <- sz_bayesian_plackett_luce(rankings, samples = 300, burn_in = 100, seed = 1)

  expect_named(r1, c("mean_worths", "ci_low", "ci_high", "p_best", "samples"))
  expect_length(r1$mean_worths, 3)
  expect_length(r1$ci_low, 3)
  expect_length(r1$ci_high, 3)
  expect_length(r1$p_best, 3)
  expect_equal(r1$samples, 300)

  # Same-seed determinism: exact list equality (bit-identical), not merely
  # close -- same convention as the Rust bayes_pl_deterministic_given_seed
  # test and py-sezgi's Task 7 test of the same name.
  expect_identical(r1, r2)

  expect_equal(sum(r1$p_best), 1.0, tolerance = 1e-9)
})

test_that("sz_bayesian_plackett_luce diverges under a different seed", {
  rankings <- list(c(1, 2, 3), c(2, 1, 3), c(1, 3, 2), c(3, 2, 1))
  r1 <- sz_bayesian_plackett_luce(rankings, samples = 300, burn_in = 100, seed = 1)
  r2 <- sz_bayesian_plackett_luce(rankings, samples = 300, burn_in = 100, seed = 2)
  expect_false(identical(r1$mean_worths, r2$mean_worths))
})

test_that("sz_bayesian_plackett_luce defaults are samples=2000, burn_in=500, seed=1", {
  rankings <- list(c(1, 2), c(1, 2), c(1, 2), c(2, 1))
  r <- sz_bayesian_plackett_luce(rankings)
  expect_equal(r$samples, 2000)
})

test_that("sz_bayesian_plackett_luce rejects invalid rankings", {
  expect_error(sz_bayesian_plackett_luce(list(c(1, 1, 2))))
})

# ---------------------------------------------------------------------
# checkpoint + log_dir resume must not re-log (T1 deferred homework, same
# mandatory requirement as py-sezgi's Task 7 test of the same purpose).
# ---------------------------------------------------------------------

tree_bytes <- function(root) {
  files <- sort(list.files(root, recursive = TRUE, full.names = TRUE))
  raw <- raw(0)
  for (f in files) {
    raw <- c(raw, readBin(f, what = "raw", n = file.info(f)$size))
  }
  raw
}

test_that("checkpoint + log_dir resume does not regrow the IOH tree", {
  spec_toml <- tiny_experiment_toml(budgets = "[500]")
  journal_path <- tempfile(fileext = ".jsonl")
  log_dir <- file.path(tempfile(), "logs")

  r1 <- sz_run_experiment(spec_toml, journal = journal_path, log_dir = log_dir, parallel = FALSE)
  expect_equal(nrow(r1), 2)
  expect_true(dir.exists(log_dir))

  first_run_bytes <- tree_bytes(log_dir)
  expect_true(length(first_run_bytes) > 0)
  lines_after_first <- length(readLines(journal_path))

  # Resume: same journal, same spec -- every run is already done, so nothing
  # should be re-executed, and hence nothing re-logged.
  r2 <- sz_run_experiment(spec_toml, journal = journal_path, log_dir = log_dir, parallel = FALSE)
  expect_equal(nrow(r2), 2)

  lines_after_second <- length(readLines(journal_path))
  expect_equal(lines_after_second, lines_after_first,
               info = "resume of a completed experiment must not append new journal lines")

  resumed_bytes <- tree_bytes(log_dir)
  expect_identical(resumed_bytes, first_run_bytes,
                    label = "resumed runs must never be re-logged -- the on-disk IOH tree must not grow")

  ord <- function(d) d[order(d$algo, d$seed), ]
  expect_identical(ord(r1)$best_f, ord(r2)$best_f)
})
