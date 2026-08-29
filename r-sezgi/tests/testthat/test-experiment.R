# Schema verified against `crates/bench/src/experiment.rs`'s `ExperimentSpec`
# deserialization structs: `name`, `seeds`, `budgets` are top-level fields (no
# `[experiment]` wrapper section); each `[[algorithms]]` entry needs a `name`
# plus either `preset = { kind = "...", pop_size = ... }` or `spec_toml =
# "..."`; each `[[problems]]` entry has `suite`, `fid` (singular), `dim`
# (singular), `instances` (array).
experiment_toml <- '
name = "r-smoke"
seeds = [1, 2]
budgets = [500]

[[algorithms]]
name = "de1"
preset = { kind = "de_rand_1", pop_size = 10 }

[[algorithms]]
name = "rs1"
preset = { kind = "random_search", pop_size = 10 }

[[problems]]
suite = "bbob"
fid = 1
dim = 5
instances = [1, 2, 3]
'

test_that("sz_run_experiment returns one row per run", {
  df <- sz_run_experiment(experiment_toml)
  expect_s3_class(df, "data.frame")
  expect_equal(nrow(df), 2 * 2 * 3 * 1)  # algos x seeds x instances x budgets
  expect_true(all(c("algo", "fid", "dim", "instance", "seed", "budget",
                    "best_f", "f_opt", "evals") %in% names(df)))
})

test_that("sz_run_experiment is deterministic across calls", {
  a <- sz_run_experiment(experiment_toml)
  b <- sz_run_experiment(experiment_toml, parallel = FALSE)
  ord <- function(d) d[order(d$algo, d$seed, d$instance), ]
  expect_identical(ord(a)$best_f, ord(b)$best_f)  # parallel == sequential, bitwise
})

test_that("journal resume does not change results", {
  j <- tempfile(fileext = ".jsonl")
  a <- sz_run_experiment(experiment_toml, journal = j)
  b <- sz_run_experiment(experiment_toml, journal = j)  # full resume, 0 re-runs
  expect_identical(a$best_f[order(a$seed)], b$best_f[order(b$seed)])
})

test_that("journal resume tolerates a formatting-only edit (M2d-1 Task 9)", {
  # The journal's spec hash is computed over the canonical TOML form, so
  # appending a comment (whitespace to the TOML parser) must not invalidate
  # an existing journal on resume.
  j <- tempfile(fileext = ".jsonl")
  a <- sz_run_experiment(experiment_toml, journal = j, parallel = FALSE)
  lines_after_first <- length(readLines(j))

  edited_toml <- paste0(experiment_toml, "\n# harmless comment\n")
  b <- sz_run_experiment(edited_toml, journal = j, parallel = FALSE)
  lines_after_second <- length(readLines(j))

  expect_equal(nrow(a), nrow(b))
  expect_equal(lines_after_first, lines_after_second,
               info = "resume with a formatting-only edit must not append new journal lines")
  ord <- function(d) d[order(d$algo, d$seed, d$instance), ]
  expect_identical(ord(a)$best_f, ord(b)$best_f)
})

test_that("journal resume rejects a real spec field change", {
  j <- tempfile(fileext = ".jsonl")
  sz_run_experiment(experiment_toml, journal = j, parallel = FALSE)

  changed_toml <- sub("budgets = [500]", "budgets = [600]", experiment_toml, fixed = TRUE)
  expect_false(identical(changed_toml, experiment_toml))

  expect_error(
    sz_run_experiment(changed_toml, journal = j, parallel = FALSE),
    regexp = "spec has changed"
  )
})

test_that("all 18 preset builders return parseable spec JSON", {
  skip_if_not_installed("jsonlite")
  specs <- list(
    sz_preset_de_rand_1(10, 500), sz_preset_de_best_1(10, 500),
    sz_preset_jde(10, 500), sz_preset_ga_real(10, 500),
    sz_preset_pso(10, 500), sz_preset_gwo(10, 500), sz_preset_woa(10, 500),
    sz_preset_harmony_search(10, 500), sz_preset_cuckoo_search(10, 500),
    sz_preset_goa(10, 500),
    sz_preset_sa(500),
    sz_preset_shade(10, 500), sz_preset_lshade(5, 500),
    sz_preset_cmaes(10, 500), sz_preset_cmaes_ipop(5, 500),
    sz_preset_nelder_mead(5, 500), sz_preset_random_search(10, 500),
    sz_preset_es_mu_plus_lambda(10, 500)
  )
  for (s in specs) expect_silent(jsonlite::fromJSON(s))
})

test_that("gwo preset solves bbob f1 (M2d-3 Task 5)", {
  spec <- sz_preset_gwo(30, 2000)
  r <- sz_solve_bbob(spec, fid = 1L, dim = 5L, instance = 1L,
                     master_seed = 1, run_id = 0)
  expect_true(is.finite(r$best_f))
  expect_true(r$evals > 0)
})

test_that("woa preset solves bbob f1 (M2d-3 Task 6)", {
  spec <- sz_preset_woa(30, 2000)
  r <- sz_solve_bbob(spec, fid = 1L, dim = 5L, instance = 1L,
                     master_seed = 1, run_id = 0)
  expect_true(is.finite(r$best_f))
  expect_true(r$evals > 0)
})

test_that("harmony_search preset solves bbob f1 (M2d-3 Task 7)", {
  spec <- sz_preset_harmony_search(30, 2000)
  r <- sz_solve_bbob(spec, fid = 1L, dim = 5L, instance = 1L,
                     master_seed = 1, run_id = 0)
  expect_true(is.finite(r$best_f))
  expect_true(r$evals > 0)
})

test_that("cuckoo_search preset solves bbob f1 (M2d-3 Task 8)", {
  spec <- sz_preset_cuckoo_search(25, 2000)
  r <- sz_solve_bbob(spec, fid = 1L, dim = 5L, instance = 1L,
                     master_seed = 1, run_id = 0)
  expect_true(is.finite(r$best_f))
  expect_true(r$evals > 0)
})

test_that("goa preset solves bbob f1 (M2d-3 Task 9)", {
  spec <- sz_preset_goa(30, 2000)
  r <- sz_solve_bbob(spec, fid = 1L, dim = 5L, instance = 1L,
                     master_seed = 1, run_id = 0)
  expect_true(is.finite(r$best_f))
  expect_true(r$evals > 0)
})

test_that("es_mu_plus_lambda default dist solves bbob f1 (M2d-1 Task 7)", {
  spec <- sz_preset_es_mu_plus_lambda(10, 500)
  r <- sz_solve_bbob(spec, fid = 1L, dim = 5L, instance = 1L,
                     master_seed = 1, run_id = 0)
  expect_true(is.finite(r$best_f))
  expect_true(r$evals > 0)
})

test_that("es_mu_plus_lambda rejects an unknown distribution", {
  expect_error(sz_preset_es_mu_plus_lambda(10, 500, dist = "banana"), "unknown distribution")
})

# M2d-1 Task 8: sz_results_matrix() / sz_per_budget_packages().
#
# 2 algos x f1 x 5 instances x 2 seeds x 2 budgets -- matches the README
# experiments quickstart and py-sezgi's reporting_grid_toml() fixture.
reporting_grid_toml <- '
name = "reporting-grid"
seeds = [1, 2]
budgets = [400, 800]

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
instances = [1, 2, 3, 4, 5]
'

test_that("sz_results_matrix has the right shape and labels", {
  df <- sz_run_experiment(reporting_grid_toml)
  rm <- sz_results_matrix(df, budget = 400)

  expect_equal(rm$algo_names, c("de", "rs"))
  expect_equal(rm$problem_labels, c("f1d5i1", "f1d5i2", "f1d5i3", "f1d5i4", "f1d5i5"))
  expect_equal(dim(rm$matrix), c(5, 2))  # rows = problems, cols = algorithms
})

test_that("sz_per_budget_packages returns ascending budgets with no NaN", {
  df <- sz_run_experiment(reporting_grid_toml)
  pkgs <- sz_per_budget_packages(df)

  expect_equal(length(pkgs), 2)
  expect_equal(names(pkgs), c("400", "800"))  # ascending budget order

  for (pkg in pkgs) {
    expect_false(grepl("NaN", pkg$latex_summary, fixed = TRUE))
    expect_setequal(names(pkg), c("friedman", "nemenyi_cd", "pairwise_wilcoxon_holm",
                                   "cliffs", "bayes", "plackett_luce",
                                   "latex_summary", "latex_tests"))
  }
})

test_that("sz_results_matrix errors on a missing cell", {
  df <- sz_run_experiment(reporting_grid_toml)
  incomplete <- df[!(df$algo == "rs" & df$instance == 3 & df$budget == 400), ]

  expect_error(sz_results_matrix(incomplete, budget = 400), "missing run record")
})

test_that("sz_results_matrix rejects a fractional budget (M2d-3 f64_to_u64 strictness)", {
  df <- sz_run_experiment(reporting_grid_toml)
  expect_error(sz_results_matrix(df, budget = 400.5), "expected a whole number")
})

test_that("sz_results_matrix and sz_per_budget_packages reject an unknown aggregate", {
  df <- sz_run_experiment(reporting_grid_toml)
  expect_error(sz_results_matrix(df, budget = 400, aggregate = "bogus"), "unknown aggregate")
  expect_error(sz_per_budget_packages(df, aggregate = "bogus"), "unknown aggregate")
})
