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

test_that("all 12 preset builders return parseable spec JSON", {
  skip_if_not_installed("jsonlite")
  specs <- list(
    sz_preset_de_rand_1(10, 500), sz_preset_de_best_1(10, 500),
    sz_preset_jde(10, 500), sz_preset_ga_real(10, 500),
    sz_preset_pso(10, 500), sz_preset_sa(500),
    sz_preset_shade(10, 500), sz_preset_lshade(5, 500),
    sz_preset_cmaes(10, 500), sz_preset_cmaes_ipop(5, 500),
    sz_preset_nelder_mead(5, 500), sz_preset_random_search(10, 500)
  )
  for (s in specs) expect_silent(jsonlite::fromJSON(s))
})
