# M4-2 Task 5: 26 table-driven preset wrapper classes + GeneticAlgorithm/
# DifferentialEvolution dispatchers + NSGA2 skin (`r-sezgi/R/builtins.R`).

# ---- helper problems --------------------------------------------------------

float_problem <- function(d = 5) {
  R6::R6Class("FloatProblem", inherit = Problem, public = list(
    d = d,
    space = function() sz_space(sz_float(-5.12, 5.12, self$d)),
    evaluate = function(x) 10 * self$d + sum(x^2 - 10 * cos(2 * pi * x)),
    optimum = function() 0
  ))$new()
}

int_problem <- function(n = 3) {
  R6::R6Class("IntProblem", inherit = Problem, public = list(
    n = n,
    space = function() sz_space(sz_int(0, 9, self$n)),
    evaluate = function(x) sum((x - 5)^2)
  ))$new()
}

cat_problem <- function(n = 3, k = 4) {
  R6::R6Class("CatProblem", inherit = Problem, public = list(
    n = n, k = k,
    space = function() sz_space(sz_categorical(self$k, self$n)),
    evaluate = function(x) sum(x != 0)
  ))$new()
}

bin_problem <- function(n = 8) {
  R6::R6Class("BinProblem", inherit = Problem, public = list(
    n = n,
    space = function() sz_space(sz_binary(self$n)),
    evaluate = function(x) as.double(sum(!x))
  ))$new()
}

perm_problem <- function(n = 5) {
  R6::R6Class("PermProblem", inherit = Problem, public = list(
    n = n,
    space = function() sz_space(sz_permutation(self$n)),
    evaluate = function(x) sum(abs(x - (seq_len(self$n) - 1)))
  ))$new()
}

mixed_problem <- function() {
  R6::R6Class("MixedProblem", inherit = Problem, public = list(
    space = function() sz_space(sz_float(-5, 5, 2), sz_int(0, 5, 2)),
    evaluate = function(x) sum(x[[1]]^2) + sum(x[[2]])
  ))$new()
}

# ---- reconciliation arithmetic (live-verified) -------------------------------

test_that("reconciliation: 26 table rows + GA(5 presets) + DE(3 presets) = 34 sz_preset_* builders", {
  n_table <- length(sezgi:::.sz_preset_table)
  expect_equal(n_table, 26L)

  n_ga_presets <- length(sezgi:::.sz_ga_representation_to_preset)
  expect_equal(n_ga_presets, 5L)

  n_de_presets <- length(sezgi:::.sz_de_variant_to_preset)
  expect_equal(n_de_presets, 3L)

  expect_equal(n_table + n_ga_presets + n_de_presets, 34L)
})

test_that("live count: exactly 34 exported sz_preset_* functions in the installed package", {
  exported <- getNamespaceExports("sezgi")
  preset_fns <- exported[grepl("^sz_preset_", exported)]
  expect_equal(length(preset_fns), 34L)
})

test_that("reconciliation: 26 table classes + GeneticAlgorithm + DifferentialEvolution = 28 preset-backed classes; +NSGA2 = 29 exported classes", {
  n_table <- length(sezgi:::.sz_preset_table)
  n_preset_backed <- n_table + 2L
  expect_equal(n_preset_backed, 28L)
  expect_equal(n_preset_backed + 1L, 29L)
})

# ---- NAMESPACE structural test: every table class + GA/DE/NSGA2 exported ----

test_that("every class in .sz_preset_table, plus GeneticAlgorithm/DifferentialEvolution/NSGA2, is exported", {
  table <- sezgi:::.sz_preset_table
  exported <- getNamespaceExports("sezgi")
  class_names <- vapply(table, function(row) row$class, character(1))
  all_names <- c(class_names, "GeneticAlgorithm", "DifferentialEvolution", "NSGA2")
  expect_length(all_names, 29L)
  for (nm in all_names) {
    expect_true(nm %in% exported, info = nm)
  }
})

# ---- table-driven bit-identity anchor: EVERY class vs. a raw sz_solve_bbob call

test_that("every table-driven class is bit-identical to sz_solve_bbob(sz_preset_X(...)) at the same seed/run_id", {
  problem <- sz_builtin_bbob(1, 5, 1)
  budget <- 300
  seed <- 7
  run_id <- 0
  table <- sezgi:::.sz_preset_table

  for (row in table) {
    cls <- get(row$class, envir = asNamespace("sezgi"))
    inst <- cls$new()
    preset_fn <- sezgi:::.sz_preset_fn(row$preset_name)

    expected_spec <- if (identical(row$mode, "pop_budget")) {
      preset_fn(inst$pop_size, budget)
    } else if (identical(row$mode, "budget_only")) {
      preset_fn(budget)
    } else {
      preset_fn(as.double(problem$dim), budget)
    }
    expected <- sz_solve_bbob(
      expected_spec,
      fid = as.integer(problem$fid), dim = as.integer(problem$dim),
      instance = as.integer(problem$instance), master_seed = seed, run_id = run_id
    )

    actual <- inst$run(problem, budget = budget, seed = seed, run_id = run_id)

    expect_identical(actual$best_f, expected$best_f, info = row$class)
    expect_identical(actual$best_x, expected$best_x, info = row$class)
    expect_identical(actual$evals_used, expected$evals, info = row$class)
    expect_identical(actual$algo, tolower(row$class), info = row$class)
    expect_null(actual$f_opt, info = row$class)
  }
})

# ---- non-default-kwarg anchors (a handful of classes) ------------------------

test_that("ArtificialBeeColony with a non-default pop_size is bit-identical to a direct sz_preset_abc(15, budget) call", {
  problem <- sz_builtin_bbob(1, 5, 1)
  budget <- 200
  expected_spec <- sz_preset_abc(15, budget)
  expected <- sz_solve_bbob(expected_spec, fid = 1L, dim = 5L, instance = 1L, master_seed = 4, run_id = 0)
  actual <- ArtificialBeeColony$new(pop_size = 15)$run(problem, budget = budget, seed = 4, run_id = 0)
  expect_identical(actual$best_f, expected$best_f)
  expect_identical(actual$best_x, expected$best_x)
})

test_that("EvolutionStrategy with non-default dist=/loc=/scale= kwargs is bit-identical to a direct call", {
  problem <- sz_builtin_bbob(1, 5, 1)
  budget <- 200
  expected_spec <- sz_preset_es_mu_plus_lambda(12, budget, dist = "cauchy", loc = 1, scale = 2)
  expected <- sz_solve_bbob(expected_spec, fid = 1L, dim = 5L, instance = 1L, master_seed = 9, run_id = 0)
  actual <- EvolutionStrategy$new(pop_size = 12, dist = "cauchy", loc = 1, scale = 2)$
    run(problem, budget = budget, seed = 9, run_id = 0)
  expect_identical(actual$best_f, expected$best_f)
  expect_identical(actual$best_x, expected$best_x)
})

test_that("ParticleSwarm with a non-default pop_size is bit-identical to a direct sz_preset_pso(15, budget) call", {
  problem <- sz_builtin_bbob(1, 5, 1)
  budget <- 200
  expected_spec <- sz_preset_pso(15, budget)
  expected <- sz_solve_bbob(expected_spec, fid = 1L, dim = 5L, instance = 1L, master_seed = 2, run_id = 1)
  actual <- ParticleSwarm$new(pop_size = 15)$run(problem, budget = budget, seed = 2, run_id = 1)
  expect_identical(actual$best_f, expected$best_f)
  expect_identical(actual$best_x, expected$best_x)
})

# ---- table-driven classes also run on the R-Problem form ---------------------

test_that("GreyWolfOptimizer runs on an R Problem subclass, bit-identical to a direct sz_solve_r_problem call, with f_opt/gap set", {
  prob <- float_problem(4)
  budget <- 150
  spec <- sz_preset_gwo(30, budget)
  blocks <- sezgi:::.sz_space_to_blocks(prob$space())
  evaluate_shim <- sezgi:::.sz_make_evaluate_shim(prob)
  expected <- sezgi:::sz_solve_r_problem(spec, blocks, evaluate_shim, master_seed = 3, run_id = 0)

  actual <- GreyWolfOptimizer$new()$run(float_problem(4), budget = budget, seed = 3, run_id = 0)
  expect_identical(actual$best_f, expected$best_f)
  expect_identical(actual$best_x, expected$best_x)
  expect_identical(actual$evals_used, expected$evals)
  expect_equal(actual$f_opt, 0)
  expect_equal(actual$gap, actual$best_f - 0)
})

# ---- dim_budget / budget_only pop_size rejection ------------------------------

test_that("dim_budget presets (CMAESIpop/LSHADE/NelderMead) reject an explicit pop_size", {
  expect_error(CMAESIpop$new(pop_size = 10), "pop_size")
  expect_error(LSHADE$new(pop_size = 10), "pop_size")
  expect_error(NelderMead$new(pop_size = 10), "pop_size")
})

test_that("dim_budget presets accept pop_size = NULL (the default) without error", {
  expect_silent(CMAESIpop$new())
  expect_silent(LSHADE$new())
  expect_silent(NelderMead$new())
})

test_that("SimulatedAnnealing (budget_only) rejects pop_size != 1 but accepts 1 or NULL", {
  expect_error(SimulatedAnnealing$new(pop_size = 5), "pop_size")
  expect_silent(SimulatedAnnealing$new(pop_size = 1))
  expect_silent(SimulatedAnnealing$new())
})

test_that("CMAESIpop/LSHADE/NelderMead derive pop_size from problem dim at run() time, bit-identical to a direct call", {
  problem <- sz_builtin_bbob(1, 6, 1)
  budget <- 250

  expected_lshade <- sz_solve_bbob(
    sz_preset_lshade(6, budget), fid = 1L, dim = 6L, instance = 1L, master_seed = 1, run_id = 0
  )
  actual_lshade <- LSHADE$new()$run(problem, budget = budget, seed = 1, run_id = 0)
  expect_identical(actual_lshade$best_f, expected_lshade$best_f)

  expected_cmaes_ipop <- sz_solve_bbob(
    sz_preset_cmaes_ipop(6, budget), fid = 1L, dim = 6L, instance = 1L, master_seed = 1, run_id = 0
  )
  actual_cmaes_ipop <- CMAESIpop$new()$run(problem, budget = budget, seed = 1, run_id = 0)
  expect_identical(actual_cmaes_ipop$best_f, expected_cmaes_ipop$best_f)

  expected_nm <- sz_solve_bbob(
    sz_preset_nelder_mead(6, budget), fid = 1L, dim = 6L, instance = 1L, master_seed = 1, run_id = 0
  )
  actual_nm <- NelderMead$new()$run(problem, budget = budget, seed = 1, run_id = 0)
  expect_identical(actual_nm$best_f, expected_nm$best_f)
})

# ---- sz_result shape --------------------------------------------------------

test_that("a table-driven class's run() returns an sz_result with exactly the 8 documented fields", {
  res <- RandomSearch$new()$run(sz_builtin_bbob(1, 4, 1), budget = 80, seed = 1)
  expect_s3_class(res, "sz_result")
  expect_named(res, c("algo", "seed", "budget", "evals_used", "best_x", "best_f", "f_opt", "gap"))
})

# ---- GeneticAlgorithm: space-kind auto-dispatch -------------------------------

test_that("GeneticAlgorithm dispatches to ga_real on a Float space (and on sz_builtin_bbob), bit-identical to a direct call", {
  problem <- sz_builtin_bbob(1, 5, 1)
  budget <- 150
  expected <- sz_solve_bbob(
    sz_preset_ga_real(20, budget), fid = 1L, dim = 5L, instance = 1L, master_seed = 1, run_id = 0
  )
  ga <- GeneticAlgorithm$new()
  actual <- ga$run(problem, budget = budget, seed = 1, run_id = 0)
  expect_identical(actual$best_f, expected$best_f)
  expect_identical(actual$best_x, expected$best_x)
  expect_equal(ga$dispatched_representation, "real")
})

test_that("GeneticAlgorithm dispatches to ga_perm on a Permutation space, bit-identical to a direct call", {
  prob <- perm_problem(5)
  budget <- 150
  spec <- sz_preset_ga_perm(20, budget)
  blocks <- sezgi:::.sz_space_to_blocks(prob$space())
  evaluate_shim <- sezgi:::.sz_make_evaluate_shim(prob)
  expected <- sezgi:::sz_solve_r_problem(spec, blocks, evaluate_shim, master_seed = 1, run_id = 0)

  ga <- GeneticAlgorithm$new()
  actual <- ga$run(perm_problem(5), budget = budget, seed = 1, run_id = 0)
  expect_identical(actual$best_f, expected$best_f)
  expect_identical(actual$best_x, expected$best_x)
  expect_equal(ga$dispatched_representation, "perm")
})

test_that("GeneticAlgorithm dispatches to ga_bin on a Binary space, bit-identical to a direct call", {
  prob <- bin_problem(8)
  budget <- 150
  spec <- sz_preset_ga_bin(20, budget)
  blocks <- sezgi:::.sz_space_to_blocks(prob$space())
  evaluate_shim <- sezgi:::.sz_make_evaluate_shim(prob)
  expected <- sezgi:::sz_solve_r_problem(spec, blocks, evaluate_shim, master_seed = 1, run_id = 0)

  ga <- GeneticAlgorithm$new()
  actual <- ga$run(bin_problem(8), budget = budget, seed = 1, run_id = 0)
  expect_identical(actual$best_f, expected$best_f)
  expect_identical(actual$best_x, expected$best_x)
  expect_equal(ga$dispatched_representation, "bin")
})

test_that("GeneticAlgorithm dispatches to ga_int on an Int space, bit-identical to a direct call", {
  prob <- int_problem(3)
  budget <- 150
  spec <- sz_preset_ga_int(20, budget)
  blocks <- sezgi:::.sz_space_to_blocks(prob$space())
  evaluate_shim <- sezgi:::.sz_make_evaluate_shim(prob)
  expected <- sezgi:::sz_solve_r_problem(spec, blocks, evaluate_shim, master_seed = 1, run_id = 0)

  ga <- GeneticAlgorithm$new()
  actual <- ga$run(int_problem(3), budget = budget, seed = 1, run_id = 0)
  expect_identical(actual$best_f, expected$best_f)
  expect_identical(actual$best_x, expected$best_x)
  expect_equal(ga$dispatched_representation, "int")
})

test_that("GeneticAlgorithm dispatches to ga_cat on a Categorical space, bit-identical to a direct call", {
  prob <- cat_problem(3, 4)
  budget <- 150
  spec <- sz_preset_ga_cat(20, budget)
  blocks <- sezgi:::.sz_space_to_blocks(prob$space())
  evaluate_shim <- sezgi:::.sz_make_evaluate_shim(prob)
  expected <- sezgi:::sz_solve_r_problem(spec, blocks, evaluate_shim, master_seed = 1, run_id = 0)

  ga <- GeneticAlgorithm$new()
  actual <- ga$run(cat_problem(3, 4), budget = budget, seed = 1, run_id = 0)
  expect_identical(actual$best_f, expected$best_f)
  expect_identical(actual$best_x, expected$best_x)
  expect_equal(ga$dispatched_representation, "cat")
})

test_that("GeneticAlgorithm on a Mixed space raises an honest error naming the limitation", {
  expect_error(
    GeneticAlgorithm$new()$run(mixed_problem(), budget = 40, seed = 1),
    "Mixed space"
  )
})

test_that("GeneticAlgorithm representation= overrides auto-dispatch", {
  ga <- GeneticAlgorithm$new(representation = "real")
  res <- ga$run(float_problem(4), budget = 100, seed = 1)
  expect_equal(ga$dispatched_representation, "real")
  expect_true(is.finite(res$best_f))
})

test_that("GeneticAlgorithm rejects an unknown representation", {
  expect_error(GeneticAlgorithm$new(representation = "bogus"), "representation must be one of")
})

# ---- DifferentialEvolution: variant= dispatch, 3 variants anchored -----------

test_that("DifferentialEvolution(variant = 'rand_1') (the default) is bit-identical to a direct sz_preset_de_rand_1 call", {
  problem <- sz_builtin_bbob(1, 5, 1)
  budget <- 150
  expected <- sz_solve_bbob(
    sz_preset_de_rand_1(20, budget), fid = 1L, dim = 5L, instance = 1L, master_seed = 1, run_id = 0
  )
  actual <- DifferentialEvolution$new()$run(problem, budget = budget, seed = 1, run_id = 0)
  expect_identical(actual$best_f, expected$best_f)
  expect_identical(actual$best_x, expected$best_x)
})

test_that("DifferentialEvolution(variant = 'best_1') is bit-identical to a direct sz_preset_de_best_1 call", {
  problem <- sz_builtin_bbob(1, 5, 1)
  budget <- 150
  expected <- sz_solve_bbob(
    sz_preset_de_best_1(20, budget), fid = 1L, dim = 5L, instance = 1L, master_seed = 1, run_id = 0
  )
  actual <- DifferentialEvolution$new(variant = "best_1")$run(problem, budget = budget, seed = 1, run_id = 0)
  expect_identical(actual$best_f, expected$best_f)
  expect_identical(actual$best_x, expected$best_x)
})

test_that("DifferentialEvolution(variant = 'jde') is bit-identical to a direct sz_preset_jde call", {
  problem <- sz_builtin_bbob(1, 5, 1)
  budget <- 150
  expected <- sz_solve_bbob(
    sz_preset_jde(20, budget), fid = 1L, dim = 5L, instance = 1L, master_seed = 1, run_id = 0
  )
  actual <- DifferentialEvolution$new(variant = "jde")$run(problem, budget = budget, seed = 1, run_id = 0)
  expect_identical(actual$best_f, expected$best_f)
  expect_identical(actual$best_x, expected$best_x)
})

test_that("DifferentialEvolution rejects an unknown variant", {
  expect_error(DifferentialEvolution$new(variant = "bogus"), "variant must be one of")
})

# ---- NSGA2: ZDT anchor AND a WFG anchor where k/l matter ----------------------

test_that("NSGA2$run() on a ZDT problem is bit-identical to a direct sz_nsga2() call", {
  expected <- sz_nsga2("zdt1", dim = 5, pop_size = 8, budget = 60, seed = 3)
  actual <- NSGA2$new(pop_size = 8)$run("zdt1", dim = 5, budget = 60, seed = 3)
  expect_identical(actual$individuals, expected$individuals)
  expect_identical(actual$objectives, expected$objectives)
  expect_identical(actual$front0, expected$front0)
  expect_identical(actual$evals_used, expected$evals_used)
})

test_that("NSGA2$run() on a WFG problem (k/l matter) is bit-identical to a direct sz_nsga2() call", {
  expected <- sz_nsga2("wfg4", dim = NULL, m = 3, pop_size = 8, budget = 24, seed = 0, k = 6, l = 4)
  actual <- NSGA2$new(pop_size = 8)$run("wfg4", dim = NULL, budget = 24, m = 3, seed = 0, k = 6, l = 4)
  expect_identical(actual$individuals, expected$individuals)
  expect_identical(actual$objectives, expected$objectives)
  expect_identical(actual$front0, expected$front0)
  expect_identical(actual$evals_used, expected$evals_used)
})

test_that("NSGA2$run() with non-default operator kwargs is bit-identical to a direct sz_nsga2() call", {
  expected <- sz_nsga2(
    "zdt2", dim = 6, pop_size = 8, budget = 40, seed = 5,
    eta_c = 15, eta_m = 25, p_c = 0.8
  )
  actual <- NSGA2$new(pop_size = 8, eta_c = 15, eta_m = 25, p_c = 0.8)$
    run("zdt2", dim = 6, budget = 40, seed = 5)
  expect_identical(actual$individuals, expected$individuals)
  expect_identical(actual$objectives, expected$objectives)
})

test_that("NSGA2$run() returns sz_nsga2's own list shape, not an sz_result", {
  res <- NSGA2$new(pop_size = 8)$run("zdt1", dim = 5, budget = 40, seed = 1)
  expect_false(inherits(res, "sz_result"))
  expect_true(all(c("individuals", "objectives", "front0", "evals_used") %in% names(res)))
})
