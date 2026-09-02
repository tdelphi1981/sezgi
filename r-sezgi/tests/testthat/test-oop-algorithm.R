# M4-2 Task 4: R6 Algorithm / PopulationAlgorithm / LocalSearch bases
# (`r-sezgi/R/algorithm.R`) -- exercises the user-facing surface built on
# top of T2's Problem bridge and T3's engine-hosted generator bridge.

# ---- helpers --------------------------------------------------------------

# A deterministic DE/rand/1-shaped PopulationAlgorithm: overrides ONLY
# `vary()`, drawing donor indices from `ctx$rng` (never R's own global
# RNG/`sample()`) so a run is reproducible from `seed`/`run_id` alone --
# the same discipline `examples/r/oop/custom_de_variant.R` uses.
MyDE <- R6::R6Class("MyDE", inherit = PopulationAlgorithm, public = list(
  F = 0.5,
  vary = function(parents, ctx) {
    n <- length(parents)
    dim <- length(parents[[1]])
    out <- matrix(0, n, dim)
    for (i in seq_len(n)) {
      r1 <- ctx$rng$next_below(as.double(n)) + 1
      r2 <- ctx$rng$next_below(as.double(n)) + 1
      r3 <- ctx$rng$next_below(as.double(n)) + 1
      out[i, ] <- parents[[r1]] + self$F * (parents[[r2]] - parents[[r3]])
    }
    out
  }
))

rastrigin_problem <- function(d = 5) {
  R6::R6Class("Rastrigin", inherit = Problem, public = list(
    d = d,
    space = function() sz_space(sz_float(-5.12, 5.12, self$d)),
    evaluate = function(x) 10 * self$d + sum(x^2 - 10 * cos(2 * pi * x)),
    optimum = function() 0
  ))$new()
}

# ---- MyDE runs on an R problem AND on the built-in form -------------------

test_that("MyDE runs end-to-end on an R Problem subclass", {
  res <- MyDE$new()$run(rastrigin_problem(5), budget = 300, pop_size = 10, seed = 1)
  expect_s3_class(res, "sz_result")
  expect_true(is.finite(res$best_f))
  expect_equal(res$evals_used, 300)
  expect_equal(res$f_opt, 0)
  expect_equal(res$gap, res$best_f - 0)
})

test_that("MyDE runs end-to-end on the built-in form (sz_builtin_bbob)", {
  res <- MyDE$new()$run(sz_builtin_bbob(1, 5, 1), budget = 300, pop_size = 10, seed = 1)
  expect_s3_class(res, "sz_result")
  expect_true(is.finite(res$best_f))
  expect_equal(res$evals_used, 300)
  expect_null(res$f_opt)
  expect_null(res$gap)
})

# ---- determinism / run_id / name ------------------------------------------

test_that("MyDE is deterministic across repeat runs at the same seed/run_id", {
  r1 <- MyDE$new()$run(sz_builtin_bbob(1, 4, 1), budget = 200, pop_size = 8, seed = 42, run_id = 0)
  r2 <- MyDE$new()$run(sz_builtin_bbob(1, 4, 1), budget = 200, pop_size = 8, seed = 42, run_id = 0)
  expect_identical(r1$best_f, r2$best_f)
  expect_identical(r1$best_x, r2$best_x)
})

test_that("a different run_id changes the result", {
  r1 <- MyDE$new()$run(sz_builtin_bbob(1, 4, 1), budget = 200, pop_size = 8, seed = 42, run_id = 0)
  r2 <- MyDE$new()$run(sz_builtin_bbob(1, 4, 1), budget = 200, pop_size = 8, seed = 42, run_id = 1)
  expect_false(identical(r1$best_f, r2$best_f))
})

test_that("name = 'x' lands in result$algo", {
  res <- MyDE$new()$run(sz_builtin_bbob(1, 4, 1), budget = 100, pop_size = 5, seed = 1, name = "x")
  expect_equal(res$algo, "x")
})

test_that("default name is the lowercased class name", {
  res <- MyDE$new()$run(sz_builtin_bbob(1, 4, 1), budget = 100, pop_size = 5, seed = 1)
  expect_equal(res$algo, "myde")
})

# ---- PopulationAlgorithm: partial-override inherits tournament select -----

test_that("a PopulationAlgorithm subclass overriding ONLY vary() inherits the default tournament select()", {
  inst <- MyDE$new()
  # select() is INHERITED unchanged -- identical to the base class's own
  # declared function (ignoring environment, since R6 rebinds every
  # method's enclosure per instance regardless of override status; see
  # R/algorithm.R's own module doc for why `ignore.environment = TRUE` is
  # the correct spelling here).
  expect_true(identical(
    inst$select, PopulationAlgorithm$public_methods$select, ignore.environment = TRUE
  ))
  # vary() itself IS overridden (sanity: proves the identity check above is
  # actually discriminating, not vacuously true for every method).
  expect_false(identical(
    inst$vary, PopulationAlgorithm$public_methods$vary, ignore.environment = TRUE
  ))

  # Load-bearing behavioral proof: select() (never overridden by MyDE) is
  # actually exercised by a real run -- generate() unconditionally calls
  # self$select(pop, n, ctx) before self$vary(), so a successful,
  # deterministic run (covered by the determinism test above) is only
  # possible if the inherited tournament select() is being reached and
  # drawing from ctx$rng correctly (a missing/broken select() would error,
  # not silently no-op).
  res <- MyDE$new()$run(sz_builtin_bbob(1, 4, 1), budget = 200, pop_size = 8, seed = 5)
  expect_true(is.finite(res$best_f))
})

# ---- LocalSearch: only neighbor overridden, pop_size = 1 -------------------

test_that("a LocalSearch subclass overriding ONLY neighbor() runs at pop_size = 1", {
  HillClimb <- R6::R6Class("HillClimb", inherit = LocalSearch, public = list(
    step = 0.3,
    neighbor = function(x, ctx) x + (ctx$rng$next_f64() - 0.5) * 2 * self$step
  ))
  res <- HillClimb$new()$run(sz_builtin_bbob(1, 3, 1), budget = 100, seed = 7)
  expect_s3_class(res, "sz_result")
  expect_true(is.finite(res$best_f))
  expect_equal(res$evals_used, 100)
})

test_that("LocalSearch$run() rejects pop_size != 1", {
  HillClimb <- R6::R6Class("HillClimb2", inherit = LocalSearch, public = list(
    neighbor = function(x, ctx) x
  ))
  expect_error(
    HillClimb$new()$run(sz_builtin_bbob(1, 3, 1), budget = 100, pop_size = 4, seed = 1),
    "pop_size = 1"
  )
})

# ---- initialize_population override detection -----------------------------

test_that("initialize_population NOT overridden: run() is bit-identical to a raw sz_solve_r_generator_bbob call with initializer = NULL", {
  algo <- MyDE$new()
  res <- algo$run(sz_builtin_bbob(1, 4, 1), budget = 240, pop_size = 8, seed = 11, run_id = 2)

  blocks <- sezgi:::.sz_space_to_blocks(sz_space(sz_float(-5, 5, 4L)))
  shim <- sezgi:::.sz_make_generate_shim(algo, blocks)
  raw <- sezgi:::sz_solve_r_generator_bbob(
    shim, fid = 1L, dim = 4L, instance = 1L, budget = 240, master_seed = 11, run_id = 2,
    pop_size = 8, init_kind = "init/uniform", replacer_kind = "replace/mu-plus-lambda",
    initializer = NULL, validate_space = NULL, algo_name = "myde"
  )
  expect_identical(res$best_f, raw$best_f)
  expect_identical(res$best_x, raw$best_x)
  expect_identical(res$evals_used, raw$evals)
})

test_that("initialize_population OVERRIDDEN: run() calls it, proven by a marker value reaching the engine", {
  MyDEInit <- R6::R6Class("MyDEInit", inherit = PopulationAlgorithm, public = list(
    vary = function(parents, ctx) do.call(rbind, parents),
    initialize_population = function(n, space, ctx) {
      # Marker: every initial individual is exactly (9, 9, 9) -- WAY
      # outside BBOB's [-5, 5] domain, so a successful run's best_x is
      # forced to the boundary-clamped (5, 5, 5) IF (and only if) this
      # initializer actually ran (the pure-Rust init/uniform default would
      # never produce a uniform population this extreme by chance).
      matrix(9.0, n, space[[1]]$n)
    }
  ))
  res <- MyDEInit$new()$run(sz_builtin_bbob(1, 3, 1), budget = 15, pop_size = 5, seed = 3)
  expect_equal(res$best_x, c(5, 5, 5))
})

test_that("initialize_population override detection is class-level: a 3-level chain (subclass of a PopulationAlgorithm subclass) inherits the non-overridden default", {
  Grandchild <- R6::R6Class("Grandchild", inherit = MyDE, public = list())
  algo <- Grandchild$new()
  overridden <- !identical(
    algo$initialize_population, Algorithm$public_methods$initialize_population,
    ignore.environment = TRUE
  )
  expect_false(overridden)
})

# ---- sz_result shape + print smoke -----------------------------------------

test_that("run() returns an sz_result with exactly the 8 documented fields", {
  res <- MyDE$new()$run(sz_builtin_bbob(1, 4, 1), budget = 100, pop_size = 5, seed = 1)
  expect_s3_class(res, "sz_result")
  expect_named(
    res, c("algo", "seed", "budget", "evals_used", "best_x", "best_f", "f_opt", "gap")
  )
  expect_equal(res$seed, 1)
  expect_equal(res$budget, 100)
})

test_that("print.sz_result() smoke test", {
  res <- MyDE$new()$run(sz_builtin_bbob(1, 4, 1), budget = 100, pop_size = 5, seed = 1)
  out <- capture.output(print(res))
  expect_true(any(grepl("sz_result", out, fixed = TRUE)))
  expect_true(any(grepl("algo=myde", out, fixed = TRUE)))

  res_opt <- MyDE$new()$run(rastrigin_problem(3), budget = 100, pop_size = 5, seed = 1)
  out_opt <- capture.output(print(res_opt))
  expect_true(any(grepl("f_opt=", out_opt, fixed = TRUE)))
})

# ---- log_dir: honest rejection ---------------------------------------------

test_that("run(log_dir = ...) is honestly rejected with a clear error", {
  expect_error(
    MyDE$new()$run(sz_builtin_bbob(1, 4, 1), budget = 100, pop_size = 5, seed = 1, log_dir = tempdir()),
    "log_dir is not supported"
  )
})

# ---- validate_space: overridden vs. not ------------------------------------

test_that("validate_space NOT overridden: no veto, run proceeds normally", {
  res <- MyDE$new()$run(sz_builtin_bbob(1, 3, 1), budget = 60, pop_size = 4, seed = 1)
  expect_true(is.finite(res$best_f))
})

test_that("validate_space OVERRIDDEN: raising rejects the run before generate() is called", {
  called <- new.env(parent = emptyenv())
  called$generate <- 0L
  MyDEVeto <- R6::R6Class("MyDEVeto", inherit = PopulationAlgorithm, public = list(
    vary = function(parents, ctx) {
      called$generate <- called$generate + 1L
      do.call(rbind, parents)
    },
    validate_space = function(space) {
      stop("nope: rejected by test")
    }
  ))
  expect_error(
    MyDEVeto$new()$run(sz_builtin_bbob(1, 3, 1), budget = 60, pop_size = 4, seed = 1),
    "nope: rejected by test"
  )
  expect_equal(called$generate, 0L)
})
