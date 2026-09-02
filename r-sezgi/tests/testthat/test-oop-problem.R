# M4-2 Task 2: R-callable problem bridge + the `Problem` R6 base
# (`R/problem.R`, `src/rust/src/solve.rs`) -- the R mirror of py-sezgi's
# subclassable `sezgi.Problem` ABC (`py-sezgi/python/sezgi/problem.py`,
# M4-1 Task 1), closing the "R callable-objective sessions" deferral
# (docs/DECISIONS.md:839-845).
#
# `sz_solve_r_problem()` is an INTERNAL entry point (no `@export`, see its
# own Rust doc) -- every test here calls it via `sezgi:::`, building its
# `blocks`/`evaluate` arguments from `.sz_space_to_blocks()`/
# `.sz_make_evaluate_shim()` by hand, exactly the way a later task's
# wrapper-class `$run()` method is expected to.

# ---- Problem base defaults -------------------------------------------

test_that("Problem base evaluate()/space() stop with 'not implemented'; optimum() defaults NULL", {
  p <- Problem$new()
  expect_error(p$evaluate(1), "not implemented")
  expect_error(p$space(), "not implemented")
  expect_null(p$optimum())
})

# ---- sz_as_problem ------------------------------------------------------

test_that("sz_as_problem accepts a Problem subclass instance unchanged", {
  Sphere1D <- R6::R6Class("Sphere1D", inherit = Problem, public = list(
    space = function() sz_space(sz_float(-1, 1, 1)),
    evaluate = function(x) x^2
  ))
  prob <- Sphere1D$new()
  expect_identical(sz_as_problem(prob), prob)
})

test_that("sz_as_problem rejects a non-Problem with a friendly error", {
  expect_error(sz_as_problem(42), "Problem")
  expect_error(sz_as_problem(list()), "Problem")
  expect_error(sz_as_problem("nope"), "Problem")
})

# ---- Sphere: solved via sz_solve_r_problem, reaches a small best_f,
# deterministic by (master_seed, run_id) ------------------------------

test_that("a Sphere Problem solved via sz_solve_r_problem reaches a small best_f; deterministic by (master_seed, run_id)", {
  Sphere <- R6::R6Class("Sphere", inherit = Problem, public = list(
    space = function() sz_space(sz_float(-5, 5, 8)),
    evaluate = function(x) sum(x^2)
  ))
  prob <- Sphere$new()
  spec <- sz_preset_gwo(20, 4000)
  blocks <- sezgi:::.sz_space_to_blocks(prob$space())
  shim <- sezgi:::.sz_make_evaluate_shim(prob)

  r1 <- sezgi:::sz_solve_r_problem(spec, blocks, shim, master_seed = 42, run_id = 0)
  expect_type(r1, "list")
  expect_named(r1, c("best_f", "evals", "best_x"))
  expect_true(is.finite(r1$best_f))
  expect_true(r1$best_f < 50)
  expect_length(r1$best_x, 8L)
  expect_equal(r1$evals, 4000)

  # Same (master_seed, run_id) -> bit-identical.
  r2 <- sezgi:::sz_solve_r_problem(spec, blocks, shim, master_seed = 42, run_id = 0)
  expect_identical(r1$best_f, r2$best_f)
  expect_identical(r1$best_x, r2$best_x)

  # Different run_id -> a different trajectory.
  r3 <- sezgi:::sz_solve_r_problem(spec, blocks, shim, master_seed = 42, run_id = 1)
  expect_false(identical(r1$best_f, r3$best_f))
})

# ---- Mixed Float+Binary problem: evaluate() receives a typed list --------

MIXED_FLOAT_BINARY_SPEC_JSON <- '{
  "name": "mixed-float-binary",
  "pop_size": 12,
  "init": {"kind": "init/uniform"},
  "boundary": {"kind": "boundary/clamp"},
  "stages": [
    {
      "generator": {
        "kind": "gen/compound",
        "blocks": [
          {"kind": "gen/ga-real", "tournament_k": 2, "pc": 0.9, "eta_c": 15.0, "eta_m": 20.0},
          {"kind": "gen/ga-bin", "tournament_k": 2, "p_c": 0.9}
        ]
      },
      "replacer": {"kind": "replace/mu-plus-lambda"}
    }
  ],
  "termination": {"budget": 120}
}'

test_that("a mixed Float+Binary Problem round-trips: evaluate() receives a 2-element list (double, logical)", {
  seen <- new.env(parent = emptyenv())
  Mixed <- R6::R6Class("MixedFloatBinary", inherit = Problem, public = list(
    space = function() sz_space(sz_float(-5, 5, 3), sz_binary(4)),
    evaluate = function(x) {
      seen$last <- x
      sum(x[[1]]^2) + sum(as.numeric(x[[2]]))
    }
  ))
  prob <- Mixed$new()
  blocks <- sezgi:::.sz_space_to_blocks(prob$space())
  shim <- sezgi:::.sz_make_evaluate_shim(prob)

  r <- sezgi:::sz_solve_r_problem(MIXED_FLOAT_BINARY_SPEC_JSON, blocks, shim, master_seed = 7, run_id = 0)

  # Assert INSIDE-evaluate() observations (stashed each call, so the LAST
  # call's shape is what we check -- every call has the same shape).
  expect_type(seen$last, "list")
  expect_length(seen$last, 2L)
  expect_true(is.double(seen$last[[1]]))
  expect_length(seen$last[[1]], 3L)
  expect_true(is.logical(seen$last[[2]]))
  expect_length(seen$last[[2]], 4L)

  # best_x mirrors the same multi-block shape.
  expect_type(r, "list")
  expect_length(r$best_x, 2L)
  expect_true(is.double(r$best_x[[1]]))
  expect_length(r$best_x[[1]], 3L)
  expect_true(is.logical(r$best_x[[2]]))
  expect_length(r$best_x[[2]], 4L)
})

# ---- error propagation: the load-bearing test of the whole error design --

test_that("an R condition raised inside evaluate() survives to the R session UNCHANGED (class + message)", {
  Boom <- R6::R6Class("Boom", inherit = Problem, public = list(
    space = function() sz_space(sz_float(-5, 5, 2)),
    evaluate = function(x) {
      stop(structure(
        class = c("myCondition", "error", "condition"),
        list(message = "boom", call = NULL)
      ))
    }
  ))
  prob <- Boom$new()
  spec <- sz_preset_gwo(10, 100)
  blocks <- sezgi:::.sz_space_to_blocks(prob$space())
  shim <- sezgi:::.sz_make_evaluate_shim(prob)

  expect_error(
    sezgi:::sz_solve_r_problem(spec, blocks, shim, master_seed = 1, run_id = 0),
    class = "myCondition"
  )

  err <- tryCatch(
    sezgi:::sz_solve_r_problem(spec, blocks, shim, master_seed = 1, run_id = 0),
    myCondition = function(e) e
  )
  expect_s3_class(err, "myCondition")
  expect_equal(conditionMessage(err), "boom")
})

# ---- gctorture: the cheapest way to turn a latent protection bug into a
# deterministic failure (research doc §F1) -----------------------------

test_that("a short solve completes correctly under gctorture(TRUE)", {
  gctorture(TRUE)
  on.exit(gctorture(FALSE))

  SphereTorture <- R6::R6Class("SphereTorture", inherit = Problem, public = list(
    space = function() sz_space(sz_float(-5, 5, 2)),
    evaluate = function(x) sum(x^2)
  ))
  prob <- SphereTorture$new()
  spec <- sz_preset_gwo(4, 20)
  blocks <- sezgi:::.sz_space_to_blocks(prob$space())
  shim <- sezgi:::.sz_make_evaluate_shim(prob)

  r <- sezgi:::sz_solve_r_problem(spec, blocks, shim, master_seed = 1, run_id = 0)
  expect_true(is.finite(r$best_f))
  expect_length(r$best_x, 2L)
})

# ---- bound validation happens at BRIDGE time (SearchSpace::new), not at
# builder time (sz_float()/sz_int() perform none, Task 1) -------------

test_that("a Problem whose space has lo >= hi is accepted at construction but rejected at solve time", {
  BadBounds <- R6::R6Class("BadBounds", inherit = Problem, public = list(
    space = function() sz_space(sz_float(5, 1, 3)),  # lo > hi
    evaluate = function(x) sum(x^2)
  ))
  prob <- BadBounds$new()

  # Construction-time: no validation at all (Task 1 ruling, spaces.R).
  expect_silent(prob$space())

  spec <- sz_preset_gwo(10, 100)
  blocks <- sezgi:::.sz_space_to_blocks(prob$space())
  shim <- sezgi:::.sz_make_evaluate_shim(prob)

  # Solve-time: SearchSpace::new rejects it with a clear error.
  expect_error(
    sezgi:::sz_solve_r_problem(spec, blocks, shim, master_seed = 1, run_id = 0),
    "invalid bounds"
  )
})

# ---- i64 round-trip: Int block lo/hi beyond R's 32-bit integer range
# (M4-2 Task 2's `.sz_block_to_list` fix, R/spaces.R) --------------------

test_that("Int block lo/hi beyond 32-bit range round-trip through the bridge without truncation", {
  BigInt <- R6::R6Class("BigIntProblem", inherit = Problem, public = list(
    space = function() sz_space(sz_int(-3e9, 3e9, 2)),
    evaluate = function(x) sum(as.numeric(x)^2)
  ))
  prob <- BigIntProblem <- BigInt$new()
  blocks <- sezgi:::.sz_space_to_blocks(prob$space())

  # The wire representation itself must carry the full bound, not an
  # NA-on-overflow 32-bit integer.
  expect_true(is.double(blocks[[1]]$lo))
  expect_equal(blocks[[1]]$lo, -3e9)
  expect_equal(blocks[[1]]$hi, 3e9)

  spec <- sz_preset_ga_int(10, 100)
  shim <- sezgi:::.sz_make_evaluate_shim(prob)
  r <- sezgi:::sz_solve_r_problem(spec, blocks, shim, master_seed = 1, run_id = 0)

  expect_true(all(r$best_x >= -3e9 & r$best_x <= 3e9))
})
