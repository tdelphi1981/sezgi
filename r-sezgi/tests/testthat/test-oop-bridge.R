# M4-2 Task 3: engine-hosted R generator bridge + the `SzRng` external-
# pointer RNG handle (`r-sezgi/src/rust/src/solve.rs`) -- an R-authored
# `generate(pop, fitness, rng, iteration)` closure running INSIDE the Rust
# engine's generate stage, the R mirror of py-sezgi's
# `solve_with_py_generator` (`py-sezgi/src/lib.rs`).
#
# Both internal entry points tested here are INTERNAL (no `@export`, see
# their own Rust doc) -- called via `sezgi:::`, exactly like T2's
# `sz_solve_r_problem`:
#   - `sz_solve_r_generator_bbob(generate, fid, dim, instance, budget,
#      master_seed, run_id, pop_size, init_kind, replacer_kind, initializer,
#      validate_space, algo_name)` -- the built-in-problem form (BBOB,
#      cheapest existing `sz_solve_*` family to mirror).
#   - `sz_solve_r_generator(generate, blocks, evaluate, budget, master_seed,
#      run_id, pop_size, init_kind, replacer_kind, initializer,
#      validate_space, algo_name)` -- the R-callable-problem form (T2's
#      `blocks`/`evaluate` bridge, same shape `sz_solve_r_problem` takes).
#
# `generate`'s PINNED calling convention (the T4 contract -- see this
# task's own report for the full detail): a closure of FOUR positional
# arguments, `function(pop, fitness, rng, iteration)`:
#   - `pop`: an (unnamed) list of length n (one entry per individual, in
#     the CURRENT population's own order), each entry converted the SAME
#     way `evaluate(x)`'s own `x` is (bare for a single-block space, an
#     unnamed list of per-block values for a multi-block one).
#   - `fitness`: a numeric vector of length n, `pop`'s own fitness values,
#     in the same order.
#   - `rng`: a RAW external pointer (EXTPTRSXP) to a fresh `SzRng` handle,
#     already positioned wherever the engine's own stage stream was left --
#     wrap it via `sezgi:::.savvy_wrap_SzRng(rng)` to get `$next_f64()`/
#     `$next_below(n)`/`$split(child_id)`. Valid ONLY for the duration of
#     THIS call: the pointer is nulled immediately after `generate()`
#     returns (a stored handle errors with "consumed or deleted" --
#     `savvy::Error::InvalidPointer` -- on later use).
#   - `iteration`: a double, the engine's own 0-based iteration counter.
#
# `generate` must return an (unnamed) list of offspring individuals, in the
# SAME bare-vs-list convention as `pop`'s own entries -- NEVER an empty
# list (an empty offspring vector silently hangs the engine, see the
# module doc at `solve.rs`'s own Task 3 section for the full citation).

# ---- helpers -----------------------------------------------------------

# Every test below passes every sz_solve_r_generator*() argument explicitly
# (this internal entry point has no R-native defaults yet -- T4's job);
# these two wrappers just keep each test_that() block's own call sites
# short and put the (documented) defaults from `solve_with_py_generator`
# (py-sezgi/src/lib.rs) -- pop_size = 20, init_kind = "init/uniform",
# replacer_kind = "replace/mu-plus-lambda" -- in exactly one place.
solve_r_generator_bbob <- function(generate, fid = 1L, dim = 2L, instance = 1L,
                                    budget, master_seed = 0, run_id = 0,
                                    pop_size = 20, init_kind = "init/uniform",
                                    replacer_kind = "replace/mu-plus-lambda",
                                    initializer = NULL, validate_space = NULL,
                                    algo_name = NULL) {
  sezgi:::sz_solve_r_generator_bbob(
    generate, fid = fid, dim = dim, instance = instance, budget = budget,
    master_seed = master_seed, run_id = run_id, pop_size = pop_size,
    init_kind = init_kind, replacer_kind = replacer_kind,
    initializer = initializer, validate_space = validate_space,
    algo_name = algo_name
  )
}

solve_r_generator_problem <- function(generate, blocks, evaluate, budget,
                                       master_seed = 0, run_id = 0,
                                       pop_size = 20, init_kind = "init/uniform",
                                       replacer_kind = "replace/mu-plus-lambda",
                                       initializer = NULL, validate_space = NULL,
                                       algo_name = NULL) {
  sezgi:::sz_solve_r_generator(
    generate, blocks, evaluate, budget = budget, master_seed = master_seed,
    run_id = run_id, pop_size = pop_size, init_kind = init_kind,
    replacer_kind = replacer_kind, initializer = initializer,
    validate_space = validate_space, algo_name = algo_name
  )
}

# A trivial "uniform resample via ctx rng" generator: ignores `pop`, draws a
# fresh uniform point in [-5, 5]^dim for every offspring individual.
uniform_resample_generate <- function(dim) {
  function(pop, fitness, rng, iteration) {
    h <- sezgi:::.savvy_wrap_SzRng(rng)
    n <- length(pop)
    lapply(seq_len(n), function(i) {
      vapply(seq_len(dim), function(j) -5 + 10 * h$next_f64(), numeric(1))
    })
  }
}

# ---- determinism ---------------------------------------------------------

test_that("sz_solve_r_generator_bbob: same seed twice is bit-identical; a different run_id differs", {
  gen <- uniform_resample_generate(3)

  r1 <- solve_r_generator_bbob(gen, dim = 3L, budget = 60, master_seed = 42,
                                run_id = 0, pop_size = 6)
  r2 <- solve_r_generator_bbob(gen, dim = 3L, budget = 60, master_seed = 42,
                                run_id = 0, pop_size = 6)
  expect_identical(r1$best_f, r2$best_f)
  expect_identical(r1$best_x, r2$best_x)
  expect_equal(r1$evals, 60)

  r3 <- solve_r_generator_bbob(gen, dim = 3L, budget = 60, master_seed = 42,
                                run_id = 1, pop_size = 6)
  expect_false(identical(r1$best_f, r3$best_f))
})

# ---- RNG continuity: the load-bearing determinism test -------------------

test_that("recorded next_f64() draws match a standalone SzRng$from_master(seed, [run_id, 1]) reconstruction", {
  seed <- 42
  run_id <- 3
  recorded <- numeric(0)

  gen <- function(pop, fitness, rng, iteration) {
    h <- sezgi:::.savvy_wrap_SzRng(rng)
    n <- length(pop)
    dim <- length(pop[[1]])
    lapply(seq_len(n), function(i) {
      vapply(seq_len(dim), function(j) {
        v <- h$next_f64()
        recorded[length(recorded) + 1] <<- v
        -5 + 10 * v
      }, numeric(1))
    })
  }

  r <- solve_r_generator_bbob(gen, dim = 3L, budget = 30, master_seed = seed,
                               run_id = run_id, pop_size = 6)
  expect_true(is.finite(r$best_f))
  expect_true(length(recorded) > 0)

  # Stage 0's generator stream lives at path [run_id, 1] (single-stage spec
  # -- `crates/core/src/engine.rs`'s own path convention, `stage_rngs[0] =
  # (RngStream::from_master(seed, [run_id, 1]), ...)`).
  standalone <- sezgi:::SzRng$from_master(seed, c(run_id, 1))
  replay <- vapply(seq_along(recorded), function(i) standalone$next_f64(), numeric(1))
  expect_equal(replay, recorded)
})

# ---- write-back is load-bearing -------------------------------------------

test_that("write-back is load-bearing: consecutive generate() calls draw different values", {
  draws <- list()
  gen <- function(pop, fitness, rng, iteration) {
    h <- sezgi:::.savvy_wrap_SzRng(rng)
    v <- h$next_f64()
    draws[[length(draws) + 1]] <<- v
    n <- length(pop)
    dim <- length(pop[[1]])
    lapply(seq_len(n), function(i) rep(v, dim))
  }

  r <- solve_r_generator_bbob(gen, dim = 2L, budget = 30, master_seed = 5,
                               run_id = 0, pop_size = 3)
  expect_true(is.finite(r$best_f))
  expect_true(length(draws) >= 2)
  expect_false(isTRUE(all.equal(draws[[1]], draws[[2]])))
})

# ---- stale handle ----------------------------------------------------------

test_that("a stale RNG handle (stored across iterations) errors cleanly, not a crash or a wrong value", {
  stored <- new.env(parent = emptyenv())
  gen <- function(pop, fitness, rng, iteration) {
    if (iteration == 0) {
      stored$rng <- sezgi:::.savvy_wrap_SzRng(rng)
    } else if (iteration == 1) {
      stored$rng$next_f64()
    }
    n <- length(pop)
    dim <- length(pop[[1]])
    lapply(seq_len(n), function(i) rep(0, dim))
  }

  expect_error(
    solve_r_generator_bbob(gen, dim = 2L, budget = 30, master_seed = 1,
                            run_id = 0, pop_size = 3),
    "consumed or deleted"
  )
})

# ---- error propagation mid-generate: the load-bearing test of the whole
# error design ---------------------------------------------------------------

test_that("an R condition raised inside generate() on iteration 3 survives to the R session UNCHANGED (class + message); session survives", {
  gen <- function(pop, fitness, rng, iteration) {
    if (iteration == 3) {
      stop(structure(
        class = c("myGenCondition", "error", "condition"),
        list(message = "boom-in-generate", call = NULL)
      ))
    }
    lapply(pop, identity)
  }

  expect_error(
    solve_r_generator_bbob(gen, dim = 2L, budget = 1000, master_seed = 1,
                            run_id = 0, pop_size = 3),
    class = "myGenCondition"
  )

  err <- tryCatch(
    solve_r_generator_bbob(gen, dim = 2L, budget = 1000, master_seed = 1,
                            run_id = 0, pop_size = 3),
    myGenCondition = function(e) e
  )
  expect_s3_class(err, "myGenCondition")
  expect_equal(conditionMessage(err), "boom-in-generate")

  # The R session survives: a subsequent, unrelated call still works.
  r <- solve_r_generator_bbob(function(pop, fitness, rng, iteration) lapply(pop, identity),
                               dim = 2L, budget = 30, master_seed = 1,
                               run_id = 0, pop_size = 3)
  expect_true(is.finite(r$best_f))
})

# ---- malformed offspring: three shapes, each an honest error -------------

test_that("malformed offspring: wrong length, wrong type, and a duplicate permutation entry each raise an honest error naming the defect", {
  # (1) wrong row length: BBOB dim = 3L, offspring individual has 2 values.
  gen_wrong_length <- function(pop, fitness, rng, iteration) {
    lapply(pop, function(x) x[1:2])
  }
  expect_error(
    solve_r_generator_bbob(gen_wrong_length, dim = 3L, budget = 30,
                            master_seed = 1, run_id = 0, pop_size = 3),
    "expected 3 values"
  )

  # (2) wrong type: a character vector instead of numeric.
  gen_wrong_type <- function(pop, fitness, rng, iteration) {
    lapply(pop, function(x) c("a", "b", "c"))
  }
  expect_error(
    solve_r_generator_bbob(gen_wrong_type, dim = 3L, budget = 30,
                            master_seed = 1, run_id = 0, pop_size = 3)
  )

  # (3) a duplicate permutation entry (R-callable problem, one Permutation
  # block -- BBOB is Float-only, so this shape needs sz_solve_r_generator).
  PermProblem <- R6::R6Class("PermProblemBridgeTest", inherit = Problem, public = list(
    space = function() sz_space(sz_permutation(4)),
    evaluate = function(x) sum(x)
  ))
  prob <- PermProblem$new()
  blocks <- sezgi:::.sz_space_to_blocks(prob$space())
  shim <- sezgi:::.sz_make_evaluate_shim(prob)
  gen_dup_perm <- function(pop, fitness, rng, iteration) {
    lapply(pop, function(x) c(0L, 0L, 1L, 2L))
  }
  expect_error(
    solve_r_generator_problem(gen_dup_perm, blocks, shim, budget = 40,
                               master_seed = 1, run_id = 0, pop_size = 4),
    "not a valid permutation"
  )
})

# ---- validate_space veto ---------------------------------------------------

test_that("a validate_space callback that returns a rejection string prevents the run at build time; generate() is never called", {
  called <- new.env(parent = emptyenv())
  called$generate <- 0L
  called$validate <- 0L

  gen <- function(pop, fitness, rng, iteration) {
    called$generate <- called$generate + 1L
    lapply(pop, identity)
  }
  veto <- function(blocks) {
    called$validate <- called$validate + 1L
    "space rejected: too many blocks"
  }

  expect_error(
    solve_r_generator_bbob(gen, dim = 2L, budget = 30, master_seed = 1,
                            run_id = 0, pop_size = 3, validate_space = veto),
    "space rejected"
  )
  expect_equal(called$validate, 1L)
  expect_equal(called$generate, 0L)
})

test_that("a validate_space callback that stop()s (a genuine R condition) also vetoes the run before generate() is called", {
  called <- new.env(parent = emptyenv())
  called$generate <- 0L

  gen <- function(pop, fitness, rng, iteration) {
    called$generate <- called$generate + 1L
    lapply(pop, identity)
  }
  veto <- function(blocks) {
    stop(structure(
      class = c("myVetoCondition", "error", "condition"),
      list(message = "veto-in-validate-space", call = NULL)
    ))
  }

  expect_error(
    solve_r_generator_bbob(gen, dim = 2L, budget = 30, master_seed = 1,
                            run_id = 0, pop_size = 3, validate_space = veto),
    class = "myVetoCondition"
  )
  expect_equal(called$generate, 0L)
})

# ---- fix round 1 regression: a large mixed-space validate_space payload
# under gctorture(TRUE) (controller review finding, empirically reproduced
# as list corruption and, once, a hard segfault against the pre-fix build:
# `blocks_to_r(space)`'s returned `Sexp` was built BEFORE `FunctionArgs::
# new()` existed, leaving it unprotected across that constructor's own
# `Rf_cons` allocation -- fixed by the same build-then-immediately-attach
# discipline `RGenerator::generate`/`RInitializer::initialize` already
# used) ---------------------------------------------------------------------

test_that("validate_space sees correct descriptors for a large (24-block) mixed space and vetoes cleanly, under gctorture(TRUE)", {
  gctorture(TRUE)
  on.exit(gctorture(FALSE))

  # 6 Float + 6 Int + 6 Categorical + 3 Binary + 3 Permutation = 24 blocks,
  # several kinds -- the reviewer's repro shape. Vetoed immediately (no
  # generate() call needed) to keep the torture run as cheap as possible.
  block_kinds <- c(rep("float", 6), rep("int", 6), rep("categorical", 6),
                    rep("binary", 3), rep("permutation", 3))
  block_objs <- lapply(block_kinds, function(kind) {
    switch(kind,
      float = sz_float(-1, 1, 2),
      int = sz_int(-3, 3, 2),
      categorical = sz_categorical(3, 2),
      binary = sz_binary(2),
      permutation = sz_permutation(3)
    )
  })
  LargeMixed <- R6::R6Class("LargeMixedBridgeRegressionTest", inherit = Problem, public = list(
    space = function() do.call(sz_space, block_objs),
    evaluate = function(x) 0
  ))
  prob <- LargeMixed$new()
  space_blocks <- sezgi:::.sz_space_to_blocks(prob$space())
  shim <- sezgi:::.sz_make_evaluate_shim(prob)

  seen <- new.env(parent = emptyenv())
  gen <- function(pop, fitness, rng, iteration) lapply(pop, identity)
  veto <- function(blocks) {
    seen$n_blocks <- length(blocks)
    seen$types <- vapply(blocks, function(b) b$type, character(1))
    "rejected: regression-test veto"
  }

  expect_error(
    solve_r_generator_problem(gen, space_blocks, shim, budget = 6,
                               master_seed = 1, run_id = 0, pop_size = 3,
                               validate_space = veto),
    "rejected: regression-test veto"
  )
  expect_equal(seen$n_blocks, 24L)
  expect_equal(seen$types, block_kinds)
})

# ---- mixed space end-to-end with an R problem -----------------------------

test_that("mixed Float+Permutation space solves end-to-end via sz_solve_r_generator (R-callable problem)", {
  Mixed <- R6::R6Class("MixedFloatPermBridgeTest", inherit = Problem, public = list(
    space = function() sz_space(sz_float(-5, 5, 2), sz_permutation(4)),
    evaluate = function(x) sum(x[[1]]^2) + sum(x[[2]])
  ))
  prob <- Mixed$new()
  blocks <- sezgi:::.sz_space_to_blocks(prob$space())
  shim <- sezgi:::.sz_make_evaluate_shim(prob)

  gen <- function(pop, fitness, rng, iteration) {
    h <- sezgi:::.savvy_wrap_SzRng(rng)
    n <- length(pop)
    lapply(seq_len(n), function(i) {
      x <- pop[[i]]
      float_part <- x[[1]] + (h$next_f64() - 0.5) * 0.1
      perm_part <- as.integer(sample(0:3, 4))
      list(float_part, perm_part)
    })
  }

  r <- solve_r_generator_problem(gen, blocks, shim, budget = 40,
                                  master_seed = 3, run_id = 0, pop_size = 4)
  expect_type(r, "list")
  expect_named(r, c("best_f", "evals", "best_x"))
  expect_length(r$best_x, 2L)
  expect_true(is.double(r$best_x[[1]]))
  expect_length(r$best_x[[1]], 2L)
  expect_length(r$best_x[[2]], 4L)
  expect_setequal(as.integer(r$best_x[[2]]), 0:3)
})

# ---- gctorture: the cheapest way to turn a latent protection bug into a
# deterministic failure (research doc §F1) -----------------------------

test_that("a short sz_solve_r_generator_bbob run completes correctly under gctorture(TRUE)", {
  gctorture(TRUE)
  on.exit(gctorture(FALSE))

  gen <- function(pop, fitness, rng, iteration) {
    h <- sezgi:::.savvy_wrap_SzRng(rng)
    lapply(pop, function(x) x + (h$next_f64() - 0.5))
  }
  r <- solve_r_generator_bbob(gen, dim = 2L, budget = 12, master_seed = 1,
                               run_id = 0, pop_size = 4)
  expect_true(is.finite(r$best_f))
  expect_length(r$best_x, 2L)
})

# ---- SzRng$next_below()/$split(): sanity coverage beyond next_f64() ------

test_that("SzRng$next_below(n) and $split(child_id) work as advertised, exercised inside generate()", {
  seen <- new.env(parent = emptyenv())
  gen <- function(pop, fitness, rng, iteration) {
    h <- sezgi:::.savvy_wrap_SzRng(rng)
    if (iteration == 0) {
      seen$below <- h$next_below(5)
      seen$child_draw <- h$split(1)$next_f64()
    } else {
      h$next_below(5)
    }
    lapply(pop, identity)
  }
  r <- solve_r_generator_bbob(gen, dim = 2L, budget = 15, master_seed = 9,
                               run_id = 0, pop_size = 3)
  expect_true(is.finite(r$best_f))
  expect_true(seen$below >= 0 && seen$below < 5)
  expect_true(is.finite(seen$child_draw))

  # A SPLIT child stream reconstructs deterministically from the SAME
  # (master_seed, path) + child_id: stage 0's generator stream lives at
  # path [run_id, 1] (see the RNG continuity test above); iteration 0's
  # `rng` is exactly that stream, positioned at its start.
  standalone_child <- sezgi:::SzRng$from_master(9, c(0, 1))$split(1)
  expect_equal(standalone_child$next_f64(), seen$child_draw)
})
