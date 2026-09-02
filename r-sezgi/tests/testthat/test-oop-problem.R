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
#
# Fix round 1 (controller review item 9): the shim/bridge dispatch
# ("Sphere ... solved via sz_solve_r_problem" etc. below) was changed from
# ONE R call per INDIVIDUAL to ONE R call per GENERATION (the whole
# population at once), mirroring py-sezgi's per-generation
# `vectorized=True` callable-problem bridge (`py-sezgi/python/sezgi/
# problem.py`'s `Problem._to_native()`). The tests below this line were
# already dispatch-cardinality-agnostic (none asserted call counts) and
# needed no changes; the block starting at "batch dispatch" further down
# is new.

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

# =========================================================================
# Fix round 1 (controller review item 9): batch dispatch (ONE R call per
# GENERATION, not per individual), mirroring py-sezgi's per-generation
# `vectorized=True` callable-problem bridge exactly.
# =========================================================================

# ---- anchor: results stay bit-identical across the dispatch-cardinality
# change ------------------------------------------------------------------

test_that("solve results are bit-identical to the pre-batch-dispatch implementation (anchor)", {
  # Captured from the CURRENT code before the batch-dispatch fix (one
  # sz_solve_r_problem() call, per-individual evaluate() dispatch):
  #   Sphere(dim=4), sz_preset_gwo(6, 60), master_seed=123, run_id=7
  #   best_f = 0.805840203291199
  #   best_x = c(-0.368450268861315, -0.303166452437549,
  #              0.265292548650735, 0.712597058942887)
  #   evals  = 60
  # If batch dispatch (one R call per generation, reshaping/looping the
  # SAME individuals in the SAME order) changes any of these values, the
  # dispatch-cardinality change altered something beyond call count -- a
  # correctness regression, not just a performance change.
  Sphere <- R6::R6Class("SphereAnchorRecheck", inherit = Problem, public = list(
    space = function() sz_space(sz_float(-5, 5, 4)),
    evaluate = function(x) sum(x^2)
  ))
  prob <- Sphere$new()
  spec <- sz_preset_gwo(6, 60)
  blocks <- sezgi:::.sz_space_to_blocks(prob$space())
  shim <- sezgi:::.sz_make_evaluate_shim(prob)
  r <- sezgi:::sz_solve_r_problem(spec, blocks, shim, master_seed = 123, run_id = 7)

  expect_equal(r$best_f, 0.805840203291199)
  expect_equal(
    r$best_x,
    c(-0.368450268861315, -0.303166452437549, 0.265292548650735, 0.712597058942887)
  )
  expect_equal(r$evals, 60)
})

# ---- call-counter: the R side is entered ONCE per generation -----------

test_that("the evaluate shim is entered ONCE per generation, not once per individual", {
  calls <- 0L
  Sphere <- R6::R6Class("SphereCallCounter", inherit = Problem, public = list(
    space = function() sz_space(sz_float(-5, 5, 3)),
    evaluate = function(x) sum(x^2)
  ))
  prob <- Sphere$new()
  pop_size <- 20
  budget <- 200
  spec <- sz_preset_gwo(pop_size, budget)
  blocks <- sezgi:::.sz_space_to_blocks(prob$space())
  raw_shim <- sezgi:::.sz_make_evaluate_shim(prob)
  counting_shim <- function(pop) {
    calls <<- calls + 1L
    # Every call must carry the WHOLE generation, not one individual at a
    # time -- the direct, load-bearing proof that this is batch dispatch.
    expect_equal(length(pop), pop_size)
    raw_shim(pop)
  }

  r <- sezgi:::sz_solve_r_problem(spec, blocks, counting_shim, master_seed = 1, run_id = 0)

  expect_true(is.finite(r$best_f))
  # Per-individual dispatch would have made `calls == budget` (200) here;
  # per-generation dispatch makes it roughly budget/pop_size (a handful),
  # a world apart from 200 -- bound it generously to avoid coupling this
  # test to GWO's exact init/generation accounting.
  expect_true(calls > 0)
  expect_true(calls <= ceiling(budget / pop_size) + 2)
})

# ---- default batch_evaluate delegates to evaluate, in order --------------

test_that("Problem's default batch_evaluate() delegates to evaluate(), in order", {
  Sq <- R6::R6Class("SqOnlyEvaluate", inherit = Problem, public = list(
    space = function() sz_space(sz_float(-10, 10, 1)),
    evaluate = function(x) x^2
  ))
  prob <- Sq$new()
  out <- prob$batch_evaluate(list(2, 3, -4))
  expect_equal(out, c(4, 9, 16))
})

# ---- an overridden batch_evaluate is honored (evaluate never called) -----

test_that("a Problem subclass overriding batch_evaluate directly is honored end to end", {
  called_evaluate <- FALSE
  Batchy <- R6::R6Class("Batchy", inherit = Problem, public = list(
    space = function() sz_space(sz_float(-5, 5, 2)),
    evaluate = function(x) {
      called_evaluate <<- TRUE
      stop("evaluate() should not be called when batch_evaluate is overridden")
    },
    batch_evaluate = function(xs) {
      vapply(xs, function(x) sum(x^2), numeric(1))
    }
  ))
  prob <- Batchy$new()
  spec <- sz_preset_gwo(10, 100)
  blocks <- sezgi:::.sz_space_to_blocks(prob$space())
  shim <- sezgi:::.sz_make_evaluate_shim(prob)

  r <- sezgi:::sz_solve_r_problem(spec, blocks, shim, master_seed = 1, run_id = 0)

  expect_false(called_evaluate)
  expect_true(is.finite(r$best_f))
})

# ---- a wrong-length batch_evaluate() return errors honestly ---------------

test_that("a batch_evaluate() returning the wrong length errors clearly, not silently", {
  BadLength <- R6::R6Class("BadLengthBatch", inherit = Problem, public = list(
    space = function() sz_space(sz_float(-5, 5, 2)),
    evaluate = function(x) sum(x^2),
    batch_evaluate = function(xs) 0  # wrong length: must be length(xs)
  ))
  prob <- BadLength$new()
  spec <- sz_preset_gwo(10, 100)
  blocks <- sezgi:::.sz_space_to_blocks(prob$space())
  shim <- sezgi:::.sz_make_evaluate_shim(prob)

  expect_error(
    sezgi:::sz_solve_r_problem(spec, blocks, shim, master_seed = 1, run_id = 0),
    "length"
  )
})

# =========================================================================
# Final whole-branch review, item M1 (MAJOR): `call_r_evaluate_batch`
# (`solve.rs`) converted its per-generation `pop_list` to a bare `Sexp`
# (dropping `OwnedListSexp`'s own preserved-list token) BEFORE
# `FunctionArgs::new()` ran -- and `FunctionArgs::new()` performs a real R
# allocation (`Rf_cons` for its own `head` pairlist) before `args.add(...)`
# re-protects the payload by attaching it. The SAME anti-pattern class was
# already fixed twice elsewhere on this branch (T3's generate/initialize
# TDD pass; `validate_space`'s block payload, commit `1b7c427`).
#
# This regression test reconstructs the reviewer's OWN repro shape
# (24-block mixed-space Problem -- 6 Float + 6 Int + 6 Categorical + 6
# Binary blocks -- solved via `sz_preset_random_search(16, 96)` under
# `gctorture(TRUE)`), and was PROVEN to trip the pre-fix bug by
# construction: run standalone (outside testthat, to avoid taking the
# whole test session down) against the installed pre-fix build, a loop of
# repeated solves in ONE R session crashed on the 4th repetition -- a
# segfault ("invalid permissions") immediately followed by a bus error
# ("invalid alignment") -- matching the reviewer's own reported symptom
# class exactly. A SINGLE solve call is not reliable enough to trip this
# class of bug (this machine's R build has no strict write barrier, so
# detection depends on the freed memory actually being reused before it
# is read back -- a matter of allocator timing, not of the bug being
# absent) -- hence the repeated-solve loop below, not a single call.
#
# Auditing `call_r_evaluate_batch` and its helpers per the review's
# instruction turned up a SECOND, closely related site while validating
# the fix above: `sz_solve_r_problem`'s own result-assembly tail called
# `out.set_name_and_value(2, "best_x", genotype_to_r(&result.best_x.blocks)?)?`
# -- and savvy 0.10.2's `set_name_and_value` calls `set_name` (which
# allocates a CHARSXP via `Rf_mkCharLenCE`) BEFORE attaching `v`, so a `v`
# that arrives already unprotected (as `genotype_to_r`'s multi-block
# branch does, for exactly the same "OwnedListSexp -> Sexp drops its
# token" reason as the MAIN bug above) sits unprotected across that
# CHARSXP allocation too. Same anti-pattern, different call site, one
# call per SOLVE rather than one call per GENERATION -- which is why it
# needed far more repetitions to manifest: a standalone repeated-solve
# script (same 24-block shape, same preset) against a build with ONLY
# the `call_r_evaluate_batch` reorder applied crashed on the 34th
# repetition (segfault, "invalid permissions"), proving this second site
# independently. Fixed in the same commit by splitting every
# `set_name_and_value` call in `sz_solve_r_problem` into `set_value`
# (attach first; nothing allocates between argument construction and
# `SET_VECTOR_ELT`) followed by `set_name` (allocate the CHARSXP only
# after the value is already reachable through the protected list) --
# see the comment at that call site in `solve.rs`. A standalone 80-
# repetition run of the same script against the build with BOTH fixes
# applied completed every repetition cleanly.
#
# The loop below is intentionally kept at a CI-reasonable width (8 reps,
# ~90s under gctorture on this machine): it reliably re-trips the MAIN
# bug's rep-4 crash point on any regression, and it exercises the SECOND
# site's exact code path every repetition too, even though tripping that
# rarer bug by allocator-timing chance alone would need the much longer
# (34+ rep) standalone width used to originally find and confirm it, which
# is impractical to ship as a routine test.
# =========================================================================

test_that("a large 24-block mixed-space Problem solves correctly across repeated gctorture(TRUE) solves", {
  gctorture(TRUE)
  on.exit(gctorture(FALSE))

  LargeMixed <- R6::R6Class("LargeMixed24Torture", inherit = Problem, public = list(
    space = function() {
      blocks <- c(
        replicate(6, sz_float(-5, 5, 1), simplify = FALSE),
        replicate(6, sz_int(-5L, 5L, 1L), simplify = FALSE),
        replicate(6, sz_categorical(3L, 1L), simplify = FALSE),
        replicate(6, sz_binary(1L), simplify = FALSE)
      )
      do.call(sz_space, blocks)
    },
    evaluate = function(x) {
      sum(vapply(x, function(b) sum(as.numeric(b)), numeric(1)))
    }
  ))
  prob <- LargeMixed$new()
  blocks <- sezgi:::.sz_space_to_blocks(prob$space())
  shim <- sezgi:::.sz_make_evaluate_shim(prob)

  # Repeated solves (not just one) within this same session -- the
  # pre-fix crash needed a handful of repetitions to manifest reliably on
  # this machine's allocator (see this block's own comment above).
  for (rep in seq_len(8)) {
    spec <- sz_preset_random_search(16, 96)
    r <- sezgi:::sz_solve_r_problem(spec, blocks, shim, master_seed = as.double(rep), run_id = 0)
    expect_true(is.finite(r$best_f))
    expect_length(r$best_x, 24L)
  }
})
