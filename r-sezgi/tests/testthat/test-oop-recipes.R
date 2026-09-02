# M4-2 Task 6: `FeatureSelection`/`MixedTuning` data recipes (`R/recipes.R`)
# -- the R mirror of py-sezgi's `sezgi.recipes.FeatureSelection`/
# `MixedTuning` (`py-sezgi/python/sezgi/recipes.py`, M4-1 Task 6, repo
# authority for the semantics mirrored here).
#
# Covers this task's own pinned test list:
# (a) FeatureSelection end-to-end recovery, anchored (fixed seed, exact
#     mask + best_f) -- a dataset/scorer/penalty/budget/seed INDEPENDENTLY
#     derived here (not copied from the Python twin's own numbers), plus a
#     brute-force 2^8 = 256-mask uniqueness cross-check computed INSIDE
#     this test file (an independent argmin, not a copied constant).
# (b) penalty term monotonicity (unit-level, no engine).
# (c) empty-mask documented behavior (unit-level): evaluate() on the
#     all-FALSE mask returns Inf and never calls scorer.
# (d) the drop = FALSE regression: a scorer that stopifnot(is.matrix(X))
#     with a 1-column mask -- the R analogue of M4-1 Task 6's Python
#     bool-indexing trap.
# (e) scorer exceptions/conditions propagate unchanged.
# (f) MixedTuning smoke, in two genuinely different exercises (mirroring
#     the Python twin's own split, see MixedTuning's own doc on why
#     GeneticAlgorithm cannot run a Mixed space): (f1) a Float-only
#     MixedTuning instance run via GeneticAlgorithm's auto-dispatch;
#     (f2) a Float+Categorical Mixed-space MixedTuning instance run
#     through a hand-built gen/compound algorithm spec (the SAME pattern
#     test-oop-problem.R's own mixed-space test uses), asserting the
#     objective receives the pinned per-block types.
# (g) determinism for both recipes.
# (h) export surface / Problem-subclass structural checks.

# =========================================================================
# (a) FeatureSelection end-to-end recovery, anchored.
#
# Dataset/scorer/penalty/budget/seed constants below are an INDEPENDENT
# derivation (base-R `set.seed()`/`rnorm()`, not numpy) -- chosen by
# sweeping a handful of candidate dataset seeds and keeping one where the
# GA-recovered mask matches the brute-force-confirmed UNIQUE global
# minimum (see this task's own report for the sweep). Not copied from
# py-sezgi's test_oop_recipes.py own numbers (which use a different
# seed/informative-column/penalty combination entirely).
# =========================================================================

DATA_SEED <- 42
N_SAMPLES <- 20L
N_FEATURES <- 8L
INFORMATIVE <- c(2L, 4L, 7L) # 1-based R column indices
COEFS <- c(3.0, -2.0, 1.5)
NOISE_SD <- 0.2
PENALTY <- 0.3
POP_SIZE <- 20
BUDGET <- 200
GA_SEED <- 42

.make_dataset <- function() {
  set.seed(DATA_SEED)
  X <- matrix(rnorm(N_SAMPLES * N_FEATURES), nrow = N_SAMPLES, ncol = N_FEATURES)
  noise <- rnorm(N_SAMPLES, sd = NOISE_SD)
  y <- as.vector(X[, INFORMATIVE, drop = FALSE] %*% COEFS + noise)
  list(X = X, y = y)
}

.ols_rss_scorer <- function(X_sub, y) {
  design <- cbind(1, X_sub)
  coefs <- qr.solve(design, y)
  resid <- y - design %*% coefs
  sum(resid^2)
}

test_that("FeatureSelection recovers the informative mask, anchored (fixed dataset/penalty/budget/seed)", {
  ds <- .make_dataset()
  problem <- FeatureSelection$new(ds$X, ds$y, .ols_rss_scorer, penalty = PENALTY)
  ga <- GeneticAlgorithm$new(pop_size = POP_SIZE)
  res <- ga$run(problem, budget = BUDGET, seed = GA_SEED)

  expect_equal(res$evals_used, BUDGET)
  expect_equal(res$best_f, 0.65621827579752678, tolerance = 1e-12)
  expect_length(res$best_x, N_FEATURES)
  expect_true(is.logical(res$best_x))

  recovered_mask <- as.logical(res$best_x)
  informative_mask <- seq_len(N_FEATURES) %in% INFORMATIVE
  expect_identical(recovered_mask, informative_mask)
})

test_that("FeatureSelection recovery is deterministic across repeated runs at the same seed", {
  ds <- .make_dataset()
  problem <- FeatureSelection$new(ds$X, ds$y, .ols_rss_scorer, penalty = PENALTY)
  a <- GeneticAlgorithm$new(pop_size = POP_SIZE)$run(problem, budget = BUDGET, seed = GA_SEED)
  b <- GeneticAlgorithm$new(pop_size = POP_SIZE)$run(problem, budget = BUDGET, seed = GA_SEED)
  expect_identical(a$best_f, b$best_f)
  expect_identical(a$best_x, b$best_x)
})

test_that("brute force over all 2^8 = 256 masks confirms the informative mask is the UNIQUE global minimum", {
  # Independent derivation (a fresh loop over every possible mask, computed
  # HERE, not copied from the anchor above) -- confirms the informative
  # mask is the unique minimizer under this penalty, so the anchored
  # recovery test above is verifying something real, not a GA fluke.
  ds <- .make_dataset()
  problem <- FeatureSelection$new(ds$X, ds$y, .ols_rss_scorer, penalty = PENALTY)
  informative_mask <- seq_len(N_FEATURES) %in% INFORMATIVE

  best_f <- Inf
  best_mask <- NULL
  second_f <- Inf
  all_masks <- expand.grid(rep(list(c(FALSE, TRUE)), N_FEATURES))
  for (i in seq_len(nrow(all_masks))) {
    mask <- as.logical(all_masks[i, ])
    f <- problem$evaluate(mask)
    if (f < best_f) {
      second_f <- best_f
      best_f <- f
      best_mask <- mask
    } else if (f < second_f) {
      second_f <- f
    }
  }

  expect_identical(best_mask, informative_mask)
  expect_true(second_f > best_f, info = "the informative mask must be the UNIQUE minimum")
  expect_equal(best_f, 0.65621827579752678, tolerance = 1e-12)
})

# =========================================================================
# (b) Penalty term monotonicity (unit-level, no engine): a scorer that
# returns the SAME value regardless of which/how-many columns are
# selected -- fitness must strictly increase with popcount whenever
# penalty > 0.
# =========================================================================

.constant_scorer <- function(X_sub, y) 5.0

test_that("the penalty term is monotonic in popcount", {
  X <- matrix(0, nrow = 4, ncol = N_FEATURES)
  y <- rep(0, 4)
  problem <- FeatureSelection$new(X, y, .constant_scorer, penalty = 0.4)

  f_one <- problem$evaluate(c(TRUE, rep(FALSE, N_FEATURES - 1)))
  f_two <- problem$evaluate(c(TRUE, TRUE, rep(FALSE, N_FEATURES - 2)))
  f_all <- problem$evaluate(rep(TRUE, N_FEATURES))

  expect_true(f_one < f_two)
  expect_true(f_two < f_all)
  expect_equal(f_one, 5.0 + 0.4 * 1 / N_FEATURES)
  expect_equal(f_two, 5.0 + 0.4 * 2 / N_FEATURES)
  expect_equal(f_all, 5.0 + 0.4 * N_FEATURES / N_FEATURES)
})

test_that("zero penalty ignores popcount", {
  X <- matrix(0, nrow = 4, ncol = N_FEATURES)
  y <- rep(0, 4)
  problem <- FeatureSelection$new(X, y, .constant_scorer, penalty = 0)

  f_one <- problem$evaluate(c(TRUE, rep(FALSE, N_FEATURES - 1)))
  f_all <- problem$evaluate(rep(TRUE, N_FEATURES))
  expect_equal(f_one, 5.0)
  expect_equal(f_all, 5.0)
})

# =========================================================================
# (c) Empty-mask documented behavior: Inf, and scorer is NEVER called.
# =========================================================================

test_that("the empty mask returns Inf and never calls scorer", {
  calls <- list()
  recording_scorer <- function(X_sub, y) {
    calls[[length(calls) + 1]] <<- dim(X_sub)
    0.0
  }

  X <- matrix(0, nrow = 4, ncol = N_FEATURES)
  y <- rep(0, 4)
  problem <- FeatureSelection$new(X, y, recording_scorer, penalty = 1)

  f <- problem$evaluate(rep(FALSE, N_FEATURES))
  expect_equal(f, Inf)
  expect_length(calls, 0)
})

test_that("the empty mask is never the minimizer among nonempty candidates", {
  X <- matrix(0, nrow = 4, ncol = N_FEATURES)
  y <- rep(0, 4)
  problem <- FeatureSelection$new(X, y, .constant_scorer, penalty = 1)
  empty_f <- problem$evaluate(rep(FALSE, N_FEATURES))
  nonempty_f <- problem$evaluate(c(TRUE, rep(FALSE, N_FEATURES - 1)))
  expect_true(nonempty_f < empty_f)
})

# =========================================================================
# (d) drop = FALSE regression: a single selected column must reach scorer
# as a MATRIX, not a bare vector -- the R analogue of M4-1 Task 6's Python
# bool-indexing trap (pinned here per this task's own brief).
# =========================================================================

test_that("a 1-column mask reaches scorer as a matrix (drop = FALSE is load-bearing)", {
  matrix_checking_scorer <- function(X_sub, y) {
    stopifnot(is.matrix(X_sub))
    stopifnot(ncol(X_sub) == 1L)
    sum(X_sub)
  }

  X <- matrix(1:12, nrow = 4, ncol = 3)
  y <- rep(0, 4)
  problem <- FeatureSelection$new(X, y, matrix_checking_scorer, penalty = 0)

  # The 1-column optimal mask: selecting column 2 alone.
  mask <- c(FALSE, TRUE, FALSE)
  expect_silent(f <- problem$evaluate(mask))
  expect_equal(f, sum(X[, 2]))
})

# =========================================================================
# (e) Scorer exceptions/conditions propagate unchanged.
# =========================================================================

test_that("a condition raised inside scorer survives to the caller unchanged (class + message)", {
  failing_scorer <- function(X_sub, y) {
    stop(structure(
      class = c("myScorerError", "error", "condition"),
      list(message = "scorer blew up on purpose", call = NULL)
    ))
  }

  X <- matrix(0, nrow = 4, ncol = N_FEATURES)
  y <- rep(0, 4)
  problem <- FeatureSelection$new(X, y, failing_scorer, penalty = 0)

  expect_error(
    problem$evaluate(c(TRUE, rep(FALSE, N_FEATURES - 1))),
    class = "myScorerError"
  )
  err <- tryCatch(
    problem$evaluate(c(TRUE, rep(FALSE, N_FEATURES - 1))),
    myScorerError = function(e) e
  )
  expect_equal(conditionMessage(err), "scorer blew up on purpose")
})

# =========================================================================
# (f) MixedTuning smoke -- (f1) Float-only through GeneticAlgorithm
# auto-dispatch, (f2) a genuinely Mixed (Float + Categorical) space
# through a hand-built gen/compound algorithm spec.
# =========================================================================

test_that("a Float-only MixedTuning instance runs via GeneticAlgorithm auto-dispatch", {
  objective <- function(x) sum((x - 1.5)^2)
  problem <- MixedTuning$new(sz_space(sz_float(-5.0, 5.0, 3)), objective)
  res <- GeneticAlgorithm$new(pop_size = 20)$run(problem, budget = 2000, seed = 1)

  expect_true(res$best_f < 1e-3)
  expect_true(all(abs(res$best_x - 1.5) < 0.05))
})

test_that("Float-only MixedTuning is deterministic across repeated runs at the same seed", {
  objective <- function(x) sum((x - 1.5)^2)
  a <- GeneticAlgorithm$new(pop_size = 20)$run(
    MixedTuning$new(sz_space(sz_float(-5.0, 5.0, 3)), objective), budget = 500, seed = 3
  )
  b <- GeneticAlgorithm$new(pop_size = 20)$run(
    MixedTuning$new(sz_space(sz_float(-5.0, 5.0, 3)), objective), budget = 500, seed = 3
  )
  expect_identical(a$best_f, b$best_f)
  expect_identical(a$best_x, b$best_x)
})

# gen/compound spec: a Float block (gen/ga-real) plus a Categorical block
# (gen/ga-cat) -- GeneticAlgorithm itself rejects a Mixed space (see its
# own docstring), so a Mixed-space MixedTuning instance is run through a
# hand-built spec, the same pattern test-oop-problem.R's own mixed
# Float+Binary test uses. spec_json (NOT TOML -- sz_solve_r_problem's own
# `spec_json` argument parses JSON only, AlgorithmSpec::from_json,
# `crates/core/src/spec.rs`).
MIXED_TUNING_COMPOUND_SPEC_JSON <- '{
  "name": "mixed-tuning-oop",
  "pop_size": 12,
  "init": {"kind": "init/uniform"},
  "boundary": {"kind": "boundary/clamp"},
  "stages": [
    {
      "generator": {
        "kind": "gen/compound",
        "blocks": [
          {"kind": "gen/ga-real", "tournament_k": 2, "pc": 0.9, "eta_c": 15.0, "eta_m": 20.0},
          {"kind": "gen/ga-cat", "tournament_k": 2, "p_c": 0.9}
        ]
      },
      "replacer": {"kind": "replace/mu-plus-lambda"}
    }
  ],
  "termination": {"budget": 200}
}'

test_that("a Float+Categorical Mixed-space MixedTuning instance gets pinned per-block types through the engine", {
  calls <- 0L
  objective <- function(x) {
    calls <<- calls + 1L
    expect_type(x, "list")
    expect_length(x, 2L)
    float_block <- x[[1]]
    cat_block <- x[[2]]
    expect_true(is.double(float_block))
    expect_true(is.integer(cat_block))
    expect_true(all(cat_block >= 0L & cat_block < 3L))
    sum(float_block^2) + sum(cat_block != 0L)
  }

  space <- sz_space(sz_float(-5.0, 5.0, 2), sz_categorical(3, 2))
  problem <- MixedTuning$new(space, objective)
  blocks <- sezgi:::.sz_space_to_blocks(problem$space())
  shim <- sezgi:::.sz_make_evaluate_shim(problem)

  r <- sezgi:::sz_solve_r_problem(MIXED_TUNING_COMPOUND_SPEC_JSON, blocks, shim, master_seed = 42, run_id = 0)

  expect_true(is.double(r$best_f))
  expect_true(calls > 0, info = "objective must have been called at least once")
})

# =========================================================================
# (h) Export surface / structural checks.
# =========================================================================

test_that("FeatureSelection and MixedTuning are exported and are Problem subclasses", {
  exported <- getNamespaceExports("sezgi")
  expect_true("FeatureSelection" %in% exported)
  expect_true("MixedTuning" %in% exported)

  fs <- FeatureSelection$new(matrix(0, 2, 2), c(0, 0), .constant_scorer)
  mt <- MixedTuning$new(sz_space(sz_float(0, 1, 1)), function(x) 0)
  expect_true(inherits(fs, "Problem"))
  expect_true(inherits(mt, "Problem"))
})
