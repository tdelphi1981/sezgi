# Data recipes (M4-2 Task 6) -- the "optimize against data" door: two
# `Problem` subclasses that bind a search space to a caller-supplied
# dataset/objective, so a data-analysis workflow reaches the engine (and
# every built-in wrapper class, `R/builtins.R`) through the SAME
# `sezgi::Problem` R6 base (`R/problem.R`) every other M4-2 class uses --
# no bespoke glue code per dataset. Mirrors py-sezgi's `sezgi.recipes`
# (`py-sezgi/python/sezgi/recipes.py`, M4-1 Task 6) exactly in semantics
# (repo authority per this task's own brief).
#
# `FeatureSelection`: wraps a `sz_binary(n_features)` search over a 2D
# dataset's columns, scored by a caller-supplied `scorer(X_sub, y) ->
# numeric(1)`.
#
# `MixedTuning`: the general "tune anything" door -- binds a caller-
# supplied `objective(x) -> numeric(1)` over ANY declared `sz_space(...)`
# (a single block or a multi-block space), with zero dataset-specific
# assumptions.
#
# Pure R; no ML-package dependency anywhere in this file (see
# `FeatureSelection`'s own doc for the sklearn-shaped scorer note, kept as
# a COMMENT, never an actual dependency).

#' @importFrom R6 R6Class
NULL

# ---- FeatureSelection -----------------------------------------------------

#' Binary feature-selection search over a 2D dataset's columns.
#'
#' Mirrors `sezgi.recipes.FeatureSelection`
#' (`py-sezgi/python/sezgi/recipes.py`) exactly.
#'
#' `space()` is `sz_space(sz_binary(n_features))`: a genotype is a
#' length-`n_features` logical mask (per [Problem]'s own genotype
#' conversion table -- a Binary block converts to a logical vector),
#' `TRUE` selecting a column.
#'
#' MINIMIZE convention: `scorer(X_sub, y)` must return a LOWER-is-better
#' number (e.g. an error/loss/residual metric, NOT accuracy/R^2/any
#' higher-is-better score) -- `evaluate()` never negates or inverts it.
#' An sklearn-shaped cross-validated ACCURACY scorer would need to be
#' wrapped to flip its sign before it could be used here (this package
#' has no ML-package dependency of its own -- the sketch below is a
#' COMMENT only, never an actual call anywhere in this file):
#'
#' \preformatted{
#' # scorer <- function(X_sub, y) {
#' #   # a cross-validated accuracy is HIGHER-is-better -- negate it so
#' #   # lower is better, matching this class's minimize convention.
#' #   -mean(cross_val_accuracy(X_sub, y))
#' # }
#' # FeatureSelection$new(X, y, scorer, penalty = 0.01)
#' }
#'
#' `evaluate(mask)` = `scorer(X[, mask, drop = FALSE], y) + penalty *
#' (popcount(mask) / n_features)` -- the penalty term is a fraction of
#' the FULL feature count (`popcount / n_features`, not a raw popcount),
#' so it stays on a comparable scale to `scorer`'s own output regardless
#' of `n_features`, and its size is `penalty` at the all-features mask,
#' `0` at the empty mask. Larger `penalty` biases the search toward
#' smaller feature subsets; `penalty = 0` (default) is a pure
#' `scorer`-value search with no feature-count preference of its own.
#'
#' `X[, mask, drop = FALSE]`: the `drop = FALSE` is LOAD-BEARING -- without
#' it, a single-`TRUE` mask would subset `X` down to a plain vector (R's
#' single-bracket auto-drop for a one-column result), silently changing
#' `X_sub`'s shape from a matrix to a vector depending on the mask's own
#' popcount and breaking any `scorer` that assumes a consistent
#' `(n_samples, k)` matrix shape (e.g. anything calling `ncol(X_sub)` or
#' doing matrix algebra on it) -- the R analogue of M4-1 Task 6's Python
#' bool-indexing trap, pinned here with its own regression test
#' (`tests/testthat/test-oop-recipes.R`).
#'
#' EMPTY MASK (`popcount == 0`): `scorer` is NEVER called -- `X[, mask,
#' drop = FALSE]` would be an `(n_samples, 0)` matrix, and most real
#' scorers (a model fit, a distance-correlation-like statistic, ...)
#' cannot meaningfully score zero columns; the `popcount == 0` case is
#' instead handled directly, as a DOCUMENTED SENTINEL:
#' `evaluate(rep(FALSE, n_features)) == Inf`. `Inf` was chosen over, say,
#' a large-but-finite number because it is unambiguous (no
#' dataset-dependent magnitude to pick or accidentally beat by a
#' legitimately bad but non-empty selection) and it sorts correctly under
#' every consumer's own comparison (`<`) with no special-casing needed --
#' the empty mask is simply never the minimizer of any run that has at
#' least one non-empty candidate in its population, which every
#' population-based search here always does.
#'
#' @section Methods:
#' \describe{
#'   \item{`new(X, y, scorer, penalty = 0)`}{`X` a 2D matrix/data.frame
#'     (coerced via `as.matrix()`), `y` the target (any vector/type
#'     `scorer` accepts), `scorer(X_sub, y) -> numeric(1)` a
#'     caller-supplied minimize-convention scoring function, `penalty` a
#'     numeric scalar (default `0`, no feature-count preference).}
#'   \item{`space()`}{`sz_space(sz_binary(n_features))`, where
#'     `n_features = ncol(X)`.}
#'   \item{`evaluate(mask)`}{See the class doc above.}
#' }
#'
#' @export
FeatureSelection <- R6::R6Class("FeatureSelection",
  inherit = Problem,
  public = list(
    X = NULL,
    y = NULL,
    scorer = NULL,
    penalty = NULL,
    n_features = NULL,

    initialize = function(X, y, scorer, penalty = 0) {
      if (is.null(dim(X)) || length(dim(X)) != 2L) {
        stop(sprintf(
          "FeatureSelection: X must be 2D, got dim %s",
          paste(dim(X), collapse = " x ")
        ))
      }
      self$X <- as.matrix(X)
      self$y <- y
      self$scorer <- scorer
      self$penalty <- as.double(penalty)
      self$n_features <- ncol(self$X)
      invisible(self)
    },

    space = function() {
      sz_space(sz_binary(self$n_features))
    },

    evaluate = function(mask) {
      mask <- as.logical(mask)
      k <- sum(mask)
      if (k == 0L) {
        return(Inf)
      }
      score <- self$scorer(self$X[, mask, drop = FALSE], self$y)
      score + self$penalty * (k / self$n_features)
    }
  )
)

# ---- MixedTuning -----------------------------------------------------------

#' The general "tune anything" door: binds a caller-supplied
#' `objective(x) -> numeric(1)` over ANY declared `sz_space(...)` -- a
#' single block (`sz_float(...)`, `sz_categorical(...)`, ...) or a
#' multi-block space. `evaluate(x)` delegates to `objective` UNCHANGED --
#' `x`'s exact shape follows [Problem]'s own genotype conversion table
#' (a bare converted value for a single-block space, an unnamed `list` of
#' per-block converted values in `space()`'s own block order for a
#' multi-block one).
#'
#' Mirrors `sezgi.recipes.MixedTuning`
#' (`py-sezgi/python/sezgi/recipes.py`) exactly.
#'
#' Note [GeneticAlgorithm] auto-dispatches only over a SINGLE-kind space
#' (all-Float, all-Binary, ...) -- it raises a clear error naming the
#' limitation on a genuinely Mixed space (see its own doc); a mixed-space
#' `MixedTuning` instance is instead run via a hand-built `gen/compound`
#' algorithm spec passed to `sezgi:::sz_solve_r_problem()` directly (see
#' `tests/testthat/test-oop-recipes.R`'s own worked example), or
#' `evaluate()`d directly for a non-engine use (grid search, a script
#' sanity check, ...). A single-kind `MixedTuning` space (e.g. Float-only,
#' tuning several continuous hyperparameters at once) runs through
#' [GeneticAlgorithm] exactly like any other single-kind [Problem].
#'
#' @section Methods:
#' \describe{
#'   \item{`new(space, objective)`}{`space` an `sz_space(...)` object,
#'     `objective(x) -> numeric(1)` a caller-supplied minimize-convention
#'     function.}
#'   \item{`space()`}{Returns the constructor's own `space`, unchanged.}
#'   \item{`evaluate(x)`}{Returns `objective(x)`, unchanged.}
#' }
#'
#' @export
MixedTuning <- R6::R6Class("MixedTuning",
  inherit = Problem,
  public = list(
    space_obj = NULL,
    objective = NULL,

    initialize = function(space, objective) {
      self$space_obj <- space
      self$objective <- objective
      invisible(self)
    },

    space = function() {
      self$space_obj
    },

    evaluate = function(x) {
      self$objective(x)
    }
  )
)
