# R-callable Problem bridge (M4-2 Task 2) -- the R mirror of py-sezgi's
# subclassable `sezgi.Problem` ABC (`py-sezgi/python/sezgi/problem.py`,
# M4-1 Task 1), closing the "R callable-objective sessions" deferral
# (docs/DECISIONS.md:839-845). Subclass `Problem`
# (`R6::R6Class(..., inherit = sezgi::Problem, ...)`), override
# `evaluate(x)`/`space()`, and solve it via the internal Rust entry point
# `sz_solve_r_problem()` (`r-sezgi/src/rust/src/solve.rs`) -- this task's
# own tests call it directly (`sezgi:::sz_solve_r_problem`), building its
# `blocks`/`evaluate` arguments from `.sz_space_to_blocks()`/
# `.sz_make_evaluate_shim()` by hand; a convenience `$run()`-style wrapper
# is left to a later task, the same way py-sezgi's own
# `_sezgi.from_callable_spaced` stayed internal in M4-1 Task 1.
#
# `.sz_make_evaluate_shim(prob)` exists because `FunctionSexp::call`
# evaluates the R side in `R_GlobalEnv`, not this package's namespace
# (research doc, `docs/superpowers/research/2026-09-02-r-class-front-
# door.md` §C4) -- constructing the shim closure HERE (inside this file,
# i.e. lexically inside the package) means its own enclosure IS the
# package namespace, so any future free-variable lookup inside it would
# resolve correctly regardless of where `Rf_eval` runs it from.

#' @importFrom R6 R6Class
NULL

# ---- Problem R6 base ----------------------------------------------------

#' Subclassable R6 base for a search-space problem.
#'
#' Mirrors `sezgi.Problem` (`py-sezgi/python/sezgi/problem.py`) exactly:
#' subclass and override `evaluate(x)` and `space()` (both required -- the
#' defaults below `stop()` with a "not implemented" message); `optimum()`
#' is an optional override, default `NULL`.
#'
#' Genotype -> R conversion table for `evaluate(x)`'s own `x` argument
#' (PINNED; mirrors M4-1's pinned block -> Python table exactly, see
#' `py-sezgi/python/sezgi/problem.py`'s own module doc):
#'
#' \describe{
#'   \item{Float block}{a numeric (double) vector}
#'   \item{Int block}{an integer vector}
#'   \item{Categorical block}{an integer vector of category INDICES `0..k`, not labels}
#'   \item{Binary block}{a logical vector}
#'   \item{Permutation block}{an integer vector, 0-based}
#' }
#'
#' A single-block space's `x` is that one block's own converted value,
#' passed BARE. A multi-block space's `x` is an (unnamed) R `list` of
#' per-block converted values, in `space()`'s own block order.
#'
#' Methods:
#' \describe{
#'   \item{`evaluate(x)`}{`x` -> a numeric scalar, the fitness/objective
#'     value at `x`. Subclasses MUST override this. See the class doc's
#'     conversion table above for `x`'s exact shape. Default: `stop()`s
#'     with "not implemented".}
#'   \item{`space()`}{Declares the search space. Subclasses MUST override
#'     this to return an `sz_space(...)` object. Default: `stop()`s with
#'     "not implemented".}
#'   \item{`optimum()`}{The problem's known optimum (a numeric scalar), or
#'     `NULL` (the default) if it has none. Optional override.}
#' }
#'
#' @export
Problem <- R6::R6Class("Problem",
  public = list(
    evaluate = function(x) {
      stop("Problem$evaluate() is not implemented -- subclass Problem and override it")
    },

    space = function() {
      stop("Problem$space() is not implemented -- subclass Problem and override it")
    },

    optimum = function() {
      NULL
    }
  )
)

# ---- sz_as_problem --------------------------------------------------------

#' Accepts a `Problem` subclass instance.
#'
#' Mirrors py-sezgi's `as_native_problem()`
#' (`py-sezgi/python/sezgi/problem.py`) in spirit -- unlike py-sezgi,
#' r-sezgi builds no separate "native problem handle" type to convert to
#' (there is no `_sezgi.Problem` counterpart), so a `Problem` subclass
#' instance already IS what this task's Rust bridge needs; this function's
#' job is purely to give a friendly, dedicated error for the "not a
#' Problem" case, at the point of use rather than deep inside the solve
#' machinery.
#'
#' @param obj An object to check.
#' @returns `obj`, unchanged, if it inherits `"Problem"`.
#' @export
sz_as_problem <- function(obj) {
  if (!inherits(obj, "Problem")) {
    stop(sprintf(
      "expected a Problem subclass instance (see sezgi::Problem), got %s",
      paste(class(obj), collapse = "/")
    ))
  }
  obj
}

# ---- evaluate shim (internal) --------------------------------------------

#' Builds the `evaluate` shim `sz_solve_r_problem()`'s own `evaluate`
#' parameter expects.
#'
#' Constructed INSIDE the package namespace on purpose (see this file's
#' own module doc, and research doc §C4): `FunctionSexp::call` evaluates in
#' `R_GlobalEnv`, so a closure looked up BY NAME from inside a callback
#' running there would not resolve a package-internal function -- but this
#' closure's own BODY resolves free variables lexically via its OWN
#' enclosure (the namespace, since it is defined here), independent of
#' where it is invoked from.
#'
#' Called once PER INDIVIDUAL with `blocks` -- a `list` of per-block typed
#' R vectors, one entry per block, in `prob$space()`'s block order, each
#' already converted to its natural R type by the Rust side (see
#' [Problem]'s own doc for the block -> R type table). Reshapes to the
#' `evaluate(x)` contract (bare for a single block, an unnamed list for
#' more than one) and dispatches to `prob$evaluate(x)`.
#'
#' @param prob A `Problem` subclass instance.
#' @returns A closure of one argument (`blocks`).
#' @noRd
.sz_make_evaluate_shim <- function(prob) {
  force(prob)
  function(blocks) {
    x <- if (length(blocks) == 1L) blocks[[1L]] else blocks
    prob$evaluate(x)
  }
}
