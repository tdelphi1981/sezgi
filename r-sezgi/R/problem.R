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
#
# Fix round 1 (controller review item 9): the shim is now a BATCH shim,
# called ONCE PER GENERATION with the whole population, mirroring
# py-sezgi's `Problem._to_native()` (`py-sezgi/python/sezgi/problem.py`),
# which always builds `vectorized=True` and calls `self.batch_evaluate`
# once per generation with the whole population -- it used to call
# `prob$evaluate(x)` once per INDIVIDUAL, a real mirror gap against the
# Python bridge this milestone is meant to match.

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
#' `batch_evaluate(xs)`'s own `xs` is a `list` of `x` values (one per
#' individual, each in the SAME per-`x` shape as `evaluate(x)`'s own `x`
#' above), matching py-sezgi's `sezgi.Problem.batch_evaluate` input shape
#' exactly (`py-sezgi/python/sezgi/problem.py`) -- the Python side is the
#' authority for this shape, not an independent R design.
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
#'   \item{`batch_evaluate(xs)`}{`xs` -> a numeric vector, one entry per
#'     `x` in `xs`, in the SAME order (default: loop `self$evaluate`,
#'     mirroring `sezgi.Problem.batch_evaluate`'s own default
#'     `[self.evaluate(x) for x in xs]` exactly). Override for a
#'     vectorized objective; called ONCE PER GENERATION with the WHOLE
#'     population as `xs` (see `.sz_make_evaluate_shim`'s own doc).}
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
    },

    batch_evaluate = function(xs) {
      vapply(xs, self$evaluate, numeric(1))
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

#' Builds the batch-evaluate shim `sz_solve_r_problem()`'s own `evaluate`
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
#' Called ONCE PER GENERATION with `pop` -- a `list` of length n (one entry
#' per individual, matching `pop`'s own generation size), each entry itself
#' a `list` of per-block typed R vectors (one entry per block, in
#' `prob$space()`'s block order, each already converted to its natural R
#' type by the Rust side -- see [Problem]'s own doc for the block -> R type
#' table). Reshapes EACH individual to the `evaluate(x)`/`batch_evaluate
#' (xs)` contract (bare for a single block, an unnamed list for more than
#' one) and dispatches ONCE to `prob$batch_evaluate(xs)` -- mirroring
#' py-sezgi's `_to_native()`, which always builds `vectorized=True` and
#' calls `self.batch_evaluate` once per generation with the whole
#' population (`py-sezgi/python/sezgi/problem.py`). A subclass that only
#' overrides `evaluate` still works correctly (and, per-generation, in one
#' R call rather than n) through [Problem]'s own default `batch_evaluate`
#' (loops `self$evaluate`); a subclass that overrides `batch_evaluate`
#' directly is honored -- its return value (a numeric vector of length n,
#' in `xs`'s own order) is returned to Rust UNCHANGED.
#'
#' @param prob A `Problem` subclass instance.
#' @returns A closure of one argument (`pop`), returning a numeric vector
#'   of length `length(pop)`.
#' @noRd
.sz_make_evaluate_shim <- function(prob) {
  force(prob)
  function(pop) {
    xs <- lapply(pop, function(blocks) {
      if (length(blocks) == 1L) blocks[[1L]] else blocks
    })
    prob$batch_evaluate(xs)
  }
}
