# Pure-R algorithm-authoring surface over T5's generalized EvalSession
# ask/tell core -- the R mirror of py-sezgi's M3-4 `sezgi.Algorithm` story
# (`py-sezgi/python/sezgi/algo.py`: `Algorithm` ABC, `AlgoContext`,
# `SolveResult`, `BudgetExhausted`). Base R only (scope ruling 4: no R6) --
# `sz_algorithm()` returns a classed list; the run context (`ctx`) is a
# plain `environment` of closures, not an R6/S4 object; a batch that would
# cross the remaining budget is signalled via a custom base-R CONDITION
# CLASS (`"sz_budget_exhausted"`), not an exception object -- R's condition
# system is base R's own analogue to Python's exception hierarchy
# (`stop(structure(class = c("sz_budget_exhausted", "error", "condition"),
# ...))`, caught with `tryCatch(..., sz_budget_exhausted = function(e) ...)`).
#
# There is no savvy/Rust involvement anywhere in this file -- `sz_algorithm`/
# `sz_algo_solve` are pure R, driving any `EvalSession` object T5 already
# exposes (`sz_eval_session`/`sz_eval_session_cec2022`/`sz_eval_session_f0`)
# through its `$evaluate()`/`$evals_used()`/`$budget()`/`$best()`/`$f_opt()`/
# `$dim()`/`$bounds()`/`$finish()` methods -- exactly like `AlgoContext`
# wraps `PyEvalSession` on the Python side, but with everything (including
# the RNG) named as callables (`ctx$dim()`, not `ctx$dim`), since R has no
# property/descriptor syntax to fall back on for a uniform `$`-access
# surface.

# ---- sz_budget_exhausted condition -------------------------------------

#' Builds the custom condition [`sz_budget_exhausted`] signals -- a base-R
#' condition object of class `c("sz_budget_exhausted", "error", "condition")`,
#' the same three-class shape `simpleError()` itself builds (drop-in
#' compatible with any generic error handler), plus the extra
#' `"sz_budget_exhausted"` class the `sz_algo_solve()` driver's `tryCatch`
#' specifically catches (and swallows) to end a run cleanly -- mirroring
#' py-sezgi's `algo.BudgetExhausted` exception class 1:1 in what it is used
#' for, expressed through R's condition system instead of an exception
#' class hierarchy.
#'
#' @param message Character scalar, the condition's `$message`.
#' @return A condition object.
#' @noRd
.sz_budget_exhausted <- function(message) {
  structure(
    class = c("sz_budget_exhausted", "error", "condition"),
    list(message = message, call = NULL)
  )
}

# ---- point counting for the pre-charge budget check ---------------------

#' Counts the number of points in `x` -- a numeric matrix (rows = points) or
#' a `list` of numeric vectors, the SAME two shapes `EvalSession$evaluate()`
#' itself accepts (`sexp_to_rows()` in `src/rust/src/session.rs`) -- without
#' fully parsing `x` into rows. Used by `ctx$evaluate()` BEFORE calling
#' `session$evaluate()`, so a batch that would cross the remaining budget is
#' rejected without spending anything (mirrors `AlgoContext.evaluate`'s own
#' `len(points) > self.remaining` pre-check in `py-sezgi/python/sezgi/algo.py`,
#' checked before the underlying session is touched at all).
#'
#' A `data.frame` is also an R `list` (one element per COLUMN, not per
#' point); `session$evaluate()` itself rejects it with a message pointing at
#' the fix, and this counts it early with the same message so the point
#' count used for the pre-check is never silently wrong for that input.
#'
#' @param x A numeric matrix or a list of numeric vectors.
#' @return An integer/numeric scalar, the number of points `x` represents.
#' @noRd
.sz_count_points <- function(x) {
  if (is.matrix(x)) {
    return(nrow(x))
  }
  if (is.list(x)) {
    if (inherits(x, "data.frame")) {
      stop(paste0(
        "evaluate() does not accept a data.frame: it would be read column-wise, ",
        "not row-wise, silently producing wrong points -- pass a numeric matrix ",
        "instead, e.g. as.matrix(x) (rows = points)"
      ))
    }
    return(length(x))
  }
  stop("evaluate() expects a numeric matrix (rows = points) or a list of numeric vectors")
}

# ---- AlgoContext (an environment of closures) ----------------------------

#' Reverses the 1-based, INCLUSIVE segment `tour[i:j]` -- positions `i`
#' through `j`, both inclusive -- returning a NEW tour (`tour` itself is not
#' mutated). The classic 2-opt move: replaces edges `(tour[i-1], tour[i])`
#' and `(tour[j], tour[j+1])` with `(tour[i-1], tour[j])` and
#' `(tour[i], tour[j+1])` (the tour's own closing/wraparound edge is
#' untouched unless `i == 1` or `j == length(tour)`).
#'
#' The R mirror of py-sezgi's `AlgoContext.two_opt(tour, i, j)`
#' (`py-sezgi/python/sezgi/algo.py`) -- SAME semantics (an inclusive
#' segment reversal), shifted to r-sezgi's 1-based tour convention: Python's
#' `0 <= i <= j < len(tour)` becomes `1 <= i <= j <= length(tour)` here, and
#' R's own `tour[i:j]` slicing is ALREADY inclusive on both ends, so this is
#' a direct 1-based translation, not a re-derivation. Pure R, no RNG, no
#' Rust involvement at all (like the Python original) -- there is no FFI
#' boundary to cross for a pure index-reversal move, unlike
#' `ctx$random_permutation()` (see `.sz_algo_context()`'s own doc).
#'
#' `i == j` reverses a single-element segment (a no-op: the returned tour
#' equals `tour`). `i == 1, j == length(tour)` reverses the WHOLE tour
#' (still a no-op on tour length/validity, but exercises both boundaries at
#' once). Does not itself validate that `tour` is a permutation (a tour
#' from `ctx$random_permutation()` or an earlier `ctx$two_opt()` call
#' already is one); does not touch the evaluation budget -- call
#' `ctx$evaluate(list(result))` to score it.
#'
#' @param tour A numeric vector (1-based tour).
#' @param i,j Whole-number scalars, `1 <= i <= j <= length(tour)`.
#' @return A new numeric vector, the same length as `tour`.
#' @noRd
.sz_two_opt <- function(tour, i, j) {
  n <- length(tour)
  if (!(i >= 1 && i <= j && j <= n)) {
    stop(sprintf(
      "two_opt: i=%s, j=%s out of range for a tour of length %d (require 1 <= i <= j <= %d)",
      i, j, n, n
    ))
  }
  # `tour[0]` (not `numeric(0)`) for an empty prefix/suffix: preserves
  # `tour`'s own type (e.g. integer) instead of coercing the whole result to
  # double via `c()`.
  c(
    if (i > 1) tour[1:(i - 1)] else tour[0],
    rev(tour[i:j]),
    if (j < n) tour[(j + 1):n] else tour[0]
  )
}

#' Builds the run context a `setup()`/`step()` closure touches, wrapping
#' `session` -- the SOLE keeper of counting/best/logging, same division of
#' responsibility as py-sezgi's `AlgoContext`/`PyEvalSession` pair. Every
#' member is a CALLABLE (`ctx$dim()`, not `ctx$dim`) since R has no
#' property/descriptor syntax -- kept uniform on purpose rather than mixing
#' plain fields (for `dim`/`bounds`) with functions (for `evaluate`/`best`).
#'
#' `ctx$random_point()` draws from R's GLOBAL RNG stream (`runif()`), not a
#' private per-context generator -- `sz_algo_solve()` calls `set.seed(seed)`
#' once up front, so R's global RNG IS the ctx RNG for the whole run. This
#' is the SAME convention the hand-written pure-R algorithm examples already
#' use (e.g. `examples/r/gwo.R`'s `pop <- lapply(1:pop_size, function(i)
#' runif(dim, lo, hi))` after its own `set.seed(seed)`), not a new
#' invention -- `sz_algorithm`/`sz_algo_solve` simply codify it.
#'
#' M3-8 Task 8 widens `ctx` to also cover a permutation-typed session
#' (`sz_eval_session_tsp()`) -- the R mirror of py-sezgi's M3-8 Task 7
#' `AlgoContext` widening (`py-sezgi/python/sezgi/algo.py`): `ctx$kind()`
#' (`"float"` or `"permutation"`), `ctx$n()` (the SAME value as `ctx$dim()`,
#' under a second, kind-neutral name -- `n` reads naturally for a tour,
#' `dim` does not), `ctx$random_permutation()` (delegates to
#' `session$random_permutation()`, which draws from the SESSION's own
#' seeded Rust-side RNG stream, NOT R's global stream -- a permutation-typed
#' run must be reproducible from the same `seed` the same way a float-typed
#' one is, so the draw comes from the session, mirroring
#' `AlgoContext.random_permutation()`'s own Rust-side-not-Python's-`random`
#' choice exactly), and `ctx$two_opt(tour, i, j)` (see `.sz_two_opt()`'s own
#' doc -- pure R, no RNG, available regardless of `ctx$kind()`).
#' `ctx$random_point()` now checks `ctx$kind()` FIRST, raising a clear error
#' naming the actual kind instead of relying on `session$bounds()`'s own
#' (less specific) rejection -- mirrors `AlgoContext.random_point()`'s own
#' "small robustness addition" from the same task on the Python side.
#'
#' @param session An `EvalSession` object (from `sz_eval_session()` or a
#'   sibling constructor).
#' @return An `environment` exposing `evaluate`, `evals_used`, `budget`,
#'   `remaining`, `best`, `f_opt`, `dim`, `bounds`, `random_point`, `kind`,
#'   `n`, `random_permutation`, `two_opt`.
#' @importFrom stats runif
#' @noRd
.sz_algo_context <- function(session) {
  ctx <- new.env(parent = emptyenv())

  ctx$budget <- function() session$budget()
  ctx$evals_used <- function() session$evals_used()
  ctx$remaining <- function() session$budget() - session$evals_used()
  ctx$best <- function() session$best()
  ctx$f_opt <- function() session$f_opt()
  ctx$dim <- function() session$dim()
  ctx$bounds <- function() session$bounds()
  ctx$kind <- function() session$kind()
  ctx$n <- function() session$dim()  # same value as dim(); see doc above

  ctx$random_point <- function() {
    k <- session$kind()
    if (k != "float") {
      stop(sprintf(
        "random_point() is only available for float-typed sessions (this session's kind() is \"%s\")",
        k
      ))
    }
    b <- session$bounds()
    runif(session$dim(), b[1], b[2])
  }

  ctx$random_permutation <- function() session$random_permutation()

  ctx$two_opt <- function(tour, i, j) .sz_two_opt(tour, i, j)

  ctx$evaluate <- function(points) {
    remaining <- session$budget() - session$evals_used()
    n <- .sz_count_points(points)
    if (n > remaining) {
      stop(.sz_budget_exhausted(sprintf(
        "batch of %d exceeds remaining budget %d", n, remaining
      )))
    }
    session$evaluate(points)
  }

  ctx
}

# ---- sz_algorithm ---------------------------------------------------------

#' Declares a pure-R metaheuristic algorithm to run over an `EvalSession`.
#'
#' Base R only (no R6/S4): the result is a plain classed `list` holding the
#' two closures and a name -- there is no method dispatch, subclassing, or
#' mutable state of its own; all run state lives in the `ctx` environment
#' `sz_algo_solve()` builds fresh for each call. Mirrors py-sezgi's
#' `sezgi.Algorithm` ABC (subclass, implement `setup()`/`step()`), expressed
#' as two plain closures instead of two abstract methods.
#'
#' @param setup A function `function(ctx)`, called once at the start of a
#'   run.
#' @param step A function `function(ctx)`, called repeatedly while budget
#'   remains. Each call MUST evaluate at least one point (via
#'   `ctx$evaluate()`) or let `ctx$evaluate()` raise `sz_budget_exhausted` --
#'   a `step()` that returns having consumed no budget is a driver error
#'   (see `sz_algo_solve()`).
#' @param name Character scalar, the algorithm's label -- recorded as the
#'   `algo` field of `sz_algo_solve()`'s result and (when the session has
#'   IOH logging enabled) the IOH archive's algorithm name. Default
#'   `"custom"`.
#' @returns A classed `list` (`"sz_algorithm"`) with elements `setup`,
#'   `step`, `name`.
#' @export
sz_algorithm <- function(setup, step, name = "custom") {
  structure(
    list(setup = setup, step = step, name = name),
    class = "sz_algorithm"
  )
}

# ---- sz_algo_solve ---------------------------------------------------------

#' Drives an `sz_algorithm` over an `EvalSession` to exhaustion.
#'
#' Semantics MIRROR py-sezgi's `Algorithm.solve()` exactly
#' (`py-sezgi/python/sezgi/algo.py`):
#' \itemize{
#'   \item `set.seed(seed)` is called ONCE, up front -- R's global RNG IS
#'     the `ctx` RNG for the whole run (see `.sz_algo_context()`'s doc);
#'     `algo$setup`/`algo$step` calling `ctx$random_point()` (or drawing
#'     from R's RNG directly, e.g. `runif()`/`sample()`) are both driven by
#'     this one seeded stream, so a run is fully reproducible from `seed`
#'     alone.
#'   \item The driver calls `algo$setup(ctx)` once, then `algo$step(ctx)`
#'     repeatedly while `ctx$remaining() > 0`. A `step()` call that leaves
#'     `ctx$evals_used()` UNCHANGED is a driver error (`stop()` with a
#'     message containing `"consumed no budget"`) -- a step must evaluate at
#'     least one point or let `ctx$evaluate()` raise `sz_budget_exhausted`.
#'   \item `ctx$evaluate(points)` signals the condition class
#'     `"sz_budget_exhausted"` when `points` would exceed the remaining
#'     budget -- checked BEFORE the session is touched, so a rejected batch
#'     spends nothing. `sz_algo_solve()`'s own `tryCatch` catches ONLY this
#'     condition class (wrapping both `algo$setup()` and the `step()` loop),
#'     ending the run cleanly; any OTHER error `setup()`/`step()` raises
#'     (including the "consumed no budget" guard above) propagates out of
#'     `sz_algo_solve()` uncaught.
#'   \item A run that ends having evaluated NOTHING at all (`ctx$best()` is
#'     `NULL`) errors.
#'   \item `session$finish()` is GUARANTEED to run EXACTLY ONCE, on every
#'     exit path -- success, the "consumed no budget"/"no evaluations"
#'     guards, or any error `setup()`/`step()` itself raises -- via
#'     `on.exit(..., add = TRUE)` registered before the run starts (base
#'     R's `finally` equivalent). This mirrors py-sezgi's own `try`/
#'     `finally` structure, which a Task 4 (M3-4) final-review finding
#'     added specifically so a run that dies mid-flight does not silently
#'     lose its IOH archive; applied here from the start rather than
#'     rediscovered later.
#' }
#'
#' @param algo An `sz_algorithm` (from `sz_algorithm()`).
#' @param session An `EvalSession` object (from `sz_eval_session()` or a
#'   sibling constructor) -- NOT yet finished; `sz_algo_solve()` finishes it.
#' @param seed Integer/numeric scalar, the master RNG seed. Passed straight
#'   to `set.seed()`; also recorded verbatim as the result's `seed` field
#'   (purely documentation of the seed used -- it plays no other role once
#'   `set.seed()` has run).
#' @returns A named `list` with fields `algo` (character, `algo$name`),
#'   `seed`, `budget` (`session$budget()`), `evals_used`, `best_x`
#'   (numeric vector), `best_f` (numeric scalar), `f_opt` (numeric scalar,
#'   or `NULL` if the problem has no known optimum -- e.g. an f0 session),
#'   `gap` (`best_f - f_opt`, or `NULL` iff `f_opt` is `NULL`) -- mirrors
#'   py-sezgi's `SolveResult` field names exactly.
#' @export
sz_algo_solve <- function(algo, session, seed) {
  if (!inherits(algo, "sz_algorithm")) {
    stop("algo must be an sz_algorithm (see sz_algorithm())")
  }

  set.seed(seed)
  algo_name <- algo$name
  budget <- session$budget()
  ctx <- .sz_algo_context(session)

  # Guarantees session$finish() runs exactly once, on every exit path from
  # this point on (normal return OR any error propagating past the
  # tryCatch below) -- base R's `finally` equivalent to py-sezgi's
  # try/finally. `add = TRUE` is not load-bearing here (this is the first
  # on.exit() call in the function), but keeps the call self-documenting
  # about not clobbering any future addition.
  on.exit(session$finish(), add = TRUE)

  tryCatch(
    {
      algo$setup(ctx)
      while (ctx$remaining() > 0) {
        before <- ctx$evals_used()
        algo$step(ctx)
        if (identical(ctx$evals_used(), before)) {
          stop(paste0(
            "step() consumed no budget; a step must evaluate at least one ",
            "point or raise sz_budget_exhausted"
          ))
        }
      }
    },
    sz_budget_exhausted = function(e) NULL
  )

  best <- ctx$best()
  if (is.null(best)) {
    stop("run ended with no evaluations at all")
  }
  f_opt <- ctx$f_opt()
  best_f <- best$f

  list(
    algo = algo_name,
    seed = seed,
    budget = budget,
    evals_used = ctx$evals_used(),
    best_x = best$x,
    best_f = best_f,
    f_opt = f_opt,
    gap = if (is.null(f_opt)) NULL else best_f - f_opt
  )
}
