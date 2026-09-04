# R6 Algorithm / PopulationAlgorithm / LocalSearch bases --
# the R mirror of py-sezgi's engine-hosted class-first authoring surface
# (`py-sezgi/python/sezgi/algorithm.py`). A subclass
# implements `generate(pop, ctx)` (REQUIRED) and may optionally override
# `initialize_population(n, space, ctx)` (RULING 8 -- NOT `initialize`,
# which R6 reserves for the constructor; a documented divergence from
# Python's `initialize(n, ctx)` name) and `validate_space(space)`; `run()`
# wires the subclass instance through `sz_solve_r_generator`/
# `sz_solve_r_generator_bbob` bridge and `sz_as_problem`/`Problem`.
#
# Hook taxonomy modeled after pymoo 0.6.2 (Apache-2.0) and jMetal v7.5
# (MIT) public APIs, same attribution as `py-sezgi/python/sezgi/
# algorithm.py`'s own module docstring (no code copied from either
# source, naming/hook-shape conventions only).
#
# ---- `pop`/`ctx` shape handed to a user's `generate`/`select`/`vary`/
# `neighbor` method (PINNED, this task's own design decision -- see the
# task-4 report for the full rationale) ----------------------------------
#
# `pop` is `list(x = <...>, f = <numeric>)`:
#   - `f`: a numeric vector of length n, the CURRENT population's fitness
#     values (T3's own `fitness` argument, unchanged).
#   - `x`: for a SINGLE Float-block space, an n x dim numeric MATRIX (rows
#     = individuals) -- the ergonomic case (`pop$x[i, ]`, `ncol(pop$x)`,
#     matching the research doc's own §B8 Variant A sketch). For every
#     OTHER space shape (multi-block, or a single NON-Float block --
#     Int/Categorical/Binary/Permutation), `x` is the raw (unnamed) `list`
#     of individuals T3's own `pop` argument already is (bare-or-
#     per-block-list per individual, per T2/T3's pinned single-vs-multi-
#     block convention) -- UNCHANGED, no reshaping. This is a DELIBERATE
#     R-idiomatic divergence from py-sezgi's `PopView` (`pop.individuals`
#     is ALWAYS a plain Python list, never a numpy matrix): R's own
#     row-major `matrix` type is the natural vectorized-code shape for the
#     common continuous case, and every `sz_algorithm`/pure-R example in
#     this repo already favors matrix/vector idioms over per-element
#     loops where practical.
#
# A `generate`/`vary`/`neighbor` return value mirrors this: an n_offspring
# x dim MATRIX (only meaningful when `pop$x` was itself a matrix) OR a
# plain `list` of x-values (bare-or-per-block-list per individual, the
# SAME convention `pop$x` falls back to for every non-single-Float-block
# space) -- `.sz_offspring_to_raw()` converts either shape back to the raw
# list-of-individuals T3's own bridge expects. `select()`/`vary()` always
# work with individual VALUES extracted via `.sz_pop_at()` (a matrix row
# or a list element, transparently), so a `PopulationAlgorithm` subclass
# never needs to branch on `is.matrix(pop$x)` itself.
#
# `ctx` is an `environment` with `$rng` (the ALREADY-WRAPPED `SzRng`
# handle -- this file's own shims call `.savvy_wrap_SzRng(rng)` on T3's
# raw external pointer BEFORE the user ever sees it, per T3's own report
# and research doc §C4's shim-in-namespace rationale: `FunctionSexp::call`
# evaluates in `R_GlobalEnv`, so `.savvy_wrap_SzRng` must be resolved from
# INSIDE the package namespace, i.e. by a shim closure built here, not by
# name from Rust), `$iteration` (a double, T3's own `iteration` argument,
# unchanged -- always `0` for an `initialize_population` call), `$space`
# (the block-descriptor `list` `.sz_space_to_blocks()` already produces --
# the SAME shape `validate_space`'s own `space` argument uses, and the
# SAME shape T3's raw `validate_space(blocks)` callback receives, reused
# rather than inventing a second space representation).
#
# ---- override detection (RULING 8) --------------------------------------
#
# `initialize_population`/`validate_space` are detected as overridden via
# `!identical(self[[name]], Algorithm$public_methods[[name]], ignore.
# environment = TRUE)`. R6 rebinds every method's ENCLOSING ENVIRONMENT
# per instance (so `self$initialize_population`'s environment is never
# literally `identical()` to the un-bound `Algorithm$public_methods$
# initialize_population`'s, even when NOT overridden -- verified live: a
# bare `identical(self$foo, Base$public_methods$foo)`, with no
# `ignore.environment`, is FALSE in every case, overridden or not, making
# it useless here) but does NOT rewrite a method's `body`/`formals` when a
# subclass does not redeclare it -- so `identical(f, g, ignore.environment
# = TRUE)` (comparing body + formals only, a documented base-R parameter
# since R 3.1.0) is TRUE iff the resolved method is, at the CLASS level,
# the exact same function object `Algorithm` itself declared, regardless
# of how many intermediate classes (e.g. `PopulationAlgorithm`) sit
# between the instance's own class and `Algorithm` in the inheritance
# chain -- verified live across a 3-level chain (`MyDE` ->
# `PopulationAlgorithm` -> `Algorithm`, neither intermediate level
# overriding `initialize_population`: still detected as NOT overridden).
# This was chosen over walking a class GENERATOR's own `public_methods`
# chain (the other candidate the brief names) because a running instance
# (`self`, inside `run()`) has no reliable built-in back-reference to its
# own generator object (R6 does not store one on `self` by default), while
# `self[[name]]` is always available; comparing against `Algorithm$
# public_methods[[name]]` specifically (not `PopulationAlgorithm`'s, even
# for a `PopulationAlgorithm` subclass) matches py-sezgi's own precedent
# exactly (`type(self).initialize is not Algorithm.initialize`, ALWAYS
# against the top base class).
#
# ---- log_dir (MIRRORS M4-1 RULING B) -------------------------------------
#
# Python's `sezgi.Algorithm.run(..., log_dir=...)` wires IOH logging
# through `_sezgi.solve_with_py_generator`'s own `log_dir` parameter.
# `sz_solve_r_generator`/`sz_solve_r_generator_bbob` (`r-sezgi/src/rust/
# src/solve.rs`) have NO `log_dir` parameter at all -- IOH logging was
# deliberately left out of scope for that bridge. There is therefore NO
# R-side logging path reachable from either entry point to wire `run()`'s
# own `log_dir` through -- this HONESTLY REJECTS it (a clear error, not a
# silently-swallowed parameter), mirroring the "Callable problem +
# log_dir -> ValueError" precedent elsewhere in this project in spirit
# (an explicit error instead of silently ignoring the parameter).

#' @importFrom R6 R6Class
NULL

# ---- pop view / offspring conversion helpers (internal) -----------------

#' Builds the `pop` view (`list(x, f)`) a user's `generate`/`select`/
#' `vary`/`neighbor` method sees -- see this file's own module doc for the
#' exact matrix-vs-list rule.
#'
#' @param raw_pop T3's own raw `pop` argument (a `list` of n individuals,
#'   bare-or-per-block-list each).
#' @param fitness T3's own raw `fitness` argument (a numeric vector).
#' @param space_blocks The block-descriptor `list` (`.sz_space_to_blocks()`
#'   shape) for the space being solved.
#' @returns `list(x, f)`.
#' @noRd
.sz_pop_view <- function(raw_pop, fitness, space_blocks) {
  x <- if (length(space_blocks) == 1L && identical(space_blocks[[1L]]$type, "float")) {
    do.call(rbind, raw_pop)
  } else {
    raw_pop
  }
  list(x = x, f = fitness)
}

#' @param pop A `pop` view (from `.sz_pop_view()`).
#' @returns The number of individuals in `pop`.
#' @noRd
.sz_pop_n <- function(pop) {
  # `as.double()`: `ctx$rng$next_below(n)` (SzRng, `r-sezgi/src/rust/src/
  # solve.rs`) requires `n` to be a double SEXP -- `nrow()`/`length()` both
  # return an R integer, which savvy rejects outright ("must be double, not
  # integer") rather than silently coercing.
  as.double(if (is.matrix(pop$x)) nrow(pop$x) else length(pop$x))
}

#' @param pop A `pop` view (from `.sz_pop_view()`).
#' @param i A 1-based individual index.
#' @returns That individual's own x-value (a numeric vector, for the
#'   matrix case, or the raw bare-or-per-block-list value, for the list
#'   case).
#' @noRd
.sz_pop_at <- function(pop, i) {
  if (is.matrix(pop$x)) pop$x[i, ] else pop$x[[i]]
}

#' Converts a `generate`/`vary`/`neighbor`-produced offspring value (an
#' n_offspring x dim matrix, or a plain `list` of x-values) back to the raw
#' (unnamed) `list`-of-individuals shape T3's own bridge expects.
#'
#' @param off A matrix or a `list`.
#' @returns A `list`.
#' @noRd
.sz_offspring_to_raw <- function(off) {
  if (is.matrix(off)) {
    lapply(seq_len(nrow(off)), function(i) off[i, ])
  } else if (is.list(off)) {
    off
  } else {
    stop(
      "generate()/vary()/neighbor() must return a matrix (n_offspring x dim) ",
      "or a list of offspring x-values"
    )
  }
}

#' Builds the `ctx` environment T3's raw `rng`/`iteration` arguments (plus
#' this task's own `space_blocks`) are assembled into.
#'
#' @param rng T3's own raw `rng` argument (an un-wrapped `SzRng` external
#'   pointer).
#' @param iteration T3's own raw `iteration` argument (a double).
#' @param space_blocks The block-descriptor `list` for the space being
#'   solved.
#' @returns An `environment` with `$rng` (WRAPPED, `.savvy_wrap_SzRng(rng)`
#'   -- see this file's own module doc for why the wrapping happens HERE),
#'   `$iteration`, `$space`.
#' @noRd
.sz_make_ctx <- function(rng, iteration, space_blocks) {
  ctx <- new.env(parent = emptyenv())
  ctx$rng <- .savvy_wrap_SzRng(rng)
  ctx$iteration <- iteration
  ctx$space <- space_blocks
  ctx
}

# ---- override detection (internal) ---------------------------------------

#' Whether `self[[name]]` was overridden relative to `Algorithm`'s own
#' declared default -- see this file's own module doc for the exact
#' spelling and why it was chosen.
#'
#' @param self An R6 instance.
#' @param name Character scalar, the method name.
#' @param base_fn `Algorithm$public_methods[[name]]` -- ALWAYS `Algorithm`'s
#'   own entry, never an intermediate subclass's.
#' @returns `TRUE`/`FALSE`.
#' @noRd
.sz_algo_overridden <- function(self, name, base_fn) {
  !identical(self[[name]], base_fn, ignore.environment = TRUE)
}

# ---- namespace shim builders (internal) ----------------------------------

#' Builds the `generate` shim `sz_solve_r_generator()`/
#' `sz_solve_r_generator_bbob()`'s own `generate` parameter expects (T3's
#' pinned `function(pop, fitness, rng, iteration)` convention) -- wraps
#' `rng`, assembles `ctx`/the `pop` view, dispatches to `algo$generate(pop,
#' ctx)`, and converts the return value back to T3's raw offspring shape.
#'
#' @param algo An `Algorithm` subclass instance.
#' @param space_blocks The block-descriptor `list` for the space being
#'   solved.
#' @returns A closure of four arguments (`pop`, `fitness`, `rng`,
#'   `iteration`).
#' @noRd
.sz_make_generate_shim <- function(algo, space_blocks) {
  force(algo)
  force(space_blocks)
  function(pop, fitness, rng, iteration) {
    ctx <- .sz_make_ctx(rng, iteration, space_blocks)
    view <- .sz_pop_view(pop, fitness, space_blocks)
    .sz_offspring_to_raw(algo$generate(view, ctx))
  }
}

#' Builds the `initializer` shim `sz_solve_r_generator()`/
#' `sz_solve_r_generator_bbob()`'s own `initializer` parameter expects
#' (T3's pinned `function(n, rng, iteration)` convention) -- only ever
#' built (and passed) when `initialize_population` was actually
#' overridden (see `Algorithm$run()`'s own override-detection logic).
#'
#' @param algo An `Algorithm` subclass instance.
#' @param space_blocks The block-descriptor `list` for the space being
#'   solved.
#' @returns A closure of three arguments (`n`, `rng`, `iteration`).
#' @noRd
.sz_make_initializer_shim <- function(algo, space_blocks) {
  force(algo)
  force(space_blocks)
  function(n, rng, iteration) {
    ctx <- .sz_make_ctx(rng, iteration, space_blocks)
    .sz_offspring_to_raw(algo$initialize_population(n, space_blocks, ctx))
  }
}

#' Builds the `validate_space` shim `sz_solve_r_generator()`/
#' `sz_solve_r_generator_bbob()`'s own `validate_space` parameter expects
#' (T3's pinned `function(blocks)` convention) -- only ever built (and
#' passed) when `validate_space` was actually overridden. Deliberately
#' ALWAYS returns `NULL` (discarding whatever `algo$validate_space()`
#' itself returns): the R6 contract mirrors py-sezgi's `Algorithm.
#' validate_space()` exactly (raise an exception to reject; a return value
#' has no meaning), which also shields a subclass from T3's raw bridge's
#' OWN separate "a non-NULL character-scalar return value is a rejection
#' message" mechanism -- an override whose last expression happens to
#' evaluate to a string (e.g. a `sprintf()` call with no trailing
#' `invisible(NULL)`) must not accidentally veto every run.
#'
#' @param algo An `Algorithm` subclass instance.
#' @returns A closure of one argument (`blocks`).
#' @noRd
.sz_make_validate_space_shim <- function(algo) {
  force(algo)
  function(blocks) {
    algo$validate_space(blocks)
    NULL
  }
}

# ---- Algorithm ------------------------------------------------------------

#' Subclassable R6 base for an engine-hosted algorithm.
#'
#' Mirrors `sezgi.Algorithm` (`py-sezgi/python/sezgi/algorithm.py`, M4-1
#' Task 3) closely: a subclass implements `generate(pop, ctx)` (REQUIRED)
#' and may optionally override `initialize_population(n, space, ctx)`
#' (RULING 8: named `initialize_population`, NOT `initialize` -- R6
#' reserves `initialize` for the constructor, so Python's `initialize(n,
#' ctx)` hook name cannot be reused here; a deliberate, documented
#' divergence) and `validate_space(space)`. `run()` is the sole entry
#' point, wiring the instance through T3's `sz_solve_r_generator`/
#' `sz_solve_r_generator_bbob` bridge.
#'
#' `pop` (handed to `generate()`/a `PopulationAlgorithm`'s `select()`/
#' `vary()`/a `LocalSearch`'s `neighbor()`) is `list(x, f)`: `f` is a
#' numeric vector, the current population's fitness values, one entry per
#' individual. `x` is an n x dim numeric MATRIX (rows = individuals) for a
#' SINGLE-Float-block space (the ergonomic, common case -- `pop$x[i, ]`,
#' `ncol(pop$x)`); for every OTHER space shape (multi-block, or a single
#' NON-Float block -- Int/Categorical/Binary/Permutation), `x` is instead
#' a plain (unnamed) `list` of per-individual values, one entry per
#' individual (bare for a single block, a further per-block `list` for
#' multiple blocks) -- the SAME single-vs-multi-block convention
#' `Problem$evaluate(x)`'s own `x` argument already uses. A
#' `generate()`/`vary()`/`neighbor()` return value mirrors this: a matrix
#' (only meaningful when `pop$x` was itself a matrix) or a plain `list` of
#' offspring x-values.
#'
#' `ctx` is an `environment` with `$rng` (an already-wrapped RNG handle --
#' `ctx$rng$next_f64()` (a double in `[0, 1)`), `ctx$rng$next_below(n)` (an
#' integer-valued double in `[0, n)`), `ctx$rng$split(child_id)` (an
#' independent child stream)), `$iteration` (a double, the engine's own
#' 0-based generation counter -- always `0` for an `initialize_population()`
#' call), and `$space` (the block-descriptor `list` for the space being
#' solved -- one entry per block, each a named `list` with a `type` field
#' plus that block's own fields; the SAME shape `validate_space(space)`'s
#' own `space` argument uses).
#'
#' Override-detection caveat: `initialize_population`/`validate_space` are
#' detected as overridden by comparing the resolved method against this
#' class's own declared default at the CLASS level (body + formals,
#' ignoring environment -- see `R/algorithm.R`'s own module doc for the
#' exact spelling) -- a subclass method whose body happens to be
#' byte-identical to the base default (e.g. a copy-pasted `stop(...)` call
#' with the same message) is therefore treated as NOT overridden even
#' though it was technically redeclared; harmless today since both
#' defaults are no-op-equivalent (`initialize_population`'s default always
#' errors if ever reached, `validate_space`'s default always accepts), but
#' worth knowing.
#'
#' See this file's own module doc (top of `R/algorithm.R`) for the full
#' rationale behind these choices and the `log_dir` rejection.
#'
#' Methods:
#' \describe{
#'   \item{`generate(pop, ctx)`}{REQUIRED: produce this generation's
#'     offspring. Default: `stop()`s with "not implemented".}
#'   \item{`initialize_population(n, space, ctx)`}{Optional: produce the
#'     initial population of size `n`. Default: NOT overridden -- `run()`
#'     never calls this default body; the engine's own Rust `init/uniform`
#'     initializer runs instead, with NO R call for population
#'     initialization at all (byte-identical to a raw `sz_solve_r_generator`
#'     call with `initializer = NULL`).}
#'   \item{`validate_space(space)`}{Optional build-time veto: `stop()` to
#'     reject `space` BEFORE any `generate()`/`initialize_population()`
#'     call. Default: no-op (accepts any space).}
#'   \item{`run(problem, budget, pop_size = 50, seed = 0, run_id = 0, name
#'     = NULL, log_dir = NULL)`}{Runs this algorithm's hooks INSIDE the
#'     Rust engine loop. `problem`: a `Problem` subclass instance (routed
#'     through `sz_as_problem()`) OR a built-in problem descriptor (see
#'     [sz_builtin_bbob()]). `log_dir`: NOT supported (see module doc) --
#'     a non-`NULL` value raises a clear error. Returns an `sz_result`
#'     (see [`.sz_wrap_result`]). NOTE -- known, INTENTIONAL divergence
#'     from `sezgi.Algorithm.run(problem, budget, seed=0, pop_size=50,
#'     log_dir=None, ...)` (py-sezgi/python/sezgi/algorithm.py): `pop_size`
#'     and `seed` are swapped, and `log_dir` sits last rather than third,
#'     kept as-is (not reordered to match) because every anchored test and
#'     example already calls `run()` positionally against THIS order --
#'     reshuffling would break them for no benefit in a language where
#'     both sides are typically called with named arguments anyway.}
#' }
#'
#' @export
Algorithm <- R6::R6Class("Algorithm",
  public = list(
    generate = function(pop, ctx) {
      stop("Algorithm$generate() is not implemented -- subclass Algorithm and override it")
    },

    initialize_population = function(n, space, ctx) {
      stop(paste0(
        "Algorithm$initialize_population() was not overridden -- run() never calls ",
        "this default body (see its own override-detection note in R/algorithm.R); ",
        "the engine's own init/uniform initializer is used instead when a subclass ",
        "does not override this method"
      ))
    },

    validate_space = function(space) {
      NULL
    },

    run = function(problem, budget, pop_size = 50, seed = 0, run_id = 0,
                    name = NULL, log_dir = NULL) {
      if (!is.null(log_dir)) {
        stop(paste0(
          "Algorithm$run(): log_dir is not supported -- sz_solve_r_generator()/",
          "sz_solve_r_generator_bbob() (r-sezgi/src/rust/src/solve.rs, M4-2 Task 3) ",
          "have no log_dir parameter (IOH logging was not wired into the R generator ",
          "bridge -- see that task's own report), so there is no R-side logging path ",
          "reachable from here to wire this parameter through (mirrors M4-1 RULING B's ",
          "\"honestly reject\" outcome, applied to the missing-plumbing case)"
        ))
      }

      algo_name <- if (is.null(name)) tolower(class(self)[1]) else name

      overridden_init <- .sz_algo_overridden(
        self, "initialize_population", Algorithm$public_methods$initialize_population
      )
      overridden_validate <- .sz_algo_overridden(
        self, "validate_space", Algorithm$public_methods$validate_space
      )

      if (inherits(problem, "sz_builtin_bbob")) {
        space_blocks <- .sz_space_to_blocks(sz_space(sz_float(-5, 5, problem$dim)))
        generate_shim <- .sz_make_generate_shim(self, space_blocks)
        initializer_shim <- if (overridden_init) .sz_make_initializer_shim(self, space_blocks) else NULL
        validate_shim <- if (overridden_validate) .sz_make_validate_space_shim(self) else NULL

        raw <- sz_solve_r_generator_bbob(
          generate_shim,
          fid = as.integer(problem$fid), dim = as.integer(problem$dim),
          instance = as.integer(problem$instance),
          budget = budget, master_seed = seed, run_id = run_id, pop_size = pop_size,
          init_kind = "init/uniform", replacer_kind = "replace/mu-plus-lambda",
          initializer = initializer_shim, validate_space = validate_shim,
          algo_name = algo_name
        )
        f_opt <- NULL
      } else {
        prob <- sz_as_problem(problem)
        space_blocks <- .sz_space_to_blocks(prob$space())
        evaluate_shim <- .sz_make_evaluate_shim(prob)
        generate_shim <- .sz_make_generate_shim(self, space_blocks)
        initializer_shim <- if (overridden_init) .sz_make_initializer_shim(self, space_blocks) else NULL
        validate_shim <- if (overridden_validate) .sz_make_validate_space_shim(self) else NULL

        raw <- sz_solve_r_generator(
          generate_shim, space_blocks, evaluate_shim,
          budget = budget, master_seed = seed, run_id = run_id, pop_size = pop_size,
          init_kind = "init/uniform", replacer_kind = "replace/mu-plus-lambda",
          initializer = initializer_shim, validate_space = validate_shim,
          algo_name = algo_name
        )
        f_opt <- prob$optimum()
      }

      .sz_wrap_result(raw, algo_name, seed, budget, f_opt)
    }
  )
)

# ---- PopulationAlgorithm ---------------------------------------------------

#' Family base for population-style algorithms (GA/DE/ES-shaped).
#'
#' Inherits [Algorithm]. A subclass implements `vary(parents, ctx)`
#' (REQUIRED -- turns a parent pool into offspring) and may optionally
#' override `select(pop, k, ctx)` (default: k-fold binary tournament,
#' tournament size 2, minimization). `generate(pop, ctx)` is NOT meant to
#' be overridden (override `select`/`vary` instead) -- it is composed here
#' from `select` + `vary`.
#'
#' Mirrors `sezgi.PopulationAlgorithm` (`py-sezgi/python/sezgi/
#' algorithm.py`, M4-1 Task 4) exactly, including the PROVEN-uniform
#' two-fixed-draw index-shift tournament formula (ported verbatim below,
#' with its own determinism comment).
#'
#' `pop` (handed to `select()`/`generate()`) is `list(x, f)`: `f` is a
#' numeric vector of per-individual fitness values; `x` is an n x dim
#' numeric MATRIX (rows = individuals) for a SINGLE-Float-block space, or
#' a plain (unnamed) `list` of per-individual values otherwise (bare for a
#' single non-Float block, a further per-block `list` for multiple
#' blocks) -- the SAME single-vs-multi-block convention `Problem$
#' evaluate(x)`'s own `x` uses. `vary()`'s return value mirrors this (a
#' matrix or a `list`). `ctx` is an `environment` with `$rng` (a wrapped
#' RNG handle -- `$rng$next_f64()`/`$rng$next_below(n)`/`$rng$split
#' (child_id)`), `$iteration` (a double), and `$space` (the block-
#' descriptor `list` for the space being solved). See [Algorithm]'s own
#' doc for the full detail on both, and for the override-detection
#' caveat that applies to any `initialize_population`/`validate_space`
#' override this subclass adds (a body byte-identical to `Algorithm`'s
#' own default is treated as NOT overridden, even if redeclared).
#'
#' Methods:
#' \describe{
#'   \item{`select(pop, k, ctx)`}{Default: k-fold binary tournament
#'     selection (tournament size 2; minimization -- lower fitness wins).
#'     For each of the `k` requested parent slots, INDEPENDENTLY draws
#'     exactly two values from `ctx$rng` (with `n = ` the current
#'     population size): `i <- ctx$rng$next_below(n)`; `j <-
#'     ctx$rng$next_below(n - 1); if (j >= i) j <- j + 1` (for `n > 1`).
#'     This is the standard "index-shift" trick for drawing two DISTINCT
#'     indices from `[0, n)` using exactly two draws and no rejection/
#'     retry loop -- the draw count per slot is always EXACTLY 2 for
#'     `n > 1`, so the whole sequence is hand-traceable given a fixed RNG
#'     seed and a known `n`. The individuals at `i` and `j` then "fight":
#'     `pop$f[i+1] <= pop$f[j+1]` wins as `i` -- this is also the EXACT
#'     tie-break rule: on an exact tie, the first-drawn contestant (`i`)
#'     always wins, never `j`. `n == 1` (a population of one) is a
#'     degenerate edge case with no second index to draw: only `i <-
#'     ctx$rng$next_below(1)` (always `0`) is drawn, and `i` wins
#'     trivially, costing 1 draw instead of 2. Returns a `list` of exactly
#'     `k` parents (individual x-values, one per slot, in slot order) --
#'     ties/duplicates across slots are possible and expected.}
#'   \item{`vary(parents, ctx)`}{REQUIRED: turn a parent pool (a `list` of
#'     `length(parents)` x-values) into this generation's offspring (a
#'     matrix or a `list`, see the module doc). Offspring COUNT is
#'     entirely this method's own choice, independent of
#'     `length(parents)`.}
#'   \item{`generate(pop, ctx)`}{Composes `select()` then `vary()` -- NOT
#'     meant to be overridden. Arity contract (PINNED): always calls
#'     `self$select(pop, n, ctx)` where `n` is the CURRENT population's
#'     own size (the standard "mating pool" convention: a full-size parent
#'     pool). The resulting `parents` list is passed to `self$vary(parents,
#'     ctx)` UNCHANGED, and `vary()`'s return value is returned UNCHANGED
#'     as this generation's offspring.}
#' }
#'
#' @export
PopulationAlgorithm <- R6::R6Class("PopulationAlgorithm",
  inherit = Algorithm,
  public = list(
    select = function(pop, k, ctx) {
      n <- .sz_pop_n(pop)
      parents <- vector("list", k)
      for (idx in seq_len(k)) {
        i <- ctx$rng$next_below(n)
        if (n > 1) {
          j <- ctx$rng$next_below(n - 1)
          if (j >= i) j <- j + 1
          winner <- if (pop$f[i + 1] <= pop$f[j + 1]) i else j
        } else {
          winner <- i
        }
        parents[[idx]] <- .sz_pop_at(pop, winner + 1)
      }
      parents
    },

    vary = function(parents, ctx) {
      stop("PopulationAlgorithm$vary() is not implemented -- subclass and override it")
    },

    generate = function(pop, ctx) {
      n <- .sz_pop_n(pop)
      parents <- self$select(pop, n, ctx)
      self$vary(parents, ctx)
    }
  )
)

# ---- LocalSearch ------------------------------------------------------------

#' Family base for single-trajectory local search (hill-climbing/
#' SA-shaped).
#'
#' Inherits [Algorithm]. A subclass implements `neighbor(x, ctx)`
#' (REQUIRED -- proposes one perturbed candidate from the current point)
#' and may optionally override `accept(f_old, f_new, ctx)` (default:
#' greedy, `f_new <= f_old`). Designed for `pop_size = 1` -- `run()` is
#' overridden here to default AND ENFORCE `pop_size = 1` (`stop()`
#' otherwise).
#'
#' Mirrors `sezgi.LocalSearch` (`py-sezgi/python/sezgi/algorithm.py`,
#' M4-1 Task 4) exactly, including its honest structural limitation:
#'
#' `pop` (handed to `generate()`, which itself calls `neighbor()`/
#' `accept()`) is `list(x, f)`, at `pop_size = 1` always a single
#' individual: `f` is a length-1 numeric vector, the current point's
#' fitness; `x` is a length-1 x dim numeric MATRIX (a single row) for a
#' SINGLE-Float-block space (`x[1, ]` is the current point), or a
#' length-1 `list` otherwise (`x[[1]]` is the current point, bare for a
#' single non-Float block, a further per-block `list` for multiple
#' blocks) -- the SAME single-vs-multi-block convention `Problem$
#' evaluate(x)`'s own `x` uses (`.sz_pop_at(pop, 1)`, used internally by
#' `generate()` below, abstracts over both shapes). `neighbor()`'s return
#' value is a SINGLE x-value in that same convention (NOT a list --
#' `generate()` wraps it). `ctx` is an `environment` with `$rng` (a
#' wrapped RNG handle -- `$rng$next_f64()`/`$rng$next_below(n)`/`$rng$
#' split(child_id)`), `$iteration` (a double), and `$space` (the block-
#' descriptor `list` for the space being solved). See [Algorithm]'s own
#' doc for the full detail on both, and for the override-detection
#' caveat that applies to any `initialize_population`/`validate_space`
#' override this subclass adds (a body byte-identical to `Algorithm`'s
#' own default is treated as NOT overridden, even if redeclared).
#'
#' ACCEPT() DESIGN (read carefully -- this is not the naive reading of
#' "accept decides whether the engine keeps the point"): this base runs
#' over the engine's default `replace/mu-plus-lambda` replacer. At
#' `pop_size = 1` (mu = 1), with `generate()` here always returning
#' exactly ONE offspring (lambda = 1), that replacer ALWAYS keeps the
#' strictly-better (or, on an exact tie, the OLD) of \{current, neighbor\}
#' as the NEXT call's `pop$x`/`pop$f`'s own first entry -- a structural,
#' UNCONDITIONAL guarantee of `replace/mu-plus-lambda` itself, not a
#' decision `accept()` makes. Two direct consequences: (1) the fitness
#' this base reads on the call following a `neighbor()` proposal is
#' ALWAYS `<= f_old` -- `accept()` can therefore NEVER actually be called
#' with an `f_new` that is objectively WORSE than `f_old`; a worse
#' neighbor's exact fitness is never even surfaced back to R by this
#' bridge. **This base's `accept()` therefore CANNOT implement true
#' simulated-annealing-style "sometimes accept a worse move" semantics --
#' that would require intercepting the replacer's own decision, which
#' this design deliberately does NOT attempt.** (2) Given (1), the DEFAULT
#' `accept` (`f_new <= f_old`) is a TAUTOLOGY over every pair this base
#' can ever actually present to it -- it is ALWAYS true, faithfully
#' mirroring (not fighting) what `replace/mu-plus-lambda` already
#' unconditionally enforces at `pop_size = 1`. What `accept()` DOES
#' meaningfully control: whether this base's OWN internally-tracked anchor
#' (`self$current_x`/`self$current_f`, the point the NEXT `neighbor()`
#' call is proposed from) ADVANCES to match the engine's already-decided
#' outcome, or stays where it was -- i.e. `accept()` governs this base's
#' own SEARCH TRAJECTORY, never population membership (which the replacer
#' alone decides, unconditionally, at `pop_size = 1`).
#'
#' @export
LocalSearch <- R6::R6Class("LocalSearch",
  inherit = Algorithm,
  public = list(
    current_x = NULL,
    current_f = NULL,

    neighbor = function(x, ctx) {
      stop("LocalSearch$neighbor() is not implemented -- subclass and override it")
    },

    accept = function(f_old, f_new, ctx) {
      f_new <= f_old
    },

    run = function(problem, budget, pop_size = 1, seed = 0, run_id = 0,
                    name = NULL, log_dir = NULL) {
      if (pop_size != 1) {
        stop(sprintf(
          "LocalSearch requires pop_size = 1 (a single search trajectory) -- got pop_size=%s",
          pop_size
        ))
      }
      self$current_x <- NULL
      self$current_f <- NULL
      super$run(problem, budget, pop_size = pop_size, seed = seed, run_id = run_id,
                name = name, log_dir = log_dir)
    },

    generate = function(pop, ctx) {
      if (is.null(self$current_x)) {
        self$current_x <- .sz_pop_at(pop, 1)
        self$current_f <- as.double(pop$f[1])
      } else {
        f_candidate <- as.double(pop$f[1])
        if (self$accept(self$current_f, f_candidate, ctx)) {
          self$current_x <- .sz_pop_at(pop, 1)
          self$current_f <- f_candidate
        }
      }
      list(self$neighbor(self$current_x, ctx))
    }
  )
)

# ---- sz_builtin_bbob --------------------------------------------------------

#' A built-in-problem descriptor for [Algorithm]'s `run()`.
#'
#' r-sezgi has no general "native problem handle" type the way py-sezgi's
#' `as_native_problem()` does (every existing `sz_solve_*`/`sz_eval_session*`
#' binding takes `fid`/`dim`/`instance` as separate scalar arguments, never
#' a wrapped object) -- this is a MINIMAL, purpose-built descriptor letting
#' `Algorithm$run()`'s single `problem` argument select T3's
#' `sz_solve_r_generator_bbob()` entry point (the "built-in form" of a
#' `run()` call, mirroring py-sezgi's `sezgi.bbob(fid, dim, instance)`
#' handles reaching `Algorithm.run()` the same way) instead of the
#' `Problem`-subclass path. Deliberately narrow: it exists ONLY to route
#' `Algorithm$run()`, not as a general native-problem abstraction (a
#' richer type, if ever needed by a future task, is out of this task's
#' scope -- see the task-4 report's concerns for T5).
#'
#' @param fid BBOB function id (`1..=24`).
#' @param dim Problem dimension (>= 1).
#' @param instance BBOB instance id (>= 1). Default `1`.
#' @returns An object of class `"sz_builtin_bbob"` with elements `fid`,
#'   `dim`, `instance`.
#' @export
sz_builtin_bbob <- function(fid, dim, instance = 1) {
  structure(list(fid = fid, dim = dim, instance = instance), class = "sz_builtin_bbob")
}

# ---- .sz_wrap_result / print.sz_result -------------------------------------

#' Wraps a raw 3-field `sz_solve_r_generator()`/`sz_solve_r_generator_bbob()`
#' result (`best_f`, `evals`, `best_x`) into the 8-field `sz_result` shape.
#'
#' Field names/gap arithmetic mirror `sz_algo_solve()`'s own return `list`
#' (`R/algo.R`) EXACTLY (`algo`, `seed`, `budget`, `evals_used`, `best_x`,
#' `best_f`, `f_opt`, `gap`) -- the SAME shape both surfaces agree on
#' (`sz_algo_solve()`'s own list is not classed; this one carries the
#' `"sz_result"` S3 class for a dedicated [print.sz_result()] method, an
#' additive difference only). This is the ONE shared helper this task
#' introduces specifically so a future task (e.g. the 29-wrapper-class
#' preset surface) does not duplicate the same gap arithmetic a second
#' time (the M4-1 fix-wave consolidation lesson, `py-sezgi/python/sezgi/
#' algo.py::_wrap_result`, applied here from the start).
#'
#' @param raw3 A named `list` with `best_f`, `evals`, `best_x` (T3's own
#'   return shape).
#' @param algo Character scalar, the algorithm's label.
#' @param seed Numeric scalar, the master RNG seed used.
#' @param budget Numeric scalar, the run's evaluation budget.
#' @param f_opt Numeric scalar, or `NULL` if the problem has no known
#'   optimum.
#' @returns An `sz_result` (`list` with `algo`, `seed`, `budget`,
#'   `evals_used`, `best_x`, `best_f`, `f_opt`, `gap`).
#' @noRd
.sz_wrap_result <- function(raw3, algo, seed, budget, f_opt) {
  best_f <- raw3$best_f
  structure(
    list(
      algo = algo,
      seed = seed,
      budget = budget,
      evals_used = raw3$evals,
      best_x = raw3$best_x,
      best_f = best_f,
      f_opt = f_opt,
      gap = if (is.null(f_opt)) NULL else best_f - f_opt
    ),
    class = "sz_result"
  )
}

#' Prints an `sz_result` object.
#'
#' @param x An `sz_result` object (from [.sz_wrap_result]/[Algorithm]'s
#'   own `run()`).
#' @param ... Ignored.
#' @returns `x`, invisibly.
#' @export
print.sz_result <- function(x, ...) {
  cat(sprintf(
    "<sz_result> algo=%s seed=%s budget=%s evals_used=%s best_f=%s\n",
    x$algo, x$seed, x$budget, x$evals_used, format(x$best_f)
  ))
  if (!is.null(x$f_opt)) {
    cat(sprintf("  f_opt=%s gap=%s\n", format(x$f_opt), format(x$gap)))
  }
  invisible(x)
}
