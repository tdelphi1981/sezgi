# Built-in preset-backed algorithm wrapper classes (M4-2 Task 5) -- the R
# mirror of py-sezgi's `sezgi.builtins` module (`py-sezgi/python/sezgi/
# builtins.py`, M4-1 Task 5): one R6 class per `sz_preset_*` builder
# (`R/000-wrappers.R` + `R/presets.R`, both mirroring `crates/components/
# src/presets.rs`), plus `GeneticAlgorithm`/`DifferentialEvolution`
# (dispatchers over several presets each) and `NSGA2` (a thin skin over
# `sz_nsga2()`, NOT a preset -- see its own doc below).
#
# ---- reconciliation arithmetic (live-verified; see test-oop-builtins.R's
# own structural tests, which assert these counts from the LIVE table/
# NAMESPACE rather than restating them as prose) --------------------------
#
# 34 total `sz_preset_*` builders in this package (`R/000-wrappers.R` +
# `R/presets.R`) = 26 table rows below (every non-GA/non-DE preset) + 5
# GA presets (`ga_real`/`ga_perm`/`ga_bin`/`ga_int`/`ga_cat`, folded into
# `GeneticAlgorithm`'s own space-kind auto-dispatch) + 3 DE presets
# (`de_rand_1`/`de_best_1`/`jde`, folded into `DifferentialEvolution`'s own
# `variant=` dispatch).
#
# 29 exported classes in this file = 26 table classes + `GeneticAlgorithm`
# + `DifferentialEvolution` (28 preset-backed classes total) + `NSGA2`
# (NOT preset-backed -- NSGA-II is not built on the scalar Engine/Registry/
# Generator/AlgorithmSpec machinery every `sz_preset_*` targets; it is a
# thin skin over `sz_nsga2()`, `R/mo.R`, mirroring py-sezgi's own `NSGA2`
# class, whose docstring makes the identical point).
#
# ---- table-driven generation (ruling: "no hand-divergence") -------------
#
# `.sz_preset_table` (below) is the single source of truth for all 26
# uniformly-shaped preset wrapper classes; `.sz_make_preset_class()` (also
# below) generates every one of them from a table row. `GeneticAlgorithm`
# and `DifferentialEvolution` are hand-written (their dispatch axis
# genuinely differs from the uniform "one class, one preset" shape -- the
# problem's own space kind for the former, a constructor kwarg for the
# latter) but both still funnel through the SAME `.sz_preset_run()` helper
# every generated class's own `run()` uses, so the actual
# spec-building-then-solve-then-`.sz_wrap_result()` machinery has exactly
# ONE call site in this file, not N (mirrors py-sezgi's own `_run_spec`
# helper and its module doc's identical claim).
#
# ---- problem-form support matrix (identical across every class here) ----
#
# `$run(problem, budget, seed = 0, run_id = 0)`'s `problem` is EITHER a
# [Problem] subclass instance (T2's bridge -- routed through
# `sz_solve_r_problem()`, `f_opt = prob$optimum()`) OR an
# [sz_builtin_bbob()] descriptor (routed through `sz_solve_bbob()`,
# `f_opt = NULL`) -- the SAME two forms [Algorithm]'s own `run()` accepts
# (T4, `R/algorithm.R`). No CEC2014/CEC2017/CEC2022/TSP/onemax/etc.
# native-descriptor path exists yet for this class family (r-sezgi has no
# general native-problem-handle type -- see T4's own report's "Concerns for
# T5" and this task's own report for the full discussion).
#
# ---- kwargs authority ----------------------------------------------------
#
# Every `sz_preset_*()` signature (which kwargs it takes, in what order,
# with what "canonical" pop_size per its own doc comment where one is
# stated) is taken VERBATIM from `R/000-wrappers.R` (the savvy-generated
# bindings) + `R/presets.R` (the one hand-written wrapper,
# `sz_preset_es_mu_plus_lambda`) -- the repo authority per this task's own
# brief. Where `000-wrappers.R` states no canonical pop_size value for a
# preset (`cmaes`/`de_rand_1`/`de_best_1`/`jde`/`es_mu_plus_lambda`/`pso`/
# `shade`/`random_search`/`ga_real`/`ga_perm`/`ga_bin`/`ga_int`/`ga_cat`),
# `20` is used -- mirroring py-sezgi's `builtins.py` own identical
# resolution for the identical presets (both wrapper layers cite the same
# underlying Rust doc comments).

#' @importFrom R6 R6Class
NULL

# ---- preset function resolver (internal) ----------------------------------

#' Resolves `sz_preset_<preset_name>` by NAME, from inside the package
#' namespace, at CALL time rather than at package-load/parse time.
#'
#' Deliberately indirect (not a direct function reference stored in
#' `.sz_preset_table`/the GA/DE dispatch tables): `R/` files are sourced in
#' plain alphabetical order (no `Collate:` field in `DESCRIPTION`), so
#' `builtins.R` loads BEFORE `presets.R` (`sz_preset_es_mu_plus_lambda`'s
#' own hand-written-wrapper file) -- a direct top-level reference to that
#' one function at `builtins.R`'s own parse time would fail with "object
#' not found". Resolving by name inside a method body (called only once
#' the package has FULLY loaded) sidesteps the ordering question entirely,
#' for every preset uniformly (not a special case for `es_mu_plus_lambda`
#' alone).
#'
#' @param preset_name Character scalar, e.g. `"abc"`, `"es_mu_plus_lambda"`.
#' @returns The `sz_preset_<preset_name>` function.
#' @noRd
.sz_preset_fn <- function(preset_name) {
  get(paste0("sz_preset_", preset_name), envir = asNamespace("sezgi"), inherits = TRUE, mode = "function")
}

# ---- shared run-tail helper (internal) -----------------------------------

#' Shared `spec_json -> solve -> sz_result` tail every wrapper class in
#' this file (table-generated and hand-written alike) funnels through --
#' the ONE place `sz_solve_bbob()`/`sz_solve_r_problem()` are actually
#' called from this file. See this file's own module doc for the
#' problem-form support matrix.
#'
#' @param spec_json Algorithm spec as JSON (from a `sz_preset_*()` call).
#' @param problem A [Problem] subclass instance, or an [sz_builtin_bbob()]
#'   descriptor.
#' @param budget Evaluation budget (recorded on the returned `sz_result`
#'   only -- already baked into `spec_json` by whichever `sz_preset_*()`
#'   built it).
#' @param seed Master RNG seed.
#' @param run_id Run id.
#' @param algo_name Character scalar recorded as the returned `sz_result`'s
#'   `algo` field.
#' @returns An `sz_result` (see [.sz_wrap_result]).
#' @noRd
.sz_preset_run <- function(spec_json, problem, budget, seed, run_id, algo_name) {
  if (inherits(problem, "sz_builtin_bbob")) {
    raw <- sz_solve_bbob(
      spec_json,
      fid = as.integer(problem$fid), dim = as.integer(problem$dim),
      instance = as.integer(problem$instance), master_seed = seed, run_id = run_id
    )
    f_opt <- NULL
  } else {
    prob <- sz_as_problem(problem)
    blocks <- .sz_space_to_blocks(prob$space())
    evaluate_shim <- .sz_make_evaluate_shim(prob)
    raw <- sz_solve_r_problem(spec_json, blocks, evaluate_shim, master_seed = seed, run_id = run_id)
    f_opt <- prob$optimum()
  }
  .sz_wrap_result(raw, algo_name, seed, budget, f_opt)
}

#' The resolved problem's own total dimensionality -- for `dim_budget`-mode
#' presets (`CMAESIpop`/`LSHADE`/`NelderMead`), which derive their own
#' population size from `dim` at `$run()` time rather than accepting a free
#' `pop_size`.
#'
#' @param problem A [Problem] subclass instance, or an [sz_builtin_bbob()]
#'   descriptor.
#' @returns A double scalar (`sz_preset_lshade()`/etc.'s own `dim` parameter
#'   is `f64`, not an R integer).
#' @noRd
.sz_problem_dim <- function(problem) {
  if (inherits(problem, "sz_builtin_bbob")) {
    return(as.double(problem$dim))
  }
  prob <- sz_as_problem(problem)
  blocks <- .sz_space_to_blocks(prob$space())
  as.double(sum(vapply(blocks, function(b) b$n, integer(1))))
}

# ---- table-driven preset classes -------------------------------------------

#' Generates one uniformly-shaped preset wrapper R6 class from a
#' `.sz_preset_table` row.
#'
#' `mode` (mirrors py-sezgi's `builtins.py` own three-way split exactly):
#' \describe{
#'   \item{`"pop_budget"`}{`preset_fn(pop_size, budget, ...)` -- covers
#'     `EvolutionStrategy`'s `dist=`/`mean=`/`sigma=`/`loc=`/`scale=`/
#'     `alpha=`/`nu=` kwargs too (they flow through `...` unchanged, no
#'     special-casing needed: `sz_preset_es_mu_plus_lambda()`'s own
#'     signature already accepts them). `pop_size` defaults to
#'     `default_pop_size` and may be overridden at `$new()`.}
#'   \item{`"budget_only"`}{`preset_fn(budget, ...)` -- the preset's own
#'     Rust signature has NO `pop_size` parameter at all
#'     (`SimulatedAnnealing`: pop_size is hard-coded to 1 inside the preset
#'     itself). `pop_size` is accepted at `$new()` ONLY for interface
#'     uniformity with every other class, fixed at `1`, and rejected (a
#'     clear error) if given any other value.}
#'   \item{`"dim_budget"`}{`preset_fn(dim, budget, ...)`, `dim` read from
#'     the resolved problem at `$run()` time (see [.sz_problem_dim]); the
#'     preset's own population size is a DERIVED formula, so an explicit
#'     `pop_size` at `$new()` is REJECTED (a clear error) -- there is no
#'     `pop_size` knob these presets expose to override.}
#' }
#'
#' @param class_name Character scalar, the generated class's own name.
#' @param preset_name Character scalar, the preset's own short name (e.g.
#'   `"abc"`) -- the `sz_preset_*` function this class wraps is resolved
#'   from this by [.sz_preset_fn()] at `$run()` time, not stored directly
#'   (see that function's own doc for why). Also used in error messages.
#' @param mode One of `"pop_budget"`, `"budget_only"`, `"dim_budget"`.
#' @param default_pop_size The class's own `pop_size` default (a numeric
#'   scalar for `"pop_budget"` mode, `NULL` for the other two modes).
#' @returns An `R6ClassGenerator`.
#' @noRd
.sz_make_preset_class <- function(class_name, preset_name, mode, default_pop_size) {
  force(class_name)
  force(preset_name)
  force(mode)
  force(default_pop_size)

  initialize_fn <- function(pop_size = NULL, ...) {
    if (identical(mode, "budget_only")) {
      if (!is.null(pop_size) && pop_size != 1) {
        stop(sprintf(
          paste0(
            "%s has a fixed population of 1 (a single search trajectory -- ",
            "sz_preset_%s()'s own Rust signature has no pop_size parameter ",
            "at all) -- pop_size is not a settable parameter of this class, ",
            "got pop_size=%s"
          ),
          class_name, preset_name, format(pop_size)
        ))
      }
      self$pop_size <- 1
    } else if (identical(mode, "dim_budget")) {
      if (!is.null(pop_size)) {
        stop(sprintf(
          paste0(
            "%s derives its population size from the problem's own ",
            "dimensionality at run() time (sz_preset_%s()'s own formula) -- ",
            "pop_size is not a settable parameter of this class, got ",
            "pop_size=%s"
          ),
          class_name, preset_name, format(pop_size)
        ))
      }
      self$pop_size <- NULL
    } else {
      self$pop_size <- if (is.null(pop_size)) default_pop_size else pop_size
    }
    self$preset_kwargs <- list(...)
    invisible(self)
  }

  run_fn <- function(problem, budget, seed = 0, run_id = 0) {
    preset_fn <- .sz_preset_fn(preset_name)
    spec_json <- if (identical(mode, "pop_budget")) {
      do.call(preset_fn, c(list(self$pop_size, budget), self$preset_kwargs))
    } else if (identical(mode, "budget_only")) {
      do.call(preset_fn, c(list(budget), self$preset_kwargs))
    } else {
      do.call(preset_fn, c(list(.sz_problem_dim(problem), budget), self$preset_kwargs))
    }
    .sz_preset_run(spec_json, problem, budget, seed, run_id, tolower(class_name))
  }

  R6::R6Class(class_name,
    public = list(
      pop_size = NULL,
      preset_kwargs = NULL,
      initialize = initialize_fn,
      run = run_fn
    )
  )
}

#' Table row: `class` (character), `preset_name` (character, the
#' `sz_preset_<preset_name>` function this class wraps -- resolved by
#' [.sz_preset_fn()] at `$run()` time, see that function's own doc for
#' why), `mode` (see [.sz_make_preset_class]), `default_pop_size` (numeric
#' or `NULL`).
#'
#' Row order/`default_pop_size` matches the research doc §E pinned class
#' list; every `default_pop_size` for `"pop_budget"` mode is the "Canonical
#' is N" value `R/000-wrappers.R`'s own roxygen doc states for that preset,
#' where one is stated (`20` otherwise -- see this file's own module doc).
#' @noRd
.sz_preset_table <- list(
  list(class = "ArtificialBeeColony", preset_name = "abc", mode = "pop_budget", default_pop_size = 20),
  list(class = "AntLion", preset_name = "alo", mode = "pop_budget", default_pop_size = 25),
  list(class = "BatAlgorithm", preset_name = "bat", mode = "pop_budget", default_pop_size = 30),
  list(class = "CMAES", preset_name = "cmaes", mode = "pop_budget", default_pop_size = 20),
  list(class = "CMAESIpop", preset_name = "cmaes_ipop", mode = "dim_budget", default_pop_size = NULL),
  list(class = "CuckooSearch", preset_name = "cuckoo_search", mode = "pop_budget", default_pop_size = 25),
  list(class = "EvolutionStrategy", preset_name = "es_mu_plus_lambda", mode = "pop_budget", default_pop_size = 20),
  list(class = "FireflyAlgorithm", preset_name = "firefly", mode = "pop_budget", default_pop_size = 25),
  list(class = "FlowerPollination", preset_name = "fpa", mode = "pop_budget", default_pop_size = 25),
  list(class = "GrasshopperOptimization", preset_name = "goa", mode = "pop_budget", default_pop_size = 30),
  list(class = "GravitationalSearch", preset_name = "gsa", mode = "pop_budget", default_pop_size = 30),
  list(class = "GreyWolfOptimizer", preset_name = "gwo", mode = "pop_budget", default_pop_size = 30),
  list(class = "HarmonySearch", preset_name = "harmony_search", mode = "pop_budget", default_pop_size = 30),
  list(class = "HarrisHawks", preset_name = "hho", mode = "pop_budget", default_pop_size = 30),
  list(class = "JAYA", preset_name = "jaya", mode = "pop_budget", default_pop_size = 30),
  list(class = "LSHADE", preset_name = "lshade", mode = "dim_budget", default_pop_size = NULL),
  list(class = "MothFlameOptimization", preset_name = "mfo", mode = "pop_budget", default_pop_size = 30),
  list(class = "NelderMead", preset_name = "nelder_mead", mode = "dim_budget", default_pop_size = NULL),
  list(class = "ParticleSwarm", preset_name = "pso", mode = "pop_budget", default_pop_size = 20),
  list(class = "RandomSearch", preset_name = "random_search", mode = "pop_budget", default_pop_size = 20),
  list(class = "SHADE", preset_name = "shade", mode = "pop_budget", default_pop_size = 20),
  list(class = "SalpSwarm", preset_name = "ssa", mode = "pop_budget", default_pop_size = 30),
  list(class = "SimulatedAnnealing", preset_name = "sa", mode = "budget_only", default_pop_size = NULL),
  list(class = "SineCosineAlgorithm", preset_name = "sca", mode = "pop_budget", default_pop_size = 30),
  list(class = "TLBO", preset_name = "tlbo", mode = "pop_budget", default_pop_size = 30),
  list(class = "WhaleOptimization", preset_name = "woa", mode = "pop_budget", default_pop_size = 30)
)

#' Table-driven preset-backed algorithm wrapper classes.
#'
#' One R6 class per non-GA/DE `sz_preset_*` builder (`R/000-wrappers.R` +
#' `R/presets.R`, both mirroring `crates/components/src/presets.rs`) -- 26
#' classes, generated from ONE row table (`.sz_preset_table`) by ONE
#' factory (`.sz_make_preset_class()`, both internal to `R/builtins.R`).
#' Every class shares the identical shape: `$new(pop_size = NULL, ...)`
#' stores constructor kwargs only (no computation happens at construction
#' time); `$run(problem, budget, seed = 0, run_id = 0)` builds the
#' preset's `spec_json` via the matching `sz_preset_*(...)` call
#' (bit-identical to calling that function directly with the same
#' arguments -- see `test-oop-builtins.R`'s own table-driven anchor test),
#' routes it through `sz_solve_bbob()`/`sz_solve_r_problem()` depending on
#' `problem`'s own form, and wraps the raw 3-field result via T4's
#' `.sz_wrap_result()` into an `sz_result` (the SAME shape [Algorithm]'s
#' own `run()` returns).
#'
#' `pop_size` semantics depend on the preset's own Rust signature (see
#' each `sz_preset_*()`'s own doc in `R/000-wrappers.R` for its "Canonical
#' is N" convention, where stated):
#'
#' \describe{
#'   \item{most presets}{`sz_preset_*(pop_size, budget, ...)` --
#'     `pop_size` defaults to the class's own documented convention (the
#'     source paper's canonical value where `000-wrappers.R` states one;
#'     `20` otherwise, matching this repo's existing dominant convention
#'     for presets with no single canonical source), and MAY be
#'     overridden at `$new()`.}
#'   \item{`SimulatedAnnealing`}{`sz_preset_sa(budget)` has NO `pop_size`
#'     parameter at all (a single search trajectory) -- `pop_size` is
#'     fixed at `1`; a non-`NULL`, non-`1` value at `$new()` raises a
#'     clear error.}
#'   \item{`CMAESIpop`/`LSHADE`/`NelderMead`}{`sz_preset_*(dim, budget,
#'     ...)` DERIVES its own population size from the problem's
#'     dimensionality at `$run()` time (`dim` is read from the resolved
#'     problem, NOT `pop_size`) -- ANY `pop_size` given at `$new()` raises
#'     a clear error naming the derivation formula.}
#' }
#'
#' `EvolutionStrategy` additionally accepts `dist=`/`mean=`/`sigma=`/
#' `loc=`/`scale=`/`alpha=`/`nu=` at `$new()` (flowing through to
#' `sz_preset_es_mu_plus_lambda()` unchanged, via `...`) -- the ONLY table
#' row whose preset takes kwargs beyond `pop_size`/`budget`.
#'
#' Problem-form support matrix (identical for every class in this
#' family, and for [GeneticAlgorithm]/[DifferentialEvolution]): `problem`
#' is EITHER a [Problem] subclass instance (routed through
#' `sz_solve_r_problem()`, `f_opt = prob$optimum()`) OR an
#' [sz_builtin_bbob()] descriptor (routed through `sz_solve_bbob()`,
#' `f_opt = NULL`) -- the SAME two forms [Algorithm]'s own `run()`
#' accepts. No CEC2014/CEC2017/CEC2022/TSP/onemax/etc. native-descriptor
#' path exists yet.
#'
#' @name sz_preset_classes
#' @aliases ArtificialBeeColony AntLion BatAlgorithm CMAES CMAESIpop
#'   CuckooSearch EvolutionStrategy FireflyAlgorithm FlowerPollination
#'   GrasshopperOptimization GravitationalSearch GreyWolfOptimizer
#'   HarmonySearch HarrisHawks JAYA LSHADE MothFlameOptimization NelderMead
#'   ParticleSwarm RandomSearch SHADE SalpSwarm SimulatedAnnealing
#'   SineCosineAlgorithm TLBO WhaleOptimization
#' @evalNamespace paste0("export(", vapply(.sz_preset_table, function(row) row$class, character(1)), ")")
NULL

for (.sz_preset_row in .sz_preset_table) {
  assign(
    .sz_preset_row$class,
    .sz_make_preset_class(
      .sz_preset_row$class, .sz_preset_row$preset_name,
      .sz_preset_row$mode, .sz_preset_row$default_pop_size
    )
  )
}
rm(.sz_preset_row)

# ---- GeneticAlgorithm ------------------------------------------------------

#' Whether `problem`'s own search space is fit for GA auto-dispatch, and
#' which `ga_*` representation it resolves to.
#'
#' @param problem A [Problem] subclass instance, or an [sz_builtin_bbob()]
#'   descriptor (always resolves to `"real"` -- a built-in BBOB problem's
#'   space is a single Float block by construction, see [sz_builtin_bbob()]
#'   and [Algorithm]'s own `run()`).
#' @returns One of `"real"`, `"perm"`, `"bin"`, `"int"`, `"cat"`.
#' @noRd
.sz_ga_representation <- function(problem) {
  if (inherits(problem, "sz_builtin_bbob")) {
    return("real")
  }
  prob <- sz_as_problem(problem)
  blocks <- .sz_space_to_blocks(prob$space())
  if (length(blocks) == 0L) {
    stop("GeneticAlgorithm auto-dispatch: empty search space")
  }
  kinds <- unique(vapply(blocks, function(b) b$type, character(1)))
  if (length(kinds) != 1L) {
    stop(sprintf(
      paste0(
        "GeneticAlgorithm auto-dispatch requires every block in the space ",
        "to share one kind; got a Mixed space with kinds %s. This ",
        "milestone has no ga_* preset for a mixed space -- pass ",
        "representation= to force a single-kind preset, or build the ",
        "AlgorithmSpec by hand around the compound generator instead."
      ),
      paste(sort(kinds), collapse = ", ")
    ))
  }
  kind <- kinds[[1]]
  representation <- switch(kind,
    float = "real", permutation = "perm", binary = "bin",
    int = "int", categorical = "cat", NULL
  )
  if (is.null(representation)) {
    stop(sprintf("GeneticAlgorithm auto-dispatch has no preset for block kind %s", kind))
  }
  representation
}

#' Wraps a single-stage `ga_*` preset spec's generator in a `gen/compound`
#' generator holding one independent copy of the generator per block
#' (mirrors py-sezgi's `GeneticAlgorithm.run()`). The spec is a JSON string
#' and the package has no JSON runtime dependency, so the generator object
#' is located and brace-matched textually (preset generators are flat
#' objects whose strings contain no braces).
#'
#' @param spec_json Preset spec JSON.
#' @param n_blocks Number of blocks (> 1).
#' @returns The rewritten spec JSON string.
#' @noRd
.sz_ga_wrap_compound <- function(spec_json, n_blocks) {
  key <- gregexpr('"generator"\\s*:\\s*\\{', spec_json)[[1]]
  if (length(key) != 1L || key[[1]] < 0L) {
    stop(
      "GeneticAlgorithm: expected a single-stage ga_* preset spec with ",
      "exactly one generator"
    )
  }
  start <- key[[1]] + attr(key, "match.length")[[1]] - 1L
  chars <- strsplit(substring(spec_json, start), "")[[1]]
  depth <- 0L
  end <- NA_integer_
  for (i in seq_along(chars)) {
    if (chars[[i]] == "{") {
      depth <- depth + 1L
    } else if (chars[[i]] == "}") {
      depth <- depth - 1L
      if (depth == 0L) {
        end <- start + i - 1L
        break
      }
    }
  }
  if (is.na(end)) {
    stop("GeneticAlgorithm: malformed preset spec (unbalanced generator object)")
  }
  gen <- substr(spec_json, start, end)
  wrapped <- sprintf(
    '{"kind": "gen/compound", "blocks": [%s]}',
    paste(rep(gen, n_blocks), collapse = ", ")
  )
  paste0(substr(spec_json, 1L, start - 1L), wrapped, substring(spec_json, end + 1L))
}

.sz_ga_representation_to_preset <- list(
  real = "ga_real", perm = "ga_perm", bin = "ga_bin", int = "ga_int", cat = "ga_cat"
)

#' Genetic Algorithm -- auto-dispatches to `sz_preset_ga_real`/`ga_perm`/
#' `ga_bin`/`ga_int`/`ga_cat` (`R/000-wrappers.R`) based on `problem`'s own
#' search space, read via `prob$space()`/`.sz_space_to_blocks()`:
#'
#' \describe{
#'   \item{all-Float}{`sz_preset_ga_real`}
#'   \item{all-Permutation}{`sz_preset_ga_perm`}
#'   \item{all-Binary}{`sz_preset_ga_bin`}
#'   \item{all-Int}{`sz_preset_ga_int`}
#'   \item{all-Categorical}{`sz_preset_ga_cat`}
#' }
#'
#' A multi-block space whose blocks all share one kind is dispatched the
#' same way, but `$run()` wraps one copy of the preset's generator per block
#' in a `gen/compound` generator (init/boundary/replacer/termination stay
#' the preset's own), so every block is recombined independently. The same
#' wrap applies when `representation` is forced.
#'
#' A space MIXING block kinds (e.g. Float + Int together) has no `ga_*`
#' preset in this milestone -- `$run()` raises a clear error naming the
#' limitation (mirrors py-sezgi's `GeneticAlgorithm`'s own identical
#' `NotImplementedError`, M4-1 Task 5 deferral (d)).
#'
#' `representation` (a constructor kwarg, one of `"real"`/`"perm"`/
#' `"bin"`/`"int"`/`"cat"`) OVERRIDES auto-dispatch entirely -- block-kind
#' introspection is skipped whenever it is given.
#'
#' After a `$run()` call, `$dispatched_representation` holds the
#' representation actually engaged -- an introspectable field proving
#' which preset ran, independent of `representation` itself (which stays
#' `NULL` under auto-dispatch).
#'
#' See [sz_preset_classes]'s own doc for the problem-form support matrix
#' and result shape (identical here).
#'
#' @export
GeneticAlgorithm <- R6::R6Class("GeneticAlgorithm",
  public = list(
    pop_size = 20,
    representation = NULL,
    preset_kwargs = NULL,
    dispatched_representation = NULL,

    initialize = function(pop_size = 20, representation = NULL, ...) {
      if (!is.null(representation) && !(representation %in% names(.sz_ga_representation_to_preset))) {
        stop(sprintf(
          "representation must be one of %s or NULL (auto-dispatch), got %s",
          paste(sort(names(.sz_ga_representation_to_preset)), collapse = ", "),
          representation
        ))
      }
      self$pop_size <- pop_size
      self$representation <- representation
      self$preset_kwargs <- list(...)
      invisible(self)
    },

    run = function(problem, budget, seed = 0, run_id = 0) {
      representation <- if (!is.null(self$representation)) {
        self$representation
      } else {
        .sz_ga_representation(problem)
      }
      self$dispatched_representation <- representation
      preset_fn <- .sz_preset_fn(.sz_ga_representation_to_preset[[representation]])
      spec_json <- do.call(preset_fn, c(list(self$pop_size, budget), self$preset_kwargs))
      if (!inherits(problem, "sz_builtin_bbob")) {
        n_blocks <- length(.sz_space_to_blocks(sz_as_problem(problem)$space()))
        if (n_blocks > 1L) {
          spec_json <- .sz_ga_wrap_compound(spec_json, n_blocks)
        }
      }
      .sz_preset_run(spec_json, problem, budget, seed, run_id, "geneticalgorithm")
    }
  )
)

# ---- DifferentialEvolution --------------------------------------------------

.sz_de_variant_to_preset <- list(
  rand_1 = "de_rand_1", best_1 = "de_best_1", jde = "jde"
)

#' Differential Evolution -- delegates to `sz_preset_de_rand_1` (default,
#' DE/rand/1/bin), `sz_preset_de_best_1` (DE/best/1/bin), or
#' `sz_preset_jde` (self-adaptive jDE), selected by `variant` (one of
#' `"rand_1"`, `"best_1"`, `"jde"`) at `$new()` time. All three presets
#' share the identical `(pop_size, budget)` signature, so `variant` is the
#' only dispatch axis -- no problem introspection needed (unlike
#' [GeneticAlgorithm]'s space-driven auto-dispatch).
#'
#' See [sz_preset_classes]'s own doc for the problem-form support matrix
#' and result shape (identical here).
#'
#' @export
DifferentialEvolution <- R6::R6Class("DifferentialEvolution",
  public = list(
    pop_size = 20,
    variant = "rand_1",
    preset_kwargs = NULL,

    initialize = function(pop_size = 20, variant = "rand_1", ...) {
      if (!(variant %in% names(.sz_de_variant_to_preset))) {
        stop(sprintf(
          "variant must be one of %s, got %s",
          paste(sort(names(.sz_de_variant_to_preset)), collapse = ", "), variant
        ))
      }
      self$pop_size <- pop_size
      self$variant <- variant
      self$preset_kwargs <- list(...)
      invisible(self)
    },

    run = function(problem, budget, seed = 0, run_id = 0) {
      preset_fn <- .sz_preset_fn(.sz_de_variant_to_preset[[self$variant]])
      spec_json <- do.call(preset_fn, c(list(self$pop_size, budget), self$preset_kwargs))
      .sz_preset_run(spec_json, problem, budget, seed, run_id, "differentialevolution")
    }
  )
)

# ---- NSGA2 ------------------------------------------------------------------

#' Class skin over [sz_nsga2()] (`R/mo.R`) -- NOT one of the 34
#' `sz_preset_*` builders (NSGA-II is not built on the scalar
#' Engine/Registry/Generator/AlgorithmSpec machinery every `sz_preset_*`
#' targets, see this file's own module doc). ZERO new MO capability:
#' `$run()` delegates to `sz_nsga2()` VERBATIM, same positional/keyword
#' arguments, same return value shape.
#'
#' Constructor kwargs mirror `sz_nsga2()`'s own "algorithm configuration"
#' parameters -- `pop_size` plus every variation-operator knob (`eta_c`,
#' `eta_m`, `p_c`, `p_m`, `p_c_bin`, `p_m_bin`, `p_c_cat`, `p_m_cat`), the
#' ones that describe the algorithm instance itself, independent of which
#' problem it is pointed at. `$run()`'s own arguments mirror `sz_nsga2()`'s
#' remaining "this particular problem/run" parameters (`problem`, `dim`,
#' `budget`, `m`, `k`, `l`, `seed`, `log_dir`, `label`) -- `m`/`k`/`l`
#' describe the PROBLEM being solved (`m` = objective count for dtlz/wfg;
#' `k`/`l` = WFG's own shape parameters), not the algorithm, so they belong
#' with `$run()` rather than `$new()`, exactly mirroring `sz_nsga2()`'s own
#' per-problem-family validation.
#'
#' `NSGA2$new(pop_size = P, ...)$run(problem, dim, budget, m = M, seed = S,
#' ...)` is IDENTICAL to calling `sz_nsga2(problem, dim, P, budget, m = M,
#' seed = S, ...)` directly (see `test-oop-builtins.R`'s own anchored
#' equivalence tests, a ZDT anchor and a WFG anchor where `k`/`l` matter).
#'
#' `$run()`'s return value is `sz_nsga2()`'s OWN list shape (`individuals`,
#' `objectives`, `front0`, `evals_used`, `violations` when constrained) --
#' NOT an `sz_result` (T4's `.sz_wrap_result()` shape assumes a
#' single-objective run with one best point, which does not fit
#' NSGA-II's multi-objective Pareto-front result).
#'
#' @export
NSGA2 <- R6::R6Class("NSGA2",
  public = list(
    pop_size = NULL,
    eta_c = 20.0,
    eta_m = 20.0,
    p_c = 0.9,
    p_m = NULL,
    p_c_bin = 0.9,
    p_m_bin = NULL,
    p_c_cat = 0.9,
    p_m_cat = NULL,

    initialize = function(pop_size, eta_c = 20.0, eta_m = 20.0, p_c = 0.9, p_m = NULL,
                          p_c_bin = 0.9, p_m_bin = NULL, p_c_cat = 0.9, p_m_cat = NULL) {
      self$pop_size <- pop_size
      self$eta_c <- eta_c
      self$eta_m <- eta_m
      self$p_c <- p_c
      self$p_m <- p_m
      self$p_c_bin <- p_c_bin
      self$p_m_bin <- p_m_bin
      self$p_c_cat <- p_c_cat
      self$p_m_cat <- p_m_cat
      invisible(self)
    },

    run = function(problem, dim, budget, m = NULL, seed = 0, k = NULL, l = NULL,
                   log_dir = NULL, label = NULL) {
      sz_nsga2(
        problem, dim, m = m, pop_size = self$pop_size, budget = budget, seed = seed,
        eta_c = self$eta_c, eta_m = self$eta_m, p_c = self$p_c, p_m = self$p_m,
        p_c_bin = self$p_c_bin, p_m_bin = self$p_m_bin, p_c_cat = self$p_c_cat,
        p_m_cat = self$p_m_cat, k = k, l = l, log_dir = log_dir, label = label
      )
    }
  )
)
