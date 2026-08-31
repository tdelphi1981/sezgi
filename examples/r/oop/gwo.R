# OOP twin of examples/r/gwo.R -- same math, same RNG draw order.
#
# Port of the pure-R Grey Wolf Optimizer script onto sz_algorithm/
# sz_algo_solve (r-sezgi's M3-5 pure-R authoring surface, the R mirror of
# py-sezgi's sezgi.Algorithm -- see examples/python/oop/gwo.py for the
# Python twin this file matches structurally). The pure script (and the
# Rust module doc it cites, crates/components/src/gwo.rs) remains the
# provenance for the update equations; this file re-derives NOTHING and
# reproduces the pure script's evals_used/best_f/gap output STRING-EXACTLY
# at the same seed -- enforced by r-sezgi/tests/testthat/test-algo.R's
# "R OOP gwo twin matches the pure gwo.R script" test (subprocess, both
# scripts run via Rscript, printed fields compared as strings).
#
# Draw-order analysis (why identity is achievable here): `sz_algo_solve`
# calls `set.seed(seed)` exactly once, before `algo$setup(ctx)` runs, and
# R's global RNG stream is the ONE stream both `ctx$random_point()` and any
# direct `runif()` call inside setup()/step() draw from (see
# r-sezgi/R/algo.R's `.sz_algo_context` doc). The pure script's own
# structure is: `set.seed(seed)` -> construct the session (no RNG use in
# session construction, verified: no rand/Rng/runif/sample symbol appears
# in r-sezgi/src/rust/src/session.rs) -> `pop <- lapply(1:pop_size,
# function(i) runif(dim, lo, hi))` -> `fitness <- s$evaluate(pop)` -> the
# generation while-loop, each iteration drawing `runif(6)` once per
# (candidate, dimension) in the exact nesting order `for (i in
# 1:pop_size) for (d in 1:dim)`. Because session construction touches no
# RNG state, building `s` before calling `sz_algo_solve` (which is where
# `set.seed` actually happens) does not shift anything -- the first draw
# either script makes is the pure script's init `runif(dim, lo, hi)` /
# this twin's init `ctx$random_point()` (`runif(session$dim(), lo, hi)`,
# byte-identical call), both immediately after `set.seed(seed)`. `setup()`
# below reproduces the init draws verbatim (same `lapply` order); `step()`
# reproduces one generation's `for (i) for (d) runif(6)` nesting verbatim.
#
# The one structural difference: the pure script's `while (used + pop_size
# <= budget)` guard is checked BEFORE a generation's body runs, so its
# final (would-be 66th) generation is never entered and draws NOTHING. This
# twin's `step()` has no such up-front guard -- the driver calls it
# whenever `ctx$remaining() > 0` (here, 20 remaining after 65 full
# generations), so the twin's final `step()` call DOES compute a full
# offspring batch (900 `runif(6)` draws) before `ctx$evaluate(offspring)`
# raises `sz_budget_exhausted` (30 > 20 remaining), caught by
# `sz_algo_solve`'s driver, ending the run. Those 900 draws are wasted --
# `ctx$evaluate` rejects the batch BEFORE touching the session (see
# `.sz_algo_context`'s pre-charge check), so `pop`/`fitness` are never
# reassigned for that failed step, and no further draws happen afterward in
# EITHER script. Since the printed fields (`evals_used`/`best_f`/`gap`) are
# read from the session's own state -- which only advances on a
# SUCCESSFUL `evaluate()` call, and every successful call up to and
# including the 65th generation drew IDENTICAL random numbers in IDENTICAL
# order in both scripts -- this trailing, discarded draw burst on the
# twin's side does not change the final printed output at all. Verified
# below, not just argued: the twin's live output matches the pure script's
# field-for-field.

library(sezgi)

DIM <- 5
BUDGET <- 2000
POP_SIZE <- 30
SEED <- 42
LO <- -5.0
HI <- 5.0

gwo_dim_step <- function(a, leaders_d, x_d, draws) {
  total <- 0.0
  for (k in 0:2) {
    r1 <- draws[2 * k + 1]
    r2 <- draws[2 * k + 2]
    big_a <- 2.0 * a * r1 - a
    c <- 2.0 * r2
    total <- total + (leaders_d[k + 1] - big_a * abs(c * leaders_d[k + 1] - x_d))
  }
  total / 3.0
}

clamp <- function(x, lo, hi) min(max(x, lo), hi)

# Cross-step state (pop/fitness) lives as local variables in
# `make_gwo()`'s own execution environment; `setup`/`step` are two
# closures defined inside that same call, so they share it as their
# common enclosing environment. Each closure updates the shared variables
# via `<<-` (assign in the nearest enclosing environment where the name
# already exists -- `make_gwo`'s frame, since `pop`/`fitness` are
# initialized there) rather than returning state through `ctx` (which has
# no slot for algorithm-private state) or using a package-global `<<-`
# into .GlobalEnv (which would leak/collide across runs). This is the
# base-R idiom the brief calls for: a closure-local environment, not a
# separate `new.env()` object threaded through by hand.
make_gwo <- function() {
  pop <- NULL
  fitness <- NULL

  setup <- function(ctx) {
    pop <<- lapply(1:POP_SIZE, function(i) ctx$random_point())
    fitness <<- ctx$evaluate(pop)
  }

  step <- function(ctx) {
    used <- ctx$evals_used()
    progress <- min(max(used / ctx$budget(), 0.0), 1.0)
    a <- 2.0 - 2.0 * progress

    # order() is a stable sort: ties keep original (lower-index) order first.
    leaders <- order(fitness)[1:3]
    leader_x <- pop[leaders]

    offspring <- vector("list", POP_SIZE)
    for (i in 1:POP_SIZE) {
      x <- pop[[i]]
      new_x <- numeric(DIM)
      for (d in 1:DIM) {
        leaders_d <- vapply(leader_x, function(lx) lx[d], numeric(1))
        draws <- runif(6)
        new_x[d] <- clamp(gwo_dim_step(a, leaders_d, x[d], draws), LO, HI)
      }
      offspring[[i]] <- new_x
    }

    # replace/generational, whole-batch-or-nothing: ctx$evaluate() raises
    # sz_budget_exhausted (caught by sz_algo_solve's driver) instead of
    # committing a partial generation -- reproduces the pure script's own
    # `while used + pop_size <= budget` guard as a boundary condition
    # rather than a loop precondition (see header comment above).
    fitness <<- ctx$evaluate(offspring)
    pop <<- offspring
  }

  sz_algorithm(setup, step, name = "gwo-r-oop")
}

s <- sz_eval_session(fid = 1, dim = DIM, instance = 1, budget = BUDGET)
res <- sz_algo_solve(make_gwo(), s, seed = SEED)
cat(sprintf("gwo (oop): evals_used=%d best_f=%.6g gap=%.6g\n",
            res$evals_used, res$best_f, res$gap))
