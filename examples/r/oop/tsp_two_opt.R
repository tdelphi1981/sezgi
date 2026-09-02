# OOP example: sz_algorithm-based local search on a vendored TSP instance --
# a seeded random start (ctx$random_permutation()) plus a first-improvement
# 2-opt sweep (ctx$two_opt()), run until the budget is exhausted.
#
# M3-8 Task 8: the R twin of examples/python/oop/tsp_two_opt.py (M3-8 Task
# 7) -- the FIRST worked example over a permutation-typed session on the R
# side, not a port of an existing pure script -- unlike every other file in
# this directory (e.g. examples/r/oop/gwo.R), it deliberately sits OUTSIDE
# r-sezgi/tests/testthat/test-algo.R's "R OOP twin matches the pure script"
# subprocess-diff gate (that gate compares an OOP script against a
# pure-script counterpart under examples/r/<algo>.R; this file has no such
# pure-script twin to reproduce -- there is no examples/r/tsp_two_opt.R).
# Its own determinism/anchored-output gate is
# r-sezgi/tests/testthat/test-tsp-two-opt.R instead, following the same
# "run the script as a subprocess, parse its printed metrics line" pattern
# test-algo.R's own gwo twin test uses.
#
# SAME instance (berlin52), SAME seed (42), SAME budget (2000), and the
# SAME first-improvement 2-opt sweep order as tsp_two_opt.py -- Python's
# 0-based `0 <= i < j < n` sweep is this script's 1-based
# `1 <= i < j <= n` sweep (ctx$two_opt's own inclusive [i, j] semantics,
# see .sz_two_opt()'s doc in R/algo.R) -- so this script reproduces the
# SAME printed evals_used/best_f/gap/tour_length numbers as the Python
# twin: tour LENGTH is invariant to whether the tour itself is indexed
# 0-based (Python) or 1-based (R), so anchoring on these numbers lets both
# scripts match without needing the same raw tour representation.
# random_permutation()'s draw is ALSO bit-identical to Python's own (both
# sides derive their RngStream from the SAME seed via the SAME
# PERM_SESSION_RNG_TAG/fisher_yates_shuffle core -- see
# r-sezgi/src/rust/src/session.rs's module doc, "Permutation-typed
# sessions") -- R's random_permutation() literally returns the Python
# script's own first draw, shifted +1 -- but this script does not need to
# rely on that fact to reproduce the SAME best_f: the 2-opt sweep from that
# identical starting tour is itself an index-order-preserving mirror of the
# Python sweep (see below), so the two scripts land on the same tour by
# construction, not by coincidence.

library(sezgi)

INSTANCE <- "berlin52"
BUDGET <- 2000
SEED <- 42

make_tsp_two_opt <- function() {
  pairs <- NULL
  k <- 0
  tour <- NULL
  best_f <- NULL

  # setup(): draws one random 1-based tour (ctx$random_permutation()) and
  # evaluates it -- the run's starting point. `pairs` enumerates every
  # 2-opt segment `(i, j)` with `1 <= i < j <= n` in a FIXED,
  # deterministic order (outer i ascending, inner j ascending -- the exact
  # 1-based translation of the Python script's `(i, j) for i in
  # range(n - 1) for j in range(i + 1, n)`), so the sweep itself draws no
  # randomness.
  setup <- function(ctx) {
    n <- ctx$n()
    idx <- expand.grid(j = 1:n, i = 1:n)
    idx <- idx[idx$i < idx$j, ]
    idx <- idx[order(idx$i, idx$j), ]
    pairs <<- idx
    k <<- 0
    tour <<- ctx$random_permutation()
    best_f <<- ctx$evaluate(matrix(tour, nrow = 1))[1]
  }

  # step(): evaluates the NEXT candidate in the fixed sweep, wrapping
  # around and restarting once every pair has been tried (so a step always
  # has a next candidate regardless of the budget -- exactly ONE evaluation
  # is charged per step() call, matching every other OOP example's "consume
  # at least one eval per step" contract). Accepts the candidate (replacing
  # the current tour) only if it strictly improves the current tour's
  # length (first-improvement, not best-improvement).
  step <- function(ctx) {
    row <- (k %% nrow(pairs)) + 1
    i <- pairs$i[row]
    j <- pairs$j[row]
    k <<- k + 1
    candidate <- ctx$two_opt(tour, i, j)
    f <- ctx$evaluate(matrix(candidate, nrow = 1))[1]
    if (f < best_f) {
      tour <<- candidate
      best_f <<- f
    }
  }

  sz_algorithm(setup, step, name = "tsp_two_opt-r-oop")
}

s <- sz_eval_session_tsp(INSTANCE, budget = BUDGET, seed = SEED)
res <- sz_algo_solve(make_tsp_two_opt(), s, seed = SEED)
cat(sprintf("tsp_two_opt (oop): evals_used=%d best_f=%.6g gap=%.6g tour_length=%.6g\n",
            res$evals_used, res$best_f, res$gap, res$best_f))
