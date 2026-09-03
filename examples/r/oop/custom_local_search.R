# OOP example: a custom sezgi::LocalSearch variant (M4-2 Task 4) --
# overrides ONLY neighbor(), keeping the base's default greedy accept()
# (f_new <= f_old). The R twin of examples/python/oop/custom_local_search.py
# (M4-1 Task 4): same coordinate-wise perturbation in [-STEP, STEP] via
# ctx$rng$next_f64(), run against sezgi::sz_builtin_bbob() -- see
# R/algorithm.R's own class doc (ACCEPT() DESIGN) for exactly what
# accept() does and does not control at pop_size = 1 under the engine's
# default replace/mu-plus-lambda replacer. Same new-style precedent as
# custom_de_variant.R: authored directly against the engine-hosted family
# base, no pure-script twin. Its own determinism/anchored-output gate is
# the stopifnot() below.

library(sezgi)

STEP <- 0.3 # perturbation half-width

CustomPerturbationSearch <- R6::R6Class("CustomPerturbationSearch",
  inherit = LocalSearch,
  public = list(
    # neighbor(): perturbs every coordinate of x by an independent uniform
    # draw in [-STEP, STEP] via ctx$rng$next_f64(). The base's default
    # greedy accept() tracks whichever of {current, neighbor} the engine's
    # own replace/mu-plus-lambda replacer already keeps at pop_size = 1.
    neighbor = function(x, ctx) {
      x + vapply(seq_along(x), function(i) (ctx$rng$next_f64() - 0.5) * 2 * STEP, numeric(1))
    }
  )
)

res <- CustomPerturbationSearch$new()$run(sz_builtin_bbob(1, 10, 1), budget = 2000, seed = 42)
cat(sprintf(
  "custom_local_search (oop): evals_used=%d best_f=%.6g\n",
  res$evals_used, res$best_f
))
# Anchored: pinned by actually running this script (M3-8 README lesson --
# examples are executable regression evidence, not just illustrations).
stopifnot(identical(res$evals_used, 2000))
stopifnot(isTRUE(all.equal(res$best_f, -84.3232356341047)))
