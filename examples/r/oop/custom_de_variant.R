# OOP example: a custom DE/rand/1-shaped sezgi::PopulationAlgorithm variant
# (M4-2 Task 4) -- overrides ONLY vary(), keeping the base's default
# tournament select(). The R twin of examples/python/oop/custom_de_variant.py
# (M4-1 Task 4): same DE/rand/1 mutation (r1 + F * (r2 - r3), donor indices
# drawn via ctx$rng$next_below(), never R's own global RNG/sample()), run
# against sezgi::sz_builtin_bbob() -- the "built-in form" of Algorithm$run()
# (see R/algorithm.R's own doc: r-sezgi has no general native-problem
# handle, so this is a purpose-built descriptor routing to T3's
# sz_solve_r_generator_bbob() entry point). No pure-script twin to
# reproduce -- authored directly against the engine-hosted
# PopulationAlgorithm base, same precedent as examples/r/oop/tsp_two_opt.R.
# Its own determinism/anchored-output gate is the stopifnot() below.

library(sezgi)

F <- 0.5 # DE differential weight

CustomDERandOne <- R6::R6Class("CustomDERandOne",
  inherit = PopulationAlgorithm,
  public = list(
    # DE/rand/1-shaped vary(): for each parent slot, draws THREE donor
    # indices via ctx$rng$next_below(n) and combines them as
    # r1 + F * (r2 - r3) -- applied directly over the tournament-selected
    # parent pool this base's default select() already built. No separate
    # crossover step (kept deliberately small, matching the Python twin).
    vary = function(parents, ctx) {
      n <- length(parents)
      dim <- length(parents[[1]])
      out <- matrix(0, n, dim)
      for (i in seq_len(n)) {
        r1 <- ctx$rng$next_below(as.double(n)) + 1
        r2 <- ctx$rng$next_below(as.double(n)) + 1
        r3 <- ctx$rng$next_below(as.double(n)) + 1
        out[i, ] <- parents[[r1]] + F * (parents[[r2]] - parents[[r3]])
      }
      out
    }
  )
)

res <- CustomDERandOne$new()$run(sz_builtin_bbob(1, 10, 1), budget = 2000, seed = 42, pop_size = 20)
cat(sprintf(
  "custom_de_variant (oop): evals_used=%d best_f=%.6g\n",
  res$evals_used, res$best_f
))

# Anchored: pinned by actually running this script (M3-8 README lesson --
# examples are executable regression evidence, not just illustrations).
stopifnot(identical(res$evals_used, 2000))
stopifnot(isTRUE(all.equal(res$best_f, -59.3945547399384)))
