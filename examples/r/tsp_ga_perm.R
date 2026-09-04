# ga-perm (a fused permutation GA -- OX1 crossover + swap mutation in one
# generator, see crates/components/src/presets.rs::ga_perm's doc) on
# berlin52, the classic TSPLIB (Reinelt) benchmark instance -- a matched
# Python/R example pair for sezgi's TSPLIB/permutation-problem suite.
# See examples/python/tsp_ga_perm.py for the Python counterpart (0-based
# tour indices there vs. this script's 1-based, per r-sezgi's own indexing
# convention -- see
# r-sezgi/src/rust/src/problems.rs's module doc).
#
# Runs through r-sezgi's own established sz_solve_* / sz_preset_* path:
# sz_preset_ga_perm(pop_size, budget) + sz_solve_tsp(spec, name, ...).
#
# berlin52's published-optimal tour length is EXACTLY 7542.0 (TSPLIB,
# Reinelt 1991) -- ga-perm is a baseline permutation GA with no local
# search (no 2-opt), so it is not expected to reach the optimum; the point
# of this example is a valid, budget-respecting solve run and an honest gap
# report.
#
# Single seed, single problem, small budget -- a SMOKE demonstration, not a
# claim about ga-perm's quality relative to any other algorithm or to a
# proper multi-seed benchmark. Comparing algorithms properly needs many
# seeds via sz_per_budget_packages/statistical tests (see README.md's
# "Experiments & Statistics" section) -- never a single-seed gap number.

library(sezgi)

instance <- "berlin52"
pop_size <- 32
budget <- 5000
master_seed <- 20260830

info <- sz_tsp_load(instance)
known_optimum <- info$known_optimum
spec <- sz_preset_ga_perm(pop_size, budget)

result <- sz_solve_tsp(spec, instance, master_seed = master_seed, run_id = 0)
best_length <- result$best_f
best_tour <- result$best_x # 1-based permutation of 1:n_cities

# Independent cross-check: recompute the reported best tour's length directly.
recomputed <- sz_tsp_tour_length(instance, best_tour)
stopifnot(identical(recomputed, best_length))

cat(sprintf("TSP %s (%d cities), ga-perm, pop_size=%d budget=%d seed=%d -- SMOKE DEMO, single seed\n",
            instance, info$n_cities, pop_size, budget, master_seed))
cat(sprintf("known optimum (TSPLIB): %s\n", known_optimum))
cat(sprintf("best tour length: %s\n", best_length))
cat(sprintf("gap (best - optimum): %s  ratio: %.4f\n", best_length - known_optimum, best_length / known_optimum))
cat(sprintf("evals_used: %s\n", result$evals))
