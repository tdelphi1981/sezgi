# ga_bin (KanGAL two-point binary crossover + bit-flip mutation, Deb,
# Pratap, Agarwal & Meyarivan 2002, Sec. IV.A -- p_c=0.9, p_m=1/L) on OneMax
# (Goldberg 1989's classic "maximize the number of set bits" toy,
# re-expressed in this crate as minimizing the ZERO-bit count -- see
# crates/problems/src/diagnostics.rs's module doc) -- a matched Python/R
# example pair for sezgi's M3-8 typed-operator milestone. See
# examples/python/onemax_ga.py for the Python counterpart: same scenario,
# same Rust core underneath (both bindings run the identical
# `AlgorithmSpec`/`Engine` path, like every other matched pair in this
# project), so this pair's numbers are bit-identical across languages --
# only the report formatting differs (per the M3-7 example-pair "numbers
# bit-identical, formatting divergence acceptable" clarification,
# docs/DECISIONS.md's M3-7 record).
#
# Runs through r-sezgi's own established sz_solve_* / sz_preset_* path:
# sz_preset_ga_bin(pop_size, budget) + sz_solve_onemax(spec, n_bits, ...).
#
# Small budget, seeded, deliberately chosen so the run does NOT reach the
# known optimum (best_f == 0, all bits set) -- a fully-converged run would
# be a trivial, uninformative anchor (any correctly terminating optimal run
# gives the identical 0 regardless of path taken, the same reasoning M3-8
# Task 10's own cross-language anchor report gives for avoiding its own
# OneMax scenario's easy exact-optimum case). At n_bits=100, pop_size=20,
# budget=400 (== 20 generations past init), seed=7, ga_bin lands on a
# genuine partial result (16 of 100 bits still zero) that pins the run's
# actual trajectory, not just "did it solve the toy."
#
# Single seed, single problem, small budget -- a SMOKE demonstration of the
# binding (typed operators + diagnostic problems reachable end-to-end from
# R), not a claim about ga_bin's quality relative to any other algorithm or
# to a proper multi-seed benchmark. Comparing algorithms properly needs many
# seeds via sz_per_budget_packages/statistical tests (see README.md's
# "Experiments & Statistics" section) -- never a single-seed gap number.

library(sezgi)

n_bits <- 100
pop_size <- 20
budget <- 400
master_seed <- 7

spec <- sz_preset_ga_bin(pop_size, budget)

result <- sz_solve_onemax(spec, n_bits = n_bits, master_seed = master_seed, run_id = 0)
best_f <- result$best_f
best_x <- result$best_x # logical[n_bits], TRUE = set bit

# Independent cross-check: recompute the reported best_f directly from
# best_x (OneMax's own zero-bit-count formula) -- the same
# "recompute, don't just trust" pattern tsp_ga_perm.R's own
# sz_tsp_tour_length cross-check uses.
ones <- sum(best_x)
zeros <- length(best_x) - ones
stopifnot(identical(as.numeric(zeros), as.numeric(best_f)))

known_optimum <- 0 # OneMax's known optimum: all bits set (Goldberg 1989)
gap <- best_f - known_optimum

cat(sprintf("OneMax n_bits=%d, ga_bin, pop_size=%d budget=%d seed=%d -- SMOKE DEMO, single seed\n",
            n_bits, pop_size, budget, master_seed))
cat(sprintf("known optimum: %s\n", known_optimum))
cat(sprintf("evals_used=%s best_f=%s gap=%s ones=%s\n", result$evals, best_f, gap, ones))
