# NSGA-II (Deb, Pratap, Agarwal & Meyarivan 2002) on ZDT1 (Zitzler, Deb &
# Thiele 2000) -- a matched Python/R example pair for sezgi's
# multi-objective baseline. See examples/python/nsga2_zdt1.py for the
# Python counterpart: same scenario, same Rust core underneath (NSGA-II is
# a self-contained, seeded Rust runner in BOTH bindings, unlike the
# pure-Python/pure-R algorithm scripts elsewhere in this directory), so
# this pair's numbers are bit-identical across languages -- only the
# report formatting differs.
#
# Calls sz_nsga2() directly rather than going through an ask/tell
# sz_eval_session() or a component spec: NSGA-II ships as a self-contained,
# seeded reference runner over the parallel MoProblem/MoEvaluator surface,
# not as a composable component graph -- MO component-graph specs (the
# ExperimentSpec/spec-JSON path the rest of this catalog uses) are
# deferred to v2 (see README.md's "Multi-objective optimization"
# section for the full spec-tension ruling). Accordingly, there is no specs/nsga2_zdt1.toml in this catalog.
#
# Small budget, seeded, for a fast illustrative run -- no cross-algorithm
# or cross-run quality claims are made here (single seed, single problem).
# pop_size must be a multiple of 4 (KanGAL-faithful tightening of the
# naive "even, >= 4" rule; see crates/components/src/nsga2.rs).

library(sezgi)

problem <- "zdt1"
dim <- 10
pop_size <- 40
budget <- 4000
seed <- 20260830
ref_point <- c(1.1, 1.1) # ZDT1 objectives lie in [0,1]x[0,1]; 1.1 dominates the whole front

result <- sz_nsga2(problem, dim = dim, pop_size = pop_size, budget = budget, seed = seed)
front0_objectives <- result$objectives[result$front0] # 1-based front0, R convention
reference_front <- sz_mo_pareto_front(problem, dim = dim, n = 200)

# sz_mo_hypervolume_2d/sz_mo_igd take R-idiomatic numeric matrices (rows =
# points), unlike py-sezgi's list-of-lists -- see r-sezgi/R/mo.R.
front0_mat <- do.call(rbind, front0_objectives)
ref_mat <- do.call(rbind, reference_front)

hv <- sz_mo_hypervolume_2d(front0_mat, ref_point)
igd_value <- sz_mo_igd(front0_mat, ref_mat)

cat(sprintf("problem: %s  dim=%d  pop_size=%d  budget=%d  seed=%d\n",
            problem, dim, pop_size, budget, seed))
cat(sprintf("front size: %d\n", length(front0_objectives)))
cat(sprintf("evals_used: %d\n", result$evals_used))
cat(sprintf("hypervolume_2d (ref_point=[%.1f,%.1f]): %s\n", ref_point[1], ref_point[2], hv))
cat(sprintf("igd (vs pareto_front(200)): %s\n", igd_value))

ord <- order(front0_mat[, 1])
cat("front points (f1, f2), sorted by f1, first 5 (plot-ready -- x=f1, y=f2):\n")
for (i in ord[1:5]) {
  cat(sprintf("  %.6f  %.6f\n", front0_mat[i, 1], front0_mat[i, 2]))
}
