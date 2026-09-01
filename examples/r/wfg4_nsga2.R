# NSGA-II (Deb, Pratap, Agarwal & Meyarivan 2002) on WFG4 (Huband, Barone,
# While & Hingston, EMO 2005, "corrected version: 22 June 2005"; later
# Huband, Hingston, Barone & While, IEEE TEC 2006) -- a matched Python/R
# example pair for sezgi's M3-7 multi-objective remainders. See
# examples/python/wfg4_nsga2.py for the Python counterpart: same scenario,
# same Rust core underneath (NSGA-II is a self-contained, seeded Rust
# runner in BOTH bindings, exactly like examples/r/nsga2_zdt1.R's M3-2
# pair), so this pair's numbers are bit-identical across languages -- only
# the report formatting differs.
#
# Unlike nsga2_zdt1.R (which reports the raw front0/hypervolume_2d/igd
# triple), this pair exercises the FULL M3-7 surface added since M3-2: the
# WFG problem suite (crates/problems/src/wfg.rs), sezgi-moa v1 archive-first
# run logging (sz_nsga2(..., log_dir=, label=), M3-7 Task 9), the reader
# (sz_mo_read_moa), and the general-M exact hypervolume (sz_mo_hypervolume,
# M3-7 Task 8) -- WFG4 is 2-objective here, but sz_mo_hypervolume (unlike
# the frozen sz_mo_hypervolume_2d) is the general-M implementation, used
# for exactly that reason: this pair is meant to demonstrate the new entry
# point, not the old 2D-only one nsga2_zdt1.R already covers.
#
# WFG4's decision-space dimension is DERIVED (dim = k + l), not given
# directly -- k/l are left at NULL here so the binding resolves them to the
# WFG toolkit's own recommended defaults (crates/problems/src/wfg.rs module
# doc, "Recommended k/l defaults" section, transcribed from
# WFG_v2006.03.28/README.txt): k=4 for m=2, l=20 unconditionally, so
# dim = k + l = 24 for this run (m=2).
#
# sezgi decision: "archive size" and "hypervolume of the final front" (this
# pair's own printed report) both refer to the SAME object: the sezgi-moa
# archive reconstructed via sz_mo_read_moa(...)$archive at the run's own
# final budget (at=NULL), not front0 (the final POPULATION's own
# non-dominated subset, a narrower and run-order-dependent set). The
# archive is the unbounded nondominated set accumulated over every
# evaluated individual across the whole run (crates/bench/src/mo_archive.rs's
# own "trajectory, not compaction" design) -- reading it back via
# log_dir/label is also how this pair exercises Task 9's logging path,
# rather than only printing numbers sz_nsga2() already returns in memory.
#
# Reference point (the WFG4/m=2 analytic front's nadir x 1.1, per
# crates/stats/src/moo_indicators.rs's own "Choosing a reference point"
# convention and its Ishibuchi, Imada, Setoguchi & Nojima 2018 citation):
# WFG4's shape functions are h_m = concave_m for every m (Huband et al.,
# Table 6), so at a Pareto-optimal point (x_M = 0) the m=2 front satisfies
# f1 = S1*sin(x1*pi/2), f2 = S2*cos(x1*pi/2) with S1=2, S2=4 (Sm=1:M = 2m),
# i.e. sum_j (f_j / (2j))^2 = 1 -- a concave arc from (0,4) to (2,0). The
# componentwise-worst point across that arc (the nadir) is (2.0, 4.0)
# (f1's max of 2.0 at x1=1, f2's max of 4.0 at x1=0 -- verified live against
# sz_mo_pareto_front("wfg4", dim=NULL, n=50, m=2)). Scaled by 1.1: (2.2, 4.4).
#
# Small budget, seeded, for a fast illustrative run -- no cross-algorithm or
# cross-run quality claims are made here (single seed, single problem).
# pop_size must be a multiple of 4 (KanGAL-faithful tightening of the naive
# "even, >= 4" rule; see crates/components/src/nsga2.rs).

library(sezgi)

problem <- "wfg4"
m <- 2
pop_size <- 40
budget <- 4000
seed <- 20260830
label <- "wfg4demo"
ref_point <- c(2.2, 4.4) # nadir (2.0, 4.0) x 1.1 -- see file header above

log_dir <- tempfile()
result <- sz_nsga2(problem, dim = NULL, m = m, pop_size = pop_size, budget = budget, seed = seed,
                    log_dir = log_dir, label = label)
archive_run <- sz_mo_read_moa(file.path(log_dir, sprintf("%s-s%d.moa", label, seed)))

archive_mat <- do.call(rbind, archive_run$archive)
front0_objectives <- result$objectives[result$front0] # 1-based front0, R convention

archive_size <- nrow(archive_mat)
hv <- sz_mo_hypervolume(archive_mat, ref_point)

cat(sprintf("problem: %s  m=%d  k=4 l=20 (toolkit defaults, dim=24)  pop_size=%d  budget=%d  seed=%d\n",
            problem, m, pop_size, budget, seed))
cat(sprintf("evals_used: %d\n", result$evals_used))
cat(sprintf("archive size: %d\n", archive_size))
cat(sprintf("hypervolume (ref_point=[%.1f,%.1f]): %s\n", ref_point[1], ref_point[2], hv))

front0_mat <- do.call(rbind, front0_objectives)
ord <- order(front0_mat[, 1])
cat("front0 points (f1, f2), sorted by f1, first 5 (plot-ready -- x=f1, y=f2):\n")
for (i in ord[1:5]) {
  cat(sprintf("  %.6f  %.6f\n", front0_mat[i, 1], front0_mat[i, 2]))
}
