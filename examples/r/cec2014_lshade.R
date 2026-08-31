# L-SHADE (Tanabe & Fukunaga 2014) on CEC 2014 (Liang, Qu & Suganthan 2013)
# function f1, dim=10 -- a matched Python/R example pair for sezgi's M3-6
# CEC 2014 benchmark suite. See examples/python/cec2014_lshade.py for the
# Python counterpart, which runs sezgi.presets.lshade through
# sezgi.solve() against sezgi.problems.cec2014(fid, dim).
#
# L-SHADE (SHADE with linear population-size reduction) placed 1st in the
# CEC 2014 competition itself, making it the natural pairing for this
# suite -- the same relationship examples/r/cec2022_shade.R has to CEC 2022
# (SHADE, the CEC 2013 winner and L-SHADE's own ancestor).
#
# f1 is CEC 2014's "Rotated High Conditioning Elliptic Function" -- a
# unimodal function with an ill-conditioned, widely spread per-coordinate
# scale, which is why even a competitive optimizer's best_f stays visibly
# above F* at a small smoke-demo budget; this is expected for f1's shape,
# not a defect. See docs/DECISIONS.md's M3-6 record for the CEC 2014
# suite's provenance, vendored-data totals, and every report-vs-official-C
# divergence found across the whole suite (f1 itself has none).
#
# Runs through sz_solve_cec2014 (spec_json, fid, dim, master_seed, run_id)
# with sz_preset_lshade(dim, budget), the SAME Rust core the Python script
# runs through sezgi.solve()/presets.lshade. sz_preset_lshade is
# parameterized by dim (initial population size 18*dim, then shrinking
# per Tanabe & Fukunaga's own linear reduction schedule), not pop_size
# directly -- unlike sz_preset_shade.
#
# Single seed, single problem, small budget -- a SMOKE demonstration of the
# CEC 2014 solve()-integrated binding, not a claim about L-SHADE's quality
# or convergence rate. Comparing algorithms properly needs many seeds via
# sz_per_budget_packages/statistical tests (see README.md's "Experiments &
# Statistics" section) -- never a single-seed gap number.
#
# Cross-language number check: this script's printed best_f matches the
# Python script's best_f BIT-FOR-BIT (verified via the writeBin/struct.pack
# byte comparison this project uses throughout -- see
# r-sezgi/tests/testthat/test-cec1417.R -- same algorithm spec, same
# problem, same master_seed, same run_id, routed through the identical
# Rust core in both languages, so there is no reason for them to differ,
# and they don't: both sides produce the byte string 4066875c33a1c67b).

library(sezgi)

fid <- 1
dim <- 10
budget <- 9000
seed <- 20260830

f_star <- sz_cec2014_f_star(fid)
spec <- sz_preset_lshade(dim, budget)

r <- sz_solve_cec2014(spec, fid = fid, dim = dim, master_seed = seed, run_id = 0)

cat(sprintf("CEC 2014 f%d (dim=%d), L-SHADE, budget=%d seed=%d -- SMOKE DEMO, single seed\n",
            fid, dim, budget, seed))
cat(sprintf("F* (report's pinned optimum): %s\n", f_star))
cat(sprintf("best_f: %s\n", r$best_f))
cat(sprintf("gap (best_f - F*): %s\n", r$best_f - f_star))
cat(sprintf("evals: %d\n", r$evals))
