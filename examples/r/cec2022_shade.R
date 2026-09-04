# CEC 2022 (Kumar, Price, Mohamed, Hadi & Suganthan 2021) function f3,
# dim=10 -- a matched Python/R example pair for sezgi's CEC 2022
# benchmark suite. See examples/python/cec2022_shade.py for the Python
# counterpart, which runs sezgi.presets.shade (Tanabe & Fukunaga 2013's
# SHADE) through sezgi.solve() against sezgi.problems.cec2022(fid, dim).
#
# r-sezgi originally bound ONLY direct evaluation for CEC 2022
# (sz_cec2022_evaluate/sz_cec2022_f_star), a real capability gap -- so
# this script used to run a small classic DE/rand/1/bin loop written
# directly in base R instead of the SHADE preset. sz_solve_cec2022 was
# added later (mirrors sz_solve_bbob/sz_solve_tsp: Engine::from_spec +
# engine.run against Cec2022::new(fid, dim)), so this script now runs THE
# SAME SHADE preset through THE SAME Rust core as the Python script --
# same algorithm, same problem, same seed, in both languages.
#
# f3 is the CEC 2022 report's "Shifted and full Rotated Expanded Schaffer's
# f6 Function" BY NAME -- but the vendored official C reference actually
# dispatches problem 3 to plain Schaffer's F7 with no scale and (due to a
# verified buffer-reuse bug in the reference code) no effective rotation;
# sezgi follows the C code, which is what scored the competition, not the
# report's printed formula. See
# crates/problems/src/cec2022/mod.rs's module doc ("F3: report says...")
# for the full, quoted discrepancy.
#
# Single seed, single problem, small budget -- a SMOKE demonstration of the
# CEC 2022 solve()-integrated binding, not a claim about SHADE's quality or
# convergence rate. Comparing algorithms properly needs many seeds via
# sz_per_budget_packages/statistical tests (see README.md's "Experiments &
# Statistics" section) -- never a single-seed gap number.
#
# Cross-language number check: this script's printed best_f matches the
# Python script's best_f BIT-FOR-BIT (verified via the writeBin/struct.pack
# byte comparison in r-sezgi/tests/testthat/test-cec-tsp.R -- "R
# sz_solve_cec2022 (SHADE) is bit-identical to the Python/Rust golden") --
# same algorithm spec, same problem, same master_seed, same run_id, routed
# through the identical Rust core in both languages, so there is no reason
# for them to differ, and they don't.

library(sezgi)

fid <- 3
dim <- 10
pop_size <- 20
budget <- 5000
seed <- 20260830

f_star <- sz_cec2022_f_star(fid)
spec <- sz_preset_shade(pop_size, budget)

r <- sz_solve_cec2022(spec, fid = fid, dim = dim, master_seed = seed, run_id = 0)

cat(sprintf("CEC 2022 f%d (dim=%d), SHADE, pop_size=%d budget=%d seed=%d -- SMOKE DEMO, single seed\n",
            fid, dim, pop_size, budget, seed))
cat(sprintf("F* (report's pinned optimum): %s\n", f_star))
cat(sprintf("best_f: %s\n", r$best_f))
cat(sprintf("gap (best_f - F*): %s\n", r$best_f - f_star))
cat(sprintf("evals: %d\n", r$evals))
