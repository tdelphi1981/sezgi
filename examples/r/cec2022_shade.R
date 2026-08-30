# CEC 2022 (Kumar, Price, Mohamed, Hadi & Suganthan 2021) function f3,
# dim=10 -- a matched Python/R example pair for sezgi's M3-3 CEC 2022
# benchmark suite. See examples/python/cec2022_shade.py for the Python
# counterpart, which runs sezgi.presets.shade (Tanabe & Fukunaga 2013's
# SHADE) through sezgi.solve() against sezgi.problems.cec2022(fid, dim).
#
# IMPORTANT ASYMMETRY, disclosed rather than hidden: r-sezgi (T10) bound
# ONLY direct evaluation for CEC 2022 (sz_cec2022_evaluate/sz_cec2022_f_star)
# -- no sz_solve_cec2022 / no generic solve()-over-arbitrary-Problem
# equivalent to py-sezgi's Inner::Cec2022 + solve() path exists yet (see
# docs/DECISIONS.md's M3-3 record). So this script does NOT run the SHADE
# preset -- there is currently no R binding through which a built-in preset
# can be pointed at a CEC2022 problem. Instead it implements a small,
# classic DE/rand/1/bin loop (Storn & Price 1997) directly in base R,
# evaluating each trial via sz_cec2022_evaluate -- a plain reference
# algorithm, NOT sezgi's own pinned `gen/de-rand-1` component and NOT
# SHADE's success-history adaptation. This is a real, disclosed capability
# gap, not a stylistic choice -- it belongs on sezgi's v1.0 checklist.
#
# f3 is the CEC 2022 report's "Shifted and full Rotated Expanded Schaffer's
# f6 Function" BY NAME -- but the vendored official C reference actually
# dispatches problem 3 to plain Schaffer's F7 with no scale and (due to a
# verified buffer-reuse bug in the reference code) no effective rotation;
# sezgi follows the C code, which is what scored the competition, not the
# report's printed formula. See docs/DECISIONS.md's M3-3 record and
# crates/problems/src/cec2022/mod.rs's module doc ("F3: report says...")
# for the full, quoted discrepancy.
#
# Single seed, single problem, small budget -- a SMOKE demonstration of the
# CEC 2022 direct-evaluation binding, not a claim about DE/rand/1/bin's
# quality or convergence rate, and NOT a comparison against the Python
# script's SHADE run (different algorithms, different language RNG
# streams -- never compare algorithms on one seed). Comparing algorithms
# properly needs many seeds via sz_per_budget_packages/statistical tests
# (see README.md's "Experiments & Statistics" section).

library(sezgi)

fid <- 3
dim <- 10
pop_size <- 20
budget <- 5000
seed <- 20260830
lo <- -100.0
hi <- 100.0
de_f <- 0.5
de_cr <- 0.9

f_star <- sz_cec2022_f_star(fid)

set.seed(seed)
pop <- matrix(runif(pop_size * dim, lo, hi), nrow = pop_size)
fitness <- apply(pop, 1, function(x) sz_cec2022_evaluate(fid = fid, dim = dim, x = x))
used <- pop_size

while (used + pop_size <= budget) {
  for (i in 1:pop_size) {
    idxs <- sample(setdiff(1:pop_size, i), 3)
    donor <- pop[idxs[1], ] + de_f * (pop[idxs[2], ] - pop[idxs[3], ])
    donor <- pmin(pmax(donor, lo), hi)

    trial <- pop[i, ]
    j_rand <- sample(1:dim, 1)
    cross <- runif(dim) < de_cr
    cross[j_rand] <- TRUE
    trial[cross] <- donor[cross]

    f_trial <- sz_cec2022_evaluate(fid = fid, dim = dim, x = trial)
    used <- used + 1
    if (f_trial <= fitness[i]) {
      pop[i, ] <- trial
      fitness[i] <- f_trial
    }
  }
}

best_idx <- which.min(fitness)
best_f <- fitness[best_idx]

cat(sprintf("CEC 2022 f%d (dim=%d), pure-R DE/rand/1/bin (NOT the SHADE preset -- see this file's header), pop_size=%d budget=%d seed=%d -- SMOKE DEMO, single seed\n",
            fid, dim, pop_size, budget, seed))
cat(sprintf("F* (report's pinned optimum): %s\n", f_star))
cat(sprintf("best_f: %s\n", best_f))
cat(sprintf("gap (best_f - F*): %s\n", best_f - f_star))
cat(sprintf("evals_used: %d\n", used))
