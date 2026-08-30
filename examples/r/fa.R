# Pure-R Firefly Algorithm (Yang, X.-S., Nature-Inspired Metaheuristic
# Algorithms, 2nd ed., Luniver Press, 2010).
#
# Teaches the SAME pinned update equations as sezgi's `gen/fa` Rust
# component (see crates/components/src/fa.rs's module doc) with base R
# only, driven through `sz_eval_session`. Does NOT reproduce the Rust
# component's RNG-stream (draw order) contract, only the same update rule;
# see examples/specs/fa.toml for the pinned-RNG preset.
#
# Tier note: a labeled metaphor preset -- faithful to fa_ndim.m/
# ffa_move.m's own reference implementation, with a pinned deterministic
# draw order, but NOT validated against the paper's reported benchmark
# numbers. Equivalence critique: Camacho-Villalon, Dorigo & Stutzle
# (International Transactions in Operational Research, six-algorithm
# critique: grey wolf, moth-flame, whale, firefly, bat, antlion) -- cited
# here conservatively, as background for why this is "labeled metaphor"
# rather than a mechanism sezgi treats as novel.
#
# Reproduces the source's hybrid live/frozen quirk VERBATIM: the pairwise
# distance and a firefly's own multiplicative self-term read the LIVE,
# in-place-mutating position array, while the additive move target reads
# a FROZEN, start-of-generation snapshot. The floored attractiveness
# formula (beta never decays below betamin) and the closed-form alpha
# decay use the same "progress substitutes a literal generation counter"
# idiom already used by sca.R/mfo.R/ssa.R.
#
# Cost note: O(pop_size^2 * dim) per generation.
#
# No cross-algorithm quality claims are made or implied here -- single
# seed, single problem, reported as a gap.

library(sezgi)

dim <- 5
budget <- 2000
pop_size <- 25  # number of fireflies
seed <- 42
lo <- -5.0
hi <- 5.0
alpha0 <- 0.5
beta0 <- 1.0
betamin <- 0.2
gamma <- 1.0

fa_beta <- function(r2) (beta0 - betamin) * exp(-gamma * r2) + betamin

fa_alpha <- function(progress) alpha0 * (1e-4 / 0.9)^progress

clamp <- function(x, lo, hi) min(max(x, lo), hi)

set.seed(seed)
# s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget,
#                       log_dir = "logs/fa-r", algo_name = "fa-r", seed = seed)
s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget)

pop <- lapply(1:pop_size, function(i) runif(dim, lo, hi))
fitness <- s$evaluate(pop)
used <- pop_size
scale <- rep(hi - lo, dim)

while (used + pop_size <= budget) {
  progress <- min(max(used / budget, 0.0), 1.0)
  alpha <- fa_alpha(progress)

  # Rank order: stable sort by fitness ascending, ties -> original index
  # -- REQUIRED for the verified in-place loop semantics.
  ord <- order(fitness)
  lighto <- fitness[ord]
  nso <- pop[ord]              # FROZEN rank-ordered snapshot
  ns <- lapply(nso, identity)  # LIVE working copy, mutated in place

  for (i in 1:pop_size) {
    for (j in 1:pop_size) {
      # r2 from the LIVE working copy on BOTH sides -- verified quirk.
      r2 <- sum((ns[[i]] - ns[[j]])^2)
      if (lighto[i] > lighto[j]) {
        beta <- fa_beta(r2)
        for (d in 1:dim) {
          u <- runif(1)
          step <- alpha * (u - 0.5) * scale[d]
          # self LIVE (accumulates across repeated j hits), target FROZEN.
          ns[[i]][d] <- ns[[i]][d] * (1.0 - beta) + nso[[j]][d] * beta + step
        }
      }
    }
  }

  # boundary/clamp applied AFTER the full double loop (same timing as the
  # preset's separate boundary stage, not mid-loop).
  offspring <- lapply(ns, function(row) vapply(row, function(v) clamp(v, lo, hi), numeric(1)))

  # replace/generational: fa_ndim.m overwrites the whole population
  # unconditionally every generation.
  fitness <- s$evaluate(offspring)
  pop <- offspring
  used <- used + pop_size
}

best <- s$best()
gap <- best$f - s$f_opt()
cat(sprintf("fa: evals_used=%d best_f=%.6g gap=%.6g\n", s$evals_used(), best$f, gap))
s$finish()
