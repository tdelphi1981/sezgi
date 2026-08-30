# Pure-R Bat Algorithm (Yang, X.-S. 2010, "A new metaheuristic
# bat-inspired algorithm", in: Nature Inspired Cooperative Strategies for
# Optimization (NICSO 2010), Studies in Computational Intelligence vol.
# 284, Springer, pp. 65-74).
#
# Teaches the SAME pinned update equations as sezgi's `gen/ba` +
# `replace/bat-loudness-greedy` Rust components (see
# crates/components/src/ba.rs's module doc) with base R only, driven
# through `sz_eval_session`. Does NOT reproduce the Rust component's
# RNG-stream (draw order) contract, only the same update rule; see
# examples/specs/ba.toml for the pinned-RNG preset.
#
# Tier note: a labeled metaphor preset -- faithful to bat_algorithm.m's
# own reference implementation, with a pinned deterministic draw order,
# but NOT validated against the paper's reported benchmark numbers.
# Equivalence critique: Camacho-Villalon, Dorigo & Stutzle (International
# Transactions in Operational Research, six-algorithm critique: grey
# wolf, moth-flame, whale, firefly, bat, antlion) -- cited here
# conservatively, as background for why this is "labeled metaphor" rather
# than a mechanism sezgi treats as novel.
#
# Reproduces two VERIFIED source quirks verbatim (not "fixed"): the
# frequency draw is Q = Qmin + (Qmin-Qmax)*u, an inverted [-2,0] range
# (not the paper-prose-intuitive [0,2]), and the velocity term uses
# (X - X_best), not (X_best - X) -- the two sign inversions compose into
# a genuine pull TOWARD best, not away from it. A=0.5/r=0.5 are FIXED
# constants in the verified source (the demo explicitly does not
# implement loudness/pulse-rate dynamics), so `velocity` is the only
# persisted state here, carried in a plain R list across generations.
#
# No cross-algorithm quality claims are made or implied here -- single
# seed, single problem, reported as a gap.

library(sezgi)

dim <- 5
budget <- 2000
pop_size <- 30  # number of bats
seed <- 42
lo <- -5.0
hi <- 5.0
qmin <- 0.0
qmax <- 2.0
r0 <- 0.5   # fixed pulse rate
a0 <- 0.5   # fixed loudness
local_walk_scale <- 0.001

# VERIFIED VERBATIM against bat_algorithm.m: Qmin + (Qmin-Qmax)*u -- note
# the sign, giving [-2, 0] for the pinned defaults, not [0, 2].
ba_frequency <- function(u) qmin + (qmin - qmax) * u

ba_velocity_step <- function(v_d, x_d, best_d, freq) v_d + (x_d - best_d) * freq

clamp <- function(x, lo, hi) min(max(x, lo), hi)

set.seed(seed)
# s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget,
#                       log_dir = "logs/ba-r", algo_name = "ba-r", seed = seed)
s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget)

pop <- lapply(1:pop_size, function(i) runif(dim, lo, hi))
fitness <- s$evaluate(pop)
used <- pop_size
velocity <- lapply(1:pop_size, function(i) numeric(dim))

while (used + pop_size <= budget) {
  # best: current-population fitness argmin, ties -> lower index, before
  # any draws (sezgi simplification, shared with the Rust preset:
  # bat_algorithm.m persists a best-ever).
  best_idx <- order(fitness)[1]
  best <- pop[[best_idx]]

  offspring <- vector("list", pop_size)
  for (i in 1:pop_size) {
    x <- pop[[i]]
    # ONE frequency draw per bat (not per dimension).
    freq <- ba_frequency(runif(1))
    v_i <- vapply(1:dim, function(d) ba_velocity_step(velocity[[i]][d], x[d], best[d], freq), numeric(1))
    new_x <- x + v_i

    # Local-walk trigger: ONE draw per bat.
    trigger <- runif(1)
    if (trigger > r0) {
      # COMPLETE OVERWRITE of the candidate; the velocity move above is
      # discarded for the candidate but v_i still commits below.
      new_x <- best + local_walk_scale * rnorm(dim)
    }

    velocity[[i]] <- v_i
    offspring[[i]] <- new_x
  }

  # boundary/clamp applied AFTER generate returns, same timing as the
  # Rust preset's separate boundary stage.
  offspring <- lapply(offspring, function(row) vapply(row, function(v) clamp(v, lo, hi), numeric(1)))

  new_fitness <- s$evaluate(offspring)
  used <- used + pop_size

  # replace/bat-loudness-greedy: accept iff off_fit[i] <= own fit[i] AND a
  # FRESH, UNCONDITIONAL loudness draw < A0 -- the draw always fires
  # (MATLAB's non-short-circuiting `&`), whether or not the fitness test
  # holds.
  for (i in 1:pop_size) {
    loud_draw <- runif(1)
    if (new_fitness[i] <= fitness[i] && loud_draw < a0) {
      pop[[i]] <- offspring[[i]]
      fitness[i] <- new_fitness[i]
    }
  }
}

best <- s$best()
gap <- best$f - s$f_opt()
cat(sprintf("ba: evals_used=%d best_f=%.6g gap=%.6g\n", s$evals_used(), best$f, gap))
s$finish()
