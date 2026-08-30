# Pure-R Salp Swarm Algorithm (Mirjalili, S., Gandomi, A.H., Mirjalili,
# S.Z., Saremi, S., Faris, H. & Mirjalili, S.M. 2017, "Salp Swarm
# Algorithm: A bio-inspired optimizer for engineering design problems",
# Advances in Engineering Software 114, 163-191).
#
# Teaches the SAME pinned update equations as sezgi's `gen/ssa` Rust
# component (see crates/components/src/ssa.rs's module doc) with base R
# only, driven through `sz_eval_session`. Does NOT reproduce the Rust
# component's RNG-stream (draw order) contract, only the same update rule;
# see examples/specs/ssa.toml for the pinned-RNG preset.
#
# Tier note: a labeled metaphor preset -- faithful to SSA.m's own
# reference implementation, with a pinned deterministic draw order, but
# NOT validated against the paper's reported benchmark numbers. No
# established equivalence-critique reference covers SSA -- cited here as
# primary-source only.
#
# sezgi simplification (shared with the Rust preset): SSA.m's
# FoodPosition is a persisted best-ever; this uses the current
# population's fitness argmin instead (this wave's established
# current-generation-best convention for single-scalar attractors).
#
# No cross-algorithm quality claims are made or implied here -- single
# seed, single problem, reported as a gap.

library(sezgi)

dim <- 5
budget <- 2000
pop_size <- 30  # number of salps
seed <- 42
lo <- -5.0
hi <- 5.0

ssa_c1 <- function(progress) 2.0 * exp(-((4.0 * progress)^2))

ssa_leader_dim_step <- function(food_d, c1, c2, c3, lo, hi) {
  term <- c1 * ((hi - lo) * c2 + lo)
  if (c3 < 0.5) food_d + term else food_d - term
}

ssa_follower_dim_step <- function(x_old_d, prev_new_d) (x_old_d + prev_new_d) / 2.0

clamp <- function(x, lo, hi) min(max(x, lo), hi)

set.seed(seed)
# s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget,
#                       log_dir = "logs/ssa-r", algo_name = "ssa-r", seed = seed)
s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget)

pop <- lapply(1:pop_size, function(i) runif(dim, lo, hi))
fitness <- s$evaluate(pop)
used <- pop_size
leader_count <- pop_size %/% 2  # fixed positional split, not fitness-based

while (used + pop_size <= budget) {
  progress <- min(max(used / budget, 0.0), 1.0)
  c1 <- ssa_c1(progress)

  # Food: current-population fitness argmin, ties -> lower index, before
  # any draws.
  food_idx <- order(fitness)[1]
  food <- pop[[food_idx]]

  # Built SEQUENTIALLY in ascending index order: followers chain off the
  # already-computed offspring entries from earlier this same sweep.
  offspring <- vector("list", pop_size)
  for (i in 1:pop_size) {
    x_old <- pop[[i]]
    if (i <= leader_count) {
      new_x <- numeric(dim)
      for (d in 1:dim) {
        # Pinned draw order: c2, then c3, per (leader, d).
        c2 <- runif(1)
        c3 <- runif(1)
        new_x[d] <- clamp(ssa_leader_dim_step(food[d], c1, c2, c3, lo, hi), lo, hi)
      }
    } else {
      # Follower: zero draws; chains off offspring[[i-1]], ALREADY
      # computed this same sweep.
      prev <- offspring[[i - 1]]
      new_x <- vapply(1:dim, function(d) clamp(ssa_follower_dim_step(x_old[d], prev[d]), lo, hi), numeric(1))
    }
    offspring[[i]] <- new_x
  }

  # replace/generational: SSA.m overwrites every salp unconditionally.
  fitness <- s$evaluate(offspring)
  pop <- offspring
  used <- used + pop_size
}

best <- s$best()
gap <- best$f - s$f_opt()
cat(sprintf("ssa: evals_used=%d best_f=%.6g gap=%.6g\n", s$evals_used(), best$f, gap))
s$finish()
