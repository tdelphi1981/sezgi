# Pure-R JAYA (Rao, R.V. 2016, "Jaya: A Simple and New Optimization
# Algorithm for Solving Constrained and Unconstrained Optimization
# Problems", International Journal of Industrial Engineering Computations
# 7(1), 19-34).
#
# Teaches the SAME pinned update equations as sezgi's `gen/jaya` Rust
# component (see crates/components/src/jaya.rs's module doc) with base R
# only, driven through `sz_eval_session`. Does NOT reproduce the Rust
# component's RNG-stream (draw order) contract, only the same update rule;
# see examples/specs/jaya.toml for the pinned-RNG preset.
#
# Tier note: a labeled metaphor preset -- faithful to the primary paper's
# stated Eq. (1) AND its own worked numerical example, with a pinned
# deterministic draw order, but NOT validated against the paper's reported
# benchmark numbers. No established equivalence-critique reference covers
# JAYA -- cited here as primary-source only.
#
# No cross-algorithm quality claims are made or implied here -- single
# seed, single problem, reported as a gap.

library(sezgi)

dim <- 5
budget <- 2000
pop_size <- 30  # candidate count
seed <- 42
lo <- -5.0
hi <- 5.0

jaya_dim_step <- function(x_d, x_best_d, x_worst_d, r1, r2) {
  abs_x <- abs(x_d)
  x_d + r1 * (x_best_d - abs_x) - r2 * (x_worst_d - abs_x)
}

clamp <- function(x, lo, hi) min(max(x, lo), hi)

# Ties -> HIGHER index (order()'s stable sort keeps ties -> lower index
# even under decreasing=TRUE, so the worst pick needs an explicit tie
# break to match the pinned "opposite-extremum" convention).
worst_index <- function(fitness) {
  idx <- which(fitness == max(fitness))
  idx[length(idx)]
}

set.seed(seed)
# s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget,
#                       log_dir = "logs/jaya-r", algo_name = "jaya-r", seed = seed)
s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget)

pop <- lapply(1:pop_size, function(i) runif(dim, lo, hi))
fitness <- s$evaluate(pop)
used <- pop_size

while (used + pop_size <= budget) {
  # Current-population argmin/argmax, both before any draws.
  best <- order(fitness)[1]
  worst <- worst_index(fitness)
  x_best <- pop[[best]]
  x_worst <- pop[[worst]]

  # Pinned: r1[d], r2[d] drawn ONCE PER DIMENSION PER GENERATION, shared
  # across every agent.
  r1 <- runif(dim)
  r2 <- runif(dim)

  offspring <- vector("list", pop_size)
  for (i in 1:pop_size) {
    x <- pop[[i]]
    new_x <- numeric(dim)
    for (d in 1:dim) {
      new_x[d] <- clamp(jaya_dim_step(x[d], x_best[d], x_worst[d], r1[d], r2[d]), lo, hi)
    }
    offspring[[i]] <- new_x
  }
  new_fitness <- s$evaluate(offspring)
  used <- used + pop_size

  # replace/one-to-one-greedy: accept only if strictly better.
  for (i in 1:pop_size) {
    if (new_fitness[i] < fitness[i]) {
      pop[[i]] <- offspring[[i]]
      fitness[i] <- new_fitness[i]
    }
  }
}

best <- s$best()
gap <- best$f - s$f_opt()
cat(sprintf("jaya: evals_used=%d best_f=%.6g gap=%.6g\n", s$evals_used(), best$f, gap))
s$finish()
