# Pure-R Sine Cosine Algorithm (Mirjalili, S. 2016, "SCA: A Sine Cosine
# Algorithm for Solving Optimization Problems", Knowledge-Based Systems
# 96, 120-133).
#
# Teaches the SAME pinned update equations as sezgi's `gen/sca` Rust
# component (see crates/components/src/sca.rs's module doc) with base R
# only, driven through `sz_eval_session`. Does NOT reproduce the Rust
# component's RNG-stream (draw order) contract, only the same update rule;
# see examples/specs/sca.toml for the pinned-RNG preset.
#
# Tier note: a labeled metaphor preset -- faithful to SCA.m's own
# reference implementation, with a pinned deterministic draw order, but
# NOT validated against the paper's reported benchmark numbers. No
# established equivalence-critique reference covers SCA -- cited here as
# primary-source only.
#
# No cross-algorithm quality claims are made or implied here -- single
# seed, single problem, reported as a gap.

library(sezgi)

dim <- 5
budget <- 2000
pop_size <- 30  # number of search agents
seed <- 42
lo <- -5.0
hi <- 5.0

sca_dim_step <- function(x_d, x_best_d, r1, r2, r3, r4) {
  delta <- abs(r3 * x_best_d - x_d)
  if (r4 < 0.5) x_d + r1 * sin(r2) * delta else x_d + r1 * cos(r2) * delta
}

clamp <- function(x, lo, hi) min(max(x, lo), hi)

set.seed(seed)
# s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget,
#                       log_dir = "logs/sca-r", algo_name = "sca-r", seed = seed)
s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget)

pop <- lapply(1:pop_size, function(i) runif(dim, lo, hi))
fitness <- s$evaluate(pop)
used <- pop_size

while (used + pop_size <= budget) {
  progress <- min(max(used / budget, 0.0), 1.0)
  a <- 2.0
  r1 <- a * (1.0 - progress)

  # Destination_position: current population's fitness argmin.
  best <- order(fitness)[1]
  x_best <- pop[[best]]

  offspring <- vector("list", pop_size)
  for (i in 1:pop_size) {
    x <- pop[[i]]
    new_x <- numeric(dim)
    for (d in 1:dim) {
      # Pinned draw order: r2, then r3, then r4, fresh for every (i, d).
      r2 <- (2.0 * pi) * runif(1)
      r3 <- 2.0 * runif(1)
      r4 <- runif(1)
      new_x[d] <- clamp(sca_dim_step(x[d], x_best[d], r1, r2, r3, r4), lo, hi)
    }
    offspring[[i]] <- new_x
  }

  # replace/generational: SCA.m overwrites every agent unconditionally.
  fitness <- s$evaluate(offspring)
  pop <- offspring
  used <- used + pop_size
}

best <- s$best()
gap <- best$f - s$f_opt()
cat(sprintf("sca: evals_used=%d best_f=%.6g gap=%.6g\n", s$evals_used(), best$f, gap))
s$finish()
