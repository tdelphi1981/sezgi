# Pure-R Whale Optimization Algorithm (Mirjalili & Lewis 2016, "The Whale
# Optimization Algorithm", Advances in Engineering Software).
#
# Teaches the SAME pinned update equations as sezgi's `gen/woa` Rust
# component (see crates/components/src/woa.rs's module doc) with base R
# only, driven through `sz_eval_session`. Does NOT reproduce the Rust
# component's RNG-stream (draw order) contract, only the same update rule;
# see examples/specs/woa.toml for the pinned-RNG preset.
#
# Equivalence critique: Camacho-Villalon, Dorigo & Stutzle (International
# Transactions in Operational Research), which names WOA ("whale")
# explicitly among six metaphor-based algorithms shown to be, component for
# component, relabeled special cases of older operators.
#
# No cross-algorithm quality claims are made or implied here -- single
# seed, single problem, reported as a gap.

library(sezgi)

dim <- 5
budget <- 2000
pop_size <- 30
seed <- 42
lo <- -5.0
hi <- 5.0

woa_encircle_step <- function(target_d, x_d, big_a, big_c) {
  target_d - big_a * abs(big_c * target_d - x_d)
}

woa_spiral_step <- function(x_best_d, x_d, l, b = 1.0) {
  d <- abs(x_best_d - x_d)
  d * exp(b * l) * cos(2.0 * pi * l) + x_best_d
}

clamp <- function(x, lo, hi) min(max(x, lo), hi)

set.seed(seed)
# s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget,
#                       log_dir = "logs/woa-r", algo_name = "woa-r", seed = seed)
s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget)

pop <- lapply(1:pop_size, function(i) runif(dim, lo, hi))
fitness <- s$evaluate(pop)
used <- pop_size

while (used + pop_size <= budget) {
  progress <- min(max(used / budget, 0.0), 1.0)
  a <- 2.0 - 2.0 * progress
  a2 <- -1.0 - progress

  best <- order(fitness)[1]
  x_best <- pop[[best]]

  offspring <- vector("list", pop_size)
  for (i in 1:pop_size) {
    x <- pop[[i]]
    # p, and then r1/r2 (search branch) or l_raw (spiral branch), are all
    # drawn ONCE per whale, before the dimension loop -- matching the
    # reference MATLAB (WOA.m) exactly. Only the random-leader index is
    # drawn per dimension.
    p <- runif(1)
    new_x <- numeric(dim)
    if (p < 0.5) {
      r1 <- runif(1)
      r2 <- runif(1)
      big_a <- 2.0 * a * r1 - a
      big_c <- 2.0 * r2
      for (d in 1:dim) {
        if (abs(big_a) < 1.0) {
          target_d <- x_best[d]
        } else {
          j <- sample.int(pop_size, 1)  # may equal i, matches reference MATLAB
          target_d <- pop[[j]][d]
        }
        val <- woa_encircle_step(target_d, x[d], big_a, big_c)
        new_x[d] <- clamp(val, lo, hi)
      }
    } else {
      l_raw <- runif(1)
      l <- (a2 - 1.0) * l_raw + 1.0
      for (d in 1:dim) {
        val <- woa_spiral_step(x_best[d], x[d], l)
        new_x[d] <- clamp(val, lo, hi)
      }
    }
    offspring[[i]] <- new_x
  }

  # replace/generational, whole-batch-or-nothing per iteration.
  fitness <- s$evaluate(offspring)
  pop <- offspring
  used <- used + pop_size
}

best_pt <- s$best()
gap <- best_pt$f - s$f_opt()
cat(sprintf("woa: evals_used=%d best_f=%.6g gap=%.6g\n", s$evals_used(), best_pt$f, gap))
s$finish()
