# Pure-R Grey Wolf Optimizer (Mirjalili, Mirjalili & Lewis 2014, "Grey Wolf
# Optimizer", Advances in Engineering Software).
#
# Teaches the SAME pinned update equations as sezgi's `gen/gwo` Rust
# component (see crates/components/src/gwo.rs's module doc) with base R
# only, driven through `sz_eval_session`. Does NOT reproduce the Rust
# component's RNG-stream (draw order) contract, only the same update rule;
# see examples/specs/gwo.toml for the pinned-RNG preset.
#
# Equivalence critique: Camacho-Villalon, Dorigo & Stutzle (ANTS 2020,
# three-algorithm study); Camacho-Villalon, Dorigo & Stutzle (International
# Transactions in Operational Research, six-algorithm journal extension).
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

gwo_dim_step <- function(a, leaders_d, x_d, draws) {
  total <- 0.0
  for (k in 0:2) {
    r1 <- draws[2 * k + 1]
    r2 <- draws[2 * k + 2]
    big_a <- 2.0 * a * r1 - a
    c <- 2.0 * r2
    total <- total + (leaders_d[k + 1] - big_a * abs(c * leaders_d[k + 1] - x_d))
  }
  total / 3.0
}

clamp <- function(x, lo, hi) min(max(x, lo), hi)

set.seed(seed)
# s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget,
#                       log_dir = "logs/gwo-r", algo_name = "gwo-r", seed = seed)
s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget)

pop <- lapply(1:pop_size, function(i) runif(dim, lo, hi))
fitness <- s$evaluate(pop)
used <- pop_size

while (used + pop_size <= budget) {
  progress <- min(max(used / budget, 0.0), 1.0)
  a <- 2.0 - 2.0 * progress

  # order() is a stable sort: ties keep original (lower-index) order first.
  leaders <- order(fitness)[1:3]
  leader_x <- pop[leaders]

  offspring <- vector("list", pop_size)
  for (i in 1:pop_size) {
    x <- pop[[i]]
    new_x <- numeric(dim)
    for (d in 1:dim) {
      leaders_d <- vapply(leader_x, function(lx) lx[d], numeric(1))
      draws <- runif(6)
      new_x[d] <- clamp(gwo_dim_step(a, leaders_d, x[d], draws), lo, hi)
    }
    offspring[[i]] <- new_x
  }

  # replace/generational, whole-batch-or-nothing per iteration.
  fitness <- s$evaluate(offspring)
  pop <- offspring
  used <- used + pop_size
}

best <- s$best()
gap <- best$f - s$f_opt()
cat(sprintf("gwo: evals_used=%d best_f=%.6g gap=%.6g\n", s$evals_used(), best$f, gap))
s$finish()
