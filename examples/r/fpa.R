# Pure-R Flower Pollination Algorithm (Yang, X.-S. 2012, "Flower
# Pollination Algorithm for Global Optimization", in: Unconventional
# Computation and Natural Computation (UCNC 2012), Lecture Notes in
# Computer Science vol. 7445, Springer, pp. 240-249).
#
# Teaches the SAME pinned update equations as sezgi's `gen/fpa` +
# `replace/one-to-one-greedy` Rust component (see
# crates/components/src/fpa.rs's module doc) with base R only, driven
# through `sz_eval_session`. Does NOT reproduce the Rust component's
# RNG-stream (draw order) contract, only the same update rule; see
# examples/specs/fpa.toml for the pinned-RNG preset.
#
# Tier note: a labeled metaphor preset -- faithful to fpa_demo.m's own
# reference implementation, with a pinned deterministic draw order, but
# NOT validated against the paper's reported benchmark numbers. No
# established equivalence critique covers FPA -- primary-source only.
#
# Reproduces two VERIFIED source deltas verbatim: the branch switch is
# u > p selecting the GLOBAL (Levy) branch (NOT u < p -- with p=0.8 global
# fires only ~20% of the time), and the local branch's j/k indices are
# distinct FROM EACH OTHER ONLY, never excluded from equaling i. The global
# step reuses the SAME Mantegna Levy step as cs.R (cs_dim_step, 0.01 scale).
#
# No cross-algorithm quality claims are made or implied here -- single
# seed, single problem, reported as a gap.

library(sezgi)

dim <- 5
budget <- 2000
pop_size <- 25  # flower count
seed <- 42
lo <- -5.0
hi <- 5.0
p_switch <- 0.8
alpha_step <- 0.01
levy_alpha <- 1.5

gauss_polar <- function() {
  repeat {
    u <- 2.0 * runif(1) - 1.0
    v <- 2.0 * runif(1) - 1.0
    s <- u * u + v * v
    if (s > 0.0 && s < 1.0) {
      return(u * sqrt(-2.0 * log(s) / s))
    }
  }
}

levy_mantegna <- function(alpha = levy_alpha) {
  num <- gamma(1.0 + alpha) * sin(pi * alpha / 2.0)
  den <- gamma((1.0 + alpha) / 2.0) * alpha * 2.0 ^ ((alpha - 1.0) / 2.0)
  sigma_u <- (num / den) ^ (1.0 / alpha)
  u <- sigma_u * gauss_polar()
  v <- abs(gauss_polar())
  u / (v ^ (1.0 / alpha))
}

# Reused verbatim from cs.R -- FPA's global-pollination step is
# algebraically identical to CS's Levy-toward-best step.
cs_dim_step <- function(x_i_d, x_best_d, levy, step = alpha_step) {
  x_i_d + step * levy * (x_i_d - x_best_d)
}

fpa_is_global_branch <- function(u, p) u > p

fpa_local_dim_step <- function(x_d, x_j_d, x_k_d, epsilon) x_d + epsilon * (x_j_d - x_k_d)

# j, k distinct FROM EACH OTHER only -- self (i) is never excluded.
pick_two_distinct <- function(n) {
  j <- sample.int(n, 1)
  repeat {
    k <- sample.int(n, 1)
    if (k != j) return(c(j, k))
  }
}

clamp <- function(x, lo, hi) min(max(x, lo), hi)

set.seed(seed)
# s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget,
#                       log_dir = "logs/fpa-r", algo_name = "fpa-r", seed = seed)
s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget)

pop <- lapply(1:pop_size, function(i) runif(dim, lo, hi))
fitness <- s$evaluate(pop)
used <- pop_size

while (used + pop_size <= budget) {
  # Attractor: current-population fitness argmin (ties -> lower index),
  # computed once per generation before any draws.
  best <- order(fitness)[1]
  x_best <- pop[[best]]

  offspring <- vector("list", pop_size)
  for (i in 1:pop_size) {
    x <- pop[[i]]
    u <- runif(1)
    if (fpa_is_global_branch(u, p_switch)) {
      new_x <- vapply(1:dim, function(d) clamp(cs_dim_step(x[d], x_best[d], levy_mantegna()), lo, hi), numeric(1))
    } else {
      epsilon <- runif(1)  # scalar, shared across every dimension
      jk <- pick_two_distinct(pop_size)
      x_j <- pop[[jk[1]]]; x_k <- pop[[jk[2]]]
      new_x <- vapply(1:dim, function(d) clamp(fpa_local_dim_step(x[d], x_j[d], x_k[d], epsilon), lo, hi), numeric(1))
    }
    offspring[[i]] <- new_x
  }

  new_fitness <- s$evaluate(offspring)
  used <- used + pop_size

  # replace/one-to-one-greedy
  for (i in 1:pop_size) {
    if (new_fitness[i] < fitness[i]) {
      pop[[i]] <- offspring[[i]]
      fitness[i] <- new_fitness[i]
    }
  }
}

best_pt <- s$best()
gap <- best_pt$f - s$f_opt()
cat(sprintf("fpa: evals_used=%d best_f=%.6g gap=%.6g\n", s$evals_used(), best_pt$f, gap))
s$finish()
