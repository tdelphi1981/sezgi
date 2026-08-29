# Pure-R Cuckoo Search (Yang, X.-S. & Deb, S. 2009, "Cuckoo Search via Levy
# Flights", 2009 World Congress on Nature & Biologically Inspired Computing
# (NaBIC), pp. 210-214, IEEE).
#
# Teaches the SAME pinned update equations as sezgi's `gen/cuckoo_levy` +
# `adapter/abandon-worst-fraction` (see crates/components/src/cs.rs's
# module doc), base R only (`gamma()` is a base-R builtin, replacing the
# Rust component's own Lanczos gamma -- same closed-form Mantegna (1994)
# algorithm), via `sz_eval_session`. Not RNG-stream-identical to the Rust
# preset -- same equations only; see examples/specs/cs.toml for the
# pinned-RNG preset. No equivalence-critique citation -- not mandated for
# CS by the task brief.
#
# sezgi simplification (shared with the Rust preset): the paper's
# Algorithm 1 compares a new egg against a RANDOM nest; this uses greedy
# same-index replacement instead, matching `presets::cuckoo_search`'s
# divergence.
#
# No cross-algorithm quality claims are made or implied here -- single
# seed, single problem, reported as a gap.

library(sezgi)

dim <- 5
budget <- 2000
pop_size <- 25  # nest count
seed <- 42
lo <- -5.0
hi <- 5.0
alpha_step <- 0.01
levy_alpha <- 1.5
pa <- 0.25  # abandoned-fraction

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

cs_dim_step <- function(x_i_d, x_best_d, levy, step = alpha_step) {
  x_i_d + step * levy * (x_i_d - x_best_d)
}

clamp <- function(x, lo, hi) min(max(x, lo), hi)

# Worst-first indices of the worst floor(pa*n) nests; ties -> higher index
# abandoned first.
abandon_order <- function(fitness, pa) {
  n <- length(fitness)
  k <- as.integer(pa * n)
  if (k == 0) return(integer(0))
  ord <- order(-fitness, -seq_along(fitness))
  ord[1:k]
}

set.seed(seed)
# s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget,
#                       log_dir = "logs/cs-r", algo_name = "cs-r", seed = seed)
s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget)

pop <- lapply(1:pop_size, function(i) runif(dim, lo, hi))
fitness <- s$evaluate(pop)
used <- pop_size

while (used + pop_size <= budget) {
  best <- order(fitness)[1]
  x_best <- pop[[best]]

  offspring <- lapply(1:pop_size, function(i) {
    vapply(1:dim, function(d) {
      clamp(cs_dim_step(pop[[i]][d], x_best[d], levy_mantegna()), lo, hi)
    }, numeric(1))
  })
  new_fitness <- s$evaluate(offspring)
  used <- used + pop_size

  # replace/one-to-one-greedy
  for (i in 1:pop_size) {
    if (new_fitness[i] < fitness[i]) {
      pop[[i]] <- offspring[[i]]
      fitness[i] <- new_fitness[i]
    }
  }

  # adapter/abandon-worst-fraction: only if the WHOLE abandon batch fits
  # (mirrors the Rust adapter silently skipping on budget exhaustion).
  ord <- abandon_order(fitness, pa)
  if (length(ord) > 0 && used + length(ord) <= budget) {
    new_nests <- lapply(ord, function(i) runif(dim, lo, hi))
    new_fit2 <- s$evaluate(new_nests)
    used <- used + length(ord)
    for (k in seq_along(ord)) {
      pop[[ord[k]]] <- new_nests[[k]]
      fitness[ord[k]] <- new_fit2[k]
    }
  }
}

best_pt <- s$best()
gap <- best_pt$f - s$f_opt()
cat(sprintf("cs: evals_used=%d best_f=%.6g gap=%.6g\n", s$evals_used(), best_pt$f, gap))
s$finish()
