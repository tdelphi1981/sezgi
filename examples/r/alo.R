# Pure-R Ant Lion Optimizer (Mirjalili, S. 2015, "The Ant Lion Optimizer",
# Advances in Engineering Software 83, 80-98).
#
# Teaches the SAME pinned update equations as sezgi's `gen/alo` +
# `replace/mu-plus-lambda` Rust component (see
# crates/components/src/alo.rs's module doc, verified against the
# author's own ALO.m/Random_walk_around_antlion.m/
# RouletteWheelSelection.m) with base R only, driven through
# `sz_eval_session`. Does NOT reproduce the Rust component's RNG-stream
# (draw order) contract, only the same update rule; see
# examples/specs/alo.toml for the pinned-RNG preset.
#
# Tier note: a labeled metaphor preset -- faithful to ALO.m's own
# equations and loop structure, pinned draw order, but NOT validated
# against the paper's reported benchmark numbers. Equivalence critique:
# Camacho-Villalon, Dorigo & Stutzle (International Transactions in
# Operational Research, six-algorithm critique) -- cited conservatively.
#
# Each ant's move is (RA + RE) / 2: a FULL-HORIZON random walk (cumsum of
# +-1 steps, t_max steps long, t_max = budget %/% pop_size) built fresh
# EVERY generation, min-max normalized into a SHRINKING [c, d] bound
# around the walk's target antlion, read at the current row -- once
# around a roulette-selected antlion (1/fitness weights, WITH a
# floor-shift when any fitness is <= 0), once around the elite
# (current-population fitness argmin). The antlion population itself
# carries elitism: pop and offspring are merged, sorted ascending by
# fitness, and truncated back to pop_size (replace/mu-plus-lambda).
#
# No cross-algorithm quality claims are made or implied here -- single
# seed, single problem, reported as a gap.

library(sezgi)

dim <- 5
budget <- 2000
pop_size <- 25  # ant/antlion count
seed <- 42
lo <- -5.0
hi <- 5.0

alo_i_ratio <- function(progress) {
  i <- 1.0
  if (progress > 0.1) i <- 1.0 + 100.0 * progress
  if (progress > 0.5) i <- 1.0 + 1000.0 * progress
  if (progress > 0.75) i <- 1.0 + 10000.0 * progress
  if (progress > 0.9) i <- 1.0 + 100000.0 * progress
  if (progress > 0.95) i <- 1.0 + 1000000.0 * progress
  i
}

alo_shift_bound <- function(scaled, antlion_d, use_plus) {
  if (use_plus) scaled + antlion_d else -scaled + antlion_d
}

alo_cumsum_walk <- function(steps) {
  x <- numeric(length(steps) + 1)
  x[1] <- 0.0
  acc <- 0.0
  for (k in seq_along(steps)) {
    acc <- acc + if (steps[k]) 1.0 else -1.0
    x[k + 1] <- acc
  }
  x
}

# row is 0-based (matching the Python/Rust convention); x is 1-indexed in R.
alo_normalize_walk <- function(x, row, c, d) {
  a <- min(x); b <- max(x)
  (x[row + 1] - a) * (d - c) / (b - a) + c
}

alo_roulette_weights <- function(fitness) {
  min_f <- min(fitness)
  if (min_f > 0.0) return(1.0 / fitness)
  shift <- -min_f + 1.0
  1.0 / (fitness + shift)
}

alo_roulette_select <- function(weights, u) {
  total <- sum(weights)
  target <- u * total
  acc <- 0.0
  for (i in seq_along(weights)) {
    acc <- acc + weights[i]
    if (acc > target) return(i)
  }
  1  # fallback (1-based) -- matches ALO.m's chosen_index==-1 -> index 1
}

alo_walk_around <- function(dim, t_max, row, lo, hi, i_ratio, antlion) {
  lb_scaled <- lo / i_ratio
  ub_scaled <- hi / i_ratio
  use_plus_lo <- runif(1) < 0.5
  use_plus_hi <- runif(1) >= 0.5
  out <- numeric(dim)
  for (d in 1:dim) {
    c <- alo_shift_bound(lb_scaled, antlion[d], use_plus_lo)
    dd <- alo_shift_bound(ub_scaled, antlion[d], use_plus_hi)
    steps <- runif(t_max) > 0.5
    x <- alo_cumsum_walk(steps)
    out[d] <- alo_normalize_walk(x, row, c, dd)
  }
  out
}

clamp <- function(x, lo, hi) min(max(x, lo), hi)

set.seed(seed)
# s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget,
#                       log_dir = "logs/alo-r", algo_name = "alo-r", seed = seed)
s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget)

pop <- lapply(1:pop_size, function(i) runif(dim, lo, hi))
fitness <- s$evaluate(pop)
used <- pop_size

# t_max = ALO.m's Max_iter, derived once from (budget, pop_size).
t_max <- max(1, budget %/% pop_size)
generation <- 0  # ALO.m's Current_iter = generation + 2 (1-based, iteration 1 = setup)

while (used + pop_size <= budget) {
  current_iter <- generation + 2
  progress <- current_iter / t_max
  i_ratio <- alo_i_ratio(progress)
  row <- min(generation + 1, t_max)

  elite_idx <- order(fitness)[1]
  elite <- pop[[elite_idx]]
  weights <- alo_roulette_weights(fitness)

  offspring <- vector("list", pop_size)
  for (k in 1:pop_size) {
    u_sel <- runif(1)
    sel_idx <- alo_roulette_select(weights, u_sel)
    selected <- pop[[sel_idx]]

    ra <- alo_walk_around(dim, t_max, row, lo, hi, i_ratio, selected)
    re <- alo_walk_around(dim, t_max, row, lo, hi, i_ratio, elite)
    xs <- vapply(1:dim, function(d) clamp((ra[d] + re[d]) / 2.0, lo, hi), numeric(1))
    offspring[[k]] <- xs
  }

  new_fitness <- s$evaluate(offspring)
  used <- used + pop_size

  # replace/mu-plus-lambda: pop first, offspring second, stable sort
  # ascending by fitness, truncate to pop_size.
  combined_fitness <- c(fitness, new_fitness)
  combined_positions <- c(pop, offspring)
  ord <- order(combined_fitness)
  keep <- ord[1:pop_size]
  fitness <- combined_fitness[keep]
  pop <- combined_positions[keep]

  generation <- generation + 1
}

best_pt <- s$best()
gap <- best_pt$f - s$f_opt()
cat(sprintf("alo: evals_used=%d best_f=%.6g gap=%.6g\n", s$evals_used(), best_pt$f, gap))
s$finish()
