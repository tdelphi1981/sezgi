# Pure-R Harris Hawks Optimization (Heidari, A.A., Mirjalili, S., Faris,
# H., Aljarah, I., Mafarja, M. & Chen, H. 2019, "Harris hawks
# optimization: Algorithm and applications", Future Generation Computer
# Systems 97, 849-872).
#
# Teaches the SAME pinned update equations as sezgi's `gen/hho` +
# `replace/generational` Rust component (see
# crates/components/src/hho.rs's module doc, verified against the paper
# AUTHOR's own HHO.m) with base R only, driven through `sz_eval_session`.
# Does NOT reproduce the Rust component's RNG-stream (draw order)
# contract, only the same update rule; see examples/specs/hho.toml for the
# pinned-RNG preset.
#
# Tier note: a labeled metaphor preset -- faithful to HHO.m's own
# equations and loop structure, pinned draw order, but NOT validated
# against the paper's reported benchmark numbers. No established
# equivalence critique covers HHO -- primary-source only.
#
# A per-hawk escape-energy branch tree (explore vs. exploit, further split
# on r) whose deepest branches (progressive rapid dives) EVALUATE mid-loop
# -- Y tried first (accepted immediately if it improves), else
# Z=Y+S*Levy (2nd eval); neither improving leaves the hawk UNCHANGED.
# Reproduces the CORRECTED soft-dive Y formula verbatim:
# Rabbit - E*|J*Rabbit - X_i| (a genuinely different equation from the
# no-dive soft besiege). mean(X) and the random-hawk lookup both read the
# LIVE, in-place-updated working array.
#
# sezgi simplification (script-specific): if the FINAL per-generation
# batch evaluation would overrun the budget (dive trials already spent
# it), this script stops at the generation boundary rather than
# reproducing the engine's own per-hawk graceful fallback.
#
# No cross-algorithm quality claims are made or implied here -- single
# seed, single problem, reported as a gap.

library(sezgi)

dim <- 5
budget <- 2000
pop_size <- 30  # number of hawks
seed <- 42
lo <- -5.0
hi <- 5.0
levy_alpha <- 1.5

gauss_polar <- function() {
  repeat {
    u <- 2.0 * runif(1) - 1.0
    v <- 2.0 * runif(1) - 1.0
    s <- u * u + v * v
    if (s > 0.0 && s < 1.0) return(u * sqrt(-2.0 * log(s) / s))
  }
}

# HHO.m's own Levy(d) has NO 0.01 scale factor -- used directly.
levy_mantegna <- function(alpha = levy_alpha) {
  num <- gamma(1.0 + alpha) * sin(pi * alpha / 2.0)
  den <- gamma((1.0 + alpha) / 2.0) * alpha * 2.0 ^ ((alpha - 1.0) / 2.0)
  sigma_u <- (num / den) ^ (1.0 / alpha)
  u <- sigma_u * gauss_polar()
  v <- abs(gauss_polar())
  u / (v ^ (1.0 / alpha))
}

hho_e1 <- function(progress) 2.0 * (1.0 - progress)

hho_explore_family_dim_step <- function(x_rand_d, r_a, r_b, x_i_d) x_rand_d - r_a * abs(x_rand_d - 2.0 * r_b * x_i_d)
hho_explore_tree_dim_step <- function(rabbit_d, mean_d, r_a, r_b, lo, hi) (rabbit_d - mean_d) - r_a * ((hi - lo) * r_b + lo)
hho_hard_besiege_dim_step <- function(rabbit_d, e, x_i_d) rabbit_d - e * abs(rabbit_d - x_i_d)
hho_soft_besiege_dim_step <- function(rabbit_d, e, jump, x_i_d) (rabbit_d - x_i_d) - e * abs(jump * rabbit_d - x_i_d)
# CORRECTED formula (HHO.m line 105's X1) -- no leading (rabbit-x_i) term.
hho_soft_dive_y_dim_step <- function(rabbit_d, e, jump, x_i_d) rabbit_d - e * abs(jump * rabbit_d - x_i_d)
hho_hard_dive_y_dim_step <- function(rabbit_d, e, jump, mean_d) rabbit_d - e * abs(jump * rabbit_d - mean_d)
hho_dive_z_dim_step <- function(y_d, s_d, levy_d) y_d + s_d * levy_d

hho_mean <- function(work) {
  d <- length(work[[1]])
  vapply(1:d, function(k) mean(vapply(work, function(row) row[k], numeric(1))), numeric(1))
}

clamp <- function(x, lo, hi) min(max(x, lo), hi)

# Evaluate Y; accept immediately if it improves. Else Z=Y+S*Levy
# (interleaved per-dim draws); accept if IT improves. Neither -> NULL
# (caller keeps the un-dived candidate). Budget exhaustion at any point
# also falls back to NULL, spending nothing further.
hho_run_dive <- function(s, state_env, y, dim, current_fitness) {
  fy <- tryCatch({
    f <- s$evaluate(list(y))[1]
    state_env$used <- state_env$used + 1
    f
  }, error = function(e) NULL)
  if (is.null(fy)) return(NULL)
  if (fy < current_fitness) return(y)

  z <- numeric(dim)
  for (d in 1:dim) {
    s_d <- runif(1)
    levy_d <- levy_mantegna()
    z[d] <- hho_dive_z_dim_step(y[d], s_d, levy_d)
  }
  fz <- tryCatch({
    f <- s$evaluate(list(z))[1]
    state_env$used <- state_env$used + 1
    f
  }, error = function(e) NULL)
  if (is.null(fz)) return(NULL)
  if (fz < current_fitness) z else NULL
}

set.seed(seed)
# s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget,
#                       log_dir = "logs/hho-r", algo_name = "hho-r", seed = seed)
s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget)

pop <- lapply(1:pop_size, function(i) runif(dim, lo, hi))
fitness <- s$evaluate(pop)
state_env <- new.env()
state_env$used <- pop_size

repeat {
  if (state_env$used + pop_size > budget) break
  progress <- min(max(state_env$used / budget, 0.0), 1.0)
  e1 <- hho_e1(progress)

  rabbit_idx <- order(fitness)[1]
  rabbit <- pop[[rabbit_idx]]

  work <- lapply(pop, function(x) x)  # live, in-place working copy

  for (i in 1:pop_size) {
    e0 <- 2.0 * runif(1) - 1.0
    e <- e1 * e0

    if (abs(e) >= 1.0) {
      # Exploration
      q <- runif(1)
      rand_idx <- sample.int(pop_size, 1)  # always drawn, even if unused
      if (q < 0.5) {
        x_rand <- work[[rand_idx]]
        r_a <- runif(1); r_b <- runif(1)
        new_x <- vapply(1:dim, function(d) hho_explore_family_dim_step(x_rand[d], r_a, r_b, work[[i]][d]), numeric(1))
      } else {
        mean_v <- hho_mean(work)
        r_a <- runif(1); r_b <- runif(1)
        new_x <- vapply(1:dim, function(d) hho_explore_tree_dim_step(rabbit[d], mean_v[d], r_a, r_b, lo, hi), numeric(1))
      }
    } else {
      # Exploitation
      r <- runif(1)
      if (r >= 0.5 && abs(e) < 0.5) {
        new_x <- vapply(1:dim, function(d) hho_hard_besiege_dim_step(rabbit[d], e, work[[i]][d]), numeric(1))
      } else if (r >= 0.5) {
        jump <- 2.0 * (1.0 - runif(1))
        new_x <- vapply(1:dim, function(d) hho_soft_besiege_dim_step(rabbit[d], e, jump, work[[i]][d]), numeric(1))
      } else if (abs(e) >= 0.5) {
        jump <- 2.0 * (1.0 - runif(1))
        y <- vapply(1:dim, function(d) hho_soft_dive_y_dim_step(rabbit[d], e, jump, work[[i]][d]), numeric(1))
        dived <- hho_run_dive(s, state_env, y, dim, fitness[i])
        new_x <- if (!is.null(dived)) dived else work[[i]]
      } else {
        jump <- 2.0 * (1.0 - runif(1))
        mean_v <- hho_mean(work)
        y <- vapply(1:dim, function(d) hho_hard_dive_y_dim_step(rabbit[d], e, jump, mean_v[d]), numeric(1))
        dived <- hho_run_dive(s, state_env, y, dim, fitness[i])
        new_x <- if (!is.null(dived)) dived else work[[i]]
      }
    }
    work[[i]] <- new_x
  }

  offspring <- lapply(work, function(row) vapply(row, function(v) clamp(v, lo, hi), numeric(1)))
  new_fitness <- tryCatch(s$evaluate(offspring), error = function(e) NULL)
  if (is.null(new_fitness)) break  # dive trials consumed the remaining budget; stop cleanly
  state_env$used <- state_env$used + pop_size
  pop <- offspring
  fitness <- new_fitness  # replace/generational: unconditional overwrite
}

best_pt <- s$best()
gap <- best_pt$f - s$f_opt()
cat(sprintf("hho: evals_used=%d best_f=%.6g gap=%.6g\n", s$evals_used(), best_pt$f, gap))
s$finish()
