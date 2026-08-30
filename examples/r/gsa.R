# Pure-R Gravitational Search Algorithm (Rashedi, E., Nezamabadi-pour, H.
# & Saryazdi, S. 2009, "GSA: A Gravitational Search Algorithm",
# Information Sciences 179(13), 2232-2248).
#
# Teaches the SAME pinned update equations as sezgi's `gen/gsa` +
# `replace/generational` Rust component (see
# crates/components/src/gsa.rs's module doc, verified against Esmat
# Rashedi's own GSA.m/Gconstant.m/massCalculation.m/Gfield.m/move.m) with
# base R only, driven through `sz_eval_session`. Does NOT reproduce the
# Rust component's RNG-stream (draw order) contract, only the same update
# rule; see examples/specs/gsa.toml for the pinned-RNG preset.
#
# Tier note: a labeled metaphor preset -- faithful to GSA.m's own
# equations and loop structure, pinned draw order, but NOT validated
# against the paper's reported benchmark numbers. No established
# equivalence critique covers GSA -- primary-source only.
#
# The wave's LAST stateful algorithm: the per-agent velocity is a genuine
# persisted memory, carried in a plain R list across generations. Per
# generation: masses are normalized ((fit-worst)/(best-worst), EXCEPT a
# degenerate all-equal-fitness guard giving uniform 1/N masses), G decays
# exponentially with progress, the Kbest elite-set SHRINKS with progress
# (2 + (1-progress)*98 percent, NO floor), and for every target agent,
# every OTHER agent in the (mass-sorted) Kbest set contributes a force
# with its OWN independent draw per dimension (rand*M_j*(x_j-x_i)/(R+eps)
# -- notably NO M_i term, since Gfield.m's own comment states
# Mp(i)/Mi(i)=1). Velocity updates as v' = rand*v + accel (the random
# factor MULTIPLIES the OLD velocity) and position as x' = x + v'.
#
# No cross-algorithm quality claims are made or implied here -- single
# seed, single problem, reported as a gap.

library(sezgi)

dim <- 5
budget <- 2000
pop_size <- 30  # number of agents
seed <- 42
lo <- -5.0
hi <- 5.0
g0 <- 100.0
alpha <- 20.0
final_percent <- 2.0
eps_val <- 2.220446049250313e-16  # f64::EPSILON / numpy.finfo(float).eps

gsa_mass <- function(fit) {
  fmax <- max(fit); fmin <- min(fit)
  if (fmax == fmin) {
    m <- rep(1.0, length(fit))
  } else {
    best <- fmin; worst <- fmax
    m <- (fit - worst) / (best - worst)
  }
  m / sum(m)
}

gsa_g <- function(progress) g0 * exp(-alpha * progress)

gsa_kbest_count <- function(n, progress) {
  percent <- final_percent + (1.0 - progress) * (100.0 - final_percent)
  round(n * percent / 100.0)  # NOT clamped to a minimum of 1 -- can reach 0
}

# Indices (1-based) sorted by mass DESCENDING -- stable, ties keep the
# original agent order.
gsa_kbest_order <- function(mass) order(-mass)

# NO M_i factor anywhere -- Gfield.m's own comment: Mp(i)/Mi(i)=1.
gsa_force_term <- function(rnd, mass_j, xj_d, xi_d, r_dist) rnd * mass_j * (xj_d - xi_d) / (r_dist + eps_val)

euclidean <- function(a, b) sqrt(sum((a - b) ^ 2))

# The random factor MULTIPLIES the OLD velocity, unlike ba.R's
# purely-additive term.
gsa_velocity_step <- function(v_d, accel_d, r1) r1 * v_d + accel_d

clamp <- function(x, lo, hi) min(max(x, lo), hi)

set.seed(seed)
# s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget,
#                       log_dir = "logs/gsa-r", algo_name = "gsa-r", seed = seed)
s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget)

pop <- lapply(1:pop_size, function(i) runif(dim, lo, hi))
fitness <- s$evaluate(pop)
used <- pop_size
velocity <- lapply(1:pop_size, function(i) numeric(dim))

while (used + pop_size <= budget) {
  progress <- min(max(used / budget, 0.0), 1.0)
  mass <- gsa_mass(fitness)
  g_const <- gsa_g(progress)
  kbest <- gsa_kbest_count(pop_size, progress)
  ord <- gsa_kbest_order(mass)

  # Gfield.m: ALL per-(i,j,d) force draws happen first, completing the
  # WHOLE acceleration array, BEFORE move.m's per-(i,d) draws start.
  accel <- lapply(1:pop_size, function(i) numeric(dim))
  for (i in 1:pop_size) {
    force_i <- numeric(dim)
    if (kbest >= 1) {
      for (ii in 1:kbest) {
        j <- ord[ii]
        if (j == i) next
        r_dist <- euclidean(pop[[i]], pop[[j]])
        for (d in 1:dim) {
          rnd <- runif(1)
          force_i[d] <- force_i[d] + gsa_force_term(rnd, mass[j], pop[[j]][d], pop[[i]][d], r_dist)
        }
      }
    }
    accel[[i]] <- g_const * force_i
  }

  # move.m: velocity + position update, a SEPARATE loop after Gfield.
  offspring <- vector("list", pop_size)
  for (i in 1:pop_size) {
    xs <- numeric(dim)
    for (d in 1:dim) {
      r1 <- runif(1)
      v_new <- gsa_velocity_step(velocity[[i]][d], accel[[i]][d], r1)
      velocity[[i]][d] <- v_new
      xs[d] <- clamp(pop[[i]][d] + v_new, lo, hi)
    }
    offspring[[i]] <- xs
  }

  new_fitness <- s$evaluate(offspring)
  used <- used + pop_size
  pop <- offspring
  fitness <- new_fitness  # replace/generational: unconditional overwrite
}

best_pt <- s$best()
gap <- best_pt$f - s$f_opt()
cat(sprintf("gsa: evals_used=%d best_f=%.6g gap=%.6g\n", s$evals_used(), best_pt$f, gap))
s$finish()
