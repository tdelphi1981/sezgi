# Pure-R Moth-Flame Optimization (Mirjalili, S. 2015, "Moth-flame
# optimization algorithm: A novel nature-inspired heuristic paradigm",
# Knowledge-Based Systems 89, 228-249).
#
# Teaches the SAME pinned update equations and flame-memory mechanism as
# sezgi's `gen/mfo` + `adapter/mfo-flame-update` Rust components (see
# crates/components/src/mfo.rs's module doc) with base R only, driven
# through `sz_eval_session`. Does NOT reproduce the Rust component's
# RNG-stream (draw order) contract, only the same update rule; see
# examples/specs/mfo.toml for the pinned-RNG preset.
#
# Tier note: a labeled metaphor preset -- faithful to MFO.m's own
# reference implementation, with a pinned deterministic draw order, but
# NOT validated against the paper's reported benchmark numbers.
# Equivalence critique: Camacho-Villalon, Dorigo & Stutzle (International
# Transactions in Operational Research, six-algorithm critique: grey
# wolf, moth-flame, whale, firefly, bat, antlion) -- cited here
# conservatively, as background for why this is "labeled metaphor" rather
# than a mechanism sezgi treats as novel.
#
# The flame memory is a genuine persisted archive here too, held in plain
# R variables across generations (this pure script has no blackboard --
# the Rust component's `adapter/mfo-flame-update` is this state's
# canonical owner there; here the main loop plays that role directly).
#
# No cross-algorithm quality claims are made or implied here -- single
# seed, single problem, reported as a gap.

library(sezgi)

dim <- 5
budget <- 2000
pop_size <- 30  # number of search agents (also the flame archive size)
seed <- 42
lo <- -5.0
hi <- 5.0

mfo_dim_step <- function(x_d, own_flame_d, target_flame_d, t) {
  b <- 1.0
  distance_to_flame <- abs(own_flame_d - x_d)
  distance_to_flame * exp(b * t) * cos(t * 2.0 * pi) + target_flame_d
}

mfo_flame_count <- function(n, progress) {
  raw <- n - progress * (n - 1)
  min(n, max(1, round(raw)))
}

# Concatenate (pop first, flames second), stable-sort ascending by
# fitness, truncate to length(pop_fitness) -- MFO.m's own elitist
# double_population=[previous_population; best_flames] step.
merge_and_truncate <- function(pop_fitness, pop_positions, flame_fitness, flame_positions) {
  combined_fitness <- c(pop_fitness, flame_fitness)
  combined_positions <- c(pop_positions, flame_positions)
  ord <- order(combined_fitness)  # stable: ties keep pop-before-flames order
  keep <- ord[1:length(pop_fitness)]
  list(fitness = combined_fitness[keep], positions = combined_positions[keep])
}

clamp <- function(x, lo, hi) min(max(x, lo), hi)

set.seed(seed)
# s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget,
#                       log_dir = "logs/mfo-r", algo_name = "mfo-r", seed = seed)
s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget)

pop <- lapply(1:pop_size, function(i) runif(dim, lo, hi))
fitness <- s$evaluate(pop)
used <- pop_size

# Bootstrap: generation 0 seeds the flame archive from the sorted initial
# population (a merge against an EMPTY old-flame set).
mem <- merge_and_truncate(fitness, pop, numeric(0), list())
flame_fitness <- mem$fitness
flames <- mem$positions

while (used + pop_size <= budget) {
  progress <- min(max(used / budget, 0.0), 1.0)
  a <- -1.0 - progress
  flame_count <- mfo_flame_count(pop_size, progress)

  offspring <- vector("list", pop_size)
  for (i in 1:pop_size) {
    x <- pop[[i]]
    own_flame <- flames[[i]]
    target_flame <- flames[[min(i, flame_count)]]
    new_x <- numeric(dim)
    for (d in 1:dim) {
      t <- (a - 1.0) * runif(1) + 1.0
      new_x[d] <- clamp(mfo_dim_step(x[d], own_flame[d], target_flame[d], t), lo, hi)
    }
    offspring[[i]] <- new_x
  }

  # replace/generational: the flames, not the moth population, carry the
  # elitism, so the moths are overwritten unconditionally.
  new_fitness <- s$evaluate(offspring)
  used <- used + pop_size
  pop <- offspring
  fitness <- new_fitness

  # adapter/mfo-flame-update: merge this generation's moved moths against
  # the PREVIOUS flames, truncate back to pop_size.
  mem <- merge_and_truncate(fitness, pop, flame_fitness, flames)
  flame_fitness <- mem$fitness
  flames <- mem$positions
}

best <- s$best()
gap <- best$f - s$f_opt()
cat(sprintf("mfo: evals_used=%d best_f=%.6g gap=%.6g\n", s$evals_used(), best$f, gap))
s$finish()
