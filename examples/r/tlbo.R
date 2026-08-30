# Pure-R Teaching-Learning-Based Optimization (Rao, R.V., Savsani, V.J.,
# Vakharia, D.P. 2011, "Teaching-learning-based optimization: A novel
# method for constrained mechanical design optimization problems",
# Computer-Aided Design 43(3), 303-315).
#
# Teaches the SAME pinned update equations as sezgi's `gen/tlbo-teacher` +
# `gen/tlbo-learner` (each paired with `replace/one-to-one-greedy`) Rust
# components -- sezgi's FIRST multi-stage preset (see
# crates/components/src/tlbo.rs's module doc) -- with base R only, driven
# through `sz_eval_session`. Does NOT reproduce the Rust component's
# RNG-stream (draw order) contract, only the same update rule; see
# examples/specs/tlbo.toml for the pinned-RNG preset.
#
# Tier note: a labeled metaphor preset -- faithful to a well-regarded
# THIRD-PARTY reimplementation's own update equations (Yarpiz's tlbo.m,
# Project Code YPEA111 -- Rao's own code was not independently locatable),
# with a pinned deterministic draw order, but NOT validated against the
# paper's reported benchmark numbers.
#
# This script models BOTH generation stages EXPLICITLY, one after the
# other: a full teacher pass (with its own greedy accept per learner)
# followed by a full learner pass (also with its own greedy accept) --
# 2*pop_size evaluations per generation. TF is randi{1,2}, drawn ONCE PER
# LEARNER. The learner phase's partner j is UNCONDITIONALLY DISTINCT from
# i (never self-selecting).
#
# sezgi does NOT implement any duplicate-removal/re-evaluation step (see
# Crepinsek, Liu & Mernik 2012, Information Sciences 212, 79-93, for the
# documented critique of TLBO's "parameter-free" framing) -- every
# evaluation is counted honestly, exactly 2*pop_size per generation.
#
# No cross-algorithm quality claims are made or implied here -- single
# seed, single problem, reported as a gap.

library(sezgi)

dim <- 5
budget <- 2000
pop_size <- 30  # number of learners
seed <- 42
lo <- -5.0
hi <- 5.0

tlbo_teacher_dim_step <- function(x_d, teacher_d, mean_d, tf, r) x_d + r * (teacher_d - tf * mean_d)

tlbo_learner_dim_step <- function(x_d, partner_d, r, partner_is_better) {
  step <- x_d - partner_d
  if (partner_is_better) step <- -step
  x_d + r * step
}

clamp <- function(x, lo, hi) min(max(x, lo), hi)

set.seed(seed)
# s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget,
#                       log_dir = "logs/tlbo-r", algo_name = "tlbo-r", seed = seed)
s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget)

pop <- lapply(1:pop_size, function(i) runif(dim, lo, hi))
fitness <- s$evaluate(pop)
used <- pop_size

while (used + 2 * pop_size <= budget) {
  # ---- Teacher phase (population mean first, then teacher -- no draw) ----
  mean_vec <- Reduce(`+`, pop) / pop_size
  teacher_idx <- order(fitness)[1]
  teacher <- pop[[teacher_idx]]

  teacher_offspring <- vector("list", pop_size)
  for (i in 1:pop_size) {
    x <- pop[[i]]
    tf <- 1.0 + sample(0:1, 1)  # randi{1,2}, once per learner
    new_x <- vapply(1:dim, function(d) clamp(tlbo_teacher_dim_step(x[d], teacher[d], mean_vec[d], tf, runif(1)), lo, hi), numeric(1))
    teacher_offspring[[i]] <- new_x
  }
  new_fitness <- s$evaluate(teacher_offspring)
  used <- used + pop_size
  for (i in 1:pop_size) {
    if (new_fitness[i] < fitness[i]) {
      pop[[i]] <- teacher_offspring[[i]]
      fitness[i] <- new_fitness[i]
    }
  }

  # ---- Learner phase (on the teacher phase's updated population) ----
  learner_offspring <- vector("list", pop_size)
  for (i in 1:pop_size) {
    j <- sample.int(pop_size, 1)
    while (j == i) j <- sample.int(pop_size, 1)
    partner_is_better <- fitness[j] < fitness[i]
    x <- pop[[i]]; partner <- pop[[j]]
    new_x <- vapply(1:dim, function(d) clamp(tlbo_learner_dim_step(x[d], partner[d], runif(1), partner_is_better), lo, hi), numeric(1))
    learner_offspring[[i]] <- new_x
  }
  new_fitness <- s$evaluate(learner_offspring)
  used <- used + pop_size
  for (i in 1:pop_size) {
    if (new_fitness[i] < fitness[i]) {
      pop[[i]] <- learner_offspring[[i]]
      fitness[i] <- new_fitness[i]
    }
  }
}

best_pt <- s$best()
gap <- best_pt$f - s$f_opt()
cat(sprintf("tlbo: evals_used=%d best_f=%.6g gap=%.6g\n", s$evals_used(), best_pt$f, gap))
s$finish()
