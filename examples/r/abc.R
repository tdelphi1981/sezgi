# Pure-R Artificial Bee Colony (Karaboga, D. 2005, "An Idea Based On Honey
# Bee Swarm For Numerical Optimization", TR-06, Erciyes University; and
# Karaboga, D., Basturk, B. 2007, "A Powerful and Efficient Algorithm for
# Numerical Function Optimization: ABC Algorithm", Journal of Global
# Optimization 39(3), 459-471).
#
# Teaches the SAME pinned update equations as sezgi's `gen/abc-employed` +
# `replace/abc-trial-greedy` + `adapter/abc-onlooker-scout` Rust
# components (see crates/components/src/abc.rs's module doc, verified
# against Karaboga & Basturk's OWN ABCorig.m) with base R only, driven
# through `sz_eval_session`. Does NOT reproduce the Rust components'
# RNG-stream (draw order) contract, only the same update rule; see
# examples/specs/abc.toml for the pinned-RNG preset.
#
# Tier note: a labeled metaphor preset -- faithful to the verified primary
# artifacts' own update equations and loop structure, pinned draw order,
# but NOT validated against any publication's reported benchmark numbers.
# No established equivalence critique covers ABC -- primary sources only.
#
# sezgi's pop_size IS the food-source count SN directly (NOT Karaboga's
# colony size NP=2*SN) -- there is only ever ONE array of solutions;
# onlookers merely SELECT one of the SAME SN sources to perturb. Models
# the THREE stages explicitly, one cycle at a time:
#
# 1. Employed -- exactly SN candidates (one dimension changed per source,
#    x_ij + phi*(x_ij - x_kj), k distinct from i), evaluated as ONE frozen
#    batch, then a per-source greedy accept coupled with a trial counter.
# 2. Onlooker -- a repeated LINEAR SCAN (not literal roulette-wheel
#    sampling): i=1; t=0; while t<SN: if rand<prob(i): t+=1; [[move]];
#    i=(i mod SN)+1. prob is computed ONCE and never recomputed mid-scan.
# 3. Scout -- at most ONE per cycle: the source with max(trial) (ties ->
#    LAST index), re-randomized if its trial count exceeds limit=SN*dim.
#
# No cross-algorithm quality claims are made or implied here -- single
# seed, single problem, reported as a gap.

library(sezgi)

dim <- 5
budget <- 2000
pop_size <- 20  # food-source count SN (== sezgi's pop_size, NOT NP=2*SN)
seed <- 42
lo <- -5.0
hi <- 5.0
limit_val <- pop_size * dim  # SN*dim (Karaboga & Akay 2009's rule of thumb)

abc_dim_step <- function(x_i_j, x_k_j, phi) x_i_j + phi * (x_i_j - x_k_j)

abc_candidate <- function(x_i, x_k, j, phi) {
  c <- x_i
  c[j] <- abc_dim_step(x_i[j], x_k[j], phi)
  c
}

abc_fitness_transform <- function(f) if (f >= 0.0) 1.0 / (f + 1.0) else 1.0 + abs(f)

abc_probabilities <- function(transformed) {
  max_fit <- max(transformed)
  0.9 * (transformed / max_fit) + 0.1
}

# argmax, ties -> LAST index (ABCorig.m's ind(end)).
abc_scout_target <- function(trials) {
  best <- 1
  for (k in 2:length(trials)) {
    if (trials[k] >= trials[best]) best <- k
  }
  best
}

abc_draw_move <- function(sn, dim, i) {
  j <- sample.int(dim, 1)
  repeat {
    k <- sample.int(sn, 1)
    if (k != i) break
  }
  phi <- (runif(1) - 0.5) * 2.0
  list(j = j, k = k, phi = phi)
}

clamp <- function(x, lo, hi) min(max(x, lo), hi)

set.seed(seed)
# s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget,
#                       log_dir = "logs/abc-r", algo_name = "abc-r", seed = seed)
s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget)

pop <- lapply(1:pop_size, function(i) runif(dim, lo, hi))
fitness <- s$evaluate(pop)
used <- pop_size
trials <- rep(0, pop_size)

while (used + pop_size <= budget) {
  # ---- Employed phase: frozen batch, one candidate per source ----
  candidates <- vector("list", pop_size)
  for (i in 1:pop_size) {
    mv <- abc_draw_move(pop_size, dim, i)
    cand <- abc_candidate(pop[[i]], pop[[mv$k]], mv$j, mv$phi)
    candidates[[i]] <- vapply(cand, function(v) clamp(v, lo, hi), numeric(1))
  }
  new_fitness <- s$evaluate(candidates)
  used <- used + pop_size
  for (i in 1:pop_size) {
    if (new_fitness[i] < fitness[i]) {
      pop[[i]] <- candidates[[i]]; fitness[i] <- new_fitness[i]; trials[i] <- 0
    } else {
      trials[i] <- trials[i] + 1
    }
  }

  # ---- Onlooker phase: repeated linear scan, exactly SN accepted visits ----
  transformed <- vapply(fitness, abc_fitness_transform, numeric(1))
  prob <- abc_probabilities(transformed)
  i <- 1; t <- 0; budget_exhausted <- FALSE
  while (t < pop_size && !budget_exhausted) {
    if (runif(1) < prob[i]) {
      t <- t + 1
      mv <- abc_draw_move(pop_size, dim, i)
      cand <- abc_candidate(pop[[i]], pop[[mv$k]], mv$j, mv$phi)
      cand[mv$j] <- clamp(cand[mv$j], lo, hi)
      f <- tryCatch(s$evaluate(list(cand))[1], error = function(e) NULL)
      if (is.null(f)) {
        budget_exhausted <- TRUE
      } else {
        used <- used + 1
        if (f < fitness[i]) {
          pop[[i]] <- cand; fitness[i] <- f; trials[i] <- 0
        } else {
          trials[i] <- trials[i] + 1
        }
      }
    }
    i <- (i %% pop_size) + 1
  }

  # ---- Scout phase: at most one re-randomized source per cycle ----
  if (!budget_exhausted) {
    scout_idx <- abc_scout_target(trials)
    if (trials[scout_idx] > limit_val) {
      new_pos <- runif(dim, lo, hi)
      f <- tryCatch(s$evaluate(list(new_pos))[1], error = function(e) NULL)
      if (!is.null(f)) {
        pop[[scout_idx]] <- new_pos; fitness[scout_idx] <- f; trials[scout_idx] <- 0
        used <- used + 1
      }
    }
  }
}

best_pt <- s$best()
gap <- best_pt$f - s$f_opt()
cat(sprintf("abc: evals_used=%d best_f=%.6g gap=%.6g\n", s$evals_used(), best_pt$f, gap))
s$finish()
