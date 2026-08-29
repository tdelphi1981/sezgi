# Pure-R Harmony Search (Geem, Kim & Loganathan 2001, "A new heuristic
# optimization algorithm: harmony search", Simulation).
#
# Teaches the SAME pinned update equations as sezgi's `gen/hs` Rust
# component (see crates/components/src/hs.rs's module doc) with base R
# only, driven through `sz_eval_session`. Does NOT reproduce the Rust
# component's RNG-stream (draw order) contract, only the same update rule;
# see examples/specs/hs.toml for the pinned-RNG preset.
#
# Equivalence critique: Weyland (2010), whose analysis shows HS's harmony-
# memory-consideration/pitch-adjustment mechanism is, component for
# component, a special case of evolution strategies.
#
# No cross-algorithm quality claims are made or implied here -- single
# seed, single problem, reported as a gap.

library(sezgi)

dim <- 5
budget <- 2000
hms <- 30  # harmony memory size (sezgi's "pop_size" for this preset)
seed <- 42
lo <- -5.0
hi <- 5.0
hmcr <- 0.9
par <- 0.3
bw_fraction <- 0.01

hs_dim_step <- function(hmcr, par, bw_d, lo_d, hi_d, memory_d) {
  u1 <- runif(1)
  if (u1 < hmcr) {
    j <- sample.int(length(memory_d), 1)
    value <- memory_d[j]
    u2 <- runif(1)
    if (u2 < par) {
      u3 <- runif(1)
      value <- value + bw_d * (2.0 * u3 - 1.0)
    }
    return(value)
  }
  u4 <- runif(1)
  lo_d + u4 * (hi_d - lo_d)
}

clamp <- function(x, lo, hi) min(max(x, lo), hi)

set.seed(seed)
# s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget,
#                       log_dir = "logs/hs-r", algo_name = "hs-r", seed = seed)
s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget)

bw <- rep(bw_fraction * (hi - lo), dim)
memory <- lapply(1:hms, function(i) runif(dim, lo, hi))
fitness <- s$evaluate(memory)
used <- hms

# Exactly ONE new harmony per iteration -- unlike the whole-population
# generators (gwo/woa/goa), a single-eval iteration always fits until the
# budget is fully exhausted.
while (used < budget) {
  new_harmony <- numeric(dim)
  for (d in 1:dim) {
    memory_d <- vapply(memory, function(row) row[d], numeric(1))
    new_harmony[d] <- clamp(hs_dim_step(hmcr, par, bw[d], lo, hi, memory_d), lo, hi)
  }
  new_f <- s$evaluate(list(new_harmony))[1]
  used <- used + 1

  # replace/worst-if-better: replace the current worst iff new_f beats it.
  worst <- which.max(fitness)
  if (new_f < fitness[worst]) {
    memory[[worst]] <- new_harmony
    fitness[worst] <- new_f
  }
}

best <- s$best()
gap <- best$f - s$f_opt()
cat(sprintf("hs: evals_used=%d best_f=%.6g gap=%.6g\n", s$evals_used(), best$f, gap))
s$finish()
