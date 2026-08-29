# Pure-R Grasshopper Optimisation Algorithm (Saremi, Mirjalili & Lewis 2017,
# "Grasshopper optimisation algorithm: theory and application", Advances in
# Engineering Software, 105, 30-47).
#
# Teaches the SAME pinned update equations as sezgi's `gen/goa` Rust
# component (see crates/components/src/goa.rs's module doc), base R only.
# GOA's core update draws NOTHING from RNG (fully deterministic given the
# population and `c`); `runif` here seeds only the initial population. Not
# RNG-stream-identical to the Rust preset; see examples/specs/goa.toml for
# that.
#
# Per the Rust module's IMPLEMENTER-VERIFY note, this follows the VERIFIED
# reference-implementation semantics (confirmed against the paper's own
# MATLAB File Exchange #61421 code, via `thieu1995/mealpy`'s `OriginalGOA`)
# rather than the paper's literal prose: pairwise distance is mapped into
# the bounded [2, 4) interval before applying the social force `s(.)`. No
# equivalence-critique citation -- none is cited by the Rust module's doc.
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
goa_f <- 0.5
goa_l <- 1.5
goa_c_max <- 1.0
goa_c_min <- 1e-5
eps <- 1e-12

# Social force s(r) = f*e^(-r/l) - e^(-r) (Eq. 2.3).
goa_s <- function(r) goa_f * exp(-r / goa_l) - exp(-r)

goa_pair_term <- function(c, half_range_d, x_i_d, x_j_d, d_ij) {
  # sezgi simplification: maps the raw Euclidean distance into [2, 4)
  # before applying s(.) -- the verified reference-implementation
  # semantics, see this file's module doc.
  dist_term <- 2.0 + (d_ij %% 2.0)
  s <- goa_s(dist_term)
  # sezgi simplification: + eps guards a zero-distance pair.
  c * half_range_d * s * (x_j_d - x_i_d) / (d_ij + eps)
}

euclidean_dist <- function(a, b) sqrt(sum((a - b) ^ 2))

clamp <- function(x, lo, hi) min(max(x, lo), hi)

set.seed(seed)
# s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget,
#                       log_dir = "logs/goa-r", algo_name = "goa-r", seed = seed)
s <- sz_eval_session(fid = 1, dim = dim, instance = 1, budget = budget)

half_range <- (hi - lo) / 2.0
pop <- lapply(1:pop_size, function(i) runif(dim, lo, hi))
fitness <- s$evaluate(pop)
used <- pop_size

while (used + pop_size <= budget) {
  progress <- min(max(used / budget, 0.0), 1.0)
  c <- goa_c_max - progress * (goa_c_max - goa_c_min)
  best <- order(fitness)[1]
  x_best <- pop[[best]]

  dists <- matrix(0.0, pop_size, pop_size)
  for (i in 1:pop_size) {
    for (j in 1:pop_size) {
      dists[i, j] <- euclidean_dist(pop[[i]], pop[[j]])
    }
  }

  offspring <- vector("list", pop_size)
  for (i in 1:pop_size) {
    x_i <- pop[[i]]
    new_x <- numeric(dim)
    for (d in 1:dim) {
      total <- 0.0
      for (j in 1:pop_size) {
        if (j == i) next
        total <- total + goa_pair_term(c, half_range, x_i[d], pop[[j]][d], dists[i, j])
      }
      new_x[d] <- clamp(c * total + x_best[d], lo, hi)
    }
    offspring[[i]] <- new_x
  }

  # replace/generational, whole-batch-or-nothing per iteration.
  fitness <- s$evaluate(offspring)
  pop <- offspring
  used <- used + pop_size
}

best_pt <- s$best()
gap <- best_pt$f - s$f_opt()
cat(sprintf("goa: evals_used=%d best_f=%.6g gap=%.6g\n", s$evals_used(), best_pt$f, gap))
s$finish()
