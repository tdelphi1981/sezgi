# OOP example: `FeatureSelection` (M4-2 Task 6, `r-sezgi/R/recipes.R`)
# recovering a known informative-column mask from a fixed synthetic
# dataset via `GeneticAlgorithm`'s space-driven auto-dispatch (M4-2 Task
# 5) -- `FeatureSelection$space()` is `sz_binary(n_features)`, so the
# wrapper auto-dispatches to `sz_preset_ga_bin`.
#
# Structural twin of examples/python/oop/feature_selection.py, but the
# dataset itself is an INDEPENDENT derivation (base-R `set.seed()`/
# `rnorm()`, not numpy) -- see r-sezgi/tests/testthat/test-oop-recipes.R
# for the brute-force 2^8 = 256-mask uniqueness cross-check of this exact
# configuration.
#
# Dataset (deterministic, pure base R, no external package anywhere): a
# fixed 20x8 design matrix `X` drawn from a seeded RNG, with exactly 3
# informative columns (1-based indices 2, 4, 7) that linearly determine
# `y` up to Gaussian noise; the other 5 columns are pure noise,
# uncorrelated with `y` beyond sampling coincidence. Least-squares
# in-sample residual sum of squares (RSS) is only ever weakly reduced by
# adding MORE columns (a noise column can never make RSS worse), so a
# pure-RSS search over-selects every column -- `FeatureSelection`'s own
# `penalty` term (a fraction of the full feature count) is what makes the
# 3-column informative subset the actual global minimum; see this
# script's own `PENALTY` constant and `test-oop-recipes.R`'s brute-force
# cross-check of that claim.
#
# `scorer(X_sub, y)`: ordinary least squares (`qr.solve()`, with an
# intercept column prepended) residual sum of squares -- a minimal,
# dependency-free stand-in for a real cross-validated scorer
# (`FeatureSelection`'s own doc sketches an sklearn-shaped example, kept
# as a comment, never an actual dependency).

library(sezgi)

DATA_SEED <- 42
N_SAMPLES <- 20L
N_FEATURES <- 8L
INFORMATIVE <- c(2L, 4L, 7L) # 1-based R column indices
COEFS <- c(3.0, -2.0, 1.5)
NOISE_SD <- 0.2
PENALTY <- 0.3
POP_SIZE <- 20
BUDGET <- 200
GA_SEED <- 42

make_dataset <- function() {
  set.seed(DATA_SEED)
  X <- matrix(rnorm(N_SAMPLES * N_FEATURES), nrow = N_SAMPLES, ncol = N_FEATURES)
  noise <- rnorm(N_SAMPLES, sd = NOISE_SD)
  y <- as.vector(X[, INFORMATIVE, drop = FALSE] %*% COEFS + noise)
  list(X = X, y = y)
}

scorer <- function(X_sub, y) {
  # Ordinary-least-squares residual sum of squares (with an intercept
  # column) -- lower is better, matching FeatureSelection's minimize
  # convention.
  design <- cbind(1, X_sub)
  coefs <- qr.solve(design, y)
  resid <- y - design %*% coefs
  sum(resid^2)
}

ds <- make_dataset()
problem <- FeatureSelection$new(ds$X, ds$y, scorer, penalty = PENALTY)
res <- GeneticAlgorithm$new(pop_size = POP_SIZE)$run(problem, budget = BUDGET, seed = GA_SEED)

mask <- as.logical(res$best_x)
informative_mask <- seq_len(N_FEATURES) %in% INFORMATIVE
recovered <- identical(mask, informative_mask)
mask_str <- paste(ifelse(mask, "1", "0"), collapse = "")

cat(sprintf(
  "feature_selection (oop): evals_used=%d best_f=%.10g popcount=%d mask=%s recovered=%s\n",
  res$evals_used, res$best_f, sum(mask), mask_str, recovered
))

stopifnot(recovered)
