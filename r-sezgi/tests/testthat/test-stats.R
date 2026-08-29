# Fixtures below are copied verbatim (inputs + expected constants) from the
# pinned #[cfg(test)] fixtures in crates/stats/src/{ranks,pairwise,bayesian,
# report}.rs. Never re-derive an expected value by running code -- the
# in-repo Rust tests are the authority; each test_that() below names the
# Rust test it was copied from.

test_that("friedman matches the pinned hand-derived fixture (ranks.rs::friedman_hand_computed_fixture)", {
  # results[i] = [A, B, C] on problem i (rows = problems, cols = algorithms).
  m <- rbind(
    c(1, 2, 3),
    c(2, 3, 1),
    c(3, 1, 2),
    c(1, 3, 2)
  )
  r <- sz_stats_friedman(m)
  expect_named(r, c("statistic", "p_value", "mean_ranks"))
  expect_equal(r$mean_ranks, c(1.75, 2.25, 2.0), tolerance = 1e-10)
  expect_equal(r$statistic, 0.5, tolerance = 1e-10)
  expect_equal(r$p_value, exp(-0.25), tolerance = 1e-8)
})

test_that("friedman matrix marshalling catches a transposition (2x3, asymmetric)", {
  # 2 problems x 3 algorithms. If the R matrix (column-major) were read as if
  # row-major (i.e. rows/cols swapped), this would be parsed as 3 problems x
  # 2 algorithms -- mean_ranks would come back length 2, not 3, and with
  # different values. Column-major data: col1=[10,30], col2=[20,10],
  # col3=[30,20] -> row0=[10,20,30] (ranks 1,2,3), row1=[30,10,20]
  # (ranks 3,1,2).
  m <- matrix(c(10, 30, 20, 10, 30, 20), nrow = 2, ncol = 3)
  r <- sz_stats_friedman(m)
  expect_length(r$mean_ranks, 3)
  # mean rank per algorithm: algo1=(1+3)/2=2, algo2=(2+1)/2=1.5, algo3=(3+2)/2=2.5
  expect_equal(r$mean_ranks, c(2.0, 1.5, 2.5), tolerance = 1e-12)
})

test_that("wilcoxon matches the pinned no-ties-no-zeros fixture (pairwise.rs::wilcoxon_no_ties_no_zeros)", {
  a <- c(101, 98, 103, 96, 105, 94, 107, 92, 109, 90)
  b <- rep(100, 10)
  r <- sz_stats_wilcoxon(a, b)
  expect_named(r, c("w_statistic", "z", "p_value", "n_effective"))
  expect_equal(r$n_effective, 10)
  expect_equal(r$w_statistic, 25.0, tolerance = 1e-12)
  expect_equal(r$z, -0.20385887657505022, tolerance = 1e-6)
  expect_equal(r$p_value, 0.8384637819224636, tolerance = 1e-6)
})

test_that("wilcoxon matches the pinned Pratt zero-handling fixture (pairwise.rs::wilcoxon_pratt_zero_handling)", {
  a <- c(100, 102, 97, 100, 105, 96, 106, 99)
  b <- rep(100, 8)
  r <- sz_stats_wilcoxon(a, b)
  expect_equal(r$n_effective, 6)
  expect_equal(r$w_statistic, 14.0, tolerance = 1e-12)
  expect_equal(r$z, -0.2835524820033343, tolerance = 1e-6)
  expect_equal(r$p_value, 0.7767533567940004, tolerance = 1e-6)
})

test_that("cliffs_delta matches the pinned known-value fixture (pairwise.rs::cliffs_delta_known_value)", {
  a <- c(1, 2, 3)
  b <- c(2, 3, 4)
  delta <- sz_stats_cliffs_delta(a, b)
  expect_equal(delta, -5 / 9, tolerance = 1e-12)
})

test_that("cliffs_magnitude matches the pinned boundary fixture (pairwise.rs::cliffs_magnitude_boundaries)", {
  expect_equal(sz_stats_cliffs_magnitude(0.1469), "negligible")
  expect_equal(sz_stats_cliffs_magnitude(0.1471), "small")
  expect_equal(sz_stats_cliffs_magnitude(0.147), "small")
  expect_equal(sz_stats_cliffs_magnitude(0.3299), "small")
  expect_equal(sz_stats_cliffs_magnitude(0.3301), "medium")
  expect_equal(sz_stats_cliffs_magnitude(0.4739), "medium")
  expect_equal(sz_stats_cliffs_magnitude(0.4741), "large")
  expect_equal(sz_stats_cliffs_magnitude(-0.5), "large")
  expect_equal(sz_stats_cliffs_magnitude(0.0), "negligible")
})

test_that("bayesian_signed_rank matches the pinned rope-boundary fixture (bayesian.rs::rope_zero_splits)", {
  a <- rep(3, 6)
  b <- rep(3, 6)
  r <- sz_stats_bayesian_signed_rank(a, b, rope = 0.0, samples = 500, seed = 1)
  expect_named(r, c("p_left", "p_rope", "p_right"))
  expect_equal(r$p_rope, 1.0, tolerance = 1e-9)
})

test_that("bayesian_signed_rank is EXACTLY reproducible given the same seed (bayesian.rs::deterministic_given_seed)", {
  a <- c(1, 2, 3, 4, 5)
  b <- c(1.5, 1.5, 3.5, 3.0, 6.0)
  r1 <- sz_stats_bayesian_signed_rank(a, b, rope = 0.2, samples = 500, seed = 777)
  r2 <- sz_stats_bayesian_signed_rank(a, b, rope = 0.2, samples = 500, seed = 777)
  # Exact equality (not tolerance): same seeded Monte Carlo in the same Rust
  # core must be bit-identical.
  expect_identical(r1, r2)
})

test_that("bayesian_signed_rank defaults are rope=0, samples=20000, seed=1", {
  a <- rep(3, 6)
  b <- rep(3, 6)
  r_default <- sz_stats_bayesian_signed_rank(a, b)
  r_explicit <- sz_stats_bayesian_signed_rank(a, b, rope = 0, samples = 20000, seed = 1)
  expect_identical(r_default, r_explicit)
})

test_that("plackett_luce matches the pinned win-fraction fixture (bayesian.rs::two_algorithms_reduce_to_win_fraction)", {
  # Rust fixture (0-based): vec![vec![0,1], vec![0,1], vec![0,1], vec![1,0]].
  # R rankings are 1-based item ids (sz_stats_plackett_luce() doc): item 1 =
  # algo A, item 2 = algo B.
  rankings <- list(c(1, 2), c(1, 2), c(1, 2), c(2, 1))
  r <- sz_stats_plackett_luce(rankings)
  expect_named(r, c("worths", "p_best", "iterations"))
  expect_equal(r$worths[1], 0.75, tolerance = 0.02)
  expect_true(r$worths[1] > r$worths[2])
})

test_that("plackett_luce p_best sums to one and equals worths (bayesian.rs::p_best_sums_to_one)", {
  # Rust fixture (0-based): vec![vec![0,1,2], vec![2,1,0], vec![1,0,2]].
  rankings <- list(c(1, 2, 3), c(3, 2, 1), c(2, 1, 3))
  r <- sz_stats_plackett_luce(rankings)
  expect_equal(sum(r$p_best), 1.0, tolerance = 1e-9)
  expect_equal(r$p_best, r$worths)
})

test_that("paper_package matches the pinned 3x6 synthetic fixture (report.rs::paper_package_3x6_synthetic)", {
  algo_names <- c("Algo_A", "Algo_B", "Algo_C")
  problem_names <- paste0("P", 0:5)
  # results[i] = [Algo_A, Algo_B, Algo_C] on problem i (rows = problems).
  results <- rbind(
    c(1.0, 2.0, 3.0),
    c(1.5, 2.2, 2.9),
    c(1.2, 1.9, 3.1),
    c(0.9, 2.3, 3.2),
    c(1.3, 2.1, 2.8),
    c(1.1, 2.0, 3.0)
  )
  r <- sz_stats_paper_package(algo_names, problem_names, results, rope = 0.1, samples = 1000, seed = 42)
  expect_named(r, c(
    "friedman", "nemenyi_cd", "pairwise_wilcoxon_holm", "cliffs", "bayes",
    "plackett_luce", "latex_summary", "latex_tests"
  ))
  expect_named(r$friedman, c("statistic", "p_value", "mean_ranks"))
  expect_length(r$friedman$mean_ranks, 3)
  expect_length(r$pairwise_wilcoxon_holm, 3) # C(3,2) = 3 pairs
  expect_length(r$cliffs, 3)
  expect_length(r$bayes, 3)
  expect_length(r$plackett_luce$worths, 3)
  expect_named(r$plackett_luce, c("worths", "p_best", "iterations"))

  for (row in r$pairwise_wilcoxon_holm) {
    expect_length(row, 3)
    p <- row[3]
    expect_true(p >= 0.0 && p <= 1.0)
  }

  for (entry in r$bayes) {
    expect_length(entry, 3)
    expect_named(entry[[3]], c("p_left", "p_rope", "p_right"))
  }

  expect_equal(sum(r$plackett_luce$worths), 1.0, tolerance = 1e-9)

  expect_true(nzchar(r$latex_summary))
  expect_true(nzchar(r$latex_tests))
  # paper_package cells are single aggregated values (n = 1): the summary
  # table must emit just the mean, never "NaN" (sample std's n-1 denominator
  # is zero for n = 1).
  expect_false(grepl("NaN", r$latex_summary, fixed = TRUE))
  expect_false(grepl("NaN", r$latex_tests, fixed = TRUE))
})

test_that("paper_package errors on too few problems (report.rs::paper_package_errors_on_too_few_problems)", {
  algo_names <- c("A", "B")
  problem_names <- c("P1", "P2", "P3")
  results <- rbind(c(1.0, 2.0), c(1.5, 1.8), c(1.2, 2.1))
  expect_error(sz_stats_paper_package(algo_names, problem_names, results, rope = 0.1, samples = 100, seed = 777))
})
