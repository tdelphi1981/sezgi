# M3-8 Task 8: sz_eval_session_tsp() / EvalSession$new_tsp -- the
# permutation-typed ask/tell session (R mirror of py-sezgi's M3-8 Task 7
# `PermSession`/`sezgi.EvalSession.for_problem(sezgi.problems.tsp(...))`).
#
# Mirrors py-sezgi/tests/test_tsp_algo.py's session-level coverage (kind/
# dim, eval counting, budget honored/exhausted, invalid-tour rejection with
# honest per-defect errors, random_permutation validity/determinism/
# divergence) through r-sezgi's own 1-based tour convention (this file's
# tours are permutations of `1:n_cities`, NOT py-sezgi's 0-based
# `0:(n_cities - 1)` -- see `problems.rs`'s module doc, "Index-convention
# decision").

# ---- kind() / dim() ---------------------------------------------------

test_that("a TSP session's kind() is \"permutation\", a BBOB session's is \"float\"", {
  s_tsp <- sz_eval_session_tsp("berlin52", budget = 10, seed = 1)
  expect_identical(s_tsp$kind(), "permutation")
  s_tsp$finish()

  s_bbob <- sz_eval_session(fid = 1, dim = 3, instance = 1, budget = 10)
  expect_identical(s_bbob$kind(), "float")
  s_bbob$finish()
})

test_that("dim() is n_cities for a TSP session", {
  s <- sz_eval_session_tsp("berlin52", budget = 10, seed = 1)
  expect_identical(s$dim(), 52)
  s$finish()

  s2 <- sz_eval_session_tsp("eil51", budget = 10, seed = 1)
  expect_identical(s2$dim(), 51)
  s2$finish()
})

test_that("bounds() errors clearly for a TSP session", {
  s <- sz_eval_session_tsp("berlin52", budget = 10, seed = 1)
  expect_error(s$bounds(), "permutation")
  s$finish()
})

# ---- random_permutation() ----------------------------------------------

test_that("random_permutation() returns a valid 1-based tour", {
  s <- sz_eval_session_tsp("berlin52", budget = 10, seed = 1)
  t <- s$random_permutation()
  expect_length(t, 52)
  expect_identical(sort(t), as.double(1:52))
  s$finish()
})

test_that("random_permutation() is deterministic under a fixed seed", {
  s1 <- sz_eval_session_tsp("berlin52", budget = 10, seed = 42)
  t1 <- s1$random_permutation()
  s1$finish()

  s2 <- sz_eval_session_tsp("berlin52", budget = 10, seed = 42)
  t2 <- s2$random_permutation()
  s2$finish()

  expect_identical(t1, t2)
})

test_that("random_permutation() diverges across different seeds", {
  s1 <- sz_eval_session_tsp("berlin52", budget = 10, seed = 1)
  t1 <- s1$random_permutation()
  s1$finish()

  s2 <- sz_eval_session_tsp("berlin52", budget = 10, seed = 2)
  t2 <- s2$random_permutation()
  s2$finish()

  expect_false(identical(t1, t2))
})

test_that("successive random_permutation() draws differ within one session", {
  s <- sz_eval_session_tsp("berlin52", budget = 10, seed = 1)
  t1 <- s$random_permutation()
  t2 <- s$random_permutation()
  expect_false(identical(t1, t2))
  s$finish()
})

test_that("random_permutation() is only available for a permutation-typed session", {
  s <- sz_eval_session(fid = 1, dim = 3, instance = 1, budget = 10)
  expect_error(s$random_permutation(), "permutation")
  s$finish()
})

test_that("random_permutation() matches the Python-derived draw (RNG parity, seed=42)", {
  # ANCHORED: the underlying 0-based draw is bit-identical to py-sezgi's own
  # PermSession::random_permutation for the same seed (this module's own
  # session.rs doc, "Permutation-typed sessions" -- SAME PERM_SESSION_RNG_TAG,
  # SAME fisher_yates_shuffle core). Derived by running:
  #   ./py-sezgi/.venv/bin/python -c "
  #   import sezgi
  #   s = sezgi.EvalSession.for_problem(sezgi.problems.tsp('berlin52'), budget=10, seed=42)
  #   print([c + 1 for c in s.random_permutation()[:10]])
  #   "
  #   -> [2, 31, 23, 36, 32, 19, 13, 28, 50, 46] (Python's 0-based draw, +1)
  s <- sz_eval_session_tsp("berlin52", budget = 10, seed = 42)
  t <- s$random_permutation()
  expect_identical(head(t, 10), c(2, 31, 23, 36, 32, 19, 13, 28, 50, 46))
  s$finish()
})

# ---- evaluate() ----------------------------------------------------------

berlin52_opt_tour_1based <- c(
  1, 49, 32, 45, 19, 41, 8, 9, 10, 43, 33, 51, 11, 52, 14, 13, 47, 26, 27, 28,
  12, 25, 4, 6, 15, 5, 24, 48, 38, 37, 40, 39, 36, 35, 34, 44, 46, 16, 29, 50,
  20, 23, 30, 2, 7, 42, 21, 17, 3, 18, 31, 22
)

test_that("evaluate() computes the published-optimal tour length exactly", {
  s <- sz_eval_session_tsp("berlin52", budget = 10, seed = 1)
  f <- s$evaluate(matrix(berlin52_opt_tour_1based, nrow = 1))
  expect_identical(f, 7542.0)
  expect_identical(s$evals_used(), 1)
  s$finish()
})

test_that("evaluate() counts across batches and honors the budget", {
  s <- sz_eval_session_tsp("berlin52", budget = 3, seed = 1)
  t <- s$random_permutation()
  s$evaluate(matrix(t, nrow = 1))
  expect_identical(s$evals_used(), 1)
  s$evaluate(rbind(t, t))
  expect_identical(s$evals_used(), 3)
  s$finish()
})

test_that("evaluate() rejects a batch exceeding the remaining budget, all-or-nothing", {
  s <- sz_eval_session_tsp("berlin52", budget = 1, seed = 1)
  t <- s$random_permutation()
  expect_error(s$evaluate(rbind(t, t)))
  expect_identical(s$evals_used(), 0) # rejected batch spends nothing
  s$finish()
})

test_that("evaluate() rejects the wrong number of entries", {
  s <- sz_eval_session_tsp("berlin52", budget = 10, seed = 1)
  expect_error(s$evaluate(matrix(as.double(1:51), nrow = 1)), "52")
  expect_identical(s$evals_used(), 0)
  s$finish()
})

test_that("evaluate() rejects an out-of-range entry (0)", {
  s <- sz_eval_session_tsp("berlin52", budget = 10, seed = 1)
  tour <- berlin52_opt_tour_1based
  tour[1] <- 0
  expect_error(s$evaluate(matrix(tour, nrow = 1)), "out-of-range")
  expect_identical(s$evals_used(), 0)
  s$finish()
})

test_that("evaluate() rejects an out-of-range entry (n + 1)", {
  s <- sz_eval_session_tsp("berlin52", budget = 10, seed = 1)
  tour <- berlin52_opt_tour_1based
  tour[1] <- 53
  expect_error(s$evaluate(matrix(tour, nrow = 1)), "out-of-range")
  expect_identical(s$evals_used(), 0)
  s$finish()
})

test_that("evaluate() rejects a repeated city", {
  s <- sz_eval_session_tsp("berlin52", budget = 10, seed = 1)
  tour <- berlin52_opt_tour_1based
  tour[2] <- tour[1]
  expect_error(s$evaluate(matrix(tour, nrow = 1)), "repeated")
  expect_identical(s$evals_used(), 0)
  s$finish()
})

test_that("evaluate() rejects a non-whole-number entry", {
  s <- sz_eval_session_tsp("berlin52", budget = 10, seed = 1)
  tour <- as.double(berlin52_opt_tour_1based)
  tour[1] <- tour[1] + 0.5
  expect_error(s$evaluate(matrix(tour, nrow = 1)))
  expect_identical(s$evals_used(), 0)
  s$finish()
})

test_that("evaluate() validates every row all-or-nothing before charging anything", {
  s <- sz_eval_session_tsp("berlin52", budget = 10, seed = 1)
  bad <- berlin52_opt_tour_1based
  bad[1] <- 0 # invalid
  expect_error(s$evaluate(rbind(berlin52_opt_tour_1based, bad)))
  expect_identical(s$evals_used(), 0) # the valid first row was NOT charged either
  s$finish()
})

test_that("evaluate() accepts a list of tours, same as a matrix", {
  s <- sz_eval_session_tsp("berlin52", budget = 10, seed = 1)
  f_matrix <- s$evaluate(matrix(berlin52_opt_tour_1based, nrow = 1))
  s$finish()

  s2 <- sz_eval_session_tsp("berlin52", budget = 10, seed = 1)
  f_list <- s2$evaluate(list(berlin52_opt_tour_1based))
  s2$finish()

  expect_identical(f_matrix, f_list)
})

# ---- best() / f_opt() -----------------------------------------------------

test_that("best() tracks the strictly-improving tour, 1-based", {
  s <- sz_eval_session_tsp("berlin52", budget = 10, seed = 1)
  bad_tour <- c(berlin52_opt_tour_1based[-1], berlin52_opt_tour_1based[1]) # a rotation, not optimal-length necessarily but valid
  s$evaluate(matrix(bad_tour, nrow = 1))
  s$evaluate(matrix(berlin52_opt_tour_1based, nrow = 1)) # strictly better (7542, the published optimum)
  b <- s$best()
  expect_identical(b$f, 7542.0)
  expect_identical(sort(b$x), as.double(1:52))
  s$finish()
})

test_that("f_opt() is the published berlin52 optimum", {
  s <- sz_eval_session_tsp("berlin52", budget = 10, seed = 1)
  expect_identical(s$f_opt(), 7542.0)
  s$finish()
})

# ---- finish() --------------------------------------------------------------

test_that("finish() marks a TSP session finished; every method errors afterward", {
  s <- sz_eval_session_tsp("berlin52", budget = 10, seed = 1)
  s$finish()
  expect_error(s$evals_used(), "session finished")
  expect_error(s$random_permutation(), "session finished")
  expect_error(s$finish(), "session finished")
})

# ---- new_tsp() rejects an unknown vendored name ---------------------------

test_that("sz_eval_session_tsp rejects an unknown vendored name", {
  expect_error(sz_eval_session_tsp("not-a-real-instance", budget = 10, seed = 1))
})
