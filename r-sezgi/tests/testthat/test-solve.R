test_that("sz_solve_bbob runs DE on BBOB f1 and returns a sane result", {
  spec <- sz_preset_de_rand_1(20, 2000)  # added in this task, see Step 2
  r <- sz_solve_bbob(spec, fid = 1L, dim = 5L, instance = 1L,
                     master_seed = 42, run_id = 0)
  expect_type(r, "list")
  expect_named(r, c("best_f", "evals", "best_x"))
  expect_length(r$best_x, 5L)
  expect_true(is.finite(r$best_f))
  expect_equal(r$evals, 2000)
})
