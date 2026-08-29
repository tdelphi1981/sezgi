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

# T11 fix round 1: every sz_preset_* builder and sz_solve_bbob used to cast
# their f64 params directly (`pop_size as usize`, `budget as u64`,
# `master_seed as u64`, `run_id as u64`), bypassing f64_to_u64/f64_to_usize
# and silently truncating a fractional value -- the one place in the R
# binding surface the task 11 "no silent truncation path remains" contract
# had missed. Now rejected with "expected a whole number".
test_that("a preset builder rejects a fractional pop_size/budget", {
  expect_error(sz_preset_de_rand_1(20.5, 2000), "expected a whole number")
  expect_error(sz_preset_de_rand_1(20, 2000.5), "expected a whole number")
})

test_that("sz_solve_bbob rejects a fractional master_seed/run_id", {
  spec <- sz_preset_de_rand_1(20, 2000)
  expect_error(
    sz_solve_bbob(spec, fid = 1L, dim = 5L, instance = 1L, master_seed = 42.5, run_id = 0),
    "expected a whole number"
  )
  expect_error(
    sz_solve_bbob(spec, fid = 1L, dim = 5L, instance = 1L, master_seed = 42, run_id = 0.5),
    "expected a whole number"
  )
})
