golden_path <- function(name) {
  # repo-relative; skip on CRAN/installed-package contexts
  p <- file.path("..", "..", "..", "tests", "golden", name)
  if (!file.exists(p)) skip("golden fixtures not available (not in repo checkout)")
  p
}

f64_bits_hex <- function(x) {
  raw <- writeBin(x, raw(), size = 8L, endian = "little")
  paste0(rev(sprintf("%02x", as.integer(raw))), collapse = "")
}

test_that("R trajectory is bit-identical to the Rust/Python golden (DE rand1)", {
  skip_if_not_installed("jsonlite")
  g <- jsonlite::fromJSON(golden_path("de_bbob_f1_seed42.json"))
  spec <- sz_preset_de_rand_1(20, 2000)
  r <- sz_solve_bbob(spec, fid = as.integer(g$problem$fid),
                     dim = as.integer(g$problem$dim),
                     instance = as.integer(g$problem$instance),
                     master_seed = as.double(g$master_seed), run_id = as.double(g$run_id))
  expect_equal(f64_bits_hex(r$best_f), g$best_f_bits)  # c05f7cc7f16d3d8b
})

test_that("R trajectory is bit-identical to the Rust/Python golden (DE best1)", {
  skip_if_not_installed("jsonlite")
  g <- jsonlite::fromJSON(golden_path("de_best1_bbob_f1_seed42.json"))
  spec <- sz_preset_de_best_1(20, 2000)
  r <- sz_solve_bbob(spec, fid = as.integer(g$problem$fid),
                     dim = as.integer(g$problem$dim),
                     instance = as.integer(g$problem$instance),
                     master_seed = as.double(g$master_seed), run_id = as.double(g$run_id))
  expect_equal(f64_bits_hex(r$best_f), g$best_f_bits)  # c05f7cc7bd2a2711
})
