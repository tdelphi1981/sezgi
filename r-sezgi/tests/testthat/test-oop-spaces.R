# M4-2 Task 1: S3 space builders (R/spaces.R) -- the R mirror of
# py-sezgi's `sezgi.Float`/`Int`/`Categorical`/`Binary`/`Permutation`/
# `Space` (`py-sezgi/python/sezgi/spaces.py`). Construction happy paths
# for all 5 block kinds, `sz_space()` with 1 and with 3 mixed blocks
# (plus its own two validation rules -- at least one block, every
# argument an `sz_block`), print smoke, and `.sz_space_to_blocks()`
# round-trip shape.
#
# NOTE on scope: `spaces.py`'s five block dataclasses perform NO
# construction-time validation of `lo`/`hi`/`n`/`k` -- no
# `__post_init__`, nothing (verified by reading the file in full; the
# ONLY validation anywhere in `spaces.py` is `Space.__init__`'s two
# rules, covered below). This test file therefore does not test
# `lo >= hi` / non-whole `n` / `k < 2` rejection at construction time --
# there is nothing to mirror; `spaces.py` accepts all of these silently
# too, and actual bound/arity checking happens only once a space reaches
# the Rust core (`SearchSpace::new`, `crates/core/src/space.rs`), which
# this task's R builders never call (no space-consuming Rust binding
# exists yet -- see Task 2).

# ---- sz_float -------------------------------------------------------------

test_that("sz_float constructs a readable sz_block_float/sz_block object", {
  b <- sz_float(-5.0, 5.0, 3)
  expect_s3_class(b, c("sz_block_float", "sz_block"), exact = TRUE)
  expect_equal(b$lo, -5.0)
  expect_equal(b$hi, 5.0)
  expect_equal(b$n, 3)
})

test_that("sz_float requires n (no default, mirroring spaces.py's Float.n)", {
  # spaces.py's `Float.n` has no default either -- omitting it is a
  # Python TypeError (missing required positional argument); the R
  # mirror is base R's own "argument is missing, with no default" for
  # the same reason (n is unbound when list(n = n) forces it).
  expect_error(sz_float(0, 1), '"n"', fixed = TRUE)
})

test_that("sz_float stores fields verbatim -- no construction-time validation", {
  # Mirrors spaces.py's Float dataclass exactly: lo >= hi and a
  # non-whole-number n are both accepted at construction (spaces.py has
  # no __post_init__ either); only the Rust core would reject them, and
  # this task builds no path there.
  b <- sz_float(5.0, 1.0, 2.5)
  expect_equal(b$lo, 5.0)
  expect_equal(b$hi, 1.0)
  expect_equal(b$n, 2.5)
})

# ---- sz_int -----------------------------------------------------------

test_that("sz_int constructs a readable sz_block_int/sz_block object", {
  b <- sz_int(0L, 10L, 4L)
  expect_s3_class(b, c("sz_block_int", "sz_block"), exact = TRUE)
  expect_equal(b$lo, 0L)
  expect_equal(b$hi, 10L)
  expect_equal(b$n, 4L)
})

test_that("sz_int requires n (no default, mirroring spaces.py's Int.n)", {
  expect_error(sz_int(0L, 10L), '"n"', fixed = TRUE)
})

# ---- sz_categorical -----------------------------------------------------

test_that("sz_categorical constructs a readable sz_block_categorical/sz_block object", {
  b <- sz_categorical(4L, 6L)
  expect_s3_class(b, c("sz_block_categorical", "sz_block"), exact = TRUE)
  expect_equal(b$k, 4L)
  expect_equal(b$n, 6L)
})

test_that("sz_categorical requires n (no default, mirroring spaces.py's Categorical.n)", {
  expect_error(sz_categorical(3L), '"n"', fixed = TRUE)
})

test_that("sz_categorical accepts k < 2 -- no construction-time validation", {
  b <- sz_categorical(1L, 2L)
  expect_equal(b$k, 1L)
})

# ---- sz_binary ----------------------------------------------------------

test_that("sz_binary constructs a readable sz_block_binary/sz_block object", {
  b <- sz_binary(8L)
  expect_s3_class(b, c("sz_block_binary", "sz_block"), exact = TRUE)
  expect_equal(b$n, 8L)
})

# ---- sz_permutation -------------------------------------------------------

test_that("sz_permutation constructs a readable sz_block_permutation/sz_block object", {
  b <- sz_permutation(5L)
  expect_s3_class(b, c("sz_block_permutation", "sz_block"), exact = TRUE)
  expect_equal(b$n, 5L)
})

# ---- sz_space -----------------------------------------------------------

test_that("sz_space with a single block", {
  s <- sz_space(sz_float(-5, 5, 3))
  expect_s3_class(s, "sz_space", exact = TRUE)
  expect_length(s$blocks, 1)
  expect_s3_class(s$blocks[[1]], "sz_block_float")
})

test_that("sz_space with 3 mixed blocks, order preserved", {
  s <- sz_space(sz_float(-5, 5, 2), sz_permutation(4), sz_categorical(3, 2))
  expect_s3_class(s, "sz_space", exact = TRUE)
  expect_length(s$blocks, 3)
  expect_s3_class(s$blocks[[1]], "sz_block_float")
  expect_s3_class(s$blocks[[2]], "sz_block_permutation")
  expect_s3_class(s$blocks[[3]], "sz_block_categorical")
})

test_that("sz_space() with no blocks errors", {
  expect_error(sz_space(), "at least one block")
})

test_that("sz_space() rejects a non-block argument", {
  expect_error(sz_space(sz_float(-5, 5, 1), 42), "sz_block")
})

# ---- print smoke ----------------------------------------------------------

test_that("print.sz_block and print.sz_space produce output without erroring", {
  expect_output(print(sz_float(-5, 5, 3)), "sz_block_float")
  expect_output(print(sz_binary(4)), "sz_block_binary")

  s <- sz_space(sz_float(-5, 5, 2), sz_binary(3))
  out <- capture.output(print(s))
  expect_true(any(grepl("sz_space: 2 blocks", out)))
  expect_true(any(grepl("sz_block_float", out)))
  expect_true(any(grepl("sz_block_binary", out)))
})

test_that("print.sz_space singular block count reads \"1 block\"", {
  out <- capture.output(print(sz_space(sz_binary(2))))
  expect_true(any(grepl("sz_space: 1 block(?!s)", out, perl = TRUE)))
})

# ---- .sz_space_to_blocks round-trip shape --------------------------------

test_that(".sz_space_to_blocks serializes a single-block space", {
  s <- sz_space(sz_float(-5, 5, 3))
  out <- sezgi:::.sz_space_to_blocks(s)
  expect_type(out, "list")
  expect_length(out, 1)
  expect_equal(out[[1]], list(type = "float", lo = -5, hi = 5, n = 3L))
})

test_that(".sz_space_to_blocks serializes a 5-block mixed space, one entry per kind", {
  s <- sz_space(
    sz_float(-5, 5, 2),
    sz_int(0L, 9L, 1L),
    sz_categorical(4L, 3L),
    sz_binary(6L),
    sz_permutation(4L)
  )
  out <- sezgi:::.sz_space_to_blocks(s)
  expect_length(out, 5)

  expect_equal(out[[1]], list(type = "float", lo = -5, hi = 5, n = 2L))
  expect_equal(out[[2]], list(type = "int", lo = 0L, hi = 9L, n = 1L))
  expect_equal(out[[3]], list(type = "categorical", k = 4L, n = 3L))
  expect_equal(out[[4]], list(type = "binary", n = 6L))
  expect_equal(out[[5]], list(type = "permutation", n = 4L))
})

test_that(".sz_space_to_blocks coerces lo/hi to double and n/k to integer", {
  s <- sz_space(sz_float(1L, 2L, 3L))
  out <- sezgi:::.sz_space_to_blocks(s)
  expect_true(is.double(out[[1]]$lo))
  expect_true(is.double(out[[1]]$hi))
  expect_true(is.integer(out[[1]]$n))
})

test_that(".sz_space_to_blocks rejects a non-sz_space argument", {
  expect_error(sezgi:::.sz_space_to_blocks(sz_float(-5, 5, 1)), "sz_space")
  expect_error(sezgi:::.sz_space_to_blocks(42), "sz_space")
})
