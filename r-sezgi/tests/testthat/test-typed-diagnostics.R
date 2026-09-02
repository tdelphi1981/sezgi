# M3-8 Task 10: R bindings for the typed operator families and the
# diagnostic problem trio -- R mirror of py-sezgi's Task 9
# (py-sezgi/tests/test_typed_diagnostics.py):
#
#   - sz_solve_onemax/sz_solve_int_quadratic/sz_solve_cat_match --
#     solve()-path bindings for sezgi_problems::diagnostics::{OneMax,
#     IntQuadratic, CatMatch} (crates/problems/src/diagnostics.rs, Task 5):
#     minimal, hand-verifiable, single-global-optimum landscapes that exist
#     to make ga_bin/ga_int/ga_cat reachable and testable from R, NOT
#     benchmark targets (that module's own "diagnostic, not benchmark"
#     framing).
#   - sz_preset_ga_bin/ga_int/ga_cat (Tasks 2-4), pairing 1:1 with the
#     diagnostics above.
#   - sz_solve_onemax/sz_solve_int_quadratic/sz_solve_cat_match end to end,
#     at the SAME (n_bits/lo/hi/n/k, pop_size, budget, master_seed) anchors
#     crates/problems/tests/solve_typed_presets.rs / py-sezgi's own
#     test_typed_diagnostics.py use -- these bindings run the IDENTICAL
#     Engine::from_spec/Engine::run core those Rust tests do, so the same
#     inputs must reach the SAME exact best_f == 0.0.
#   - sz_solve_mixed_diagnostic -- a Task 10 addition (NOT one of Task 5's
#     brief-pinned diagnostics), the R mirror of py-sezgi's
#     `mixed_diagnostic` Problem (Task 9), proving gen/compound (Task 5) is
#     reachable end to end through the NORMAL R solve path from an
#     R-authored, mixed-space TOML AlgorithmSpec (see `solve.rs`'s own doc
#     for why this one binding takes `spec_toml` + `AlgorithmSpec::from_toml`
#     rather than `spec_json`, unlike its three siblings above).
#
# Typed-result-genotype design decision (implemented + documented in
# r-sezgi/src/rust/src/solve.rs's block_values_to_r/genotype_to_r, restated
# here -- the R mirror of py-sezgi's block_values_to_py decision):
#   - Float  -> a numeric (double) vector -- BYTE-IDENTICAL to every
#     sz_solve_bbob/sz_solve_cec2022/... result before this task (no
#     behavior change for existing callers).
#   - Int    -> an integer vector.
#   - Cat    -> an integer vector of category INDICES 0..k (no label
#     concept -- CatMatch/gen/ga-cat have none).
#   - Bin    -> a logical vector, one per bit (R's native boolean, matching
#     Rust's own Vec<bool> 1:1 -- the SAME choice `sz_mo_read_moa`'s
#     `MoArchiveGenotype::Binary -> OwnedLogicalSexp` conversion already
#     makes, per mo.rs).
#   - A MULTI-block genotype (reachable only via sz_solve_mixed_diagnostic)
#     is an unnamed list of per-block vectors, one per SearchSpace block in
#     order, each typed as above.

f64_bits_hex <- function(x) {
  raw <- writeBin(x, raw(), size = 8L, endian = "little")
  paste0(rev(sprintf("%02x", as.integer(raw))), collapse = "")
}

# ---------------------------------------------------------------------
# sz_preset_ga_bin / sz_solve_onemax
# ---------------------------------------------------------------------

test_that("sz_preset_ga_bin returns a JSON spec naming gen/ga-bin", {
  spec <- sz_preset_ga_bin(20, 100)
  expect_type(spec, "character")
  expect_match(spec, "gen/ga-bin", fixed = TRUE)
})

test_that("sz_solve_onemax reaches OneMax's exact optimum", {
  # Anchor mirrors crates/problems/tests/solve_typed_presets.rs::
  # ga_bin_solves_one_max exactly: n_bits=20, pop_size=40, budget=6000,
  # master_seed=1.
  spec <- sz_preset_ga_bin(40, 6000)
  r <- sz_solve_onemax(spec, n_bits = 20, master_seed = 1, run_id = 0)
  expect_identical(r$best_f, 0)
  expect_length(r$best_x, 20L)
  expect_type(r$best_x, "logical")
  expect_true(all(r$best_x))  # the all-true genotype, at OneMax's optimum
})

test_that("sz_solve_onemax rejects a fractional n_bits/master_seed", {
  spec <- sz_preset_ga_bin(8, 40)
  expect_error(sz_solve_onemax(spec, n_bits = 20.5, master_seed = 0, run_id = 0), "expected a whole number")
  expect_error(sz_solve_onemax(spec, n_bits = 20, master_seed = 0.5, run_id = 0), "expected a whole number")
})

# ---------------------------------------------------------------------
# sz_preset_ga_int / sz_solve_int_quadratic
# ---------------------------------------------------------------------

test_that("sz_preset_ga_int returns a JSON spec naming gen/ga-int", {
  spec <- sz_preset_ga_int(20, 100)
  expect_match(spec, "gen/ga-int", fixed = TRUE)
})

test_that("sz_solve_int_quadratic reaches IntQuadratic's exact optimum", {
  # Anchor mirrors solve_typed_presets.rs::ga_int_solves_int_quadratic:
  # lo=-10, hi=10, n=5, pop_size=40, budget=8000, master_seed=3.
  spec <- sz_preset_ga_int(40, 8000)
  r <- sz_solve_int_quadratic(spec, lo = -10, hi = 10, n = 5, master_seed = 3, run_id = 0)
  expect_identical(r$best_f, 0)
  expect_length(r$best_x, 5L)
  expect_type(r$best_x, "integer")
})

test_that("sz_solve_int_quadratic rejects lo >= hi", {
  spec <- sz_preset_ga_int(10, 100)
  expect_error(sz_solve_int_quadratic(spec, lo = 5, hi = 5, n = 3, master_seed = 0, run_id = 0), "lo")
  expect_error(sz_solve_int_quadratic(spec, lo = 5, hi = -5, n = 3, master_seed = 0, run_id = 0), "lo")
})

# ---------------------------------------------------------------------
# sz_preset_ga_cat / sz_solve_cat_match
# ---------------------------------------------------------------------

test_that("sz_preset_ga_cat returns a JSON spec naming gen/ga-cat", {
  spec <- sz_preset_ga_cat(20, 100)
  expect_match(spec, "gen/ga-cat", fixed = TRUE)
})

test_that("sz_solve_cat_match reaches CatMatch's exact optimum", {
  # Anchor mirrors solve_typed_presets.rs::ga_cat_solves_cat_match:
  # k=4, n=8, seed=123, pop_size=40, budget=8000, master_seed=5.
  spec <- sz_preset_ga_cat(40, 8000)
  r <- sz_solve_cat_match(spec, k = 4, n = 8, seed = 123, master_seed = 5, run_id = 0)
  expect_identical(r$best_f, 0)
  expect_length(r$best_x, 8L)
  expect_type(r$best_x, "integer")
  expect_true(all(r$best_x >= 0 & r$best_x < 4))
})

test_that("sz_solve_cat_match's different construction seed yields a different target", {
  # CatMatch::new's seed IS caller-supplied (unlike int_quadratic's fixed
  # internal seed) -- a different construction seed derives a different
  # target, so two ga_cat runs each converging to their OWN problem's exact
  # optimum (best_f == 0) land on different genotypes.
  spec <- sz_preset_ga_cat(40, 8000)
  r1 <- sz_solve_cat_match(spec, k = 4, n = 8, seed = 123, master_seed = 5, run_id = 0)
  r2 <- sz_solve_cat_match(spec, k = 4, n = 8, seed = 999, master_seed = 5, run_id = 0)
  expect_identical(r1$best_f, 0)
  expect_identical(r2$best_f, 0)
  expect_false(identical(r1$best_x, r2$best_x))
})

# ---------------------------------------------------------------------
# gen/compound reachable via an R-authored, mixed-space TOML
# AlgorithmSpec (Task 10's own reachability scaffold: sz_solve_mixed_diagnostic)
# ---------------------------------------------------------------------

MIXED_COMPOUND_TOML <- '
name = "mixed-compound"
pop_size = 12

[init]
kind = "init/uniform"

[boundary]
kind = "boundary/clamp"

[[stages]]

[stages.generator]
kind = "gen/compound"
blocks = [
  { kind = "gen/ga-real", tournament_k = 2, pc = 0.9, eta_c = 15.0, eta_m = 20.0 },
  { kind = "gen/ga-int", tournament_k = 2, p_c = 0.9, eta_c = 15.0, eta_m = 20.0 },
  { kind = "gen/ga-cat", tournament_k = 2, p_c = 0.9 },
  { kind = "gen/ga-bin", tournament_k = 2, p_c = 0.9 },
]

[stages.replacer]
kind = "replace/mu-plus-lambda"

[termination]
budget = 200
'

test_that("gen/compound mixed-space TOML spec solves end to end through the normal R solve path", {
  # Mirrors crates/components/src/compound.rs's own
  # spec_validation_accepts_gen_compound_on_matching_mixed_space test's
  # block layout (Float(2)+Int(2)+Categorical(k=3,n=2)+Binary(4)) and its
  # compound_spec_json()'s exact sub-generator params -- proving the SAME
  # mixed-space gen/compound spec is reachable from an R-authored TOML
  # document via AlgorithmSpec::from_toml (no R-side TOML library: R has no
  # stdlib TOML parser the way Python's tomllib is, so parsing happens on
  # the Rust side -- see sz_solve_mixed_diagnostic's own doc, "solve.rs").
  r <- sz_solve_mixed_diagnostic(MIXED_COMPOUND_TOML, n_float = 2, n_int = 2,
                                  k_cat = 3, n_cat = 2, n_bin = 4, master_seed = 42)

  expect_true(is.numeric(r$best_f))
  # Multi-block genotype: best_x is a list of 4 per-block vectors, in
  # SearchSpace::blocks() order (Float, Int, Categorical, Binary).
  expect_length(r$best_x, 4L)
  float_block <- r$best_x[[1]]
  int_block <- r$best_x[[2]]
  cat_block <- r$best_x[[3]]
  bin_block <- r$best_x[[4]]
  expect_length(float_block, 2L); expect_type(float_block, "double")
  expect_length(int_block, 2L); expect_type(int_block, "integer")
  expect_length(cat_block, 2L); expect_type(cat_block, "integer")
  expect_true(all(cat_block >= 0 & cat_block < 3))
  expect_length(bin_block, 4L); expect_type(bin_block, "logical")
})

# ---------------------------------------------------------------------
# Cross-language anchor (house precedent: test-tsp-two-opt.R's "R best_f is
# bit-identical to the Python twin" pattern, test-mo.R's wfg4_nsga2 pair
# anchor): sz_solve_onemax(sz_preset_ga_bin(...)) on OneMax(20), compared to
# an independently-derived Python golden hex.
#
# ga_bin_solves_one_max's OWN anchor (n_bits=20, pop_size=40, budget=6000,
# master_seed=1, tested above) reaches best_f == 0.0 EXACTLY -- a trivial
# anchor that pins nothing about the run's actual trajectory (any correctly
# terminating algorithm reaching the known optimum gives the same 0.0). This
# anchor instead uses a MUCH smaller budget (40, vs. the exact-optimum
# anchor's 6000) that does NOT reach OneMax's optimum, so the pinned best_f
# depends on the full generational trajectory (selection/crossover/mutation
# draw order), not just on eventual convergence.
# ---------------------------------------------------------------------

test_that("R sz_solve_onemax(ga_bin) is bit-identical to the Python twin (cross-language hex anchor)", {
  # Golden hex derived by running the Python side directly:
  #
  #   ./py-sezgi/.venv/bin/python -c "
  #   import struct
  #   import sezgi
  #   p = sezgi.problems.onemax(20)
  #   r = sezgi.solve(sezgi.presets.ga_bin(8, 40), p, master_seed=7)
  #   print(repr(r['best_f']), struct.pack('>d', r['best_f']).hex())
  #   "
  #   -> best_f = 4.0 (exact -- OneMax's fitness is always an integer
  #   zero-bit count), big-endian hex 4010000000000000 (struct.pack('>d',
  #   ...), matching f64_bits_hex()'s own little-endian-raw-then-reversed =
  #   big-endian output convention -- see test-mo.R's identical note).
  #   evals_used = 40 (== budget: no early termination). Confirmed
  #   NON-optimal: OneMax(20)'s known optimum is 0.0, budget=40 with
  #   pop_size=8 (5 generations) does not reach it.
  spec <- sz_preset_ga_bin(8, 40)
  r <- sz_solve_onemax(spec, n_bits = 20, master_seed = 7, run_id = 0)

  expect_identical(r$best_f, 4)
  expect_identical(r$evals, 40)
  expect_identical(f64_bits_hex(r$best_f), "4010000000000000")
})
