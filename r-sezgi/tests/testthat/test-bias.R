# M3-1 Task 9: R bindings for the bias-scanning module (sz_bias_structural /
# sz_bias_central / sz_bias_report).
#
# Mirrors crates/bias's public structs 1:1 by field name -- see py-sezgi's
# "Bias-scanning bindings" section (py-sezgi/src/lib.rs) and its own doc
# comment for the full method provenance (that crate's own `structural`/
# `central`/`report` modules are the ultimate source). Task 6 (the
# Rajwar-Deep signature test) is DEFERRED in `sezgi-bias` itself -- there is
# no `sz_bias_signature()`; `sz_bias_report()`'s `signature` element is
# always NULL, matching py-sezgi's `sezgi.bias.report()["signature"] is
# None`.
#
# Budgets/dims are kept tiny throughout (test speed), except `runs = 30` for
# the structural scan, which is the BIAS-toolbox method's own verified
# minimum (`sezgi_bias::structural::DEFAULT_RUNS`) -- not something this
# suite shrinks without changing what is actually being tested.

rs_spec <- function(pop_size = 5, budget = 50) {
  sz_preset_random_search(pop_size, budget)
}

# Corrupts the (unique) generator kind string inside a `sz_preset_random_search()`
# JSON spec, to build an invalid spec the same way py-sezgi's tests do
# (`spec["stages"][0]["generator"]["kind"] = "yok/boyle"`) -- the R preset
# builders return the spec as a JSON character scalar, not a list, so string
# substitution on the (uniquely-named) "gen/uniform-resample" kind stands in
# for that structural mutation.
corrupt_generator_kind <- function(spec_json) {
  gsub('"gen/uniform-resample"', '"yok/boyle"', spec_json, fixed = TRUE)
}

f64_bits_hex <- function(x) {
  raw <- writeBin(x, raw(), size = 8L, endian = "little")
  paste0(rev(sprintf("%02x", as.integer(raw))), collapse = "")
}

# ---- sz_bias_structural -----------------------------------------------------

test_that("sz_bias_structural returns the mirrored dict shape", {
  spec <- rs_spec()
  r <- sz_bias_structural(spec, dim = 2, budget = 50, runs = 30, seed = 20260830)

  expect_named(r, c(
    "per_dim_ks", "per_dim_ad", "holm_rejections_ks", "holm_rejections_ad",
    "verdict", "detail", "final_positions"
  ))
  expect_length(r$per_dim_ks, 2)
  expect_length(r$per_dim_ad, 2)
  for (row in r$per_dim_ks) {
    expect_named(row, c("d", "p_value", "n"))
    expect_true(is.double(row$d))
    expect_true(is.double(row$p_value))
  }
  for (row in r$per_dim_ad) {
    expect_named(row, c("a2", "p_value", "n"))
    expect_true(is.double(row$a2))
    expect_true(is.double(row$p_value))
  }
  expect_true(is.double(r$holm_rejections_ks))
  expect_true(is.double(r$holm_rejections_ad))
  expect_true(r$verdict %in% c("no_evidence", "evidence"))
  if (identical(r$verdict, "no_evidence")) {
    expect_null(r$detail)
  } else {
    expect_true(is.character(r$detail))
  }
  expect_length(r$final_positions, 30)
  expect_true(all(vapply(r$final_positions, length, integer(1)) == 2))
})

test_that("sz_bias_structural is deterministic and matches the anchored no-evidence verdict", {
  spec <- rs_spec()
  r1 <- sz_bias_structural(spec, dim = 2, budget = 50, runs = 30, seed = 20260830)
  r2 <- sz_bias_structural(spec, dim = 2, budget = 50, runs = 30, seed = 20260830)
  expect_identical(r1, r2)

  # random_search (pop=5, budget=50) on f0(dim=2), runs=30, seed=20260830 has
  # no directional operator (uniform resampling + elitist replacement), so it
  # should show no structural bias -- ANCHORED (per this project's
  # convention): the actual measured verdict at this seed/config, matching
  # py-sezgi's own anchored fixture
  # (test_structural_deterministic_and_anchored_verdict in
  # py-sezgi/tests/test_bias.py):
  #   per_dim_ks p-values = [0.8931464115659824, 0.7346814853531314]
  #   per_dim_ad p-values = [0.8926490729470692, 0.7520431448530165]
  expect_identical(r1$holm_rejections_ks, 0)
  expect_identical(r1$holm_rejections_ad, 0)
  expect_identical(r1$verdict, "no_evidence")
  expect_null(r1$detail)
  expect_identical(r1$per_dim_ks[[1]]$p_value, 0.8931464115659824)
  expect_identical(r1$per_dim_ad[[1]]$a2, r2$per_dim_ad[[1]]$a2)
  expect_identical(r1$final_positions[[1]][1], r2$final_positions[[1]][1])
})

# ---- Cross-language bit-equality (M3-1 Task 9's own mandate) ---------------
#
# The scan below (random_search(pop=5, budget=50) on f0(dim=2), runs=30,
# seed=20260830) is the SAME tiny seeded structural scan as py-sezgi's own
# anchored fixture above. R and Python both call the identical seeded Rust
# core (`sezgi_bias::structural::structural_bias_scan`), so every f64 they
# return must be BIT-EQUAL, not merely numerically close.
#
# The expected values below were captured by running this exact scan through
# the Python binding directly (`./py-sezgi/.venv/bin/python`, from the
# worktree root):
#
#   ./py-sezgi/.venv/bin/python - <<'PYEOF'
#   import sezgi, struct
#   bits = lambda x: struct.pack('>d', x).hex()
#   spec = sezgi.presets.random_search(pop_size=5, budget=50)
#   r = sezgi.bias.structural(spec, dim=2, budget=50, runs=30, seed=20260830)
#   for i, row in enumerate(r["per_dim_ks"]):
#       print(f'ks[{i}] d={bits(row["d"])} p={bits(row["p_value"])}')
#   for i, row in enumerate(r["per_dim_ad"]):
#       print(f'ad[{i}] a2={bits(row["a2"])} p={bits(row["p_value"])}')
#   print("final_positions[0]:", [bits(v) for v in r["final_positions"][0]])
#   print("final_positions[29]:", [bits(v) for v in r["final_positions"][29]])
#   PYEOF
#
# Python output (raw IEEE-754 big-endian hex of each f64, captured verbatim,
# not re-derived from R):
#
#   ks[0] d=3fba4c36ae1a7a28 p=3fec94a7c886e69c
#   ks[1] d=3fbf4167c14dec88 p=3fe78282bf12305a
#   ad[0] a2=3fd6aaabcb2e91c0 p=3fec9094c9e3989f
#   ad[1] a2=3fdfac30a48967c0 p=3fe810bcc90a63ea
#   final_positions[0]:  [3fe8837493192d02, 3f9fbbbe79789ca0]
#   final_positions[29]: [3fc2516ff5541c08, 3feccad02773d0f1]
#
# NOTE (finding from implementing this test): the comparison below is done
# via raw IEEE-754 bit hex, not `expect_identical()` against a typed decimal
# literal -- R's own decimal-string-to-double parser (`as.numeric()`/the
# literal parser, both routed through the platform strtod) was empirically
# found to NOT always be correctly rounded at full (17-significant-digit)
# precision on this R 4.5.2 build: the literal "0.12209175557466001" (one of
# the pinned per_dim_ks[[2]]$d values below) parses in R to bits
# `...dec87`, one ULP away from the correctly-rounded `...dec88` Python (and
# the Rust core itself, confirmed by reading the SAME field straight off a
# live `sz_bias_structural()` call) produce for that exact decimal text --
# every other literal in this fixture happens to round-trip correctly, but
# nothing here should be trusted to. Comparing via `f64_bits_hex()` (as
# `test-cross-language.R`'s own golden tests already do) sidesteps R's
# decimal-parser entirely: it reads the ACTUAL bytes of the computed double,
# so it cannot be fooled by this quirk in either direction.
test_that("R structural bias scan is bit-identical to the Python binding (cross-language)", {
  spec <- rs_spec()
  r <- sz_bias_structural(spec, dim = 2, budget = 50, runs = 30, seed = 20260830)

  expect_identical(r$holm_rejections_ks, 0)
  expect_identical(r$holm_rejections_ad, 0)
  expect_identical(r$verdict, "no_evidence")

  expect_equal(f64_bits_hex(r$per_dim_ks[[1]]$d), "3fba4c36ae1a7a28")
  expect_equal(f64_bits_hex(r$per_dim_ks[[1]]$p_value), "3fec94a7c886e69c")
  expect_equal(f64_bits_hex(r$per_dim_ks[[2]]$d), "3fbf4167c14dec88")
  expect_equal(f64_bits_hex(r$per_dim_ks[[2]]$p_value), "3fe78282bf12305a")

  expect_equal(f64_bits_hex(r$per_dim_ad[[1]]$a2), "3fd6aaabcb2e91c0")
  expect_equal(f64_bits_hex(r$per_dim_ad[[1]]$p_value), "3fec9094c9e3989f")
  expect_equal(f64_bits_hex(r$per_dim_ad[[2]]$a2), "3fdfac30a48967c0")
  expect_equal(f64_bits_hex(r$per_dim_ad[[2]]$p_value), "3fe810bcc90a63ea")

  expect_equal(f64_bits_hex(r$final_positions[[1]][1]), "3fe8837493192d02")
  expect_equal(f64_bits_hex(r$final_positions[[1]][2]), "3f9fbbbe79789ca0")
  expect_equal(f64_bits_hex(r$final_positions[[30]][1]), "3fc2516ff5541c08")
  expect_equal(f64_bits_hex(r$final_positions[[30]][2]), "3feccad02773d0f1")
})

test_that("sz_bias_structural rejects dim = 0", {
  spec <- rs_spec()
  expect_error(sz_bias_structural(spec, dim = 0, budget = 50, runs = 30, seed = 1))
})

test_that("sz_bias_structural rejects too few runs", {
  spec <- rs_spec()
  expect_error(sz_bias_structural(spec, dim = 2, budget = 50, runs = 3, seed = 1))
})

test_that("sz_bias_structural rejects an invalid spec", {
  spec <- corrupt_generator_kind(rs_spec())
  expect_error(sz_bias_structural(spec, dim = 2, budget = 50, runs = 30, seed = 1), "yok/boyle")
})

test_that("sz_bias_structural rejects malformed JSON", {
  expect_error(sz_bias_structural("{not json", dim = 2, budget = 50, runs = 30, seed = 1))
})

test_that("a fractional runs/dim/seed is rejected, not silently truncated", {
  spec <- rs_spec()
  expect_error(sz_bias_structural(spec, dim = 2, budget = 50, runs = 30.5, seed = 1), "expected a whole number")
  expect_error(sz_bias_structural(spec, dim = 2.5, budget = 50, runs = 30, seed = 1), "expected a whole number")
  expect_error(sz_bias_structural(spec, dim = 2, budget = 50, runs = 30, seed = 1.5), "expected a whole number")
  expect_error(sz_bias_structural(spec, dim = 2, budget = 50.5, runs = 30, seed = 1), "expected a whole number")
})

# ---- sz_bias_central ---------------------------------------------------------

test_that("sz_bias_central returns the mirrored dict shape", {
  spec <- rs_spec()
  r <- sz_bias_central(spec, dim = 2, budget = 50, fids = 1, instances_shifted = 1,
                        runs_per = 5, seed = 20260830)

  expect_named(r, c("gap_centered", "gap_shifted", "wilcoxon", "effect", "verdict", "detail"))
  expect_length(r$gap_centered, 5)
  expect_length(r$gap_shifted, 5)
  expect_true(is.double(r$gap_centered))
  expect_true(is.double(r$gap_shifted))
  expect_named(r$wilcoxon, c("w_statistic", "z", "p_value", "n_effective", "method"))
  expect_true(r$wilcoxon$method %in% c("exact", "normal_approx"))
  expect_true(is.double(r$effect))
  expect_true(r$verdict %in% c("no_evidence", "evidence"))
  if (identical(r$verdict, "no_evidence")) {
    expect_null(r$detail)
  } else {
    expect_true(is.character(r$detail))
  }
})

test_that("sz_bias_central is deterministic given the same seed", {
  spec <- rs_spec()
  r1 <- sz_bias_central(spec, dim = 2, budget = 50, fids = 1, instances_shifted = 1,
                         runs_per = 5, seed = 20260830)
  r2 <- sz_bias_central(spec, dim = 2, budget = 50, fids = 1, instances_shifted = 1,
                         runs_per = 5, seed = 20260830)
  expect_identical(r1, r2)
})

test_that("sz_bias_central defaults fids/instances_shifted to c(1,4,13) x c(1,2)", {
  spec <- rs_spec()
  r <- sz_bias_central(spec, dim = 2, budget = 50, runs_per = 1, seed = 1)
  # 3 fids * 2 instances * 1 run_per = 6 pairs.
  expect_length(r$gap_centered, 6)
  expect_length(r$gap_shifted, 6)
})

test_that("sz_bias_central rejects every non-translation-invariant fid", {
  spec <- rs_spec()
  for (fid in c(5, 6, 20, 24)) {
    expect_error(
      sz_bias_central(spec, dim = 5, budget = 50, fids = fid, instances_shifted = c(1, 2),
                       runs_per = 3, seed = 1)
    )
  }
})

test_that("sz_bias_central rejects too few (fid, instance, run) pairs", {
  spec <- rs_spec()
  expect_error(
    sz_bias_central(spec, dim = 2, budget = 50, fids = 1, instances_shifted = 1,
                     runs_per = 1, seed = 1)
  )
})

# ---- sz_bias_report -----------------------------------------------------------

test_that("sz_bias_report returns the mirrored dict shape, signature NULL, latex NaN-free", {
  spec <- rs_spec()
  r <- sz_bias_report(spec, dim = 2, budget = 50, seed = 20260830,
                       structural_runs = 30, central_fids = 1,
                       central_instances = 1, central_runs_per = 5)

  expect_named(r, c("structural", "central", "signature", "latex_summary", "plot_data"))
  expect_named(r$structural, c(
    "per_dim_ks", "per_dim_ad", "holm_rejections_ks", "holm_rejections_ad",
    "verdict", "detail", "final_positions"
  ))
  expect_named(r$central, c("gap_centered", "gap_shifted", "wilcoxon", "effect", "verdict", "detail"))
  # T6 (Rajwar-Deep signature test) is deferred in sezgi-bias itself: always
  # NULL -- the method could not be pinned from accessible sources.
  expect_null(r$signature)
  expect_false(grepl("NaN", r$latex_summary, fixed = TRUE))
  expect_true(grepl("\\toprule", r$latex_summary, fixed = TRUE))
  expect_true(grepl("not run: the Rajwar-Deep method could not be pinned", r$latex_summary, fixed = TRUE))
  expect_named(r$plot_data, c("final_positions", "gap_centered", "gap_shifted"))
  expect_identical(r$plot_data$final_positions, r$structural$final_positions)
  expect_identical(r$plot_data$gap_centered, r$central$gap_centered)
  expect_identical(r$plot_data$gap_shifted, r$central$gap_shifted)
})

test_that("sz_bias_report is deterministic given the same seed", {
  spec <- rs_spec()
  r1 <- sz_bias_report(spec, dim = 2, budget = 50, seed = 20260830,
                        structural_runs = 30, central_fids = 1,
                        central_instances = 1, central_runs_per = 5)
  r2 <- sz_bias_report(spec, dim = 2, budget = 50, seed = 20260830,
                        structural_runs = 30, central_fids = 1,
                        central_instances = 1, central_runs_per = 5)
  expect_identical(r1, r2)
})

test_that("sz_bias_report defaults use the documented verified-method defaults", {
  # Omitting every optional knob falls back to BiasReportConfig::new's own
  # documented defaults: structural_runs=30, central_fids=c(1,4,13),
  # central_instances=c(1,2), central_runs_per=20 -- so the default report
  # has 30 final_positions rows and 3*2*20=120 gap pairs.
  spec <- rs_spec()
  r <- sz_bias_report(spec, dim = 2, budget = 50, seed = 1)
  expect_length(r$structural$final_positions, 30)
  expect_length(r$central$gap_centered, 120)
  expect_length(r$central$gap_shifted, 120)
})

test_that("sz_bias_report rejects an invalid spec", {
  spec <- corrupt_generator_kind(rs_spec())
  expect_error(
    sz_bias_report(spec, dim = 2, budget = 50, seed = 1,
                    structural_runs = 30, central_fids = 1,
                    central_instances = 1, central_runs_per = 5),
    "yok/boyle"
  )
})

test_that("sz_bias_report rejects an invalid central fid", {
  spec <- rs_spec()
  expect_error(
    sz_bias_report(spec, dim = 2, budget = 50, seed = 1,
                    structural_runs = 30, central_fids = 5,
                    central_instances = 1, central_runs_per = 5)
  )
})
