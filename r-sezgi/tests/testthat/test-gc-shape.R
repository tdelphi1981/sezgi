# GC shape audit, Task 1 (`.superpowers/sdd/2026-09-03-sezgi-gc-shape-audit/
# task-1-brief.md`) -- crate-wide `set_name_and_value` provenance audit and
# fix, closing the M4-2 deferral (`docs/DECISIONS.md`'s M4-2 "Deferrals
# onward" entry).
#
# Mechanism (full explanation: `solve.rs`'s `sz_solve_r_problem` result-tail
# comment, and this crate's own M4-2 `test-oop-problem.R`/`test-oop-bridge.R`
# gctorture loops): savvy 0.10.2's `set_name_and_value(i, k, v)` calls
# `set_name` (an UNCONDITIONAL CHARSXP allocation, `R_MakeUnwindCont` inside
# `unwind_protect_impl`) BEFORE attaching `v`. If `v` arrives already a bare,
# unprotected `Sexp` (its `Owned*Sexp` token already dropped via `.into()`
# before reaching the call), that intervening allocation can GC the orphaned
# value out from under it -- a real, gctorture-provable crasher (M4-2 proved
# a live segfault at `sz_solve_onemax`, rep 5). An `Owned*Sexp` value still
# holding its own token at the call is unaffected: `set_name_and_value` takes
# `v: T` BY VALUE and only runs `v.into()` AFTER `set_name` returns, so the
# owned token is still alive across the allocation.
#
# This task's exhaustive, provenance-traced audit covered all 130
# `.set_name_and_value(` call sites in `r-sezgi/src/rust/src/*.rs` at the
# audit's base commit (100 outside `solve.rs` + `solve.rs`'s own 30; the 6
# `solve.rs` sites M4-2 had already converted away from `set_name_and_value`
# entirely are not among them). It found exactly 9 REAL (bare-`Sexp`) sites,
# all fixed via the identical `set_value`-then-`set_name` split, with the
# same mechanism comment:
#   - `mo.rs` (1): `geno`, `sz_mo_read_moa`'s per-record `genotype` field
#   - `bias.rs` (7): `structural_result_list`'s `verdict`/`detail`,
#     `central_result_list`'s `verdict`/`detail`, and `sz_bias_report_raw`'s
#     `signature_sexp` plus the `sig` sub-list's own `verdict_sexp`/
#     `detail_sexp` inside the `Some(verdict)` arm
#   - `problems.rs` (1): `known_optimum`, `sz_tsp_load`
# See `docs/DECISIONS.md`'s "GC shape audit (Task 1) closed" record for the
# full classification and the exact grep/line references.
#
# Three modules have a fixed site reachable from an exported R function; one
# test below per module, mirroring the M4-2 loop style (`gctorture(TRUE)`,
# tiny workloads, result-correctness assertions INSIDE the loop, not just
# survival).
#
# Assertion shape / runtime note: under `gctorture(TRUE)` EVERY R allocation
# forces a full GC, and testthat's own `expect_*` machinery allocates
# heavily -- a per-field expectation style costs ~2.5s PER EXPECTATION here
# (measured: the naive first draft of this file ran >10 minutes). Each loop
# therefore makes exactly ONE bundled `expect_identical()` per rep, over a
# list that pins every field the fix touches (and, for `mo`/`bias`, the exact
# values from an UNTORTURED baseline call captured before the loop). This is
# strictly stronger than per-field shape checks -- it is an exact-value
# comparison, which is precisely what the GC bug corrupts (M4-2's own symptom
# was a collected value's slot coming back holding a NEIGHBOURING field's
# value) -- while keeping the whole file at ~75s.
#
# Honest gap: `bias.rs`'s `sig.set_value(0/1, ...)` / `sig.set_name(0/1,
# ...)` split (inside `sz_bias_report_raw`'s `Some(verdict) => { ... }` arm)
# is NOT exercised by any loop here, and cannot be from R today -- the
# Rajwar-Deep signature test (T6) is DEFERRED in `sezgi-bias` itself (no
# method could be pinned; see `crates/bias/src/report.rs`'s module doc),
# so `BiasReport::signature` is unconditionally `None` and that arm is
# unreachable from any exported function. It was still fixed (mechanically
# identical, zero-behavior-change split; the build confirms it type-checks)
# for when T6 eventually lands, matching `sz_bias_report_raw`'s own doc
# comment's stated policy of mapping `signature` through properly rather
# than hardcoding around the gap. The THIRD bias fix in that same function
# -- `out.set_value(2, signature_sexp)` / `out.set_name(2, "signature")`,
# which runs unconditionally regardless of which arm built `signature_sexp`
# -- IS exercised below: every `sz_bias_report()` call passes through it.

# ---- mo.rs: sz_mo_read_moa's per-record `genotype` field -------------------
#
# The fixed site (`geno`, `mo.rs`) lives entirely inside `sz_mo_read_moa`'s
# own parse-and-build path, not inside the `sz_nsga2` run that produces the
# archive file it reads -- so the (untortured, one-shot) run that generates
# a small `.moa` archive is hoisted OUTSIDE the `gctorture(TRUE)` window,
# and only the repeated `sz_mo_read_moa` calls (the actual fixed code path)
# are tortured. This keeps the loop cheap while still exercising the fixed
# site 8 times.

test_that("repeated sz_mo_read_moa reads complete correctly under gctorture(TRUE)", {
  log_dir <- tempfile("sezgi-moa-gcshape-")
  dir.create(log_dir)
  sz_nsga2("zdt1", dim = 2, pop_size = 4, budget = 12, seed = 1,
           log_dir = log_dir, label = "gcshape")
  path <- file.path(log_dir, "gcshape-s1.moa")
  expect_true(file.exists(path))

  # Untortured baseline: reading the same file is deterministic, so every
  # tortured rep below must reproduce these EXACT values -- including each
  # record's `genotype`, the field whose `set_value`/`set_name` split this
  # loop exists to cover.
  moa_probe <- function(d) {
    list(
      names = names(d),
      algo = d$algo,
      problem = d$problem,
      kind = d$kind,
      rec_names = lapply(d$records, names),
      genotype = lapply(d$records, function(r) r$genotype),
      objectives = lapply(d$records, function(r) r$objectives),
      archive = d$archive
    )
  }
  baseline <- moa_probe(sz_mo_read_moa(path))
  expect_identical(baseline$algo, "nsga2")
  expect_identical(baseline$kind, "float")
  expect_true(length(baseline$genotype) >= 1)

  gctorture(TRUE)
  on.exit(gctorture(FALSE))

  for (rep in seq_len(8)) {
    expect_identical(moa_probe(sz_mo_read_moa(path)), baseline)
  }
})

# ---- bias.rs: sz_bias_report's verdict/detail (structural + central) and
# its own signature field -----------------------------------------------
#
# Each `sz_bias_report()` call drives `structural_bias_scan`'s own verified
# minimum (`MIN_RUNS = 5`) plus `central_bias_scan`'s own verified minimum
# (`MIN_PAIRS = 5` total `(fid, instance, run)` triples; both crate-side
# invariants, not shrinkable) worth of tiny algorithm runs -- but all of that
# is pure Rust compute inside ONE call, allocating no R objects, so it stays
# cheap even under torture. 4 tortured reps at a fixed seed; the same seed
# gives a deterministic result, so each rep is compared field-by-field to an
# untortured baseline (this doubles as the post-fix determinism check cited
# in `docs/DECISIONS.md`'s behavior-neutrality note).

test_that("repeated sz_bias_report runs complete correctly under gctorture(TRUE)", {
  spec <- sz_preset_random_search(4, 8)
  bias_args <- list(dim = 2, budget = 8, seed = 1, structural_runs = 5,
                    central_runs_per = 5, central_fids = c(1),
                    central_instances = c(1))
  run_report <- function() do.call(sz_bias_report, c(list(spec), bias_args))

  bias_probe <- function(r) {
    list(
      names = names(r),
      structural_names = names(r$structural),
      structural_verdict = r$structural$verdict,
      structural_detail = r$structural$detail,
      central_names = names(r$central),
      central_verdict = r$central$verdict,
      central_detail = r$central$detail,
      signature = r$signature,
      latex_summary = r$latex_summary,
      plot_names = names(r$plot_data)
    )
  }
  baseline <- bias_probe(run_report())
  expect_identical(
    baseline$names,
    c("structural", "central", "signature", "latex_summary", "plot_data")
  )
  expect_true(baseline$structural_verdict %in% c("no_evidence", "evidence"))
  expect_true(baseline$central_verdict %in% c("no_evidence", "evidence"))
  # T6 (signature test) is deferred crate-side -- always NULL today (see this
  # file's header); still the field the `out.set_value(2, signature_sexp)` /
  # `out.set_name(2, "signature")` split attaches on every call.
  expect_null(baseline$signature)
  expect_false(grepl("NaN", baseline$latex_summary, fixed = TRUE))

  gctorture(TRUE)
  on.exit(gctorture(FALSE))

  for (rep in seq_len(4)) {
    expect_identical(bias_probe(run_report()), baseline)
  }
})

# ---- problems.rs: sz_tsp_load's `known_optimum` field ----------------------
#
# `known_optimum` is the field whose pre-fix bare-`Sexp` window was directly
# observable from R: under torture it intermittently came back holding
# `n_cities`'s value (52) or `name`'s value ("berlin52"). The bundled
# expectation below pins all four fields' exact values, so any such slot
# reuse fails the rep it happens in.

test_that("repeated sz_tsp_load runs complete correctly under gctorture(TRUE)", {
  gctorture(TRUE)
  on.exit(gctorture(FALSE))

  for (rep in seq_len(8)) {
    t <- sz_tsp_load("berlin52")
    expect_identical(
      list(names(t), t$name, t$n_cities, dim(t$coords), t$known_optimum),
      list(c("name", "n_cities", "coords", "known_optimum"), "berlin52", 52,
           c(52L, 2L), 7542)
    )
  }
})
