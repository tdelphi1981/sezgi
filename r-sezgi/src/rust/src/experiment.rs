use savvy::{
    savvy, savvy_err, OwnedIntegerSexp, OwnedListSexp, OwnedRealSexp, OwnedStringSexp, RealSexp,
    Sexp,
};
use sezgi_bench::{
    ioh_records as bench_ioh_records, read_ioh_root, run_experiment_logged,
    run_experiment_parallel, run_experiment_sequential, run_experiment_with_checkpoint,
    ExperimentSpec, RunRecord,
};
use std::path::Path;

/// Builds the `data.frame` returned to R: one row per [`RunRecord`], columns
/// `algo, fid, dim, instance, seed, budget, suite, best_f, f_opt, evals`
/// (same fields as py-sezgi's `run_experiment` record dicts, minus
/// `wall_secs`, `gap`, and `evals_used` renamed to `evals`).
///
/// `suite` (M3-5 Task 1): always emitted -- BBOB runs get
/// `sezgi_bench::SUITE_BBOB` (`"sezgi-bbob"`), matching every record built
/// before this column existed. See `records_from_columns` in `stats.rs` for
/// the read side, and its `suite` parameter's default (a data.frame lacking
/// this column) for the backward-compat path.
fn records_to_data_frame(records: &[RunRecord]) -> savvy::Result<Sexp> {
    let n = records.len();

    let mut algo = OwnedStringSexp::new(n)?;
    let mut fid = OwnedRealSexp::new(n)?;
    let mut dim = OwnedRealSexp::new(n)?;
    let mut instance = OwnedRealSexp::new(n)?;
    let mut seed = OwnedRealSexp::new(n)?;
    let mut budget = OwnedRealSexp::new(n)?;
    let mut suite = OwnedStringSexp::new(n)?;
    let mut best_f = OwnedRealSexp::new(n)?;
    let mut f_opt = OwnedRealSexp::new(n)?;
    let mut evals = OwnedRealSexp::new(n)?;

    for (i, r) in records.iter().enumerate() {
        algo.set_elt(i, &r.key.algo)?;
        fid.set_elt(i, r.key.fid as f64)?;
        dim.set_elt(i, r.key.dim as f64)?;
        instance.set_elt(i, r.key.instance as f64)?;
        seed.set_elt(i, r.key.seed as f64)?;
        budget.set_elt(i, r.key.budget as f64)?;
        suite.set_elt(i, &r.key.suite)?;
        best_f.set_elt(i, r.best_f)?;
        f_opt.set_elt(i, r.f_opt)?;
        evals.set_elt(i, r.evals_used as f64)?;
    }

    let mut out = OwnedListSexp::new(10, true)?;
    out.set_name_and_value(0, "algo", algo)?;
    out.set_name_and_value(1, "fid", fid)?;
    out.set_name_and_value(2, "dim", dim)?;
    out.set_name_and_value(3, "instance", instance)?;
    out.set_name_and_value(4, "seed", seed)?;
    out.set_name_and_value(5, "budget", budget)?;
    out.set_name_and_value(6, "suite", suite)?;
    out.set_name_and_value(7, "best_f", best_f)?;
    out.set_name_and_value(8, "f_opt", f_opt)?;
    out.set_name_and_value(9, "evals", evals)?;

    out.set_class(["data.frame"])?;
    let row_names = OwnedIntegerSexp::try_from_iter(1..=(n as i32))?;
    out.set_attrib("row.names", row_names.into())?;

    Ok(out.into())
}

/// Runs an [`ExperimentSpec`] (parsed from `spec_toml`) and returns its
/// `RunRecord`s as a data.frame. This mirrors py-sezgi's `run_experiment`
/// exactly (see `py-sezgi/src/lib.rs`): `journal=None` runs via
/// `run_experiment_parallel`/`run_experiment_sequential` (chosen by
/// `parallel`); `journal=Some(path)` runs via `run_experiment_with_checkpoint`,
/// which resumes from -- and appends to -- an existing journal file at
/// `path`.
///
/// This is the raw savvy-generated binding; the public R entry point with
/// R-native defaults (`parallel = TRUE`) is the hand-written wrapper
/// `sz_run_experiment()` in `R/experiment.R`, which calls this function.
///
/// @param spec_toml Experiment spec as TOML.
/// @param parallel Whether to run in parallel via rayon.
/// @param journal Optional path to a checkpoint journal file.
/// @param threads Optional rayon thread-pool size. `usize` is not a
///   supported savvy scalar arg type, so this comes in as `i32` and is cast;
///   negative values are rejected explicitly (savvy has no unsigned integer
///   scalar type to enforce this at the signature level).
/// @param log_dir Optional directory to log every run this call actually
///   EXECUTES in IOH-profiler format (via `run_experiment_logged` when
///   there is no `journal`, or via `run_experiment_with_checkpoint`'s own
///   `log_dir` pass-through when there is -- see that function's doc
///   comment: a run resumed from the journal was executed in a PRIOR
///   process and is never re-logged).
/// @returns A data.frame with one row per run.
///
/// The journal's spec hash is computed by `run_experiment_with_checkpoint`
/// from `spec` (the already-parsed `ExperimentSpec`), not from the raw
/// `spec_toml` text, so whitespace/comment-only edits to `spec_toml` never
/// invalidate a journal -- see `crates/bench/src/checkpoint.rs`.
///
/// @noRd
#[savvy]
fn sz_run_experiment_raw(
    spec_toml: &str,
    parallel: bool,
    journal: Option<&str>,
    threads: Option<i32>,
    log_dir: Option<&str>,
) -> savvy::Result<Sexp> {
    let spec = ExperimentSpec::from_toml(spec_toml).map_err(|e| savvy_err!("{e}"))?;
    let threads = match threads {
        Some(t) if t < 0 => return Err(savvy_err!("threads must be >= 0")),
        Some(t) => Some(t as usize),
        None => None,
    };
    let log_dir_path = log_dir.map(Path::new);

    let records = if let Some(journal_path) = journal {
        let path = Path::new(journal_path);
        run_experiment_with_checkpoint(&spec, path, parallel, threads, log_dir_path)
    } else if let Some(dir) = log_dir_path {
        run_experiment_logged(&spec, dir, parallel, threads).map(|(r, _)| r)
    } else if parallel {
        run_experiment_parallel(&spec, threads)
    } else {
        run_experiment_sequential(&spec, |_| {})
    }
    .map_err(|e| savvy_err!("{e}"))?;

    records_to_data_frame(&records)
}

/// Casts a non-negative-checked, WHOLE-NUMBER-checked `f64` (as passed
/// from R, which has no native unsigned integer type) to `u64`, rejecting
/// negative, non-finite, or fractional values. Duplicated from the
/// identically-named private helper in `stats.rs` / `session.rs` (no
/// shared private cross-module import -- same "no shared private crate
/// imports" rule as `parse_distribution`'s duplication between `solve.rs`
/// and py-sezgi's `lib.rs`).
///
/// M2d-3: previously silently truncated a fractional `x` toward zero; now
/// rejects it (see `session.rs`'s copy of this doc comment for the
/// py-sezgi asymmetry this closes).
fn f64_to_u64(name: &str, x: f64) -> savvy::Result<u64> {
    if !x.is_finite() || x < 0.0 {
        return Err(savvy_err!(
            "{} must be a non-negative finite number, got {}",
            name,
            x
        ));
    }
    if x.fract() != 0.0 {
        return Err(savvy_err!("{} expected a whole number, got {}", name, x));
    }
    Ok(x as u64)
}

/// Reconstructs `RunRecord`s from an on-disk IOH archive at `log_root` (as
/// written by `sz_run_experiment(..., log_dir = ...)`), one record per
/// `(run, budget)` pair -- see `sezgi_bench::ioh_records`'s doc comment for
/// the exact `best_f`/`evals_used` semantics and the curtailed-view-vs-
/// independent-run distinction for budgets smaller than a run's logged
/// budget.
///
/// Returns the SAME data.frame shape `sz_run_experiment()` returns (via
/// [`records_to_data_frame`]), so `sz_results_matrix()`/
/// `sz_per_budget_packages()` accept it unchanged.
///
/// @param log_root Path to the IOH archive directory (as passed to
///   `sz_run_experiment(..., log_dir = ...)`).
/// @param budgets Numeric vector of evaluation budgets to reconstruct
///   records at.
/// @returns A data.frame with the same columns as `sz_run_experiment()`.
/// @export
#[savvy]
fn sz_read_ioh_records(log_root: &str, budgets: RealSexp) -> savvy::Result<Sexp> {
    let budgets_u: Vec<u64> = budgets
        .as_slice()
        .iter()
        .map(|&b| f64_to_u64("budgets", b))
        .collect::<savvy::Result<Vec<u64>>>()?;

    let scenarios = read_ioh_root(Path::new(log_root)).map_err(|e| savvy_err!("{e}"))?;
    let records = bench_ioh_records(&scenarios, &budgets_u).map_err(|e| savvy_err!("{e}"))?;

    records_to_data_frame(&records)
}
