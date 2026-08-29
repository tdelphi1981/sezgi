use savvy::{savvy, savvy_err, OwnedIntegerSexp, OwnedListSexp, OwnedRealSexp, OwnedStringSexp, Sexp};
use sezgi_bench::{
    run_experiment_parallel, run_experiment_sequential, run_experiment_with_checkpoint,
    ExperimentSpec, RunRecord,
};
use std::path::Path;

/// Builds the `data.frame` returned to R: one row per [`RunRecord`], columns
/// `algo, fid, dim, instance, seed, budget, best_f, f_opt, evals` (same
/// fields as py-sezgi's `run_experiment` record dicts, minus `wall_secs`).
fn records_to_data_frame(records: &[RunRecord]) -> savvy::Result<Sexp> {
    let n = records.len();

    let mut algo = OwnedStringSexp::new(n)?;
    let mut fid = OwnedRealSexp::new(n)?;
    let mut dim = OwnedRealSexp::new(n)?;
    let mut instance = OwnedRealSexp::new(n)?;
    let mut seed = OwnedRealSexp::new(n)?;
    let mut budget = OwnedRealSexp::new(n)?;
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
        best_f.set_elt(i, r.best_f)?;
        f_opt.set_elt(i, r.f_opt)?;
        evals.set_elt(i, r.evals_used as f64)?;
    }

    let mut out = OwnedListSexp::new(9, true)?;
    out.set_name_and_value(0, "algo", algo)?;
    out.set_name_and_value(1, "fid", fid)?;
    out.set_name_and_value(2, "dim", dim)?;
    out.set_name_and_value(3, "instance", instance)?;
    out.set_name_and_value(4, "seed", seed)?;
    out.set_name_and_value(5, "budget", budget)?;
    out.set_name_and_value(6, "best_f", best_f)?;
    out.set_name_and_value(7, "f_opt", f_opt)?;
    out.set_name_and_value(8, "evals", evals)?;

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
/// @returns A data.frame with one row per run.
///
/// The journal's spec hash is computed by `run_experiment_with_checkpoint`
/// from `spec` (the already-parsed `ExperimentSpec`), not from the raw
/// `spec_toml` text, so whitespace/comment-only edits to `spec_toml` never
/// invalidate a journal -- see `crates/bench/src/checkpoint.rs`.
#[savvy]
fn sz_run_experiment_raw(
    spec_toml: &str,
    parallel: bool,
    journal: Option<&str>,
    threads: Option<i32>,
) -> savvy::Result<Sexp> {
    let spec = ExperimentSpec::from_toml(spec_toml).map_err(|e| savvy_err!("{e}"))?;
    let threads = match threads {
        Some(t) if t < 0 => return Err(savvy_err!("threads must be >= 0")),
        Some(t) => Some(t as usize),
        None => None,
    };

    let records = if let Some(journal_path) = journal {
        let path = Path::new(journal_path);
        run_experiment_with_checkpoint(&spec, path, parallel, threads)
    } else if parallel {
        run_experiment_parallel(&spec, threads)
    } else {
        run_experiment_sequential(&spec, |_| {})
    }
    .map_err(|e| savvy_err!("{e}"))?;

    records_to_data_frame(&records)
}
