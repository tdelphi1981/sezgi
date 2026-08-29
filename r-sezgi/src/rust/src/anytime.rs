// M2d-2 Task 8: R bindings for the analysis cluster's anytime-performance
// and COCO-export functions (1:1 mirror of py-sezgi's Task 7 additions --
// see `py-sezgi/src/lib.rs`'s `ecdf`/`coco_export`).

use savvy::{
    savvy, savvy_err, OwnedListSexp, OwnedRealSexp, OwnedStringSexp, RealSexp, Sexp,
};
use sezgi_bench::{
    coco_export as bench_coco_export, default_targets as bench_default_targets,
    ecdf as bench_ecdf, ecdf_per_algo as bench_ecdf_per_algo, read_ioh_root, EcdfCurve,
};
use std::path::Path;

/// Builds a `list(evals = ..., proportion = ...)` from an [`EcdfCurve`].
/// `evals` (`Vec<u64>` in Rust) comes back as an R double vector -- R has
/// no native 64-bit integer type, and evaluation counts are always small
/// enough (well under 2^53) for this to lose no precision.
fn ecdf_curve_to_list(curve: &EcdfCurve) -> savvy::Result<OwnedListSexp> {
    let evals_f: Vec<f64> = curve.evals.iter().map(|&e| e as f64).collect();
    let mut out = OwnedListSexp::new(2, true)?;
    out.set_name_and_value(0, "evals", OwnedRealSexp::try_from_slice(evals_f.as_slice())?)?;
    out.set_name_and_value(
        1,
        "proportion",
        OwnedRealSexp::try_from_slice(curve.proportion.as_slice())?,
    )?;
    Ok(out)
}

/// ECDF/anytime curve(s) over the IOH archive at `log_root` -- see
/// `sezgi_bench::ecdf`/`sezgi_bench::ecdf_per_algo`.
///
/// This is the raw savvy-generated binding (required args only; savvy has
/// no way to express a non-`NULL` default for a required argument in the
/// generated signature -- same raw/wrapper pattern as
/// `sz_run_experiment()`). The public R entry point with R-native defaults
/// (`per_algo = TRUE`) is the hand-written wrapper `sz_ecdf()` in
/// `R/anytime.R`, which calls this function.
///
/// @param log_root Path to the IOH archive directory (as passed to
///   `sz_run_experiment(..., log_dir = ...)`).
/// @param per_algo `TRUE` returns a named list, one `list(evals=,
///   proportion=)` entry per distinct algo in the archive, named by algo
///   (first-appearance order); `FALSE` returns a single pooled
///   `list(evals=, proportion=)` over every scenario.
/// @param targets Optional numeric vector of precision targets; `NULL`
///   uses `sezgi_bench::default_targets` (the COCO-convention 51-value
///   set).
/// @returns A named list (see `per_algo`).
#[savvy]
fn sz_ecdf_raw(log_root: &str, per_algo: bool, targets: Option<RealSexp>) -> savvy::Result<Sexp> {
    let scenarios = read_ioh_root(Path::new(log_root)).map_err(|e| savvy_err!("{e}"))?;
    let targets_v: Vec<f64> = match targets {
        Some(t) => t.as_slice().to_vec(),
        None => bench_default_targets(),
    };

    if per_algo {
        let curves = bench_ecdf_per_algo(&scenarios, &targets_v).map_err(|e| savvy_err!("{e}"))?;
        let mut out = OwnedListSexp::new(curves.len(), true)?;
        for (idx, (algo, curve)) in curves.iter().enumerate() {
            out.set_name_and_value(idx, algo.as_str(), ecdf_curve_to_list(curve)?)?;
        }
        Ok(out.into())
    } else {
        let curve = bench_ecdf(&scenarios, &targets_v).map_err(|e| savvy_err!("{e}"))?;
        Ok(ecdf_curve_to_list(&curve)?.into())
    }
}

/// Exports the IOH archive at `log_root` as a COCO/BBOB "old format"
/// archive rooted at `out_dir` -- see `sezgi_bench::coco_export`. Returns
/// the list of written file paths (as strings), sorted for determinism.
///
/// @param log_root Path to the IOH archive directory (as passed to
///   `sz_run_experiment(..., log_dir = ...)`).
/// @param out_dir Directory to write the COCO/BBOB archive under.
/// @returns A character vector of written file paths, sorted.
/// @export
#[savvy]
fn sz_coco_export(log_root: &str, out_dir: &str) -> savvy::Result<Sexp> {
    let scenarios = read_ioh_root(Path::new(log_root)).map_err(|e| savvy_err!("{e}"))?;
    let written = bench_coco_export(&scenarios, Path::new(out_dir)).map_err(|e| savvy_err!("{e}"))?;
    let paths: Vec<String> = written.iter().map(|p| p.display().to_string()).collect();
    let out = OwnedStringSexp::try_from(paths.as_slice())?;
    Ok(out.into())
}
