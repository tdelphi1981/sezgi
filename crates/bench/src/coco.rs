//! COCO/BBOB "old format" archive export from [`IohScenario`]/[`IohRun`]
//! logs (M2d-2 Task 4), so a sezgi experiment can be post-processed with
//! `cocopp` (the COCO postprocessing tool) alongside archives produced by
//! `cocoex`-based experiments.
//!
//! ## Layout produced
//!
//! For each distinct `(algo, fid)` pair found in the input scenarios,
//! [`coco_export`] writes:
//! - `<out>/<algo>/bbobexp_f<fid>.info` — one header/comment/data-line
//!   TRIPLET per distinct `dim` present for that `(algo, fid)` (the
//!   reference format's convention for a multi-dimension experiment: all
//!   dims of one function share a single `.info` file).
//! - `<out>/<algo>/data_f<fid>/bbobexp_f<fid>_DIM<dim>.dat` — one file per
//!   dim, holding one run BLOCK per run (each block starting with the
//!   repeated `%`-comment header line), one row per improvement record in
//!   [`IohRun::rows`].
//!
//! ## `// sezgi simplification:` tags
//!
//! Every place this module's output diverges from what the reference COCO
//! C/Python postprocessing tool would itself emit is tagged inline with
//! `// sezgi simplification:`, per the sezgi-bbob precedent
//! (`crates/problems`). The two load-bearing ones:
//! - **3 `.dat` columns, not 5.** The reference noiseless-BBOB `.dat` has 5
//!   columns (evals, noise-free current f - Fopt, noise-free best f - Fopt,
//!   plus 2 columns that only matter for the *noisy* BBOB testbed). sezgi
//!   only ever tracks the noiseless testbed and only records BEST-SO-FAR
//!   values ([`IohRun::rows`] is exactly the improvement sequence — see
//!   `crate::ioh_read`'s module doc), so "current f" and "best f" are
//!   identically the same value here; we emit 3 columns and drop the two
//!   noisy-testbed columns outright rather than duplicating a column that
//!   would carry no information.
//! - **No restarts / no independent-restart bookkeeping.** The reference
//!   format supports multiple "restart" sub-blocks per run entry
//!   (`maxevals` bookkeeping for restarted algorithms); sezgi's engine has
//!   no restart mechanism, so every run is exactly one block.
//!
//! `Precision = 1.000e-08` in the `.info` header is likewise a fixed,
//! non-configurable literal (sezgi simplification: the reference tool lets
//! this vary per experiment; every sezgi export pins it since sezgi has no
//! analogous per-experiment precision setting).

use crate::experiment::ExperimentError;
use crate::ioh_read::IohScenario;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn coco_err(msg: impl std::fmt::Display) -> ExperimentError {
    ExperimentError::CocoExport(msg.to_string())
}

/// Formats `value` in scientific notation with `decimals` digits after the
/// point and an explicitly-signed, zero-padded (minimum 2 digits) exponent
/// — i.e. C's `%.<decimals>e`, which is what the reference COCO tooling
/// emits (e.g. `1.000e-08`, `3.5e-02`, `1.2e+01`).
///
/// Rust's built-in `{:e}` formatter is close but not identical: it omits
/// the `+` sign on non-negative exponents and never zero-pads them (e.g.
/// `format!("{:.1e}", 0.035)` is `"3.5e-2"`, not `"3.5e-02"`). This helper
/// post-processes the exponent to match the C convention the reference
/// format uses everywhere.
fn fmt_exp(value: f64, decimals: usize) -> String {
    let s = format!("{value:.decimals$e}");
    let (mantissa, exp) = s.split_once('e').expect("scientific format always contains 'e'");
    let (sign, digits) = match exp.strip_prefix('-') {
        Some(d) => ('-', d),
        None => ('+', exp.strip_prefix('+').unwrap_or(exp)),
    };
    format!("{mantissa}e{sign}{digits:0>2}")
}

/// Exports `scenarios` as a COCO/BBOB "old format" archive rooted at
/// `out_dir` (created if absent), returning every file path written,
/// sorted for determinism.
///
/// Every run in `scenarios` MUST carry `f_opt` (see [`IohRun::f_opt`]) and
/// at least one row in [`IohRun::rows`] — a legacy log missing either is an
/// error naming the offending run (algo/fid/dim/instance) BEFORE any file
/// is written (validation is a full up-front pass, so a rejected export
/// never leaves a partial archive on disk).
///
/// Scenarios are grouped by `(algo, fid)`: every dim present for that pair
/// shares one `.info` file, with DIM blocks emitted in ascending `dim`
/// order regardless of `scenarios`' input order (matching
/// `crate::ioh_read`'s existing "sort for deterministic output" convention
/// — directory iteration / caller-supplied order is not something this
/// module's output should depend on).
pub fn coco_export(scenarios: &[IohScenario], out_dir: &Path) -> Result<Vec<PathBuf>, ExperimentError> {
    // ---- Up-front validation pass: every run needs f_opt + >=1 row. ----
    for sc in scenarios {
        for run in &sc.runs {
            if run.f_opt.is_none() {
                return Err(coco_err(format!(
                    "run instance {} (algo `{}`, fid {}, dim {}) has no f_opt \
                     (legacy log?) — COCO export requires f_opt on every run",
                    run.instance, sc.algo, sc.fid, sc.dim
                )));
            }
            if run.rows.is_empty() {
                return Err(coco_err(format!(
                    "run instance {} (algo `{}`, fid {}, dim {}) has no data rows \
                     — COCO export requires at least one row per run",
                    run.instance, sc.algo, sc.fid, sc.dim
                )));
            }
        }
    }

    // ---- Group by (algo, fid); collect every dim's scenario. ----
    let mut groups: BTreeMap<(String, u32), Vec<&IohScenario>> = BTreeMap::new();
    for sc in scenarios {
        groups.entry((sc.algo.clone(), sc.fid)).or_default().push(sc);
    }

    let mut written = Vec::new();
    for ((algo, fid), mut scs) in groups {
        scs.sort_by_key(|s| s.dim);

        let algo_dir = out_dir.join(&algo);
        let data_dir_name = format!("data_f{fid}");
        std::fs::create_dir_all(algo_dir.join(&data_dir_name)).map_err(coco_err)?;

        let mut info = String::new();
        for sc in &scs {
            let dat_rel = format!("{data_dir_name}/bbobexp_f{fid}_DIM{}.dat", sc.dim);
            let dat_path = algo_dir.join(&dat_rel);

            info.push_str(&format!(
                "suite = 'bbob', funcId = {fid}, DIM = {}, Precision = 1.000e-08, algId = '{algo}'\n",
                sc.dim
            ));
            info.push_str("% exported by sezgi\n");
            info.push_str(&dat_rel);

            let mut dat = String::new();
            let mut tokens: Vec<String> = Vec::with_capacity(sc.runs.len());
            for run in &sc.runs {
                // Validated above: f_opt is Some and rows is non-empty.
                let f_opt = run.f_opt.expect("validated non-None above");

                dat.push_str(&format!(
                    "% function evaluation | noise-free fitness - Fopt ({}) | best noise-free fitness - Fopt\n",
                    fmt_exp(f_opt, 15)
                ));
                for &(evals, raw_y) in &run.rows {
                    let gap = raw_y - f_opt;
                    // sezgi simplification: reference emits 5 columns
                    // (current + best noise-free fitness, plus 2
                    // noisy-testbed-only columns); sezgi only records
                    // best-so-far values (see module doc), so "current"
                    // and "best" gap are identically the same number here
                    // — we emit 3 columns total (evals, gap, gap) and drop
                    // the two noisy-testbed columns rather than
                    // duplicating information that carries no signal.
                    dat.push_str(&format!("{evals} {} {}\n", fmt_exp(gap, 9), fmt_exp(gap, 9)));
                }

                let (_, final_raw_y) = *run.rows.last().expect("validated non-empty above");
                let final_gap = final_raw_y - f_opt;
                tokens.push(format!("{}:{}|{}", run.instance, run.evals, fmt_exp(final_gap, 1)));
            }
            std::fs::write(&dat_path, dat).map_err(coco_err)?;
            written.push(dat_path);

            info.push_str(", ");
            info.push_str(&tokens.join(", "));
            info.push('\n');
        }

        let info_path = algo_dir.join(format!("bbobexp_f{fid}.info"));
        std::fs::write(&info_path, info).map_err(coco_err)?;
        written.push(info_path);
    }

    written.sort();
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiment::{
        AlgoEntry, AlgoSource, ExperimentSpec, ProblemEntry, run_experiment_logged,
    };
    use crate::ioh_read::{IohRun, read_ioh_root};

    fn run(instance: u32, f_opt: f64, rows: Vec<(u64, f64)>, evals: u64) -> IohRun {
        IohRun { instance, seed: Some(1), f_opt: Some(f_opt), rows, evals }
    }

    fn scenario(algo: &str, fid: u32, dim: usize, runs: Vec<IohRun>) -> IohScenario {
        IohScenario {
            algo: algo.into(),
            suite: "bbob".into(),
            fid,
            fname: "Sphere".into(),
            dim,
            runs,
        }
    }

    #[test]
    fn fmt_exp_matches_c_style_scientific_notation() {
        assert_eq!(fmt_exp(1e-8, 3), "1.000e-08");
        assert_eq!(fmt_exp(0.035, 1), "3.5e-02");
        assert_eq!(fmt_exp(12.0, 1), "1.2e+01");
        assert_eq!(fmt_exp(0.0, 9), "0.000000000e+00");
    }

    #[test]
    fn single_dim_info_first_line_and_run_summary_tokens_exact() {
        // f_opt = 10.0; run 0's final gap = 12.0 - 10.0 = 2.0 -> "2.0e+00";
        // run 1's final gap = 10.035 - 10.0 = 0.035 -> "3.5e-02".
        let scenarios = vec![scenario(
            "de",
            1,
            5,
            vec![
                run(1, 10.0, vec![(50, 20.0), (100, 12.0)], 200),
                run(2, 10.0, vec![(30, 15.0), (150, 10.035)], 300),
            ],
        )];

        let tmp = tempfile::tempdir().unwrap();
        let written = coco_export(&scenarios, tmp.path()).unwrap();
        assert!(!written.is_empty());

        let info_path = tmp.path().join("de/bbobexp_f1.info");
        assert!(written.contains(&info_path));
        let info = std::fs::read_to_string(&info_path).unwrap();
        let mut lines = info.lines();
        assert_eq!(
            lines.next().unwrap(),
            "suite = 'bbob', funcId = 1, DIM = 5, Precision = 1.000e-08, algId = 'de'"
        );
        assert_eq!(lines.next().unwrap(), "% exported by sezgi");
        assert_eq!(
            lines.next().unwrap(),
            "data_f1/bbobexp_f1_DIM5.dat, 1:200|2.0e+00, 2:300|3.5e-02"
        );
        assert_eq!(lines.next(), None);
    }

    #[test]
    fn dat_rows_equal_raw_y_minus_f_opt_with_repeated_header_per_block() {
        let scenarios = vec![scenario(
            "de",
            1,
            5,
            vec![
                run(1, 10.0, vec![(50, 20.0), (100, 12.0)], 200),
                run(2, 10.0, vec![(30, 15.0)], 300),
            ],
        )];

        let tmp = tempfile::tempdir().unwrap();
        coco_export(&scenarios, tmp.path()).unwrap();

        let dat = std::fs::read_to_string(tmp.path().join("de/data_f1/bbobexp_f1_DIM5.dat")).unwrap();
        let header = "% function evaluation | noise-free fitness - Fopt (1.000000000000000e+01) | best noise-free fitness - Fopt";
        let expected = format!(
            "{header}\n50 1.000000000e+01 1.000000000e+01\n100 2.000000000e+00 2.000000000e+00\n\
             {header}\n30 5.000000000e+00 5.000000000e+00\n"
        );
        assert_eq!(dat, expected);
    }

    #[test]
    fn multi_dim_same_fid_shares_one_info_with_one_line_per_dim() {
        let scenarios = vec![
            scenario("de", 1, 10, vec![run(1, 0.0, vec![(10, 1.0)], 10)]),
            scenario("de", 1, 5, vec![run(1, 0.0, vec![(10, 1.0)], 10)]),
        ];

        let tmp = tempfile::tempdir().unwrap();
        let written = coco_export(&scenarios, tmp.path()).unwrap();

        let info_path = tmp.path().join("de/bbobexp_f1.info");
        let info = std::fs::read_to_string(&info_path).unwrap();
        let lines: Vec<&str> = info.lines().collect();
        assert_eq!(lines.len(), 6, "two DIM blocks x 3 lines each: {info}");
        // Ascending dim order regardless of input order.
        assert!(lines[0].contains("DIM = 5"));
        assert!(lines[3].contains("DIM = 10"));

        assert!(written.contains(&tmp.path().join("de/data_f1/bbobexp_f1_DIM5.dat")));
        assert!(written.contains(&tmp.path().join("de/data_f1/bbobexp_f1_DIM10.dat")));
    }

    #[test]
    fn distinct_fids_get_separate_info_files() {
        let scenarios = vec![
            scenario("de", 1, 5, vec![run(1, 0.0, vec![(10, 1.0)], 10)]),
            scenario("de", 2, 5, vec![run(1, 0.0, vec![(10, 1.0)], 10)]),
        ];

        let tmp = tempfile::tempdir().unwrap();
        let written = coco_export(&scenarios, tmp.path()).unwrap();
        assert!(written.contains(&tmp.path().join("de/bbobexp_f1.info")));
        assert!(written.contains(&tmp.path().join("de/bbobexp_f2.info")));
    }

    #[test]
    fn missing_f_opt_errors_naming_the_run_before_writing_anything() {
        let scenarios = vec![scenario(
            "de",
            1,
            5,
            vec![IohRun { instance: 3, seed: None, f_opt: None, rows: vec![(1, 5.0)], evals: 1 }],
        )];

        let tmp = tempfile::tempdir().unwrap();
        let err = coco_export(&scenarios, tmp.path()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("instance 3"), "{msg}");
        assert!(msg.contains("fid 1"), "{msg}");
        assert!(msg.contains("dim 5"), "{msg}");
        assert!(!tmp.path().join("de").exists(), "no partial output on validation failure");
    }

    #[test]
    fn empty_rows_errors_naming_the_run() {
        let scenarios = vec![scenario(
            "de",
            1,
            5,
            vec![IohRun { instance: 4, seed: Some(1), f_opt: Some(0.0), rows: vec![], evals: 0 }],
        )];

        let tmp = tempfile::tempdir().unwrap();
        let err = coco_export(&scenarios, tmp.path()).unwrap_err();
        assert!(err.to_string().contains("instance 4"));
    }

    #[test]
    fn end_to_end_smoke_from_real_run_experiment_logged() {
        let spec = ExperimentSpec {
            name: "coco-smoke".into(),
            seeds: vec![11, 22],
            budgets: vec![200],
            algorithms: vec![AlgoEntry {
                name: "de".into(),
                source: AlgoSource::Preset { kind: "de_rand_1".into(), pop_size: Some(6) },
            }],
            problems: vec![ProblemEntry { suite: "bbob".into(), fid: 1, dim: 5, instances: vec![1] }],
        };
        let log_dir = tempfile::tempdir().unwrap();
        run_experiment_logged(&spec, log_dir.path(), false, None).unwrap();
        let scenarios = read_ioh_root(log_dir.path()).unwrap();

        let out_dir = tempfile::tempdir().unwrap();
        let written = coco_export(&scenarios, out_dir.path()).unwrap();
        assert!(!written.is_empty());
        for p in &written {
            assert!(p.exists());
        }
        assert!(out_dir.path().join("de/bbobexp_f1.info").exists());
        assert!(out_dir.path().join("de/data_f1/bbobexp_f1_DIM5.dat").exists());
    }
}
