//! IOH log reader (M2d Task 2): the read-side counterpart to
//! [`crate::ioh::IohLogger::finish`].
//!
//! ## On-disk layout consumed
//!
//! For each algo directory `<root>/<algo>/`:
//! - `IOHprofiler_f<fid>_<fname>.json` — a meta file. Its top-level
//!   `algorithm.name`, `suite`, `function_id`, `function_name` apply to
//!   every entry in its `scenarios` array; each scenario carries
//!   `dimension`, a `path` (relative to the `<algo>` directory) pointing
//!   at the `.dat` file, and a `runs` array — one entry per completed run,
//!   with `instance`, `evals`, and optional `seed`/`f_opt` (absent on
//!   legacy pre-M2d logs, which used [`crate::ioh::IohLogger::start_run`]
//!   rather than `start_run_with`).
//! - the referenced `.dat` file: one or more run blocks, each starting
//!   with the literal header line `"evaluations" "raw_y"`, followed by
//!   whitespace-separated `<eval> <raw_y>` rows (`eval` a `u64`, `raw_y`
//!   an `f64` parsed via `str::parse`, which — like the `serde_json`
//!   `float_roundtrip` feature this workspace enables for writing — is a
//!   correctly-rounded parse).
//!
//! ## Format contract: run pairing by ORDER
//!
//! A scenario's `.dat` file holds one run block per completed run, in the
//! SAME order `finish()` wrote that scenario's `runs` array — i.e. the
//! order `start_run`/`start_run_with` was called on the logger. There is
//! no run id, seed, or instance number embedded in the `.dat` file itself
//! that could be used to re-associate a block with its meta entry after
//! the fact: [`read_ioh_root`] pairs the Nth `.dat` run block with the Nth
//! entry of that scenario's meta `runs` array POSITIONALLY, and nothing
//! else. A `.dat` file whose block count does not match its scenario's
//! `runs` array length is treated as corrupt (an error naming the file).
//!
//! Note: [`crate::ioh::IohLogger::finish`] currently names the meta file
//! from `(fid, fname)` alone, not `dim`, so two `finish()` calls for the
//! same `(algo, fid)` at different `dim`s target the same meta path and
//! the later call's write overwrites the earlier one's `scenarios` entry
//! (the `.dat` files, whose names DO embed `dim`, are unaffected). This
//! reader has no such limitation — it reads whatever `scenarios` entries
//! a meta file actually contains — but a caller relying on multiple dims
//! of the same `(algo, fid)` surviving on disk needs that write-side
//! collision fixed first.

use crate::experiment::ExperimentError;
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// One run within an [`IohScenario`], reconstructed from a `.dat` run
/// block paired positionally with its meta `runs[]` entry.
#[derive(Debug, Clone, PartialEq)]
pub struct IohRun {
    pub instance: u32,
    /// `None` for a legacy log whose meta entry omitted `seed` (written by
    /// [`crate::ioh::IohLogger::start_run`], not `start_run_with`).
    pub seed: Option<u64>,
    /// `None` for a legacy log whose meta entry omitted `f_opt`.
    pub f_opt: Option<f64>,
    /// Improvement rows (plus the always-written final row) exactly as
    /// parsed from the `.dat` run block, in file order.
    pub rows: Vec<(u64, f64)>,
    pub evals: u64,
}

/// One `(algo, fid, dim)` scenario: the meta file's `algorithm`/`suite`/
/// `function_id`/`function_name` plus one `scenarios[]` entry's
/// `dimension` and its runs.
#[derive(Debug, Clone, PartialEq)]
pub struct IohScenario {
    pub algo: String,
    pub suite: String,
    pub fid: u32,
    pub fname: String,
    pub dim: usize,
    pub runs: Vec<IohRun>,
}

// ---------------------------------------------------------------------
// Meta JSON shape (mirrors the object literal in `IohLogger::finish`)
// ---------------------------------------------------------------------

#[derive(Deserialize)]
struct MetaFile {
    suite: String,
    function_id: u32,
    function_name: String,
    algorithm: MetaAlgorithm,
    scenarios: Vec<MetaScenario>,
}

#[derive(Deserialize)]
struct MetaAlgorithm {
    name: String,
}

#[derive(Deserialize)]
struct MetaScenario {
    dimension: usize,
    path: String,
    runs: Vec<MetaRun>,
}

#[derive(Deserialize)]
struct MetaRun {
    instance: u32,
    evals: u64,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    f_opt: Option<f64>,
    // `best` (evals/y) is written by `finish()` but not needed by any
    // `IohRun` field; left undeclared so serde ignores it.
}

const DAT_HEADER: &str = "\"evaluations\" \"raw_y\"";

fn read_err(path: &Path, msg: impl std::fmt::Display) -> ExperimentError {
    ExperimentError::IohRead(format!("{}: {msg}", path.display()))
}

fn read_meta_file(path: &Path) -> Result<MetaFile, ExperimentError> {
    let text = std::fs::read_to_string(path).map_err(|e| read_err(path, e))?;
    serde_json::from_str(&text).map_err(|e| read_err(path, e))
}

/// Splits a `.dat` file's text into run blocks (one `Vec<(eval, raw_y)>`
/// per block), split on the literal [`DAT_HEADER`] line. Any row that is
/// not exactly two whitespace-separated tokens parsing as `u64`/`f64` -
/// including a torn or otherwise corrupt final line - is an error naming
/// `path`.
fn split_dat_blocks(path: &Path, text: &str) -> Result<Vec<Vec<(u64, f64)>>, ExperimentError> {
    let mut blocks: Vec<Vec<(u64, f64)>> = Vec::new();
    for line in text.lines() {
        let line = line.trim_end_matches('\r');
        if line == DAT_HEADER {
            blocks.push(Vec::new());
            continue;
        }
        let block = blocks.last_mut().ok_or_else(|| {
            read_err(path, format_args!("data row {line:?} before any header line"))
        })?;
        let mut tokens = line.split_whitespace();
        let (Some(ev_tok), Some(y_tok), None) = (tokens.next(), tokens.next(), tokens.next())
        else {
            return Err(read_err(
                path,
                format_args!("malformed .dat row {line:?} (torn or corrupt final line?)"),
            ));
        };
        let ev: u64 = ev_tok.parse().map_err(|_| {
            read_err(path, format_args!("malformed .dat row {line:?} (torn or corrupt final line?)"))
        })?;
        let y: f64 = y_tok.parse().map_err(|_| {
            read_err(path, format_args!("malformed .dat row {line:?} (torn or corrupt final line?)"))
        })?;
        block.push((ev, y));
    }
    Ok(blocks)
}

/// Lists a directory's immediate entries matching `keep`, sorted by path
/// for deterministic output (directory iteration order is otherwise
/// platform-dependent).
fn list_sorted(dir: &Path, keep: impl Fn(&std::fs::DirEntry) -> bool) -> Result<Vec<PathBuf>, ExperimentError> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| read_err(dir, e))?
        .filter_map(|e| e.ok())
        .filter(keep)
        .map(|e| e.path())
        .collect();
    paths.sort();
    Ok(paths)
}

/// Walks `root` for `<algo>/IOHprofiler_f*.json` meta files (one level of
/// algo subdirectories, per the layout `IohLogger::finish` writes),
/// parses each one, resolves and parses its scenarios' `.dat` files, and
/// returns one [`IohScenario`] per `scenarios[]` entry found — see the
/// module doc for the exact layout and the run-pairing contract.
pub fn read_ioh_root(root: &Path) -> Result<Vec<IohScenario>, ExperimentError> {
    let algo_dirs = list_sorted(root, |e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))?;

    let mut scenarios = Vec::new();
    for algo_dir in algo_dirs {
        let json_paths = list_sorted(&algo_dir, |e| {
            e.file_type().map(|t| t.is_file()).unwrap_or(false)
                && e.file_name()
                    .to_str()
                    .is_some_and(|n| n.starts_with("IOHprofiler_f") && n.ends_with(".json"))
        })?;

        for meta_path in json_paths {
            let meta = read_meta_file(&meta_path)?;
            for sc in &meta.scenarios {
                let dat_path = algo_dir.join(&sc.path);
                let text = std::fs::read_to_string(&dat_path).map_err(|e| read_err(&dat_path, e))?;
                let blocks = split_dat_blocks(&dat_path, &text)?;
                if blocks.len() != sc.runs.len() {
                    return Err(read_err(
                        &dat_path,
                        format_args!(
                            "{} run block(s) but meta {} lists {} run(s)",
                            blocks.len(),
                            meta_path.display(),
                            sc.runs.len()
                        ),
                    ));
                }
                let runs = sc
                    .runs
                    .iter()
                    .zip(blocks)
                    .map(|(mr, rows)| IohRun {
                        instance: mr.instance,
                        seed: mr.seed,
                        f_opt: mr.f_opt,
                        rows,
                        evals: mr.evals,
                    })
                    .collect();
                scenarios.push(IohScenario {
                    algo: meta.algorithm.name.clone(),
                    suite: meta.suite.clone(),
                    fid: meta.function_id,
                    fname: meta.function_name.clone(),
                    dim: sc.dimension,
                    runs,
                });
            }
        }
    }
    Ok(scenarios)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiment::{
        AlgoEntry, AlgoSource, ExperimentSpec, PlannedRun, ProblemEntry, enumerate, execute_run,
        run_experiment_logged,
    };
    use crate::ioh::IohLogger;
    use sezgi_components::register_builtins;
    use sezgi_core::component::Registry;
    use sezgi_core::problem::EvalObserver;
    use std::sync::{Arc, Mutex};

    fn demo_spec() -> ExperimentSpec {
        ExperimentSpec {
            name: "roundtrip".into(),
            seeds: vec![11, 22],
            budgets: vec![300],
            algorithms: vec![AlgoEntry {
                name: "de".into(),
                source: AlgoSource::Preset { kind: "de_rand_1".into(), pop_size: Some(6) },
            }],
            problems: vec![
                ProblemEntry { suite: "bbob".into(), fid: 1, dim: 5, instances: vec![1, 2] },
                ProblemEntry { suite: "bbob".into(), fid: 2, dim: 5, instances: vec![1] },
            ],
        }
    }

    /// Independently reproduces the exact filter `IohRunObserver::on_eval`
    /// applies (improvement rows + always-written final row), by observing
    /// a second, separately-driven execution of the same [`PlannedRun`].
    /// The engine is deterministic and the observer is a passive side
    /// channel (see `execute_run`'s doc), so this must match the rows
    /// `IohLogger` actually wrote, bit for bit.
    #[derive(Default)]
    struct CapturedData {
        prev_best: Option<f64>,
        rows: Vec<(u64, f64)>,
        last: Option<(u64, f64)>,
    }
    struct CapturingObserver(Arc<Mutex<CapturedData>>);
    impl EvalObserver for CapturingObserver {
        fn on_eval(&mut self, eval_index: u64, _f: f64, best_so_far: f64) {
            let mut d = self.0.lock().unwrap();
            d.last = Some((eval_index, best_so_far));
            let improved = d.prev_best.is_none_or(|p| best_so_far < p);
            if improved {
                d.rows.push((eval_index, best_so_far));
                d.prev_best = Some(best_so_far);
            }
        }
    }

    #[test]
    fn round_trip_matches_enumeration_and_reproduces_rows_bit_exactly() {
        let spec = demo_spec();
        let tmp = tempfile::tempdir().unwrap();
        run_experiment_logged(&spec, tmp.path(), false, None).unwrap();

        let scenarios = read_ioh_root(tmp.path()).unwrap();

        let planned = enumerate(&spec).unwrap();

        // scenario count = distinct (algo, fid, dim)
        let mut expected_groups: Vec<(String, u32, usize)> =
            planned.iter().map(|r| (r.key.algo.clone(), r.fid, r.dim)).collect();
        expected_groups.sort();
        expected_groups.dedup();
        assert_eq!(scenarios.len(), expected_groups.len());

        // Group planned runs by (algo, fid, dim) in first-seen order, same as
        // `build_ioh_observers` does — this is the order each scenario's
        // `.dat` run blocks / meta `runs[]` were written in.
        let mut groups: Vec<(String, u32, usize)> = Vec::new();
        let mut group_runs: Vec<Vec<&PlannedRun>> = Vec::new();
        for run in &planned {
            let key = (run.key.algo.clone(), run.fid, run.dim);
            let idx = match groups.iter().position(|g| *g == key) {
                Some(i) => i,
                None => {
                    groups.push(key);
                    group_runs.push(Vec::new());
                    groups.len() - 1
                }
            };
            group_runs[idx].push(run);
        }

        let mut reg = Registry::new();
        register_builtins(&mut reg);

        for (key, runs) in groups.iter().zip(&group_runs) {
            let sc = scenarios
                .iter()
                .find(|s| (&s.algo, s.fid, s.dim) == (&key.0, key.1, key.2))
                .expect("scenario present for every enumerated group");
            assert_eq!(sc.runs.len(), runs.len(), "run count for group {key:?}");

            for (planned_run, ioh_run) in runs.iter().zip(&sc.runs) {
                assert_eq!(ioh_run.instance, planned_run.instance);
                assert_eq!(ioh_run.seed, Some(planned_run.seed));

                let problem = sezgi_problems::BbobProblem::new(
                    planned_run.fid, planned_run.dim, planned_run.instance,
                ).unwrap();
                assert_eq!(ioh_run.f_opt.map(f64::to_bits), Some(problem.f_opt().to_bits()));

                let captured = Arc::new(Mutex::new(CapturedData::default()));
                execute_run(&reg, planned_run, Some(Box::new(CapturingObserver(captured.clone()))))
                    .unwrap();
                let d = captured.lock().unwrap();
                let mut expected_rows = d.rows.clone();
                if let Some(last) = d.last {
                    if expected_rows.last() != Some(&last) {
                        expected_rows.push(last);
                    }
                }
                assert_eq!(expected_rows.len(), ioh_run.rows.len(), "row count for {key:?}");
                for ((e1, y1), (e2, y2)) in expected_rows.iter().zip(&ioh_run.rows) {
                    assert_eq!(e1, e2);
                    assert_eq!(y1.to_bits(), y2.to_bits(), "raw_y bits must match exactly");
                }
            }
        }
    }

    #[test]
    fn torn_final_line_errors_naming_the_file() {
        let tmp = tempfile::tempdir().unwrap();
        let algo_dir = tmp.path().join("de");
        std::fs::create_dir_all(&algo_dir).unwrap();
        let data_dir = algo_dir.join("data_f1_Sphere");
        std::fs::create_dir_all(&data_dir).unwrap();
        let dat_path = data_dir.join("IOHprofiler_f1_DIM5.dat");
        // Torn: the writer got cut off mid-value on the final line.
        std::fs::write(&dat_path, "\"evaluations\" \"raw_y\"\n1 10\n2 -\n").unwrap();

        let meta = serde_json::json!({
            "version": "sezgi-0.1", "suite": "sezgi-bbob", "function_id": 1,
            "function_name": "Sphere", "maximization": false,
            "algorithm": {"name": "de"},
            "scenarios": [{
                "dimension": 5,
                "path": "data_f1_Sphere/IOHprofiler_f1_DIM5.dat",
                "runs": [{"instance": 1, "evals": 2, "best": {"evals": 1, "y": 10.0}}],
            }],
        });
        std::fs::write(
            algo_dir.join("IOHprofiler_f1_Sphere.json"),
            serde_json::to_string_pretty(&meta).unwrap(),
        ).unwrap();

        let err = read_ioh_root(tmp.path()).unwrap_err();
        assert!(
            err.to_string().contains(dat_path.to_str().unwrap()),
            "error must name the corrupt file: {err}"
        );
    }

    #[test]
    fn legacy_run_missing_seed_and_f_opt_reads_as_none() {
        let tmp = tempfile::tempdir().unwrap();
        let mut lg = IohLogger::new(tmp.path(), "de", "sezgi-bbob", 1, "Sphere", 3);
        {
            let mut o = lg.start_run(7); // legacy path: no seed/f_opt known
            o.on_eval(1, 5.0, 5.0);
        }
        lg.finish().unwrap();

        let scenarios = read_ioh_root(tmp.path()).unwrap();
        assert_eq!(scenarios.len(), 1);
        let run0 = &scenarios[0].runs[0];
        assert_eq!(run0.instance, 7);
        assert_eq!(run0.seed, None);
        assert_eq!(run0.f_opt, None);
        assert_eq!(run0.rows, vec![(1, 5.0)]);
    }
}
