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
//! Note: the meta file is named from `(fid, fname)` alone, not `dim` — one
//! meta file per `(algo, fid)`, per the IOH convention. [`crate::ioh::IohLogger::finish`]
//! MERGES into it rather than overwriting: two `finish()` calls for the
//! same `(algo, fid)` at different `dim`s both leave their scenario in the
//! same meta file's `scenarios[]` array (keyed by `dimension`; re-`finish()`ing
//! the same dim replaces that entry rather than duplicating it), so
//! [`read_ioh_root`] sees every dimension's scenario for that `(algo, fid)` —
//! it simply iterates whatever `scenarios[]` entries a meta file contains,
//! producing one [`IohScenario`] per entry.
//!
//! ## Records bridge (M2d Task 5)
//!
//! [`ioh_records`] is the read-side counterpart to
//! [`crate::experiment::run_experiment_sequential`]/`_parallel`/`_logged`:
//! it turns [`read_ioh_root`]'s `Vec<IohScenario>` back into
//! `Vec<`[`crate::experiment::RunRecord`]`>`, so a paper package
//! ([`crate::reporting::per_budget_packages`]) can be built purely from a
//! disk archive, with no in-memory experiment run required. See its doc
//! comment for the exact `best_f`/`evals_used` semantics and — important —
//! the curtailed-view-vs-independent-run distinction for budgets smaller
//! than a run's logged budget.

use crate::experiment::{ExperimentError, RunKey, RunRecord};
use serde::Deserialize;
use std::collections::HashMap;
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
    /// `None` for a legacy log whose meta entry omitted `budget` — either a
    /// pre-M2d-3 archive (no `start_run_with` budget parameter existed yet)
    /// or a fully-legacy [`crate::ioh::IohLogger::start_run`] entry (which
    /// omits `seed`/`f_opt`/`budget` together). See [`dedupe_runs`] for how
    /// a missing `budget` is handled when disambiguating same-`(instance,
    /// seed)` runs.
    pub budget: Option<u64>,
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
    #[serde(default)]
    budget: Option<u64>,
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
                        budget: mr.budget,
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

// ---------------------------------------------------------------------
// Dedupe (M2d-3 Task 1): canonicalize multi-budget archives to one run
// per (instance, seed) — the single canonicalization point [`ecdf`],
// [`ecdf_per_algo`], [`coco_export`], and [`ioh_records`] all call first.
// ---------------------------------------------------------------------

/// Canonicalizes `scenarios` in place so that, within each scenario, every
/// `(instance, seed)` pair maps to exactly one run.
///
/// A budget-aware archive (written since M2d-3, via
/// [`crate::ioh::IohLogger::start_run_with`]'s `budget` parameter) can
/// legitimately hold MULTIPLE runs sharing the same `(instance, seed)` —
/// one per budget the spec swept over, each a fully independent trajectory
/// (see `run_experiment_logged`'s doc comment on why running each queried
/// budget as its own `PlannedRun`, rather than deriving smaller budgets
/// from one big run, is how sezgi avoids budget-adaptive-schedule
/// divergence). For each such group, this function keeps ONLY the run with
/// the LARGEST `budget` — its trajectory strictly subsumes every smaller
/// budget's curtailed view (see [`ioh_records`]'s doc comment) — and drops
/// the rest.
///
/// A group of size 1 is always left as-is, `budget` present or not (a
/// single run is never ambiguous, `budget`-carrying or legacy).
///
/// A group of size > 1 where ANY member's `budget` is `None` is an ERROR
/// (message contains `"ambiguous"`): this is a pre-M2d-3 archive whose
/// duplicate `(instance, seed)` runs cannot be told apart — the archive
/// must be re-logged with a budget-aware `start_run_with` call, or read as
/// a genuinely single-budget archive (whose `run_experiment_logged` call
/// never produces duplicate `(instance, seed)` runs in the first place).
// `&mut Vec<_>` (not `&mut [_]`) is the pinned interface (M2d-3 task brief):
// canonicalization is in-place mutation of `scenarios`, and `&mut Vec` is
// what every call site already owns (a `to_vec()`'d clone or a caller's
// owned archive) — clippy's `ptr_arg` suggestion would just push the same
// `&mut Vec` at the call site into a `.as_mut_slice()`, with no meaningful
// prevention gained.
#[allow(clippy::ptr_arg)]
pub fn dedupe_runs(scenarios: &mut Vec<IohScenario>) -> Result<(), ExperimentError> {
    for sc in scenarios.iter_mut() {
        // Group run indices by (instance, seed), preserving first-seen
        // group order for a deterministic result.
        let mut order: Vec<(u32, Option<u64>)> = Vec::new();
        let mut groups: HashMap<(u32, Option<u64>), Vec<usize>> = HashMap::new();
        for (i, run) in sc.runs.iter().enumerate() {
            let key = (run.instance, run.seed);
            if !groups.contains_key(&key) {
                order.push(key);
            }
            groups.entry(key).or_default().push(i);
        }

        let mut kept: Vec<usize> = Vec::with_capacity(order.len());
        for key in &order {
            let idxs = &groups[key];
            if idxs.len() == 1 {
                kept.push(idxs[0]);
                continue;
            }
            if idxs.iter().any(|&i| sc.runs[i].budget.is_none()) {
                return Err(ExperimentError::IohRead(format!(
                    "{}/f{}d{}: ambiguous pre-M2d-3 archive: {} runs share instance {} \
                     seed {:?} with no budget key to disambiguate them — re-log with a \
                     budget-aware archive (IohLogger::start_run_with) or read a \
                     single-budget archive",
                    sc.algo, sc.fid, sc.dim, idxs.len(), key.0, key.1,
                )));
            }
            let winner = *idxs.iter()
                .max_by_key(|&&i| sc.runs[i].budget.expect("checked Some above"))
                .expect("idxs is non-empty");
            kept.push(winner);
        }
        kept.sort_unstable();

        sc.runs = kept.into_iter().map(|i| sc.runs[i].clone()).collect();
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Records bridge (M2d Task 5): on-disk IOH archives -> paper packages
// ---------------------------------------------------------------------

/// Reconstructs [`RunRecord`]s from on-disk [`IohScenario`]s, for every
/// `(run, budget)` pair — one record per run per requested budget — so the
/// result feeds [`crate::reporting::results_matrix`]/
/// [`crate::reporting::per_budget_packages`] UNCHANGED, exactly as if it had
/// come from [`crate::experiment::run_experiment_sequential`] et al.
///
/// ## `best_f` semantics: best-so-far AT that budget
///
/// A run's [`IohRun::rows`] are improvement rows (plus the always-written
/// final row) in file order — `evals` strictly increasing, `raw_y`
/// non-increasing (each row only records a NEW best). For a requested
/// `budget`, `best_f` is the `raw_y` of the LAST row with `evals <= budget`:
/// because the sequence is monotone, that row's `raw_y` is not merely *a*
/// value seen by that budget, it IS the best value seen by that budget — no
/// later (higher-eval) row within the budget could have a smaller `raw_y`
/// without having been recorded as an improvement row itself. It is an
/// error — naming the run and the budget — for a run to have NO row with
/// `evals <= budget` (the run's first recorded eval count already exceeds
/// the budget, so no best-so-far value exists at that budget).
///
/// `evals_used = min(budget, run.evals)`: the run may have used fewer than
/// `budget` evaluations in total (if it terminated early), in which case
/// `evals_used` reports the run's actual eval count rather than overstating
/// it as `budget`.
///
/// `f_opt` and `seed` are both taken from the run's meta and are REQUIRED —
/// `None` (a legacy archive written via `IohLogger::start_run` rather than
/// `start_run_with`) is an error naming the missing key, since `RunKey.seed`
/// and `RunRecord.f_opt` are non-optional. [`ioh_records`] therefore only
/// accepts SELF-CONTAINED archives (every run meta carries `seed`+`f_opt`).
///
/// ## Curtailed view vs. independent run — READ THIS before comparing
/// ## against records from the live runner
///
/// For a budget EQUAL TO the run's actual logged budget, the reconstructed
/// `best_f` is bit-identical to what the runner returned in memory (both
/// are the same trajectory's final best-so-far value) — the disk round trip
/// loses nothing.
///
/// For a budget SMALLER than the run's logged budget, `ioh_records` returns
/// a CURTAILED VIEW: it truncates the SAME trajectory (the one recorded at
/// the larger, actual budget) to the prefix visible by the smaller budget.
/// This is semantically a DIFFERENT thing from what
/// [`crate::experiment::enumerate`] produces for that smaller budget as an
/// independent [`crate::experiment::PlannedRun`]: `budget` is part of
/// [`RunKey`], but `run_id` is derived only from the seed's index in
/// `ExperimentSpec::seeds` (see `enumerate`), NOT from budget — so the
/// runner's smaller-budget run reuses the very same RNG stream, but as a
/// SEPARATE `Engine::run` invocation whose `AlgorithmSpec::termination.budget`
/// is set to the smaller value up front. For most presets this changes
/// nothing beyond where the run stops (so the trajectories usually agree on
/// their shared prefix) — but any budget-adaptive component (e.g. a mutation
/// or population schedule that reads `termination.budget` to plan its
/// trajectory over the FULL run, such as `lshade`'s population-shrinking
/// schedule) can make the independent smaller-budget run diverge from the
/// larger run's prefix. Curtailing a trajectory recorded at budget B to a
/// view at budget b < B is therefore NOT guaranteed to equal — and must
/// never be asserted equal to — an independent run planned at budget b from
/// the start. Callers wanting the reconstructed curtailed view and the live
/// runner's own smaller-budget records to match bit-for-bit are asserting
/// something this function does not promise.
///
/// For a `budget` LARGER than the run's own logged budget, `best_f` is the
/// run's final recorded value and `evals_used` is `run.evals` (never
/// `budget` itself, since `evals_used = min(budget, run.evals)`) — the same
/// "run has already finished, no more evaluations happen" semantics the
/// live runner itself exhibits once a run's own termination budget is
/// reached, so this is silent, not an error.
///
/// `wall_secs` is always `0.0`: wall-clock time is not recorded in the
/// on-disk IOH log, so it cannot be reconstructed here; this is an inert
/// placeholder rather than a real measurement, and the reporting pipeline
/// (`results_matrix`/`per_budget_packages`) never reads this field.
///
/// [`dedupe_runs`] is applied to (a private clone of) `scenarios` FIRST, so
/// a multi-budget archive (one written since M2d-3 with duplicate
/// `(instance, seed)` runs across budgets) is canonicalized to one run per
/// `(instance, seed)` — the largest-budget one — before any record is
/// built; the returned `RunKey`s are therefore always unique per queried
/// budget, never duplicated with conflicting `best_f`.
pub fn ioh_records(scenarios: &[IohScenario], budgets: &[u64]) -> Result<Vec<RunRecord>, ExperimentError> {
    let mut scenarios = scenarios.to_vec();
    dedupe_runs(&mut scenarios)?;

    let mut records = Vec::new();
    for sc in &scenarios {
        for run in &sc.runs {
            let run_label = format!("{}/f{}d{}i{}", sc.algo, sc.fid, sc.dim, run.instance);
            let seed = run.seed.ok_or_else(|| ExperimentError::IohRead(format!(
                "{run_label}: missing `seed` in meta (ioh_records requires self-contained \
                 archives written by IohLogger::start_run_with)"
            )))?;
            let f_opt = run.f_opt.ok_or_else(|| ExperimentError::IohRead(format!(
                "{run_label}/s{seed}: missing `f_opt` in meta (ioh_records requires \
                 self-contained archives written by IohLogger::start_run_with)"
            )))?;

            for &budget in budgets {
                let best_f = run.rows.iter()
                    .rev()
                    .find(|&&(evals, _)| evals <= budget)
                    .map(|&(_, y)| y)
                    .ok_or_else(|| ExperimentError::IohRead(format!(
                        "{run_label}/s{seed}: no row with evals <= budget {budget} \
                         (the run's first recorded eval count already exceeds this budget)"
                    )))?;

                records.push(RunRecord {
                    key: RunKey {
                        algo: sc.algo.clone(),
                        fid: sc.fid,
                        dim: sc.dim,
                        instance: run.instance,
                        seed,
                        budget,
                    },
                    best_f,
                    f_opt,
                    evals_used: budget.min(run.evals),
                    wall_secs: 0.0,
                });
            }
        }
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiment::{
        AlgoEntry, AlgoSource, ExperimentSpec, PlannedRun, ProblemEntry, enumerate, execute_run,
        run_experiment_logged,
    };
    use crate::ioh::IohLogger;
    use crate::reporting::{Aggregate, per_budget_packages};
    use sezgi_components::register_builtins;
    use sezgi_core::component::Registry;
    use sezgi_core::problem::EvalObserver;
    use std::collections::HashMap;
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
                assert_eq!(ioh_run.budget, Some(planned_run.key.budget));

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

    /// Regression for the multi-dim meta-collision bug: `run_experiment_logged`
    /// swept over one fid at TWO dims used to lose all but the
    /// last-finished dim's meta scenario (the dim-5 `.dat` orphaned,
    /// `read_ioh_root` silently returning only one scenario). `finish()`
    /// now merges into the shared `(algo, fid)` meta file instead of
    /// overwriting it, so both dims' scenarios must come back.
    #[test]
    fn multi_dim_same_fid_both_scenarios_survive_end_to_end() {
        let spec = ExperimentSpec {
            name: "multi-dim".into(),
            seeds: vec![11, 22],
            budgets: vec![300],
            algorithms: vec![AlgoEntry {
                name: "de".into(),
                source: AlgoSource::Preset { kind: "de_rand_1".into(), pop_size: Some(6) },
            }],
            problems: vec![
                ProblemEntry { suite: "bbob".into(), fid: 1, dim: 5, instances: vec![1] },
                ProblemEntry { suite: "bbob".into(), fid: 1, dim: 10, instances: vec![1] },
            ],
        };
        let tmp = tempfile::tempdir().unwrap();
        run_experiment_logged(&spec, tmp.path(), false, None).unwrap();

        let scenarios = read_ioh_root(tmp.path()).unwrap();
        assert_eq!(scenarios.len(), 2, "both dims of the same (algo, fid) must survive");

        let dims: Vec<usize> = {
            let mut d: Vec<usize> = scenarios.iter().map(|s| s.dim).collect();
            d.sort();
            d
        };
        assert_eq!(dims, vec![5, 10]);

        for dim in [5usize, 10usize] {
            let sc = scenarios.iter().find(|s| s.dim == dim).unwrap();
            assert_eq!(sc.fid, 1);
            assert_eq!(sc.algo, "de");
            // Both seeds' runs present for this dim, each with its seed
            // and f_opt intact.
            assert_eq!(sc.runs.len(), 2, "dim {dim} must keep both runs");
            let expected_f_opt = sezgi_problems::BbobProblem::new(1, dim, 1).unwrap().f_opt();
            for (run, &seed) in sc.runs.iter().zip(&[11u64, 22u64]) {
                assert_eq!(run.seed, Some(seed));
                assert_eq!(run.f_opt.map(f64::to_bits), Some(expected_f_opt.to_bits()));
                assert!(!run.rows.is_empty(), "dim {dim} run must have data rows");
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
        assert_eq!(run0.budget, None);
        assert_eq!(run0.rows, vec![(1, 5.0)]);
    }

    // -------------------------------------------------------------
    // ioh_records (M2d Task 5)
    // -------------------------------------------------------------

    fn hand_scenario(rows: Vec<(u64, f64)>, evals: u64, seed: Option<u64>, f_opt: Option<f64>) -> IohScenario {
        IohScenario {
            algo: "de".into(),
            suite: "sezgi-bbob".into(),
            fid: 1,
            fname: "Sphere".into(),
            dim: 5,
            runs: vec![IohRun { instance: 1, seed, f_opt, budget: None, rows, evals }],
        }
    }

    #[test]
    fn mid_budget_selects_last_row_within_budget() {
        // Rows at evals 1/3/10; budget 5 must pick the eval-3 value (the
        // last row with evals <= 5), not the eval-1 or eval-10 value.
        let sc = hand_scenario(vec![(1, 10.0), (3, 4.0), (10, 1.0)], 10, Some(42), Some(0.0));
        let records = ioh_records(std::slice::from_ref(&sc), &[5]).unwrap();
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert_eq!(r.best_f.to_bits(), 4.0f64.to_bits(), "must select the eval-3 row, not eval-1 or eval-10");
        assert_eq!(r.evals_used, 5, "evals_used = min(budget, run.evals) = min(5, 10)");
        assert_eq!(r.f_opt.to_bits(), 0.0f64.to_bits());
        assert_eq!(r.key.seed, 42);
        assert_eq!(r.key.budget, 5);
    }

    #[test]
    fn budget_at_or_above_full_evals_selects_final_row() {
        let sc = hand_scenario(vec![(1, 10.0), (3, 4.0), (10, 1.0)], 10, Some(42), Some(0.0));
        let records = ioh_records(std::slice::from_ref(&sc), &[10, 999]).unwrap();
        for r in &records {
            assert_eq!(r.best_f.to_bits(), 1.0f64.to_bits());
        }
        assert_eq!(records[0].evals_used, 10, "min(10, 10)");
        assert_eq!(records[1].evals_used, 10, "min(999, 10)");
    }

    #[test]
    fn budget_below_first_row_errors_naming_run_and_budget() {
        let sc = hand_scenario(vec![(5, 4.0), (10, 1.0)], 10, Some(42), Some(0.0));
        let err = ioh_records(std::slice::from_ref(&sc), &[3]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("de/f1d5i1"), "error must name the run: {msg}");
        assert!(msg.contains('3'), "error must name the budget: {msg}");
    }

    #[test]
    fn missing_seed_errors_mentioning_seed() {
        let sc = hand_scenario(vec![(1, 4.0)], 1, None, Some(0.0));
        let err = ioh_records(std::slice::from_ref(&sc), &[1]).unwrap_err();
        assert!(err.to_string().contains("seed"), "error must mention the missing `seed` key: {err}");
    }

    #[test]
    fn missing_f_opt_errors_mentioning_f_opt() {
        let sc = hand_scenario(vec![(1, 4.0)], 1, Some(42), None);
        let err = ioh_records(std::slice::from_ref(&sc), &[1]).unwrap_err();
        assert!(err.to_string().contains("f_opt"), "error must mention the missing `f_opt` key: {err}");
    }

    /// The CORE test: a disk-log-driven paper package from a single-budget
    /// experiment. `budgets = [500]` on the spec (a SINGLE budget) is
    /// deliberate — see `ioh_records`'s doc comment on why a curtailed view
    /// at a SMALLER budget must not be compared bit-for-bit against the
    /// runner's own (independently planned) records at that smaller budget.
    /// Here we only assert bit-identity at the run's actual, full budget
    /// (500) — the disk round trip losing nothing — and separately exercise
    /// the curtailed view at a smaller budget (200) purely to prove
    /// `per_budget_packages` accepts it and produces a NaN-free package,
    /// without asserting it equals anything from a live smaller-budget run.
    #[test]
    fn full_budget_disk_round_trip_is_bit_identical_and_feeds_paper_packages() {
        let spec = ExperimentSpec {
            name: "disk-package".into(),
            seeds: vec![101, 202],
            budgets: vec![500],
            algorithms: vec![
                AlgoEntry {
                    name: "de".into(),
                    source: AlgoSource::Preset { kind: "de_rand_1".into(), pop_size: Some(8) },
                },
                AlgoEntry {
                    name: "rs".into(),
                    source: AlgoSource::Preset { kind: "random_search".into(), pop_size: Some(8) },
                },
            ],
            problems: vec![ProblemEntry {
                suite: "bbob".into(), fid: 1, dim: 5, instances: vec![1, 2, 3, 4, 5],
            }],
        };

        let tmp = tempfile::tempdir().unwrap();
        let (runner_records, _finishes) = run_experiment_logged(&spec, tmp.path(), false, None).unwrap();
        // 2 algos x 5 instances x 2 seeds x 1 budget
        assert_eq!(runner_records.len(), 20);

        let scenarios = read_ioh_root(tmp.path()).unwrap();
        let disk_records = ioh_records(&scenarios, &[200, 500]).unwrap();
        // 2 algos x 5 instances x 2 seeds x 2 queried budgets
        assert_eq!(disk_records.len(), 40);

        // Full-budget (500) records must be bit-identical to the runner's
        // in-memory RunRecords — matched by key, since ioh_records' nested
        // scenario/run/budget iteration order need not equal the runner's
        // flat cartesian enumeration order.
        let mut runner_by_key: HashMap<RunKey, (f64, u64)> = HashMap::new();
        for r in &runner_records {
            runner_by_key.insert(r.key.clone(), (r.best_f, r.evals_used));
        }
        assert_eq!(runner_by_key.len(), 20);

        let full_budget_disk: Vec<&RunRecord> = disk_records.iter().filter(|r| r.key.budget == 500).collect();
        assert_eq!(full_budget_disk.len(), 20);
        for r in &full_budget_disk {
            let (expected_best_f, expected_evals_used) = *runner_by_key.get(&r.key)
                .unwrap_or_else(|| panic!("no runner record for key {}", r.key));
            assert_eq!(
                r.best_f.to_bits(), expected_best_f.to_bits(),
                "disk round trip at the full budget must be bit-identical for {}: disk={} runner={}",
                r.key, r.best_f, expected_best_f,
            );
            assert_eq!(
                r.evals_used, expected_evals_used,
                "evals_used at the full budget must match the runner's for {}", r.key,
            );
        }

        // Curtailed 200-budget view: present, distinct records, no assertion
        // that they match any independent smaller-budget run (there is none
        // in this spec — see the doc comment above).
        let curtailed: Vec<&RunRecord> = disk_records.iter().filter(|r| r.key.budget == 200).collect();
        assert_eq!(curtailed.len(), 20);
        for r in &curtailed {
            assert_eq!(r.evals_used, 200, "evals_used = min(200, run.evals)");
        }

        let packages = per_budget_packages(&disk_records, 0.1, 200, 7, Aggregate::Mean).unwrap();
        assert_eq!(packages.len(), 2, "one package per distinct queried budget");
        assert_eq!(packages[0].0, 200);
        assert_eq!(packages[1].0, 500);
        for (budget, pkg) in &packages {
            assert!(
                !pkg.latex_summary.to_lowercase().contains("nan"),
                "budget {budget}: latex_summary must be NaN-free: {}", pkg.latex_summary
            );
            assert!(
                !pkg.latex_tests.to_lowercase().contains("nan"),
                "budget {budget}: latex_tests must be NaN-free: {}", pkg.latex_tests
            );
        }
    }

    // -------------------------------------------------------------
    // dedupe_runs / multi-budget logging (M2d-3 Task 1)
    // -------------------------------------------------------------

    /// CORE test: `run_experiment_logged` with TWO budgets — previously
    /// rejected by the (now-removed) `MultiBudgetLogging` guard — succeeds,
    /// and the resulting archive is unambiguous end to end: `read_ioh_root`
    /// plus `dedupe_runs` collapses each `(instance, seed)` pair down to
    /// its one largest-budget run, and `ioh_records` (which dedupes
    /// internally) returns unique `RunKey`s with no conflicting `best_f`
    /// for the same key. This is the M2d-2 final-review probe scenario the
    /// multi-budget guard was originally added to guard against.
    #[test]
    fn multi_budget_logging_dedupes_to_unique_records() {
        let spec = ExperimentSpec {
            name: "multi-budget-probe".into(),
            seeds: vec![101, 202],
            budgets: vec![200, 400],
            algorithms: vec![
                AlgoEntry {
                    name: "de".into(),
                    source: AlgoSource::Preset { kind: "de_rand_1".into(), pop_size: Some(8) },
                },
                AlgoEntry {
                    name: "rs".into(),
                    source: AlgoSource::Preset { kind: "random_search".into(), pop_size: Some(8) },
                },
            ],
            problems: vec![ProblemEntry {
                suite: "bbob".into(), fid: 1, dim: 5, instances: vec![1, 2, 3, 4, 5],
            }],
        };

        let tmp = tempfile::tempdir().unwrap();
        let (runner_records, _finishes) = run_experiment_logged(&spec, tmp.path(), false, None).unwrap();
        // 2 algos x 5 instances x 2 seeds x 2 budgets
        assert_eq!(runner_records.len(), 40);

        let raw_scenarios = read_ioh_root(tmp.path()).unwrap();
        assert_eq!(raw_scenarios.len(), 2, "one scenario per algo");
        for sc in &raw_scenarios {
            // 5 instances x 2 seeds x 2 budgets: both budgets' runs present
            // pre-dedupe, i.e. duplicate (instance, seed) pairs on disk.
            assert_eq!(sc.runs.len(), 20);
        }

        // read_ioh_root + dedupe_runs: exactly one run per (instance, seed),
        // and it must be the larger-budget (400) one.
        let mut deduped = raw_scenarios.clone();
        dedupe_runs(&mut deduped).unwrap();
        for sc in &deduped {
            assert_eq!(sc.runs.len(), 10, "5 instances x 2 seeds, one run each after dedupe");
            for run in &sc.runs {
                assert_eq!(run.budget, Some(400), "dedupe must keep the largest-budget run");
            }
        }

        // ioh_records dedupes internally, so passing the RAW (non-deduped)
        // scenarios must still yield unique (key -> best_f) pairs: 2 algos x
        // 10 (instance, seed) groups x 2 queried budgets = 40 records, 40
        // unique keys, and every key's best_f must agree with the runner's
        // own in-memory record at the run's actual, full (400) budget.
        let records = ioh_records(&raw_scenarios, &[200, 400]).unwrap();
        assert_eq!(records.len(), 40);

        let mut seen: HashMap<RunKey, f64> = HashMap::new();
        for r in &records {
            let prior = seen.insert(r.key.clone(), r.best_f);
            assert!(prior.is_none(), "duplicate key {} with conflicting best_f", r.key);
        }
        assert_eq!(seen.len(), 40, "all 40 keys must be unique");

        let runner_by_key: HashMap<RunKey, f64> =
            runner_records.iter().map(|r| (r.key.clone(), r.best_f)).collect();
        for r in records.iter().filter(|r| r.key.budget == 400) {
            let expected = runner_by_key.get(&r.key)
                .unwrap_or_else(|| panic!("no runner record for key {}", r.key));
            assert_eq!(
                r.best_f.to_bits(), expected.to_bits(),
                "budget-400 disk record must be bit-identical to the runner's for {}", r.key,
            );
        }
    }

    /// Legacy ambiguity: a hand-written meta with two runs sharing the same
    /// `(instance, seed)` and neither carrying a `budget` key (a pre-M2d-3
    /// archive) is genuinely ambiguous — `dedupe_runs` must error, matching
    /// "ambiguous", rather than silently picking one or pooling both.
    #[test]
    fn dedupe_runs_errors_on_ambiguous_duplicate_without_budget() {
        let mut scenarios = vec![IohScenario {
            algo: "de".into(),
            suite: "sezgi-bbob".into(),
            fid: 1,
            fname: "Sphere".into(),
            dim: 5,
            runs: vec![
                IohRun { instance: 1, seed: Some(42), f_opt: Some(0.0), budget: None,
                          rows: vec![(1, 5.0)], evals: 1 },
                IohRun { instance: 1, seed: Some(42), f_opt: Some(0.0), budget: None,
                          rows: vec![(1, 3.0)], evals: 1 },
            ],
        }];

        let err = dedupe_runs(&mut scenarios).unwrap_err();
        assert!(err.to_string().contains("ambiguous"), "error must mention 'ambiguous': {err}");
    }

    /// Regression guard alongside the ambiguity test above: a single
    /// legacy run (no duplicate, no budget) is never ambiguous and must
    /// read through `dedupe_runs` unchanged.
    #[test]
    fn dedupe_runs_leaves_single_legacy_run_without_budget_untouched() {
        let mut scenarios = vec![IohScenario {
            algo: "de".into(),
            suite: "sezgi-bbob".into(),
            fid: 1,
            fname: "Sphere".into(),
            dim: 5,
            runs: vec![IohRun {
                instance: 1, seed: None, f_opt: None, budget: None,
                rows: vec![(1, 5.0)], evals: 1,
            }],
        }];

        dedupe_runs(&mut scenarios).unwrap();
        assert_eq!(scenarios[0].runs.len(), 1, "a single run is never ambiguous");
        assert_eq!(scenarios[0].runs[0].budget, None);
    }
}
