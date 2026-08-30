use sezgi_core::problem::EvalObserver;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// The `.dat` file's per-run block header line (IOH convention).
const DAT_HEADER: &str = "\"evaluations\" \"raw_y\"";

/// Splits a `.dat` file's text (as written by this module, one header line
/// per run followed by its rows) back into per-run blocks (each block's
/// first line is [`DAT_HEADER`]) for [`IohLogger::finish`]'s run-identity
/// merge. Trusts the input's own formatting (this module's writer is the
/// only producer); unlike `crate::ioh_read`'s reader, it does not validate
/// row shape — it only needs to preserve each block's lines verbatim so
/// they can be re-written unchanged.
fn split_dat_into_blocks(text: &str) -> Vec<Vec<String>> {
    let mut blocks: Vec<Vec<String>> = Vec::new();
    for line in text.lines() {
        if line.trim_end_matches('\r') == DAT_HEADER {
            blocks.push(vec![line.to_string()]);
        } else if let Some(last) = blocks.last_mut() {
            last.push(line.to_string());
        }
    }
    blocks
}

/// One of this logger's own runs, staged in memory during [`IohLogger::finish`]
/// before it is merged (by `identity`) against any existing scenario for the
/// same dimension.
struct NewRun {
    /// `(instance, seed, budget)` — a legacy `start_run` run always carries
    /// `(instance, 0, 0)`, matching its own never-serialized `RunData`
    /// defaults, so it identifies consistently across `finish()` calls.
    identity: (u64, u64, u64),
    entry: serde_json::Value,
    /// This run's `.dat` block, header line included.
    block: Vec<String>,
}

#[derive(Debug, Clone)]
struct RunData {
    instance: u32,
    seed: u64,
    f_opt: f64,                  // f64::NAN means "unknown" (legacy `start_run` path):
                                  // never serialized — see `finish()`.
    budget: u64,                 // meaningless (never serialized) when f_opt is NaN —
                                  // see `finish()`'s single NaN-gated omission block.
    rows: Vec<(u64, f64)>,       // improvement rows
    last: Option<(u64, f64)>,    // final eval (always written)
    evals: u64,
    best: Option<(u64, f64)>,
}

#[derive(Debug)]
pub struct IohFinish {
    pub meta_path: PathBuf,
    pub skipped_empty_runs: usize,
}

pub struct IohLogger {
    root: PathBuf,
    algo: String,
    suite: String,
    fid: u32,
    fname: String,
    dim: usize,
    runs: Vec<Arc<Mutex<RunData>>>,
}

pub struct IohRunObserver {
    data: Arc<Mutex<RunData>>,
    prev_best: Option<f64>,
}

impl EvalObserver for IohRunObserver {
    fn on_eval(&mut self, eval_index: u64, _f: f64, best_so_far: f64) {
        let mut d = self.data.lock().unwrap();
        d.evals = eval_index;
        d.last = Some((eval_index, best_so_far));
        let improved = self.prev_best.is_none_or(|p| best_so_far < p);
        if improved {
            d.rows.push((eval_index, best_so_far));
            d.best = Some((eval_index, best_so_far));
            self.prev_best = Some(best_so_far);
        }
    }
}

impl IohLogger {
    pub fn new(root: &Path, algo: &str, suite: &str, fid: u32,
               fname: &str, dim: usize) -> Self {
        Self { root: root.to_path_buf(), algo: algo.into(), suite: suite.into(),
               fid, fname: fname.into(), dim, runs: vec![] }
    }

    /// Legacy entry point (M2c): no seed/f_opt/budget known at call time.
    /// Delegates to [`Self::start_run_with`] with `seed = 0`,
    /// `f_opt = f64::NAN`, and `budget = 0`; `finish()` omits the
    /// `"seed"`/`"f_opt"`/`"budget"` meta keys for such a run (NaN is never
    /// serialized into JSON, and the omission is gated on that same NaN
    /// check, so `budget` is dropped alongside them even though it carries
    /// no sentinel of its own), keeping the meta backward-compatible with
    /// pre-M2d readers.
    pub fn start_run(&mut self, instance: u32) -> IohRunObserver {
        self.start_run_with(instance, 0, f64::NAN, 0)
    }

    /// Starts a run with its master `seed`, the problem instance's known
    /// `f_opt`, and the run's `budget` (its `AlgorithmSpec::termination.budget`),
    /// all three of which `finish()` writes into that run's meta entry —
    /// `budget` is what lets a later read tell apart multiple runs of the
    /// same `(instance, seed)` logged at different budgets. See
    /// `crate::ioh_read`'s module doc ("Two canonicalization policies") for
    /// how: [`crate::ioh_read::ioh_records`] prefers each budget's own
    /// genuine run, while [`crate::ioh_read::canonical_anytime`] instead
    /// keeps only the largest-budget one. Pass `f_opt = f64::NAN` only via
    /// [`Self::start_run`] (legacy path); a caller that knows the real
    /// f_opt must always pass it here.
    pub fn start_run_with(&mut self, instance: u32, seed: u64, f_opt: f64, budget: u64) -> IohRunObserver {
        let data = Arc::new(Mutex::new(RunData {
            instance, seed, f_opt, budget, rows: vec![], last: None, evals: 0, best: None }));
        self.runs.push(data.clone());
        IohRunObserver { data, prev_best: None }
    }

    /// Writes this logger's `.dat` file and merges its scenario entry into
    /// the `(algo, fid)` meta JSON.
    ///
    /// The meta file is one-per-`(algo, fid)` — the IOH convention (spec
    /// §7: logs are read directly by IOHinspector/IOHanalyzer, which expect
    /// a single meta file per function covering every dimension it was run
    /// at) — never per-dim, even though the `.dat` files are (their names
    /// embed `DIM<dim>`). If a meta file already exists at that path, it is
    /// parsed and this call's scenario is merged into its `scenarios[]`
    /// array, keyed by `dimension`: a dimension with no existing entry gets
    /// one appended; any other dimension's entry is left untouched. The
    /// file's other top-level fields (`suite`, `function_id`,
    /// `function_name`, `algorithm.name`) must already agree with this
    /// logger's — a mismatch (e.g. two different algorithms writing into
    /// the same `(fid, fname)` meta path) is an error rather than a silent
    /// clobber.
    ///
    /// // sezgi decision: within one dimension's scenario, this logger's
    /// // runs are merged into any existing scenario BY RUN IDENTITY — the
    /// // `(instance, seed, budget)` triple (a legacy `start_run` run, whose
    /// // seed/budget are never serialized, identifies by `(instance, 0,
    /// // 0)`, matching its own always-`0` `RunData` defaults). A run whose
    /// // identity already exists in that scenario is REWRITTEN in place
    /// // (both its meta entry and its `.dat` block) — this is what keeps
    /// // re-`finish()`ing the SAME logger idempotent: re-running the exact
    /// // same `(instance, seed, budget)` and finishing again reproduces
    /// // that identity, so it replaces its own prior entry rather than
    /// // duplicating it. A run whose identity is new to the scenario is
    /// // APPENDED. This is what lets `bbob_records`-style callers — a
    /// // fresh `IohLogger`/`finish()` per run, one seed at a time, into the
    /// // same `(algo, fid, dim)` target — accumulate every seed's run
    /// // into one archive instead of each `finish()` clobbering the last:
    /// // distinct seeds are distinct identities, so they append; the same
    /// // seed re-run replaces only itself. The single-`finish()`,
    /// // many-`start_run_with()`-calls pattern (`run_experiment_logged`)
    /// // is unaffected: its one `finish()` call per `(algo, fid, dim)`
    /// // never re-enters this merge (no existing scenario for that
    /// // dimension yet), so all of its runs are appended together exactly
    /// // as before.
    ///
    /// The merge above is an UNLOCKED read-modify-write on the meta file (no
    /// file lock, no atomic rename): it reads `meta_path`, computes the
    /// merged `scenarios[]` in memory, and overwrites the file. This is safe
    /// for one process writing to a given `log_dir` (all `finish()` calls
    /// happen sequentially, after every run has completed — see
    /// `run_experiment_logged`), but NOT for multiple processes sharing the
    /// same `log_dir` concurrently: two processes racing to merge into the
    /// same `(algo, fid)` meta file can each read the file before the
    /// other's write lands, and the loser's write silently drops the
    /// winner's scenario. Keep one process per `log_dir`.
    pub fn finish(self) -> std::io::Result<IohFinish> {
        let dir = self.root.join(&self.algo);
        let data_rel = format!("data_f{}_{}", self.fid, self.fname);
        let data_dir = dir.join(&data_rel);
        std::fs::create_dir_all(&data_dir)?;

        let dat_name = format!("IOHprofiler_f{}_DIM{}.dat", self.fid, self.dim);
        let dat_path = data_dir.join(&dat_name);

        // Build this logger's own run entries + `.dat` blocks in memory
        // first (no file writes yet) so they can be merged by identity
        // against whatever the dat file/meta already hold, below.
        let mut new_runs: Vec<NewRun> = Vec::new();
        let mut skipped_empty_runs = 0;
        for run in &self.runs {
            let d = run.lock().unwrap();
            // Skip runs that never saw an on_eval call
            if d.best.is_none() {
                skipped_empty_runs += 1;
                continue;
            }
            let mut block = vec![DAT_HEADER.to_string()];
            for (e, y) in &d.rows { block.push(format!("{e} {y}")); }
            if let Some((e, y)) = d.last {
                if d.rows.last() != Some(&(e, y)) { block.push(format!("{e} {y}")); }
            }
            let (be, by) = d.best.unwrap();
            let mut entry = serde_json::json!({
                "instance": d.instance, "evals": d.evals,
                "best": {"evals": be, "y": by},
            });
            // NaN f_opt marks the legacy `start_run` path: omit all three
            // keys rather than ever serializing a NaN into JSON.
            if !d.f_opt.is_nan() {
                entry["seed"] = serde_json::json!(d.seed);
                entry["f_opt"] = serde_json::json!(d.f_opt);
                entry["budget"] = serde_json::json!(d.budget);
            }
            new_runs.push(NewRun { identity: (d.instance as u64, d.seed, d.budget), entry, block });
        }

        let meta_path = dir.join(format!("IOHprofiler_f{}_{}.json", self.fid, self.fname));

        // Values this logger expects the meta file's top-level fields to
        // carry, computed before `self.suite`/`self.fname`/`self.algo` are
        // moved into the freshly-built `meta` object below.
        let want_suite = serde_json::Value::String(self.suite.clone());
        let want_fname = serde_json::Value::String(self.fname.clone());
        let want_fid = serde_json::Value::from(self.fid);
        let want_algo = serde_json::Value::String(self.algo.clone());

        let mut scenarios: Vec<serde_json::Value> = if meta_path.exists() {
            let existing_text = std::fs::read_to_string(&meta_path)?;
            let existing: serde_json::Value = serde_json::from_str(&existing_text)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!(
                    "{}: could not parse existing meta file for merge: {e}", meta_path.display(),
                )))?;

            let mismatch = |field: &str, existing_val: &serde_json::Value, want_val: &serde_json::Value| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, format!(
                    "{}: existing meta file has {field} = {existing_val}, but this finish() call has \
                     {field} = {want_val} — one meta file is shared per (algo, fid) across all dimensions, \
                     so its top-level fields must agree on every finish()",
                    meta_path.display(),
                ))
            };
            if existing["suite"] != want_suite {
                return Err(mismatch("suite", &existing["suite"], &want_suite));
            }
            if existing["function_id"] != want_fid {
                return Err(mismatch("function_id", &existing["function_id"], &want_fid));
            }
            if existing["function_name"] != want_fname {
                return Err(mismatch("function_name", &existing["function_name"], &want_fname));
            }
            if existing["algorithm"]["name"] != want_algo {
                return Err(mismatch("algorithm.name", &existing["algorithm"]["name"], &want_algo));
            }

            existing["scenarios"].as_array().cloned().unwrap_or_default()
        } else {
            Vec::new()
        };

        // Merge this logger's runs into this dimension's scenario by run
        // identity (see the `finish()` doc's "sezgi decision" paragraph):
        // an existing entry for this dim gets its runs merged in-place
        // (matching identities REWRITTEN, new identities APPENDED); no
        // existing entry means this dim's scenario is new and its runs are
        // simply this logger's own.
        let dim_json = serde_json::Value::from(self.dim);
        let (merged_runs, merged_blocks): (Vec<serde_json::Value>, Vec<Vec<String>>) =
            match scenarios.iter().position(|s| s["dimension"] == dim_json) {
                Some(idx) => {
                    let existing_runs = scenarios[idx]["runs"].as_array().cloned().unwrap_or_default();
                    let existing_blocks = if dat_path.exists() {
                        split_dat_into_blocks(&std::fs::read_to_string(&dat_path)?)
                    } else {
                        Vec::new()
                    };

                    let mut consumed = vec![false; new_runs.len()];
                    let mut merged_runs = Vec::with_capacity(existing_runs.len());
                    let mut merged_blocks = Vec::with_capacity(existing_blocks.len());
                    for (i, existing_entry) in existing_runs.iter().enumerate() {
                        let ident = (
                            existing_entry["instance"].as_u64().unwrap_or(0),
                            existing_entry["seed"].as_u64().unwrap_or(0),
                            existing_entry["budget"].as_u64().unwrap_or(0),
                        );
                        match new_runs.iter().position(|nr| nr.identity == ident) {
                            Some(pos) => {
                                // Rewrite: this identity is present in the new
                                // logger's runs — take its entry/block instead
                                // of the existing one, in the same position.
                                merged_runs.push(new_runs[pos].entry.clone());
                                merged_blocks.push(new_runs[pos].block.clone());
                                consumed[pos] = true;
                            }
                            None => {
                                merged_runs.push(existing_entry.clone());
                                if let Some(b) = existing_blocks.get(i) {
                                    merged_blocks.push(b.clone());
                                }
                            }
                        }
                    }
                    // Append every new run whose identity wasn't already
                    // present (and thus wasn't consumed as a rewrite above).
                    for (i, nr) in new_runs.iter().enumerate() {
                        if !consumed[i] {
                            merged_runs.push(nr.entry.clone());
                            merged_blocks.push(nr.block.clone());
                        }
                    }
                    (merged_runs, merged_blocks)
                }
                None => (
                    new_runs.iter().map(|nr| nr.entry.clone()).collect(),
                    new_runs.iter().map(|nr| nr.block.clone()).collect(),
                ),
            };

        let mut dat = std::fs::File::create(&dat_path)?;
        for block in &merged_blocks {
            for line in block { writeln!(dat, "{line}")?; }
        }

        let new_scenario = serde_json::json!({
            "dimension": self.dim,
            "path": format!("{data_rel}/{dat_name}"),
            "runs": merged_runs,
        });
        match scenarios.iter().position(|s| s["dimension"] == dim_json) {
            Some(idx) => scenarios[idx] = new_scenario,
            None => scenarios.push(new_scenario),
        }

        let meta = serde_json::json!({
            "version": "sezgi-0.1",
            "suite": self.suite,
            "function_id": self.fid,
            "function_name": self.fname,
            "maximization": false,
            "algorithm": {"name": self.algo},
            "scenarios": scenarios,
        });
        std::fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)?;
        Ok(IohFinish { meta_path, skipped_empty_runs })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::problem::EvalObserver;

    #[test]
    fn writes_improvement_rows_and_meta() {
        let tmp = tempfile::tempdir().unwrap();
        let mut lg = IohLogger::new(tmp.path(), "de-rand-1", "sezgi-bbob",
                                    1, "Sphere", 5);
        {
            let mut obs = lg.start_run(1);
            obs.on_eval(1, 10.0, 10.0);
            obs.on_eval(2, 12.0, 10.0);  // no improvement → no row
            obs.on_eval(3, 4.0, 4.0);    // improvement → row
            obs.on_eval(4, 6.0, 4.0);    // final eval → always written
        }
        let fin = lg.finish().unwrap();
        let meta: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&fin.meta_path).unwrap()).unwrap();
        assert_eq!(meta["function_id"], 1);
        assert_eq!(meta["scenarios"][0]["runs"][0]["best"]["y"], 4.0);

        let dat = std::fs::read_to_string(
            tmp.path().join("de-rand-1/data_f1_Sphere/IOHprofiler_f1_DIM5.dat")).unwrap();
        let lines: Vec<&str> = dat.lines().collect();
        assert_eq!(lines[0], "\"evaluations\" \"raw_y\"");
        assert_eq!(lines[1], "1 10");
        assert_eq!(lines[2], "3 4");
        assert_eq!(lines[3], "4 4"); // final state row
    }

    #[test]
    fn multiple_runs_share_dat_file_with_headers() {
        let tmp = tempfile::tempdir().unwrap();
        let mut lg = IohLogger::new(tmp.path(), "a", "s", 3, "Rastrigin", 2);
        { let mut o = lg.start_run(1); o.on_eval(1, 5.0, 5.0); }
        { let mut o = lg.start_run(2); o.on_eval(1, 7.0, 7.0); }
        lg.finish().unwrap();
        let dat = std::fs::read_to_string(
            tmp.path().join("a/data_f3_Rastrigin/IOHprofiler_f3_DIM2.dat")).unwrap();
        assert_eq!(dat.matches("\"evaluations\"").count(), 2, "one header per run");
    }

    #[test]
    fn empty_run_skipped_not_null() {
        let tmp = tempfile::tempdir().unwrap();
        let mut lg = IohLogger::new(tmp.path(), "a", "s", 1, "Sphere", 3);
        { let _o = lg.start_run(1); }                    // no on_eval calls at all
        { let mut o = lg.start_run(2); o.on_eval(1, 5.0, 5.0); }
        let fin = lg.finish().unwrap();
        assert_eq!(fin.skipped_empty_runs, 1);
        let meta: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&fin.meta_path).unwrap()).unwrap();
        let runs = meta["scenarios"][0]["runs"].as_array().unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0]["instance"], 2);
        assert!(runs[0]["best"]["y"].is_f64(), "should not leak a null");
        let dat = std::fs::read_to_string(
            tmp.path().join("a/data_f1_Sphere/IOHprofiler_f1_DIM3.dat")).unwrap();
        assert_eq!(dat.matches("\"evaluations\"").count(), 1, "no header should be written for an empty run");
    }

    #[test]
    fn all_empty_runs_still_writes_valid_meta() {
        let tmp = tempfile::tempdir().unwrap();
        let mut lg = IohLogger::new(tmp.path(), "a", "s", 1, "Sphere", 3);
        { let _o = lg.start_run(1); }
        let fin = lg.finish().unwrap();
        assert_eq!(fin.skipped_empty_runs, 1);
        let meta: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&fin.meta_path).unwrap()).unwrap();
        assert_eq!(meta["scenarios"][0]["runs"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn start_run_with_writes_seed_and_f_opt_and_budget() {
        let tmp = tempfile::tempdir().unwrap();
        let mut lg = IohLogger::new(tmp.path(), "a", "s", 1, "Sphere", 3);
        { let mut o = lg.start_run_with(1, 42, -12.5, 300); o.on_eval(1, 5.0, 5.0); }
        let fin = lg.finish().unwrap();
        let meta: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&fin.meta_path).unwrap()).unwrap();
        let run0 = &meta["scenarios"][0]["runs"][0];
        assert_eq!(run0["seed"], 42);
        assert_eq!(run0["f_opt"], -12.5);
        assert_eq!(run0["budget"], 300);
    }

    #[test]
    fn finish_merges_multiple_dims_of_same_algo_fid_into_one_meta() {
        let tmp = tempfile::tempdir().unwrap();

        let mut lg5 = IohLogger::new(tmp.path(), "de-rand-1", "sezgi-bbob", 1, "Sphere", 5);
        { let mut o = lg5.start_run(1); o.on_eval(1, 10.0, 10.0); }
        let fin5 = lg5.finish().unwrap();

        let mut lg10 = IohLogger::new(tmp.path(), "de-rand-1", "sezgi-bbob", 1, "Sphere", 10);
        { let mut o = lg10.start_run(1); o.on_eval(1, 20.0, 20.0); }
        let fin10 = lg10.finish().unwrap();

        // Both finish() calls target the same (algo, fid) meta path.
        assert_eq!(fin5.meta_path, fin10.meta_path);

        let meta: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&fin10.meta_path).unwrap()).unwrap();
        let scenarios = meta["scenarios"].as_array().unwrap();
        assert_eq!(scenarios.len(), 2, "both dims must survive in the merged meta");

        let dims: Vec<u64> = scenarios.iter().map(|s| s["dimension"].as_u64().unwrap()).collect();
        assert!(dims.contains(&5) && dims.contains(&10), "got dims {dims:?}");

        for s in scenarios {
            let dim = s["dimension"].as_u64().unwrap();
            let path = s["path"].as_str().unwrap();
            assert!(path.contains(&format!("DIM{dim}")), "path {path} must embed its own dim {dim}");
            // Each dim's .dat file must actually exist where the meta says it does.
            assert!(tmp.path().join("de-rand-1").join(path).exists());
        }

        // Re-finishing dim 5 replaces its scenario, not duplicates it.
        let mut lg5_again = IohLogger::new(tmp.path(), "de-rand-1", "sezgi-bbob", 1, "Sphere", 5);
        { let mut o = lg5_again.start_run(1); o.on_eval(1, 99.0, 99.0); }
        let fin5_again = lg5_again.finish().unwrap();

        let meta: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&fin5_again.meta_path).unwrap()).unwrap();
        let scenarios = meta["scenarios"].as_array().unwrap();
        assert_eq!(scenarios.len(), 2, "re-finishing dim 5 must replace, not append");
        let sc5 = scenarios.iter().find(|s| s["dimension"] == 5).unwrap();
        assert_eq!(sc5["runs"][0]["best"]["y"], 99.0, "dim 5's scenario must reflect the re-finish");
        let sc10 = scenarios.iter().find(|s| s["dimension"] == 10).unwrap();
        assert_eq!(sc10["runs"][0]["best"]["y"], 20.0, "dim 10's scenario must be untouched");
    }

    #[test]
    fn finish_appends_new_identity_within_same_dim() {
        let tmp = tempfile::tempdir().unwrap();

        let mut lg1 = IohLogger::new(tmp.path(), "de-rand-1", "sezgi-bbob", 1, "Sphere", 5);
        { let mut o = lg1.start_run_with(1, 0, -1.0, 100); o.on_eval(1, 10.0, 10.0); }
        lg1.finish().unwrap();

        // A second finish() call into the SAME (algo, fid, dim) but with a
        // DIFFERENT run identity (instance 2, not instance 1) must be
        // APPENDED, not replace the first — this is the mechanism
        // `bbob_records`-style multi-seed sweeps rely on (a fresh
        // IohLogger/finish() per run, into the same target).
        let mut lg2 = IohLogger::new(tmp.path(), "de-rand-1", "sezgi-bbob", 1, "Sphere", 5);
        { let mut o = lg2.start_run_with(2, 0, -1.0, 100); o.on_eval(1, 20.0, 20.0); }
        let fin2 = lg2.finish().unwrap();

        let meta: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&fin2.meta_path).unwrap()).unwrap();
        let runs = meta["scenarios"][0]["runs"].as_array().unwrap();
        assert_eq!(runs.len(), 2, "a new run identity must append, not replace");
        let instances: Vec<u64> = runs.iter().map(|r| r["instance"].as_u64().unwrap()).collect();
        assert!(instances.contains(&1) && instances.contains(&2), "got instances {instances:?}");

        let dat = std::fs::read_to_string(
            tmp.path().join("de-rand-1/data_f1_Sphere/IOHprofiler_f1_DIM5.dat")).unwrap();
        assert_eq!(dat.matches("\"evaluations\"").count(), 2,
            "both runs' blocks must survive in the .dat file");
    }

    #[test]
    fn finish_rewrites_matching_identity_leaving_one_copy() {
        let tmp = tempfile::tempdir().unwrap();

        let mut lg1 = IohLogger::new(tmp.path(), "de-rand-1", "sezgi-bbob", 1, "Sphere", 5);
        { let mut o = lg1.start_run_with(1, 7, -1.0, 100); o.on_eval(1, 10.0, 10.0); }
        lg1.finish().unwrap();

        let mut lg2 = IohLogger::new(tmp.path(), "de-rand-1", "sezgi-bbob", 1, "Sphere", 5);
        { let mut o = lg2.start_run_with(2, 8, -1.0, 100); o.on_eval(1, 20.0, 20.0); }
        lg2.finish().unwrap();

        // Re-finishing with the SAME identity (instance 1, seed 7,
        // budget 100) as the first call must REWRITE that run in place,
        // leaving exactly one copy of it (not duplicating), while the
        // unrelated instance-2/seed-8 run from the second call is
        // untouched.
        let mut lg1_again = IohLogger::new(tmp.path(), "de-rand-1", "sezgi-bbob", 1, "Sphere", 5);
        { let mut o = lg1_again.start_run_with(1, 7, -1.0, 100); o.on_eval(1, 99.0, 99.0); }
        let fin_again = lg1_again.finish().unwrap();

        let meta: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&fin_again.meta_path).unwrap()).unwrap();
        let runs = meta["scenarios"][0]["runs"].as_array().unwrap();
        assert_eq!(runs.len(), 2, "a matching identity must rewrite in place, not duplicate");

        let matching: Vec<&serde_json::Value> = runs.iter()
            .filter(|r| r["instance"] == 1 && r["seed"] == 7).collect();
        assert_eq!(matching.len(), 1, "exactly one copy of the rewritten identity");
        assert_eq!(matching[0]["best"]["y"], 99.0, "rewritten run must reflect the re-finish");

        let untouched = runs.iter().find(|r| r["instance"] == 2).unwrap();
        assert_eq!(untouched["best"]["y"], 20.0, "unrelated identity must be untouched by the rewrite");

        let dat = std::fs::read_to_string(
            tmp.path().join("de-rand-1/data_f1_Sphere/IOHprofiler_f1_DIM5.dat")).unwrap();
        assert_eq!(dat.matches("\"evaluations\"").count(), 2,
            "still exactly two .dat blocks after rewrite (no duplication)");
    }

    #[test]
    fn finish_errors_on_mismatched_top_level_fields() {
        let tmp = tempfile::tempdir().unwrap();

        let mut lg = IohLogger::new(tmp.path(), "de-rand-1", "sezgi-bbob", 1, "Sphere", 5);
        { let mut o = lg.start_run(1); o.on_eval(1, 10.0, 10.0); }
        lg.finish().unwrap();

        // Same (algo, fid) meta path, but a different algorithm name inside
        // the logger's `algorithm.name` field — simulated here by writing
        // to the same dir/fid/fname with a mismatched `suite`.
        let mut lg_bad = IohLogger::new(tmp.path(), "de-rand-1", "other-suite", 1, "Sphere", 10);
        { let mut o = lg_bad.start_run(1); o.on_eval(1, 1.0, 1.0); }
        let err = lg_bad.finish().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("suite"), "error must name the mismatched field: {msg}");
    }

    #[test]
    fn legacy_start_run_omits_seed_and_f_opt_and_budget_keys() {
        let tmp = tempfile::tempdir().unwrap();
        let mut lg = IohLogger::new(tmp.path(), "a", "s", 1, "Sphere", 3);
        { let mut o = lg.start_run(1); o.on_eval(1, 5.0, 5.0); }
        let fin = lg.finish().unwrap();
        let meta: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&fin.meta_path).unwrap()).unwrap();
        let run0 = meta["scenarios"][0]["runs"][0].as_object().unwrap();
        assert!(!run0.contains_key("seed"), "legacy run must not carry a seed key");
        assert!(!run0.contains_key("f_opt"), "legacy run must not carry an f_opt key (NaN never serialized)");
        assert!(!run0.contains_key("budget"), "legacy run must not carry a budget key");
    }
}
