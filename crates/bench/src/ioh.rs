use sezgi_core::problem::EvalObserver;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
struct RunData {
    instance: u32,
    seed: u64,
    f_opt: f64,                  // f64::NAN means "unknown" (legacy `start_run` path):
                                  // never serialized — see `finish()`.
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

    /// Legacy entry point (M2c): no seed/f_opt known at call time. Delegates
    /// to [`Self::start_run_with`] with `seed = 0` and `f_opt = f64::NAN`;
    /// `finish()` omits the `"seed"`/`"f_opt"` meta keys for such a run
    /// (NaN is never serialized into JSON), keeping the meta backward-
    /// compatible with pre-M2d readers.
    pub fn start_run(&mut self, instance: u32) -> IohRunObserver {
        self.start_run_with(instance, 0, f64::NAN)
    }

    /// Starts a run with its master `seed` and the problem instance's known
    /// `f_opt`, both of which `finish()` writes into that run's meta entry.
    /// Pass `f_opt = f64::NAN` only via [`Self::start_run`] (legacy path);
    /// a caller that knows the real f_opt must always pass it here.
    pub fn start_run_with(&mut self, instance: u32, seed: u64, f_opt: f64) -> IohRunObserver {
        let data = Arc::new(Mutex::new(RunData {
            instance, seed, f_opt, rows: vec![], last: None, evals: 0, best: None }));
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
    /// array, keyed by `dimension`: an existing entry for this logger's
    /// `dim` is REPLACED (re-`finish()`ing the same dimension is
    /// idempotent, not additive); any other dimension's entry is left
    /// untouched and this one is appended. The file's other top-level
    /// fields (`suite`, `function_id`, `function_name`, `algorithm.name`)
    /// must already agree with this logger's — a mismatch (e.g. two
    /// different algorithms writing into the same `(fid, fname)` meta
    /// path) is an error rather than a silent clobber.
    pub fn finish(self) -> std::io::Result<IohFinish> {
        let dir = self.root.join(&self.algo);
        let data_rel = format!("data_f{}_{}", self.fid, self.fname);
        let data_dir = dir.join(&data_rel);
        std::fs::create_dir_all(&data_dir)?;

        let dat_name = format!("IOHprofiler_f{}_DIM{}.dat", self.fid, self.dim);
        let mut dat = std::fs::File::create(data_dir.join(&dat_name))?;
        let mut runs_json = vec![];
        let mut skipped_empty_runs = 0;
        for run in &self.runs {
            let d = run.lock().unwrap();
            // Skip runs that never saw an on_eval call
            if d.best.is_none() {
                skipped_empty_runs += 1;
                continue;
            }
            writeln!(dat, "\"evaluations\" \"raw_y\"")?;
            for (e, y) in &d.rows { writeln!(dat, "{e} {y}")?; }
            if let Some((e, y)) = d.last {
                if d.rows.last() != Some(&(e, y)) { writeln!(dat, "{e} {y}")?; }
            }
            let (be, by) = d.best.unwrap();
            let mut entry = serde_json::json!({
                "instance": d.instance, "evals": d.evals,
                "best": {"evals": be, "y": by},
            });
            // NaN f_opt marks the legacy `start_run` path: omit both keys
            // rather than ever serializing a NaN into JSON.
            if !d.f_opt.is_nan() {
                entry["seed"] = serde_json::json!(d.seed);
                entry["f_opt"] = serde_json::json!(d.f_opt);
            }
            runs_json.push(entry);
        }

        let new_scenario = serde_json::json!({
            "dimension": self.dim,
            "path": format!("{data_rel}/{dat_name}"),
            "runs": runs_json,
        });

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

        // Replace this dimension's scenario if the file already had one
        // (a re-finish of the same dim), otherwise append.
        let dim_json = serde_json::Value::from(self.dim);
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
    fn start_run_with_writes_seed_and_f_opt() {
        let tmp = tempfile::tempdir().unwrap();
        let mut lg = IohLogger::new(tmp.path(), "a", "s", 1, "Sphere", 3);
        { let mut o = lg.start_run_with(1, 42, -12.5); o.on_eval(1, 5.0, 5.0); }
        let fin = lg.finish().unwrap();
        let meta: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&fin.meta_path).unwrap()).unwrap();
        let run0 = &meta["scenarios"][0]["runs"][0];
        assert_eq!(run0["seed"], 42);
        assert_eq!(run0["f_opt"], -12.5);
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
    fn legacy_start_run_omits_seed_and_f_opt_keys() {
        let tmp = tempfile::tempdir().unwrap();
        let mut lg = IohLogger::new(tmp.path(), "a", "s", 1, "Sphere", 3);
        { let mut o = lg.start_run(1); o.on_eval(1, 5.0, 5.0); }
        let fin = lg.finish().unwrap();
        let meta: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&fin.meta_path).unwrap()).unwrap();
        let run0 = meta["scenarios"][0]["runs"][0].as_object().unwrap();
        assert!(!run0.contains_key("seed"), "legacy run must not carry a seed key");
        assert!(!run0.contains_key("f_opt"), "legacy run must not carry an f_opt key (NaN never serialized)");
    }
}
