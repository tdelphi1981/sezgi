use sezgi_core::problem::EvalObserver;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
struct RunData {
    instance: u32,
    rows: Vec<(u64, f64)>,       // improvement rows
    last: Option<(u64, f64)>,    // final eval (always written)
    evals: u64,
    best: Option<(u64, f64)>,
}

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
        let improved = self.prev_best.map_or(true, |p| best_so_far < p);
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

    pub fn start_run(&mut self, instance: u32) -> IohRunObserver {
        let data = Arc::new(Mutex::new(RunData {
            instance, rows: vec![], last: None, evals: 0, best: None }));
        self.runs.push(data.clone());
        IohRunObserver { data, prev_best: None }
    }

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
            runs_json.push(serde_json::json!({
                "instance": d.instance, "evals": d.evals,
                "best": {"evals": be, "y": by},
            }));
        }

        let meta = serde_json::json!({
            "version": "sezgi-0.1",
            "suite": self.suite,
            "function_id": self.fid,
            "function_name": self.fname,
            "maximization": false,
            "algorithm": {"name": self.algo},
            "scenarios": [{
                "dimension": self.dim,
                "path": format!("{data_rel}/{dat_name}"),
                "runs": runs_json,
            }],
        });
        let meta_path = dir.join(format!("IOHprofiler_f{}_{}.json", self.fid, self.fname));
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
}
