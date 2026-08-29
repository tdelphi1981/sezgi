use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sezgi_bench::{
    coco_export as bench_coco_export, default_targets as bench_default_targets,
    ecdf as bench_ecdf, ecdf_per_algo as bench_ecdf_per_algo, ioh_records as bench_ioh_records,
    per_budget_packages as bench_per_budget_packages, read_ioh_root,
    results_matrix as bench_results_matrix, run_experiment_logged, run_experiment_parallel,
    run_experiment_sequential, run_experiment_with_checkpoint, Aggregate, EcdfCurve, EvalSession,
    ExperimentSpec, IohLogger, RunKey, RunRecord,
};
use sezgi_components::{presets, register_builtins};
use sezgi_core::component::Registry;
use sezgi_core::dist::Distribution;
use sezgi_core::engine::{Engine, RunConfig};
use sezgi_core::problem::Problem;
use sezgi_core::space::{Block, BlockValues, Genotype, SearchSpace};
use sezgi_core::spec::AlgorithmSpec;
use sezgi_problems::BbobProblem;
use sezgi_stats::{
    bayesian_plackett_luce, bayesian_signed_rank, cliffs_delta, cliffs_magnitude, friedman,
    paper_package, plackett_luce, wilcoxon_signed_rank, PaperPackage, WilcoxonMethod,
};
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;

enum Inner {
    Bbob(BbobProblem),
    Callable { f: Py<PyAny>, space: SearchSpace },
}

#[pyclass(name = "Problem")]
struct PyProblem { inner: Inner }

struct CallableProblem<'a> { f: &'a Py<PyAny>, space: &'a SearchSpace }

impl Problem for CallableProblem<'_> {
    fn space(&self) -> &SearchSpace { self.space }
    fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<f64> {
        Python::with_gil(|py| {
            let rows: Vec<Vec<f64>> = pop.iter().map(|g| match &g.blocks[0] {
                BlockValues::Float(v) => v.clone(),
                _ => unreachable!("from_callable only builds Float spaces"),
            }).collect();
            // The population is written into a SINGLE (n, d) float64 numpy array;
            // a copy happens only at the Rust->numpy boundary (zero-copy return: PyReadonlyArray1).
            let arr = match PyArray2::from_vec2(py, &rows) {
                Ok(a) => a,
                Err(e) => panic::panic_any(PyValueError::new_err(
                    format!("could not convert population to numpy array: {e}"))),
            };
            let out = match self.f.call1(py, (arr,)) {
                Ok(o) => o,
                // A Python exception inside the callback is thrown here as a
                // panic; `catch_unwind` inside `solve` downcasts it and
                // returns it to the caller as the original exception.
                Err(e) => panic::panic_any(e),
            };
            let bound = out.bind(py);
            if let Ok(ro) = bound.extract::<PyReadonlyArray1<f64>>() {
                ro.as_array().iter().copied().collect()
            } else {
                match bound.extract::<Vec<f64>>() {
                    Ok(v) => v,
                    Err(e) => panic::panic_any(e),
                }
            }
        })
    }
}

/// Releases the GIL for the duration of the run (`py.allow_threads`) and wraps
/// the run in `catch_unwind`: an exception thrown from the callback via
/// `panic_any(PyErr)` is caught and returned in its original form; any other
/// panic — such as the core's length assert — is converted to a
/// `PyRuntimeError` carrying its message. This way no panic leaks into
/// Python as a `PanicException`.
fn run_with_bridge<F, T>(py: Python<'_>, f: F) -> PyResult<T>
where
    F: FnOnce() -> PyResult<T> + Send,
    T: Send,
{
    // NOTE: do not swap the global panic hook to suppress panic stderr
    // output — it is process-global and races under concurrent solve calls.
    match py.allow_threads(|| panic::catch_unwind(AssertUnwindSafe(f))) {
        Ok(inner) => inner,
        Err(payload) => match payload.downcast::<PyErr>() {
            Ok(pyerr) => Err(*pyerr),
            Err(payload) => {
                let msg = payload
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown panic".to_string());
                Err(PyRuntimeError::new_err(msg))
            }
        },
    }
}

#[pyfunction]
fn bbob(fid: u32, dim: usize, instance: u32) -> PyResult<PyProblem> {
    Ok(PyProblem { inner: Inner::Bbob(
        BbobProblem::new(fid, dim, instance)
            .map_err(|e| PyValueError::new_err(e.to_string()))?) })
}

#[pyfunction]
fn from_callable(f: Py<PyAny>, lo: f64, hi: f64, dim: usize) -> PyResult<PyProblem> {
    let space = SearchSpace::new(vec![Block::Float { lo, hi, n: dim }])
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyProblem { inner: Inner::Callable { f, space } })
}

/// Raised by every [`PyEvalSession`] method once the session has been
/// [`PyEvalSession::finish`]ed (including a second call to `finish` itself).
fn session_finished_err() -> PyErr {
    PyValueError::new_err("session finished")
}

/// Python binding for [`sezgi_bench::EvalSession`] — the ask/tell core
/// behind the spec's engine-inside-out promise: an external (here, pure
/// Python) algorithm generates candidate points and this session stays the
/// sole keeper of evaluation, tamper-proof counting, best-tracking and IOH
/// logging.
///
/// `finish()` consumes the underlying Rust session (matching its own
/// consuming signature); since `#[pymethods]` cannot take `self` by value
/// through a Python handle, the consuming step is modeled with an
/// `Option<EvalSession>` inner slot that `finish` takes, leaving `None`
/// behind. Every method (including a second `finish()`) then raises
/// `ValueError("session finished")` if called afterward.
///
/// `with_log` is wired up ONLY in the constructor, applied before any
/// `evaluate()` call is possible from Python — so `EvalSession::with_log`'s
/// own "called after evaluation has started" guard (`LogAfterEval`) is
/// unreachable through this binding by construction; it exists purely as a
/// Rust-level invariant, not a case Python callers can trigger.
#[pyclass(name = "EvalSession")]
struct PyEvalSession {
    inner: Option<EvalSession>,
}

#[pymethods]
impl PyEvalSession {
    #[new]
    #[pyo3(signature = (fid, dim, instance, budget, log_dir=None, algo_name="custom", seed=0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        fid: u32,
        dim: usize,
        instance: u32,
        budget: u64,
        log_dir: Option<&str>,
        algo_name: &str,
        seed: u64,
    ) -> PyResult<Self> {
        let mut session = EvalSession::new_bbob(fid, dim, instance, budget)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if let Some(dir) = log_dir {
            session = session
                .with_log(Path::new(dir), algo_name, seed)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
        Ok(Self { inner: Some(session) })
    }

    /// Batch-evaluates `xs` (a list of rows, each a list of `dim` floats).
    /// All-or-nothing: on any error (dimension mismatch, a non-finite
    /// coordinate, or budget overrun) nothing is counted.
    fn evaluate(&mut self, xs: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let session = self.inner.as_mut().ok_or_else(session_finished_err)?;
        session.evaluate(&xs).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn evals_used(&self) -> PyResult<u64> {
        Ok(self.inner.as_ref().ok_or_else(session_finished_err)?.evals_used())
    }

    fn budget(&self) -> PyResult<u64> {
        Ok(self.inner.as_ref().ok_or_else(session_finished_err)?.budget())
    }

    /// `(x, f)` of the best evaluation seen so far, or `None` if nothing has
    /// been evaluated yet.
    fn best(&self) -> PyResult<Option<(Vec<f64>, f64)>> {
        let session = self.inner.as_ref().ok_or_else(session_finished_err)?;
        Ok(session.best().map(|(x, f)| (x.to_vec(), f)))
    }

    fn f_opt(&self) -> PyResult<f64> {
        Ok(self.inner.as_ref().ok_or_else(session_finished_err)?.f_opt())
    }

    /// Flushes the IOH log (if logging was enabled) and consumes the
    /// session. Any method call afterward, including a second `finish()`,
    /// raises `ValueError("session finished")`.
    fn finish(&mut self) -> PyResult<()> {
        let session = self.inner.take().ok_or_else(session_finished_err)?;
        session.finish().map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

fn registry() -> Registry {
    let mut r = Registry::new();
    register_builtins(&mut r);
    r
}

#[pyfunction]
#[pyo3(signature = (spec_json, problem, master_seed=0, run_id=0, log_dir=None, algo_name=None))]
fn solve(py: Python<'_>, spec_json: &str, problem: &PyProblem, master_seed: u64,
         run_id: u64, log_dir: Option<&str>, algo_name: Option<&str>)
         -> PyResult<Py<PyDict>> {
    let spec = AlgorithmSpec::from_json(spec_json)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let reg = registry();

    let run = |p: &dyn Problem, obs| -> PyResult<_> {
        let engine = Engine::from_spec(&spec, &reg, p.space())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        engine.run(p, RunConfig { master_seed, run_id }, obs)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    };

    let (result, skipped_empty_runs_opt) = match &problem.inner {
        Inner::Bbob(p) => {
            if let Some(dir) = log_dir {
                let name = algo_name.unwrap_or(&spec.name).to_string();
                let mut lg = IohLogger::new(std::path::Path::new(dir), &name,
                    "sezgi-bbob", p.fid(), p.name(),
                    p.space().dim());
                let obs = lg.start_run_with(p.instance, master_seed, p.f_opt(), spec.termination.budget);
                // Logger observer is Rust-native (no GIL needed); GIL is released for the run.
                let r = run_with_bridge(py, || run(p, Some(Box::new(obs))))?;
                let fin = lg.finish().map_err(|e| PyValueError::new_err(e.to_string()))?;
                (r, Some(fin.skipped_empty_runs as u64))
            } else { (run_with_bridge(py, || run(p, None))?, None) }
        }
        Inner::Callable { f, space } => {
            if log_dir.is_some() {
                return Err(PyValueError::new_err(
                    "log_dir is only supported for builtin (bbob) problems"));
            }
            let cp = CallableProblem { f, space };
            // GIL is released for the run; the callback reacquires it each
            // batch via Python::with_gil (standard PyO3 pattern).
            (run_with_bridge(py, || run(&cp, None))?, None)
        }
    };

    let d = PyDict::new(py);
    d.set_item("best_f", result.best_f)?;
    let BlockValues::Float(xs) = &result.best_x.blocks[0] else {
        return Err(PyValueError::new_err("unexpected genotype"));
    };
    d.set_item("best_x", PyList::new(py, xs)?)?;
    d.set_item("evals_used", result.evals_used)?;
    d.set_item("iterations", result.iterations)?;
    if let Some(skipped) = skipped_empty_runs_opt {
        d.set_item("skipped_empty_runs", skipped)?;
    }
    Ok(d.into())
}

/// Builds a record dict from a [`RunRecord`], with the SAME shape
/// `run_experiment` returns (keys: `algo, fid, dim, instance, seed, budget,
/// best_f, f_opt, gap, evals_used, wall_secs`). Shared by `run_experiment`
/// and `read_ioh_records` so a disk-reconstructed record and a freshly-run
/// one are interchangeable to any downstream consumer (e.g.
/// `per_budget_packages`).
fn record_to_dict<'py>(py: Python<'py>, r: &RunRecord) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("algo", &r.key.algo)?;
    d.set_item("fid", r.key.fid)?;
    d.set_item("dim", r.key.dim)?;
    d.set_item("instance", r.key.instance)?;
    d.set_item("seed", r.key.seed)?;
    d.set_item("budget", r.key.budget)?;
    d.set_item("best_f", r.best_f)?;
    d.set_item("f_opt", r.f_opt)?;
    d.set_item("gap", r.best_f - r.f_opt)?;
    d.set_item("evals_used", r.evals_used)?;
    d.set_item("wall_secs", r.wall_secs)?;
    Ok(d)
}

/// Runs an [`ExperimentSpec`] (parsed from `spec_toml`) and returns its
/// `RunRecord`s as a list of dicts. `journal=None` runs via
/// `run_experiment_parallel`/`run_experiment_sequential` (chosen by
/// `parallel`); `journal=Some(path)` runs via
/// `run_experiment_with_checkpoint`, which resumes from — and appends to —
/// an existing journal file at `path`. The GIL is released
/// (`py.allow_threads`) for the duration of the run: experiment problems
/// are bbob-only (no Python callbacks), so no Python object is touched
/// while the GIL is released.
///
/// The journal's spec hash is computed by `run_experiment_with_checkpoint`
/// from `spec` (the already-parsed `ExperimentSpec`), not from the raw
/// `spec_toml` text, so whitespace/comment-only edits to `spec_toml` never
/// invalidate a journal — see `crates/bench/src/checkpoint.rs`.
///
/// `log_dir`: when given, every run this call actually EXECUTES is also
/// logged in IOH-profiler format under that directory (via
/// `run_experiment_logged` when there is no `journal`, or via
/// `run_experiment_with_checkpoint`'s own `log_dir` pass-through when there
/// is — see that function's doc comment: a run resumed from the journal was
/// executed in a PRIOR process and is never re-logged).
#[pyfunction]
#[pyo3(signature = (spec_toml, journal=None, parallel=true, threads=None, log_dir=None))]
fn run_experiment(
    py: Python<'_>,
    spec_toml: &str,
    journal: Option<&str>,
    parallel: bool,
    threads: Option<usize>,
    log_dir: Option<&str>,
) -> PyResult<Py<PyList>> {
    let spec = ExperimentSpec::from_toml(spec_toml)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let log_dir_path = log_dir.map(Path::new);

    let records = if let Some(journal_path) = journal {
        let path = Path::new(journal_path);
        py.allow_threads(|| {
            run_experiment_with_checkpoint(&spec, path, parallel, threads, log_dir_path)
        })
    } else if let Some(dir) = log_dir_path {
        py.allow_threads(|| run_experiment_logged(&spec, dir, parallel, threads).map(|(r, _)| r))
    } else if parallel {
        py.allow_threads(|| run_experiment_parallel(&spec, threads))
    } else {
        py.allow_threads(|| run_experiment_sequential(&spec, |_| {}))
    }
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let mut rows = Vec::with_capacity(records.len());
    for r in &records {
        rows.push(record_to_dict(py, r)?);
    }
    Ok(PyList::new(py, rows)?.into())
}

// ---------------------------------------------------------------------
// Analysis cluster bindings (M2d-2 Task 7): on-disk IOH archives ->
// records / ECDF curves / COCO export.
// ---------------------------------------------------------------------

/// Reconstructs [`RunRecord`]s from an on-disk IOH archive at `log_root`
/// (as written by `run_experiment(..., log_dir=...)` or `solve(...,
/// log_dir=...)`), one record per `(run, budget)` pair — see
/// [`sezgi_bench::ioh_records`]'s doc comment for the exact `best_f`/
/// `evals_used` semantics and the curtailed-view-vs-independent-run
/// distinction for budgets smaller than a run's logged budget.
///
/// Returns the SAME record-dict shape `run_experiment` returns (via
/// [`record_to_dict`]), so `per_budget_packages`/`results_matrix` accept it
/// unchanged.
#[pyfunction]
fn read_ioh_records(py: Python<'_>, log_root: &str, budgets: Vec<u64>) -> PyResult<Py<PyList>> {
    let scenarios =
        read_ioh_root(Path::new(log_root)).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let records = bench_ioh_records(&scenarios, &budgets)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let mut rows = Vec::with_capacity(records.len());
    for r in &records {
        rows.push(record_to_dict(py, r)?);
    }
    Ok(PyList::new(py, rows)?.into())
}

/// Builds an `{"evals": [...], "proportion": [...]}` dict from an
/// [`EcdfCurve`].
fn ecdf_curve_to_dict<'py>(py: Python<'py>, curve: &EcdfCurve) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("evals", PyList::new(py, &curve.evals)?)?;
    d.set_item("proportion", PyList::new(py, &curve.proportion)?)?;
    Ok(d)
}

/// ECDF/anytime curve(s) over the IOH archive at `log_root` — see
/// [`sezgi_bench::ecdf`]/[`sezgi_bench::ecdf_per_algo`].
///
/// `targets`: precision targets; `None` uses [`sezgi_bench::default_targets`]
/// (the COCO-convention 51-value set).
/// `per_algo`: `True` (default) returns a list of `(algo, curve_dict)` pairs
/// (first-appearance order, one curve per distinct algo in the archive);
/// `False` returns a single pooled `curve_dict` over every scenario.
/// Each `curve_dict` is `{"evals": [...], "proportion": [...]}`.
#[pyfunction]
#[pyo3(signature = (log_root, targets=None, per_algo=true))]
fn ecdf(py: Python<'_>, log_root: &str, targets: Option<Vec<f64>>, per_algo: bool) -> PyResult<Py<PyAny>> {
    let scenarios =
        read_ioh_root(Path::new(log_root)).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let targets = targets.unwrap_or_else(bench_default_targets);

    if per_algo {
        let curves = bench_ecdf_per_algo(&scenarios, &targets)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let out = PyList::empty(py);
        for (algo, curve) in &curves {
            // Tuple, not a 2-element list, per the docstring ("a list of
            // `(algo, curve_dict)` pairs").
            let row = (algo.clone(), ecdf_curve_to_dict(py, curve)?);
            out.append(row)?;
        }
        Ok(out.into_any().unbind())
    } else {
        let curve = bench_ecdf(&scenarios, &targets)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(ecdf_curve_to_dict(py, &curve)?.into_any().unbind())
    }
}

/// Exports the IOH archive at `log_root` as a COCO/BBOB "old format"
/// archive rooted at `out_dir` — see [`sezgi_bench::coco_export`]. Returns
/// the list of written file paths (as strings), sorted for determinism.
#[pyfunction]
fn coco_export(py: Python<'_>, log_root: &str, out_dir: &str) -> PyResult<Py<PyList>> {
    let scenarios =
        read_ioh_root(Path::new(log_root)).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let written = bench_coco_export(&scenarios, Path::new(out_dir))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let paths: Vec<String> = written.iter().map(|p| p.display().to_string()).collect();
    Ok(PyList::new(py, paths)?.into())
}

// ---------------------------------------------------------------------
// Statistics bindings (sezgi.stats)
// ---------------------------------------------------------------------

/// Extracts a `Vec<f64>` from either a Python list/sequence of floats or a
/// 1-D numpy float64 array (tried first, falling back to plain `extract`).
fn extract_f64_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(ro) = obj.extract::<PyReadonlyArray1<f64>>() {
        return Ok(ro.as_array().iter().copied().collect());
    }
    obj.extract::<Vec<f64>>()
}

/// Extracts a results matrix (`Vec<Vec<f64>>`) from either a Python list of
/// lists (rows may themselves be numpy arrays) or a 2-D numpy float64
/// array.
fn extract_matrix(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
    if let Ok(ro) = obj.extract::<PyReadonlyArray2<f64>>() {
        let arr = ro.as_array();
        let mut out = Vec::with_capacity(arr.nrows());
        for row in arr.rows() {
            out.push(row.to_vec());
        }
        return Ok(out);
    }
    let mut result = Vec::new();
    for item in obj.try_iter()? {
        result.push(extract_f64_vec(&item?)?);
    }
    Ok(result)
}

#[pyfunction]
fn stats_friedman(py: Python<'_>, results: &Bound<'_, PyAny>) -> PyResult<Py<PyDict>> {
    let matrix = extract_matrix(results)?;
    let r = friedman(&matrix).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let d = PyDict::new(py);
    d.set_item("statistic", r.statistic)?;
    d.set_item("p_value", r.p_value)?;
    d.set_item("mean_ranks", PyList::new(py, &r.mean_ranks)?)?;
    Ok(d.into())
}

#[pyfunction]
fn stats_wilcoxon(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
) -> PyResult<Py<PyDict>> {
    let av = extract_f64_vec(a)?;
    let bv = extract_f64_vec(b)?;
    let r = wilcoxon_signed_rank(&av, &bv).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let d = PyDict::new(py);
    d.set_item("w_statistic", r.w_statistic)?;
    d.set_item("z", r.z)?;
    d.set_item("p_value", r.p_value)?;
    d.set_item("n_effective", r.n_effective)?;
    d.set_item(
        "method",
        match r.method {
            WilcoxonMethod::Exact => "exact",
            WilcoxonMethod::NormalApprox => "normal_approx",
        },
    )?;
    Ok(d.into())
}

#[pyfunction]
fn stats_cliffs_delta(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<f64> {
    let av = extract_f64_vec(a)?;
    let bv = extract_f64_vec(b)?;
    cliffs_delta(&av, &bv).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn stats_cliffs_magnitude(delta: f64) -> &'static str {
    cliffs_magnitude(delta)
}

#[pyfunction]
#[pyo3(signature = (a, b, rope=0.0, samples=20000, seed=1))]
fn stats_bayesian_signed_rank(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    rope: f64,
    samples: u64,
    seed: u64,
) -> PyResult<Py<PyDict>> {
    let av = extract_f64_vec(a)?;
    let bv = extract_f64_vec(b)?;
    let r = bayesian_signed_rank(&av, &bv, rope, samples, seed)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let d = PyDict::new(py);
    d.set_item("p_left", r.p_left)?;
    d.set_item("p_rope", r.p_rope)?;
    d.set_item("p_right", r.p_right)?;
    Ok(d.into())
}

#[pyfunction]
fn stats_plackett_luce(py: Python<'_>, rankings: Vec<Vec<usize>>) -> PyResult<Py<PyDict>> {
    let r = plackett_luce(&rankings).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let d = PyDict::new(py);
    d.set_item("worths", PyList::new(py, &r.worths)?)?;
    d.set_item("p_best", PyList::new(py, &r.p_best)?)?;
    d.set_item("iterations", r.iterations)?;
    Ok(d.into())
}

#[pyfunction]
#[pyo3(signature = (rankings, samples=2000, burn_in=500, seed=1))]
fn stats_bayesian_plackett_luce(
    py: Python<'_>,
    rankings: Vec<Vec<usize>>,
    samples: u64,
    burn_in: u64,
    seed: u64,
) -> PyResult<Py<PyDict>> {
    let r = bayesian_plackett_luce(&rankings, samples, burn_in, seed)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let d = PyDict::new(py);
    d.set_item("mean_worths", PyList::new(py, &r.mean_worths)?)?;
    d.set_item("ci_low", PyList::new(py, &r.ci_low)?)?;
    d.set_item("ci_high", PyList::new(py, &r.ci_high)?)?;
    d.set_item("p_best", PyList::new(py, &r.p_best)?)?;
    d.set_item("samples", r.samples)?;
    Ok(d.into())
}

/// Builds the dict mirroring `stats_paper_package`'s return shape from an
/// already-computed [`PaperPackage`]. Shared by `stats_paper_package` and
/// `per_budget_packages` so both produce identically-shaped package dicts.
fn paper_package_to_dict(py: Python<'_>, pkg: &PaperPackage) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);

    let friedman_d = PyDict::new(py);
    friedman_d.set_item("statistic", pkg.friedman.statistic)?;
    friedman_d.set_item("p_value", pkg.friedman.p_value)?;
    friedman_d.set_item("mean_ranks", PyList::new(py, &pkg.friedman.mean_ranks)?)?;
    d.set_item("friedman", friedman_d)?;

    d.set_item("nemenyi_cd", pkg.nemenyi_cd)?;

    let pw_list = PyList::empty(py);
    for &(i, j, p) in &pkg.pairwise_wilcoxon_holm {
        let row = PyList::empty(py);
        row.append(i)?;
        row.append(j)?;
        row.append(p)?;
        pw_list.append(row)?;
    }
    d.set_item("pairwise_wilcoxon_holm", pw_list)?;

    let cliffs_list = PyList::empty(py);
    for &(i, j, delta) in &pkg.cliffs {
        let row = PyList::empty(py);
        row.append(i)?;
        row.append(j)?;
        row.append(delta)?;
        cliffs_list.append(row)?;
    }
    d.set_item("cliffs", cliffs_list)?;

    let bayes_list = PyList::empty(py);
    for (i, j, res) in &pkg.bayes {
        let bd = PyDict::new(py);
        bd.set_item("p_left", res.p_left)?;
        bd.set_item("p_rope", res.p_rope)?;
        bd.set_item("p_right", res.p_right)?;
        let row = PyList::empty(py);
        row.append(*i)?;
        row.append(*j)?;
        row.append(bd)?;
        bayes_list.append(row)?;
    }
    d.set_item("bayes", bayes_list)?;

    let pl_d = PyDict::new(py);
    pl_d.set_item("worths", PyList::new(py, &pkg.plackett_luce.worths)?)?;
    pl_d.set_item("p_best", PyList::new(py, &pkg.plackett_luce.p_best)?)?;
    pl_d.set_item("iterations", pkg.plackett_luce.iterations)?;
    d.set_item("plackett_luce", pl_d)?;

    d.set_item("latex_summary", pkg.latex_summary.clone())?;
    d.set_item("latex_tests", pkg.latex_tests.clone())?;

    Ok(d.into())
}

#[pyfunction]
#[pyo3(signature = (algo_names, problem_names, results, rope=0.0, samples=20000, seed=1))]
fn stats_paper_package(
    py: Python<'_>,
    algo_names: Vec<String>,
    problem_names: Vec<String>,
    results: &Bound<'_, PyAny>,
    rope: f64,
    samples: u64,
    seed: u64,
) -> PyResult<Py<PyDict>> {
    let matrix = extract_matrix(results)?;
    let pkg = paper_package(&algo_names, &problem_names, &matrix, rope, samples, seed)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    paper_package_to_dict(py, &pkg)
}

// ---------------------------------------------------------------------
// Reporting bindings (sezgi.results_matrix / sezgi.per_budget_packages)
// ---------------------------------------------------------------------

/// Parses an aggregate string (`"mean"` | `"median"`) into
/// [`sezgi_bench::Aggregate`], raising `ValueError` on anything else.
fn parse_aggregate(aggregate: &str) -> PyResult<Aggregate> {
    match aggregate {
        "mean" => Ok(Aggregate::Mean),
        "median" => Ok(Aggregate::Median),
        other => Err(PyValueError::new_err(format!(
            "unknown aggregate `{other}` (expected \"mean\" or \"median\")"
        ))),
    }
}

/// Rebuilds `RunRecord`s from the record dicts `run_experiment` returns
/// (fields `algo, fid, dim, instance, seed, budget, best_f, f_opt,
/// evals_used`; `wall_secs` is read if present, defaulted to `0.0`
/// otherwise — it plays no role in `results_matrix`/`per_budget_packages`).
fn records_from_pylist(records: &Bound<'_, PyAny>) -> PyResult<Vec<RunRecord>> {
    let mut out = Vec::new();
    for item in records.try_iter()? {
        let item = item?;
        let d = item.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err("records must be a list of dicts (as returned by run_experiment)")
        })?;
        let get = |k: &str| -> PyResult<Bound<'_, PyAny>> {
            d.get_item(k)?.ok_or_else(|| PyValueError::new_err(format!("record is missing field `{k}`")))
        };
        let key = RunKey {
            algo: get("algo")?.extract::<String>()?,
            fid: get("fid")?.extract::<u32>()?,
            dim: get("dim")?.extract::<usize>()?,
            instance: get("instance")?.extract::<u32>()?,
            seed: get("seed")?.extract::<u64>()?,
            budget: get("budget")?.extract::<u64>()?,
        };
        let evals_used = get("evals_used")?.extract::<u64>()?;
        let wall_secs = match d.get_item("wall_secs")? {
            Some(v) => v.extract::<f64>()?,
            None => 0.0,
        };
        out.push(RunRecord {
            key,
            best_f: get("best_f")?.extract::<f64>()?,
            f_opt: get("f_opt")?.extract::<f64>()?,
            evals_used,
            wall_secs,
        });
    }
    Ok(out)
}

/// Returns `(algo_names, problem_labels, matrix)` for one `budget` — see
/// `sezgi_bench::reporting::results_matrix`. `records` is the list of dicts
/// `run_experiment` returns (or any list of dicts with the same fields).
pub type ResultsMatrix = (Vec<String>, Vec<String>, Vec<Vec<f64>>);

#[pyfunction]
#[pyo3(signature = (records, budget, aggregate="mean"))]
fn results_matrix(
    records: &Bound<'_, PyAny>,
    budget: u64,
    aggregate: &str,
) -> PyResult<ResultsMatrix> {
    let recs = records_from_pylist(records)?;
    let agg = parse_aggregate(aggregate)?;
    bench_results_matrix(&recs, budget, agg).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Returns a list of `(budget, package_dict)` pairs, one per distinct
/// budget present in `records`, in ascending budget order — see
/// `sezgi_bench::reporting::per_budget_packages`. Each `package_dict` has
/// exactly the shape `stats_paper_package` returns.
#[pyfunction]
#[pyo3(signature = (records, rope=0.0, samples=20000, seed=1, aggregate="mean"))]
fn per_budget_packages(
    py: Python<'_>,
    records: &Bound<'_, PyAny>,
    rope: f64,
    samples: u64,
    seed: u64,
    aggregate: &str,
) -> PyResult<Py<PyList>> {
    let recs = records_from_pylist(records)?;
    let agg = parse_aggregate(aggregate)?;
    let packages = bench_per_budget_packages(&recs, rope, samples, seed, agg)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let out = PyList::empty(py);
    for (budget, pkg) in &packages {
        let pkg_dict = paper_package_to_dict(py, pkg)?;
        let row = PyList::empty(py);
        row.append(*budget)?;
        row.append(pkg_dict)?;
        out.append(row)?;
    }
    Ok(out.into())
}

#[pyfunction] fn preset_de_rand_1(pop_size: usize, budget: u64) -> String {
    presets::de_rand_1(pop_size, budget).to_json()
}
#[pyfunction] fn preset_de_best_1(pop_size: usize, budget: u64) -> String {
    presets::de_best_1(pop_size, budget).to_json()
}
#[pyfunction] fn preset_jde(pop_size: usize, budget: u64) -> String {
    presets::jde(pop_size, budget).to_json()
}
#[pyfunction] fn preset_ga_real(pop_size: usize, budget: u64) -> String {
    presets::ga_real(pop_size, budget).to_json()
}
#[pyfunction] fn preset_pso(pop_size: usize, budget: u64) -> String {
    presets::pso(pop_size, budget).to_json()
}
#[pyfunction] fn preset_gwo(pop_size: usize, budget: u64) -> String {
    presets::gwo(pop_size, budget).to_json()
}
#[pyfunction] fn preset_woa(pop_size: usize, budget: u64) -> String {
    presets::woa(pop_size, budget).to_json()
}
#[pyfunction] fn preset_harmony_search(pop_size: usize, budget: u64) -> String {
    presets::harmony_search(pop_size, budget).to_json()
}
#[pyfunction] fn preset_cuckoo_search(pop_size: usize, budget: u64) -> String {
    presets::cuckoo_search(pop_size, budget).to_json()
}
#[pyfunction] fn preset_goa(pop_size: usize, budget: u64) -> String {
    presets::goa(pop_size, budget).to_json()
}
#[pyfunction] fn preset_sca(pop_size: usize, budget: u64) -> String {
    presets::sca(pop_size, budget).to_json()
}
#[pyfunction] fn preset_jaya(pop_size: usize, budget: u64) -> String {
    presets::jaya(pop_size, budget).to_json()
}
#[pyfunction] fn preset_mfo(pop_size: usize, budget: u64) -> String {
    presets::mfo(pop_size, budget).to_json()
}
#[pyfunction] fn preset_ssa(pop_size: usize, budget: u64) -> String {
    presets::ssa(pop_size, budget).to_json()
}
#[pyfunction] fn preset_firefly(pop_size: usize, budget: u64) -> String {
    presets::firefly(pop_size, budget).to_json()
}
#[pyfunction] fn preset_sa(budget: u64) -> String {
    presets::sa(budget).to_json()
}
#[pyfunction] fn preset_shade(pop_size: usize, budget: u64) -> String {
    presets::shade(pop_size, budget).to_json()
}
#[pyfunction] fn preset_lshade(dim: usize, budget: u64) -> String {
    presets::lshade(dim, budget).to_json()
}
#[pyfunction] fn preset_cmaes(pop_size: usize, budget: u64) -> String {
    presets::cmaes(pop_size, budget).to_json()
}
#[pyfunction] fn preset_cmaes_ipop(dim: usize, budget: u64) -> String {
    presets::cmaes_ipop(dim, budget).to_json()
}
#[pyfunction] fn preset_nelder_mead(dim: usize, budget: u64) -> String {
    presets::nelder_mead(dim, budget).to_json()
}
#[pyfunction] fn preset_random_search(pop_size: usize, budget: u64) -> String {
    presets::random_search(pop_size, budget).to_json()
}
/// Parses the dist string + flattened params accepted by
/// `preset_es_mu_plus_lambda` into a `Distribution`. Shared with the R
/// binding's semantics (duplicated there per the "no shared private crate
/// imports" rule — see `r-sezgi/src/rust/src/solve.rs`).
fn parse_distribution(
    dist: &str,
    mean: f64,
    sigma: f64,
    loc: f64,
    scale: f64,
    alpha: f64,
    nu: f64,
) -> Result<Distribution, String> {
    match dist {
        "uniform" => Ok(Distribution::Uniform),
        "gaussian" => Ok(Distribution::Gaussian { mean, sigma }),
        "cauchy" => Ok(Distribution::Cauchy { loc, scale }),
        "levy" => Ok(Distribution::Levy { alpha }),
        "student_t" => Ok(Distribution::StudentT { nu }),
        "laplace" => Ok(Distribution::Laplace { loc, scale }),
        other => Err(format!(
            "unknown distribution '{other}' (expected uniform|gaussian|cauchy|levy|student_t|laplace)"
        )),
    }
}

#[pyfunction]
#[pyo3(signature = (pop_size, budget, dist="gaussian", mean=0.0, sigma=0.5, loc=0.0, scale=1.0, alpha=1.5, nu=3.0))]
#[allow(clippy::too_many_arguments)]
fn preset_es_mu_plus_lambda(
    pop_size: usize,
    budget: u64,
    dist: &str,
    mean: f64,
    sigma: f64,
    loc: f64,
    scale: f64,
    alpha: f64,
    nu: f64,
) -> PyResult<String> {
    let d = parse_distribution(dist, mean, sigma, loc, scale, alpha, nu)
        .map_err(PyValueError::new_err)?;
    Ok(presets::es_mu_plus_lambda(pop_size, budget, d).to_json())
}

#[pymodule]
fn _sezgi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProblem>()?;
    m.add_class::<PyEvalSession>()?;
    m.add_function(wrap_pyfunction!(bbob, m)?)?;
    m.add_function(wrap_pyfunction!(from_callable, m)?)?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(run_experiment, m)?)?;
    m.add_function(wrap_pyfunction!(read_ioh_records, m)?)?;
    m.add_function(wrap_pyfunction!(ecdf, m)?)?;
    m.add_function(wrap_pyfunction!(coco_export, m)?)?;
    m.add_function(wrap_pyfunction!(stats_friedman, m)?)?;
    m.add_function(wrap_pyfunction!(stats_wilcoxon, m)?)?;
    m.add_function(wrap_pyfunction!(stats_cliffs_delta, m)?)?;
    m.add_function(wrap_pyfunction!(stats_cliffs_magnitude, m)?)?;
    m.add_function(wrap_pyfunction!(stats_bayesian_signed_rank, m)?)?;
    m.add_function(wrap_pyfunction!(stats_plackett_luce, m)?)?;
    m.add_function(wrap_pyfunction!(stats_bayesian_plackett_luce, m)?)?;
    m.add_function(wrap_pyfunction!(stats_paper_package, m)?)?;
    m.add_function(wrap_pyfunction!(results_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(per_budget_packages, m)?)?;
    m.add_function(wrap_pyfunction!(preset_de_rand_1, m)?)?;
    m.add_function(wrap_pyfunction!(preset_de_best_1, m)?)?;
    m.add_function(wrap_pyfunction!(preset_jde, m)?)?;
    m.add_function(wrap_pyfunction!(preset_ga_real, m)?)?;
    m.add_function(wrap_pyfunction!(preset_pso, m)?)?;
    m.add_function(wrap_pyfunction!(preset_gwo, m)?)?;
    m.add_function(wrap_pyfunction!(preset_woa, m)?)?;
    m.add_function(wrap_pyfunction!(preset_harmony_search, m)?)?;
    m.add_function(wrap_pyfunction!(preset_cuckoo_search, m)?)?;
    m.add_function(wrap_pyfunction!(preset_goa, m)?)?;
    m.add_function(wrap_pyfunction!(preset_sca, m)?)?;
    m.add_function(wrap_pyfunction!(preset_jaya, m)?)?;
    m.add_function(wrap_pyfunction!(preset_mfo, m)?)?;
    m.add_function(wrap_pyfunction!(preset_ssa, m)?)?;
    m.add_function(wrap_pyfunction!(preset_firefly, m)?)?;
    m.add_function(wrap_pyfunction!(preset_sa, m)?)?;
    m.add_function(wrap_pyfunction!(preset_shade, m)?)?;
    m.add_function(wrap_pyfunction!(preset_lshade, m)?)?;
    m.add_function(wrap_pyfunction!(preset_cmaes, m)?)?;
    m.add_function(wrap_pyfunction!(preset_cmaes_ipop, m)?)?;
    m.add_function(wrap_pyfunction!(preset_nelder_mead, m)?)?;
    m.add_function(wrap_pyfunction!(preset_random_search, m)?)?;
    m.add_function(wrap_pyfunction!(preset_es_mu_plus_lambda, m)?)?;
    Ok(())
}
