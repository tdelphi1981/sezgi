use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sezgi_bench::{
    run_experiment_parallel, run_experiment_sequential, run_experiment_with_checkpoint,
    ExperimentSpec, IohLogger,
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
    bayesian_signed_rank, cliffs_delta, cliffs_magnitude, friedman, paper_package,
    plackett_luce, wilcoxon_signed_rank, WilcoxonMethod,
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
                let obs = lg.start_run(p.instance);
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

/// Runs an [`ExperimentSpec`] (parsed from `spec_toml`) and returns its
/// `RunRecord`s as a list of dicts. `journal=None` runs via
/// `run_experiment_parallel`/`run_experiment_sequential` (chosen by
/// `parallel`); `journal=Some(path)` runs via
/// `run_experiment_with_checkpoint`, which resumes from — and appends to —
/// an existing journal file at `path`. The GIL is released
/// (`py.allow_threads`) for the duration of the run: experiment problems
/// are bbob-only (no Python callbacks), so no Python object is touched
/// while the GIL is released.
#[pyfunction]
#[pyo3(signature = (spec_toml, journal=None, parallel=true, threads=None))]
fn run_experiment(
    py: Python<'_>,
    spec_toml: &str,
    journal: Option<&str>,
    parallel: bool,
    threads: Option<usize>,
) -> PyResult<Py<PyList>> {
    let spec = ExperimentSpec::from_toml(spec_toml)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let records = if let Some(journal_path) = journal {
        let path = Path::new(journal_path);
        py.allow_threads(|| {
            run_experiment_with_checkpoint(&spec, spec_toml, path, parallel, threads)
        })
    } else if parallel {
        py.allow_threads(|| run_experiment_parallel(&spec, threads))
    } else {
        py.allow_threads(|| run_experiment_sequential(&spec, |_| {}))
    }
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let mut rows = Vec::with_capacity(records.len());
    for r in records {
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
        rows.push(d);
    }
    Ok(PyList::new(py, rows)?.into())
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
    m.add_function(wrap_pyfunction!(bbob, m)?)?;
    m.add_function(wrap_pyfunction!(from_callable, m)?)?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(run_experiment, m)?)?;
    m.add_function(wrap_pyfunction!(stats_friedman, m)?)?;
    m.add_function(wrap_pyfunction!(stats_wilcoxon, m)?)?;
    m.add_function(wrap_pyfunction!(stats_cliffs_delta, m)?)?;
    m.add_function(wrap_pyfunction!(stats_cliffs_magnitude, m)?)?;
    m.add_function(wrap_pyfunction!(stats_bayesian_signed_rank, m)?)?;
    m.add_function(wrap_pyfunction!(stats_plackett_luce, m)?)?;
    m.add_function(wrap_pyfunction!(stats_paper_package, m)?)?;
    m.add_function(wrap_pyfunction!(preset_de_rand_1, m)?)?;
    m.add_function(wrap_pyfunction!(preset_de_best_1, m)?)?;
    m.add_function(wrap_pyfunction!(preset_jde, m)?)?;
    m.add_function(wrap_pyfunction!(preset_ga_real, m)?)?;
    m.add_function(wrap_pyfunction!(preset_pso, m)?)?;
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
