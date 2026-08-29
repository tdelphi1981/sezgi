use numpy::{PyArray2, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sezgi_bench::IohLogger;
use sezgi_components::{presets, register_builtins};
use sezgi_core::component::Registry;
use sezgi_core::engine::{Engine, RunConfig};
use sezgi_core::problem::Problem;
use sezgi_core::space::{Block, BlockValues, Genotype, SearchSpace};
use sezgi_core::spec::AlgorithmSpec;
use sezgi_problems::BbobProblem;
use std::panic::{self, AssertUnwindSafe};

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
// preset_es_mu_plus_lambda is intentionally NOT exposed here: its Rust
// signature takes a `Distribution` (an enum with nested params, e.g.
// gaussian mean/sigma), which does not have a clean pyfunction argument
// mapping. Bridging it (accepting a JSON-encoded dist, or a richer PyO3
// type) is deferred to M2d.

#[pymodule]
fn _sezgi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProblem>()?;
    m.add_function(wrap_pyfunction!(bbob, m)?)?;
    m.add_function(wrap_pyfunction!(from_callable, m)?)?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
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
    Ok(())
}
