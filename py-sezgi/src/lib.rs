use pyo3::exceptions::PyValueError;
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
            let xs: Vec<Vec<f64>> = pop.iter().map(|g| match &g.blocks[0] {
                BlockValues::Float(v) => v.clone(),
                _ => vec![],
            }).collect();
            let out = self.f.call1(py, (xs,)).expect("python callback hatası");
            out.extract::<Vec<f64>>(py).expect("callback list[float] döndürmeli")
        })
    }
}

// Not: `Python::with_gil` + `Py<PyAny>` Send+Sync gereksinimini karşılar;
// callback istisnaları M1'de panic→PyErr olarak yüzer, temiz istisna
// köprüsü (PyErr saklama) M2 iyileştirmesi.

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

    let result = match &problem.inner {
        Inner::Bbob(p) => {
            if let Some(dir) = log_dir {
                let name = algo_name.unwrap_or(&spec.name).to_string();
                let mut lg = IohLogger::new(std::path::Path::new(dir), &name,
                    "sezgi-bbob", p.fid(), p.name(),
                    p.space().dim());
                let obs = lg.start_run(p.instance);
                let r = run(p, Some(Box::new(obs)))?;
                let _fin = lg.finish().map_err(|e| PyValueError::new_err(e.to_string()))?;
                r
            } else { run(p, None)? }
        }
        Inner::Callable { f, space } => {
            if log_dir.is_some() {
                return Err(PyValueError::new_err(
                    "log_dir yalnız yerleşik (bbob) problemlerde destekleniyor"));
            }
            let cp = CallableProblem { f, space };
            // GIL callback sırasında yeniden alınır; çağıran thread'de koşuyoruz
            run(&cp, None)?
        }
    };

    let d = PyDict::new(py);
    d.set_item("best_f", result.best_f)?;
    let BlockValues::Float(xs) = &result.best_x.blocks[0] else {
        return Err(PyValueError::new_err("beklenmeyen genotip"));
    };
    d.set_item("best_x", PyList::new(py, xs)?)?;
    d.set_item("evals_used", result.evals_used)?;
    d.set_item("iterations", result.iterations)?;
    Ok(d.into())
}

#[pyfunction] fn preset_de_rand_1(pop_size: usize, budget: u64) -> String {
    presets::de_rand_1(pop_size, budget).to_json()
}
#[pyfunction] fn preset_ga_real(pop_size: usize, budget: u64) -> String {
    presets::ga_real(pop_size, budget).to_json()
}
#[pyfunction] fn preset_pso(pop_size: usize, budget: u64) -> String {
    presets::pso(pop_size, budget).to_json()
}

#[pymodule]
fn _sezgi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProblem>()?;
    m.add_function(wrap_pyfunction!(bbob, m)?)?;
    m.add_function(wrap_pyfunction!(from_callable, m)?)?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(preset_de_rand_1, m)?)?;
    m.add_function(wrap_pyfunction!(preset_ga_real, m)?)?;
    m.add_function(wrap_pyfunction!(preset_pso, m)?)?;
    Ok(())
}
