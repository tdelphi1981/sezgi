use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sezgi_bench::{
    coco_export as bench_coco_export, default_targets as bench_default_targets,
    ecdf as bench_ecdf, ecdf_per_algo as bench_ecdf_per_algo, ioh_records as bench_ioh_records,
    per_budget_packages as bench_per_budget_packages, read_ioh_root,
    results_matrix as bench_results_matrix, run_experiment_logged, run_experiment_parallel,
    run_experiment_sequential, run_experiment_with_checkpoint, Aggregate, EcdfCurve, EvalSession,
    ExperimentSpec, IohLogger, RunKey, RunRecord, SessionMeta,
};
use sezgi_bias::{
    central_bias_scan, structural_bias_scan, BiasReportConfig, BiasVerdict, CentralBiasConfig,
    CentralBiasResult, StructuralBiasConfig, StructuralBiasResult,
};
use sezgi_components::nsga2::{nsga2_run, Nsga2Config};
use sezgi_components::{presets, register_builtins};
use sezgi_core::component::Registry;
use sezgi_core::dist::Distribution;
use sezgi_core::engine::{Engine, RunConfig};
use sezgi_core::mo::MoProblem;
use sezgi_core::problem::Problem;
use sezgi_core::space::{Block, BlockValues, Genotype, SearchSpace};
use sezgi_core::spec::AlgorithmSpec;
use sezgi_problems::{BbobProblem, Cec2022, Dtlz, Tsp, TspError, Zdt};
use sezgi_stats::{
    bayesian_plackett_luce, bayesian_signed_rank, cliffs_delta, cliffs_magnitude, friedman,
    hypervolume_2d as stats_hypervolume_2d, igd as stats_igd, paper_package, plackett_luce,
    wilcoxon_signed_rank, PaperPackage, WilcoxonMethod, WilcoxonResult,
};
use sezgi_stats::uniformity::{AdResult, KsResult};
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;

enum Inner {
    Bbob(BbobProblem),
    Cec2022(Cec2022),
    Tsp(Tsp),
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

/// Owned counterpart of [`CallableProblem`], used only by
/// `EvalSession::for_problem` (which needs a `'static` `Box<dyn Problem>`,
/// unlike `solve()`'s synchronous, borrow-for-the-run-lifetime use of
/// `CallableProblem`). Unlike `CallableProblem`'s frozen, population-batched
/// convention (one call per `evaluate_batch`, a 2-D array in and an array
/// out — `solve()`'s engine always evaluates a full population at once),
/// this variant calls `f` once PER POINT, a 1-D length-`dim` array in and a
/// scalar `float` out: `EvalSession`'s ask/tell callers evaluate
/// individually-generated candidate points, so a plain scalar-in/scalar-out
/// callback is the natural shape there, mirroring an ordinary single-point
/// objective function.
struct OwnedCallableProblem { f: Py<PyAny>, space: SearchSpace }

impl Problem for OwnedCallableProblem {
    fn space(&self) -> &SearchSpace { &self.space }
    // sezgi decision: per-point calling convention (see struct doc above), deliberately diverging from CallableProblem's batched one.
    fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<f64> {
        Python::with_gil(|py| {
            pop.iter().map(|g| {
                let BlockValues::Float(v) = &g.blocks[0] else {
                    unreachable!("from_callable only builds Float spaces")
                };
                let arr = PyArray1::from_vec(py, v.clone());
                let out = match self.f.call1(py, (arr,)) {
                    Ok(o) => o,
                    Err(e) => panic::panic_any(e),
                };
                match out.extract::<f64>(py) {
                    Ok(f) => f,
                    Err(e) => panic::panic_any(e),
                }
            }).collect()
        })
    }
}

/// The uniform `(lo, hi)` bounds of a continuous (all-`Float`-block) space:
/// every block must be `Block::Float` and share the SAME `(lo, hi)` pair —
/// true for every space this crate ever builds (`BbobProblem`, `Cec2022`,
/// `from_callable` all build a single `Float` block), but checked explicitly
/// rather than just reading the first block, since `Problem::bounds` is a
/// documented API surface a T4-era problem could in principle multi-block.
///
/// # Errors
/// `ValueError` if any block is not `Float` (e.g. TSP's permutation space),
/// or if the space is empty (dim 0), or if `Float` blocks disagree on
/// `(lo, hi)`.
fn bounds_of(space: &SearchSpace) -> PyResult<(f64, f64)> {
    let mut bounds: Option<(f64, f64)> = None;
    for b in space.blocks() {
        match *b {
            Block::Float { lo, hi, .. } => match bounds {
                None => bounds = Some((lo, hi)),
                Some((rlo, rhi)) if rlo == lo && rhi == hi => {}
                Some(_) => return Err(PyValueError::new_err(
                    "space has non-uniform bounds across its float blocks")),
            },
            _ => return Err(PyValueError::new_err(
                "bounds() is only defined for continuous (float) spaces")),
        }
    }
    bounds.ok_or_else(|| PyValueError::new_err(
        "bounds() is only defined for continuous (float) spaces"))
}

/// `Problem`-handle accessors: `dim()`, `bounds()`, `optimum()` — usable on
/// any handle `sezgi.bbob(...)` / `sezgi.problems.cec2022(...)` /
/// `sezgi.from_callable(...)` / `sezgi.problems.tsp(...)` returns.
#[pymethods]
impl PyProblem {
    /// The search space's dimensionality (`space().dim()`).
    fn dim(&self) -> usize {
        match &self.inner {
            Inner::Bbob(p) => p.space().dim(),
            Inner::Cec2022(p) => p.space().dim(),
            Inner::Tsp(p) => p.space().dim(),
            Inner::Callable { space, .. } => space.dim(),
        }
    }

    /// The uniform `(lo, hi)` bounds of a continuous (float) space.
    ///
    /// # Errors
    /// `ValueError` for a non-continuous space (e.g. TSP's permutation
    /// space) — see [`bounds_of`].
    fn bounds(&self) -> PyResult<(f64, f64)> {
        let space = match &self.inner {
            Inner::Bbob(p) => p.space(),
            Inner::Cec2022(p) => p.space(),
            Inner::Tsp(p) => p.space(),
            Inner::Callable { space, .. } => space,
        };
        bounds_of(space)
    }

    /// The problem's known optimum, or `None` if it has none (a
    /// `from_callable` handle always returns `None`: an arbitrary Python
    /// function has no analytically known optimum).
    fn optimum(&self) -> Option<f64> {
        match &self.inner {
            Inner::Bbob(p) => p.optimum(),
            Inner::Cec2022(p) => p.optimum(),
            Inner::Tsp(p) => p.optimum(),
            Inner::Callable { .. } => None,
        }
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

/// `sezgi.problems.cec2022(fid, dim)` -- a [`Problem`] handle for a CEC 2022
/// function (`sezgi_problems::Cec2022::new`), usable with `solve()` (via
/// `sezgi.presets.*` + `sezgi.solve`) exactly like `sezgi.bbob(...)`. See
/// [`Cec2022::new`]'s own doc for the exact `fid`/`dim` domain.
///
/// # Errors
/// `ValueError` for [`sezgi_problems::Cec2022Error`]: `fid` outside `1..=12`,
/// `dim` outside `{2, 10, 20}`, or `dim=2` for a hybrid function (`fid`
/// 6-8).
#[pyfunction]
fn cec2022(fid: u32, dim: usize) -> PyResult<PyProblem> {
    Ok(PyProblem { inner: Inner::Cec2022(
        Cec2022::new(fid, dim).map_err(|e| PyValueError::new_err(e.to_string()))?) })
}

/// `sezgi.problems.cec2022_evaluate(fid, dim, x)` -- direct, one-shot
/// evaluation of a CEC 2022 function at `x` (a length-`dim` list of floats),
/// bypassing `solve()`'s budget/engine machinery entirely. Returns the exact
/// `f64` [`Cec2022::evaluate_batch`] computes (`Problem::evaluate_batch`'s
/// own `xs.len() == self.dim` check is redundant with the explicit length
/// check below, but the explicit check gives a clear `ValueError` message
/// instead of a silent `f64::INFINITY` sentinel).
///
/// # Errors
/// `ValueError` for the same [`sezgi_problems::Cec2022Error`] cases as
/// `cec2022(...)`, plus a `ValueError` if `len(x) != dim`.
#[pyfunction]
fn cec2022_evaluate(fid: u32, dim: usize, x: Vec<f64>) -> PyResult<f64> {
    let p = Cec2022::new(fid, dim).map_err(|e| PyValueError::new_err(e.to_string()))?;
    if x.len() != dim {
        return Err(PyValueError::new_err(format!(
            "x must have exactly {dim} coordinates (dim={dim}), got {}", x.len())));
    }
    let g = Genotype { blocks: vec![BlockValues::Float(x)] };
    Ok(p.evaluate_batch(&[g])[0])
}

/// `sezgi.problems.cec2022_f_star(fid)` -- the report's pinned `F_i*` bias
/// ([`Cec2022::f_star`], module doc section 1.2's table). `f_star` does not
/// depend on `dim`, so an internal probe `dim=10` is used purely to validate
/// `fid` (every `fid` in `1..=12` accepts `dim=10`, hybrids included) --
/// `dim` is not itself a parameter of this function.
///
/// # Errors
/// `ValueError` if `fid` is outside `1..=12`.
#[pyfunction]
fn cec2022_f_star(fid: u32) -> PyResult<f64> {
    let p = Cec2022::new(fid, 10).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(p.f_star())
}

/// `sezgi.problems.tsp(name)` -- a [`Problem`] handle for a VENDORED TSPLIB
/// instance (`sezgi_problems::Tsp::vendored`: `"berlin52"`, `"eil51"`, or
/// `"st70"`), usable with `solve()` exactly like `sezgi.bbob(...)` /
/// `sezgi.problems.cec2022(...)`.
///
/// # Errors
/// `ValueError` if `name` is not one of the three vendored instances.
#[pyfunction]
fn tsp(name: &str) -> PyResult<PyProblem> {
    Ok(PyProblem { inner: Inner::Tsp(
        Tsp::vendored(name).map_err(|e| PyValueError::new_err(e.to_string()))?) })
}

/// Shared `name_or_text` resolver for `tsp_load`/`tsp_tour_length`: tries
/// `Tsp::vendored(name_or_text)` first (a vendored instance name), and on
/// [`TspError::UnknownVendored`] only, falls back to parsing `name_or_text`
/// as raw TSPLIB `.tsp` file text via `Tsp::from_tsplib`. Any other
/// [`TspError`] (from either path) is surfaced as `ValueError` directly.
fn load_tsp(name_or_text: &str) -> PyResult<Tsp> {
    match Tsp::vendored(name_or_text) {
        Ok(t) => Ok(t),
        Err(TspError::UnknownVendored(_)) => {
            Tsp::from_tsplib(name_or_text).map_err(|e| PyValueError::new_err(e.to_string()))
        }
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
}

/// `sezgi.problems.tsp_load(name_or_text)` -- loads a TSPLIB `EUC_2D`
/// instance, either a vendored instance name or raw TSPLIB file text (see
/// `load_tsp`), and returns a dict describing it: `name` (str), `n_cities`
/// (int), `coords` (list of `(x, y)` tuples, index `c` is the coordinate
/// pair `Tsp::coords()[c]` -- genotype index `c` maps to TSPLIB node
/// `c + 1`, see `tsp.rs`'s module doc), `known_optimum` (float, or `None` for
/// an instance parsed from raw text rather than a vendored name).
///
/// # Errors
/// `ValueError` for any [`TspError`] (unknown vendored name that also fails
/// to parse as TSPLIB text, malformed TSPLIB text, unsupported
/// `EDGE_WEIGHT_TYPE`, ...).
#[pyfunction]
fn tsp_load(py: Python<'_>, name_or_text: &str) -> PyResult<Py<PyDict>> {
    let t = load_tsp(name_or_text)?;
    let d = PyDict::new(py);
    d.set_item("name", t.name())?;
    d.set_item("n_cities", t.n_cities())?;
    let coords = PyList::empty(py);
    for &(x, y) in t.coords() {
        coords.append((x, y))?;
    }
    d.set_item("coords", coords)?;
    d.set_item("known_optimum", t.known_optimum())?;
    Ok(d.into())
}

/// `sezgi.problems.tsp_tour_length(name_or_text, tour)` -- closed-tour
/// length of a 0-based `tour` (a permutation of `0..n_cities`) on the
/// instance named/parsed by `name_or_text` (see `load_tsp`), via
/// `Tsp::evaluate_batch`'s `nint`-rounded `EUC_2D` sum (`tsp.rs`'s module
/// doc). UNLIKE `Tsp::evaluate_batch` itself (which returns `f64::INFINITY`
/// for a malformed genotype, since genotype validity is normally the
/// engine's `SearchSpace::validate` job, not `Tsp`'s own) -- this binding
/// validates `tour` itself (exact length, every entry in `0..n_cities`, no
/// repeats) and raises a precise `ValueError` instead, since a `tour` coming
/// directly from Python has no engine-side validation gate in front of it.
///
/// # Errors
/// `ValueError` for any [`TspError`] resolving `name_or_text`, or if `tour`
/// is not a permutation of `0..n_cities` (wrong length, an out-of-range
/// entry, or a repeated entry).
#[pyfunction]
fn tsp_tour_length(name_or_text: &str, tour: Vec<i64>) -> PyResult<f64> {
    let t = load_tsp(name_or_text)?;
    let n = t.n_cities();
    if tour.len() != n {
        return Err(PyValueError::new_err(format!(
            "tour must have exactly {n} entries (one per city), got {}", tour.len())));
    }
    let mut seen = vec![false; n];
    let mut order = Vec::with_capacity(n);
    for &c in &tour {
        if c < 0 || c as usize >= n {
            return Err(PyValueError::new_err(format!(
                "tour entry {c} out of range for a {n}-city instance (expected 0..{n})")));
        }
        let ci = c as usize;
        if seen[ci] {
            return Err(PyValueError::new_err(format!(
                "tour has a repeated city {ci}; not a valid permutation")));
        }
        seen[ci] = true;
        order.push(ci as u32);
    }
    let g = Genotype { blocks: vec![BlockValues::Perm(order)] };
    Ok(t.evaluate_batch(&[g])[0])
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

    /// Builds a session over any continuous [`PyProblem`] handle —
    /// `sezgi.bbob(...)`, `sezgi.problems.cec2022(...)`, or
    /// `sezgi.from_callable(...)`. See [`sezgi_bench::EvalSession::new_owned`]
    /// for the generalization this delegates to; each `Inner` arm below
    /// builds its own [`SessionMeta`] (suite/fid/name/instance/f_opt), so
    /// adding a new continuous-problem arm elsewhere in this crate is a
    /// self-contained extension of this match.
    ///
    /// # Errors
    /// - `ValueError` for `sezgi.problems.tsp(...)`: its permutation space
    ///   is not a continuous problem `EvalSession` can evaluate.
    /// - `ValueError` if `log_dir` is given for a problem with no known
    ///   optimum (currently only `sezgi.from_callable(...)`): the IOH
    ///   archive's meta has nothing to record as `f_opt`.
    #[staticmethod]
    #[pyo3(signature = (problem, budget, log_dir=None, algo_name="custom", seed=0))]
    fn for_problem(
        py: Python<'_>,
        problem: &PyProblem,
        budget: u64,
        log_dir: Option<&str>,
        algo_name: &str,
        seed: u64,
    ) -> PyResult<Self> {
        let (boxed, meta): (Box<dyn Problem>, SessionMeta) = match &problem.inner {
            Inner::Bbob(p) => {
                // Fresh instance from the same (fid, dim, instance): matches
                // `EvalSession::new_bbob`'s own construction exactly (BBOB's
                // instance-seeded RNG makes this deterministic and
                // bit-identical to `p`).
                let fresh = BbobProblem::new(p.fid(), p.space().dim(), p.instance)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let meta = SessionMeta {
                    suite: "sezgi-bbob".into(),
                    fid: p.fid(),
                    name: p.name().to_string(),
                    instance: p.instance,
                    f_opt: Some(p.f_opt()),
                };
                (Box::new(fresh), meta)
            }
            Inner::Cec2022(p) => {
                let fresh = Cec2022::new(p.fid(), p.dim())
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let meta = SessionMeta {
                    suite: "sezgi-cec2022".into(),
                    fid: p.fid(),
                    name: format!("cec2022-f{}", p.fid()),
                    instance: 1,
                    f_opt: p.optimum(),
                };
                (Box::new(fresh), meta)
            }
            Inner::Callable { f, space } => {
                let owned = OwnedCallableProblem { f: f.clone_ref(py), space: space.clone() };
                let meta = SessionMeta {
                    suite: "sezgi-custom".into(),
                    fid: 0,
                    name: "callable".into(),
                    instance: 1,
                    f_opt: None,
                };
                (Box::new(owned), meta)
            }
            Inner::Tsp(_) => return Err(PyValueError::new_err(
                "EvalSession supports continuous (float) problems only")),
        };

        if log_dir.is_some() && meta.f_opt.is_none() {
            return Err(PyValueError::new_err(
                "log_dir requires a problem with a known optimum (f_opt)"));
        }

        let mut session = EvalSession::new_owned(boxed, meta, budget)
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

    /// The problem's known optimum, or `None` if it has none (e.g. a
    /// `for_problem`-built session over a `from_callable` handle). A
    /// session built via the `EvalSession(...)` (BBOB) constructor always
    /// returns a `float`.
    fn f_opt(&self) -> PyResult<Option<f64>> {
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
        // sezgi decision: CEC 2022/TSP are solve()-eligible (Inner::Cec2022,
        // Inner::Tsp) additively -- same shape as Inner::Bbob minus IOH
        // logging (no fid/instance/name scenario metadata to log against for
        // either, so log_dir is rejected the same way Inner::Callable
        // rejects it).
        Inner::Cec2022(p) => {
            if log_dir.is_some() {
                return Err(PyValueError::new_err(
                    "log_dir is only supported for builtin (bbob) problems"));
            }
            (run_with_bridge(py, || run(p, None))?, None)
        }
        Inner::Tsp(p) => {
            if log_dir.is_some() {
                return Err(PyValueError::new_err(
                    "log_dir is only supported for builtin (bbob) problems"));
            }
            (run_with_bridge(py, || run(p, None))?, None)
        }
    };

    let d = PyDict::new(py);
    d.set_item("best_f", result.best_f)?;
    // best_x's block shape follows the problem's own space: Float for
    // Bbob/Cec2022/Callable, Perm for Tsp (a permutation genotype, per
    // tsp.rs's module doc) -- both are surfaced as a plain list of Python
    // numbers (float or int respectively).
    match &result.best_x.blocks[0] {
        BlockValues::Float(xs) => d.set_item("best_x", PyList::new(py, xs)?)?,
        BlockValues::Perm(xs) => d.set_item("best_x", PyList::new(py, xs)?)?,
        _ => return Err(PyValueError::new_err("unexpected genotype")),
    }
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

// ---------------------------------------------------------------------
// Bias-scanning bindings (sezgi.bias) — M3-1 Task 8.
//
// Mirrors `crates/bias`'s public structs 1:1 by field name (see that
// crate's `structural`/`central`/`report` modules for the full method
// provenance). `BiasVerdict` is surfaced as two flat keys on every dict
// that carries one — `verdict`: the string "no_evidence" or "evidence",
// and `detail`: `None` for `NoEvidence`, the `Evidence::detail` string
// otherwise — rather than a nested sub-dict, matching this module's own
// flat-dict convention elsewhere (e.g. `stats_wilcoxon`'s `method` key).
//
// T6 (the Rajwar-Deep signature-bias test) is DEFERRED in `sezgi-bias`
// itself (no `signature_scan` exists yet — see `crates/bias/src/report.rs`'s
// module doc, "T6 (signature test): deferred"); `bias_report`'s `signature`
// key is mapped through `r.signature` properly (a `Some`/`None` match, not a
// hardcoded `py.None()`), so it is Python `None` today only because
// `BiasReport::signature` itself is always `None` today — the moment T6
// lands in the crate, this binding reflects it with no code change needed
// here, rather than silently misreporting `None` forever.
//
// Every scalar statistic (`d`, `p_value`, `a2`, `w_statistic`, `z`,
// `effect`, ...) is passed through as the exact `f64` PyO3 already returns
// bit-for-bit for a Rust `f64` argument to `PyDict::set_item` — no
// rounding/formatting anywhere in this section.
// ---------------------------------------------------------------------

fn ks_result_to_dict<'py>(py: Python<'py>, r: &KsResult) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("d", r.d)?;
    d.set_item("p_value", r.p_value)?;
    d.set_item("n", r.n)?;
    Ok(d)
}

fn ad_result_to_dict<'py>(py: Python<'py>, r: &AdResult) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("a2", r.a2)?;
    d.set_item("p_value", r.p_value)?;
    d.set_item("n", r.n)?;
    Ok(d)
}

fn wilcoxon_result_to_dict<'py>(py: Python<'py>, r: &WilcoxonResult) -> PyResult<Bound<'py, PyDict>> {
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
    Ok(d)
}

/// Sets `verdict` ("no_evidence"/"evidence") and `detail` (`None`, or the
/// `Evidence::detail` string) on `d` — shared by every dict this module
/// builds from a [`BiasVerdict`].
fn set_verdict(d: &Bound<'_, PyDict>, verdict: &BiasVerdict) -> PyResult<()> {
    match verdict {
        BiasVerdict::NoEvidence => {
            d.set_item("verdict", "no_evidence")?;
            d.set_item("detail", Option::<String>::None)?;
        }
        BiasVerdict::Evidence { detail } => {
            d.set_item("verdict", "evidence")?;
            d.set_item("detail", detail.clone())?;
        }
    }
    Ok(())
}

fn structural_result_to_dict<'py>(
    py: Python<'py>,
    r: &StructuralBiasResult,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let ks_list = PyList::empty(py);
    for k in &r.per_dim_ks {
        ks_list.append(ks_result_to_dict(py, k)?)?;
    }
    d.set_item("per_dim_ks", ks_list)?;
    let ad_list = PyList::empty(py);
    for a in &r.per_dim_ad {
        ad_list.append(ad_result_to_dict(py, a)?)?;
    }
    d.set_item("per_dim_ad", ad_list)?;
    d.set_item("holm_rejections_ks", r.holm_rejections_ks)?;
    d.set_item("holm_rejections_ad", r.holm_rejections_ad)?;
    set_verdict(&d, &r.verdict)?;
    let positions = PyList::empty(py);
    for row in &r.final_positions {
        positions.append(PyList::new(py, row)?)?;
    }
    d.set_item("final_positions", positions)?;
    Ok(d)
}

fn central_result_to_dict<'py>(
    py: Python<'py>,
    r: &CentralBiasResult,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("gap_centered", PyList::new(py, &r.gap_centered)?)?;
    d.set_item("gap_shifted", PyList::new(py, &r.gap_shifted)?)?;
    d.set_item("wilcoxon", wilcoxon_result_to_dict(py, &r.wilcoxon)?)?;
    d.set_item("effect", r.effect)?;
    set_verdict(&d, &r.verdict)?;
    Ok(d)
}

/// `sezgi.bias.structural(spec, dim, budget, runs=30, seed=0)` — see
/// [`sezgi_bias::structural::structural_bias_scan`]. Returns a dict with
/// keys `per_dim_ks` (list of `{d, p_value, n}`), `per_dim_ad` (list of
/// `{a2, p_value, n}`), `holm_rejections_ks`, `holm_rejections_ad`,
/// `verdict`/`detail` (see this section's own doc), and `final_positions`
/// (`runs` x `dim`).
#[pyfunction]
#[pyo3(signature = (spec_json, dim, budget, runs=sezgi_bias::structural::DEFAULT_RUNS, seed=0))]
fn bias_structural(
    py: Python<'_>,
    spec_json: &str,
    dim: usize,
    budget: u64,
    runs: u32,
    seed: u64,
) -> PyResult<Py<PyDict>> {
    let spec =
        AlgorithmSpec::from_json(spec_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let cfg = StructuralBiasConfig { runs, dim, budget, seed };
    let r = structural_bias_scan(&spec, &cfg).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(structural_result_to_dict(py, &r)?.into())
}

/// `sezgi.bias.central(spec, dim, budget, fids=None, instances_shifted=None,
/// runs_per=20, seed=0)` — see [`sezgi_bias::central::central_bias_scan`].
/// `fids`/`instances_shifted` default to `sezgi_bias::report`'s own
/// `DEFAULT_CENTRAL_FIDS`/`DEFAULT_CENTRAL_INSTANCES` when omitted (`None`).
/// Returns a dict with keys `gap_centered`, `gap_shifted`, `wilcoxon` (a
/// dict: `w_statistic, z, p_value, n_effective, method`), `effect`, and
/// `verdict`/`detail`.
///
/// # Errors
/// `ValueError` for every [`sezgi_bias::BiasError`] case, including a fid in
/// `{5, 6, 20, 24}` (not translation-invariant — see `central.rs`'s module
/// doc, "Construction fix").
#[pyfunction]
#[pyo3(signature = (
    spec_json, dim, budget,
    fids=None, instances_shifted=None,
    runs_per=sezgi_bias::report::DEFAULT_CENTRAL_RUNS_PER, seed=0,
))]
#[allow(clippy::too_many_arguments)]
fn bias_central(
    py: Python<'_>,
    spec_json: &str,
    dim: usize,
    budget: u64,
    fids: Option<Vec<u32>>,
    instances_shifted: Option<Vec<u32>>,
    runs_per: u32,
    seed: u64,
) -> PyResult<Py<PyDict>> {
    let spec =
        AlgorithmSpec::from_json(spec_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let cfg = CentralBiasConfig {
        fids: fids.unwrap_or_else(|| sezgi_bias::report::DEFAULT_CENTRAL_FIDS.to_vec()),
        dim,
        instances_shifted: instances_shifted
            .unwrap_or_else(|| sezgi_bias::report::DEFAULT_CENTRAL_INSTANCES.to_vec()),
        runs_per,
        budget,
        seed,
    };
    let r = central_bias_scan(&spec, &cfg).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(central_result_to_dict(py, &r)?.into())
}

/// `sezgi.bias.report(spec, dim, budget, seed=0, structural_runs=None,
/// central_fids=None, central_instances=None, central_runs_per=None)` — see
/// [`sezgi_bias::bias_report`]. Every optional knob left `None` falls back
/// to [`BiasReportConfig::new`]'s own documented verified-method default
/// (`structural_runs` -> 30, `central_fids` -> `[1, 4, 13]`,
/// `central_instances` -> `[1, 2]`, `central_runs_per` -> 20).
///
/// Returns a dict with keys `structural` (same shape as `bias_structural`'s
/// return), `central` (same shape as `bias_central`'s return), `signature`
/// (Python `None` today, since `BiasReport::signature` is always `None` — T6
/// is deferred, see this section's own doc; mapped through properly rather
/// than hardcoded, so it becomes `{"verdict": ..., "detail": ...}`,
/// mirroring `structural`/`central`'s own `set_verdict` shape, the moment T6
/// lands), `latex_summary` (str, never containing the literal `NaN`), and
/// `plot_data` (`{final_positions, gap_centered, gap_shifted}`, the same raw
/// vectors already inside `structural`/`central`, surfaced for a caller that
/// wants to plot them directly).
#[pyfunction]
#[pyo3(signature = (
    spec_json, dim, budget, seed=0,
    structural_runs=None, central_fids=None, central_instances=None, central_runs_per=None,
))]
#[allow(clippy::too_many_arguments)]
fn bias_report(
    py: Python<'_>,
    spec_json: &str,
    dim: usize,
    budget: u64,
    seed: u64,
    structural_runs: Option<u32>,
    central_fids: Option<Vec<u32>>,
    central_instances: Option<Vec<u32>>,
    central_runs_per: Option<u32>,
) -> PyResult<Py<PyDict>> {
    let spec =
        AlgorithmSpec::from_json(spec_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut cfg = BiasReportConfig::new(dim, budget, seed);
    if let Some(v) = structural_runs {
        cfg.structural_runs = v;
    }
    if let Some(v) = central_fids {
        cfg.central_fids = v;
    }
    if let Some(v) = central_instances {
        cfg.central_instances = v;
    }
    if let Some(v) = central_runs_per {
        cfg.central_runs_per = v;
    }
    let r = sezgi_bias::bias_report(&spec, &cfg)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let d = PyDict::new(py);
    d.set_item("structural", structural_result_to_dict(py, &r.structural)?)?;
    d.set_item("central", central_result_to_dict(py, &r.central)?)?;
    // T6 is deferred in sezgi-bias itself -- `r.signature` is always `None`
    // TODAY (see this section's own doc), but this maps it through properly
    // (rather than hardcoding `py.None()`) so a future T6 landing in the
    // crate is reflected here automatically, with no silent misreport.
    match &r.signature {
        Some(verdict) => {
            let sig = PyDict::new(py);
            set_verdict(&sig, verdict)?;
            d.set_item("signature", sig)?;
        }
        None => d.set_item("signature", py.None())?,
    }
    d.set_item("latex_summary", r.latex_summary.clone())?;

    let plot = PyDict::new(py);
    let positions = PyList::empty(py);
    for row in &r.plot_data.final_positions {
        positions.append(PyList::new(py, row)?)?;
    }
    plot.set_item("final_positions", positions)?;
    plot.set_item("gap_centered", PyList::new(py, &r.plot_data.gap_centered)?)?;
    plot.set_item("gap_shifted", PyList::new(py, &r.plot_data.gap_shifted)?)?;
    d.set_item("plot_data", plot)?;

    Ok(d.into())
}

// ---------------------------------------------------------------------
// Multi-objective bindings (sezgi.mo) -- M3-2 Task 9.
//
// Binds T6's NSGA-II runner (`sezgi_components::nsga2::nsga2_run`), the
// ZDT/DTLZ benchmark suites (`sezgi_problems::{Zdt, Dtlz}`), and the exact
// 2-objective hypervolume / IGD indicators (`sezgi_stats::{hypervolume_2d,
// igd}`). Every scalar/vector f64 is passed through EXACTLY as the Rust
// core computed it -- no rounding/formatting anywhere in this section (T10's
// R bindings assert bit-equality against these same values).
//
// Problem-string mapping (shared by `mo_nsga2` and `mo_pareto_front`, via
// `mo_problem_from_str`): `"zdt1"`, `"zdt2"`, `"zdt3"`, `"zdt4"`, `"zdt6"`
// (ZDT5 is a binary-coded problem, out of scope -- see
// `sezgi_problems::zdt`'s module doc; `"zdt5"` is rejected the same way any
// other unrecognized ZDT number is, via `Zdt::new`'s own `UnknownWhich`
// error) and `"dtlz1"`..`"dtlz7"`. **`m` (number of objectives) is DTLZ-only
// and REQUIRED there** (`Dtlz::new` has no default `m` to fall back to);
// **passing `m` for a `zdt*` problem is a `ValueError`** (zdt problems are
// always 2-objective by construction, so a caller-supplied `m` could never
// be honored silently -- rejecting it outright surfaces the mistake instead
// of quietly ignoring the argument).
// ---------------------------------------------------------------------

/// Shared problem-string -> `Box<dyn MoProblem>` builder for `mo_nsga2` and
/// `mo_pareto_front`. See this section's own doc for the full mapping.
fn mo_problem_from_str(problem: &str, dim: usize, m: Option<usize>) -> PyResult<Box<dyn MoProblem>> {
    let unknown = || {
        PyValueError::new_err(format!(
            "unknown problem `{problem}` (expected one of zdt1, zdt2, zdt3, zdt4, zdt6, or dtlz1..dtlz7)"
        ))
    };
    if let Some(rest) = problem.strip_prefix("zdt") {
        let which: u32 = rest.parse().map_err(|_| unknown())?;
        if m.is_some() {
            return Err(PyValueError::new_err(
                "m is DTLZ-only (number of objectives); zdt problems are always 2-objective -- \
                 omit m (or pass m=None) for a zdt problem",
            ));
        }
        let p = Zdt::new(which, dim).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Box::new(p))
    } else if let Some(rest) = problem.strip_prefix("dtlz") {
        let which: u32 = rest.parse().map_err(|_| unknown())?;
        let m = m.ok_or_else(|| {
            PyValueError::new_err("m (number of objectives) is required for dtlz problems")
        })?;
        let p = Dtlz::new(which, m, dim).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Box::new(p))
    } else {
        Err(unknown())
    }
}

/// Flattens a [`Genotype`] into a single `Vec<f64>` (concatenating every
/// block in order) -- these are always single- or multi-Float-block
/// genotypes here, since `nsga2_run` validates an all-`Block::Float` space
/// before ever constructing one (`Nsga2Error::NonFloatSpace`). Mirrors
/// `solve`'s own `best_x` conversion above (`BlockValues::Float(xs) => ...`),
/// generalized to however many Float blocks the space has (ZDT4 has two:
/// `x1` and the rest).
fn genotype_to_flat_vec(g: &Genotype) -> Vec<f64> {
    let mut out = Vec::new();
    for b in &g.blocks {
        if let BlockValues::Float(xs) = b {
            out.extend_from_slice(xs);
        }
    }
    out
}

/// `sezgi.mo.nsga2(problem, dim, pop_size, budget, m=None, seed=0,
/// eta_c=20.0, eta_m=20.0, p_c=0.9, p_m=None)` -- binds
/// [`sezgi_components::nsga2::nsga2_run`]. `eta_c`/`eta_m`/`p_c` default to
/// the paper's own pinned experimental settings (Deb et al. 2002, Sec.
/// IV.A: eta_c=20, eta_m=20, p_c=0.9 -- see that module's "Defaults"
/// doc section); `p_m=None` resolves on the Rust side to `1 / n_variables`
/// (the paper's own default), never re-derived here.
///
/// Returns a dict mirroring `MoRunResult` 1:1: `individuals` (list of
/// float-lists, one per final-population member, flattened across every
/// Float block), `objectives` (list of float-lists, parallel to
/// `individuals`), `front0` (list of ints: indices of the final
/// population's non-dominated set), `evals_used` (int).
///
/// # Errors
/// `ValueError` for an unrecognized `problem` string, an `m` given for a
/// zdt problem, a missing `m` for a dtlz problem, or any
/// [`sezgi_components::nsga2::Nsga2Error`] (including `pop_size` failing
/// the `>= 4 && pop_size % 4 == 0` check -- NOT merely "even, >= 4").
#[pyfunction]
#[pyo3(signature = (problem, dim, pop_size, budget, m=None, seed=0, eta_c=20.0, eta_m=20.0, p_c=0.9, p_m=None))]
#[allow(clippy::too_many_arguments)]
fn mo_nsga2(
    py: Python<'_>,
    problem: &str,
    dim: usize,
    pop_size: usize,
    budget: u64,
    m: Option<usize>,
    seed: u64,
    eta_c: f64,
    eta_m: f64,
    p_c: f64,
    p_m: Option<f64>,
) -> PyResult<Py<PyDict>> {
    let prob = mo_problem_from_str(problem, dim, m)?;
    let cfg = Nsga2Config { pop_size, budget, seed, eta_c, eta_m, p_c, p_m };

    let result = run_with_bridge(py, || {
        nsga2_run(prob.as_ref(), &cfg).map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    let d = PyDict::new(py);

    let individuals = PyList::empty(py);
    for g in &result.individuals {
        individuals.append(PyList::new(py, genotype_to_flat_vec(g))?)?;
    }
    d.set_item("individuals", individuals)?;

    let objectives = PyList::empty(py);
    for row in &result.objectives {
        objectives.append(PyList::new(py, row)?)?;
    }
    d.set_item("objectives", objectives)?;

    d.set_item("front0", PyList::new(py, &result.front0)?)?;
    d.set_item("evals_used", result.evals_used)?;

    Ok(d.into())
}

/// `sezgi.mo.hypervolume_2d(front, ref_point)` -- binds
/// [`sezgi_stats::hypervolume_2d`] exactly (see that function's doc for the
/// pinned S-metric definition and reference-point convention). `front`: a
/// list of `[f1, f2]` rows (minimization). `ref_point`: a 2-element
/// `[f64; 2]`-shaped list.
///
/// # Errors
/// `ValueError` if `ref_point` does not have exactly 2 values, or for any
/// [`sezgi_stats::StatsError`] (empty front, a non-2-objective row, or a
/// non-finite value).
#[pyfunction]
fn mo_hypervolume_2d(front: Vec<Vec<f64>>, ref_point: Vec<f64>) -> PyResult<f64> {
    let rp: [f64; 2] = ref_point.clone().try_into().map_err(|_| {
        PyValueError::new_err(format!(
            "ref_point must have exactly 2 values, got {}",
            ref_point.len()
        ))
    })?;
    stats_hypervolume_2d(&front, &rp).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// `sezgi.mo.igd(front, reference_front)` -- binds [`sezgi_stats::igd`]
/// exactly (Ishibuchi et al. 2015, eq. 12, `p = 1`; see that function's doc
/// for the pinned definition). Any (equal, consistent) number of objectives
/// across both `front` and `reference_front`.
///
/// # Errors
/// `ValueError` for any [`sezgi_stats::StatsError`] (an empty `front` or
/// `reference_front`, a dimension mismatch, or a non-finite value).
#[pyfunction]
fn mo_igd(front: Vec<Vec<f64>>, reference_front: Vec<Vec<f64>>) -> PyResult<f64> {
    stats_igd(&front, &reference_front).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// `sezgi.mo.pareto_front(problem, dim, n, m=None)` -- a deterministic
/// `n`-point sample of the analytic Pareto front in OBJECTIVE space, via
/// [`sezgi_core::mo::MoProblem::pareto_front`]. Same `problem`/`m` mapping
/// as `mo_nsga2` (see this section's own doc). Returns `None` when the
/// problem has no known analytic front sample at this `m` (e.g. DTLZ5/DTLZ6
/// with `m > 3` -- verified only for `m <= 3`, see `sezgi_problems::dtlz`'s
/// module doc), a list of `n` float-lists otherwise.
///
/// # Errors
/// Same as `mo_nsga2`'s problem-construction errors (unrecognized `problem`,
/// `m` given for zdt, `m` missing for dtlz, or any
/// `ZdtError`/`DtlzError`).
#[pyfunction]
#[pyo3(signature = (problem, dim, n, m=None))]
fn mo_pareto_front(problem: &str, dim: usize, n: usize, m: Option<usize>) -> PyResult<Option<Vec<Vec<f64>>>> {
    let prob = mo_problem_from_str(problem, dim, m)?;
    Ok(prob.pareto_front(n))
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
#[pyfunction] fn preset_bat(pop_size: usize, budget: u64) -> String {
    presets::bat(pop_size, budget).to_json()
}
#[pyfunction] fn preset_fpa(pop_size: usize, budget: u64) -> String {
    presets::fpa(pop_size, budget).to_json()
}
#[pyfunction] fn preset_tlbo(pop_size: usize, budget: u64) -> String {
    presets::tlbo(pop_size, budget).to_json()
}
#[pyfunction] fn preset_hho(pop_size: usize, budget: u64) -> String {
    presets::hho(pop_size, budget).to_json()
}
#[pyfunction] fn preset_alo(pop_size: usize, budget: u64) -> String {
    presets::alo(pop_size, budget).to_json()
}
#[pyfunction] fn preset_abc(pop_size: usize, budget: u64) -> String {
    presets::abc(pop_size, budget).to_json()
}
#[pyfunction] fn preset_gsa(pop_size: usize, budget: u64) -> String {
    presets::gsa(pop_size, budget).to_json()
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
#[pyfunction] fn preset_ga_perm(pop_size: usize, budget: u64) -> String {
    presets::ga_perm(pop_size, budget).to_json()
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
    m.add_function(wrap_pyfunction!(cec2022, m)?)?;
    m.add_function(wrap_pyfunction!(cec2022_evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(cec2022_f_star, m)?)?;
    m.add_function(wrap_pyfunction!(tsp, m)?)?;
    m.add_function(wrap_pyfunction!(tsp_load, m)?)?;
    m.add_function(wrap_pyfunction!(tsp_tour_length, m)?)?;
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
    m.add_function(wrap_pyfunction!(bias_structural, m)?)?;
    m.add_function(wrap_pyfunction!(bias_central, m)?)?;
    m.add_function(wrap_pyfunction!(bias_report, m)?)?;
    m.add_function(wrap_pyfunction!(mo_nsga2, m)?)?;
    m.add_function(wrap_pyfunction!(mo_hypervolume_2d, m)?)?;
    m.add_function(wrap_pyfunction!(mo_igd, m)?)?;
    m.add_function(wrap_pyfunction!(mo_pareto_front, m)?)?;
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
    m.add_function(wrap_pyfunction!(preset_bat, m)?)?;
    m.add_function(wrap_pyfunction!(preset_fpa, m)?)?;
    m.add_function(wrap_pyfunction!(preset_tlbo, m)?)?;
    m.add_function(wrap_pyfunction!(preset_hho, m)?)?;
    m.add_function(wrap_pyfunction!(preset_alo, m)?)?;
    m.add_function(wrap_pyfunction!(preset_abc, m)?)?;
    m.add_function(wrap_pyfunction!(preset_gsa, m)?)?;
    m.add_function(wrap_pyfunction!(preset_sa, m)?)?;
    m.add_function(wrap_pyfunction!(preset_shade, m)?)?;
    m.add_function(wrap_pyfunction!(preset_lshade, m)?)?;
    m.add_function(wrap_pyfunction!(preset_cmaes, m)?)?;
    m.add_function(wrap_pyfunction!(preset_cmaes_ipop, m)?)?;
    m.add_function(wrap_pyfunction!(preset_nelder_mead, m)?)?;
    m.add_function(wrap_pyfunction!(preset_random_search, m)?)?;
    m.add_function(wrap_pyfunction!(preset_es_mu_plus_lambda, m)?)?;
    m.add_function(wrap_pyfunction!(preset_ga_perm, m)?)?;
    Ok(())
}
