use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sezgi_bench::{
    coco_export as bench_coco_export, default_targets as bench_default_targets,
    ecdf as bench_ecdf, ecdf_per_algo as bench_ecdf_per_algo, ioh_records as bench_ioh_records,
    nsga2_run_logged, per_budget_packages as bench_per_budget_packages,
    read_ioh_root, read_moa as bench_read_moa, results_matrix as bench_results_matrix,
    run_experiment_logged, run_experiment_parallel, run_experiment_sequential,
    run_experiment_with_checkpoint, Aggregate, EcdfCurve, EvalSession, ExperimentSpec, GenoKind,
    IohLogger, MoArchiveGenotype, RunKey, RunRecord, SessionMeta, SUITE_BBOB,
};
use sezgi_bias::{
    central_bias_scan, f0 as bias_f0_mod, scan_from_positions, structural_bias_scan,
    BiasReportConfig, BiasVerdict, CentralBiasConfig, CentralBiasResult, F0Random,
    StructuralBiasConfig, StructuralBiasResult,
};
use sezgi_components::nsga2::{nsga2_run, Nsga2Config};
use sezgi_components::perm::fisher_yates_shuffle;
use sezgi_components::{presets, register_builtins};
use sezgi_core::component::Registry;
use sezgi_core::dist::Distribution;
use sezgi_core::engine::{Engine, RunConfig};
use sezgi_core::mo::MoProblem;
use sezgi_core::problem::{Evaluator, Problem};
use sezgi_core::rng::RngStream;
use sezgi_core::space::{Block, BlockValues, Genotype, SearchSpace};
use sezgi_core::spec::AlgorithmSpec;
use sezgi_problems::{
    BbobProblem, CatMatch, Cec2014, Cec2017, Cec2022, Dtlz, IntQuadratic, OneMax, Tsp, TspError,
    Wfg, Zdt,
};
// Zdt5 (M3-7 Task 6) is not re-exported at `sezgi_problems`'s crate root
// (only `Zdt`/`ZdtError` are, per that crate's `lib.rs`) -- imported by its
// full module path instead, rather than widening `crates/problems/src/lib.rs`'s
// own `pub use` (out of this task's scope: only `py-sezgi/src/lib.rs` is
// touched, per this task's own gate).
use sezgi_problems::zdt::Zdt5;
use sezgi_stats::{
    bayesian_plackett_luce, bayesian_signed_rank, cliffs_delta, cliffs_magnitude, friedman,
    hypervolume as stats_hypervolume, hypervolume_2d as stats_hypervolume_2d, igd as stats_igd,
    paper_package, plackett_luce, wilcoxon_signed_rank, PaperPackage, WilcoxonMethod,
    WilcoxonResult,
};
use sezgi_stats::uniformity::{AdResult, KsResult};
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;

enum Inner {
    Bbob(BbobProblem),
    Cec2022(Cec2022),
    /// M3-6 Task 9: CEC 2014 (`sezgi_problems::cec2014::Cec2014`, fid
    /// `1..=30`) -- same handle shape as `Inner::Cec2022`, one Rust struct
    /// stored directly (bit-exact `Clone`-free reuse, no rebuild needed for
    /// `dim()`/`bounds()`/`optimum()`).
    Cec2014(Cec2014),
    /// M3-6 Task 9: CEC 2017 (`sezgi_problems::cec2017::Cec2017`, fid `{1}
    /// union {3..=30}`; fid 2 is officially withdrawn -- `Cec2017::new`
    /// itself rejects it before a handle can ever exist, see `cec2017(...)`'s
    /// own doc). Same handle shape as `Inner::Cec2022`/`Inner::Cec2014`.
    Cec2017(Cec2017),
    Tsp(Tsp),
    /// M3-8 Task 9: `sezgi.problems.onemax(n_bits)` -- Goldberg 1989's
    /// classic Binary-block diagnostic (`sezgi_problems::diagnostics::OneMax`).
    /// Solve()-eligible exactly like `Inner::Bbob`; pairs with
    /// `sezgi.presets.ga_bin(...)`.
    OneMax(OneMax),
    /// M3-8 Task 9: `sezgi.problems.int_quadratic(lo, hi, n)` -- an Int-block
    /// diagnostic (`sezgi_problems::diagnostics::IntQuadratic`). Pairs with
    /// `sezgi.presets.ga_int(...)`.
    IntQuadratic(IntQuadratic),
    /// M3-8 Task 9: `sezgi.problems.cat_match(k, n, seed)` -- a Categorical-
    /// block diagnostic (`sezgi_problems::diagnostics::CatMatch`). Pairs with
    /// `sezgi.presets.ga_cat(...)`.
    CatMatch(CatMatch),
    /// M3-8 Task 9: `sezgi.problems.mixed_diagnostic(n_float, n_int, k_cat,
    /// n_cat, n_bin)` -- a Float+Int+Categorical+Binary mixed-space
    /// scaffold problem, added SOLELY so `gen/compound` (M3-8 Task 5) is
    /// reachable end to end from a Python-authored, mixed-space TOML
    /// `AlgorithmSpec` (parsed via Python's stdlib `tomllib` into the same
    /// dict `solve()` already accepts -- no new solve-side API). Mirrors
    /// `crates/components/src/compound.rs`'s own test-local `MixedProblem`
    /// exactly (same block bounds, same `evaluate_batch` formula) -- see
    /// [`MixedDiagnostic`]'s own doc. Not one of Task 5's brief-pinned
    /// diagnostics (those are each single-block); `optimum()` returns
    /// `None` accordingly -- no verified target is claimed.
    Mixed(MixedDiagnostic),
    Callable { f: Py<PyAny>, space: SearchSpace, vectorized: bool },
    /// `sezgi.bias.f0(dim, seed)` -- the BIAS-toolbox null problem
    /// ([`F0Random`]). `dim`/`seed` are stored (not an `F0Random` instance
    /// itself) so `for_problem`/`solve()` can rebuild a fresh, independently
    /// seeded instance for each run -- the same "rebuild from stored params"
    /// pattern `Inner::Bbob` uses, and necessary here besides: `F0Random`
    /// holds a `Mutex<RngStream>` (interior mutability, not `Clone`), so a
    /// single stored instance could not be shared/reused across runs even if
    /// we wanted to. `space` is precomputed at construction time so
    /// `dim()`/`bounds()` need no `F0Random` instance at all.
    F0 { dim: usize, seed: u64, space: SearchSpace },
}

/// Mixed Float+Int+Categorical+Binary scaffold problem (M3-8 Task 9), living
/// only in this crate (not `crates/problems`) -- it exists purely to give
/// `sezgi.solve()` a Python-reachable target whose search space has one
/// block of each non-Permutation kind, so a mixed-space TOML `AlgorithmSpec`
/// using `gen/compound` (Task 5) can be run end to end from Python. Mirrors
/// `crates/components/src/compound.rs`'s own test-local `MixedProblem`
/// byte-for-byte: `Block::Float{-5,5,n_float}` + `Block::Int{-5,5,n_int}` +
/// `Block::Categorical{k_cat,n_cat}` + `Block::Binary{n_bin}`, and
/// `evaluate_batch` = (sum of the float block) + (sum of the int block) +
/// (count of categorical genes != 0) + (count of `false` bits). Not a
/// benchmark and not one of Task 5's brief-pinned diagnostics -- `optimum()`
/// returns `None` (no verified reachable target is claimed; a real value
/// would need to account for the float block's continuous infimum, which no
/// finite-budget GA run is guaranteed to hit exactly).
struct MixedDiagnostic {
    space: SearchSpace,
}

impl MixedDiagnostic {
    fn new(n_float: usize, n_int: usize, k_cat: u32, n_cat: usize, n_bin: usize) -> Self {
        let space = SearchSpace::new(vec![
            Block::Float { lo: -5.0, hi: 5.0, n: n_float },
            Block::Int { lo: -5, hi: 5, n: n_int },
            Block::Categorical { k: k_cat, n: n_cat },
            Block::Binary { n: n_bin },
        ])
        .expect("Float{-5,5,..}/Int{-5,5,..} bounds are fixed and valid (lo < hi); \
                 Categorical/Binary have no bounds for SearchSpace::new to reject");
        Self { space }
    }
}

impl Problem for MixedDiagnostic {
    fn space(&self) -> &SearchSpace { &self.space }

    fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<f64> {
        pop.iter()
            .map(|g| {
                g.blocks.iter().fold(0.0, |f, b| f + match b {
                    BlockValues::Float(xs) => xs.iter().sum::<f64>(),
                    BlockValues::Int(xs) => xs.iter().sum::<i64>() as f64,
                    BlockValues::Cat(xs) => xs.iter().filter(|&&c| c != 0).count() as f64,
                    BlockValues::Bin(xs) => xs.iter().filter(|&&b| !b).count() as f64,
                    BlockValues::Perm(_) => 0.0,
                })
            })
            .collect()
    }
}

#[pyclass(name = "Problem")]
struct PyProblem { inner: Inner }

/// Shared calling convention for a `from_callable(...)` handle's Python
/// callback `f`, dispatched on `vectorized` -- the flag lives on the HANDLE
/// (`Inner::Callable`/`from_callable`'s own `vectorized` parameter), not on
/// the consumer, so a given handle has exactly ONE contract everywhere it is
/// used (`solve()`'s [`CallableProblem`] and
/// `EvalSession::for_problem`'s [`OwnedCallableProblem`] both call this).
///
/// - `vectorized=true` (the default): `f` is called ONCE per
///   `evaluate_batch` call, with the WHOLE population written into a single
///   2-D `(n, d)` float64 numpy array, and must return `n` values (a numpy
///   array or a plain list) -- this is the ORIGINAL, frozen convention
///   (unchanged byte-for-byte: `solve()`'s existing tests, including "must
///   be a SINGLE call per population", still pass unmodified).
/// - `vectorized=false`: `f` is called ONCE PER POINT, with a 1-D
///   length-`dim` float64 numpy array, and must return a scalar `float`.
///   Natural for `EvalSession`'s ask/tell callers, which evaluate
///   individually-generated candidate points, and for an ordinary
///   single-point objective function generally -- without requiring every
///   `from_callable` caller to write batch-aware code.
fn call_callable(f: &Py<PyAny>, vectorized: bool, pop: &[Genotype]) -> Vec<f64> {
    Python::with_gil(|py| {
        if vectorized {
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
            let out = match f.call1(py, (arr,)) {
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
        } else {
            pop.iter().map(|g| {
                let BlockValues::Float(v) = &g.blocks[0] else {
                    unreachable!("from_callable only builds Float spaces")
                };
                let arr = PyArray1::from_vec(py, v.clone());
                let out = match f.call1(py, (arr,)) {
                    Ok(o) => o,
                    Err(e) => panic::panic_any(e),
                };
                match out.extract::<f64>(py) {
                    Ok(f) => f,
                    Err(e) => panic::panic_any(e),
                }
            }).collect()
        }
    })
}

/// Borrowed [`Problem`] wrapper around a `from_callable(...)` handle, used
/// by `solve()` (which runs synchronously and can borrow the `PyProblem`'s
/// own fields for the run's lifetime). Calling convention: see
/// [`call_callable`]'s doc.
struct CallableProblem<'a> { f: &'a Py<PyAny>, space: &'a SearchSpace, vectorized: bool }

impl Problem for CallableProblem<'_> {
    fn space(&self) -> &SearchSpace { self.space }
    fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<f64> { call_callable(self.f, self.vectorized, pop) }
}

/// Owned counterpart of [`CallableProblem`], used only by
/// `EvalSession::for_problem` (which needs a `'static` `Box<dyn Problem>`,
/// unlike `solve()`'s borrow-for-the-run-lifetime use of `CallableProblem`
/// above). Same calling convention (dispatched on the SAME `vectorized`
/// flag carried on the handle) -- see [`call_callable`]'s doc.
struct OwnedCallableProblem { f: Py<PyAny>, space: SearchSpace, vectorized: bool }

impl Problem for OwnedCallableProblem {
    fn space(&self) -> &SearchSpace { &self.space }
    fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<f64> { call_callable(&self.f, self.vectorized, pop) }
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
            Inner::Cec2014(p) => p.space().dim(),
            Inner::Cec2017(p) => p.space().dim(),
            Inner::Tsp(p) => p.space().dim(),
            Inner::OneMax(p) => p.space().dim(),
            Inner::IntQuadratic(p) => p.space().dim(),
            Inner::CatMatch(p) => p.space().dim(),
            Inner::Mixed(p) => p.space().dim(),
            Inner::Callable { space, .. } => space.dim(),
            Inner::F0 { space, .. } => space.dim(),
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
            Inner::Cec2014(p) => p.space(),
            Inner::Cec2017(p) => p.space(),
            Inner::Tsp(p) => p.space(),
            Inner::OneMax(p) => p.space(),
            Inner::IntQuadratic(p) => p.space(),
            Inner::CatMatch(p) => p.space(),
            Inner::Mixed(p) => p.space(),
            Inner::Callable { space, .. } => space,
            Inner::F0 { space, .. } => space,
        };
        bounds_of(space)
    }

    /// The problem's known optimum, or `None` if it has none (a
    /// `from_callable` handle always returns `None`: an arbitrary Python
    /// function has no analytically known optimum; `sezgi.bias.f0(...)`
    /// likewise -- it has no landscape at all, see `sezgi_bias::f0`'s module
    /// doc).
    fn optimum(&self) -> Option<f64> {
        match &self.inner {
            Inner::Bbob(p) => p.optimum(),
            Inner::Cec2022(p) => p.optimum(),
            Inner::Cec2014(p) => p.optimum(),
            Inner::Cec2017(p) => p.optimum(),
            Inner::Tsp(p) => p.optimum(),
            Inner::OneMax(p) => p.optimum(),
            Inner::IntQuadratic(p) => p.optimum(),
            Inner::CatMatch(p) => p.optimum(),
            // MixedDiagnostic (Task 9 scaffold): no verified target claimed
            // -- see its own doc.
            Inner::Mixed(_) => None,
            Inner::Callable { .. } => None,
            Inner::F0 { .. } => None,
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

/// `sezgi.from_callable(f, lo, hi, dim, vectorized=True)` -- a [`Problem`]
/// handle wrapping a Python function `f`. `vectorized` fixes `f`'s calling
/// convention for every consumer of this handle (`solve()`,
/// `EvalSession.for_problem`) -- see [`call_callable`]'s doc for the exact
/// contract of each value. Defaults to `True`: the original, frozen
/// population-batched convention, unchanged for every existing caller.
#[pyfunction]
#[pyo3(signature = (f, lo, hi, dim, vectorized=true))]
fn from_callable(f: Py<PyAny>, lo: f64, hi: f64, dim: usize, vectorized: bool) -> PyResult<PyProblem> {
    let space = SearchSpace::new(vec![Block::Float { lo, hi, n: dim }])
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyProblem { inner: Inner::Callable { f, space, vectorized } })
}

/// `sezgi.bias.f0(dim, seed)` -- a [`Problem`] handle for
/// [`sezgi_bias::f0::F0Random`], the BIAS-toolbox null problem (M3-4 Task
/// 4): every evaluation is an independent U(0,1) draw, uncorrelated with the
/// queried point, over the domain `[0,1]^dim`. Usable with `EvalSession.
/// for_problem` / `Algorithm.solve` exactly like any other continuous
/// [`Problem`] handle; `optimum()` is always `None` (there is no landscape
/// to have an optimum), so `log_dir` is rejected the same way it is for
/// `from_callable(...)` (nothing to record as `f_opt`). Collecting `best_x`
/// over repeated runs and passing them to `sezgi.bias.structural_positions`
/// is how a researcher's OWN algorithm gets the library's structural-bias
/// scan without spec-driven engine involvement.
#[pyfunction]
fn bias_f0(dim: usize, seed: u64) -> PyResult<PyProblem> {
    let space = SearchSpace::new(vec![Block::Float {
        lo: bias_f0_mod::F0_LO,
        hi: bias_f0_mod::F0_HI,
        n: dim,
    }])
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyProblem { inner: Inner::F0 { dim, seed, space } })
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

/// `sezgi.problems.cec2014(fid, dim)` -- a [`Problem`] handle for a CEC 2014
/// function (`sezgi_problems::Cec2014::new`), usable with `solve()` and
/// `EvalSession.for_problem` exactly like `sezgi.problems.cec2022(...)`. See
/// [`Cec2014::new`]'s own doc for the exact `fid`/`dim` domain.
///
/// # Errors
/// `ValueError` for [`sezgi_problems::Cec2014Error`]: `fid` outside `1..=30`,
/// or `dim` outside `{10, 30}`.
#[pyfunction]
fn cec2014(fid: u32, dim: usize) -> PyResult<PyProblem> {
    Ok(PyProblem { inner: Inner::Cec2014(
        Cec2014::new(fid, dim).map_err(|e| PyValueError::new_err(e.to_string()))?) })
}

/// `sezgi.problems.cec2014_evaluate(fid, dim, x)` -- direct, one-shot
/// evaluation of a CEC 2014 function at `x` (a length-`dim` list of floats),
/// bypassing `solve()`'s budget/engine machinery entirely. Same shape as
/// `cec2022_evaluate(...)`.
///
/// # Errors
/// `ValueError` for the same [`sezgi_problems::Cec2014Error`] cases as
/// `cec2014(...)`, plus a `ValueError` if `len(x) != dim`.
#[pyfunction]
fn cec2014_evaluate(fid: u32, dim: usize, x: Vec<f64>) -> PyResult<f64> {
    let p = Cec2014::new(fid, dim).map_err(|e| PyValueError::new_err(e.to_string()))?;
    if x.len() != dim {
        return Err(PyValueError::new_err(format!(
            "x must have exactly {dim} coordinates (dim={dim}), got {}", x.len())));
    }
    let g = Genotype { blocks: vec![BlockValues::Float(x)] };
    Ok(p.evaluate_batch(&[g])[0])
}

/// `sezgi.problems.cec2014_f_star(fid)` -- the pinned `F_i* = 100*fid` bias
/// ([`Cec2014::f_star`]). Does not depend on `dim`; an internal probe
/// `dim=10` validates `fid` only.
///
/// # Errors
/// `ValueError` if `fid` is outside `1..=30`.
#[pyfunction]
fn cec2014_f_star(fid: u32) -> PyResult<f64> {
    let p = Cec2014::new(fid, 10).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(p.f_star())
}

/// `sezgi.problems.cec2017(fid, dim)` -- a [`Problem`] handle for a CEC 2017
/// function (`sezgi_problems::Cec2017::new`), usable with `solve()` and
/// `EvalSession.for_problem` exactly like `sezgi.problems.cec2022(...)`. See
/// [`Cec2017::new`]'s own doc for the exact `fid`/`dim` domain.
///
/// # Errors
/// `ValueError` for [`sezgi_problems::Cec2017Error`]: `fid` outside `{1}
/// union {3..=30}`, `fid == 2` (officially withdrawn -- the Rust
/// [`sezgi_problems::Cec2017Error::Withdrawn`] message is surfaced VERBATIM
/// so a caller sees the honest reason, distinct from an ordinary
/// out-of-range `fid`), or `dim` outside `{10, 30}`.
#[pyfunction]
fn cec2017(fid: u32, dim: usize) -> PyResult<PyProblem> {
    Ok(PyProblem { inner: Inner::Cec2017(
        Cec2017::new(fid, dim).map_err(|e| PyValueError::new_err(e.to_string()))?) })
}

/// `sezgi.problems.cec2017_evaluate(fid, dim, x)` -- direct, one-shot
/// evaluation of a CEC 2017 function at `x` (a length-`dim` list of floats),
/// bypassing `solve()`'s budget/engine machinery entirely. Same shape as
/// `cec2022_evaluate(...)`.
///
/// # Errors
/// `ValueError` for the same [`sezgi_problems::Cec2017Error`] cases as
/// `cec2017(...)` (withdrawn `fid == 2` included), plus a `ValueError` if
/// `len(x) != dim`.
#[pyfunction]
fn cec2017_evaluate(fid: u32, dim: usize, x: Vec<f64>) -> PyResult<f64> {
    let p = Cec2017::new(fid, dim).map_err(|e| PyValueError::new_err(e.to_string()))?;
    if x.len() != dim {
        return Err(PyValueError::new_err(format!(
            "x must have exactly {dim} coordinates (dim={dim}), got {}", x.len())));
    }
    let g = Genotype { blocks: vec![BlockValues::Float(x)] };
    Ok(p.evaluate_batch(&[g])[0])
}

/// `sezgi.problems.cec2017_f_star(fid)` -- the pinned `F_i* = 100*fid` bias
/// ([`Cec2017::f_star`]). Does not depend on `dim`; an internal probe
/// `dim=10` validates `fid` only.
///
/// # Errors
/// `ValueError` if `fid` is outside `{1} union {3..=30}` (`fid == 2`
/// included, via the [`sezgi_problems::Cec2017Error::Withdrawn`] message).
#[pyfunction]
fn cec2017_f_star(fid: u32) -> PyResult<f64> {
    let p = Cec2017::new(fid, 10).map_err(|e| PyValueError::new_err(e.to_string()))?;
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

/// `sezgi.problems.onemax(n_bits)` -- a [`Problem`] handle for
/// [`sezgi_problems::diagnostics::OneMax`] (M3-8 Task 9), the classic
/// Binary-block GA diagnostic (Goldberg 1989; this crate's own
/// minimize-the-zero-bit-count re-expression -- see that module's doc).
/// Solve()-eligible exactly like `sezgi.bbob(...)`. Pairs with
/// `sezgi.presets.ga_bin(pop_size, budget)`.
#[pyfunction]
fn onemax(n_bits: usize) -> PyProblem {
    PyProblem { inner: Inner::OneMax(OneMax::new(n_bits)) }
}

/// `sezgi.problems.int_quadratic(lo, hi, n)` -- a [`Problem`] handle for
/// [`sezgi_problems::diagnostics::IntQuadratic`] (M3-8 Task 9), an Int-block
/// quadratic bowl around a fixed target derived deterministically from
/// `(lo, hi, n)` (see that module's own doc; the constructor takes no `seed`
/// argument, by the Task 5 brief's pinned signature). Pairs with
/// `sezgi.presets.ga_int(pop_size, budget)`.
///
/// # Errors
/// `ValueError` if `lo >= hi` -- checked explicitly here (rather than
/// letting `IntQuadratic::new`'s own internal `SearchSpace::new(...).expect(..)`
/// panic) so a bad Python call gets a clean exception instead of a Rust
/// panic, matching every other bounds-checked constructor in this file.
#[pyfunction]
fn int_quadratic(lo: i64, hi: i64, n: usize) -> PyResult<PyProblem> {
    if lo >= hi {
        return Err(PyValueError::new_err(format!(
            "int_quadratic: lo ({lo}) must be < hi ({hi})")));
    }
    Ok(PyProblem { inner: Inner::IntQuadratic(IntQuadratic::new(lo, hi, n)) })
}

/// `sezgi.problems.cat_match(k, n, seed)` -- a [`Problem`] handle for
/// [`sezgi_problems::diagnostics::CatMatch`] (M3-8 Task 9), a Categorical-
/// block Hamming-distance-to-target matching problem; unlike
/// `int_quadratic`, `seed` IS an explicit caller-supplied parameter (Task 5
/// brief's pinned signature). Pairs with `sezgi.presets.ga_cat(pop_size,
/// budget)`.
#[pyfunction]
fn cat_match(k: u32, n: usize, seed: u64) -> PyProblem {
    PyProblem { inner: Inner::CatMatch(CatMatch::new(k, n, seed)) }
}

/// `sezgi.problems.mixed_diagnostic(n_float, n_int, k_cat, n_cat, n_bin)` --
/// a [`Problem`] handle for [`MixedDiagnostic`] (M3-8 Task 9), a
/// Float+Int+Categorical+Binary mixed-space scaffold problem added SOLELY
/// so `gen/compound` (Task 5) is reachable end to end through a
/// Python-authored, mixed-space TOML `AlgorithmSpec` -- see
/// [`MixedDiagnostic`]'s own doc for the exact block layout/`evaluate_batch`
/// formula (mirrors `crates/components/src/compound.rs`'s own test-local
/// `MixedProblem`). Not a benchmark, not one of Task 5's brief-pinned
/// diagnostics; `optimum()` is always `None`.
#[pyfunction]
fn mixed_diagnostic(n_float: usize, n_int: usize, k_cat: u32, n_cat: usize, n_bin: usize) -> PyProblem {
    PyProblem { inner: Inner::Mixed(MixedDiagnostic::new(n_float, n_int, k_cat, n_cat, n_bin)) }
}

/// Raised by every [`PyEvalSession`] method once the session has been
/// [`PyEvalSession::finish`]ed (including a second call to `finish` itself).
fn session_finished_err() -> PyErr {
    PyValueError::new_err("session finished")
}

/// Path tag folded into [`PermSession`]'s [`RngStream`] (see
/// [`RngStream::from_master`]) -- an arbitrary but FIXED value distinguishing
/// this stream from any other stream this crate might ever derive from the
/// same `seed`. Pinned as of `random_permutation()`'s introduction (M3-8
/// Task 7); changing it would silently change every future draw sequence.
const PERM_SESSION_RNG_TAG: u64 = 0x5045524D; // "PERM", arbitrary ASCII-hex mnemonic

/// Task 7 (M3-8): ask/tell session over a PERMUTATION-typed (e.g. TSP)
/// [`Problem`] handle -- the typed counterpart [`PyEvalSession::for_problem`]
/// now builds for `sezgi.problems.tsp(...)` instead of rejecting it (the
/// rejection this replaces used to live right where [`Self::new`] is called
/// from `for_problem` below).
///
/// Lives entirely in THIS crate (not `crates/bench`), deliberately: this
/// task's change surface is `py-sezgi/src/lib.rs` only (the
/// `cargo test --workspace` gate must report the same PASS/FAIL numbers as
/// before this task -- per the M3-8 Task 7 approved scope ruling).
/// [`sezgi_bench::EvalSession`] itself is not reused because its own
/// `evaluate` is hard-coded to `Vec<Vec<f64>>` rows / `BlockValues::Float`
/// genotypes (`crates/bench/src/session.rs`) -- widening THAT signature
/// would touch a workspace crate outside py-sezgi. Instead this struct is
/// built directly on [`sezgi_core::problem::Evaluator`] -- the SAME
/// tamper-proof budget-check/counting primitive `EvalSession` itself is
/// built on (see that module's own doc for the reuse rationale) -- so the
/// budget/counting/best-tracking behavior is the identical primitive, not a
/// re-implementation of it, even though the two session types don't share a
/// common Rust struct.
///
/// Unlike `EvalSession` (which owns no RNG of its own: candidate points are
/// entirely caller-supplied, and a Float-typed `Algorithm` draws from
/// `AlgoContext.rng`, a plain Python `random.Random(seed)`), this session
/// owns a seeded [`RngStream`] for [`Self::random_permutation`] -- a
/// permutation-typed algorithm has no Python-side equivalent of
/// `random_point()` to draw a candidate from, so the shuffle itself must
/// come from somewhere deterministic. It draws from THIS session's own
/// house `RngStream` (seeded from the same `seed` `for_problem` was built
/// with, folded through [`PERM_SESSION_RNG_TAG`]) rather than Python's
/// `random` module, so a run is reproducible the same way every other
/// seeded draw in this codebase is (`crates/core/src/rng.rs`).
struct PermSession {
    problem: Box<dyn Problem>,
    n: usize,
    budget: u64,
    used: u64,
    best: Option<(Vec<u32>, f64)>,
    rng: RngStream,
}

impl PermSession {
    fn new(problem: Box<dyn Problem>, seed: u64, budget: u64) -> Self {
        let n = problem.space().dim(); // Block::Permutation { n }.dim() == n
        let rng = RngStream::from_master(seed, &[PERM_SESSION_RNG_TAG]);
        Self { problem, n, budget, used: 0, best: None, rng }
    }

    /// A uniformly random permutation of `0..n` (a 0-based tour, per this
    /// task's Python-side convention -- R's Task 8 twin is 1-based), drawn
    /// from this session's own [`RngStream`]. Delegates to
    /// [`sezgi_components::perm::fisher_yates_shuffle`] -- the SAME
    /// Fisher-Yates/Durstenfeld shuffle (for `i` from `n-1` down to `1`,
    /// swap `v[i]` with `v[j]` for a uniformly random `j` in `0..=i`, via
    /// `RngStream::next_below`'s rejection sampling, no modulo bias) that
    /// `gen/perm-swap`/`gen/ox`'s own init path already uses (`perm.rs`'s
    /// module doc), reused here rather than re-implemented inline (fix
    /// round 1 review) -- bit-identical draws, same struct field, same
    /// `RngStream`, just called through the shared, already-tested core.
    /// Each call advances the stream, so successive calls draw DIFFERENT
    /// permutations; the same `seed` and call sequence reproduce the same
    /// permutations every time.
    fn random_permutation(&mut self) -> Vec<u32> {
        fisher_yates_shuffle(self.n, &mut self.rng)
    }

    /// Batch-evaluates `tours` (0-based; each must be a permutation of
    /// `0..n`). All-or-nothing, mirroring `EvalSession::evaluate`'s own
    /// error semantics (`crates/bench/src/session.rs`'s module doc): every
    /// row is validated BEFORE the counter moves, so a rejected batch (any
    /// row the wrong length, containing an out-of-range city, or containing
    /// a repeated city) leaves `used` untouched and evaluates nothing --
    /// same all-or-nothing contract as the Float path's dimension/
    /// non-finite pre-pass, plus the budget check itself (via
    /// [`Evaluator::evaluate`], the same primitive `EvalSession` is built
    /// on).
    fn evaluate(&mut self, tours: &[Vec<i64>]) -> PyResult<Vec<f64>> {
        for (row, t) in tours.iter().enumerate() {
            if t.len() != self.n {
                return Err(PyValueError::new_err(format!(
                    "PermSession::evaluate: row {row} has {got} entries, expected {n} \
                     (one per city)", got = t.len(), n = self.n)));
            }
            let mut seen = vec![false; self.n];
            for &c in t {
                if c < 0 || c as usize >= self.n {
                    return Err(PyValueError::new_err(format!(
                        "PermSession::evaluate: row {row} has out-of-range entry {c} \
                         (expected 0..{n})", n = self.n)));
                }
                let ci = c as usize;
                if seen[ci] {
                    return Err(PyValueError::new_err(format!(
                        "PermSession::evaluate: row {row} has a repeated city {ci}; \
                         not a valid permutation")));
                }
                seen[ci] = true;
            }
        }

        let pop: Vec<Genotype> = tours.iter()
            .map(|t| Genotype {
                blocks: vec![BlockValues::Perm(t.iter().map(|&c| c as u32).collect())],
            })
            .collect();

        // Short-lived Evaluator sized to the remaining budget -- same reuse
        // pattern as EvalSession::evaluate (crates/bench/src/session.rs):
        // its own all-or-nothing check IS this session's check.
        let remaining = self.budget - self.used;
        let mut ev = Evaluator::new(&*self.problem, remaining);
        let fs = ev.evaluate(&pop).map_err(|e| PyValueError::new_err(format!(
            "PermSession::evaluate: budget exceeded ({used}/{budget} used, \
             {requested} requested)", used = self.used, budget = self.budget,
            requested = e.requested)))?;

        for (t, &f) in tours.iter().zip(&fs) {
            self.used += 1;
            // Same tie rule as Evaluator::evaluate / EvalSession::evaluate:
            // update only on strict improvement, ties keep the earlier tour.
            let improved = !matches!(&self.best, Some((_, b)) if *b <= f);
            if improved {
                self.best = Some((t.iter().map(|&c| c as u32).collect(), f));
            }
        }
        Ok(fs)
    }

    fn evals_used(&self) -> u64 { self.used }
    fn budget(&self) -> u64 { self.budget }
    fn best(&self) -> Option<(Vec<u32>, f64)> { self.best.clone() }
    /// The problem's known optimum, or `None` if it has none. A vendored TSP
    /// instance (`sezgi.problems.tsp(...)`) returns `Some` for all three
    /// (`berlin52`/`eil51`/`st70`, `tsp.rs`'s pinned published optima).
    fn f_opt(&self) -> Option<f64> { self.problem.optimum() }
}

/// One [`PyEvalSession`]'s underlying Rust session -- `Float` for every
/// continuous problem handle (BBOB, CEC 2022/2014/2017, `from_callable`,
/// `bias.f0`), `Perm` for a permutation-typed one (`sezgi.problems.tsp`,
/// Task 7). `PyEvalSession::kind()` surfaces which one a given session is
/// (`"float"`/`"permutation"`) to Python, driving `AlgoContext.kind`/
/// `AlgoContext.n`/`AlgoContext.bounds` in `py-sezgi/python/sezgi/algo.py`.
enum SessionKind {
    Float(EvalSession),
    Perm(PermSession),
}

/// Python binding for [`sezgi_bench::EvalSession`] (Float path) / the
/// crate-local [`PermSession`] (permutation path) — the ask/tell core
/// behind the spec's engine-inside-out promise: an external (here, pure
/// Python) algorithm generates candidate points and this session stays the
/// sole keeper of evaluation, tamper-proof counting, best-tracking and (for
/// the Float path) IOH logging.
///
/// `finish()` consumes the underlying Rust session (matching its own
/// consuming signature); since `#[pymethods]` cannot take `self` by value
/// through a Python handle, the consuming step is modeled with an
/// `Option<SessionKind>` inner slot that `finish` takes, leaving `None`
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
    inner: Option<SessionKind>,
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
        Ok(Self { inner: Some(SessionKind::Float(session)) })
    }

    /// Builds a session over any [`PyProblem`] handle -- continuous
    /// (`sezgi.bbob(...)`, `sezgi.problems.cec2022(...)`,
    /// `sezgi.from_callable(...)`, `sezgi.bias.f0(...)`) OR, as of Task 7
    /// (M3-8), permutation-typed (`sezgi.problems.tsp(...)`). A
    /// permutation-typed handle builds a crate-local [`PermSession`]
    /// instead (see its own doc) -- `PyEvalSession::kind()` tells the two
    /// apart from Python. See [`sezgi_bench::EvalSession::new_owned`] for
    /// the continuous-path generalization the Float branch below delegates
    /// to; each of ITS `Inner` arms builds its own [`SessionMeta`]
    /// (suite/fid/name/instance/f_opt), so adding a new continuous-problem
    /// arm elsewhere in this crate is a self-contained extension of that
    /// match.
    ///
    /// # Errors
    /// - `ValueError` if `tour` validation would ever be needed here (it
    ///   isn't -- `tour` validation happens per-batch in
    ///   [`PermSession::evaluate`], not at session-construction time).
    /// - `ValueError` if `log_dir` is given for `sezgi.problems.tsp(...)`:
    ///   IOH logging is not currently wired up for permutation-typed
    ///   sessions (no suite/fid identity exists for a TSP run to log
    ///   against, and this task's scope is the minimal ask/tell surface,
    ///   not IOH support).
    /// - `ValueError` if `log_dir` is given for `sezgi.from_callable(...)` or
    ///   `sezgi.bias.f0(...)`: IOH logging is restricted to problems with a
    ///   real fid identity and a known optimum (BBOB, CEC 2022, CEC 2014, and
    ///   CEC 2017), matching `solve()`'s existing policy. A known optimum
    ///   is necessary but not sufficient — the on-disk IOH record key needs
    ///   a suite discriminator too (`RunKey::suite`, M3-5 Task 1), so a
    ///   non-BBOB run sharing `(fid, dim, instance, seed, budget)` with a
    ///   BBOB run does not silently merge into one `results_matrix` cell.
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
        // Permutation path (Task 7): handled up front, entirely separately
        // from the continuous match below -- PermSession is a different
        // Rust type with no SessionMeta/EvalSession::new_owned involvement.
        if let Inner::Tsp(p) = &problem.inner {
            if log_dir.is_some() {
                return Err(PyValueError::new_err(
                    "IOH logging is not currently supported for permutation-typed \
                     (e.g. TSP) problems"));
            }
            // Fresh instance rebuilt from the stored vendored name: same
            // "rebuild fresh" pattern every continuous arm below uses,
            // though Tsp itself holds no RNG state to reseed (unlike
            // BbobProblem) -- Tsp::vendored is a pure function of its name,
            // so this is bit-identical to `p` regardless.
            let fresh = Tsp::vendored(p.name())
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let session = PermSession::new(Box::new(fresh), seed, budget);
            return Ok(Self { inner: Some(SessionKind::Perm(session)) });
        }

        // M3-8 Task 9: onemax/int_quadratic/cat_match/mixed_diagnostic have
        // no ask/tell session type of their own (no "BinSession"/
        // "IntSession"/"CatSession"/"MixedSession" analogous to PermSession
        // exists -- out of this task's scope, which is solve()-only for
        // these typed families). Rejected up front with an honest message
        // rather than silently mis-building a Float-typed EvalSession over a
        // non-Float space.
        if matches!(&problem.inner,
            Inner::OneMax(_) | Inner::IntQuadratic(_) | Inner::CatMatch(_) | Inner::Mixed(_)) {
            return Err(PyValueError::new_err(
                "EvalSession.for_problem is not currently supported for onemax(...)/\
                 int_quadratic(...)/cat_match(...)/mixed_diagnostic(...) problems (no \
                 ask/tell session type exists yet for Binary/Int/Categorical/mixed-typed \
                 spaces); use sezgi.solve(...) with presets.ga_bin/ga_int/ga_cat instead"));
        }

        let (boxed, meta): (Box<dyn Problem>, SessionMeta) = match &problem.inner {
            Inner::Bbob(p) => {
                // Fresh instance from the same (fid, dim, instance): matches
                // `EvalSession::new_bbob`'s own construction exactly (BBOB's
                // instance-seeded RNG makes this deterministic and
                // bit-identical to `p`).
                let fresh = BbobProblem::new(p.fid(), p.space().dim(), p.instance)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let meta = SessionMeta {
                    suite: SUITE_BBOB.into(),
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
            // M3-6 Task 9: same shape as Inner::Cec2022 -- a fresh instance
            // rebuilt from (fid, dim) (Cec2014/Cec2017 hold no RNG state, so
            // this is bit-identical to `p`, same as Cec2022's own rebuild).
            // Suite/name follow the M3-5 label helper's convention
            // (`problem_segment` in crates/bench/src/experiment.rs): suite
            // `"sezgi-cec2014"`/`"sezgi-cec2017"` strips its `sezgi-` prefix
            // to `cec2014`/`cec2017`, so `name` = `"cec2014-f{fid}"` /
            // `"cec2017-f{fid}"` combines with `d{dim}i{instance}` (appended
            // by that shared helper) into the exact `cec2014-f1d10i1` /
            // `cec2017-f1d10i1` label shape.
            Inner::Cec2014(p) => {
                let fresh = Cec2014::new(p.fid(), p.dim())
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let meta = SessionMeta {
                    suite: "sezgi-cec2014".into(),
                    fid: p.fid(),
                    name: format!("cec2014-f{}", p.fid()),
                    instance: 1,
                    f_opt: p.optimum(),
                };
                (Box::new(fresh), meta)
            }
            Inner::Cec2017(p) => {
                let fresh = Cec2017::new(p.fid(), p.dim())
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let meta = SessionMeta {
                    suite: "sezgi-cec2017".into(),
                    fid: p.fid(),
                    name: format!("cec2017-f{}", p.fid()),
                    instance: 1,
                    f_opt: p.optimum(),
                };
                (Box::new(fresh), meta)
            }
            Inner::Callable { f, space, vectorized } => {
                let owned = OwnedCallableProblem {
                    f: f.clone_ref(py), space: space.clone(), vectorized: *vectorized,
                };
                let meta = SessionMeta {
                    suite: "sezgi-custom".into(),
                    fid: 0,
                    name: "callable".into(),
                    instance: 1,
                    f_opt: None,
                };
                (Box::new(owned), meta)
            }
            Inner::F0 { dim, seed, .. } => {
                // Fresh instance from the stored (dim, seed): F0Random is
                // not Clone (interior-mutable RNG stream), and rebuilding is
                // deterministic anyway (F0Random::new derives its stream
                // purely from dim/seed) -- see the `Inner::F0` doc.
                let fresh = F0Random::new(*dim, *seed);
                let meta = SessionMeta {
                    suite: "sezgi-f0".into(),
                    fid: 0,
                    name: "f0".into(),
                    instance: 1,
                    f_opt: None,
                };
                (Box::new(fresh), meta)
            }
            // Handled and returned above -- this match is unreachable for
            // Inner::Tsp, but the match must still be exhaustive.
            Inner::Tsp(_) => unreachable!("Inner::Tsp is handled and returned before this match"),
            Inner::OneMax(_) | Inner::IntQuadratic(_) | Inner::CatMatch(_) | Inner::Mixed(_) =>
                unreachable!("rejected and returned before this match"),
        };

        // sezgi decision (M3-5 scope ruling 2, widened again by M3-6 Task 9):
        // IOH logging via for_problem is restricted to sessions whose
        // SessionMeta carries a real fid identity and a known optimum --
        // BBOB, CEC 2022, CEC 2014, and CEC 2017 -- not to "any problem with
        // a known optimum" (Callable/F0 have neither). M3-4's final review
        // blocked CEC 2022 logging because the on-disk IOH record key
        // (algo, fid, dim, instance, seed, budget) carried no suite
        // discriminator, so a CEC 2022 run and a BBOB run sharing that key
        // would silently merge into one results_matrix cell. M3-5 Task 1
        // added a "suite" discriminator to that record key (RunKey::suite,
        // threaded through read_ioh_records/results_matrix), closing that
        // gap, so CEC 2022 was admitted; CEC 2014/CEC 2017 ride the same
        // suite-aware machinery (RunKey::suite/problem_segment derive their
        // folder/label from the suite string automatically -- no ioh.rs
        // change needed), so they are admitted the same way. Callable and F0
        // arms keep the rejection: their SessionMeta has no fid identity or
        // f_opt, so there is nothing to build an IOH archive against
        // (with_log itself also rejects a None f_opt) -- see
        // docs/DECISIONS.md's M3-5 record.
        if log_dir.is_some() && !matches!(&problem.inner,
            Inner::Bbob(_) | Inner::Cec2022(_) | Inner::Cec2014(_) | Inner::Cec2017(_)) {
            return Err(PyValueError::new_err(
                "IOH logging is currently supported for BBOB, CEC 2022, CEC 2014, \
                 and CEC 2017 problems only"));
        }

        let mut session = EvalSession::new_owned(boxed, meta, budget)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if let Some(dir) = log_dir {
            session = session
                .with_log(Path::new(dir), algo_name, seed)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
        Ok(Self { inner: Some(SessionKind::Float(session)) })
    }

    /// Batch-evaluates `xs`: a list of rows, each a list of `dim` floats
    /// for a Float-typed session, or each a length-`n` 0-based tour (a
    /// permutation of `0..n`) for a permutation-typed one (`kind() ==
    /// "permutation"`) -- dispatched on this session's own [`SessionKind`],
    /// not on `xs`'s shape. All-or-nothing either way: on any error
    /// (dimension mismatch / an invalid tour, a non-finite coordinate, or
    /// budget overrun) nothing is counted. The Float path is BYTE-IDENTICAL
    /// to before Task 7 (same extraction, same error mapping); see
    /// [`PermSession::evaluate`] for the permutation path's own validation.
    fn evaluate(&mut self, xs: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let session = self.inner.as_mut().ok_or_else(session_finished_err)?;
        match session {
            SessionKind::Float(s) => {
                let rows: Vec<Vec<f64>> = xs.extract()?;
                s.evaluate(&rows).map_err(|e| PyValueError::new_err(e.to_string()))
            }
            SessionKind::Perm(s) => {
                let rows: Vec<Vec<i64>> = xs.extract()?;
                s.evaluate(&rows)
            }
        }
    }

    fn evals_used(&self) -> PyResult<u64> {
        Ok(match self.inner.as_ref().ok_or_else(session_finished_err)? {
            SessionKind::Float(s) => s.evals_used(),
            SessionKind::Perm(s) => s.evals_used(),
        })
    }

    fn budget(&self) -> PyResult<u64> {
        Ok(match self.inner.as_ref().ok_or_else(session_finished_err)? {
            SessionKind::Float(s) => s.budget(),
            SessionKind::Perm(s) => s.budget(),
        })
    }

    /// `(x, f)` of the best evaluation seen so far, or `None` if nothing has
    /// been evaluated yet. `x` is a list of floats for a Float-typed
    /// session, a list of ints (a 0-based tour) for a permutation-typed one.
    fn best(&self, py: Python<'_>) -> PyResult<Option<(Py<PyList>, f64)>> {
        let session = self.inner.as_ref().ok_or_else(session_finished_err)?;
        match session {
            SessionKind::Float(s) => match s.best() {
                None => Ok(None),
                Some((x, f)) => Ok(Some((PyList::new(py, x)?.unbind(), f))),
            },
            SessionKind::Perm(s) => match s.best() {
                None => Ok(None),
                Some((x, f)) => Ok(Some((PyList::new(py, &x)?.unbind(), f))),
            },
        }
    }

    /// The problem's known optimum, or `None` if it has none (e.g. a
    /// `for_problem`-built session over a `from_callable` handle). A
    /// session built via the `EvalSession(...)` (BBOB) constructor always
    /// returns a `float`; a permutation-typed session returns `Some` for
    /// every vendored TSP instance (`tsp.rs`'s pinned published optima).
    fn f_opt(&self) -> PyResult<Option<f64>> {
        Ok(match self.inner.as_ref().ok_or_else(session_finished_err)? {
            SessionKind::Float(s) => s.f_opt(),
            SessionKind::Perm(s) => s.f_opt(),
        })
    }

    /// This session's kind: `"float"` for every continuous problem handle,
    /// `"permutation"` for `sezgi.problems.tsp(...)` (Task 7, M3-8). Drives
    /// `AlgoContext.kind`/`AlgoContext.bounds` in
    /// `py-sezgi/python/sezgi/algo.py`.
    fn kind(&self) -> PyResult<&'static str> {
        Ok(match self.inner.as_ref().ok_or_else(session_finished_err)? {
            SessionKind::Float(_) => "float",
            SessionKind::Perm(_) => "permutation",
        })
    }

    /// A uniformly random permutation of `0..n` (a 0-based tour), drawn from
    /// this PERMUTATION-typed session's own seeded [`RngStream`] -- see
    /// [`PermSession::random_permutation`]'s own doc for the shuffle
    /// algorithm and determinism contract. Does not count against the
    /// evaluation budget (drawing a candidate is not evaluating one).
    ///
    /// # Errors
    /// `ValueError` if this session's `kind()` is `"float"` -- there is no
    /// RNG stream to draw a permutation from on a continuous session.
    fn random_permutation(&mut self) -> PyResult<Vec<u32>> {
        match self.inner.as_mut().ok_or_else(session_finished_err)? {
            SessionKind::Perm(s) => Ok(s.random_permutation()),
            SessionKind::Float(_) => Err(PyValueError::new_err(
                "random_permutation() is only available for permutation-typed \
                 sessions (this session's kind is \"float\")")),
        }
    }

    /// Flushes the IOH log (if logging was enabled) and consumes the
    /// session. Any method call afterward, including a second `finish()`,
    /// raises `ValueError("session finished")`. A permutation-typed session
    /// never has a log to flush (`for_problem` rejects `log_dir` for
    /// `sezgi.problems.tsp(...)`, see its own doc), so `finish()` is a no-op
    /// for it beyond consuming the session.
    fn finish(&mut self) -> PyResult<()> {
        match self.inner.take().ok_or_else(session_finished_err)? {
            SessionKind::Float(s) => s.finish().map_err(|e| PyValueError::new_err(e.to_string())),
            SessionKind::Perm(_) => Ok(()),
        }
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
                    SUITE_BBOB, p.fid(), p.name(),
                    p.space().dim());
                let obs = lg.start_run_with(p.instance, master_seed, p.f_opt(), spec.termination.budget);
                // Logger observer is Rust-native (no GIL needed); GIL is released for the run.
                let r = run_with_bridge(py, || run(p, Some(Box::new(obs))))?;
                let fin = lg.finish().map_err(|e| PyValueError::new_err(e.to_string()))?;
                (r, Some(fin.skipped_empty_runs as u64))
            } else { (run_with_bridge(py, || run(p, None))?, None) }
        }
        Inner::Callable { f, space, vectorized } => {
            if log_dir.is_some() {
                return Err(PyValueError::new_err(
                    "log_dir is only supported for builtin (bbob) problems"));
            }
            let cp = CallableProblem { f, space, vectorized: *vectorized };
            // GIL is released for the run; the callback reacquires it each
            // batch via Python::with_gil (standard PyO3 pattern).
            (run_with_bridge(py, || run(&cp, None))?, None)
        }
        // sezgi decision (M3-5 scope ruling 2, fix round 1): CEC 2022 is
        // solve()-eligible (Inner::Cec2022) additively -- same shape as
        // Inner::Bbob, IOH logging included. Widened alongside
        // `EvalSession::for_problem`'s own CEC 2022 log_dir widening: the
        // two entry points must agree on one handle's logging behavior (an
        // asymmetry here would re-create the exact asymmetry M3-4's final
        // review fixed, just in the opposite direction). Suite/fid/name/
        // instance mirror `for_problem`'s Inner::Cec2022 SessionMeta
        // exactly ("sezgi-cec2022", p.fid(), "cec2022-f{fid}", instance 1),
        // so a run logged via solve() and one logged via for_problem
        // reconstruct with identical identity keys. TSP still has no
        // fid/instance/name scenario metadata to log against, so it keeps
        // rejecting log_dir the same way Inner::Callable does.
        Inner::Cec2022(p) => {
            if let Some(dir) = log_dir {
                let name = algo_name.unwrap_or(&spec.name).to_string();
                let scenario_name = format!("cec2022-f{}", p.fid());
                let mut lg = IohLogger::new(std::path::Path::new(dir), &name,
                    "sezgi-cec2022", p.fid(), &scenario_name,
                    p.space().dim());
                let obs = lg.start_run_with(1, master_seed, p.f_star(), spec.termination.budget);
                let r = run_with_bridge(py, || run(p, Some(Box::new(obs))))?;
                let fin = lg.finish().map_err(|e| PyValueError::new_err(e.to_string()))?;
                (r, Some(fin.skipped_empty_runs as u64))
            } else { (run_with_bridge(py, || run(p, None))?, None) }
        }
        // M3-6 Task 9: CEC 2014/CEC 2017 are solve()-eligible additively,
        // same shape as Inner::Cec2022 (IOH logging included) -- widened
        // alongside for_problem's own CEC 2014/CEC 2017 log_dir widening
        // above, keeping the two entry points symmetric (M3-5 T2 asymmetry
        // ruling: both or neither). Suite/fid/name/instance mirror
        // for_problem's own SessionMeta exactly ("sezgi-cec2014"/
        // "sezgi-cec2017", p.fid(), "cec2014-f{fid}"/"cec2017-f{fid}",
        // instance 1).
        Inner::Cec2014(p) => {
            if let Some(dir) = log_dir {
                let name = algo_name.unwrap_or(&spec.name).to_string();
                let scenario_name = format!("cec2014-f{}", p.fid());
                let mut lg = IohLogger::new(std::path::Path::new(dir), &name,
                    "sezgi-cec2014", p.fid(), &scenario_name,
                    p.space().dim());
                let obs = lg.start_run_with(1, master_seed, p.f_star(), spec.termination.budget);
                let r = run_with_bridge(py, || run(p, Some(Box::new(obs))))?;
                let fin = lg.finish().map_err(|e| PyValueError::new_err(e.to_string()))?;
                (r, Some(fin.skipped_empty_runs as u64))
            } else { (run_with_bridge(py, || run(p, None))?, None) }
        }
        Inner::Cec2017(p) => {
            if let Some(dir) = log_dir {
                let name = algo_name.unwrap_or(&spec.name).to_string();
                let scenario_name = format!("cec2017-f{}", p.fid());
                let mut lg = IohLogger::new(std::path::Path::new(dir), &name,
                    "sezgi-cec2017", p.fid(), &scenario_name,
                    p.space().dim());
                let obs = lg.start_run_with(1, master_seed, p.f_star(), spec.termination.budget);
                let r = run_with_bridge(py, || run(p, Some(Box::new(obs))))?;
                let fin = lg.finish().map_err(|e| PyValueError::new_err(e.to_string()))?;
                (r, Some(fin.skipped_empty_runs as u64))
            } else { (run_with_bridge(py, || run(p, None))?, None) }
        }
        Inner::Tsp(p) => {
            if log_dir.is_some() {
                return Err(PyValueError::new_err(
                    "log_dir is only supported for builtin (bbob) problems"));
            }
            (run_with_bridge(py, || run(p, None))?, None)
        }
        // M3-8 Task 9: onemax/int_quadratic/cat_match/mixed_diagnostic are
        // solve()-eligible additively, same shape as Inner::Tsp -- no IOH
        // identity (fid/instance/f_opt-for-logging) exists for any of them,
        // so log_dir is rejected the same way.
        Inner::OneMax(p) => {
            if log_dir.is_some() {
                return Err(PyValueError::new_err(
                    "log_dir is only supported for builtin (bbob) problems"));
            }
            (run_with_bridge(py, || run(p, None))?, None)
        }
        Inner::IntQuadratic(p) => {
            if log_dir.is_some() {
                return Err(PyValueError::new_err(
                    "log_dir is only supported for builtin (bbob) problems"));
            }
            (run_with_bridge(py, || run(p, None))?, None)
        }
        Inner::CatMatch(p) => {
            if log_dir.is_some() {
                return Err(PyValueError::new_err(
                    "log_dir is only supported for builtin (bbob) problems"));
            }
            (run_with_bridge(py, || run(p, None))?, None)
        }
        Inner::Mixed(p) => {
            if log_dir.is_some() {
                return Err(PyValueError::new_err(
                    "log_dir is only supported for builtin (bbob) problems"));
            }
            (run_with_bridge(py, || run(p, None))?, None)
        }
        Inner::F0 { dim, seed, .. } => {
            if log_dir.is_some() {
                return Err(PyValueError::new_err(
                    "log_dir is only supported for builtin (bbob) problems"));
            }
            let p = F0Random::new(*dim, *seed);
            (run_with_bridge(py, || run(&p, None))?, None)
        }
    };

    let d = PyDict::new(py);
    d.set_item("best_f", result.best_f)?;
    // best_x's block shape follows the problem's own space -- M3-8 Task 9
    // "typed result genotype" decision, documented here and in the task's
    // report:
    //
    // - A SINGLE-block genotype (every problem before this task, plus
    //   onemax/int_quadratic/cat_match, each of which is single-block) is
    //   surfaced as a FLAT list, in each block kind's own natural Python
    //   type: Float -> list[float] (BYTE-IDENTICAL to every solve() result
    //   before this task -- no behavior change for existing callers), Perm
    //   -> list[int] (unchanged from Task 7/8), Int -> list[int], Cat ->
    //   list[int] (category INDICES 0..k, not labels -- CatMatch/gen/ga-cat
    //   have no notion of a label), Bin -> list[bool] (Python's native
    //   boolean, matching Rust's own `Vec<bool>` 1:1 -- chosen over a 0/1-int
    //   encoding as the more natural per-bit Python type; UNLIKE
    //   `sezgi.mo.nsga2`'s OWN Binary encoding, which flattens bits to
    //   0.0/1.0 floats so `individuals` stays uniformly float-typed across
    //   every MO problem family -- see `genotype_to_flat_vec`'s doc -- there
    //   is no such cross-family uniformity constraint on solve()'s
    //   single-problem `best_x`).
    // - A MULTI-block genotype (reachable only via
    //   `problems.mixed_diagnostic(...)`, Task 9's `gen/compound` TOML
    //   reachability scaffold) is surfaced as a list of per-block lists, one
    //   sub-list per block in `SearchSpace::blocks()` order, each typed as
    //   above -- the natural generalization that preserves block structure
    //   rather than collapsing every block into one ambiguously-typed flat
    //   list.
    fn block_values_to_py(py: Python<'_>, bv: &BlockValues) -> PyResult<Py<PyAny>> {
        Ok(match bv {
            BlockValues::Float(xs) => PyList::new(py, xs)?.into_any().unbind(),
            BlockValues::Perm(xs) => PyList::new(py, xs)?.into_any().unbind(),
            BlockValues::Int(xs) => PyList::new(py, xs)?.into_any().unbind(),
            BlockValues::Cat(xs) => PyList::new(py, xs)?.into_any().unbind(),
            BlockValues::Bin(xs) => PyList::new(py, xs)?.into_any().unbind(),
        })
    }
    let best_x_py: Py<PyAny> = if result.best_x.blocks.len() == 1 {
        block_values_to_py(py, &result.best_x.blocks[0])?
    } else {
        let blocks = PyList::empty(py);
        for b in &result.best_x.blocks {
            blocks.append(block_values_to_py(py, b)?)?;
        }
        blocks.into_any().unbind()
    };
    d.set_item("best_x", best_x_py)?;
    d.set_item("evals_used", result.evals_used)?;
    d.set_item("iterations", result.iterations)?;
    if let Some(skipped) = skipped_empty_runs_opt {
        d.set_item("skipped_empty_runs", skipped)?;
    }
    Ok(d.into())
}

/// Builds a record dict from a [`RunRecord`], with the SAME shape
/// `run_experiment` returns (keys: `algo, fid, dim, instance, seed, budget,
/// suite, best_f, f_opt, gap, evals_used, wall_secs`). Shared by
/// `run_experiment` and `read_ioh_records` so a disk-reconstructed record
/// and a freshly-run one are interchangeable to any downstream consumer
/// (e.g. `per_budget_packages`).
///
/// `"suite"` (M3-5 Task 1): always emitted -- BBOB runs get [`SUITE_BBOB`]
/// (`"sezgi-bbob"`), same as every record built before this key existed.
/// See [`records_from_pylist`] for the read side.
fn record_to_dict<'py>(py: Python<'py>, r: &RunRecord) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("algo", &r.key.algo)?;
    d.set_item("fid", r.key.fid)?;
    d.set_item("dim", r.key.dim)?;
    d.set_item("instance", r.key.instance)?;
    d.set_item("seed", r.key.seed)?;
    d.set_item("budget", r.key.budget)?;
    d.set_item("suite", &r.key.suite)?;
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
/// Each `curve_dict` is `{"evals": [...], "proportion": [...]}`. Grouping
/// (in both modes) is by algo only, not `(algo, suite)` -- see
/// [`sezgi_bench::ecdf_per_algo`]'s doc.
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
/// BBOB-only: raises `ValueError` (naming the offending suite) if the tree
/// holds any non-BBOB scenario — see `coco_export`'s doc.
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
///
/// `"suite"` (M3-5 Task 1): read if present, defaulted to [`SUITE_BBOB`]
/// otherwise — an old-shape dict from before this key existed (or any
/// hand-built dict that omits it) is a BBOB record, matching every
/// pre-Task-1 record. See [`record_to_dict`] for the write side.
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
        let suite = match d.get_item("suite")? {
            Some(v) => v.extract::<String>()?,
            None => SUITE_BBOB.to_string(),
        };
        let key = RunKey {
            algo: get("algo")?.extract::<String>()?,
            fid: get("fid")?.extract::<u32>()?,
            dim: get("dim")?.extract::<usize>()?,
            instance: get("instance")?.extract::<u32>()?,
            seed: get("seed")?.extract::<u64>()?,
            budget: get("budget")?.extract::<u64>()?,
            suite,
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

/// `sezgi.bias.structural_positions(final_positions)` — the bias bridge for
/// externally-authored algorithms (M3-4 Task 4): runs the SAME KS/AD/Holm
/// battery as `sezgi.bias.structural`, but over caller-supplied
/// `final_positions` (each row one run's final best `x`) instead of driving
/// an `AlgorithmSpec` through the engine itself — see
/// [`sezgi_bias::structural::scan_from_positions`]. `dim` is inferred from
/// row length; every row must have the SAME length. Returns a dict with the
/// IDENTICAL keys `sezgi.bias.structural` returns (`per_dim_ks`,
/// `per_dim_ad`, `holm_rejections_ks`, `holm_rejections_ad`, `verdict`,
/// `detail`, `final_positions`) — built via the same
/// [`structural_result_to_dict`] helper, so the two are interchangeable to
/// any downstream consumer.
///
/// # Errors
/// `ValueError` if `final_positions` has fewer than 5 rows, is ragged (rows
/// of differing length), or any row is empty (`dim == 0`).
#[pyfunction]
fn bias_structural_positions(py: Python<'_>, final_positions: Vec<Vec<f64>>) -> PyResult<Py<PyDict>> {
    let dim = final_positions.first().map_or(0, |row| row.len());
    let r = scan_from_positions(final_positions, dim)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
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
// Multi-objective bindings (sezgi.mo) -- M3-2 Task 9, extended in M3-7.
//
// Binds the NSGA-II runner (`sezgi_components::nsga2::nsga2_run`, incl. the
// M3-7 constrained and binary paths), the ZDT/DTLZ/WFG benchmark suites
// (`sezgi_problems::{Zdt, Zdt5, Dtlz, Wfg}`), the exact hypervolume / IGD
// indicators (`sezgi_stats::{hypervolume_2d, hypervolume, igd}`), and the
// sezgi-moa v1 archive logging (`sezgi_bench::mo_archive`). Every
// scalar/vector f64 is passed through EXACTLY as the Rust core computed it
// -- no rounding/formatting anywhere in this section (the R bindings assert
// bit-equality against these same values).
//
// Problem-string mapping (shared by `mo_nsga2` and `mo_pareto_front`, via
// `mo_problem_from_str`): `"zdt1"`..`"zdt4"`, `"zdt6"` (real-coded);
// `"zdt5"` (the binary-coded T5, its own `Zdt5` type -- fixed 80-bit
// layout, so `dim` is rejected for it); `"dtlz1"`..`"dtlz9"` (`m` REQUIRED;
// 8/9 are the constrained pair and surface `violations`); `"wfg1"`..
// `"wfg9"` (`m` required, optional `k`/`l` with the toolkit-recommended
// defaults). **Passing `m` for a `zdt*` problem is a `ValueError`** (zdt
// problems are always 2-objective by construction, so a caller-supplied
// `m` could never be honored silently -- rejecting it outright surfaces
// the mistake instead of quietly ignoring the argument); `k`/`l` are
// likewise WFG-only.
// ---------------------------------------------------------------------

/// WFG's own recommended `k` default (module doc's "Recommended `k`/`l`
/// defaults" section, `sezgi_problems::wfg`'s own `// sezgi decision:` on
/// the toolkit README's typo -- the widely-used literature reading: `k=4`
/// for `m=2`, `k=2*(m-1)` for `m>=3`). Guarded at `m<=2` (not `m==2`) purely
/// to avoid a `usize` underflow computing a THROWAWAY value for `m<2`: an
/// `m<2` call always fails [`Wfg::new`]'s own `BadM` check first (checked
/// before the `k%(m-1)==0` line that would otherwise divide by zero), so
/// this default is never actually used in that case.
fn wfg_default_k(m: usize) -> usize {
    if m <= 2 { 4 } else { 2 * (m - 1) }
}

/// WFG's own recommended `l` default (same module-doc section): `l=20`,
/// unconditional on `m`.
const WFG_DEFAULT_L: usize = 20;

/// Shared problem-string -> `Box<dyn MoProblem>` builder for `mo_nsga2`,
/// `mo_pareto_front`, `mo_evaluate`, and `mo_evaluate_constraints`. See this
/// section's own doc for the full mapping.
///
/// `dim`/`m`/`k`/`l` are each meaningful for only SOME problem families
/// (`dim`: zdt1-4/6, dtlz1-9; `m`: dtlz1-9, wfg1-9; `k`/`l`: wfg1-9 only) --
/// a parameter given where it does not apply, or omitted where it is
/// required, is an honest `ValueError`, mirroring the pre-existing
/// `m`-is-dtlz-only rejection style exactly (never silently ignored).
fn mo_problem_from_str(
    problem: &str,
    dim: Option<usize>,
    m: Option<usize>,
    k: Option<usize>,
    l: Option<usize>,
) -> PyResult<Box<dyn MoProblem>> {
    let unknown = || {
        PyValueError::new_err(format!(
            "unknown problem `{problem}` (expected one of zdt1, zdt2, zdt3, zdt4, zdt5, zdt6, \
             dtlz1..dtlz9, or wfg1..wfg9)"
        ))
    };

    // sezgi decision: ZDT5 (M3-7 Task 6) is checked BEFORE the generic
    // "zdt"-prefix branch below -- `Zdt::new(5, dim)` exists as a type but
    // deliberately REJECTS which=5 with its own "use Zdt5::new() instead"
    // error (zdt.rs's own doc), since ZDT5 is a separate, binary-coded type
    // with no free `dim`/real-coded space at all. Letting the generic
    // branch's `strip_prefix("zdt")` catch "zdt5" would just re-surface
    // that Rust-internal redirect error instead of actually constructing
    // it, so it is special-cased here first.
    if problem == "zdt5" {
        if m.is_some() {
            return Err(PyValueError::new_err(
                "m is DTLZ-only (number of objectives); zdt5 is always 2-objective -- \
                 omit m (or pass m=None) for zdt5",
            ));
        }
        // sezgi decision: `dim` is REJECTED for zdt5 (mirroring how `m` is
        // rejected for every zdt problem), not merely ignored -- zdt5's
        // search space is a FIXED 80-bit layout (one 30-bit block plus ten
        // 5-bit blocks, Zitzler/Deb/Thiele 2000 Definition 4; see
        // `Zdt5::new`'s own doc), not a free-dimension real-coded space, so
        // a caller-supplied `dim` can never be honored and silently
        // dropping it would hide a caller's wrong assumption.
        if dim.is_some() {
            return Err(PyValueError::new_err(
                "dim is not accepted for zdt5: its search space is a FIXED 80-bit layout \
                 (one 30-bit block plus ten 5-bit blocks) -- omit dim (or pass dim=None) for zdt5",
            ));
        }
        if k.is_some() || l.is_some() {
            return Err(PyValueError::new_err(
                "k/l are wfg-only; omit them (or pass None) for zdt5",
            ));
        }
        return Ok(Box::new(Zdt5::new()));
    }

    if let Some(rest) = problem.strip_prefix("zdt") {
        let which: u32 = rest.parse().map_err(|_| unknown())?;
        if m.is_some() {
            return Err(PyValueError::new_err(
                "m is DTLZ-only (number of objectives); zdt problems are always 2-objective -- \
                 omit m (or pass m=None) for a zdt problem",
            ));
        }
        if k.is_some() || l.is_some() {
            return Err(PyValueError::new_err(
                "k/l are wfg-only; omit them (or pass None) for a zdt problem",
            ));
        }
        let dim = dim.ok_or_else(|| {
            PyValueError::new_err("dim is required for zdt problems")
        })?;
        let p = Zdt::new(which, dim).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Box::new(p))
    } else if let Some(rest) = problem.strip_prefix("wfg") {
        let which: u32 = rest.parse().map_err(|_| unknown())?;
        // sezgi decision: `dim` is REJECTED for wfg (same reasoning as
        // zdt5 above) -- a WFG instance's dimension `n = k + l` is DERIVED
        // from `k`/`l` (`Wfg::new`'s own signature takes no `dim` at all),
        // so there is no free `dim` slot to fill; a caller must use `k`/`l`
        // instead.
        if dim.is_some() {
            return Err(PyValueError::new_err(
                "dim is not accepted for wfg problems: n = k + l is derived from k and l -- \
                 omit dim (or pass dim=None) and use k/l instead",
            ));
        }
        let m = m.ok_or_else(|| {
            PyValueError::new_err("m (number of objectives) is required for wfg problems")
        })?;
        // sezgi decision: `k`/`l` default to the toolkit's own recommended
        // values (`wfg_default_k`/`WFG_DEFAULT_L` above) when omitted,
        // mirroring `p_m`'s own `None`-resolves-to-a-formula precedent
        // rather than requiring every caller to spell them out.
        let k = k.unwrap_or_else(|| wfg_default_k(m));
        let l = l.unwrap_or(WFG_DEFAULT_L);
        let p = Wfg::new(which, m, k, l).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Box::new(p))
    } else if let Some(rest) = problem.strip_prefix("dtlz") {
        let which: u32 = rest.parse().map_err(|_| unknown())?;
        if k.is_some() || l.is_some() {
            return Err(PyValueError::new_err(
                "k/l are wfg-only; omit them (or pass None) for a dtlz problem",
            ));
        }
        let m = m.ok_or_else(|| {
            PyValueError::new_err("m (number of objectives) is required for dtlz problems")
        })?;
        let dim = dim.ok_or_else(|| {
            PyValueError::new_err("dim is required for dtlz problems")
        })?;
        // dtlz8/dtlz9 (M3-7 Task 2) reuse `Dtlz::new` unchanged -- its own
        // `dim > m` constraint-surface check (`DtlzError::BadDimConstraintSurface`)
        // surfaces via the SAME `map_err` path as every other dtlz error.
        let p = Dtlz::new(which, m, dim).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Box::new(p))
    } else {
        Err(unknown())
    }
}

/// Flattens a [`Genotype`] into a single `Vec<f64>` (concatenating every
/// block in order) -- always either an all-`Block::Float` genotype (zdt1-4/
/// 6, dtlz1-9, wfg1-9) or an all-`Block::Binary` one (zdt5 only), since
/// [`mo_problem_from_str`]'s own problem catalog never builds a problem
/// whose space mixes block kinds (M3-8 Task 6 makes `nsga2_run` itself
/// ACCEPT mixed spaces too, but no problem registered here ever constructs
/// one, so this function's own all-Float/all-Binary assumption still holds
/// for every input it actually sees). Mirrors `solve`'s own `best_x` conversion above (`BlockValues::Float(xs)
/// => ...`), generalized to however many Float blocks the space has (ZDT4
/// has two: `x1` and the rest).
///
/// `# sezgi decision:` (M3-7 Task 10) a `Block::Binary` block's bits are
/// flattened to `0.0`/`1.0` floats, the SAME 0/1 convention
/// `genotype_from_flat` (below) reads back -- keeping `individuals`
/// uniformly typed as "list of float-lists" across every problem family
/// (rather than introducing a differently-typed field only for zdt5) is
/// simpler for every existing consumer of this dict shape (Python callers,
/// R bindings via M3-7 Task 11) that already expects floats.
fn genotype_to_flat_vec(g: &Genotype) -> Vec<f64> {
    let mut out = Vec::new();
    for b in &g.blocks {
        match b {
            BlockValues::Float(xs) => out.extend_from_slice(xs),
            BlockValues::Bin(bits) => out.extend(bits.iter().map(|&b| if b { 1.0 } else { 0.0 })),
            _ => {}
        }
    }
    out
}

/// Builds a [`Genotype`] matching `space`'s own block layout from a flat
/// `x` (the inverse of [`genotype_to_flat_vec`]'s flattening, used by
/// `mo_evaluate`/`mo_evaluate_constraints` to construct a one-off individual
/// from a caller-supplied decision vector). `Block::Float` blocks take their
/// slice of `x` verbatim; `Block::Binary` blocks read each value as a bit
/// via `!= 0.0` (matching `genotype_to_flat_vec`'s own `0.0`/`1.0` encoding
/// on the way back out).
///
/// # Errors
/// `ValueError` if `x.len() != space.dim()`, or if `space` has a block that
/// is neither `Block::Float` nor `Block::Binary` (unreachable through
/// `mo_problem_from_str`'s own zdt/dtlz/wfg constructors today, but checked
/// honestly rather than silently skipped).
fn genotype_from_flat(space: &SearchSpace, x: &[f64]) -> PyResult<Genotype> {
    if x.len() != space.dim() {
        return Err(PyValueError::new_err(format!(
            "x must have exactly {} coordinates (dim={}), got {}",
            space.dim(), space.dim(), x.len()
        )));
    }
    let mut blocks = Vec::with_capacity(space.blocks().len());
    let mut i = 0usize;
    for b in space.blocks() {
        match *b {
            Block::Float { n, .. } => {
                blocks.push(BlockValues::Float(x[i..i + n].to_vec()));
                i += n;
            }
            Block::Binary { n } => {
                blocks.push(BlockValues::Bin(x[i..i + n].iter().map(|&v| v != 0.0).collect()));
                i += n;
            }
            _ => return Err(PyValueError::new_err(
                "mo.evaluate only supports all-Float or all-Binary spaces")),
        }
    }
    Ok(Genotype { blocks })
}

/// `sezgi.mo.evaluate(problem, x, dim=None, m=None, k=None, l=None) ->
/// list[float]` -- direct, one-shot objective evaluation of a decision
/// vector `x` against any `sezgi.mo` problem string, bypassing `nsga2`'s
/// population/budget machinery entirely (`problem`/`dim`/`m`/`k`/`l` share
/// `mo_nsga2`'s own `mo_problem_from_str` mapping).
///
/// `# sezgi decision:` (M3-7 Task 10) added purely so this binding's OWN
/// test suite can pin exact fixture values (zdt5's all-ones/all-zeros hand
/// fixtures, dtlz8/9's hand fixtures, the committed
/// `wfg_reference_values.json` points) directly from Python -- mirroring
/// the existing `cec2022_evaluate`/`cec2014_evaluate`/`cec2017_evaluate`
/// one-shot-evaluation convention already established in this file. Not
/// literally named in this task's brief (which only names
/// `nsga2`/`pareto_front` for the new problem families), but required by
/// the brief's own Test section, since `nsga2`'s randomly-initialized
/// population cannot pin a fixture at a chosen `x`.
///
/// `x`: for an all-Float problem, its raw decision values; for zdt5 (the
/// only all-Binary problem reachable here), each entry is read as a bit
/// (`!= 0.0` -> `true`) -- see [`genotype_from_flat`]'s own doc.
///
/// # Errors
/// Same problem-construction errors as `mo.nsga2`/`mo.pareto_front`, plus a
/// `ValueError` if `len(x)` does not match the problem's own dimension.
#[pyfunction]
#[pyo3(signature = (problem, x, dim=None, m=None, k=None, l=None))]
fn mo_evaluate(
    problem: &str,
    x: Vec<f64>,
    dim: Option<usize>,
    m: Option<usize>,
    k: Option<usize>,
    l: Option<usize>,
) -> PyResult<Vec<f64>> {
    let prob = mo_problem_from_str(problem, dim, m, k, l)?;
    let g = genotype_from_flat(prob.space(), &x)?;
    Ok(prob
        .evaluate_batch(std::slice::from_ref(&g))
        .into_iter()
        .next()
        .expect("evaluate_batch returns one row per input genotype"))
}

/// `sezgi.mo.evaluate_constraints(problem, x, dim=None, m=None, k=None,
/// l=None) -> Optional[list[float]]` -- direct, one-shot constraint-row
/// evaluation, mirroring `mo.evaluate`'s calling convention exactly (same
/// [`mo_problem_from_str`] mapping, same [`genotype_from_flat`] decoding).
/// Returns `None` for an unconstrained problem (every zdt/wfg problem, and
/// dtlz1-7), or the constraint row (`g_1..g_ncon`, `g_j >= 0` meaning
/// SATISFIED -- see [`sezgi_core::mo::MoProblem::evaluate_constraints_batch`]'s
/// own doc for the pinned sign convention) for a constrained one (dtlz8/
/// dtlz9).
///
/// # Errors
/// Same as `mo.evaluate`.
#[pyfunction]
#[pyo3(signature = (problem, x, dim=None, m=None, k=None, l=None))]
fn mo_evaluate_constraints(
    problem: &str,
    x: Vec<f64>,
    dim: Option<usize>,
    m: Option<usize>,
    k: Option<usize>,
    l: Option<usize>,
) -> PyResult<Option<Vec<f64>>> {
    let prob = mo_problem_from_str(problem, dim, m, k, l)?;
    let g = genotype_from_flat(prob.space(), &x)?;
    Ok(prob
        .evaluate_constraints_batch(std::slice::from_ref(&g))
        .map(|rows| rows.into_iter().next().expect("one row per input genotype")))
}

/// `sezgi.mo.nsga2(problem, dim, pop_size, budget, m=None, seed=0,
/// eta_c=20.0, eta_m=20.0, p_c=0.9, p_m=None, p_c_bin=0.9, p_m_bin=None,
/// k=None, l=None, log_dir=None, label=None)` -- binds
/// [`sezgi_components::nsga2::nsga2_run`] (or, when `log_dir` is given,
/// [`nsga2_run_logged`] -- see below). `eta_c`/`eta_m`/`p_c` default to the
/// paper's own pinned experimental settings (Deb et al. 2002, Sec. IV.A:
/// eta_c=20, eta_m=20, p_c=0.9 -- see that module's "Defaults" doc
/// section); `p_m=None` resolves on the Rust side to `1 / n_variables` (the
/// paper's own default), never re-derived here.
///
/// `dim`: required (may be `None`, but the argument itself must be
/// supplied) for zdt1-4/6 and dtlz1-9; REJECTED (must be `None`) for zdt5
/// and wfg1-9, whose dimension is fixed (zdt5) or derived from `k`/`l`
/// (wfg) -- see [`mo_problem_from_str`]'s own doc.
///
/// `p_c_bin`/`p_m_bin` (M3-7 Task 3/10): the binary-genotype counterparts
/// of `p_c`/`p_m`, consulted ONLY when `problem` builds an all-Binary space
/// (zdt5 today) -- `Nsga2Config`'s own doc: "Only consulted when
/// `nsga2_run`'s search space is all-Block::Binary -- never read on the
/// all-Float path". `# sezgi decision:` `p_c_bin` defaults to `0.9`,
/// mirroring `p_c`'s own paper-pinned default -- the paper itself gives NO
/// verified formula for a binary-specific crossover probability (KanGAL's
/// C reference requires it as a REQUIRED CLI input with no built-in
/// default either, `Nsga2Config::p_c_bin`'s own doc), so `p_c`'s value is
/// the closest defensible choice, not a verified paper quote. `p_m_bin`
/// defaults to `None`, which the Rust side resolves to `1 / l` (`l` = the
/// space's total bit count) -- the paper's OWN stated binary-coded default,
/// mirroring `p_m`'s identical `None`-resolves-to-a-formula design exactly.
/// Both are passed through UNCONDITIONALLY (validated by
/// [`sezgi_components::nsga2::Nsga2Config`]'s own range check even for an
/// all-Float problem, where they are simply unused) -- no problem-family
/// gating is added on the Python side, matching the Rust config's own
/// unconditional-validation design.
///
/// `k`/`l` (M3-7 Task 10): wfg-only parameters, forwarded to
/// [`mo_problem_from_str`] -- see that function's own doc for their
/// defaults and the `dim`-rejection this implies for wfg.
///
/// `log_dir`/`label` (M3-7 Task 9/10): when `log_dir` is given, the run is
/// executed via [`nsga2_run_logged`] instead of the plain `nsga2_run`,
/// streaming every feasible archive insertion to
/// `<log_dir>/<label>-s<seed>.moa` (sezgi-moa v1 format; see
/// `crates/bench/src/mo_archive.rs`'s own module doc for the exact format
/// grammar) -- `label` is REQUIRED whenever `log_dir` is given (an honest
/// `ValueError` otherwise, mirroring how `EvalSession`'s own `for_problem`
/// constructor requires enough identity to build a meaningful on-disk key
/// before it will accept a `log_dir`); omitting `log_dir` runs exactly as
/// before (byte-identical `nsga2_run` path), and `label` is simply ignored
/// if given without `log_dir`.
///
/// Returns a dict mirroring `MoRunResult` 1:1: `individuals` (list of
/// float-lists, one per final-population member, flattened across every
/// block -- see [`genotype_to_flat_vec`]'s own doc for the Binary-block
/// encoding), `objectives` (list of float-lists, parallel to
/// `individuals`), `front0` (list of ints: indices of the final
/// population's non-dominated set), `evals_used` (int), and `violations`
/// (M3-7 Task 1/10) -- **present ONLY when the problem is constrained**
/// (dtlz8/dtlz9 today; `MoRunResult::violations`'s own `Some` iff
/// `evaluate_constraints_batch` returned `Some` rule), a list of floats
/// (`<= 0.0`, `0.0` = fully feasible) parallel to `individuals`/
/// `objectives`. `# sezgi decision:` the key is ABSENT (not present with a
/// `None` value) for an unconstrained problem -- the SAME
/// present-only-when-meaningful convention `solve()`'s own
/// `skipped_empty_runs` key already uses in this file, rather than a
/// dict key every caller must always check for `None`.
///
/// # Errors
/// `ValueError` for an unrecognized `problem` string, a `dim`/`m`/`k`/`l`
/// given where the problem does not accept it (or missing where required --
/// see [`mo_problem_from_str`]'s own doc), a `log_dir` given without
/// `label`, or any [`sezgi_components::nsga2::Nsga2Error`] (including
/// `pop_size` failing the `>= 4 && pop_size % 4 == 0` check -- NOT merely
/// "even, >= 4") / [`sezgi_bench::MoRunLoggedError`] (when `log_dir` is
/// given).
#[pyfunction]
#[pyo3(signature = (problem, dim, pop_size, budget, m=None, seed=0, eta_c=20.0, eta_m=20.0,
                     p_c=0.9, p_m=None, p_c_bin=0.9, p_m_bin=None, k=None, l=None,
                     log_dir=None, label=None))]
#[allow(clippy::too_many_arguments)]
fn mo_nsga2(
    py: Python<'_>,
    problem: &str,
    dim: Option<usize>,
    pop_size: usize,
    budget: u64,
    m: Option<usize>,
    seed: u64,
    eta_c: f64,
    eta_m: f64,
    p_c: f64,
    p_m: Option<f64>,
    p_c_bin: f64,
    p_m_bin: Option<f64>,
    k: Option<usize>,
    l: Option<usize>,
    log_dir: Option<&str>,
    label: Option<&str>,
) -> PyResult<Py<PyDict>> {
    let prob = mo_problem_from_str(problem, dim, m, k, l)?;
    // p_c_cat/p_m_cat (M3-8 Task 6): mechanical Nsga2Config spillover, kept
    // INERT this task (binding surface untouched -- see
    // sezgi_components::nsga2's own module doc, "M3-8 Task 6" section,
    // "Nsga2Config field spillover"). Fixed defaults (0.9/None), matching
    // the Rust API's own defaults; these remain internal-only knobs --
    // exposing them as real, user-tunable `mo_nsga2` parameters is deferred
    // (no milestone currently owns this work).
    let cfg = Nsga2Config {
        pop_size, budget, seed, eta_c, eta_m, p_c, p_m, p_c_bin, p_m_bin,
        p_c_cat: 0.9, p_m_cat: None,
    };

    let result = if let Some(dir) = log_dir {
        let label = label.ok_or_else(|| PyValueError::new_err(
            "label is required when log_dir is given (sezgi-moa file naming: \
             <log_dir>/<label>-s<seed>.moa)"))?;
        run_with_bridge(py, || {
            nsga2_run_logged(prob.as_ref(), &cfg, Path::new(dir), label)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?
    } else {
        run_with_bridge(py, || {
            nsga2_run(prob.as_ref(), &cfg).map_err(|e| PyValueError::new_err(e.to_string()))
        })?
    };

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
    if let Some(v) = &result.violations {
        d.set_item("violations", PyList::new(py, v)?)?;
    }

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

/// `sezgi.mo.hypervolume(front, ref_point)` -- binds [`sezgi_stats::hypervolume`]
/// (M3-7 Task 8/10), the exact general-M hypervolume via the WFG algorithm
/// (While, Bradstreet & Barone, "A Fast Way of Calculating Exact
/// Hypervolumes", IEEE TEC 2012). Unlike `hypervolume_2d`, `front`/
/// `ref_point` may have any number `M >= 1` of objectives (`M == 2`
/// delegates internally to the SAME `hypervolume_2d`, that module's own "2D
/// shortcut" doc section).
///
/// `ref_point` is **REQUIRED, with no default** (a plain positional/keyword
/// argument -- calling this with only `front` raises Python's own
/// `TypeError` for a missing argument, by signature, not a `ValueError`
/// this binding raises itself): `sezgi_stats::moo_indicators`'s own module
/// doc, "Choosing a reference point: explicit-always, contested in the
/// literature" section, deliberately never picks one FOR the caller. One
/// common convention from that literature (also the module's own examples/
/// tests' choice, and the specific one critiqued by Ishibuchi, Imada,
/// Setoguchi & Nojima 2018, "How to Specify a Reference Point in
/// Hypervolume Calculation for Fair Performance Comparison", GECCO
/// Companion) is the analytic front's nadir point (the componentwise worst
/// value across the front) scaled by `1.1` -- a caller-supplied choice,
/// never defaulted here.
///
/// # Errors
/// `ValueError` for any [`sezgi_stats::StatsError`] (`ref_point` empty, a
/// `front` row with a different number of objectives than `ref_point`, or a
/// non-finite value). An EMPTY `front` is NOT an error -- it returns `0.0`
/// (the algorithm's own base case, `hypervolume`'s own "empty front"
/// doc section).
#[pyfunction]
fn mo_hypervolume(front: Vec<Vec<f64>>, ref_point: Vec<f64>) -> PyResult<f64> {
    stats_hypervolume(&front, &ref_point).map_err(|e| PyValueError::new_err(e.to_string()))
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

/// `sezgi.mo.pareto_front(problem, dim, n, m=None, k=None, l=None)` -- a
/// deterministic `n`-point sample of the analytic Pareto front in OBJECTIVE
/// space, via [`sezgi_core::mo::MoProblem::pareto_front`]. Same
/// `problem`/`dim`/`m`/`k`/`l` mapping as `mo_nsga2` (see this section's own
/// doc). Returns `None` when the problem has no known analytic front sample
/// at this `m` (e.g. DTLZ5/DTLZ6 with `m > 3` -- verified only for `m <= 3`,
/// see `sezgi_problems::dtlz`'s module doc; or WFG1/WFG2 unconditionally --
/// see `sezgi_problems::wfg`'s own `pareto_front` decisions section), a
/// list of `n` float-lists otherwise.
///
/// # Errors
/// Same as `mo_nsga2`'s problem-construction errors (unrecognized `problem`,
/// a `dim`/`m`/`k`/`l` given where the problem does not accept it or
/// missing where required, or any `ZdtError`/`DtlzError`/`WfgError`).
#[pyfunction]
#[pyo3(signature = (problem, dim, n, m=None, k=None, l=None))]
fn mo_pareto_front(
    problem: &str,
    dim: Option<usize>,
    n: usize,
    m: Option<usize>,
    k: Option<usize>,
    l: Option<usize>,
) -> PyResult<Option<Vec<Vec<f64>>>> {
    let prob = mo_problem_from_str(problem, dim, m, k, l)?;
    Ok(prob.pareto_front(n))
}

/// `sezgi.mo.read_moa(path, at=None)` -- binds [`sezgi_bench::read_moa`] (a
/// "sezgi-moa v1" archive file written by `mo.nsga2(..., log_dir=...,
/// label=...)`, M3-7 Task 9/10). Returns a dict:
/// - `algo` (str): the logging algorithm name -- always `"nsga2"` today
///   (`nsga2_run_logged`'s own fixed `NSGA2_ALGO_NAME`).
/// - `problem` (str): the `label` `mo.nsga2` was called with. **Kept as the
///   literal on-disk header key name** (`crates/bench/src/mo_archive.rs`'s
///   own format grammar: the header line is `problem <label>`, not
///   `label <label>`) rather than renamed here to `"label"` -- this
///   binding stays a thin, direct mirror of [`MoArchiveRun`]'s own field
///   names, so a reader cross-checking against the Rust struct (or the R
///   binding, M3-7 Task 11) sees the SAME key everywhere.
/// - `m` (int), `seed` (int), `budget` (int).
/// - `kind` (str): `"float"` or `"binary"`.
/// - `records` (list of dicts, in file/eval order): `eval_index` (int),
///   `objectives` (list of float), `genotype` (list of float for
///   `kind="float"`, list of bool for `kind="binary"` -- [`MoArchiveGenotype`]'s
///   own two variants).
/// - `archive` (list of float-lists): the reconstructed nondominated
///   archive at evaluation budget `at`, via [`MoArchiveRun::archive_at`].
///
/// `# sezgi decision:` `at=None` resolves to the file's own logged `budget`
/// header field (the full run's final archive) -- `archive_at` itself takes
/// a REQUIRED `evals: u64` with no Rust-side default, and the file's own
/// `budget` is the one value guaranteed to reconstruct the run's COMPLETE
/// trajectory (`nsga2_run_logged`'s own `MoEvaluator` budget enforcement
/// means no `eval_index` beyond it was ever charged) -- a natural,
/// self-contained default rather than requiring every caller to pass the
/// budget back in by hand after already reading it out of the same dict.
///
/// # Errors
/// `ValueError` for any [`sezgi_bench::MoArchiveError`] (missing file, a
/// malformed header, or a malformed record line).
#[pyfunction]
#[pyo3(signature = (path, at=None))]
fn mo_read_moa(py: Python<'_>, path: &str, at: Option<u64>) -> PyResult<Py<PyDict>> {
    let run = bench_read_moa(path).map_err(|e| PyValueError::new_err(e.to_string()))?;

    let d = PyDict::new(py);
    d.set_item("algo", &run.algo)?;
    d.set_item("problem", &run.problem)?;
    d.set_item("m", run.m)?;
    d.set_item("seed", run.seed)?;
    d.set_item("budget", run.budget)?;
    d.set_item("kind", match run.kind {
        GenoKind::Float => "float",
        GenoKind::Binary => "binary",
    })?;

    let records = PyList::empty(py);
    for r in &run.records {
        let rd = PyDict::new(py);
        rd.set_item("eval_index", r.eval_index)?;
        rd.set_item("objectives", PyList::new(py, &r.objectives)?)?;
        match &r.genotype {
            MoArchiveGenotype::Float(xs) => rd.set_item("genotype", PyList::new(py, xs)?)?,
            MoArchiveGenotype::Binary(bits) => rd.set_item("genotype", PyList::new(py, bits)?)?,
        }
        records.append(rd)?;
    }
    d.set_item("records", records)?;

    let evals = at.unwrap_or(run.budget);
    let archive = run.archive_at(evals);
    let archive_list = PyList::empty(py);
    for row in &archive {
        archive_list.append(PyList::new(py, row)?)?;
    }
    d.set_item("archive", archive_list)?;

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
/// `sezgi.presets.ga_bin(pop_size, budget)` (M3-8 Task 9) -- mirrors
/// `preset_ga_perm`'s own shape; pairs with `sezgi.problems.onemax(...)`.
#[pyfunction] fn preset_ga_bin(pop_size: usize, budget: u64) -> String {
    presets::ga_bin(pop_size, budget).to_json()
}
/// `sezgi.presets.ga_int(pop_size, budget)` (M3-8 Task 9) -- pairs with
/// `sezgi.problems.int_quadratic(...)`.
#[pyfunction] fn preset_ga_int(pop_size: usize, budget: u64) -> String {
    presets::ga_int(pop_size, budget).to_json()
}
/// `sezgi.presets.ga_cat(pop_size, budget)` (M3-8 Task 9) -- pairs with
/// `sezgi.problems.cat_match(...)`.
#[pyfunction] fn preset_ga_cat(pop_size: usize, budget: u64) -> String {
    presets::ga_cat(pop_size, budget).to_json()
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
    m.add_function(wrap_pyfunction!(bias_f0, m)?)?;
    m.add_function(wrap_pyfunction!(cec2022, m)?)?;
    m.add_function(wrap_pyfunction!(cec2022_evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(cec2022_f_star, m)?)?;
    m.add_function(wrap_pyfunction!(cec2014, m)?)?;
    m.add_function(wrap_pyfunction!(cec2014_evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(cec2014_f_star, m)?)?;
    m.add_function(wrap_pyfunction!(cec2017, m)?)?;
    m.add_function(wrap_pyfunction!(cec2017_evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(cec2017_f_star, m)?)?;
    m.add_function(wrap_pyfunction!(tsp, m)?)?;
    m.add_function(wrap_pyfunction!(tsp_load, m)?)?;
    m.add_function(wrap_pyfunction!(tsp_tour_length, m)?)?;
    m.add_function(wrap_pyfunction!(onemax, m)?)?;
    m.add_function(wrap_pyfunction!(int_quadratic, m)?)?;
    m.add_function(wrap_pyfunction!(cat_match, m)?)?;
    m.add_function(wrap_pyfunction!(mixed_diagnostic, m)?)?;
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
    m.add_function(wrap_pyfunction!(bias_structural_positions, m)?)?;
    m.add_function(wrap_pyfunction!(bias_central, m)?)?;
    m.add_function(wrap_pyfunction!(bias_report, m)?)?;
    m.add_function(wrap_pyfunction!(mo_nsga2, m)?)?;
    m.add_function(wrap_pyfunction!(mo_hypervolume_2d, m)?)?;
    m.add_function(wrap_pyfunction!(mo_hypervolume, m)?)?;
    m.add_function(wrap_pyfunction!(mo_igd, m)?)?;
    m.add_function(wrap_pyfunction!(mo_pareto_front, m)?)?;
    m.add_function(wrap_pyfunction!(mo_evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(mo_evaluate_constraints, m)?)?;
    m.add_function(wrap_pyfunction!(mo_read_moa, m)?)?;
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
    m.add_function(wrap_pyfunction!(preset_ga_bin, m)?)?;
    m.add_function(wrap_pyfunction!(preset_ga_int, m)?)?;
    m.add_function(wrap_pyfunction!(preset_ga_cat, m)?)?;
    Ok(())
}
