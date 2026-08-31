//! R binding for `sezgi_bench::EvalSession` -- the ask/tell core behind the
//! spec's engine-inside-out promise (see `crates/bench/src/session.rs`'s
//! module doc). This is the first STATEFUL object in r-sezgi: savvy 0.10's
//! `#[savvy]` on a struct + impl block generates an external-pointer-backed
//! R object (a sealed `environment` with one closure per method -- see
//! `savvy-bindgen-0.10.2/src/codegen/r.rs::generate_r_impl_for_impl`), which
//! is exactly the mutable-handle mechanism the standing ruling asked to be
//! investigated. `py-sezgi/src/lib.rs`'s `PyEvalSession` is the semantic
//! reference this mirrors 1:1.
//!
//! ## Why this does NOT use savvy's consuming-`self` finish()
//!
//! savvy also supports a plain (non-reference) `self` receiver on a method
//! (`SavvyFnType::Method { reference: false, .. }` in `savvy-bindgen`):
//! generated code takes the value out of the external pointer via
//! `take_external_pointer_value`, which **nulls the R-side pointer**
//! (`R_ClearExternalPtr`), and any later `&self`/`&mut self` access on that
//! same R object would then hit `get_external_pointer_addr`'s own
//! null-pointer check and fail with savvy's `Error::InvalidPointer` --
//! so reuse-after-finish is already memory-safe by construction, with no
//! unsafe workaround needed.
//!
//! That path is not used here, though: `Error::InvalidPointer`'s message
//! does not contain the string `"session finished"` that Task 3's tests
//! (mirrored by this task's tests) match on. So, exactly like
//! `PyEvalSession`, every method here takes `&self`/`&mut self`, and
//! `finish` clears an internal `Option<CoreSession>` slot instead of
//! consuming the R handle -- giving the exact same post-finish error message
//! Python raises, by choice, not because savvy required the extra
//! indirection.
//!
//! ## `evaluate()`'s input shape
//!
//! `evaluate(x)` accepts EITHER a numeric matrix (rows = points, columns =
//! `dim`, R's column-major storage -- the exact convention already used by
//! `matrix_to_rows()` in `stats.rs`) OR a `list` of numeric vectors, one per
//! point. Both are cheap to support from one `Sexp`-typed argument (branch
//! on `Sexp::is_list()`), so both are implemented and both are exercised in
//! `tests/testthat/test-session.R`, rather than picking only one.
//!
//! ## Generic sessions (M3-5 Task 5; CEC 2014/CEC 2017 added M3-6 Task 10)
//!
//! [`EvalSession::new`] remains BBOB-only, unchanged. Four more associated
//! functions build a session over any other continuous problem this crate
//! exposes: [`EvalSession::new_cec2022`] / [`EvalSession::new_cec2014`] /
//! [`EvalSession::new_cec2017`] (logging allowed for all three -- each
//! `SessionMeta` matches py-sezgi's `PyEvalSession::for_problem`
//! `Inner::Cec2022`/`Inner::Cec2014`/`Inner::Cec2017` arm exactly, so R- and
//! Python-produced IOH records reconstruct identically) and
//! [`EvalSession::new_f0`] (the BIAS-toolbox null problem, `sezgi_bias::
//! F0Random` -- NO log parameters at all, since f0 has no known optimum and
//! `with_log` requires one). savvy generates each additional associated
//! function as `EvalSession$new_cec2022(...)` / `EvalSession$new_cec2014(...)`
//! / `EvalSession$new_cec2017(...)` / `EvalSession$new_f0(...)` (any
//! associated-function name other than `new` is namespaced under the type --
//! see `savvy-bindgen-0.10.2/src/codegen/r.rs::SavvyFn::name_r`'s "Special
//! convention" comment), each with its own hand-written R wrapper
//! (`sz_eval_session_cec2022()` / `sz_eval_session_cec2014()` /
//! `sz_eval_session_cec2017()` / `sz_eval_session_f0()` in `R/session.R`)
//! giving it R-native default arguments, same raw/wrapper pattern as
//! `sz_eval_session()` above. `$dim()`/`$bounds()` (see below) are available
//! on every session regardless of which constructor built it.

use savvy::{savvy, savvy_err, ListSexp, NullSexp, NumericSexp, OwnedListSexp, OwnedRealSexp, Sexp};
use sezgi_bench::{EvalSession as CoreSession, SessionMeta};
use sezgi_bias::F0Random;
use sezgi_core::problem::Problem;
use sezgi_problems::{Cec2014, Cec2017, Cec2022};
use std::path::Path;

/// Casts a non-negative-checked, WHOLE-NUMBER-checked `f64` (as passed from
/// R, which has no native unsigned integer type) to `u64`, rejecting
/// negative, non-finite, or fractional values. Duplicated from the
/// identically-named private helpers in `stats.rs` / `experiment.rs` -- no
/// shared private cross-module import, same rule as `parse_distribution`'s
/// duplication between `solve.rs` and py-sezgi's `lib.rs`.
///
/// M2d-3: previously silently truncated a fractional `x` toward zero (an
/// asymmetry with py-sezgi, whose `u64`-typed parameters make pyo3 reject a
/// fractional Python value outright at the FFI boundary). Now rejects it
/// the same way pyo3 does, so `sz_eval_session(budget = 10.5)` errors
/// instead of silently running a budget-10 session.
fn f64_to_u64(name: &str, x: f64) -> savvy::Result<u64> {
    if !x.is_finite() || x < 0.0 {
        return Err(savvy_err!(
            "{} must be a non-negative finite number, got {}",
            name,
            x
        ));
    }
    if x.fract() != 0.0 {
        return Err(savvy_err!("{} expected a whole number, got {}", name, x));
    }
    Ok(x as u64)
}

/// Raised by every [`EvalSession`] method once the session has been
/// [`EvalSession::finish`]ed (including a second call to `finish` itself) --
/// same message py-sezgi's `PyEvalSession` raises.
fn session_finished_err() -> savvy::Error {
    savvy_err!("session finished")
}

/// Converts `x` -- either a numeric matrix (rows = points, R's column-major
/// storage; identical transpose-avoidance logic to `matrix_to_rows()` in
/// `stats.rs`) or a `list` of numeric vectors, one per point -- into
/// `Vec<Vec<f64>>` for `CoreSession::evaluate`. Row/list-element order is
/// preserved either way; `CoreSession::evaluate` itself is what checks each
/// row's length against `dim` (`DimensionMismatch`), so this makes no
/// assumption about a fixed row width.
///
/// A `data.frame` is ALSO a `VECSXP` (R's `list` representation) -- one
/// element per COLUMN, not per point -- so `Sexp::is_list()` alone cannot
/// tell it apart from the intended "list of points" shape. Left unchecked,
/// a `data.frame` would silently be read column-wise (plausible-looking but
/// wrong values, no warning: reported by review as a real footgun). So the
/// `class` attribute is checked explicitly here and a `data.frame` is
/// rejected with a message pointing at the fix (`as.matrix(x)`), before it
/// ever reaches the list branch below.
fn sexp_to_rows(x: Sexp) -> savvy::Result<Vec<Vec<f64>>> {
    if x.is_list() {
        if let Some(classes) = x.get_class()
            && classes.contains(&"data.frame")
        {
            return Err(savvy_err!(
                "evaluate() does not accept a data.frame: it would be read column-wise, \
                 not row-wise, silently producing wrong points -- pass a numeric matrix \
                 instead, e.g. as.matrix(x) (rows = points)"
            ));
        }
        let list: ListSexp = x.try_into()?;
        let mut rows = Vec::with_capacity(list.len());
        for i in 0..list.len() {
            let elt = list
                .get_by_index(i)
                .ok_or_else(|| savvy_err!("x[[{}]] is missing", i + 1))?;
            let num: NumericSexp = elt.try_into()?;
            rows.push(num.as_slice_f64().to_vec());
        }
        Ok(rows)
    } else {
        let num: NumericSexp = x.try_into()?;
        let dim = num.get_dim().ok_or_else(|| {
            savvy_err!(
                "evaluate() expects a numeric matrix (rows = points) or a list of numeric vectors"
            )
        })?;
        if dim.len() != 2 {
            return Err(savvy_err!(
                "expected a 2-D matrix, got {} dimensions",
                dim.len()
            ));
        }
        let nrow = dim[0] as usize;
        let ncol = dim[1] as usize;
        let data = num.as_slice_f64();
        if data.len() != nrow * ncol {
            return Err(savvy_err!(
                "matrix data length {} does not match dim {}x{}",
                data.len(),
                nrow,
                ncol
            ));
        }

        let mut rows = vec![vec![0.0_f64; ncol]; nrow];
        for c in 0..ncol {
            for r in 0..nrow {
                rows[r][c] = data[c * nrow + r];
            }
        }
        Ok(rows)
    }
}

/// The ask/tell evaluation session -- see the module doc.
#[savvy]
struct EvalSession {
    inner: Option<CoreSession>,
}

#[savvy]
impl EvalSession {
    /// Builds a new ask/tell evaluation session over a BBOB problem.
    ///
    /// This is the savvy-generated constructor (`EvalSession$new(...)`); the
    /// public R entry point with R-native defaults (`algo_name = "custom"`,
    /// `seed = 0`) is the hand-written wrapper `sz_eval_session()` in
    /// `R/session.R`, which calls `EvalSession$new()` -- same raw/wrapper
    /// pattern used throughout this package (e.g. `sz_run_experiment()` /
    /// `sz_run_experiment_raw()`).
    ///
    /// `log_dir` is wired up ONLY here, before any `evaluate()` call is
    /// possible from R -- so `EvalSession::with_log`'s own "called after
    /// evaluation has started" guard is unreachable through this binding by
    /// construction (same constructor-only wiring as `PyEvalSession`).
    ///
    /// @param fid BBOB function id (>= 1).
    /// @param dim Problem dimension (>= 1).
    /// @param instance BBOB instance id (>= 1).
    /// @param budget Evaluation budget (non-negative).
    /// @param algo_name Algorithm label recorded in the IOH archive (only
    ///   meaningful when `log_dir` is given).
    /// @param seed Caller-declared reproducibility label recorded in the
    ///   IOH archive meta only -- the session itself never draws random
    ///   numbers, so this is purely documentation of the CALLER's RNG seed.
    /// @param log_dir Optional directory: when given, every evaluation is
    ///   logged in IOH-profiler format. savvy requires optional args to
    ///   come after mandatory ones in the generated signature, so this is
    ///   last here even though the public `sz_eval_session()` wrapper
    ///   places it right after `budget` (matching py-sezgi's argument
    ///   order) and reorders before calling `EvalSession$new()`.
    #[allow(clippy::too_many_arguments)]
    fn new(
        fid: i32,
        dim: i32,
        instance: i32,
        budget: f64,
        algo_name: &str,
        seed: f64,
        log_dir: Option<&str>,
    ) -> savvy::Result<Self> {
        if fid < 1 {
            return Err(savvy_err!("fid must be >= 1"));
        }
        if dim < 1 {
            return Err(savvy_err!("dim must be >= 1"));
        }
        if instance < 1 {
            return Err(savvy_err!("instance must be >= 1"));
        }
        let budget_u = f64_to_u64("budget", budget)?;
        let seed_u = f64_to_u64("seed", seed)?;

        let mut session =
            CoreSession::new_bbob(fid as u32, dim as usize, instance as u32, budget_u)
                .map_err(|e| savvy_err!("{e}"))?;
        if let Some(dir) = log_dir {
            session = session
                .with_log(Path::new(dir), algo_name, seed_u)
                .map_err(|e| savvy_err!("{e}"))?;
        }
        Ok(Self { inner: Some(session) })
    }

    /// Builds a new ask/tell evaluation session over a CEC 2022 problem
    /// (M3-5 Task 5 -- the generic-session counterpart to [`Self::new`]'s
    /// BBOB-only constructor). Generated as `EvalSession$new_cec2022(...)`;
    /// the public R entry point is the hand-written wrapper
    /// `sz_eval_session_cec2022()` in `R/session.R`.
    ///
    /// `SessionMeta` is built to match py-sezgi's `PyEvalSession::for_problem`
    /// `Inner::Cec2022` arm EXACTLY (`py-sezgi/src/lib.rs`): `suite =
    /// "sezgi-cec2022"`, `name = "cec2022-f{fid}"`, `instance = 1`, `f_opt =
    /// Cec2022::optimum()` -- so an IOH record produced by an R session and
    /// one produced by an equivalent Python session reconstruct to the
    /// identical `(suite, fid, name, instance)` key and `f_opt`.
    ///
    /// `log_dir` is wired up here, before any `evaluate()` call is
    /// possible -- same constructor-only placement as [`Self::new`], so
    /// `EvalSession::with_log`'s "called after evaluation has started"
    /// guard stays unreachable through this binding.
    ///
    /// @param fid CEC 2022 function id (`1..=12`).
    /// @param dim Problem dimension, one of `2`, `10`, `20` (`dim = 2` is
    ///   additionally rejected for a hybrid function, `fid` 6-8).
    /// @param budget Evaluation budget (non-negative).
    /// @param algo_name Algorithm label recorded in the IOH archive (only
    ///   meaningful when `log_dir` is given).
    /// @param seed Caller-declared reproducibility label recorded in the
    ///   IOH archive meta only.
    /// @param log_dir Optional directory: when given, every evaluation is
    ///   logged in IOH-profiler format.
    #[allow(clippy::too_many_arguments)]
    fn new_cec2022(
        fid: i32,
        dim: i32,
        budget: f64,
        algo_name: &str,
        seed: f64,
        log_dir: Option<&str>,
    ) -> savvy::Result<Self> {
        if fid < 1 {
            return Err(savvy_err!("fid must be >= 1"));
        }
        if dim < 1 {
            return Err(savvy_err!("dim must be >= 1"));
        }
        let budget_u = f64_to_u64("budget", budget)?;
        let seed_u = f64_to_u64("seed", seed)?;

        let fid_u = fid as u32;
        let problem = Cec2022::new(fid_u, dim as usize).map_err(|e| savvy_err!("{e}"))?;
        let meta = SessionMeta {
            suite: "sezgi-cec2022".to_string(),
            fid: fid_u,
            name: format!("cec2022-f{fid_u}"),
            instance: 1,
            f_opt: problem.optimum(),
        };
        let mut session = CoreSession::new_owned(Box::new(problem), meta, budget_u)
            .map_err(|e| savvy_err!("{e}"))?;
        if let Some(dir) = log_dir {
            session = session
                .with_log(Path::new(dir), algo_name, seed_u)
                .map_err(|e| savvy_err!("{e}"))?;
        }
        Ok(Self { inner: Some(session) })
    }

    /// Builds a new ask/tell evaluation session over a CEC 2014 problem
    /// (M3-6 Task 10 -- the generic-session counterpart to [`Self::new`]'s
    /// BBOB-only constructor and [`Self::new_cec2022`]'s CEC 2022
    /// counterpart, mirrored exactly). Generated as
    /// `EvalSession$new_cec2014(...)`; the public R entry point is the
    /// hand-written wrapper `sz_eval_session_cec2014()` in `R/session.R`.
    ///
    /// `SessionMeta` is built to match py-sezgi's `PyEvalSession::for_problem`
    /// `Inner::Cec2014` arm EXACTLY: `suite = "sezgi-cec2014"`, `name =
    /// "cec2014-f{fid}"`, `instance = 1`, `f_opt = Cec2014::f_star()` -- so an
    /// IOH record produced by an R session and one produced by an equivalent
    /// Python session reconstruct to the identical `(suite, fid, name,
    /// instance)` key and `f_opt`.
    ///
    /// `log_dir` is wired up here, before any `evaluate()` call is
    /// possible -- same constructor-only placement as [`Self::new_cec2022`].
    ///
    /// @param fid CEC 2014 function id (`1..=30`).
    /// @param dim Problem dimension, one of `10`, `30`.
    /// @param budget Evaluation budget (non-negative).
    /// @param algo_name Algorithm label recorded in the IOH archive (only
    ///   meaningful when `log_dir` is given).
    /// @param seed Caller-declared reproducibility label recorded in the
    ///   IOH archive meta only.
    /// @param log_dir Optional directory: when given, every evaluation is
    ///   logged in IOH-profiler format.
    #[allow(clippy::too_many_arguments)]
    fn new_cec2014(
        fid: i32,
        dim: i32,
        budget: f64,
        algo_name: &str,
        seed: f64,
        log_dir: Option<&str>,
    ) -> savvy::Result<Self> {
        if fid < 1 {
            return Err(savvy_err!("fid must be >= 1"));
        }
        if dim < 1 {
            return Err(savvy_err!("dim must be >= 1"));
        }
        let budget_u = f64_to_u64("budget", budget)?;
        let seed_u = f64_to_u64("seed", seed)?;

        let fid_u = fid as u32;
        let problem = Cec2014::new(fid_u, dim as usize).map_err(|e| savvy_err!("{e}"))?;
        let meta = SessionMeta {
            suite: "sezgi-cec2014".to_string(),
            fid: fid_u,
            name: format!("cec2014-f{fid_u}"),
            instance: 1,
            f_opt: problem.optimum(),
        };
        let mut session = CoreSession::new_owned(Box::new(problem), meta, budget_u)
            .map_err(|e| savvy_err!("{e}"))?;
        if let Some(dir) = log_dir {
            session = session
                .with_log(Path::new(dir), algo_name, seed_u)
                .map_err(|e| savvy_err!("{e}"))?;
        }
        Ok(Self { inner: Some(session) })
    }

    /// Builds a new ask/tell evaluation session over a CEC 2017 problem
    /// (M3-6 Task 10 -- mirrors [`Self::new_cec2014`] exactly). Generated as
    /// `EvalSession$new_cec2017(...)`; the public R entry point is the
    /// hand-written wrapper `sz_eval_session_cec2017()` in `R/session.R`.
    ///
    /// `SessionMeta` matches py-sezgi's `PyEvalSession::for_problem`
    /// `Inner::Cec2017` arm EXACTLY: `suite = "sezgi-cec2017"`, `name =
    /// "cec2017-f{fid}"`, `instance = 1`, `f_opt = Cec2017::f_star()`.
    ///
    /// `fid = 2` ("Sum of Different Powers") is officially withdrawn from the
    /// CEC 2017 suite -- [`Cec2017::new`] rejects it with
    /// [`sezgi_problems::Cec2017Error::Withdrawn`] before a problem handle
    /// can ever exist, surfaced VERBATIM here.
    ///
    /// @param fid CEC 2017 function id (`1` or `3..=30`).
    /// @param dim Problem dimension, one of `10`, `30`.
    /// @param budget Evaluation budget (non-negative).
    /// @param algo_name Algorithm label recorded in the IOH archive (only
    ///   meaningful when `log_dir` is given).
    /// @param seed Caller-declared reproducibility label recorded in the
    ///   IOH archive meta only.
    /// @param log_dir Optional directory: when given, every evaluation is
    ///   logged in IOH-profiler format.
    #[allow(clippy::too_many_arguments)]
    fn new_cec2017(
        fid: i32,
        dim: i32,
        budget: f64,
        algo_name: &str,
        seed: f64,
        log_dir: Option<&str>,
    ) -> savvy::Result<Self> {
        if fid < 1 {
            return Err(savvy_err!("fid must be >= 1"));
        }
        if dim < 1 {
            return Err(savvy_err!("dim must be >= 1"));
        }
        let budget_u = f64_to_u64("budget", budget)?;
        let seed_u = f64_to_u64("seed", seed)?;

        let fid_u = fid as u32;
        let problem = Cec2017::new(fid_u, dim as usize).map_err(|e| savvy_err!("{e}"))?;
        let meta = SessionMeta {
            suite: "sezgi-cec2017".to_string(),
            fid: fid_u,
            name: format!("cec2017-f{fid_u}"),
            instance: 1,
            f_opt: problem.optimum(),
        };
        let mut session = CoreSession::new_owned(Box::new(problem), meta, budget_u)
            .map_err(|e| savvy_err!("{e}"))?;
        if let Some(dir) = log_dir {
            session = session
                .with_log(Path::new(dir), algo_name, seed_u)
                .map_err(|e| savvy_err!("{e}"))?;
        }
        Ok(Self { inner: Some(session) })
    }

    /// Builds a new ask/tell evaluation session over the f0 BIAS-toolbox
    /// null problem ([`sezgi_bias::F0Random`]; see that module's own doc).
    /// Generated as `EvalSession$new_f0(...)`; the public R entry point is
    /// the hand-written wrapper `sz_eval_session_f0()` in `R/session.R`.
    ///
    /// Deliberately has NO `log_dir`/`algo_name`/`seed` (log) parameters at
    /// all -- f0 has no known optimum (`SessionMeta::f_opt = None`), and
    /// `EvalSession::with_log` rejects a `None` `f_opt` (see
    /// `crates/bench/src/session.rs`'s `LogRequiresKnownOptimum`), so
    /// offering log arguments here would advertise a capability this
    /// constructor can never honor. `SessionMeta` otherwise matches
    /// py-sezgi's `PyEvalSession::for_problem` `Inner::F0` arm: `suite =
    /// "sezgi-f0"`, `fid = 0`, `name = "f0"`, `instance = 1`, `f_opt =
    /// None`.
    ///
    /// @param dim f0's domain dimensionality (>= 1); the space is
    ///   `[0,1]^dim`.
    /// @param f0_seed Seed for f0's own RNG stream (see
    ///   [`sezgi_bias::F0Random::new`]'s doc) -- distinct from any engine
    ///   RNG seed.
    /// @param budget Evaluation budget (non-negative).
    fn new_f0(dim: i32, f0_seed: f64, budget: f64) -> savvy::Result<Self> {
        if dim < 1 {
            return Err(savvy_err!("dim must be >= 1"));
        }
        let seed_u = f64_to_u64("f0_seed", f0_seed)?;
        let budget_u = f64_to_u64("budget", budget)?;

        let problem = F0Random::new(dim as usize, seed_u);
        let meta = SessionMeta {
            suite: "sezgi-f0".to_string(),
            fid: 0,
            name: "f0".to_string(),
            instance: 1,
            f_opt: None,
        };
        let session = CoreSession::new_owned(Box::new(problem), meta, budget_u)
            .map_err(|e| savvy_err!("{e}"))?;
        Ok(Self { inner: Some(session) })
    }

    /// Batch-evaluates `x` -- see the module doc for the accepted shapes.
    /// All-or-nothing: on any error (dimension mismatch, a non-finite
    /// coordinate, or budget overrun) nothing is counted.
    ///
    /// @param x A numeric matrix (rows = points) or a `list` of numeric
    ///   vectors.
    /// @returns A numeric vector of `f` values, one per point (row/element
    ///   order preserved).
    fn evaluate(&mut self, x: Sexp) -> savvy::Result<Sexp> {
        let session = self.inner.as_mut().ok_or_else(session_finished_err)?;
        let xs = sexp_to_rows(x)?;
        let fs = session.evaluate(&xs).map_err(|e| savvy_err!("{e}"))?;
        fs.try_into()
    }

    /// Number of evaluations counted so far.
    /// @returns A numeric scalar.
    fn evals_used(&self) -> savvy::Result<Sexp> {
        let session = self.inner.as_ref().ok_or_else(session_finished_err)?;
        (session.evals_used() as f64).try_into()
    }

    /// The session's total evaluation budget.
    /// @returns A numeric scalar.
    fn budget(&self) -> savvy::Result<Sexp> {
        let session = self.inner.as_ref().ok_or_else(session_finished_err)?;
        (session.budget() as f64).try_into()
    }

    /// The best evaluation seen so far.
    /// @returns A named list `list(x = <numeric vector>, f = <numeric
    ///   scalar>)`, or R `NULL` if nothing has been evaluated yet.
    fn best(&self) -> savvy::Result<Sexp> {
        let session = self.inner.as_ref().ok_or_else(session_finished_err)?;
        match session.best() {
            None => Ok(NullSexp.into()),
            Some((x, f)) => {
                let mut out = OwnedListSexp::new(2, true)?;
                out.set_name_and_value(0, "x", OwnedRealSexp::try_from_slice(x)?)?;
                out.set_name_and_value(1, "f", OwnedRealSexp::try_from_scalar(f)?)?;
                Ok(out.into())
            }
        }
    }

    /// The problem's known optimum value, or R `NULL` if it has none.
    /// @returns A numeric scalar, or `NULL` (e.g. an f0 session -- see
    ///   `sz_eval_session_f0()`).
    fn f_opt(&self) -> savvy::Result<Sexp> {
        let session = self.inner.as_ref().ok_or_else(session_finished_err)?;
        // `CoreSession::f_opt()` is `Option<f64>` (M3-4 Task 1 generalized
        // `EvalSession` beyond BBOB). Task 5 (M3-5) adds `new_cec2022`/
        // `new_f0` constructors alongside the BBOB-only `new()` above; BBOB
        // and CEC 2022 sessions always have a known optimum (`Some`), but
        // an f0 session never does, so this now maps `Option<f64>` to
        // R `NULL`/scalar honestly instead of `.expect()`-ing `Some`.
        match session.f_opt() {
            Some(v) => v.try_into(),
            None => Ok(NullSexp.into()),
        }
    }

    /// The search space's dimensionality.
    /// @returns A numeric scalar (whole number).
    fn dim(&self) -> savvy::Result<Sexp> {
        let session = self.inner.as_ref().ok_or_else(session_finished_err)?;
        (session.dim() as f64).try_into()
    }

    /// The uniform `(lo, hi)` bounds of this session's continuous (float)
    /// space -- see `sezgi_bench::EvalSession::bounds`'s doc for the exact
    /// rule (errors for a non-uniform/non-float space; unreachable through
    /// every constructor this binding exposes today, since BBOB, CEC 2022,
    /// and f0 are each a single uniform `Float` block, but the error path
    /// is kept honest rather than assumed away).
    /// @returns A length-2 numeric vector `c(lo, hi)`.
    fn bounds(&self) -> savvy::Result<Sexp> {
        let session = self.inner.as_ref().ok_or_else(session_finished_err)?;
        let (lo, hi) = session.bounds().map_err(|e| savvy_err!("{e}"))?;
        Ok(OwnedRealSexp::try_from_slice([lo, hi])?.into())
    }

    /// Flushes the IOH log (if logging was enabled) and marks the session
    /// as finished. Every method call afterward, including a second
    /// `finish()`, raises an error whose message contains
    /// `"session finished"`.
    fn finish(&mut self) -> savvy::Result<()> {
        let session = self.inner.take().ok_or_else(session_finished_err)?;
        session.finish().map_err(|e| savvy_err!("{e}"))
    }
}
