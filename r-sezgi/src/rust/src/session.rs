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

use savvy::{savvy, savvy_err, ListSexp, NullSexp, NumericSexp, OwnedListSexp, OwnedRealSexp, Sexp};
use sezgi_bench::EvalSession as CoreSession;
use std::path::Path;

/// Casts a non-negative-checked `f64` (as passed from R, which has no native
/// unsigned integer type) to `u64`, rejecting negative or non-finite values.
/// Duplicated from the identically-named private helpers in `stats.rs` /
/// `experiment.rs` -- no shared private cross-module import, same rule as
/// `parse_distribution`'s duplication between `solve.rs` and py-sezgi's
/// `lib.rs`.
fn f64_to_u64(name: &str, x: f64) -> savvy::Result<u64> {
    if !x.is_finite() || x < 0.0 {
        return Err(savvy_err!(
            "{} must be a non-negative finite number, got {}",
            name,
            x
        ));
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

    /// The problem's known optimum value.
    /// @returns A numeric scalar.
    fn f_opt(&self) -> savvy::Result<Sexp> {
        let session = self.inner.as_ref().ok_or_else(session_finished_err)?;
        session.f_opt().try_into()
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
