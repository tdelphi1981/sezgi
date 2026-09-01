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
//!
//! ## Permutation-typed sessions (M3-8 Task 8: TSP)
//!
//! [`EvalSession::new_tsp`] builds a session over a vendored TSPLIB instance
//! (`Tsp::vendored`) -- the R mirror of py-sezgi's M3-8 Task 7 `PermSession`/
//! `sezgi.problems.tsp(...)` typed session (`py-sezgi/src/lib.rs`'s module
//! doc is the semantic reference this section mirrors). Unlike every
//! constructor above, whose [`CoreSession`] is `sezgi_bench::EvalSession`
//! itself, `new_tsp` builds a crate-local [`PermSession`] instead -- the
//! SAME reason py-sezgi keeps its own `PermSession` crate-local:
//! `CoreSession::evaluate` is hard-coded to `Vec<Vec<f64>>` rows /
//! `BlockValues::Float` genotypes, so a permutation-typed session cannot
//! reuse it; [`PermSession`] is instead built directly on
//! [`sezgi_core::problem::Evaluator`], the SAME tamper-proof budget-check/
//! counting primitive `CoreSession` itself is built on. `EvalSession$inner`
//! becomes [`SessionKind`] (`Float`/`Perm`) to hold either kind behind one R
//! handle; every existing method (`evaluate`/`evals_used`/`budget`/`best`/
//! `f_opt`/`dim`/`bounds`/`finish`) now dispatches on it, with the `Float`
//! arm of each UNCHANGED from before this task. Two new methods,
//! `$kind()` and `$random_permutation()`, are added (generated as
//! `EvalSession$kind()`/`EvalSession$random_permutation()`, both callable on
//! ANY session regardless of kind -- `kind()` always answers, and
//! `random_permutation()` errors with a clear message on a Float-typed one).
//!
//! **1-based tours, converted at THIS Rust boundary**: per this crate's
//! standing index-convention ruling (`problems.rs`'s module doc,
//! "Index-convention decision"), every tour value crossing the R/Rust
//! boundary here -- `evaluate()`'s input rows, `random_permutation()`'s
//! return value, `best()`'s `x` field -- is 1-based (a permutation of
//! `1..=n_cities`) on the R side, converted to/from the underlying 0-based
//! `BlockValues::Perm` genotype entirely within [`PermSession`]'s own
//! methods (mirrors `sz_tsp_tour_length`'s own `-1`/`+1` conversion in
//! `problems.rs` exactly). `sezgi_problems::Tsp` itself is untouched (still
//! 0-based).
//!
//! **RNG parity with py-sezgi**: [`PermSession::random_permutation`] draws
//! from a Rust-side [`RngStream`] seeded via `RngStream::from_master(seed,
//! &[PERM_SESSION_RNG_TAG])` -- the identical tag value and derivation
//! py-sezgi's own `PermSession` uses (`py-sezgi/src/lib.rs`'s
//! `PERM_SESSION_RNG_TAG`) -- over the SAME shared
//! [`sezgi_components::perm::fisher_yates_shuffle`] core. So for the SAME
//! `seed`, the R and Python sessions draw the bit-identical UNDERLYING
//! 0-based permutation on their first `random_permutation()` call; R simply
//! displays it shifted `+1`. `two_opt(tour, i, j)` (the third piece of the
//! Task 8 `ctx` surface) has NO Rust involvement at all -- like py-sezgi's
//! own `AlgoContext.two_opt` (pure Python, no RNG), it is pure R, added to
//! `.sz_algo_context()` in `R/algo.R`, operating directly on a 1-based tour
//! with no FFI boundary to cross.

use savvy::{
    savvy, savvy_err, ListSexp, NullSexp, NumericSexp, OwnedListSexp, OwnedRealSexp,
    OwnedStringSexp, Sexp,
};
use sezgi_bench::{EvalSession as CoreSession, SessionMeta};
use sezgi_bias::F0Random;
use sezgi_components::perm::fisher_yates_shuffle;
use sezgi_core::problem::{Evaluator, Problem};
use sezgi_core::rng::RngStream;
use sezgi_core::space::{BlockValues, Genotype};
use sezgi_problems::{Cec2014, Cec2017, Cec2022, Tsp};
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

/// Path tag folded into [`PermSession`]'s [`RngStream`] (see
/// [`RngStream::from_master`]) -- IDENTICAL value and role to py-sezgi's
/// `PERM_SESSION_RNG_TAG` (`py-sezgi/src/lib.rs`): an arbitrary but FIXED
/// value distinguishing this stream from any other stream this crate might
/// ever derive from the same `seed`. Kept equal to the Python side ON
/// PURPOSE -- this is what makes `EvalSession::new_tsp(...)`'s
/// `random_permutation()` draw the bit-identical underlying 0-based
/// permutation as `sezgi.EvalSession.for_problem(sezgi.problems.tsp(...))`'s
/// own `random_permutation()`, for the same `seed`. Changing it would
/// silently change every future draw sequence AND break that cross-language
/// parity.
const PERM_SESSION_RNG_TAG: u64 = 0x5045524D; // "PERM", arbitrary ASCII-hex mnemonic (matches py-sezgi)

/// M3-8 Task 8: ask/tell session over a permutation-typed (TSP) problem --
/// the R mirror of py-sezgi's crate-local `PermSession`
/// (`py-sezgi/src/lib.rs`, M3-8 Task 7). See this module's own doc,
/// "Permutation-typed sessions", for the full design rationale.
///
/// UNLIKE py-sezgi's `PermSession` (0-based throughout, matching Python's
/// own `tour`/`coords` convention), every tour value this struct's public
/// methods accept or return is 1-BASED (r-sezgi's standing index-convention
/// ruling, `problems.rs`'s module doc) -- the `-1`/`+1` conversion to/from
/// the underlying 0-based `BlockValues::Perm` genotype happens entirely
/// inside [`Self::evaluate`]/[`Self::random_permutation`], never leaking a
/// 0-based value to a caller. `self.best`, in particular, stores the
/// 1-based tour (not the 0-based genotype order) so [`Self::best`] needs no
/// further conversion at its own call site.
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

    /// A uniformly random 1-based tour (a permutation of `1..=n`), drawn
    /// from this session's own [`RngStream`] via
    /// [`sezgi_components::perm::fisher_yates_shuffle`] -- the SAME core
    /// py-sezgi's `PermSession::random_permutation` calls, over a stream
    /// seeded identically (see [`PERM_SESSION_RNG_TAG`]'s doc) -- so the
    /// underlying 0-based draw is bit-identical to Python's for the same
    /// `seed` and call sequence; the `+1` shift to r-sezgi's 1-based
    /// convention happens ONLY here, at this Rust boundary. Each call
    /// advances the stream, so successive calls draw DIFFERENT
    /// permutations.
    fn random_permutation(&mut self) -> Vec<u32> {
        fisher_yates_shuffle(self.n, &mut self.rng)
            .into_iter()
            .map(|c| c + 1)
            .collect()
    }

    /// Batch-evaluates `tours` -- 1-based (a permutation of `1..=n`, one row
    /// per point; R's numeric doubles, converted/validated entry-by-entry
    /// here). All-or-nothing, mirroring `CoreSession::evaluate`'s own error
    /// semantics (this module's doc) and py-sezgi's `PermSession::evaluate`:
    /// every row is fully validated (length, whole-number entries,
    /// `1..=n` range, no repeats) BEFORE the counter moves, so a rejected
    /// batch leaves `used` untouched and evaluates nothing. Once validated,
    /// each row is shifted `-1` into a 0-based `BlockValues::Perm` genotype
    /// and run through a short-lived `Evaluator` sized to the remaining
    /// budget -- the identical all-or-nothing budget-check reuse pattern
    /// `CoreSession::evaluate` itself uses. Best-tracking uses the same
    /// strict-improvement tie rule (`!(b <= f)`) as `Evaluator`/
    /// `CoreSession`, storing the tour 1-based (see the struct doc).
    fn evaluate(&mut self, tours: &[Vec<f64>]) -> savvy::Result<Vec<f64>> {
        let mut orders: Vec<Vec<u32>> = Vec::with_capacity(tours.len());
        for (row, t) in tours.iter().enumerate() {
            if t.len() != self.n {
                return Err(savvy_err!(
                    "PermSession::evaluate: row {} has {} entries, expected {} \
                     (one per city)",
                    row,
                    t.len(),
                    self.n
                ));
            }
            let mut seen = vec![false; self.n];
            let mut order = Vec::with_capacity(self.n);
            for &v in t {
                if !v.is_finite() || v.fract() != 0.0 {
                    return Err(savvy_err!(
                        "PermSession::evaluate: row {} has a non-whole-number entry {}; \
                         expected a 1-based city index",
                        row,
                        v
                    ));
                }
                let city1 = v as i64;
                if city1 < 1 || city1 as usize > self.n {
                    return Err(savvy_err!(
                        "PermSession::evaluate: row {} has out-of-range entry {} \
                         (expected 1..={})",
                        row,
                        city1,
                        self.n
                    ));
                }
                let ci = (city1 - 1) as usize;
                if seen[ci] {
                    return Err(savvy_err!(
                        "PermSession::evaluate: row {} has a repeated city {}; \
                         not a valid permutation",
                        row,
                        city1
                    ));
                }
                seen[ci] = true;
                order.push(ci as u32);
            }
            orders.push(order);
        }

        let pop: Vec<Genotype> = orders
            .iter()
            .map(|o| Genotype { blocks: vec![BlockValues::Perm(o.clone())] })
            .collect();

        // Short-lived Evaluator sized to the remaining budget -- same reuse
        // pattern as CoreSession::evaluate: its own all-or-nothing check IS
        // this session's check.
        let remaining = self.budget - self.used;
        let mut ev = Evaluator::new(&*self.problem, remaining);
        let fs = ev.evaluate(&pop).map_err(|e| {
            savvy_err!(
                "PermSession::evaluate: budget exceeded ({}/{} used, {} requested)",
                self.used,
                self.budget,
                e.requested
            )
        })?;

        for (order, &f) in orders.iter().zip(&fs) {
            self.used += 1;
            // Same tie rule as Evaluator::evaluate / CoreSession::evaluate:
            // update only on strict improvement, ties keep the earlier tour.
            let improved = !matches!(&self.best, Some((_, b)) if *b <= f);
            if improved {
                // +1: store the R-facing 1-based tour, not the 0-based order.
                self.best = Some((order.iter().map(|&c| c + 1).collect(), f));
            }
        }
        Ok(fs)
    }

    fn evals_used(&self) -> u64 {
        self.used
    }
    fn budget(&self) -> u64 {
        self.budget
    }
    fn best(&self) -> Option<(Vec<u32>, f64)> {
        self.best.clone()
    }
    /// The problem's known optimum, or `None` if it has none. A vendored TSP
    /// instance (`Tsp::vendored`) returns `Some` for all three
    /// (`berlin52`/`eil51`/`st70`, `tsp.rs`'s pinned published optima).
    fn f_opt(&self) -> Option<f64> {
        self.problem.optimum()
    }
}

/// One [`EvalSession`]'s underlying Rust session -- `Float` for every
/// continuous-problem constructor above (`new`/`new_cec2022`/`new_cec2014`/
/// `new_cec2017`/`new_f0`), `Perm` for [`EvalSession::new_tsp`] (M3-8 Task
/// 8). `EvalSession::kind()` surfaces which one a given session is
/// (`"float"`/`"permutation"`) to R.
enum SessionKind {
    Float(CoreSession),
    Perm(PermSession),
}

/// The ask/tell evaluation session -- see the module doc.
#[savvy]
struct EvalSession {
    inner: Option<SessionKind>,
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
        Ok(Self { inner: Some(SessionKind::Float(session)) })
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
        Ok(Self { inner: Some(SessionKind::Float(session)) })
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
        Ok(Self { inner: Some(SessionKind::Float(session)) })
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
        Ok(Self { inner: Some(SessionKind::Float(session)) })
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
        Ok(Self { inner: Some(SessionKind::Float(session)) })
    }

    /// Builds a new ask/tell evaluation session over a vendored TSPLIB
    /// instance (M3-8 Task 8 -- the R mirror of py-sezgi's M3-8 Task 7
    /// `PermSession`/`sezgi.problems.tsp(...)` typed session; see this
    /// module's own doc, "Permutation-typed sessions", for the full design).
    /// Generated as `EvalSession$new_tsp(...)`; the public R entry point is
    /// the hand-written wrapper `sz_eval_session_tsp()` in `R/session.R`.
    ///
    /// Deliberately has NO `log_dir`/`algo_name` parameters at all -- IOH
    /// logging is not wired up for permutation-typed sessions (no
    /// suite/fid identity exists to log a TSP run against), same choice
    /// [`Self::new_f0`] already makes for its own unsupported-logging case
    /// (offering and then rejecting the parameter would be worse than never
    /// offering it).
    ///
    /// @param name A vendored TSPLIB instance name (`"berlin52"`, `"eil51"`,
    ///   `"st70"` -- see [`Tsp::vendored`]). UNLIKE `sz_tsp_load()`/
    ///   `sz_tsp_tour_length()` in `problems.rs`, raw TSPLIB file text is
    ///   not accepted here -- mirrors `sz_solve_tsp()`'s own vendored-only
    ///   restriction, and py-sezgi's `sezgi.problems.tsp(name)`.
    /// @param budget Evaluation budget (non-negative).
    /// @param seed Master RNG seed for this session's own
    ///   `random_permutation()` draws (see [`PERM_SESSION_RNG_TAG`]'s doc
    ///   for the cross-language parity this seed drives) -- distinct from
    ///   any IOH-log seed, since this session never logs.
    fn new_tsp(name: &str, budget: f64, seed: f64) -> savvy::Result<Self> {
        let budget_u = f64_to_u64("budget", budget)?;
        let seed_u = f64_to_u64("seed", seed)?;

        let problem = Tsp::vendored(name).map_err(|e| savvy_err!("{e}"))?;
        let session = PermSession::new(Box::new(problem), seed_u, budget_u);
        Ok(Self { inner: Some(SessionKind::Perm(session)) })
    }

    /// Batch-evaluates `x` -- see the module doc for the accepted shapes.
    /// A Float-typed session expects each row to be `dim` coordinates; a
    /// permutation-typed one (`kind() == "permutation"`) expects each row
    /// to be a 1-based tour (a permutation of `1..=n`, this module's doc,
    /// "Permutation-typed sessions"). All-or-nothing either way: on any
    /// error (dimension mismatch / an invalid tour, a non-finite
    /// coordinate, or budget overrun) nothing is counted. The Float path is
    /// UNCHANGED from before this task; see [`PermSession::evaluate`] for
    /// the permutation path's own validation.
    ///
    /// @param x A numeric matrix (rows = points) or a `list` of numeric
    ///   vectors.
    /// @returns A numeric vector of `f` values, one per point (row/element
    ///   order preserved).
    fn evaluate(&mut self, x: Sexp) -> savvy::Result<Sexp> {
        let session = self.inner.as_mut().ok_or_else(session_finished_err)?;
        let rows = sexp_to_rows(x)?;
        match session {
            SessionKind::Float(s) => {
                let fs = s.evaluate(&rows).map_err(|e| savvy_err!("{e}"))?;
                fs.try_into()
            }
            SessionKind::Perm(s) => {
                let fs = s.evaluate(&rows)?;
                fs.try_into()
            }
        }
    }

    /// Number of evaluations counted so far.
    /// @returns A numeric scalar.
    fn evals_used(&self) -> savvy::Result<Sexp> {
        let used = match self.inner.as_ref().ok_or_else(session_finished_err)? {
            SessionKind::Float(s) => s.evals_used(),
            SessionKind::Perm(s) => s.evals_used(),
        };
        (used as f64).try_into()
    }

    /// The session's total evaluation budget.
    /// @returns A numeric scalar.
    fn budget(&self) -> savvy::Result<Sexp> {
        let budget = match self.inner.as_ref().ok_or_else(session_finished_err)? {
            SessionKind::Float(s) => s.budget(),
            SessionKind::Perm(s) => s.budget(),
        };
        (budget as f64).try_into()
    }

    /// The best evaluation seen so far. For a permutation-typed session,
    /// `x` is the 1-based tour (this module's doc, "Permutation-typed
    /// sessions") -- no further conversion needed at this call site, since
    /// [`PermSession`] already stores it 1-based.
    /// @returns A named list `list(x = <numeric vector>, f = <numeric
    ///   scalar>)`, or R `NULL` if nothing has been evaluated yet.
    fn best(&self) -> savvy::Result<Sexp> {
        let (x, f) = match match self.inner.as_ref().ok_or_else(session_finished_err)? {
            SessionKind::Float(s) => s.best().map(|(x, f)| (x.to_vec(), f)),
            SessionKind::Perm(s) => {
                s.best().map(|(x, f)| (x.into_iter().map(f64::from).collect::<Vec<f64>>(), f))
            }
        } {
            None => return Ok(NullSexp.into()),
            Some(xf) => xf,
        };
        let mut out = OwnedListSexp::new(2, true)?;
        out.set_name_and_value(0, "x", OwnedRealSexp::try_from_slice(x.as_slice())?)?;
        out.set_name_and_value(1, "f", OwnedRealSexp::try_from_scalar(f)?)?;
        Ok(out.into())
    }

    /// The problem's known optimum value, or R `NULL` if it has none.
    /// @returns A numeric scalar, or `NULL` (e.g. an f0 session -- see
    ///   `sz_eval_session_f0()`).
    fn f_opt(&self) -> savvy::Result<Sexp> {
        // `f_opt()` is `Option<f64>` on both session kinds (M3-4 Task 1
        // generalized `EvalSession` beyond BBOB; M3-8 Task 8 adds the Perm
        // arm the same way). BBOB, CEC 2022/2014/2017, and every vendored
        // TSP instance always have a known optimum (`Some`), but an f0
        // session never does, so this maps `Option<f64>` to R `NULL`/scalar
        // honestly instead of `.expect()`-ing `Some`.
        match match self.inner.as_ref().ok_or_else(session_finished_err)? {
            SessionKind::Float(s) => s.f_opt(),
            SessionKind::Perm(s) => s.f_opt(),
        } {
            Some(v) => v.try_into(),
            None => Ok(NullSexp.into()),
        }
    }

    /// The search space's dimensionality -- for a permutation-typed
    /// session, the number of cities (`PermSession::n`, the same value
    /// `random_permutation()`/`evaluate()` expect a tour's length to be).
    /// @returns A numeric scalar (whole number).
    fn dim(&self) -> savvy::Result<Sexp> {
        let dim = match self.inner.as_ref().ok_or_else(session_finished_err)? {
            SessionKind::Float(s) => s.dim(),
            SessionKind::Perm(s) => s.n,
        };
        (dim as f64).try_into()
    }

    /// The uniform `(lo, hi)` bounds of this session's continuous (float)
    /// space -- see `sezgi_bench::EvalSession::bounds`'s doc for the exact
    /// rule (errors for a non-uniform/non-float space; unreachable through
    /// every Float-typed constructor this binding exposes today, since
    /// BBOB, CEC 2022/2014/2017, and f0 are each a single uniform `Float`
    /// block, but the error path is kept honest rather than assumed away).
    /// A permutation-typed session (`kind() == "permutation"`) has no
    /// uniform domain at all -- errors unconditionally, mirroring
    /// py-sezgi's `AlgoContext.random_point()` raising a clear `ValueError`
    /// for a permutation-typed context rather than an opaque one.
    /// @returns A length-2 numeric vector `c(lo, hi)`.
    fn bounds(&self) -> savvy::Result<Sexp> {
        match self.inner.as_ref().ok_or_else(session_finished_err)? {
            SessionKind::Float(s) => {
                let (lo, hi) = s.bounds().map_err(|e| savvy_err!("{e}"))?;
                Ok(OwnedRealSexp::try_from_slice([lo, hi])?.into())
            }
            SessionKind::Perm(_) => Err(savvy_err!(
                "bounds() is not available for a permutation-typed session \
                 (this session's kind() is \"permutation\")"
            )),
        }
    }

    /// This session's kind: `"float"` for every continuous-problem
    /// constructor (`new`/`new_cec2022`/`new_cec2014`/`new_cec2017`/
    /// `new_f0`), `"permutation"` for [`Self::new_tsp`] (M3-8 Task 8).
    /// Drives `ctx$kind()`/`ctx$n()`/`ctx$bounds()` in `R/algo.R`, mirroring
    /// py-sezgi's `AlgoContext.kind` exactly.
    /// @returns A character scalar, `"float"` or `"permutation"`.
    fn kind(&self) -> savvy::Result<Sexp> {
        let k = match self.inner.as_ref().ok_or_else(session_finished_err)? {
            SessionKind::Float(_) => "float",
            SessionKind::Perm(_) => "permutation",
        };
        Ok(OwnedStringSexp::try_from(k)?.into())
    }

    /// A uniformly random 1-based tour (a permutation of `1..=n`) drawn from
    /// this session's own seeded Rust-side RNG stream -- see
    /// [`PermSession::random_permutation`]'s own doc for the shuffle
    /// algorithm and this module's doc ("Permutation-typed sessions") for
    /// the cross-language RNG-parity guarantee with py-sezgi.
    ///
    /// @returns A numeric vector of length `dim()`, a permutation of
    ///   `1:dim()`.
    ///
    /// # Errors
    /// A savvy error if this session's `kind()` is `"float"` -- there is no
    /// permutation to draw for a continuous-typed session.
    fn random_permutation(&mut self) -> savvy::Result<Sexp> {
        let session = self.inner.as_mut().ok_or_else(session_finished_err)?;
        match session {
            SessionKind::Perm(s) => {
                let tour: Vec<f64> = s.random_permutation().into_iter().map(f64::from).collect();
                Ok(OwnedRealSexp::try_from_slice(tour.as_slice())?.into())
            }
            SessionKind::Float(_) => Err(savvy_err!(
                "random_permutation() is only available for permutation-typed \
                 sessions (this session's kind() is \"float\")"
            )),
        }
    }

    /// Flushes the IOH log (if logging was enabled) and marks the session
    /// as finished. Every method call afterward, including a second
    /// `finish()`, raises an error whose message contains
    /// `"session finished"`. A permutation-typed session never logs (see
    /// [`Self::new_tsp`]'s doc), so `finish()` on one is a no-op beyond
    /// clearing the slot -- mirrors py-sezgi's `PyEvalSession::finish`'s own
    /// `SessionKind::Perm` arm exactly.
    fn finish(&mut self) -> savvy::Result<()> {
        match self.inner.take().ok_or_else(session_finished_err)? {
            SessionKind::Float(s) => s.finish().map_err(|e| savvy_err!("{e}")),
            SessionKind::Perm(_) => Ok(()),
        }
    }
}
