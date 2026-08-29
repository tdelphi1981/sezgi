//! `EvalSession`: the ask/tell core behind the spec's engine-inside-out
//! promise (spec §2). An outer algorithm (Rust today; pure Python/R later,
//! via FFI) *generates* candidate points; this session stays the sole
//! keeper of evaluation, tamper-proof counting, best-tracking and IOH
//! logging — the parts that must be bit-identical to the engine-driven path
//! regardless of who is proposing candidates.
//!
//! ## Design: reusing `Evaluator` without duplicating its counting logic
//!
//! [`sezgi_core::problem::Evaluator`] already owns the tricky, must-not-drift
//! logic: the all-or-nothing budget check, calling `Problem::evaluate_batch`,
//! and asserting the returned batch has the right length. Its `used`/`best`
//! fields are private (by design — see `problem.rs`'s comment on tamper
//! -proofing), so they cannot be seeded from another crate, and its
//! lifetime-parameterized `&'a dyn Problem` borrow does not fit a
//! long-lived, owned session.
//!
//! [`EvalSession`] resolves this by owning the [`BbobProblem`] itself and
//! constructing a **short-lived `Evaluator` per `evaluate()` call**, sized
//! to `budget - used_so_far` (the session's *remaining* budget). That
//! Evaluator's own all-or-nothing check — `used + req > budget` — thus IS
//! the session's check, verbatim, not a reimplementation of it: the
//! borrow-checker-friendliest way to reuse the exact code path without
//! copying it. The same is true of the `evaluate_batch` call and its length
//! assertion.
//!
//! What IS re-derived (about three lines) is best-tracking: `Evaluator`'s
//! own `best` is call-local (it starts fresh at `None` every time a new
//! Evaluator is built) and cannot report a *cross-call* minimum. So after
//! each successful call, `EvalSession` walks the returned `Vec<f64>` once
//! more and folds each value into its own `best: Option<(Vec<f64>, f64)>`
//! using the *identical* comparison `Evaluator::evaluate` uses internally
//! (`Some(b) if b <= f => b, _ => f`, i.e. update only on strict
//! improvement — ties keep the earlier `x`). Only the *value* side of this
//! comparison is mirrored from `Evaluator`, which tracks `f64` only, no `x`;
//! keeping the associated `x` alongside the winning value is necessarily new
//! session behavior (there is nothing in `Evaluator` to mirror it from),
//! required because [`Self::best`] returns `(&[f64], f64)`. This is the
//! only piece of "counting logic" not reused verbatim, and it is
//! unavoidable given `Evaluator`'s private fields.
//!
//! ## Error semantics
//!
//! - **Budget**: all-or-nothing, exactly like `Evaluator::evaluate` — a
//!   batch that would cross the remaining budget is rejected *before* any
//!   row is evaluated, and the counter does not move. (Confirmed by reading
//!   `crates/core/src/problem.rs::Evaluator::evaluate`: the check happens
//!   before `problem.evaluate_batch` is even called.)
//! - **Dimension mismatch**: `Evaluator`/`BbobProblem` do **not** check this
//!   at all — `BbobProblem::evaluate_batch` just `zip`s `xs` against
//!   `x_opt`, silently truncating a too-long row or under-consuming a
//!   too-short one, with no panic and no error. That silent truncation is
//!   fine for the internal engine (every `Genotype` it builds is already
//!   sized by `SearchSpace`), but it is exactly the kind of bug an external
//!   ask/tell caller (raw `Vec<f64>` rows from Python/R) needs a real error
//!   for. So `EvalSession::evaluate` adds its **own** dimension check
//!   (against `self.problem.space().dim()`) up front, before constructing
//!   any `Genotype` or touching the counter — all-or-nothing, and it names
//!   the offending row index.
//! - **Non-finite coordinates (NaN/±Inf) in x**: `BbobProblem::evaluate_batch`
//!   does not reject them itself — a non-finite coordinate just flows through
//!   the arithmetic (typically producing a non-finite `f`), and (as read in
//!   `Evaluator::evaluate`) its best-tracking comparison
//!   `Some(b) if b <= f => b, _ => f` would treat a NaN `f` as *strictly
//!   better* than any existing best (`b <= NaN` is `false` in IEEE 754, so
//!   the `_ => f` arm always fires) — meaning a NaN could silently become
//!   "the best" and, with logging on, get written into the IOH archive as
//!   the run's improvement/final row, permanently if it's the last eval.
//!   `EvalSession` does **not** let this happen: like `crates/stats`'s
//!   `check_finite` precedent (M2d-1), which rejects non-finite input at
//!   every FFI-facing entry point in that crate before any computation runs,
//!   `EvalSession::evaluate` is the rawest FFI-facing boundary in this crate
//!   (raw `Vec<f64>` rows straight from Python/R), so it rejects any
//!   non-finite coordinate explicitly, in the same pre-counter all-or-nothing
//!   pass as the dimension check, naming the offending row index. This is a
//!   deliberate *departure* from mirroring `Evaluator`/`BbobProblem` (which
//!   do not check), chosen because the FFI boundary is exactly where
//!   `check_finite`'s precedent says user input must be validated, not
//!   silently propagated.
//! - **`with_log` after evaluation has started**: calling [`Self::with_log`]
//!   once `evals_used() > 0` would silently produce an incomplete archive —
//!   the new `IohRunObserver` only sees evals from that point on, so earlier
//!   evals vanish from the `.dat` file and the first *logged* row gets
//!   marked as an "improvement" even if it wasn't the run's actual first
//!   improvement. `with_log` therefore errors instead of allowing this.

use crate::ioh::{IohLogger, IohRunObserver};
use crate::experiment::ExperimentError;
use sezgi_core::problem::{EvalObserver, Evaluator, Problem};
use sezgi_core::space::{BlockValues, Genotype};
use sezgi_problems::BbobProblem;
use std::path::Path;

const SUITE: &str = "sezgi-bbob";

/// The ask/tell evaluation core: owns the problem, the tamper-proof eval
/// counter, the running best, and (optionally) an IOH logger + run
/// observer. See the module doc for the reuse design and error semantics.
pub struct EvalSession {
    problem: BbobProblem,
    budget: u64,
    used: u64,                              // no public setter anywhere: see `evaluate`
    best: Option<(Vec<f64>, f64)>,
    log: Option<(IohLogger, IohRunObserver)>,
}

impl EvalSession {
    /// Builds a session over a fresh BBOB problem instance. No RNG of its
    /// own: candidate points are supplied entirely by the caller via
    /// [`Self::evaluate`].
    pub fn new_bbob(fid: u32, dim: usize, instance: u32, budget: u64) -> Result<Self, ExperimentError> {
        let problem = BbobProblem::new(fid, dim, instance)
            .map_err(|e| ExperimentError::Problem(e.to_string()))?;
        Ok(Self { problem, budget, used: 0, best: None, log: None })
    }

    /// Enables IOH logging for this session: one run (`instance` is the
    /// problem's own instance, `f_opt` its own known optimum) at this
    /// session's `budget`. `algo_name` labels the archive directory;
    /// `seed` is a caller-declared reproducibility label recorded in the
    /// meta only — the session itself never draws random numbers, so this
    /// is purely documentation of what the *caller's* RNG was seeded with.
    ///
    /// Errors if any evaluation has already happened (`evals_used() > 0`):
    /// attaching a logger mid-session would silently produce an incomplete
    /// archive (see module doc).
    pub fn with_log(mut self, log_dir: &Path, algo_name: &str, seed: u64) -> Result<Self, ExperimentError> {
        if self.used > 0 {
            return Err(ExperimentError::LogAfterEval { used: self.used });
        }
        let mut logger = IohLogger::new(
            log_dir, algo_name, SUITE,
            self.problem.fid(), self.problem.name(), self.problem.space().dim(),
        );
        let obs = logger.start_run_with(self.problem.instance, seed, self.problem.f_opt(), self.budget);
        self.log = Some((logger, obs));
        Ok(self)
    }

    /// Batch-evaluates `xs`. Counts every row on success; on any error
    /// (dimension mismatch, a non-finite coordinate, or budget overrun)
    /// nothing is counted and no row is evaluated — all-or-nothing,
    /// mirroring `Evaluator::evaluate`'s budget check and going further for
    /// the two checks `Evaluator`/`BbobProblem` don't make themselves (see
    /// module doc).
    pub fn evaluate(&mut self, xs: &[Vec<f64>]) -> Result<Vec<f64>, ExperimentError> {
        let dim = self.problem.space().dim();
        for (row, x) in xs.iter().enumerate() {
            if x.len() != dim {
                return Err(ExperimentError::DimensionMismatch { row, expected: dim, got: x.len() });
            }
            if x.iter().any(|v| !v.is_finite()) {
                return Err(ExperimentError::NonFiniteInput { row });
            }
        }

        let pop: Vec<Genotype> = xs.iter()
            .map(|x| Genotype { blocks: vec![BlockValues::Float(x.clone())] })
            .collect();

        // Short-lived Evaluator sized to the remaining budget: its own
        // all-or-nothing check IS the session's check (see module doc).
        let remaining = self.budget - self.used;
        let mut ev = Evaluator::new(&self.problem, remaining);
        let fs = ev.evaluate(&pop).map_err(|e| ExperimentError::BudgetExceeded {
            used: self.used, budget: self.budget, requested: e.requested,
        })?;

        for (x, &f) in xs.iter().zip(&fs) {
            self.used += 1;
            // Same tie rule as Evaluator::evaluate: update only on strict
            // improvement, tracking x too (necessarily new — see module
            // doc). x is finite by construction (checked above); f could
            // still be non-finite in principle if the objective itself
            // produces one from finite input, in which case this mirrors
            // Evaluator's own (unvalidated) tie behavior for f.
            let improved = !matches!(&self.best, Some((_, b)) if *b <= f);
            if improved { self.best = Some((x.clone(), f)); }
            let best_f = self.best.as_ref().unwrap().1;
            if let Some((_, obs)) = &mut self.log {
                obs.on_eval(self.used, f, best_f);
            }
        }
        Ok(fs)
    }

    pub fn evals_used(&self) -> u64 { self.used }
    pub fn budget(&self) -> u64 { self.budget }
    pub fn best(&self) -> Option<(&[f64], f64)> { self.best.as_ref().map(|(x, f)| (x.as_slice(), *f)) }
    pub fn f_opt(&self) -> f64 { self.problem.f_opt() }

    /// Flushes the IOH log, if logging was enabled. No-op (Ok) otherwise.
    /// Consumes the session — matches `IohLogger::finish`'s own consuming
    /// signature.
    pub fn finish(self) -> Result<(), ExperimentError> {
        if let Some((logger, _obs)) = self.log {
            logger.finish().map_err(|e| ExperimentError::IohWrite(e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(xs: &[f64]) -> Vec<f64> { xs.to_vec() }

    #[test]
    fn counting_across_two_batches() {
        let mut s = EvalSession::new_bbob(1, 3, 1, 100).unwrap();
        s.evaluate(&[row(&[0.0, 0.0, 0.0]), row(&[1.0, 1.0, 1.0]), row(&[2.0, 2.0, 2.0])]).unwrap();
        assert_eq!(s.evals_used(), 3);
        s.evaluate(&[row(&[3.0, 3.0, 3.0]), row(&[4.0, 4.0, 4.0])]).unwrap();
        assert_eq!(s.evals_used(), 5);
    }

    #[test]
    fn batch_crossing_budget_errs_and_does_not_count() {
        // Same all-or-nothing semantics as Evaluator::evaluate (verified by
        // reading crates/core/src/problem.rs: the budget check happens
        // before evaluate_batch is ever called, so a rejected batch leaves
        // `used` untouched).
        let mut s = EvalSession::new_bbob(1, 3, 1, 2).unwrap();
        let err = s.evaluate(&[row(&[0.0, 0.0, 0.0]), row(&[1.0, 1.0, 1.0]), row(&[2.0, 2.0, 2.0])]);
        assert!(err.is_err(), "batch of 3 requested against budget 2 must error");
        assert_eq!(s.evals_used(), 0, "a rejected batch must not count any rows");

        // A batch that fits exactly still succeeds and counts fully.
        s.evaluate(&[row(&[0.0, 0.0, 0.0]), row(&[1.0, 1.0, 1.0])]).unwrap();
        assert_eq!(s.evals_used(), 2);
    }

    #[test]
    fn best_tracking_matches_hand_tracked_min() {
        let mut s = EvalSession::new_bbob(1, 2, 1, 100).unwrap();
        let xs = [row(&[0.5, 0.5]), row(&[-1.0, 2.0]), row(&[3.0, -3.0]), row(&[0.1, 0.1])];
        let fs = s.evaluate(&xs).unwrap();

        let mut hand_best: Option<(usize, f64)> = None;
        for (i, &f) in fs.iter().enumerate() {
            hand_best = match hand_best {
                Some((_, b)) if b <= f => hand_best,
                _ => Some((i, f)),
            };
        }
        let (idx, want_f) = hand_best.unwrap();
        let (got_x, got_f) = s.best().unwrap();
        assert_eq!(got_f, want_f);
        assert_eq!(got_x, xs[idx].as_slice());
    }

    #[test]
    fn dimension_mismatch_names_the_row_index() {
        let mut s = EvalSession::new_bbob(1, 3, 1, 100).unwrap();
        let err = s.evaluate(&[row(&[0.0, 0.0, 0.0]), row(&[1.0, 1.0])]).unwrap_err();
        match err {
            ExperimentError::DimensionMismatch { row, expected, got } => {
                assert_eq!(row, 1);
                assert_eq!(expected, 3);
                assert_eq!(got, 2);
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
        assert_eq!(s.evals_used(), 0, "a rejected batch must not count any rows");
    }

    #[test]
    fn non_finite_coordinate_errors_and_does_not_count() {
        // Departure from mirroring Evaluator/BbobProblem (neither checks):
        // check_finite's precedent (crates/stats, M2d-1) says FFI-facing
        // input must be validated, not silently propagated — see module
        // doc. Covers NaN and +/-Inf, and every offending row is rejected
        // (all-or-nothing), including a row that itself has valid dimension.
        let mut s = EvalSession::new_bbob(1, 2, 1, 100).unwrap();
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = s.evaluate(&[row(&[0.0, 0.0]), row(&[bad, 0.0])]).unwrap_err();
            match err {
                ExperimentError::NonFiniteInput { row } => assert_eq!(row, 1),
                other => panic!("expected NonFiniteInput, got {other:?}"),
            }
            assert_eq!(s.evals_used(), 0, "a rejected batch must not count any rows, including the good ones before it");
        }
    }

    #[test]
    fn session_is_deterministic() {
        let xs = [row(&[0.5, -0.5]), row(&[1.5, 1.5]), row(&[-2.0, 0.25])];
        let run = || {
            let mut s = EvalSession::new_bbob(3, 2, 1, 100).unwrap();
            let fs = s.evaluate(&xs).unwrap();
            (fs, s.evals_used(), s.best().map(|(x, f)| (x.to_vec(), f)))
        };
        let a = run();
        let b = run();
        assert_eq!(a.0, b.0, "same input sequence must give bit-exact outputs");
        assert_eq!(a.1, b.1);
        assert_eq!(a.2, b.2);
    }

    #[test]
    fn finish_without_log_is_ok() {
        let s = EvalSession::new_bbob(1, 2, 1, 10).unwrap();
        assert!(s.finish().is_ok());
    }

    #[test]
    fn with_log_after_evaluation_errors() {
        // A logger attached after evals have already happened would
        // silently produce an incomplete archive (earlier evals invisible,
        // first logged row falsely marked as an improvement) — see module
        // doc. Must error instead, and leave the counter untouched.
        let tmp = tempfile::tempdir().unwrap();
        let mut s = EvalSession::new_bbob(1, 2, 1, 100).unwrap();
        s.evaluate(&[row(&[0.0, 0.0]), row(&[1.0, 1.0]), row(&[2.0, 2.0])]).unwrap();
        assert_eq!(s.evals_used(), 3);

        match s.with_log(tmp.path(), "test-algo", 42) {
            Err(ExperimentError::LogAfterEval { used }) => assert_eq!(used, 3),
            Err(other) => panic!("expected LogAfterEval, got {other:?}"),
            Ok(_) => panic!("with_log after evaluation must error"),
        }
    }

    #[test]
    fn with_log_before_any_evaluation_succeeds() {
        let tmp = tempfile::tempdir().unwrap();
        let s = EvalSession::new_bbob(1, 2, 1, 100).unwrap();
        assert!(s.with_log(tmp.path(), "test-algo", 42).is_ok());
    }

    /// The core deliverable: session-driven IOH output must be byte-identical
    /// to observer-driven output (the engine's own path, e.g. py-sezgi's
    /// `solve()`) for the same eval sequence.
    #[test]
    fn ioh_parity_session_vs_raw_observer() {
        let tmp = tempfile::tempdir().unwrap();
        let seq: Vec<Vec<f64>> = vec![
            row(&[0.5, -1.0, 2.0]),
            row(&[1.0, 1.0, 1.0]),
            row(&[-2.0, 0.5, 0.0]),
            row(&[0.0, 0.0, 0.0]),
            row(&[3.0, -3.0, 1.0]),
        ];

        // --- Path A: EvalSession-driven ---
        let dir_a = tmp.path().join("a");
        let mut s = EvalSession::new_bbob(1, 3, 1, 100).unwrap()
            .with_log(&dir_a, "test-algo", 42).unwrap();
        s.evaluate(&seq).unwrap();
        s.finish().unwrap();

        // --- Path B: raw IohLogger + start_run_with + Evaluator, driven by hand ---
        let dir_b = tmp.path().join("b");
        let p = BbobProblem::new(1, 3, 1).unwrap();
        let mut lg = IohLogger::new(&dir_b, "test-algo", SUITE, p.fid(), p.name(), 3);
        {
            let obs = lg.start_run_with(p.instance, 42, p.f_opt(), 100);
            let mut ev = Evaluator::new(&p, 100);
            ev.set_observer(Box::new(obs));
            let pop: Vec<Genotype> = seq.iter()
                .map(|x| Genotype { blocks: vec![BlockValues::Float(x.clone())] })
                .collect();
            ev.evaluate(&pop).unwrap();
        }
        lg.finish().unwrap();

        let dat_a = std::fs::read(dir_a.join("test-algo/data_f1_Sphere/IOHprofiler_f1_DIM3.dat")).unwrap();
        let dat_b = std::fs::read(dir_b.join("test-algo/data_f1_Sphere/IOHprofiler_f1_DIM3.dat")).unwrap();
        assert_eq!(dat_a, dat_b, ".dat files must be byte-identical");

        let meta_a = std::fs::read(dir_a.join("test-algo/IOHprofiler_f1_Sphere.json")).unwrap();
        let meta_b = std::fs::read(dir_b.join("test-algo/IOHprofiler_f1_Sphere.json")).unwrap();
        assert_eq!(meta_a, meta_b, "meta files must be byte-identical");
    }
}
