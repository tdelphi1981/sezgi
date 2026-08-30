//! sezgi-bias: structural/central bias scanning for metaheuristic algorithms
//! (spec §7 "Bias scanning" — M3-1). This crate hosts the BIAS-toolbox-style
//! null problems and the tests run against them; the statistical routines
//! themselves (KS, Anderson-Darling) live in `sezgi-stats` and are consumed
//! here, not reimplemented.
//!
//! # f0: the structural-bias null problem
//!
//! [`F0Random`] is the toolbox's reference null problem: every evaluation
//! returns an independent U(0,1) draw, uncorrelated with the queried point.
//! An algorithm with no structural bias should leave its final positions
//! uniformly distributed over the search domain when run repeatedly on f0;
//! systematic departures from uniformity are evidence of a bias in the
//! algorithm's own operators, not of the (nonexistent) fitness landscape.
//! See `f0` for the exact behavioral contract, domain, and RNG stream path.
//!
//! # Structural-bias scan (M3-1 Task 4)
//!
//! [`structural::structural_bias_scan`] runs an [`sezgi_core::spec::
//! AlgorithmSpec`] repeatedly on [`F0Random`] and tests its final positions
//! for departure from uniformity, per the BIAS-toolbox method -- see
//! `structural`'s module doc for the full, citation-backed provenance.
//!
//! # Central-bias scan (M3-1 Task 5)
//!
//! [`central::central_bias_scan`] runs an [`sezgi_core::spec::
//! AlgorithmSpec`] on paired centered/shifted BBOB conditions (via
//! [`sezgi_problems::BbobProblem::recentered`]) and tests whether its
//! performance gap differs between them, per Kudela's
//! center-bias-exploitation method -- see `central`'s module doc for the
//! full, citation-backed provenance.
//!
//! [`BiasVerdict`] is shared by every bias-scan flavour this crate hosts
//! (structural bias, central bias here; T6's own scan reuses it): its
//! `Evidence` variant's `detail` string MUST read as "evidence of
//! structural/center bias toward/..." -- language documenting what the DATA
//! shows, never an accusation ("algorithm X is biased") the data alone
//! cannot support.

pub mod central;
pub mod f0;
pub mod structural;

pub use central::{central_bias_scan, CentralBiasConfig, CentralBiasResult};
pub use f0::F0Random;
pub use structural::{structural_bias_scan, StructuralBiasConfig, StructuralBiasResult};

/// Verdict of a bias scan (structural, or any future scan this crate hosts).
/// "Evidence not accusation": [`BiasVerdict::Evidence`]'s `detail` says
/// "evidence of structural bias toward/..."; it must never claim an
/// algorithm categorically IS biased -- a finite scan can only report
/// evidence found (or not) in the data it collected.
#[derive(Debug, Clone, PartialEq)]
pub enum BiasVerdict {
    /// No statistically significant departure from uniformity was found.
    NoEvidence,
    /// A statistically significant departure was found; `detail` names what
    /// it was ("evidence of structural bias toward ...").
    Evidence { detail: String },
}

/// Errors produced by this crate's bias scans.
#[derive(Debug, thiserror::Error)]
pub enum BiasError {
    /// A statistics-crate computation rejected its input (should not happen
    /// for in-domain f0 output, but is not assumed away).
    #[error(transparent)]
    Stats(#[from] sezgi_stats::StatsError),
    /// `AlgorithmSpec` failed to validate against f0's own search space.
    #[error(transparent)]
    Spec(#[from] sezgi_core::spec::SpecError),
    /// An engine run failed (e.g. budget smaller than population size).
    #[error(transparent)]
    Engine(#[from] sezgi_core::engine::EngineError),
    /// A `(fid, dim, instance)` combination failed to construct a
    /// [`sezgi_problems::BbobProblem`] (e.g. `dim < 2`, or an unimplemented
    /// `fid`) -- used by [`central::central_bias_scan`].
    #[error(transparent)]
    Bbob(#[from] sezgi_problems::BbobError),
    /// A scan configuration was invalid on its own terms (e.g. `dim == 0`,
    /// too few runs, or a ragged `final_positions` matrix).
    #[error("invalid bias-scan configuration: {0}")]
    InvalidConfig(String),
}
