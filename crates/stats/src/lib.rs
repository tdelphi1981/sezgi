//! Statistical routines for comparing metaheuristic algorithms across benchmark
//! problems.
//!
//! # Input convention
//!
//! Functions that accept a "results matrix" use the shape `&[Vec<f64>]`:
//! - **Rows** correspond to problems/datasets (`n` of them).
//! - **Columns** correspond to algorithms (`k` of them).
//! - Every row must have exactly `k` entries (a "ragged" matrix, where rows
//!   have differing lengths, is an error).
//! - Values are performance scores where **lower is better** (minimization).
//!   Rank 1 is assigned to the best (lowest) value in a row.

pub mod bayesian;
pub mod pairwise;
pub mod ranks;
pub mod report;
pub mod special;

pub use bayesian::{
    bayesian_plackett_luce, bayesian_signed_rank, plackett_luce, BayesPlackettLuceResult,
    BayesSignedRankResult, PlackettLuceResult,
};
pub use pairwise::{
    cliffs_delta, cliffs_magnitude, wilcoxon_signed_rank, WilcoxonMethod, WilcoxonResult,
};
pub use ranks::{friedman, hochberg, holm, nemenyi_cd, rank_matrix, FriedmanResult};
pub use report::{paper_package, summary_table_latex, AlgorithmSummary, PaperPackage};
pub use special::{chi_square_sf, erf, gamma_p, ln_gamma, normal_cdf, normal_sf};

/// Errors produced by statistical routines in this crate.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum StatsError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

/// Validates that every value in `values` is finite (not NaN, not
/// +/-infinity), returning `StatsError::InvalidInput` naming `func` and the
/// flat (0-based) index of the first offending value otherwise.
///
/// Called FIRST (before any other validation or computation) by every public
/// entry point in this crate that accepts `f64` data, so a non-finite input
/// is always rejected explicitly rather than propagating NaN through a
/// computation or silently producing a meaningless result. Matrix inputs are
/// flattened row-major before being passed in, so the reported index is the
/// flat index into that row-major flattening.
pub(crate) fn check_finite(func: &str, values: impl Iterator<Item = f64>) -> Result<(), StatsError> {
    for (i, v) in values.enumerate() {
        if !v.is_finite() {
            return Err(StatsError::InvalidInput(format!(
                "{func}: non-finite value at flat index {i}"
            )));
        }
    }
    Ok(())
}
