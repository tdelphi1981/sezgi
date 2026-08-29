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

pub mod pairwise;
pub mod ranks;
pub mod special;

pub use pairwise::{cliffs_delta, cliffs_magnitude, wilcoxon_signed_rank, WilcoxonResult};
pub use ranks::{friedman, hochberg, holm, nemenyi_cd, rank_matrix, FriedmanResult};
pub use special::{chi_square_sf, erf, gamma_p, ln_gamma, normal_cdf, normal_sf};

/// Errors produced by statistical routines in this crate.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum StatsError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
}
