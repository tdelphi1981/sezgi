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

pub mod f0;

pub use f0::F0Random;
