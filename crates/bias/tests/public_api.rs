//! Integration test (a separate compilation unit, per Rust's own crate
//! boundary) proving `sezgi_bias::scan_from_positions` is callable from
//! OUTSIDE the crate -- the honest proof that Task 4's `pub(crate)` -> `pub`
//! visibility change (and its re-export from `crates/bias/src/lib.rs`) is
//! real, not merely a compile-checked claim inside the crate's own
//! `#[cfg(test)]` module (which can see private items regardless).

use sezgi_bias::{scan_from_positions, BiasVerdict};

#[test]
fn scan_from_positions_is_public_api() {
    // 30 runs, dim 2, drawn from a fixed uniform-ish grid: deterministic,
    // no engine/problem machinery needed -- this is exactly the
    // statistics-only entry point Task 4's bridge (`sezgi.bias.
    // structural_positions`) calls from outside this crate (via py-sezgi).
    let runs = 30;
    let final_positions: Vec<Vec<f64>> = (0..runs)
        .map(|i| {
            let t = i as f64 / (runs as f64 - 1.0);
            vec![t, 1.0 - t]
        })
        .collect();

    let r = scan_from_positions(final_positions, 2).expect("valid input must not error");
    assert_eq!(r.per_dim_ks.len(), 2);
    assert_eq!(r.per_dim_ad.len(), 2);
    assert!(matches!(r.verdict, BiasVerdict::NoEvidence | BiasVerdict::Evidence { .. }));
}

#[test]
fn scan_from_positions_still_validates_from_outside_the_crate() {
    let err = scan_from_positions(vec![vec![0.5]; 3], 1).unwrap_err();
    assert!(err.to_string().contains("at least"));
}
