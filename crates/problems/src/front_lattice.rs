//! Shared deterministic front-sampling lattice helpers, extracted verbatim
//! from `dtlz::Dtlz`'s own private associated functions in M3-8 T1 for
//! reuse by `wfg::Wfg` (whose `axis_grid`/`grid_r`/`cartesian_product` were
//! a byte-identical copy-paste of the DTLZ trio -- same names, same bodies,
//! same construction, per `wfg.rs`'s own "mirrors `crate::dtlz`'s own ...
//! helpers" comment above them prior to this extraction). This is a PURE
//! RELOCATION -- every function body below is byte-identical to the code it
//! was extracted from (only visibility changed: private `impl Dtlz`/`impl
//! Wfg` associated function to `pub(crate)` free function taking explicit
//! parameters instead of a `Self::` receiver -- these never used `self`
//! anyway, so no receiver was dropped). Both `dtlz.rs` and `wfg.rs` call
//! sites are re-pointed to `crate::front_lattice::*`; no formula, ordering,
//! or numeric behavior changed.
//!
//! Used by both DTLZ's and WFG's `pareto_front` lattice sampling: `m - 1`
//! free position parameters are sampled on a common axis grid (`axis_grid`,
//! `grid_r` choosing the per-axis resolution from the requested point
//! count), then combined via `cartesian_product` into every combination of
//! axis values across `dim` free dimensions -- each caller's own
//! `pareto_front` maps each combination through its problem-specific
//! closed-form front point.

/// `r` evenly spaced points in `[0.0, 1.0]` (inclusive of both ends when
/// `r > 1`); `r <= 1` returns the single point `[0.0]` (avoids a
/// division by zero in the `r - 1` denominator).
pub(crate) fn axis_grid(r: usize) -> Vec<f64> {
    if r <= 1 { vec![0.0] } else { (0..r).map(|i| i as f64 / (r - 1) as f64).collect() }
}

/// The per-axis resolution `r` that makes an `r^dim_free`-point lattice
/// come closest to (at least) `n` total points, without exceeding it by
/// more than one axis step: `ceil(n^(1/dim_free))`, floored at `1`.
pub(crate) fn grid_r(n: usize, dim_free: usize) -> usize {
    ((n.max(1) as f64).powf(1.0 / dim_free as f64)).ceil().max(1.0) as usize
}

/// Every combination of `axis`'s values across `dim` free dimensions, in
/// nested (outer-to-inner) axis order -- the full cartesian product
/// `axis^dim`, as `Vec<Vec<f64>>` (one inner `Vec` per combination, length
/// `dim`).
pub(crate) fn cartesian_product(axis: &[f64], dim: usize) -> Vec<Vec<f64>> {
    let mut out = vec![Vec::new()];
    for _ in 0..dim {
        let mut next = Vec::with_capacity(out.len() * axis.len());
        for combo in &out {
            for &v in axis {
                let mut c = combo.clone();
                c.push(v);
                next.push(c);
            }
        }
        out = next;
    }
    out
}
