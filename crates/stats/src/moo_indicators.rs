//! Multi-objective quality indicators: exact 2-objective hypervolume and
//! Inverted Generational Distance (IGD).
//!
//! # Input convention
//!
//! Both functions take a "front" as `&[Vec<f64>]`: each row is one
//! objective vector (minimization -- lower is every-coordinate better).
//! `hypervolume_2d` requires every row to have exactly 2 objectives; `igd`
//! accepts any (consistent) number of objectives.
//!
//! # Provenance
//!
//! ## Hypervolume (`hypervolume_2d`): the S-metric definition
//!
//! Source: Zitzler, E., & Thiele, L. (1999). "Multiobjective Evolutionary
//! Algorithms: A Comparative Case Study and the Strength Pareto Approach."
//! *IEEE Transactions on Evolutionary Computation*, 3(4), 257-271. Section
//! III-B-1 (p. 260), "Size of the space covered," quoted verbatim
//! (converting the paper's typeset notation to prose):
//!
//! ```text
//! Let X' = (x1, x2, ..., xk) be a set of k decision vectors. The function
//! S(X') gives the volume enclosed by the union of the polytopes
//! p1, p2, ..., pk, where each pi is formed by the intersections of the
//! following hyperplanes arising out of xi, along with the axes: for each
//! axis in the objective space, there exists a hyperplane perpendicular to
//! the axis and passing through the point (f1(xi), f2(xi), ..., fn(xi)).
//! In the two-dimensional (2-D) case, each pi represents a rectangle
//! defined by the points (0,0) and (f1(xi), f2(xi)).
//! ```
//!
//! This module implements the now-standard reference-point variant of the
//! same S-metric: instead of anchoring every rectangle at the coordinate
//! origin `(0,0)` (as in the paper's maximization/profit setting), each
//! rectangle is anchored at a caller-supplied `ref_point` and spans between
//! it and the (minimization) objective vector, i.e. the corners
//! `(f1(xi), f2(xi))` and `ref_point`. `S` (the union-of-rectangles volume)
//! is then computed exactly, without double-counting overlaps, by an
//! elementary sort-and-sweep derived directly from the definition above:
//! discard dominated points (their rectangle is a subset of a surviving
//! rectangle's), sort the remainder ascending by `f1`, and sum
//! non-overlapping strips `(next_f1 - f1_i) * (ref2 - f2_i)` (the last
//! strip's `next_f1` is `ref_point[0]`). **Convention (this
//! implementation)**: a point that does not strictly dominate `ref_point`
//! (`f1 >= ref_point[0]` or `f2 >= ref_point[1]`) contributes an
//! empty/degenerate rectangle and is dropped before the sweep -- it
//! contributes nothing to `S`.
//!
//! General-M (M > 2 objectives) hypervolume requires the WFG algorithm (or
//! similar); that is explicitly OUT of scope here, which implements only
//! the exact 2-objective case.
//!
//! ## IGD (`igd`): pinned definition and variant
//!
//! Source: Ishibuchi, H., Masuda, H., Tanigaki, Y., & Nojima, Y. (2015).
//! "Modified Distance Calculation in Generational Distance and Inverted
//! Generational Distance." *Proceedings of the 8th International
//! Conference on Evolutionary Multi-Criterion Optimization (EMO 2015)*,
//! Part I, 110-125. Eq. (12), quoted verbatim (converting the paper's
//! typeset notation to prose):
//!
//! ```text
//! Inverted Generational Distance:
//!   IGD(A) = (1/|Z|) * ( sum_{j=1}^{|Z|} d^_j^p )^(1/p)
//! where d^_j is the Euclidean distance from z_j to its nearest objective
//! vector in A.
//! ```
//!
//! and, on `p` (same section): "In this paper, we always specify the value
//! of p as p = 1. ... the IGD with p = 1 is the average Euclidean distance
//! from each reference point to its nearest objective vector." **PINNED
//! variant**: this module implements exactly `p = 1` -- the plain
//! arithmetic mean of per-reference-point nearest-neighbor Euclidean
//! distances: `IGD(A) = (1/|Z|) * sum_{z in Z} min_{a in A} d(z,a)`. This is
//! NOT the general power-mean form, nor Schutze et al.'s `IGD_p` (which
//! moves the `1/|Z|` factor inside the outer power); per the same section,
//! that variant coincides with the plain-mean `IGD` only at `p = 1`. The
//! Ishibuchi et al. (2015) paper itself attributes the term "inverted
//! generational distance" to Coello & Sierra (2004) and Sierra & Coello
//! (2004).
//!
//! # Determinism
//!
//! Both functions are pure and deterministic (no RNG involved); repeat
//! calls on identical inputs are bit-identical, asserted directly in tests.

use crate::{check_finite, StatsError};

/// Exact 2-objective hypervolume (minimization) of `front` with respect to
/// `ref_point`. See the module doc for the pinned S-metric definition
/// (Zitzler & Thiele, 1999) and the reference-point sort-and-sweep
/// computation derived from it.
///
/// Points not strictly dominating `ref_point` contribute nothing (see
/// module doc, "Convention").
///
/// # Errors
/// [`StatsError::InvalidInput`] if any value (in `front` or `ref_point`) is
/// non-finite, if `front` is empty, or if any row of `front` does not have
/// exactly 2 objectives.
pub fn hypervolume_2d(front: &[Vec<f64>], ref_point: &[f64; 2]) -> Result<f64, StatsError> {
    check_finite("hypervolume_2d", front.iter().flatten().copied())?;
    check_finite("hypervolume_2d", ref_point.iter().copied())?;

    if front.is_empty() {
        return Err(StatsError::InvalidInput(
            "hypervolume_2d: front is empty".to_string(),
        ));
    }

    for (i, row) in front.iter().enumerate() {
        if row.len() != 2 {
            return Err(StatsError::InvalidInput(format!(
                "hypervolume_2d: front row {i} has {} objectives, expected exactly 2",
                row.len()
            )));
        }
    }

    let ref1 = ref_point[0];
    let ref2 = ref_point[1];

    // Keep only points that strictly dominate ref_point; everything else
    // contributes an empty/degenerate rectangle (module doc "Convention").
    let mut candidates: Vec<(f64, f64)> = front
        .iter()
        .filter_map(|p| {
            let (f1, f2) = (p[0], p[1]);
            if f1 < ref1 && f2 < ref2 {
                Some((f1, f2))
            } else {
                None
            }
        })
        .collect();

    candidates.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .expect("non-finite already rejected above")
            .then(
                a.1.partial_cmp(&b.1)
                    .expect("non-finite already rejected above"),
            )
    });

    // Sweep ascending by f1, keeping only points that strictly improve f2
    // (non-dominated front); this also drops exact duplicates and any
    // point dominated by an earlier (smaller-f1) survivor.
    let mut kept: Vec<(f64, f64)> = Vec::with_capacity(candidates.len());
    let mut best_f2 = f64::INFINITY;
    for (f1, f2) in candidates {
        if f2 < best_f2 {
            kept.push((f1, f2));
            best_f2 = f2;
        }
    }

    let mut area = 0.0_f64;
    for (i, &(f1, f2)) in kept.iter().enumerate() {
        let next_f1 = if i + 1 < kept.len() {
            kept[i + 1].0
        } else {
            ref1
        };
        area += (next_f1 - f1) * (ref2 - f2);
    }

    Ok(area)
}

/// Inverted Generational Distance: the mean, over reference-front points,
/// of the Euclidean distance to the nearest point in `front`. Any (equal,
/// consistent) number of objectives. See the module doc for the pinned
/// definition (Ishibuchi et al., 2015, eq. 12, `p = 1`).
///
/// # Errors
/// [`StatsError::InvalidInput`] if any value is non-finite, if `front` or
/// `reference_front` is empty, or if the rows of `front` and
/// `reference_front` are not all of the same (nonzero) length.
pub fn igd(front: &[Vec<f64>], reference_front: &[Vec<f64>]) -> Result<f64, StatsError> {
    check_finite("igd", front.iter().flatten().copied())?;
    check_finite("igd", reference_front.iter().flatten().copied())?;

    if front.is_empty() {
        return Err(StatsError::InvalidInput("igd: front is empty".to_string()));
    }
    if reference_front.is_empty() {
        return Err(StatsError::InvalidInput(
            "igd: reference_front is empty".to_string(),
        ));
    }

    let d = front[0].len();
    if d == 0 {
        return Err(StatsError::InvalidInput(
            "igd: front rows must have at least 1 objective, got 0".to_string(),
        ));
    }
    for (i, row) in front.iter().enumerate() {
        if row.len() != d {
            return Err(StatsError::InvalidInput(format!(
                "igd: ragged front, row 0 has {d} objectives, row {i} has {}",
                row.len()
            )));
        }
    }
    for (i, row) in reference_front.iter().enumerate() {
        if row.len() != d {
            return Err(StatsError::InvalidInput(format!(
                "igd: reference_front row {i} has {} objectives, expected {d} (matching front)",
                row.len()
            )));
        }
    }

    let mut sum = 0.0_f64;
    for z in reference_front {
        let mut min_dist = f64::INFINITY;
        for a in front {
            let dist_sq: f64 = z.iter().zip(a.iter()).map(|(zi, ai)| (zi - ai).powi(2)).sum();
            let dist = dist_sq.sqrt();
            if dist < min_dist {
                min_dist = dist;
            }
        }
        sum += min_dist;
    }

    Ok(sum / reference_front.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Hand fixture: 3-point 2D non-dominated front, full rectangle
    // arithmetic.
    //
    // front = [(0.25, 0.75), (0.5, 0.5), (0.75, 0.25)], ref_point = (1, 1).
    // All three points strictly dominate ref_point and none dominates
    // another (each is better in one coordinate, worse in the other), so
    // all three survive the sweep, already sorted ascending by f1.
    //
    // i=1 (0.25, 0.75): width = 0.5  - 0.25 = 0.25, height = 1.0 - 0.75 = 0.25
    //                    area = 0.25 * 0.25 = 0.0625
    // i=2 (0.5, 0.5):    width = 0.75 - 0.5  = 0.25, height = 1.0 - 0.5  = 0.5
    //                    area = 0.25 * 0.5  = 0.125
    // i=3 (0.75, 0.25):  width = 1.0  - 0.75 = 0.25, height = 1.0 - 0.25 = 0.75
    //                    area = 0.25 * 0.75 = 0.1875
    //
    // total = 0.0625 + 0.125 + 0.1875 = 0.375
    //
    // Every value above (0.25, 0.5, 0.75, 0.0625, 0.125, 0.1875, 0.375) is
    // exactly representable in f64, so the assertion below is exact
    // equality, not a tolerance check.
    #[test]
    fn hypervolume_2d_hand_fixture_exact() {
        let front = vec![vec![0.25, 0.75], vec![0.5, 0.5], vec![0.75, 0.25]];
        let ref_point = [1.0, 1.0];

        let hv = hypervolume_2d(&front, &ref_point).expect("valid fixture");
        assert_eq!(hv, 0.375);
    }

    // Dominated point contributes nothing: (0.6, 0.6) is dominated by
    // (0.5, 0.5) (0.5 <= 0.6 and 0.5 <= 0.6), so it is dropped by the sweep
    // and the hypervolume is unchanged from the base fixture (0.375).
    #[test]
    fn hypervolume_2d_dominated_point_contributes_nothing() {
        let front = vec![
            vec![0.25, 0.75],
            vec![0.5, 0.5],
            vec![0.75, 0.25],
            vec![0.6, 0.6],
        ];
        let ref_point = [1.0, 1.0];

        let hv = hypervolume_2d(&front, &ref_point).expect("valid fixture");
        assert_eq!(hv, 0.375);
    }

    // Point outside the ref box contributes nothing: (1.5, 0.1) has
    // f1 = 1.5 > ref_point[0] = 1.0, so it does not strictly dominate
    // ref_point and is filtered out before the sweep; hypervolume is
    // unchanged from the base fixture (0.375).
    #[test]
    fn hypervolume_2d_point_outside_ref_box_contributes_nothing() {
        let front = vec![
            vec![0.25, 0.75],
            vec![0.5, 0.5],
            vec![0.75, 0.25],
            vec![1.5, 0.1],
        ];
        let ref_point = [1.0, 1.0];

        let hv = hypervolume_2d(&front, &ref_point).expect("valid fixture");
        assert_eq!(hv, 0.375);
    }

    // Duplicate points don't double-count: (0.5, 0.5) appears twice; the
    // sweep's strict "f2 < best_f2" survivor test drops the second copy
    // (its f2 does not strictly improve on the first), so the hypervolume
    // is unchanged from the base fixture (0.375).
    #[test]
    fn hypervolume_2d_duplicate_points_do_not_double_count() {
        let front = vec![
            vec![0.25, 0.75],
            vec![0.5, 0.5],
            vec![0.75, 0.25],
            vec![0.5, 0.5],
        ];
        let ref_point = [1.0, 1.0];

        let hv = hypervolume_2d(&front, &ref_point).expect("valid fixture");
        assert_eq!(hv, 0.375);
    }

    // Ref-point boundary-equality convention guard: a front point with
    // f1 == ref_point[0] exactly, and one with f2 == ref_point[1] exactly,
    // do NOT strictly dominate ref_point (the code's filter is `<`, not
    // `<=`), so both contribute zero area. Exact-representable values
    // (0.5, 1.0) so this is checked by exact equality, not tolerance.
    // Guards against a future "fix" of `<` to `<=` silently changing the
    // pinned convention.
    #[test]
    fn hypervolume_2d_boundary_equality_contributes_nothing() {
        let front = vec![
            vec![0.25, 0.75],
            vec![0.5, 0.5],
            vec![0.75, 0.25],
            vec![1.0, 0.5], // f1 == ref_point[0] exactly
            vec![0.5, 1.0], // f2 == ref_point[1] exactly
        ];
        let ref_point = [1.0, 1.0];

        let hv = hypervolume_2d(&front, &ref_point).expect("valid fixture");
        assert_eq!(hv, 0.375);
    }

    // Same-f1 tie that is NOT an exact duplicate: front =
    // [(1,3), (2,2), (2,1), (3,0.5)], ref_point = (4,4). (2,2) and (2,1)
    // share f1 = 2 but (2,2) is weakly dominated by (2,1) (1 <= 2, 2 <= 2,
    // not equal), so (2,2) must contribute nothing; the running-min sweep
    // (sorted ascending by (f1, f2)) keeps (1,3), then (2,1) (f2=1 < 3,
    // survives), then skips (2,2) (f2=2 is not < best_f2=1), then keeps
    // (3,0.5) (f2=0.5 < 1).
    //
    // Kept, ascending f1: (1,3), (2,1), (3,0.5).
    //
    // i=1 (1,3):   width = 2 - 1 = 1,   height = 4 - 3   = 1   -> area = 1
    // i=2 (2,1):   width = 3 - 2 = 1,   height = 4 - 1   = 3   -> area = 3
    // i=3 (3,0.5): width = 4 - 3 = 1,   height = 4 - 0.5 = 3.5 -> area = 3.5
    //
    // total = 1 + 3 + 3.5 = 7.5
    #[test]
    fn hypervolume_2d_same_f1_tie_not_double_counted() {
        let front = vec![
            vec![1.0, 3.0],
            vec![2.0, 2.0],
            vec![2.0, 1.0],
            vec![3.0, 0.5],
        ];
        let ref_point = [4.0, 4.0];

        let hv = hypervolume_2d(&front, &ref_point).expect("valid fixture");
        assert_eq!(hv, 7.5);
    }

    #[test]
    fn hypervolume_2d_is_deterministic() {
        let front = vec![vec![0.25, 0.75], vec![0.5, 0.5], vec![0.75, 0.25]];
        let ref_point = [1.0, 1.0];

        let hv1 = hypervolume_2d(&front, &ref_point).unwrap();
        let hv2 = hypervolume_2d(&front, &ref_point).unwrap();
        assert_eq!(hv1.to_bits(), hv2.to_bits());
    }

    #[test]
    fn hypervolume_2d_errors_on_empty_front() {
        let front: Vec<Vec<f64>> = vec![];
        let ref_point = [1.0, 1.0];
        let err = hypervolume_2d(&front, &ref_point).expect_err("empty front must be rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("hypervolume_2d"), "message: {msg}");
        assert!(msg.contains("empty"), "message: {msg}");
    }

    #[test]
    fn hypervolume_2d_errors_on_non_2d_row() {
        let front = vec![vec![0.25, 0.75, 0.1], vec![0.5, 0.5]];
        let ref_point = [1.0, 1.0];
        let err = hypervolume_2d(&front, &ref_point).expect_err("non-2D row must be rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("hypervolume_2d"), "message: {msg}");
        assert!(msg.contains("expected exactly 2"), "message: {msg}");
    }

    #[test]
    fn hypervolume_2d_errors_on_non_finite_front_value() {
        let front = vec![vec![0.25, f64::NAN], vec![0.5, 0.5]];
        let ref_point = [1.0, 1.0];
        let err = hypervolume_2d(&front, &ref_point).expect_err("NaN must be rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("hypervolume_2d"), "message: {msg}");
        assert!(msg.contains("non-finite"), "message: {msg}");
    }

    #[test]
    fn hypervolume_2d_errors_on_non_finite_ref_point() {
        let front = vec![vec![0.25, 0.75]];
        let ref_point = [1.0, f64::INFINITY];
        let err = hypervolume_2d(&front, &ref_point).expect_err("non-finite ref_point rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("hypervolume_2d"), "message: {msg}");
        assert!(msg.contains("non-finite"), "message: {msg}");
    }

    // Hand fixture: 2 reference points, 2 front points, distances computed
    // by hand.
    //
    // reference_front = [(0,0), (4,0)], front = [(1,0), (3,0)].
    //
    // z=(0,0): d to (1,0) = sqrt((0-1)^2 + (0-0)^2) = sqrt(1) = 1.0
    //          d to (3,0) = sqrt((0-3)^2 + (0-0)^2) = sqrt(9) = 3.0
    //          min = 1.0  (nearest front point is (1,0))
    // z=(4,0): d to (1,0) = sqrt((4-1)^2 + (0-0)^2) = sqrt(9) = 3.0
    //          d to (3,0) = sqrt((4-3)^2 + (0-0)^2) = sqrt(1) = 1.0
    //          min = 1.0  (nearest front point is (3,0), different from z=(0,0)'s)
    //
    // IGD = (1.0 + 1.0) / 2 = 1.0, exact (both nearest distances are
    // sqrt(1) = 1.0 exactly).
    #[test]
    fn igd_hand_fixture_exact() {
        let front = vec![vec![1.0, 0.0], vec![3.0, 0.0]];
        let reference_front = vec![vec![0.0, 0.0], vec![4.0, 0.0]];

        let value = igd(&front, &reference_front).expect("valid fixture");
        assert_eq!(value, 1.0);
    }

    #[test]
    fn igd_is_zero_when_front_covers_reference_front() {
        let reference_front = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 0.5]];
        // front is a superset of reference_front (order doesn't matter).
        let front = vec![
            vec![5.0, 5.0],
            vec![2.0, 0.5],
            vec![0.0, 0.0],
            vec![1.0, 1.0],
        ];

        let value = igd(&front, &reference_front).expect("valid fixture");
        assert_eq!(value, 0.0);
    }

    #[test]
    fn igd_is_deterministic() {
        let front = vec![vec![1.0, 0.0], vec![3.0, 0.0]];
        let reference_front = vec![vec![0.0, 0.0], vec![4.0, 0.0]];

        let v1 = igd(&front, &reference_front).unwrap();
        let v2 = igd(&front, &reference_front).unwrap();
        assert_eq!(v1.to_bits(), v2.to_bits());
    }

    #[test]
    fn igd_errors_on_empty_front() {
        let front: Vec<Vec<f64>> = vec![];
        let reference_front = vec![vec![0.0, 0.0]];
        let err = igd(&front, &reference_front).expect_err("empty front must be rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("igd"), "message: {msg}");
        assert!(msg.contains("front is empty"), "message: {msg}");
    }

    #[test]
    fn igd_errors_on_empty_reference_front() {
        let front = vec![vec![0.0, 0.0]];
        let reference_front: Vec<Vec<f64>> = vec![];
        let err = igd(&front, &reference_front).expect_err("empty reference_front rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("igd"), "message: {msg}");
        assert!(msg.contains("reference_front is empty"), "message: {msg}");
    }

    #[test]
    fn igd_errors_on_dimension_mismatch() {
        let front = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let reference_front = vec![vec![0.0, 0.0, 0.0]];
        let err = igd(&front, &reference_front).expect_err("dimension mismatch rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("igd"), "message: {msg}");
        assert!(msg.contains("expected 2"), "message: {msg}");
    }

    #[test]
    fn igd_errors_on_ragged_front() {
        let front = vec![vec![0.0, 0.0], vec![1.0, 1.0, 1.0]];
        let reference_front = vec![vec![0.0, 0.0]];
        let err = igd(&front, &reference_front).expect_err("ragged front rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("igd"), "message: {msg}");
        assert!(msg.contains("ragged"), "message: {msg}");
    }

    #[test]
    fn igd_errors_on_non_finite_value() {
        let front = vec![vec![0.0, 0.0], vec![f64::NAN, 1.0]];
        let reference_front = vec![vec![0.0, 0.0]];
        let err = igd(&front, &reference_front).expect_err("NaN must be rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("igd"), "message: {msg}");
        assert!(msg.contains("non-finite"), "message: {msg}");
    }
}
