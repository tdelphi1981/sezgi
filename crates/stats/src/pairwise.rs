//! Pairwise comparison statistics for two paired samples (e.g. one
//! algorithm's per-problem results vs. another's): the Wilcoxon
//! signed-rank test and Cliff's delta effect size.
//!
//! See the crate-level documentation in `lib.rs` for the general
//! results-matrix convention used elsewhere in this crate; the functions
//! here instead take two same-length slices representing a single paired
//! comparison (e.g. two columns of a results matrix).

use crate::special::normal_cdf;
use crate::StatsError;

/// Result of a Wilcoxon signed-rank test ([`wilcoxon_signed_rank`]).
#[derive(Debug, Clone, PartialEq)]
pub struct WilcoxonResult {
    /// W = min(W+, W-), the smaller of the two signed rank sums (over the
    /// effective/nonzero pairs).
    pub w_statistic: f64,
    /// Normal-approximation z score (continuity-corrected), `z <= 0` since
    /// `w_statistic` is the min of the two rank sums.
    pub z: f64,
    /// Two-sided p-value, `min(1, 2 * normal_cdf(z))`.
    pub p_value: f64,
    /// Number of pairs with a nonzero difference (`n - t0`), the effective
    /// sample size the normal approximation is computed over.
    pub n_effective: usize,
}

/// Computes average-tie ranks (1-based) of `values`, ascending, together
/// with the `(value, group_size)` of each distinct-value tie group in
/// ascending order. Ties share the average of the ranks they span.
///
/// Shared by [`wilcoxon_signed_rank`] to rank `|d|` (Pratt's zero-handling
/// requires ranking the zeros along with everything else before dropping
/// them, see below).
fn rank_with_tie_groups(values: &[f64]) -> (Vec<f64>, Vec<(f64, usize)>) {
    let k = values.len();
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .expect("NaN in wilcoxon_signed_rank input")
    });

    let mut ranks = vec![0.0_f64; k];
    let mut groups = Vec::new();
    let mut i = 0;
    while i < k {
        let mut j = i;
        while j + 1 < k && values[order[j + 1]] == values[order[i]] {
            j += 1;
        }
        // Positions order[i..=j] are tied; they share ranks (i+1)..=(j+1)
        // (1-based), whose average is ((i+1)+(j+1))/2.
        let avg_rank = ((i + 1) + (j + 1)) as f64 / 2.0;
        for &pos in &order[i..=j] {
            ranks[pos] = avg_rank;
        }
        groups.push((values[order[i]], j - i + 1));
        i = j + 1;
    }
    (ranks, groups)
}

/// Wilcoxon signed-rank test for two paired samples `a` and `b`
/// (`d_i = a_i - b_i`), using the normal approximation with **Pratt's
/// (1959) zero-handling and tie correction**.
///
/// # Zero-handling (pinned choice: Pratt, not Wilcoxon)
///
/// The classic Wilcoxon procedure *discards* zero-difference pairs before
/// ranking. Pratt's variant instead ranks `|d_i|` over **all** `n` pairs
/// (zeros included, tied with each other at the bottom of the ranking via
/// the usual average-tie rule), and only *afterward* drops the ranks
/// belonging to the zero-`d` pairs from the W+ / W- sums. This changes the
/// ranks assigned to the nonzero pairs (a zero-tie group "uses up" some of
/// the lowest ranks) and is the semantics of SciPy's
/// `wilcoxon(a, b, zero_method="pratt")`. This is the pinned choice for
/// this crate; document any future divergence.
///
/// # Statistic
///
/// Let `n = a.len()`, `d_i = a_i - b_i`, `t0` = number of zero `d_i`,
/// `n_effective = n - t0`. Rank `|d_1|, ..., |d_n|` (all `n` values,
/// average-tie) to get ranks `R_1, ..., R_n`. Then, summing only over
/// pairs with nonzero `d_i`:
///
/// ```text
/// W+ = sum of R_i for d_i > 0
/// W- = sum of R_i for d_i < 0
/// W  = min(W+, W-)
/// ```
///
/// Normal approximation (Pratt 1959; zeros excluded from the sums but
/// still accounted for in the moments since they consumed low ranks):
///
/// ```text
/// mu     = [ n(n+1) - t0(t0+1) ] / 4
/// sigma^2 = [ n(n+1)(2n+1) - t0(t0+1)(2t0+1) ] / 24 - sum(t^3 - t) / 48
/// ```
///
/// where the tie-correction sum `sum(t^3 - t)` runs over the tie groups
/// among the **nonzero** `|d_i|` values only (`t` = size of each such
/// group; the all-zero group, if any, is already accounted for via `t0`
/// in the leading terms and must not be double-counted here).
///
/// Continuity-corrected z (toward the mean, `W` being the smaller side so
/// `z <= 0`):
///
/// ```text
/// z = (W - mu + 0.5) / sigma
/// ```
///
/// Two-sided p-value: `p = min(1, 2 * normal_cdf(z))`.
///
/// # Errors
/// Returns `StatsError::InvalidInput` if `a.len() != b.len()`, or if
/// `n_effective < 5` (below this, the normal approximation is unreliable
/// and the small-sample exact tables are not implemented in this crate —
/// deferred; see M2c task brief).
///
/// # Reference
/// Pratt, J. W. (1959). "Remarks on Zeros and Ties in the Wilcoxon Signed
/// Rank Procedures." *Journal of the American Statistical Association*,
/// 54(287), 655-667.
pub fn wilcoxon_signed_rank(a: &[f64], b: &[f64]) -> Result<WilcoxonResult, StatsError> {
    if a.len() != b.len() {
        return Err(StatsError::InvalidInput(format!(
            "wilcoxon_signed_rank requires paired samples of equal length, got {} and {}",
            a.len(),
            b.len()
        )));
    }

    let n = a.len();
    let d: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
    let abs_d: Vec<f64> = d.iter().map(|x| x.abs()).collect();

    let (ranks, groups) = rank_with_tie_groups(&abs_d);

    let t0 = d.iter().filter(|&&x| x == 0.0).count();
    let n_effective = n - t0;
    if n_effective < 5 {
        return Err(StatsError::InvalidInput(format!(
            "wilcoxon_signed_rank requires at least 5 nonzero-difference pairs (n_effective), got {}",
            n_effective
        )));
    }

    let mut w_pos = 0.0_f64;
    let mut w_neg = 0.0_f64;
    for i in 0..n {
        if d[i] > 0.0 {
            w_pos += ranks[i];
        } else if d[i] < 0.0 {
            w_neg += ranks[i];
        }
    }
    let w_statistic = w_pos.min(w_neg);

    let n_f = n as f64;
    let t0_f = t0 as f64;

    let mu = (n_f * (n_f + 1.0) - t0_f * (t0_f + 1.0)) / 4.0;

    let tie_sum: f64 = groups
        .iter()
        .filter(|&&(value, _)| value != 0.0)
        .map(|&(_, t)| {
            let t_f = t as f64;
            t_f.powi(3) - t_f
        })
        .sum();

    let sigma2 = (n_f * (n_f + 1.0) * (2.0 * n_f + 1.0) - t0_f * (t0_f + 1.0) * (2.0 * t0_f + 1.0))
        / 24.0
        - tie_sum / 48.0;
    let sigma = sigma2.sqrt();

    let z = (w_statistic - mu + 0.5) / sigma;
    let p_value = (2.0 * normal_cdf(z)).min(1.0);

    Ok(WilcoxonResult {
        w_statistic,
        z,
        p_value,
        n_effective,
    })
}

/// Cliff's delta effect size for two independent (unpaired) samples `a`
/// and `b`:
///
/// ```text
/// delta = ( #{a_i > b_j} - #{a_i < b_j} ) / (n * m)
/// ```
///
/// summed over all `n * m` pairs `(a_i, b_j)`. Ties (`a_i == b_j`)
/// contribute to neither count. `delta` ranges over `[-1, 1]`:
/// `delta = -1` means every element of `a` is smaller than every element
/// of `b`; `delta = +1` the reverse; `delta = 0` indicates no stochastic
/// dominance (e.g. identical samples).
///
/// Direct O(n*m) computation (no rank-based shortcut), matching the
/// definition exactly.
pub fn cliffs_delta(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let m = b.len();

    let mut greater = 0_i64;
    let mut less = 0_i64;
    for &ai in a {
        for &bj in b {
            if ai > bj {
                greater += 1;
            } else if ai < bj {
                less += 1;
            }
        }
    }

    (greater - less) as f64 / (n as f64 * m as f64)
}

/// Qualitative magnitude label for a Cliff's delta value, using Romano et
/// al. (2006) thresholds on `|delta|`:
///
/// - `|delta| < 0.147` -> "negligible"
/// - `|delta| < 0.33`  -> "small"
/// - `|delta| < 0.474` -> "medium"
/// - otherwise         -> "large"
///
/// **Boundary convention (pinned):** each threshold is a strict `<`, so a
/// value landing exactly *on* a boundary is classified into the
/// **higher** magnitude bin (e.g. `|delta| == 0.147` is "small", not
/// "negligible"). Document any future divergence from this convention.
///
/// # Reference
/// Romano, J., Kromrey, J. D., Coraggio, J., & Skowronek, J. (2006).
/// "Appropriate statistics for ordinal level data: Should we really be
/// using t-test and Cohen's d for evaluating group differences on the
/// NSSE and other surveys?" Annual meeting of the Florida Association of
/// Institutional Research.
pub fn cliffs_magnitude(delta: f64) -> &'static str {
    let ad = delta.abs();
    if ad < 0.147 {
        "negligible"
    } else if ad < 0.33 {
        "small"
    } else if ad < 0.474 {
        "medium"
    } else {
        "large"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // No-ties, no-zeros n=10 fixture.
    //
    // a = [101, 98, 103, 96, 105, 94, 107, 92, 109, 90]
    // b = [100]*10
    // d = a - b = [1, -2, 3, -4, 5, -6, 7, -8, 9, -10]
    //
    // |d| = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] -- already sorted ascending
    // with no ties, so rank(|d_i|) = |d_i| exactly (1-based).
    //
    // t0 = 0 (no zeros), n = 10, n_effective = 10.
    //
    // W+ (positive d: 1, 3, 5, 7, 9 -> ranks 1, 3, 5, 7, 9):
    //   W+ = 1 + 3 + 5 + 7 + 9 = 25
    // W- (negative d: -2, -4, -6, -8, -10 -> ranks 2, 4, 6, 8, 10):
    //   W- = 2 + 4 + 6 + 8 + 10 = 30
    // Check: W+ + W- = 55 = n(n+1)/2 = 10*11/2 = 55  [OK]
    // W = min(25, 30) = 25
    //
    // mu = n(n+1)/4 = 10*11/4 = 27.5
    // sigma^2 = n(n+1)(2n+1)/24 = 10*11*21/24 = 2310/24 = 96.25
    // sigma = sqrt(96.25) = 9.810708435174291...
    //
    // z = (W - mu + 0.5) / sigma = (25 - 27.5 + 0.5) / 9.810708435174291
    //   = -2 / 9.810708435174291 = -0.20385887657505022...
    //
    // p = min(1, 2 * normal_cdf(z)) with z = -0.20385887657505022:
    //   Phi(z) = 0.5*(1 + erf(z/sqrt(2))) = 0.4192318909612318...
    //   p = 2 * 0.4192318909612318... = 0.8384637819224636...
    // (computed independently, standard double-precision erf; the crate's
    // A&S 7.1.26 erf approximation has max pointwise error ~1.5e-7, well
    // within the 1e-6 tolerance used below)
    #[test]
    fn wilcoxon_no_ties_no_zeros() {
        let a = vec![
            101.0, 98.0, 103.0, 96.0, 105.0, 94.0, 107.0, 92.0, 109.0, 90.0,
        ];
        let b = vec![100.0; 10];

        let result = wilcoxon_signed_rank(&a, &b).expect("valid fixture");

        assert_eq!(result.n_effective, 10);
        assert!(
            (result.w_statistic - 25.0).abs() < 1e-12,
            "w_statistic: got {}",
            result.w_statistic
        );

        let expected_z = -0.20385887657505022_f64;
        assert!(
            (result.z - expected_z).abs() < 1e-6,
            "z: got {}, expected {}",
            result.z,
            expected_z
        );

        let expected_p = 0.8384637819224636_f64;
        assert!(
            (result.p_value - expected_p).abs() < 1e-6,
            "p_value: got {}, expected {}",
            result.p_value,
            expected_p
        );
    }

    // Zero-handling fixture (Pratt): 8 pairs, 2 of which have d_i = 0.
    //
    // a = [100, 102, 97, 100, 105, 96, 106, 99]
    // b = [100]*8
    // d = a - b = [0, 2, -3, 0, 5, -4, 6, -1]
    // |d| = [0, 2, 3, 0, 5, 4, 6, 1]
    //
    // Rank ALL 8 |d| values (Pratt: zeros included), average-tie:
    //   sorted ascending: 0(pos0), 0(pos3), 1(pos7), 2(pos1), 3(pos2),
    //                      4(pos5), 5(pos4), 6(pos6)
    //   pos0, pos3 tie for ranks 1,2 -> avg 1.5 each
    //   pos7 -> rank 3
    //   pos1 -> rank 4
    //   pos2 -> rank 5
    //   pos5 -> rank 6
    //   pos4 -> rank 7
    //   pos6 -> rank 8
    //   ranks by position: [1.5, 4, 5, 1.5, 7, 6, 8, 3]
    //
    // Drop the zero-d positions (pos0, pos3) from the W+/W- sums.
    // Remaining (nonzero) positions and their (d, rank):
    //   pos1: d=+2, rank=4
    //   pos2: d=-3, rank=5
    //   pos4: d=+5, rank=7
    //   pos5: d=-4, rank=6
    //   pos6: d=+6, rank=8
    //   pos7: d=-1, rank=3
    //
    // W+ = 4 + 7 + 8 = 19
    // W- = 5 + 6 + 3 = 14
    // Check: W+ + W- = 33 = sum of nonzero ranks (4+5+7+6+8+3=33)  [OK]
    // W = min(19, 14) = 14
    //
    // t0 = 2, n = 8, n_effective = 6 (>= 5, OK).
    //
    // mu = [n(n+1) - t0(t0+1)] / 4 = [8*9 - 2*3] / 4 = [72 - 6]/4 = 66/4 = 16.5
    //
    // Nonzero-|d| tie groups: {2, 3, 5, 4, 6, 1} are all distinct -> no
    // ties among the nonzero values, so the tie-correction sum is 0.
    //
    // sigma^2 = [n(n+1)(2n+1) - t0(t0+1)(2t0+1)] / 24 - 0/48
    //   n(n+1)(2n+1) = 8*9*17 = 1224
    //   t0(t0+1)(2t0+1) = 2*3*5 = 30
    //   sigma^2 = (1224 - 30) / 24 = 1194/24 = 49.75
    //   sigma = sqrt(49.75) = 7.053367989832942...
    //
    // z = (W - mu + 0.5) / sigma = (14 - 16.5 + 0.5) / 7.053367989832942
    //   = -2 / 7.053367989832942 = -0.2835524820033343...
    //
    // p = min(1, 2 * normal_cdf(z)) with z = -0.2835524820033343:
    //   Phi(z) = 0.5*(1 + erf(z/sqrt(2))) = 0.3883766783970002...
    //   p = 2 * 0.3883766783970002... = 0.7767533567940004...
    #[test]
    fn wilcoxon_pratt_zero_handling() {
        let a = vec![100.0, 102.0, 97.0, 100.0, 105.0, 96.0, 106.0, 99.0];
        let b = vec![100.0; 8];

        let result = wilcoxon_signed_rank(&a, &b).expect("valid fixture");

        assert_eq!(result.n_effective, 6);
        assert!(
            (result.w_statistic - 14.0).abs() < 1e-12,
            "w_statistic: got {}",
            result.w_statistic
        );

        let expected_z = -0.2835524820033343_f64;
        assert!(
            (result.z - expected_z).abs() < 1e-6,
            "z: got {}, expected {}",
            result.z,
            expected_z
        );

        let expected_p = 0.7767533567940004_f64;
        assert!(
            (result.p_value - expected_p).abs() < 1e-6,
            "p_value: got {}, expected {}",
            result.p_value,
            expected_p
        );
    }

    // Ties fixture: n=8, no zeros, two tie groups among the nonzero |d|.
    //
    // a = [102, 97, 104, 98, 105, 94, 103, 93]
    // b = [100]*8
    // d = a - b = [2, -3, 4, -2, 5, -6, 3, -7]
    // |d| = [2, 3, 4, 2, 5, 6, 3, 7]
    //
    // Rank all 8 |d| values, average-tie:
    //   sorted ascending (index, |d|): (0,2),(3,2),(1,3),(6,3),(2,4),(4,5),(5,6),(7,7)
    //   pos idx0, idx3 tie for ranks 1,2 -> avg 1.5 each
    //   pos idx1, idx6 tie for ranks 3,4 -> avg 3.5 each
    //   idx2 -> rank 5
    //   idx4 -> rank 6
    //   idx5 -> rank 7
    //   idx7 -> rank 8
    //   ranks by position: [1.5, 3.5, 5, 1.5, 6, 7, 3.5, 8]
    //
    // t0 = 0, n = 8, n_effective = 8.
    //
    // W+ (d>0 at idx0,2,4,6): 1.5 + 5 + 6 + 3.5 = 16
    // W- (d<0 at idx1,3,5,7): 3.5 + 1.5 + 7 + 8 = 20
    // Check: 16 + 20 = 36 = n(n+1)/2 = 8*9/2 = 36  [OK]
    // W = min(16, 20) = 16
    //
    // mu = n(n+1)/4 (t0=0) = 8*9/4 = 18
    //
    // Nonzero-|d| tie groups: {2: size 2, 3: size 2, 4:1, 5:1, 6:1, 7:1}.
    // tie_sum = sum(t^3 - t) over groups with t > 1 (t=1 groups contribute
    // 1-1=0 anyway): (2^3-2) + (2^3-2) = 6 + 6 = 12
    //
    // sigma^2 = n(n+1)(2n+1)/24 - tie_sum/48
    //   n(n+1)(2n+1) = 8*9*17 = 1224 -> 1224/24 = 51
    //   sigma^2 = 51 - 12/48 = 51 - 0.25 = 50.75
    //   sigma = sqrt(50.75) = 7.123903424387503...
    //
    // z = (W - mu + 0.5)/sigma = (16 - 18 + 0.5)/7.123903424387503
    //   = -1.5/7.123903424387503 = -0.21055872190307892...
    //
    // p = min(1, 2 * normal_cdf(z)) with z = -0.21055872190307892:
    //   Phi(z) = 0.5*(1 + erf(z/sqrt(2))) = 0.4166158126352...
    //   p = 2 * 0.4166158126352... = 0.8332316252704...
    #[test]
    fn wilcoxon_ties_correction() {
        let a = vec![102.0, 97.0, 104.0, 98.0, 105.0, 94.0, 103.0, 93.0];
        let b = vec![100.0; 8];

        let result = wilcoxon_signed_rank(&a, &b).expect("valid fixture");

        assert_eq!(result.n_effective, 8);
        assert!(
            (result.w_statistic - 16.0).abs() < 1e-12,
            "w_statistic: got {}",
            result.w_statistic
        );

        let expected_z = -0.21055872190307892_f64;
        assert!(
            (result.z - expected_z).abs() < 1e-6,
            "z: got {}, expected {}",
            result.z,
            expected_z
        );

        let expected_p = 0.8332316252704_f64;
        assert!(
            (result.p_value - expected_p).abs() < 1e-6,
            "p_value: got {}, expected {}",
            result.p_value,
            expected_p
        );
    }

    #[test]
    fn wilcoxon_errors_on_unequal_lengths() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        assert!(matches!(
            wilcoxon_signed_rank(&a, &b),
            Err(StatsError::InvalidInput(_))
        ));
    }

    #[test]
    fn wilcoxon_errors_on_too_few_effective_pairs() {
        // n = 4, all nonzero -> n_effective = 4 < 5.
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![0.0, 0.0, 0.0, 0.0];
        assert!(matches!(
            wilcoxon_signed_rank(&a, &b),
            Err(StatsError::InvalidInput(_))
        ));
    }

    #[test]
    fn wilcoxon_errors_on_too_few_effective_pairs_with_zeros() {
        // n = 8, but 4 zero pairs leave n_effective = 4 < 5.
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0];
        let b = vec![0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0];
        assert!(matches!(
            wilcoxon_signed_rank(&a, &b),
            Err(StatsError::InvalidInput(_))
        ));
    }

    #[test]
    fn cliffs_delta_identical_arrays_is_zero() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let delta = cliffs_delta(&a, &b);
        assert!((delta - 0.0).abs() < 1e-12, "delta: got {}", delta);
    }

    #[test]
    fn cliffs_delta_disjoint_extremes() {
        // Every element of a is smaller than every element of b ->
        // delta = -1 (sign convention: a-dominates-lower gives negative).
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![10.0, 20.0, 30.0];
        let delta_ab = cliffs_delta(&a, &b);
        assert!((delta_ab - (-1.0)).abs() < 1e-12, "got {}", delta_ab);

        // Reversed: every element of a is larger than every element of b
        // -> delta = +1.
        let delta_ba = cliffs_delta(&b, &a);
        assert!((delta_ba - 1.0).abs() < 1e-12, "got {}", delta_ba);
    }

    // a=[1,2,3], b=[2,3,4]. All 9 pairs (a_i, b_j):
    //   a=1: vs 2(less), vs 3(less), vs 4(less)          -> 0 greater, 3 less
    //   a=2: vs 2(equal), vs 3(less), vs 4(less)          -> 0 greater, 2 less
    //   a=3: vs 2(greater), vs 3(equal), vs 4(less)       -> 1 greater, 1 less
    // total greater = 1, total less = 6
    // delta = (1 - 6) / 9 = -5/9
    #[test]
    fn cliffs_delta_known_value() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 3.0, 4.0];
        let delta = cliffs_delta(&a, &b);
        let expected = -5.0 / 9.0;
        assert!(
            (delta - expected).abs() < 1e-12,
            "got {}, expected {}",
            delta,
            expected
        );
    }

    #[test]
    fn cliffs_magnitude_boundaries() {
        // Boundary convention: strict `<`, so landing exactly on a
        // threshold goes to the HIGHER bin.
        assert_eq!(cliffs_magnitude(0.1469), "negligible");
        assert_eq!(cliffs_magnitude(0.1471), "small");
        assert_eq!(cliffs_magnitude(0.147), "small");

        assert_eq!(cliffs_magnitude(0.3299), "small");
        assert_eq!(cliffs_magnitude(0.3301), "medium");
        assert_eq!(cliffs_magnitude(0.33), "medium");

        assert_eq!(cliffs_magnitude(0.4739), "medium");
        assert_eq!(cliffs_magnitude(0.4741), "large");
        assert_eq!(cliffs_magnitude(0.474), "large");

        // Sign is irrelevant -- magnitude uses |delta|.
        assert_eq!(cliffs_magnitude(-0.5), "large");
        assert_eq!(cliffs_magnitude(0.0), "negligible");
    }
}
