//! Rank-based statistics for comparing multiple algorithms across multiple
//! problems: per-row average ranking, the Friedman test, Holm/Hochberg
//! p-value adjustment, and the Nemenyi critical difference.
//!
//! See the crate-level documentation in `lib.rs` for the results-matrix
//! input convention (`&[Vec<f64>]`, rows = problems, columns = algorithms,
//! lower is better).

use crate::special::chi_square_sf;
use crate::{check_finite, StatsError};

/// Result of a Friedman test.
#[derive(Debug, Clone, PartialEq)]
pub struct FriedmanResult {
    /// The Friedman chi-square statistic, χ²_F.
    pub statistic: f64,
    /// The p-value, computed as `chi_square_sf(statistic, k - 1)`.
    pub p_value: f64,
    /// Mean rank per algorithm (column), length k.
    pub mean_ranks: Vec<f64>,
}

/// Computes the average rank of a single row (lower value -> lower rank,
/// 1-based). Tied values share the average of the ranks they span.
fn rank_row(row: &[f64]) -> Vec<f64> {
    let k = row.len();
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| row[a].partial_cmp(&row[b]).expect("NaN in results row"));

    let mut ranks = vec![0.0_f64; k];
    let mut i = 0;
    while i < k {
        let mut j = i;
        while j + 1 < k && row[order[j + 1]] == row[order[i]] {
            j += 1;
        }
        // Positions order[i..=j] are tied; they share ranks (i+1)..=(j+1)
        // (1-based), whose average is ((i+1)+(j+1))/2.
        let avg_rank = ((i + 1) + (j + 1)) as f64 / 2.0;
        for &pos in &order[i..=j] {
            ranks[pos] = avg_rank;
        }
        i = j + 1;
    }
    ranks
}

/// Computes the rank matrix of a results matrix: for each row (problem),
/// the average rank of each column (algorithm), 1-based, lower value ->
/// lower rank, ties shared as the average of the tied positions' ranks.
///
/// See the crate-level doc for the input convention. Rows are ranked
/// independently, so this function does not validate that the matrix is
/// rectangular (well-formedness is enforced by [`friedman`], which returns
/// a `Result`).
pub fn rank_matrix(results: &[Vec<f64>]) -> Vec<Vec<f64>> {
    results.iter().map(|row| rank_row(row)).collect()
}

/// Runs the Friedman test on a results matrix.
///
/// Input convention: `results[i][j]` is the score of algorithm `j` on
/// problem `i`; lower is better. `n = results.len()` (problems),
/// `k = results[0].len()` (algorithms).
///
/// Statistic (mean-rank form):
///
/// ```text
/// chi2_F = [12n / (k(k+1))] * [ sum_j( mean_rank_j^2 ) - k(k+1)^2/4 ]
/// ```
///
/// where `mean_rank_j` is the average rank of algorithm `j` across the `n`
/// problems. This is algebraically equivalent to the rank-sum form
/// `chi2_F = [12 / (n k (k+1))] * sum_j(R_j^2) - 3n(k+1)` where
/// `R_j = n * mean_rank_j` is the rank SUM of algorithm `j` (see the
/// `friedman_chi2_forms_agree` test for a numeric check on a fixture).
///
/// `p_value = chi_square_sf(chi2_F, k - 1)`, i.e. the statistic is treated
/// as approximately chi-square distributed with `k - 1` degrees of freedom.
///
/// # Errors
/// Returns `StatsError::InvalidInput` if `n < 2`, `k < 2`, or the matrix is
/// ragged (rows of differing length).
pub fn friedman(results: &[Vec<f64>]) -> Result<FriedmanResult, StatsError> {
    check_finite("friedman", results.iter().flatten().copied())?;

    let n = results.len();
    if n < 2 {
        return Err(StatsError::InvalidInput(format!(
            "friedman requires at least 2 problems (rows), got {}",
            n
        )));
    }

    let k = results[0].len();
    if k < 2 {
        return Err(StatsError::InvalidInput(format!(
            "friedman requires at least 2 algorithms (columns), got {}",
            k
        )));
    }

    for (i, row) in results.iter().enumerate() {
        if row.len() != k {
            return Err(StatsError::InvalidInput(format!(
                "ragged results matrix: row 0 has {} columns, row {} has {}",
                k,
                i,
                row.len()
            )));
        }
    }

    let ranks = rank_matrix(results);

    let mut mean_ranks = vec![0.0_f64; k];
    for row in &ranks {
        for (j, &r) in row.iter().enumerate() {
            mean_ranks[j] += r;
        }
    }
    for m in &mut mean_ranks {
        *m /= n as f64;
    }

    let n_f = n as f64;
    let k_f = k as f64;
    let sum_sq_mean_ranks: f64 = mean_ranks.iter().map(|m| m * m).sum();
    let statistic =
        (12.0 * n_f / (k_f * (k_f + 1.0))) * (sum_sq_mean_ranks - k_f * (k_f + 1.0).powi(2) / 4.0);

    let p_value = chi_square_sf(statistic, k - 1);

    Ok(FriedmanResult {
        statistic,
        p_value,
        mean_ranks,
    })
}

/// Holm step-down adjustment of a set of p-values (family-wise error rate).
///
/// Sorts the p-values ascending, p_(1) <= ... <= p_(m). The adjusted value
/// at sorted rank `i` (1-based) is
/// `adj_(i) = max_{j <= i} [ (m - j + 1) * p_(j) ]`, i.e. a running maximum
/// of `(m - j + 1) * p_(j)` enforcing monotonicity, clamped to at most 1.
/// The result is returned in the original input order.
pub fn holm(p_values: &[f64]) -> Vec<f64> {
    let m = p_values.len();
    if m == 0 {
        return Vec::new();
    }

    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| {
        p_values[a]
            .partial_cmp(&p_values[b])
            .expect("NaN in p_values")
    });

    let mut adjusted_sorted = vec![0.0_f64; m];
    let mut running_max = 0.0_f64;
    for (i, &idx) in order.iter().enumerate() {
        // 1-based rank is i + 1; multiplier is (m - rank + 1) = m - i.
        let mult = (m - i) as f64;
        let candidate = mult * p_values[idx];
        running_max = running_max.max(candidate);
        adjusted_sorted[i] = running_max.min(1.0);
    }

    let mut result = vec![0.0_f64; m];
    for (i, &idx) in order.iter().enumerate() {
        result[idx] = adjusted_sorted[i];
    }
    result
}

/// Hochberg step-up adjustment of a set of p-values (family-wise error
/// rate, valid under independence/positive dependence).
///
/// Sorts the p-values ascending, p_(1) <= ... <= p_(m). Working from the
/// largest (rank m) down to the smallest (rank 1), the adjusted value at
/// sorted rank `i` (1-based) is
/// `adj_(i) = min_{j >= i} [ (m - j + 1) * p_(j) ]`, i.e. a running minimum
/// of `(m - j + 1) * p_(j)` enforcing monotonicity, clamped to at most 1.
/// The result is returned in the original input order.
pub fn hochberg(p_values: &[f64]) -> Vec<f64> {
    let m = p_values.len();
    if m == 0 {
        return Vec::new();
    }

    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| {
        p_values[a]
            .partial_cmp(&p_values[b])
            .expect("NaN in p_values")
    });

    let mut adjusted_sorted = vec![0.0_f64; m];
    let mut running_min = f64::INFINITY;
    for i in (0..m).rev() {
        let idx = order[i];
        // 1-based rank is i + 1; multiplier is (m - rank + 1) = m - i.
        let mult = (m - i) as f64;
        let candidate = mult * p_values[idx];
        running_min = running_min.min(candidate);
        adjusted_sorted[i] = running_min.min(1.0);
    }

    let mut result = vec![0.0_f64; m];
    for (i, &idx) in order.iter().enumerate() {
        result[idx] = adjusted_sorted[i];
    }
    result
}

/// Studentized range critical values q_{0.05}(k) for the Nemenyi test,
/// k = 2..=20 (Demšar 2006, Table 5(b) / standard Nemenyi tables).
/// `Q_ALPHA_05[i]` is the value for `k = i + 2`.
const Q_ALPHA_05: [f64; 19] = [
    1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164, 3.219, 3.268, 3.313, 3.354,
    3.391, 3.426, 3.458, 3.489, 3.517, 3.544,
];

/// Nemenyi critical difference for post-hoc comparison of `k` algorithms
/// evaluated on `n` problems, at alpha = 0.05:
///
/// ```text
/// CD = q_0.05(k) * sqrt( k(k+1) / (6n) )
/// ```
///
/// # Errors
/// Returns `StatsError::InvalidInput` if `k` is outside `2..=20` (the
/// hard-coded table range) or `n == 0`.
pub fn nemenyi_cd(k: usize, n: usize) -> Result<f64, StatsError> {
    if !(2..=20).contains(&k) {
        return Err(StatsError::InvalidInput(format!(
            "nemenyi_cd: k must be in 2..=20 (got {}); q_0.05 table not available outside this range",
            k
        )));
    }
    if n == 0 {
        return Err(StatsError::InvalidInput(
            "nemenyi_cd: n must be at least 1".to_string(),
        ));
    }

    let q = Q_ALPHA_05[k - 2];
    let k_f = k as f64;
    let n_f = n as f64;
    Ok(q * (k_f * (k_f + 1.0) / (6.0 * n_f)).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_matrix_with_ties() {
        // Hand-computed 3x3 fixture, results[i][j] = score of algorithm j
        // on problem i, lower is better.
        //
        // Row 0: [1, 2, 2]
        //   sorted ascending: 1 (pos0), 2 (pos1), 2 (pos2)
        //   pos0 -> rank 1 (no tie)
        //   pos1, pos2 tie for ranks 2 and 3 -> average rank 2.5 each
        //   expected ranks: [1, 2.5, 2.5]
        //
        // Row 1: [5, 3, 3]
        //   sorted ascending: 3 (pos1), 3 (pos2), 5 (pos0)
        //   pos1, pos2 tie for ranks 1 and 2 -> average rank 1.5 each
        //   pos0 -> rank 3 (no tie)
        //   expected ranks: [3, 1.5, 1.5]
        //
        // Row 2: [4, 4, 1]
        //   sorted ascending: 1 (pos2), 4 (pos0), 4 (pos1)
        //   pos2 -> rank 1 (no tie)
        //   pos0, pos1 tie for ranks 2 and 3 -> average rank 2.5 each
        //   expected ranks: [2.5, 2.5, 1]
        let results = vec![
            vec![1.0, 2.0, 2.0],
            vec![5.0, 3.0, 3.0],
            vec![4.0, 4.0, 1.0],
        ];

        let ranks = rank_matrix(&results);

        let expected = vec![
            vec![1.0, 2.5, 2.5],
            vec![3.0, 1.5, 1.5],
            vec![2.5, 2.5, 1.0],
        ];

        for (row_r, row_e) in ranks.iter().zip(expected.iter()) {
            for (r, e) in row_r.iter().zip(row_e.iter()) {
                assert!(
                    (r - e).abs() < 1e-12,
                    "rank mismatch: got {:?}, expected {:?}",
                    ranks,
                    expected
                );
            }
        }

        // Each row's ranks must sum to n(n+1)/2 = 6 regardless of ties.
        for row in &ranks {
            let sum: f64 = row.iter().sum();
            assert!(
                (sum - 6.0).abs() < 1e-12,
                "row rank sum should be 6, got {}",
                sum
            );
        }
    }

    // Fully hand-computed Friedman fixture: 3 algorithms (A, B, C) x 4
    // problems. Values are already a permutation of {1, 2, 3} in every row
    // (no ties), so the rank of an algorithm in a row equals its raw value
    // directly -- ranks are trivial to verify by inspection.
    //
    // results[i] = [A, B, C] on problem i:
    //   problem 0: [1, 2, 3]  -> ranks [1, 2, 3]
    //   problem 1: [2, 3, 1]  -> ranks [2, 3, 1]
    //   problem 2: [3, 1, 2]  -> ranks [3, 1, 2]
    //   problem 3: [1, 3, 2]  -> ranks [1, 3, 2]
    //
    // Rank sums per column (algorithm):
    //   R_A = 1 + 2 + 3 + 1 = 7
    //   R_B = 2 + 3 + 1 + 3 = 9
    //   R_C = 3 + 1 + 2 + 2 = 8
    //   check: R_A + R_B + R_C = 24 = n*k*(k+1)/2 = 4*3*4/2 = 24  [OK]
    //
    // n = 4, k = 3.
    //
    // Mean ranks: R_A/n = 1.75, R_B/n = 2.25, R_C/n = 2.0
    //
    // Mean-rank form:
    //   chi2_F = [12n / (k(k+1))] * [ sum(mean_rank_j^2) - k(k+1)^2/4 ]
    //   sum(mean_rank_j^2) = 1.75^2 + 2.25^2 + 2.0^2
    //                      = 3.0625 + 5.0625 + 4.0 = 12.125
    //   k(k+1)^2/4 = 3 * 16 / 4 = 12
    //   [12n / (k(k+1))] = 12*4 / (3*4) = 48/12 = 4
    //   chi2_F = 4 * (12.125 - 12) = 4 * 0.125 = 0.5
    //
    // Rank-sum form (cross-check):
    //   sum(R_j^2) = 7^2 + 9^2 + 8^2 = 49 + 81 + 64 = 194
    //   [12 / (n*k*(k+1))] = 12 / (4*3*4) = 12/48 = 0.25
    //   chi2_F = 0.25 * 194 - 3*n*(k+1) = 48.5 - 3*4*4 = 48.5 - 48 = 0.5
    //   Both forms agree: chi2_F = 0.5
    //
    // p-value: chi_square_sf(0.5, k-1=2). For 2 degrees of freedom the
    // chi-square survival function has the closed form sf(x, 2) = exp(-x/2).
    //   p = exp(-0.5/2) = exp(-0.25) = 0.7788007830714049...
    #[test]
    fn friedman_hand_computed_fixture() {
        let results = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 1.0],
            vec![3.0, 1.0, 2.0],
            vec![1.0, 3.0, 2.0],
        ];

        let result = friedman(&results).expect("valid fixture");

        assert!((result.mean_ranks[0] - 1.75).abs() < 1e-10, "mean rank A");
        assert!((result.mean_ranks[1] - 2.25).abs() < 1e-10, "mean rank B");
        assert!((result.mean_ranks[2] - 2.0).abs() < 1e-10, "mean rank C");

        assert!(
            (result.statistic - 0.5).abs() < 1e-10,
            "statistic: got {}, expected 0.5",
            result.statistic
        );

        let expected_p = (-0.25_f64).exp();
        assert!(
            (result.p_value - expected_p).abs() < 1e-8,
            "p_value: got {}, expected {}",
            result.p_value,
            expected_p
        );
    }

    #[test]
    fn friedman_chi2_forms_agree() {
        // Same fixture as friedman_hand_computed_fixture; verify the two
        // algebraic forms of chi2_F agree to within 1e-10.
        let results = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 1.0],
            vec![3.0, 1.0, 2.0],
            vec![1.0, 3.0, 2.0],
        ];
        let n = results.len();
        let k = results[0].len();

        let ranks = rank_matrix(&results);
        let mut rank_sums = vec![0.0_f64; k];
        for row in &ranks {
            for (j, &r) in row.iter().enumerate() {
                rank_sums[j] += r;
            }
        }
        let mean_ranks: Vec<f64> = rank_sums.iter().map(|s| s / n as f64).collect();

        let n_f = n as f64;
        let k_f = k as f64;

        // Mean-rank form.
        let sum_sq_mean: f64 = mean_ranks.iter().map(|m| m * m).sum();
        let chi2_mean_form =
            (12.0 * n_f / (k_f * (k_f + 1.0))) * (sum_sq_mean - k_f * (k_f + 1.0).powi(2) / 4.0);

        // Rank-sum form.
        let sum_sq_r: f64 = rank_sums.iter().map(|r| r * r).sum();
        let chi2_sum_form = (12.0 / (n_f * k_f * (k_f + 1.0))) * sum_sq_r - 3.0 * n_f * (k_f + 1.0);

        assert!(
            (chi2_mean_form - chi2_sum_form).abs() < 1e-10,
            "forms disagree: mean-rank form = {}, rank-sum form = {}",
            chi2_mean_form,
            chi2_sum_form
        );

        // Both forms should also match friedman()'s reported statistic.
        let result = friedman(&results).expect("valid fixture");
        assert!((result.statistic - chi2_mean_form).abs() < 1e-10);
    }

    #[test]
    fn friedman_errors_on_too_few_problems() {
        let results = vec![vec![1.0, 2.0]];
        assert!(matches!(
            friedman(&results),
            Err(StatsError::InvalidInput(_))
        ));
    }

    #[test]
    fn friedman_errors_on_too_few_algorithms() {
        let results = vec![vec![1.0], vec![2.0]];
        assert!(matches!(
            friedman(&results),
            Err(StatsError::InvalidInput(_))
        ));
    }

    #[test]
    fn friedman_errors_on_ragged_rows() {
        let results = vec![vec![1.0, 2.0, 3.0], vec![1.0, 2.0]];
        assert!(matches!(
            friedman(&results),
            Err(StatsError::InvalidInput(_))
        ));
    }

    #[test]
    fn friedman_errors_on_nan() {
        let results = vec![vec![1.0, f64::NAN, 3.0], vec![2.0, 3.0, 1.0]];
        let err = friedman(&results).expect_err("NaN must be rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("friedman"), "message should name the function: {msg}");
        assert!(msg.contains("non-finite"), "message should say non-finite: {msg}");
    }

    #[test]
    fn friedman_errors_on_infinity() {
        let results = vec![vec![1.0, f64::INFINITY, 3.0], vec![2.0, 3.0, 1.0]];
        let err = friedman(&results).expect_err("infinity must be rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("friedman"), "message should name the function: {msg}");
        assert!(msg.contains("non-finite"), "message should say non-finite: {msg}");
    }

    // Textbook Holm/Hochberg fixture: p = [0.01, 0.02, 0.03, 0.04], m = 4.
    // p is already sorted ascending: p_(1)=.01, p_(2)=.02, p_(3)=.03, p_(4)=.04.
    //
    // Holm (step-down, running max of (m-rank+1)*p_(rank)):
    //   rank 1: (4-1+1)*.01 = 4*.01 = .04                 -> adj_(1) = .04
    //   rank 2: (4-2+1)*.02 = 3*.02 = .06                 -> adj_(2) = max(.04, .06) = .06
    //   rank 3: (4-3+1)*.03 = 2*.03 = .06                 -> adj_(3) = max(.06, .06) = .06
    //   rank 4: (4-4+1)*.04 = 1*.04 = .04                 -> adj_(4) = max(.06, .04) = .06
    //   Holm-adjusted (in ascending-p order) = [0.04, 0.06, 0.06, 0.06]
    //
    // Hochberg (step-up, running min of (m-rank+1)*p_(rank), from rank m down to 1):
    //   rank 4: (4-4+1)*.04 = 1*.04 = .04                 -> adj_(4) = .04
    //   rank 3: (4-3+1)*.03 = 2*.03 = .06                 -> adj_(3) = min(.04, .06) = .04
    //   rank 2: (4-2+1)*.02 = 3*.02 = .06                 -> adj_(2) = min(.04, .06) = .04
    //   rank 1: (4-1+1)*.01 = 4*.01 = .04                 -> adj_(1) = min(.04, .04) = .04
    //   Hochberg-adjusted (in ascending-p order) = [0.04, 0.04, 0.04, 0.04]
    //
    // The input is already sorted ascending, so ascending-p order equals
    // input order here.
    #[test]
    fn holm_hochberg_textbook() {
        let p = vec![0.01, 0.02, 0.03, 0.04];

        let holm_adj = holm(&p);
        let expected_holm = vec![0.04, 0.06, 0.06, 0.06];
        for (a, e) in holm_adj.iter().zip(expected_holm.iter()) {
            assert!(
                (a - e).abs() < 1e-12,
                "holm: got {:?}, expected {:?}",
                holm_adj,
                expected_holm
            );
        }

        let hochberg_adj = hochberg(&p);
        let expected_hochberg = vec![0.04, 0.04, 0.04, 0.04];
        for (a, e) in hochberg_adj.iter().zip(expected_hochberg.iter()) {
            assert!(
                (a - e).abs() < 1e-12,
                "hochberg: got {:?}, expected {:?}",
                hochberg_adj,
                expected_hochberg
            );
        }
    }

    #[test]
    fn holm_restores_original_order() {
        // Same values as the textbook fixture but permuted in the input;
        // adjusted values must come back in the same permuted order.
        let p = vec![0.04, 0.01, 0.03, 0.02];
        let holm_adj = holm(&p);
        // original order: [0.04, 0.01, 0.03, 0.02]
        // sorted ascending: p_(1)=0.01(idx1) p_(2)=0.02(idx3) p_(3)=0.03(idx2) p_(4)=0.04(idx0)
        // adjusted sorted:  [0.04, 0.06, 0.06, 0.06] (idx1, idx3, idx2, idx0)
        // restored to original order (idx0, idx1, idx2, idx3): [0.06, 0.04, 0.06, 0.06]
        let expected = vec![0.06, 0.04, 0.06, 0.06];
        for (a, e) in holm_adj.iter().zip(expected.iter()) {
            assert!(
                (a - e).abs() < 1e-12,
                "got {:?}, expected {:?}",
                holm_adj,
                expected
            );
        }
    }

    #[test]
    fn hochberg_restores_original_order() {
        // Non-uniform, shuffled input: p = [0.03, 0.001, 0.04, 0.002]
        // (original order: idx0=0.03, idx1=0.001, idx2=0.04, idx3=0.002).
        //
        // Deliberately NOT a uniformly-spaced fixture (unlike the earlier
        // holm_restores_original_order / textbook fixtures) so that a
        // broken order-restoration mapping -- e.g. one that accidentally
        // returns the values in sorted order, or off-by-one shifts the
        // index mapping -- cannot coincidentally produce the right answer.
        //
        // Sorted ascending: p_(1)=0.001(idx1), p_(2)=0.002(idx3),
        //                    p_(3)=0.03(idx0), p_(4)=0.04(idx2).
        //
        // Hochberg (step-up, running min of (m-rank+1)*p_(rank), from
        // rank m=4 down to rank 1):
        //   rank 4 (idx2, p=.04):  mult=4-4+1=1, 1*.04  = .04   -> adj = .04
        //   rank 3 (idx0, p=.03):  mult=4-3+1=2, 2*.03  = .06   -> adj = min(.04,.06) = .04
        //   rank 2 (idx3, p=.002): mult=4-2+1=3, 3*.002 = .006  -> adj = min(.04,.006) = .006
        //   rank 1 (idx1, p=.001): mult=4-1+1=4, 4*.001 = .004  -> adj = min(.006,.004) = .004
        //   adjusted in ascending-p order (idx1, idx3, idx0, idx2)
        //     = [0.004, 0.006, 0.04, 0.04]
        //
        // Restored to original order (idx0, idx1, idx2, idx3):
        //   idx0 -> 0.04 (from rank 3)
        //   idx1 -> 0.004 (from rank 1)
        //   idx2 -> 0.04 (from rank 4)
        //   idx3 -> 0.006 (from rank 2)
        //   expected = [0.04, 0.004, 0.04, 0.006]
        let p = vec![0.03, 0.001, 0.04, 0.002];
        let hochberg_adj = hochberg(&p);
        let expected = vec![0.04, 0.004, 0.04, 0.006];
        for (a, e) in hochberg_adj.iter().zip(expected.iter()) {
            assert!(
                (a - e).abs() < 1e-12,
                "got {:?}, expected {:?}",
                hochberg_adj,
                expected
            );
        }
    }

    // Nemenyi CD(4, 14): q_0.05(4) = 2.569 (table index k-2 = 2).
    // CD = 2.569 * sqrt(4*5 / (6*14)) = 2.569 * sqrt(20/84)
    //    = 2.569 * sqrt(0.238095238095...)
    //    = 2.569 * 0.48795003650... = 1.25354364...
    #[test]
    fn nemenyi_cd_known() {
        let cd = nemenyi_cd(4, 14).expect("k=4 is in range");
        let expected = 1.2535436437023908_f64;
        assert!(
            (cd - expected).abs() < 1e-4,
            "CD(4,14): got {}, expected {}",
            cd,
            expected
        );
    }

    #[test]
    fn nemenyi_cd_out_of_range_k() {
        assert!(matches!(
            nemenyi_cd(21, 14),
            Err(StatsError::InvalidInput(_))
        ));
        assert!(matches!(
            nemenyi_cd(1, 14),
            Err(StatsError::InvalidInput(_))
        ));
    }

    #[test]
    fn nemenyi_cd_zero_n() {
        assert!(matches!(nemenyi_cd(4, 0), Err(StatsError::InvalidInput(_))));
    }
}
