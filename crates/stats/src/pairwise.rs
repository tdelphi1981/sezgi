//! Pairwise comparison statistics for two paired samples (e.g. one
//! algorithm's per-problem results vs. another's): the Wilcoxon
//! signed-rank test and Cliff's delta effect size.
//!
//! See the crate-level documentation in `lib.rs` for the general
//! results-matrix convention used elsewhere in this crate; the functions
//! here instead take two same-length slices representing a single paired
//! comparison (e.g. two columns of a results matrix).

use crate::special::normal_cdf;
use crate::{check_finite, StatsError};

/// Which distribution [`WilcoxonResult::p_value`] was drawn from.
///
/// See [`wilcoxon_signed_rank`]'s "Exact vs. normal approximation" section
/// for the eligibility rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WilcoxonMethod {
    /// `p_value` is the exact two-sided permutation p-value (see
    /// [`exact_wilcoxon_p_value`]).
    Exact,
    /// `p_value` is `min(1, 2 * normal_cdf(z))` (Pratt's normal
    /// approximation with continuity correction).
    NormalApprox,
}

/// Result of a Wilcoxon signed-rank test ([`wilcoxon_signed_rank`]).
#[derive(Debug, Clone, PartialEq)]
pub struct WilcoxonResult {
    /// W = min(W+, W-), the smaller of the two signed rank sums (over the
    /// effective/nonzero pairs).
    pub w_statistic: f64,
    /// Normal-approximation z score (continuity-corrected), `z <= 0` since
    /// `w_statistic` is the min of the two rank sums. Always computed the
    /// same way regardless of [`WilcoxonResult::method`] (informational when
    /// `method == Exact`, since `p_value` then comes from the exact
    /// distribution instead).
    pub z: f64,
    /// Two-sided p-value: the exact permutation p-value when
    /// `method == Exact`, else `min(1, 2 * normal_cdf(z))`.
    pub p_value: f64,
    /// Number of pairs with a nonzero difference (`n - t0`), the effective
    /// sample size the normal approximation (or, for `Exact`, `n` itself
    /// since eligibility requires zero-free input) is computed over.
    pub n_effective: usize,
    /// Which distribution `p_value` was drawn from; see [`WilcoxonMethod`].
    pub method: WilcoxonMethod,
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
/// # Exact vs. normal approximation
///
/// When `n_effective <= 25` **and** there are no zero differences (`t0 ==
/// 0`) **and** no tied `|d_i|` ranks (every rank is an untied integer),
/// `p_value` is instead the exact two-sided permutation p-value (`method ==
/// WilcoxonMethod::Exact`); otherwise the normal approximation above is used
/// (`method == WilcoxonMethod::NormalApprox`). `z` is always computed via
/// the normal-approximation formula regardless of `method`.
///
/// The exact p-value is
///
/// ```text
/// p = min(1, 2 * count(sum <= W) / 2^n)
/// ```
///
/// where `count(sum <= W)` is the number of subsets of `{1, ..., n}` whose
/// elements sum to at most `W` (`W` is `w_statistic`, an integer since
/// eligibility requires untied, zero-free ranks), computed by the standard
/// rank-sum DP:
///
/// ```text
/// c[0] = 1;
/// for r in 1..=n {
///     for s in (r..=max).rev() {
///         c[s] += c[s - r];
///     }
/// }
/// ```
///
/// **This formula (both the `min(1, 2 * count / 2^n)` two-sided convention
/// and the DP recurrence) is PINNED: changing it is a semver event.**
///
/// # Errors
/// Returns `StatsError::InvalidInput` if `a.len() != b.len()`. Also returns
/// it if `n_effective < 5` **and** the input is not exact-eligible (i.e.
/// there are zeros, or tied `|d_i|` ranks, or `n_effective > 25`) — below 5,
/// the normal approximation is unreliable and, absent exact eligibility,
/// this crate has no other basis for a p-value. Exact-eligible input with
/// `n_effective < 5` is accepted (the exact distribution is well-defined at
/// any `n_effective >= 1`).
///
/// # Reference
/// Pratt, J. W. (1959). "Remarks on Zeros and Ties in the Wilcoxon Signed
/// Rank Procedures." *Journal of the American Statistical Association*,
/// 54(287), 655-667.
pub fn wilcoxon_signed_rank(a: &[f64], b: &[f64]) -> Result<WilcoxonResult, StatsError> {
    check_finite("wilcoxon_signed_rank", a.iter().chain(b.iter()).copied())?;

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

    // Exact eligibility: no zeros, n_effective <= 25, and no tied |d| ranks
    // (every nonzero-|d| tie group has size 1; since t0 == 0 here, `groups`
    // contains only nonzero-value groups).
    let has_tied_ranks = groups.iter().any(|&(_, size)| size > 1);
    let eligible_exact = t0 == 0 && n_effective <= 25 && !has_tied_ranks;

    if !eligible_exact && n_effective < 5 {
        return Err(StatsError::InvalidInput(format!(
            "wilcoxon_signed_rank requires at least 5 nonzero-difference pairs (n_effective) unless the input is exact-eligible (no zeros, no tied |d| ranks, n_effective <= 25); got n_effective = {}",
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

    let (p_value, method) = if eligible_exact {
        // Eligibility guarantees integer, untied ranks 1..=n_effective (==
        // n, since t0 == 0), so w_statistic is an exact integer.
        (
            exact_wilcoxon_p_value(w_statistic.round() as usize, n_effective),
            WilcoxonMethod::Exact,
        )
    } else {
        ((2.0 * normal_cdf(z)).min(1.0), WilcoxonMethod::NormalApprox)
    };

    Ok(WilcoxonResult {
        w_statistic,
        z,
        p_value,
        n_effective,
        method,
    })
}

/// Exact two-sided Wilcoxon signed-rank p-value for `n` untied, zero-free
/// ranks `1..=n`, given the observed rank-sum statistic `w` (`w_statistic`,
/// the smaller of W+/W-).
///
/// `p = min(1, 2 * count(sum <= w) / 2^n)`, where `count(sum <= w)` is the
/// number of subsets of `{1, ..., n}` summing to at most `w`, computed via
/// the standard rank-sum DP (`c[s]` = number of subsets summing to exactly
/// `s`):
///
/// ```text
/// c[0] = 1;
/// for r in 1..=n {
///     for s in (r..=max).rev() {
///         c[s] += c[s - r];
///     }
/// }
/// ```
///
/// **PINNED**: this formula (the DP recurrence and the
/// `min(1, 2 * count / 2^n)` two-sided convention) must not change without
/// a semver bump; see [`wilcoxon_signed_rank`]'s doc comment.
///
/// `u64` counts suffice: `n <= 25` (the caller's eligibility bound) makes
/// `2^n <= 2^25`, well within `u64` range, and every per-sum subset count is
/// bounded by the same `2^n` total.
fn exact_wilcoxon_p_value(w: usize, n: usize) -> f64 {
    let max_sum = n * (n + 1) / 2;
    let w = w.min(max_sum);

    let mut c = vec![0_u64; max_sum + 1];
    c[0] = 1;
    for r in 1..=n {
        for s in (r..=max_sum).rev() {
            c[s] += c[s - r];
        }
    }

    let count: u64 = c[0..=w].iter().sum();
    let total = 2_u64.pow(n as u32);
    (2.0 * count as f64 / total as f64).min(1.0)
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
///
/// # Errors
/// Returns `StatsError::InvalidInput` if any value in `a` or `b` is
/// non-finite (NaN or +/-infinity).
pub fn cliffs_delta(a: &[f64], b: &[f64]) -> Result<f64, StatsError> {
    check_finite("cliffs_delta", a.iter().chain(b.iter()).copied())?;

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

    Ok((greater - less) as f64 / (n as f64 * m as f64))
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
    // (z is always computed this way, regardless of method.)
    //
    // *** TASK 6 RE-PIN (M2d-1, flagged prominently per the brief's
    // disclosure rule): this fixture is n=10, t0=0, and its ranks are
    // untied (rank(|d_i|) = |d_i| exactly, see above) -- i.e. it is now
    // EXACT-eligible (n_effective=10 <= 25, no zeros, no ties), so
    // `p_value` switches from the normal-approximation value below to the
    // exact permutation p-value. The superseded normal-approx value was:
    //   p = min(1, 2 * normal_cdf(z)) with z = -0.20385887657505022:
    //     Phi(z) = 0.5*(1 + erf(z/sqrt(2))) = 0.4192318909612318...
    //     p = 2 * 0.4192318909612318... = 0.8384637819224636...
    // `z` itself is unchanged (method-independent) -- see the companion
    // test `wilcoxon_no_ties_no_zeros_normal_approx_z_unaffected` below.
    //
    // Exact p, W = 25, n = 10: p = min(1, 2 * count(sum <= 25) / 2^10),
    // count(sum <= 25) from the pinned DP over subsets of {1..10}:
    //   count(sum <= 25) = 433 (of 2^10 = 1024 total subsets)
    //   p = 2 * 433 / 1024 = 866 / 1024 = 0.845703125 (exact dyadic
    //   rational). Independently cross-checked against SciPy's
    //   `wilcoxon(a, b, zero_method="wilcox", correction=False,
    //   mode="exact")` on this exact fixture, which returns
    //   statistic=25.0, pvalue=0.845703125 -- see task-6-report.md.
    #[test]
    fn wilcoxon_no_ties_no_zeros() {
        let a = vec![
            101.0, 98.0, 103.0, 96.0, 105.0, 94.0, 107.0, 92.0, 109.0, 90.0,
        ];
        let b = vec![100.0; 10];

        let result = wilcoxon_signed_rank(&a, &b).expect("valid fixture");

        assert_eq!(result.n_effective, 10);
        assert_eq!(result.method, WilcoxonMethod::Exact);
        assert!(
            (result.w_statistic - 25.0).abs() < 1e-12,
            "w_statistic: got {}",
            result.w_statistic
        );

        let expected_p = 0.845703125_f64;
        assert_eq!(
            result.p_value, expected_p,
            "exact dyadic rational -- bit-exact assert"
        );
    }

    // Companion to `wilcoxon_no_ties_no_zeros`: `z` (the normal-
    // approximation z score) is unaffected by Task 6 -- it is always
    // computed via the normal formula regardless of `method`.
    #[test]
    fn wilcoxon_no_ties_no_zeros_normal_approx_z_unaffected() {
        let a = vec![
            101.0, 98.0, 103.0, 96.0, 105.0, 94.0, 107.0, 92.0, 109.0, 90.0,
        ];
        let b = vec![100.0; 10];

        let result = wilcoxon_signed_rank(&a, &b).expect("valid fixture");

        let expected_z = -0.20385887657505022_f64;
        assert!(
            (result.z - expected_z).abs() < 1e-6,
            "z: got {}, expected {}",
            result.z,
            expected_z
        );
    }

    // Exact small-n fixture (Task 6, M2d-1). CONTROLLER RULING: the task
    // brief's illustrative a/b vectors for this test were defective (they
    // produce all-negative differences, contradicting the normative
    // comment); this fixture instead follows the brief's normative
    // construction text directly: |d| ranks are 1..8, all distinct, and
    // exactly the smallest-|d| pair has the sign opposite the rest.
    //
    // a = [99, 102, 103, 104, 105, 106, 107, 108]
    // b = [100]*8
    // d = a - b = [-1, 2, 3, 4, 5, 6, 7, 8]
    // |d| = [1, 2, 3, 4, 5, 6, 7, 8] -- already sorted, no ties, so
    // rank(|d_i|) = |d_i| exactly.
    //
    // t0 = 0, n = 8, n_effective = 8 (<= 25, no zeros, no ties -> exact-
    // eligible).
    //
    // W- (only negative d, |d|=1 -> rank 1): W- = 1
    // W+ (positive d, ranks 2..8): W+ = 2+3+4+5+6+7+8 = 35
    // Check: W- + W+ = 36 = n(n+1)/2 = 8*9/2 = 36  [OK]
    // W = min(1, 35) = 1
    //
    // Exact p: count(sum <= 1) over subsets of {1..8} = 2 (the empty
    // subset, sum=0; and {1}, sum=1). 2^8 = 256.
    // p = min(1, 2 * 2 / 256) = 4/256 = 0.015625 (exact dyadic rational).
    // Independently cross-checked against SciPy's
    // `wilcoxon([99,102,103,104,105,106,107,108], [100]*8,
    // zero_method="wilcox", correction=False, mode="exact")`, which
    // returns statistic=1.0, pvalue=0.015625 -- see task-6-report.md.
    #[test]
    fn exact_small_n_no_ties_uses_exact_distribution() {
        let a = vec![99.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0];
        let b = vec![100.0; 8];

        let r = wilcoxon_signed_rank(&a, &b).unwrap();
        assert_eq!(r.method, WilcoxonMethod::Exact);
        assert_eq!(r.n_effective, 8);
        assert_eq!(r.w_statistic, 1.0);
        assert_eq!(r.p_value, 0.015625); // exact dyadic rational -- bit-exact assert
    }

    // Ties fixture: two equal |d| values (tied ranks) force NormalApprox
    // even though n_effective = 5 <= 25 and there are no zeros.
    //
    // a = [102, 98, 103, 104, 105], b = [100]*5
    // d = a - b = [2, -2, 3, 4, 5], |d| = [2, 2, 3, 4, 5]
    // idx0, idx1 tie for ranks 1,2 -> avg 1.5 each; idx2->3, idx3->4, idx4->5
    // ranks by position: [1.5, 1.5, 3, 4, 5]
    //
    // t0 = 0, n = 5, n_effective = 5. Tied ranks -> NOT exact-eligible.
    //
    // W+ (d>0: idx0,2,3,4) = 1.5+3+4+5 = 13.5
    // W- (d<0: idx1) = 1.5
    // W = min(13.5, 1.5) = 1.5
    //
    // mu = n(n+1)/4 = 5*6/4 = 7.5
    // tie_sum = 2^3-2 = 6 (one size-2 tie group among the nonzero |d|)
    // sigma^2 = n(n+1)(2n+1)/24 - tie_sum/48 = 5*6*11/24 - 6/48
    //   = 330/24 - 0.125 = 13.75 - 0.125 = 13.625
    // sigma = sqrt(13.625) = 3.69120576505835...
    // z = (1.5 - 7.5 + 0.5)/3.69120576505835 = -5.5/3.69120576505835
    //   = -1.490028015252912...
    // p = min(1, 2*normal_cdf(z)) = 0.13621686984456766...
    // (cross-checked against SciPy's wilcoxon(..., zero_method="pratt",
    // correction=True, mode="approx") on this fixture: statistic=1.5,
    // pvalue=0.13621686984456766 -- see task-6-report.md)
    #[test]
    fn ties_fall_back_to_normal_approx() {
        let a = vec![102.0, 98.0, 103.0, 104.0, 105.0];
        let b = vec![100.0; 5];

        let r = wilcoxon_signed_rank(&a, &b).expect("valid fixture");
        assert_eq!(r.method, WilcoxonMethod::NormalApprox);
        assert_eq!(r.n_effective, 5);
        assert!(
            (r.w_statistic - 1.5).abs() < 1e-12,
            "w_statistic: got {}",
            r.w_statistic
        );

        let expected_z = -1.490028015252912_f64;
        assert!(
            (r.z - expected_z).abs() < 1e-6,
            "z: got {}, expected {}",
            r.z,
            expected_z
        );

        let expected_p = 0.13621686984456766_f64;
        assert!(
            (r.p_value - expected_p).abs() < 1e-6,
            "p_value: got {}, expected {}",
            r.p_value,
            expected_p
        );
    }

    // Zero-difference fixture: one zero difference forces NormalApprox even
    // though the remaining ranks are untied and n_effective = 5 <= 25.
    //
    // a = [100, 101, 98, 103, 96, 105], b = [100]*6
    // d = a - b = [0, 1, -2, 3, -4, 5], |d| = [0, 1, 2, 3, 4, 5]
    // All 6 |d| values distinct (Pratt: rank over all n including the
    // zero) -> ranks by position: [1, 2, 3, 4, 5, 6].
    //
    // t0 = 1, n = 6, n_effective = 5. Zero present -> NOT exact-eligible.
    //
    // Drop the zero-d position (pos0) from the W+/W- sums:
    //   pos1: d=+1, rank=2; pos2: d=-2, rank=3; pos3: d=+3, rank=4;
    //   pos4: d=-4, rank=5; pos5: d=+5, rank=6
    // W+ = 2+4+6 = 12
    // W- = 3+5 = 8
    // Check: W+ + W- = 20 = sum of nonzero ranks (2+3+4+5+6=20)  [OK]
    // W = min(12, 8) = 8
    //
    // mu = [n(n+1) - t0(t0+1)]/4 = [6*7 - 1*2]/4 = 40/4 = 10
    // Nonzero-|d| tie groups: {1,2,3,4,5} all distinct -> tie_sum = 0
    // sigma^2 = [n(n+1)(2n+1) - t0(t0+1)(2t0+1)]/24
    //   = [6*7*13 - 1*2*3]/24 = [546-6]/24 = 540/24 = 22.5
    // sigma = sqrt(22.5) = 4.743416490252569...
    // z = (8 - 10 + 0.5)/4.743416490252569 = -1.5/4.743416490252569
    //   = -0.31622776601683794...
    // p = min(1, 2*normal_cdf(z)) = 0.7518296340458492...
    // (cross-checked against SciPy's wilcoxon(..., zero_method="pratt",
    // correction=True, mode="approx") on this fixture: statistic=8.0,
    // pvalue=0.7518296340458492 -- see task-6-report.md)
    #[test]
    fn zeros_fall_back_to_normal_approx() {
        let a = vec![100.0, 101.0, 98.0, 103.0, 96.0, 105.0];
        let b = vec![100.0; 6];

        let r = wilcoxon_signed_rank(&a, &b).expect("valid fixture");
        assert_eq!(r.method, WilcoxonMethod::NormalApprox);
        assert_eq!(r.n_effective, 5);
        assert!(
            (r.w_statistic - 8.0).abs() < 1e-12,
            "w_statistic: got {}",
            r.w_statistic
        );

        let expected_z = -0.31622776601683794_f64;
        assert!(
            (r.z - expected_z).abs() < 1e-6,
            "z: got {}, expected {}",
            r.z,
            expected_z
        );

        let expected_p = 0.7518296340458492_f64;
        assert!(
            (r.p_value - expected_p).abs() < 1e-6,
            "p_value: got {}, expected {}",
            r.p_value,
            expected_p
        );
    }

    // n_effective = 26 > 25: NormalApprox regardless of the (untied,
    // zero-free) ranks.
    //
    // a[i] = 100 + i for i = 1..=26, b = [100]*26 -> d_i = i (all
    // positive, distinct: |d| = 1..26, no ties, no zeros).
    //
    // t0 = 0, n = 26, n_effective = 26 > 25 -> NOT exact-eligible.
    //
    // W+ = sum(1..26) = 26*27/2 = 351, W- = 0, W = min(351, 0) = 0.
    // mu = n(n+1)/4 = 26*27/4 = 175.5
    // sigma^2 = n(n+1)(2n+1)/24 = 26*27*53/24 = 37206/24 = 1550.25
    // sigma = sqrt(1550.25) = 39.37321424522006...
    // z = (0 - 175.5 + 0.5)/39.37321424522006 = -175/39.37321424522006
    //   = -4.444646020263513...
    // p = min(1, 2*normal_cdf(z)) = 8.803669768964184e-06...
    // (cross-checked against SciPy: statistic=0.0,
    // pvalue=8.803669768907958e-06 -- see task-6-report.md; the ~5e-14
    // relative discrepancy is within the crate's erf-approximation error
    // budget, well inside the 1e-6 tolerance used below)
    #[test]
    fn n_26_uses_normal_approx() {
        let a: Vec<f64> = (1..=26).map(|i| 100.0 + i as f64).collect();
        let b = vec![100.0; 26];

        let r = wilcoxon_signed_rank(&a, &b).expect("valid fixture");
        assert_eq!(r.method, WilcoxonMethod::NormalApprox);
        assert_eq!(r.n_effective, 26);
        assert!(
            (r.w_statistic - 0.0).abs() < 1e-12,
            "w_statistic: got {}",
            r.w_statistic
        );

        let expected_p = 8.803669768964184e-06_f64;
        assert!(
            (r.p_value - expected_p).abs() < 1e-6,
            "p_value: got {}, expected {}",
            r.p_value,
            expected_p
        );
    }

    // DP-matches-closed-form fixture (Task 6 brief step 1): n=3, W+
    // distribution over subsets of {1,2,3} has sums 0..6 with counts
    // 1,1,1,2,1,1,1 (total 8 = 2^3). P(W+ <= 1) two-sided = 2*2/8 = 0.5.
    //
    // a = [99, 102, 103], b = [100]*3
    // d = a - b = [-1, 2, 3], |d| = [1, 2, 3] (distinct, ranks = |d|).
    // t0 = 0, n = 3, n_effective = 3 (< 5, but exact-eligible, so the
    // n_effective < 5 error does NOT apply -- exercising the public API at
    // n=3 is exactly the point of this fixture).
    //
    // W- (only negative d, |d|=1 -> rank 1) = 1
    // W+ (d=2,3 -> ranks 2,3) = 5
    // Check: 1+5=6=n(n+1)/2=3*4/2=6  [OK]
    // W = min(1,5) = 1
    //
    // Exact p = min(1, 2*count(sum<=1)/2^3) = 2*2/8 = 0.5 (exact dyadic
    // rational). Independently cross-checked against SciPy's
    // `wilcoxon([99,102,103], [100]*3, zero_method="wilcox",
    // correction=False, mode="exact")`, which returns statistic=1.0,
    // pvalue=0.5 -- see task-6-report.md.
    #[test]
    fn exact_distribution_dp_matches_closed_form_small_case() {
        let a = vec![99.0, 102.0, 103.0];
        let b = vec![100.0; 3];

        let r = wilcoxon_signed_rank(&a, &b).expect("n=3 is exact-eligible, not an error");
        assert_eq!(r.method, WilcoxonMethod::Exact);
        assert_eq!(r.n_effective, 3);
        assert_eq!(r.w_statistic, 1.0);
        assert_eq!(r.p_value, 0.5); // exact dyadic rational -- bit-exact assert
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
        // t0 = 2 (zeros present) -> not exact-eligible -> unchanged
        // NormalApprox path; see the exact-eligibility carve-out doc on
        // wilcoxon_signed_rank.
        assert_eq!(result.method, WilcoxonMethod::NormalApprox);
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
        // Tied |d| ranks (two size-2 tie groups) -> not exact-eligible ->
        // unchanged NormalApprox path.
        assert_eq!(result.method, WilcoxonMethod::NormalApprox);
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
    fn wilcoxon_errors_on_nan() {
        let a = vec![1.0, f64::NAN, 3.0];
        let b = vec![0.0, 0.0, 0.0];
        let err = wilcoxon_signed_rank(&a, &b).expect_err("NaN must be rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("wilcoxon_signed_rank"), "message should name the function: {msg}");
        assert!(msg.contains("non-finite"), "message should say non-finite: {msg}");
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

    // *** TASK 6 BEHAVIOR CHANGE (M2d-1, flagged prominently per the
    // brief's disclosure rule): renamed from
    // `wilcoxon_errors_on_too_few_effective_pairs`. n = 4, all nonzero, no
    // ties -> n_effective = 4 < 5, which USED TO error unconditionally
    // (before Task 6, the crate had no exact small-n distribution to fall
    // back on -- see the superseded doc comment: "the small-sample exact
    // tables are not implemented in this crate -- deferred; see M2c task
    // brief"). Task 6 implements exactly that deferred capability: n=4 is
    // exact-eligible (t0=0, no ties, n_effective=4 <= 25), so the call now
    // SUCCEEDS via the exact distribution instead of erroring. This is a
    // direct, intended consequence of Task 6 (also required by the brief's
    // own `exact_distribution_dp_matches_closed_form_small_case` test,
    // which likewise exercises n=3 < 5 via this same public API and must
    // succeed) -- flagged here since it changes a previously-pinned Err
    // assertion to Ok.
    //
    // d = a - b = [1, 2, 3, 4], |d| = [1, 2, 3, 4] (distinct, ranks = |d|).
    // All d > 0, so W+ = 1+2+3+4 = 10, W- = 0, W = min(10, 0) = 0.
    //
    // Exact p, W = 0, n = 4: count(sum <= 0) = 1 (only the empty subset
    // sums to 0; the pinned DP over subsets of {1,2,3,4}). Total = 2^4 =
    // 16. p = min(1, 2 * 1 / 16) = 2/16 = 0.125 (exact dyadic rational).
    // Independently cross-checked against SciPy's
    // `wilcoxon([1,2,3,4], [0,0,0,0], zero_method="wilcox",
    // correction=False, mode="exact")`, which returns statistic=0.0,
    // pvalue=0.125 -- see task-6-report.md.
    #[test]
    fn wilcoxon_few_effective_pairs_now_exact_eligible() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![0.0, 0.0, 0.0, 0.0];

        let result = wilcoxon_signed_rank(&a, &b).expect("n=4 is now exact-eligible, not an error");

        assert_eq!(result.n_effective, 4);
        assert_eq!(result.method, WilcoxonMethod::Exact);
        assert!(
            (result.w_statistic - 0.0).abs() < 1e-12,
            "w_statistic: got {}",
            result.w_statistic
        );
        let expected_p = 0.125_f64;
        assert_eq!(
            result.p_value, expected_p,
            "exact dyadic rational -- bit-exact assert"
        );
    }

    #[test]
    fn wilcoxon_errors_on_too_few_effective_pairs_with_zeros() {
        // n = 8, but 4 zero pairs leave n_effective = 4 < 5. Unaffected by
        // Task 6: t0 = 4 != 0 makes this NOT exact-eligible regardless of
        // n_effective, so it still falls through to the (unchanged)
        // NormalApprox-path n_effective < 5 error.
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
        let delta = cliffs_delta(&a, &b).expect("valid fixture");
        assert!((delta - 0.0).abs() < 1e-12, "delta: got {}", delta);
    }

    #[test]
    fn cliffs_delta_disjoint_extremes() {
        // Every element of a is smaller than every element of b ->
        // delta = -1 (sign convention: a-dominates-lower gives negative).
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![10.0, 20.0, 30.0];
        let delta_ab = cliffs_delta(&a, &b).expect("valid fixture");
        assert!((delta_ab - (-1.0)).abs() < 1e-12, "got {}", delta_ab);

        // Reversed: every element of a is larger than every element of b
        // -> delta = +1.
        let delta_ba = cliffs_delta(&b, &a).expect("valid fixture");
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
        let delta = cliffs_delta(&a, &b).expect("valid fixture");
        let expected = -5.0 / 9.0;
        assert!(
            (delta - expected).abs() < 1e-12,
            "got {}, expected {}",
            delta,
            expected
        );
    }

    #[test]
    fn cliffs_delta_errors_on_nan() {
        let a = vec![1.0, f64::NAN, 3.0];
        let b = vec![2.0, 3.0, 4.0];
        let err = cliffs_delta(&a, &b).expect_err("NaN must be rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("cliffs_delta"), "message should name the function: {msg}");
        assert!(msg.contains("non-finite"), "message should say non-finite: {msg}");
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
