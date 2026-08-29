//! Bayesian signed-rank test with a region of practical equivalence (ROPE),
//! and Plackett-Luce ranking of full permutations (Hunter 2004 MM
//! algorithm).
//!
//! See the crate-level documentation in `lib.rs` for the general
//! conventions used elsewhere in this crate.

use sezgi_core::rng::RngStream;

use crate::{check_finite, StatsError};

/// Result of [`bayesian_signed_rank`]: posterior probability mass assigned
/// to each of the three regions of the (A - B) pseudo-median, relative to a
/// region of practical equivalence (ROPE) of half-width `rope`.
#[derive(Debug, Clone, PartialEq)]
pub struct BayesSignedRankResult {
    /// P(pseudo-median of A-B < -rope) — A is practically better than B.
    pub p_left: f64,
    /// P(|pseudo-median of A-B| <= rope) — A and B are practically
    /// equivalent.
    pub p_rope: f64,
    /// P(pseudo-median of A-B > rope) — B is practically better than A.
    pub p_right: f64,
}

/// Bayesian signed-rank test with a region of practical equivalence (ROPE),
/// following the Dirichlet-weighted Monte Carlo scheme of Benavoli et al.
/// (2017) — **this is a documented sezgi simplification**, not a literal
/// reimplementation of their algorithm.
///
/// # Model (sezgi simplification of Benavoli et al. 2017)
///
/// Let `n = a.len() == b.len()`, `d_i = a_i - b_i` for `i = 1..=n`, plus a
/// pseudo-observation `d_0 = 0` representing the Bayesian nonparametric
/// prior mass at zero (Dirichlet Process prior with a point mass at 0, as
/// in Benavoli et al.). This gives `m = n + 1` "observations" `d_0..=d_n`.
///
/// Each Monte Carlo iteration:
///
/// 1. **Pinned draw order**: draw `m = n + 1` uniforms `u_0, u_1, ..., u_n`
///    via `rng.next_f64()`, in that index order (`u_0` for the pseudo-
///    observation first, then `u_1..=u_n` for the real differences in
///    input order). Each `u` is clamped to `>= 1e-300` before taking its
///    log (guards `u == 0`, which would otherwise produce `-inf`).
/// 2. Convert to Dirichlet(1, ..., 1) weights via the standard
///    exponential-spacings construction: `w_i = -ln(u_i)`, then normalize
///    so `sum(w_i) = 1`.
/// 3. For **every pair `(i, j)` with `i <= j`** (including `i == j`),
///    compute the midpoint `m_ij = (d_i + d_j) / 2` and its symmetric
///    weight contribution `w_i * w_j` for `i == j`, or `2 * w_i * w_j` for
///    `i != j` (accounting for both `(i,j)` and `(j,i)` in the implicit
///    full `m x m` weighted sum, whose total mass is exactly
///    `(sum(w_i))^2 = 1`). This weighted sum over pairwise midpoints is the
///    Monte Carlo draw's realization of the posterior pseudo-median
///    distribution of `A - B`.
/// 4. Accumulate each pair's weight into one of three masses: `left` if
///    `m_ij < -rope`, `right` if `m_ij > rope`, otherwise `rope` (i.e. the
///    boundary `|m_ij| == rope` is classified as `rope`, not `left`/
///    `right` — see the `rope_zero_splits` test for the `rope = 0`
///    boundary case).
/// 5. The **iteration votes** for whichever of the three masses is
///    largest. Tie-break (deterministic, for exact floating-point ties):
///    `left` beats `rope` beats `right`.
///
/// After `samples` iterations, `p_left = votes_left / samples`, and
/// likewise for `p_rope` and `p_right`.
///
/// This whole-vote-per-iteration scheme (rather than averaging masses
/// directly across iterations) matches the region classification used by
/// Benavoli et al.'s decision rule while keeping the RNG draw order fully
/// pinned for reproducibility across sezgi runs.
///
/// # Determinism
/// `RngStream::from_master(master_seed, &[0])` is used as a single stream;
/// the same `(a, b, rope, samples, master_seed)` always produces a
/// bit-identical result.
///
/// # Errors
/// Returns `StatsError::InvalidInput` if `a.len() != b.len()`, if `a` (and
/// `b`) is empty, if `samples == 0`, or if `rope < 0.0`.
///
/// # Reference
/// Benavoli, A., Corani, G., Demšar, J., & Zaffalon, M. (2017). "Time for a
/// Change: a Tutorial for Comparing Multiple Classifiers Through Bayesian
/// Analysis." *Journal of Machine Learning Research*, 18(77), 1-36.
pub fn bayesian_signed_rank(
    a: &[f64],
    b: &[f64],
    rope: f64,
    samples: u64,
    master_seed: u64,
) -> Result<BayesSignedRankResult, StatsError> {
    check_finite("bayesian_signed_rank", a.iter().chain(b.iter()).copied())?;

    if a.len() != b.len() {
        return Err(StatsError::InvalidInput(format!(
            "bayesian_signed_rank requires paired samples of equal length, got {} and {}",
            a.len(),
            b.len()
        )));
    }
    if a.is_empty() {
        return Err(StatsError::InvalidInput(
            "bayesian_signed_rank requires at least 1 pair, got 0".to_string(),
        ));
    }
    if samples == 0 {
        return Err(StatsError::InvalidInput(
            "bayesian_signed_rank requires samples > 0, got 0".to_string(),
        ));
    }
    if rope < 0.0 {
        return Err(StatsError::InvalidInput(format!(
            "bayesian_signed_rank requires rope >= 0.0, got {}",
            rope
        )));
    }

    let n = a.len();
    // d[0] = pseudo-observation 0; d[1..=n] = a_i - b_i.
    let mut d = vec![0.0_f64; n + 1];
    for i in 0..n {
        d[i + 1] = a[i] - b[i];
    }
    let m = n + 1;

    let mut rng = RngStream::from_master(master_seed, &[0]);

    let mut votes_left = 0_u64;
    let mut votes_rope = 0_u64;
    let mut votes_right = 0_u64;

    let mut w = vec![0.0_f64; m];
    for _ in 0..samples {
        // Pinned draw order: i = 0..=n, one next_f64() call per weight.
        let mut sum_w = 0.0_f64;
        for wi in w.iter_mut() {
            let u = rng.next_f64().max(1e-300);
            let e = -u.ln();
            *wi = e;
            sum_w += e;
        }
        for wi in w.iter_mut() {
            *wi /= sum_w;
        }

        let mut mass_left = 0.0_f64;
        let mut mass_rope = 0.0_f64;
        let mut mass_right = 0.0_f64;
        for i in 0..m {
            for j in i..m {
                let mid = (d[i] + d[j]) / 2.0;
                let weight = if i == j {
                    w[i] * w[j]
                } else {
                    2.0 * w[i] * w[j]
                };
                if mid < -rope {
                    mass_left += weight;
                } else if mid > rope {
                    mass_right += weight;
                } else {
                    mass_rope += weight;
                }
            }
        }

        // Vote for the largest mass; deterministic tie-break left > rope > right.
        let max_mass = mass_left.max(mass_rope).max(mass_right);
        if mass_left == max_mass {
            votes_left += 1;
        } else if mass_rope == max_mass {
            votes_rope += 1;
        } else {
            votes_right += 1;
        }
    }

    let samples_f = samples as f64;
    Ok(BayesSignedRankResult {
        p_left: votes_left as f64 / samples_f,
        p_rope: votes_rope as f64 / samples_f,
        p_right: votes_right as f64 / samples_f,
    })
}

/// Result of [`plackett_luce`].
#[derive(Debug, Clone, PartialEq)]
pub struct PlackettLuceResult {
    /// Estimated Plackett-Luce "worth" parameters, one per item, normalized
    /// to sum to 1.
    pub worths: Vec<f64>,
    /// PL-model probability that each item is ranked first (the rank-1
    /// choice probability). Under the PL model / Luce's choice axiom this
    /// is numerically identical to `worths` (`p_best[i] == worths[i]`),
    /// since `P(item i chosen first | all k items) = w_i / sum(w)` and
    /// `sum(w) = 1` by construction. Kept as a distinct field for API
    /// clarity (documented here rather than silently aliased).
    pub p_best: Vec<f64>,
    /// Number of MM iterations actually run (converged when
    /// `max|delta w| < 1e-10`, capped at 10_000).
    pub iterations: usize,
}

const PL_MAX_ITERATIONS: usize = 10_000;
const PL_CONVERGENCE_TOL: f64 = 1e-10;
const PL_WEIGHT_FLOOR: f64 = 1e-300;

/// Plackett-Luce maximum-likelihood ranking via Hunter's (2004) Minorize-
/// Maximize (MM) algorithm.
///
/// # Input
/// `rankings`: each row is a **full ranking** of the same `k` items — a
/// permutation of `0..k`, best (rank 1) first.
///
/// # Algorithm (Hunter 2004, standard PL MM update)
///
/// For a full ranking of `k` items there are `k - 1` "stages" `s = 0..k-2`
/// (0-indexed): stage `s`'s **choice set** is the items ranked at positions
/// `s..k` (i.e. not yet "chosen"), and the item **chosen** at stage `s` is
/// the one ranked at position `s`. The item ranked last (position `k-1`) is
/// never itself the chosen item at any stage — it is only ever a bystander
/// in earlier choice sets — so for a full ranking, every item except the
/// last-ranked one is "chosen" exactly once.
///
/// - `W_t` = total number of stages (summed across all rankings) at which
///   item `t` is the chosen item = number of rankings in which `t` is NOT
///   ranked last (constant across MM iterations; computed once).
/// - `D_t` = sum, over every stage (across all rankings) whose choice set
///   contains `t`, of `1 / (sum of current w_u for u in that choice set)`.
///
/// Each iteration updates `w_t <- W_t / D_t` for every item, then
/// renormalizes so `sum(w) = 1`. This is repeated until
/// `max_t |w_t^(new) - w_t^(old)| < 1e-10` or 10_000 iterations have run.
///
/// `D_t` is computed efficiently per ranking via a suffix-sum /
/// prefix-of-reciprocals pass: for a ranking with items `r_0, r_1, ...,
/// r_{k-1}` (best to worst) and current weights `w`, let
/// `S_s = sum_{u=s}^{k-1} w[r_u]` (the choice-set weight sum at stage `s`).
/// The item at position `p` belongs to the choice sets of stages
/// `s = 0..=min(p, k-2)`, so its contribution to `D_{r_p}` from this
/// ranking is `sum_{s=0}^{min(p,k-2)} 1/S_s` — a running cumulative sum as
/// `p` increases from `0` to `k-1`.
///
/// # Degenerate items (guard against NaN)
/// An item that is **always ranked last** has `W_t = 0`, so its MM update
/// gives `w_t = 0 / D_t = 0` — correct maximum-likelihood behavior (an
/// item that is never chosen has zero estimated worth). To avoid `0` ever
/// participating in a later division by a choice-set sum of exactly zero,
/// every updated weight is floored at `1e-300` before renormalization;
/// `D_t` itself is always strictly positive since every item appears in
/// every ranking (enforced by input validation) and all weights are
/// strictly positive.
///
/// # Errors
/// Returns `StatsError::InvalidInput` if:
/// - `rankings` is empty,
/// - `k = rankings[0].len() < 2`,
/// - any row's length differs from `k` (ragged input), or
/// - any row is not a valid permutation of `0..k` (out-of-range or
///   repeated item).
///
/// No `check_finite` call: `rankings` are `usize` permutations of item
/// indices, not `f64` data, so there is no non-finite value they could ever
/// contain.
///
/// # Reference
/// Hunter, D. R. (2004). "MM algorithms for generalized Bradley-Terry
/// models." *The Annals of Statistics*, 32(1), 384-406.
pub fn plackett_luce(rankings: &[Vec<usize>]) -> Result<PlackettLuceResult, StatsError> {
    if rankings.is_empty() {
        return Err(StatsError::InvalidInput(
            "plackett_luce requires at least 1 ranking, got 0".to_string(),
        ));
    }

    let k = rankings[0].len();
    if k < 2 {
        return Err(StatsError::InvalidInput(format!(
            "plackett_luce requires at least 2 items (k), got {}",
            k
        )));
    }

    for (r, ranking) in rankings.iter().enumerate() {
        if ranking.len() != k {
            return Err(StatsError::InvalidInput(format!(
                "plackett_luce: ranking {} has {} items, expected {} (ragged input: all rankings must rank the same k items)",
                r,
                ranking.len(),
                k
            )));
        }
        let mut seen = vec![false; k];
        for &item in ranking {
            if item >= k {
                return Err(StatsError::InvalidInput(format!(
                    "plackett_luce: ranking {} contains out-of-range item {} (k={})",
                    r, item, k
                )));
            }
            if seen[item] {
                return Err(StatsError::InvalidInput(format!(
                    "plackett_luce: ranking {} is not a valid permutation of 0..{} (item {} repeated)",
                    r, k, item
                )));
            }
            seen[item] = true;
        }
    }

    // W_t: number of rankings in which item t is NOT last. Constant across
    // MM iterations, computed once.
    let mut w_num = vec![0.0_f64; k];
    for ranking in rankings {
        for &item in &ranking[..k - 1] {
            w_num[item] += 1.0;
        }
    }

    let mut w = vec![1.0 / k as f64; k];
    let mut iterations = 0_usize;

    loop {
        let mut denom = vec![0.0_f64; k];
        for ranking in rankings {
            // suffix[s] = sum of w over items ranked at positions s..k.
            let mut suffix = vec![0.0_f64; k + 1];
            for s in (0..k).rev() {
                suffix[s] = suffix[s + 1] + w[ranking[s]];
            }
            // Running cumulative sum of 1/S_s for s = 0..=min(p, k-2), as p
            // increases; assigned to the item at each position p.
            let mut running = 0.0_f64;
            for (p, &item) in ranking.iter().enumerate() {
                if p <= k - 2 {
                    running += 1.0 / suffix[p];
                }
                denom[item] += running;
            }
        }

        let mut w_new = vec![0.0_f64; k];
        for t in 0..k {
            w_new[t] = (w_num[t] / denom[t]).max(PL_WEIGHT_FLOOR);
        }
        let sum: f64 = w_new.iter().sum();
        for wt in w_new.iter_mut() {
            *wt /= sum;
        }

        let max_delta = w_new
            .iter()
            .zip(w.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        w = w_new;
        iterations += 1;

        if max_delta < PL_CONVERGENCE_TOL || iterations >= PL_MAX_ITERATIONS {
            break;
        }
    }

    let p_best = w.clone();
    Ok(PlackettLuceResult {
        worths: w,
        p_best,
        iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Degenerate fixture: a == b elementwise, so every d_i = 0 (including
    // the pseudo-observation d_0 = 0). Every pairwise midpoint m_ij = 0
    // regardless of the Dirichlet weights drawn, so |m_ij| <= rope holds
    // for ANY rope >= 0 -- mass_rope = 1.0 (all weight, since the total
    // pairwise weighted mass is (sum w)^2 = 1) on every single iteration.
    // p_rope must therefore be (near-)exactly 1.0.
    #[test]
    fn all_zero_differences_vote_rope() {
        let a = vec![5.0; 10];
        let b = vec![5.0; 10];
        let result = bayesian_signed_rank(&a, &b, 0.1, 2000, 42).expect("valid fixture");
        assert!(result.p_rope > 0.99, "p_rope: got {}", result.p_rope);
        assert!((result.p_left).abs() < 1e-12);
        assert!((result.p_right).abs() < 1e-12);
    }

    // Clearly separated fixture: a_i = b_i - 10 for all 10 pairs, so every
    // real difference d_i = -10 (i = 1..=10), plus the pseudo-observation
    // d_0 = 0. For 11 items (0..=10) under Dirichlet(1,...,1) weights, the
    // only pair landing in the ROPE (|mid| <= 0.1) is (0,0) itself (mid=0);
    // every other pair -- (i,i) with mid=-10, (0,j) with mid=-5, and (i,j)
    // i,j>=1 with mid=-10 -- lands in "left". mass_rope = w_0^2 has
    // expectation 2/(11*12) = 1/66 ~= 0.015 under Dirichlet(1,...,1) with
    // 11 categories, so "left" wins the per-iteration vote with very high
    // probability; p_left should be well above 0.95 over 2000 samples.
    #[test]
    fn separated_votes_left() {
        let b: Vec<f64> = (1..=10).map(|i| i as f64 * 10.0).collect();
        let a: Vec<f64> = b.iter().map(|&x| x - 10.0).collect();
        let result = bayesian_signed_rank(&a, &b, 0.1, 2000, 42).expect("valid fixture");
        assert!(result.p_left > 0.95, "p_left: got {}", result.p_left);
    }

    #[test]
    fn deterministic_given_seed() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.5, 1.5, 3.5, 3.0, 6.0];
        let r1 = bayesian_signed_rank(&a, &b, 0.2, 500, 777).expect("valid fixture");
        let r2 = bayesian_signed_rank(&a, &b, 0.2, 500, 777).expect("valid fixture");
        assert_eq!(r1, r2, "same (data, rope, samples, seed) must be bit-identical");
    }

    // Boundary convention sanity: rope = 0.0 exactly. a == b elementwise so
    // every midpoint is exactly 0.0, and the classification rule is
    // `|m| <= rope` for the rope bucket (not strict `<`), so 0.0 <= 0.0
    // still lands in "rope". Documents that rope's boundary is inclusive.
    #[test]
    fn rope_zero_splits() {
        let a = vec![3.0; 6];
        let b = vec![3.0; 6];
        let result = bayesian_signed_rank(&a, &b, 0.0, 500, 1).expect("valid fixture");
        assert!(
            (result.p_rope - 1.0).abs() < 1e-9,
            "p_rope: got {}",
            result.p_rope
        );
    }

    #[test]
    fn errors_on_nan() {
        let a = vec![1.0, f64::NAN, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let err = bayesian_signed_rank(&a, &b, 0.1, 100, 1).expect_err("NaN must be rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("bayesian_signed_rank"), "message should name the function: {msg}");
        assert!(msg.contains("non-finite"), "message should say non-finite: {msg}");
    }

    #[test]
    fn errors_on_length_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        assert!(matches!(
            bayesian_signed_rank(&a, &b, 0.1, 100, 1),
            Err(StatsError::InvalidInput(_))
        ));
    }

    #[test]
    fn errors_on_empty_input() {
        let a: Vec<f64> = vec![];
        let b: Vec<f64> = vec![];
        assert!(matches!(
            bayesian_signed_rank(&a, &b, 0.1, 100, 1),
            Err(StatsError::InvalidInput(_))
        ));
    }

    #[test]
    fn errors_on_zero_samples() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0];
        assert!(matches!(
            bayesian_signed_rank(&a, &b, 0.1, 0, 1),
            Err(StatsError::InvalidInput(_))
        ));
    }

    #[test]
    fn errors_on_negative_rope() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0];
        assert!(matches!(
            bayesian_signed_rank(&a, &b, -0.1, 100, 1),
            Err(StatsError::InvalidInput(_))
        ));
    }

    // Two-algorithm fixture: A (item 0) beats B (item 1) in 3 of 4
    // rankings. For k=2 every ranking has a single stage whose choice set
    // is both items, so W_A=3, W_B=1, and D_A=D_B=4/(w_A+w_B) identically
    // (both items are in that single choice set every time regardless of
    // which one is listed first). After renormalizing,
    // w_A = W_A/(W_A+W_B) = 3/4 exactly -- i.e. PL with k=2 reduces exactly
    // to the Bradley-Terry / raw win fraction.
    #[test]
    fn two_algorithms_reduce_to_win_fraction() {
        let rankings = vec![vec![0, 1], vec![0, 1], vec![0, 1], vec![1, 0]];
        let result = plackett_luce(&rankings).expect("valid fixture");
        assert!(
            (result.worths[0] - 0.75).abs() < 0.02,
            "w_A: got {}",
            result.worths[0]
        );
        assert!(result.worths[0] > result.worths[1]);
    }

    // Three-algorithm fixture with a known dominance chain: item 0 beats
    // item 1 which beats item 2 in the large majority of rankings.
    #[test]
    fn dominance_chain_monotone() {
        let rankings = vec![
            vec![0, 1, 2],
            vec![0, 1, 2],
            vec![0, 1, 2],
            vec![1, 0, 2],
        ];
        let result = plackett_luce(&rankings).expect("valid fixture");
        assert!(
            result.worths[0] > result.worths[1],
            "w0={} w1={}",
            result.worths[0],
            result.worths[1]
        );
        assert!(
            result.worths[1] > result.worths[2],
            "w1={} w2={}",
            result.worths[1],
            result.worths[2]
        );
    }

    // Item 2 is ranked last in every single ranking -> W_2 = 0 -> its MM
    // update always yields 0 (floored to 1e-300 pre-normalization), so its
    // worth should collapse to (numerically) zero.
    #[test]
    fn always_last_worth_vanishes() {
        let rankings = vec![
            vec![0, 1, 2],
            vec![1, 0, 2],
            vec![0, 1, 2],
            vec![1, 0, 2],
        ];
        let result = plackett_luce(&rankings).expect("valid fixture");
        assert!(result.worths[2] < 1e-6, "w2: got {}", result.worths[2]);
    }

    #[test]
    fn p_best_sums_to_one() {
        let rankings = vec![vec![0, 1, 2], vec![2, 1, 0], vec![1, 0, 2]];
        let result = plackett_luce(&rankings).expect("valid fixture");
        let sum: f64 = result.p_best.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum: got {}", sum);
        assert_eq!(result.p_best, result.worths);
    }

    #[test]
    fn errors_on_empty_rankings() {
        let rankings: Vec<Vec<usize>> = vec![];
        assert!(matches!(
            plackett_luce(&rankings),
            Err(StatsError::InvalidInput(_))
        ));
    }

    #[test]
    fn errors_on_k_less_than_2() {
        let rankings = vec![vec![0], vec![0]];
        assert!(matches!(
            plackett_luce(&rankings),
            Err(StatsError::InvalidInput(_))
        ));
    }

    #[test]
    fn errors_on_ragged_row() {
        let rankings = vec![vec![0, 1, 2], vec![0, 1]];
        assert!(matches!(
            plackett_luce(&rankings),
            Err(StatsError::InvalidInput(_))
        ));
    }

    #[test]
    fn errors_on_out_of_range_item() {
        let rankings = vec![vec![0, 1, 3]]; // k=3, item 3 is out of range
        assert!(matches!(
            plackett_luce(&rankings),
            Err(StatsError::InvalidInput(_))
        ));
    }

    #[test]
    fn errors_on_repeated_item() {
        let rankings = vec![vec![0, 0, 1]];
        assert!(matches!(
            plackett_luce(&rankings),
            Err(StatsError::InvalidInput(_))
        ));
    }
}
