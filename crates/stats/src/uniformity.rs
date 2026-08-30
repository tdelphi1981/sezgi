//! One-sample uniformity goodness-of-fit tests against U(0,1): the
//! Kolmogorov-Smirnov (KS) test and the Anderson-Darling (AD) case-0 test
//! (fully specified reference distribution, no parameters estimated from
//! the data).
//!
//! Callers are expected to have already mapped their domain-specific values
//! into `[0,1]` (e.g. via a PIT/CDF transform); that mapping is out of
//! scope for this module (it lives in the `bias` crate). A small numerical
//! tolerance is applied here purely to absorb floating-point noise at the
//! `[0,1]` boundary from that upstream mapping.
//!
//! # Provenance
//!
//! ## KS: statistic, `Q_KS` series, and the `lambda` correction
//!
//! Source: Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery,
//! B. P. (1992). *Numerical Recipes in Fortran 77: The Art of Scientific
//! Computing*, 2nd ed. Cambridge University Press. Section 14.3, "Are Two
//! Distributions Different?", pp. 617-620, equations (14.3.7) and (14.3.9),
//! and the `probks` FUNCTION listing on p. 619. Equation (14.3.9) there
//! cites Stephens, M. A. (1970). "Use of the Kolmogorov-Smirnov, Cramer-von
//! Mises and Related Statistics Without Extensive Tables." *Journal of the
//! Royal Statistical Society, Series B*, 32(1), 115-122, as reference [1]
//! for the approximation.
//!
//! Quoted verbatim (converting the book's typeset notation to prose):
//!
//! ```text
//! Q_KS(lambda) = 2 * sum_{j=1}^{infinity} (-1)^(j-1) * exp(-2 j^2 lambda^2)   (14.3.7)
//!
//! Probability(D > observed) = Q_KS( [sqrt(N_e) + 0.12 + 0.11/sqrt(N_e)] * D )   (14.3.9)
//! ```
//!
//! where `N_e` is the effective sample size (`N_e = n` for the one-sample
//! case this module implements). The book states: "The nature of the
//! approximation involved in (14.3.9) is that it becomes asymptotically
//! accurate as the `N_e` becomes large, but is already quite good for
//! `N_e >= 4`."
//!
//! The `probks` FUNCTION (p. 619) evaluates (14.3.7) as an alternating
//! series with early termination, which this module reproduces exactly
//! (`q_ks`, PINNED):
//!
//! ```text
//! FUNCTION probks(alam)
//!   PARAMETER (EPS1=0.001, EPS2=1.e-8)
//!   a2 = -2.*alam**2
//!   fac = 2.
//!   probks = 0.
//!   termbf = 0.        ! previous term in sum
//!   do j = 1, 100
//!     term = fac*exp(a2*j**2)
//!     probks = probks + term
//!     if (abs(term).le.EPS1*termbf .or. abs(term).le.EPS2*probks) return
//!     fac = -fac        ! alternating signs in sum
//!     termbf = abs(term)
//!   enddo
//!   probks = 1.        ! get here only by failing to converge
//! END
//! ```
//!
//! ## AD (case 0): statistic and the asymptotic p-value mapping
//!
//! Source: Marsaglia, G., & Marsaglia, J. (2004). "Evaluating the
//! Anderson-Darling Distribution." *Journal of Statistical Software*, 9(2),
//! 1-5. Page 1 gives the statistic (their `A_n`, our `a2`) for an ordered
//! set `x_1 < x_2 < ... < x_n` of purported U(0,1) variates, quoted
//! verbatim:
//!
//! ```text
//! A_n = -n - (1/n)[ln(x_1(1-x_n)) + 3 ln(x_2(1-x_{n-1}))
//!             + 5 ln(x_3(1-x_{n-2})) + ... + (2n-1) ln(x_n(1-x_1))]
//! ```
//!
//! i.e. `A_n = -n - (1/n) * sum_{i=1}^{n} (2i-1) [ln(x_i) + ln(1 - x_{n+1-i})]`.
//!
//! For the p-value, page 4 gives a closed-form, two-piece Horner-polynomial
//! approximation to the **limiting** (`n -> infinity`) CDF of `A_n`, which
//! the paper calls `adinf(z)` (lowercase, to distinguish it from the
//! full-precision `ADinf(z)` computed via their series method), quoted
//! verbatim including the paper's own stated accuracy:
//!
//! ```text
//! adinf(z) =
//!   for 0 < z < 2, with |error| < .000002:
//!     z^(-1/2) * e^(-1.2337141/z)
//!       * (2.00012 + (.247105 - (.0649821 - (.0347962
//!           - (.0116720 - .00168691 z) z) z) z) z)
//!   for 2 <= z < infinity, with |error| < .0000008:
//!     exp(-exp(1.0776 - (2.30695 - (.43424 - (.082433
//!           - (.008056 - .0003146 z) z) z) z) z))
//! ```
//!
//! `p_value = 1 - adinf(A_n)`.
//!
//! // sezgi simplification: the paper also provides a finite-`n`
//! // `errfix(n,x)` correction (piecewise, fit to n = 8,16,32,64,128, to
//! // within +/-.0005) layered on top of `adinf`; this module implements
//! // only the asymptotic `adinf` mapping (verified accuracy above), not
//! // `errfix`, so `ad_uniform`'s `p_value` is the `n -> infinity` limiting
//! // approximation and carries additional, undocumented finite-`n` error
//! // for small `n` (the paper reports the worst-case error of using
//! // `A_infinity` in place of the exact `A_n` distribution is about
//! // `.044/n`, "near the 33rd percentile"). Revisit if Task 4 needs
//! // tighter small-`n` accuracy.

use crate::{check_finite, StatsError};

/// Numerical tolerance for the `[0,1]` boundary check in
/// [`validate_uniform_sample`]: absorbs floating-point noise from an
/// upstream domain-to-`[0,1]` mapping, not part of either pinned formula.
const BOUNDARY_TOL: f64 = 1e-9;

/// Minimum sample size accepted by [`ks_uniform`] and [`ad_uniform`].
const MIN_N: usize = 5;

/// Result of [`ks_uniform`]. **PINNED** field set/names once merged.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KsResult {
    /// The Kolmogorov-Smirnov D statistic: `D = sup_x |F_emp(x) - F(x)|`.
    pub d: f64,
    /// Two-sided asymptotic p-value: `Q_KS([sqrt(n) + 0.12 + 0.11/sqrt(n)] * d)`.
    pub p_value: f64,
    /// Sample size.
    pub n: usize,
}

/// Result of [`ad_uniform`]. **PINNED** field set/names once merged.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdResult {
    /// The Anderson-Darling A^2 statistic (case 0: fully specified U(0,1)
    /// reference, no estimated parameters).
    pub a2: f64,
    /// Asymptotic (n -> infinity) p-value: `1 - adinf(a2)`.
    pub p_value: f64,
    /// Sample size.
    pub n: usize,
}

/// Shared input validation for [`ks_uniform`] and [`ad_uniform`]: checks
/// finiteness (via [`check_finite`]), the `n >= 5` minimum, and that every
/// value lies within `[0,1]` up to [`BOUNDARY_TOL`] (values just outside
/// that band, e.g. from floating-point noise in an upstream mapping, are
/// clamped into `[0,1]`; values further outside are rejected). Returns the
/// clamped values sorted ascending, ready for order-statistic use by either
/// test.
///
/// The out-of-tolerance error, if any, names the **original** (pre-sort)
/// index, matching [`check_finite`]'s convention.
fn validate_uniform_sample(func: &str, samples: &[f64]) -> Result<Vec<f64>, StatsError> {
    check_finite(func, samples.iter().copied())?;

    let n = samples.len();
    if n < MIN_N {
        return Err(StatsError::InvalidInput(format!(
            "{func} requires at least {MIN_N} samples, got {n}"
        )));
    }

    let mut sorted = Vec::with_capacity(n);
    for (i, &v) in samples.iter().enumerate() {
        if !(-BOUNDARY_TOL..=1.0 + BOUNDARY_TOL).contains(&v) {
            return Err(StatsError::InvalidInput(format!(
                "{func}: value at index {i} = {v} is outside [0,1] (tolerance {BOUNDARY_TOL})"
            )));
        }
        sorted.push(v.clamp(0.0, 1.0));
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("non-finite already rejected above"));
    Ok(sorted)
}

/// KS D statistic against U(0,1) for an ascending-sorted sample: for each
/// order statistic `x_(i)` (1-based `i`), compares `F(x_(i)) = x_(i)`
/// against both the empirical CDF just before (`(i-1)/n`) and just after
/// (`i/n`) its jump at `x_(i)`, per the `dt=max(abs(fo-ff),abs(fn-ff))` step
/// of NR's `ksone` (see module doc). `D` is the max `dt` over all `i`.
fn ks_statistic(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    let n_f = n as f64;
    let mut d = 0.0_f64;
    for (idx, &x) in sorted.iter().enumerate() {
        let i = (idx + 1) as f64;
        let f_before = (i - 1.0) / n_f;
        let f_after = i / n_f;
        let dt = (f_before - x).abs().max((f_after - x).abs());
        if dt > d {
            d = dt;
        }
    }
    d
}

/// `Q_KS(lambda)`, NR equation (14.3.7), evaluated via the `probks`
/// alternating-series algorithm (p. 619) exactly as quoted in the module
/// doc. **PINNED**: the series form, the `EPS1`/`EPS2` early-termination
/// thresholds, the 100-term cap, and the `1.0`-on-non-convergence fallback
/// must not change without a semver bump.
fn q_ks(lambda: f64) -> f64 {
    const EPS1: f64 = 0.001;
    const EPS2: f64 = 1e-8;

    let a2 = -2.0 * lambda * lambda;
    let mut fac = 2.0_f64;
    let mut prob = 0.0_f64;
    let mut termbf = 0.0_f64;

    for j in 1..=100 {
        let j_f = j as f64;
        let term = fac * (a2 * j_f * j_f).exp();
        prob += term;
        if term.abs() <= EPS1 * termbf || term.abs() <= EPS2 * prob {
            return prob;
        }
        fac = -fac;
        termbf = term.abs();
    }
    1.0
}

/// One-sample Kolmogorov-Smirnov test against U(0,1).
///
/// `D = sup_x |F_emp(x) - F(x)|`, computed exactly (both sup-branches at
/// every order statistic, see [`ks_statistic`]); `p_value` is the
/// asymptotic Kolmogorov distribution `Q_KS(lambda)` with
/// `lambda = (sqrt(n) + 0.12 + 0.11/sqrt(n)) * D` (NR eq. 14.3.9, `N_e = n`
/// for the one-sample case). See the module doc for the full provenance
/// (Numerical Recipes Sec. 14.3, citing Stephens 1970).
///
/// # Errors
/// [`StatsError::InvalidInput`] if any value is non-finite, if
/// `samples.len() < 5`, or if any value lies outside `[0,1]` beyond a small
/// tolerance; see [`validate_uniform_sample`].
pub fn ks_uniform(samples: &[f64]) -> Result<KsResult, StatsError> {
    let sorted = validate_uniform_sample("ks_uniform", samples)?;
    let n = sorted.len();

    let d = ks_statistic(&sorted);

    let n_f = n as f64;
    let en = n_f.sqrt();
    let lambda = (en + 0.12 + 0.11 / en) * d;
    let p_value = q_ks(lambda);

    Ok(KsResult { d, p_value, n })
}

/// Anderson-Darling A^2 statistic (case 0) for an ascending-sorted sample,
/// per Marsaglia & Marsaglia (2004) p. 1 (see module doc, quoted verbatim):
/// `A_n = -n - (1/n) * sum_{i=1}^{n} (2i-1) [ln(x_i) + ln(1 - x_{n+1-i})]`.
fn ad_statistic(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    let n_f = n as f64;
    let mut total = 0.0_f64;
    for i in 1..=n {
        let xi = sorted[i - 1];
        let x_np1_mi = sorted[n - i]; // x_{n+1-i}, 0-based
        let coeff = (2 * i - 1) as f64;
        total += coeff * (xi.ln() + (1.0 - x_np1_mi).ln());
    }
    -n_f - total / n_f
}

/// `adinf(z)`, the Marsaglia & Marsaglia (2004) asymptotic (n -> infinity)
/// closed-form CDF approximation for the case-0 AD statistic, exactly as
/// quoted in the module doc (their two-piece Horner-polynomial form, p. 4).
/// **PINNED**: coefficients and the `z < 2` / `z >= 2` split must not
/// change without a semver bump.
fn adinf(z: f64) -> f64 {
    if z < 2.0 {
        let poly = 2.00012
            + z * (0.247105
                - z * (0.0649821 - z * (0.0347962 - z * (0.0116720 - z * 0.00168691))));
        z.powf(-0.5) * (-1.2337141 / z).exp() * poly
    } else {
        let poly = 1.0776
            - z * (2.30695 - z * (0.43424 - z * (0.082433 - z * (0.008056 - z * 0.0003146))));
        (-poly.exp()).exp()
    }
}

/// One-sample Anderson-Darling test against U(0,1) (case 0: fully specified
/// reference distribution).
///
/// `a2` is the A^2 statistic (see [`ad_statistic`]); `p_value` is
/// `1 - adinf(a2)` (see the module doc for the full provenance, Marsaglia &
/// Marsaglia 2004), clamped to `[0,1]` to guard against `adinf`'s own
/// stated (small) approximation error pushing the result marginally outside
/// that range.
///
/// // sezgi simplification: see the module-level doc's provenance section
/// // for the `errfix(n,x)` finite-sample correction this function omits.
///
/// # Errors
/// [`StatsError::InvalidInput`] if any value is non-finite, if
/// `samples.len() < 5`, if any value lies outside `[0,1]` beyond a small
/// tolerance (see [`validate_uniform_sample`]), or if any (post-clamp)
/// value is exactly `0.0` or `1.0` (the statistic takes `ln(x)` and
/// `ln(1-x)`, both undefined at those boundaries; the index named in that
/// error is into the internally sorted copy, not the caller's original
/// order).
pub fn ad_uniform(samples: &[f64]) -> Result<AdResult, StatsError> {
    let sorted = validate_uniform_sample("ad_uniform", samples)?;
    let n = sorted.len();

    for (i, &v) in sorted.iter().enumerate() {
        if v == 0.0 || v == 1.0 {
            return Err(StatsError::InvalidInput(format!(
                "ad_uniform: sorted sample at index {i} = {v} touches the domain boundary (0 or 1); \
                 the Anderson-Darling statistic requires ln(x) and ln(1-x), both undefined there"
            )));
        }
    }

    let a2 = ad_statistic(&sorted);
    let p_value = (1.0 - adinf(a2)).clamp(0.0, 1.0);

    Ok(AdResult { a2, p_value, n })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Hand-derived n=5 KS fixture, showing BOTH sup-branches at every
    // order statistic (per NR's ksone: dt = max(|F_emp_before - F|,
    // |F_emp_after - F|)).
    //
    // x = [0.05, 0.15, 0.4, 0.6, 0.95] (already ascending), n = 5.
    // i/n for i=1..5: 0.2, 0.4, 0.6, 0.8, 1.0
    // (i-1)/n for i=1..5: 0, 0.2, 0.4, 0.6, 0.8
    //
    // i=1: F_emp before=0, after=0.2, F(x_1)=0.05
    //   branch (after - F)  = 0.2 - 0.05 = 0.15
    //   branch (F - before) = 0.05 - 0   = 0.05
    //   dt = max(0.15, 0.05) = 0.15
    // i=2: before=0.2, after=0.4, F(x_2)=0.15
    //   branch (after - F)  = 0.4 - 0.15 = 0.25
    //   branch (before - F) = 0.2 - 0.15 = 0.05  (F below F_emp_before)
    //   dt = max(0.25, 0.05) = 0.25
    // i=3: before=0.4, after=0.6, F(x_3)=0.4
    //   branch (after - F)  = 0.6 - 0.4 = 0.2
    //   branch (F - before) = 0.4 - 0.4 = 0
    //   dt = max(0.2, 0) = 0.2
    // i=4: before=0.6, after=0.8, F(x_4)=0.6
    //   branch (after - F)  = 0.8 - 0.6 = 0.2
    //   branch (F - before) = 0.6 - 0.6 = 0
    //   dt = max(0.2, 0) = 0.2
    // i=5: before=0.8, after=1.0, F(x_5)=0.95
    //   branch (after - F)  = 1.0 - 0.95 = 0.05
    //   branch (F - before) = 0.95 - 0.8 = 0.15
    //   dt = max(0.05, 0.15) = 0.15
    //
    // D = max(0.15, 0.25, 0.2, 0.2, 0.15) = 0.25, attained at i=2 (the
    // "after" branch), with the "before" branch also material at i=1, i=2,
    // and i=5 -- both branches are exercised by this fixture.
    //
    // lambda = (sqrt(5) + 0.12 + 0.11/sqrt(5)) * D
    //        = (2.23606797749979 + 0.12 + 0.04919349550499537) * 0.25
    //        = 2.4052614730047854 * 0.25 = 0.6013153682511964
    //
    // p_value = Q_KS(lambda), evaluated via the pinned `probks` series;
    // independently computed (Python mirror of the exact same pinned
    // algorithm, see task-3-report.md) as 0.8625362880828501.
    #[test]
    fn ks_uniform_hand_derived_d_both_branches() {
        let x = vec![0.05, 0.15, 0.4, 0.6, 0.95];
        let r = ks_uniform(&x).expect("valid fixture");

        assert_eq!(r.n, 5);
        assert!((r.d - 0.25).abs() < 1e-12, "d: got {}", r.d);

        let expected_p = 0.8625362880828501_f64;
        assert!(
            (r.p_value - expected_p).abs() < 1e-9,
            "p_value: got {}, expected {}",
            r.p_value,
            expected_p
        );
    }

    // Evenly spaced (perfectly "uniform-looking") n=5 sample:
    // x = [0.1, 0.3, 0.5, 0.7, 0.9]. Every order statistic sits exactly at
    // the midpoint of its (i-1)/n .. i/n empirical-CDF step, so both
    // branches equal 0.1 at every i; D = 0.1.
    // p_value (pinned algorithm, independently computed): 0.9999999945629237.
    #[test]
    fn ks_uniform_evenly_spaced_sample_large_p() {
        let x = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let r = ks_uniform(&x).expect("valid fixture");

        assert!((r.d - 0.1).abs() < 1e-12, "d: got {}", r.d);
        assert!(r.p_value > 0.99, "p_value: got {}", r.p_value);
    }

    // Tightly clustered n=5 sample near 0: x = [0.01, 0.02, 0.03, 0.04, 0.05].
    // D = 0.95 (dominant branch at i=5: after=1.0, F=0.05 -> 0.95), a very
    // large deviation from uniform for n=5.
    // p_value (pinned algorithm, independently computed): 5.833617325364261e-05.
    #[test]
    fn ks_uniform_clustered_sample_tiny_p() {
        let x = vec![0.01, 0.02, 0.03, 0.04, 0.05];
        let r = ks_uniform(&x).expect("valid fixture");

        assert!((r.d - 0.95).abs() < 1e-12, "d: got {}", r.d);
        assert!(r.p_value < 1e-3, "p_value: got {}", r.p_value);
    }

    // Property: an all-equal sample (n=20, every value 0.5) is rejected by
    // KS at a conventional significance level. Independently computed
    // (pinned algorithm): D = 0.5, p_value = 4.706583254931542e-05.
    #[test]
    fn ks_uniform_rejects_all_equal_sample() {
        let x = vec![0.5; 20];
        let r = ks_uniform(&x).expect("valid fixture");

        assert!((r.d - 0.5).abs() < 1e-12, "d: got {}", r.d);
        assert!(r.p_value < 0.01, "p_value: got {}", r.p_value);
    }

    // Determinism/property: n=500 sample drawn from this project's own
    // RngStream (sezgi-core), seeded so the draw is fixed across runs.
    // ANCHORED THRESHOLD (per this task's convention): the measured p-value
    // for this exact seed/path was p_ks = 0.7405573402342792 (independently
    // computed with the pinned algorithm, cross-checked against the Rust
    // implementation's own output). The assertion below (p > 0.1) is a
    // loose, stable threshold well below that measured value -- it is not
    // meant to re-derive the exact figure, only to confirm a genuinely
    // uniform sample is accepted.
    #[test]
    fn ks_uniform_seeded_n500_large_p() {
        let mut rng = sezgi_core::rng::RngStream::from_master(20260830, &[3, 1]);
        let x: Vec<f64> = (0..500).map(|_| rng.next_f64()).collect();

        let r = ks_uniform(&x).expect("valid sample");
        assert_eq!(r.n, 500);
        assert!(r.p_value > 0.1, "p_value: got {}", r.p_value);
    }

    #[test]
    fn ks_uniform_errors_on_too_few_samples() {
        let x = vec![0.1, 0.2, 0.3, 0.4];
        assert!(matches!(
            ks_uniform(&x),
            Err(StatsError::InvalidInput(_))
        ));
    }

    #[test]
    fn ks_uniform_errors_on_out_of_range_value() {
        let x = vec![0.1, 0.2, 0.3, 0.4, 1.5];
        let err = ks_uniform(&x).expect_err("out-of-range value must be rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("ks_uniform"), "message should name the function: {msg}");
        assert!(msg.contains("outside [0,1]"), "message: {msg}");
    }

    #[test]
    fn ks_uniform_errors_on_nan() {
        let x = vec![0.1, 0.2, f64::NAN, 0.4, 0.5];
        let err = ks_uniform(&x).expect_err("NaN must be rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("ks_uniform"), "message: {msg}");
        assert!(msg.contains("non-finite"), "message: {msg}");
    }

    #[test]
    fn ks_uniform_accepts_tiny_boundary_noise() {
        // -1e-10 and 1.0 + 1e-10 are within BOUNDARY_TOL (1e-9) and get
        // clamped into [0,1] rather than rejected.
        let x = vec![-1e-10, 0.25, 0.5, 0.75, 1.0 + 1e-10];
        assert!(ks_uniform(&x).is_ok());
    }

    #[test]
    fn ks_uniform_is_deterministic() {
        let x = vec![0.05, 0.15, 0.4, 0.6, 0.95];
        let r1 = ks_uniform(&x).unwrap();
        let r2 = ks_uniform(&x).unwrap();
        assert_eq!(r1.d.to_bits(), r2.d.to_bits());
        assert_eq!(r1.p_value.to_bits(), r2.p_value.to_bits());
    }

    // Hand-derived n=5 AD fixture, full Sigma arithmetic.
    //
    // x = [0.1, 0.2, 0.4, 0.6, 0.9] (ascending), n = 5.
    // A_n = -n - (1/n) * sum_{i=1}^{5} (2i-1)[ln(x_i) + ln(1 - x_{6-i})]
    //
    // i=1: coeff=1, x_1=0.1, x_5=0.9 -> ln(0.1) + ln(1-0.9) = ln(0.1)+ln(0.1)
    //      = -2.302585092994046 + -2.302585092994046 = -4.605170185988092
    //      term = 1 * -4.605170185988092 = -4.605170185988092
    // i=2: coeff=3, x_2=0.2, x_4=0.6 -> ln(0.2) + ln(1-0.6) = ln(0.2)+ln(0.4)
    //      = -1.6094379124341003 + -0.916290731874155 = -2.525728644308255
    //      term = 3 * -2.525728644308255 = -7.577185932924765
    // i=3: coeff=5, x_3=0.4, x_3=0.4 -> ln(0.4) + ln(1-0.4) = ln(0.4)+ln(0.6)
    //      = -0.916290731874155 + -0.5108256237659907 = -1.4271163556401457
    //      term = 5 * -1.4271163556401457 = -7.135581778200728
    // i=4: coeff=7, x_4=0.6, x_2=0.2 -> ln(0.6) + ln(1-0.2) = ln(0.6)+ln(0.8)
    //      = -0.5108256237659907 + -0.2231435513142097 = -0.7339691750802004
    //      term = 7 * -0.7339691750802004 = -5.137784225561403
    // i=5: coeff=9, x_5=0.9, x_1=0.1 -> ln(0.9) + ln(1-0.1) = ln(0.9)+ln(0.9)
    //      = -0.10536051565782628 + -0.10536051565782628 = -0.21072103131565257
    //      term = 9 * -0.21072103131565257 = -1.8964892818408732
    //
    // sum = -4.605170185988092 -7.577185932924765 -7.135581778200728
    //       -5.137784225561403 -1.8964892818408732 = -26.352211404515863
    //
    // A_n = -5 - (-26.352211404515863 / 5) = -5 + 5.270442280903173
    //     = 0.2704422809031728
    //
    // p_value = 1 - adinf(A_n), A_n < 2 branch (pinned Horner polynomial);
    // independently computed: 0.9585768354526911.
    #[test]
    fn ad_uniform_hand_derived_a2_full_sum() {
        let x = vec![0.1, 0.2, 0.4, 0.6, 0.9];
        let r = ad_uniform(&x).expect("valid fixture");

        assert_eq!(r.n, 5);
        let expected_a2 = 0.2704422809031728_f64;
        assert!(
            (r.a2 - expected_a2).abs() < 1e-9,
            "a2: got {}, expected {}",
            r.a2,
            expected_a2
        );

        let expected_p = 0.9585768354526911_f64;
        assert!(
            (r.p_value - expected_p).abs() < 1e-9,
            "p_value: got {}, expected {}",
            r.p_value,
            expected_p
        );
    }

    // Same evenly spaced sample as the KS trio: x = [0.1, 0.3, 0.5, 0.7, 0.9].
    // A_n (independently computed): 0.13008346290525719.
    // p_value: 0.9995716562595469.
    #[test]
    fn ad_uniform_evenly_spaced_sample_large_p() {
        let x = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let r = ad_uniform(&x).expect("valid fixture");

        let expected_a2 = 0.13008346290525719_f64;
        assert!(
            (r.a2 - expected_a2).abs() < 1e-9,
            "a2: got {}, expected {}",
            r.a2,
            expected_a2
        );
        assert!(r.p_value > 0.99, "p_value: got {}", r.p_value);
    }

    // Same clustered sample as the KS trio: x = [0.01, 0.02, 0.03, 0.04, 0.05].
    // A_n (independently computed): 11.785135442842353 (n=5, z >= 2 branch).
    // p_value: 4.8297186472368026e-08.
    #[test]
    fn ad_uniform_clustered_sample_tiny_p() {
        let x = vec![0.01, 0.02, 0.03, 0.04, 0.05];
        let r = ad_uniform(&x).expect("valid fixture");

        let expected_a2 = 11.785135442842353_f64;
        assert!(
            (r.a2 - expected_a2).abs() < 1e-6,
            "a2: got {}, expected {}",
            r.a2,
            expected_a2
        );
        assert!(r.p_value < 1e-6, "p_value: got {}", r.p_value);
    }

    // Property: an all-equal sample (n=20, every value 0.5) is rejected by
    // AD. Independently computed: A_n = 7.725887222397812,
    // p_value = 0.00015102743069206337.
    #[test]
    fn ad_uniform_rejects_all_equal_sample() {
        let x = vec![0.5; 20];
        let r = ad_uniform(&x).expect("valid fixture");

        let expected_a2 = 7.725887222397812_f64;
        assert!(
            (r.a2 - expected_a2).abs() < 1e-6,
            "a2: got {}, expected {}",
            r.a2,
            expected_a2
        );
        assert!(r.p_value < 0.01, "p_value: got {}", r.p_value);
    }

    // Same n=500 seeded sample as the KS determinism test (identical
    // RngStream seed/path). ANCHORED THRESHOLD: measured p_value for this
    // exact draw was 0.6706588748626475 (A_n = 0.5763282960501783,
    // independently computed and cross-checked against this
    // implementation's own output). Assertion uses a loose, stable
    // threshold well below the measured value.
    #[test]
    fn ad_uniform_seeded_n500_large_p() {
        let mut rng = sezgi_core::rng::RngStream::from_master(20260830, &[3, 1]);
        let x: Vec<f64> = (0..500).map(|_| rng.next_f64()).collect();

        let r = ad_uniform(&x).expect("valid sample");
        assert_eq!(r.n, 500);
        assert!(r.p_value > 0.1, "p_value: got {}", r.p_value);
    }

    #[test]
    fn ad_uniform_errors_on_too_few_samples() {
        let x = vec![0.1, 0.2, 0.3, 0.4];
        assert!(matches!(
            ad_uniform(&x),
            Err(StatsError::InvalidInput(_))
        ));
    }

    #[test]
    fn ad_uniform_errors_on_out_of_range_value() {
        let x = vec![0.1, 0.2, 0.3, 0.4, -0.5];
        let err = ad_uniform(&x).expect_err("out-of-range value must be rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("ad_uniform"), "message: {msg}");
        assert!(msg.contains("outside [0,1]"), "message: {msg}");
    }

    #[test]
    fn ad_uniform_errors_on_boundary_zero() {
        // Exactly 0.0 makes ln(x) diverge; must be rejected explicitly
        // rather than propagating +/-infinity.
        let x = vec![0.0, 0.2, 0.4, 0.6, 0.8];
        let err = ad_uniform(&x).expect_err("boundary 0.0 must be rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("ad_uniform"), "message: {msg}");
        assert!(msg.contains("boundary"), "message: {msg}");
    }

    #[test]
    fn ad_uniform_errors_on_boundary_one() {
        let x = vec![0.2, 0.4, 0.6, 0.8, 1.0];
        let err = ad_uniform(&x).expect_err("boundary 1.0 must be rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("ad_uniform"), "message: {msg}");
        assert!(msg.contains("boundary"), "message: {msg}");
    }

    #[test]
    fn ad_uniform_errors_on_nan() {
        let x = vec![0.1, 0.2, f64::NAN, 0.4, 0.5];
        let err = ad_uniform(&x).expect_err("NaN must be rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("ad_uniform"), "message: {msg}");
        assert!(msg.contains("non-finite"), "message: {msg}");
    }

    #[test]
    fn ad_uniform_is_deterministic() {
        let x = vec![0.1, 0.2, 0.4, 0.6, 0.9];
        let r1 = ad_uniform(&x).unwrap();
        let r2 = ad_uniform(&x).unwrap();
        assert_eq!(r1.a2.to_bits(), r2.a2.to_bits());
        assert_eq!(r1.p_value.to_bits(), r2.p_value.to_bits());
    }
}
