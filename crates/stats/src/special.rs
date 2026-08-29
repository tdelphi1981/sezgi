/// Error function using Abramowitz & Stegun 7.1.26 rational approximation.
///
/// Max absolute error: 1.5e-7
///
/// Formula:
/// t = 1 / (1 + 0.3275911 * |x|)
/// erf(x) = sign(x) * (1 - (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5) * exp(-x^2))
///
/// Coefficients (pinned):
/// a1 = 0.254829592
/// a2 = -0.284496736
/// a3 = 1.421413741
/// a4 = -1.453152027
/// a5 = 1.061405429
pub fn erf(x: f64) -> f64 {
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;
    const P: f64 = 0.327_591_1;

    if x == 0.0 {
        return 0.0;
    }

    let abs_x = x.abs();
    let t = 1.0 / (1.0 + P * abs_x);
    let exp_neg_x2 = (-x * x).exp();

    let poly = A1 * t + A2 * t.powi(2) + A3 * t.powi(3) + A4 * t.powi(4) + A5 * t.powi(5);

    x.signum() * (1.0 - poly * exp_neg_x2)
}

/// Normal cumulative distribution function.
///
/// Φ(x) = 0.5 * (1 + erf(x / √2))
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Normal survival function (complementary CDF).
///
/// S(x) = 1 - Φ(x)
pub fn normal_sf(x: f64) -> f64 {
    1.0 - normal_cdf(x)
}

/// Natural logarithm of the absolute value of the gamma function.
///
/// Uses Lanczos approximation with g=7, n=9 (pinned coefficients from crates/core/src/dist.rs).
/// Valid for x > 0.
pub fn ln_gamma(x: f64) -> f64 {
    // Lanczos g=7, n=9 fixed coefficients
    const G: [f64; 9] = [
        0.999_999_999_999_809_9, 676.5203681218851, -1259.1392167224028,
        771.323_428_777_653_1, -176.615_029_162_140_6, 12.507343278686905,
        -0.13857109526572012, 9.984_369_578_019_572e-6, 1.5056327351493116e-7,
    ];

    if x <= 0.0 {
        return f64::NAN;
    }

    if x < 0.5 {
        // Use reflection formula: Γ(x) * Γ(1-x) = π / sin(πx)
        // ln(Γ(x)) = ln(π) - ln(sin(πx)) - ln(Γ(1-x))
        let pi_x = std::f64::consts::PI * x;
        return std::f64::consts::PI.ln() - pi_x.sin().ln() - ln_gamma(1.0 - x);
    }

    let x = x - 1.0;
    let mut a = G[0];
    let t = x + 7.5;
    for (i, &g) in G.iter().enumerate().skip(1) {
        a += g / (x + i as f64);
    }

    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Regularized lower incomplete gamma function.
///
/// γ(a, x) / Γ(a) where a > 0, x >= 0
///
/// Uses series approximation when x < a + 1,
/// otherwise uses Lentz continued fraction (Numerical Recipes).
/// Relative tolerance: 1e-12
/// Iteration cap: 500
///
/// Edge cases:
/// - γ(a, 0) = 0
/// - x < 0 → NaN
pub fn gamma_p(a: f64, x: f64) -> f64 {
    const ITMAX: usize = 500;
    const EPS: f64 = 1e-12;

    if a <= 0.0 || x < 0.0 {
        return f64::NAN;
    }

    if x == 0.0 {
        return 0.0;
    }

    if x < a + 1.0 {
        // Series approximation
        gamma_p_series(a, x)
    } else {
        // Continued fraction (Lentz algorithm)
        1.0 - gamma_p_frac(a, x, ITMAX, EPS)
    }
}

/// Gamma series: computes γ(a, x) / Γ(a) using series expansion.
fn gamma_p_series(a: f64, x: f64) -> f64 {
    const ITMAX: usize = 500;
    const EPS: f64 = 1e-12;

    let ln_gamma_a = ln_gamma(a);
    let ln_exp_part = a * x.ln() - x - ln_gamma_a;

    if ln_exp_part < -700.0 {
        return 0.0;
    }

    let exp_part = ln_exp_part.exp();
    let mut sum = 1.0 / a;
    let mut delta = sum;

    for i in 1..ITMAX {
        delta *= x / (a + i as f64);
        sum += delta;
        if delta.abs() < sum.abs() * EPS {
            return sum * exp_part;
        }
    }

    sum * exp_part
}

/// Gamma complementary: computes Γ(a, x) / Γ(a) using continued fraction (Lentz).
fn gamma_p_frac(a: f64, x: f64, itmax: usize, eps: f64) -> f64 {
    const FPMIN: f64 = 1e-30;

    let ln_gamma_a = ln_gamma(a);
    let ln_exp_part = a * x.ln() - x - ln_gamma_a;

    if ln_exp_part < -700.0 {
        return 1.0;
    }

    let exp_part = ln_exp_part.exp();

    let mut b = x + 1.0 - a;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b;
    let mut h = d;

    for i in 1..=itmax {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = b + an / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < eps {
            return exp_part * h;
        }
    }

    exp_part * h
}

/// Chi-square survival function (complementary CDF).
///
/// P(X > x) where X ~ χ²(k)
///
/// Formula: χ²_sf(x, k) = 1 - γ(k/2, x/2) / Γ(k/2)
pub fn chi_square_sf(x: f64, k: usize) -> f64 {
    if k == 0 || x < 0.0 {
        return f64::NAN;
    }

    let a = k as f64 / 2.0;
    let x_half = x / 2.0;

    1.0 - gamma_p(a, x_half)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erf_zero() {
        let result = erf(0.0);
        assert!((result - 0.0).abs() < 1e-15, "erf(0) should be 0, got {}", result);
    }

    #[test]
    fn test_erf_one() {
        let result = erf(1.0);
        let expected = 0.8427007929;
        assert!((result - expected).abs() < 2e-7,
                "erf(1) should be {}, got {}, error: {}",
                expected, result, (result - expected).abs());
    }

    #[test]
    fn test_erf_symmetry() {
        let x = 0.5;
        let erf_x = erf(x);
        let erf_neg_x = erf(-x);
        assert!((erf_x + erf_neg_x).abs() < 1e-15,
                "erf(-x) should equal -erf(x)");
    }

    #[test]
    fn test_normal_cdf() {
        let result = normal_cdf(1.959964);
        let expected = 0.975;
        assert!((result - expected).abs() < 1e-6,
                "normal_cdf(1.959964) should be {}, got {}, error: {}",
                expected, result, (result - expected).abs());
    }

    #[test]
    fn test_ln_gamma_five() {
        let result = ln_gamma(5.0);
        let expected = (24.0_f64).ln(); // ln(4!) = ln(24)
        assert!((result - expected).abs() < 1e-10,
                "ln_gamma(5) should be ln(24), got {}, expected {}, error: {}",
                result, expected, (result - expected).abs());
    }

    #[test]
    fn test_ln_gamma_half() {
        let result = ln_gamma(0.5);
        let expected = (std::f64::consts::PI.sqrt()).ln(); // ln(√π)
        assert!((result - expected).abs() < 1e-8,
                "ln_gamma(0.5) should be ln(√π), got {}, expected {}, error: {}",
                result, expected, (result - expected).abs());
    }

    #[test]
    fn test_ln_gamma_reflection_branch() {
        // Tests the reflection branch (x < 0.5) using ln_gamma(0.3)
        let result = ln_gamma(0.3);
        let expected = 1.0957979948180756;
        assert!((result - expected).abs() < 1e-8,
                "ln_gamma(0.3) should be {}, got {}, error: {}",
                expected, result, (result - expected).abs());
    }

    #[test]
    fn test_ln_gamma_reflection_point_25() {
        // Tests the reflection branch (x < 0.5) using ln_gamma(0.25)
        let result = ln_gamma(0.25);
        let expected = 1.2880225246980774;
        assert!((result - expected).abs() < 1e-8,
                "ln_gamma(0.25) should be {}, got {}, error: {}",
                expected, result, (result - expected).abs());
    }

    #[test]
    fn test_gamma_p_one_one() {
        let result = gamma_p(1.0, 1.0);
        let expected = 1.0 - std::f64::consts::E.recip(); // 1 - e^(-1)
        assert!((result - expected).abs() < 1e-10,
                "gamma_p(1, 1) should be 1-e^(-1), got {}, expected {}, error: {}",
                result, expected, (result - expected).abs());
    }

    #[test]
    fn test_chi_square_sf_one() {
        let result = chi_square_sf(3.841459, 1);
        let expected = 0.05;
        assert!((result - expected).abs() < 1e-5,
                "chi_square_sf(3.841459, 1) should be ~0.05, got {}, error: {}",
                result, (result - expected).abs());
    }

    #[test]
    fn test_chi_square_sf_two() {
        let result = chi_square_sf(5.991465, 2);
        let expected = 0.05;
        assert!((result - expected).abs() < 1e-5,
                "chi_square_sf(5.991465, 2) should be ~0.05, got {}, error: {}",
                result, (result - expected).abs());
    }
}
