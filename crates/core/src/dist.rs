use crate::rng::RngStream;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Distribution {
    Uniform,
    Gaussian { mean: f64, sigma: f64 },
    Cauchy { loc: f64, scale: f64 },
    Levy { alpha: f64 },
    StudentT { nu: f64 },
    Laplace { loc: f64, scale: f64 },
}

impl Distribution {
    pub fn sample(&self, rng: &mut RngStream) -> f64 {
        match *self {
            Distribution::Uniform => rng.next_f64(),
            Distribution::Gaussian { mean, sigma } => mean + sigma * gauss_polar(rng),
            Distribution::Cauchy { loc, scale } =>
                loc + scale * (std::f64::consts::PI * (rng.next_f64() - 0.5)).tan(),
            Distribution::Levy { alpha } => levy_mantegna(rng, alpha),
            Distribution::StudentT { nu } => student_t(rng, nu),
            Distribution::Laplace { loc, scale } => {
                let u = rng.next_f64() - 0.5;
                loc - scale * u.signum() * (1.0 - 2.0 * u.abs()).ln()
            }
        }
    }

    pub fn sample_n(&self, rng: &mut RngStream, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.sample(rng)).collect()
    }

    pub fn validate(&self) -> Result<(), String> {
        let err = |m: String| Err(m);
        let fin = |name: &str, v: f64| if v.is_finite() { Ok(()) }
            else { Err(format!("{name} sonlu olmalı: {v}")) };
        match *self {
            Distribution::Uniform => Ok(()),
            Distribution::Gaussian { mean, sigma } => {
                fin("mean", mean)?; fin("sigma", sigma)?;
                if sigma < 0.0 { return err(format!("sigma >= 0 olmalı: {sigma}")); } Ok(())
            }
            Distribution::Cauchy { loc, scale } | Distribution::Laplace { loc, scale } => {
                fin("loc", loc)?; fin("scale", scale)?;
                if scale <= 0.0 { return err(format!("scale > 0 olmalı: {scale}")); } Ok(())
            }
            Distribution::Levy { alpha } => {
                fin("alpha", alpha)?;
                if !(0.0 < alpha && alpha <= 2.0) {
                    return err(format!("alpha (0,2] aralığında olmalı: {alpha}")); } Ok(())
            }
            Distribution::StudentT { nu } => {
                fin("nu", nu)?;
                if nu <= 0.0 { return err(format!("nu > 0 olmalı: {nu}")); } Ok(())
            }
        }
    }
}

/// Polar Box–Muller (Marsaglia). SABİT — değişiklik semver'e işlenir.
fn gauss_polar(rng: &mut RngStream) -> f64 {
    loop {
        let u = 2.0 * rng.next_f64() - 1.0;
        let v = 2.0 * rng.next_f64() - 1.0;
        let s = u * u + v * v;
        if s > 0.0 && s < 1.0 {
            return u * (-2.0 * s.ln() / s).sqrt();
        }
    }
}

/// Mantegna (1994) Lévy adım üreteci; 0 < alpha <= 2.
fn levy_mantegna(rng: &mut RngStream, alpha: f64) -> f64 {
    fn gamma(x: f64) -> f64 {
        // Lanczos g=7, n=9 sabit katsayılar
        const G: [f64; 9] = [
            0.99999999999980993, 676.5203681218851, -1259.1392167224028,
            771.32342877765313, -176.61502916214059, 12.507343278686905,
            -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
        ];
        if x < 0.5 {
            std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma(1.0 - x))
        } else {
            let x = x - 1.0;
            let mut a = G[0];
            let t = x + 7.5;
            for (i, &g) in G.iter().enumerate().skip(1) {
                a += g / (x + i as f64);
            }
            (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * a
        }
    }
    let num = gamma(1.0 + alpha) * (std::f64::consts::PI * alpha / 2.0).sin();
    let den = gamma((1.0 + alpha) / 2.0) * alpha * 2f64.powf((alpha - 1.0) / 2.0);
    let sigma_u = (num / den).powf(1.0 / alpha);
    let u = sigma_u * gauss_polar(rng);
    let v = gauss_polar(rng).abs();
    u / v.powf(1.0 / alpha)
}

/// Student-t: t = Z / sqrt(ChiSq(nu)/nu); ChiSq, Gaussian karelerinin
/// toplamı yerine Marsaglia–Tsang gamma örnekleyicisiyle (sabit).
fn student_t(rng: &mut RngStream, nu: f64) -> f64 {
    let z = gauss_polar(rng);
    let chi2 = 2.0 * gamma_mt(rng, nu / 2.0);
    z / (chi2 / nu).sqrt()
}

/// Marsaglia–Tsang (2000) gamma(shape, scale=1); shape >= 1 dalı + boost.
fn gamma_mt(rng: &mut RngStream, shape: f64) -> f64 {
    if shape < 1.0 {
        let u = rng.next_f64();
        return gamma_mt(rng, shape + 1.0) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = gauss_polar(rng);
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 { continue; }
        let u = rng.next_f64();
        if u < 1.0 - 0.0331 * x.powi(4)
            || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::RngStream;

    fn stats(xs: &[f64]) -> (f64, f64) {
        let m = xs.iter().sum::<f64>() / xs.len() as f64;
        let v = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64;
        (m, v)
    }

    #[test]
    fn gaussian_moments() {
        let mut r = RngStream::from_master(7, &[0]);
        let d = Distribution::Gaussian { mean: 2.0, sigma: 0.5 };
        let (m, v) = stats(&d.sample_n(&mut r, 200_000));
        assert!((m - 2.0).abs() < 0.01, "mean={m}");
        assert!((v - 0.25).abs() < 0.01, "var={v}");
    }

    #[test]
    fn uniform_bounds_and_mean() {
        let mut r = RngStream::from_master(7, &[1]);
        let d = Distribution::Uniform;
        let xs = d.sample_n(&mut r, 100_000);
        assert!(xs.iter().all(|x| (0.0..1.0).contains(x)));
        assert!((stats(&xs).0 - 0.5).abs() < 0.01);
    }

    #[test]
    fn cauchy_median_at_loc() {
        let mut r = RngStream::from_master(7, &[2]);
        let d = Distribution::Cauchy { loc: 1.0, scale: 2.0 };
        let mut xs = d.sample_n(&mut r, 100_001);
        xs.sort_by(f64::total_cmp);
        assert!((xs[50_000] - 1.0).abs() < 0.05);
    }

    #[test]
    fn serde_roundtrip() {
        let d = Distribution::Levy { alpha: 1.5 };
        let j = serde_json::to_string(&d).unwrap();
        assert_eq!(j, r#"{"kind":"levy","alpha":1.5}"#);
        assert_eq!(serde_json::from_str::<Distribution>(&j).unwrap(), d);
    }

    #[test]
    fn validate_rules() {
        assert!(Distribution::Uniform.validate().is_ok());
        assert!(Distribution::Gaussian { mean: 0.0, sigma: 1.0 }.validate().is_ok());
        assert!(Distribution::Gaussian { mean: 0.0, sigma: -0.1 }.validate().is_err());
        assert!(Distribution::Levy { alpha: 1.5 }.validate().is_ok());
        assert!(Distribution::Levy { alpha: 0.0 }.validate().is_err());
        assert!(Distribution::Levy { alpha: 2.1 }.validate().is_err());
        assert!(Distribution::StudentT { nu: -1.0 }.validate().is_err());
        assert!(Distribution::Cauchy { loc: 0.0, scale: 0.0 }.validate().is_err());
        assert!(Distribution::Laplace { loc: 0.0, scale: f64::NAN }.validate().is_err());
    }
}
