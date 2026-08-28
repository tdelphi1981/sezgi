use sezgi_core::rng::RngStream;
use sezgi_core::dist::Distribution;

/// Gauss matrisinin Gram–Schmidt ortogonalleştirmesi → rastgele döndürme.
pub fn rotation_matrix(d: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = RngStream::from_master(seed, &[0]);
    let gauss = Distribution::Gaussian { mean: 0.0, sigma: 1.0 };
    let mut m: Vec<Vec<f64>> = (0..d).map(|_| gauss.sample_n(&mut rng, d)).collect();
    for i in 0..d {
        for j in 0..i {
            let dot: f64 = (0..d).map(|k| m[i][k] * m[j][k]).sum();
            for k in 0..d { m[i][k] -= dot * m[j][k]; }
        }
        let norm: f64 = m[i].iter().map(|x| x * x).sum::<f64>().sqrt();
        for k in 0..d { m[i][k] /= norm; }
    }
    // satır vektörleri ortonormal; R[k][i] erişimi için transpoze saklamaya gerek yok,
    // apply() satır-vektör konvansiyonunu kullanır
    m
}

pub fn apply(r: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    r.iter().map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum()).collect()
}

pub fn t_osz(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&xi| {
        if xi == 0.0 { return 0.0; }
        let xhat = xi.abs().ln();
        let (c1, c2) = if xi > 0.0 { (10.0, 7.9) } else { (5.5, 3.1) };
        xi.signum() * (xhat + 0.049 * ((c1 * xhat).sin() + (c2 * xhat).sin())).exp()
    }).collect()
}

pub fn t_asy(x: &[f64], beta: f64) -> Vec<f64> {
    let d = x.len() as f64;
    x.iter().enumerate().map(|(i, &xi)| {
        if xi > 0.0 {
            xi.powf(1.0 + beta * i as f64 / (d - 1.0).max(1.0) * xi.sqrt())
        } else { xi }
    }).collect()
}

pub fn lambda_alpha(x: &[f64], alpha: f64) -> Vec<f64> {
    let d = x.len() as f64;
    x.iter().enumerate()
        .map(|(i, &xi)| alpha.powf(0.5 * i as f64 / (d - 1.0).max(1.0)) * xi)
        .collect()
}

pub fn f_pen(x: &[f64]) -> f64 {
    x.iter().map(|xi| (xi.abs() - 5.0).max(0.0).powi(2)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_osz_fixed_points_and_sign() {
        assert_eq!(t_osz(&[0.0]), vec![0.0]);
        let y = t_osz(&[1.0, -1.0]);
        assert!(y[0] > 0.0 && y[1] < 0.0);
        // x=1: xhat=0 → sin terimleri 0 → exp(0)=1
        assert!((y[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn t_asy_only_bends_positive() {
        let y = t_asy(&[-2.0, 2.0], 0.5);
        assert_eq!(y[0], -2.0);          // negatif dokunulmaz
        assert!(y[1] > 2.0);             // pozitif büker (üs > 1)
    }

    #[test]
    fn lambda_alpha_endpoints() {
        let y = lambda_alpha(&[1.0, 1.0, 1.0], 100.0);
        assert!((y[0] - 1.0).abs() < 1e-12);
        assert!((y[2] - 10.0).abs() < 1e-9); // 100^0.5 = 10
    }

    #[test]
    fn f_pen_zero_inside_bounds() {
        assert_eq!(f_pen(&[5.0, -5.0, 0.0]), 0.0);
        assert!((f_pen(&[6.0]) - 1.0).abs() < 1e-12);
    }
}
