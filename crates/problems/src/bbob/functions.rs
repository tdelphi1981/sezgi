pub fn sphere(z: &[f64]) -> f64 { z.iter().map(|x| x * x).sum() }

pub fn ellipsoidal(z: &[f64]) -> f64 {
    let d = z.len() as f64;
    z.iter().enumerate()
        .map(|(i, x)| 1e6f64.powf(i as f64 / (d - 1.0).max(1.0)) * x * x)
        .sum()
}

pub fn rastrigin(z: &[f64]) -> f64 {
    10.0 * (z.len() as f64
        - z.iter().map(|x| (2.0 * std::f64::consts::PI * x).cos()).sum::<f64>())
        + sphere(z)
}

pub fn rosenbrock(z: &[f64]) -> f64 {
    // BBOB f8 konvansiyonu: optimumda z = 1 vektörü olacak şekilde kaydırılmış çağrılır
    z.windows(2).map(|w| 100.0 * (w[0] * w[0] - w[1]).powi(2) + (w[0] - 1.0).powi(2)).sum()
}
