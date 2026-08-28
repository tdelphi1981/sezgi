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

/// f4 çekirdeği: z = t_osz(x - x_opt) SONRASI çağrılır. s_i: tek indeksli
/// ve pozitif bileşenlere 10x ölçek; taban ölçek 10^(0.5·i/(d-1)).
pub fn buche_rastrigin(z: &[f64]) -> f64 {
    let d = z.len() as f64;
    let s: Vec<f64> = z.iter().enumerate().map(|(i, &zi)| {
        let base = 10f64.powf(0.5 * i as f64 / (d - 1.0).max(1.0));
        let extra = if i % 2 == 0 && zi > 0.0 { 10.0 } else { 1.0 }; // BBOB: tek (1-tabanlı) indeks
        base * extra * zi
    }).collect();
    10.0 * (d - s.iter().map(|v| (2.0 * std::f64::consts::PI * v).cos()).sum::<f64>())
        + s.iter().map(|v| v * v).sum::<f64>()
}

/// f5: kaydırmasız, ham x. s_i = sign(x_opt_i)·10^(i/(d-1));
/// z_i = x_i eğer x_opt_i·x_i < 25, değilse x_opt_i (plato).
pub fn linear_slope(x: &[f64], x_opt: &[f64]) -> f64 {
    let d = x.len() as f64;
    x.iter().zip(x_opt).enumerate().map(|(i, (&xi, &oi))| {
        let s = oi.signum() * 10f64.powf(i as f64 / (d - 1.0).max(1.0));
        let z = if oi * xi < 25.0 { xi } else { oi };
        5.0 * s.abs() - s * z
    }).sum()
}

/// f6: Attractive Sector
/// Boru hattı: z = Q·Λ^10·R·(x - x_opt)
/// Çekirdek: s_i = 100 eğer z_i·x_opt_i > 0, değilse 1
pub fn attractive_sector(z: &[f64], x_opt: &[f64]) -> f64 {
    let s: f64 = z.iter().zip(x_opt).map(|(&zi, &oi)| {
        let si = if zi * oi > 0.0 { 100.0 } else { 1.0 };
        (si * zi).powi(2)
    }).sum();
    // t_osz skaler hali: tek elemanlı dilim üzerinden
    crate::bbob::transform::t_osz(&[s])[0].powf(0.9)
}

/// f7: Step Ellipsoidal
/// zhat0: ilk Λ^10 bileşeni, z: Q·ztilde sonrası
pub fn step_ellipsoidal(zhat0: f64, z: &[f64]) -> f64 {
    let d = z.len() as f64;
    let ell: f64 = z.iter().enumerate()
        .map(|(i, &zi)| 10f64.powf(2.0 * i as f64 / (d - 1.0).max(1.0)) * zi * zi)
        .sum();
    0.1 * (zhat0.abs() / 1e4).max(ell)
}
