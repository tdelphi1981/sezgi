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
