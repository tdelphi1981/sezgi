pub mod functions;
pub mod transform;

use std::f64::consts::PI;
use sezgi_core::problem::Problem;
use sezgi_core::rng::RngStream;
use sezgi_core::space::{Block, BlockValues, Genotype, SearchSpace};

pub const BBOB_SEED_BASE: u64 = 0x5EC1;
pub const IMPLEMENTED_FIDS: &[u32] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24];

#[derive(Debug, Clone)]
pub struct GallagherData {
    pub peaks: Vec<Vec<f64>>,
    pub weights: Vec<f64>,
    pub alphas: Vec<f64>,
}

#[derive(Debug, thiserror::Error)]
pub enum BbobError {
    #[error("fid {0} henüz implemente edilmedi")]
    NotImplemented(u32),
    #[error("dim >= 2 olmalı")]
    BadDim,
}

#[derive(Debug, Clone, Copy)]
enum XOptPolicy {
    Uniform44,
    BoundaryPm5,
    ScaledPattern { scale: f64 },
}

fn needs_r(fid: u32) -> bool {
    !matches!(fid, 1 | 5 | 20)
}

fn needs_q(fid: u32) -> bool {
    matches!(fid, 6 | 7 | 13 | 15 | 16 | 17 | 18 | 23 | 24)
}

fn x_opt_policy(fid: u32) -> XOptPolicy {
    match fid {
        5 => XOptPolicy::BoundaryPm5,
        20 => XOptPolicy::ScaledPattern { scale: 4.2096874633 / 2.0 },
        24 => XOptPolicy::ScaledPattern { scale: 2.5 / 2.0 },
        _ => XOptPolicy::Uniform44,
    }
}

fn shift(xs: &[f64], x_opt: &[f64]) -> Vec<f64> {
    xs.iter().zip(x_opt).map(|(x, o)| x - o).collect()
}

/// z = Q·Λ^10·R·(x - x_opt) — f6/f13 tarzı boru hattı
fn apply2(p: &BbobProblem, xs: &[f64]) -> Vec<f64> {
    let s = shift(xs, &p.x_opt);
    let r = transform::apply(p.rot.as_ref().unwrap(), &s);
    let l = transform::lambda_alpha(&r, 10.0);
    transform::apply(p.rot2.as_ref().unwrap(), &l)
}

pub struct BbobProblem {
    fid: u32,
    x_opt: Vec<f64>,
    f_opt: f64,
    rot: Option<Vec<Vec<f64>>>,
    rot2: Option<Vec<Vec<f64>>>,
    gallagher: Option<GallagherData>,
    space: SearchSpace,
    pub instance: u32,
}

impl BbobProblem {
    pub fn new(fid: u32, dim: usize, instance: u32) -> Result<Self, BbobError> {
        if dim < 2 { return Err(BbobError::BadDim); }
        if !(1..=24).contains(&fid) { return Err(BbobError::NotImplemented(fid)); }

        let mut rng = RngStream::from_master(BBOB_SEED_BASE + fid as u64,
                                             &[instance as u64]);

        // Drawing order (CRITICAL): [x_opt d draws][f_opt 1 draw][R seed if needed][Q seed if needed]
        let x_opt: Vec<f64> = match x_opt_policy(fid) {
            XOptPolicy::Uniform44 =>
                (0..dim).map(|_| -4.0 + 8.0 * rng.next_f64()).collect(),
            XOptPolicy::BoundaryPm5 =>
                (0..dim).map(|_| if rng.next_f64() < 0.5 { -5.0 } else { 5.0 }).collect(),
            XOptPolicy::ScaledPattern { scale } =>
                (0..dim).map(|_| if rng.next_f64() < 0.5 { -scale } else { scale }).collect(),
        };
        let f_opt = -200.0 + 400.0 * rng.next_f64();
        let rot = needs_r(fid).then(|| transform::rotation_matrix(dim, rng.next_u64()));
        let rot2 = needs_q(fid).then(|| transform::rotation_matrix(dim, rng.next_u64()));

        // Gallagher tepe verileri (fid 21/22 için)
        let gallagher = matches!(fid, 21 | 22).then(|| {
            let p = if fid == 21 { 101usize } else { 21 };
            let mut peaks = Vec::with_capacity(p);
            peaks.push(x_opt.clone());
            for _ in 1..p {
                peaks.push((0..dim).map(|_| -4.9 + 9.8 * rng.next_f64()).collect());
            }
            let weights: Vec<f64> = (0..p).map(|i| if i == 0 { 10.0 }
                else { 1.1 + 8.0 * (i as f64 - 1.0) / (p as f64 - 2.0) }).collect();
            let alphas: Vec<f64> = (0..p).map(|i| if i == 0 { 1000.0 }
                else { 1000f64.powf(2.0 * rng.next_f64()) }).collect();
            GallagherData { peaks, weights, alphas }
        });

        let space = SearchSpace::new(vec![Block::Float { lo: -5.0, hi: 5.0, n: dim }])
            .expect("sabit sınırlar geçerli");

        // Check if fid is implemented
        if !IMPLEMENTED_FIDS.contains(&fid) {
            return Err(BbobError::NotImplemented(fid));
        }

        Ok(Self { fid, x_opt, f_opt, rot, rot2, gallagher, space, instance })
    }

    pub fn x_opt(&self) -> &[f64] { &self.x_opt }
    pub fn f_opt(&self) -> f64 { self.f_opt }
    pub fn fid(&self) -> u32 { self.fid }
    pub fn name(&self) -> &'static str {
        match self.fid { 1 => "Sphere", 2 => "Ellipsoidal", 3 => "Rastrigin", 4 => "BucheRastrigin",
                         5 => "LinearSlope", 6 => "AttractiveSector", 7 => "StepEllipsoidal", 8 => "Rosenbrock", 9 => "RosenbrockRotated",
                         10 => "EllipsoidalRotated", 11 => "Discus", 12 => "BentCigar", 13 => "SharpRidge", 14 => "DifferentPowers",
                         15 => "RastriginRotated", 16 => "Weierstrass", 17 => "SchaffersF7", 18 => "SchaffersF7Ill", 19 => "GriewankRosenbrock",
                         20 => "Schwefel", 21 => "Gallagher101", 22 => "Gallagher21", 23 => "Katsuura", 24 => "LunacekBiRastrigin", _ => unreachable!() }
    }
}

impl Problem for BbobProblem {
    fn space(&self) -> &SearchSpace { &self.space }
    fn optimum(&self) -> Option<f64> { Some(self.f_opt) }

    fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<f64> {
        pop.iter().map(|g| {
            let BlockValues::Float(xs) = &g.blocks[0] else { return f64::INFINITY };
            let raw = match self.fid {
                1 | 2 | 3 | 8 => {
                    let shifted: Vec<f64> = xs.iter().zip(&self.x_opt).map(|(x, o)| x - o).collect();
                    let z = match &self.rot {
                        Some(r) => transform::apply(r, &shifted),
                        None => shifted,
                    };
                    match self.fid {
                        1 => functions::sphere(&z),
                        2 => functions::ellipsoidal(&z),
                        3 => functions::rastrigin(&z),
                        8 => {
                            // optimumda z=0 → w=1 kaydırması
                            let w: Vec<f64> = z.iter().map(|v| v + 1.0).collect();
                            functions::rosenbrock(&w)
                        }
                        _ => unreachable!(),
                    }
                }
                4 => {
                    let shifted: Vec<f64> = xs.iter().zip(&self.x_opt).map(|(x, o)| x - o).collect();
                    functions::buche_rastrigin(&transform::t_osz(&shifted)) + 100.0 * transform::f_pen(xs)
                }
                5 => functions::linear_slope(xs, &self.x_opt),
                6 => {
                    let z = apply2(self, xs);
                    functions::attractive_sector(&z, &self.x_opt)
                }
                7 => {
                    let shifted = shift(xs, &self.x_opt);
                    let zhat = transform::lambda_alpha(&transform::apply(self.rot.as_ref().unwrap(), &shifted), 10.0);
                    let ztilde: Vec<f64> = zhat.iter().map(|&v| {
                        if v.abs() > 0.5 { (0.5 + v).floor() } else { (0.5 + 10.0 * v).floor() / 10.0 }
                    }).collect();
                    let z = transform::apply(self.rot2.as_ref().unwrap(), &ztilde);
                    functions::step_ellipsoidal(zhat[0], &z) + transform::f_pen(xs)
                }
                9 => {
                    let shifted = shift(xs, &self.x_opt);
                    let scale = 1f64.max((xs.len() as f64).sqrt() / 8.0);
                    let z: Vec<f64> = transform::apply(self.rot.as_ref().unwrap(), &shifted)
                        .iter().map(|v| scale * v + 1.0).collect();
                    functions::rosenbrock(&z)
                }
                10 => {
                    let z = transform::t_osz(&transform::apply(self.rot.as_ref().unwrap(), &shift(xs, &self.x_opt)));
                    functions::ellipsoidal(&z)
                }
                11 => {
                    let z = transform::t_osz(&transform::apply(self.rot.as_ref().unwrap(), &shift(xs, &self.x_opt)));
                    functions::discus(&z)
                }
                12 => {
                    let r1 = transform::apply(self.rot.as_ref().unwrap(), &shift(xs, &self.x_opt));
                    let z = transform::apply(self.rot.as_ref().unwrap(), &transform::t_asy(&r1, 0.5));
                    functions::bent_cigar(&z)
                }
                13 => functions::sharp_ridge(&apply2(self, xs)),
                14 => functions::different_powers(&transform::apply(self.rot.as_ref().unwrap(), &shift(xs, &self.x_opt))),
                15 => {
                    let r1 = transform::apply(self.rot.as_ref().unwrap(), &shift(xs, &self.x_opt));
                    let a = transform::t_asy(&transform::t_osz(&r1), 0.2);
                    let q = transform::apply(self.rot2.as_ref().unwrap(), &a);
                    let z = transform::apply(self.rot.as_ref().unwrap(), &transform::lambda_alpha(&q, 10.0));
                    functions::rastrigin(&z)
                }
                16 => {
                    let r1 = transform::apply(self.rot.as_ref().unwrap(), &shift(xs, &self.x_opt));
                    let q = transform::apply(self.rot2.as_ref().unwrap(), &transform::t_osz(&r1));
                    let z = transform::apply(self.rot.as_ref().unwrap(), &transform::lambda_alpha(&q, 0.01));
                    functions::weierstrass(&z) + 10.0 / xs.len() as f64 * transform::f_pen(xs)
                }
                17 | 18 => {
                    let alpha = if self.fid == 17 { 10.0 } else { 1000.0 };
                    let r1 = transform::apply(self.rot.as_ref().unwrap(), &shift(xs, &self.x_opt));
                    let q = transform::apply(self.rot2.as_ref().unwrap(), &transform::t_asy(&r1, 0.5));
                    let z = transform::lambda_alpha(&q, alpha);
                    functions::schaffers_f7(&z) + 10.0 * transform::f_pen(xs)
                }
                19 => {
                    let scale = 1f64.max((xs.len() as f64).sqrt() / 8.0);
                    let z: Vec<f64> = transform::apply(self.rot.as_ref().unwrap(), &shift(xs, &self.x_opt))
                        .iter().map(|v| scale * v + 1.0).collect();
                    functions::griewank_rosenbrock(&z)
                }
                20 => {
                    let d = xs.len();
                    let xhat: Vec<f64> = xs.iter().zip(&self.x_opt).map(|(x, o)| 2.0 * o.signum() * x).collect();
                    let two_abs: Vec<f64> = self.x_opt.iter().map(|o| 2.0 * o.abs()).collect();
                    let mut zhat = xhat.clone();
                    for i in 1..d { zhat[i] = xhat[i] + 0.25 * (xhat[i - 1] - two_abs[i - 1]); }
                    let inner: Vec<f64> = zhat.iter().zip(&two_abs).map(|(z, t)| z - t).collect();
                    let z: Vec<f64> = transform::lambda_alpha(&inner, 10.0).iter()
                        .zip(&two_abs).map(|(l, t)| 100.0 * (l + t)).collect();
                    let s: f64 = z.iter().map(|&zi| zi * (zi.abs().sqrt()).sin()).sum();
                    let z100: Vec<f64> = z.iter().map(|v| v / 100.0).collect();
                    -s / (100.0 * d as f64) + 4.189828872724339 + 100.0 * transform::f_pen(&z100)
                }
                21 | 22 => {
                    let gd = self.gallagher.as_ref().unwrap();
                    let d = xs.len() as f64;
                    let best: f64 = gd.peaks.iter().zip(&gd.weights).zip(&gd.alphas).map(|((y, &w), &a)| {
                        let diff = shift(xs, y);
                        let rd = transform::apply(self.rot.as_ref().unwrap(), &diff);
                        let c = transform::lambda_alpha(&rd, a);
                        let q: f64 = c.iter().zip(&rd).map(|(ci, ri)| ci * ri).sum::<f64>() / a.powf(0.25);
                        w * (-q / (2.0 * d)).exp()
                    }).fold(f64::NEG_INFINITY, f64::max);
                    transform::t_osz(&[10.0 - best])[0].powi(2) + transform::f_pen(xs)
                }
                23 => {
                    let r1 = transform::apply(self.rot.as_ref().unwrap(), &shift(xs, &self.x_opt));
                    let z = transform::apply(self.rot2.as_ref().unwrap(), &transform::lambda_alpha(&r1, 100.0));
                    let d = xs.len() as f64;
                    let prod: f64 = z.iter().enumerate().map(|(i, &zi)| {
                        let s: f64 = (1..=32).map(|j| {
                            let t = 2f64.powi(j) * zi;
                            (t - t.round()).abs() / 2f64.powi(j)
                        }).sum();
                        (1.0 + (i as f64 + 1.0) * s).powf(10.0 / d.powf(1.2))
                    }).product();
                    10.0 / (d * d) * prod - 10.0 / (d * d) + transform::f_pen(xs)
                }
                24 => {
                    let d = xs.len() as f64;
                    let mu0 = 2.5f64;
                    let s = 1.0 - 1.0 / (2.0 * (d + 20.0).sqrt() - 8.2);
                    let mu1 = -((mu0 * mu0 - 1.0) / s).sqrt();
                    let xhat: Vec<f64> = xs.iter().zip(&self.x_opt).map(|(x, o)| 2.0 * o.signum() * x).collect();
                    let inner: Vec<f64> = xhat.iter().map(|v| v - mu0).collect();
                    let r1 = transform::apply(self.rot.as_ref().unwrap(), &inner);
                    let z = transform::apply(self.rot2.as_ref().unwrap(), &transform::lambda_alpha(&r1, 100.0));
                    let sum0: f64 = xhat.iter().map(|v| (v - mu0).powi(2)).sum();
                    let sum1: f64 = xhat.iter().map(|v| (v - mu1).powi(2)).sum();
                    let cos_sum: f64 = z.iter().map(|v| (2.0 * PI * v).cos()).sum();
                    sum0.min(d + s * sum1) + 10.0 * (d - cos_sum) + 1e4 * transform::f_pen(xs)
                }
                _ => unreachable!(),
            };
            raw + self.f_opt
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::problem::Problem;
    use sezgi_core::space::{BlockValues, Genotype};

    fn g(xs: Vec<f64>) -> Genotype { Genotype { blocks: vec![BlockValues::Float(xs)] } }

    #[test]
    fn optimum_is_attained_at_x_opt() {
        for fid in [1u32, 2, 3, 8] {
            let p = BbobProblem::new(fid, 5, 1).unwrap();
            let f = p.evaluate_batch(&[g(p.x_opt().to_vec())])[0];
            assert!((f - p.f_opt()).abs() < 1e-9, "fid={fid}: {f} != {}", p.f_opt());
        }
    }

    #[test]
    fn x_opt_not_at_center() {
        let p = BbobProblem::new(1, 10, 1).unwrap();
        assert!(p.x_opt().iter().any(|&x| x.abs() > 1e-3), "optimum merkezde olmamalı");
    }

    #[test]
    fn instances_differ_and_are_deterministic() {
        let a1 = BbobProblem::new(3, 5, 1).unwrap();
        let a2 = BbobProblem::new(3, 5, 1).unwrap();
        let b = BbobProblem::new(3, 5, 2).unwrap();
        assert_eq!(a1.x_opt(), a2.x_opt());
        assert_ne!(a1.x_opt(), b.x_opt());
    }

    #[test]
    fn away_from_optimum_is_worse() {
        let p = BbobProblem::new(2, 5, 3).unwrap();
        let mut x = p.x_opt().to_vec();
        x[0] += 1.0;
        assert!(p.evaluate_batch(&[g(x)])[0] > p.f_opt());
    }

    #[test]
    fn unknown_fid_is_error() {
        assert!(matches!(BbobProblem::new(99, 5, 1), Err(BbobError::NotImplemented(99))));
    }

    #[test]
    fn rotation_is_orthogonal() {
        let r = crate::bbob::transform::rotation_matrix(6, 12345);
        for i in 0..6 {
            for j in 0..6 {
                let dot: f64 = (0..6).map(|k| r[k][i] * r[k][j]).sum();
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expect).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn m1_fids_instance_values_pinned() {
        // Bu değerler bu testin İLK koşusunda mevcut koddan alınıp sabitlenir (PIN-ME
        // prosedürü, T17/M1 ile aynı): her fid için instance=1, dim=5,
        // x=[0.5,-1.0,2.0,0.0,-3.0] noktasında f değeri bit'leri.
        let probe = g(vec![0.5, -1.0, 2.0, 0.0, -3.0]);
        for (fid, expected_hex) in [(1u32, "c05a1069cca05d30"), (2, "414d7d483f2ad056"), (3, "c03b12604f3ed308"), (8, "40d41e5080f1a8d2")] {
            let p = BbobProblem::new(fid, 5, 1).unwrap();
            let f = p.evaluate_batch(std::slice::from_ref(&probe))[0];
            let got = format!("{:016x}", f.to_bits());
            assert_eq!(got, expected_hex, "fid {fid}: M1 davranışı drift etti!");
        }
    }

    #[test]
    fn f4_f5_optimum_attained_and_offcenter_worse() {
        for fid in [4u32, 5] {
            let p = BbobProblem::new(fid, 5, 1).unwrap();
            let at_opt = p.evaluate_batch(&[g(p.x_opt().to_vec())])[0];
            assert!((at_opt - p.f_opt()).abs() < 1e-6, "fid {fid}: {at_opt} != {}", p.f_opt());
            let mut x = p.x_opt().to_vec();
            x[0] = (x[0] - 0.7).clamp(-5.0, 5.0); // f5'te sınırdan içeri it
            if x == p.x_opt() { x[0] += 0.7; }
            assert!(p.evaluate_batch(&[g(x)])[0] > p.f_opt(), "fid {fid}: optimum-dışı daha iyi çıktı");
        }
    }

    #[test]
    fn f5_optimum_on_boundary() {
        let p = BbobProblem::new(5, 6, 2).unwrap();
        assert!(p.x_opt().iter().all(|&x| x == 5.0 || x == -5.0));
    }

    #[test]
    fn f6_f7_f9_optimum_and_determinism() {
        for fid in [6u32, 7, 9] {
            let p = BbobProblem::new(fid, 5, 1).unwrap();
            let at_opt = p.evaluate_batch(&[g(p.x_opt().to_vec())])[0];
            assert!((at_opt - p.f_opt()).abs() < 1e-6, "fid {fid}: {}", at_opt - p.f_opt());
            let mut x = p.x_opt().to_vec();
            x[1] += 1.0;
            assert!(p.evaluate_batch(&[g(x)])[0] > p.f_opt(), "fid {fid}");
            let p2 = BbobProblem::new(fid, 5, 1).unwrap();
            assert_eq!(p.x_opt(), p2.x_opt(), "fid {fid} instance determinizmi");
        }
    }

    #[test]
    fn f7_has_plateaus() {
        // Step-ellipsoidal: optimumdan yeterince uzak iki yakın nokta aynı f'i vermeli
        let p = BbobProblem::new(7, 5, 1).unwrap();
        let mut a = p.x_opt().to_vec(); a[0] += 2.0;
        let mut b = a.clone(); b[0] += 1e-4;
        let fa = p.evaluate_batch(&[g(a)])[0];
        let fb = p.evaluate_batch(&[g(b)])[0];
        assert_eq!(fa, fb, "basamak platosu bekleniyordu");
    }

    #[test]
    fn f10_to_f14_optimum_and_conditioning() {
        for fid in 10u32..=14 {
            let p = BbobProblem::new(fid, 5, 3).unwrap();
            let at_opt = p.evaluate_batch(&[g(p.x_opt().to_vec())])[0];
            assert!((at_opt - p.f_opt()).abs() < 1e-6, "fid {fid}: {}", at_opt - p.f_opt());
            let mut x = p.x_opt().to_vec();
            x[2] -= 0.5;
            assert!(p.evaluate_batch(&[g(x)])[0] > p.f_opt(), "fid {fid}");
        }
    }

    #[test]
    fn discus_and_cigar_axis_asymmetry() {
        // saf çekirdek testi: discus'ta ilk eksen 1e6 kat ağır, cigar'da tersi
        assert!(functions::discus(&[1.0, 0.0]) > functions::discus(&[0.0, 1.0]) * 1e5);
        assert!(functions::bent_cigar(&[0.0, 1.0]) > functions::bent_cigar(&[1.0, 0.0]) * 1e5);
    }

    #[test]
    fn f15_to_f19_optimum_attained() {
        for fid in 15u32..=19 {
            let p = BbobProblem::new(fid, 5, 1).unwrap();
            let at_opt = p.evaluate_batch(&[g(p.x_opt().to_vec())])[0];
            assert!((at_opt - p.f_opt()).abs() < 1e-6, "fid {fid}: fark {}", at_opt - p.f_opt());
            let mut x = p.x_opt().to_vec();
            x[0] += 0.9; x[3] -= 0.4;
            assert!(p.evaluate_batch(&[g(x)])[0] > p.f_opt(), "fid {fid}");
        }
    }

    #[test]
    fn weierstrass_core_zero_at_origin() {
        assert!(functions::weierstrass(&[0.0; 5]).abs() < 1e-9);
    }

    #[test]
    fn schaffers_core_zero_at_origin_positive_elsewhere() {
        assert!(functions::schaffers_f7(&[0.0; 4]).abs() < 1e-12);
        assert!(functions::schaffers_f7(&[1.0, -2.0, 0.5, 3.0]) > 0.0);
    }

    #[test]
    fn f20_to_f24_optimum_attained() {
        for fid in 20u32..=24 {
            let p = BbobProblem::new(fid, 5, 1).unwrap();
            let at_opt = p.evaluate_batch(&[g(p.x_opt().to_vec())])[0];
            assert!((at_opt - p.f_opt()).abs() < 1e-6, "fid {fid}: fark {}", at_opt - p.f_opt());
        }
    }

    #[test]
    fn gallagher_first_peak_dominates() {
        let p = BbobProblem::new(21, 5, 1).unwrap();
        // optimumdan uzakta değer f_opt+10²'ye (t_osz olmadan kaba sınır) yaklaşmalı ama altında kalmalı
        let far = g(vec![4.9; 5]);
        let f_far = p.evaluate_batch(&[far])[0];
        assert!(f_far > p.f_opt());
    }
}
