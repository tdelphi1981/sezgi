pub mod functions;
pub mod transform;

use sezgi_core::problem::Problem;
use sezgi_core::rng::RngStream;
use sezgi_core::space::{Block, BlockValues, Genotype, SearchSpace};

pub const BBOB_SEED_BASE: u64 = 0x5EC1;

#[derive(Debug, thiserror::Error)]
pub enum BbobError {
    #[error("fid {0} bu sürümde yok (M1: 1,2,3,8)")]
    NotImplemented(u32),
    #[error("dim >= 2 olmalı")]
    BadDim,
}

pub struct BbobProblem {
    fid: u32,
    x_opt: Vec<f64>,
    f_opt: f64,
    rot: Option<Vec<Vec<f64>>>,
    space: SearchSpace,
    pub instance: u32,
}

impl BbobProblem {
    pub fn new(fid: u32, dim: usize, instance: u32) -> Result<Self, BbobError> {
        if dim < 2 { return Err(BbobError::BadDim); }
        if ![1, 2, 3, 8].contains(&fid) { return Err(BbobError::NotImplemented(fid)); }
        let mut rng = RngStream::from_master(BBOB_SEED_BASE + fid as u64,
                                             &[instance as u64]);
        let x_opt: Vec<f64> = (0..dim).map(|_| -4.0 + 8.0 * rng.next_f64()).collect();
        let f_opt = -200.0 + 400.0 * rng.next_f64();
        let rot = match fid {
            2 | 3 | 8 => Some(transform::rotation_matrix(dim, rng.next_u64())),
            _ => None,
        };
        let space = SearchSpace::new(vec![Block::Float { lo: -5.0, hi: 5.0, n: dim }])
            .expect("sabit sınırlar geçerli");
        Ok(Self { fid, x_opt, f_opt, rot, space, instance })
    }

    pub fn x_opt(&self) -> &[f64] { &self.x_opt }
    pub fn f_opt(&self) -> f64 { self.f_opt }
    pub fn fid(&self) -> u32 { self.fid }
    pub fn name(&self) -> &'static str {
        match self.fid { 1 => "Sphere", 2 => "Ellipsoidal", 3 => "Rastrigin",
                         8 => "Rosenbrock", _ => unreachable!() }
    }
}

impl Problem for BbobProblem {
    fn space(&self) -> &SearchSpace { &self.space }
    fn optimum(&self) -> Option<f64> { Some(self.f_opt) }

    fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<f64> {
        pop.iter().map(|g| {
            let BlockValues::Float(xs) = &g.blocks[0] else { return f64::INFINITY };
            let shifted: Vec<f64> = xs.iter().zip(&self.x_opt).map(|(x, o)| x - o).collect();
            let z = match &self.rot {
                Some(r) => transform::apply(r, &shifted),
                None => shifted,
            };
            let raw = match self.fid {
                1 => functions::sphere(&z),
                2 => functions::ellipsoidal(&z),
                3 => functions::rastrigin(&z),
                8 => {
                    // optimumda z=0 → w=1 kaydırması
                    let w: Vec<f64> = z.iter().map(|v| v + 1.0).collect();
                    functions::rosenbrock(&w)
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
}
