use sezgi_core::component::*;
use sezgi_core::problem::Population;
use sezgi_core::rng::RngStream;
use sezgi_core::space::{Block, BlockValues, Genotype};

/// SBX: two children for a single gene pair (Deb & Agrawal 1995).
pub fn sbx_pair(p1: f64, p2: f64, eta: f64, rng: &mut RngStream) -> (f64, f64) {
    let u = rng.next_f64();
    let beta = if u <= 0.5 {
        (2.0 * u).powf(1.0 / (eta + 1.0))
    } else {
        (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (eta + 1.0))
    };
    (0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2),
     0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2))
}

/// Polynomial mutation, boundary-aware (Deb & Goyal 1996).
pub fn polynomial_mutate(x: f64, lo: f64, hi: f64, eta: f64, rng: &mut RngStream) -> f64 {
    let u = rng.next_f64();
    let d = hi - lo;
    let (d1, d2) = ((x - lo) / d, (hi - x) / d);
    let dq = if u <= 0.5 {
        let v = 2.0 * u + (1.0 - 2.0 * u) * (1.0 - d1).powf(eta + 1.0);
        v.powf(1.0 / (eta + 1.0)) - 1.0
    } else {
        let v = 2.0 * (1.0 - u) + (2.0 * u - 1.0) * (1.0 - d2).powf(eta + 1.0);
        1.0 - v.powf(1.0 / (eta + 1.0))
    };
    (x + dq * d).clamp(lo, hi)
}

pub struct GaRealGenerator {
    pub tournament_k: usize,
    pub pc: f64,
    pub pm_per_gene: Option<f64>,
    pub eta_c: f64,
    pub eta_m: f64,
}

impl GaRealGenerator {
    pub fn from_params(p: &serde_json::Value) -> Result<Self, ComponentError> {
        let err = |reason: String| ComponentError::InvalidParams {
            kind: "gen/ga-real".into(), reason };
        let g = Self {
            tournament_k: p.get("tournament_k").and_then(|v| v.as_u64()).unwrap_or(2) as usize,
            pc: p.get("pc").and_then(|v| v.as_f64()).unwrap_or(0.9),
            pm_per_gene: p.get("pm_per_gene").and_then(|v| v.as_f64()),
            eta_c: p.get("eta_c").and_then(|v| v.as_f64()).unwrap_or(15.0),
            eta_m: p.get("eta_m").and_then(|v| v.as_f64()).unwrap_or(20.0),
        };
        if g.tournament_k == 0 { return Err(err("tournament_k must be >= 1".into())); }
        if !(0.0..=1.0).contains(&g.pc) { return Err(err(format!("pc outside [0,1]: {}", g.pc))); }
        Ok(g)
    }

    fn tournament(&self, pop: &Population, rng: &mut RngStream) -> usize {
        let mut best = rng.next_below(pop.len() as u64) as usize;
        for _ in 1..self.tournament_k {
            let c = rng.next_below(pop.len() as u64) as usize;
            if pop.fitness[c] < pop.fitness[best] { best = c; }
        }
        best
    }
}

impl Generator for GaRealGenerator {
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        let Block::Float { lo, hi, n: dim } = ctx.space.blocks()[0] else { unreachable!() };
        let pm = self.pm_per_gene.unwrap_or(1.0 / dim as f64);
        let mut out = Vec::with_capacity(pop.len());
        while out.len() < pop.len() {
            let (i1, i2) = (self.tournament(pop, ctx.rng), self.tournament(pop, ctx.rng));
            let g1 = match &pop.individuals[i1].blocks[0] {
                BlockValues::Float(x) => x.clone(), _ => unreachable!() };
            let g2 = match &pop.individuals[i2].blocks[0] {
                BlockValues::Float(x) => x.clone(), _ => unreachable!() };
            let (mut c1, mut c2) = (g1.clone(), g2.clone());
            if ctx.rng.next_f64() < self.pc {
                for j in 0..dim {
                    let (a, b) = sbx_pair(g1[j], g2[j], self.eta_c, ctx.rng);
                    c1[j] = a; c2[j] = b;
                }
            }
            for c in [&mut c1, &mut c2] {
                for x in c.iter_mut() {
                    if ctx.rng.next_f64() < pm {
                        *x = polynomial_mutate(*x, lo, hi, self.eta_m, ctx.rng);
                    }
                }
            }
            out.push(Genotype { blocks: vec![BlockValues::Float(c1)] });
            if out.len() < pop.len() {
                out.push(Genotype { blocks: vec![BlockValues::Float(c2)] });
            }
        }
        out
    }
    fn meta(&self) -> ComponentMeta {
        ComponentMeta { kind: "gen/ga-real",
            supported_blocks: SupportedBlocks::Only(vec!["float"]),
            requires: vec![], provides: vec![] }
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_generator("gen/ga-real", |p| Ok(Box::new(GaRealGenerator::from_params(p)?)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::rng::RngStream;

    #[test]
    fn sbx_children_average_preserves_parents_mean() {
        let mut rng = RngStream::from_master(3, &[0]);
        let (c1, c2) = sbx_pair(1.0, 3.0, 15.0, &mut rng);
        assert!(((c1 + c2) / 2.0 - 2.0).abs() < 1e-9);
    }

    #[test]
    fn polynomial_mutation_stays_in_bounds() {
        let mut rng = RngStream::from_master(4, &[0]);
        for _ in 0..1000 {
            let y = polynomial_mutate(0.9, -1.0, 1.0, 20.0, &mut rng);
            assert!((-1.0..=1.0).contains(&y));
        }
    }

    #[test]
    fn params_validate() {
        assert!(GaRealGenerator::from_params(&serde_json::json!({"pc": 1.5})).is_err());
        assert!(GaRealGenerator::from_params(&serde_json::json!({"tournament_k": 0})).is_err());
    }
}
