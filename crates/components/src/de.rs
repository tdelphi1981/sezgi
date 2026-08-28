use sezgi_core::component::*;
use sezgi_core::problem::Population;
use sezgi_core::space::{BlockValues, Genotype};

#[derive(Debug, Clone, Copy)]
pub enum DeStrategy { Rand1, Best1 }

pub struct DeGenerator { pub strategy: DeStrategy, pub f: f64, pub cr: f64 }

impl DeGenerator {
    pub fn from_params(p: &serde_json::Value) -> Result<Self, ComponentError> {
        let err = |reason: String| ComponentError::InvalidParams {
            kind: "gen/de".into(), reason };
        let strategy = match p.get("strategy").and_then(|v| v.as_str()).unwrap_or("rand1") {
            "rand1" => DeStrategy::Rand1,
            "best1" => DeStrategy::Best1,
            s => return Err(err(format!("unknown strategy: {s}"))),
        };
        let f = p.get("f").and_then(|v| v.as_f64()).unwrap_or(0.5);
        let cr = p.get("cr").and_then(|v| v.as_f64()).unwrap_or(0.9);
        if !(0.0..=2.0).contains(&f) { return Err(err(format!("f outside [0,2]: {f}"))); }
        if !(0.0..=1.0).contains(&cr) { return Err(err(format!("cr outside [0,1]: {cr}"))); }
        Ok(Self { strategy, f, cr })
    }

    fn float_view(g: &Genotype) -> &Vec<f64> {
        match &g.blocks[0] { BlockValues::Float(xs) => xs, _ => unreachable!() }
    }
}

impl Generator for DeGenerator {
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        assert!(pop.len() >= 4, "gen/de requires a population of at least 4 (pop_size={})", pop.len());
        let n = pop.len();
        let best = pop.best_index().unwrap_or(0);
        (0..n).map(|i| {
            // three indices distinct from each other and from i
            let mut pick_distinct = |excluded: &[usize]| loop {
                let r = ctx.rng.next_below(n as u64) as usize;
                if r != i && !excluded.contains(&r) { break r; }
            };
            let r1 = pick_distinct(&[]);
            let r2 = pick_distinct(&[r1]);
            let r3 = pick_distinct(&[r1, r2]);

            let base = match self.strategy {
                DeStrategy::Rand1 => Self::float_view(&pop.individuals[r1]),
                DeStrategy::Best1 => Self::float_view(&pop.individuals[best]),
            };
            let a = Self::float_view(&pop.individuals[r2]);
            let b = Self::float_view(&pop.individuals[r3]);
            let target = Self::float_view(&pop.individuals[i]);
            let d = target.len();
            let j_rand = ctx.rng.next_below(d as u64) as usize;
            let xs = (0..d).map(|j| {
                if j == j_rand || ctx.rng.next_f64() < self.cr {
                    base[j] + self.f * (a[j] - b[j])
                } else { target[j] }
            }).collect();
            Genotype { blocks: vec![BlockValues::Float(xs)] }
        }).collect()
    }
    fn meta(&self) -> ComponentMeta {
        ComponentMeta { kind: "gen/de",
            supported_blocks: SupportedBlocks::Only(vec!["float"]),
            requires: vec![], provides: vec![] }
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_generator("gen/de", |p| Ok(Box::new(DeGenerator::from_params(p)?)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::problem::{Problem, SphereShifted, Evaluator};
    use sezgi_core::rng::RngStream;
    use sezgi_core::state::Blackboard;

    #[test]
    fn params_parse_and_validate() {
        let g = DeGenerator::from_params(&serde_json::json!(
            {"strategy": "best1", "f": 0.7, "cr": 0.8})).unwrap();
        assert!(matches!(g.strategy, DeStrategy::Best1));
        assert!(DeGenerator::from_params(&serde_json::json!({"f": -1.0})).is_err());
        assert!(DeGenerator::from_params(&serde_json::json!({"cr": 2.0})).is_err());
    }

    #[test]
    #[should_panic(expected = "at least 4")]
    fn small_population_panics() {
        use sezgi_core::problem::Population;

        let p = SphereShifted::new(vec![0.0], -5.0, 5.0);
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(42, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };

        // Create 3-individual population to trigger panic
        let pop = Population {
            individuals: vec![
                Genotype { blocks: vec![BlockValues::Float(vec![0.0])] },
                Genotype { blocks: vec![BlockValues::Float(vec![0.0])] },
                Genotype { blocks: vec![BlockValues::Float(vec![0.0])] },
            ],
            fitness: vec![0.0; 3],
        };

        let gen = DeGenerator::from_params(&serde_json::json!({})).unwrap();
        let _ = gen.generate(&pop, &mut ctx);
    }
}
