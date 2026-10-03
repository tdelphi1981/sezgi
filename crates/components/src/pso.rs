use sezgi_core::component::*;
use sezgi_core::problem::Population;
use sezgi_core::space::{BlockValues, Genotype};
use sezgi_core::state::StateReq;

pub struct PsoGenerator { pub w: f64, pub c1: f64, pub c2: f64 }

impl PsoGenerator {
    pub fn from_params(p: &serde_json::Value) -> Result<Self, ComponentError> {
        Ok(Self {
            w: p.get("w").and_then(|v| v.as_f64()).unwrap_or(0.7298),
            c1: p.get("c1").and_then(|v| v.as_f64()).unwrap_or(1.49618),
            c2: p.get("c2").and_then(|v| v.as_f64()).unwrap_or(1.49618),
        })
    }
}

fn floats(g: &Genotype) -> &Vec<f64> {
    match &g.blocks[0] { BlockValues::Float(x) => x, _ => unreachable!() }
}

impl Generator for PsoGenerator {
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        let n = pop.len();
        let dim = floats(&pop.individuals[0]).len();
        if !ctx.bb.contains("pso_velocity") {
            ctx.bb.insert("pso_velocity", vec![vec![0.0f64; dim]; n]);
            ctx.bb.insert("pso_pbest",
                pop.individuals.iter().map(|g| floats(g).clone()).collect::<Vec<_>>());
            ctx.bb.insert("pso_pbest_f", pop.fitness.clone());
        } else {
            // Since 0.1.6 the pbest fold lives here (it used to run in
            // `replace/pso-commit` at the end of the previous iteration):
            // `pop` is the previous iteration's committed offspring, so a
            // strictly better fitness replaces the personal best. RNG-free.
            let new_pos: Vec<&Vec<f64>> = pop.individuals.iter().map(floats).collect();
            let pf = ctx.bb.get_mut::<Vec<f64>>("pso_pbest_f").unwrap();
            let mut improved: Vec<usize> = Vec::new();
            for i in 0..n.min(pf.len()) {
                if pop.fitness[i] < pf[i] { pf[i] = pop.fitness[i]; improved.push(i); }
            }
            let pb = ctx.bb.get_mut::<Vec<Vec<f64>>>("pso_pbest").unwrap();
            for i in improved { pb[i] = new_pos[i].clone(); }
        }
        let gbest = {
            let pf = ctx.bb.get::<Vec<f64>>("pso_pbest_f").unwrap();
            let gi = pf.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
            ctx.bb.get::<Vec<Vec<f64>>>("pso_pbest").unwrap()[gi].clone()
        };
        let pbest = ctx.bb.get::<Vec<Vec<f64>>>("pso_pbest").unwrap().clone();
        let vel = ctx.bb.get_mut::<Vec<Vec<f64>>>("pso_velocity").unwrap();
        (0..n).map(|i| {
            let x = floats(&pop.individuals[i]);
            let xs: Vec<f64> = (0..dim).map(|j| {
                let (r1, r2) = (ctx.rng.next_f64(), ctx.rng.next_f64());
                vel[i][j] = self.w * vel[i][j]
                    + self.c1 * r1 * (pbest[i][j] - x[j])
                    + self.c2 * r2 * (gbest[j] - x[j]);
                x[j] + vel[i][j]
            }).collect();
            Genotype { blocks: vec![BlockValues::Float(xs)] }
        }).collect()
    }
    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/pso", SupportedBlocks::Only(vec!["float"]))
            .with_provides(vec![
                StateReq::of::<Vec<Vec<f64>>>("pso_velocity"),
                StateReq::of::<Vec<Vec<f64>>>("pso_pbest"),
                StateReq::of::<Vec<f64>>("pso_pbest_f"),
            ])
    }
}

/// Commit replacer for `gen/pso`: positions carry over unconditionally
/// (`pop = offspring`).
///
/// Since 0.1.6 the pbest/pbest_f fold lives in `gen/pso` (it runs at the top
/// of the next `generate()` call from the committed population); this
/// component is kept for spec compatibility and only performs the commit.
pub struct PsoCommit;
impl Replacer for PsoCommit {
    fn replace(&self, pop: &mut Population, oi: Vec<Genotype>, of: Vec<f64>, _ctx: &mut Ctx) {
        pop.individuals = oi;
        pop.fitness = of;
    }
    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("replace/pso-commit", SupportedBlocks::All)
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_generator("gen/pso", |p| Ok(Box::new(PsoGenerator::from_params(p)?)));
    reg.register_replacer("replace/pso-commit", |_| Ok(Box::new(PsoCommit)));
}
