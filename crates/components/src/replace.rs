use sezgi_core::component::*;
use sezgi_core::problem::Population;
use sezgi_core::space::Genotype;

pub fn one_to_one(pop: &mut Population, off_i: Vec<Genotype>, off_f: Vec<f64>) {
    for (i, (gi, fi)) in off_i.into_iter().zip(off_f).enumerate() {
        if i < pop.len() && fi < pop.fitness[i] {
            pop.individuals[i] = gi;
            pop.fitness[i] = fi;
        }
    }
}

pub fn mu_plus_lambda(pop: &mut Population, off_i: Vec<Genotype>, off_f: Vec<f64>) {
    let mu = pop.len();
    let mut all: Vec<(Genotype, f64)> = pop.individuals.drain(..)
        .zip(pop.fitness.drain(..))
        .chain(off_i.into_iter().zip(off_f))
        .collect();
    all.sort_by(|a, b| a.1.total_cmp(&b.1));
    all.truncate(mu);
    for (g, f) in all { pop.individuals.push(g); pop.fitness.push(f); }
}

pub struct OneToOneGreedy;
impl Replacer for OneToOneGreedy {
    fn replace(&self, pop: &mut Population, oi: Vec<Genotype>, of: Vec<f64>, _c: &mut Ctx) {
        one_to_one(pop, oi, of);
    }
    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("replace/one-to-one-greedy", SupportedBlocks::All)
    }
}

pub struct MuPlusLambda;
impl Replacer for MuPlusLambda {
    fn replace(&self, pop: &mut Population, oi: Vec<Genotype>, of: Vec<f64>, _c: &mut Ctx) {
        mu_plus_lambda(pop, oi, of);
    }
    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("replace/mu-plus-lambda", SupportedBlocks::All)
    }
}

/// Unconditional generational replacement: the offspring become the new
/// population outright (no per-individual fitness comparison, no coupled
/// blackboard state). Added for `gen/gwo` (Task 5, M2d-3), after checking
/// that no existing kind fits: `replace/one-to-one-greedy` only keeps an
/// offspring when it beats its parent, `replace/mu-plus-lambda` is elitist
/// (selects the best `mu` from the combined pool), and the only other
/// unconditional replacer in this crate -- `replace/pso-commit` (`pso.rs`)
/// and `replace/cma-update` (`cma.rs`) -- both bundle algorithm-specific
/// bookkeeping (pbest/velocity, CMA state) that a plain GWO spec doesn't
/// provide, so their `meta().requires` would fail spec validation. This is
/// therefore a new, clearly-named, state-free kind, reusable by any future
/// non-elitist metaphor preset that needs unconditional replacement without
/// extra coupling.
pub struct Generational;
impl Replacer for Generational {
    fn replace(&self, pop: &mut Population, oi: Vec<Genotype>, of: Vec<f64>, _c: &mut Ctx) {
        pop.individuals = oi;
        pop.fitness = of;
    }
    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("replace/generational", SupportedBlocks::All)
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_replacer("replace/one-to-one-greedy", |_| Ok(Box::new(OneToOneGreedy)));
    reg.register_replacer("replace/mu-plus-lambda", |_| Ok(Box::new(MuPlusLambda)));
    reg.register_replacer("replace/generational", |_| Ok(Box::new(Generational)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::problem::Population;
    use sezgi_core::space::{BlockValues, Genotype};

    fn g(x: f64) -> Genotype { Genotype { blocks: vec![BlockValues::Float(vec![x])] } }
    fn pop() -> Population {
        Population { individuals: vec![g(1.0), g(2.0)], fitness: vec![10.0, 20.0] }
    }
    // Note: the replace logic is factored into standalone functions so it can be
    // tested as pure logic without a Ctx: one_to_one(pop, off_i, off_f) and mu_plus_lambda(pop, off_i, off_f)

    #[test]
    fn one_to_one_keeps_better_parent() {
        let mut p = pop();
        one_to_one(&mut p, vec![g(9.0), g(8.0)], vec![15.0, 5.0]);
        assert_eq!(p.fitness, vec![10.0, 5.0]); // 0: parent stayed, 1: offspring won
    }

    #[test]
    fn mu_plus_lambda_takes_global_best() {
        let mut p = pop();
        mu_plus_lambda(&mut p, vec![g(9.0), g(8.0)], vec![5.0, 30.0]);
        assert_eq!(p.fitness, vec![5.0, 10.0]); // best 2 of the combined pool
    }
}
