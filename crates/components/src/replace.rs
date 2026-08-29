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

/// Replace the population's single worst member (highest fitness --
/// lower-is-better; ties -> lower index, mirroring `Population::best_index`'s
/// tie convention for the opposite extremum) with each offspring in turn,
/// but only if that offspring is strictly better than the CURRENT worst at
/// the time it is considered -- so with more than one offspring, the worst
/// slot is re-evaluated after each acceptance. Added for `gen/hs` (Task 7,
/// M2d-3), after checking that no existing kind fits: `one_to_one` compares
/// offspring `i` to parent `i` BY INDEX, which is meaningless for a
/// single-harmony generator whose one offspring is always at index 0
/// regardless of which population slot is actually worst. `mu_plus_lambda`
/// happens to be numerically equivalent for exactly one offspring (keeping
/// the best `mu` of the combined `mu+1` pool drops exactly the worst
/// individual when the offspring doesn't beat it, or drops the original
/// worst otherwise) -- but as a side effect it fully re-sorts and
/// re-indexes the ENTIRE population on every call, which is not what
/// "replace the worst in place" means and would be a surprising,
/// undocumented reordering for any future caller (HS's own memory-index
/// draw doesn't care about order, but a general "worst-replacement" kind
/// shouldn't silently reorder). This kind replaces IN PLACE: every other
/// slot's identity and index are left untouched.
pub fn replace_worst_if_better(pop: &mut Population, off_i: Vec<Genotype>, off_f: Vec<f64>) {
    for (gi, fi) in off_i.into_iter().zip(off_f) {
        let mut worst = 0;
        for k in 1..pop.len() {
            if pop.fitness[k] > pop.fitness[worst] {
                worst = k;
            }
        }
        if fi < pop.fitness[worst] {
            pop.individuals[worst] = gi;
            pop.fitness[worst] = fi;
        }
    }
}

pub struct ReplaceWorstIfBetter;
impl Replacer for ReplaceWorstIfBetter {
    fn replace(&self, pop: &mut Population, oi: Vec<Genotype>, of: Vec<f64>, _c: &mut Ctx) {
        replace_worst_if_better(pop, oi, of);
    }
    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("replace/worst-if-better", SupportedBlocks::All)
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_replacer("replace/one-to-one-greedy", |_| Ok(Box::new(OneToOneGreedy)));
    reg.register_replacer("replace/mu-plus-lambda", |_| Ok(Box::new(MuPlusLambda)));
    reg.register_replacer("replace/generational", |_| Ok(Box::new(Generational)));
    reg.register_replacer("replace/worst-if-better", |_| Ok(Box::new(ReplaceWorstIfBetter)));
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

    fn pop3() -> Population {
        // worst (highest fitness) is index 2.
        Population { individuals: vec![g(1.0), g(2.0), g(3.0)], fitness: vec![10.0, 20.0, 30.0] }
    }

    #[test]
    fn worst_if_better_replaces_worst_slot_in_place_when_better() {
        let mut p = pop3();
        replace_worst_if_better(&mut p, vec![g(99.0)], vec![15.0]); // beats worst (30.0), not slots 0/1
        assert_eq!(p.fitness, vec![10.0, 20.0, 15.0]);
        assert_eq!(p.individuals[2], g(99.0));
        // Other slots' identity/index untouched.
        assert_eq!(p.individuals[0], g(1.0));
        assert_eq!(p.individuals[1], g(2.0));
    }

    #[test]
    fn worst_if_better_leaves_pop_unchanged_when_offspring_not_better() {
        let mut p = pop3();
        let before = p.clone();
        replace_worst_if_better(&mut p, vec![g(0.0)], vec![30.0]); // ties the worst, not strictly better
        assert_eq!(p.fitness, before.fitness);
        assert_eq!(p.individuals, before.individuals);
    }

    #[test]
    fn worst_if_better_ties_break_to_lower_index_as_worst() {
        // Two ties at the max fitness (20.0): indices 1 and 2. Lower index (1)
        // is tracked as "the worst" and is the one replaced.
        let mut p = Population {
            individuals: vec![g(1.0), g(2.0), g(3.0)],
            fitness: vec![10.0, 20.0, 20.0],
        };
        replace_worst_if_better(&mut p, vec![g(99.0)], vec![5.0]);
        assert_eq!(p.fitness, vec![10.0, 5.0, 20.0]);
        assert_eq!(p.individuals[1], g(99.0));
        assert_eq!(p.individuals[2], g(3.0)); // untouched
    }

    #[test]
    fn worst_if_better_reevaluates_worst_across_multiple_offspring() {
        // Two offspring in sequence: the first beats the original worst
        // (30.0) and takes its slot; the SECOND offspring must then be
        // compared against the NEW worst (20.0 at index 1), not the stale
        // original worst.
        let mut p = pop3();
        replace_worst_if_better(&mut p, vec![g(50.0), g(60.0)], vec![15.0, 12.0]);
        // After offspring 1 (f=15 < 30): fitness = [10, 20, 15], individuals[2]=g(50.0)
        // After offspring 2 (f=12 < 20, new worst is index 1): fitness = [10, 12, 15]
        assert_eq!(p.fitness, vec![10.0, 12.0, 15.0]);
        assert_eq!(p.individuals[1], g(60.0));
        assert_eq!(p.individuals[2], g(50.0));
    }
}
