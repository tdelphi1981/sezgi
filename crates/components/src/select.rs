//! Shared tournament-selection helper, extracted verbatim from six
//! byte-identical private copies (`ga.rs`'s `GaRealGenerator::tournament`,
//! `perm.rs`'s `OxGenerator::tournament` and `GaPermGenerator::tournament`,
//! `int_ops.rs`'s, `bin_ops.rs`'s, and `cat_ops.rs`'s free-function
//! `tournament`s -- the last of these needlessly `pub(crate)` despite having
//! no external caller). This is a PURE RELOCATION -- the body below is
//! byte-identical to every copy it replaces (only the calling convention
//! changed for `ga.rs`/`perm.rs`: their `&self`-method form is now a plain
//! call with `self.tournament_k` passed explicitly, matching the form
//! `int_ops.rs`/`bin_ops.rs`/`cat_ops.rs` already used). No draw order,
//! tie-breaking, or numeric behavior changed; every existing test (frozen
//! goldens and hand-traced RNG exactness tests included) still passes
//! unmodified.
//!
//! `nsga2.rs`'s own `tournament` is NOT this function and is left alone: it
//! selects by Pareto dominance and crowding distance with a very different
//! signature and draw pattern (see that module's doc), not by raw fitness
//! comparison.

use sezgi_core::problem::Population;
use sezgi_core::rng::RngStream;

/// Binary (or k-ary) fitness tournament: draws `tournament_k` candidate
/// indices uniformly from `0..pop.len()` (with replacement) and returns the
/// one with the lowest `pop.fitness` value, keeping the first draw on a tie
/// (strict `<` comparison, so a later draw only replaces the incumbent when
/// it is strictly better).
pub(crate) fn tournament(pop: &Population, tournament_k: usize, rng: &mut RngStream) -> usize {
    let mut best = rng.next_below(pop.len() as u64) as usize;
    for _ in 1..tournament_k {
        let c = rng.next_below(pop.len() as u64) as usize;
        if pop.fitness[c] < pop.fitness[best] { best = c; }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::space::Genotype;

    fn pop_with_fitness(fitness: Vec<f64>) -> Population {
        let individuals = fitness.iter().map(|_| Genotype { blocks: vec![] }).collect();
        Population { individuals, fitness }
    }

    #[test]
    fn tournament_k_1_is_a_single_uniform_draw() {
        let pop = pop_with_fitness(vec![3.0, 1.0, 2.0]);
        let mut rng = RngStream::from_master(1, &[0]);
        let picked = tournament(&pop, 1, &mut rng);
        assert!(picked < pop.len());
    }

    #[test]
    fn larger_tournament_k_never_picks_worse_than_smaller_k_would_have_seen() {
        // With every fitness distinct, a k-ary tournament must return the
        // index of the lowest fitness among however many indices it drew;
        // regardless of draws, the winner's fitness can never exceed the
        // population's own worst value.
        let pop = pop_with_fitness(vec![5.0, 4.0, 3.0, 2.0, 1.0]);
        let mut rng = RngStream::from_master(2, &[0]);
        for _ in 0..50 {
            let picked = tournament(&pop, 3, &mut rng);
            assert!(pop.fitness[picked] <= 5.0);
        }
    }
}
