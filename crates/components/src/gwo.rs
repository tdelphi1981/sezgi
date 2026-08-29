use sezgi_core::component::*;
use sezgi_core::problem::Population;
use sezgi_core::space::{BlockValues, Genotype};

/// Grey Wolf Optimizer (Mirjalili, Mirjalili & Lewis 2014, "Grey Wolf
/// Optimizer", *Advances in Engineering Software*) — a **labeled metaphor
/// preset**: faithful to the primary source's equations, with a pinned
/// deterministic draw order and property tests, but NOT validated against
/// the paper's (or any other publication's) reported benchmark numbers. See
/// Camacho-Villalón, Birattari & Stützle (ANTS 2020 / *ITOR*) for the
/// equivalence critique showing GWO's update mechanism is, component for
/// component, a relabeled special case of older PSO-family search
/// operators — cited here conservatively, as background on why this is
/// "labeled metaphor" rather than a mechanism sezgi treats as novel.
///
/// **Pinned update rule** (part of the RNG-stream contract):
/// - `a = 2 − 2·progress`, where `progress = ctx.eval.used() / ctx.eval.budget()`
///   (clamped to `[0, 1]`; `0` if `budget == 0`), computed **once per
///   `generate` call**, before any RNG draws.
/// - Leaders: sort population indices by fitness ascending (ties → lower
///   index first); alpha, beta, delta = the best three. `min_pop = 3`.
/// - For each wolf `i` (population order), each dimension `d` (index
///   order), for each leader `L` in `(alpha, beta, delta)` order: draw
///   `r1 = rng.next_f64()`, then `r2 = rng.next_f64()` (**exactly 6 draws
///   per `(i, d)`, always both, regardless of `a`**); `A = 2a·r1 − a`,
///   `C = 2·r2`; `contrib_L = X_L[d] − A·|C·X_L[d] − X_i[d]|`. The new
///   `X_i[d]` is the mean of the three leaders' contributions.
///
/// The per-dimension math is factored into [`gwo_dim_step`] so it can be
/// unit-tested directly (e.g. the `a = 0` limit, where every contribution
/// collapses exactly to its leader's coordinate) without needing to fake
/// `Ctx`/`Evaluator` progress. [`GwoGenerator::generate`] is a thin loop
/// over `(i, d)` that draws the 6 values in the pinned order and calls the
/// helper.
///
/// Boundary handling and replacement are NOT part of this generator: the
/// `presets::gwo` preset reuses `boundary/clamp` (same as `presets::pso`)
/// and the new `replace/generational` kind (see `replace.rs`) for GWO's
/// unconditional generational replacement.
pub fn gwo_dim_step(a: f64, leaders_d: [f64; 3], x_d: f64, draws: &[f64; 6]) -> f64 {
    let mut sum = 0.0;
    for k in 0..3 {
        let r1 = draws[2 * k];
        let r2 = draws[2 * k + 1];
        let big_a = 2.0 * a * r1 - a;
        let c = 2.0 * r2;
        sum += leaders_d[k] - big_a * (c * leaders_d[k] - x_d).abs();
    }
    sum / 3.0
}

pub struct GwoGenerator;

impl GwoGenerator {
    pub fn from_params(_p: &serde_json::Value) -> Result<Self, ComponentError> {
        // No tunable parameters (a is fully determined by ctx.eval progress).
        Ok(Self)
    }

    fn floats(g: &Genotype) -> &Vec<f64> {
        match &g.blocks[0] { BlockValues::Float(x) => x, _ => unreachable!() }
    }
}

impl Generator for GwoGenerator {
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        assert!(pop.len() >= 3, "gen/gwo requires a population of at least 3 (pop_size={})", pop.len());
        let n = pop.len();
        let dim = Self::floats(&pop.individuals[0]).len();

        let used = ctx.eval.used() as f64;
        let budget = ctx.eval.budget() as f64;
        let progress = if budget > 0.0 { (used / budget).clamp(0.0, 1.0) } else { 0.0 };
        let a = 2.0 - 2.0 * progress;

        // Leaders: best 3 by fitness ascending, ties -> lower index -- pinned.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&x, &y| pop.fitness[x].total_cmp(&pop.fitness[y]).then(x.cmp(&y)));
        let leaders = [order[0], order[1], order[2]];
        let leader_x: [&Vec<f64>; 3] = [
            Self::floats(&pop.individuals[leaders[0]]),
            Self::floats(&pop.individuals[leaders[1]]),
            Self::floats(&pop.individuals[leaders[2]]),
        ];

        (0..n).map(|i| {
            let x = Self::floats(&pop.individuals[i]);
            let xs: Vec<f64> = (0..dim).map(|d| {
                let leaders_d = [leader_x[0][d], leader_x[1][d], leader_x[2][d]];
                // Pinned draw order: for this (i, d), r1 then r2 for alpha,
                // then r1 then r2 for beta, then r1 then r2 for delta.
                let mut draws = [0.0f64; 6];
                for slot in draws.iter_mut() {
                    *slot = ctx.rng.next_f64();
                }
                gwo_dim_step(a, leaders_d, x[d], &draws)
            }).collect();
            Genotype { blocks: vec![BlockValues::Float(xs)] }
        }).collect()
    }

    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/gwo", SupportedBlocks::Only(vec!["float"])).with_min_pop(3)
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_generator("gen/gwo", |p| Ok(Box::new(GwoGenerator::from_params(p)?)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::problem::{Problem, SphereShifted, Evaluator};
    use sezgi_core::rng::RngStream;
    use sezgi_core::state::Blackboard;

    fn g(xs: Vec<f64>) -> Genotype { Genotype { blocks: vec![BlockValues::Float(xs)] } }

    fn pop_5d(n: usize, dim: usize) -> Population {
        // Distinct fitness so leader selection is unambiguous.
        Population {
            individuals: (0..n).map(|i| g((0..dim).map(|d| (i * dim + d) as f64 * 0.37 - 1.0).collect())).collect(),
            fitness: (0..n).map(|i| (n - i) as f64).collect(), // higher index = better (lower fitness)
        }
    }

    #[test]
    fn a_zero_limit_gives_exact_leader_centroid() {
        let leaders_d = [1.0, 2.0, 3.0];
        // Arbitrary draws: when a == 0, A == 2*a*r1 - a == 0 exactly regardless
        // of r1/r2, so every contribution collapses to its leader's coordinate.
        let draws = [0.11, 0.92, 0.03, 0.71, 0.58, 0.14];
        let result = gwo_dim_step(0.0, leaders_d, 100.0, &draws);
        assert_eq!(result, 2.0, "mean of leaders (1+2+3)/3 exactly, x_d and draws must be irrelevant when a=0");
    }

    #[test]
    fn determinism_same_seed_bit_identical() {
        let n = 6; let dim = 4;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_5d(n, dim);

        let run = || {
            let mut evaluator = Evaluator::new(&p, 1000);
            let mut rng = RngStream::from_master(42, &[]);
            let mut bb = Blackboard::new();
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
            GwoGenerator.generate(&pop, &mut ctx)
        };

        let a = run();
        let b = run();
        for (ga, gb) in a.iter().zip(b.iter()) {
            let (xa, xb) = (GwoGenerator::floats(ga), GwoGenerator::floats(gb));
            assert_eq!(xa, xb, "same seed must produce bit-identical offspring");
        }
    }

    #[test]
    fn draw_count_is_exactly_6_times_n_times_dim() {
        let n = 5; let dim = 3;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_5d(n, dim);

        let mut rng = RngStream::from_master(7, &[]);
        let rng_before = rng.clone();
        let mut evaluator = Evaluator::new(&p, 1000);
        let mut bb = Blackboard::new();
        {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
            let _ = GwoGenerator.generate(&pop, &mut ctx);
        }

        // Twin stream: advance a clone of the pre-generate rng by exactly
        // 6*n*dim draws by hand, then compare the NEXT draw from each stream --
        // if generate() consumed a different number of draws, the two streams
        // would be at different internal states and the next draws would (with
        // overwhelming probability) differ.
        let mut twin = rng_before;
        for _ in 0..(6 * n * dim) { twin.next_f64(); }
        assert_eq!(rng.next_f64(), twin.next_f64(),
            "gen/gwo must consume exactly 6*n*dim RNG draws");
    }

    #[test]
    fn min_pop_below_3_panics() {
        let n = 2; let dim = 2;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_5d(n, dim);
        let mut evaluator = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            GwoGenerator.generate(&pop, &mut ctx)
        }));
        assert!(result.is_err(), "gen/gwo must reject pop_size < 3 at runtime as a backstop");
    }

    #[test]
    fn leaders_are_best_three_ties_broken_by_lower_index() {
        let dim = 1;
        // Two ties at the best value (0.0): indices 0 and 2. Third-best is index 4.
        let pop = Population {
            individuals: (0..5).map(|i| g(vec![i as f64])).collect(),
            fitness: vec![0.0, 5.0, 0.0, 3.0, 1.0],
        };
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 100);
        // Use a=0 (progress=1, budget fully consumed) so every offspring is
        // exactly the leader centroid regardless of RNG draws -- this exposes
        // exactly which three individuals were picked as leaders.
        evaluator.evaluate(&vec![g(vec![0.0]); 100]).unwrap();
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let off = GwoGenerator.generate(&pop, &mut ctx);
        // leaders (by fitness ascending, tie -> lower index) are 0, 2, 4 with x = 0, 2, 4.
        // Centroid = (0+2+4)/3 = 2.0 for every offspring.
        for o in &off {
            assert_eq!(GwoGenerator::floats(o), &vec![2.0]);
        }
    }
}
