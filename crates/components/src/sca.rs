use sezgi_core::component::*;
use sezgi_core::problem::Population;
use sezgi_core::space::{BlockValues, Genotype};

/// Sine Cosine Algorithm (Mirjalili 2016, "SCA: A Sine Cosine Algorithm for
/// Solving Optimization Problems", *Knowledge-Based Systems* 96, 120-133) --
/// a **labeled metaphor preset**: faithful to the primary source's reference
/// MATLAB implementation, with a pinned deterministic draw order and
/// property tests, but NOT validated against the paper's (or any other
/// publication's) reported benchmark numbers. No established equivalence
/// critique covers SCA (the Camacho-Villalón/Dorigo/Stützle *ITOR*
/// six-algorithm critique covers GWO/MFO/WOA/FA/BA/ALO, not SCA) -- cited
/// here as primary-source only.
///
/// **Pinned update rule** (part of the RNG-stream contract), verified
/// directly against the author's reference `SCA.m` (MATLAB Central File
/// Exchange #54948/#55980, "SCA: A Sine Cosine Algorithm" / "Sine Cosine
/// Algorithm Toolbox", Seyedali Mirjalili) -- NOT assumed from the paper's
/// prose:
/// - `a = 2` (fixed constant, not a tunable parameter -- the source hardcodes
///   it inside the loop). `r1 = a − t·(a / Max_iteration)`, i.e.
///   `r1 = 2 − 2·progress` where `progress = ctx.eval.used() / ctx.eval.budget()`
///   (clamped to `[0, 1]`; `0` if `budget == 0`). `SCA.m` computes `r1` ONCE
///   PER ITERATION, before the loop over search agents -- this project's
///   equivalent is computing it once per `generate` call, before any RNG
///   draws (same convention as `gwo.rs`'s `a` / `woa.rs`'s `a`/`a2`).
/// - `Destination_position` (the paper's tracked best-so-far): here, the
///   **current population's** fitness argmin, ties -> lower index (via
///   [`Population::best_index`]) -- the same "current-generation best, not a
///   separately-persisted historical best" convention already pinned by
///   `gwo.rs` and `woa.rs` (the engine's `global_best` is a reporting-only
///   value not fed back into generators; see those modules' docs).
/// - For each search agent `i` (population order), each dimension `d`
///   (index order): draw `r2 = 2π · rng.next_f64()` (range `[0, 2π)`), then
///   `r3 = 2 · rng.next_f64()` (range `[0, 2)`), then `r4 = rng.next_f64()`
///   (range `[0, 1)`) -- **exactly 3 draws per `(i, d)`, in this order,
///   always all three, regardless of branch** (matches `SCA.m`'s literal
///   nested loop: `for i = 1:size(X,1) ... for j = 1:size(X,2) ...
///   r2=(2*pi)*rand(); r3=2*rand; r4=rand();`). No draw happens per-agent or
///   per-generation other than the deterministic `r1` above -- unlike
///   `woa.rs`'s per-whale-then-per-dimension trap, SCA's reference draws
///   ALL of `r2`/`r3`/`r4` fresh for every single dimension of every single
///   agent; there is no coarser-grained draw to miss.
///   - if `r4 < 0.5`: `X'[d] = X[d] + r1·sin(r2)·|r3·X_best[d] − X[d]|`
///     (Eq. 3.1).
///   - else: `X'[d] = X[d] + r1·cos(r2)·|r3·X_best[d] − X[d]|` (Eq. 3.2).
/// - Replacement is **unconditional** in the reference: `SCA.m` overwrites
///   every agent's position every iteration with no per-agent
///   fitness-improvement test (no greedy comparison). This is `replace/
///   generational` (GWO's kind, reused as-is).
///
/// DELTA FROM THE SECONDARY (fallback) SOURCE: mealpy's `OriginalSCA`
/// (`mealpy/math_based/SCA.py`) matches the per-agent/per-dimension draw
/// order and the sine/cosine update above, but additionally applies a
/// greedy same-index replacement (`get_better_agent` / `greedy_selection_
/// population`) that `SCA.m` does not have. Since `SCA.m` is the paper's own
/// linked reference implementation (the provenance protocol's first
/// choice), `SCA.m` governs here: replacement is unconditional generational,
/// not greedy. This is a real delta found during Step 1, not a sketch
/// discrepancy -- the plan's sketch did not specify a replacer contradicting
/// this, so no correction to the sketch's "Replacer: generational" line was
/// needed, but the mealpy divergence is worth recording for reviewers.
///
/// The per-dimension math is factored into [`sca_dim_step`] so it can be
/// unit-tested directly (e.g. the `r1 = 0` limit, where the update
/// collapses exactly to `X'[d] = X[d]` regardless of `r2`/`r3`/`r4`/branch)
/// without needing to fake `Ctx`/`Evaluator` progress. [`ScaGenerator::generate`]
/// is a thin loop over `(i, d)` that draws the 3 values in the pinned order
/// and calls the helper.
///
/// `min_pop = 2`: SCA needs a best-so-far individual plus at least one other
/// agent for the update to be meaningful (with `pop_size == 1` the "best"
/// individual is always the agent itself, degenerating the algorithm
/// entirely) -- same rationale as `woa.rs`'s `min_pop = 2`.
///
/// Boundary handling and replacement are NOT part of this generator: the
/// `presets::sca` preset reuses `boundary/clamp` (same as `gwo`/`woa` --
/// `SCA.m`'s own bound-flag logic is functionally a clamp to `[lb, ub]`) and
/// `replace/generational` (GWO's kind, reused as-is) for SCA's unconditional
/// generational replacement.
pub fn sca_dim_step(x_d: f64, x_best_d: f64, r1: f64, r2: f64, r3: f64, r4: f64) -> f64 {
    let delta = (r3 * x_best_d - x_d).abs();
    if r4 < 0.5 {
        x_d + r1 * r2.sin() * delta
    } else {
        x_d + r1 * r2.cos() * delta
    }
}

pub struct ScaGenerator;

impl ScaGenerator {
    pub fn from_params(_p: &serde_json::Value) -> Result<Self, ComponentError> {
        // No tunable parameters (a is fixed at 2; r1 is fully determined by
        // ctx.eval progress).
        Ok(Self)
    }

    fn floats(g: &Genotype) -> &Vec<f64> {
        match &g.blocks[0] { BlockValues::Float(x) => x, _ => unreachable!() }
    }
}

impl Generator for ScaGenerator {
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        assert!(pop.len() >= 2, "gen/sca requires a population of at least 2 (pop_size={})", pop.len());
        let n = pop.len();
        let dim = Self::floats(&pop.individuals[0]).len();

        let used = ctx.eval.used() as f64;
        let budget = ctx.eval.budget() as f64;
        let progress = if budget > 0.0 { (used / budget).clamp(0.0, 1.0) } else { 0.0 };
        let a = 2.0;
        let r1 = a * (1.0 - progress); // = a - t*(a/T) = 2 - 2*progress

        // Best-so-far: current-population fitness argmin, ties -> lower
        // index -- pinned (see module doc's "Destination_position" note).
        let best = pop.best_index().unwrap_or(0);
        let x_best = Self::floats(&pop.individuals[best]);

        (0..n).map(|i| {
            let x = Self::floats(&pop.individuals[i]);
            let xs: Vec<f64> = (0..dim).map(|d| {
                // Pinned draw order: r2, then r3, then r4, fresh for every
                // (i, d) -- see the module doc's SCA.m trace.
                let r2 = 2.0 * std::f64::consts::PI * ctx.rng.next_f64();
                let r3 = 2.0 * ctx.rng.next_f64();
                let r4 = ctx.rng.next_f64();
                sca_dim_step(x[d], x_best[d], r1, r2, r3, r4)
            }).collect();
            Genotype { blocks: vec![BlockValues::Float(xs)] }
        }).collect()
    }

    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/sca", SupportedBlocks::Only(vec!["float"])).with_min_pop(2)
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_generator("gen/sca", |p| Ok(Box::new(ScaGenerator::from_params(p)?)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::problem::{Problem, SphereShifted, Evaluator};
    use sezgi_core::rng::RngStream;
    use sezgi_core::state::Blackboard;

    fn g(xs: Vec<f64>) -> Genotype { Genotype { blocks: vec![BlockValues::Float(xs)] } }

    fn pop_nd(n: usize, dim: usize) -> Population {
        // Distinct fitness so best-index selection is unambiguous.
        Population {
            individuals: (0..n).map(|i| g((0..dim).map(|d| (i * dim + d) as f64 * 0.37 - 1.0).collect())).collect(),
            fitness: (0..n).map(|i| (n - i) as f64).collect(), // higher index = better (lower fitness)
        }
    }

    #[test]
    fn r1_zero_limit_gives_x_unchanged_exact() {
        // r1 = 0 => both the sin and cos branches multiply their delta term
        // by 0, so X'[d] = X[d] exactly, regardless of r2/r3/r4/branch.
        let x_d = 3.5;
        let cases = [
            (0.11, 0.92, 0.03), // r4 < 0.5 -> sin branch
            (0.71, 0.58, 0.74), // r4 >= 0.5 -> cos branch
        ];
        for (r2, r3, r4) in cases {
            let result = sca_dim_step(x_d, 100.0, 0.0, r2, r3, r4);
            assert_eq!(result, x_d, "r1=0 must leave X[d] exactly unchanged (r4={r4})");
        }
    }

    #[test]
    fn sin_and_cos_branches_match_the_pinned_formula() {
        // Hand-computed check of both branches against Eq. 3.1/3.2 directly,
        // independent of the r1=0 degenerate case above.
        let x_d: f64 = 2.0;
        let x_best_d: f64 = 5.0;
        let r1: f64 = 1.5;
        let r2: f64 = 0.7;
        let r3: f64 = 1.2;
        let delta = (r3 * x_best_d - x_d).abs(); // |1.2*5 - 2| = 4.0
        let expect_sin = x_d + r1 * r2.sin() * delta;
        let expect_cos = x_d + r1 * r2.cos() * delta;
        assert_eq!(sca_dim_step(x_d, x_best_d, r1, r2, r3, 0.49), expect_sin);
        assert_eq!(sca_dim_step(x_d, x_best_d, r1, r2, r3, 0.5), expect_cos);
    }

    #[test]
    fn determinism_same_seed_bit_identical() {
        let n = 6; let dim = 4;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_nd(n, dim);

        let run = || {
            let mut evaluator = Evaluator::new(&p, 1000);
            let mut rng = RngStream::from_master(42, &[]);
            let mut bb = Blackboard::new();
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
            ScaGenerator.generate(&pop, &mut ctx)
        };

        let a = run();
        let b = run();
        for (ga, gb) in a.iter().zip(b.iter()) {
            let (xa, xb) = (ScaGenerator::floats(ga), ScaGenerator::floats(gb));
            assert_eq!(xa, xb, "same seed must produce bit-identical offspring");
        }
    }

    #[test]
    fn draw_count_is_exactly_3_times_n_times_dim() {
        let n = 5; let dim = 3;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_nd(n, dim);

        let mut rng = RngStream::from_master(7, &[]);
        let rng_before = rng.clone();
        let mut evaluator = Evaluator::new(&p, 1000);
        let mut bb = Blackboard::new();
        {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
            let _ = ScaGenerator.generate(&pop, &mut ctx);
        }

        // Twin stream: advance a clone of the pre-generate rng by exactly
        // 3*n*dim draws by hand (r2, r3, r4 per (i,d), the pinned SCA.m
        // structure -- no per-agent or per-generation draws, no branch that
        // changes draw count), then compare the NEXT draw from each stream --
        // if generate() consumed a different number of draws, the two
        // streams would be at different internal states and the next draws
        // would (with overwhelming probability) differ.
        let mut twin = rng_before;
        for _ in 0..(3 * n * dim) { twin.next_f64(); }
        assert_eq!(rng.next_f64(), twin.next_f64(),
            "gen/sca must consume exactly 3*n*dim RNG draws");
    }

    #[test]
    fn min_pop_below_2_panics() {
        let n = 1; let dim = 2;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_nd(n, dim);
        let mut evaluator = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ScaGenerator.generate(&pop, &mut ctx)
        }));
        assert!(result.is_err(), "gen/sca must reject pop_size < 2 at runtime as a backstop");
    }

    #[test]
    fn best_index_is_current_population_argmin_ties_lower_index() {
        let dim = 1;
        // Two ties at the best value (0.0): indices 0 and 2. With r1=0
        // (progress=1, budget fully consumed), every offspring equals its
        // own X[d] exactly regardless of X_best -- so this test instead
        // checks best selection directly via Population::best_index, the
        // same helper gwo.rs/woa.rs pin their leader/best selection to.
        let pop = Population {
            individuals: (0..5).map(|i| g(vec![i as f64])).collect(),
            fitness: vec![0.0, 5.0, 0.0, 3.0, 1.0],
        };
        assert_eq!(pop.best_index(), Some(0), "ties must resolve to the lower index");
        let _ = dim;
    }
}
