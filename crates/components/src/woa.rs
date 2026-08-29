use sezgi_core::component::*;
use sezgi_core::problem::Population;
use sezgi_core::space::{BlockValues, Genotype};

/// Whale Optimization Algorithm (Mirjalili & Lewis 2016, "The Whale
/// Optimization Algorithm", *Advances in Engineering Software*) — a
/// **labeled metaphor preset**: faithful to the primary source's reference
/// MATLAB implementation, with a pinned deterministic draw order and
/// property tests, but NOT validated against the paper's (or any other
/// publication's) reported benchmark numbers. See the equivalence critique
/// that names WOA ("whale") explicitly among six metaphor-based algorithms
/// shown to be, component for component, relabeled special cases of older
/// operators — cited here conservatively, as background on why this is
/// "labeled metaphor" rather than a mechanism sezgi treats as novel:
/// Camacho-Villalón, Dorigo & Stützle (*International Transactions in
/// Operational Research*, six-algorithm critique: grey wolf, moth-flame,
/// whale, firefly, bat, antlion).
///
/// **Pinned update rule** (part of the RNG-stream contract), following the
/// paper's reference MATLAB (which recomputes `A` and the branch choice
/// per-dimension, not once per individual):
/// - `a = 2 − 2·progress`, `a2 = −1 − progress` (spiral parameter range),
///   where `progress = ctx.eval.used() / ctx.eval.budget()` (clamped to
///   `[0, 1]`; `0` if `budget == 0`), computed **once per `generate` call**,
///   before any RNG draws.
/// - Best-so-far individual `X_best`: fitness argmin over the population,
///   ties → lower index (same convention as `gwo.rs`'s leader selection,
///   via [`Population::best_index`]).
/// - For each whale `i` (population order): draw `p = rng.next_f64()`
///   (1 draw). Then, for each dimension `d` (index order):
///   - if `p < 0.5` (**encircle/search branch**): draw `r1 = rng.next_f64()`,
///     then `r2 = rng.next_f64()` (2 draws); `A = 2a·r1 − a`, `C = 2·r2`.
///     - if `|A| < 1` (**encircling prey**): target = `X_best`.
///     - else (**search for prey**): draw a random whale index
///       `j = rng.next_below(pop_len)` (1 draw — the project's uniform-index
///       idiom, see `de.rs`'s `pick_distinct`, minus the exclusion: `j` MAY
///       equal `i`, matching the reference MATLAB which does not exclude
///       self); target = `X_j`.
///
///     `X'[d] = X_target[d] − A·|C·X_target[d] − X_i[d]|`.
///   - else (`p ≥ 0.5`, **spiral bubble-net branch**): draw
///     `l_raw = rng.next_f64()` (1 draw); `l = (a2 − 1)·l_raw + 1`;
///     `D = |X_best[d] − X_i[d]|`;
///     `X'[d] = D·e^{b·l}·cos(2π·l) + X_best[d]`, `b = 1` (fixed, per the
///     paper's convention).
/// - Draw order: `p` first per whale; then, per dimension, the
///   branch-dependent draws in the order listed above. This matches the
///   reference MATLAB, where the encircle/search choice (and hence `A`) is
///   recomputed per dimension, not once per individual — pinned here as the
///   RNG-stream contract.
///
/// The per-dimension math is factored into [`woa_encircle_step`] and
/// [`woa_spiral_step`] so both can be unit-tested directly (the `A = 0` and
/// `l = 0` limits) without needing to fake `Ctx`/`Evaluator` progress.
/// [`WoaGenerator::generate`] is a thin loop over `(i, d)` that draws the
/// pinned values in order and calls the appropriate helper.
///
/// `min_pop = 2`: WOA needs a best-so-far individual plus at least one other
/// whale for the search-for-prey (random-whale) branch to be meaningful. The
/// random index `j` MAY equal `i` itself (this matches the reference
/// MATLAB, which does not exclude self-selection, so it is not an error
/// condition) — but with `pop_size == 1` there is no other whale `X_best`
/// could ever differ from, degenerating the algorithm entirely. `pop_size
/// == 2` is the smallest population where encircling and search can still
/// target genuinely different individuals.
///
/// Boundary handling and replacement are NOT part of this generator: the
/// `presets::woa` preset reuses `boundary/clamp` (same as `presets::gwo`)
/// and the `replace/generational` kind (see `replace.rs`, added for GWO)
/// for WOA's unconditional generational replacement.
pub fn woa_encircle_step(target_d: f64, x_d: f64, big_a: f64, big_c: f64) -> f64 {
    target_d - big_a * (big_c * target_d - x_d).abs()
}

pub fn woa_spiral_step(x_best_d: f64, x_d: f64, l: f64, b: f64) -> f64 {
    let d = (x_best_d - x_d).abs();
    d * (b * l).exp() * (2.0 * std::f64::consts::PI * l).cos() + x_best_d
}

pub struct WoaGenerator;

impl WoaGenerator {
    pub fn from_params(_p: &serde_json::Value) -> Result<Self, ComponentError> {
        // No tunable parameters (a/a2 are fully determined by ctx.eval progress).
        Ok(Self)
    }

    fn floats(g: &Genotype) -> &Vec<f64> {
        match &g.blocks[0] { BlockValues::Float(x) => x, _ => unreachable!() }
    }
}

impl Generator for WoaGenerator {
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        assert!(pop.len() >= 2, "gen/woa requires a population of at least 2 (pop_size={})", pop.len());
        let n = pop.len();
        let dim = Self::floats(&pop.individuals[0]).len();

        let used = ctx.eval.used() as f64;
        let budget = ctx.eval.budget() as f64;
        let progress = if budget > 0.0 { (used / budget).clamp(0.0, 1.0) } else { 0.0 };
        let a = 2.0 - 2.0 * progress;
        let a2 = -1.0 - progress;

        // Best-so-far: fitness argmin, ties -> lower index -- pinned.
        let best = pop.best_index().unwrap_or(0);
        let x_best = Self::floats(&pop.individuals[best]);

        (0..n).map(|i| {
            let x = Self::floats(&pop.individuals[i]);
            // Pinned draw order: p first, per whale.
            let p = ctx.rng.next_f64();
            let xs: Vec<f64> = (0..dim).map(|d| {
                if p < 0.5 {
                    let r1 = ctx.rng.next_f64();
                    let r2 = ctx.rng.next_f64();
                    let big_a = 2.0 * a * r1 - a;
                    let big_c = 2.0 * r2;
                    let target_d = if big_a.abs() < 1.0 {
                        x_best[d]
                    } else {
                        let j = ctx.rng.next_below(n as u64) as usize;
                        Self::floats(&pop.individuals[j])[d]
                    };
                    woa_encircle_step(target_d, x[d], big_a, big_c)
                } else {
                    let l_raw = ctx.rng.next_f64();
                    let l = (a2 - 1.0) * l_raw + 1.0;
                    woa_spiral_step(x_best[d], x[d], l, 1.0)
                }
            }).collect();
            Genotype { blocks: vec![BlockValues::Float(xs)] }
        }).collect()
    }

    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/woa", SupportedBlocks::Only(vec!["float"])).with_min_pop(2)
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_generator("gen/woa", |p| Ok(Box::new(WoaGenerator::from_params(p)?)));
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
    fn spiral_limit_l_zero_gives_d_plus_best() {
        // l = 0 => e^{b*0} = 1, cos(0) = 1, so result = D + X_best exactly,
        // regardless of b.
        let x_best_d = 5.0;
        let x_d = 2.0; // D = |5 - 2| = 3
        let result = woa_spiral_step(x_best_d, x_d, 0.0, 1.0);
        assert_eq!(result, 3.0 + 5.0, "l=0 must collapse to D + X_best exactly");
    }

    #[test]
    fn encircle_limit_a_zero_gives_target_exactly() {
        // A = 0 => result = target_d - 0 = target_d exactly, regardless of
        // C or x_d. With target = X_best, this is the pinned X' = X_best[d].
        let target_d = 7.0;
        let result = woa_encircle_step(target_d, 100.0, 0.0, 999.0);
        assert_eq!(result, target_d, "A=0 must collapse to the target coordinate exactly");
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
            WoaGenerator.generate(&pop, &mut ctx)
        };

        let a = run();
        let b = run();
        for (ga, gb) in a.iter().zip(b.iter()) {
            let (xa, xb) = (WoaGenerator::floats(ga), WoaGenerator::floats(gb));
            assert_eq!(xa, xb, "same seed must produce bit-identical offspring");
        }
    }

    #[test]
    fn draw_count_two_whale_case_covers_both_branches() {
        // Crafted case: n=2, dim=2, master seed=0, budget=1000/used=0 so
        // progress=0 (a=2, a2=-1). Manually traced against the raw
        // `RngStream` sequence for master seed 0 (independent of the
        // implementation, from the pinned draw-order contract): with a=2,
        //   whale 0 (p0 = 0.3246... < 0.5 -> encircle/search branch):
        //     dim 0: r1,r2 give |A| < 1 -> encircle (2 draws)
        //     dim 1: r1,r2 give |A| >= 1 -> search, +1 draw for j (3 draws)
        //   whale 1 (p1 = 0.8572... >= 0.5 -> spiral branch, both dims):
        //     dim 0: l_raw (1 draw)
        //     dim 1: l_raw (1 draw)
        // Total = 1 (p0) + 2 + 3 + 1 (p1) + 1 + 1 = 9 draws, exercising both
        // top-level branches (encircle/search vs spiral) and both
        // sub-branches of the p<0.5 branch (encircle vs search).
        let n = 2; let dim = 2;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_nd(n, dim);

        let mut rng = RngStream::from_master(0, &[]);
        let rng_before = rng.clone();
        let mut evaluator = Evaluator::new(&p, 1000); // used=0, budget=1000 -> progress=0
        let mut bb = Blackboard::new();
        {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
            let off = WoaGenerator.generate(&pop, &mut ctx);
            assert_eq!(off.len(), n);
        }

        // Twin stream: replay the exact call sequence derived above (branch
        // structure known in advance from the RNG trace, not from reading
        // generate()'s code) on a clone of the pre-generate rng, then
        // compare the NEXT draw from each stream -- if generate() consumed
        // a different number/kind of draws, the two streams would (with
        // overwhelming probability) diverge.
        let mut twin = rng_before;
        let _p0 = twin.next_f64();
        // whale 0, dim 0: encircle
        let _r1 = twin.next_f64(); let _r2 = twin.next_f64();
        // whale 0, dim 1: search (extra random-index draw)
        let _r1 = twin.next_f64(); let _r2 = twin.next_f64(); let _j = twin.next_below(n as u64);
        let _p1 = twin.next_f64();
        // whale 1, dim 0: spiral
        let _l0 = twin.next_f64();
        // whale 1, dim 1: spiral
        let _l1 = twin.next_f64();

        assert_eq!(rng.next_f64(), twin.next_f64(),
            "gen/woa must consume exactly the pinned draw sequence for this crafted two-whale case");
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
            WoaGenerator.generate(&pop, &mut ctx)
        }));
        assert!(result.is_err(), "gen/woa must reject pop_size < 2 at runtime as a backstop");
    }
}
