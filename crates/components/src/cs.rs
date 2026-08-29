use crate::init::sample_uniform;
use sezgi_core::component::*;
use sezgi_core::dist::Distribution;
use sezgi_core::problem::Population;
use sezgi_core::space::{BlockValues, Genotype};

/// Cuckoo Search (Yang, X.-S. & Deb, S. 2009, "Cuckoo Search via Lévy
/// Flights", *2009 World Congress on Nature & Biologically Inspired
/// Computing (NaBIC)*, pp. 210-214, IEEE) — a **labeled metaphor preset**:
/// faithful to the primary source's update rule, with a pinned deterministic
/// draw order and property tests, but NOT validated against the paper's (or
/// any other publication's) reported benchmark numbers. Unlike `gwo.rs`/
/// `woa.rs`, no equivalence-critique citation is included here — not
/// mandated for CS by the task brief — so this module cites only the
/// primary source.
///
/// **Pinned update rule** (part of the RNG-stream contract):
/// - Best-so-far individual `X_best`: fitness argmin over the population,
///   ties → lower index (same convention as `gwo.rs`'s leader selection /
///   `woa.rs`'s `X_best`, via [`Population::best_index`]), computed **once
///   per `generate` call, before any RNG draws**.
/// - For each nest `i` (population order), each dimension `d` (index
///   order): draw exactly ONE Lévy step via
///   `Distribution::Levy { alpha: 1.5 }.sample(ctx.rng)` (see below for the
///   sampler's exact scheme and its per-sample raw-draw-count contract);
///   `X'[d] = X_i[d] + alpha_step · levy · (X_i[d] − X_best[d])`, with
///   `alpha_step = 0.01` (fixed, not tunable via `from_params`, same idiom
///   as `gwo.rs`/`woa.rs`'s fixed `a`/`a2`).
///
/// **The project's Lévy sampler** (`Distribution::Levy { alpha }` in
/// `sezgi_core::dist`, private fn `levy_mantegna`) implements the Mantegna
/// (1994) algorithm: `levy = sigma_u * gauss_polar(rng) / gauss_polar(rng).abs().powf(1/alpha)`,
/// where `sigma_u` is a deterministic, no-draw closed-form constant derived
/// from `alpha` via the (also no-draw, Lanczos-approximated) Gamma function,
/// and `gauss_polar` is the project's polar Box–Muller (Marsaglia) Gaussian
/// sampler. ONE Lévy sample therefore consumes EXACTLY TWO `gauss_polar`
/// calls — but `gauss_polar` is itself a rejection-sampling loop (draw `u, v`
/// uniform in `[-1,1)` via 2 raw `next_f64()` draws, accept iff
/// `0 < u^2+v^2 < 1`, else redraw) whose iteration count is data-dependent
/// (acceptance probability = π/4 ≈ 0.785, so ≥1 iteration almost always,
/// unboundedly more with vanishing probability). The number of RAW uniform
/// draws behind one `Distribution::Levy { alpha: 1.5 }.sample()` call is
/// therefore data-dependent (minimum 4 raw draws, no fixed upper bound) —
/// unlike GWO's fixed 6 draws/`(i,d)` or HS's `{2,3,4}`-draw branches. The
/// pin here is expressed at the `Distribution::sample()` call boundary:
/// exactly ONE `Distribution::Levy { alpha: 1.5 }.sample()` call per
/// `(i, d)`, in that order — NOT a fixed raw-draw count. The draw-count test
/// below (`levy_draw_count_via_raw_replay`) verifies this boundary by
/// twin-stream REPLAYING actual `Distribution::Levy{..}.sample()` calls (not
/// hand-counted raw draws), per the task brief's guidance for a variable
/// per-sample draw count.
///
/// The per-dimension math is factored into [`cs_dim_step`] so it can be
/// unit-tested directly (e.g. the `X_i == X_best` zero-step limit, where the
/// `(X_i[d] − X_best[d])` factor is exactly `0.0` and the Lévy draw's value
/// becomes irrelevant) without needing to fake `Ctx`/`RngStream`.
/// [`CsGenerator::generate`] is a thin loop over `(i, d)` that draws the
/// pinned Lévy sample and calls the helper.
///
/// `min_pop = 2`: CS needs a best-so-far individual distinct from `i` for
/// the Lévy step to move `X_i` at all (same rationale as `woa.rs`'s
/// `min_pop = 2`).
///
/// Boundary handling and (same-index greedy) replacement are NOT part of
/// this generator: `presets::cuckoo_search` reuses `boundary/clamp` (same as
/// `gwo`/`woa`/`harmony_search` — a Lévy step can push a coordinate outside
/// `[lo, hi]`) and DE's existing `replace/one-to-one-greedy` kind (see
/// `replace.rs`) for the greedy same-index (offspring `i` vs parent `i`)
/// replacement —
// sezgi simplification: the paper's Algorithm 1 compares the new cuckoo
// egg against a RANDOMLY chosen nest j (not necessarily i), replacing j if
// the new egg is better. sezgi reuses the existing same-index greedy
// replacer instead (offspring i vs parent i), because (a) it is already a
// tested, reusable in-repo kind with no CS-specific coupling, and (b) a
// "pick a random OTHER nest to compare against" replacer has no existing
// analog and would need its own new RNG-stream contract for a mechanism
// that, unlike the Lévy-flight generation itself, isn't the part of CS this
// preset aims to demonstrate faithfully. This is a genuine, deliberate
// divergence from the paper's Algorithm 1 — called out here and in
// `presets::cuckoo_search`'s doc comment.
/// plus a NEW `adapter/abandon-worst-fraction` component for the
/// abandonment step (worst `pa = 0.25` fraction re-randomized after
/// replacement, see [`AbandonWorstFraction`]'s doc for the full
/// component-chain-placement and RNG-stream rationale).
pub fn cs_dim_step(x_i_d: f64, x_best_d: f64, levy: f64, alpha_step: f64) -> f64 {
    x_i_d + alpha_step * levy * (x_i_d - x_best_d)
}

const ALPHA_STEP: f64 = 0.01;
const LEVY_ALPHA: f64 = 1.5;

pub struct CsGenerator;

impl CsGenerator {
    pub fn from_params(_p: &serde_json::Value) -> Result<Self, ComponentError> {
        // No tunable parameters (alpha_step/levy alpha are fixed, per the pinned spec).
        Ok(Self)
    }

    fn floats(g: &Genotype) -> &Vec<f64> {
        match &g.blocks[0] { BlockValues::Float(x) => x, _ => unreachable!() }
    }
}

impl Generator for CsGenerator {
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        assert!(pop.len() >= 2, "gen/cuckoo_levy requires a population of at least 2 (pop_size={})", pop.len());
        let n = pop.len();
        let dim = Self::floats(&pop.individuals[0]).len();

        // Best-so-far: fitness argmin, ties -> lower index -- pinned,
        // computed once per generate() call, before any RNG draws (same
        // convention as gwo.rs's leaders / woa.rs's X_best).
        let best = pop.best_index().unwrap_or(0);
        let x_best = Self::floats(&pop.individuals[best]);

        (0..n).map(|i| {
            let x = Self::floats(&pop.individuals[i]);
            let xs: Vec<f64> = (0..dim).map(|d| {
                // Pinned: exactly one Distribution::Levy{alpha:1.5}.sample() per (i, d).
                let levy = Distribution::Levy { alpha: LEVY_ALPHA }.sample(ctx.rng);
                cs_dim_step(x[d], x_best[d], levy, ALPHA_STEP)
            }).collect();
            Genotype { blocks: vec![BlockValues::Float(xs)] }
        }).collect()
    }

    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/cuckoo_levy", SupportedBlocks::Only(vec!["float"])).with_min_pop(2)
    }
}

const PA: f64 = 0.25;

/// Indices of the worst `floor(pa * n)` nests, worst-first (fitness
/// descending; ties → higher index abandoned first — scanning further among
/// equal-fitness nests keeps replacing the running "worst" pick with the
/// higher index, mirroring the tie-break idiom used elsewhere in this crate
/// for the opposite extremum, e.g. `Population::best_index`/`gwo.rs`'s
/// leader selection use "ties → lower index"). Pure helper, factored out
/// for direct unit testing without `Ctx`/`Evaluator`/`RngStream`.
pub fn abandon_order(fitness: &[f64], pa: f64) -> Vec<usize> {
    let n = fitness.len();
    let k = (pa * n as f64).floor() as usize;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| fitness[b].total_cmp(&fitness[a]).then(b.cmp(&a)));
    order.truncate(k);
    order
}

/// `pa = 0.25` fixed (not tunable via `from_params`, same idiom as `hs.rs`'s
/// fixed `hmcr`/`par`). Re-randomizes the worst `floor(pa*n)` nests (per
/// [`abandon_order`], worst-first) uniformly in bounds via `sample_uniform`
/// (see `init.rs` — the same uniform-in-bounds routine `gen/uniform-resample`
/// uses in `resample.rs`), then evaluates the new nests via
/// `ctx.eval.evaluate(..)` so `pop.fitness` stays consistent for the NEXT
/// iteration's `best_index`/greedy comparisons. If the evaluation budget is
/// exhausted mid-call, the already-drawn new nests are discarded and `pop`
/// is left entirely untouched — the engine's own next `eval.evaluate` call
/// (the following iteration's generator stage) will then cleanly hit the
/// same budget wall and exit; this mirrors `restart.rs`'s "budget
/// exhaustion is clean" contract for its own re-init evaluate call.
///
/// **Component-chain placement** (part of the RNG-stream contract): this is
/// an [`Adapter`], not a [`Restart`] or a second stage.
/// `Engine::run` (`core/src/engine.rs`) runs, per stage, once per iteration:
/// generate → boundary repair → evaluate → replace → **adapter (if any)** —
/// so an adapter is the only pinned hook that runs strictly AFTER
/// replacement within the same stage/iteration, exactly matching "after
/// replacement" per the pinned spec. `Restart` only checks once per
/// iteration AFTER all stages, and unconditionally re-initializes the WHOLE
/// population on a stagnation trigger — the wrong shape entirely (partial,
/// every-iteration, fitness-ordered abandonment vs whole-population,
/// conditional, stagnation-triggered reset). No existing `resample.rs`/
/// `restart.rs` kind expresses partial worst-fraction abandonment, so this
/// is registered as a NEW kind: `adapter/abandon-worst-fraction` (dash-case,
/// matching the project's `adapter/jde-commit`/`adapter/shade-history`/
/// `adapter/lpsr` naming convention). It lives in `cs.rs` rather than
/// `resample.rs`, mirroring how `shade.rs` keeps its own algorithm-specific
/// adapters (`ShadeHistoryAdapter`, `LpsrAdapter`) colocated with the
/// generator/replacer they pair with, rather than in a shared file.
///
/// **RNG stream** (part of the RNG-stream contract): `Ctx::rng` inside
/// `Adapter::adapt` is already bound, by `Engine::run`, to that stage's
/// dedicated adapter stream — `adapter_rngs[stage_index]`, path
/// `[run_id, 1_000_000 + stage_index]` (the M1 adapter-stream convention;
/// see `engine.rs`'s `adapter_rngs` construction and the doc comment on
/// [`Adapter::adapt`]). `sample_uniform`'s draws for each abandoned nest are
/// therefore automatically isolated from `gen/cuckoo_levy`'s own stream
/// (path `[run_id, 1 + 2*stage_index]`) — no extra plumbing needed. Draws
/// happen in `abandon_order`'s (worst-first) sequence, one full genotype
/// draw per abandoned nest.
///
/// This is the first adapter in this crate to call `ctx.eval.evaluate(..)`:
/// the existing adapters (`adapter/jde-commit`, `adapter/shade-history`,
/// `adapter/lpsr`) only read `ctx.eval.used()`/`budget()` or mutate
/// `ctx.bb`, never introduce brand-new unevaluated individuals into `pop`.
/// Known, documented limitation inherited from `Engine::run`'s existing
/// architecture (not fixed by this task): `Engine::run`'s own `global_best`
/// bookkeeping (used for `RunResult::best_f`/`best_x`) is only updated right
/// after the GENERATOR stage's own `eval.evaluate(&offspring)` call and
/// after a `Restart`'s re-init evaluate — an adapter-issued `evaluate` call
/// is invisible to it. `ctx.eval.best_so_far()` DOES still see it (the
/// `Evaluator` tracks its own best across ALL `evaluate` calls, regardless
/// of caller), but a re-randomized nest that happens to be the single best
/// point of the entire run, and is never subsequently matched or improved
/// upon by a later generator-stage evaluation, would not surface in
/// `RunResult`. In practice this is low-impact: abandoned nests are drawn
/// from the CURRENT worst fraction and replaced with UNIFORM noise, so they
/// only rarely improve on the population's best, let alone the run's best.
pub struct AbandonWorstFraction;

impl AbandonWorstFraction {
    pub fn from_params(_p: &serde_json::Value) -> Result<Self, ComponentError> {
        // No tunable parameters (pa = 0.25 is fixed, per the pinned spec).
        Ok(Self)
    }
}

impl Adapter for AbandonWorstFraction {
    fn adapt(&self, pop: &mut Population, ctx: &mut Ctx) {
        let order = abandon_order(&pop.fitness, PA);
        if order.is_empty() { return; }

        let new_genotypes: Vec<Genotype> =
            order.iter().map(|_| sample_uniform(ctx.space, ctx.rng)).collect();
        if let Ok(new_fitness) = ctx.eval.evaluate(&new_genotypes) {
            for (&idx, (gi, fi)) in order.iter().zip(new_genotypes.into_iter().zip(new_fitness)) {
                pop.individuals[idx] = gi;
                pop.fitness[idx] = fi;
            }
        }
    }

    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("adapter/abandon-worst-fraction", SupportedBlocks::All)
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_generator("gen/cuckoo_levy", |p| Ok(Box::new(CsGenerator::from_params(p)?)));
    reg.register_adapter("adapter/abandon-worst-fraction", |p| Ok(Box::new(AbandonWorstFraction::from_params(p)?)));
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

    // ---- cs_dim_step / CsGenerator ----

    #[test]
    fn zero_step_when_x_i_equals_x_best_exactly() {
        // X_i == X_best (for ANY i, not just i == best_index) makes the
        // (X_i[d] - X_best[d]) factor exactly 0.0 for every d, so offspring
        // i must equal parent i EXACTLY regardless of the Levy draw's value.
        let dim = 3;
        // individuals[0] is the best (fitness 0.0); individuals[2] is a
        // DIFFERENT population slot with the IDENTICAL genotype but worse
        // fitness (5.0) -- so index 2 != best_index (0) yet X_2 == X_best.
        let shared = vec![1.25, -3.5, 0.75];
        let pop = Population {
            individuals: vec![
                g(shared.clone()),
                g(vec![10.0, 10.0, 10.0]),
                g(shared.clone()),
                g(vec![-10.0, -10.0, -10.0]),
            ],
            fitness: vec![0.0, 100.0, 5.0, 200.0],
        };
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 1000);
        let mut rng = RngStream::from_master(3, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let off = CsGenerator.generate(&pop, &mut ctx);
        assert_eq!(CsGenerator::floats(&off[2]), &shared,
            "X_i == X_best must leave offspring i UNCHANGED exactly (zero-step property)");
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
            CsGenerator.generate(&pop, &mut ctx)
        };

        let a = run();
        let b = run();
        assert_eq!(a.len(), n);
        for (ga, gb) in a.iter().zip(b.iter()) {
            let (xa, xb) = (CsGenerator::floats(ga), CsGenerator::floats(gb));
            assert_eq!(xa, xb, "same seed must produce bit-identical offspring");
        }
    }

    #[test]
    fn levy_draw_count_via_raw_replay() {
        // Per dist.rs's `levy_mantegna`, one `Distribution::Levy{alpha:1.5}.sample()`
        // call consumes a DATA-DEPENDENT number of raw uniform draws (two
        // `gauss_polar` calls, each itself a variable-length rejection loop)
        // -- see this file's module doc. So the pin is expressed at the
        // `Distribution::sample()` call boundary (exactly one call per
        // (i, d)), not as a fixed raw-draw count -- this test verifies that
        // boundary by RAW-REPLAYING actual `Distribution::Levy{..}.sample()`
        // calls (not hand-counted `next_f64()` calls), n*dim times, on a
        // twin stream.
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
            let off = CsGenerator.generate(&pop, &mut ctx);
            assert_eq!(off.len(), n);
        }

        let mut twin = rng_before;
        for _ in 0..(n * dim) {
            Distribution::Levy { alpha: 1.5 }.sample(&mut twin);
        }
        assert_eq!(rng.next_f64(), twin.next_f64(),
            "gen/cuckoo_levy must consume exactly one Distribution::Levy {{alpha:1.5}}.sample() call per (i,d), n*dim total");
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
            CsGenerator.generate(&pop, &mut ctx)
        }));
        assert!(result.is_err(), "gen/cuckoo_levy must reject pop_size < 2 at runtime as a backstop");
    }

    // ---- abandon_order / AbandonWorstFraction ----

    #[test]
    fn abandon_order_ties_break_to_higher_index_first() {
        // Two ties at the max (worst) fitness (20.0): indices 1 and 2.
        // Higher index (2) must be abandoned first.
        let fitness = vec![10.0, 20.0, 20.0, 5.0];
        let order = abandon_order(&fitness, 0.5); // floor(0.5*4) = 2
        assert_eq!(order, vec![2, 1], "worst-first, ties -> higher index abandoned first");
    }

    #[test]
    fn abandon_order_pa_quarter_n8_picks_exactly_two_worst() {
        let fitness: Vec<f64> = (0..8).map(|i| i as f64 * 10.0).collect(); // index 7 worst
        let order = abandon_order(&fitness, 0.25);
        assert_eq!(order, vec![7, 6]);
    }

    #[test]
    fn abandon_worst_fraction_pa_quarter_n8_abandons_exactly_two_worst() {
        let n = 8; let dim = 2;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 1000);
        // Distinct fitness per index, strictly increasing -- so the worst
        // two (highest fitness) are unambiguous: index 7 (worst), index 6
        // (second worst).
        let individuals: Vec<Genotype> = (0..n).map(|i| g(vec![i as f64; dim])).collect();
        let fitness: Vec<f64> = (0..n).map(|i| (i * 10) as f64).collect();
        let mut pop = Population { individuals: individuals.clone(), fitness: fitness.clone() };
        let mut rng = RngStream::from_master(9, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        AbandonWorstFraction.adapt(&mut pop, &mut ctx);

        // Exactly floor(0.25*8) = 2 nests changed: indices 7 and 6.
        for i in 0..6 {
            assert_eq!(pop.individuals[i], individuals[i], "nest {i} must be bit-unchanged");
            assert_eq!(pop.fitness[i], fitness[i], "nest {i}'s fitness must be bit-unchanged");
        }
        assert_ne!(pop.individuals[6], individuals[6], "worst-but-one nest (6) must be re-randomized");
        assert_ne!(pop.individuals[7], individuals[7], "worst nest (7) must be re-randomized");
    }

    #[test]
    fn abandon_worst_fraction_pa_below_one_over_n_abandons_nothing() {
        // n=2: floor(0.25*2) = 0 -- no abandonment, pop entirely unchanged.
        let dim = 2;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 1000);
        let individuals: Vec<Genotype> = vec![g(vec![1.0; dim]), g(vec![2.0; dim])];
        let fitness = vec![1.0, 2.0];
        let mut pop = Population { individuals: individuals.clone(), fitness: fitness.clone() };
        let mut rng = RngStream::from_master(9, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        AbandonWorstFraction.adapt(&mut pop, &mut ctx);
        assert_eq!(pop.individuals, individuals);
        assert_eq!(pop.fitness, fitness);
    }
}
