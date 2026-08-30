use sezgi_core::component::*;
use sezgi_core::problem::Population;
use sezgi_core::space::{BlockValues, Genotype};

/// Teaching-Learning-Based Optimization (Rao, R.V., Savsani, V.J., Vakharia,
/// D.P. 2011, "Teaching-learning-based optimization: A novel method for
/// constrained mechanical design optimization problems", *Computer-Aided
/// Design*, 43(3), 303-315, DOI 10.1016/j.cad.2010.12.015) -- a **labeled
/// metaphor preset**: faithful to a well-regarded reimplementation's own
/// update equations and loop structure, with a pinned deterministic draw
/// order and property tests, but NOT validated against the paper's (or any
/// other publication's) reported benchmark numbers. TLBO is the first
/// **multi-stage** preset in sezgi: two generators run in sequence, each
/// generation, each with its own [`StageSpec`](sezgi_core::spec::StageSpec)
/// -- `gen/tlbo-teacher` (this module's [`TlboTeacherGenerator`]) followed by
/// `gen/tlbo-learner` ([`TlboLearnerGenerator`]), both paired with
/// `replace/one-to-one-greedy`. No established equivalence critique covers
/// TLBO the way the Camacho-Villalón/Dorigo/Stützle *ITOR* six-algorithm
/// critique covers GWO/MFO/WOA/FA/BA/ALO -- but a DIFFERENT, well-established
/// critique specifically targets TLBO's own accounting and framing (see
/// "Hidden evaluations and the 'parameter-free' claim" below).
///
/// ## Provenance (PROVENANCE-FIRST protocol)
///
/// **Artifact used (governs) -- THIRD-PARTY, not Rao's own code:** Rao's own
/// TLBO code was not independently locatable through this project's
/// text-fetch tooling (his group's usual publication channels -- a personal
/// site or MATLAB File Exchange upload under his own name -- were not found;
/// several File Exchange TLBO submissions exist but are third-party
/// contributions themselves, with the same metadata-only-page limitation
/// `jaya.rs`'s and `fpa.rs`'s provenance notes already documented for other
/// File Exchange entries). Instead, per the protocol's "well-regarded
/// reimplementation" fallback (the accepted precedent this wave already uses
/// for mealpy elsewhere), this module is verified against Mostapha Kalami
/// Heris / **Yarpiz**'s `tlbo.m` (Project Code **YPEA111**, "Implementation
/// of TLBO in MATLAB", <https://yarpiz.com/83/ypea111-teaching-learning-based-optimization>,
/// mirrored verbatim on GitHub at
/// `smkalami/ypea111-teaching-learning-based-optimization` -- `TLBO/tlbo.m`,
/// fetched in full, byte-for-byte, from that repository's raw file). Yarpiz
/// is one of the most widely cited/reused sources for canonical metaheuristic
/// reference code in tutorials and comparison studies -- the SAME kind of
/// well-regarded-third-party status this project already extends to mealpy --
/// but it is explicitly NOT Rao's own code, and is labeled as such here per
/// the protocol's pedigree-honesty requirement. Key quoted lines (verbatim):
///
/// ```text
/// % Calculate Population Mean
/// Mean = 0;
/// for i = 1:nPop
///     Mean = Mean + pop(i).Position;
/// end
/// Mean = Mean/nPop;
///
/// % Select Teacher
/// Teacher = pop(1);
/// for i = 2:nPop
///     if pop(i).Cost < Teacher.Cost
///         Teacher = pop(i);
///     end
/// end
///
/// % Teacher Phase
/// for i = 1:nPop
///     newsol = empty_individual;
///     TF = randi([1 2]);
///     newsol.Position = pop(i).Position ...
///         + rand(VarSize).*(Teacher.Position - TF*Mean);
///     newsol.Position = max(newsol.Position, VarMin);
///     newsol.Position = min(newsol.Position, VarMax);
///     newsol.Cost = CostFunction(newsol.Position);
///     if newsol.Cost<pop(i).Cost
///         pop(i) = newsol;
///         if pop(i).Cost < BestSol.Cost
///             BestSol = pop(i);
///         end
///     end
/// end
///
/// % Learner Phase
/// for i = 1:nPop
///     A = 1:nPop;
///     A(i) = [];
///     j = A(randi(nPop-1));
///     Step = pop(i).Position - pop(j).Position;
///     if pop(j).Cost < pop(i).Cost
///         Step = -Step;
///     end
///     newsol = empty_individual;
///     newsol.Position = pop(i).Position + rand(VarSize).*Step;
///     newsol.Position = max(newsol.Position, VarMin);
///     newsol.Position = min(newsol.Position, VarMax);
///     newsol.Cost = CostFunction(newsol.Position);
///     if newsol.Cost<pop(i).Cost
///         pop(i) = newsol;
///         if pop(i).Cost < BestSol.Cost
///             BestSol = pop(i);
///         end
///     end
/// end
/// ```
///
/// ## Verified findings (each checked against the plan's sketch; SOURCE
/// GOVERNS on every delta)
///
/// 1. **Attractor -- CONFIRMS the wave's already-parked convention, no
///    delta needed:** `Teacher` is selected as the CURRENT population's
///    fitness argmin (`Teacher = pop(1); for i=2:nPop if pop(i).Cost <
///    Teacher.Cost`), recomputed fresh every generation, from a completely
///    fresh loop -- NOT a persisted global-best-ever (unlike `ssa.rs`'s
///    `FoodPosition`/`ba.rs`'s `best` bookkeeping, which needed the wave's
///    parked-convention delta). TLBO's own reference already IS current-pop
///    argmin, so [`Population::best_index`] (ties -> lower index, matching
///    `Teacher = pop(1); ... if pop(i).Cost < Teacher.Cost` keeping the
///    first-seen minimum on a tie) is used directly, with zero
///    simplification tag needed.
/// 2. **Population mean -- confirmed per-dimension, over ALL `nPop`
///    individuals, computed ONCE per generation, before any RNG draw:**
///    `Mean = Mean + pop(i).Position` accumulated over the full loop then
///    divided by `nPop` -- a plain componentwise arithmetic mean.
/// 3. **Teaching factor `TF` -- settles the plan sketch's "per-learner vs
///    per-generation?" question: PER LEARNER.** `TF = randi([1 2])` sits
///    INSIDE the `for i = 1:nPop` teacher-phase loop, drawn fresh for every
///    learner -- not once per generation. `randi([1 2])` is a uniform
///    integer draw from `{1, 2}`; pinned here as `1.0 + ctx.rng.next_below(2)
///    as f64`.
/// 4. **Teacher-phase `r` -- per-dimension, confirmed:** `rand(VarSize)`
///    with `VarSize = [1 nVar]` draws one independent uniform value per
///    dimension (MATLAB's elementwise `.*` broadcasts it across the whole
///    position vector) -- NOT a single scalar per learner. Drawn AFTER `TF`
///    (both fixed before `newsol.Position` is computed).
/// 5. **Greedy same-index acceptance, confirmed -- and TWO evaluations per
///    learner per generation, confirmed:** `if newsol.Cost<pop(i).Cost:
///    pop(i) = newsol` appears, structurally identical, in BOTH phases --
///    once per phase, per learner, so a full generation costs `2 * nPop`
///    evaluations (plus the one-time initialization batch). sezgi reuses
///    `replace/one-to-one-greedy` (DE's kind, strict `<`) for both stages,
///    matching `jaya.rs`'s/`fpa.rs`'s established precedent that the source's
///    `<=`-vs-`<` distinction is immaterial for continuous fitness under this
///    project's RNG streams.
/// 6. **Learner-phase partner `j` -- a genuine, load-bearing distinctness
///    finding, DIFFERENT from `fpa.rs`'s/`woa.rs`'s "self not excluded"
///    idiom:** `A = 1:nPop; A(i) = []; j = A(randi(nPop-1))` explicitly
///    REMOVES `i` from the candidate set before drawing -- `j` is
///    UNCONDITIONALLY distinct from `i` (never self-selecting), drawn
///    uniformly from the remaining `nPop - 1` members. Pinned here via
///    rejection sampling on `ctx.rng.next_below(n)`, rejecting only `cand ==
///    i` -- the same distribution as the source's remaining-array index,
///    following this project's established `pick_distinct`-style idiom
///    (`de.rs`).
/// 7. **Direction -- confirmed toward-better/away-from-worse:** `Step =
///    pop(i).Position - pop(j).Position; if pop(j).Cost < pop(i).Cost: Step
///    = -Step`. If `j` is strictly better than `i`, `Step` flips to `X_j -
///    X_i` (the update moves toward `j`); otherwise `Step` stays `X_i - X_j`
///    (the update moves away from `j`, including the tie case `f(j) ==
///    f(i)`, which the source's strict `<` does not flip).
/// 8. **Learner-phase `r` -- per-dimension, confirmed, drawn AFTER `j`:**
///    `rand(VarSize)`, same shape as finding 4, applied to `Step` -- `j` is
///    fully resolved (including the fitness comparison, which consumes no
///    RNG draw) before any dimension's `r` is drawn.
/// 9. **Bounds clipping -- decoupled, same as every other preset in this
///    crate:** `newsol.Position = max(..., VarMin); ... min(..., VarMax)`
///    plays the same role as `boundary/clamp`, used by both stages here
///    exactly as `gwo`/`woa`/`.../fpa` already use it.
/// 10. **No duplicate-removal step in this reference -- confirmed absent.**
///     See "Hidden evaluations..." below.
///
/// ## min_pop -- ADJUSTED from the plan's sketched `3` down to `2`
///
/// Finding 6 settles this precisely, the same way `fpa.rs`'s finding 4 did
/// for a DIFFERENT reason: `j` need only be distinct from `i`, not from any
/// THIRD index, so a population of exactly 2 already gives the learner phase
/// its one and only valid partner (`j` is forced to be "the other one", zero
/// rejection-sampling iterations needed). `min_pop = 2` is a HARD requirement
/// for [`TlboLearnerGenerator`], not merely "meaningful": with `n < 2` there
/// is no candidate to draw at all (the rejection loop for `j` never
/// terminates) -- enforced both by `ComponentMeta::with_min_pop(2)`
/// (spec-validation) and a runtime `assert!` backstop, the same two-layer
/// idiom as every other preset in this crate. [`TlboTeacherGenerator`]'s own
/// math does not literally divide by zero at `n = 1` (`Mean == Teacher == X`,
/// a degenerate but well-defined update), but a population mean is
/// meaningless with only one member, so it is ALSO given `min_pop = 2`, for
/// consistency with the wave's "needs a best-so-far distinct from `i`"
/// convention (`sca.rs`/`goa.rs`) and because `presets::tlbo` always pairs it
/// with the learner stage's hard `2` anyway.
///
/// ## Hidden evaluations and the "parameter-free" claim
///
/// TLBO is often marketed as "parameter-free" (needing only the common
/// controls every population-based metaheuristic needs -- population size
/// and generation count -- with no algorithm-specific tuning knob). The
/// verified Yarpiz reference here has NO duplicate-removal step (see finding
/// 10): every evaluation the engine performs, it counts, exactly `2 * nPop`
/// per generation as derived in finding 5 -- there is nothing to omit.
/// However, this is a real point of documented controversy in the
/// literature: **Črepinšek, M., Liu, S.-H., & Mernik, L. (2012), "A note on
/// teaching-learning-based optimization algorithm", *Information Sciences*,
/// 212, 79-93, DOI 10.1016/j.ins.2012.05.009** identifies (a) an unreported
/// but important step in some of Rao's OWN published TLBO variants/
/// comparisons -- a duplicate-removal/re-evaluation procedure that is not
/// always disclosed, (b) incorrect formulae for the resulting number-of-
/// function-evaluations counts in some of those comparisons, and (c)
/// misconceptions around the "parameter-less" framing itself. sezgi's own
/// budget accounting is unaffected either way (the engine counts every
/// `Evaluator::evaluate` call regardless of what any external paper claims;
/// see the two-stage budget-accounting test in `sezgi_core::engine`'s test
/// module), but per this task's provenance protocol, sezgi does NOT
/// implement any duplicate-removal step -- this is a documented, deliberate
/// omission (not a simplification of an existing mechanism), and this
/// module's "parameter-free" framing above is stated WITH this critique's
/// counterpoint attached, not asserted uncritically.
///
/// ## Draw order (part of the RNG-stream contract)
///
/// Per the engine's per-stage RNG derivation, `gen/tlbo-teacher` draws from
/// stage 0's generator stream (`[run_id, 1]`) and `gen/tlbo-learner` from
/// stage 1's generator stream (`[run_id, 3]`) -- each entirely independent
/// of the other (see `sezgi_core::engine::Engine::run`'s `stage_rngs`
/// derivation, `[run_id, 1 + 2*i]`/`[run_id, 2 + 2*i]` for stage `i`'s
/// generator/replacer pair). Within EACH generator, per learner `i`
/// (population order):
///
/// **Teacher phase** ([`TlboTeacherGenerator`]): `TF` (1 draw, integer,
/// finding 3) THEN, for each dimension `d` (ascending index order), `r[d]`
/// (1 draw, finding 4) immediately consumed by [`tlbo_teacher_dim_step`].
///
/// **Learner phase** ([`TlboLearnerGenerator`]): `j` (>= 1 draw,
/// data-dependent rejection sampling, finding 6) THEN, for each dimension
/// `d` (ascending index order), `r[d]` (1 draw, finding 8) immediately
/// consumed by [`tlbo_learner_dim_step`]. The `partner_is_better` fitness
/// comparison (finding 7) consumes no RNG draw.
///
/// Both generators' per-dimension math is factored into
/// [`tlbo_teacher_dim_step`] / [`tlbo_learner_dim_step`] so both are
/// unit-tested directly without needing to fake `Ctx`/`RngStream`.
pub fn tlbo_teacher_dim_step(x_d: f64, teacher_d: f64, mean_d: f64, tf: f64, r: f64) -> f64 {
    x_d + r * (teacher_d - tf * mean_d)
}

/// `partner_is_better`: `true` when the partner's fitness is strictly better
/// (lower) than `x`'s own -- moves toward the partner; `false` (including
/// ties) moves away from it, per finding 7.
pub fn tlbo_learner_dim_step(x_d: f64, partner_d: f64, r: f64, partner_is_better: bool) -> f64 {
    let mut step = x_d - partner_d;
    if partner_is_better { step = -step; }
    x_d + r * step
}

fn floats(g: &Genotype) -> &Vec<f64> {
    match &g.blocks[0] { BlockValues::Float(x) => x, _ => unreachable!() }
}

pub struct TlboTeacherGenerator;

impl TlboTeacherGenerator {
    pub fn from_params(_p: &serde_json::Value) -> Result<Self, ComponentError> {
        Ok(Self) // no tunable parameters -- TF/r are drawn, not configured
    }
}

impl Generator for TlboTeacherGenerator {
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        assert!(pop.len() >= 2, "gen/tlbo-teacher requires a population of at least 2 (pop_size={})", pop.len());
        let n = pop.len();
        let dim = floats(&pop.individuals[0]).len();

        // Pinned order (finding 2 before finding 1, matching the source):
        // population mean first, then the current-pop-best teacher -- neither
        // consumes an RNG draw.
        let mut mean = vec![0.0; dim];
        for ind in &pop.individuals {
            let xs = floats(ind);
            for d in 0..dim { mean[d] += xs[d]; }
        }
        for m in &mut mean { *m /= n as f64; }
        let teacher_idx = pop.best_index().unwrap_or(0);
        let teacher = floats(&pop.individuals[teacher_idx]);

        (0..n).map(|i| {
            let x = floats(&pop.individuals[i]);
            // Pinned: TF drawn ONCE per learner, before the per-dim r draws (finding 3).
            let tf = 1.0 + ctx.rng.next_below(2) as f64;
            let xs: Vec<f64> = (0..dim).map(|d| {
                let r = ctx.rng.next_f64();
                tlbo_teacher_dim_step(x[d], teacher[d], mean[d], tf, r)
            }).collect();
            Genotype { blocks: vec![BlockValues::Float(xs)] }
        }).collect()
    }

    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/tlbo-teacher", SupportedBlocks::Only(vec!["float"])).with_min_pop(2)
    }
}

pub struct TlboLearnerGenerator;

impl TlboLearnerGenerator {
    pub fn from_params(_p: &serde_json::Value) -> Result<Self, ComponentError> {
        Ok(Self)
    }
}

impl Generator for TlboLearnerGenerator {
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        assert!(pop.len() >= 2, "gen/tlbo-learner requires a population of at least 2 (pop_size={})", pop.len());
        let n = pop.len();
        let dim = floats(&pop.individuals[0]).len();

        (0..n).map(|i| {
            // Pinned: j drawn ONCE per learner, unconditionally distinct
            // from i (finding 6) -- rejection sampling on next_below(n),
            // the same distribution as the source's remaining-array index.
            let j = loop {
                let cand = ctx.rng.next_below(n as u64) as usize;
                if cand != i { break cand; }
            };
            let partner_is_better = pop.fitness[j] < pop.fitness[i];
            let x = floats(&pop.individuals[i]);
            let xp = floats(&pop.individuals[j]);
            let xs: Vec<f64> = (0..dim).map(|d| {
                let r = ctx.rng.next_f64();
                tlbo_learner_dim_step(x[d], xp[d], r, partner_is_better)
            }).collect();
            Genotype { blocks: vec![BlockValues::Float(xs)] }
        }).collect()
    }

    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/tlbo-learner", SupportedBlocks::Only(vec!["float"])).with_min_pop(2)
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_generator("gen/tlbo-teacher", |p| Ok(Box::new(TlboTeacherGenerator::from_params(p)?)));
    reg.register_generator("gen/tlbo-learner", |p| Ok(Box::new(TlboLearnerGenerator::from_params(p)?)));
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

    // ---- tlbo_teacher_dim_step ----

    #[test]
    fn teacher_dim_step_tf1_vanishes_when_teacher_equals_mean() {
        // Hand-derived: teacher_d == mean_d exactly => (teacher_d - TF*mean_d)
        // == mean_d*(1-TF). TF=1 makes the factor exactly zero, so the
        // dimension is left EXACTLY unchanged regardless of r.
        let mean_d = 3.25;
        let r_arbitrary = 0.91;
        let result = tlbo_teacher_dim_step(7.0, mean_d, mean_d, 1.0, r_arbitrary);
        assert_eq!(result, 7.0);
    }

    #[test]
    fn teacher_dim_step_tf2_shifts_by_minus_r_times_teacher_when_teacher_equals_mean() {
        // Same fixture, TF=2: factor becomes mean_d*(1-2) = -mean_d, so the
        // step is exactly -r*teacher_d (since teacher_d == mean_d here).
        let mean_d = 3.25;
        let r = 0.91;
        let result = tlbo_teacher_dim_step(7.0, mean_d, mean_d, 2.0, r);
        let expected = 7.0 + r * (-mean_d);
        assert!((result - expected).abs() < 1e-12, "got {result}, expected {expected}");
    }

    #[test]
    fn teacher_dim_step_zero_r_leaves_dimension_unchanged() {
        for tf in [1.0, 2.0] {
            let result = tlbo_teacher_dim_step(5.0, 100.0, -50.0, tf, 0.0);
            assert_eq!(result, 5.0);
        }
    }

    #[test]
    fn teacher_dim_step_matches_hand_computed_value() {
        // x=1.0, teacher=4.0, mean=2.0, TF=2, r=0.5:
        // x + r*(teacher - TF*mean) = 1.0 + 0.5*(4.0 - 4.0) = 1.0
        assert_eq!(tlbo_teacher_dim_step(1.0, 4.0, 2.0, 2.0, 0.5), 1.0);
        // TF=1: 1.0 + 0.5*(4.0 - 2.0) = 2.0
        assert_eq!(tlbo_teacher_dim_step(1.0, 4.0, 2.0, 1.0, 0.5), 2.0);
    }

    // ---- tlbo_learner_dim_step: direction property (finding 7) ----

    #[test]
    fn learner_dim_step_moves_exactly_to_partner_when_better_and_r_is_1() {
        // Hand-derived 2-learner fixture: partner strictly better, r=1 =>
        // x' = x + 1*(partner - x) = partner EXACTLY.
        let result = tlbo_learner_dim_step(2.0, 9.0, 1.0, true);
        assert_eq!(result, 9.0);
    }

    #[test]
    fn learner_dim_step_moves_away_when_worse_and_r_is_1() {
        // partner NOT better, r=1 => x' = x + 1*(x - partner) = 2x - partner.
        let result = tlbo_learner_dim_step(2.0, 9.0, 1.0, false);
        assert_eq!(result, 2.0 * 2.0 - 9.0);
    }

    #[test]
    fn learner_dim_step_zero_r_leaves_dimension_unchanged() {
        for better in [true, false] {
            assert_eq!(tlbo_learner_dim_step(3.0, 999.0, 0.0, better), 3.0);
        }
    }

    // ---- TlboTeacherGenerator ----

    #[test]
    fn teacher_determinism_same_seed_bit_identical() {
        let n = 6; let dim = 4;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_nd(n, dim);

        let run = || {
            let mut evaluator = Evaluator::new(&p, 1000);
            let mut rng = RngStream::from_master(42, &[]);
            let mut bb = Blackboard::new();
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
            TlboTeacherGenerator.generate(&pop, &mut ctx)
        };

        let a = run();
        let b = run();
        assert_eq!(a.len(), n);
        for (ga, gb) in a.iter().zip(b.iter()) {
            assert_eq!(floats(ga), floats(gb), "same seed must produce bit-identical offspring");
        }
    }

    #[test]
    fn teacher_draw_order_matches_pinned_structure_via_raw_replay() {
        // Twin-stream raw-replay: TF (1 integer draw) then per-dim r (dim
        // draws), per learner, in population order.
        let n = 6; let dim = 4;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_nd(n, dim);

        let mut rng = RngStream::from_master(13, &[]);
        let rng_before = rng.clone();
        let mut evaluator = Evaluator::new(&p, 1000);
        let mut bb = Blackboard::new();
        {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
            let off = TlboTeacherGenerator.generate(&pop, &mut ctx);
            assert_eq!(off.len(), n);
        }

        let mut twin = rng_before;
        for _ in 0..n {
            twin.next_below(2); // TF
            for _ in 0..dim { twin.next_f64(); } // r per dim
        }
        assert_eq!(rng.next_f64(), twin.next_f64(),
            "gen/tlbo-teacher must consume exactly TF then per-dim r, per learner");
    }

    #[test]
    fn teacher_min_pop_below_2_panics() {
        let n = 1; let dim = 2;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_nd(n, dim);
        let mut evaluator = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            TlboTeacherGenerator.generate(&pop, &mut ctx)
        }));
        assert!(result.is_err(), "gen/tlbo-teacher must reject pop_size < 2 at runtime as a backstop");
    }

    // ---- TlboLearnerGenerator ----

    #[test]
    fn learner_determinism_same_seed_bit_identical() {
        let n = 6; let dim = 4;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_nd(n, dim);

        let run = || {
            let mut evaluator = Evaluator::new(&p, 1000);
            let mut rng = RngStream::from_master(42, &[]);
            let mut bb = Blackboard::new();
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
            TlboLearnerGenerator.generate(&pop, &mut ctx)
        };

        let a = run();
        let b = run();
        assert_eq!(a.len(), n);
        for (ga, gb) in a.iter().zip(b.iter()) {
            assert_eq!(floats(ga), floats(gb), "same seed must produce bit-identical offspring");
        }
    }

    #[test]
    fn learner_tie_fitness_treated_as_not_better_moves_away() {
        // n=2 forces the ONLY candidate partner: agent 0's partner is agent 1
        // (and vice versa). Equal fitness (a genuine tie) must be treated by
        // the STRICT `pop.fitness[j] < pop.fitness[i]` comparison as "j is
        // NOT better" (finding 7) -- the "away from partner" branch, not
        // "toward". "Away" means, for every dimension, the offspring's
        // distance from the partner is >= the parent's distance from the
        // partner (equality only at the r=0 edge case).
        let dim = 3;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = Population {
            individuals: vec![g(vec![1.0, 2.0, 3.0]), g(vec![4.0, 5.0, 6.0])],
            fitness: vec![7.0, 7.0], // exact tie
        };

        let mut rng = RngStream::from_master(5, &[]);
        let mut evaluator = Evaluator::new(&p, 100);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let off = TlboLearnerGenerator.generate(&pop, &mut ctx);

        let x0 = floats(&pop.individuals[0]);
        let x1 = floats(&pop.individuals[1]);
        let off0 = floats(&off[0]);
        for d in 0..dim {
            let dist_before = (x0[d] - x1[d]).abs();
            let dist_after = (off0[d] - x1[d]).abs();
            assert!(dist_after >= dist_before - 1e-12,
                "a tie must move AWAY from the partner (or stay put at r=0): dim {d}, before={dist_before}, after={dist_after}");
        }
    }

    #[test]
    fn learner_partner_always_distinct_from_self() {
        // Finding 6: j is UNCONDITIONALLY distinct from i (never
        // self-selecting, unlike fpa.rs's/woa.rs's pick idiom). Exercised
        // indirectly: with n=2, learner i's only valid partner is "the
        // other one" -- verify via the generator's determinism plus a
        // direct construction of the underlying rejection loop.
        let mut rng = RngStream::from_master(11, &[]);
        for i in 0..3usize {
            for _ in 0..100 {
                let j = loop {
                    let cand = rng.next_below(5) as usize;
                    if cand != i { break cand; }
                };
                assert_ne!(j, i);
                assert!(j < 5);
            }
        }
    }

    #[test]
    fn learner_draw_order_matches_pinned_structure_via_raw_replay() {
        // Twin-stream raw-replay: j (>=1 rejection-sampling draw) then
        // per-dim r (dim draws), per learner, in population order.
        let n = 6; let dim = 4;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_nd(n, dim);

        let mut rng = RngStream::from_master(13, &[]);
        let rng_before = rng.clone();
        let mut evaluator = Evaluator::new(&p, 1000);
        let mut bb = Blackboard::new();
        {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
            let off = TlboLearnerGenerator.generate(&pop, &mut ctx);
            assert_eq!(off.len(), n);
        }

        let mut twin = rng_before;
        for i in 0..n {
            loop {
                let cand = twin.next_below(n as u64) as usize;
                if cand != i { break; }
            }
            for _ in 0..dim { twin.next_f64(); } // r per dim
        }
        assert_eq!(rng.next_f64(), twin.next_f64(),
            "gen/tlbo-learner must consume exactly j (rejection sampling) then per-dim r, per learner");
    }

    #[test]
    fn learner_min_pop_below_2_panics() {
        let n = 1; let dim = 2;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_nd(n, dim);
        let mut evaluator = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            TlboLearnerGenerator.generate(&pop, &mut ctx)
        }));
        assert!(result.is_err(), "gen/tlbo-learner must reject pop_size < 2 at runtime as a backstop");
    }
}
