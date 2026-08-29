use sezgi_core::component::*;
use sezgi_core::problem::Population;
use sezgi_core::space::{BlockValues, Genotype};
use sezgi_core::state::StateReq;

/// Moth-Flame Optimization (Mirjalili, S. 2015, "Moth-flame optimization
/// algorithm: A novel nature-inspired heuristic paradigm", *Knowledge-Based
/// Systems* 89, 228-249, DOI 10.1016/j.knosys.2015.07.006) -- a **labeled
/// metaphor preset**: faithful to the primary source's own reference MATLAB
/// implementation's update equations and loop structure, with a pinned
/// deterministic draw order and property tests, but NOT validated against
/// the paper's (or any other publication's) reported benchmark numbers.
/// Equivalence critique: Camacho-Villalón, Dorigo & Stützle (*International
/// Transactions in Operational Research*, six-algorithm critique: grey wolf,
/// moth-flame, whale, firefly, bat, antlion) -- cited here conservatively,
/// as background on why this is "labeled metaphor" rather than a mechanism
/// sezgi treats as novel.
///
/// **This is the wave's first STATEFUL algorithm**: the "flames" are a
/// genuine persisted memory (blackboard `mfo/flames` + `mfo/flame_fitness`),
/// not a simplified current-population convention -- the flame memory *is*
/// MFO's defining mechanism (an elitist archive that only ever improves or
/// holds steady, generation to generation), so it is implemented as real
/// blackboard state, per the controller's ruling that the wave-wide
/// current-pop-vs-persisted-best convention (parked for attractor-selection
/// conventions like `X_best`) does NOT cover MFO's core flame mechanism.
///
/// ## Provenance (PROVENANCE-FIRST protocol)
///
/// **Primary artifact used (governs):** `MFO.m`, Seyedali Mirjalili's own
/// reference MATLAB implementation, MATLAB Central File Exchange submission
/// **#52269** ("Moth-flame Optimization (MFO) Algorithm"), authored by
/// Mirjalili himself and explicitly linked from the paper ("This is the
/// source codes of the paper: S. Mirjalili, Moth-Flame Optimization
/// Algorithm..."). This is a MORE DIRECT route than the task brief's
/// suggested fallback (reusing the #55980 "A new MATLAB optimization
/// toolbox" bundle from Task 1/SCA's provenance, which also happens to
/// contain a copy of `MFO.m`): #52269 is MFO's own dedicated, author-linked
/// submission, so it was fetched and used instead. Accessed via the
/// MathWorks File Exchange preview mirror (`mlc-downloads/downloads/
/// submissions/52269/versions/2/previews/MFO/MFO.m/index.html`) -- full
/// listing not reproducible verbatim here (fair-use limit on the fetch, the
/// same constraint Task 1/SCA's provenance note recorded), but the loop
/// structure and equations were extracted and quoted in fragments directly
/// from the source, not inferred from the paper's prose.
///
/// **Extracted equations / loop structure (from `MFO.m`, per-iteration
/// order):**
/// - Fitness of the current `Moth_pos` is evaluated fresh every iteration.
/// - **Flame update (the "double-population sort"):** on iteration 1,
///   `sorted_population = Moth_pos` sorted by `Moth_fitness` ascending
///   (`best_flames = sorted_population`, `best_flame_fitness =
///   fitness_sorted`). On every subsequent iteration: `double_population =
///   [previous_population; best_flames]` (concatenation: THIS iteration's
///   freshly-evaluated `Moth_pos` first, then the PREVIOUS iteration's
///   `best_flames`), `double_fitness = [previous_fitness
///   best_flame_fitness]`, sorted ascending, truncated to the first `N` --
///   those become the new `sorted_population`/`best_flames`/
///   `best_flame_fitness`. This is a genuine elitist merge-sort-truncate: a
///   flame can only be displaced by a fitter individual, never simply
///   overwritten each generation, and the "double population" concatenation
///   ORDER (current moths before old flames) matters for the tie-break
///   (MATLAB's `sort` is stable), which this module reproduces bit-exactly
///   -- see [`mfo_merge_and_truncate`]'s tests.
/// - `a = -1 + Iteration·(-1/Max_iteration)`, i.e. `a` decreases linearly
///   from `-1` (iteration 0) to `-2` (iteration `Max_iteration`) -- computed
///   ONCE per iteration. In sezgi's convention, `Max_iteration`-relative
///   progress is approximated by the established `progress = ctx.eval.used()
///   / ctx.eval.budget()` (clamped `[0,1]`) already used by `gwo`/`woa`/
///   `sca` (an existing wave-wide convention, not a new decision for this
///   task): `a = -1 - progress`.
/// - **Flame count**, Eq. (3.14) of the paper: `Flame_no = round(N -
///   Iteration·(N-1)/Max_iteration)` -- `round(N - progress·(N-1))` in
///   sezgi's progress convention. Linear from `N` (progress 0) down to `1`
///   (progress 1); MATLAB's `round` is round-half-away-from-zero, matching
///   Rust's `f64::round`.
/// - `b = 1` -- a literal constant re-assigned inside the loop body every
///   `(i,j)` pass in `MFO.m` (not a schedule, not tunable); sezgi hoists it
///   to a `const`-equivalent local since re-assigning a constant inside the
///   loop has no observable effect.
/// - **The move, and the wave's WOA-class draw-placement trap, VERIFIED
///   against `MFO.m` line-for-line:** `t` is drawn **once per (moth,
///   dimension) pair** (`t=(a-1)*rand+1` sits inside BOTH nested loops, `for
///   i=1:size(Moth_pos,1)` then `for j=1:size(Moth_pos,2)`) -- NOT once per
///   moth. (mealpy's `OriginalMFO`, consulted as the protocol's secondary/
///   fallback source, draws `t` only ONCE PER MOTH -- `t =
///   self.generator.uniform()` sits OUTSIDE mealpy's per-dimension
///   vectorized ops, one call per `idx` -- diverging from `MFO.m`'s own
///   per-dimension draws. Per the provenance protocol, `MFO.m` (the paper's
///   own linked code) governs.) Total draw count per `generate()` call:
///   exactly `n·dim`, in `(i, j)` row-major order -- the same idiom SCA/GWO/
///   WOA already use for their per-agent-per-dimension draws.
/// - **A second, subtler VERIFIED delta from this plan's sketch -- the
///   real trap in this task:** `MFO.m`'s two branches (`i<=Flame_no` vs
///   `i>Flame_no`) use **different row indices for different parts of the
///   formula**, not a single clamped index for both:
///   `distance_to_flame = abs(sorted_population(i,j) - Moth_pos(i,j))` is
///   IDENTICAL in both branches -- it ALWAYS uses row `i` (the moth's own
///   index into the full `N`-row sorted array, which always exists, even
///   past `Flame_no`). Only the position TARGET added at the end differs:
///   `+ sorted_population(i,j)` when `i<=Flame_no`, but `+
///   sorted_population(Flame_no,j)` when `i>Flame_no`. The plan's sketch
///   ("`D = |flame_k[d] − X_i[d]|` with `k = min(i, flame_count−1)`")
///   conflated these into a single clamped index `k` used for BOTH the
///   distance and the target -- that is WRONG per the verified source: `D`
///   ALWAYS uses the moth's own unclamped row `i`; only the additive target
///   flame position is clamped to `min(i, flame_count-1)`. This module
///   implements the verified (two-index) form -- see
///   [`MfoGenerator::generate`] and the `own_flame`-vs-`target_flame`
///   distinction in [`mfo_dim_step`]'s tests.
/// - `Flame_no`'s target-index mapping in 0-indexed terms: `target_idx =
///   min(i, flame_count - 1)` (the sketch's clamp formula IS correct for
///   this one term). mealpy's fallback code diverges here too: it uses
///   `g_best = pop_flames[0]` (the single best flame) for the
///   beyond-`Flame_no` branch, not `sorted_population(Flame_no,:)`
///   (`min(i, flame_count-1)`, the LAST *active* flame, not the single
///   best) -- `MFO.m` (verified, primary) governs; mealpy's simplification
///   is not reproduced here.
/// - Replacement: `MFO.m` applies the position update UNCONDITIONALLY to
///   `Moth_pos` (no per-moth greedy fitness comparison in the sequential
///   non-parallel branch) -- the flames, not the moth population, carry the
///   elitism. `replace/generational` (reused as-is) is the correct pin.
///
/// ## Blackboard state (this is the wave's first stateful/blackboard
/// algorithm; state keys namespaced per algorithm, per this wave's
/// convention, e.g. `ba/velocity`)
///
/// - `mfo/flames`: `Vec<Vec<f64>>`, length `pop_size`, the flame positions
///   (row `i` = the `i`-th best-ranked flame after the most recent
///   merge-sort-truncate).
/// - `mfo/flame_fitness`: `Vec<f64>`, length `pop_size`, the flames'
///   fitness values, co-indexed with `mfo/flames`.
///
/// **Ownership** ([`MfoGenerator::meta`] declares `requires`, NOT
/// `provides`, for both keys; [`MfoFlameAdapter::meta`] declares BOTH
/// `requires` (it reads the existing flames to merge) AND `provides` (it
/// writes the merged/truncated result) -- the adapter is this state's
/// canonical owner, per the task brief's design). **Bootstrap exception,
/// documented here for reviewers so the requires-vs-provides asymmetry
/// isn't mistaken for a bug:** on this preset's very FIRST `generate()`
/// call (generation 0), the blackboard has no flame memory yet -- no
/// adapter has ever run. `MfoGenerator::generate` itself performs a
/// ONE-TIME bootstrap seed when `mfo/flames` is absent (mirrors `pso.rs`'s/
/// `shade.rs`'s established lazy-init-when-absent pattern), using
/// [`mfo_merge_and_truncate`] with an EMPTY old-flame set -- which
/// degenerates to exactly `MFO.m`'s own `Iteration==1` special case
/// (`sorted_population = sort(Moth_pos)`, no merge partner). // sezgi
/// simplification: `MFO.m` hand-codes the `Iteration==1` case as a
/// separate branch; sezgi unifies it as a merge against an empty flame set
/// through the SAME [`mfo_merge_and_truncate`] helper the adapter uses --
/// behaviorally identical (merging with nothing is a no-op beyond the sort
/// itself), verified by [`mfo_dim_step`]'s and this module's bootstrap
/// test.
///
/// The per-dimension math is factored into [`mfo_dim_step`] and the flame
/// memory's merge-sort-truncate into [`mfo_merge_and_truncate`] /
/// [`mfo_flame_count`] so each can be unit-tested directly without needing
/// to fake `Ctx`/`RngStream`.
///
/// `min_pop = 2`: MFO needs at least 2 flames for the `i<=Flame_no` /
/// `i>Flame_no` branch split to be meaningful (with `pop_size == 1`,
/// `Flame_no` is always `1` and every moth's own row IS the only flame,
/// degenerating but not breaking the algorithm -- `min_pop = 2` matches the
/// wave's established floor for single-best/worst-style presets, e.g.
/// `sca.rs`/`jaya.rs`).
///
/// Boundary handling is NOT part of this generator: `presets::mfo` reuses
/// `boundary/clamp` (same as `gwo`/`woa`/`sca`/`jaya` -- `MFO.m` does clamp
/// out-of-bound coordinates via its own `Flag4ub`/`Flag4lb` masks, so this
/// is both this project's standard choice AND consistent with the source).
pub fn mfo_dim_step(x_d: f64, own_flame_d: f64, target_flame_d: f64, t: f64) -> f64 {
    let b = 1.0_f64;
    let distance_to_flame = (own_flame_d - x_d).abs();
    distance_to_flame * (b * t).exp() * (t * 2.0 * std::f64::consts::PI).cos() + target_flame_d
}

/// Flame count, Eq. (3.14): `round(N − progress·(N−1))`, `progress ∈
/// [0,1]`. Linear from `N` (progress 0) to `1` (progress 1); MATLAB's
/// `round` is round-half-away-from-zero (matches `f64::round`). Clamped to
/// `[1, n]` as a defensive belt-and-suspenders measure -- with `progress`
/// already clamped to `[0,1]` by the caller, the unclamped formula is
/// already exactly bounded in `[1, n]`, so this never actually fires, but
/// costs nothing to state explicitly.
pub fn mfo_flame_count(n: usize, progress: f64) -> usize {
    let raw = n as f64 - progress * (n as f64 - 1.0);
    (raw.round() as i64).clamp(1, n as i64) as usize
}

/// The flame memory's merge-sort-truncate (pinned, part of the RNG-stream-
/// free state-update contract -- no RNG draws occur here): concatenates
/// `(pop_fitness, pop_positions)` FIRST, then `(flame_fitness,
/// flame_positions)`, sorts the combined list ascending by fitness (a
/// STABLE sort, so ties keep the population-before-flames order --
/// reproducing `MFO.m`'s `double_population=[previous_population;
/// best_flames]` concatenation order and MATLAB `sort`'s stability), then
/// truncates to the first `pop_fitness.len()` entries. Used both by
/// [`MfoFlameAdapter::adapt`] (every generation, against the previous
/// flames) and by [`MfoGenerator::generate`]'s one-time bootstrap (against
/// an empty flame set).
pub fn mfo_merge_and_truncate(
    pop_fitness: &[f64],
    pop_positions: &[Vec<f64>],
    flame_fitness: &[f64],
    flame_positions: &[Vec<f64>],
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = pop_fitness.len();
    let mut combined: Vec<(f64, Vec<f64>)> = Vec::with_capacity(n + flame_fitness.len());
    for i in 0..n {
        combined.push((pop_fitness[i], pop_positions[i].clone()));
    }
    for i in 0..flame_fitness.len() {
        combined.push((flame_fitness[i], flame_positions[i].clone()));
    }
    combined.sort_by(|a, b| a.0.total_cmp(&b.0));
    combined.truncate(n);
    combined.into_iter().unzip()
}

fn floats(g: &Genotype) -> &Vec<f64> {
    match &g.blocks[0] { BlockValues::Float(x) => x, _ => unreachable!() }
}

pub struct MfoGenerator;

impl MfoGenerator {
    pub fn from_params(_p: &serde_json::Value) -> Result<Self, ComponentError> {
        // No tunable parameters (b=1 is a fixed constant in the source; a's
        // schedule is fully determined by ctx.eval progress).
        Ok(Self)
    }
}

impl Generator for MfoGenerator {
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        assert!(pop.len() >= 2, "gen/mfo requires a population of at least 2 (pop_size={})", pop.len());
        let n = pop.len();
        let dim = floats(&pop.individuals[0]).len();

        // Bootstrap (generation 0 only): no adapter has run yet, so seed the
        // flame memory from the initial (sorted) population -- see the
        // module doc's "Bootstrap exception" note.
        if !ctx.bb.contains("mfo/flames") {
            let pop_positions: Vec<Vec<f64>> =
                pop.individuals.iter().map(|g| floats(g).clone()).collect();
            let (flame_fitness, flames) =
                mfo_merge_and_truncate(&pop.fitness, &pop_positions, &[], &[]);
            ctx.bb.insert("mfo/flames", flames);
            ctx.bb.insert("mfo/flame_fitness", flame_fitness);
        }

        let flames = ctx.bb.get::<Vec<Vec<f64>>>("mfo/flames").unwrap().clone();

        let used = ctx.eval.used() as f64;
        let budget = ctx.eval.budget() as f64;
        let progress = if budget > 0.0 { (used / budget).clamp(0.0, 1.0) } else { 0.0 };
        let a = -1.0 - progress;
        let flame_count = mfo_flame_count(n, progress);

        (0..n).map(|i| {
            let x = floats(&pop.individuals[i]);
            let own_flame = &flames[i];
            let target_flame = &flames[i.min(flame_count - 1)];
            let xs: Vec<f64> = (0..dim).map(|d| {
                let t = (a - 1.0) * ctx.rng.next_f64() + 1.0;
                mfo_dim_step(x[d], own_flame[d], target_flame[d], t)
            }).collect();
            Genotype { blocks: vec![BlockValues::Float(xs)] }
        }).collect()
    }

    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/mfo", SupportedBlocks::Only(vec!["float"]))
            .with_requires(vec![
                StateReq::of::<Vec<Vec<f64>>>("mfo/flames"),
                StateReq::of::<Vec<f64>>("mfo/flame_fitness"),
            ])
            .with_min_pop(2)
    }
}

/// Companion adapter for `gen/mfo`: the flame-memory merge-sort-truncate
/// (see the module doc's "Flame update" extraction), run once per
/// generation AFTER `replace/generational` has installed the freshly-moved
/// moths into `pop`. Reads the PREVIOUS flames (this generation's move
/// target) out of the blackboard, merges them against the just-replaced
/// `pop` (this generation's freshly-evaluated moths) via
/// [`mfo_merge_and_truncate`], and writes the truncated result back as the
/// NEXT generation's flames -- exactly `MFO.m`'s `double_population =
/// [previous_population; best_flames]` step, with `previous_population` ==
/// this stage's just-evaluated `pop` and `best_flames` == the flames this
/// same generation's `gen/mfo` call just consumed.
pub struct MfoFlameAdapter;

impl Adapter for MfoFlameAdapter {
    fn adapt(&self, pop: &mut Population, ctx: &mut Ctx) {
        let pop_positions: Vec<Vec<f64>> =
            pop.individuals.iter().map(|g| floats(g).clone()).collect();
        let old_flames = ctx.bb.get::<Vec<Vec<f64>>>("mfo/flames").unwrap().clone();
        let old_flame_fitness = ctx.bb.get::<Vec<f64>>("mfo/flame_fitness").unwrap().clone();

        let (new_flame_fitness, new_flames) =
            mfo_merge_and_truncate(&pop.fitness, &pop_positions, &old_flame_fitness, &old_flames);

        ctx.bb.insert("mfo/flames", new_flames);
        ctx.bb.insert("mfo/flame_fitness", new_flame_fitness);
    }

    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("adapter/mfo-flame-update", SupportedBlocks::Only(vec!["float"]))
            .with_requires(vec![
                StateReq::of::<Vec<Vec<f64>>>("mfo/flames"),
                StateReq::of::<Vec<f64>>("mfo/flame_fitness"),
            ])
            .with_provides(vec![
                StateReq::of::<Vec<Vec<f64>>>("mfo/flames"),
                StateReq::of::<Vec<f64>>("mfo/flame_fitness"),
            ])
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_generator("gen/mfo", |p| Ok(Box::new(MfoGenerator::from_params(p)?)));
    reg.register_adapter("adapter/mfo-flame-update", |_| Ok(Box::new(MfoFlameAdapter)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::problem::{Problem, SphereShifted, Evaluator};
    use sezgi_core::rng::RngStream;
    use sezgi_core::state::Blackboard;

    fn g(xs: Vec<f64>) -> Genotype { Genotype { blocks: vec![BlockValues::Float(xs)] } }

    fn pop_nd(n: usize, dim: usize) -> Population {
        // Distinct fitness so sort order is unambiguous.
        Population {
            individuals: (0..n).map(|i| g((0..dim).map(|d| (i * dim + d) as f64 * 0.37 - 1.0).collect())).collect(),
            fitness: (0..n).map(|i| (n - i) as f64).collect(), // higher index = better (lower fitness)
        }
    }

    // ---- mfo_dim_step: spiral t=0 limit property ----

    #[test]
    fn t_zero_limit_gives_distance_plus_target_exact() {
        // exp(b*0) = 1, cos(0) = 1 exactly (no floating rounding at these
        // special inputs), so the formula collapses EXACTLY to
        // distance_to_flame + target_flame_d, regardless of own_flame_d
        // (as long as distance is computed from it).
        for (x_d, own_flame_d, target_flame_d) in
            [(-3.5, 2.0, 7.0), (0.0, 0.0, 0.0), (10.0, -4.0, -1.5)] {
            let result = mfo_dim_step(x_d, own_flame_d, target_flame_d, 0.0);
            let expected = (own_flame_d - x_d).abs() + target_flame_d;
            assert_eq!(result, expected,
                "t=0 must give distance_to_flame + target_flame_d exactly (x_d={x_d}, own={own_flame_d}, target={target_flame_d})");
        }
    }

    #[test]
    fn own_flame_and_target_flame_are_independent_inputs() {
        // The verified delta vs the plan's sketch: distance_to_flame uses
        // own_flame_d (the moth's own unclamped row), the additive term
        // uses target_flame_d (the clamped row) -- these can legitimately
        // differ, and the helper must treat them as two separate
        // parameters, not a single shared index.
        let with_same = mfo_dim_step(1.0, 5.0, 5.0, 0.3);
        let with_different = mfo_dim_step(1.0, 5.0, 9.0, 0.3);
        assert_ne!(with_same, with_different,
            "changing only target_flame_d (own_flame_d/distance held fixed) must change the result");
        // The distance term (own_flame_d - x_d) must NOT depend on target_flame_d.
        let d1 = mfo_dim_step(1.0, 5.0, 0.0, 0.3) - 0.0;
        let d2 = mfo_dim_step(1.0, 5.0, 100.0, 0.3) - 100.0;
        assert!((d1 - d2).abs() < 1e-12,
            "distance_to_flame*exp(b*t)*cos(2*pi*t) term must be identical regardless of target_flame_d");
    }

    // ---- mfo_flame_count: Eq. (3.14) schedule ----

    #[test]
    fn flame_count_schedule_progress_zero_is_pop_size_progress_one_is_one() {
        assert_eq!(mfo_flame_count(30, 0.0), 30, "progress=0 must give the full population as flames");
        assert_eq!(mfo_flame_count(30, 1.0), 1, "progress=1 must collapse to a single flame");
        assert_eq!(mfo_flame_count(2, 0.0), 2);
        assert_eq!(mfo_flame_count(2, 1.0), 1);
    }

    #[test]
    fn flame_count_schedule_intermediate_rounds_half_away_from_zero() {
        // n=10, progress=0.5: 10 - 0.5*9 = 5.5 -> round-half-away-from-zero -> 6
        // (MATLAB's round() and Rust's f64::round() agree on this tie).
        assert_eq!(mfo_flame_count(10, 0.5), 6);
        // n=11, progress=0.5: 11 - 0.5*10 = 6.0 -> exact, no rounding ambiguity.
        assert_eq!(mfo_flame_count(11, 0.5), 6);
    }

    // ---- mfo_merge_and_truncate: the flame-update mechanism, hand-built ----

    #[test]
    fn merge_and_truncate_matches_hand_built_mechanism_bit_exact() {
        // 2 moths (fitness 5.0, 1.0) merged against 2 old flames
        // (fitness 3.0, 10.0) -> combined sorted ascending: 1.0(moth1),
        // 3.0(flame0), 5.0(moth0), 10.0(flame1) -> truncate to N=2:
        // [1.0(moth1), 3.0(flame0)].
        let pop_fitness = vec![5.0, 1.0];
        let pop_positions = vec![vec![50.0], vec![10.0]];
        let flame_fitness = vec![3.0, 10.0];
        let flame_positions = vec![vec![30.0], vec![100.0]];

        let (new_fitness, new_positions) =
            mfo_merge_and_truncate(&pop_fitness, &pop_positions, &flame_fitness, &flame_positions);

        assert_eq!(new_fitness, vec![1.0, 3.0]);
        assert_eq!(new_positions, vec![vec![10.0], vec![30.0]]);
    }

    #[test]
    fn merge_and_truncate_ties_break_pop_before_flames() {
        // A tie in fitness (2.0) between a pop entry and a flame entry: the
        // pop entry (inserted first in the concatenation, per MFO.m's
        // [previous_population; best_flames] order) must win the tie under
        // a stable sort.
        let pop_fitness = vec![2.0];
        let pop_positions = vec![vec![999.0]]; // tagged so we can tell which one survived
        let flame_fitness = vec![2.0, 50.0];
        let flame_positions = vec![vec![-1.0], vec![-2.0]];

        let (new_fitness, new_positions) =
            mfo_merge_and_truncate(&pop_fitness, &pop_positions, &flame_fitness, &flame_positions);

        assert_eq!(new_fitness, vec![2.0]);
        assert_eq!(new_positions, vec![vec![999.0]],
            "on a fitness tie, the population entry (concatenated first) must win under a stable sort");
    }

    #[test]
    fn merge_and_truncate_against_empty_flames_is_just_sorted_pop() {
        // The bootstrap case (generation 0): merging against an empty flame
        // set must degenerate to exactly sort(pop by fitness) -- MFO.m's
        // Iteration==1 special case, reproduced via the general helper.
        let pop_fitness = vec![3.0, 1.0, 2.0];
        let pop_positions = vec![vec![30.0], vec![10.0], vec![20.0]];

        let (new_fitness, new_positions) =
            mfo_merge_and_truncate(&pop_fitness, &pop_positions, &[], &[]);

        assert_eq!(new_fitness, vec![1.0, 2.0, 3.0]);
        assert_eq!(new_positions, vec![vec![10.0], vec![20.0], vec![30.0]]);
    }

    // ---- MfoGenerator ----

    #[test]
    fn bootstrap_seeds_flames_from_sorted_initial_population_when_absent() {
        let n = 4; let dim = 2;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_nd(n, dim); // fitness = [4,3,2,1] for indices [0,1,2,3]

        let mut evaluator = Evaluator::new(&p, 1000);
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();
        assert!(!bb.contains("mfo/flames"));

        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        MfoGenerator.generate(&pop, &mut ctx);

        let flames = ctx.bb.get::<Vec<Vec<f64>>>("mfo/flames").unwrap();
        let flame_fitness = ctx.bb.get::<Vec<f64>>("mfo/flame_fitness").unwrap();
        assert_eq!(flame_fitness, &vec![1.0, 2.0, 3.0, 4.0], "flames must be sorted ascending by fitness");
        // Sorted by fitness ascending means original indices [3,2,1,0].
        let expected_positions: Vec<Vec<f64>> =
            [3usize, 2, 1, 0].iter().map(|&i| floats(&pop.individuals[i]).clone()).collect();
        assert_eq!(flames, &expected_positions);
    }

    #[test]
    fn own_flame_index_unclamped_target_flame_index_clamped() {
        // n=3, progress=1.0 -> flame_count=1 (Eq. 3.14: round(3 - 1*2) = 1).
        // Moth index 2 (i=2 > flame_count-1=0): distance_to_flame must use
        // flames[2] (its OWN row, unclamped), but the additive target must
        // use flames[0] (the clamped row) -- the verified two-index
        // mechanism, checked end-to-end through the generator.
        let dim = 1;
        let p = SphereShifted::new(vec![0.0; dim], -100.0, 100.0);
        let space = p.space();
        let pop = Population {
            individuals: vec![g(vec![1.0]), g(vec![2.0]), g(vec![3.0])],
            fitness: vec![3.0, 2.0, 1.0], // sorted ascending -> flames = [idx2, idx1, idx0] = [3.0, 2.0, 1.0]
        };

        let mut evaluator = Evaluator::new(&p, 100);
        evaluator.evaluate(&vec![g(vec![0.0]); 100]).unwrap(); // drive used==budget -> progress=1.0
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };

        let off = MfoGenerator.generate(&pop, &mut ctx);

        // Flames (sorted by fitness ascending from pop=[3,2,1] fitness=[3.0,2.0,1.0]):
        // flame[0]=pop[2]=3.0 (fitness 1.0), flame[1]=pop[1]=2.0 (fitness 2.0), flame[2]=pop[0]=1.0 (fitness 3.0).
        // flame_count=1 -> target_idx = min(i,0) = 0 for all i -> target_flame_d = flames[0][0] = 3.0.
        // own_flame for i=2 is flames[2][0] = 1.0 (moth 2's OWN row, unclamped).
        // moth 2's x_d = pop.individuals[2][0] = 3.0.
        // distance_to_flame = |own_flame(1.0) - x_d(3.0)| = 2.0 (NOT |target(3.0)-x_d(3.0)|=0.0).
        let x2 = floats(&off[2])[0];
        // a=-2 (progress=1), t=(a-1)*rand+1=(-3)*rand+1; can't predict rand
        // without replaying the stream, so instead assert the moth actually
        // MOVED in a way consistent with a nonzero distance_to_flame: if
        // the (wrong) clamped-both-terms formula had been used instead,
        // distance_to_flame would be exactly 0.0 and x2 would equal
        // target_flame_d (3.0) for EVERY t, since distance*anything=0.
        // With the verified two-index formula, distance_to_flame=2.0 != 0,
        // so x2 depends on t and is (with overwhelming probability, since
        // rand is continuous) not exactly 3.0.
        assert_ne!(x2, 3.0,
            "moth 2's own (unclamped) row must feed distance_to_flame, not the clamped target row -- \
             if both terms used the clamped index the distance would collapse to exactly 0 and x2 would equal target_flame_d=3.0");
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
            MfoGenerator.generate(&pop, &mut ctx)
        };

        let a = run();
        let b = run();
        for (ga, gb) in a.iter().zip(b.iter()) {
            assert_eq!(floats(ga), floats(gb), "same seed must produce bit-identical offspring");
        }
    }

    #[test]
    fn draw_count_is_exactly_n_times_dim_per_generate_call() {
        // Pinned: t is drawn ONCE PER (moth, dimension) pair -- n*dim draws
        // total, i outer / j inner -- verified against MFO.m line-for-line
        // (see module doc). Twin-stream raw replay.
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
            let off = MfoGenerator.generate(&pop, &mut ctx);
            assert_eq!(off.len(), n);
        }

        let mut twin = rng_before;
        for _ in 0..(n * dim) { twin.next_f64(); }
        assert_eq!(rng.next_f64(), twin.next_f64(),
            "gen/mfo must consume exactly n*dim RNG draws (n={n}, dim={dim}), i outer / j inner");
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
            MfoGenerator.generate(&pop, &mut ctx)
        }));
        assert!(result.is_err(), "gen/mfo must reject pop_size < 2 at runtime as a backstop");
    }

    // ---- MfoFlameAdapter ----

    #[test]
    fn adapter_updates_flames_via_merge_and_truncate() {
        let dim = 1;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();
        bb.insert("mfo/flames", vec![vec![30.0], vec![100.0]]);
        bb.insert("mfo/flame_fitness", vec![3.0, 10.0]);

        let mut pop = Population {
            individuals: vec![g(vec![50.0]), g(vec![10.0])],
            fitness: vec![5.0, 1.0],
        };

        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        MfoFlameAdapter.adapt(&mut pop, &mut ctx);

        // Same numbers as merge_and_truncate_matches_hand_built_mechanism_bit_exact:
        // combined sorted -> [1.0(moth1,10.0), 3.0(flame0,30.0), 5.0(moth0,50.0), 10.0(flame1,100.0)]
        // truncate to N=2 -> [1.0/10.0, 3.0/30.0].
        let new_flames = ctx.bb.get::<Vec<Vec<f64>>>("mfo/flames").unwrap();
        let new_fitness = ctx.bb.get::<Vec<f64>>("mfo/flame_fitness").unwrap();
        assert_eq!(new_fitness, &vec![1.0, 3.0]);
        assert_eq!(new_flames, &vec![vec![10.0], vec![30.0]]);
    }

    // ---- End-to-end: two hand-traced generations (generator + replace/generational + adapter) ----

    #[test]
    fn two_generations_flame_evolution_hand_traced() {
        // n=2, dim=1. Generation 0: bootstrap seeds flames from sorted
        // init pop. Generation 1: adapter must have merged generation 0's
        // (moved, replaced) offspring against generation 0's flames.
        let n = 2; let dim = 1;
        let p = SphereShifted::new(vec![0.0; dim], -1000.0, 1000.0);
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 100_000);
        let mut rng = RngStream::from_master(3, &[]);
        let mut bb = Blackboard::new();

        // Init population: fitness 10.0 (worse) at x=0.0, fitness 1.0 (better) at x=5.0.
        let mut pop = Population {
            individuals: vec![g(vec![0.0]), g(vec![5.0])],
            fitness: vec![10.0, 1.0],
        };

        // Generation 0: generate (bootstraps flames = sorted([idx1(5.0,f=1.0), idx0(0.0,f=10.0)])).
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let off0 = MfoGenerator.generate(&pop, &mut ctx);
        let flames_after_gen0_generate = ctx.bb.get::<Vec<Vec<f64>>>("mfo/flames").unwrap().clone();
        assert_eq!(flames_after_gen0_generate, vec![vec![5.0], vec![0.0]],
            "bootstrap flames must be the initial population sorted by fitness ascending");

        // replace/generational: pop <- off0 (unconditional); assign made-up
        // fresh fitness values for the offspring to drive the adapter.
        let off0_fitness = vec![7.0, 0.5]; // hand-picked, not re-derived from the real objective
        pop.individuals = off0;
        pop.fitness = off0_fitness.clone();

        // adapter/mfo-flame-update: merge pop (fitness [7.0, 0.5]) against
        // the flames generation 0's generate() just used ([5.0]@1.0, [0.0]@10.0).
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        MfoFlameAdapter.adapt(&mut pop, &mut ctx);

        // Combined: (7.0, off0[0]), (0.5, off0[1]), (1.0, [5.0]), (10.0, [0.0])
        // sorted ascending by fitness: 0.5(off0[1]), 1.0([5.0]), 7.0(off0[0]), 10.0([0.0])
        // truncate to N=2: [0.5(off0[1]), 1.0([5.0])].
        let flames = ctx.bb.get::<Vec<Vec<f64>>>("mfo/flames").unwrap();
        let flame_fitness = ctx.bb.get::<Vec<f64>>("mfo/flame_fitness").unwrap();
        assert_eq!(flame_fitness, &vec![0.5, 1.0]);
        assert_eq!(flames[1], vec![5.0], "the second flame must be the OLD flame [5.0] (fitness 1.0), still beating off0[0]'s 7.0");

        // Generation 1: generate() must now read these merged flames (not
        // re-bootstrap -- mfo/flames already present).
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 1 };
        let off1 = MfoGenerator.generate(&pop, &mut ctx);
        assert_eq!(off1.len(), n);
        let flames_unchanged = ctx.bb.get::<Vec<Vec<f64>>>("mfo/flames").unwrap();
        assert_eq!(flames_unchanged, &vec![flames_unchanged[0].clone(), vec![5.0]],
            "generate() at generation 1 must NOT modify or re-bootstrap the existing flame memory");
    }
}
