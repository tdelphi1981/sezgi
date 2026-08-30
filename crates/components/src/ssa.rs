use sezgi_core::component::*;
use sezgi_core::problem::Population;
use sezgi_core::space::{Block, BlockValues, Genotype};

/// Salp Swarm Algorithm (Mirjalili, S., Gandomi, A.H., Mirjalili, S.Z.,
/// Saremi, S., Faris, H. & Mirjalili, S.M. 2017, "Salp Swarm Algorithm: A
/// bio-inspired optimizer for engineering design problems", *Advances in
/// Engineering Software* 114, 163-191, DOI 10.1016/j.advengsoft.2017.07.002)
/// -- a **labeled metaphor preset**: faithful to the primary source's own
/// reference MATLAB implementation's update equations and loop structure,
/// with a pinned deterministic draw order and property tests, but NOT
/// validated against the paper's (or any other publication's) reported
/// benchmark numbers. No established equivalence critique covers SSA (the
/// Camacho-Villalón/Dorigo/Stützle *ITOR* six-algorithm critique covers
/// GWO/MFO/WOA/FA/BA/ALO, not SSA) -- cited here as primary-source only.
///
/// ## Provenance (PROVENANCE-FIRST protocol)
///
/// **Primary artifact used (governs):** `SSA.m`, Seyedali Mirjalili's own
/// reference MATLAB implementation, MATLAB Central File Exchange submission
/// **#63745** ("SSA: Salp Swarm Algorithm"), explicitly linked from the
/// paper ("source codes of the paper"). Accessed via the MathWorks File
/// Exchange preview mirror
/// (`mlc-downloads/downloads/submissions/63745/versions/1/previews/SSA/
/// SSA.m/index.html`); the loop structure and equations below are quoted
/// directly from the fetched source, not inferred from the paper's prose.
/// **Secondary/fallback source consulted for corroboration:** mealpy's
/// `OriginalSSO` (`mealpy/swarm_based/SSO.py`, the Salp Swarm Optimization
/// class -- NOTE: mealpy's module named literally `SSA.py` is a DIFFERENT
/// algorithm, the Sparrow Search Algorithm, a naming collision caught while
/// resolving this task's provenance; `SSO.py` is the actual Salp Swarm
/// reimplementation). `OriginalSSO` independently agrees with `SSA.m` on
/// every structural point extracted below (the half-population leader/
/// follower split, the `c1` schedule, the per-dimension `c2`/`c3` leader
/// draws, the sign-branch structure, and the in-place follower chain) --
/// see the one genuine delta it also confirms, below.
///
/// **Extracted equations / loop structure (from `SSA.m`, quoted/paraphrased
/// verbatim from the fetched source):**
///
/// ```text
/// c1 = 2*exp(-(4*l/Max_iter)^2);
/// for i=1:size(SalpPositions,1)
///     SalpPositions = SalpPositions';           % transpose to dim x N
///     if i<=N/2
///         for j=1:1:dim
///             c2=rand(); c3=rand();
///             if c3<0.5
///                 SalpPositions(j,i)=FoodPosition(j)+c1*((ub(j)-lb(j))*c2+lb(j));
///             else
///                 SalpPositions(j,i)=FoodPosition(j)-c1*((ub(j)-lb(j))*c2+lb(j));
///             end
///         end
///     elseif i>N/2 && i<N+1
///         point1=SalpPositions(:,i-1);
///         point2=SalpPositions(:,i);
///         SalpPositions(:,i)=(point2+point1)/2;
///     end
///     SalpPositions = SalpPositions';           % transpose back
/// end
/// ```
///
/// - `c1 = 2·e^{−(4·l/Max_iter)²}` -- matches the plan sketch's exact
///   exponent form. `SSA.m` uses a 1-based iteration counter `l` that starts
///   at `2` (the loop's first pass, since the pre-loop code already performs
///   the equivalent of "iteration 1": initial-population evaluation and
///   sort). sezgi does not replicate this specific off-by-one counter (the
///   wave's established convention, already used by `sca.rs`/`mfo.rs`, maps
///   the whole schedule onto `progress = ctx.eval.used() / ctx.eval.budget()`
///   clamped to `[0,1]` instead of a literal iteration count): `c1 =
///   2·e^{−(4·progress)²}` -- see [`ssa_c1`].
/// - **The leader/follower split -- the wave's flagged trap, VERIFIED
///   against `SSA.m` line-for-line and independently confirmed by mealpy's
///   `OriginalSSO`:** it is **HALF the population**, not a single leader at
///   index 0 as the plan's sketch phrased it ("leader (index 0)"). `SSA.m`'s
///   `if i<=N/2` (1-indexed) makes indices `1..floor(N/2)` leaders and
///   `floor(N/2)+1..N` followers -- in sezgi's 0-indexed terms,
///   `leader_count = n / 2` (integer division, i.e. `floor(n/2)`), indices
///   `0..leader_count` are leaders and `leader_count..n` are followers. This
///   is a genuine delta from the plan's sketch, not a sketch-consistent
///   detail: the sketch's "leader (index 0)" phrasing described a
///   single-leader mechanism (as in GWO's alpha or MFO's best flame);
///   `SSA.m`'s actual mechanism is a **fixed positional split of the
///   population array** (NOT a fitness-based leader selection -- the array
///   is sorted only ONCE, before the main loop, purely to seed the initial
///   `FoodPosition`; inside the loop `SalpPositions` itself is never
///   re-sorted by fitness, so "leader" vs "follower" is determined solely by
///   an individual's INDEX/POSITION in the population array, which this
///   crate's engine preserves across generations via
///   `replace/generational`'s unconditional overwrite-in-place).
/// - **Leader update, per dimension:** draw `c2` THEN `c3` (this exact
///   order, per `(leader, d)` pair); sign branch on `c3 < 0.5` -- `+` if
///   `c3 < 0.5`, `−` otherwise (the plan sketch named the branch but not
///   which sign goes with which condition; `SSA.m` pins `c3<0.5` to the `+`
///   branch): `X'[d] = food[d] ± c1·((hi[d]−lo[d])·c2 + lo[d])`. Exactly TWO
///   draws per `(leader, d)` pair, always -- see [`ssa_leader_dim_step`].
/// - **Followers draw ZERO random numbers** (`SSA.m`'s `elseif` branch
///   contains no `rand()` call at all) -- confirmed, matching the plan
///   sketch.
/// - **The follower chain -- the wave's second flagged trap, VERIFIED: the
///   UPDATING (in-place) array, not the previous generation's snapshot.**
///   `SSA.m` processes indices `i=1..N` in a single ascending loop, and
///   EVERY iteration re-transposes and re-assigns directly into
///   `SalpPositions` (the SAME variable being read by later iterations of
///   the SAME loop). For a follower at index `i`, `point1 =
///   SalpPositions(:,i-1)` is read from the CURRENT state of the array at
///   the moment index `i` is processed -- since indices are visited in
///   ascending order and index `i-1` was already visited (and its new
///   position written back into `SalpPositions`) earlier in this SAME
///   sweep, `point1` reflects `i-1`'s FRESH, this-generation position
///   (whether `i-1` is itself a leader or an earlier-processed follower),
///   NOT `i-1`'s pre-generation (previous-generation) position. `point2 =
///   SalpPositions(:,i)` is index `i`'s own OLD (pre-generation) position,
///   since index `i` has not yet been visited/overwritten by this sweep.
///   sezgi reproduces this exactly: `X'[i][d] = (X_old[i][d] +
///   X'[i-1][d]) / 2`, where `X'[i-1][d]` is the offspring array entry this
///   SAME `generate()` call already computed for index `i-1` -- a genuine
///   chain, since a follower several positions past the leader block
///   transitively incorporates every predecessor follower's move. See
///   [`ssa_follower_dim_step`] and `follower_chain_uses_in_place_updates_
///   not_old_snapshot` for the hand-derived 3-salp property test that
///   distinguishes this from the (wrong) old-snapshot alternative.
/// - **Food source -- VERIFIED as a persisted best-ever, and a deliberate
///   delta from that verified mechanism, per this task's explicit brief:**
///   `SSA.m`'s `FoodPosition`/`FoodFitness` are genuinely PERSISTED across
///   the whole run -- initialized once before the loop from the sorted
///   initial population, then updated (if-improved only, never regresses)
///   after each iteration's fitness re-evaluation. mealpy's `OriginalSSO`
///   independently confirms this (`self.g_best.solution`, its own persisted
///   global-best tracker). This is a real, verified persisted-best-ever
///   mechanism -- NOT an ambiguous case defaulting to the sketch's "use
///   pop-best if ambiguous" fallback. However, per this task's explicit
///   direction (distinct from `mfo.rs`'s flame memory, which the controller
///   separately ruled DOES warrant real blackboard state as MFO's core
///   defining mechanism): SSA's `FoodPosition` is functionally the SAME
///   kind of single-best-attractor value that `gwo.rs`/`woa.rs`/`sca.rs`/
///   `jaya.rs` already normalize to the **current-population argmin**
///   (ties → lower index, via [`Population::best_index`]) under this wave's
///   established "current-generation best, not a separately-persisted
///   historical best" convention (parked for attractor-selection values,
///   first pinned in Task 1/SCA's review) -- SSA does not get its own
///   blackboard exception the way MFO's N-sized flame archive did, because a
///   single persisted scalar-attractor is exactly the case the parked
///   convention already covers, not a new mechanism. `// sezgi simplification:`
///   this module uses the current population's fitness argmin as `food`,
///   not a persisted best-ever -- a genuine, deliberately NOT-implemented
///   delta from the verified source, recorded here rather than reproduced.
/// - **Replacement:** `SSA.m` applies the position update UNCONDITIONALLY
///   to `SalpPositions` every iteration (no per-agent greedy fitness
///   comparison -- only the separate `FoodPosition`/`FoodFitness` scalar is
///   conditionally updated, which sezgi does not persist per the delta
///   above). `replace/generational` (reused as-is) is the correct pin.
/// - **Boundary handling:** `SSA.m` clamps out-of-bound coordinates to
///   `[lb, ub]` via its own `Tp`/`Tm` boolean-mask logic, AFTER the full
///   position-update loop -- functionally a clamp, matching this project's
///   standard `boundary/clamp` (same as `gwo`/`woa`/`sca`/`jaya`/`mfo`),
///   applied by the preset, not this generator.
///
/// The per-dimension math is factored into [`ssa_leader_dim_step`] and
/// [`ssa_follower_dim_step`] (and the schedule into [`ssa_c1`] and the split
/// into [`ssa_leader_count`]) so each can be unit-tested directly without
/// needing to fake `Ctx`/`RngStream`. [`SsaGenerator::generate`] builds the
/// offspring array SEQUENTIALLY in ascending index order (required for the
/// follower chain to see already-computed predecessor entries, exactly
/// `SSA.m`'s single ascending sweep).
///
/// `min_pop = 2`: with `pop_size == 1`, `ssa_leader_count(1) == 0`, making
/// the single individual a follower with no valid predecessor index -- the
/// algorithm is not meaningful (and would panic on the chain lookup) below
/// 2; `pop_size == 2` gives exactly one leader (index 0) and one follower
/// (index 1, chaining off the leader), the minimal meaningful case.
///
/// Boundary handling and replacement are NOT part of this generator:
/// `presets::ssa` reuses `boundary/clamp` and `replace/generational` (both
/// reused as-is, see above).
pub fn ssa_c1(progress: f64) -> f64 {
    2.0 * (-((4.0 * progress).powi(2))).exp()
}

/// `SSA.m`'s `i<=N/2` (1-indexed) test, translated to a 0-indexed leader
/// COUNT: `floor(n/2)`. Indices `0..leader_count` are leaders; indices
/// `leader_count..n` are followers. This is a FIXED positional split of the
/// population array's index order (NOT a fitness-based re-selection every
/// generation) -- see the module doc's leader/follower delta note.
pub fn ssa_leader_count(n: usize) -> usize {
    n / 2
}

/// One leader's per-dimension move: `food_d ± c1·((hi−lo)·c2 + lo)`, `+` if
/// `c3 < 0.5` else `−` -- pinned sign branch, verified against `SSA.m`.
pub fn ssa_leader_dim_step(food_d: f64, c1: f64, c2: f64, c3: f64, lo: f64, hi: f64) -> f64 {
    let term = c1 * ((hi - lo) * c2 + lo);
    if c3 < 0.5 { food_d + term } else { food_d - term }
}

/// One follower's per-dimension move: `(x_old_d + prev_new_d) / 2` --
/// `prev_new_d` MUST be the immediate predecessor's ALREADY-COMPUTED (this
/// same `generate()` call's) new position, per the module doc's verified
/// in-place chain semantics, not the predecessor's previous-generation
/// position.
pub fn ssa_follower_dim_step(x_old_d: f64, prev_new_d: f64) -> f64 {
    (x_old_d + prev_new_d) / 2.0
}

fn floats(g: &Genotype) -> &Vec<f64> {
    match &g.blocks[0] { BlockValues::Float(x) => x, _ => unreachable!() }
}

pub struct SsaGenerator;

impl SsaGenerator {
    pub fn from_params(_p: &serde_json::Value) -> Result<Self, ComponentError> {
        // No tunable parameters (the c1 schedule is fully determined by
        // ctx.eval progress; the leader/follower split by pop_size).
        Ok(Self)
    }
}

impl Generator for SsaGenerator {
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        assert!(pop.len() >= 2, "gen/ssa requires a population of at least 2 (pop_size={})", pop.len());
        let n = pop.len();
        let dim = floats(&pop.individuals[0]).len();

        let (lo, hi): (Vec<f64>, Vec<f64>) = match &ctx.space.blocks()[0] {
            Block::Float { lo, hi, n: bn } => {
                debug_assert_eq!(*bn, dim);
                (vec![*lo; dim], vec![*hi; dim])
            }
            _ => unreachable!("gen/ssa only supports float blocks"),
        };

        let used = ctx.eval.used() as f64;
        let budget = ctx.eval.budget() as f64;
        let progress = if budget > 0.0 { (used / budget).clamp(0.0, 1.0) } else { 0.0 };
        let c1 = ssa_c1(progress);

        // Food: current-population fitness argmin, ties -> lower index --
        // pinned (see module doc's "Food source" delta note), computed once
        // per generate() call, before any RNG draws.
        let food_idx = pop.best_index().unwrap_or(0);
        let food = floats(&pop.individuals[food_idx]).clone();

        let leader_count = ssa_leader_count(n);

        let mut offspring: Vec<Genotype> = Vec::with_capacity(n);
        for i in 0..n {
            let x_old = floats(&pop.individuals[i]);
            let xs: Vec<f64> = if i < leader_count {
                (0..dim).map(|d| {
                    // Pinned draw order: c2, then c3, per (leader, d).
                    let c2 = ctx.rng.next_f64();
                    let c3 = ctx.rng.next_f64();
                    ssa_leader_dim_step(food[d], c1, c2, c3, lo[d], hi[d])
                }).collect()
            } else {
                // Follower: zero draws. prev's ALREADY-COMPUTED entry (this
                // same generate() call) -- the verified in-place chain.
                let prev = floats(&offspring[i - 1]);
                (0..dim).map(|d| ssa_follower_dim_step(x_old[d], prev[d])).collect()
            };
            offspring.push(Genotype { blocks: vec![BlockValues::Float(xs)] });
        }
        offspring
    }

    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/ssa", SupportedBlocks::Only(vec!["float"])).with_min_pop(2)
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_generator("gen/ssa", |p| Ok(Box::new(SsaGenerator::from_params(p)?)));
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

    // ---- ssa_c1: schedule endpoints ----

    #[test]
    fn c1_schedule_endpoints() {
        // progress=0: c1 = 2*e^0 = 2.0 exactly.
        assert_eq!(ssa_c1(0.0), 2.0);
        // progress=1: c1 = 2*e^-16, a small positive value, exact formula match.
        let expected = 2.0 * (-16.0_f64).exp();
        assert_eq!(ssa_c1(1.0), expected);
        assert!(ssa_c1(1.0) > 0.0 && ssa_c1(1.0) < 1e-6, "c1 at progress=1 must be small but positive");
    }

    #[test]
    fn c1_is_monotonically_decreasing_in_progress() {
        let c1_0 = ssa_c1(0.0);
        let c1_half = ssa_c1(0.5);
        let c1_1 = ssa_c1(1.0);
        assert!(c1_0 > c1_half, "c1(0) must exceed c1(0.5)");
        assert!(c1_half > c1_1, "c1(0.5) must exceed c1(1)");
    }

    // ---- ssa_leader_count: half-population split ----

    #[test]
    fn leader_count_is_floor_half() {
        assert_eq!(ssa_leader_count(2), 1);
        assert_eq!(ssa_leader_count(3), 1);
        assert_eq!(ssa_leader_count(4), 2);
        assert_eq!(ssa_leader_count(5), 2);
        assert_eq!(ssa_leader_count(30), 15);
    }

    // ---- ssa_leader_dim_step: sign branch ----

    #[test]
    fn leader_sign_branch_c3_below_half_is_plus() {
        let food_d = 10.0; let c1 = 0.5; let c2 = 0.4; let lo = -5.0; let hi = 5.0;
        let term = c1 * ((hi - lo) * c2 + lo);
        assert_eq!(ssa_leader_dim_step(food_d, c1, c2, 0.49, lo, hi), food_d + term,
            "c3 < 0.5 must select the + branch");
        assert_eq!(ssa_leader_dim_step(food_d, c1, c2, 0.5, lo, hi), food_d - term,
            "c3 >= 0.5 must select the - branch");
    }

    // ---- ssa_follower_dim_step ----

    #[test]
    fn follower_step_is_midpoint_of_old_and_predecessor_new() {
        assert_eq!(ssa_follower_dim_step(4.0, 10.0), 7.0);
        assert_eq!(ssa_follower_dim_step(-3.0, -3.0), -3.0);
    }

    // ---- SsaGenerator: the follower-chain hand-derived property, 3 salps ----

    #[test]
    fn follower_chain_uses_in_place_updates_not_old_snapshot() {
        // n=3, dim=1: leader_count = ssa_leader_count(3) = 1 -> index 0 is
        // the ONLY leader; indices 1, 2 are followers. Hand-derive the
        // EXACT expected offspring bit-for-bit from the pinned RNG draw
        // order, then separately show the (wrong) old-snapshot alternative
        // gives a DIFFERENT result -- distinguishing the two semantics.
        let dim = 1;
        let p = SphereShifted::new(vec![0.0; dim], -10.0, 10.0);
        let space = p.space();
        // x_old: [0]=leader's own old value (irrelevant to its own move,
        // which reads only `food`), [1]=5.0, [2]=100.0 -- picked far apart
        // so old-vs-new-predecessor substitution is unmistakable.
        let pop = Population {
            individuals: vec![g(vec![0.0]), g(vec![5.0]), g(vec![100.0])],
            fitness: vec![1.0, 2.0, 3.0], // index 0 is best -> food = [0.0]
        };

        let mut evaluator = Evaluator::new(&p, 1000); // progress=0 -> c1=2.0
        let mut rng = RngStream::from_master(11, &[]);
        let rng_before = rng.clone();
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let off = SsaGenerator.generate(&pop, &mut ctx);

        // Re-derive the leader's draws by hand from the SAME stream.
        let mut twin = rng_before;
        let c2 = twin.next_f64();
        let c3 = twin.next_f64();
        let food_d = 0.0; let c1 = ssa_c1(0.0); let lo = -10.0; let hi = 10.0;
        let leader_new = ssa_leader_dim_step(food_d, c1, c2, c3, lo, hi);
        assert_eq!(floats(&off[0])[0], leader_new, "leader's own move must match the hand-derived formula");

        // Follower 1 (index 1): must use leader_new (index 0's NEW value),
        // not x_old[0]=0.0.
        let expected_f1 = ssa_follower_dim_step(5.0, leader_new);
        assert_eq!(floats(&off[1])[0], expected_f1, "follower 1 must chain off the leader's NEW position");
        // Confirm this actually depends on using the NEW (not old) value:
        // the old-snapshot alternative would differ whenever leader_new != x_old[0] (0.0).
        assert_ne!(leader_new, 0.0, "sanity: leader must have actually moved from its old position");
        let wrong_using_old_snapshot = ssa_follower_dim_step(5.0, 0.0);
        assert_ne!(floats(&off[1])[0], wrong_using_old_snapshot,
            "follower 1 must NOT match the old-snapshot (previous-generation) alternative");

        // Follower 2 (index 2): must chain off follower 1's NEW value
        // (expected_f1), not x_old[1]=5.0 -- the transitive chain link.
        let expected_f2 = ssa_follower_dim_step(100.0, expected_f1);
        assert_eq!(floats(&off[2])[0], expected_f2, "follower 2 must chain off follower 1's NEW (already-updated) position");
        let wrong_using_old_snapshot_f2 = ssa_follower_dim_step(100.0, 5.0);
        assert_ne!(floats(&off[2])[0], wrong_using_old_snapshot_f2,
            "follower 2 must NOT match the old-snapshot alternative (chaining off x_old[1] instead of the updated follower 1)");
    }

    // ---- SsaGenerator: draw count (twin-stream, leaders-only) ----

    #[test]
    fn draw_count_is_exactly_2_times_leader_count_times_dim() {
        // Pinned: leaders draw c2,c3 per dimension (2*leader_count*dim
        // total); followers draw ZERO. Twin-stream raw replay.
        let n = 6; let dim = 4; // leader_count = 3
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_nd(n, dim);

        let mut rng = RngStream::from_master(7, &[]);
        let rng_before = rng.clone();
        let mut evaluator = Evaluator::new(&p, 1000);
        let mut bb = Blackboard::new();
        {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
            let off = SsaGenerator.generate(&pop, &mut ctx);
            assert_eq!(off.len(), n);
        }

        let leader_count = ssa_leader_count(n);
        assert_eq!(leader_count, 3);
        let mut twin = rng_before;
        for _ in 0..(2 * leader_count * dim) { twin.next_f64(); }
        assert_eq!(rng.next_f64(), twin.next_f64(),
            "gen/ssa must consume exactly 2*leader_count*dim RNG draws (n={n}, dim={dim}, leader_count={leader_count})");
    }

    #[test]
    fn followers_alone_draw_nothing_beyond_leaders() {
        // Same total-population size, but compare n where leader_count
        // stays fixed while more followers are appended (impossible to
        // vary n without changing leader_count in this algorithm's split,
        // so instead directly confirm zero draws are consumed between
        // computing the last leader and the first follower by checking the
        // exact count above already isolates followers' contribution to
        // zero -- this test instead pins the SPECIFIC n=2 minimal case
        // (1 leader, 1 follower) end-to-end).
        let n = 2; let dim = 3; // leader_count = 1
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_nd(n, dim);

        let mut rng = RngStream::from_master(3, &[]);
        let rng_before = rng.clone();
        let mut evaluator = Evaluator::new(&p, 1000);
        let mut bb = Blackboard::new();
        {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
            let _ = SsaGenerator.generate(&pop, &mut ctx);
        }
        let mut twin = rng_before;
        let leader_count = 1; // n=2 -> ssa_leader_count(2) == 1
        for _ in 0..(2 * leader_count * dim) { twin.next_f64(); }
        assert_eq!(rng.next_f64(), twin.next_f64(), "n=2 (1 leader, 1 follower) must consume exactly 2*dim draws");
    }

    // ---- determinism ----

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
            SsaGenerator.generate(&pop, &mut ctx)
        };

        let a = run();
        let b = run();
        for (ga, gb) in a.iter().zip(b.iter()) {
            assert_eq!(floats(ga), floats(gb), "same seed must produce bit-identical offspring");
        }
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
            SsaGenerator.generate(&pop, &mut ctx)
        }));
        assert!(result.is_err(), "gen/ssa must reject pop_size < 2 at runtime as a backstop");
    }

    #[test]
    fn best_index_is_current_population_argmin_ties_lower_index() {
        let pop = Population {
            individuals: (0..5).map(|i| g(vec![i as f64])).collect(),
            fitness: vec![0.0, 5.0, 0.0, 3.0, 1.0],
        };
        assert_eq!(pop.best_index(), Some(0), "ties must resolve to the lower index");
    }
}
