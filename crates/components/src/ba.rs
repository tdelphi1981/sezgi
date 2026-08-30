use sezgi_core::component::*;
use sezgi_core::dist::Distribution;
use sezgi_core::problem::Population;
use sezgi_core::space::{BlockValues, Genotype};
use sezgi_core::state::StateReq;

/// Bat Algorithm (Yang, X.-S. 2010, "A new metaheuristic bat-inspired
/// algorithm", in: *Nature Inspired Cooperative Strategies for Optimization
/// (NICSO 2010)* (eds. J. R. González et al.), Studies in Computational
/// Intelligence, vol. 284, Springer, pp. 65-74) -- a **labeled metaphor
/// preset**: faithful to the primary source's own reference MATLAB
/// implementation's update equations and loop structure, with a pinned
/// deterministic draw order and property tests, but NOT validated against
/// the paper's (or any other publication's) reported benchmark numbers.
/// Equivalence critique: Camacho-Villalón, Dorigo & Stützle (*International
/// Transactions in Operational Research*, six-algorithm critique: grey wolf,
/// moth-flame, whale, firefly, bat, antlion) -- cited here conservatively,
/// as background on why this is "labeled metaphor" rather than a mechanism
/// sezgi treats as novel.
///
/// **This is the wave's second stateful/blackboard algorithm** (after
/// `mfo.rs`'s flame memory): the per-bat velocity `v` is a genuine persisted
/// memory (blackboard `ba/velocity`), exactly matching PSO's own precedent
/// (`pso.rs`'s `pso_velocity`) -- it accumulates additively (`v = v + ...`)
/// across generations with no decay/inertia term at all, and is owned
/// entirely by [`BaGenerator`] itself (reads it, updates it, writes it back,
/// no separate adapter -- same self-owning shape as `pso.rs`'s
/// `PsoGenerator`, not `mfo.rs`'s generator+adapter split).
///
/// ## Provenance (PROVENANCE-FIRST protocol)
///
/// **Primary artifact used (governs):** `bat_algorithm.m`, Xin-She Yang's own
/// MATLAB Central File Exchange submission **#37582** ("Bat algorithm
/// (demo)"), fetched and downloaded in full (the fetch returned the actual
/// submission archive, not a truncated preview -- unlike `sca.rs`/`mfo.rs`/
/// `fa.rs`'s fair-use-limited preview-mirror fetches, this task obtained the
/// complete, exact 135-line source file). The file's own header: "Files of
/// the Matlab programs included in the book: Xin-She Yang, Nature-Inspired
/// Metaheuristic Algorithms, Second Edition, Luniver Press, (2010)" /
/// "Bat-inspired algorithm for continuous optimization (demo), Programmed by
/// Xin-She Yang @Cambridge University 2010" -- explicitly the NICSO-2010-era
/// demo the task brief anticipated, author-linked and citing the NICSO 2010
/// paper (and the Yang & Gandomi 2012 *Engineering Computations* follow-up)
/// directly in its own comment header.
///
/// **THE task brief's flagged VERIFY, settled directly from the source: A
/// and r are FIXED CONSTANTS, not dynamic per-bat state.** The file's own
/// comment, verbatim: *"As this is a demo, here we did not implement the
/// reduction of loudness and increase of emission rates. Interested readers
/// can do some parametric studies and also implementation various changes of
/// A and r etc."* `A=para(3)` and `r=para(4)` are plain scalars (defaults
/// `A=0.5`, `r=0.5`, from the file's own `para=[20 1000 0.5 0.5]` usage
/// example), shared by every bat, and NEVER reassigned anywhere in the main
/// loop. Per the provenance protocol ("if the authoritative artifact fixes
/// them, THAT is the pin ... do not invent dynamics the artifact lacks"),
/// sezgi does **not** implement the paper-prose `A_i(t+1)=alpha*A_i(t)`,
/// `r_i(t+1)=r_i(0)*(1-e^{-gamma*t})` dynamics the plan's sketch described
/// (and which some *other*, non-governing File Exchange submissions by other
/// authors do implement) -- this is a genuine, documented paper-vs-code
/// delta, not an oversight: `BA_A0`/`BA_R0` below are plain pinned module
/// constants (the same idiom as `fa.rs`'s `FA_ALPHA0` etc.), and there is no
/// `ba/loudness`/`ba/pulse_rate` blackboard state, because the verified
/// artifact has nothing dynamic to persist.
///
/// **Extracted equations / loop structure (from `bat_algorithm.m`, quoted
/// verbatim from the fetched source, per-bat per-generation order):**
///
/// ```text
/// Q(i)=Qmin+(Qmin-Qmax)*rand;
/// v(i,:)=v(i,:)+(Sol(i,:)-best)*Q(i);
/// S(i,:)=Sol(i,:)+v(i,:);
/// Sol(i,:)=simplebounds(Sol(i,:),Lb,Ub);
/// if rand>r
///     S(i,:)=best+0.001*randn(1,d);
/// end
/// Fnew=Fun(S(i,:));
/// if (Fnew<=Fitness(i)) & (rand<A) ,
///      Sol(i,:)=S(i,:);
///      Fitness(i)=Fnew;
/// end
/// if Fnew<=fmin,
///      best=S(i,:);
///      fmin=Fnew;
/// end
/// ```
///
/// - **Frequency draw -- VERIFIED, and a genuine artifact-level anomaly,
///   reproduced VERBATIM (not "fixed"), per the same provenance-fidelity
///   discipline `fa.rs`'s in-place/live-vs-frozen quirk and `mfo.rs`'s
///   two-index formula already established for this wave:** the exact line
///   is `Q(i)=Qmin+(Qmin-Qmax)*rand;` -- note `(Qmin-Qmax)`, NOT
///   `(Qmax-Qmin)`. With the file's own `Qmin=0`, `Qmax=2`, this makes
///   `Q(i) = 0 + (0-2)*rand = -2*rand`, i.e. `Q(i) ∈ [-2, 0]` -- the
///   OPPOSITE sign range from the paper's own prose formula (`f_i = f_min +
///   (f_max-f_min)*beta`, intended range `[0, 2]`) and from essentially
///   every independent reimplementation checked while resolving this task's
///   provenance (a Python port explicitly labeled "Implementation of
///   standard Bat Algorithm (Xin She Yang, 2010)" uses
///   `self.fmin + (self.fmax-self.fmin)*random`; a MATLAB Answers thread
///   shows a user who copied `bat_algorithm.m` verbatim hitting this exact
///   line and being told it "should" read `Qmin+(Qmax-Qmin)*rand`). This
///   reads as an unintentional sign transposition in Yang's own
///   book-published demo, not a deliberate design choice -- but per the
///   provenance protocol the verified, author-linked, book-published
///   artifact governs, so sezgi ships it exactly as found: see
///   [`ba_frequency`] and the
///   `frequency_range_is_verified_inverted_not_the_paper_intuitive_range`
///   test, which explicitly demonstrates the `[-2,0]` range (not `[0,2]`).
///   **This composes with the next verified quirk into net-attractive
///   dynamics, not repulsive ones -- see below.**
/// - **Velocity update -- sign CONFIRMED exactly as the task brief flagged:**
///   `v(i,:)=v(i,:)+(Sol(i,:)-best)*Q(i);`, i.e. `v[d] += (X[d] -
///   best[d]) * Q`, using `(X - X_best)` (not the naively-expected
///   `(X_best - X)`) -- see [`ba_velocity_step`]. Read in isolation against
///   the paper's textbook description this also looks inverted (a positive
///   frequency would push bats AWAY from `best`). **But composed with the
///   frequency quirk above (`Q <= 0` always, per the verified formula), the
///   two sign inversions cancel: `(X-best)*Q` with `Q<=0` is
///   `-(X-best)*|Q| = (best-X)*|Q|`, i.e. a genuine pull TOWARD `best`.**
///   This is not sezgi editorializing -- it is a direct algebraic
///   consequence of the two verified formulas composed together, and it is
///   why this "buggy-looking" demo (the file's own comment: "though this
///   demo works very well") is not in fact broken. See the
///   `combined_velocity_and_frequency_signs_produce_net_attraction_to_best`
///   hand-derived test, which shows a bat exactly at the midpoint frequency
///   draw landing EXACTLY on `best` after one step.
/// - `Q(i)` is drawn **ONCE PER BAT** (not per dimension) -- `Q=zeros(n,1)`
///   is a per-bat scalar, and `(Sol(i,:)-best)*Q(i)` applies that single
///   scalar across the whole dimension vector. No per-dimension draws for
///   the frequency/velocity/position step at all.
/// - **Boundary clamp -- VERIFIED clamp-target mismatch, sidestepped
///   structurally by this project's architecture (not a bespoke BA
///   decision):** the source's `Sol(i,:)=simplebounds(Sol(i,:),Lb,Ub);`
///   clamps `Sol(i,:)` (the OLD, pre-move position) in place, NOT `S(i,:)`
///   (the actual new candidate about to be evaluated) -- almost certainly a
///   second typo (the candidate `S(i,:)` is therefore never bounded at all
///   in the literal source, and `Sol(i,:)` is pointlessly re-clamped to
///   itself). sezgi's architecture -- established since `gwo.rs`/`woa.rs`
///   and followed by every metaphor preset in this wave -- keeps boundary
///   handling OUT of the generator entirely: `presets::bat` applies
///   `boundary/clamp` to whatever [`BaGenerator::generate`] returns (i.e.
///   functionally to `S(i,:)`, the actual candidate), which is the more
///   sensible target and a structural side effect of the wave's established
///   generator/boundary separation, not a BA-specific fix.
/// - **Local-walk trigger:** `if rand>r` -- ONE draw per bat (r is the
///   shared, fixed scalar `BA_R0`). See [`ba_local_walk_fires`].
/// - **Local-walk formula -- the task's flagged VERIFY, settled:**
///   `S(i,:)=best+0.001*randn(1,d);` -- a COMPLETE OVERWRITE of the
///   candidate (not additive to the frequency/velocity move), `best[d] +
///   0.001 * randn` per dimension, **Gaussian** (not `epsilon*mean(A)` as
///   the plan's sketch guessed -- there is no `mean(A)` to speak of anyway,
///   since `A` is a shared scalar, not a per-bat array). `d` INDEPENDENT
///   draws, one per dimension, only when the trigger fires. See
///   [`ba_local_walk_step`]. **The velocity `v(i,:)` computed just above is
///   NOT touched by this branch** -- it still commits to the persisted
///   `ba/velocity` state below even when the local walk discards its
///   candidate-position contribution; `bat_algorithm.m` never reassigns `v`
///   inside the `if rand>r` block either.
/// - **Evaluation count -- THE finding that settles this task's design
///   adjudication (see `presets::bat`'s doc and below):** `Fnew=Fun(S(i,:));`
///   runs EXACTLY ONCE per bat per generation, unconditionally, AFTER the
///   local-walk conditional (whether or not it fired). There is no second,
///   separate evaluation anywhere in the loop -- BA's local walk is a
///   candidate-REPLACEMENT mechanism (it overwrites what gets evaluated),
///   not a candidate-ADDITION mechanism (unlike the design the plan's sketch
///   and HHO's later mid-generate-evaluation pattern might suggest). One
///   bat, one candidate, one evaluation -- the SAME honest per-generation
///   evaluation budget as every other generational preset in this wave.
/// - **Acceptance -- VERIFIED against the bat's OWN fitness, and a
///   non-short-circuiting draw order:** `if (Fnew<=Fitness(i)) & (rand<A)`.
///   MATLAB's `&` (unlike `&&`) is NOT short-circuiting: `rand<A`'s `rand`
///   call fires EVERY bat, EVERY generation, regardless of whether
///   `Fnew<=Fitness(i)` holds. The comparison target is `Fitness(i)` -- the
///   bat's OWN current fitness (same-index), NOT `fmin`/`best` -- settling
///   the task brief's flagged VERIFY in favor of the same-index reading. See
///   [`ba_accept`] and [`BaLoudnessGreedy`] below.
/// - **Attractor `best` -- VERIFIED as a persisted best-ever in the source,
///   and a deliberate, precedented delta from that verified mechanism:**
///   `bat_algorithm.m`'s `best`/`fmin` genuinely persist across the WHOLE
///   run (updated unconditionally, if-improved-only, by the SEPARATE
///   `if Fnew<=fmin` check -- independent of whether the loudness gate
///   above accepted the same candidate into `Sol`/`Fitness`). This is
///   functionally the SAME kind of single-scalar best-so-far attractor that
///   `ssa.rs`'s `FoodPosition` delta already analyzed and ruled does NOT
///   warrant its own blackboard exception (unlike `mfo.rs`'s N-sized flame
///   ARCHIVE, which is a genuinely different, bigger mechanism): under this
///   wave's established "current-generation best, not a separately-
///   persisted historical best" convention (parked for attractor-selection
///   values, pinned since Task 1/SCA, re-confirmed by `ssa.rs`'s identical
///   reasoning for `FoodPosition`), `// sezgi simplification:` this module
///   uses the current population's fitness argmin (ties -> lower index, via
///   [`Population::best_index`]) as `best`, computed once per `generate()`
///   call -- not a separately persisted best-ever immune to the loudness
///   gate's rejections. A genuine, deliberately NOT-implemented delta from
///   the verified source, recorded here rather than reproduced, following
///   the SAME precedent `ssa.rs` already established (not a new decision for
///   this task).
///
/// ## Design adjudication: the acceptance-coupled replacement
///
/// The plan's sketch anticipated needing either a new replacer (with RNG +
/// blackboard `A_i` access) or an in-generator-evaluation pattern (the
/// design HHO will need later). **The verified source settles this in favor
/// of the simpler option, and rules out the more complex one:** because
/// `bat_algorithm.m` spends EXACTLY ONE evaluation per bat per generation
/// (see "Evaluation count" above) -- there is no second candidate requiring
/// a mid-`generate()` evaluation, unlike HHO's progressive-rapid-dive
/// acceptance test (planned for Task 9). The engine's standard
/// generate -> boundary -> evaluate -> replace pipeline (`engine.rs`)
/// therefore already produces exactly the `Fnew` value BA's acceptance rule
/// needs, handed to the replacer as `off_fit`. The remaining question is
/// only whether an EXISTING replacer expresses `accept iff off_fit[i] <=
/// pop.fitness[i] AND a fresh random draw < loudness` -- checked and
/// rejected: `replace/one-to-one-greedy` has no RNG gate at all (and uses
/// strict `<`, not BA's `<=`); `replace/metropolis` (`sa.rs`) is the
/// closest existing precedent for an RNG-gated Replacer (`Ctx::rng` +
/// `Ctx::iteration`-driven acceptance), but its `accept` SHORT-CIRCUITS
/// (only draws when the deterministic condition fails) -- exactly the
/// OPPOSITE of BA's verified non-short-circuiting `&`, where the draw is
/// unconditional. Reusing it would silently change the pinned RNG-stream
/// contract (fewer draws whenever an offspring already fails the fitness
/// test). So this task adds a new kind, `replace/bat-loudness-greedy`
/// ([`BaLoudnessGreedy`]), following the `replace/metropolis` PATTERN
/// (params-configurable constant, `Ctx::rng`-gated, no blackboard coupling
/// needed since `A`/`r` are fixed per the verified artifact) but with the
/// UNCONDITIONAL-draw, same-bat-fitness-comparison, `<=` semantics BA's
/// source actually has. This is the T7-M2d-3-style rationale the plan's
/// Global Constraints require for a new replacer: no existing kind fits
/// without silently changing the pinned draw contract or the comparison
/// target, and the new kind needs nothing more than `Ctx::rng` (which the
/// `Replacer::replace` signature already provides), no blackboard state at
/// all (since `A` is fixed, not `A_i` per-bat evolving state the plan's
/// sketch imagined -- see the provenance note above).
///
/// The per-bat/per-dimension math is factored into [`ba_frequency`],
/// [`ba_velocity_step`], [`ba_local_walk_fires`], [`ba_local_walk_step`] and
/// [`ba_accept`] so each can be unit-tested directly without needing to fake
/// `Ctx`/`RngStream`.
///
/// `min_pop = 2`: BA needs a best-so-far individual distinct from `i` for
/// the velocity step to move `X_i` at all (same rationale as `cs.rs`'s and
/// `ssa.rs`'s `min_pop = 2`).
///
/// Boundary handling and the acceptance-coupled replacement are NOT part of
/// this generator: `presets::bat` reuses `boundary/clamp` (see the
/// "Boundary clamp" delta note above) and pairs `gen/ba` with the new
/// `replace/bat-loudness-greedy` (see "Design adjudication" above).
pub const BA_QMIN: f64 = 0.0;
pub const BA_QMAX: f64 = 2.0;
pub const BA_R0: f64 = 0.5;
pub const BA_A0: f64 = 0.5;
pub const BA_LOCAL_WALK_SCALE: f64 = 0.001;

/// Frequency draw, VERIFIED VERBATIM against `bat_algorithm.m`:
/// `qmin + (qmin - qmax) * u`, `u ∈ [0,1)`. NOTE the sign: with `qmin <
/// qmax` (sezgi's pinned `BA_QMIN=0.0`/`BA_QMAX=2.0`) this gives a range of
/// `[2*qmin - qmax, qmin]`, i.e. `[-2, 0]` for the pinned defaults -- the
/// OPPOSITE of the naively-expected `[qmin, qmax]` -- see the module doc's
/// "Frequency draw" delta note for why this is reproduced as found, not
/// "corrected".
pub fn ba_frequency(qmin: f64, qmax: f64, u: f64) -> f64 {
    qmin + (qmin - qmax) * u
}

/// One dimension's velocity update, VERIFIED against `bat_algorithm.m`:
/// `v_new = v_old + (x_d - best_d) * freq` -- the `(X - X_best)` sign the
/// task brief flagged, confirmed exactly. See the module doc's note on how
/// this composes with [`ba_frequency`]'s sign to produce net attraction.
pub fn ba_velocity_step(v_d: f64, x_d: f64, best_d: f64, freq: f64) -> f64 {
    v_d + (x_d - best_d) * freq
}

/// The local-walk trigger, VERIFIED against `bat_algorithm.m`'s `if
/// rand>r`: fires (returns `true`) iff `trigger > r`, strictly. `r=0` means
/// the walk fires for every `trigger` except the (probability-zero)
/// `trigger==0.0` case -- "always local-walk", per the verified semantics.
pub fn ba_local_walk_fires(trigger: f64, r: f64) -> bool {
    trigger > r
}

/// One dimension's local-walk candidate, VERIFIED against `bat_algorithm.m`:
/// `best_d + scale * gaussian` (`scale = BA_LOCAL_WALK_SCALE = 0.001`,
/// `gaussian` a standard-normal draw) -- a COMPLETE replacement of that
/// dimension's candidate value, not an additive term.
pub fn ba_local_walk_step(best_d: f64, gaussian: f64, scale: f64) -> f64 {
    best_d + scale * gaussian
}

/// The acceptance rule, VERIFIED against `bat_algorithm.m`'s `if
/// (Fnew<=Fitness(i)) & (rand<A)`: accept iff the offspring's fitness is
/// `<=` the SAME bat's own current fitness (same-index, `<=` not strict
/// `<`) AND a (separately, unconditionally drawn -- see [`BaLoudnessGreedy`])
/// random draw is `< loudness`. `loudness=0.0` -> never accepts (`loud_draw
/// < 0.0` is never true for `loud_draw ∈ [0,1)`).
pub fn ba_accept(off_fit: f64, own_fit: f64, loud_draw: f64, loudness: f64) -> bool {
    off_fit <= own_fit && loud_draw < loudness
}

fn floats(g: &Genotype) -> &Vec<f64> {
    match &g.blocks[0] { BlockValues::Float(x) => x, _ => unreachable!() }
}

pub struct BaGenerator;

impl BaGenerator {
    pub fn from_params(_p: &serde_json::Value) -> Result<Self, ComponentError> {
        // No tunable parameters (Qmin/Qmax/r0/local-walk scale are the
        // pinned verified defaults/constants -- A/r are fixed per the
        // verified source, see the module doc's provenance note).
        Ok(Self)
    }
}

impl Generator for BaGenerator {
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        assert!(pop.len() >= 2, "gen/ba requires a population of at least 2 (pop_size={})", pop.len());
        let n = pop.len();
        let dim = floats(&pop.individuals[0]).len();

        // Persisted velocity: real blackboard state (bat_algorithm.m's own
        // `v`, `v=zeros(n,d)` at t=0), bootstrapped to zeros the first time
        // this preset's blackboard sees it -- mirrors pso.rs's
        // lazy-init-when-absent pattern. Owned entirely by this generator
        // (read, updated, written back below) -- no separate adapter, same
        // self-owning shape as pso.rs's PsoGenerator.
        if !ctx.bb.contains("ba/velocity") {
            ctx.bb.insert("ba/velocity", vec![vec![0.0_f64; dim]; n]);
        }
        let mut velocity = ctx.bb.get::<Vec<Vec<f64>>>("ba/velocity").unwrap().clone();

        // best: current-population fitness argmin, ties -> lower index --
        // pinned (see module doc's "Attractor `best`" delta note), computed
        // once per generate() call, before any RNG draws.
        let best_idx = pop.best_index().unwrap_or(0);
        let best = floats(&pop.individuals[best_idx]).clone();

        let mut offspring: Vec<Genotype> = Vec::with_capacity(n);
        // `i` indexes THREE independent collections (`pop.individuals`,
        // `velocity` read-then-write, and the `offspring` push) in lockstep
        // -- not a single-array pattern `iter_mut()` could express cleanly.
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let x = floats(&pop.individuals[i]);

            // Pinned draw order: ONE frequency draw per bat (not per
            // dimension) -- see module doc.
            let freq = ba_frequency(BA_QMIN, BA_QMAX, ctx.rng.next_f64());
            let v_i: Vec<f64> = (0..dim)
                .map(|d| ba_velocity_step(velocity[i][d], x[d], best[d], freq))
                .collect();
            let mut s: Vec<f64> = (0..dim).map(|d| x[d] + v_i[d]).collect();

            // Local-walk trigger: ONE draw per bat.
            let trigger = ctx.rng.next_f64();
            if ba_local_walk_fires(trigger, BA_R0) {
                // d draws (Distribution::Gaussian standard-normal call
                // boundary, same pinning idiom as cs.rs's Levy draws),
                // COMPLETE OVERWRITE of `s` -- the velocity move above is
                // discarded for the CANDIDATE, but `v_i` still commits to
                // the persisted state below (see module doc).
                for d in 0..dim {
                    let gaussian = Distribution::Gaussian { mean: 0.0, sigma: 1.0 }.sample(ctx.rng);
                    s[d] = ba_local_walk_step(best[d], gaussian, BA_LOCAL_WALK_SCALE);
                }
            }

            velocity[i] = v_i;
            offspring.push(Genotype { blocks: vec![BlockValues::Float(s)] });
        }

        ctx.bb.insert("ba/velocity", velocity);
        offspring
    }

    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/ba", SupportedBlocks::Only(vec!["float"]))
            .with_provides(vec![StateReq::of::<Vec<Vec<f64>>>("ba/velocity")])
            .with_min_pop(2)
    }
}

/// The new replacer this task adds (see the module doc's "Design
/// adjudication" section for the full rationale): accepts offspring `i`
/// over parent `i` (same-index) iff `off_fit[i] <= pop.fitness[i]` AND a
/// FRESH `ctx.rng.next_f64() < loudness` draw succeeds -- that draw is
/// UNCONDITIONAL, fired for every bat every call, matching
/// `bat_algorithm.m`'s non-short-circuiting `&` (see [`ba_accept`]).
/// `loudness` defaults to `BA_A0 = 0.5` (the verified source's own default,
/// `para(3)`), configurable via `from_params` (unlike the fixed-constant
/// idiom this crate otherwise uses for un-tunable pinned defaults) because,
/// unlike `Qmin`/`Qmax`/`r0`, `A` is a plain acceptance-probability
/// parameter with an obvious, safe tunable range (`[0,1]`) and no
/// draw-order or loop-structure implications -- the same rationale
/// `sa.rs`'s `MetropolisReplacer` already applies to its own `t0`/`alpha`.
pub struct BaLoudnessGreedy {
    pub loudness: f64,
}

impl BaLoudnessGreedy {
    pub fn from_params(p: &serde_json::Value) -> Result<Self, ComponentError> {
        let err = |reason: String| ComponentError::InvalidParams {
            kind: "replace/bat-loudness-greedy".into(),
            reason,
        };
        let loudness = p.get("loudness").and_then(|v| v.as_f64()).unwrap_or(BA_A0);
        if !loudness.is_finite() || !(0.0..=1.0).contains(&loudness) {
            return Err(err(format!("loudness must be in [0, 1], got {}", loudness)));
        }
        Ok(Self { loudness })
    }
}

impl Replacer for BaLoudnessGreedy {
    fn replace(&self, pop: &mut Population, off_i: Vec<Genotype>, off_f: Vec<f64>, ctx: &mut Ctx) {
        for (i, (gi, fi)) in off_i.into_iter().zip(off_f).enumerate() {
            if i >= pop.len() { continue; }
            // Unconditional per-bat draw, matching MATLAB's
            // non-short-circuiting `&` (see module doc) -- ALWAYS drawn,
            // whether or not `fi <= pop.fitness[i]` holds.
            let loud_draw = ctx.rng.next_f64();
            if ba_accept(fi, pop.fitness[i], loud_draw, self.loudness) {
                pop.individuals[i] = gi;
                pop.fitness[i] = fi;
            }
        }
    }

    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("replace/bat-loudness-greedy", SupportedBlocks::All)
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_generator("gen/ba", |p| Ok(Box::new(BaGenerator::from_params(p)?)));
    reg.register_replacer("replace/bat-loudness-greedy", |p| {
        Ok(Box::new(BaLoudnessGreedy::from_params(p)?))
    });
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

    // ---- ba_frequency: the verified inverted-sign range ----

    #[test]
    fn frequency_range_is_verified_inverted_not_the_paper_intuitive_range() {
        // qmin=0, qmax=2 (sezgi's pinned defaults): u=0 -> exactly qmin
        // (0.0); u=1 -> exactly 2*qmin-qmax (-2.0). The naively-expected
        // "paper prose" range would be [0,2]; the VERIFIED formula gives
        // [-2,0] instead -- see module doc.
        assert_eq!(ba_frequency(BA_QMIN, BA_QMAX, 0.0), 0.0);
        assert_eq!(ba_frequency(BA_QMIN, BA_QMAX, 1.0), -2.0);
        let mid = ba_frequency(BA_QMIN, BA_QMAX, 0.5);
        assert_eq!(mid, -1.0);
        assert!((-2.0..=0.0).contains(&mid), "frequency must land in the VERIFIED [-2,0] range, got {mid}");
    }

    // ---- ba_velocity_step: sign, hand-derived ----

    #[test]
    fn velocity_step_sign_hand_derived() {
        // v_d=0, x_d=5, best_d=2, freq=3 -> v_new = 0 + (5-2)*3 = 9.
        assert_eq!(ba_velocity_step(0.0, 5.0, 2.0, 3.0), 9.0);
        // Negative freq (the verified, actually-occurring case): v_d=0,
        // x_d=5, best_d=2, freq=-1 -> v_new = 0 + (5-2)*(-1) = -3.
        assert_eq!(ba_velocity_step(0.0, 5.0, 2.0, -1.0), -3.0);
    }

    #[test]
    fn combined_velocity_and_frequency_signs_produce_net_attraction_to_best() {
        // The module doc's central claim, verified exactly: at the midpoint
        // draw u=0.5, freq = ba_frequency(0,2,0.5) = -1.0 exactly. With
        // v_d=0 (a bat's first-ever move), x_d=5.0, best_d=0.0:
        // v_new = 0 + (5.0-0.0)*(-1.0) = -5.0
        // new_x = x_d + v_new = 5.0 + (-5.0) = 0.0 == best_d EXACTLY.
        // Despite each individual sign (the frequency's [-2,0] range, and
        // the velocity term's (X-best) rather than (best-X) form) looking
        // inverted in isolation, together they pull the bat EXACTLY onto
        // `best` here -- a genuine attraction, not a repulsion.
        let freq = ba_frequency(BA_QMIN, BA_QMAX, 0.5);
        assert_eq!(freq, -1.0);
        let v_new = ba_velocity_step(0.0, 5.0, 0.0, freq);
        let new_x = 5.0 + v_new;
        assert_eq!(new_x, 0.0, "the combined verified formulas must pull the bat onto best at this exact draw");
    }

    // ---- ba_local_walk_fires: r=0 always-fires property ----

    #[test]
    fn local_walk_fires_iff_trigger_strictly_exceeds_r() {
        // r=0: fires for essentially every trigger draw in [0,1) except the
        // probability-zero trigger==0.0 case -- "always local-walk", per
        // the verified semantics.
        assert!(!ba_local_walk_fires(0.0, 0.0), "trigger==r must NOT fire (strict >)");
        assert!(ba_local_walk_fires(1e-12, 0.0), "any positive trigger with r=0 must fire");
        assert!(ba_local_walk_fires(0.999999, 0.0));
        // r=1.0: can never fire for trigger in [0,1) (trigger>1.0 is never true).
        assert!(!ba_local_walk_fires(0.999999, 1.0), "r=1.0 must never fire for trigger < 1.0");
    }

    // ---- ba_accept: A=0 never-accepts property, both-conditions-required ----

    #[test]
    fn accept_with_loudness_zero_never_accepts() {
        // loud_draw < 0.0 is never true for loud_draw in [0,1) -- A=0 means
        // "never accept", regardless of how good the offspring is.
        for loud_draw in [0.0, 0.1, 0.5, 0.999999] {
            assert!(!ba_accept(-1000.0, 1000.0, loud_draw, 0.0),
                "loudness=0.0 must never accept even a hugely improved offspring (loud_draw={loud_draw})");
        }
    }

    #[test]
    fn accept_requires_both_the_fitness_and_loudness_conditions() {
        // Improved (fi <= own) but loudness gate fails.
        assert!(!ba_accept(1.0, 2.0, 0.9, 0.5), "fitness improved but loud_draw >= loudness must reject");
        // Loudness gate passes but not improved.
        assert!(!ba_accept(3.0, 2.0, 0.1, 0.5), "loud_draw < loudness but fitness NOT improved must reject");
        // Both pass.
        assert!(ba_accept(1.0, 2.0, 0.1, 0.5), "both conditions satisfied must accept");
        // Equal fitness (<=, not strict <) with loudness gate passing.
        assert!(ba_accept(2.0, 2.0, 0.1, 0.5), "equal fitness (<=) with loud_draw < loudness must accept");
    }

    // ---- BaGenerator ----

    #[test]
    fn bootstrap_seeds_velocity_to_zeros_when_absent() {
        let n = 4; let dim = 3;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_nd(n, dim);
        let mut evaluator = Evaluator::new(&p, 1000);
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();
        assert!(!bb.contains("ba/velocity"));

        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        BaGenerator.generate(&pop, &mut ctx);

        let velocity = ctx.bb.get::<Vec<Vec<f64>>>("ba/velocity").unwrap();
        assert_eq!(velocity.len(), n);
        // Velocity is bootstrapped to zeros BEFORE this call's own update,
        // then updated in place -- so after ONE generate() call it need not
        // still be zero (freq is nonzero with overwhelming probability);
        // just check shape here (the round-trip test below checks the
        // exact accumulated values via a hand-traced, twin-stream replay).
        for row in velocity { assert_eq!(row.len(), dim); }
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
            BaGenerator.generate(&pop, &mut ctx)
        };

        let a = run();
        let b = run();
        for (ga, gb) in a.iter().zip(b.iter()) {
            assert_eq!(floats(ga), floats(gb), "same seed must produce bit-identical offspring");
        }
    }

    #[test]
    fn draw_count_matches_hand_traced_conditional_local_walk() {
        // Twin-stream RAW REPLAY (the fa.rs/cs.rs technique): replay the
        // SAME per-bat draw sequence (freq, trigger, conditionally d
        // Distribution::Gaussian samples) on an independent clone of the
        // pre-call stream, letting the SAME threshold decide the SAME
        // branches -- both streams must therefore end up at the identical
        // position, regardless of exactly which bats fired the local walk.
        let n = 6; let dim = 3;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_nd(n, dim);

        let mut rng = RngStream::from_master(7, &[]);
        let rng_before = rng.clone();
        let mut evaluator = Evaluator::new(&p, 1000);
        let mut bb = Blackboard::new();
        {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
            let off = BaGenerator.generate(&pop, &mut ctx);
            assert_eq!(off.len(), n);
        }

        let mut twin = rng_before;
        for _ in 0..n {
            let _freq = twin.next_f64();
            let trigger = twin.next_f64();
            if ba_local_walk_fires(trigger, BA_R0) {
                for _ in 0..dim {
                    Distribution::Gaussian { mean: 0.0, sigma: 1.0 }.sample(&mut twin);
                }
            }
        }
        assert_eq!(rng.next_f64(), twin.next_f64(),
            "gen/ba must consume exactly (freq + trigger) per bat, plus dim Gaussian draws per bat that fires the local walk");
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
            BaGenerator.generate(&pop, &mut ctx)
        }));
        assert!(result.is_err(), "gen/ba must reject pop_size < 2 at runtime as a backstop");
    }

    // ---- State round-trip across two generations (hand-traced, MFO-flame-test style) ----

    #[test]
    fn velocity_state_round_trips_and_accumulates_across_two_generations() {
        let dim = 1;
        let p = SphereShifted::new(vec![0.0; dim], -1000.0, 1000.0);
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 100_000);
        let mut rng = RngStream::from_master(3, &[]);
        let mut bb = Blackboard::new();

        // Init population: bat 0 (fitness 10.0, worse) at x=0.0, bat 1
        // (fitness 1.0, better) at x=5.0 -- so best = bat 1's position (5.0).
        let mut pop = Population {
            individuals: vec![g(vec![0.0]), g(vec![5.0])],
            fitness: vec![10.0, 1.0],
        };

        // Generation 0: hand-trace bat 0's exact draws against a twin
        // stream (bootstraps velocity=[[0.0],[0.0]]).
        let rng_before_gen0 = rng.clone();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let off0 = BaGenerator.generate(&pop, &mut ctx);
        let velocity_after_gen0 = ctx.bb.get::<Vec<Vec<f64>>>("ba/velocity").unwrap().clone();

        let mut twin = rng_before_gen0;
        // Bat 0: x=0.0, best=5.0 (bat 1). freq0, v0 = 0 + (0.0-5.0)*freq0.
        let freq0_bat0 = ba_frequency(BA_QMIN, BA_QMAX, twin.next_f64());
        let v0_bat0 = ba_velocity_step(0.0, 0.0, 5.0, freq0_bat0);
        let mut s0_bat0 = 0.0 + v0_bat0;
        let trig0_bat0 = twin.next_f64();
        if ba_local_walk_fires(trig0_bat0, BA_R0) {
            let gaussian = Distribution::Gaussian { mean: 0.0, sigma: 1.0 }.sample(&mut twin);
            s0_bat0 = ba_local_walk_step(5.0, gaussian, BA_LOCAL_WALK_SCALE);
        }
        // Bat 1: x=5.0, best=5.0 (itself). freq1, v1 = 0 + (5.0-5.0)*freq1 = 0.
        let freq0_bat1 = ba_frequency(BA_QMIN, BA_QMAX, twin.next_f64());
        let v0_bat1 = ba_velocity_step(0.0, 5.0, 5.0, freq0_bat1);
        assert_eq!(v0_bat1, 0.0, "the current best's own velocity contribution is exactly zero (x==best)");
        let mut s0_bat1 = 5.0 + v0_bat1;
        let trig0_bat1 = twin.next_f64();
        if ba_local_walk_fires(trig0_bat1, BA_R0) {
            let gaussian = Distribution::Gaussian { mean: 0.0, sigma: 1.0 }.sample(&mut twin);
            s0_bat1 = ba_local_walk_step(5.0, gaussian, BA_LOCAL_WALK_SCALE);
        }

        assert_eq!(floats(&off0[0])[0], s0_bat0, "bat 0's generation-0 candidate must match the hand-derived trace");
        assert_eq!(floats(&off0[1])[0], s0_bat1, "bat 1's generation-0 candidate must match the hand-derived trace");
        assert_eq!(velocity_after_gen0, vec![vec![v0_bat0], vec![v0_bat1]],
            "velocity must persist the EXACT per-bat values computed this generation, regardless of local-walk overwrite");

        // replace/bat-loudness-greedy (hand-applied): pretend both offspring
        // are accepted, with made-up fresh fitness values, to drive the NEXT
        // generation's `best` selection deterministically.
        pop.individuals = off0;
        pop.fitness = vec![7.0, 0.5]; // bat 1 (index 1) is still best

        // Generation 1: velocity must be READ BACK (not re-bootstrapped) and
        // ACCUMULATE on top of generation 0's values -- hand-trace again.
        let rng_before_gen1 = rng.clone();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 1 };
        let off1 = BaGenerator.generate(&pop, &mut ctx);
        let velocity_after_gen1 = ctx.bb.get::<Vec<Vec<f64>>>("ba/velocity").unwrap().clone();

        let best1 = floats(&pop.individuals[1])[0]; // still bat 1
        let x1_bat0 = floats(&pop.individuals[0])[0];
        let x1_bat1 = floats(&pop.individuals[1])[0];

        let mut twin1 = rng_before_gen1;
        let freq1_bat0 = ba_frequency(BA_QMIN, BA_QMAX, twin1.next_f64());
        // MUST accumulate on top of v0_bat0, not reset to 0.
        let v1_bat0 = ba_velocity_step(v0_bat0, x1_bat0, best1, freq1_bat0);
        let mut s1_bat0 = x1_bat0 + v1_bat0;
        let trig1_bat0 = twin1.next_f64();
        if ba_local_walk_fires(trig1_bat0, BA_R0) {
            let gaussian = Distribution::Gaussian { mean: 0.0, sigma: 1.0 }.sample(&mut twin1);
            s1_bat0 = ba_local_walk_step(best1, gaussian, BA_LOCAL_WALK_SCALE);
        }
        let freq1_bat1 = ba_frequency(BA_QMIN, BA_QMAX, twin1.next_f64());
        let v1_bat1 = ba_velocity_step(v0_bat1, x1_bat1, best1, freq1_bat1);
        let mut s1_bat1 = x1_bat1 + v1_bat1;
        let trig1_bat1 = twin1.next_f64();
        if ba_local_walk_fires(trig1_bat1, BA_R0) {
            let gaussian = Distribution::Gaussian { mean: 0.0, sigma: 1.0 }.sample(&mut twin1);
            s1_bat1 = ba_local_walk_step(best1, gaussian, BA_LOCAL_WALK_SCALE);
        }

        assert_eq!(floats(&off1[0])[0], s1_bat0, "bat 0's generation-1 candidate must match the hand-derived accumulated trace");
        assert_eq!(floats(&off1[1])[0], s1_bat1, "bat 1's generation-1 candidate must match the hand-derived accumulated trace");
        assert_eq!(velocity_after_gen1, vec![vec![v1_bat0], vec![v1_bat1]],
            "velocity after generation 1 must be the ACCUMULATED (not reset) per-bat values");
        assert_ne!(velocity_after_gen1[0][0], v0_bat0,
            "sanity: velocity must actually have changed across the two generations (not silently frozen)");
    }

    // ---- BaLoudnessGreedy ----

    #[test]
    fn replacer_default_loudness_is_ba_a0() {
        let r = BaLoudnessGreedy::from_params(&serde_json::json!({})).unwrap();
        assert_eq!(r.loudness, BA_A0);
    }

    #[test]
    fn replacer_rejects_loudness_out_of_range() {
        assert!(BaLoudnessGreedy::from_params(&serde_json::json!({"loudness": -0.1})).is_err());
        assert!(BaLoudnessGreedy::from_params(&serde_json::json!({"loudness": 1.1})).is_err());
        assert!(BaLoudnessGreedy::from_params(&serde_json::json!({"loudness": 0.0})).is_ok());
        assert!(BaLoudnessGreedy::from_params(&serde_json::json!({"loudness": 1.0})).is_ok());
    }

    #[test]
    fn replacer_loudness_zero_never_accepts_even_strict_improvement() {
        let p = SphereShifted::new(vec![0.0; 1], -5.0, 5.0);
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 1000);
        let mut rng = RngStream::from_master(5, &[]);
        let mut bb = Blackboard::new();
        let mut pop = Population { individuals: vec![g(vec![1.0]), g(vec![2.0])], fitness: vec![10.0, 20.0] };
        let before = pop.clone();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };

        let replacer = BaLoudnessGreedy { loudness: 0.0 };
        // Hugely improved offspring for both slots.
        replacer.replace(&mut pop, vec![g(vec![9.0]), g(vec![8.0])], vec![0.0, 0.0], &mut ctx);

        assert_eq!(pop.individuals, before.individuals, "loudness=0.0 must leave the population entirely unchanged");
        assert_eq!(pop.fitness, before.fitness);
    }

    #[test]
    fn replacer_accepts_improved_offspring_when_loudness_one() {
        let p = SphereShifted::new(vec![0.0; 1], -5.0, 5.0);
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 1000);
        let mut rng = RngStream::from_master(5, &[]);
        let mut bb = Blackboard::new();
        let mut pop = Population { individuals: vec![g(vec![1.0]), g(vec![2.0])], fitness: vec![10.0, 20.0] };
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };

        // loudness=1.0: loud_draw < 1.0 is true for every draw in [0,1) --
        // acceptance collapses to the plain fitness test.
        let replacer = BaLoudnessGreedy { loudness: 1.0 };
        replacer.replace(&mut pop, vec![g(vec![9.0]), g(vec![8.0])], vec![5.0, 25.0], &mut ctx);

        assert_eq!(pop.fitness, vec![5.0, 20.0], "slot 0 improved (5.0<10.0) and must be accepted; slot 1 did not (25.0>20.0) and must be rejected");
        assert_eq!(pop.individuals[0], g(vec![9.0]));
        assert_eq!(pop.individuals[1], g(vec![2.0]), "unchanged slot must keep its original parent");
    }

    #[test]
    fn replacer_draws_unconditionally_once_per_bat_regardless_of_acceptance() {
        // Twin-stream: EXACTLY n draws, one per bat, whether or not the
        // fitness condition held -- matching MATLAB's non-short-circuiting
        // `&` (see module doc).
        let n = 5;
        let p = SphereShifted::new(vec![0.0; 1], -5.0, 5.0);
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 1000);
        let mut rng = RngStream::from_master(11, &[]);
        let rng_before = rng.clone();
        let mut bb = Blackboard::new();
        let mut pop = Population {
            individuals: (0..n).map(|i| g(vec![i as f64])).collect(),
            fitness: (0..n).map(|i| i as f64).collect(),
        };
        // Offspring fitness alternates improved/not-improved, to exercise both branches.
        let off_i: Vec<Genotype> = (0..n).map(|i| g(vec![100.0 + i as f64])).collect();
        let off_f: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { -1.0 } else { 1000.0 }).collect();
        {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
            let replacer = BaLoudnessGreedy { loudness: BA_A0 };
            replacer.replace(&mut pop, off_i, off_f, &mut ctx);
        }

        let mut twin = rng_before;
        for _ in 0..n { twin.next_f64(); }
        assert_eq!(rng.next_f64(), twin.next_f64(), "replace/bat-loudness-greedy must consume exactly one draw per bat, unconditionally");
    }
}
