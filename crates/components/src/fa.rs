use sezgi_core::component::*;
use sezgi_core::problem::Population;
use sezgi_core::space::{Block, BlockValues, Genotype};

/// Firefly Algorithm (Yang, X.-S. -- originally *Nature-Inspired
/// Metaheuristic Algorithms*, 1st ed., Luniver Press, 2008; the artifact
/// fetched and verified here is the 2nd-edition (2010) release of the same
/// book's companion MATLAB code) -- a **labeled metaphor preset**: faithful
/// to the primary source's own reference MATLAB implementation's update
/// equations and loop structure, with a pinned deterministic draw order and
/// property tests, but NOT validated against the paper's (or any other
/// publication's) reported benchmark numbers. Equivalence critique:
/// Camacho-Villalón, Dorigo & Stützle (*International Transactions in
/// Operational Research*, six-algorithm critique: grey wolf, moth-flame,
/// whale, firefly, bat, antlion) -- cited here conservatively, as background
/// on why this is "labeled metaphor" rather than a mechanism sezgi treats as
/// novel.
///
/// ## Provenance (PROVENANCE-FIRST protocol)
///
/// **Primary artifact used (governs):** `fa_ndim.m` / its `ffa_mincon`
/// subfunction, from Xin-She Yang's own MATLAB Central File Exchange
/// submission **#29693** ("Firefly Algorithm", author Xin-She Yang) --
/// "For unconstrained functions in higher dimensions, please use
/// `fa_ndim.m`" per the submission's own description, matching this task's
/// N-dimensional requirement (as opposed to the same submission's
/// `firefly_simple.m`, a 2D-only toy demo -- see the delta note below).
/// Accessed via the MathWorks File Exchange preview mirror
/// (`mlc-downloads/downloads/submissions/29693/versions/3/previews/
/// fa_ndim.m/index.html`); the file's own header states it is "Files of the
/// Matlab programs included in the book: Xin-She Yang, Nature-Inspired
/// Metaheuristic Algorithms, Second Edition, Luniver Press, (2010)". The
/// full source (driver `fa_ndim`, core mover `ffa_mincon`, its movement
/// subfunction `ffa_move`, and helpers `alpha_new`/`init_ffa`/`findlimits`)
/// was fetched and quoted verbatim across several targeted fetches (the
/// mirror's per-request output is truncated, the same constraint noted by
/// Task 1/SCA's and Task 3/MFO's provenance notes).
///
/// **Secondary/fallback source consulted, per protocol, WITH a caveat found
/// and recorded (not used to override the primary):** mealpy's `FFA.py`
/// (`OriginalFFA`) turned out, on inspection, to implement a *different*
/// published algorithm -- Łukasik & Żak (2009), "Firefly algorithm for
/// continuous constrained optimization tasks" -- not a reimplementation of
/// Yang's own `fa_ndim.m`/`ffa_move.m` mechanism (its hyperparameter set is
/// materially different: `gamma` default `0.001`, `beta_base` default `2`,
/// `alpha_damp` a fixed `0.99` per-generation multiplicative decay, plus a
/// `delta` mutation-step-size parameter that has no analogue in
/// `ffa_move.m` at all). This is exactly the kind of naming-collision
/// caveat the protocol anticipates (compare Task 4/SSA's `SSA.py`-vs-
/// `SSO.py` collision): mealpy's `OriginalFFA` is NOT independent
/// corroboration of `fa_ndim.m`'s loop structure, so it is not relied on
/// for any claim below -- `fa_ndim.m`/`ffa_move.m` (Yang's own,
/// author-linked, book-published reference code) is the sole governing
/// artifact.
///
/// **Also fetched for contrast (not governing):** `firefly_simple.m`, the
/// same submission's 2D-only demo. It differs from `fa_ndim.m` in three
/// ways worth recording so reviewers don't mistake one for the other: (1)
/// its `beta` formula has **no `betamin` floor** (`beta=beta0*exp(-gamma*
/// r.^2)`, not `fa_ndim.m`'s `(beta0-betamin)*exp(-gamma*r.^2)+betamin`);
/// (2) its `alpha` decay is a fixed per-generation multiplier
/// (`alpha=alpha*delta` with a hardcoded `delta=0.97`), not `fa_ndim.m`'s
/// closed-form-derived decay; (3) it is written for a **maximization**
/// convention (`if Lightn(i)<Lighto(j)`) vs `fa_ndim.m`'s **minimization**
/// convention (`if Lightn(i)>Lighto(j)`, since `Lightn`/`Lighto` there hold
/// the raw objective value, not an inverted brightness). `fa_ndim.m`
/// governs throughout (it also matches sezgi's minimization convention
/// directly, with no sign flip needed).
///
/// **Extracted equations / loop structure (from `fa_ndim.m`/`ffa_mincon`/
/// `ffa_move`, quoted verbatim from the fetched source):**
///
/// ```text
/// % fa_ndim's driver parameters:
/// para=[20 500 0.5 0.2 1];   % [n MaxGeneration alpha betamin gamma]
///
/// % ffa_mincon, once per generation (k=1:MaxGeneration):
/// alpha=alpha_new(alpha,MaxGeneration);      % decay BEFORE this generation's move
/// for i=1:n, zn(i)=fhandle(ns(i,:)); Lightn(i)=zn(i); end   % fresh evaluation
/// [Lightn,Index]=sort(zn);                   % sort ascending (ties: MATLAB sort is stable)
/// ns_tmp=ns; for i=1:n, ns(i,:)=ns_tmp(Index(i),:); end     % reorder positions by rank
/// nso=ns; Lighto=Lightn;                     % FROZEN snapshot of the just-sorted state
/// [ns]=ffa_move(n,d,ns,Lightn,nso,Lighto,nbest,Lightbest,alpha,betamin,gamma,Lb,Ub);
///
/// % ffa_move (the actual double loop):
/// scale=abs(Ub-Lb);
/// for i=1:n,
///    for j=1:n,
///       r=sqrt(sum((ns(i,:)-ns(j,:)).^2));       % LIVE ns on BOTH sides
///       if Lightn(i)>Lighto(j),                  % i.e. Lighto(i) > Lighto(j) -- see note below
///          beta0=1; beta=(beta0-betamin)*exp(-gamma*r.^2)+betamin;
///          tmpf=alpha.*(rand(1,d)-0.5).*scale;
///          ns(i,:)=ns(i,:).*(1-beta)+nso(j,:).*beta+tmpf;   % self LIVE, target FROZEN
///       end
///    end
/// end
/// [ns]=findlimits(n,ns,Lb,Ub);
///
/// % alpha_new, once per generation:
/// delta=1-(10^(-4)/0.9)^(1/NGen); alpha=(1-delta)*alpha;
/// ```
///
/// - **Defaults -- three genuine deltas from the plan's sketch, VERIFIED:**
///   `alpha=0.5` (the sketch guessed `0.2`, which is actually `fa_ndim.m`'s
///   *`betamin`*, a different parameter entirely), `betamin=0.2`, `gamma=1`,
///   `beta0=1` (hardcoded inside `ffa_move`, not part of the tunable
///   parameter vector at all).
///   `n=20`, `MaxGeneration=500` are `fa_ndim.m`'s own demo pop/budget
///   choices, irrelevant here -- sezgi's `presets::firefly` uses this wave's
///   directed `pop_size=25` instead.
/// - **The attractiveness formula -- the task's flagged VERIFY, and the
///   biggest delta from the plan's sketch:** `beta = (beta0 - betamin)·
///   e^{−γ·r²} + betamin`, the **floored** variant, NOT the plan sketch's
///   plain `β0·e^{−γr²}` -- see [`fa_beta`]. This means `γ→∞` does **not**
///   drive attraction to zero; it drives `beta` down to the `betamin=0.2`
///   floor, a residual, non-vanishing pull -- a real behavioral consequence
///   of the verified formula, exercised by this module's `gamma_to_infinity_
///   limit_is_betamin_floor_not_zero` test (which explicitly demonstrates
///   this diverges from the plan sketch's naive expectation).
/// - **`alpha` DOES decay, once per generation (confirmed, per the sketch's
///   flagged VERIFY) -- via `alpha_new`'s closed-form-derived geometric
///   decay, not a literal per-iteration counter.** `alpha_new` is called
///   ONCE per outer generation loop (`for k=1:MaxGeneration`), computing a
///   FIXED per-generation shrink ratio `(1-delta) = (10^-4/0.9)^{1/NGen}`
///   from the TOTAL generation count `NGen=MaxGeneration`, then multiplying
///   the running `alpha` by that ratio -- so after `k` generations,
///   `alpha_k = alpha_0 · (10^-4/0.9)^{k/NGen}`. Following this wave's
///   established convention of substituting a literal iteration/generation
///   counter with `progress = ctx.eval.used()/ctx.eval.budget()` (the SAME
///   substitution already reviewed and accepted for `sca.rs`'s `r1`,
///   `gwo.rs`/`woa.rs`'s `a`, `mfo.rs`'s `a`/flame-count schedule, and
///   `ssa.rs`'s `c1` -- NOT a new decision for this task), `k/NGen ==
///   progress` in the continuous limit (since `NGen` generations each
///   consume `pop_size` evaluations, `k/NGen == (k·pop_size)/(NGen·
///   pop_size) == used/budget`), giving the closed form `alpha(progress) =
///   alpha_0 · (10^-4/0.9)^progress` -- see [`fa_alpha`]. This is a
///   deliberate, documented departure from `alpha_new`'s literal per-
///   generation multiplicative *recursion* (which would require this
///   generator to carry running state across calls, contradicting this
///   task's stateless-generator design direction) in favor of reproducing
///   the SAME decay *shape* (a geometric decay from `alpha_0` down to
///   `alpha_0 · 10^-4/0.9` as the run progresses) via a pure function of
///   `progress`, exactly the wave's established schedule-substitution
///   idiom -- not a novel invention for this task, and not tagged as a
///   `// sezgi simplification:` for that reason (same status as `ssa.rs`'s
///   analogous note about its own `l`-counter substitution).
/// - **The double-loop order -- the task's flagged VERIFY, and the module's
///   central pin.** `i` outer ascending, `j` inner ascending, **over the
///   population SORTED BY FITNESS ASCENDING** (rank 0 = best/brightest).
///   Sorting first is not optional bookkeeping: it is REQUIRED to
///   faithfully reproduce the verified in-place semantics below, because
///   `fa_ndim.m`'s array-index processing order (`i=1..n`) is made to
///   coincide with fitness rank ONLY by the pre-sort step -- see
///   [`FaGenerator::generate`]'s sort-first implementation. Since sezgi's
///   incoming `Population` is not guaranteed sorted, this module sorts a
///   WORKING COPY (stable sort by fitness ascending, ties -> original
///   index, matching MATLAB `sort`'s stability) before running the double
///   loop, and returns the offspring in that SAME rank order (there is no
///   "restore to original input order" step in `fa_ndim.m` either -- the
///   population is conceptually re-sorted from scratch every generation
///   regardless of what order it arrives in, so no canonical "original
///   order" is semantically meaningful to restore).
/// - **The move is a HYBRID of in-place (immediate) and frozen (batch)
///   semantics -- VERIFIED, and the wave's subtlest trap found so far:**
///   - The condition `Lightn(i)>Lighto(j)` compares FROZEN, per-generation
///     fixed fitness values on both sides (`Lightn`/`Lighto` are never
///     reassigned inside `ffa_move` -- only `ns`, the position array,
///     mutates) -- so, given the pre-sort, this reduces to "j is
///     strictly brighter (lower fitness) than i", a pure pairwise fitness
///     comparison independent of population order -- see
///     [`FaGenerator::generate`]'s `lighto[i] > lighto[j]` check. Ties
///     (including `i==j`, always a tie against itself) never fire (strict
///     `>`, matching the sketch's flagged "does the i==j / equal-brightness
///     case move" question: **no**). The single brightest individual
///     (rank 0) can never satisfy `lighto[0] > lighto[j]` for any `j` (it
///     is the global minimum by construction of the sort) -- it **never
///     moves**, and `ffa_move` has no separate random-walk-for-the-best
///     mechanism (some FOLKLORE FA variants add one; `fa_ndim.m` does not)
///     -- answering the sketch's flagged "does the BEST firefly do a
///     random walk or stay": **it stays**.
///   - The MOVE TARGET `nso(j,:)` is the FROZEN, pre-generation-movement
///     snapshot, fixed for the entire double loop -- a firefly's move is
///     always pulled toward where its brighter neighbor STARTED this
///     generation, never toward where that neighbor has moved TO so far
///     this generation.
///   - The MOVE'S OWN BASE `ns(i,:)` (the multiplicative `(1-beta)` term)
///     and the DISTANCE computation `r=sqrt(sum((ns(i,:)-ns(j,:)).^2))`
///     BOTH read the LIVE, currently-mutating `ns` array -- NOT the frozen
///     `nso` snapshot. Concretely this means: (a) if a firefly `i` has
///     MULTIPLE brighter attractors `j`, its position accumulates
///     IN-PLACE across the inner `j` loop -- the second pull's base is the
///     position AFTER the first pull, not the original pre-generation
///     position (an in-place, "immediate update" self-accumulation); AND
///     (b) the DISTANCE `r` used to compute `beta` for a pair `(i,j)` can
///     reflect `j`'s ALREADY-MOVED position from an EARLIER outer-loop
///     pass (when `j` itself was processed as the outer index, i.e.
///     whenever `j`'s own rank is better than some other rank that pulled
///     it, which happens before `i`'s later, worse-ranked pass reads it) --
///     even though the actual position BLENDED IN for that same `(i,j)`
///     pair still uses `j`'s FROZEN `nso(j,:)`, not its live position. This
///     is a genuine mismatch inside `fa_ndim.m`'s own reference code (the
///     distance driving `beta` and the position blended into the update
///     are computed from two DIFFERENT snapshots of `j`) -- reproduced
///     here VERBATIM, not "fixed", per the provenance protocol. See
///     [`FaGenerator::generate`]'s single `ns: Vec<Vec<f64>>` working copy
///     (mutated in place, read for both `r2` and the `(1-beta)` self term)
///     alongside the separate, immutable `nso: Vec<Vec<f64>>` (the frozen
///     rank-order snapshot, read only for the additive target term), and
///     the `in_place_self_accumulation_and_live_distance_vs_frozen_target_
///     hand_traced_3_firefly` test, which hand-derives a 3-firefly, 1-D
///     fixture specifically constructed so the live-vs-frozen distinction
///     changes the numeric result, and asserts the verified (live-distance,
///     frozen-target, self-accumulating) semantics bit-exact against the
///     (wrong) fully-frozen alternative.
/// - **Random draws -- CONDITIONAL on the brightness comparison, per
///   dimension, redrawn on EVERY qualifying `(i,j)` hit (not once per
///   `i`):** `tmpf=alpha.*(rand(1,d)-0.5).*scale` is drawn fresh, inside the
///   `if` branch, for every `(i,j)` pair where `j` is strictly brighter than
///   `i` -- a firefly with `k` brighter attractors draws `k·dim` random
///   numbers this generation, not `dim`. Zero draws occur for any `(i,j)`
///   pair that does not satisfy the condition (including all pairs
///   involving the rank-0 best individual as `i`). See [`fa_step_term`] and
///   the `draw_count_matches_hand_traced_firing_pairs` /
///   `draw_count_is_order_independent_of_input_array_order` twin-stream
///   tests.
/// - `r` is computed via `sqrt(sum(diff.^2))` then immediately squared again
///   inside `exp(-gamma*r.^2)` -- a redundant round-trip through `sqrt` in
///   the MATLAB source (almost certainly a copy-paste artifact from a
///   version that used `r` directly elsewhere). This module computes `r2`
///   as the raw sum-of-squared-differences directly (skipping the redundant
///   `sqrt`-then-`.^2`), which is the SAME formula, not a behavioral delta
///   -- MATLAB's `sqrt(x)^2` is not bit-exact-equal to `x` in general due to
///   floating-point rounding, but sezgi does not claim MATLAB bit-parity
///   anywhere in this wave (only internal Rust same-seed determinism, tested
///   separately) -- so this is a cleaner restatement of the identical pinned
///   formula, not tagged `// sezgi simplification:` (compare `mfo.rs`'s `b=1`
///   constant-hoisting note, the same category of non-semantic cleanup).
/// - **Boundary handling:** `ffa_move`'s own trailing call to `findlimits`
///   clamps to `[Lb,Ub]` -- functionally sezgi's standard `boundary/clamp`
///   (same as `gwo`/`woa`/`sca`/`jaya`/`mfo`/`ssa`), applied by the preset,
///   not this generator.
/// - **Replacement:** `fa_ndim.m`'s outer generation loop unconditionally
///   overwrites `ns` with `ffa_move`'s result every generation (no per-
///   firefly greedy fitness comparison) -- `replace/generational` (reused
///   as-is) is the correct pin.
///
/// ## Cost note
///
/// `O(n²·dim)` per `generate()` call: the double loop visits every `(i,j)`
/// pair (computing `r2`, an `O(dim)` reduction, unconditionally for each,
/// exactly mirroring `ffa_move.m`'s own unconditional `r=sqrt(...)` before
/// its `if` check) regardless of how many pairs actually satisfy the
/// brightness condition. This is markedly more expensive per generation
/// than this wave's other `O(n·dim)` presets (SCA/JAYA/MFO/SSA) -- an
/// intrinsic property of FA's pairwise-comparison mechanism, not a sezgi
/// implementation inefficiency.
///
/// The per-pair math is factored into [`fa_beta`] (the attractiveness
/// formula) and [`fa_step_term`] (the per-dimension random-step term), and
/// the per-generation schedule into [`fa_alpha`], so each can be
/// unit-tested directly without needing to fake `Ctx`/`RngStream`.
///
/// `min_pop = 2`: with `pop_size == 1` the double loop degenerates to the
/// single `(0,0)` self-pair, which never satisfies the strict brightness
/// inequality (a tie against itself) -- harmless but not a meaningful
/// firefly interaction; `min_pop = 2` matches this wave's established floor
/// for pairwise-comparison presets (SSA/MFO/SCA/JAYA), enforced via
/// `AlgorithmSpec::validate`.
///
/// Boundary handling and replacement are NOT part of this generator:
/// `presets::firefly` reuses `boundary/clamp` and `replace/generational`
/// (both reused as-is, see above).
pub fn fa_beta(beta0: f64, betamin: f64, gamma: f64, r2: f64) -> f64 {
    (beta0 - betamin) * (-gamma * r2).exp() + betamin
}

/// Closed-form progress substitute for `alpha_new`'s per-generation
/// geometric decay (see the module doc's "alpha DOES decay" note):
/// `alpha(progress) = alpha0 · (10^-4/0.9)^progress`. At `progress=0`:
/// `alpha0` exactly. At `progress=1`: `alpha0 · (10^-4/0.9)`, matching
/// `alpha_new`'s own intended end-of-run shrink ratio.
pub fn fa_alpha(alpha0: f64, progress: f64) -> f64 {
    alpha0 * (1e-4_f64 / 0.9_f64).powf(progress)
}

/// One dimension's random-step term: `alpha·(u−0.5)·scale_d`, `u ∈ [0,1)`
/// freshly drawn, `scale_d = hi_d − lo_d` (the domain width, matching
/// `ffa_move.m`'s `scale=abs(Ub-Lb)`).
pub fn fa_step_term(alpha: f64, u: f64, scale_d: f64) -> f64 {
    alpha * (u - 0.5) * scale_d
}

/// Pinned defaults, verified against `fa_ndim.m`'s `para=[20 500 0.5 0.2 1]`
/// and `ffa_move.m`'s hardcoded `beta0=1`.
pub const FA_ALPHA0: f64 = 0.5;
pub const FA_BETA0: f64 = 1.0;
pub const FA_BETAMIN: f64 = 0.2;
pub const FA_GAMMA: f64 = 1.0;

fn floats(g: &Genotype) -> &Vec<f64> {
    match &g.blocks[0] { BlockValues::Float(x) => x, _ => unreachable!() }
}

pub struct FaGenerator;

impl FaGenerator {
    pub fn from_params(_p: &serde_json::Value) -> Result<Self, ComponentError> {
        // No tunable parameters (alpha0/beta0/betamin/gamma are the pinned
        // verified defaults; alpha's decay is fully determined by ctx.eval
        // progress).
        Ok(Self)
    }
}

impl Generator for FaGenerator {
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        assert!(pop.len() >= 2, "gen/fa requires a population of at least 2 (pop_size={})", pop.len());
        let n = pop.len();
        let dim = floats(&pop.individuals[0]).len();

        let (lo, hi): (Vec<f64>, Vec<f64>) = match &ctx.space.blocks()[0] {
            Block::Float { lo, hi, n: bn } => {
                debug_assert_eq!(*bn, dim);
                (vec![*lo; dim], vec![*hi; dim])
            }
            _ => unreachable!("gen/fa only supports float blocks"),
        };
        let scale: Vec<f64> = (0..dim).map(|d| hi[d] - lo[d]).collect();

        // Rank order: stable sort by fitness ascending, ties -> original
        // index -- mirrors ffa_mincon's `[Lightn,Index]=sort(zn)` pre-sort.
        // REQUIRED for the verified in-place loop semantics (see module
        // doc's "double-loop order" note), not merely cosmetic.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| pop.fitness[a].total_cmp(&pop.fitness[b]));

        let lighto: Vec<f64> = order.iter().map(|&i| pop.fitness[i]).collect();
        // nso: FROZEN rank-ordered snapshot (read-only for the rest of this
        // call -- the additive move target). ns: the LIVE working copy,
        // mutated in place across the double loop (read for both the
        // distance r2 and the multiplicative self term).
        let nso: Vec<Vec<f64>> = order.iter().map(|&i| floats(&pop.individuals[i]).clone()).collect();
        let mut ns: Vec<Vec<f64>> = nso.clone();

        let used = ctx.eval.used() as f64;
        let budget = ctx.eval.budget() as f64;
        let progress = if budget > 0.0 { (used / budget).clamp(0.0, 1.0) } else { 0.0 };
        let alpha = fa_alpha(FA_ALPHA0, progress);

        for i in 0..n {
            for j in 0..n {
                // r2 from the LIVE working copy on BOTH sides -- verified
                // quirk (see module doc): ffa_move.m's own
                // `r=sqrt(sum((ns(i,:)-ns(j,:)).^2))` reads `ns`, the same
                // array mutating in place, not the frozen `nso` snapshot.
                let r2: f64 = (0..dim).map(|d| (ns[i][d] - ns[j][d]).powi(2)).sum();
                if lighto[i] > lighto[j] {
                    let beta = fa_beta(FA_BETA0, FA_BETAMIN, FA_GAMMA, r2);
                    for d in 0..dim {
                        let u = ctx.rng.next_f64();
                        let step = fa_step_term(alpha, u, scale[d]);
                        // self LIVE (ns[i][d], accumulates in place across
                        // repeated j hits), target FROZEN (nso[j][d]).
                        ns[i][d] = ns[i][d] * (1.0 - beta) + nso[j][d] * beta + step;
                    }
                }
            }
        }

        ns.into_iter().map(|xs| Genotype { blocks: vec![BlockValues::Float(xs)] }).collect()
    }

    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/fa", SupportedBlocks::Only(vec!["float"])).with_min_pop(2)
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_generator("gen/fa", |p| Ok(Box::new(FaGenerator::from_params(p)?)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::problem::{Problem, SphereShifted, Evaluator};
    use sezgi_core::rng::RngStream;
    use sezgi_core::state::Blackboard;

    fn g(xs: Vec<f64>) -> Genotype { Genotype { blocks: vec![BlockValues::Float(xs)] } }

    fn pop_nd(n: usize, dim: usize) -> Population {
        // Distinct fitness so rank order is unambiguous.
        Population {
            individuals: (0..n).map(|i| g((0..dim).map(|d| (i * dim + d) as f64 * 0.37 - 1.0).collect())).collect(),
            fitness: (0..n).map(|i| (i + 1) as f64).collect(), // index i already in rank order
        }
    }

    // ---- fa_beta: attractiveness formula ----

    #[test]
    fn beta_at_r_zero_is_beta0() {
        // r2=0 -> exp(0)=1 -> beta = (beta0-betamin)*1+betamin = beta0 exactly.
        assert_eq!(fa_beta(1.0, 0.2, 1.0, 0.0), 1.0);
        assert_eq!(fa_beta(2.0, 0.5, 3.0, 0.0), 2.0);
    }

    #[test]
    fn gamma_to_infinity_limit_is_betamin_floor_not_zero() {
        // The verified delta from the plan's sketch: the floored variant
        // means attraction does NOT vanish to 0 as gamma (or r2) grows --
        // it floors at betamin, a residual, non-vanishing pull.
        let beta_huge_gamma = fa_beta(1.0, 0.2, 1e6, 1.0);
        assert!((beta_huge_gamma - 0.2).abs() < 1e-9,
            "beta must approach betamin=0.2, not 0, as gamma -> infinity: got {beta_huge_gamma}");
        assert_ne!(beta_huge_gamma, 0.0, "beta must NOT vanish to 0 (that would be the un-floored sketch formula's behavior)");
    }

    #[test]
    fn betamin_zero_recovers_the_plain_vanishing_form() {
        // With betamin=0 (not sezgi's pinned default, but a parametric
        // property of the helper), the formula DOES collapse to the plain
        // beta0*exp(-gamma*r2) form, and the r2->infinity limit IS 0 --
        // confirming betamin is genuinely the source of the floor above.
        let beta = fa_beta(1.0, 0.0, 1.0, 50.0);
        let expected = 1.0 * (-50.0_f64).exp();
        assert_eq!(beta, expected);
        let beta_huge = fa_beta(1.0, 0.0, 1e6, 1.0);
        assert!(beta_huge.abs() < 1e-12, "with betamin=0, attraction must vanish as gamma -> infinity");
    }

    #[test]
    fn beta0_zero_gives_a_negative_of_betamin_times_the_decay() {
        // beta0=0: beta = (0-betamin)*exp(-gamma*r2)+betamin = betamin*(1-exp(-gamma*r2)).
        let beta = fa_beta(0.0, 0.2, 1.0, 3.0);
        let expected = 0.2 * (1.0 - (-3.0_f64).exp());
        assert!((beta - expected).abs() < 1e-12);
        // At r2=0 this must be exactly 0 (no pull toward a target at zero distance).
        assert_eq!(fa_beta(0.0, 0.2, 1.0, 0.0), 0.0);
    }

    // ---- fa_alpha: decay schedule ----

    #[test]
    fn alpha_schedule_endpoints() {
        assert_eq!(fa_alpha(0.5, 0.0), 0.5, "progress=0 must give alpha0 exactly");
        let expected_end = 0.5 * (1e-4_f64 / 0.9_f64);
        assert_eq!(fa_alpha(0.5, 1.0), expected_end);
        assert!(fa_alpha(0.5, 1.0) > 0.0 && fa_alpha(0.5, 1.0) < 1e-3,
            "alpha at progress=1 must be small but positive");
    }

    #[test]
    fn alpha_is_monotonically_decreasing_in_progress() {
        let a0 = fa_alpha(0.5, 0.0);
        let a_half = fa_alpha(0.5, 0.5);
        let a1 = fa_alpha(0.5, 1.0);
        assert!(a0 > a_half, "alpha(0) must exceed alpha(0.5)");
        assert!(a_half > a1, "alpha(0.5) must exceed alpha(1)");
    }

    // ---- fa_step_term ----

    #[test]
    fn step_term_is_zero_at_u_half() {
        assert_eq!(fa_step_term(0.5, 0.5, 10.0), 0.0);
    }

    #[test]
    fn step_term_scales_linearly_with_alpha_and_scale() {
        assert_eq!(fa_step_term(0.4, 0.9, 20.0), 0.4 * 0.4 * 20.0);
    }

    // ---- FaGenerator: best (rank 0) never moves, ties never fire ----

    #[test]
    fn best_ranked_individual_never_moves() {
        let n = 5; let dim = 3;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_nd(n, dim); // fitness = [1,2,3,4,5], index 0 is already best

        let mut evaluator = Evaluator::new(&p, 1000);
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let off = FaGenerator.generate(&pop, &mut ctx);

        assert_eq!(floats(&off[0]), floats(&pop.individuals[0]),
            "the rank-0 (brightest/best) firefly must never move: nothing is strictly brighter than it");
    }

    #[test]
    fn two_fireflies_with_equal_fitness_never_move_each_other() {
        // A tie: neither strictly brighter than the other -> zero draws, no movement.
        let dim = 2;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = Population {
            individuals: vec![g(vec![1.0, 2.0]), g(vec![3.0, 4.0])],
            fitness: vec![5.0, 5.0], // exact tie
        };

        let mut evaluator = Evaluator::new(&p, 1000);
        let mut rng = RngStream::from_master(1, &[]);
        let rng_before = rng.clone();
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let off = FaGenerator.generate(&pop, &mut ctx);

        assert_eq!(floats(&off[0]), &vec![1.0, 2.0]);
        assert_eq!(floats(&off[1]), &vec![3.0, 4.0]);
        // Zero draws consumed.
        let mut twin = rng_before;
        assert_eq!(rng.next_f64(), twin.next_f64(), "an all-tied population must consume zero RNG draws");
    }

    // ---- Twin-stream draw-count: crafted, hand-traced firing pattern ----

    #[test]
    fn draw_count_matches_hand_traced_firing_pairs() {
        // n=4, dim=3, all distinct fitness, already rank-ordered by index:
        // firing pairs (i,j) with j strictly brighter than i:
        //   i=0: none (rank 0, never fires)
        //   i=1: j=0                          -> 1 pair
        //   i=2: j=0, j=1                      -> 2 pairs
        //   i=3: j=0, j=1, j=2                 -> 3 pairs
        // total firing pairs = 0+1+2+3 = 6 = n*(n-1)/2 (fully distinct ranks).
        let n = 4; let dim = 3;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_nd(n, dim);

        let mut rng = RngStream::from_master(7, &[]);
        let rng_before = rng.clone();
        let mut evaluator = Evaluator::new(&p, 1000);
        let mut bb = Blackboard::new();
        {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
            let off = FaGenerator.generate(&pop, &mut ctx);
            assert_eq!(off.len(), n);
        }

        let firing_pairs = n * (n - 1) / 2;
        let mut twin = rng_before;
        for _ in 0..(firing_pairs * dim) { twin.next_f64(); }
        assert_eq!(rng.next_f64(), twin.next_f64(),
            "gen/fa must consume exactly firing_pairs*dim draws (n={n}, dim={dim}, firing_pairs={firing_pairs})");
    }

    #[test]
    fn draw_count_is_order_independent_of_input_array_order() {
        // Same fitness VALUES, but scrambled array order (not pre-sorted by
        // the caller) -- the draw count (a function of pairwise fitness
        // comparisons, not array position) must be identical to the
        // already-sorted case above: 6 firing pairs for n=4 distinct ranks.
        let dim = 2;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = Population {
            individuals: vec![g(vec![30.0, 0.0]), g(vec![10.0, 0.0]), g(vec![40.0, 0.0]), g(vec![20.0, 0.0])],
            fitness: vec![3.0, 1.0, 4.0, 2.0], // scrambled: rank order is index 1,3,0,2
        };

        let mut rng = RngStream::from_master(9, &[]);
        let rng_before = rng.clone();
        let mut evaluator = Evaluator::new(&p, 1000);
        let mut bb = Blackboard::new();
        {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
            let off = FaGenerator.generate(&pop, &mut ctx);
            assert_eq!(off.len(), 4);
        }

        let firing_pairs = 4 * 3 / 2; // 6, same as the pre-sorted case
        let mut twin = rng_before;
        for _ in 0..(firing_pairs * dim) { twin.next_f64(); }
        assert_eq!(rng.next_f64(), twin.next_f64(),
            "draw count must depend only on pairwise fitness comparisons, not input array order");
    }

    // ---- The in-place order property: hand-derived 3-firefly, 1-D fixture ----

    #[test]
    fn in_place_self_accumulation_and_live_distance_vs_frozen_target_hand_traced_3_firefly() {
        // n=3, dim=1, already rank-ordered: idx0 best (f=1.0, x=0.0), idx1
        // (f=2.0, x=5.0), idx2 worst (f=3.0, x=20.0). Wide domain so the
        // random step term is not degenerate. progress=0 -> alpha=FA_ALPHA0.
        let dim = 1;
        let lo = -1000.0; let hi = 1000.0;
        let p = SphereShifted::new(vec![0.0; dim], lo, hi);
        let space = p.space();
        let pop = Population {
            individuals: vec![g(vec![0.0]), g(vec![5.0]), g(vec![20.0])],
            fitness: vec![1.0, 2.0, 3.0],
        };
        let scale = hi - lo;

        let mut evaluator = Evaluator::new(&p, 1_000_000);
        let mut rng = RngStream::from_master(13, &[]);
        let rng_before = rng.clone();
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let off = FaGenerator.generate(&pop, &mut ctx);

        // Hand-trace using the SAME rng stream (twin replay) and the pinned helpers.
        let mut twin = rng_before;
        let alpha = fa_alpha(FA_ALPHA0, 0.0);

        // idx0: never fires, stays 0.0.
        assert_eq!(floats(&off[0])[0], 0.0, "rank-0 must not move");

        // idx1 (i=1): only j=0 fires. r2 = (5.0-0.0)^2 = 25.0 (ns[0] unmoved).
        let r2_1_0 = (5.0_f64 - 0.0).powi(2);
        let beta_1_0 = fa_beta(FA_BETA0, FA_BETAMIN, FA_GAMMA, r2_1_0);
        let u1 = twin.next_f64();
        let step1 = fa_step_term(alpha, u1, scale);
        let new1 = 5.0 * (1.0 - beta_1_0) + 0.0 * beta_1_0 + step1;
        assert_eq!(floats(&off[1])[0], new1, "idx1's single move must match the hand-derived formula");
        assert_ne!(new1, 5.0, "sanity: idx1 must have actually moved");

        // idx2 (i=2): j=0 fires first (self LIVE=20.0, target FROZEN=0.0).
        let r2_2_0 = (20.0_f64 - 0.0).powi(2);
        let beta_2_0 = fa_beta(FA_BETA0, FA_BETAMIN, FA_GAMMA, r2_2_0);
        let u2 = twin.next_f64();
        let step2 = fa_step_term(alpha, u2, scale);
        let ns2_after_j0 = 20.0 * (1.0 - beta_2_0) + 0.0 * beta_2_0 + step2;

        // j=1 fires next: self LIVE = ns2_after_j0 (accumulated in place, NOT
        // the original 20.0), distance uses LIVE ns[1] = new1 (idx1's
        // ALREADY-MOVED position from its own earlier outer-loop pass, NOT
        // the frozen nso[1]=5.0) -- but the BLENDED-IN target still uses
        // FROZEN nso[1]=5.0, not new1. This is the verified hybrid.
        let r2_2_1 = (ns2_after_j0 - new1).powi(2);
        let beta_2_1 = fa_beta(FA_BETA0, FA_BETAMIN, FA_GAMMA, r2_2_1);
        let u3 = twin.next_f64();
        let step3 = fa_step_term(alpha, u3, scale);
        let new2 = ns2_after_j0 * (1.0 - beta_2_1) + 5.0 * beta_2_1 + step3;

        assert_eq!(floats(&off[2])[0], new2,
            "idx2's final position must match the hand-derived in-place-self / live-distance / frozen-target formula");

        // Discriminate against the (WRONG) fully-frozen alternative: distance
        // computed from nso[1]=5.0 instead of the live new1, self base
        // re-started from the original 20.0 instead of accumulating.
        let r2_2_1_wrong = (20.0_f64 - 5.0).powi(2);
        let beta_2_1_wrong = fa_beta(FA_BETA0, FA_BETAMIN, FA_GAMMA, r2_2_1_wrong);
        let new2_wrong = 20.0 * (1.0 - beta_2_1_wrong) + 5.0 * beta_2_1_wrong + step3;
        assert_ne!(floats(&off[2])[0], new2_wrong,
            "idx2's final position must NOT match the fully-frozen (non-in-place) alternative");
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
            FaGenerator.generate(&pop, &mut ctx)
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
            FaGenerator.generate(&pop, &mut ctx)
        }));
        assert!(result.is_err(), "gen/fa must reject pop_size < 2 at runtime as a backstop");
    }
}
