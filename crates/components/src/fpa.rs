use crate::cs::cs_dim_step;
use sezgi_core::component::*;
use sezgi_core::dist::Distribution;
use sezgi_core::problem::Population;
use sezgi_core::rng::RngStream;
use sezgi_core::space::{BlockValues, Genotype};

/// Flower Pollination Algorithm (Yang, X.-S. 2012, "Flower Pollination
/// Algorithm for Global Optimization", in: *Unconventional Computation and
/// Natural Computation* (UCNC 2012), Lecture Notes in Computer Science vol.
/// 7445, Springer, pp. 240-249, DOI 10.1007/978-3-642-32894-7_27) -- a
/// **labeled metaphor preset**: faithful to the primary source's own
/// reference MATLAB implementation's update equations and loop structure,
/// with a pinned deterministic draw order and property tests, but NOT
/// validated against the paper's (or any other publication's) reported
/// benchmark numbers. No established equivalence critique covers FPA (the
/// Camacho-Villalón/Dorigo/Stützle *ITOR* six-algorithm critique covers
/// GWO/MFO/WOA/FA/BA/ALO, not FPA) -- cited here as primary-source only,
/// per the task brief.
///
/// ## Provenance (PROVENANCE-FIRST protocol)
///
/// **Primary artifact used (governs):** Yang's own `fpa_demo.m` (MATLAB
/// Central File Exchange **#45112**, "Flower Pollination Algorithm",
/// published under Yang's own account; the demo's own header cites *Nature-
/// Inspired Optimization Algorithms*, Elsevier Insights (2014), and the UCNC
/// paper above). File Exchange's own page exposes only metadata/prose to
/// this project's text-fetch tooling -- the same limitation `jaya.rs`'s
/// provenance note already documented for a different File Exchange
/// submission -- but the FULL verbatim source was independently recovered
/// from a SlideShare deck hosted under Yang's own account
/// (`slideshare.net/xinshe/flower-pollination-algorithm-matlab-code`),
/// which reproduces `fpa_demo.m` in full, including its exact variable names
/// (`Sol`, `Fitness`, `best`, `fmin`, `S`) matching the SAME per-agent /
/// evaluate / greedy-accept / unconditional-`fmin`-update loop skeleton this
/// crate already verified for `bat_algorithm.m` (see `ba.rs`'s provenance
/// note) -- strong cross-artifact consistency confirming an authentic
/// Yang-authored source, not a third-party reimplementation. Key quoted
/// lines (verbatim, per-flower):
///
/// ```text
/// p=para(2);                          % switch probability, default 0.8
/// for i=1:n
///    if rand>p
///       L=Levy(d);
///       dS=L.*(Sol(i,:)-best);
///       S(i,:)=Sol(i,:)+dS;           % GLOBAL pollination
///    else
///       epsilon=rand;
///       JK=randperm(n);
///       S(i,:)=S(i,:)+epsilon*(Sol(JK(1),:)-Sol(JK(2),:));  % LOCAL pollination
///    end
///    S(i,:)=simplebounds(S(i,:),Lb,Ub);
///    Fnew=Fun(S(i,:));
///    if (Fnew<=Fitness(i))
///       Sol(i,:)=S(i,:); Fitness(i)=Fnew;
///    end
///    if Fnew<=fmin
///       best=S(i,:); fmin=Fnew;
///    end
/// end
/// ```
///
/// and the demo's own `Levy(d)` helper: `beta=3/2;` (Mantegna exponent, i.e.
/// `alpha=1.5`), `sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*
/// beta*2^((beta-1)/2)))^(1/beta);`, `u=randn(1,d)*sigma; v=randn(1,d);
/// step=u./abs(v).^(1/beta); L=0.01*step;` -- the classic Mantegna (1994)
/// algorithm, EXACTLY the scheme sezgi's own `Distribution::Levy { alpha }`
/// implements (see `cs.rs`'s module doc for the sampler's full description),
/// with the SAME fixed `0.01` scale factor `cs.rs`'s `ALPHA_STEP` already
/// uses (verified independently for the Cuckoo Search demo -- both are
/// Yang's own demos sharing this exact idiom).
///
/// ## Verified findings (each checked against the plan's sketch; SOURCE
/// GOVERNS on every delta)
///
/// 1. **Switch orientation -- a genuine, notable delta from the plan's
///    sketch:** `if rand>p` selects the GLOBAL (Lévy) branch, NOT `u<p` as
///    the plan's sketch guessed. With the demo's own default `p=0.8`, this
///    means GLOBAL fires only ~20% of the time and LOCAL fires ~80% --
///    exactly matching the algorithm's well-known prose description ("a
///    slight bias towards local pollination"), which is strong independent
///    confirmation this is the correct orientation, not a
///    mistranscription. Pinned as [`fpa_is_global_branch`]: `u > p`.
/// 2. **Global-step sign -- a genuine delta from the plan's sketch:** the
///    demo computes `dS=L.*(Sol(i,:)-best)`, i.e. `L·(X_i − X_best)`, NOT
///    `L·(X_best − X_i)` as the plan's sketch phrased it. Composed with the
///    demo's `L=0.01*step` scale, this is `X'[d] = X[d] + 0.01·L_raw[d]·
///    (X[d] − X_best[d])` -- ALGEBRAICALLY IDENTICAL, sign for sign and
///    constant for constant, to `cs.rs`'s already-pinned and tested
///    [`cs_dim_step`] (`x_i_d + alpha_step * levy * (x_i_d - x_best_d)`,
///    `alpha_step = 0.01`, `Distribution::Levy { alpha: 1.5 }`). Per the
///    REUSE-before-writing global constraint, FPA's global-pollination step
///    calls [`cs_dim_step`] directly rather than duplicating an identical
///    formula.
/// 3. **Local-step epsilon -- confirmed scalar per flower, matching the
///    plan's guess:** `epsilon=rand;` is a single scalar draw, applied (via
///    MATLAB broadcasting) uniformly across every dimension of `S(i,:)`,
///    NOT redrawn per dimension.
/// 4. **Local-step index selection `j,k` -- a genuine delta from the plan's
///    sketch's "distinct from each other AND from i?" question:**
///    `JK=randperm(n)` permutes ALL `n` indices (`1..n` in MATLAB, `0..n`
///    here), INCLUDING `i` itself -- `JK(1)`/`JK(2)` are guaranteed
///    distinct FROM EACH OTHER (a permutation has no duplicate entries) but
///    are NOT excluded from equaling `i`. This is the demo's own design,
///    not a bug (the same self-selection-not-excluded idiom this crate
///    already verified for `woa.rs`'s random-whale index -- "matching the
///    reference MATLAB, which does not exclude self"). Pinned here via
///    [`pick_two_distinct`]: draw `j` uniformly from `0..n`, then draw `k`
///    uniformly from `0..n`, REJECTING only `k == j` (never rejecting on
///    `== i`, since `pick_two_distinct` doesn't even take an `i` parameter)
///    -- the same distribution as taking the first two entries of a random
///    permutation of `0..n`, at a fraction of the draw cost (2 raw index
///    draws in the typical case, instead of materializing a full
///    length-`n` permutation), following this project's established
///    distinct-index rejection-sampling idiom (`de.rs`'s `pick_distinct`).
/// 5. **Acceptance -- confirmed greedy same-index, `<=` not `<`:**
///    `if (Fnew<=Fitness(i))`. sezgi reuses `replace/one-to-one-greedy`
///    (DE's kind, strict `<`) as-is, matching `cs.rs`'s and `jaya.rs`'s
///    same idiom -- the `<=` vs `<` distinction is immaterial for
///    continuous-valued fitness (an exact tie has probability zero under
///    this project's RNG streams) and every other metaphor preset in this
///    wave that reuses this replacer carries the identical,
///    undocumented-as-a-delta simplification.
/// 6. **Attractor `best`/`fmin` -- the wave's parked current-pop-attractor
///    convention applies, per the `ba.rs`/`ssa.rs` precedent:** the
///    verified source persists `best`/`fmin` as a global-best-EVER, updated
///    unconditionally and potentially MID-generation (after every flower,
///    before the next one is processed) -- structurally identical to
///    `bat_algorithm.m`'s `best`/`fmin` bookkeeping (see `ba.rs`'s delta
///    7). Per the wave's already-parked convention, `// sezgi simplification:`
///    this module uses the CURRENT population's fitness argmin instead
///    ([`Population::best_index`], computed ONCE per `generate()` call,
///    before any RNG draws) -- a documented, wave-wide simplification, not
///    an FPA-specific one.
///
/// ## min_pop -- ADJUSTED from the plan's sketched `3` to `2`
///
/// Finding 4 above settles this: `j`/`k` need only be distinct FROM EACH
/// OTHER, not from `i`, so a population of exactly 2 already provides two
/// valid, distinct indices for the local branch. `min_pop = 2` is in fact a
/// HARD requirement, not merely "meaningful": [`pick_two_distinct`]'s
/// rejection loop for `k` never terminates when `n == 1` (there is no
/// candidate `!= j` to find), so `pop_size >= 2` is required for
/// correctness, not just fidelity -- enforced both by
/// `ComponentMeta::with_min_pop(2)` (spec-validation) and the runtime
/// `assert!` backstop in [`FpaGenerator::generate`] (the same two-layer
/// enforcement idiom as every other preset in this crate).
///
/// ## Draw order per flower `i` (population order) -- part of the
/// RNG-stream contract
///
/// 1. `u = ctx.rng.next_f64()` (1 draw) -- branch switch.
/// 2. If `u > p` (GLOBAL): for each dimension `d` (index order), exactly
///    ONE `Distribution::Levy { alpha: 1.5 }.sample(ctx.rng)` call (the
///    call-boundary pin, `cs.rs`'s precedent -- see that module's doc for
///    why the RAW draw count behind one such call is data-dependent).
/// 3. Else (LOCAL): `epsilon = ctx.rng.next_f64()` (1 draw, scalar);
///    `j = ctx.rng.next_below(n)` (1 draw); `k` via rejection sampling
///    (>= 1 draw, data-dependent) rejecting only `k == j`. No further draws
///    in the dimension loop -- `epsilon`/`j`/`k` are all fixed before it.
///
/// The per-dimension math is factored into [`fpa_local_dim_step`] (local
/// branch) and reuses [`cs_dim_step`] (global branch, see finding 2) so
/// both are unit-tested directly without needing to fake `Ctx`/`RngStream`.
/// [`pick_two_distinct`] and [`fpa_is_global_branch`] are similarly
/// factored out for direct unit testing. [`FpaGenerator::generate`] is a
/// thin loop over `i` that draws the pinned values in order and calls the
/// appropriate helper(s).
///
/// Boundary handling and replacement are NOT part of this generator:
/// `presets::fpa` reuses `boundary/clamp` (same as `cs`/`gwo`/`woa` -- the
/// demo's own `simplebounds` clamps into `[Lb,Ub]`, structurally the same
/// role, decoupled from the generator the same way `ba.rs`'s boundary-clamp
/// -target note describes) and `replace/one-to-one-greedy` (DE's kind, see
/// finding 5) for the greedy same-index replacement.
pub fn fpa_local_dim_step(x_d: f64, x_j_d: f64, x_k_d: f64, epsilon: f64) -> f64 {
    x_d + epsilon * (x_j_d - x_k_d)
}

/// Branch-switch orientation, pinned per finding 1: `u > p` selects the
/// GLOBAL (Lévy) branch. Factored out as a pure predicate for direct
/// property testing (`p=1` => always local, `p=0` => always global).
pub fn fpa_is_global_branch(u: f64, p: f64) -> bool {
    u > p
}

/// Draw two indices from `0..n`, distinct FROM EACH OTHER but NOT
/// necessarily from any other index a caller may hold (including its own
/// `i`) -- pinned per finding 4 (`JK=randperm(n)`'s first two entries, same
/// distribution, cheaper draw count than materializing a full permutation).
/// Requires `n >= 2`: with `n < 2` the rejection loop for `k` never
/// terminates (see the module doc's min_pop rationale) -- callers must
/// enforce `n >= 2` before calling.
pub fn pick_two_distinct(rng: &mut RngStream, n: usize) -> (usize, usize) {
    let j = rng.next_below(n as u64) as usize;
    let k = loop {
        let cand = rng.next_below(n as u64) as usize;
        if cand != j { break cand; }
    };
    (j, k)
}

const FPA_P: f64 = 0.8;
const ALPHA_STEP: f64 = 0.01;
const LEVY_ALPHA: f64 = 1.5;

pub struct FpaGenerator;

impl FpaGenerator {
    pub fn from_params(_p: &serde_json::Value) -> Result<Self, ComponentError> {
        // No tunable parameters (p/alpha_step/levy alpha are fixed, per the pinned spec).
        Ok(Self)
    }

    fn floats(g: &Genotype) -> &Vec<f64> {
        match &g.blocks[0] { BlockValues::Float(x) => x, _ => unreachable!() }
    }
}

impl Generator for FpaGenerator {
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        assert!(pop.len() >= 2, "gen/fpa requires a population of at least 2 (pop_size={})", pop.len());
        let n = pop.len();
        let dim = Self::floats(&pop.individuals[0]).len();

        // Current-population best (fitness argmin, ties -> lower index) --
        // pinned per finding 6, computed once per generate() call, before
        // any RNG draws.
        let best = pop.best_index().unwrap_or(0);
        let x_best = Self::floats(&pop.individuals[best]);

        (0..n).map(|i| {
            let x = Self::floats(&pop.individuals[i]);
            // Pinned draw order: branch switch first, per flower.
            let u = ctx.rng.next_f64();
            let xs: Vec<f64> = if fpa_is_global_branch(u, FPA_P) {
                (0..dim).map(|d| {
                    // Pinned: exactly one Distribution::Levy{alpha:1.5}.sample() per (i, d).
                    let levy = Distribution::Levy { alpha: LEVY_ALPHA }.sample(ctx.rng);
                    cs_dim_step(x[d], x_best[d], levy, ALPHA_STEP)
                }).collect()
            } else {
                // Pinned: epsilon drawn ONCE (scalar), then j, k -- all
                // before the dimension loop (finding 3/4).
                let epsilon = ctx.rng.next_f64();
                let (j, k) = pick_two_distinct(ctx.rng, n);
                let x_j = Self::floats(&pop.individuals[j]);
                let x_k = Self::floats(&pop.individuals[k]);
                (0..dim).map(|d| fpa_local_dim_step(x[d], x_j[d], x_k[d], epsilon)).collect()
            };
            Genotype { blocks: vec![BlockValues::Float(xs)] }
        }).collect()
    }

    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/fpa", SupportedBlocks::Only(vec!["float"])).with_min_pop(2)
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_generator("gen/fpa", |p| Ok(Box::new(FpaGenerator::from_params(p)?)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::problem::{Problem, SphereShifted, Evaluator};
    use sezgi_core::state::Blackboard;

    fn g(xs: Vec<f64>) -> Genotype { Genotype { blocks: vec![BlockValues::Float(xs)] } }

    fn pop_nd(n: usize, dim: usize) -> Population {
        // Distinct fitness so best-index selection is unambiguous.
        Population {
            individuals: (0..n).map(|i| g((0..dim).map(|d| (i * dim + d) as f64 * 0.37 - 1.0).collect())).collect(),
            fitness: (0..n).map(|i| (n - i) as f64).collect(), // higher index = better (lower fitness)
        }
    }

    // ---- fpa_local_dim_step ----

    #[test]
    fn fpa_local_dim_step_uses_the_same_epsilon_across_all_dimensions() {
        // Crafted fixture: the SAME scalar epsilon applied across two
        // different dimensions' (x_j - x_k) gaps must give results
        // consistent with a single shared epsilon -- exactly what
        // FpaGenerator::generate does (draw epsilon ONCE, reuse it for
        // every dimension), not an independently-redrawn-per-dimension one.
        let epsilon = 0.37;
        let d0 = fpa_local_dim_step(1.0, 5.0, 2.0, epsilon); // 1.0 + 0.37*3.0
        let d1 = fpa_local_dim_step(10.0, -1.0, -4.0, epsilon); // 10.0 + 0.37*3.0
        assert!((d0 - 2.11).abs() < 1e-12, "got {d0}");
        assert!((d1 - 11.11).abs() < 1e-12, "got {d1}");
    }

    #[test]
    fn fpa_local_dim_step_zero_when_j_equals_k() {
        // j == k (structurally impossible per pick_two_distinct, but the
        // pure helper itself has no such guard) makes (x_j - x_k) vanish.
        assert_eq!(fpa_local_dim_step(5.0, 3.0, 3.0, 0.9), 5.0);
    }

    // ---- fpa_is_global_branch: switch-orientation property ----

    #[test]
    fn switch_branch_orientation_p_extremes() {
        // p=1: u in [0,1) can never exceed 1 -> ALWAYS local (verified orientation).
        for u in [0.0, 0.3, 0.8, 0.999999] {
            assert!(!fpa_is_global_branch(u, 1.0), "p=1 must always select local, got global for u={u}");
        }
        // p=0: u>0 for any u this project's RngStream::next_f64() can
        // produce that isn't the exact boundary -> effectively always
        // global.
        for u in [0.000001, 0.3, 0.8, 0.999999] {
            assert!(fpa_is_global_branch(u, 0.0), "p=0 must select global for any u>0, got local for u={u}");
        }
        assert!(!fpa_is_global_branch(0.0, 0.0), "u=0.0 exactly must NOT exceed p=0.0 (strict > per the verified source)");
    }

    // ---- global-branch reuse of cs_dim_step: zero-distance property ----

    #[test]
    fn global_step_unchanged_when_x_equals_best_reusing_cs_dim_step() {
        // FPA's global-pollination step reuses cs.rs's cs_dim_step
        // VERBATIM (finding 2) -- so X_i == X_best makes the
        // (X_i - X_best) factor exactly 0.0, leaving the offspring
        // dimension EXACTLY unchanged regardless of the Levy draw's value
        // (see cs.rs's own zero_step_when_x_i_equals_x_best_exactly for the
        // original proof; this test documents the same property continuing
        // to hold through FPA's reuse).
        let levy_arbitrary = 12345.6789;
        let result = cs_dim_step(3.5, 3.5, levy_arbitrary, ALPHA_STEP);
        assert_eq!(result, 3.5);
    }

    // ---- pick_two_distinct ----

    #[test]
    fn pick_two_distinct_always_returns_distinct_in_range_indices() {
        let mut rng = RngStream::from_master(11, &[]);
        for _ in 0..200 {
            let (j, k) = pick_two_distinct(&mut rng, 5);
            assert_ne!(j, k);
            assert!(j < 5 && k < 5);
        }
    }

    #[test]
    fn pick_two_distinct_n2_always_the_only_possible_pair() {
        let mut rng = RngStream::from_master(3, &[]);
        for _ in 0..50 {
            let (j, k) = pick_two_distinct(&mut rng, 2);
            assert_ne!(j, k);
            assert!((j == 0 && k == 1) || (j == 1 && k == 0), "n=2 must always give {{0,1}}, got ({j},{k})");
        }
    }

    #[test]
    fn pick_two_distinct_does_not_exclude_an_external_index() {
        // Per finding 4: j,k are distinct from EACH OTHER but NOT from any
        // caller-held index (pick_two_distinct doesn't even take one) --
        // demonstrate this is a REAL, exercised behavior: across many
        // draws with n=3, index 2 is selected as j or k with overwhelming
        // probability (not "never", which excluding it would force).
        let mut rng = RngStream::from_master(21, &[]);
        let mut saw_index_2_selected = false;
        for _ in 0..100 {
            let (j, k) = pick_two_distinct(&mut rng, 3);
            if j == 2 || k == 2 { saw_index_2_selected = true; }
        }
        assert!(saw_index_2_selected, "with n=3, index 2 should appear as j or k across 100 draws");
    }

    // ---- FpaGenerator ----

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
            FpaGenerator.generate(&pop, &mut ctx)
        };

        let a = run();
        let b = run();
        assert_eq!(a.len(), n);
        for (ga, gb) in a.iter().zip(b.iter()) {
            let (xa, xb) = (FpaGenerator::floats(ga), FpaGenerator::floats(gb));
            assert_eq!(xa, xb, "same seed must produce bit-identical offspring");
        }
    }

    #[test]
    fn draw_order_matches_pinned_structure_via_raw_replay() {
        // Twin-stream raw-replay (cs.rs's / ba.rs's technique): independently
        // reproduce the PINNED draw structure -- per flower, a branch-switch
        // draw, then either per-dim Levy calls (global) or epsilon +
        // pick_two_distinct (local) -- on a cloned pre-generate() stream,
        // and confirm the real generator consumed EXACTLY the same
        // sequence (verified by both streams' next call landing on the
        // identical value). This is also the "ε scalar-vs-per-dim" and
        // "Lévy call-boundary" pin checks: had epsilon been drawn per-dim,
        // or a Levy call skipped/duplicated per dim, the twin's hand-coded
        // replay (which hard-codes the pinned counts) would diverge from
        // the real generator's stream and this assertion would fail.
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
            let off = FpaGenerator.generate(&pop, &mut ctx);
            assert_eq!(off.len(), n);
        }

        let mut twin = rng_before;
        for _ in 0..n {
            let u = twin.next_f64();
            if fpa_is_global_branch(u, FPA_P) {
                for _ in 0..dim { Distribution::Levy { alpha: LEVY_ALPHA }.sample(&mut twin); }
            } else {
                twin.next_f64(); // epsilon (scalar, one draw)
                pick_two_distinct(&mut twin, n);
            }
        }
        assert_eq!(rng.next_f64(), twin.next_f64(),
            "gen/fpa must consume exactly the pinned draw sequence: branch switch, then per-dim Levy (global) or epsilon+pick_two_distinct (local), per flower");
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
            FpaGenerator.generate(&pop, &mut ctx)
        }));
        assert!(result.is_err(), "gen/fpa must reject pop_size < 2 at runtime as a backstop");
    }
}
