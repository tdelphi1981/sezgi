use sezgi_core::component::*;
use sezgi_core::problem::Population;
use sezgi_core::space::{BlockValues, Genotype};

/// JAYA (Rao, R.V. 2016, "Jaya: A Simple and New Optimization Algorithm for
/// Solving Constrained and Unconstrained Optimization Problems",
/// *International Journal of Industrial Engineering Computations* 7(1),
/// 19-34, DOI 10.5267/j.ijiec.2015.8.004) -- a **labeled metaphor preset**:
/// faithful to the primary source's stated update equation AND its own
/// worked numerical example (Section 2.1, Tables 1-5), with a pinned
/// deterministic draw order and property tests, but NOT validated against
/// the paper's (or any other publication's) reported benchmark numbers. No
/// established equivalence critique covers JAYA (the Camacho-Villalón/
/// Dorigo/Stützle *ITOR* six-algorithm critique covers GWO/MFO/WOA/FA/BA/
/// ALO, not JAYA) -- cited here as primary-source only, per the task brief.
///
/// **Provenance note** (part of the PROVENANCE-FIRST protocol): the paper's
/// own linked/hosting site (growingscience.com/ijiec) does not distribute a
/// separate reference-implementation source file. The most prominent
/// MATLAB File Exchange submission titled "Jaya: A simple and new
/// optimization algorithm" (#74004) is NOT by R. Venkata Rao himself (its
/// author is "iraj faraji," a third party) and its `.m` source could not be
/// extracted through the tooling available here (the File Exchange page
/// exposes only metadata/prose to a text fetch, not the file body) -- so it
/// was NOT used as a claimed-verified source. Instead, this module verifies
/// DIRECTLY against the primary paper itself (the actual PDF, fetched from
/// its own publisher, growingscience.com), including hand-reproducing its
/// worked numerical example (Tables 1-2, Section 2.1: population of 5,
/// 2 design variables, one full iteration) via the pinned formula below --
/// see [`jaya_dim_step`]'s tests for the exact reproduction. mealpy's
/// `swarm_based/JA.py` (`OriginalJA`/`DevJA` classes, fetched from
/// `github.com/thieu1995/mealpy`, `master` branch) was consulted as the
/// protocol's secondary/fallback source; see the DELTA notes below for
/// where it agrees and where the primary paper (verified numerically, so
/// authoritative here) overrides it.
///
/// **Pinned update rule** (part of the RNG-stream contract), Eq. (1) of the
/// paper: for candidate `k`, variable `j`, iteration `i`:
/// `X'[j,k,i] = X[j,k,i] + r1[j,i]·(X[j,best,i] − |X[j,k,i]|) − r2[j,i]·(X[j,worst,i] − |X[j,k,i]|)`.
/// The `|X[j,k,i]|` absolute-value terms are the paper's own literal Eq. (1)
/// -- confirmed real, not a sketch artifact (the "odd but canonical form"
/// the task brief flagged for verification): the worked example's own
/// arithmetic (e.g. `X'_{1,1,1} = -5 + 0.58·(-8 − |−5|) − 0.81·(70 − |−5|) = -65.19`)
/// only reproduces the published table when `|X[j,k,i]|` is a genuine
/// absolute value, not `X[j,k,i]` itself.
///
/// - `best`/`worst`: the CURRENT population's fitness argmin/argmax
///   (best: [`Population::best_index`], ties → lower index, same
///   convention as every other preset in this crate; worst: this module's
///   own [`worst_index`], ties → HIGHER index -- mirroring `cs.rs`'s
///   `abandon_order` and `nm.rs`'s `worst_index`, both of which use
///   "worst-selection ties → higher index" as this crate's established
///   opposite-extremum convention to `best_index`'s "ties → lower index"),
///   computed once per `generate` call, before any RNG draws. This is the
///   **current-population argmin/argmax convention** already ruled on for
///   the wave (parked in Task 1/SCA's review, recorded in T15's DECISIONS):
///   the paper's own Fig. 1 flowchart re-identifies best/worst from the
///   population at the TOP of every iteration (after the PREVIOUS
///   iteration's greedy acceptance step) -- it does not persist a
///   separately-tracked historical best/worst across iterations -- so there
///   is no delta to record here; the ruling and the verified source agree.
/// - **Draw order/count -- the wave's "WOA-class loop-structure trap" for
///   this task, VERIFIED, and a genuine delta from BOTH this plan's sketch
///   and mealpy's fallback source:** the paper states "`r1,j,i` and `r2,j,i`
///   are the two random numbers for the **j-th variable during the i-th
///   iteration**" -- indexed only by dimension `j` and iteration `i`, with
///   NO candidate index `k`. This is not just prose: the worked example
///   (Tables 1-2) explicitly REUSES the SAME `r1=0.58, r2=0.81` (drawn once
///   for `x1` this iteration) across candidates 1, 2, 3, 4 and 5 alike (and
///   likewise `r1=0.92, r2=0.49` once for `x2`) -- hand-verified here by
///   reproducing THREE different candidates' new `x1`/`x2` values bit-for-
///   bit from the published table using those two shared pairs (see this
///   module's tests, `matches_the_papers_worked_example_*`). So `r1[d]`,
///   `r2[d]` are drawn **ONCE PER DIMENSION PER GENERATION (iteration),
///   SHARED across every agent in the population** -- NOT fresh per
///   `(i, d)` as the plan's sketch phrased it ("per agent, per dimension:
///   draw r1, r2"), and NOT fresh per `(i, d)` as mealpy's `OriginalJA`/
///   `DevJA` both implement it either (`self.generator.uniform(0, 1,
///   self.problem.n_dims)` is called INSIDE mealpy's per-candidate loop,
///   redrawing both vectors fresh for every agent) -- mealpy's source
///   genuinely diverges from the paper's own stated-and-demonstrated
///   convention here. Per the provenance protocol, the SOURCE GOVERNS: the
///   primary paper, numerically verified against its own published table,
///   is authoritative over the secondary/fallback reimplementation. Total
///   draw count per `generate` call: exactly `2 * dim` (`dim` values of
///   `r1`, `dim` values of `r2`) -- INDEPENDENT of population size `n`, a
///   sharp contrast to SCA's `3*n*dim` or GWO's `6*n*dim`.
/// - **Draw sequence within the dimension loop:** the paper's worked
///   example presents `r1` then `r2` GROUPED per dimension (`x1`'s `r1,r2`
///   together, then `x2`'s `r1,r2`), which is the only ordering constraint
///   the primary source's own presentation supplies (no source code was
///   available to pin an implementation-level order beyond that). sezgi
///   pins this same per-dimension-grouped order: for `d` in ascending index
///   order, draw `r1[d]` then `r2[d]` -- consistent with this project's
///   established per-dimension-grouped draw idiom elsewhere (`sca.rs`'s
///   `r2,r3,r4` grouped per `(i,d)`; `gwo.rs`'s `r1,r2` grouped per
///   `(leader,d)`).
///
/// **DELTA -- replacement (mealpy naming is misleading here, VERIFIED
/// against the primary source):** the paper's Fig. 1 flowchart states
/// `X'` is accepted (replacing the previous solution) ONLY if it gives a
/// BETTER function value, else the previous solution is kept -- explicit
/// per-candidate greedy same-index comparison. The worked example's Table 3
/// confirms this directly: of the 5 candidates in iteration 1, only
/// candidates 2 and 3 (whose new function values improved) are replaced;
/// candidates 1, 4 and 5 (whose new values were worse) keep their PREVIOUS
/// values exactly. mealpy's class actually named `OriginalJA`
/// (`swarm_based/JA.py`) does NOT do this -- it unconditionally overwrites
/// `self.pop = pop_new` every iteration; mealpy's OWN "developed" variant,
/// `DevJA` (not the paper's algorithm -- mealpy's own modification), is the
/// one that applies `get_better_agent`/`greedy_selection_population`. Since
/// the primary paper's text AND its own worked table unambiguously specify
/// and demonstrate greedy acceptance, the paper governs (mealpy's
/// `OriginalJA` naming notwithstanding): this preset uses greedy same-index
/// replacement, matching the plan's sketch (`replace/one-to-one-greedy`,
/// DE's kind, reused as-is -- see `replace.rs`).
///
/// The per-dimension math is factored into [`jaya_dim_step`] so it can be
/// unit-tested directly -- both the `X_best == X_worst == |X|` zero-step
/// property (from the plan's sketch, algebraically exact: both correction
/// terms vanish identically regardless of `r1`/`r2`) and a direct
/// reproduction of the paper's own worked numbers -- without needing to
/// fake `Ctx`/`RngStream`. [`JayaGenerator::generate`] draws the `2*dim`
/// shared values once, then loops `(i, d)` applying the helper.
///
/// `min_pop = 2`: JAYA needs a best-so-far AND a worst-so-far individual
/// (best and worst indices must both be selectable) for the update to be
/// meaningful -- same rationale as `woa.rs`'s/`sca.rs`'s `min_pop = 2` (with
/// `pop_size == 1` the single individual is simultaneously its own best and
/// worst, degenerating the algorithm to a fixed point since `X_best -
/// |X| == X_worst - |X|`).
///
/// Boundary handling is NOT part of this generator: `presets::jaya` reuses
/// `boundary/clamp` (same as `gwo`/`woa`/`sca` -- the update can push a
/// coordinate outside `[lb, ub]`, and the paper's own worked example uses
/// unconstrained ranges `[-100, 100]` without a described bound-repair
/// step, so clamping is this project's standard boundary-repair choice, not
/// a paper-specified mechanism) and `replace/one-to-one-greedy` (DE's kind,
/// reused as-is -- see the replacement DELTA note above) for the greedy
/// same-index replacement.
pub fn jaya_dim_step(x_d: f64, x_best_d: f64, x_worst_d: f64, r1: f64, r2: f64) -> f64 {
    let abs_x = x_d.abs();
    x_d + r1 * (x_best_d - abs_x) - r2 * (x_worst_d - abs_x)
}

/// Index of the worst (highest-fitness) individual, ties → HIGHER index --
/// mirrors `cs.rs`'s `abandon_order`/`nm.rs`'s `worst_index` tie-break
/// convention (this crate's established opposite-extremum counterpart to
/// [`Population::best_index`]'s "ties → lower index"). Pure helper, factored
/// out for direct unit testing without `Ctx`/`Evaluator`/`RngStream`.
pub fn worst_index(fitness: &[f64]) -> Option<usize> {
    fitness.iter().enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
}

pub struct JayaGenerator;

impl JayaGenerator {
    pub fn from_params(_p: &serde_json::Value) -> Result<Self, ComponentError> {
        // No tunable parameters (JAYA is parameter-free by design -- the
        // paper's own selling point, see its abstract/Section 1).
        Ok(Self)
    }

    fn floats(g: &Genotype) -> &Vec<f64> {
        match &g.blocks[0] { BlockValues::Float(x) => x, _ => unreachable!() }
    }
}

impl Generator for JayaGenerator {
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        assert!(pop.len() >= 2, "gen/jaya requires a population of at least 2 (pop_size={})", pop.len());
        let n = pop.len();
        let dim = Self::floats(&pop.individuals[0]).len();

        // Best/worst: current-population argmin/argmax -- pinned (see
        // module doc's "current-population argmin/argmax convention" note),
        // computed once per generate() call, before any RNG draws.
        let best = pop.best_index().unwrap_or(0);
        let worst = worst_index(&pop.fitness).unwrap_or(0);
        let x_best = Self::floats(&pop.individuals[best]);
        let x_worst = Self::floats(&pop.individuals[worst]);

        // Pinned: r1[d], r2[d] drawn ONCE PER DIMENSION PER GENERATION,
        // shared across every agent -- see the module doc's worked-example
        // verification. Exactly 2*dim draws total, independent of n.
        let mut r1 = Vec::with_capacity(dim);
        let mut r2 = Vec::with_capacity(dim);
        for _ in 0..dim {
            r1.push(ctx.rng.next_f64());
            r2.push(ctx.rng.next_f64());
        }

        (0..n).map(|i| {
            let x = Self::floats(&pop.individuals[i]);
            let xs: Vec<f64> = (0..dim).map(|d| {
                jaya_dim_step(x[d], x_best[d], x_worst[d], r1[d], r2[d])
            }).collect();
            Genotype { blocks: vec![BlockValues::Float(xs)] }
        }).collect()
    }

    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/jaya", SupportedBlocks::Only(vec!["float"])).with_min_pop(2)
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_generator("gen/jaya", |p| Ok(Box::new(JayaGenerator::from_params(p)?)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::problem::{Problem, SphereShifted, Evaluator};
    use sezgi_core::rng::RngStream;
    use sezgi_core::state::Blackboard;

    fn g(xs: Vec<f64>) -> Genotype { Genotype { blocks: vec![BlockValues::Float(xs)] } }

    fn pop_nd(n: usize, dim: usize) -> Population {
        // Distinct fitness so best-index/worst-index selection is unambiguous.
        Population {
            individuals: (0..n).map(|i| g((0..dim).map(|d| (i * dim + d) as f64 * 0.37 - 1.0).collect())).collect(),
            fitness: (0..n).map(|i| (n - i) as f64).collect(), // higher index = better (lower fitness)
        }
    }

    // ---- jaya_dim_step: property + paper's own worked-example reproduction ----

    #[test]
    fn zero_step_when_best_equals_worst_equals_abs_x() {
        // X_best == X_worst == |X| makes BOTH correction terms vanish
        // identically (X_best - |X| == 0 and X_worst - |X| == 0), so
        // offspring == X exactly, regardless of r1/r2 -- the plan sketch's
        // named property test.
        let x_d: f64 = -3.5;
        let abs_x = x_d.abs(); // 3.5
        for (r1, r2) in [(0.0, 0.0), (0.37, 0.92), (1.0, 1.0)] {
            let result = jaya_dim_step(x_d, abs_x, abs_x, r1, r2);
            assert_eq!(result, x_d, "X_best==X_worst==|X| must leave X[d] exactly unchanged (r1={r1}, r2={r2})");
        }
    }

    #[test]
    fn matches_the_papers_worked_example_candidate1_iteration1() {
        // Rao (2016) Section 2.1, Tables 1-2: candidate 1, x1=-5, x2=18;
        // best (candidate 4) x1=-8, x2=7; worst (candidate 3) x1=70, x2=-6.
        // r1=0.58, r2=0.81 for x1; r1=0.92, r2=0.49 for x2.
        let x1_new = jaya_dim_step(-5.0, -8.0, 70.0, 0.58, 0.81);
        let x2_new = jaya_dim_step(18.0, 7.0, -6.0, 0.92, 0.49);
        assert!((x1_new - (-65.19)).abs() < 1e-9, "x1' should match the paper's -65.19: got {x1_new}");
        assert!((x2_new - 19.64).abs() < 1e-9, "x2' should match the paper's 19.64: got {x2_new}");
    }

    #[test]
    fn matches_the_papers_worked_example_shared_r_across_other_candidates() {
        // The paper's own worked example reuses the SAME r1/r2 pair for a
        // given dimension across EVERY candidate in the iteration -- this
        // is the verified loop-structure finding (see module doc). Confirm
        // by reproducing candidates 2, 3 and 4's new x1/x2 from Table 2
        // using the identical r1=0.58/r2=0.81 (x1) and r1=0.92/r2=0.49 (x2)
        // pairs used for candidate 1 above.
        let best = (-8.0, 7.0); // candidate 4
        let worst = (70.0, -6.0); // candidate 3

        // Candidate 2: x1=14, x2=63 -> expect -44.12, 45.29.
        let c2_x1 = jaya_dim_step(14.0, best.0, worst.0, 0.58, 0.81);
        let c2_x2 = jaya_dim_step(63.0, best.1, worst.1, 0.92, 0.49);
        assert!((c2_x1 - (-44.12)).abs() < 1e-9, "candidate 2 x1: got {c2_x1}");
        assert!((c2_x2 - 45.29).abs() < 1e-9, "candidate 2 x2: got {c2_x2}");

        // Candidate 3 (the worst itself): x1=70, x2=-6 -> expect 24.76, 0.8.
        let c3_x1 = jaya_dim_step(70.0, best.0, worst.0, 0.58, 0.81);
        let c3_x2 = jaya_dim_step(-6.0, best.1, worst.1, 0.92, 0.49);
        assert!((c3_x1 - 24.76).abs() < 1e-9, "candidate 3 x1: got {c3_x1}");
        assert!((c3_x2 - 0.8).abs() < 1e-9, "candidate 3 x2: got {c3_x2}");

        // Candidate 4 (the best itself): x1=-8, x2=7 -> expect -67.5, 13.37.
        let c4_x1 = jaya_dim_step(-8.0, best.0, worst.0, 0.58, 0.81);
        let c4_x2 = jaya_dim_step(7.0, best.1, worst.1, 0.92, 0.49);
        assert!((c4_x1 - (-67.5)).abs() < 1e-9, "candidate 4 x1: got {c4_x1}");
        assert!((c4_x2 - 13.37).abs() < 1e-9, "candidate 4 x2: got {c4_x2}");
    }

    // ---- worst_index ----

    #[test]
    fn worst_index_ties_break_to_higher_index() {
        // Two ties at the max (worst) fitness (20.0): indices 1 and 2.
        let fitness = vec![10.0, 20.0, 20.0, 5.0];
        assert_eq!(worst_index(&fitness), Some(2), "ties -> higher index must be picked as worst");
    }

    // ---- JayaGenerator ----

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
            JayaGenerator.generate(&pop, &mut ctx)
        };

        let a = run();
        let b = run();
        for (ga, gb) in a.iter().zip(b.iter()) {
            let (xa, xb) = (JayaGenerator::floats(ga), JayaGenerator::floats(gb));
            assert_eq!(xa, xb, "same seed must produce bit-identical offspring");
        }
    }

    #[test]
    fn draw_count_is_exactly_2_times_dim_independent_of_pop_size() {
        // Pinned: r1[d]/r2[d] drawn ONCE PER DIMENSION PER GENERATION,
        // shared across all agents -- so the draw count is 2*dim, NOT
        // 2*n*dim (unlike SCA's/GWO's per-(i,d) draws). Verify via
        // twin-stream raw replay, and separately confirm the SAME dim with
        // a DIFFERENT n consumes the identical number of draws.
        let dim = 3;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();

        for n in [2usize, 5, 9] {
            let pop = pop_nd(n, dim);
            let mut rng = RngStream::from_master(7, &[]);
            let rng_before = rng.clone();
            let mut evaluator = Evaluator::new(&p, 1000);
            let mut bb = Blackboard::new();
            {
                let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
                let off = JayaGenerator.generate(&pop, &mut ctx);
                assert_eq!(off.len(), n);
            }

            let mut twin = rng_before;
            for _ in 0..(2 * dim) { twin.next_f64(); }
            assert_eq!(rng.next_f64(), twin.next_f64(),
                "gen/jaya must consume exactly 2*dim RNG draws (n={n}, dim={dim}), independent of pop size");
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
            JayaGenerator.generate(&pop, &mut ctx)
        }));
        assert!(result.is_err(), "gen/jaya must reject pop_size < 2 at runtime as a backstop");
    }

    #[test]
    fn best_index_is_current_population_argmin_ties_lower_index() {
        let pop = Population {
            individuals: (0..5).map(|i| g(vec![i as f64])).collect(),
            fitness: vec![0.0, 5.0, 0.0, 3.0, 1.0],
        };
        assert_eq!(pop.best_index(), Some(0), "ties must resolve to the lower index");
        assert_eq!(worst_index(&pop.fitness), Some(1), "worst ties would resolve to the higher index (no tie here: index 1 is the unique max)");
    }
}
