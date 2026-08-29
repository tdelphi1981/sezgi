use sezgi_core::component::*;
use sezgi_core::problem::Population;
use sezgi_core::space::{Block, BlockValues, Genotype, SearchSpace};

/// Grasshopper Optimisation Algorithm (Saremi, Mirjalili & Lewis 2017,
/// "Grasshopper optimisation algorithm: theory and application", *Advances
/// in Engineering Software*, 105, 30-47) — a **labeled metaphor preset**:
/// faithful to the primary source's Eq. (2.7)/(2.8) and its reference
/// implementation's distance handling (see IMPLEMENTER-VERIFY note below),
/// with a pinned deterministic arithmetic order and property tests, but NOT
/// validated against the paper's (or any other publication's) reported
/// benchmark numbers.
///
/// **IMPLEMENTER-VERIFY note (distance normalization, resolved by
/// verification, not by falling back to the uncertain default):** the
/// paper's Eq. (2.7) formula, read literally, applies the social force
/// `s(.)` directly to the raw Euclidean distance `d_ij`. The reference
/// MATLAB implementation (Saremi/Mirjalili/Lewis, MATLAB Central File
/// Exchange #61421, linked from the paper) instead maps `d_ij` into the
/// bounded interval `[2, 4)` before applying `s(.)`, via `xj_xi = 2 +
/// rem(d_ij, 2)`. This was confirmed in this session (not merely recalled)
/// by fetching `thieu1995/mealpy`'s `OriginalGOA` — an actively-maintained,
/// widely-used reimplementation whose docstring links directly to the same
/// MATLAB File Exchange page and whose code comments cite the paper's own
/// equation numbers (`# Eq.(2.3)`, `# Eq.(2.7)`, `# Eq.(2.8)`) — which
/// contains the line `xj_xi = 2 + np.remainder(dist, 2)  # |xjd - xid| in
/// Eq. (2.7)`, i.e. exactly the `[2, 4)` mapping. sezgi therefore implements
/// `dist_term_ij = 2.0 + (d_ij % 2.0)` as the VERIFIED reference-
/// implementation semantics (not the brief's uncertain fallback route),
/// tagged `// sezgi simplification:` below only because it is a genuine
/// departure from the paper's literal prose formula (which never mentions
/// bounding the distance) — the reference code, not the prose, is what
/// sezgi pins here, consistent with the project's "faithful to the primary
/// source" tier meaning "faithful to what the source's own authors actually
/// shipped," not to a possibly-idealized reading of the prose alone. Note
/// also that `mealpy`'s `OriginalGOA` additionally multiplies each new
/// position by an extra `c * randn(dim)` factor not present in the paper's
/// Eq. (2.7) prose and not part of sezgi's pinned spec (the task brief's
/// pinned update rule has NO RNG in the core update); sezgi does not
/// reproduce that extra term, since the pinned spec is authoritative here
/// and an RNG-free deterministic core update is itself part of the pinned
/// RNG-stream contract (see [`GoaGenerator::generate`]'s zero-draw
/// contract, verified by `zero_rng_consumption_twin_stream` below).
///
/// **Pinned update rule** (part of the RNG-stream contract — note this
/// generator draws NOTHING from `ctx.rng`; the pin is purely an f64
/// arithmetic-order contract):
/// - `c = cmax − progress·(cmax − cmin)`, `cmax = 1.0`, `cmin = 1e-5`, where
///   `progress = ctx.eval.used() / ctx.eval.budget()` (clamped to `[0, 1]`;
///   `0` if `budget == 0`), computed **once per `generate` call**.
/// - Social force `s(r) = f·e^{−r/l} − e^{−r}`, `f = 0.5`, `l = 1.5` (Eq.
///   2.3).
/// - Best-so-far individual `X_best`: fitness argmin over the population,
///   ties → lower index (same convention as `gwo.rs`/`woa.rs`/`cs.rs`, via
///   [`Population::best_index`]), computed once per `generate` call.
/// - For each grasshopper `i` (population order), each dimension `d` (index
///   order): `X_i'[d] = c·Σ_{j≠i, j in ascending order} [ c·(hi[d]−lo[d])/2
///   · s(dist_term_ij) · (X_j[d]−X_i[d]) / (d_ij + ε) ] + X_best[d]`, with
///   `d_ij` the full (all-dimensions) Euclidean distance between
///   grasshoppers `i` and `j`, `ε = 1e-12` guarding the division
// sezgi simplification: `ε = 1e-12` added to guard the division by `d_ij`
// against a zero-distance pair (two grasshoppers occupying the exact same
// point) -- not present in the paper's prose, which does not address the
// degenerate case; same idiom as `de.rs`/other generators' zero-guards.
///   , and `dist_term_ij = 2.0 + (d_ij mod 2.0)` per the verified reference
///   semantics above
// sezgi simplification: `dist_term_ij` maps the raw Euclidean distance into
// the bounded [2, 4) interval before applying `s(.)`, per the reference
// MATLAB implementation (not the paper's literal prose) -- see the
// IMPLEMENTER-VERIFY note above for the verification source.
///   . The `f64` sum is accumulated in `j`-ascending order for each fixed
///   `(i, d)` — THIS accumulation order is the pin (floating-point addition
///   is not associative, so a different summation order could produce a
///   bit-different result even though the mathematical value is the same).
/// - No RNG draws anywhere in this generator: GOA's update is fully
///   deterministic given the population and `c` (unlike `gwo.rs`'s/
///   `woa.rs`'s `A`/`C`/branch draws or `hs.rs`'s/`cs.rs`'s stochastic
///   terms) — the pin here is the ZERO-draw contract plus the arithmetic
///   order above, verified by `zero_rng_consumption_twin_stream` below.
///
/// The per-pair, per-dimension math is factored into [`goa_s`] (the social
/// force) and [`goa_pair_term`] (one `j`'s contribution to `X_i'[d]`,
/// including the INNER `c` factor from the pinned Σ) so both can be
/// unit-tested directly — e.g. the pairwise-force antisymmetry property
/// (swapping `i↔j` negates the numerator `(X_j[d]−X_i[d])` exactly while
/// `d_ij`, `dist_term_ij` and `s(.)` stay identical, since both are
/// symmetric in `i↔j`) — without needing to fake `Ctx`/`Evaluator`.
/// [`GoaGenerator::generate`] is a thin loop over `(i, d, j)` that computes
/// `d_ij` once per `(i, j)` pair (not per `(i, j, d)` — `d_ij` does not
/// depend on `d`, only the summation ORDER over `j` for fixed `(i, d)` is
/// pinned) and calls the helpers.
///
/// `min_pop = 2`: with a single grasshopper, the `Σ_{j≠i}` sum is empty for
/// every `i` and every offspring collapses trivially to `X_best` (itself,
/// since the sole grasshopper IS `X_best`) — degenerate, same rationale as
/// `woa.rs`'s/`cs.rs`'s `min_pop = 2` (needs a best-so-far distinct from
/// `i` for the swarm interaction term to be meaningful).
///
/// Boundary handling and replacement are NOT part of this generator: the
/// `presets::goa` preset reuses `boundary/clamp` (same as `gwo`/`woa`/
/// `harmony_search`/`cuckoo_search` — the swarm term can push a coordinate
/// outside `[lo, hi]`) and GWO's `replace/generational` kind (see
/// `replace.rs`) for GOA's unconditional generational replacement (GOA is
/// non-elitist by construction in the reference algorithm, same rationale
/// as `gwo`/`woa`/`pso`/`cma-es`).
pub const GOA_F: f64 = 0.5;
pub const GOA_L: f64 = 1.5;
pub const GOA_C_MAX: f64 = 1.0;
pub const GOA_C_MIN: f64 = 1e-5;
const EPS: f64 = 1e-12;

/// Social force function `s(r) = f·e^{−r/l} − e^{−r}` (Eq. 2.3), `f = 0.5`,
/// `l = 1.5` (fixed, per the pinned spec).
pub fn goa_s(r: f64) -> f64 {
    GOA_F * (-r / GOA_L).exp() - (-r).exp()
}

/// One pairwise contribution to grasshopper `i`'s dimension-`d` update from
/// grasshopper `j` — the inner term of the pinned Σ, INCLUDING the inner
/// `c` factor (see the module doc's pinned update rule; the caller applies
/// the outer `c` to the accumulated sum separately). `d_ij` is the full
/// (all-dimensions) Euclidean distance between grasshopper `i` and `j`.
///
/// This function is antisymmetric under swapping `(x_i_d, x_j_d)` (with
/// `d_ij` held fixed, since Euclidean distance is itself symmetric in
/// `i↔j`): only the `(x_j_d − x_i_d)` numerator changes sign; `dist_term`
/// and `s(.)` are unaffected.
pub fn goa_pair_term(c: f64, half_range_d: f64, x_i_d: f64, x_j_d: f64, d_ij: f64) -> f64 {
    // sezgi simplification: dist_term maps the raw Euclidean distance into
    // the bounded [2, 4) interval before applying s(.), per the reference
    // MATLAB implementation -- see the module doc's IMPLEMENTER-VERIFY note.
    let dist_term = 2.0 + (d_ij % 2.0);
    let s = goa_s(dist_term);
    // sezgi simplification: `+ EPS` guards against d_ij == 0 (two
    // grasshoppers at the exact same point) -- not addressed by the paper's
    // prose.
    c * half_range_d * s * (x_j_d - x_i_d) / (d_ij + EPS)
}

fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| (ai - bi) * (ai - bi)).sum::<f64>().sqrt()
}

/// Per-coordinate `(lo, hi)` bounds, expanding each `Block::Float { lo, hi, n }`
/// to `n` repeated entries — matches the flat single-Float-block genotype
/// convention `gen/goa` uses (same idiom as `hs.rs`'s/`cma.rs`'s private
/// `coordinate_bounds`, duplicated here rather than shared, per the
/// project's "no cross-metaphor-file private imports" convention).
fn coordinate_bounds(space: &SearchSpace) -> Vec<(f64, f64)> {
    space.blocks().iter().flat_map(|b| match *b {
        Block::Float { lo, hi, n } => vec![(lo, hi); n],
        _ => unreachable!("gen/goa requires a float-only space (meta().supported_blocks)"),
    }).collect()
}

pub struct GoaGenerator;

impl GoaGenerator {
    pub fn from_params(_p: &serde_json::Value) -> Result<Self, ComponentError> {
        // No tunable parameters (f/l/cmax/cmin are fixed, per the pinned spec).
        Ok(Self)
    }

    fn floats(g: &Genotype) -> &Vec<f64> {
        match &g.blocks[0] { BlockValues::Float(x) => x, _ => unreachable!() }
    }
}

impl Generator for GoaGenerator {
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        assert!(pop.len() >= 2, "gen/goa requires a population of at least 2 (pop_size={})", pop.len());
        let n = pop.len();
        let dim = Self::floats(&pop.individuals[0]).len();
        let bounds = coordinate_bounds(ctx.space);
        let half_range: Vec<f64> = bounds.iter().map(|&(lo, hi)| (hi - lo) / 2.0).collect();

        let used = ctx.eval.used() as f64;
        let budget = ctx.eval.budget() as f64;
        let progress = if budget > 0.0 { (used / budget).clamp(0.0, 1.0) } else { 0.0 };
        let c = GOA_C_MAX - progress * (GOA_C_MAX - GOA_C_MIN);

        // Best-so-far: fitness argmin, ties -> lower index -- pinned,
        // computed once per generate() call (no RNG draws anywhere in this
        // generator -- see module doc's zero-draw contract).
        let best = pop.best_index().unwrap_or(0);
        let x_best = Self::floats(&pop.individuals[best]);

        let xs_all: Vec<&Vec<f64>> = (0..n).map(|k| Self::floats(&pop.individuals[k])).collect();

        (0..n).map(|i| {
            let x_i = xs_all[i];
            // d_ij does not depend on d -- computed once per (i, j) pair,
            // in ascending-j order, and reused for every dimension. The
            // pinned arithmetic-order contract is the ascending-j
            // SUMMATION order per (i, d) below, not when d_ij itself is
            // computed.
            let dists: Vec<f64> = (0..n).map(|j| euclidean_dist(x_i, xs_all[j])).collect();
            let new_x: Vec<f64> = (0..dim).map(|d| {
                let mut sum = 0.0_f64;
                for (j, &x_j) in xs_all.iter().enumerate() {
                    if j == i { continue; }
                    sum += goa_pair_term(c, half_range[d], x_i[d], x_j[d], dists[j]);
                }
                c * sum + x_best[d]
            }).collect();
            Genotype { blocks: vec![BlockValues::Float(new_x)] }
        }).collect()
    }

    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/goa", SupportedBlocks::Only(vec!["float"])).with_min_pop(2)
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_generator("gen/goa", |p| Ok(Box::new(GoaGenerator::from_params(p)?)));
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
    fn pair_term_antisymmetric_and_matches_hand_derivation() {
        // Hand-derived 2-grasshopper, 1-D fixture:
        //   c = 0.3 (arbitrary mid-range value -- NOT the preset's c=1.0
        //   default, chosen specifically so the c*c-appears-twice structure
        //   of the pinned formula is actually exercised by a non-trivial
        //   value), half_range_d = (hi-lo)/2 = 2.5 (hi=5.0, lo=0.0).
        //   Grasshopper 0 at x=1.0, grasshopper 1 at x=4.0.
        //   d_01 = |4.0 - 1.0| = 3.0 (Euclidean distance in 1-D).
        //   dist_term = 2.0 + (3.0 mod 2.0) = 2.0 + 1.0 = 3.0.
        //   s(3.0) = f*e^{-3.0/1.5} - e^{-3.0} = 0.5*e^{-2.0} - e^{-3.0}
        //          ~= 0.5*0.135335 - 0.049787 ~= 0.017881 (positive: the
        //          attraction term dominates the repulsion term at r=3.0,
        //          since GOA's s(.) has a comfort zone at r0~=1.42 -- see
        //          the paper's Fig. 2 -- and 3.0 is past it into net
        //          attraction).
        //   term_{i=0, from j=1} = c*half_range_d*s(3.0)*(x1-x0)/(d01+eps)
        //                        = 0.3*2.5*s(3.0)*3.0/(3.0+1e-12)
        //   term_{i=1, from j=0} = c*half_range_d*s(3.0)*(x0-x1)/(d01+eps)
        //                        = 0.3*2.5*s(3.0)*(-3.0)/(3.0+1e-12)
        //                        = -term_{i=0, from j=1}  (only the
        //                          numerator's sign flips; d_ij, dist_term
        //                          and s(.) are all symmetric in i<->j).
        let c = 0.3;
        let half_range_d = 2.5;
        let x0 = 1.0_f64;
        let x1 = 4.0_f64;
        let d01 = (x1 - x0).abs(); // 3.0, Euclidean distance in 1-D

        let term_0_from_1 = goa_pair_term(c, half_range_d, x0, x1, d01);
        let term_1_from_0 = goa_pair_term(c, half_range_d, x1, x0, d01);

        // Exact hand-derived value, computed independently (literal f64
        // arithmetic, not by calling goa_pair_term/goa_s) from the pinned
        // formula in the derivation above.
        let dist_term = 2.0 + (d01 % 2.0);
        let s = 0.5 * (-dist_term / 1.5_f64).exp() - (-dist_term).exp();
        let expected_0_from_1 = c * half_range_d * s * (x1 - x0) / (d01 + 1e-12);
        assert_eq!(term_0_from_1, expected_0_from_1,
            "pair term must match the hand-derived pinned-formula expression bit-for-bit");

        assert_eq!(term_1_from_0, -term_0_from_1,
            "pairwise force must be exactly antisymmetric under i<->j swap (same d_ij/dist_term/s, numerator sign flips)");
    }

    #[test]
    fn two_grasshopper_generate_matches_full_hand_derivation() {
        // Full-pipeline hand derivation: n=2, dim=1, bounds [0.0, 5.0]
        // (half_range = 2.5), grasshopper 0 at x=1.0 (worse fitness),
        // grasshopper 1 at x=4.0 (better fitness, so X_best = [4.0]).
        // budget=1000, used=0 => progress=0 => c = cmax = 1.0.
        let dim = 1;
        let p = SphereShifted::new(vec![0.0; dim], 0.0, 5.0);
        let space = p.space();
        let pop = Population {
            individuals: vec![g(vec![1.0]), g(vec![4.0])],
            fitness: vec![5.0, 1.0], // grasshopper 1 is best (lower fitness)
        };

        let mut evaluator = Evaluator::new(&p, 1000);
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();
        let off = {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
            GoaGenerator.generate(&pop, &mut ctx)
        };
        assert_eq!(off.len(), 2);

        // Hand-derived expected values (c=1.0, X_best=[4.0]):
        //   d_01 = |4.0-1.0| = 3.0; dist_term = 2.0 + (3.0 mod 2.0) = 3.0
        //   s(3.0) = 0.5*e^{-2.0} - e^{-3.0}
        //   term_0_from_1 = 1.0*2.5*s(3.0)*(4.0-1.0)/(3.0+1e-12)
        //   X_0'[0] = 1.0*term_0_from_1 + 4.0
        //   term_1_from_0 = 1.0*2.5*s(3.0)*(1.0-4.0)/(3.0+1e-12) = -term_0_from_1
        //   X_1'[0] = 1.0*term_1_from_0 + 4.0 = 4.0 - term_0_from_1
        let d01 = 3.0_f64;
        let dist_term = 2.0 + (d01 % 2.0);
        let s = 0.5 * (-dist_term / 1.5_f64).exp() - (-dist_term).exp();
        let term_0_from_1 = 1.0 * 2.5 * s * (4.0 - 1.0) / (d01 + 1e-12);
        let expected_x0 = 1.0 * term_0_from_1 + 4.0;
        let expected_x1 = 1.0 * (-term_0_from_1) + 4.0;

        assert_eq!(GoaGenerator::floats(&off[0])[0], expected_x0,
            "grasshopper 0's full update must match the hand-derived value bit-for-bit");
        assert_eq!(GoaGenerator::floats(&off[1])[0], expected_x1,
            "grasshopper 1's full update must match the hand-derived value bit-for-bit");
        // The two offspring's displacement from X_best must be exactly
        // antisymmetric (mirrors the pairwise-force antisymmetry property
        // through the full generate() pipeline, not just the pure helper).
        assert_eq!(expected_x0 - 4.0, -(expected_x1 - 4.0));
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
            GoaGenerator.generate(&pop, &mut ctx)
        };

        let a = run();
        let b = run();
        for (ga, gb) in a.iter().zip(b.iter()) {
            let (xa, xb) = (GoaGenerator::floats(ga), GoaGenerator::floats(gb));
            assert_eq!(xa, xb, "same seed must produce bit-identical offspring");
        }
    }

    #[test]
    fn zero_rng_consumption_twin_stream() {
        // GOA's core update draws NOTHING from ctx.rng (see module doc's
        // zero-draw contract). Twin stream: clone rng before generate(),
        // run generate() on the real stream, then compare the NEXT draw
        // from each stream -- if generate() consumed ANY draws, the two
        // streams would (with overwhelming probability) diverge.
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
            let _ = GoaGenerator.generate(&pop, &mut ctx);
        }

        let mut twin = rng_before;
        assert_eq!(rng.next_f64(), twin.next_f64(),
            "gen/goa must consume exactly ZERO RNG draws -- the core update is fully deterministic given the population");
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
            GoaGenerator.generate(&pop, &mut ctx)
        }));
        assert!(result.is_err(), "gen/goa must reject pop_size < 2 at runtime as a backstop");
    }
}
