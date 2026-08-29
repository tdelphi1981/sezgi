use sezgi_core::component::*;
use sezgi_core::problem::Population;
use sezgi_core::rng::RngStream;
use sezgi_core::space::{Block, BlockValues, Genotype, SearchSpace};

/// Harmony Search (Geem, Kim & Loganathan 2001, *Simulation* — "A new
/// heuristic optimization algorithm: harmony search") — a **labeled
/// metaphor preset**: faithful to the primary source's update rule, with a
/// pinned deterministic draw order and property tests, but NOT validated
/// against the paper's (or any other publication's) reported benchmark
/// numbers. See Weyland's (2010) analysis showing HS's harmony-memory-
/// consideration/pitch-adjustment mechanism is, component for component, a
/// special case of evolution strategies — cited here conservatively, as
/// background on why this is "labeled metaphor" rather than a mechanism
/// sezgi treats as novel.
///
/// **Pinned update rule** (part of the RNG-stream contract):
/// - Parameters (fixed, not tunable via `from_params`): `hmcr = 0.9`
///   (harmony memory considering rate), `par = 0.3` (pitch adjusting rate),
///   `bw[d] = 0.01·(hi[d] − lo[d])` (bandwidth, per-dimension, derived from
///   the block bounds — the paper's `bw` is a fixed absolute constant, but
///   sezgi's presets are used across arbitrarily-scaled search spaces, so
///   `bw` is expressed as a bounds-relative fraction here, same idiom as
///   `gen/cma`'s `sigma0` default `0.3·(hi − lo)`).
/// - Exactly ONE new harmony is generated per `generate` call, regardless of
///   `pop_size` (HMS, Harmony Memory Size) — SA's `gen/step` is the in-repo
///   precedent for a stage whose offspring count need not match `pop_size`
///   (there, `pop_size == 1` so the two coincide; here `pop_size` is HMS 30
///   but the offspring vector returned by `generate` still has length 1).
/// - Per dimension `d` (index order): draw `u1 = rng.next_f64()`.
///   - if `u1 < hmcr` (**memory consideration**): draw a memory index
///     `j = rng.next_below(pop_len)` (the project's uniform-index idiom,
///     see `de.rs`'s `pick_distinct`/`woa.rs`'s random-whale draw), take
///     `X_j[d]`; draw `u2 = rng.next_f64()`; if `u2 < par` (**pitch
///     adjustment**): draw `u3 = rng.next_f64()`;
///     `value += bw[d]·(2·u3 − 1)`.
///   - else (`u1 ≥ hmcr`, **random selection**): draw `u4 = rng.next_f64()`;
///     `value = lo[d] + u4·(hi[d] − lo[d])`.
/// - Draw order per `d`: `u1` first; then, on the memory branch, `j`, `u2`,
///   and (only if the pitch-adjustment sub-branch is taken) `u3`; on the
///   random branch, `u4`. This is pinned as the RNG-stream contract.
///
/// The per-dimension math is factored into [`hs_dim_step`], which is
/// unit-testable directly (the `hmcr = 1, par = 0` limit — pure memory
/// recombination — and the `hmcr = 0` limit — uniform random within bounds
/// — without needing to fake `Ctx`). Unlike [`gwo_dim_step`](super::gwo)/
/// [`woa_encircle_step`](super::woa)'s design (a fixed-size pre-drawn
/// `[f64; N]` slice), `hs_dim_step` takes `&mut RngStream` directly: the
/// number and kind of draws genuinely varies by branch (2, 3, or 2 draws —
/// see above), so a fixed-size slice would either over- or under-draw
/// depending on which branch is actually taken, breaking the pinned
/// draw-order contract. Unit tests construct a real
/// `RngStream::from_master(seed, &[])` and pass it directly, which stays
/// fully deterministic and testable without `Ctx`/`Evaluator`.
/// [`HsGenerator::generate`] is a thin loop over `d` that calls the helper
/// with that dimension's memory column and bounds.
///
/// Boundary handling and replacement are NOT part of this generator: the
/// `presets::harmony_search` preset reuses `boundary/clamp` (same as
/// `presets::gwo`/`presets::woa` — pitch adjustment can push a value outside
/// `[lo, hi]`) and the new `replace/worst-if-better` kind (see `replace.rs`
/// — no existing kind matched an in-place "replace the worst iff the
/// offspring is better" contract).
///
/// `min_pop = 1`: with a single harmony in memory, memory consideration
/// degenerates to always drawing `j = 0` (the sole member) — the algorithm
/// stays well-defined, just without any actual memory diversity, so there is
/// no structural reason to require more than one harmony (unlike
/// `gen/gwo`'s three leaders or `gen/woa`'s best-plus-one-other). The
/// canonical HMS is 30 (`presets::harmony_search`'s default).
pub fn hs_dim_step(
    hmcr: f64,
    par: f64,
    bw_d: f64,
    lo_d: f64,
    hi_d: f64,
    memory_d: &[f64],
    rng: &mut RngStream,
) -> f64 {
    let u1 = rng.next_f64();
    if u1 < hmcr {
        let j = rng.next_below(memory_d.len() as u64) as usize;
        let mut value = memory_d[j];
        let u2 = rng.next_f64();
        if u2 < par {
            let u3 = rng.next_f64();
            value += bw_d * (2.0 * u3 - 1.0);
        }
        value
    } else {
        let u4 = rng.next_f64();
        lo_d + u4 * (hi_d - lo_d)
    }
}

/// Per-coordinate `(lo, hi)` bounds, expanding each `Block::Float { lo, hi, n }`
/// to `n` repeated entries — matches the flat single-Float-block genotype
/// convention `gen/hs` uses (same idiom as `gen/cma`'s private
/// `coordinate_bounds`, duplicated here rather than shared, per the
/// project's "no cross-metaphor-file private imports" convention).
fn coordinate_bounds(space: &SearchSpace) -> Vec<(f64, f64)> {
    space.blocks().iter().flat_map(|b| match *b {
        Block::Float { lo, hi, n } => vec![(lo, hi); n],
        _ => unreachable!("gen/hs requires a float-only space (meta().supported_blocks)"),
    }).collect()
}

const HMCR: f64 = 0.9;
const PAR: f64 = 0.3;
const BW_FRACTION: f64 = 0.01;

pub struct HsGenerator;

impl HsGenerator {
    pub fn from_params(_p: &serde_json::Value) -> Result<Self, ComponentError> {
        // No tunable parameters (hmcr/par/bw are fixed, per the pinned spec).
        Ok(Self)
    }

    fn floats(g: &Genotype) -> &Vec<f64> {
        match &g.blocks[0] { BlockValues::Float(x) => x, _ => unreachable!() }
    }
}

impl Generator for HsGenerator {
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        assert!(!pop.is_empty(), "gen/hs requires a non-empty population (pop_size={})", pop.len());
        let dim = Self::floats(&pop.individuals[0]).len();
        let bounds = coordinate_bounds(ctx.space);

        let xs: Vec<f64> = (0..dim).map(|d| {
            let (lo_d, hi_d) = bounds[d];
            let bw_d = BW_FRACTION * (hi_d - lo_d);
            let memory_d: Vec<f64> = pop.individuals.iter().map(|g| Self::floats(g)[d]).collect();
            hs_dim_step(HMCR, PAR, bw_d, lo_d, hi_d, &memory_d, ctx.rng)
        }).collect();

        vec![Genotype { blocks: vec![BlockValues::Float(xs)] }]
    }

    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/hs", SupportedBlocks::Only(vec!["float"])).with_min_pop(1)
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_generator("gen/hs", |p| Ok(Box::new(HsGenerator::from_params(p)?)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::problem::{Problem, SphereShifted, Evaluator};
    use sezgi_core::state::Blackboard;

    fn g(xs: Vec<f64>) -> Genotype { Genotype { blocks: vec![BlockValues::Float(xs)] } }

    fn pop_nd(n: usize, dim: usize) -> Population {
        Population {
            individuals: (0..n).map(|i| g((0..dim).map(|d| (i * dim + d) as f64 * 0.37 - 1.0).collect())).collect(),
            fitness: (0..n).map(|i| (n - i) as f64).collect(),
        }
    }

    #[test]
    fn hmcr_one_par_zero_limit_is_pure_memory_pick() {
        // hmcr=1 => u1 < hmcr always (u1 in [0,1)), so the memory branch is
        // always taken; par=0 => u2 < par never holds (u2 in [0,1)), so no
        // pitch adjustment ever fires. The result must equal EXACTLY the
        // chosen memory column entry, regardless of bw/lo/hi/u3.
        let memory_d = [10.0, 20.0, 30.0];
        let mut rng = RngStream::from_master(7, &[]);
        for _ in 0..50 {
            let result = hs_dim_step(1.0, 0.0, 999.0, -1e9, 1e9, &memory_d, &mut rng);
            assert!(memory_d.contains(&result), "hmcr=1,par=0 must pick exactly a memory value, got {result}");
        }
    }

    #[test]
    fn hmcr_one_par_zero_property_every_memory_coordinate_reachable() {
        // Over enough draws, every row of a small memory column should be
        // selected at least once (uniform index draw over pop_len).
        let memory_d = [1.0, 2.0, 3.0, 4.0];
        let mut rng = RngStream::from_master(11, &[]);
        let mut seen = [false; 4];
        for _ in 0..500 {
            let result = hs_dim_step(1.0, 0.0, 0.0, -1.0, 1.0, &memory_d, &mut rng);
            let idx = memory_d.iter().position(|&v| v == result).expect("must be a memory value");
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&s| s), "every memory row should appear reachable: {seen:?}");
    }

    #[test]
    fn hmcr_zero_limit_is_uniform_in_bounds() {
        // hmcr=0 => u1 < hmcr never holds (u1 in [0,1)), so the random
        // branch always fires: value = lo + u4*(hi-lo), always within
        // [lo, hi), and (with overwhelming probability) not equal to any
        // memory entry.
        let memory_d = [0.0, 0.0, 0.0];
        let mut rng = RngStream::from_master(3, &[]);
        for _ in 0..50 {
            let result = hs_dim_step(0.0, 1.0, 1.0, -5.0, 5.0, &memory_d, &mut rng);
            assert!((-5.0..5.0).contains(&result), "hmcr=0 must be uniform in [lo,hi): got {result}");
        }
    }

    #[test]
    fn hmcr_zero_limit_exact_reconstruction_fixed_seed() {
        // Deterministic exact-value check: hmcr=0 always takes the random
        // branch, drawing exactly one f64 (u4) per call. Reconstruct the
        // expected value from a twin RngStream at the same seed.
        let mut rng = RngStream::from_master(99, &[]);
        let mut twin = rng.clone();
        let result = hs_dim_step(0.0, 1.0, 1.0, 2.0, 8.0, &[0.0], &mut rng);
        let _u1 = twin.next_f64(); // unconditional first draw, always fails u1 < hmcr=0.0
        let u4 = twin.next_f64();
        let expected = 2.0 + u4 * (8.0 - 2.0);
        assert_eq!(result, expected, "hmcr=0 branch must draw u1 then exactly u4, computing lo + u4*(hi-lo)");
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
            HsGenerator.generate(&pop, &mut ctx)
        };

        let a = run();
        let b = run();
        assert_eq!(a.len(), 1, "gen/hs must return exactly one new harmony");
        for (ga, gb) in a.iter().zip(b.iter()) {
            let (xa, xb) = (HsGenerator::floats(ga), HsGenerator::floats(gb));
            assert_eq!(xa, xb, "same seed must produce bit-identical offspring");
        }
    }

    #[test]
    fn draw_count_memory_branch_with_and_without_pitch_adjustment() {
        // Crafted case: dim=2, pop n=3, master seed=5. Trace the raw
        // RngStream sequence to determine, independently of the
        // implementation, which branch each dimension takes -- then
        // twin-stream replay that exact sequence and compare the NEXT draw.
        let n = 3; let dim = 2;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_nd(n, dim);

        let mut rng = RngStream::from_master(5, &[]);
        let rng_before = rng.clone();
        let mut evaluator = Evaluator::new(&p, 1000);
        let mut bb = Blackboard::new();
        let off = {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
            HsGenerator.generate(&pop, &mut ctx)
        };
        assert_eq!(off.len(), 1);

        // Determine the actual branch sequence taken by tracing a twin
        // stream by hand, one draw at a time, mirroring the pinned draw
        // order documented on hs_dim_step (not read from generate()'s code).
        let mut twin = rng_before;
        for _ in 0..dim {
            let u1 = twin.next_f64();
            if u1 < HMCR {
                let _j = twin.next_below(n as u64);
                let u2 = twin.next_f64();
                if u2 < PAR {
                    let _u3 = twin.next_f64();
                }
            } else {
                let _u4 = twin.next_f64();
            }
        }

        assert_eq!(rng.next_f64(), twin.next_f64(),
            "gen/hs must consume exactly the pinned draw sequence for this crafted case");
    }

    #[test]
    fn min_pop_one_does_not_panic() {
        let n = 1; let dim = 3;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_nd(n, dim);
        let mut evaluator = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let off = HsGenerator.generate(&pop, &mut ctx);
        assert_eq!(off.len(), 1);
    }
}
