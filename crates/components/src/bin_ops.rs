use sezgi_core::component::*;
use sezgi_core::problem::Population;
use sezgi_core::rng::RngStream;
use sezgi_core::space::{Block, BlockValues, Genotype, SearchSpace};

/// Binary operators: `gen/bin-2pt`, `gen/bit-flip`, `gen/ga-bin` -- the
/// binary-genotype counterpart to `perm.rs`'s permutation triple
/// (`gen/ox`/`gen/perm-swap`/`gen/ga-perm`) and `ga.rs`'s Float pair
/// (`gen/ga-real`). Until this module, `Block::Binary` spaces were reachable
/// ONLY through NSGA-II's dedicated binary loop (`nsga2.rs`'s
/// `nsga2_run_binary`, M3-7 Task 3) -- there was no single-objective,
/// registry/preset path for a binary search space. This module (M3-8 Task
/// 2) adds one, mirroring `perm.rs`'s own shape exactly: a standalone
/// crossover-only generator, a standalone mutation-only generator, and a
/// fused crossover+mutation generator for the `ga_bin` preset.
///
/// ## PROVENANCE -- moved, not re-derived
///
/// The two pure cores below ([`bin_cross_pair`], [`bin_flip_mutation`],
/// plus the [`rnd`] helper [`bin_cross_pair`] depends on) are **relocated
/// verbatim** from `nsga2.rs`, where M3-7 Task 3 originally transcribed them
/// from the KanGAL reference C (`darnir/nsga2`'s `crossover.c`'s `bincross`
/// and `mutation.c`'s `bin_mutate_ind`, sha-verified against the GitHub
/// API's own blob shas -- see `nsga2.rs`'s module doc, "Binary genotype
/// path" section, and `docs/DECISIONS.md`'s M3-2/M3-7 records for the full
/// quoted C and the sha table). No logic changed in the move: same
/// draw-order, same branch structure, same numeric outputs for the same
/// seed -- `nsga2.rs`'s own frozen test suite (including its hand-traced
/// `bin_cross_pair`/`bin_flip_mutation` fixtures and its M3-7/M3-8
/// constrained-binary regression golden) is the acceptance evidence: it
/// passes UNMODIFIED after `nsga2.rs` is re-pointed to call this module's
/// copies instead of defining its own (see "The seam" below).
///
/// **Two-point crossover** (`bincross`): per call, ONE gate draw
/// (`rand <= p_c`); on failure, both children are verbatim copies of their
/// respective parent, no further draws. On success: TWO more draws,
/// `site1 = rnd(0, n-1)` then `site2 = rnd(0, n-1)` (via [`rnd`], the SAME
/// "no draw when `low >= high`" helper KanGAL's own `rnd()` implements --
/// this degenerates a single-bit block's crossover to a no-op even when the
/// gate passes, since both site draws are skipped), sorted ascending with NO
/// further draw (compare-and-swap only). The genotype is partitioned into
/// three segments by `[site1, site2)`: `[0,site1)` and `[site2,n)` come from
/// each child's OWN parent, the MIDDLE segment `[site1,site2)` is SWAPPED.
///
/// **Bit-flip mutation** (`bin_mutate_ind`): per bit, in order, ONE draw
/// (`prob <= p_m`); on success, flip the bit; on failure, leave it
/// unchanged. No second draw per bit (unlike Float's `polynomial_mutation`,
/// which draws a gate then a magnitude) -- a bit-flip needs no magnitude.
///
/// ## The seam: what moved, what stayed, how `nsga2.rs` calls it now
///
/// [`rnd`], [`bin_cross_pair`], [`bin_flip_mutation`] moved here, `pub`/
/// `pub(crate)` exactly as they were in `nsga2.rs` (see each function's own
/// doc for the visibility rationale). `nsga2.rs`'s multi-block, MO-specific
/// wrappers -- `bin_cross_genome`/`bin_mutate_genome` (which loop this
/// module's per-block functions over `Vec<Vec<bool>>`, one call per KanGAL
/// `nbin` variable) and `shuffle_two_interleaved` (the representation-
/// agnostic tournament-pairing shuffle, which also calls [`rnd`]) -- STAYED
/// in `nsga2.rs`, now re-pointed via `use crate::bin_ops::{bin_cross_pair,
/// bin_flip_mutation, rnd};` at that module's top, so every existing
/// caller/test inside `nsga2.rs` (which reaches these three names through
/// its own `use super::*;` in `mod tests`) resolves to the exact same
/// functions with zero test-file edits. This is a pure DRY relocation, not
/// a rewrite: `bin_cross_genome`/`bin_mutate_genome` are unchanged one-line-
/// per-block loops around the (also unchanged) per-block cores.
///
/// `nsga2.rs`'s own binary genotype handles MULTIPLE `Block::Binary` blocks
/// per genome (one call per KanGAL `nbin` index) because NSGA-II's MO
/// surface supports arbitrarily many blocks. This module's OWN generators
/// (below) instead follow `perm.rs`'s/`ga.rs`'s established single-objective
/// convention: exactly ONE block, read via `ctx.space.blocks()[0]`
/// ([`bin_dim`]/[`bin_values`], mirroring `perm.rs`'s `perm_dim`/
/// `perm_values` and `ga.rs`'s own direct `blocks()[0]` match) -- consistent
/// with `gen/ga-real` (Float, one block) and `gen/ga-perm` (Permutation, one
/// block), not with `nsga2_run`'s multi-block MO path. A general multi-block
/// binary composition, if ever needed, is out of scope here (M3-8 Task 5's
/// `gen/compound` may revisit it).
///
/// ## `gen/bin-2pt` / `gen/ga-bin`'s pair loop (mirrors `gen/ox`/
/// `gen/ga-perm` exactly)
///
/// Tournament selection ([`crate::select::tournament`], shared with
/// `gen/ga-real`'s/`gen/ox`'s/`gen/ga-perm`'s own generators): draw a
/// candidate, then `tournament_k - 1` more, keeping the best (lowest
/// fitness) seen. Per pair: tournament for parent 1 (`tournament_k` draws of
/// `next_below(pop.len())`), THEN tournament for parent 2 (`tournament_k`
/// more draws), THEN [`bin_cross_pair`] directly -- **structural difference
/// from `gen/ox`**: `gen/ox` draws its OWN external `pc` gate BEFORE calling
/// `ox_pair`, because OX1's own source (Eiben & Smith) has no internal gate.
/// KanGAL's `bincross` is different: the `p_c` gate is INSIDE the per-block
/// crossover routine itself (module doc above, "Two-point crossover"), so
/// [`bin_cross_pair`] already draws its own gate -- callers here do NOT draw
/// a second, redundant gate. `gen/ga-bin` additionally mutates both children
/// via [`bin_flip_mutation`] (`c1` then `c2`, each already internally gated
/// per bit) after crossover, mirroring `gen/ga-perm`'s own
/// crossover-then-mutate-both-children shape.
///
/// ## `gen/bit-flip`'s per-individual loop
///
/// Population order (mirrors `gen/perm-swap`'s own per-individual loop
/// shape), but with NO outer per-individual gate: [`bin_flip_mutation`]
/// already gates PER BIT internally (module doc above), so a redundant
/// per-individual gate on top of it would double-gate and is not part of
/// the KanGAL source (`bin_mutate_ind` has no individual-level gate, only
/// the per-bit one) -- unlike `gen/perm-swap`'s own `p_m` (per-Eiben&Smith,
/// PER-INDIVIDUAL semantics for a permutation-length mutation).
///
/// ## Defaults
///
/// **`p_c` (both `gen/bin-2pt` and `gen/ga-bin`): `0.9`.** VERIFIED from the
/// KanGAL convention, not invented and NOT `gen/ox`'s `0.8` (a DIFFERENT,
/// Eiben & Smith SS13.3-derived default for a DIFFERENT representation/
/// operator pair, permutation OX -- unrelated to KanGAL). Source: Deb,
/// Pratap, Agarwal & Meyarivan (2002), Sec. IV.A, "The crossover probability
/// of p_c = 0.9 ... are used" (quoted verbatim in `nsga2.rs`'s own module
/// doc, "Defaults" section) -- the SAME value `nsga2.rs`'s `Nsga2Config`
/// uses for BOTH its `p_c` (real-coded) and `p_c_bin` (binary) fields in
/// every example/test config in that file (`p_c: 0.9, ..., p_c_bin: 0.9`
/// appears throughout `nsga2.rs`'s own tests) -- `Nsga2Config.p_c_bin` itself
/// carries NO in-code default (a REQUIRED field, per that module's own doc:
/// "the paper itself never splits the CROSSOVER probability's default by
/// representation... so there is no verified auto-default formula"), but
/// `0.9` is the value the paper's own Sec. IV.A prescribes and every
/// existing NSGA-II binary config in this repo actually uses -- the correct
/// KanGAL-convention default to bake into `gen/bin-2pt`/`gen/ga-bin` here,
/// where (unlike `Nsga2Config`) a registry component DOES need an in-code
/// default for an omitted param.
///
/// **`p_m` (`gen/bit-flip` and `gen/ga-bin`): `1/L`**, `L` = the space's
/// single `Block::Binary`'s bit count (`ctx.space.blocks()[0]`'s `n`).
/// Source: the SAME Sec. IV.A sentence quoted above, continued: "...and a
/// mutation probability of p_m = 1/n or 1/l (where n is the number of
/// decision variables for real-coded GAs and l is the string length for
/// binary-coded GAs)" -- the exact formula `nsga2.rs`'s own `p_m_bin: None`
/// resolution already uses (`1/l`, `l` = total flattened bit count).
///
/// **`tournament_k`: `2`**, mirroring `gen/ga-real`'s/`gen/ox`'s/
/// `gen/ga-perm`'s own default.
/// `rand.c`'s `rnd(low, high)`: a uniform integer in `[low, high]`
/// inclusive, via `low + floor(U(0,1) * (high-low+1))` clamped down to
/// `high`. **No draw at all** when `low >= high` (moved verbatim from
/// `nsga2.rs`, where it is also used by the representation-agnostic
/// tournament-pairing shuffle, `shuffle_two_interleaved` -- `pub(crate)` so
/// that caller can still reach it via `use crate::bin_ops::rnd;`).
pub(crate) fn rnd(low: usize, high: usize, rng: &mut RngStream) -> usize {
    if low >= high {
        return low;
    }
    let width = (high - low + 1) as f64;
    let mut res = low + (rng.next_f64() * width).floor() as usize;
    if res > high {
        res = high;
    }
    res
}

/// KanGAL reference C's `crossover.c`, `bincross`, on ONE binary block (a
/// single pair's worth of bits). See the module doc's "PROVENANCE" section
/// for the verified quote and draw structure. `p1`/`p2` must be the same
/// length. `pub`: both a registered-component internal (used by
/// [`BinTwoPointGenerator`]/[`GaBinGenerator`] below) and `nsga2.rs`'s own
/// `bin_cross_genome` reuse this directly, per-block, in a loop.
pub fn bin_cross_pair(p1: &[bool], p2: &[bool], p_c: f64, rng: &mut RngStream) -> (Vec<bool>, Vec<bool>) {
    debug_assert_eq!(p1.len(), p2.len(), "bin_cross_pair: parent bit vectors must have equal length");
    let n = p1.len();

    // Per-block gate: ONE draw.
    if rng.next_f64() > p_c {
        return (p1.to_vec(), p2.to_vec());
    }

    let hi = n.saturating_sub(1);
    let mut site1 = rnd(0, hi, rng);
    let mut site2 = rnd(0, hi, rng);
    if site1 > site2 {
        std::mem::swap(&mut site1, &mut site2);
    }

    let mut c1 = Vec::with_capacity(n);
    let mut c2 = Vec::with_capacity(n);
    for j in 0..n {
        if j < site1 || j >= site2 {
            c1.push(p1[j]);
            c2.push(p2[j]);
        } else {
            c1.push(p2[j]);
            c2.push(p1[j]);
        }
    }
    (c1, c2)
}

/// KanGAL reference C's `mutation.c`, `bin_mutate_ind`, on ONE binary block,
/// in place. See the module doc's "PROVENANCE" section for the verified
/// quote and draw structure. `pub`: both a registered-component internal
/// (used by [`BitFlipGenerator`]/[`GaBinGenerator`] below) and `nsga2.rs`'s
/// own `bin_mutate_genome` reuse this directly, per-block, in a loop.
pub fn bin_flip_mutation(x: &mut [bool], p_m: f64, rng: &mut RngStream) {
    for bit in x.iter_mut() {
        if rng.next_f64() <= p_m {
            *bit = !*bit;
        }
    }
}

fn bin_dim(space: &SearchSpace) -> usize {
    let Block::Binary { n } = space.blocks()[0] else {
        unreachable!("binary operators require a single Binary block")
    };
    n
}

fn bin_values(g: &Genotype) -> &Vec<bool> {
    match &g.blocks[0] {
        BlockValues::Bin(xs) => xs,
        _ => unreachable!("binary operators require a Binary block value"),
    }
}

pub struct BinTwoPointGenerator {
    pub tournament_k: usize,
    pub p_c: f64,
}

impl BinTwoPointGenerator {
    pub fn from_params(p: &serde_json::Value) -> Result<Self, ComponentError> {
        let err = |reason: String| ComponentError::InvalidParams { kind: "gen/bin-2pt".into(), reason };
        let g = Self {
            tournament_k: p.get("tournament_k").and_then(|v| v.as_u64()).unwrap_or(2) as usize,
            p_c: crate::params::resolve(p, "p_c", "pc").and_then(|v| v.as_f64()).unwrap_or(0.9),
        };
        if g.tournament_k == 0 { return Err(err("tournament_k must be >= 1".into())); }
        if !(0.0..=1.0).contains(&g.p_c) { return Err(err(format!("p_c outside [0,1]: {}", g.p_c))); }
        Ok(g)
    }
}

impl Generator for BinTwoPointGenerator {
    /// Pair loop mirroring `gen/ox`'s own EXACTLY, minus the external `pc`
    /// gate (see this module's doc: [`bin_cross_pair`] already gates
    /// internally, per KanGAL's `bincross`).
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        assert!(pop.len() >= 2, "gen/bin-2pt requires a population of at least 2 (pop_size={})", pop.len());
        let mut out = Vec::with_capacity(pop.len());
        while out.len() < pop.len() {
            let i1 = crate::select::tournament(pop, self.tournament_k, ctx.rng);
            let i2 = crate::select::tournament(pop, self.tournament_k, ctx.rng);
            let p1 = bin_values(&pop.individuals[i1]).clone();
            let p2 = bin_values(&pop.individuals[i2]).clone();
            let (c1, c2) = bin_cross_pair(&p1, &p2, self.p_c, ctx.rng);
            out.push(Genotype { blocks: vec![BlockValues::Bin(c1)] });
            if out.len() < pop.len() {
                out.push(Genotype { blocks: vec![BlockValues::Bin(c2)] });
            }
        }
        out
    }
    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/bin-2pt", SupportedBlocks::Only(vec!["binary"])).with_min_pop(2)
    }
}

pub struct BitFlipGenerator {
    pub p_m: Option<f64>,
}

impl BitFlipGenerator {
    pub fn from_params(p: &serde_json::Value) -> Result<Self, ComponentError> {
        let err = |reason: String| ComponentError::InvalidParams { kind: "gen/bit-flip".into(), reason };
        let p_m = p.get("p_m").and_then(|v| v.as_f64());
        if let Some(pm) = p_m {
            if !(0.0..=1.0).contains(&pm) { return Err(err(format!("p_m outside [0,1]: {pm}"))); }
        }
        Ok(Self { p_m })
    }
}

impl Generator for BitFlipGenerator {
    /// Per-individual loop, population order (mirrors `gen/perm-swap`'s own
    /// shape) -- but NO outer per-individual gate: [`bin_flip_mutation`]
    /// already gates per bit internally (see this module's doc).
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        let dim = bin_dim(ctx.space);
        let pm = self.p_m.unwrap_or(1.0 / dim as f64);
        (0..pop.len())
            .map(|i| {
                let mut bits = bin_values(&pop.individuals[i]).clone();
                bin_flip_mutation(&mut bits, pm, ctx.rng);
                Genotype { blocks: vec![BlockValues::Bin(bits)] }
            })
            .collect()
    }
    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/bit-flip", SupportedBlocks::Only(vec!["binary"]))
    }
}

/// `gen/ga-bin` (M3-8 Task 2): the FUSED binary-GA generator -- two-point
/// crossover AND bit-flip mutation inside ONE `Generator`, one
/// evaluate-and-replace per generation, mirroring [`crate::ga::GaRealGenerator`]'s
/// / [`crate::perm::GaPermGenerator`]'s own single-stage shape for the
/// Binary representation. Reuses [`bin_cross_pair`]/[`bin_flip_mutation`]
/// directly -- see this module's doc for the full draw-order contract and
/// defaults provenance.
pub struct GaBinGenerator {
    pub tournament_k: usize,
    pub p_c: f64,
    pub p_m: Option<f64>,
}

impl GaBinGenerator {
    pub fn from_params(p: &serde_json::Value) -> Result<Self, ComponentError> {
        let err = |reason: String| ComponentError::InvalidParams { kind: "gen/ga-bin".into(), reason };
        let g = Self {
            tournament_k: p.get("tournament_k").and_then(|v| v.as_u64()).unwrap_or(2) as usize,
            p_c: crate::params::resolve(p, "p_c", "pc").and_then(|v| v.as_f64()).unwrap_or(0.9),
            p_m: p.get("p_m").and_then(|v| v.as_f64()),
        };
        if g.tournament_k == 0 { return Err(err("tournament_k must be >= 1".into())); }
        if !(0.0..=1.0).contains(&g.p_c) { return Err(err(format!("p_c outside [0,1]: {}", g.p_c))); }
        if let Some(pm) = g.p_m {
            if !(0.0..=1.0).contains(&pm) { return Err(err(format!("p_m outside [0,1]: {pm}"))); }
        }
        Ok(g)
    }
}

impl Generator for GaBinGenerator {
    /// See this module's doc for the full draw-order contract: per pair,
    /// tournament x2 (`tournament_k` draws each), THEN [`bin_cross_pair`]
    /// (its own internal gate + conditional site draws) producing `(c1,
    /// c2)`, THEN [`bin_flip_mutation`] on `c1` then `c2` (each internally
    /// gated per bit).
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        let dim = bin_dim(ctx.space);
        assert!(pop.len() >= 2, "gen/ga-bin requires a population of at least 2 (pop_size={})", pop.len());
        let pm = self.p_m.unwrap_or(1.0 / dim as f64);
        let mut out = Vec::with_capacity(pop.len());
        while out.len() < pop.len() {
            let i1 = crate::select::tournament(pop, self.tournament_k, ctx.rng);
            let i2 = crate::select::tournament(pop, self.tournament_k, ctx.rng);
            let p1 = bin_values(&pop.individuals[i1]).clone();
            let p2 = bin_values(&pop.individuals[i2]).clone();
            let (mut c1, mut c2) = bin_cross_pair(&p1, &p2, self.p_c, ctx.rng);
            bin_flip_mutation(&mut c1, pm, ctx.rng);
            bin_flip_mutation(&mut c2, pm, ctx.rng);
            out.push(Genotype { blocks: vec![BlockValues::Bin(c1)] });
            if out.len() < pop.len() {
                out.push(Genotype { blocks: vec![BlockValues::Bin(c2)] });
            }
        }
        out
    }
    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/ga-bin", SupportedBlocks::Only(vec!["binary"])).with_min_pop(2)
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_generator("gen/bin-2pt", |p| Ok(Box::new(BinTwoPointGenerator::from_params(p)?)));
    reg.register_generator("gen/bit-flip", |p| Ok(Box::new(BitFlipGenerator::from_params(p)?)));
    reg.register_generator("gen/ga-bin", |p| Ok(Box::new(GaBinGenerator::from_params(p)?)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::problem::{Evaluator, Problem};
    use sezgi_core::space::SearchSpace;
    use sezgi_core::state::Blackboard;

    /// A minimal `Problem` with a single `Binary { n }` block, used only to
    /// construct a valid `Evaluator`/`Ctx` for these tests -- mirrors
    /// `perm.rs`'s own `PermProblem` scaffold. Fitness = number of `false`
    /// bits (a OneMax-shaped minimization target: all-`true` is optimal at
    /// fitness 0), though most tests here don't rely on the fitness being
    /// meaningful, only `pop.fitness` existing for tournament selection.
    struct BinProblem { space: SearchSpace }
    impl BinProblem {
        fn new(n: usize) -> Self {
            Self { space: SearchSpace::new(vec![Block::Binary { n }]).unwrap() }
        }
    }
    impl Problem for BinProblem {
        fn space(&self) -> &SearchSpace { &self.space }
        fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<f64> {
            pop.iter().map(|g| bin_values(g).iter().filter(|&&b| !b).count() as f64).collect()
        }
    }

    fn g(xs: Vec<bool>) -> Genotype { Genotype { blocks: vec![BlockValues::Bin(xs)] } }

    fn pop_n(n: usize, dim: usize) -> Population {
        // Individual i: bit j is true iff (j + i) % dim == 0 -- distinct
        // patterns, distinct fitness (fewer trues => higher "false count").
        Population {
            individuals: (0..n).map(|i| {
                g((0..dim).map(|j| (j + i) % dim == 0).collect())
            }).collect(),
            fitness: (0..n).map(|i| (n - i) as f64).collect(),
        }
    }

    // ==================================================================
    // Hand-traced seeded-RNG exactness (task-2 required test 1)
    // ==================================================================

    // Seed 0's first three draws are
    // [0.3245752680314067, 0.38223929651167343, 0.3596172076473553]
    // (same reconnaissance-probe values `nsga2.rs`'s own hand-traced
    // `bin_cross_pair_gate_passes_two_point_swap_hand_trace` test anchors --
    // this is the SAME function, moved, not re-derived). With p_c=0.5 and
    // n=8 bits (nbits-1=7):
    //   - gate: 0.3246 <= 0.5 -> passes (1 draw).
    //   - site1 = rnd(0,7,rng): floor(0.38223929651167343*8) = 3 (1 draw).
    //   - site2 = rnd(0,7,rng): floor(0.3596172076473553*8) = 2 (1 draw).
    //   - site1=3 > site2=2 -> swapped: site1=2, site2=3.
    // Segments [0,2) and [3,8) come from each child's own parent; the
    // single-index middle segment [2,3) is swapped. p1=all-true,
    // p2=all-false makes the swap visually obvious: c1 all-true except
    // index 2 (false); c2 all-false except index 2 (true). 3 draws total.
    #[test]
    fn bin_cross_pair_hand_trace_exact_bit_pattern() {
        let n = 8;
        let p1 = vec![true; n];
        let p2 = vec![false; n];
        let rng_before = RngStream::from_master(0, &[]);
        let mut rng = rng_before.clone();
        let (c1, c2) = bin_cross_pair(&p1, &p2, 0.5, &mut rng);

        let mut expect_c1 = vec![true; n];
        expect_c1[2] = false;
        let mut expect_c2 = vec![false; n];
        expect_c2[2] = true;
        assert_eq!(c1, expect_c1, "only bit index 2 (the swapped middle segment) should differ from parent1");
        assert_eq!(c2, expect_c2, "only bit index 2 (the swapped middle segment) should differ from parent2");

        let mut twin = rng_before;
        for _ in 0..3 { let _ = twin.next_f64(); }
        assert_eq!(rng.next_f64(), twin.next_f64(),
            "bin_cross_pair (gate pass, n=8) must consume exactly 3 draws (1 gate + site1 + site2)");
    }

    // Seed 9's first five draws are
    // [0.5990316791291411, 0.4297364011687632, 0.19864982391454744,
    //  0.8838122874587226, 0.06898027406099494] (same seed `nsga2.rs`'s own
    // `bin_flip_mutation_hand_trace` test anchors -- same function, moved).
    // With p_m=0.5, gate is `draw <= 0.5`:
    //   bit0: 0.5990 > 0.5 -> unchanged.
    //   bit1: 0.4297 <= 0.5 -> flipped.
    //   bit2: 0.1986 <= 0.5 -> flipped.
    //   bit3: 0.8838 > 0.5 -> unchanged.
    //   bit4: 0.0690 <= 0.5 -> flipped.
    // Starting from all-false: expected [false,true,true,false,true]. 5
    // draws total (one gate draw per bit, no second draw).
    #[test]
    fn bin_flip_mutation_hand_trace_exact_bit_pattern() {
        let mut x = vec![false; 5];
        let rng_before = RngStream::from_master(9, &[]);
        let mut rng = rng_before.clone();
        bin_flip_mutation(&mut x, 0.5, &mut rng);
        assert_eq!(x, vec![false, true, true, false, true]);

        let mut twin = rng_before;
        for _ in 0..5 { let _ = twin.next_f64(); }
        assert_eq!(rng.next_f64(), twin.next_f64(),
            "bin_flip_mutation must consume exactly 1 draw per bit (5 bits -> 5 draws)");
    }

    #[test]
    fn bin_cross_pair_gate_fail_is_verbatim_parents() {
        let p1 = vec![true, true, true, false];
        let p2 = vec![false, false, false, true];
        let mut rng = RngStream::from_master(55, &[]);
        let (c1, c2) = bin_cross_pair(&p1, &p2, 0.0, &mut rng);
        assert_eq!(c1, p1, "p_c=0 must leave the block unchanged (gate always fails)");
        assert_eq!(c2, p2, "p_c=0 must leave the block unchanged (gate always fails)");
    }

    #[test]
    fn bin_flip_mutation_zero_p_m_is_identity() {
        let x0 = vec![true, false, true, true, false, false];
        let mut x = x0.clone();
        let mut rng = RngStream::from_master(2024, &[]);
        bin_flip_mutation(&mut x, 0.0, &mut rng);
        assert_eq!(x, x0, "p_m=0 must leave every bit unchanged");
    }

    #[test]
    fn bin_flip_mutation_one_p_m_flips_every_bit() {
        let x0 = vec![true, false, true, true, false, false];
        let mut x = x0.clone();
        let mut rng = RngStream::from_master(2024, &[]);
        bin_flip_mutation(&mut x, 1.0, &mut rng);
        let expect: Vec<bool> = x0.iter().map(|b| !b).collect();
        assert_eq!(x, expect, "p_m=1 must flip every bit");
    }

    #[test]
    fn bin_cross_pair_children_are_recombinations_of_the_parent_bits() {
        let mut rng = RngStream::from_master(2024, &[]);
        for _ in 0..200 {
            let n = 1 + (rng.next_f64() * 12.0) as usize;
            let p1: Vec<bool> = (0..n).map(|_| rng.next_f64() < 0.5).collect();
            let p2: Vec<bool> = (0..n).map(|_| rng.next_f64() < 0.5).collect();
            let (c1, c2) = bin_cross_pair(&p1, &p2, 0.7, &mut rng);
            for j in 0..n {
                assert!(
                    (c1[j] == p1[j] || c1[j] == p2[j]) && (c2[j] == p1[j] || c2[j] == p2[j]),
                    "child bit at {j} must come from one of the two parents"
                );
            }
        }
    }

    // ==================================================================
    // gen/bin-2pt (BinTwoPointGenerator)
    // ==================================================================

    #[test]
    fn bin_two_pt_generator_validity_property_seeded_batch() {
        let n = 10usize;
        let p = BinProblem::new(n);
        let space = p.space();
        let gen = BinTwoPointGenerator::from_params(&serde_json::json!({})).unwrap();
        let mut eval = Evaluator::new(&p, 1_000_000);
        let mut rng = RngStream::from_master(3, &[0]);
        let mut bb = Blackboard::new();
        for _ in 0..200 {
            let pop = pop_n(9, n);
            let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
            let off = gen.generate(&pop, &mut ctx);
            assert_eq!(off.len(), pop.len());
            for ind in &off { assert_eq!(bin_values(ind).len(), n); }
        }
    }

    #[test]
    fn bin_two_pt_generator_deterministic_same_seed() {
        let n = 8usize;
        let p = BinProblem::new(n);
        let space = p.space();
        let gen = BinTwoPointGenerator::from_params(&serde_json::json!({})).unwrap();
        let pop = pop_n(6, n);
        let run = || {
            let mut eval = Evaluator::new(&p, 1000);
            let mut rng = RngStream::from_master(9, &[]);
            let mut bb = Blackboard::new();
            let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
            gen.generate(&pop, &mut ctx)
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn bin_two_pt_generator_draw_order_matches_pinned_structure_via_raw_replay() {
        // Twin-stream: per pair, tournament_k*2 draws (next_below(pop.len())),
        // THEN bin_cross_pair's own internal gate (+ conditional site draws)
        // -- NO separate external pc gate (unlike gen/ox), per this module's
        // doc.
        let n = 8usize;
        let p = BinProblem::new(n);
        let space = p.space();
        let gen = BinTwoPointGenerator::from_params(&serde_json::json!({"tournament_k": 3, "p_c": 0.9})).unwrap();
        let pop = pop_n(7, n);

        let mut rng = RngStream::from_master(17, &[]);
        let rng_before = rng.clone();
        let mut eval = Evaluator::new(&p, 1000);
        let mut bb = Blackboard::new();
        {
            let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
            let off = gen.generate(&pop, &mut ctx);
            assert_eq!(off.len(), pop.len());
        }

        let mut twin = rng_before;
        let mut produced = 0usize;
        while produced < pop.len() {
            for _ in 0..gen.tournament_k { twin.next_below(pop.len() as u64); }
            for _ in 0..gen.tournament_k { twin.next_below(pop.len() as u64); }
            let gate = twin.next_f64();
            if gate <= gen.p_c {
                let hi = n - 1;
                rnd(0, hi, &mut twin);
                rnd(0, hi, &mut twin);
            }
            produced += 1;
            if produced < pop.len() { produced += 1; }
        }
        assert_eq!(rng.next_f64(), twin.next_f64(),
            "gen/bin-2pt must consume exactly: tournament x2, then bin_cross_pair's own gate + (if gated) site draws, per pair");
    }

    #[test]
    fn bin_two_pt_generator_min_pop_below_2_panics() {
        let n = 4usize;
        let p = BinProblem::new(n);
        let space = p.space();
        let gen = BinTwoPointGenerator::from_params(&serde_json::json!({})).unwrap();
        let pop = pop_n(1, n);
        let mut eval = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gen.generate(&pop, &mut ctx)
        }));
        assert!(result.is_err(), "gen/bin-2pt must reject pop_size < 2 at runtime as a backstop");
    }

    #[test]
    fn bin_two_pt_generator_meta_min_pop_2_and_block_restricted() {
        let m = BinTwoPointGenerator::from_params(&serde_json::json!({})).unwrap().meta();
        assert_eq!(m.min_pop, 2);
        assert!(m.supports_block("binary"));
        assert!(!m.supports_block("float"));
        assert_eq!(m.kind, "gen/bin-2pt");
    }

    #[test]
    fn bin_two_pt_params_validate() {
        assert!(BinTwoPointGenerator::from_params(&serde_json::json!({"p_c": 1.5})).is_err());
        assert!(BinTwoPointGenerator::from_params(&serde_json::json!({"tournament_k": 0})).is_err());
        let g = BinTwoPointGenerator::from_params(&serde_json::json!({})).unwrap();
        assert_eq!(g.tournament_k, 2);
        assert_eq!(g.p_c, 0.9);
    }

    /// `gen/bin-2pt` is an M3-8 typed family (historically parsed only
    /// `p_c`) -- it now ALSO accepts legacy `pc`, resolving identically to
    /// the equivalent `p_c` spec (M3-8 deferral cleanup, `params.rs`).
    #[test]
    fn bin_two_pt_pc_alias_resolves_identically_to_canonical_p_c() {
        let legacy = BinTwoPointGenerator::from_params(&serde_json::json!({"pc": 0.42})).unwrap();
        let canonical = BinTwoPointGenerator::from_params(&serde_json::json!({"p_c": 0.42})).unwrap();
        assert_eq!(legacy.p_c, canonical.p_c);
        assert_eq!(legacy.p_c, 0.42);
    }

    /// When both spellings are present on the same block, canonical `p_c`
    /// wins outright -- `pc`'s value is never used.
    #[test]
    fn bin_two_pt_p_c_wins_over_pc_when_both_present() {
        let g = BinTwoPointGenerator::from_params(&serde_json::json!({"p_c": 0.42, "pc": 0.99})).unwrap();
        assert_eq!(g.p_c, 0.42);
    }

    // ==================================================================
    // gen/bit-flip (BitFlipGenerator)
    // ==================================================================

    #[test]
    fn bit_flip_generator_validity_property_seeded_batch() {
        let n = 9usize;
        let p = BinProblem::new(n);
        let space = p.space();
        let gen = BitFlipGenerator::from_params(&serde_json::json!({"p_m": 0.9})).unwrap();
        let mut eval = Evaluator::new(&p, 10_000_000);
        let mut rng = RngStream::from_master(6, &[0]);
        let mut bb = Blackboard::new();
        for _ in 0..500 {
            let pop = pop_n(5, n);
            let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
            let off = gen.generate(&pop, &mut ctx);
            assert_eq!(off.len(), pop.len());
            for ind in &off { assert_eq!(bin_values(ind).len(), n); }
        }
    }

    #[test]
    fn bit_flip_generator_deterministic_same_seed() {
        let n = 7usize;
        let p = BinProblem::new(n);
        let space = p.space();
        let gen = BitFlipGenerator::from_params(&serde_json::json!({"p_m": 0.5})).unwrap();
        let pop = pop_n(5, n);
        let run = || {
            let mut eval = Evaluator::new(&p, 1000);
            let mut rng = RngStream::from_master(13, &[]);
            let mut bb = Blackboard::new();
            let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
            gen.generate(&pop, &mut ctx)
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn bit_flip_generator_zero_p_m_leaves_population_unchanged() {
        let n = 6usize;
        let p = BinProblem::new(n);
        let space = p.space();
        let gen = BitFlipGenerator::from_params(&serde_json::json!({"p_m": 0.0})).unwrap();
        let pop = pop_n(4, n);
        let mut eval = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(4, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
        let off = gen.generate(&pop, &mut ctx);
        for (o, ind) in off.iter().zip(&pop.individuals) {
            assert_eq!(bin_values(o), bin_values(ind));
        }
    }

    #[test]
    fn bit_flip_generator_meta_default_min_pop_and_block_restricted() {
        let m = BitFlipGenerator::from_params(&serde_json::json!({})).unwrap().meta();
        assert_eq!(m.min_pop, 1);
        assert!(m.supports_block("binary"));
        assert!(!m.supports_block("float"));
        assert_eq!(m.kind, "gen/bit-flip");
    }

    #[test]
    fn bit_flip_default_p_m_is_one_over_l() {
        let n = 6usize;
        let p = BinProblem::new(n);
        let space = p.space();
        let gen = BitFlipGenerator::from_params(&serde_json::json!({})).unwrap();
        assert_eq!(gen.p_m, None); // resolved lazily against L inside generate()
        let pop = Population { individuals: vec![g(vec![false; 6])], fitness: vec![0.0] };
        let mut eval = Evaluator::new(&p, 100);
        // Find a seed whose first next_f64() draw is >= 1/6 (bit0's gate
        // fails under the 1/6 default) so bit0 must stay false.
        let seed = (0..2000u64).find(|&s| {
            let mut r = RngStream::from_master(s, &[0]);
            r.next_f64() >= 1.0 / 6.0
        }).unwrap();
        let mut rng = RngStream::from_master(seed, &[0]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
        let off = gen.generate(&pop, &mut ctx);
        assert!(!bin_values(&off[0])[0], "bit0's gate must fail (unchanged) when the draw is >= 1/L");
    }

    #[test]
    fn bit_flip_params_validate() {
        assert!(BitFlipGenerator::from_params(&serde_json::json!({"p_m": 1.5})).is_err());
        assert!(BitFlipGenerator::from_params(&serde_json::json!({"p_m": -0.1})).is_err());
        assert!(BitFlipGenerator::from_params(&serde_json::json!({"p_m": 0.3})).is_ok());
    }

    // ==================================================================
    // gen/ga-bin (GaBinGenerator)
    // ==================================================================

    #[test]
    fn ga_bin_generator_meta_min_pop_2_and_block_restricted() {
        let m = GaBinGenerator::from_params(&serde_json::json!({})).unwrap().meta();
        assert_eq!(m.min_pop, 2);
        assert!(m.supports_block("binary"));
        assert!(!m.supports_block("float"));
        assert_eq!(m.kind, "gen/ga-bin");
    }

    #[test]
    fn ga_bin_params_validate() {
        assert!(GaBinGenerator::from_params(&serde_json::json!({"p_c": 1.5})).is_err());
        assert!(GaBinGenerator::from_params(&serde_json::json!({"tournament_k": 0})).is_err());
        assert!(GaBinGenerator::from_params(&serde_json::json!({"p_m": -0.1})).is_err());
        assert!(GaBinGenerator::from_params(&serde_json::json!({"p_m": 1.1})).is_err());
        let g = GaBinGenerator::from_params(&serde_json::json!({})).unwrap();
        assert_eq!(g.tournament_k, 2);
        assert_eq!(g.p_c, 0.9);
        assert_eq!(g.p_m, None); // resolved lazily against L inside generate()
    }

    /// `gen/ga-bin` is an M3-8 typed family (historically parsed only
    /// `p_c`) -- it now ALSO accepts legacy `pc`, resolving identically to
    /// the equivalent `p_c` spec (M3-8 deferral cleanup, `params.rs`).
    #[test]
    fn ga_bin_pc_alias_resolves_identically_to_canonical_p_c() {
        let legacy = GaBinGenerator::from_params(&serde_json::json!({"pc": 0.42})).unwrap();
        let canonical = GaBinGenerator::from_params(&serde_json::json!({"p_c": 0.42})).unwrap();
        assert_eq!(legacy.p_c, canonical.p_c);
        assert_eq!(legacy.p_c, 0.42);
    }

    /// When both spellings are present on the same block, canonical `p_c`
    /// wins outright -- `pc`'s value is never used.
    #[test]
    fn ga_bin_p_c_wins_over_pc_when_both_present() {
        let g = GaBinGenerator::from_params(&serde_json::json!({"p_c": 0.42, "pc": 0.99})).unwrap();
        assert_eq!(g.p_c, 0.42);
    }

    #[test]
    fn ga_bin_generator_validity_property_seeded_batch() {
        let n = 10usize;
        let p = BinProblem::new(n);
        let space = p.space();
        let gen = GaBinGenerator::from_params(&serde_json::json!({})).unwrap();
        let mut eval = Evaluator::new(&p, 1_000_000);
        let mut rng = RngStream::from_master(31, &[0]);
        let mut bb = Blackboard::new();
        for _ in 0..200 {
            let pop = pop_n(9, n);
            let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
            let off = gen.generate(&pop, &mut ctx);
            assert_eq!(off.len(), pop.len());
            for ind in &off { assert_eq!(bin_values(ind).len(), n); }
        }
    }

    #[test]
    fn ga_bin_generator_deterministic_same_seed() {
        let n = 8usize;
        let p = BinProblem::new(n);
        let space = p.space();
        let gen = GaBinGenerator::from_params(&serde_json::json!({})).unwrap();
        let pop = pop_n(6, n);
        let run = || {
            let mut eval = Evaluator::new(&p, 1000);
            let mut rng = RngStream::from_master(41, &[]);
            let mut bb = Blackboard::new();
            let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
            gen.generate(&pop, &mut ctx)
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn ga_bin_generator_min_pop_below_2_panics() {
        let n = 4usize;
        let p = BinProblem::new(n);
        let space = p.space();
        let gen = GaBinGenerator::from_params(&serde_json::json!({})).unwrap();
        let pop = pop_n(1, n);
        let mut eval = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gen.generate(&pop, &mut ctx)
        }));
        assert!(result.is_err(), "gen/ga-bin must reject pop_size < 2 at runtime as a backstop");
    }

    #[test]
    fn ga_bin_generator_draw_order_matches_pinned_structure_via_raw_replay() {
        // Twin-stream: per pair, tournament_k*2 draws, THEN bin_cross_pair's
        // own gate (+ conditional site draws), THEN bin_flip_mutation on c1
        // (n draws, one gate per bit) then c2 (n more).
        let n = 8usize;
        let p = BinProblem::new(n);
        let space = p.space();
        let gen = GaBinGenerator::from_params(&serde_json::json!({"tournament_k": 3, "p_c": 0.9, "p_m": 0.5})).unwrap();
        let pop = pop_n(7, n);

        let mut rng = RngStream::from_master(19, &[]);
        let rng_before = rng.clone();
        let mut eval = Evaluator::new(&p, 1000);
        let mut bb = Blackboard::new();
        {
            let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
            let off = gen.generate(&pop, &mut ctx);
            assert_eq!(off.len(), pop.len());
        }

        let mut twin = rng_before;
        let mut produced = 0usize;
        while produced < pop.len() {
            for _ in 0..gen.tournament_k { twin.next_below(pop.len() as u64); }
            for _ in 0..gen.tournament_k { twin.next_below(pop.len() as u64); }
            let gate = twin.next_f64();
            if gate <= gen.p_c {
                let hi = n - 1;
                rnd(0, hi, &mut twin);
                rnd(0, hi, &mut twin);
            }
            for _ in 0..(2 * n) { twin.next_f64(); } // bin_flip_mutation on c1 then c2
            produced += 1;
            if produced < pop.len() { produced += 1; }
        }
        assert_eq!(rng.next_f64(), twin.next_f64(),
            "gen/ga-bin must consume exactly: tournament x2, bin_cross_pair's own gate + (if gated) site draws, \
             then bin_flip_mutation on c1 then c2 (n draws each)");
    }

    // ==================================================================
    // registry / spec-validation (task-2 required test 3)
    // ==================================================================

    #[test]
    fn registry_resolves_all_three_kinds() {
        let mut reg = Registry::new();
        register(&mut reg);
        assert!(reg.build_generator("gen/bin-2pt", &serde_json::json!({})).is_ok());
        assert!(reg.build_generator("gen/bit-flip", &serde_json::json!({})).is_ok());
        assert!(reg.build_generator("gen/ga-bin", &serde_json::json!({})).is_ok());
    }

    #[test]
    fn spec_validation_rejects_gen_ga_bin_on_float_only_space() {
        use sezgi_core::spec::{AlgorithmSpec, ComponentSpec, SpecError, StageSpec, TerminationSpec};
        let mut reg = Registry::new();
        register(&mut reg);
        crate::init::register(&mut reg);
        crate::boundary::register(&mut reg);
        crate::replace::register(&mut reg);
        let space = SearchSpace::new(vec![Block::Float { lo: 0.0, hi: 1.0, n: 3 }]).unwrap();
        let spec = AlgorithmSpec {
            name: "t".into(), pop_size: 10,
            init: ComponentSpec { kind: "init/uniform".into(), params: serde_json::json!({}) },
            boundary: ComponentSpec { kind: "boundary/clamp".into(), params: serde_json::json!({}) },
            stages: vec![StageSpec {
                generator: ComponentSpec { kind: "gen/ga-bin".into(), params: serde_json::json!({}) },
                replacer: ComponentSpec { kind: "replace/mu-plus-lambda".into(), params: serde_json::json!({}) },
                adapter: None,
            }],
            termination: TerminationSpec { budget: 100, target: None },
            restart: None,
        };
        let e = spec.validate(&reg, &space);
        match e {
            Err(SpecError::UnsupportedBlock { kind, block }) => {
                assert_eq!(kind, "gen/ga-bin");
                assert_eq!(block, "float");
            }
            other => panic!("expected SpecError::UnsupportedBlock, got {other:?}"),
        }
    }

    #[test]
    fn spec_validation_accepts_gen_ga_bin_on_binary_space() {
        use sezgi_core::spec::{AlgorithmSpec, ComponentSpec, StageSpec, TerminationSpec};
        let mut reg = Registry::new();
        register(&mut reg);
        crate::init::register(&mut reg);
        crate::boundary::register(&mut reg);
        crate::replace::register(&mut reg);
        let space = SearchSpace::new(vec![Block::Binary { n: 12 }]).unwrap();
        let spec = AlgorithmSpec {
            name: "t".into(), pop_size: 10,
            init: ComponentSpec { kind: "init/uniform".into(), params: serde_json::json!({}) },
            boundary: ComponentSpec { kind: "boundary/clamp".into(), params: serde_json::json!({}) },
            stages: vec![StageSpec {
                generator: ComponentSpec { kind: "gen/ga-bin".into(), params: serde_json::json!({}) },
                replacer: ComponentSpec { kind: "replace/mu-plus-lambda".into(), params: serde_json::json!({}) },
                adapter: None,
            }],
            termination: TerminationSpec { budget: 100, target: None },
            restart: None,
        };
        assert!(spec.validate(&reg, &space).is_ok());
    }
}
