use sezgi_core::component::*;
use sezgi_core::problem::Population;
use sezgi_core::rng::RngStream;
use sezgi_core::space::{Block, BlockValues, Genotype};

/// Permutation operators: `init/perm-random`, `gen/ox`, `gen/perm-swap` --
/// the FIRST non-Float components in the catalog. All three operate on a
/// single `Block::Permutation { n }` block (block tag `"permutation"`, per
/// `sezgi_core::spec::block_tag` -- the exact string checked by
/// `AlgorithmSpec::validate`'s `SupportedBlocks::Only` gate; NOT guessed).
///
/// ## Composition decision (mirrors `tlbo.rs`'s SHAPE, NOT its SEMANTICS as
/// "the" GA -- read this honestly, the way `tlbo.rs`'s own doc names its
/// "Hidden evaluations" caveat rather than papering over it)
///
/// `gen/ga-real` ([`crate::ga::GaRealGenerator`]) does crossover AND
/// mutation inside ONE `Generator`, in a single stage, followed by ONE
/// evaluate-and-replace per generation -- the classic GA loop: select
/// parents, crossover, mutate the SAME children, evaluate ONCE, replace
/// ONCE. The task brief mandates two SEPARATE registry components here,
/// `gen/ox` (crossover only) and `gen/perm-swap` (mutation only), and
/// `tlbo.rs` already establishes the SHAPE for composing two independent
/// `Generator`s as two `[[stages]]` entries, each with its own generator +
/// replacer pair, run in sequence every generation (`gen/tlbo-teacher` then
/// `gen/tlbo-learner`; see `presets::tlbo`). `gen/ox` and `gen/perm-swap`
/// CAN be composed the same way -- but doing so does NOT reproduce the
/// classic GA loop above, and this must not be overclaimed:
///
/// Per `sezgi_core::engine::Engine::run`'s per-stage loop, composing
/// `gen/ox` as stage 0 and `gen/perm-swap` as stage 1 means stage 0's `gen/
/// ox` offspring are boundary-repaired, EVALUATED, and GREEDILY REPLACED
/// into `pop` BEFORE stage 1's `gen/perm-swap` ever runs -- which then
/// mutates the ALREADY-COMMITTED post-crossover population, generating its
/// OWN offspring, evaluated and replaced AGAIN. That is: (a) mutation acts
/// on the population AFTER crossover's greedy acceptance decision has
/// already been made, not on the SAME crossover children before any
/// evaluation happens; (b) each generation costs `2 * pop_size` evaluations
/// (one full evaluate-and-replace cycle per stage), not `pop_size`; (c) the
/// replacement policy fires TWICE per generation, mid-generation, not once
/// at the end. This is the exact TLBO-style two-phase shape (see `tlbo.rs`'s
/// own "Hidden evaluations" section for the same `2 * nPop`-per-generation
/// accounting), NOT the classic crossover-then-mutate-then-evaluate-once GA
/// loop `gen/ga-real` implements for the Float representation.
///
/// **Consequence for `ga-perm` (Task 4), per controller ruling:** since
/// `gen/ga-real` itself is a FUSED crossover+mutation generator and the
/// two-stage `gen/ox` + `gen/perm-swap` composition above is a materially
/// DIFFERENT algorithm (TLBO-like, not GA-like), the `ga-perm` preset does
/// NOT compose these two components as two stages. It instead uses a small
/// FUSED generator (arriving in Task 4) that reuses this module's pinned
/// operator CORES directly -- [`ox_children_from_cuts`]/[`ox_child`] for
/// crossover and [`swap_positions`] for mutation, both pure and RNG-free --
/// inside one `Generator`, mirroring `gen/ga-real`'s own single-stage,
/// single-evaluate-per-generation shape for the Permutation representation.
///
/// `gen/ox` and `gen/perm-swap` REMAIN in the catalog as standalone,
/// independently composable registry components (e.g. for a spec that
/// deliberately wants the TLBO-style two-phase permutation algorithm, or
/// that wants crossover and mutation on separate, independently-tunable
/// replacement policies) -- they are not deprecated by `ga-perm`'s fused
/// design, just not what `ga-perm` itself uses. `gen/ox`'s own
/// PARENT-SELECTION structure (tournament, `pc` gate, push-pair-until-full
/// loop) DOES mirror `gen/ga-real`'s directly regardless of which preset
/// composes it (see [`OxGenerator::generate`]'s doc).
///
/// ## PROVENANCE-FIRST
///
/// ### Fisher-Yates / Durstenfeld shuffle (`init/perm-random`)
///
/// Pinned to the modern (Durstenfeld) in-place form, quoted verbatim from
/// Wikipedia's "Fisher-Yates shuffle" article (itself citing Durstenfeld,
/// R. (1964), "Algorithm 235: Random permutation", *Communications of the
/// ACM*, 7(7), 420, DOI 10.1145/364520.364540; and Knuth, D.E. (1969),
/// *The Art of Computer Programming, Vol. 2: Seminumerical Algorithms*,
/// Addison-Wesley, pp. 139-140, Algorithm P):
///
/// ```text
/// -- To shuffle an array a of n elements (indices 0..n-1):
/// for i from n-1 down to 1 do
///      j <- random integer such that 0 <= j <= i
///      exchange a[j] and a[i]
/// ```
///
/// [`fisher_yates_shuffle`] implements this EXACTLY, byte-for-byte the same
/// draw convention already established in this crate: `sezgi_components::
/// init::sample_uniform`'s own `Block::Permutation` branch (`for i in
/// (1..n).rev() { let j = rng.next_below(i as u64 + 1) as usize; xs.swap(i,
/// j); }`) -- `rng.next_below(i+1)` draws uniformly from `[0, i]` inclusive
/// (rejection sampling, no modulo bias -- `RngStream::next_below`'s own doc).
/// This module duplicates that ~5-line body rather than importing it from
/// `init.rs`, to keep `init/perm-random`'s `SupportedBlocks::Only(["permutation"])`
/// restriction independent of `init/uniform`'s `SupportedBlocks::All` (which
/// ALSO already produces a valid, correctly-shuffled permutation for a
/// `Permutation` block -- `init/perm-random` exists as a dedicated,
/// block-restricted registry kind for specs that want to declare that
/// restriction explicitly, e.g. failing fast on a mixed space).
///
/// ### Order Crossover (`gen/ox`) -- pinned variant: **OX1**
///
/// Two named sources were fetched, and they genuinely DIFFER on one detail
/// (exactly the "sources differ" case the task brief anticipated):
///
/// **Source A -- Eiben, A.E. & Smith, J.E. (2015), *Introduction to
/// Evolutionary Computing*, 2nd ed., Natural Computing Series, Springer,
/// SS4.5.2 "Recombination for Permutation Representation", "Order crossover",
/// pp. 72-73 (fetched and extracted directly from the PDF; ISBN
/// 978-3-662-44873-1, DOI 10.1007/978-3-662-44874-8).** Quoted verbatim:
///
/// ```text
/// Order crossover This operator was designed by Davis for order-based
/// permutation problems [98]. It begins in a similar fashion to PMX, by
/// copying a randomly chosen segment of the first parent into the offspring.
/// However, it proceeds differently because the intention is to transmit
/// information about relative order from the second parent.
/// 1. Choose two crossover points at random, and copy the segment between
///    them from the first parent (P1) into the first offspring.
/// 2. Starting from the second crossover point in the second parent, copy
///    the remaining unused numbers into the first child in the order that
///    they appear in the second parent, wrapping around at the end of the
///    list.
/// 3. Create the second offspring in an analogous manner, with the parent
///    roles reversed.
/// ```
///
/// Also quoted (SS4.5.1, p.69), pinning the MUTATION-PROBABILITY semantics
/// used by [`PermSwapGenerator`]: *"the mutation parameter is interpreted as
/// the probability that the chromosome undergoes mutation, rather than that
/// a single gene in the chromosome is altered"* -- i.e. `p_m` below is
/// PER-INDIVIDUAL, not per-gene (unlike `gen/ga-real`'s `pm_per_gene`).
///
/// **Source B -- Cicirello, V.A. (2023), "A Survey and Analysis of
/// Evolutionary Operators for Permutations", *Proceedings of the 15th
/// International Joint Conference on Computational Intelligence (IJCCI
/// 2023)*, pp. 288-299 (arXiv:2311.14595, fetched in full as raw HTML from
/// arxiv.org/html/2311.14595 -- NOT the AI-summarized version, the actual
/// page text).** Quoted verbatim (SS4, "Order Crossover (OX)"):
///
/// ```text
/// OX [Davis, 1985] begins by selecting two random indexes to define a
/// cross region similar to a two-point crossover for bit strings. Child c1
/// gets the positions of the elements in the region from parent p1, and the
/// relative ordering of the remaining elements from p2 but populated into c1
/// beginning after the cross region in a cyclic manner. ... Consider an
/// example with p1=[0,1,2,3,4,5,6,7] and p2=[1,2,0,5,6,7,4,3]. Let the
/// random cross region consist of indexes 2 through 4. Child c1 gets the
/// elements at those indexes from p1, such as c1=[x,x,2,3,4,x,x,x] ... The
/// rest of the elements are relatively ordered as in p2, i.e., in the order
/// 1,0,5,6,7, but populated into c1 beginning after the cross region to
/// obtain c1=[6,7,2,3,4,1,0,5].
/// ```
///
/// **The discrepancy, found by hand-verifying Source B's own worked example
/// (confirmed programmatically, not just by eye -- see this module's test
/// derivation below):** Source A's prose reads as "start SCANNING parent 2
/// AT the second crossover point" (mirroring PMX's parallel step-2 wording,
/// "Starting from the first crossover point look for elements ... in P2"),
/// i.e. read-pointer and write-pointer share the SAME starting index. That
/// reading does NOT reproduce Source B's own numeric example: re-deriving
/// c1 with `p1=[0,1,2,3,4,5,6,7]`, `p2=[1,2,0,5,6,7,4,3]`, cross region
/// `[2,4]` inclusive, and BOTH pointers starting at index 5 (right after the
/// cross region) gives `c1=[5,6,2,3,4,7,1,0]` -- NOT `[6,7,2,3,4,1,0,5]`.
/// The ONLY reading that reproduces Source B's stated result (verified by
/// hand and by a standalone script trying four candidate readings) is: scan
/// parent 2 in ITS OWN natural left-to-right array order (starting at index
/// 0, no offset), filtering out values already in the copied segment, to
/// build the "remaining, relatively-ordered" list -- THEN write that list
/// into the child's empty slots starting right after the cross region,
/// wrapping. Source B's own prose is consistent with this reading too ("the
/// relative ordering of the remaining elements from p2" names no starting
/// offset for the READ side; only the WRITE side is anchored, "beginning
/// after the cross region").
///
/// **Pinned: this decoupled-pointer reading ("OX1", the classic Order
/// Crossover -- read parent 2 start-to-end, write starting after the
/// second cut, both cyclic where needed) is what [`ox_pair`]/[`ox_child`]
/// implement**, chosen by EXAMPLE-REPRODUCTION, not by a majority survey:
/// it is the ONLY one of four candidate readings tried that reproduces
/// Source B's own worked numeric example exactly (verified both by hand and
/// by a standalone script). This is the strongest evidence available here,
/// but it is not exhaustive -- Source A's own prose is also credibly read
/// as the COUPLED-pointer variant (same start index for both read and
/// write), and no worked numeric example from Source A was available to
/// check that reading against (its figures are images, not extracted by
/// this project's fetch tooling). Source A's structural description (two
/// random cut points, segment copied from P1, remainder from P2's relative
/// order, wrapping) is followed exactly; only the fine-grained "where does
/// the P2 scan start" detail is resolved in Source B's favor, for the
/// reason above. Both
/// sources attribute OX to Davis (1985) -- but this project's OWN direct
/// fetch of Davis's IJCAI paper (`Applying Adaptive Algorithms to Epistatic
/// Domains`, *Proceedings of IJCAI-85*, pp.162-164) found that Davis's own
/// text describes a DIFFERENT, single-cut-point operator he calls
/// MODIFIED-CROSSOVER ("takes the first part of a solution, broken at
/// random, and orders the rest of its members in accordance with their
/// order in another solution" -- worked example: `(3 1 2 6 4 5)` broken
/// after its 2nd member, crossed with `(4 1 6 5 2 3)`, yields `(3 1 4 6 5
/// 2)`), with NO second cut point and NO cyclic wraparound. The
/// two-cut-point cyclic "OX" taught in textbooks and implemented here is a
/// later refinement/synthesis, conventionally still attributed to Davis
/// 1985 by the field (both Source A and Source B do so) -- documented here
/// for pedigree honesty, per this project's provenance protocol.
///
/// **Cut-point draw convention (this module's own choice, undocumented by
/// either source beyond "at random"):** two DISTINCT indices are drawn
/// uniformly from `[0, n)` via rejection sampling (the same
/// `pick_distinct`-style idiom `de.rs`/`tlbo.rs` already use for "distinct
/// index" draws), then sorted so `lo <= hi`; the copied segment is
/// `[lo, hi]` INCLUSIVE of both endpoints -- matching Source B's own
/// "indexes 2 through 4" example (3 positions: 2, 3, 4).
///
/// ### Swap mutation (`gen/perm-swap`)
///
/// Pinned to Source A (Eiben & Smith 2015), SS4.5.1, p.69, quoted verbatim:
/// *"Swap Mutation Two positions (genes) in the chromosome are selected at
/// random and their allele values swapped."* Corroborated by Source B
/// (Cicirello 2023), SS3 "Mutation Operators": *"Swap: Swap mutation (also
/// known as exchange) chooses two different elements uniformly at random
/// and swaps them."* Applied PER INDIVIDUAL with probability `p_m` (per
/// Source A's SS4.5.1 framing quoted above): swap two DISTINCT positions
/// (rejection sampling, same idiom as the OX cut points) drawn uniformly
/// from `[0, n)`.
///
/// **`p_m` default:** Eiben & Smith's OWN worked example (SS13.3, "Example
/// Application: Graph Three-Colouring", p.213) configures a permutation-
/// representation EA with *"swap mutation with pm = 1/n and order crossover
/// with pc = 0.8"* -- quoted verbatim, `n` there being the chromosome
/// (permutation) length. [`PermSwapGenerator`] and [`OxGenerator`] adopt
/// these exact values as their defaults (`p_m = 1/n`, `pc = 0.8`), honestly
/// cited to this worked example rather than invented. `p_m`'s `1/n` default
/// mirrors `gen/ga-real`'s own `pm_per_gene.unwrap_or(1.0 / dim as f64)`
/// pattern structurally (an `Option<f64>` param defaulting to the
/// dimension-scaled value) -- `gen/ox`'s `tournament_k` default (`2`) is
/// its OWN mirror of `gen/ga-real`'s `tournament_k` default, per the task
/// brief ("parent-selection params mirroring ga-real's").
///
/// ## Draw order (part of the RNG-stream contract)
///
/// **`init/perm-random`**: per individual, `n-1` draws (`fisher_yates_
/// shuffle`'s loop, `i` from `n-1` down to `1`), each `rng.next_below(i+1)`.
///
/// **`gen/ox`** ([`OxGenerator::generate`]): identical PAIR-LOOP structure
/// to `gen/ga-real` (`while out.len() < pop.len()`). Per pair: tournament
/// for parent 1 (`tournament_k` draws of `next_below(pop.len())`), THEN
/// tournament for parent 2 (`tournament_k` more draws), THEN the `pc` gate
/// (1 draw, `next_f64()`); IF the gate passes, [`ox_pair`] draws the first
/// cut index (`next_below(n)`) then rejection-samples the second (`>= 1`
/// draw, `next_below(n)` until distinct from the first) -- IF the gate
/// fails, zero further draws (children are direct clones of the parents,
/// mirroring `gen/ga-real`'s own SBX-gated-by-`pc` structure).
///
/// **`gen/perm-swap`** ([`PermSwapGenerator::generate`]): per individual (
/// population order, mirroring `gen/tlbo-learner`'s per-learner loop
/// shape): the `p_m` gate (1 draw, `next_f64()`); IF it passes, position `a`
/// (`next_below(n)`) then rejection-sampled position `b` (`>= 1` draw,
/// `next_below(n)` until distinct from `a`) -- IF it fails, zero further
/// draws (the individual passes through unchanged).
pub fn fisher_yates_shuffle(n: usize, rng: &mut RngStream) -> Vec<u32> {
    let mut xs: Vec<u32> = (0..n as u32).collect();
    for i in (1..n).rev() {
        let j = rng.next_below(i as u64 + 1) as usize;
        xs.swap(i, j);
    }
    xs
}

fn perm_dim(space: &sezgi_core::space::SearchSpace) -> usize {
    let Block::Permutation { n } = space.blocks()[0] else {
        unreachable!("permutation operators require a single Permutation block")
    };
    n
}

fn perm_values(g: &Genotype) -> &Vec<u32> {
    match &g.blocks[0] {
        BlockValues::Perm(xs) => xs,
        _ => unreachable!("permutation operators require a Permutation block value"),
    }
}

pub struct PermRandomInit;

impl Initializer for PermRandomInit {
    fn initialize(&self, n: usize, ctx: &mut Ctx) -> Vec<Genotype> {
        let dim = perm_dim(ctx.space);
        (0..n)
            .map(|_| Genotype { blocks: vec![BlockValues::Perm(fisher_yates_shuffle(dim, ctx.rng))] })
            .collect()
    }
    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("init/perm-random", SupportedBlocks::Only(vec!["permutation"]))
    }
}

/// Draw two DISTINCT indices uniformly from `[0, n)` via rejection sampling
/// (the shared `pick_distinct`-style idiom `de.rs`/`tlbo.rs` already use for
/// "distinct index" draws) -- the ONE draw primitive both `gen/ox`'s cut
/// points and `gen/perm-swap`'s swap positions consume (see this module's
/// doc "Draw order"). Returns `(first_drawn, second_drawn)` in DRAW ORDER,
/// unsorted -- callers that need an ordered pair (e.g. [`ox_pair`]'s cut
/// points) sort afterward; this keeps the draw sequence itself identical
/// regardless of what the caller does with the result, so factoring this
/// out changes NO existing draw-order test's expected values.
///
/// `pub(crate)`, not private: this is one of the two pinned, RNG-consuming
/// primitives Task 4's fused `ga-perm` generator reuses directly (alongside
/// [`ox_children_from_cuts`]/[`ox_child`] and [`swap_positions`]) rather
/// than re-deriving the same logic -- see this module's doc "Composition
/// decision".
pub(crate) fn draw_distinct_pair(n: usize, rng: &mut RngStream) -> (usize, usize) {
    let i1 = rng.next_below(n as u64) as usize;
    let mut i2 = rng.next_below(n as u64) as usize;
    while i2 == i1 {
        i2 = rng.next_below(n as u64) as usize;
    }
    (i1, i2)
}

/// One child of Order Crossover (OX1): copies `a`'s values at positions
/// `[lo, hi]` inclusive, then fills the remaining (empty) positions, in
/// order starting right after `hi` (wrapping to 0), with `b`'s values that
/// are NOT already copied, read in `b`'s own left-to-right order. See this
/// module's doc for the full provenance derivation pinning this exact
/// read/write pointer convention against a verified worked example.
///
/// Pure and RNG-free -- `pub(crate)` so Task 4's fused `ga-perm` generator
/// can call this EXACT pinned core directly (see this module's doc
/// "Composition decision").
pub(crate) fn ox_child(a: &[u32], b: &[u32], lo: usize, hi: usize) -> Vec<u32> {
    let n = a.len();
    let mut child: Vec<Option<u32>> = vec![None; n];
    let mut used = vec![false; n];
    for i in lo..=hi {
        child[i] = Some(a[i]);
        used[a[i] as usize] = true;
    }
    let remaining = b.iter().copied().filter(|v| !used[*v as usize]);
    let mut pos = (hi + 1) % n;
    for v in remaining {
        while child[pos].is_some() {
            pos = (pos + 1) % n;
        }
        child[pos] = Some(v);
    }
    child.into_iter().map(|x| x.expect("every position filled by segment or fill loop")).collect()
}

/// Both children of Order Crossover (OX1) for a FIXED pair of cut points --
/// the pure, RNG-free core of [`ox_pair`] (which just draws `(lo, hi)` via
/// [`draw_distinct_pair`] and calls straight through to this). `pub(crate)`
/// for the same reason as [`ox_child`]: Task 4's fused generator reuses
/// this directly.
pub(crate) fn ox_children_from_cuts(p1: &[u32], p2: &[u32], lo: usize, hi: usize) -> (Vec<u32>, Vec<u32>) {
    (ox_child(p1, p2, lo, hi), ox_child(p2, p1, lo, hi))
}

/// Order Crossover (OX1) on a parent pair: draws two DISTINCT cut indices
/// via [`draw_distinct_pair`] (uniform over `[0,n)`, sorted), then derives
/// both children via [`ox_children_from_cuts`].
///
/// The `n >= 2` check below is a RUNTIME-ONLY backstop (mirrors `tlbo.rs`'s
/// own two-layer idiom for its `min_pop` checks) -- but unlike `min_pop`
/// (a POPULATION-size property, checked STATICALLY by `AlgorithmSpec::
/// validate` via `ComponentMeta::min_pop`), this is a PERMUTATION-LENGTH
/// (block DIMENSION) property. `ComponentMeta` has no dimension-minimum
/// field -- only `min_pop` -- so there is no static spec-validation
/// equivalent to add here; a `Block::Permutation { n: 1 }` or `{ n: 0 }`
/// space passes `SupportedBlocks::Only(["permutation"])` and `min_pop`
/// checks identically to any other permutation length, and can only be
/// caught here, at the point two distinct cut indices are actually needed.
/// Adding a general "minimum block dimension" concept to `ComponentMeta`
/// is out of scope for this module; documenting the gap here (rather than
/// silently relying on the runtime `assert!`) is the deliberate choice.
pub fn ox_pair(p1: &[u32], p2: &[u32], rng: &mut RngStream) -> (Vec<u32>, Vec<u32>) {
    let n = p1.len();
    assert_eq!(p1.len(), p2.len(), "gen/ox requires equal-length parents");
    assert!(n >= 2, "gen/ox requires a permutation of length >= 2 (n={n})");
    let (i1, i2) = draw_distinct_pair(n, rng);
    let (lo, hi) = if i1 <= i2 { (i1, i2) } else { (i2, i1) };
    ox_children_from_cuts(p1, p2, lo, hi)
}

pub struct OxGenerator {
    pub tournament_k: usize,
    pub pc: f64,
}

impl OxGenerator {
    pub fn from_params(p: &serde_json::Value) -> Result<Self, ComponentError> {
        let err = |reason: String| ComponentError::InvalidParams { kind: "gen/ox".into(), reason };
        let g = Self {
            tournament_k: p.get("tournament_k").and_then(|v| v.as_u64()).unwrap_or(2) as usize,
            pc: p.get("pc").and_then(|v| v.as_f64()).unwrap_or(0.8),
        };
        if g.tournament_k == 0 { return Err(err("tournament_k must be >= 1".into())); }
        if !(0.0..=1.0).contains(&g.pc) { return Err(err(format!("pc outside [0,1]: {}", g.pc))); }
        Ok(g)
    }

    /// Identical structure to `gen/ga-real`'s own `tournament` (mirrored
    /// deliberately -- see this module's doc "Composition decision").
    fn tournament(&self, pop: &Population, rng: &mut RngStream) -> usize {
        let mut best = rng.next_below(pop.len() as u64) as usize;
        for _ in 1..self.tournament_k {
            let c = rng.next_below(pop.len() as u64) as usize;
            if pop.fitness[c] < pop.fitness[best] { best = c; }
        }
        best
    }
}

impl Generator for OxGenerator {
    /// Mirrors `gen/ga-real`'s `GaRealGenerator::generate` pair-loop
    /// EXACTLY: two tournament selections, a `pc`-gated recombination, push
    /// child 1 (and child 2 if room remains). The only structural
    /// difference is that `gen/ga-real` ALSO applies per-gene mutation
    /// unconditionally inside the same generator; `gen/ox` does not -- it
    /// is crossover ONLY. `gen/perm-swap` provides the mutation half as a
    /// separate, standalone component; composing the two as sequential
    /// engine stages yields a TLBO-style two-phase algorithm, NOT the
    /// classic GA loop -- see this module's doc "Composition decision" for
    /// exactly why, and for `ga-perm`'s own (different, fused) design.
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        assert!(pop.len() >= 2, "gen/ox requires a population of at least 2 (pop_size={})", pop.len());
        let mut out = Vec::with_capacity(pop.len());
        while out.len() < pop.len() {
            let (i1, i2) = (self.tournament(pop, ctx.rng), self.tournament(pop, ctx.rng));
            let p1 = perm_values(&pop.individuals[i1]).clone();
            let p2 = perm_values(&pop.individuals[i2]).clone();
            let (c1, c2) = if ctx.rng.next_f64() < self.pc {
                ox_pair(&p1, &p2, ctx.rng)
            } else {
                (p1, p2)
            };
            out.push(Genotype { blocks: vec![BlockValues::Perm(c1)] });
            if out.len() < pop.len() {
                out.push(Genotype { blocks: vec![BlockValues::Perm(c2)] });
            }
        }
        out
    }
    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/ox", SupportedBlocks::Only(vec!["permutation"])).with_min_pop(2)
    }
}

pub struct PermSwapGenerator {
    pub p_m: Option<f64>,
}

impl PermSwapGenerator {
    pub fn from_params(p: &serde_json::Value) -> Result<Self, ComponentError> {
        let err = |reason: String| ComponentError::InvalidParams { kind: "gen/perm-swap".into(), reason };
        let p_m = p.get("p_m").and_then(|v| v.as_f64());
        if let Some(pm) = p_m {
            if !(0.0..=1.0).contains(&pm) { return Err(err(format!("p_m outside [0,1]: {pm}"))); }
        }
        Ok(Self { p_m })
    }
}

/// Swap-mutation core: exchange positions `i` and `j` of `perm` in place.
/// Pure, RNG-free, and trivial by design (`slice::swap`) -- named and made
/// `pub(crate)` anyway, alongside [`ox_child`]/[`ox_children_from_cuts`],
/// so Task 4's fused `ga-perm` generator has ONE discoverable, pinned
/// primitive to call for "the mutation step", rather than re-deriving
/// which two positions a bare `.swap(a, b)` call implements. Positions are
/// NOT validated here (callers, e.g. [`PermSwapGenerator::generate`] via
/// [`draw_distinct_pair`], already draw distinct in-bounds positions) --
/// out-of-bounds `i`/`j` panic via the underlying slice index, same as a
/// direct `.swap()` call would.
pub(crate) fn swap_positions(perm: &mut [u32], i: usize, j: usize) {
    perm.swap(i, j);
}

impl Generator for PermSwapGenerator {
    /// Per-individual loop (population order), mirroring `gen/tlbo-
    /// learner`'s per-learner shape (see this module's doc "Draw order").
    ///
    /// The `dim >= 2` check below is the same kind of runtime-only
    /// dimension backstop as [`ox_pair`]'s -- see that function's doc for
    /// why there is no static `ComponentMeta` equivalent to add.
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        let dim = perm_dim(ctx.space);
        assert!(dim >= 2, "gen/perm-swap requires a permutation of length >= 2 (n={dim})");
        let pm = self.p_m.unwrap_or(1.0 / dim as f64);
        (0..pop.len())
            .map(|i| {
                let mut xs = perm_values(&pop.individuals[i]).clone();
                if ctx.rng.next_f64() < pm {
                    let (a, b) = draw_distinct_pair(dim, ctx.rng);
                    swap_positions(&mut xs, a, b);
                }
                Genotype { blocks: vec![BlockValues::Perm(xs)] }
            })
            .collect()
    }
    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/perm-swap", SupportedBlocks::Only(vec!["permutation"]))
    }
}

/// `gen/ga-perm` (M3-3 Task 4): the FUSED permutation-GA generator --
/// crossover AND mutation inside ONE `Generator`, one evaluate-and-replace
/// per generation, mirroring [`crate::ga::GaRealGenerator`]'s own
/// single-stage shape for the Permutation representation, per the
/// controller ruling recorded above in "Composition decision". Reuses this
/// module's pinned, RNG-consuming cores directly rather than re-deriving
/// them: [`draw_distinct_pair`] for both the OX cut points and the swap
/// positions, [`ox_children_from_cuts`] for crossover, [`swap_positions`]
/// for mutation.
///
/// **Structure** (mirrors `GaRealGenerator::generate`'s pair-loop exactly:
/// `while out.len() < pop.len()`, two tournaments, a `pc`-gated
/// recombination, THEN mutation of both children, push child 1 then child 2
/// if room remains): the one deliberate representation-specific difference
/// is mutation's GRANULARITY -- `gen/ga-real`'s mutation is per-gene
/// (`pm_per_gene`, one gate draw per dimension); `gen/ga-perm`'s is
/// PER-INDIVIDUAL (one gate draw per child, same as [`PermSwapGenerator`]),
/// per Eiben & Smith's own semantics quoted above ("the mutation parameter
/// is interpreted as the probability that the chromosome undergoes
/// mutation, rather than that a single gene in the chromosome is altered").
///
/// **Params, mirroring `gen/ox`/`gen/ga-real`/`gen/perm-swap`'s own
/// defaults exactly, for the reasons cited in those sections above**:
/// `tournament_k` defaults to `2` (`gen/ga-real`'s/`gen/ox`'s default),
/// `pc` defaults to `0.8` (`gen/ox`'s default, Eiben & Smith SS13.3's worked
/// example), `p_m` defaults to `1/n` (`gen/perm-swap`'s default, same
/// worked example).
///
/// **Draw order** (twin-stream contract, per pair): tournament for parent 1
/// (`tournament_k` draws of `next_below(pop.len())`), THEN tournament for
/// parent 2 (`tournament_k` more draws), THEN the `pc` gate (1 draw,
/// `next_f64()`) -- IF it passes, [`draw_distinct_pair`]'s cut-point draws
/// (`>= 2`, rejection-sampled) feed [`ox_children_from_cuts`]; IF it fails,
/// `c1`/`c2` are direct clones of the parents, zero further draws for this
/// step (mirrors `gen/ga-real`'s SBX-gated-by-`pc` structure exactly).
/// THEN, for `c1` then `c2` in that order: the `p_m` gate (1 draw,
/// `next_f64()`) -- IF it passes, [`draw_distinct_pair`]'s swap-position
/// draws (`>= 2`, rejection-sampled) feed [`swap_positions`]; IF it fails,
/// zero further draws, the child passes through unmutated (mirrors
/// `gen/perm-swap`'s own per-individual gate exactly).
pub struct GaPermGenerator {
    pub tournament_k: usize,
    pub pc: f64,
    pub p_m: Option<f64>,
}

impl GaPermGenerator {
    pub fn from_params(p: &serde_json::Value) -> Result<Self, ComponentError> {
        let err = |reason: String| ComponentError::InvalidParams { kind: "gen/ga-perm".into(), reason };
        let g = Self {
            tournament_k: p.get("tournament_k").and_then(|v| v.as_u64()).unwrap_or(2) as usize,
            pc: p.get("pc").and_then(|v| v.as_f64()).unwrap_or(0.8),
            p_m: p.get("p_m").and_then(|v| v.as_f64()),
        };
        if g.tournament_k == 0 { return Err(err("tournament_k must be >= 1".into())); }
        if !(0.0..=1.0).contains(&g.pc) { return Err(err(format!("pc outside [0,1]: {}", g.pc))); }
        if let Some(pm) = g.p_m {
            if !(0.0..=1.0).contains(&pm) { return Err(err(format!("p_m outside [0,1]: {pm}"))); }
        }
        Ok(g)
    }

    /// Identical structure to `gen/ga-real`'s/`gen/ox`'s own `tournament`
    /// (mirrored deliberately, per this preset's design).
    fn tournament(&self, pop: &Population, rng: &mut RngStream) -> usize {
        let mut best = rng.next_below(pop.len() as u64) as usize;
        for _ in 1..self.tournament_k {
            let c = rng.next_below(pop.len() as u64) as usize;
            if pop.fitness[c] < pop.fitness[best] { best = c; }
        }
        best
    }
}

impl Generator for GaPermGenerator {
    /// See this struct's doc for the full draw-order contract and the
    /// controller ruling this design follows.
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        let dim = perm_dim(ctx.space);
        assert!(pop.len() >= 2, "gen/ga-perm requires a population of at least 2 (pop_size={})", pop.len());
        assert!(dim >= 2, "gen/ga-perm requires a permutation of length >= 2 (n={dim})");
        let pm = self.p_m.unwrap_or(1.0 / dim as f64);
        let mut out = Vec::with_capacity(pop.len());
        while out.len() < pop.len() {
            let (i1, i2) = (self.tournament(pop, ctx.rng), self.tournament(pop, ctx.rng));
            let p1 = perm_values(&pop.individuals[i1]).clone();
            let p2 = perm_values(&pop.individuals[i2]).clone();
            let (mut c1, mut c2) = if ctx.rng.next_f64() < self.pc {
                let (a, b) = draw_distinct_pair(dim, ctx.rng);
                let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
                ox_children_from_cuts(&p1, &p2, lo, hi)
            } else {
                (p1, p2)
            };
            for c in [&mut c1, &mut c2] {
                if ctx.rng.next_f64() < pm {
                    let (a, b) = draw_distinct_pair(dim, ctx.rng);
                    swap_positions(c, a, b);
                }
            }
            out.push(Genotype { blocks: vec![BlockValues::Perm(c1)] });
            if out.len() < pop.len() {
                out.push(Genotype { blocks: vec![BlockValues::Perm(c2)] });
            }
        }
        out
    }
    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/ga-perm", SupportedBlocks::Only(vec!["permutation"])).with_min_pop(2)
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_initializer("init/perm-random", |_| Ok(Box::new(PermRandomInit)));
    reg.register_generator("gen/ox", |p| Ok(Box::new(OxGenerator::from_params(p)?)));
    reg.register_generator("gen/perm-swap", |p| Ok(Box::new(PermSwapGenerator::from_params(p)?)));
    reg.register_generator("gen/ga-perm", |p| Ok(Box::new(GaPermGenerator::from_params(p)?)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::problem::{Evaluator, Problem};
    use sezgi_core::space::SearchSpace;
    use sezgi_core::state::Blackboard;

    /// A minimal `Problem` with a single `Permutation { n }` block, used
    /// only to construct a valid `Evaluator`/`Ctx` for these tests -- no
    /// test here relies on its fitness values being meaningful (mirrors how
    /// `tlbo.rs`'s tests use `SphereShifted` purely as a `Ctx` scaffold).
    struct PermProblem { space: SearchSpace }
    impl PermProblem {
        fn new(n: usize) -> Self {
            Self { space: SearchSpace::new(vec![Block::Permutation { n }]).unwrap() }
        }
    }
    impl Problem for PermProblem {
        fn space(&self) -> &SearchSpace { &self.space }
        fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<f64> {
            pop.iter().map(|g| perm_values(g)[0] as f64).collect()
        }
    }

    fn g(xs: Vec<u32>) -> Genotype { Genotype { blocks: vec![BlockValues::Perm(xs)] } }

    fn pop_n(n: usize, dim: u32) -> Population {
        // Each individual is a distinct rotation of 0..dim, distinct fitness.
        Population {
            individuals: (0..n).map(|i| {
                let i = i as u32;
                g((0..dim).map(|d| (d + i) % dim).collect())
            }).collect(),
            fitness: (0..n).map(|i| (n - i) as f64).collect(),
        }
    }

    fn is_permutation(xs: &[u32], n: usize) -> bool {
        let mut seen = vec![false; n];
        xs.len() == n && xs.iter().all(|&x| {
            let x = x as usize;
            x < n && !std::mem::replace(&mut seen[x], true)
        })
    }

    // ---- fisher_yates_shuffle / init/perm-random ----

    #[test]
    fn fisher_yates_produces_a_valid_permutation_every_time() {
        let mut rng = RngStream::from_master(1, &[0]);
        for _ in 0..2000 {
            let xs = fisher_yates_shuffle(9, &mut rng);
            assert!(is_permutation(&xs, 9), "not a permutation: {xs:?}");
        }
    }

    #[test]
    fn fisher_yates_draw_count_is_n_minus_1() {
        // Twin-stream: n-1 draws, next_below(i+1) for i from n-1 downto 1.
        let mut rng = RngStream::from_master(7, &[0]);
        let rng_before = rng.clone();
        let n = 6;
        let _ = fisher_yates_shuffle(n, &mut rng);

        let mut twin = rng_before;
        for i in (1..n).rev() { twin.next_below(i as u64 + 1); }
        assert_eq!(rng.next_u64(), twin.next_u64(),
            "fisher_yates_shuffle must consume exactly n-1 draws, next_below(i+1) for i=n-1..1");
    }

    #[test]
    fn fisher_yates_matches_init_uniform_permutation_branch_bit_for_bit() {
        // Same draw convention as sezgi_components::init::sample_uniform's
        // own Permutation branch, verified directly against its literal
        // swap sequence (not merely "looks the same").
        let mut rng = RngStream::from_master(42, &[3]);
        let mut twin = rng.clone();
        let got = fisher_yates_shuffle(10, &mut rng);

        let mut xs: Vec<u32> = (0..10u32).collect();
        for i in (1..10).rev() {
            let j = twin.next_below(i as u64 + 1) as usize;
            xs.swap(i, j);
        }
        assert_eq!(got, xs);
    }

    #[test]
    fn init_perm_random_all_valid_permutations_seeded_batch() {
        let n = 7;
        let p = PermProblem::new(n);
        let space = p.space();
        let mut eval = Evaluator::new(&p, 10_000);
        let mut rng = RngStream::from_master(11, &[0]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
        let pop = PermRandomInit.initialize(500, &mut ctx);
        assert_eq!(pop.len(), 500);
        for ind in &pop {
            space.validate(ind).unwrap();
            assert!(is_permutation(perm_values(ind), n));
        }
    }

    #[test]
    fn init_perm_random_deterministic_same_seed() {
        let n = 8;
        let p = PermProblem::new(n);
        let space = p.space();
        let run = || {
            let mut eval = Evaluator::new(&p, 1000);
            let mut rng = RngStream::from_master(5, &[]);
            let mut bb = Blackboard::new();
            let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
            PermRandomInit.initialize(20, &mut ctx)
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn init_perm_random_meta_is_permutation_only() {
        let m = PermRandomInit.meta();
        assert!(m.supports_block("permutation"));
        assert!(!m.supports_block("float"));
        assert_eq!(m.kind, "init/perm-random");
    }

    // ---- OX hand fixture (Source B's own worked example, independently
    // re-derived and verified -- see this module's doc "Order Crossover") ----

    #[test]
    fn ox_child_matches_hand_derived_fixture() {
        // p1=[0,1,2,3,4,5,6,7], p2=[1,2,0,5,6,7,4,3], cross region [2,4] incl.
        //
        // c1: segment from p1 at [2,4] = [2,3,4]. Remaining = p2 read in
        // p2's OWN left-to-right order [1,2,0,5,6,7,4,3], filtering out
        // {2,3,4}: 1,(2 excl),0,5,6,7,(4 excl),(3 excl) => [1,0,5,6,7].
        // Filled into c1's empty slots starting at position 5 (right after
        // the cross region), wrapping: pos5=1,pos6=0,pos7=5,pos0=6,pos1=7.
        // c1 = [6,7,2,3,4,1,0,5].
        let p1: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let p2: Vec<u32> = vec![1, 2, 0, 5, 6, 7, 4, 3];
        let c1 = ox_child(&p1, &p2, 2, 4);
        assert_eq!(c1, vec![6, 7, 2, 3, 4, 1, 0, 5]);

        // c2: segment from p2 at [2,4] = [0,5,6]. Remaining = p1 read in
        // p1's own order [0,1,2,3,4,5,6,7], filtering {0,5,6}:
        // (0 excl),1,2,3,4,(5 excl),(6 excl),7 => [1,2,3,4,7]. Filled into
        // c2's empty slots starting at position 5, wrapping:
        // pos5=1,pos6=2,pos7=3,pos0=4,pos1=7. c2 = [4,7,0,5,6,1,2,3].
        let c2 = ox_child(&p2, &p1, 2, 4);
        assert_eq!(c2, vec![4, 7, 0, 5, 6, 1, 2, 3]);
        assert!(is_permutation(&c1, 8));
        assert!(is_permutation(&c2, 8));
    }

    #[test]
    fn ox_pair_via_replayed_raw_draws_matches_hand_fixture() {
        // Twin-stream: force the cut points to (2,4) by replaying the exact
        // draw sequence ox_pair consumes (i1 = next_below(n), then
        // rejection-sampled i2 != i1), then compare to the hand fixture.
        let p1: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let p2: Vec<u32> = vec![1, 2, 0, 5, 6, 7, 4, 3];
        let n = 8u64;

        // Search for a seed whose first two `next_below(8)` draws (with the
        // second forced distinct from the first, matching ox_pair's own
        // rejection loop) are exactly 2 then 4 -- so the actually-exercised
        // code path IS the hand-derived fixture above, not a re-implementation.
        let seed = (0..10_000u64).find(|&s| {
            let mut r = RngStream::from_master(s, &[0]);
            let i1 = r.next_below(n);
            let mut i2 = r.next_below(n);
            while i2 == i1 { i2 = r.next_below(n); }
            i1 == 2 && i2 == 4
        }).expect("some seed in range must produce cut points (2,4)");

        let mut rng = RngStream::from_master(seed, &[0]);
        let (c1, c2) = ox_pair(&p1, &p2, &mut rng);
        assert_eq!(c1, vec![6, 7, 2, 3, 4, 1, 0, 5]);
        assert_eq!(c2, vec![4, 7, 0, 5, 6, 1, 2, 3]);
    }

    #[test]
    fn ox_pair_validity_property_seeded_batch() {
        let mut rng = RngStream::from_master(21, &[0]);
        for _ in 0..2000 {
            let p1 = fisher_yates_shuffle(11, &mut rng);
            let p2 = fisher_yates_shuffle(11, &mut rng);
            let (c1, c2) = ox_pair(&p1, &p2, &mut rng);
            assert!(is_permutation(&c1, 11), "c1 not a permutation: {c1:?}");
            assert!(is_permutation(&c2, 11), "c2 not a permutation: {c2:?}");
        }
    }

    // ---- OxGenerator ----

    #[test]
    fn ox_generator_validity_property_seeded_batch() {
        let n = 10usize;
        let p = PermProblem::new(n);
        let space = p.space();
        let gen = OxGenerator::from_params(&serde_json::json!({})).unwrap();
        let mut eval = Evaluator::new(&p, 1_000_000);
        let mut rng = RngStream::from_master(3, &[0]);
        let mut bb = Blackboard::new();
        for _ in 0..200 {
            let pop = pop_n(9, n as u32);
            let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
            let off = gen.generate(&pop, &mut ctx);
            assert_eq!(off.len(), pop.len());
            for ind in &off { assert!(is_permutation(perm_values(ind), n)); }
        }
    }

    #[test]
    fn ox_generator_deterministic_same_seed() {
        let n = 8usize;
        let p = PermProblem::new(n);
        let space = p.space();
        let gen = OxGenerator::from_params(&serde_json::json!({})).unwrap();
        let pop = pop_n(6, n as u32);
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
    fn ox_generator_draw_order_matches_pinned_structure_via_raw_replay() {
        // Twin-stream raw-replay: per pair, tournament_k*2 draws
        // (next_below(pop.len())), then the pc gate (next_f64), then IF it
        // passes, ox_pair's own cut-point draws.
        let n = 8usize;
        let p = PermProblem::new(n);
        let space = p.space();
        let gen = OxGenerator::from_params(&serde_json::json!({"tournament_k": 3, "pc": 0.8})).unwrap();
        let pop = pop_n(7, n as u32);

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
            if gate < gen.pc {
                let i1 = twin.next_below(n as u64);
                let mut i2 = twin.next_below(n as u64);
                while i2 == i1 { i2 = twin.next_below(n as u64); }
            }
            produced += 1;
            if produced < pop.len() { produced += 1; }
        }
        assert_eq!(rng.next_f64(), twin.next_f64(),
            "gen/ox must consume exactly: tournament x2, pc gate, then (if gated) ox_pair's cut draws, per pair");
    }

    #[test]
    fn ox_generator_min_pop_below_2_panics() {
        let n = 4usize;
        let p = PermProblem::new(n);
        let space = p.space();
        let gen = OxGenerator::from_params(&serde_json::json!({})).unwrap();
        let pop = pop_n(1, n as u32);
        let mut eval = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gen.generate(&pop, &mut ctx)
        }));
        assert!(result.is_err(), "gen/ox must reject pop_size < 2 at runtime as a backstop");
    }

    #[test]
    fn ox_generator_meta_min_pop_2_and_block_restricted() {
        let m = OxGenerator::from_params(&serde_json::json!({})).unwrap().meta();
        assert_eq!(m.min_pop, 2);
        assert!(m.supports_block("permutation"));
        assert!(!m.supports_block("float"));
        assert_eq!(m.kind, "gen/ox");
    }

    #[test]
    fn ox_params_validate() {
        assert!(OxGenerator::from_params(&serde_json::json!({"pc": 1.5})).is_err());
        assert!(OxGenerator::from_params(&serde_json::json!({"tournament_k": 0})).is_err());
        let g = OxGenerator::from_params(&serde_json::json!({})).unwrap();
        assert_eq!(g.tournament_k, 2);
        assert_eq!(g.pc, 0.8);
    }

    // ---- swap-mutation hand fixture ----

    #[test]
    fn perm_swap_hand_fixture_via_replayed_raw_draws() {
        // Force: pm gate passes (next_f64() < pm), positions a=1, b=4.
        // Individual: [0,1,2,3,4,5,6,7] with positions 1 and 4 swapped =>
        // [0,4,2,3,1,5,6,7] (values at index1<->index4 swapped by hand).
        let n = 8u64;
        let seed = (0..10_000u64).find(|&s| {
            let mut r = RngStream::from_master(s, &[0]);
            let gate = r.next_f64();
            if gate >= 0.5 { return false; } // pm default here forced to 0.5 for the search
            let a = r.next_below(n);
            let mut b = r.next_below(n);
            while b == a { b = r.next_below(n); }
            a == 1 && b == 4
        }).expect("some seed in range must produce gate-pass + positions (1,4)");

        let p = PermProblem::new(8);
        let space = p.space();
        let gen = PermSwapGenerator::from_params(&serde_json::json!({"p_m": 0.5})).unwrap();
        let pop = Population { individuals: vec![g(vec![0, 1, 2, 3, 4, 5, 6, 7])], fitness: vec![0.0] };
        let mut eval = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(seed, &[0]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
        let off = gen.generate(&pop, &mut ctx);
        assert_eq!(perm_values(&off[0]), &vec![0, 4, 2, 3, 1, 5, 6, 7]);
    }

    #[test]
    fn perm_swap_gate_fail_leaves_individual_unchanged() {
        // p_m = 0.0 => gate always fails => output is a bit-identical clone.
        let p = PermProblem::new(6);
        let space = p.space();
        let gen = PermSwapGenerator::from_params(&serde_json::json!({"p_m": 0.0})).unwrap();
        let pop = Population { individuals: vec![g(vec![3, 1, 4, 0, 5, 2])], fitness: vec![0.0] };
        let mut eval = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(4, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
        let off = gen.generate(&pop, &mut ctx);
        assert_eq!(perm_values(&off[0]), perm_values(&pop.individuals[0]));
    }

    #[test]
    fn perm_swap_validity_property_seeded_batch() {
        let n = 9usize;
        let p = PermProblem::new(n);
        let space = p.space();
        let gen = PermSwapGenerator::from_params(&serde_json::json!({"p_m": 0.9})).unwrap();
        let mut eval = Evaluator::new(&p, 10_000_000);
        let mut rng = RngStream::from_master(6, &[0]);
        let mut bb = Blackboard::new();
        for _ in 0..500 {
            let pop = pop_n(5, n as u32);
            let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
            let off = gen.generate(&pop, &mut ctx);
            assert_eq!(off.len(), pop.len());
            for ind in &off { assert!(is_permutation(perm_values(ind), n)); }
        }
    }

    #[test]
    fn perm_swap_deterministic_same_seed() {
        let n = 7usize;
        let p = PermProblem::new(n);
        let space = p.space();
        let gen = PermSwapGenerator::from_params(&serde_json::json!({"p_m": 0.5})).unwrap();
        let pop = pop_n(5, n as u32);
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
    fn perm_swap_draw_order_matches_pinned_structure_via_raw_replay() {
        // Twin-stream: per individual, the pm gate (next_f64), then IF it
        // passes, position a (next_below(n)) and rejection-sampled b.
        let n = 7usize;
        let p = PermProblem::new(n);
        let space = p.space();
        let gen = PermSwapGenerator::from_params(&serde_json::json!({"p_m": 0.6})).unwrap();
        let pop = pop_n(6, n as u32);

        let mut rng = RngStream::from_master(23, &[]);
        let rng_before = rng.clone();
        let mut eval = Evaluator::new(&p, 1000);
        let mut bb = Blackboard::new();
        {
            let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
            let off = gen.generate(&pop, &mut ctx);
            assert_eq!(off.len(), pop.len());
        }

        let mut twin = rng_before;
        for _ in 0..pop.len() {
            let gate = twin.next_f64();
            if gate < 0.6 {
                let a = twin.next_below(n as u64);
                let mut b = twin.next_below(n as u64);
                while b == a { b = twin.next_below(n as u64); }
            }
        }
        assert_eq!(rng.next_f64(), twin.next_f64(),
            "gen/perm-swap must consume exactly: pm gate, then (if gated) position a and rejection-sampled b, per individual");
    }

    #[test]
    fn perm_swap_meta_block_restricted_default_min_pop() {
        let m = PermSwapGenerator::from_params(&serde_json::json!({})).unwrap().meta();
        assert_eq!(m.min_pop, 1);
        assert!(m.supports_block("permutation"));
        assert!(!m.supports_block("float"));
        assert_eq!(m.kind, "gen/perm-swap");
    }

    #[test]
    fn perm_swap_default_p_m_is_one_over_n() {
        let n = 6usize;
        let p = PermProblem::new(n);
        let space = p.space();
        let gen = PermSwapGenerator::from_params(&serde_json::json!({})).unwrap();
        assert_eq!(gen.p_m, None); // resolved lazily against dim inside generate()
        // Exercise: with p_m defaulting to 1/6, a next_f64() draw of exactly
        // 1/6 must fail the gate (strict `<`), confirming the resolved value.
        let pop = Population { individuals: vec![g(vec![0, 1, 2, 3, 4, 5])], fitness: vec![0.0] };
        let mut eval = Evaluator::new(&p, 100);
        // Find a seed whose first next_f64() draw is >= 1/6 (gate fails under 1/6 default).
        let seed = (0..2000u64).find(|&s| {
            let mut r = RngStream::from_master(s, &[0]);
            r.next_f64() >= 1.0 / 6.0
        }).unwrap();
        let mut rng = RngStream::from_master(seed, &[0]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
        let off = gen.generate(&pop, &mut ctx);
        assert_eq!(perm_values(&off[0]), &vec![0, 1, 2, 3, 4, 5], "gate must fail (unchanged) when the draw is >= 1/n");
    }

    #[test]
    fn perm_swap_params_validate() {
        assert!(PermSwapGenerator::from_params(&serde_json::json!({"p_m": 1.5})).is_err());
        assert!(PermSwapGenerator::from_params(&serde_json::json!({"p_m": -0.1})).is_err());
        assert!(PermSwapGenerator::from_params(&serde_json::json!({"p_m": 0.3})).is_ok());
    }

    // ---- registry / spec-validation ----

    #[test]
    fn registry_resolves_all_three_kinds() {
        let mut reg = Registry::new();
        register(&mut reg);
        assert!(reg.build_initializer("init/perm-random", &serde_json::json!({})).is_ok());
        assert!(reg.build_generator("gen/ox", &serde_json::json!({})).is_ok());
        assert!(reg.build_generator("gen/perm-swap", &serde_json::json!({})).is_ok());
    }

    #[test]
    fn spec_validation_rejects_gen_ox_on_float_only_space() {
        use sezgi_core::spec::{AlgorithmSpec, ComponentSpec, SpecError, StageSpec, TerminationSpec};
        let mut reg = Registry::new();
        register(&mut reg);
        crate::boundary::register(&mut reg);
        crate::replace::register(&mut reg);
        let space = SearchSpace::new(vec![Block::Float { lo: 0.0, hi: 1.0, n: 3 }]).unwrap();
        let spec = AlgorithmSpec {
            name: "t".into(), pop_size: 10,
            init: ComponentSpec { kind: "init/perm-random".into(), params: serde_json::json!({}) },
            boundary: ComponentSpec { kind: "boundary/clamp".into(), params: serde_json::json!({}) },
            stages: vec![StageSpec {
                generator: ComponentSpec { kind: "gen/ox".into(), params: serde_json::json!({}) },
                replacer: ComponentSpec { kind: "replace/one-to-one-greedy".into(), params: serde_json::json!({}) },
                adapter: None,
            }],
            termination: TerminationSpec { budget: 100, target: None },
            restart: None,
        };
        let e = spec.validate(&reg, &space);
        match e {
            Err(SpecError::UnsupportedBlock { kind, block }) => {
                assert_eq!(kind, "init/perm-random", "the FIRST unsupported component encountered (init, checked before generator) must be reported");
                assert_eq!(block, "float");
            }
            other => panic!("expected SpecError::UnsupportedBlock, got {other:?}"),
        }
    }

    #[test]
    fn spec_validation_accepts_gen_ox_on_permutation_space() {
        use sezgi_core::spec::{AlgorithmSpec, ComponentSpec, StageSpec, TerminationSpec};
        let mut reg = Registry::new();
        register(&mut reg);
        crate::boundary::register(&mut reg);
        crate::replace::register(&mut reg);
        let space = SearchSpace::new(vec![Block::Permutation { n: 6 }]).unwrap();
        let spec = AlgorithmSpec {
            name: "t".into(), pop_size: 10,
            init: ComponentSpec { kind: "init/perm-random".into(), params: serde_json::json!({}) },
            boundary: ComponentSpec { kind: "boundary/clamp".into(), params: serde_json::json!({}) },
            stages: vec![
                StageSpec {
                    generator: ComponentSpec { kind: "gen/ox".into(), params: serde_json::json!({}) },
                    replacer: ComponentSpec { kind: "replace/one-to-one-greedy".into(), params: serde_json::json!({}) },
                    adapter: None,
                },
                StageSpec {
                    generator: ComponentSpec { kind: "gen/perm-swap".into(), params: serde_json::json!({}) },
                    replacer: ComponentSpec { kind: "replace/one-to-one-greedy".into(), params: serde_json::json!({}) },
                    adapter: None,
                },
            ],
            termination: TerminationSpec { budget: 100, target: None },
            restart: None,
        };
        assert!(spec.validate(&reg, &space).is_ok());
    }

    #[test]
    fn spec_validation_rejects_gen_perm_swap_on_float_only_space() {
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
                generator: ComponentSpec { kind: "gen/perm-swap".into(), params: serde_json::json!({}) },
                replacer: ComponentSpec { kind: "replace/one-to-one-greedy".into(), params: serde_json::json!({}) },
                adapter: None,
            }],
            termination: TerminationSpec { budget: 100, target: None },
            restart: None,
        };
        // init/uniform is SupportedBlocks::All so it will not be the one that trips; gen/perm-swap must.
        let e = spec.validate(&reg, &space);
        match e {
            Err(SpecError::UnsupportedBlock { kind, block }) => {
                assert_eq!(kind, "gen/perm-swap");
                assert_eq!(block, "float");
            }
            other => panic!("expected SpecError::UnsupportedBlock, got {other:?}"),
        }
    }

    // ---- gen/ga-perm (M3-3 Task 4, fused generator) ----

    #[test]
    fn ga_perm_generator_meta_min_pop_2_and_block_restricted() {
        let m = GaPermGenerator::from_params(&serde_json::json!({})).unwrap().meta();
        assert_eq!(m.min_pop, 2);
        assert!(m.supports_block("permutation"));
        assert!(!m.supports_block("float"));
        assert_eq!(m.kind, "gen/ga-perm");
    }

    #[test]
    fn ga_perm_params_validate() {
        assert!(GaPermGenerator::from_params(&serde_json::json!({"pc": 1.5})).is_err());
        assert!(GaPermGenerator::from_params(&serde_json::json!({"tournament_k": 0})).is_err());
        assert!(GaPermGenerator::from_params(&serde_json::json!({"p_m": -0.1})).is_err());
        assert!(GaPermGenerator::from_params(&serde_json::json!({"p_m": 1.1})).is_err());
        let g = GaPermGenerator::from_params(&serde_json::json!({})).unwrap();
        assert_eq!(g.tournament_k, 2);
        assert_eq!(g.pc, 0.8);
        assert_eq!(g.p_m, None); // resolved lazily against dim inside generate()
    }

    #[test]
    fn ga_perm_generator_validity_property_seeded_batch() {
        let n = 10usize;
        let p = PermProblem::new(n);
        let space = p.space();
        let gen = GaPermGenerator::from_params(&serde_json::json!({})).unwrap();
        let mut eval = Evaluator::new(&p, 1_000_000);
        let mut rng = RngStream::from_master(31, &[0]);
        let mut bb = Blackboard::new();
        for _ in 0..200 {
            let pop = pop_n(9, n as u32);
            let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
            let off = gen.generate(&pop, &mut ctx);
            assert_eq!(off.len(), pop.len());
            for ind in &off { assert!(is_permutation(perm_values(ind), n)); }
        }
    }

    #[test]
    fn ga_perm_generator_deterministic_same_seed() {
        let n = 8usize;
        let p = PermProblem::new(n);
        let space = p.space();
        let gen = GaPermGenerator::from_params(&serde_json::json!({})).unwrap();
        let pop = pop_n(6, n as u32);
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
    fn ga_perm_generator_min_pop_below_2_panics() {
        let n = 4usize;
        let p = PermProblem::new(n);
        let space = p.space();
        let gen = GaPermGenerator::from_params(&serde_json::json!({})).unwrap();
        let pop = pop_n(1, n as u32);
        let mut eval = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gen.generate(&pop, &mut ctx)
        }));
        assert!(result.is_err(), "gen/ga-perm must reject pop_size < 2 at runtime as a backstop");
    }

    #[test]
    fn ga_perm_generator_draw_order_matches_pinned_structure_via_raw_replay() {
        // Twin-stream raw-replay: per pair, tournament_k*2 draws
        // (next_below(pop.len())), the pc gate (next_f64), THEN IF it
        // passes, draw_distinct_pair's cut-point draws -- THEN, for c1 then
        // c2 in that order, the p_m gate (next_f64), and IF it passes,
        // draw_distinct_pair's swap-position draws. See GaPermGenerator's
        // doc "Draw order" for the full contract this replays.
        let n = 8usize;
        let p = PermProblem::new(n);
        let space = p.space();
        let gen = GaPermGenerator::from_params(
            &serde_json::json!({"tournament_k": 3, "pc": 0.8, "p_m": 0.5})).unwrap();
        let pop = pop_n(7, n as u32);

        let mut rng = RngStream::from_master(19, &[]);
        let rng_before = rng.clone();
        let mut eval = Evaluator::new(&p, 1000);
        let mut bb = Blackboard::new();
        {
            let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
            let off = gen.generate(&pop, &mut ctx);
            assert_eq!(off.len(), pop.len());
        }

        let draw_distinct_pair_twin = |twin: &mut RngStream| {
            let i1 = twin.next_below(n as u64);
            let mut i2 = twin.next_below(n as u64);
            while i2 == i1 { i2 = twin.next_below(n as u64); }
        };

        let mut twin = rng_before;
        let mut produced = 0usize;
        while produced < pop.len() {
            for _ in 0..gen.tournament_k { twin.next_below(pop.len() as u64); }
            for _ in 0..gen.tournament_k { twin.next_below(pop.len() as u64); }
            let pc_gate = twin.next_f64();
            if pc_gate < gen.pc { draw_distinct_pair_twin(&mut twin); }
            // Both children are mutation-gated, in order (c1 then c2),
            // regardless of whether a second child is actually pushed to
            // `out` -- generate()'s own loop mutates c1/c2 BEFORE checking
            // whether room remains for c2.
            for _ in 0..2 {
                let pm_gate = twin.next_f64();
                if pm_gate < 0.5 { draw_distinct_pair_twin(&mut twin); }
            }
            produced += 1;
            if produced < pop.len() { produced += 1; }
        }
        assert_eq!(rng.next_f64(), twin.next_f64(),
            "gen/ga-perm must consume exactly: tournament x2, pc gate, (if gated) OX cut draws, \
             then per-child (c1,c2) pm gate + (if gated) swap-position draws");
    }

    #[test]
    fn ga_perm_default_p_m_is_one_over_n() {
        // Same technique as perm_swap_default_p_m_is_one_over_n: with pc
        // forced to 0.0 (no crossover, so c1=p1/c2=p2 with zero extra
        // draws), the very first RNG draw after the two tournaments is the
        // c1 mutation gate -- find a seed where that draw is >= 1/n (n=6),
        // so the default p_m=1/6 gate must fail and c1 stays unchanged.
        let n = 6usize;
        let p = PermProblem::new(n);
        let space = p.space();
        let gen = GaPermGenerator::from_params(&serde_json::json!({"pc": 0.0})).unwrap();
        let pop = Population {
            individuals: vec![g(vec![0, 1, 2, 3, 4, 5]), g(vec![5, 4, 3, 2, 1, 0])],
            fitness: vec![0.0, 1.0],
        };
        let mut eval = Evaluator::new(&p, 100);
        let seed = (0..2000u64).find(|&s| {
            let mut r = RngStream::from_master(s, &[0]);
            for _ in 0..4 { r.next_below(2); } // two tournaments, tournament_k=2, pop_size=2
            let _pc_gate = r.next_f64(); // pc=0.0, gate always fails (next_f64() in [0,1) is never < 0.0)
            r.next_f64() >= 1.0 / 6.0 // c1's pm gate
        }).unwrap();
        let mut rng = RngStream::from_master(seed, &[0]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
        let off = gen.generate(&pop, &mut ctx);
        // c1 is whichever parent tournament 1 selected, unmutated (pc=0.0
        // means no crossover, and the found seed's pm gate fails for c1).
        assert!(is_permutation(perm_values(&off[0]), n));
        assert!(perm_values(&off[0]) == &vec![0, 1, 2, 3, 4, 5]
            || perm_values(&off[0]) == &vec![5, 4, 3, 2, 1, 0],
            "with pc=0.0, c1 must be an unmodified clone of a parent when its pm gate (default 1/n) fails: {:?}",
            perm_values(&off[0]));
    }
}
