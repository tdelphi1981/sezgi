//! `gen/compound` (M3-8 Task 5): the per-block dispatch `Generator` that
//! makes MIXED search spaces reachable. Every typed generator family added
//! in M3-8 Tasks 2-4 (`bin_ops::GaBinGenerator`, `int_ops::GaIntGenerator`,
//! `cat_ops::GaCatGenerator`) -- plus the pre-existing `ga::GaRealGenerator`/
//! `perm::GaPermGenerator` -- reads and writes exactly ONE block of its own
//! type (`ctx.space.blocks()[0]`, `pop.individuals[i].blocks[0]`), by design
//! (see each module's own doc, "single block, not `nsga2_run`'s multi-block
//! MO path"). `gen/compound` is the component that composes several of
//! these single-block generators, one per block of a real multi-block
//! space, into one fused offspring genotype -- "jMetal `CompositeCrossover`-
//! shaped" (Task 5 brief): apply one operator per variable-type segment,
//! then concatenate the segments back into a whole individual.
//!
//! ## Design spec §3
//!
//! "Operators declare the block types they support as metadata... the spec
//! validator checks type compatibility before run time, when binding the
//! graph to the problem, and gives a clear error. In mixed spaces, operators
//! are applied per block (compound operator)." This module IS that "per
//! block (compound operator)" mechanism.
//!
//! ## Controller ruling (binding, plan decision 1 -- recorded in this
//! milestone's `progress.md`): compound dispatch is a COMPONENT, not engine
//! surgery
//!
//! `sezgi_core::spec::AlgorithmSpec::validate`'s space-wide block-support
//! check (`spec.rs`, "(2) block support") requires EVERY built component's
//! `ComponentMeta` to support EVERY block tag present in the space -- it has
//! no per-block-INDEX granularity (a component either supports a tag
//! everywhere in the space, or nowhere). [`CompoundGenerator`] therefore
//! declares [`SupportedBlocks::All`] (it genuinely can dispatch on any
//! block type its sub-generator table below covers), and performs its OWN,
//! finer-grained per-block-INDEX validation internally
//! ([`CompoundGenerator::validate_against_space`]).
//!
//! **Update (M3-8 deferral, closed):** at the time this ruling was recorded,
//! `spec.rs`/`sezgi_core::engine::Engine` had no hook a component could plug
//! a per-block check into during `AlgorithmSpec::validate`, so
//! [`CompoundGenerator::validate_against_space`]'s result was only reachable
//! through a direct call or (indirectly, as a panic) through `generate()` --
//! see the paragraph below. `sezgi_core::component::Generator` has since
//! grown exactly such a hook, `validate_space` (default no-op; every other
//! generator is unaffected), called from `AlgorithmSpec::validate` right
//! after each stage's generator is built. [`CompoundGenerator`]'s
//! `Generator` impl overrides it to delegate straight to
//! `validate_against_space`, so a mis-assigned/wrong-arity compound spec now
//! fails `AlgorithmSpec::validate`/`Engine::from_spec` directly, with a real
//! `Result`, before any `generate()` call. `spec.rs`'s change is minimal and
//! generic (the new hook, not a compound-specific one) -- `gen/compound` is
//! still registered exactly like any other generator
//! (`Registry::register_generator`), and its dispatch table below still
//! calls straight into `bin_ops`/`int_ops`/`cat_ops`/`ga`/`perm`'s own
//! already-registered, already-tested `from_params` constructors (consumed,
//! not re-implemented).
//!
//! `Generator::generate`'s signature (`fn generate(&self, pop: &Population,
//! ctx: &mut Ctx) -> Vec<Genotype>`, no `Result`) still cannot itself return
//! a validation error, so [`CompoundGenerator::generate`] still calls
//! [`CompoundGenerator::validate_against_space`] as its very first action
//! and PANICS with the same message a direct call to that method would
//! return, on mismatch -- now unreachable through the normal
//! `AlgorithmSpec::validate`/`Engine::from_spec` path (which fails earlier,
//! per the update above), and kept only as a defensive backstop for the
//! direct-construction path (`from_params` followed directly by
//! `.generate()`, bypassing `AlgorithmSpec::validate`) -- the same "dynamic
//! backstop" pattern `sezgi_core::component::ComponentMeta`'s own doc
//! describes for space-dependent requirements a fixed `usize` can't express
//! (e.g. Nelder-Mead's `pop_size >= dim + 1`), and the same pattern every
//! typed generator's own `assert!(pop.len() >= 2, ...)` already uses. Tests
//! below exercise [`CompoundGenerator::validate_against_space`] DIRECTLY (a
//! real `Result`, no panic, no engine run needed) for the "mis-assignment
//! validation errors... naming both" requirement (Task 5 brief), AND (per
//! the M3-8 deferral closed above) at the `AlgorithmSpec::validate` level --
//! this is now the earliest point in this framework's component model a
//! per-block-index check can run, since no `from_params` (not
//! `CompoundGenerator`'s, not
//! any other component's) ever receives the target `SearchSpace`.
//!
//! ## Sub-generator eligibility (registry-driven dispatch)
//!
//! `blocks[i].kind` may be ANY registered `gen/*` kind that passes the
//! eligibility gates below. Sub-generators are built through a fresh
//! builtins `Registry` (`register_builtins`) constructed inside `from_params`
//! (registration is cheap and non-recursive), so every registered
//! generator's own `from_params` runs unmodified. A sub is rejected, with an
//! error naming its kind, the block index and the reason, unless:
//!
//! 1. it is not `gen/compound` itself ("gen/compound cannot nest inside
//!    itself");
//! 2. its `ComponentMeta::requires` and `provides` are both empty, OR its
//!    kind is listed in [`SELF_CONTAINED_COMPOUND_KINDS`] (`gen/gsa`,
//!    `gen/ba`: generators that bootstrap, read and write only their own
//!    blackboard keys inside `generate`). A generator that keeps
//!    engine-level blackboard state paired with a replacer/adapter (PSO,
//!    CMA-ES, SHADE, jDE, ...) is rejected, since gen/compound cannot route
//!    that pairing per block;
//! 3. its `ComponentMeta::offspring` is `OffspringCount::PopLen` (rejects
//!    `gen/hs`, `gen/nelder-mead`, which emit one offspring per call);
//! 4. its `ComponentMeta::internal_eval` is false (rejects `gen/hho`, which
//!    evaluates single-block genotypes against the full problem inside
//!    `generate()`).
//!
//! Block-type support (e.g. `gen/de` on a Binary block) is still checked per
//! block index in [`CompoundGenerator::validate_against_space`]. The
//! compound's own merged meta has empty requires/provides by construction
//! (an allow-listed sub's state lives in its nested blackboard, below, not
//! on the engine's blackboard); `min_pop` is the max over the subs.
//!
//! ## Nested per-block blackboards (0.1.5)
//!
//! Each block `i` gets its OWN [`Blackboard`], stored as a slot of the
//! parent blackboard under the key `"cmp{i}/bb"`. Before calling sub `i`,
//! `generate` takes that slot out (`take_raw`), downcasts it (or starts from
//! an empty default), hands `&mut` of it to the sub's `Ctx`, and inserts it
//! back afterwards. Every sub's keys are therefore isolated: two
//! `gen/gsa` Float blocks of different sizes no longer share
//! `gsa/velocity`. The swap consumes no RNG, so all-stateless specs are
//! bit-identical to the pre-0.1.5 behavior. The engine's restart wipe drops
//! the parent slots, so allow-listed subs re-bootstrap after a restart.
//!
//! ## Index-alignment caveat (velocity-style subs)
//!
//! `gen/gsa`/`gen/ba` keep a velocity matrix aligned by row to population
//! index (`velocity[i]` <-> `pop[i]`). That is only correct under
//! index-preserving stage replacers: generational, one-to-one, or
//! bat-loudness-greedy. A mu+lambda-style replacer that reorders survivors
//! breaks the alignment. The Python hybrid wrap always uses the host
//! preset's own replacer, so gsa/ba stay aligned and are never mixed with a
//! reordering one.
//!
//! The standalone crossover-only / mutation-only typed operators
//! (`gen/bin-2pt`, `gen/bit-flip`, `gen/int-sbx`, `gen/int-pm`, `gen/cat-ux`,
//! `gen/cat-reset`, `gen/ox`, `gen/perm-swap`) pass the gates and are
//! therefore admitted, but each is only half of a variation step; the fused
//! presets (`gen/ga-bin`, `gen/ga-int`, `gen/ga-cat`, `gen/ga-real`,
//! `gen/ga-perm`) remain the recommended choices for discrete blocks.
//!
//! ## Config schema (spec.rs's `ComponentSpec` shape, reused directly)
//!
//! ```json
//! {
//!   "blocks": [
//!     {"kind": "gen/ga-real", "tournament_k": 2, "pc": 0.9, "eta_c": 15.0, "eta_m": 20.0},
//!     {"kind": "gen/ga-int", "p_c": 0.9},
//!     {"kind": "gen/ga-cat", "p_c": 0.9},
//!     {"kind": "gen/ga-bin", "p_c": 0.9}
//!   ]
//! }
//! ```
//! Note the crossover-probability spelling shown mixed above (`pc` on
//! `gen/ga-real`, `p_c` on the typed families): both spellings are now
//! accepted EVERYWHERE, canonical `p_c` (`p_m` for the analogous mutation
//! key, where a family has one) -- `pc`/`pm` are accepted legacy aliases on
//! every family, including `gen/ga-real`/`gen/ox`/`gen/ga-perm` (see
//! `params.rs`'s own doc for the full resolution rule and history). If a
//! block sets BOTH spellings, the canonical key wins and the legacy key is
//! ignored outright. Every `from_params` here is still permissive for any
//! OTHER unknown key (silently ignored), so a genuinely misspelled key
//! (neither `p_c` nor `pc`) still silently falls back to the default instead
//! of erroring.
//!
//! `blocks` is an ORDERED array, one entry per block of the target space, in
//! the SAME order `SearchSpace::blocks()` lists them. Each entry deserializes
//! directly as `sezgi_core::spec::ComponentSpec` (`{"kind": ..., ...flattened
//! params}` -- the exact shape every `[[stages]].generator`/`.replacer`
//! entry already uses in a TOML/JSON `AlgorithmSpec`), so a `blocks[i]` entry
//! is written exactly like a top-level stage's `generator` field would be.
//!
//! ## The single-block VIEW contract (mirrors bin_ops.rs's own Task 2 note,
//! carried forward per this milestone's pre-flight scan)
//!
//! Every typed sub-generator reads `ctx.space.blocks()[0]` and
//! `pop.individuals[i].blocks[0]` ONLY -- in a space with more than one
//! block, handing a sub-generator the FULL space/population would make it
//! silently operate on block 0 only, regardless of which block index it was
//! configured for. [`CompoundGenerator::generate`] therefore constructs, for
//! each block index `i`, a ONE-BLOCK [`SearchSpace`] (`vec![space.blocks()[i]
//! .clone()]`) and a ONE-BLOCK [`Population`] (each individual's genotype
//! reduced to `vec![individual.blocks[i].clone()]`, fitness carried over
//! unchanged -- sub-generators that select parents via tournament need
//! `pop.fitness`, and fitness is a whole-genotype property, the same value
//! for every block's view), and calls `sub.generate(&sub_pop, &mut sub_ctx)`
//! on that view -- NEVER the full space/population. The `pop.len()`-many
//! per-block offspring segments are then stitched back together
//! index-for-index: `offspring[j].blocks[i] = sub_output[i][j].blocks[0]`.
//!
//! ## RNG stream design + exactness-proof strategy (Task 5 brief: "you own
//! this decision -- document it")
//!
//! `gen/compound` does NOT split or fork `ctx.rng` per block -- it threads
//! the SAME live `RngStream` through every sub-generator call, SEQUENTIALLY,
//! in block order (index `0, 1, 2, ...`): block 0's sub-generator consumes
//! whatever draws it needs first, then block 1's sub-generator continues
//! drawing from the exact stream position block 0's call left behind, and so
//! on. This is a deliberate, simple, provable choice over alternatives (e.g.
//! `ctx.rng.split(i)` per block, which would let blocks draw independently
//! but would also mean two different compound configs with a different
//! NUMBER of preceding blocks could no longer be compared draw-for-draw
//! against a standalone run of block `i`'s own generator): with sequential
//! reuse, a genotype's block `i` output depends only on (a) block `i`'s own
//! sub-generator/params, (b) the population's per-block-`i` values +
//! fitness, and (c) the RNG stream'S POSITION when block `i`'s turn starts
//! -- which is fully determined by how many draws blocks `0..i` consumed,
//! not by anything else about them. This makes the composition's own
//! "adds NOTHING beyond routing" claim (Task 5 brief) MECHANICALLY provable:
//! run `gen/compound` once from a known starting `RngStream` state; then,
//! from a FRESH CLONE of that same starting state, replay each block `i` in
//! order by calling ITS OWN standalone generator directly (same sub-pop
//! view, same params) on the twin stream -- block `i`'s replayed output must
//! equal `gen/compound`'s own block-`i` segment, and after the last block,
//! the two `RngStream`s must be in the identical state (same next draw).
//! `compound_generator_engine_run_exactness_matches_standalone_sub_generators_per_block`
//! below is exactly this proof, run through a real `sezgi_core::engine::Engine`
//! (not a bare `generate()` call) on a Float+Int+Categorical+Binary mixed
//! space.

use sezgi_core::component::*;
use sezgi_core::problem::Population;
use sezgi_core::space::{Block, Genotype, SearchSpace};
use sezgi_core::spec::ComponentSpec;
use sezgi_core::state::Blackboard;

/// Local counterpart to `sezgi_core::spec`'s private `block_tag` -- same
/// tag strings (`"float"`/`"int"`/`"categorical"`/`"permutation"`/
/// `"binary"`), same convention every `ComponentMeta::supports_block` call
/// in this crate already uses (see e.g. `bin_ops.rs`'s own
/// `SupportedBlocks::Only(vec!["binary"])`). Duplicated here (not exported
/// from `sezgi_core::spec`) rather than widening that module's public
/// surface for one caller -- a small, stable, five-arm match.
fn block_tag(b: &Block) -> &'static str {
    match b {
        Block::Float { .. } => "float",
        Block::Int { .. } => "int",
        Block::Categorical { .. } => "categorical",
        Block::Permutation { .. } => "permutation",
        Block::Binary { .. } => "binary",
    }
}

/// Builds one sub-generator from its `ComponentSpec` through a builtins
/// registry (R-A) and applies the eligibility gates (R-B). `index` is the
/// position in the `blocks` list, which IS the target block index. Errors
/// carry `gen/compound` as the reporting component (`InvalidParams { kind:
/// "gen/compound", .. }`) for registry misses and eligibility rejections --
/// distinct from a sub-generator's OWN `from_params` errors (which still
/// surface as-is, carrying THEIR kind, e.g. `"gen/ga-bin"` for an
/// out-of-range `p_c`).
/// Stateful generator kinds admitted as `gen/compound` subs. The list is
/// MANUAL on purpose: `ComponentMeta` cannot see engine-level pairing (e.g.
/// `gen/pso` provides keys that `replace/pso-commit` maintains outside
/// `generate`), so requires/provides alone cannot distinguish self-contained
/// state from paired state. A kind qualifies only if its generator
/// bootstraps, reads and writes ONLY its own keys inside `generate`, and its
/// preset replacer is stateless, `SupportedBlocks::All`, index-preserving,
/// and needs no adapter. State lives in the per-block nested blackboard.
const SELF_CONTAINED_COMPOUND_KINDS: &[&str] = &["gen/gsa", "gen/ba"];

fn build_sub_generator(
    reg: &Registry,
    index: usize,
    spec: &ComponentSpec,
) -> Result<Box<dyn Generator>, ComponentError> {
    let err = |reason: String| ComponentError::InvalidParams { kind: "gen/compound".into(), reason };
    let kind = spec.kind.as_str();
    if kind == "gen/compound" {
        return Err(err(format!("block {index}: gen/compound cannot nest inside itself")));
    }
    let sub = match reg.build_generator(kind, &spec.params) {
        Ok(g) => g,
        Err(ComponentError::UnknownKind(k)) => {
            return Err(err(format!(
                "unsupported sub-generator kind `{k}` at block {index} (not a registered generator)"
            )))
        }
        Err(e) => return Err(e),
    };
    let meta = sub.meta();
    if (!meta.requires.is_empty() || !meta.provides.is_empty())
        && !SELF_CONTAINED_COMPOUND_KINDS.contains(&kind)
    {
        return Err(err(format!(
            "sub-generator `{kind}` at block {index} is not eligible: it keeps engine-level \
             blackboard state and needs its paired replacer/adapter, which gen/compound cannot \
             route per block"
        )));
    }
    if meta.offspring != OffspringCount::PopLen {
        return Err(err(format!(
            "sub-generator `{kind}` at block {index} is not eligible: it does not produce \
             pop.len() offspring per call (gen/compound stitches pop.len() offspring per block)"
        )));
    }
    if meta.internal_eval {
        return Err(err(format!(
            "sub-generator `{kind}` at block {index} is not eligible: it evaluates genotypes \
             internally inside generate(), which breaks on the single-block view gen/compound \
             hands each sub-generator"
        )));
    }
    Ok(sub)
}

/// `gen/compound` -- see module doc for the full design (schema, single-
/// block view contract, RNG stream design). `subs[i]` is the sub-generator
/// for block index `i` of the target space; `min_pop` is the max of every
/// sub-generator's own `ComponentMeta::min_pop` (so `gen/compound`'s own
/// `meta()` reports a real, non-trivial floor to `AlgorithmSpec::validate`'s
/// "(4) minimum population size" check, rather than always reporting `1`
/// and relying purely on each sub-generator's own runtime assert as a
/// backstop).
pub struct CompoundGenerator {
    subs: Vec<Box<dyn Generator>>,
    min_pop: usize,
}

impl CompoundGenerator {
    /// Parses `{"blocks": [{...sub-generator spec...}, ...]}` (module doc's
    /// pinned schema) and builds every sub-generator via
    /// [`build_sub_generator`] (registry-driven, eligibility-gated). Does NOT (cannot -- see module doc) check
    /// sub-generator `i`'s block-type support against a real
    /// `SearchSpace`'s block `i` here; call
    /// [`CompoundGenerator::validate_against_space`] once the target space
    /// is known (this is exactly what [`CompoundGenerator::generate`] does,
    /// as its first action, on every call).
    pub fn from_params(p: &serde_json::Value) -> Result<Self, ComponentError> {
        let err = |reason: String| ComponentError::InvalidParams { kind: "gen/compound".into(), reason };
        let blocks_val = p
            .get("blocks")
            .ok_or_else(|| err("missing required `blocks` array (one sub-generator spec per space block)".into()))?;
        let specs: Vec<ComponentSpec> = serde_json::from_value(blocks_val.clone())
            .map_err(|e| err(format!("`blocks` must be an array of {{\"kind\": ..., ...}} sub-generator specs: {e}")))?;
        if specs.is_empty() {
            return Err(err("`blocks` must contain at least one sub-generator spec".into()));
        }
        // R-A: fresh builtins registry; registration is cheap and non-recursive.
        let mut reg = Registry::new();
        crate::register_builtins(&mut reg);
        let subs: Vec<Box<dyn Generator>> = specs
            .iter()
            .enumerate()
            .map(|(i, spec)| build_sub_generator(&reg, i, spec))
            .collect::<Result<_, _>>()?;
        let min_pop = subs.iter().map(|s| s.meta().min_pop).max().unwrap_or(1);
        Ok(Self { subs, min_pop })
    }

    /// The "mis-assignment"/"wrong-arity" build-time check (Task 5 brief):
    /// `self.subs.len()` must equal `space.blocks().len()` (wrong-arity, if
    /// not), and each `self.subs[i]` must
    /// `.meta().supports_block(block_tag(space.blocks()[i]))` (mis-
    /// assignment, if not) -- both errors name the offending sub-generator
    /// kind AND the block tag/index, per the brief ("naming both"). Callable
    /// directly (no engine/panic involved) -- see module doc's "Controller
    /// ruling" section for why this, not a `spec.rs`/`Engine` hook, is where
    /// per-block-index validation lives.
    pub fn validate_against_space(&self, space: &SearchSpace) -> Result<(), ComponentError> {
        let err = |reason: String| ComponentError::InvalidParams { kind: "gen/compound".into(), reason };
        if self.subs.len() != space.blocks().len() {
            return Err(err(format!(
                "blocks list length {} does not match the space's block count {} (one sub-generator spec is required per block, in order)",
                self.subs.len(),
                space.blocks().len()
            )));
        }
        for (i, (sub, block)) in self.subs.iter().zip(space.blocks()).enumerate() {
            let tag = block_tag(block);
            let meta = sub.meta();
            if !meta.supports_block(tag) {
                return Err(err(format!(
                    "sub-generator `{}` at block index {i} does not support block type `{tag}`",
                    meta.kind
                )));
            }
        }
        Ok(())
    }
}

impl Generator for CompoundGenerator {
    /// Delegates to [`CompoundGenerator::validate_against_space`] -- this is
    /// the M3-8 deferral this module's doc anticipated (see "Controller
    /// ruling" above, written when no such hook existed): `sezgi_core`'s
    /// `Generator::validate_space` build-time hook now lets `gen/compound`
    /// veto a mis-assigned space through `AlgorithmSpec::validate`/
    /// `Engine::from_spec` directly, with a real `Result`, instead of only
    /// through the `generate()`-time panic below.
    fn validate_space(&self, space: &SearchSpace) -> Result<(), ComponentError> {
        self.validate_against_space(space)
    }

    /// See module doc's "The single-block VIEW contract" and "RNG stream
    /// design" sections for the full algorithm and its exactness proof.
    ///
    /// Panics (via `validate_against_space`'s `Err`, unwrapped) if the space
    /// doesn't match this compound's `blocks` config. Since the addition of
    /// `Generator::validate_space` (delegated to `validate_against_space`
    /// above), this panic is UNREACHABLE through the normal
    /// `AlgorithmSpec::validate`/`Engine::from_spec` path -- that path now
    /// fails earlier, at build/validate time, with a real `Result`. This
    /// panic remains only as a defensive backstop for the direct
    /// construction path (`CompoundGenerator::from_params` followed
    /// directly by `.generate()`, bypassing `AlgorithmSpec::validate`
    /// entirely -- e.g. this module's own
    /// `generate_panics_on_mis_assigned_space` test below), which
    /// `Generator::generate`'s panic-only signature still can't express as
    /// a `Result`.
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        if let Err(e) = self.validate_against_space(ctx.space) {
            panic!("gen/compound: {e}");
        }
        let n_blocks = self.subs.len();
        let mut per_block_offspring: Vec<Vec<Genotype>> = Vec::with_capacity(n_blocks);
        for (i, sub) in self.subs.iter().enumerate() {
            let sub_space = SearchSpace::new(vec![ctx.space.blocks()[i].clone()])
                .expect("a single block taken from an already-valid SearchSpace is itself valid");
            let sub_pop = Population {
                individuals: pop
                    .individuals
                    .iter()
                    .map(|g| Genotype { blocks: vec![g.blocks[i].clone()] })
                    .collect(),
                fitness: pop.fitness.clone(),
            };
            // R-A: per-block nested blackboard, parent slot `cmp{i}/bb`.
            let bb_key = format!("cmp{i}/bb");
            let mut sub_bb: Blackboard = match ctx.bb.take_raw(&bb_key) {
                Some(raw) => raw.downcast::<Blackboard>().map(|b| *b).unwrap_or_default(),
                None => Blackboard::default(),
            };
            let off = {
                let mut sub_ctx = Ctx {
                    space: &sub_space,
                    rng: &mut *ctx.rng,
                    bb: &mut sub_bb,
                    eval: &mut *ctx.eval,
                    iteration: ctx.iteration,
                };
                sub.generate(&sub_pop, &mut sub_ctx)
            };
            ctx.bb.insert(&bb_key, sub_bb);
            assert_eq!(
                off.len(),
                pop.len(),
                "gen/compound: sub-generator `{}` for block {i} produced {} offspring, expected {} (pop.len())",
                sub.meta().kind,
                off.len(),
                pop.len()
            );
            per_block_offspring.push(off);
        }
        (0..pop.len())
            .map(|j| Genotype {
                blocks: (0..n_blocks).map(|i| per_block_offspring[i][j].blocks[0].clone()).collect(),
            })
            .collect()
    }

    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/compound", SupportedBlocks::All).with_min_pop(self.min_pop)
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_generator("gen/compound", |p| Ok(Box::new(CompoundGenerator::from_params(p)?)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::problem::{Evaluator, Problem};
    use sezgi_core::rng::RngStream;
    use sezgi_core::space::BlockValues;

    // A minimal mixed-space `Problem` (Float + Int + Categorical + Binary,
    // in that block order) used only to construct a valid
    // `Evaluator`/`Ctx`/`Engine` for these tests -- mirrors `bin_ops.rs`'s
    // own `BinProblem` scaffold. Fitness is an arbitrary but deterministic
    // combination of every block (sum of float genes, plus int genes, plus
    // categorical mismatch-from-0 count, plus binary zero-count) so that
    // tournament selection has real signal across every block type at
    // once.
    struct MixedProblem { space: SearchSpace }
    impl MixedProblem {
        fn new(n_float: usize, n_int: usize, k_cat: u32, n_cat: usize, n_bin: usize) -> Self {
            Self {
                space: SearchSpace::new(vec![
                    Block::Float { lo: -5.0, hi: 5.0, n: n_float },
                    Block::Int { lo: -5, hi: 5, n: n_int },
                    Block::Categorical { k: k_cat, n: n_cat },
                    Block::Binary { n: n_bin },
                ])
                .unwrap(),
            }
        }
    }
    impl Problem for MixedProblem {
        fn space(&self) -> &SearchSpace { &self.space }
        fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<f64> {
            pop.iter()
                .map(|g| {
                    let mut f = 0.0;
                    for b in &g.blocks {
                        f += match b {
                            BlockValues::Float(xs) => xs.iter().sum::<f64>(),
                            BlockValues::Int(xs) => xs.iter().sum::<i64>() as f64,
                            BlockValues::Cat(xs) => xs.iter().filter(|&&c| c != 0).count() as f64,
                            BlockValues::Bin(xs) => xs.iter().filter(|&&b| !b).count() as f64,
                            BlockValues::Perm(_) => 0.0,
                        };
                    }
                    f
                })
                .collect()
        }
    }

    fn mixed_pop(n: usize, n_float: usize, n_int: usize, n_cat: usize, n_bin: usize) -> Population {
        Population {
            individuals: (0..n)
                .map(|i| Genotype {
                    blocks: vec![
                        BlockValues::Float((0..n_float).map(|j| ((i + j) % 5) as f64 - 2.0).collect()),
                        BlockValues::Int((0..n_int).map(|j| ((i + j) % 5) as i64 - 2).collect()),
                        BlockValues::Cat((0..n_cat).map(|j| ((i + j) % 3) as u32).collect()),
                        BlockValues::Bin((0..n_bin).map(|j| (i + j) % 2 == 0).collect()),
                    ],
                })
                .collect(),
            fitness: (0..n).map(|i| (n - i) as f64).collect(),
        }
    }

    fn compound_spec_json() -> serde_json::Value {
        serde_json::json!({
            "blocks": [
                {"kind": "gen/ga-real", "tournament_k": 2, "pc": 0.9, "eta_c": 15.0, "eta_m": 20.0},
                {"kind": "gen/ga-int", "tournament_k": 2, "p_c": 0.9, "eta_c": 15.0, "eta_m": 20.0},
                {"kind": "gen/ga-cat", "tournament_k": 2, "p_c": 0.9},
                {"kind": "gen/ga-bin", "tournament_k": 2, "p_c": 0.9},
            ]
        })
    }

    // ==================================================================
    // from_params / registry
    // ==================================================================

    #[test]
    fn from_params_missing_blocks_is_error() {
        let e = CompoundGenerator::from_params(&serde_json::json!({}));
        assert!(matches!(e, Err(ComponentError::InvalidParams { kind, .. }) if kind == "gen/compound"));
    }

    #[test]
    fn from_params_empty_blocks_is_error() {
        let e = CompoundGenerator::from_params(&serde_json::json!({"blocks": []}));
        assert!(e.is_err());
    }

    #[test]
    fn from_params_unknown_sub_kind_is_error() {
        let e = CompoundGenerator::from_params(&serde_json::json!({"blocks": [{"kind": "gen/nope"}]}));
        match e {
            Err(ComponentError::InvalidParams { kind, reason }) => {
                assert_eq!(kind, "gen/compound");
                assert!(reason.contains("gen/nope"), "{reason}");
            }
            other => panic!("expected InvalidParams, got {}", other.is_ok()),
        }
    }

    #[test]
    fn from_params_propagates_sub_generator_param_error() {
        // gen/ga-bin's own p_c validation (bin_ops.rs) must still fire
        // through gen/compound's dispatch, unmodified.
        let e = CompoundGenerator::from_params(&serde_json::json!({
            "blocks": [{"kind": "gen/ga-bin", "p_c": 5.0}]
        }));
        assert!(matches!(e, Err(ComponentError::InvalidParams { kind, .. }) if kind == "gen/ga-bin"));
    }

    #[test]
    fn from_params_min_pop_is_max_of_subs() {
        // gen/ga-real has no explicit min_pop override (default 1); every
        // other fused typed generator declares min_pop=2 (bin_ops.rs/
        // int_ops.rs/cat_ops.rs/perm.rs). max over the four below is 2.
        let g = CompoundGenerator::from_params(&compound_spec_json()).unwrap();
        assert_eq!(g.meta().min_pop, 2);
    }

    #[test]
    fn registry_resolves_gen_compound() {
        let mut reg = Registry::new();
        register(&mut reg);
        crate::bin_ops::register(&mut reg);
        crate::int_ops::register(&mut reg);
        crate::cat_ops::register(&mut reg);
        crate::ga::register(&mut reg);
        crate::perm::register(&mut reg);
        assert!(reg.build_generator("gen/compound", &compound_spec_json()).is_ok());
    }

    #[test]
    fn meta_is_supported_blocks_all() {
        let g = CompoundGenerator::from_params(&compound_spec_json()).unwrap();
        assert!(g.meta().supports_block("float"));
        assert!(g.meta().supports_block("int"));
        assert!(g.meta().supports_block("categorical"));
        assert!(g.meta().supports_block("binary"));
        assert!(g.meta().supports_block("permutation"));
        assert_eq!(g.meta().kind, "gen/compound");
    }

    // ==================================================================
    // registry-driven dispatch + eligibility gates (R-A / R-B / R-D)
    // ==================================================================

    fn err_reason(spec: serde_json::Value) -> String {
        match CompoundGenerator::from_params(&spec) {
            Err(ComponentError::InvalidParams { kind, reason }) => {
                assert_eq!(kind, "gen/compound");
                reason
            }
            Err(e) => panic!("unexpected error variant: {e}"),
            Ok(_) => panic!("expected rejection"),
        }
    }

    fn run_generate(g: &CompoundGenerator, space: &SearchSpace, pop: &Population, seed: u64) -> Vec<Genotype> {
        let p = MixedProblem::new(1, 1, 3, 1, 1);
        let mut eval = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(seed, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
        g.generate(pop, &mut ctx)
    }

    #[test]
    fn de_on_float_with_typed_ga_on_discrete_blocks_generates_valid_shapes() {
        let g = CompoundGenerator::from_params(&serde_json::json!({
            "blocks": [
                {"kind": "gen/de", "strategy": "rand1"},
                {"kind": "gen/ga-int"},
                {"kind": "gen/ga-cat"},
                {"kind": "gen/ga-bin"},
            ]
        }))
        .unwrap();
        let p = MixedProblem::new(4, 3, 3, 2, 6);
        g.validate_against_space(p.space()).unwrap();
        let pop = mixed_pop(8, 4, 3, 2, 6);
        let off = run_generate(&g, p.space(), &pop, 11);
        assert_eq!(off.len(), 8);
        for o in &off {
            assert_eq!(o.blocks.len(), 4);
            assert!(matches!(&o.blocks[0], BlockValues::Float(v) if v.len() == 4));
            assert!(matches!(&o.blocks[1], BlockValues::Int(v) if v.len() == 3));
            assert!(matches!(&o.blocks[2], BlockValues::Cat(v) if v.len() == 2));
            assert!(matches!(&o.blocks[3], BlockValues::Bin(v) if v.len() == 6));
        }
    }

    fn float_bin_accept_and_generate(float_kind: &str) -> Result<(), String> {
        let spec = serde_json::json!({"blocks": [{"kind": float_kind}, {"kind": "gen/ga-bin"}]});
        let g = CompoundGenerator::from_params(&spec).map_err(|e| e.to_string())?;
        let space = SearchSpace::new(vec![
            Block::Float { lo: -5.0, hi: 5.0, n: 3 },
            Block::Binary { n: 6 },
        ])
        .unwrap();
        g.validate_against_space(&space).map_err(|e| e.to_string())?;
        let pop = Population {
            individuals: (0..8)
                .map(|i| Genotype {
                    blocks: vec![
                        BlockValues::Float((0..3).map(|j| ((i + j) % 5) as f64 - 2.0).collect()),
                        BlockValues::Bin((0..6).map(|j| (i + j) % 2 == 0).collect()),
                    ],
                })
                .collect(),
            fitness: (0..8).map(|i| (8 - i) as f64).collect(),
        };
        let off = run_generate(&g, &space, &pop, 3);
        assert_eq!(off.len(), 8);
        for o in &off {
            assert!(matches!(&o.blocks[0], BlockValues::Float(v) if v.len() == 3));
            assert!(matches!(&o.blocks[1], BlockValues::Bin(v) if v.len() == 6));
        }
        Ok(())
    }

    #[test]
    fn sweep_alo_with_ga_bin_acceptance() {
        float_bin_accept_and_generate("gen/alo").unwrap();
    }

    #[test]
    fn sweep_step_with_ga_bin_acceptance() {
        float_bin_accept_and_generate("gen/step").unwrap();
    }

    #[test]
    fn gwo_on_float_with_ga_int_generates_valid_shapes() {
        let g = CompoundGenerator::from_params(&serde_json::json!({
            "blocks": [{"kind": "gen/gwo"}, {"kind": "gen/ga-int"}]
        }))
        .unwrap();
        let space = SearchSpace::new(vec![
            Block::Float { lo: -5.0, hi: 5.0, n: 3 },
            Block::Int { lo: -5, hi: 5, n: 4 },
        ])
        .unwrap();
        g.validate_against_space(&space).unwrap();
        let pop = Population {
            individuals: (0..8)
                .map(|i| Genotype {
                    blocks: vec![
                        BlockValues::Float((0..3).map(|j| ((i + j) % 5) as f64 - 2.0).collect()),
                        BlockValues::Int((0..4).map(|j| ((i + j) % 5) as i64 - 2).collect()),
                    ],
                })
                .collect(),
            fitness: (0..8).map(|i| (8 - i) as f64).collect(),
        };
        let off = run_generate(&g, &space, &pop, 5);
        assert_eq!(off.len(), 8);
        for o in &off {
            assert!(matches!(&o.blocks[0], BlockValues::Float(v) if v.len() == 3));
            assert!(matches!(&o.blocks[1], BlockValues::Int(v) if v.len() == 4));
        }
    }

    #[test]
    fn rejects_nested_compound() {
        let r = err_reason(serde_json::json!({
            "blocks": [{"kind": "gen/compound", "blocks": [{"kind": "gen/ga-bin"}]}]
        }));
        assert!(r.contains("cannot nest inside itself"), "{r}");
    }

    #[test]
    fn rejects_stateful_generator_with_blackboard_reason() {
        let r = err_reason(serde_json::json!({
            "blocks": [{"kind": "gen/ga-bin"}, {"kind": "gen/pso"}]
        }));
        assert!(r.contains("gen/pso"), "{r}");
        assert!(r.contains("block 1"), "{r}");
        assert!(r.contains("blackboard state"), "{r}");
        assert!(r.contains("paired replacer/adapter"), "{r}");
    }

    #[test]
    fn rejects_single_offspring_generators() {
        let r = err_reason(serde_json::json!({"blocks": [{"kind": "gen/hs"}]}));
        assert!(r.contains("gen/hs") && r.contains("block 0"), "{r}");
        assert!(r.contains("pop.len() offspring"), "{r}");
        // gen/nelder-mead is also offspring-One but keeps blackboard state,
        // so the earlier (stateful) gate fires first; it is rejected either way.
        let r = err_reason(serde_json::json!({"blocks": [{"kind": "gen/nelder-mead"}]}));
        assert!(r.contains("gen/nelder-mead") && r.contains("block 0"), "{r}");
    }

    #[test]
    fn rejects_internal_eval_generator() {
        let r = err_reason(serde_json::json!({"blocks": [{"kind": "gen/hho"}]}));
        assert!(r.contains("gen/hho") && r.contains("block 0"), "{r}");
        assert!(r.contains("internally"), "{r}");
    }

    #[test]
    fn rejects_unknown_kind_naming_it() {
        let r = err_reason(serde_json::json!({"blocks": [{"kind": "gen/does-not-exist"}]}));
        assert!(r.contains("gen/does-not-exist"), "{r}");
    }

    #[test]
    fn de_on_binary_block_fails_block_support_validation() {
        let g = CompoundGenerator::from_params(&serde_json::json!({
            "blocks": [{"kind": "gen/de"}]
        }))
        .unwrap();
        let space = SearchSpace::new(vec![Block::Binary { n: 4 }]).unwrap();
        match g.validate_against_space(&space) {
            Err(ComponentError::InvalidParams { reason, .. }) => {
                assert!(reason.contains("gen/de") && reason.contains("binary"), "{reason}");
            }
            other => panic!("expected InvalidParams, got ok={}", other.is_ok()),
        }
    }

    #[test]
    fn compound_meta_stays_stateless_for_mixed_eligible_subs() {
        let g = CompoundGenerator::from_params(&serde_json::json!({
            "blocks": [
                {"kind": "gen/de"},
                {"kind": "gen/ga-int"},
                {"kind": "gen/ga-cat"},
                {"kind": "gen/ga-bin"},
                {"kind": "gen/gwo"},
            ]
        }))
        .unwrap();
        let m = g.meta();
        assert!(m.requires.is_empty());
        assert!(m.provides.is_empty());
        let max_sub = ["gen/de", "gen/ga-int", "gen/ga-cat", "gen/ga-bin", "gen/gwo"]
            .iter()
            .map(|k| {
                let mut reg = Registry::new();
                crate::register_builtins(&mut reg);
                reg.build_generator(k, &serde_json::json!({})).unwrap().meta().min_pop
            })
            .max()
            .unwrap();
        assert_eq!(m.min_pop, max_sub);
    }

    #[test]
    fn de_plus_ga_bin_is_deterministic_for_same_seed() {
        let g = CompoundGenerator::from_params(&serde_json::json!({
            "blocks": [{"kind": "gen/de"}, {"kind": "gen/ga-bin"}]
        }))
        .unwrap();
        let space = SearchSpace::new(vec![Block::Float { lo: -5.0, hi: 5.0, n: 3 }, Block::Binary { n: 6 }]).unwrap();
        let pop = Population {
            individuals: (0..8)
                .map(|i| Genotype {
                    blocks: vec![
                        BlockValues::Float((0..3).map(|j| ((i + j) % 5) as f64 - 2.0).collect()),
                        BlockValues::Bin((0..6).map(|j| (i + j) % 2 == 0).collect()),
                    ],
                })
                .collect(),
            fitness: (0..8).map(|i| (8 - i) as f64).collect(),
        };
        let a = run_generate(&g, &space, &pop, 7);
        let b = run_generate(&g, &space, &pop, 7);
        assert_eq!(a, b);
    }

    // ==================================================================
    // validate_against_space -- mis-assignment / wrong-arity (task-5
    // required test 4)
    // ==================================================================

    #[test]
    fn validate_wrong_arity_names_both_counts() {
        let g = CompoundGenerator::from_params(&compound_spec_json()).unwrap(); // 4 sub-specs
        let space = SearchSpace::new(vec![Block::Float { lo: 0.0, hi: 1.0, n: 2 }]).unwrap(); // 1 block
        let e = g.validate_against_space(&space);
        match e {
            Err(ComponentError::InvalidParams { kind, reason }) => {
                assert_eq!(kind, "gen/compound");
                assert!(reason.contains('4'), "{reason}");
                assert!(reason.contains('1'), "{reason}");
            }
            other => panic!("expected InvalidParams, got {other:?}"),
        }
    }

    #[test]
    fn validate_mis_assignment_ga_bin_on_float_block_names_both() {
        // Single sub-generator, gen/ga-bin, assigned to block index 0 of a
        // Float-only space -- the brief's own example.
        let g = CompoundGenerator::from_params(&serde_json::json!({
            "blocks": [{"kind": "gen/ga-bin"}]
        })).unwrap();
        let space = SearchSpace::new(vec![Block::Float { lo: 0.0, hi: 1.0, n: 3 }]).unwrap();
        let e = g.validate_against_space(&space);
        match e {
            Err(ComponentError::InvalidParams { kind, reason }) => {
                assert_eq!(kind, "gen/compound");
                assert!(reason.contains("gen/ga-bin"), "{reason}");
                assert!(reason.contains("float"), "{reason}");
                assert!(reason.contains('0'), "{reason} (should name block index 0)");
            }
            other => panic!("expected InvalidParams, got {other:?}"),
        }
    }

    #[test]
    fn validate_mis_assignment_reports_correct_index_for_second_block() {
        // First block matches (gen/ga-real on Float), second is wrong
        // (gen/ga-bin on Int) -- error must name index 1, not 0.
        let g = CompoundGenerator::from_params(&serde_json::json!({
            "blocks": [{"kind": "gen/ga-real"}, {"kind": "gen/ga-bin"}]
        })).unwrap();
        let space = SearchSpace::new(vec![
            Block::Float { lo: 0.0, hi: 1.0, n: 2 },
            Block::Int { lo: 0, hi: 5, n: 2 },
        ]).unwrap();
        let e = g.validate_against_space(&space);
        match e {
            Err(ComponentError::InvalidParams { reason, .. }) => {
                assert!(reason.contains("gen/ga-bin"), "{reason}");
                assert!(reason.contains("int"), "{reason}");
                assert!(reason.contains('1'), "{reason} (should name block index 1)");
            }
            other => panic!("expected InvalidParams, got {other:?}"),
        }
    }

    #[test]
    fn validate_matching_assignment_is_ok() {
        let g = CompoundGenerator::from_params(&compound_spec_json()).unwrap();
        let space = MixedProblem::new(2, 2, 3, 2, 4).space;
        assert!(g.validate_against_space(&space).is_ok());
    }

    // Adapted for the M3-8 deferral closed by this change (`sezgi_core`'s
    // `Generator::validate_space` build-time hook, delegated to here by
    // `CompoundGenerator::validate_space` above): through the NORMAL
    // `AlgorithmSpec::validate`/`Engine::from_spec` path, this exact
    // mis-assignment now fails earlier, with a real `Result` naming the
    // sub-generator and block (see
    // `spec_validation_rejects_gen_compound_mis_assigned_block` below) --
    // the panic below is no longer reachable that way. This test now
    // exercises only the remaining defensive-backstop path: calling
    // `.generate()` directly on a `CompoundGenerator` built via
    // `from_params`, WITHOUT ever going through `AlgorithmSpec::validate`.
    #[test]
    #[should_panic(expected = "gen/compound")]
    fn generate_panics_on_mis_assigned_space_when_validate_is_bypassed() {
        let g = CompoundGenerator::from_params(&serde_json::json!({
            "blocks": [{"kind": "gen/ga-bin"}]
        })).unwrap();
        let p = MixedProblem::new(1, 0, 0, 0, 0); // space: Float(1), Int(0)... irrelevant, mismatched anyway
        let space = SearchSpace::new(vec![Block::Float { lo: 0.0, hi: 1.0, n: 1 }]).unwrap();
        let pop = Population { individuals: vec![Genotype { blocks: vec![BlockValues::Float(vec![0.0])] }], fitness: vec![0.0] };
        let mut eval = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space: &space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
        let _ = g.generate(&pop, &mut ctx);
    }

    // ==================================================================
    // spec-level integration (mirrors bin_ops.rs's own
    // spec_validation_accepts/rejects tests, at the AlgorithmSpec level)
    // ==================================================================

    /// Shared scaffold for the spec-level `gen/compound` tests below: a
    /// fully-registered `Registry` plus an `AlgorithmSpec` whose single
    /// stage's generator is `gen/compound` with the given sub-generator
    /// `params`. The caller supplies the target `SearchSpace` separately to
    /// `.validate`/`Engine::from_spec`.
    fn mixed_compound_spec(generator_params: serde_json::Value) -> (Registry, sezgi_core::spec::AlgorithmSpec) {
        use sezgi_core::spec::{AlgorithmSpec, StageSpec, TerminationSpec};
        let mut reg = Registry::new();
        register(&mut reg);
        crate::bin_ops::register(&mut reg);
        crate::int_ops::register(&mut reg);
        crate::cat_ops::register(&mut reg);
        crate::ga::register(&mut reg);
        crate::perm::register(&mut reg);
        crate::init::register(&mut reg);
        crate::boundary::register(&mut reg);
        crate::replace::register(&mut reg);
        let spec = AlgorithmSpec {
            name: "t".into(), pop_size: 10,
            init: ComponentSpec { kind: "init/uniform".into(), params: serde_json::json!({}) },
            boundary: ComponentSpec { kind: "boundary/clamp".into(), params: serde_json::json!({}) },
            stages: vec![StageSpec {
                generator: ComponentSpec { kind: "gen/compound".into(), params: generator_params },
                replacer: ComponentSpec { kind: "replace/mu-plus-lambda".into(), params: serde_json::json!({}) },
                adapter: None,
            }],
            termination: TerminationSpec { budget: 1000, target: None },
            restart: None,
        };
        (reg, spec)
    }

    /// M3-8 deferral (task A of the deferral-cleanup batch): a mis-assigned
    /// `gen/compound` sub-generator (`gen/ga-bin` on a Float block, the same
    /// mismatch `generate_panics_on_mis_assigned_space_when_validate_is_bypassed`
    /// above exercises through the panic backstop) now fails
    /// `AlgorithmSpec::validate` -- and, through it, `Engine::from_spec` --
    /// directly, with `SpecError::Component(ComponentError::InvalidParams)`
    /// naming both the sub-generator kind and the block, BEFORE any
    /// `generate()` call.
    #[test]
    fn spec_validation_rejects_gen_compound_mis_assigned_block() {
        use sezgi_core::engine::Engine;
        use sezgi_core::spec::SpecError;
        let space = SearchSpace::new(vec![Block::Float { lo: 0.0, hi: 1.0, n: 3 }]).unwrap();
        let (reg, spec) = mixed_compound_spec(serde_json::json!({"blocks": [{"kind": "gen/ga-bin"}]}));
        match spec.validate(&reg, &space) {
            Err(SpecError::Component(ComponentError::InvalidParams { kind, reason })) => {
                assert_eq!(kind, "gen/compound");
                assert!(reason.contains("gen/ga-bin"), "{reason}");
                assert!(reason.contains("float"), "{reason}");
            }
            other => panic!("expected SpecError::Component(InvalidParams), got {other:?}"),
        }
        // Engine::from_spec calls spec.validate internally -- same failure,
        // same error, reached through the engine's own build path.
        assert!(matches!(
            Engine::from_spec(&spec, &reg, &space),
            Err(SpecError::Component(ComponentError::InvalidParams { .. }))
        ));
    }

    /// M3-8 deferral: a wrong-arity `gen/compound` spec (4 sub-generator
    /// entries against a 1-block space, the same mismatch
    /// `validate_wrong_arity_names_both_counts` above exercises directly)
    /// now fails `AlgorithmSpec::validate` as well, not only a direct
    /// `validate_against_space` call.
    #[test]
    fn spec_validation_rejects_gen_compound_wrong_arity() {
        let space = SearchSpace::new(vec![Block::Float { lo: 0.0, hi: 1.0, n: 2 }]).unwrap();
        let (reg, spec) = mixed_compound_spec(compound_spec_json()); // 4 sub-specs, 1-block space
        match spec.validate(&reg, &space) {
            Err(sezgi_core::spec::SpecError::Component(ComponentError::InvalidParams { kind, reason })) => {
                assert_eq!(kind, "gen/compound");
                assert!(reason.contains('4'), "{reason}");
                assert!(reason.contains('1'), "{reason}");
            }
            other => panic!("expected SpecError::Component(InvalidParams), got {other:?}"),
        }
    }

    #[test]
    fn spec_validation_accepts_gen_compound_on_matching_mixed_space() {
        use sezgi_core::spec::{AlgorithmSpec, StageSpec, TerminationSpec};
        let mut reg = Registry::new();
        register(&mut reg);
        crate::bin_ops::register(&mut reg);
        crate::int_ops::register(&mut reg);
        crate::cat_ops::register(&mut reg);
        crate::ga::register(&mut reg);
        crate::perm::register(&mut reg);
        crate::init::register(&mut reg);
        crate::boundary::register(&mut reg);
        crate::replace::register(&mut reg);
        let space = MixedProblem::new(2, 2, 3, 2, 4).space;
        let spec = AlgorithmSpec {
            name: "t".into(), pop_size: 10,
            init: ComponentSpec { kind: "init/uniform".into(), params: serde_json::json!({}) },
            boundary: ComponentSpec { kind: "boundary/clamp".into(), params: serde_json::json!({}) },
            stages: vec![StageSpec {
                generator: ComponentSpec { kind: "gen/compound".into(), params: compound_spec_json() },
                replacer: ComponentSpec { kind: "replace/mu-plus-lambda".into(), params: serde_json::json!({}) },
                adapter: None,
            }],
            termination: TerminationSpec { budget: 1000, target: None },
            restart: None,
        };
        assert!(spec.validate(&reg, &space).is_ok());
    }

    // ==================================================================
    // generate() shape / determinism
    // ==================================================================

    #[test]
    fn generate_produces_pop_len_full_mixed_genotypes() {
        let p = MixedProblem::new(2, 2, 3, 2, 4);
        let space = p.space();
        let g = CompoundGenerator::from_params(&compound_spec_json()).unwrap();
        let pop = mixed_pop(7, 2, 2, 2, 4);
        let mut eval = Evaluator::new(&p, 10_000);
        let mut rng = RngStream::from_master(5, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
        let off = g.generate(&pop, &mut ctx);
        assert_eq!(off.len(), pop.len());
        for ind in &off {
            assert_eq!(ind.blocks.len(), 4);
            assert!(matches!(&ind.blocks[0], BlockValues::Float(xs) if xs.len() == 2));
            assert!(matches!(&ind.blocks[1], BlockValues::Int(xs) if xs.len() == 2));
            assert!(matches!(&ind.blocks[2], BlockValues::Cat(xs) if xs.len() == 2));
            assert!(matches!(&ind.blocks[3], BlockValues::Bin(xs) if xs.len() == 4));
            assert!(space.validate(ind).is_ok(), "gen/compound output must validate against the full mixed space");
        }
    }

    #[test]
    fn generate_deterministic_same_seed() {
        let p = MixedProblem::new(2, 2, 3, 2, 4);
        let space = p.space();
        let g = CompoundGenerator::from_params(&compound_spec_json()).unwrap();
        let pop = mixed_pop(6, 2, 2, 2, 4);
        let run = || {
            let mut eval = Evaluator::new(&p, 10_000);
            let mut rng = RngStream::from_master(11, &[]);
            let mut bb = Blackboard::new();
            let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
            g.generate(&pop, &mut ctx)
        };
        assert_eq!(run(), run());
    }

    // ==================================================================
    // THE EXACTNESS TEST (task-5 required test 3): gen/compound, through a
    // real Engine run on a Float+Int+Categorical+Binary mixed space, must
    // produce -- per block, per offspring index -- EXACTLY what each
    // sub-generator would produce standalone on the same per-block view,
    // consuming the RNG stream in the exact same sequential block order.
    // See module doc's "RNG stream design + exactness-proof strategy".
    // ==================================================================

    #[test]
    fn compound_generator_engine_run_exactness_matches_standalone_sub_generators_per_block() {
        use sezgi_core::engine::{Engine, RunConfig};

        let n_float = 2usize;
        let n_int = 2usize;
        let k_cat = 3u32;
        let n_cat = 2usize;
        let n_bin = 4usize;
        let pop_size = 8usize;
        let budget = 400u64; // enough for init (8) + a handful of stage iterations

        let problem = MixedProblem::new(n_float, n_int, k_cat, n_cat, n_bin);
        let space = problem.space().clone();

        let mut reg = Registry::new();
        crate::init::register(&mut reg);
        crate::boundary::register(&mut reg);
        crate::replace::register(&mut reg);
        register(&mut reg); // gen/compound
        crate::bin_ops::register(&mut reg);
        crate::int_ops::register(&mut reg);
        crate::cat_ops::register(&mut reg);
        crate::ga::register(&mut reg);
        crate::perm::register(&mut reg);

        let spec = sezgi_core::spec::AlgorithmSpec {
            name: "compound-exactness".into(),
            pop_size,
            init: ComponentSpec { kind: "init/uniform".into(), params: serde_json::json!({}) },
            boundary: ComponentSpec { kind: "boundary/clamp".into(), params: serde_json::json!({}) },
            stages: vec![sezgi_core::spec::StageSpec {
                generator: ComponentSpec { kind: "gen/compound".into(), params: compound_spec_json() },
                replacer: ComponentSpec { kind: "replace/mu-plus-lambda".into(), params: serde_json::json!({}) },
                adapter: None,
            }],
            termination: sezgi_core::spec::TerminationSpec { budget, target: None },
            restart: None,
        };

        let engine = Engine::from_spec(&spec, &reg, &space).unwrap();
        let cfg = RunConfig { master_seed: 4242, run_id: 0 };

        // 1) Run the real engine once: this exercises init, gen/compound
        //    (through Engine's own internal RNG-stream-per-stage wiring),
        //    boundary repair, and replacement, over more than one
        //    iteration. We only need the RunResult to prove the run
        //    completes and produces valid output; the exactness claim
        //    itself is proven directly against gen/compound's own
        //    generate(), below, which is the exact call the engine makes
        //    internally on every iteration.
        let result = engine.run(&problem, cfg, None).unwrap();
        assert!(space.validate(&result.best_x).is_ok());
        assert!(result.iterations >= 1, "the exactness proof below needs at least one real generate() call");

        // 2) Reconstruct gen/compound's OWN generator/params (identical to
        //    the spec above) plus a population and an RngStream state
        //    equal to what the FIRST stage iteration's generator draw
        //    would see (Engine's own documented per-stage RNG path,
        //    engine.rs: `RngStream::from_master(master_seed, &[run_id, 1 +
        //    2*stage_index])` for stage 0's generator stream, `1 + 0 = 1`).
        let compound = CompoundGenerator::from_params(&compound_spec_json()).unwrap();
        let gen_rng_start = sezgi_core::rng::RngStream::from_master(cfg.master_seed, &[cfg.run_id, 1]);
        let pop = mixed_pop(pop_size, n_float, n_int, n_cat, n_bin);

        let mut compound_rng = gen_rng_start.clone();
        let mut eval_a = Evaluator::new(&problem, 10_000);
        let mut bb_a = Blackboard::new();
        let compound_off = {
            let mut ctx = Ctx { space: &space, rng: &mut compound_rng, bb: &mut bb_a, eval: &mut eval_a, iteration: 0 };
            compound.generate(&pop, &mut ctx)
        };

        // 3) Replay the SAME starting RngStream state through each
        //    sub-generator, standalone, in block order -- the module doc's
        //    "RNG stream design" contract: block i's sub-generator, given
        //    the SAME single-block view and the SAME stream position, must
        //    produce EXACTLY compound's own block-i segment.
        let ga_real = crate::ga::GaRealGenerator::from_params(&serde_json::json!(
            {"tournament_k": 2, "pc": 0.9, "eta_c": 15.0, "eta_m": 20.0})).unwrap();
        let ga_int = crate::int_ops::GaIntGenerator::from_params(&serde_json::json!(
            {"tournament_k": 2, "p_c": 0.9, "eta_c": 15.0, "eta_m": 20.0})).unwrap();
        let ga_cat = crate::cat_ops::GaCatGenerator::from_params(&serde_json::json!(
            {"tournament_k": 2, "p_c": 0.9})).unwrap();
        let ga_bin = crate::bin_ops::GaBinGenerator::from_params(&serde_json::json!(
            {"tournament_k": 2, "p_c": 0.9})).unwrap();

        let mut twin_rng = gen_rng_start;
        let mut eval_b = Evaluator::new(&problem, 10_000);
        let mut bb_b = Blackboard::new();

        // Block 0: Float, gen/ga-real.
        let float_space = SearchSpace::new(vec![space.blocks()[0].clone()]).unwrap();
        let float_pop = Population {
            individuals: pop.individuals.iter().map(|g| Genotype { blocks: vec![g.blocks[0].clone()] }).collect(),
            fitness: pop.fitness.clone(),
        };
        let float_off = {
            let mut ctx = Ctx { space: &float_space, rng: &mut twin_rng, bb: &mut bb_b, eval: &mut eval_b, iteration: 0 };
            ga_real.generate(&float_pop, &mut ctx)
        };
        for j in 0..pop_size {
            assert_eq!(compound_off[j].blocks[0], float_off[j].blocks[0], "block 0 (Float) offspring {j} must match ga-real standalone");
        }

        // Block 1: Int, gen/ga-int.
        let int_space = SearchSpace::new(vec![space.blocks()[1].clone()]).unwrap();
        let int_pop = Population {
            individuals: pop.individuals.iter().map(|g| Genotype { blocks: vec![g.blocks[1].clone()] }).collect(),
            fitness: pop.fitness.clone(),
        };
        let int_off = {
            let mut ctx = Ctx { space: &int_space, rng: &mut twin_rng, bb: &mut bb_b, eval: &mut eval_b, iteration: 0 };
            ga_int.generate(&int_pop, &mut ctx)
        };
        for j in 0..pop_size {
            assert_eq!(compound_off[j].blocks[1], int_off[j].blocks[0], "block 1 (Int) offspring {j} must match ga-int standalone");
        }

        // Block 2: Categorical, gen/ga-cat.
        let cat_space = SearchSpace::new(vec![space.blocks()[2].clone()]).unwrap();
        let cat_pop = Population {
            individuals: pop.individuals.iter().map(|g| Genotype { blocks: vec![g.blocks[2].clone()] }).collect(),
            fitness: pop.fitness.clone(),
        };
        let cat_off = {
            let mut ctx = Ctx { space: &cat_space, rng: &mut twin_rng, bb: &mut bb_b, eval: &mut eval_b, iteration: 0 };
            ga_cat.generate(&cat_pop, &mut ctx)
        };
        for j in 0..pop_size {
            assert_eq!(compound_off[j].blocks[2], cat_off[j].blocks[0], "block 2 (Categorical) offspring {j} must match ga-cat standalone");
        }

        // Block 3: Binary, gen/ga-bin.
        let bin_space = SearchSpace::new(vec![space.blocks()[3].clone()]).unwrap();
        let bin_pop = Population {
            individuals: pop.individuals.iter().map(|g| Genotype { blocks: vec![g.blocks[3].clone()] }).collect(),
            fitness: pop.fitness.clone(),
        };
        let bin_off = {
            let mut ctx = Ctx { space: &bin_space, rng: &mut twin_rng, bb: &mut bb_b, eval: &mut eval_b, iteration: 0 };
            ga_bin.generate(&bin_pop, &mut ctx)
        };
        for j in 0..pop_size {
            assert_eq!(compound_off[j].blocks[3], bin_off[j].blocks[0], "block 3 (Binary) offspring {j} must match ga-bin standalone");
        }

        // 4) Full-genotype equality (composition adds nothing beyond
        //    routing -- every block already checked, this is belt-and-
        //    suspenders over the whole Vec<Genotype>).
        let stitched: Vec<Genotype> = (0..pop_size)
            .map(|j| Genotype { blocks: vec![
                float_off[j].blocks[0].clone(),
                int_off[j].blocks[0].clone(),
                cat_off[j].blocks[0].clone(),
                bin_off[j].blocks[0].clone(),
            ]})
            .collect();
        assert_eq!(compound_off, stitched);

        // 5) The RNG streams must end in the identical state -- gen/compound
        //    consumed EXACTLY the four sub-generators' own draws, in block
        //    order, nothing more.
        assert_eq!(compound_rng.next_f64(), twin_rng.next_f64(),
            "gen/compound must consume exactly the four sub-generators' own draws, in block order, no more no less");
    }

    // ==================================================================
    // 0.1.5: self-contained stateful subs + nested per-block blackboards
    // ==================================================================

    struct ZeroProblem { space: SearchSpace }
    impl Problem for ZeroProblem {
        fn space(&self) -> &SearchSpace { &self.space }
        fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<f64> { pop.iter().map(|_| 0.0).collect() }
    }

    fn float_bin_space(d: usize, nb: usize) -> SearchSpace {
        SearchSpace::new(vec![Block::Float { lo: -5.0, hi: 5.0, n: d }, Block::Binary { n: nb }]).unwrap()
    }

    fn float_bin_pop(n: usize, d: usize, nb: usize) -> Population {
        Population {
            individuals: (0..n)
                .map(|i| Genotype {
                    blocks: vec![
                        BlockValues::Float((0..d).map(|j| ((i + j) % 4) as f64 - 1.5).collect()),
                        BlockValues::Bin((0..nb).map(|j| (i + j) % 2 == 0).collect()),
                    ],
                })
                .collect(),
            fitness: (0..n).map(|i| (n - i) as f64).collect(),
        }
    }

    fn two_float_space(d0: usize, d1: usize) -> SearchSpace {
        SearchSpace::new(vec![
            Block::Float { lo: -5.0, hi: 5.0, n: d0 },
            Block::Float { lo: -5.0, hi: 5.0, n: d1 },
        ])
        .unwrap()
    }

    fn two_float_pop(n: usize, d0: usize, d1: usize) -> Population {
        Population {
            individuals: (0..n)
                .map(|i| Genotype {
                    blocks: vec![
                        BlockValues::Float((0..d0).map(|j| ((i + j) % 7) as f64 - 3.0).collect()),
                        BlockValues::Float((0..d1).map(|j| ((i * 2 + j) % 5) as f64 - 2.0).collect()),
                    ],
                })
                .collect(),
            fitness: (0..n).map(|i| (n - i) as f64).collect(),
        }
    }

    /// `gens` successive generate calls on one shared blackboard (each
    /// call's offspring becomes the next population; fitness is reused).
    fn run_gens(
        g: &CompoundGenerator,
        space: &SearchSpace,
        mut pop: Population,
        seed: u64,
        gens: usize,
    ) -> (Vec<Genotype>, Blackboard) {
        let p = ZeroProblem { space: space.clone() };
        let mut eval = Evaluator::new(&p, 100_000);
        let mut rng = RngStream::from_master(seed, &[]);
        let mut bb = Blackboard::new();
        let mut last = Vec::new();
        for it in 0..gens {
            let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: it as u64 };
            last = g.generate(&pop, &mut ctx);
            pop = Population { individuals: last.clone(), fitness: pop.fitness.clone() };
        }
        (last, bb)
    }

    fn nested_vel(bb: &Blackboard, idx: usize, key: &str) -> Vec<Vec<f64>> {
        bb.get::<Blackboard>(&format!("cmp{idx}/bb"))
            .expect("parent nested slot present")
            .get::<Vec<Vec<f64>>>(key)
            .expect("sub velocity present")
            .clone()
    }

    #[test]
    fn gsa_plus_ga_bin_accepted_and_generates_valid_shapes() {
        let g = CompoundGenerator::from_params(&serde_json::json!({
            "blocks": [{"kind": "gen/gsa"}, {"kind": "gen/ga-bin"}]
        }))
        .expect("gen/gsa is on the self-contained allow-list");
        let space = float_bin_space(3, 5);
        let (off, bb) = run_gens(&g, &space, float_bin_pop(6, 3, 5), 11, 1);
        assert_eq!(off.len(), 6);
        for o in &off {
            assert!(matches!(&o.blocks[0], BlockValues::Float(x) if x.len() == 3));
            assert!(matches!(&o.blocks[1], BlockValues::Bin(x) if x.len() == 5));
            // gsa/ba emit unclamped positions (engine repair handles bounds), so
            // only shapes are asserted here, not box validity.
        }
        assert!(bb.contains("cmp0/bb") && bb.contains("cmp1/bb"));
        assert!(!bb.contains("gsa/velocity"), "sub state must not leak into the parent blackboard");
        assert_eq!(nested_vel(&bb, 0, "gsa/velocity").len(), 6);
    }

    #[test]
    fn ba_plus_ga_bin_accepted_and_generates_valid_shapes() {
        let g = CompoundGenerator::from_params(&serde_json::json!({
            "blocks": [{"kind": "gen/ba"}, {"kind": "gen/ga-bin"}]
        }))
        .expect("gen/ba is on the self-contained allow-list");
        let space = float_bin_space(4, 6);
        let (off, bb) = run_gens(&g, &space, float_bin_pop(6, 4, 6), 12, 2);
        assert_eq!(off.len(), 6);
        for o in &off {
            assert!(matches!(&o.blocks[0], BlockValues::Float(x) if x.len() == 4));
            assert!(matches!(&o.blocks[1], BlockValues::Bin(x) if x.len() == 6));
        }
        assert!(nested_vel(&bb, 0, "ba/velocity").iter().all(|r| r.len() == 4));
    }

    #[test]
    fn pso_cma_shade_still_rejected_with_blackboard_message() {
        for kind in ["gen/pso", "gen/cma", "gen/de-shade"] {
            let r = err_reason(serde_json::json!({"blocks": [{"kind": "gen/ga-bin"}, {"kind": kind}]}));
            assert!(r.contains(kind), "{r}");
            assert!(r.contains("block 1"), "{r}");
            assert!(r.contains("blackboard state"), "{kind}: {r}");
        }
    }

    #[test]
    fn two_gsa_float_blocks_of_different_dims_do_not_collide() {
        // R-C. Under the pre-0.1.5 shared blackboard both blocks would share
        // one `gsa/velocity` slot: block 1 (dim 5) would read block 0's
        // (dim 2) matrix and mis-size/index out of range. Nested per-block
        // blackboards keep one velocity matrix per block.
        let g = CompoundGenerator::from_params(&serde_json::json!({
            "blocks": [{"kind": "gen/gsa"}, {"kind": "gen/gsa"}]
        }))
        .unwrap();
        let (d0, d1) = (2usize, 5usize);
        let space = two_float_space(d0, d1);
        let run = || run_gens(&g, &space, two_float_pop(6, d0, d1), 21, 4);
        let (off_a, bb_a) = run();
        let (off_b, _) = run();
        assert_eq!(off_a, off_b, "deterministic across identical runs");
        assert!(bb_a.contains("cmp0/bb") && bb_a.contains("cmp1/bb"));
        let v0 = nested_vel(&bb_a, 0, "gsa/velocity");
        let v1 = nested_vel(&bb_a, 1, "gsa/velocity");
        assert_eq!((v0.len(), v1.len()), (6, 6));
        assert!(v0.iter().all(|r| r.len() == d0), "block 0 velocity rows must be dim {d0}");
        assert!(v1.iter().all(|r| r.len() == d1), "block 1 velocity rows must be dim {d1}");
    }

    #[test]
    fn nested_state_persists_and_evolves_across_generate_calls() {
        let g = CompoundGenerator::from_params(&serde_json::json!({
            "blocks": [{"kind": "gen/gsa"}, {"kind": "gen/ga-bin"}]
        }))
        .unwrap();
        let space = float_bin_space(3, 4);
        let (_, bb1) = run_gens(&g, &space, float_bin_pop(6, 3, 4), 5, 1);
        let (_, bb3) = run_gens(&g, &space, float_bin_pop(6, 3, 4), 5, 3);
        let (_, bb3b) = run_gens(&g, &space, float_bin_pop(6, 3, 4), 5, 3);
        assert!(bb3.contains("cmp0/bb"));
        assert_ne!(
            nested_vel(&bb1, 0, "gsa/velocity"),
            nested_vel(&bb3, 0, "gsa/velocity"),
            "velocity must evolve between generations, not reset"
        );
        assert_eq!(nested_vel(&bb3, 0, "gsa/velocity"), nested_vel(&bb3b, 0, "gsa/velocity"));
    }
}
