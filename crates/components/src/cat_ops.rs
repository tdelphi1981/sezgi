//! Categorical operators: `gen/cat-ux`, `gen/cat-reset`, `gen/ga-cat` -- the
//! Categorical-genotype counterpart to `int_ops.rs`'s Int triple
//! (`gen/int-sbx`/`gen/int-pm`/`gen/ga-int`), `bin_ops.rs`'s Binary triple
//! (`gen/bin-2pt`/`gen/bit-flip`/`gen/ga-bin`) and `ga.rs`'s Float pair
//! (`gen/ga-real`). Like those modules, all three operate on a single
//! `Block::Categorical { k, n }` block (block tag `"categorical"`, per
//! `sezgi_core::spec::block_tag` -- the exact string checked by
//! `AlgorithmSpec::validate`'s `SupportedBlocks::Only` gate).
//!
//! ## PROVENANCE -- pymoo 0.6.2's Choice convention (Apache-2.0)
//!
//! This module structurally ports pymoo 0.6.2's Choice-variable operator
//! wiring -- its uniform-crossover-mask genotype recombination and its
//! random-reset mutation. pymoo is licensed Apache-2.0; this is attributed
//! structural porting, not a verbatim code copy: pymoo
//! (github.com/anyoptimization/pymoo, tag `0.6.2`) --
//!
//! | File | sha256 |
//! |---|---|
//! | `pymoo/core/mixed.py` | `2218e749b5f945df05ba76998933b405363eba00260461188dd4829f26fc2e47` |
//! | `pymoo/operators/crossover/ux.py` | `561bef264612954e5babe7555b88733fd4467e8687120abcba0fa9e4e46a5e12` |
//! | `pymoo/operators/mutation/rm.py` | `d392c104deb40f2302c516cb39b5c049074af89e10acc02ecd64f316d2338430` |
//! | `pymoo/core/variable.py` | `20a3b34b8b217242029328b1f31d4216ad6e4370a1d6f053f19eb44b60746ced` |
//! | `pymoo/core/crossover.py` | `93a9d4e9e65337342ee3a015f3e9269e0a068829c25de78b39dcfdfc66de8f1c` |
//! | `pymoo/core/mutation.py` | `3259175177b8814ce128e912d3b62e3ae511a020ee5652c7e4c339638b6048a7` |
//! | `pymoo/util/misc.py` | `8d32099e34554561b90a17a2d740bae0dae2f59b95b4eae2cb861af3fc779b73` |
//!
//! Every file above was fetched TWICE -- once via `curl` from
//! `raw.githubusercontent.com/anyoptimization/pymoo/0.6.2/...` (the GitHub
//! tag), once via `uv pip install --python .venv/bin/python pymoo==0.6.2` in
//! the scratchpad venv (the PyPI release) -- and `diff`'d byte-identical
//! (`exit=0`) between the two, confirming the tag and the release are the
//! same code (same technique as `int_ops.rs`'s own Task 3 re-verification).
//! `variable.py`/`mixed.py`/`crossover.py`/`mutation.py`'s sha256s here are
//! IDENTICAL to `int_ops.rs`'s own PROVENANCE table (same files, re-fetched
//! independently, same bytes -- cross-confirms both tasks read the same
//! pinned pymoo release).
//!
//! - `pymoo/core/mixed.py`, `MixedVariableMating.__init__` -- the wiring:
//!   ```python
//!   if crossover is None:
//!       crossover = {
//!           Binary: UX(), Real: SBX(),
//!           Integer: SBX(vtype=float, repair=RoundingRepair()),
//!           Choice: UX(),
//!       }
//!   if mutation is None:
//!       mutation = {
//!           Binary: BFM(), Real: PM(),
//!           Integer: PM(vtype=float, repair=RoundingRepair()),
//!           Choice: ChoiceRandomMutation(),
//!       }
//!   ```
//!   `Choice` maps to `UX()` (uniform crossover, no override args) for
//!   crossover and `ChoiceRandomMutation()` (no override args) for mutation
//!   -- UNLIKE `Integer`/`Real`, no `vtype=float`/`repair=RoundingRepair()`
//!   wrapping: Choice values are never floated or rounded, they are moved
//!   around/resampled as opaque category indices throughout.
//!
//! ### (a) Uniform-crossover mask semantics -- quoted verbatim, per-gene 0.5
//!
//! `pymoo/operators/crossover/ux.py`, `UniformCrossover._do` (`UX` is a
//! trivial subclass, no override):
//! ```python
//! def _do(self, _, X, random_state=None, **kwargs):
//!     _, n_matings, n_var = X.shape
//!     M = random_state.random((n_matings, n_var)) < 0.5
//!     _X = crossover_mask(X, M)
//!     return _X
//! ```
//! `pymoo/util/misc.py`, `crossover_mask`:
//! ```python
//! def crossover_mask(X, M):
//!     _X = np.copy(X)
//!     _X[0][M] = X[1][M]
//!     _X[1][M] = X[0][M]
//!     return _X
//! ```
//! One draw per gene, ONE fixed threshold `0.5` (NOT parameterized by
//! `prob_var` the way `Integer`'s SBX exchange rate is -- `0.5` is intrinsic
//! to "uniform" crossover itself, not a tunable gate): `M[j] = draw < 0.5`.
//! Where `M[j]` is `True`, child0's gene `j` and child1's gene `j` are
//! SWAPPED (`_X[0][M]=X[1][M]`, `_X[1][M]=X[0][M]`); where `False`, each
//! child keeps its own-index parent's gene. VERIFIED exactly via the
//! scratchpad `cat_oracle_probe.py::ux_mask_scenario` (a scripted
//! `random_state` returning `[0.1, 0.9, 0.49999, 0.5]` for a 4-gene pair
//! produces gene0/gene2 swapped, gene1/gene3 kept -- matching `< 0.5`
//! exactly, including the `0.5` boundary itself being EXCLUDED (not
//! swapped), confirmed against [`cat_cross_pair`]'s own strict `<` below).
//!
//! `UX._do` has NO pair-level gate of its own -- that lives in the base
//! `Crossover.do` wrapper (`pymoo/core/crossover.py`,
//! `Crossover.__init__(self, n_parents, n_offsprings, prob=0.9, **kwargs)`,
//! NOT overridden by `UX`), quoted verbatim:
//! ```python
//! prob = get(self.prob, size=n_matings)
//! cross = random_state.random(n_matings) < prob
//! ```
//! i.e. an outer per-PAIR gate (default `p_c = 0.9`) wraps the per-gene mask
//! draws, mirroring the SAME single-outer-gate-then-per-gene-action shape
//! `bin_ops.rs`'s `bin_cross_pair`/`int_ops.rs`'s `sbx_pair` already use
//! (KanGAL's `bincross`'s single `p_c` gate, `sbx_pair`'s own internal
//! whole-pair gate). [`cat_cross_pair`] below follows the SAME shape: one
//! `p_c` gate (default `0.9`, VERIFIED as `UX`'s own resolved default via
//! `cat_oracle_probe.py::get_prob_var_default_scenario`'s
//! `UX().prob.value == 0.9`), THEN, on pass, one fixed-`0.5` mask draw per
//! gene.
//!
//! ### (b) Random-reset sampling -- quoted verbatim, verified: does NOT
//! exclude the current value
//!
//! `pymoo/operators/mutation/rm.py`, `ChoiceRandomMutation._do`:
//! ```python
//! def _do(self, problem, X, random_state=None, **kwargs):
//!     assert problem.vars is not None
//!     X = X.astype(object)
//!     prob_var = self.get_prob_var(problem, size=len(X))
//!     for k, (_, var) in enumerate(problem.vars.items()):
//!         mut = np.where(random_state.random(len(X)) < prob_var)[0]
//!         v = var.sample(len(mut), random_state=random_state)
//!         X[mut, k] = v
//!     return X
//! ```
//! `var.sample(...)` (`pymoo/core/variable.py`, `Variable.sample` ->
//! `Choice._sample`), quoted verbatim:
//! ```python
//! def _sample(self, n, random_state=None):
//!     return random_state.choice(self.options, size=n)
//! ```
//! **VERDICT (settled from source, not assumed): the reset draws UNIFORMLY
//! over ALL `k` options, and does NOT exclude the currently-held value.**
//! `Choice._sample` takes `self.options` (the FULL option set) and the
//! COUNT `n` only -- there is no reference anywhere in this call chain (nor
//! in `Choice.__init__`/`Variable.__init__`) to the variable's current
//! value, no filtering, no "exclude self" logic, no shrunk option pool. A
//! "reset" can therefore reproduce the SAME category the gene already held,
//! with probability `1/k`, exactly like an ordinary independent resample
//! (indistinguishable from `init/uniform`'s own `Block::Categorical` sampler
//! -- `boundary.rs`'s `sample_uniform`'s `rng.next_below(k as u64)` -- by
//! design). VERIFIED two ways in the scratchpad venv:
//! - Structurally: read `Choice._sample`/`Choice.__init__`/`Variable.sample`
//!   above -- no exclusion code exists to find.
//! - Statistically, through pymoo's OWN REAL mutation code path (not just
//!   `Choice._sample` in isolation): `cat_oracle_probe_stats.py` constructs a
//!   `FakeProblem` with one `Choice(options=[0..5))` variable, a population
//!   of 20000 individuals all holding value `0`, `ChoiceRandomMutation(
//!   prob_var=1.0)` (forces every trial's gate to pass, isolating the reset
//!   kernel), calls `ChoiceRandomMutation()._do(problem, X,
//!   random_state=rng)` directly (numpy's own `default_rng(31415)`), and
//!   measures `frac(new_value == 0)` over the 20000 trials: **0.201550**,
//!   matching the `1/k = 1/5 = 0.2` prediction for "no exclusion" (an
//!   EXCLUDING reset would instead land at exactly `0.0`). [`cat_reset`]
//!   below mirrors this exactly: on gate pass, `rng.next_below(k)` -- a
//!   plain uniform draw over `0..k`, no filtering of the current value.
//!
//! ### (c) Parameter defaults -- the two-level-to-single-level gating
//! mapping (mirrors `int_ops.rs`'s own documented mapping exactly)
//!
//! `Crossover.__init__(self, n_parents, n_offsprings, prob=0.9, **kwargs)`
//! (`UX` inherits, no override) -> **`p_c = 0.9`** for [`cat_cross_pair`]/
//! `gen/cat-ux`/`gen/ga-cat`'s own pair-level gate (VERIFIED via
//! `cat_oracle_probe.py`'s `UX().prob.value == 0.9`).
//!
//! `Mutation.__init__(self, prob=1.0, prob_var=None, **kwargs)`
//! (`pymoo/core/mutation.py`) -- `ChoiceRandomMutation` does NOT override
//! `__init__` (no override args on the class at all, only `_do`), so it
//! inherits the BASE `Mutation` class's default `prob=1.0` for its OUTER,
//! individual-level gate (`Mutation.do`: `mut = random_state.random(
//! size=n_mut) <= prob`) -- VERIFIED via `cat_oracle_probe.py`'s
//! `ChoiceRandomMutation().prob.value == 1.0`. This is DIFFERENT from
//! `int_ops.rs`'s own finding for `PM` (whose own constructor sets
//! `prob=0.9`, overriding `Mutation`'s base default): **Choice's outer gate
//! defaults to `1.0`, i.e. it is a NO-OP pass-through** (always fires) --
//! unlike `Integer`'s PM, which has a meaningful `0.9` outer gate on top of
//! its per-variable `prob_var` gate. The ONLY mutation gate with a
//! non-trivial default for Choice is therefore the per-variable one:
//! `Mutation.get_prob_var` (`pymoo/core/mutation.py`), quoted verbatim:
//! ```python
//! def get_prob_var(self, problem, **kwargs):
//!     prob_var = (
//!         self.prob_var if self.prob_var is not None else min(0.5, 1 / problem.n_var)
//!     )
//!     return get(prob_var, **kwargs)
//! ```
//! `ChoiceRandomMutation` never sets `self.prob_var` (no such constructor
//! arg), so this resolves to `min(0.5, 1/n_var) = 1/n_var` for every
//! `n_var >= 2` -- VERIFIED via `cat_oracle_probe.py`'s
//! `get_prob_var_default_scenario` (`n_var in {1,2,4,10}` all match exactly)
//! -- the SAME `1/n` convention `gen/ga-real`/`gen/ga-bin`/`gen/ga-perm`/
//! `gen/ga-int` already default `p_m` to.
//!
//! **The gating-convention mapping (mirrors `int_ops.rs`'s own documented
//! mapping)**: pymoo's Choice mutation is ALSO nominally two-level
//! (individual `prob` x per-variable `prob_var`), but since Choice's own
//! individual-level default (`1.0`) is a no-op, its EFFECTIVE resolved
//! mutation rate for `n_var >= 2` is ALREADY `1.0 * 1/n_var = 1/n_var`,
//! numerically IDENTICAL to what a single-level `1/n` gate would give --
//! unlike `Integer`'s PM, where the two-level gate resolves to `0.9/n`
//! (an ACTUAL divergence from sezgi's single-level `1/n`, documented in
//! `int_ops.rs`'s own module doc). [`cat_reset`]/`gen/cat-reset`/
//! `gen/ga-cat` below use a single per-variable gate at `p_m` (default
//! `1/n`, matching `int_ops.rs`'s/`bin_ops.rs`'s established `Option<f64>`
//! pattern) with NO outer individual-level gate on top -- for Choice
//! specifically this is NOT an approximation of pymoo's own semantics, it
//! is EXACTLY pymoo's own resolved rate (pymoo's outer gate being `1.0` by
//! default makes the two conventions coincide here, unlike the Integer
//! case).
//!
//! For crossover, pymoo's own structure IS already single-level (one `p_c`
//! pair gate, `0.9` default, wrapping the fixed-`0.5` per-gene mask) -- no
//! mapping/divergence needed at all: [`cat_cross_pair`] reproduces this
//! shape directly, verbatim.
//!
//! ## Single-block scope (mirrors `bin_ops.rs`'s/`int_ops.rs`'s own house
//! convention)
//!
//! Like `bin_ops.rs`'s/`int_ops.rs`'s own generators, this module's
//! generators read exactly ONE `Block::Categorical` block, via
//! `ctx.space.blocks()[0]` ([`cat_dim`]/[`cat_k`]/[`cat_values`]) --
//! consistent with `gen/ga-real` (Float, one block), `gen/ga-perm`
//! (Permutation, one block), `gen/ga-bin` (Binary, one block) and
//! `gen/ga-int` (Int, one block), NOT with `nsga2_run`'s multi-block MO
//! path. A general multi-block composition, if ever needed, is out of scope
//! here (M3-8 Task 5's `gen/compound` may revisit it).
//!
//! ## Standalone `gen/cat-ux`/`gen/cat-reset` (implementer judgment: shipped)
//!
//! The brief's PINNED interface names only `gen/ga-cat` explicitly, leaving
//! standalone crossover-only/mutation-only registry IDs to implementer
//! judgment ("ship only if they factor naturally"). [`cat_cross_pair`]/
//! [`cat_reset`] factor exactly as naturally here as `bin_cross_pair`/
//! `bin_flip_mutation` did for `bin_ops.rs` and `sbx_round_pair`/
//! `pm_round_mutation` did for `int_ops.rs` (both of which DID ship
//! standalones, per Task 2/Task 3) -- there is no structural reason
//! Categorical would be different, so this module ships `gen/cat-ux`
//! (crossover-only) and `gen/cat-reset` (mutation-only) alongside the fused
//! `gen/ga-cat`, for the same composability reasons (a future
//! `gen/compound`/two-phase algorithm could want just one half).
//!
//! ## `gen/cat-ux` / `gen/ga-cat`'s pair loop (mirrors `gen/bin-2pt`/
//! `gen/int-sbx` exactly)
//!
//! Tournament selection ([`crate::select::tournament`], shared with every
//! other representation in this crate): draw a candidate, then
//! `tournament_k - 1` more, keeping the best (lowest fitness) seen. Per
//! pair: tournament for parent 1 (`tournament_k` draws), THEN tournament for
//! parent 2 (`tournament_k` more draws), THEN [`cat_cross_pair`] directly --
//! [`cat_cross_pair`]'s own internal whole-pair gate (`p_c`) is the ONLY
//! gate; callers here do NOT draw a second, redundant gate (same structural
//! reasoning as `bin_ops.rs`'s/`int_ops.rs`'s own fused generators).
//! `gen/ga-cat` additionally mutates both children via [`cat_reset`] (`c1`
//! then `c2`) after crossover, mirroring `gen/ga-bin`'s/`gen/ga-int`'s own
//! crossover-then-mutate-both-children shape.
//!
//! ## `gen/cat-reset`'s per-individual loop
//!
//! Population order (mirrors `gen/bit-flip`'s/`gen/int-pm`'s own
//! per-individual loop shape), but with NO outer per-individual gate:
//! [`cat_reset`] already gates PER VARIABLE internally, so a redundant
//! per-individual gate on top of it would double-gate -- and, per the
//! gating-convention mapping above, would ALSO diverge from pymoo's own
//! resolved Choice semantics (whose outer gate is a `1.0` no-op), unlike
//! `int_ops.rs`'s PM case.
//!
//! ## Defaults
//!
//! **`p_c` (`gen/cat-ux` and `gen/ga-cat`): `0.9`.** **`p_m` (`gen/cat-reset`
//! and `gen/ga-cat`): `1/n`**, `n` = the space's single `Block::Categorical`'s
//! dimension. Both VERIFIED against pymoo 0.6.2's own Choice wiring -- see
//! "PROVENANCE" above for the quoted source lines and oracle-probe output
//! each is drawn from.
//!
//! **`tournament_k`: `2`**, mirroring every other representation's own
//! default in this crate (tournament selection itself now lives in
//! `crate::select::tournament`, shared with every other representation --
//! see that module's doc).

use sezgi_core::component::*;
use sezgi_core::problem::Population;
use sezgi_core::rng::RngStream;
use sezgi_core::space::{Block, BlockValues, Genotype, SearchSpace};

/// pymoo 0.6.2's `UX`/`UniformCrossover._do` + `crossover_mask` (see this
/// module's PROVENANCE doc, "(a)"), wrapped in the SAME single-outer-gate
/// shape `bin_cross_pair`/`sbx_pair` already use: ONE pair-level gate draw
/// (`rand > p_c` fails, verbatim-copy parents, no further draws); on pass,
/// ONE mask draw per gene (`draw < 0.5` -> swap this gene between the two
/// children; else each child keeps its own-index parent's gene). `p1`/`p2`
/// must be the same length.
pub fn cat_cross_pair(p1: &[u32], p2: &[u32], p_c: f64, rng: &mut RngStream) -> (Vec<u32>, Vec<u32>) {
    debug_assert_eq!(p1.len(), p2.len(), "cat_cross_pair: parent vectors must have equal length");

    // Pair-level gate: ONE draw (mirrors bin_cross_pair's/sbx_pair's own
    // single whole-pair gate).
    if rng.next_f64() > p_c {
        return (p1.to_vec(), p2.to_vec());
    }

    let n = p1.len();
    let mut c1 = Vec::with_capacity(n);
    let mut c2 = Vec::with_capacity(n);
    for j in 0..n {
        // Per-gene mask draw, fixed 0.5 threshold (pymoo's UX._do:
        // `M = random_state.random((n_matings, n_var)) < 0.5`) -- strict
        // `<`, verified boundary-exact via the scratchpad oracle probe (a
        // draw of exactly 0.5 does NOT swap).
        if rng.next_f64() < 0.5 {
            c1.push(p2[j]);
            c2.push(p1[j]);
        } else {
            c1.push(p1[j]);
            c2.push(p2[j]);
        }
    }
    (c1, c2)
}

/// pymoo 0.6.2's `ChoiceRandomMutation._do` + `Choice._sample` (see this
/// module's PROVENANCE doc, "(b)"): per gene, in order, ONE gate draw
/// (`draw < p_m`); on pass, ONE replacement draw, UNIFORM over `0..k` (NOT
/// excluding the gene's current value -- see this module's PROVENANCE doc,
/// "(b)", for the source-verified verdict and the statistical oracle
/// confirmation); on failure, leave the gene unchanged. `k` is the block's
/// category count (`Block::Categorical { k, .. }`).
pub fn cat_reset(x: &mut [u32], k: u32, p_m: f64, rng: &mut RngStream) {
    for gene in x.iter_mut() {
        if rng.next_f64() < p_m {
            *gene = rng.next_below(k as u64) as u32;
        }
    }
}

fn cat_dim(space: &SearchSpace) -> usize {
    let Block::Categorical { n, .. } = space.blocks()[0] else {
        unreachable!("categorical operators require a single Categorical block")
    };
    n
}

fn cat_k(space: &SearchSpace) -> u32 {
    let Block::Categorical { k, .. } = space.blocks()[0] else {
        unreachable!("categorical operators require a single Categorical block")
    };
    k
}

fn cat_values(g: &Genotype) -> &Vec<u32> {
    match &g.blocks[0] {
        BlockValues::Cat(xs) => xs,
        _ => unreachable!("categorical operators require a Categorical block value"),
    }
}

pub struct CatUxGenerator {
    pub tournament_k: usize,
    pub p_c: f64,
}

impl CatUxGenerator {
    pub fn from_params(p: &serde_json::Value) -> Result<Self, ComponentError> {
        let err = |reason: String| ComponentError::InvalidParams { kind: "gen/cat-ux".into(), reason };
        let g = Self {
            tournament_k: p.get("tournament_k").and_then(|v| v.as_u64()).unwrap_or(2) as usize,
            p_c: crate::params::resolve(p, "p_c", "pc").and_then(|v| v.as_f64()).unwrap_or(0.9),
        };
        if g.tournament_k == 0 { return Err(err("tournament_k must be >= 1".into())); }
        if !(0.0..=1.0).contains(&g.p_c) { return Err(err(format!("p_c outside [0,1]: {}", g.p_c))); }
        Ok(g)
    }
}

impl Generator for CatUxGenerator {
    /// Pair loop mirroring `gen/bin-2pt`'s/`gen/int-sbx`'s own EXACTLY,
    /// minus a second external gate ([`cat_cross_pair`] already gates
    /// internally).
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        assert!(pop.len() >= 2, "gen/cat-ux requires a population of at least 2 (pop_size={})", pop.len());
        let mut out = Vec::with_capacity(pop.len());
        while out.len() < pop.len() {
            let i1 = crate::select::tournament(pop, self.tournament_k, ctx.rng);
            let i2 = crate::select::tournament(pop, self.tournament_k, ctx.rng);
            let p1 = cat_values(&pop.individuals[i1]).clone();
            let p2 = cat_values(&pop.individuals[i2]).clone();
            let (c1, c2) = cat_cross_pair(&p1, &p2, self.p_c, ctx.rng);
            out.push(Genotype { blocks: vec![BlockValues::Cat(c1)] });
            if out.len() < pop.len() {
                out.push(Genotype { blocks: vec![BlockValues::Cat(c2)] });
            }
        }
        out
    }
    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/cat-ux", SupportedBlocks::Only(vec!["categorical"])).with_min_pop(2)
    }
}

pub struct CatResetGenerator {
    pub p_m: Option<f64>,
}

impl CatResetGenerator {
    pub fn from_params(p: &serde_json::Value) -> Result<Self, ComponentError> {
        let err = |reason: String| ComponentError::InvalidParams { kind: "gen/cat-reset".into(), reason };
        let p_m = p.get("p_m").and_then(|v| v.as_f64());
        if let Some(pm) = p_m {
            if !(0.0..=1.0).contains(&pm) { return Err(err(format!("p_m outside [0,1]: {pm}"))); }
        }
        Ok(Self { p_m })
    }
}

impl Generator for CatResetGenerator {
    /// Per-individual loop, population order (mirrors `gen/bit-flip`'s/
    /// `gen/int-pm`'s own shape) -- NO outer per-individual gate: see this
    /// module's doc, "gating-convention mapping" section.
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        let dim = cat_dim(ctx.space);
        let k = cat_k(ctx.space);
        let pm = self.p_m.unwrap_or(1.0 / dim as f64);
        (0..pop.len())
            .map(|i| {
                let mut xs = cat_values(&pop.individuals[i]).clone();
                cat_reset(&mut xs, k, pm, ctx.rng);
                Genotype { blocks: vec![BlockValues::Cat(xs)] }
            })
            .collect()
    }
    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/cat-reset", SupportedBlocks::Only(vec!["categorical"]))
    }
}

/// `gen/ga-cat` (M3-8 Task 4): the FUSED categorical-GA generator -- uniform
/// crossover AND random-reset mutation, both directly on category indices
/// (no float/round detour, unlike `Integer`), inside ONE `Generator`, one
/// evaluate-and-replace per generation, mirroring [`crate::bin_ops::GaBinGenerator`]'s
/// / [`crate::int_ops::GaIntGenerator`]'s own single-stage shape for the
/// Categorical representation. Reuses [`cat_cross_pair`]/[`cat_reset`]
/// directly -- see this module's doc for the full draw-order contract and
/// defaults provenance.
pub struct GaCatGenerator {
    pub tournament_k: usize,
    pub p_c: f64,
    pub p_m: Option<f64>,
}

impl GaCatGenerator {
    pub fn from_params(p: &serde_json::Value) -> Result<Self, ComponentError> {
        let err = |reason: String| ComponentError::InvalidParams { kind: "gen/ga-cat".into(), reason };
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

impl Generator for GaCatGenerator {
    /// See this module's doc for the full draw-order contract: per pair,
    /// tournament x2 (`tournament_k` draws each), THEN [`cat_cross_pair`]
    /// (its own internal gate + conditional per-gene mask draws) producing
    /// `(c1, c2)`, THEN [`cat_reset`] on `c1` then `c2` (each internally
    /// gated per gene).
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        assert!(pop.len() >= 2, "gen/ga-cat requires a population of at least 2 (pop_size={})", pop.len());
        let dim = cat_dim(ctx.space);
        let k = cat_k(ctx.space);
        let pm = self.p_m.unwrap_or(1.0 / dim as f64);
        let mut out = Vec::with_capacity(pop.len());
        while out.len() < pop.len() {
            let i1 = crate::select::tournament(pop, self.tournament_k, ctx.rng);
            let i2 = crate::select::tournament(pop, self.tournament_k, ctx.rng);
            let p1 = cat_values(&pop.individuals[i1]).clone();
            let p2 = cat_values(&pop.individuals[i2]).clone();
            let (mut c1, mut c2) = cat_cross_pair(&p1, &p2, self.p_c, ctx.rng);
            cat_reset(&mut c1, k, pm, ctx.rng);
            cat_reset(&mut c2, k, pm, ctx.rng);
            out.push(Genotype { blocks: vec![BlockValues::Cat(c1)] });
            if out.len() < pop.len() {
                out.push(Genotype { blocks: vec![BlockValues::Cat(c2)] });
            }
        }
        out
    }
    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/ga-cat", SupportedBlocks::Only(vec!["categorical"])).with_min_pop(2)
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_generator("gen/cat-ux", |p| Ok(Box::new(CatUxGenerator::from_params(p)?)));
    reg.register_generator("gen/cat-reset", |p| Ok(Box::new(CatResetGenerator::from_params(p)?)));
    reg.register_generator("gen/ga-cat", |p| Ok(Box::new(GaCatGenerator::from_params(p)?)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::problem::{Evaluator, Problem};
    use sezgi_core::space::SearchSpace;
    use sezgi_core::state::Blackboard;

    /// A minimal `Problem` with a single `Categorical { k, n }` block, used
    /// only to construct a valid `Evaluator`/`Ctx` for these tests -- mirrors
    /// `int_ops.rs`'s own `IntQuadratic` scaffold. Fitness = Hamming distance
    /// to `target` (a CatMatch-shaped inline toy, per the task brief -- not a
    /// public problem type, Task 5's scope): count of genes NOT matching the
    /// target category vector, minimized at 0.
    struct CatMatch { space: SearchSpace, target: Vec<u32> }
    impl CatMatch {
        fn new(k: u32, target: Vec<u32>) -> Self {
            let n = target.len();
            Self { space: SearchSpace::new(vec![Block::Categorical { k, n }]).unwrap(), target }
        }
    }
    impl Problem for CatMatch {
        fn space(&self) -> &SearchSpace { &self.space }
        fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<f64> {
            pop.iter().map(|g| {
                cat_values(g).iter().zip(&self.target)
                    .filter(|(&x, &t)| x != t)
                    .count() as f64
            }).collect()
        }
    }

    fn g(xs: Vec<u32>) -> Genotype { Genotype { blocks: vec![BlockValues::Cat(xs)] } }

    fn pop_n(n: usize, dim: usize, k: u32) -> Population {
        Population {
            individuals: (0..n).map(|i| {
                g((0..dim).map(|j| ((j + i) as u32) % k).collect())
            }).collect(),
            fitness: (0..n).map(|i| (n - i) as f64).collect(),
        }
    }

    // ==================================================================
    // Hand-traced seeded-RNG exactness (required test 1)
    // ==================================================================

    // Seed 0's first four draws are:
    //   d0 = 0.32457526803140668292  (pair gate: <= p_c=1.0 -> always passes; here compared as > p_c so 0.3246 <= 1.0 passes)
    //   d1 = 0.38223929651167343202  (gene0 mask: < 0.5 -> swap)
    //   d2 = 0.35961720764735527478  (gene1 mask: < 0.5 -> swap)
    //   d3 = 0.01145550893465363540  (gene2 mask: < 0.5 -> swap)
    // (same seed-0 reconnaissance values `bin_ops.rs`'s/`int_ops.rs`'s own
    // hand-trace tests anchor -- independently re-confirmed here). p1=[0,1,2],
    // p2=[9,8,7], p_c=1.0 (gate always passes, isolating the mask kernel --
    // matches `cat_oracle_probe.py`'s own gate-bypass technique). All three
    // draws are < 0.5 -> all three genes swap: c1=[9,8,7] (p2, fully
    // swapped), c2=[0,1,2] (p1, fully swapped). 4 draws total (1 gate + 3
    // mask, n=3).
    #[test]
    fn cat_cross_pair_hand_trace_matches_pymoo_oracle_mask_semantics() {
        let rng_before = RngStream::from_master(0, &[]);
        let mut rng = rng_before.clone();
        let (c1, c2) = cat_cross_pair(&[0, 1, 2], &[9, 8, 7], 1.0, &mut rng);
        assert_eq!(c1, vec![9, 8, 7], "all three seed-0 mask draws are < 0.5 -> full swap");
        assert_eq!(c2, vec![0, 1, 2]);

        let mut twin = rng_before;
        for _ in 0..4 { let _ = twin.next_f64(); }
        assert_eq!(rng.next_f64(), twin.next_f64(),
            "cat_cross_pair (gate pass, n=3) must consume exactly 4 draws (1 gate + 3 mask)");
    }

    // Seed 42: the gate draw (`next_f64()`) and the replacement draw
    // (`next_below(k)`) are DIFFERENT RNG primitives (`next_below` is
    // rejection-sampled directly off the raw `next_u64()` stream, not
    // derived from `next_f64()`'s `>>11` float conversion -- see
    // `rng.rs`'s own `next_below`), so the replacement value was READ from
    // the actual implementation (`RngStream::from_master(42,
    // &[]).next_f64()` then `.next_below(5)`), not hand-derived by algebra:
    //   d0 = next_f64()    = 0.81430514512290985696  (gene0 mutation gate: p_m=1.0 -> always passes)
    //   d1 = next_below(5) = 3                        (replacement category)
    // x=[3], k=5, p_m=1.0 -> gene0 always resets; the replacement happens to
    // reproduce the SAME value the gene already held (3 -> 3) -- a live,
    // in-repo demonstration of this module's PROVENANCE-verified "no
    // exclusion of the current value" semantics (see the module doc's "(b)"
    // section and `cat_reset_can_reproduce_the_current_value` below), not a
    // coincidence worth hiding.
    #[test]
    fn cat_reset_hand_trace_matches_pymoo_oracle_reset_semantics() {
        let rng_before = RngStream::from_master(42, &[]);
        let mut rng = rng_before.clone();
        let mut x = vec![3u32];
        cat_reset(&mut x, 5, 1.0, &mut rng);
        assert_eq!(x, vec![3], "the reset happens to redraw the SAME category (3) -- expected under no-exclusion semantics");

        let mut twin = rng_before;
        let _ = twin.next_f64();
        let _ = twin.next_below(5);
        assert_eq!(rng.next_f64(), twin.next_f64(),
            "cat_reset (gate pass, 1 gene) must consume exactly 2 draws (1 next_f64 gate + 1 next_below replacement)");
    }

    #[test]
    fn cat_cross_pair_gate_fail_is_verbatim_parents() {
        let mut rng = RngStream::from_master(55, &[]);
        let (c1, c2) = cat_cross_pair(&[0, 1, 2], &[4, 3, 2], 0.0, &mut rng);
        assert_eq!(c1, vec![0, 1, 2], "p_c=0 must leave the block unchanged (gate always fails)");
        assert_eq!(c2, vec![4, 3, 2], "p_c=0 must leave the block unchanged (gate always fails)");
    }

    #[test]
    fn cat_reset_zero_p_m_is_identity() {
        let x0 = vec![0u32, 1, 2, 3, 4];
        let mut x = x0.clone();
        let mut rng = RngStream::from_master(2024, &[]);
        cat_reset(&mut x, 5, 0.0, &mut rng);
        assert_eq!(x, x0, "p_m=0 must leave every gene unchanged");
    }

    #[test]
    fn cat_reset_can_reproduce_the_current_value() {
        // Per this module's PROVENANCE doc "(b)": pymoo's Choice reset does
        // NOT exclude the current value -- a "no-op" reset (new == old) is
        // possible with probability ~1/k. Regression: find a seed/gene value
        // where the k=1 degenerate case (only one possible category) proves
        // this structurally -- with k=1, next_below(1) is ALWAYS 0, so a
        // reset can ONLY reproduce the current value. This is a boundary
        // case that would be nonsensical under an "exclude current" design
        // (which would need to loop forever with k=1) but is well-defined
        // and trivially correct under the verified "no exclusion" semantics.
        let mut rng = RngStream::from_master(7, &[]);
        let mut x = vec![0u32; 3];
        cat_reset(&mut x, 1, 1.0, &mut rng);
        assert_eq!(x, vec![0, 0, 0], "k=1: reset must reproduce the only category (no-exclusion semantics)");
    }

    #[test]
    fn cat_cross_pair_children_are_recombinations_of_the_parent_genes() {
        let mut rng = RngStream::from_master(2024, &[]);
        for _ in 0..200 {
            let n = 1 + (rng.next_f64() * 12.0) as usize;
            let p1: Vec<u32> = (0..n).map(|_| rng.next_below(6) as u32).collect();
            let p2: Vec<u32> = (0..n).map(|_| rng.next_below(6) as u32).collect();
            let (c1, c2) = cat_cross_pair(&p1, &p2, 0.7, &mut rng);
            for j in 0..n {
                assert!(
                    (c1[j] == p1[j] || c1[j] == p2[j]) && (c2[j] == p1[j] || c2[j] == p2[j]),
                    "child gene at {j} must come from one of the two parents"
                );
            }
        }
    }

    // ==================================================================
    // Statistical oracle anchor (required test 2, statistical half):
    // cat_reset's/cat_cross_pair's empirical output DISTRIBUTIONs against
    // pymoo 0.6.2's own OWN empirical distribution -- both run with their
    // OWN independent RNG (numpy's `default_rng` for pymoo, this crate's
    // `RngStream` for Rust -- no shared draws), N=20000 trials (scratchpad
    // `cat_oracle_probe_stats.py`, checked into the task-4 report). Anchor
    // numbers (pymoo, N=20000, k=5, all individuals start at value 0,
    // prob_var forced to 1.0 -- isolates the reset kernel):
    //   frac(new == old == 0) = 0.201550  (matches 1/k=0.2, confirming NO
    //   exclusion of the current value)
    // ==================================================================

    #[test]
    fn cat_reset_distribution_matches_pymoo_no_exclusion_verdict_statistically() {
        let k = 5u32;
        let trials = 20_000;
        let mut rng = RngStream::from_master(4242, &[]);
        let mut same_count = 0usize;
        let mut hist = [0usize; 5];
        for _ in 0..trials {
            let mut x = vec![0u32];
            cat_reset(&mut x, k, 1.0, &mut rng);
            if x[0] == 0 { same_count += 1; }
            hist[x[0] as usize] += 1;
        }
        let frac_same = same_count as f64 / trials as f64;
        // pymoo anchor: frac(new==old)=0.201550, i.e. NOT 0.0 (which an
        // excluding-reset would produce) and close to 1/k=0.2.
        assert!((0.15..0.25).contains(&frac_same),
            "cat_reset frac(new==old) {frac_same} should be near pymoo's oracle anchor 0.2015 (no exclusion of current value)");
        for &count in &hist {
            let frac = count as f64 / trials as f64;
            assert!((0.15..0.25).contains(&frac), "cat_reset category distribution should be roughly uniform over k={k}, got {frac}");
        }
    }

    // ==================================================================
    // Validity property test (required test 4): offspring category values
    // always < k, across many seeds, for gen/cat-ux, gen/cat-reset and
    // gen/ga-cat.
    // ==================================================================

    #[test]
    fn validity_property_offspring_categories_always_below_k_across_seeds() {
        let k = 6u32;
        let n = 5usize;
        let p = CatMatch::new(k, vec![0; n]);
        let space = p.space();
        let ux = CatUxGenerator::from_params(&serde_json::json!({})).unwrap();
        let reset = CatResetGenerator::from_params(&serde_json::json!({"p_m": 0.9})).unwrap();
        let ga = GaCatGenerator::from_params(&serde_json::json!({"p_m": 0.9})).unwrap();
        let mut eval = Evaluator::new(&p, 100_000_000);
        let mut bb = Blackboard::new();
        for seed in 0..300u64 {
            let pop = pop_n(9, n, k);
            let mut rng = RngStream::from_master(seed, &[0]);
            {
                let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
                for ind in ux.generate(&pop, &mut ctx) {
                    for &x in cat_values(&ind) { assert!(x < k, "gen/cat-ux value >= k: {x}"); }
                }
            }
            {
                let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
                for ind in reset.generate(&pop, &mut ctx) {
                    for &x in cat_values(&ind) { assert!(x < k, "gen/cat-reset value >= k: {x}"); }
                }
            }
            {
                let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
                for ind in ga.generate(&pop, &mut ctx) {
                    for &x in cat_values(&ind) { assert!(x < k, "gen/ga-cat value >= k: {x}"); }
                }
            }
        }
    }

    // ==================================================================
    // gen/cat-ux (CatUxGenerator)
    // ==================================================================

    #[test]
    fn cat_ux_generator_validity_property_seeded_batch() {
        let n = 5usize;
        let p = CatMatch::new(4, vec![0; n]);
        let space = p.space();
        let gen = CatUxGenerator::from_params(&serde_json::json!({})).unwrap();
        let mut eval = Evaluator::new(&p, 1_000_000);
        let mut rng = RngStream::from_master(3, &[0]);
        let mut bb = Blackboard::new();
        for _ in 0..100 {
            let pop = pop_n(9, n, 4);
            let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
            let off = gen.generate(&pop, &mut ctx);
            assert_eq!(off.len(), pop.len());
            for ind in &off { assert_eq!(cat_values(ind).len(), n); }
        }
    }

    #[test]
    fn cat_ux_generator_deterministic_same_seed() {
        let n = 4usize;
        let p = CatMatch::new(5, vec![0; n]);
        let space = p.space();
        let gen = CatUxGenerator::from_params(&serde_json::json!({})).unwrap();
        let pop = pop_n(6, n, 5);
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
    fn cat_ux_generator_min_pop_below_2_panics() {
        let n = 3usize;
        let p = CatMatch::new(3, vec![0; n]);
        let space = p.space();
        let gen = CatUxGenerator::from_params(&serde_json::json!({})).unwrap();
        let pop = pop_n(1, n, 3);
        let mut eval = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gen.generate(&pop, &mut ctx)
        }));
        assert!(result.is_err(), "gen/cat-ux must reject pop_size < 2 at runtime as a backstop");
    }

    #[test]
    fn cat_ux_generator_meta_min_pop_2_and_block_restricted() {
        let m = CatUxGenerator::from_params(&serde_json::json!({})).unwrap().meta();
        assert_eq!(m.min_pop, 2);
        assert!(m.supports_block("categorical"));
        assert!(!m.supports_block("float"));
        assert_eq!(m.kind, "gen/cat-ux");
    }

    #[test]
    fn cat_ux_params_validate() {
        assert!(CatUxGenerator::from_params(&serde_json::json!({"p_c": 1.5})).is_err());
        assert!(CatUxGenerator::from_params(&serde_json::json!({"tournament_k": 0})).is_err());
        let g = CatUxGenerator::from_params(&serde_json::json!({})).unwrap();
        assert_eq!(g.tournament_k, 2);
        assert_eq!(g.p_c, 0.9);
    }

    /// `gen/cat-ux` is an M3-8 typed family (historically parsed only
    /// `p_c`) -- it now ALSO accepts legacy `pc`, resolving identically to
    /// the equivalent `p_c` spec (M3-8 deferral cleanup, `params.rs`).
    #[test]
    fn cat_ux_pc_alias_resolves_identically_to_canonical_p_c() {
        let legacy = CatUxGenerator::from_params(&serde_json::json!({"pc": 0.42})).unwrap();
        let canonical = CatUxGenerator::from_params(&serde_json::json!({"p_c": 0.42})).unwrap();
        assert_eq!(legacy.p_c, canonical.p_c);
        assert_eq!(legacy.p_c, 0.42);
    }

    /// When both spellings are present on the same block, canonical `p_c`
    /// wins outright -- `pc`'s value is never used.
    #[test]
    fn cat_ux_p_c_wins_over_pc_when_both_present() {
        let g = CatUxGenerator::from_params(&serde_json::json!({"p_c": 0.42, "pc": 0.99})).unwrap();
        assert_eq!(g.p_c, 0.42);
    }

    // ==================================================================
    // gen/cat-reset (CatResetGenerator)
    // ==================================================================

    #[test]
    fn cat_reset_generator_validity_property_seeded_batch() {
        let n = 5usize;
        let p = CatMatch::new(4, vec![0; n]);
        let space = p.space();
        let gen = CatResetGenerator::from_params(&serde_json::json!({"p_m": 0.9})).unwrap();
        let mut eval = Evaluator::new(&p, 1_000_000);
        let mut rng = RngStream::from_master(6, &[0]);
        let mut bb = Blackboard::new();
        for _ in 0..200 {
            let pop = pop_n(5, n, 4);
            let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
            let off = gen.generate(&pop, &mut ctx);
            assert_eq!(off.len(), pop.len());
            for ind in &off { assert_eq!(cat_values(ind).len(), n); }
        }
    }

    #[test]
    fn cat_reset_generator_deterministic_same_seed() {
        let n = 4usize;
        let p = CatMatch::new(5, vec![0; n]);
        let space = p.space();
        let gen = CatResetGenerator::from_params(&serde_json::json!({"p_m": 0.5})).unwrap();
        let pop = pop_n(5, n, 5);
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
    fn cat_reset_generator_zero_p_m_leaves_population_unchanged() {
        let n = 4usize;
        let p = CatMatch::new(5, vec![0; n]);
        let space = p.space();
        let gen = CatResetGenerator::from_params(&serde_json::json!({"p_m": 0.0})).unwrap();
        let pop = pop_n(4, n, 5);
        let mut eval = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(4, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
        let off = gen.generate(&pop, &mut ctx);
        for (o, ind) in off.iter().zip(&pop.individuals) {
            assert_eq!(cat_values(o), cat_values(ind));
        }
    }

    #[test]
    fn cat_reset_generator_meta_default_min_pop_and_block_restricted() {
        let m = CatResetGenerator::from_params(&serde_json::json!({})).unwrap().meta();
        assert_eq!(m.min_pop, 1);
        assert!(m.supports_block("categorical"));
        assert!(!m.supports_block("int"));
        assert_eq!(m.kind, "gen/cat-reset");
    }

    #[test]
    fn cat_reset_default_p_m_is_one_over_n() {
        let n = 6usize;
        let k = 5u32;
        let p = CatMatch::new(k, vec![0; n]);
        let space = p.space();
        let gen = CatResetGenerator::from_params(&serde_json::json!({})).unwrap();
        assert_eq!(gen.p_m, None); // resolved lazily against n inside generate()
        let pop = Population { individuals: vec![g(vec![2; n])], fitness: vec![0.0] };
        let mut eval = Evaluator::new(&p, 100);
        // Find a seed whose first next_f64() draw is >= 1/6 (gene0's gate
        // fails under the 1/6 default) so gene0 must stay unchanged.
        let seed = (0..2000u64).find(|&s| {
            let mut r = RngStream::from_master(s, &[0]);
            r.next_f64() >= 1.0 / 6.0
        }).unwrap();
        let mut rng = RngStream::from_master(seed, &[0]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
        let off = gen.generate(&pop, &mut ctx);
        assert_eq!(cat_values(&off[0])[0], 2, "gene0's gate must fail (unchanged) when the draw is >= 1/n");
    }

    #[test]
    fn cat_reset_params_validate() {
        assert!(CatResetGenerator::from_params(&serde_json::json!({"p_m": 1.5})).is_err());
        assert!(CatResetGenerator::from_params(&serde_json::json!({"p_m": -0.1})).is_err());
        assert!(CatResetGenerator::from_params(&serde_json::json!({"p_m": 0.3})).is_ok());
    }

    // ==================================================================
    // gen/ga-cat (GaCatGenerator)
    // ==================================================================

    #[test]
    fn ga_cat_generator_meta_min_pop_2_and_block_restricted() {
        let m = GaCatGenerator::from_params(&serde_json::json!({})).unwrap().meta();
        assert_eq!(m.min_pop, 2);
        assert!(m.supports_block("categorical"));
        assert!(!m.supports_block("binary"));
        assert_eq!(m.kind, "gen/ga-cat");
    }

    #[test]
    fn ga_cat_params_validate() {
        assert!(GaCatGenerator::from_params(&serde_json::json!({"p_c": 1.5})).is_err());
        assert!(GaCatGenerator::from_params(&serde_json::json!({"tournament_k": 0})).is_err());
        assert!(GaCatGenerator::from_params(&serde_json::json!({"p_m": -0.1})).is_err());
        assert!(GaCatGenerator::from_params(&serde_json::json!({"p_m": 1.1})).is_err());
        let g = GaCatGenerator::from_params(&serde_json::json!({})).unwrap();
        assert_eq!(g.tournament_k, 2);
        assert_eq!(g.p_c, 0.9);
        assert_eq!(g.p_m, None); // resolved lazily against n inside generate()
    }

    /// `gen/ga-cat` is an M3-8 typed family (historically parsed only
    /// `p_c`) -- it now ALSO accepts legacy `pc`, resolving identically to
    /// the equivalent `p_c` spec (M3-8 deferral cleanup, `params.rs`).
    #[test]
    fn ga_cat_pc_alias_resolves_identically_to_canonical_p_c() {
        let legacy = GaCatGenerator::from_params(&serde_json::json!({"pc": 0.42})).unwrap();
        let canonical = GaCatGenerator::from_params(&serde_json::json!({"p_c": 0.42})).unwrap();
        assert_eq!(legacy.p_c, canonical.p_c);
        assert_eq!(legacy.p_c, 0.42);
    }

    /// When both spellings are present on the same block, canonical `p_c`
    /// wins outright -- `pc`'s value is never used.
    #[test]
    fn ga_cat_p_c_wins_over_pc_when_both_present() {
        let g = GaCatGenerator::from_params(&serde_json::json!({"p_c": 0.42, "pc": 0.99})).unwrap();
        assert_eq!(g.p_c, 0.42);
    }

    #[test]
    fn ga_cat_generator_validity_property_seeded_batch() {
        let n = 5usize;
        let p = CatMatch::new(4, vec![0; n]);
        let space = p.space();
        let gen = GaCatGenerator::from_params(&serde_json::json!({})).unwrap();
        let mut eval = Evaluator::new(&p, 1_000_000);
        let mut rng = RngStream::from_master(31, &[0]);
        let mut bb = Blackboard::new();
        for _ in 0..100 {
            let pop = pop_n(9, n, 4);
            let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
            let off = gen.generate(&pop, &mut ctx);
            assert_eq!(off.len(), pop.len());
            for ind in &off { assert_eq!(cat_values(ind).len(), n); }
        }
    }

    #[test]
    fn ga_cat_generator_deterministic_same_seed() {
        let n = 4usize;
        let p = CatMatch::new(5, vec![0; n]);
        let space = p.space();
        let gen = GaCatGenerator::from_params(&serde_json::json!({})).unwrap();
        let pop = pop_n(6, n, 5);
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
    fn ga_cat_generator_min_pop_below_2_panics() {
        let n = 3usize;
        let p = CatMatch::new(3, vec![0; n]);
        let space = p.space();
        let gen = GaCatGenerator::from_params(&serde_json::json!({})).unwrap();
        let pop = pop_n(1, n, 3);
        let mut eval = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gen.generate(&pop, &mut ctx)
        }));
        assert!(result.is_err(), "gen/ga-cat must reject pop_size < 2 at runtime as a backstop");
    }

    // ==================================================================
    // registry / spec-validation (required test 5)
    // ==================================================================

    #[test]
    fn registry_resolves_all_three_kinds() {
        let mut reg = Registry::new();
        register(&mut reg);
        assert!(reg.build_generator("gen/cat-ux", &serde_json::json!({})).is_ok());
        assert!(reg.build_generator("gen/cat-reset", &serde_json::json!({})).is_ok());
        assert!(reg.build_generator("gen/ga-cat", &serde_json::json!({})).is_ok());
    }

    #[test]
    fn spec_validation_rejects_gen_ga_cat_on_non_categorical_space() {
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
                generator: ComponentSpec { kind: "gen/ga-cat".into(), params: serde_json::json!({}) },
                replacer: ComponentSpec { kind: "replace/mu-plus-lambda".into(), params: serde_json::json!({}) },
                adapter: None,
            }],
            termination: TerminationSpec { budget: 100, target: None },
            restart: None,
        };
        let e = spec.validate(&reg, &space);
        match e {
            Err(SpecError::UnsupportedBlock { kind, block }) => {
                assert_eq!(kind, "gen/ga-cat");
                assert_eq!(block, "float");
            }
            other => panic!("expected SpecError::UnsupportedBlock, got {other:?}"),
        }
    }

    #[test]
    fn spec_validation_accepts_gen_ga_cat_on_categorical_space() {
        use sezgi_core::spec::{AlgorithmSpec, ComponentSpec, StageSpec, TerminationSpec};
        let mut reg = Registry::new();
        register(&mut reg);
        crate::init::register(&mut reg);
        crate::boundary::register(&mut reg);
        crate::replace::register(&mut reg);
        let space = SearchSpace::new(vec![Block::Categorical { k: 6, n: 4 }]).unwrap();
        let spec = AlgorithmSpec {
            name: "t".into(), pop_size: 10,
            init: ComponentSpec { kind: "init/uniform".into(), params: serde_json::json!({}) },
            boundary: ComponentSpec { kind: "boundary/clamp".into(), params: serde_json::json!({}) },
            stages: vec![StageSpec {
                generator: ComponentSpec { kind: "gen/ga-cat".into(), params: serde_json::json!({}) },
                replacer: ComponentSpec { kind: "replace/mu-plus-lambda".into(), params: serde_json::json!({}) },
                adapter: None,
            }],
            termination: TerminationSpec { budget: 100, target: None },
            restart: None,
        };
        assert!(spec.validate(&reg, &space).is_ok());
    }

    // ==================================================================
    // CatMatch-shaped toy convergence via the spec path AND via
    // presets::ga_cat (required test 3)
    // ==================================================================

    #[test]
    fn ga_cat_converges_on_cat_match_via_engine_spec_path() {
        use sezgi_core::spec::{AlgorithmSpec, ComponentSpec, StageSpec, TerminationSpec};
        use sezgi_core::engine::{Engine, RunConfig};

        let target = vec![3u32, 0, 5, 1, 4];
        let p = CatMatch::new(6, target.clone());
        let mut reg = Registry::new();
        register(&mut reg);
        crate::init::register(&mut reg);
        crate::boundary::register(&mut reg);
        crate::replace::register(&mut reg);

        let spec = AlgorithmSpec {
            name: "ga-cat-toy".into(), pop_size: 40,
            init: ComponentSpec { kind: "init/uniform".into(), params: serde_json::json!({}) },
            boundary: ComponentSpec { kind: "boundary/clamp".into(), params: serde_json::json!({}) },
            stages: vec![StageSpec {
                generator: ComponentSpec { kind: "gen/ga-cat".into(), params: serde_json::json!({}) },
                replacer: ComponentSpec { kind: "replace/mu-plus-lambda".into(), params: serde_json::json!({}) },
                adapter: None,
            }],
            termination: TerminationSpec { budget: 40_000, target: None },
            restart: None,
        };
        let e = Engine::from_spec(&spec, &reg, p.space()).unwrap();
        let r = e.run(&p, RunConfig { master_seed: 2026, run_id: 0 }, None).unwrap();
        assert!(r.best_f < 1.0,
            "gen/ga-cat must drive a CatMatch toy to its exact optimum within budget, got {}", r.best_f);
    }

    #[test]
    fn ga_cat_converges_on_cat_match_via_presets_ga_cat() {
        use sezgi_core::engine::{Engine, RunConfig};

        let target = vec![2u32, 2, 0, 5, 3, 1];
        let p = CatMatch::new(6, target.clone());
        let mut reg = Registry::new();
        crate::init::register(&mut reg);
        crate::boundary::register(&mut reg);
        crate::replace::register(&mut reg);
        register(&mut reg);

        let spec = crate::presets::ga_cat(40, 40_000);
        let e = Engine::from_spec(&spec, &reg, p.space()).unwrap();
        let r = e.run(&p, RunConfig { master_seed: 77, run_id: 0 }, None).unwrap();
        assert!(r.best_f < 1.0,
            "presets::ga_cat must drive a CatMatch toy to its exact optimum within budget, got {}", r.best_f);
    }
}
