use sezgi_core::component::*;
use sezgi_core::problem::Population;
use sezgi_core::rng::RngStream;
use sezgi_core::space::{Block, BlockValues, Genotype, SearchSpace};

use crate::nsga2::{polynomial_mutation, sbx_pair};

/// Integer operators: `gen/int-sbx`, `gen/int-pm`, `gen/ga-int` -- the
/// Int-genotype counterpart to `bin_ops.rs`'s binary triple
/// (`gen/bin-2pt`/`gen/bit-flip`/`gen/ga-bin`) and `ga.rs`'s Float pair
/// (`gen/ga-real`). Like `bin_ops.rs`/`perm.rs`, all three operate on a
/// single `Block::Int { lo, hi, n }` block (block tag `"int"`, per
/// `sezgi_core::spec::block_tag` -- the exact string checked by
/// `AlgorithmSpec::validate`'s `SupportedBlocks::Only` gate).
///
/// ## PROVENANCE -- pymoo 0.6.2's Integer convention (Apache-2.0)
///
/// This module structurally ports pymoo 0.6.2's Integer-variable operator
/// wiring -- specifically its choice to run the real-coded SBX crossover and
/// polynomial mutation cores IN FLOAT, clamped to `[lo, hi]`, then round the
/// result to the nearest integer via `RoundingRepair`. pymoo is licensed
/// Apache-2.0; this is attributed structural porting, not a verbatim code
/// copy: pymoo (github.com/anyoptimization/pymoo, tag `0.6.2`) --
///
/// - `pymoo/core/mixed.py`, `MixedVariableMating.__init__` -- the file that
///   WIRES `Integer` to its operators with their defaults:
///   ```python
///   if crossover is None:
///       crossover = {
///           Binary: UX(), Real: SBX(),
///           Integer: SBX(vtype=float, repair=RoundingRepair()),
///           Choice: UX(),
///       }
///   if mutation is None:
///       mutation = {
///           Binary: BFM(), Real: PM(),
///           Integer: PM(vtype=float, repair=RoundingRepair()),
///           Choice: ChoiceRandomMutation(),
///       }
///   ```
///   `vtype=float` on both is the key finding: pymoo computes Integer SBX/PM
///   in FLOAT (the SAME `SBX`/`PM` classes used for `Real` variables, no
///   Integer-specific crossover/mutation math exists in pymoo), and ONLY
///   converts to `int` afterward, via the paired `repair=RoundingRepair()`.
/// - `pymoo/operators/repair/rounding.py`, `RoundingRepair._do` -- the
///   rounding step, quoted verbatim:
///   ```python
///   def _do(self, problem, X, **kwargs):
///       return np.around(X).astype(int)
///   ```
///   `np.around` rounds HALF TO EVEN (banker's rounding, IEEE 754 default),
///   NOT half-away-from-zero -- `np.around(0.5) == 0.0`,
///   `np.around(1.5) == 2.0`. Ported here via Rust's `f64::round_ties_even`
///   ([`round_repair`]) for exact numeric parity at tie values, not the more
///   common `f64::round` (half-away-from-zero), which would silently diverge
///   from pymoo at exact `.5` boundaries.
/// - `pymoo/core/operator.py`, `Operator.__call__` -- WHERE rounding runs
///   relative to bound repair, quoted verbatim (trimmed to the relevant
///   branch):
///   ```python
///   def __call__(self, problem, elem, *args, to_numpy=False, **kwargs):
///       out = self.do(problem, elem, *args, **kwargs)
///       if self.vtype is not None:
///           for ind in out:
///               ind.X = ind.X.astype(self.vtype)
///       # allow to have a built-in repair (can be useful to customize standard crossover)
///       if self.repair is not None:
///           self.repair.do(problem, out)
///       ...
///   ```
///   `self.repair.do(...)` (`RoundingRepair`) runs AFTER `self.do(...)`
///   (the SBX/PM float computation) has ALREADY returned. Bound repair is
///   NOT a separate outer step here -- it happens INSIDE `self.do(...)`,
///   before rounding ever sees the value (next point).
/// - `pymoo/operators/crossover/sbx.py`, `cross_sbx` -- confirms bound
///   repair is CLAMP, not resample, and runs BEFORE rounding (it is the last
///   thing `cross_sbx` itself does, quoted verbatim):
///   ```python
///   Q[0] = repair_clamp(Q[0], xl, xu)
///   Q[1] = repair_clamp(Q[1], xl, xu)
///   return Q
///   ```
///   `pymoo/operators/repair/bounds_repair.py`, `repair_clamp`, quoted
///   with docstring and lint comments elided (code lines verbatim):
///   ```python
///   def repair_clamp(Xp, xl, xu):
///       XL, XU = repeat_bounds(xl, xu, len(Xp))
///       I = np.where(Xp < XL)
///       Xp[I] = XL[I]
///       I = np.where(Xp > XU)
///       Xp[I] = XU[I]
///       return Xp
///   ```
///   i.e. an ordinary clamp to `[xl, xu]`, no resampling. `pymoo/operators/
///   mutation/pm.py`'s `mut_pm` does the analogous clamp for mutation
///   (`_Y[_Y < _xl] = _xl[...]`; `_Y[_Y > _xu] = _xu[...]`, followed by
///   `set_to_bounds_if_outside` as a floating-point-safety backstop) before
///   `Operator.__call__` rounds. **The verified order is CLAMP then ROUND**
///   (not round-then-clamp, and not resample) -- and since `xl`/`xu` for an
///   Integer variable are themselves whole numbers, clamping into `[xl, xu]`
///   BEFORE rounding already guarantees the rounded result cannot escape
///   `[xl, xu]` (rounding is monotonic and fixes integers), so a SECOND
///   clamp after rounding would be redundant. This module follows the exact
///   same order: [`sbx_pair`]/[`polynomial_mutation`] (both reused from
///   `nsga2.rs`, see "The reuse seam" below) ALREADY clamp their float
///   result to `[lo, hi]` internally (KanGAL's own `v1 = v1.clamp(yl, yu)`
///   / `y_new = (...).clamp(yl, yu)`, unrelated to pymoo but numerically the
///   same clamp-to-bounds contract) -- [`sbx_round_pair`]/
///   [`pm_round_mutation`] below round ONLY, with no second clamp, matching
///   pymoo's own no-second-clamp finding.
/// - `pymoo/core/crossover.py`, `Crossover.__init__` -- the pair-level gate
///   probability's default, quoted verbatim: `def __init__(self, n_parents,
///   n_offsprings, prob=0.9, **kwargs)`. `SimulatedBinaryCrossover` (SBX)
///   does not override `prob`, so Integer's SBX crossover gate defaults to
///   **`p_c = 0.9`**.
/// - `pymoo/operators/crossover/sbx.py`, `SimulatedBinaryCrossover.__init__`
///   -- the distribution index's default, quoted verbatim: `def __init__(
///   self, prob_var=0.5, eta=15, prob_exch=1.0, prob_bin=0.5, n_offsprings=2,
///   **kwargs)`. Integer's SBX is constructed as `SBX(vtype=float,
///   repair=RoundingRepair())` (no `eta=` override), so **`eta_c = 15.0`**.
/// - `pymoo/operators/mutation/pm.py`, `PolynomialMutation.__init__` --
///   quoted verbatim: `def __init__(self, prob=0.9, eta=20, at_least_once=
///   False, **kwargs)`. Integer's PM is `PM(vtype=float,
///   repair=RoundingRepair())` (no `eta=` override), so **`eta_m = 20.0`**.
/// - `pymoo/core/mutation.py`, `Mutation.get_prob_var` -- the PER-VARIABLE
///   mutation probability's default (distinct from the individual-level
///   `prob=0.9` above, which this module has no equivalent of -- see next
///   paragraph), quoted verbatim:
///   ```python
///   def get_prob_var(self, problem, **kwargs):
///       prob_var = (
///           self.prob_var if self.prob_var is not None else min(0.5, 1 / problem.n_var)
///       )
///       return get(prob_var, **kwargs)
///   ```
///   `PM.__init__` never sets `self.prob_var` (no `prob_var=` parameter on
///   `PolynomialMutation` at all), so Integer's per-variable mutation
///   probability resolves to **`min(0.5, 1/n_var)`** -- which EQUALS
///   `1/n_var` for every `n_var >= 2` (the only case where `min` picks the
///   non-0.5 branch), i.e. the SAME `1/n` convention this crate's
///   `gen/ga-real`/`gen/ga-bin`/`gen/ga-perm` already default `p_m` to (Deb,
///   Pratap, Agarwal & Meyarivan 2002, Sec. IV.A -- see `nsga2.rs`'s/
///   `bin_ops.rs`'s own "Defaults" doc sections). `gen/int-pm`/`gen/ga-int`
///   below default `p_m` to `1/n` (matching that existing convention
///   exactly, and pymoo's own resolved value for every realistic `n >= 2`).
///
/// pymoo's PM class ALSO has an outer, INDIVIDUAL-level gate (`prob=0.9`,
/// `pymoo/core/mutation.py`'s `Mutation.do`: `mut = random_state.random(
/// size=n_mut) <= prob`) layered ON TOP of the per-variable `prob_var` gate
/// above -- a two-level gate structure. This module has NO equivalent:
/// [`polynomial_mutation`] (reused verbatim from `nsga2.rs`, KanGAL-pinned)
/// gates PER VARIABLE ONLY, with no outer individual-level gate, matching
/// this crate's EXISTING convention for every other representation
/// (`gen/ga-real`'s/`gen/bit-flip`'s/`gen/perm-swap`'s own mutation
/// generators are all single-level-gated) -- reproducing pymoo's second gate
/// level here would silently change `nsga2.rs`'s frozen, KanGAL-validated
/// core's semantics, which the reuse seam below explicitly forbids. This is
/// a documented, deliberate delta from pymoo's own two-level PM, not an
/// oversight.
///
/// ## The reuse seam: `sbx_pair`/`polynomial_mutation` are NOT relocated
///
/// Unlike `bin_ops.rs` (M3-8 Task 2), which had to MOVE `nsga2.rs`'s
/// `pub(crate)` binary cores out to make them reachable from a new module,
/// `nsga2.rs`'s Float cores -- [`crate::nsga2::sbx_pair`] and
/// [`crate::nsga2::polynomial_mutation`] (M3-2, KanGAL-validated,
/// `crossover.c`'s `realcross` / `mutation.c`'s `real_mutate_ind`) -- are
/// ALREADY declared fully `pub fn` inside a `pub mod nsga2`, i.e. already
/// crate-visible (and, incidentally, publicly exported) with NO visibility
/// change needed. This module calls them DIRECTLY (`use crate::nsga2::{
/// polynomial_mutation, sbx_pair};` above) on `f64` copies of the `Int`
/// block's values/bounds, then rounds the result -- see [`sbx_round_pair`]/
/// [`pm_round_mutation`] below. `nsga2.rs` itself is byte-for-byte
/// unmodified by this task: its own frozen test suite is unaffected, and no
/// relocation-acceptance test is needed (contrast `bin_ops.rs`'s own
/// "PROVENANCE -- moved, not re-derived" section, which DOES need one).
///
/// ## Single-block scope (mirrors `perm.rs`'s/`bin_ops.rs`'s own house
/// convention)
///
/// Like `bin_ops.rs`'s own generators (its module doc, "`nsga2.rs`'s own
/// binary genotype handles MULTIPLE ... blocks" section), this module's
/// generators read exactly ONE `Block::Int` block, via `ctx.space.blocks()
/// [0]` ([`int_dim`]/[`int_bounds`]/[`int_values`]) -- consistent with
/// `gen/ga-real` (Float, one block), `gen/ga-perm` (Permutation, one block)
/// and `gen/ga-bin` (Binary, one block), NOT with `nsga2_run`'s multi-block
/// MO path. A general multi-block composition, if ever needed, is out of
/// scope here (M3-8 Task 5's `gen/compound` may revisit it).
///
/// ## `gen/int-sbx` / `gen/ga-int`'s pair loop (mirrors `gen/bin-2pt`/
/// `gen/ga-bin` exactly)
///
/// Tournament selection ([`crate::select::tournament`], shared with every
/// other representation in this crate): draw a candidate, then
/// `tournament_k - 1` more, keeping the best (lowest fitness) seen. Per
/// pair: tournament for parent 1 (`tournament_k` draws), THEN tournament for
/// parent 2 (`tournament_k` more draws), THEN [`sbx_round_pair`] directly --
/// [`sbx_pair`]'s own internal whole-pair gate (`p_c`) is the ONLY gate;
/// callers here do NOT draw a second, redundant gate (same structural
/// reasoning as `bin_ops.rs`'s own `gen/bin-2pt`/`gen/ga-bin`, whose
/// `bin_cross_pair` is likewise internally gated). `gen/ga-int` additionally
/// mutates both children via [`pm_round_mutation`] (`c1` then `c2`) after
/// crossover, mirroring `gen/ga-bin`'s/`gen/ga-perm`'s own
/// crossover-then-mutate-both-children shape.
///
/// ## `gen/int-pm`'s per-individual loop
///
/// Population order (mirrors `gen/bit-flip`'s own per-individual loop
/// shape), but with NO outer per-individual gate: [`polynomial_mutation`]
/// already gates PER VARIABLE internally, so a redundant per-individual gate
/// on top of it would double-gate and is not part of the KanGAL source
/// (`real_mutate_ind` has no individual-level gate, only the per-variable
/// one) -- same reasoning as `gen/bit-flip`'s own doc section.
///
/// ## Defaults
///
/// **`eta_c` (`gen/int-sbx` and `gen/ga-int`): `15.0`.** **`eta_m`
/// (`gen/int-pm` and `gen/ga-int`): `20.0`.** **`p_c` (`gen/int-sbx` and
/// `gen/ga-int`): `0.9`.** **`p_m` (`gen/int-pm` and `gen/ga-int`): `1/n`**,
/// `n` = the space's single `Block::Int`'s dimension. All four VERIFIED
/// against pymoo 0.6.2's own Integer wiring -- see "PROVENANCE" above for
/// the quoted source lines each is drawn from.
///
/// **`tournament_k`: `2`**, mirroring every other representation's own
/// default in this crate.
pub(crate) fn round_repair(x: f64) -> i64 {
    x.round_ties_even() as i64
}

fn to_f64(xs: &[i64]) -> Vec<f64> {
    xs.iter().map(|&v| v as f64).collect()
}

fn round_all(xs: &[f64]) -> Vec<i64> {
    xs.iter().map(|&v| round_repair(v)).collect()
}

/// `sbx_pair` ([`crate::nsga2::sbx_pair`], KanGAL-pinned, reused verbatim --
/// see this module's doc, "The reuse seam") run on `f64` copies of the two
/// `Int` parents, bounded by `[lo, hi]` (also cast to `f64`), then rounded
/// coordinate-wise via [`round_repair`] (pymoo's `RoundingRepair`, no second
/// clamp -- see this module's "PROVENANCE" doc section for why one is not
/// needed). `p1`/`p2`/`lo`/`hi` must be the same length.
pub fn sbx_round_pair(
    p1: &[i64],
    p2: &[i64],
    lo: &[i64],
    hi: &[i64],
    eta_c: f64,
    p_c: f64,
    rng: &mut RngStream,
) -> (Vec<i64>, Vec<i64>) {
    let (c1f, c2f) = sbx_pair(&to_f64(p1), &to_f64(p2), &to_f64(lo), &to_f64(hi), eta_c, p_c, rng);
    (round_all(&c1f), round_all(&c2f))
}

/// `polynomial_mutation` ([`crate::nsga2::polynomial_mutation`],
/// KanGAL-pinned, reused verbatim -- see this module's doc, "The reuse
/// seam") run on an `f64` copy of `x`, bounded by `[lo, hi]` (also cast to
/// `f64`), then rounded coordinate-wise via [`round_repair`] back into `x`,
/// in place.
pub fn pm_round_mutation(x: &mut [i64], lo: &[i64], hi: &[i64], eta_m: f64, p_m: f64, rng: &mut RngStream) {
    let mut xf = to_f64(x);
    polynomial_mutation(&mut xf, &to_f64(lo), &to_f64(hi), eta_m, p_m, rng);
    for (xi, &v) in x.iter_mut().zip(xf.iter()) {
        *xi = round_repair(v);
    }
}

fn int_dim(space: &SearchSpace) -> usize {
    let Block::Int { n, .. } = space.blocks()[0] else {
        unreachable!("integer operators require a single Int block")
    };
    n
}

fn int_bounds(space: &SearchSpace) -> (Vec<i64>, Vec<i64>) {
    let Block::Int { lo, hi, n } = space.blocks()[0] else {
        unreachable!("integer operators require a single Int block")
    };
    (vec![lo; n], vec![hi; n])
}

fn int_values(g: &Genotype) -> &Vec<i64> {
    match &g.blocks[0] {
        BlockValues::Int(xs) => xs,
        _ => unreachable!("integer operators require an Int block value"),
    }
}

pub struct IntSbxGenerator {
    pub tournament_k: usize,
    pub eta_c: f64,
    pub p_c: f64,
}

impl IntSbxGenerator {
    pub fn from_params(p: &serde_json::Value) -> Result<Self, ComponentError> {
        let err = |reason: String| ComponentError::InvalidParams { kind: "gen/int-sbx".into(), reason };
        let g = Self {
            tournament_k: p.get("tournament_k").and_then(|v| v.as_u64()).unwrap_or(2) as usize,
            eta_c: p.get("eta_c").and_then(|v| v.as_f64()).unwrap_or(15.0),
            p_c: crate::params::resolve(p, "p_c", "pc").and_then(|v| v.as_f64()).unwrap_or(0.9),
        };
        if g.tournament_k == 0 { return Err(err("tournament_k must be >= 1".into())); }
        if !(0.0..=1.0).contains(&g.p_c) { return Err(err(format!("p_c outside [0,1]: {}", g.p_c))); }
        if g.eta_c.is_nan() || g.eta_c <= 0.0 { return Err(err(format!("eta_c must be > 0: {}", g.eta_c))); }
        Ok(g)
    }
}

impl Generator for IntSbxGenerator {
    /// Pair loop mirroring `gen/bin-2pt`'s own exactly, minus a second
    /// external gate ([`sbx_pair`] already gates internally).
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        assert!(pop.len() >= 2, "gen/int-sbx requires a population of at least 2 (pop_size={})", pop.len());
        let (lo, hi) = int_bounds(ctx.space);
        let mut out = Vec::with_capacity(pop.len());
        while out.len() < pop.len() {
            let i1 = crate::select::tournament(pop, self.tournament_k, ctx.rng);
            let i2 = crate::select::tournament(pop, self.tournament_k, ctx.rng);
            let p1 = int_values(&pop.individuals[i1]).clone();
            let p2 = int_values(&pop.individuals[i2]).clone();
            let (c1, c2) = sbx_round_pair(&p1, &p2, &lo, &hi, self.eta_c, self.p_c, ctx.rng);
            out.push(Genotype { blocks: vec![BlockValues::Int(c1)] });
            if out.len() < pop.len() {
                out.push(Genotype { blocks: vec![BlockValues::Int(c2)] });
            }
        }
        out
    }
    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/int-sbx", SupportedBlocks::Only(vec!["int"])).with_min_pop(2)
    }
}

pub struct IntPmGenerator {
    pub eta_m: f64,
    pub p_m: Option<f64>,
}

impl IntPmGenerator {
    pub fn from_params(p: &serde_json::Value) -> Result<Self, ComponentError> {
        let err = |reason: String| ComponentError::InvalidParams { kind: "gen/int-pm".into(), reason };
        let eta_m = p.get("eta_m").and_then(|v| v.as_f64()).unwrap_or(20.0);
        let p_m = p.get("p_m").and_then(|v| v.as_f64());
        if eta_m.is_nan() || eta_m <= 0.0 { return Err(err(format!("eta_m must be > 0: {eta_m}"))); }
        if let Some(pm) = p_m {
            if !(0.0..=1.0).contains(&pm) { return Err(err(format!("p_m outside [0,1]: {pm}"))); }
        }
        Ok(Self { eta_m, p_m })
    }
}

impl Generator for IntPmGenerator {
    /// Per-individual loop, population order (mirrors `gen/bit-flip`'s own
    /// shape) -- NO outer per-individual gate: [`polynomial_mutation`]
    /// already gates per variable internally (see this module's doc).
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        let dim = int_dim(ctx.space);
        let (lo, hi) = int_bounds(ctx.space);
        let pm = self.p_m.unwrap_or(1.0 / dim as f64);
        (0..pop.len())
            .map(|i| {
                let mut xs = int_values(&pop.individuals[i]).clone();
                pm_round_mutation(&mut xs, &lo, &hi, self.eta_m, pm, ctx.rng);
                Genotype { blocks: vec![BlockValues::Int(xs)] }
            })
            .collect()
    }
    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/int-pm", SupportedBlocks::Only(vec!["int"]))
    }
}

/// `gen/ga-int` (M3-8 Task 3): the FUSED integer-GA generator -- SBX
/// crossover AND polynomial mutation, both computed in float then rounded
/// (see this module's "PROVENANCE" doc), inside ONE `Generator`, one
/// evaluate-and-replace per generation, mirroring [`crate::ga::GaRealGenerator`]'s
/// / [`crate::bin_ops::GaBinGenerator`]'s own single-stage shape for the Int
/// representation. Reuses [`sbx_round_pair`]/[`pm_round_mutation`] directly
/// -- see this module's doc for the full draw-order contract and defaults
/// provenance.
pub struct GaIntGenerator {
    pub tournament_k: usize,
    pub eta_c: f64,
    pub eta_m: f64,
    pub p_c: f64,
    pub p_m: Option<f64>,
}

impl GaIntGenerator {
    pub fn from_params(p: &serde_json::Value) -> Result<Self, ComponentError> {
        let err = |reason: String| ComponentError::InvalidParams { kind: "gen/ga-int".into(), reason };
        let g = Self {
            tournament_k: p.get("tournament_k").and_then(|v| v.as_u64()).unwrap_or(2) as usize,
            eta_c: p.get("eta_c").and_then(|v| v.as_f64()).unwrap_or(15.0),
            eta_m: p.get("eta_m").and_then(|v| v.as_f64()).unwrap_or(20.0),
            p_c: crate::params::resolve(p, "p_c", "pc").and_then(|v| v.as_f64()).unwrap_or(0.9),
            p_m: p.get("p_m").and_then(|v| v.as_f64()),
        };
        if g.tournament_k == 0 { return Err(err("tournament_k must be >= 1".into())); }
        if !(0.0..=1.0).contains(&g.p_c) { return Err(err(format!("p_c outside [0,1]: {}", g.p_c))); }
        if g.eta_c.is_nan() || g.eta_c <= 0.0 { return Err(err(format!("eta_c must be > 0: {}", g.eta_c))); }
        if g.eta_m.is_nan() || g.eta_m <= 0.0 { return Err(err(format!("eta_m must be > 0: {}", g.eta_m))); }
        if let Some(pm) = g.p_m {
            if !(0.0..=1.0).contains(&pm) { return Err(err(format!("p_m outside [0,1]: {pm}"))); }
        }
        Ok(g)
    }
}

impl Generator for GaIntGenerator {
    /// See this module's doc for the full draw-order contract: per pair,
    /// tournament x2 (`tournament_k` draws each), THEN [`sbx_round_pair`]
    /// (its own internal gate + conditional per-variable draws) producing
    /// `(c1, c2)`, THEN [`pm_round_mutation`] on `c1` then `c2` (each
    /// internally gated per variable).
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        assert!(pop.len() >= 2, "gen/ga-int requires a population of at least 2 (pop_size={})", pop.len());
        let dim = int_dim(ctx.space);
        let (lo, hi) = int_bounds(ctx.space);
        let pm = self.p_m.unwrap_or(1.0 / dim as f64);
        let mut out = Vec::with_capacity(pop.len());
        while out.len() < pop.len() {
            let i1 = crate::select::tournament(pop, self.tournament_k, ctx.rng);
            let i2 = crate::select::tournament(pop, self.tournament_k, ctx.rng);
            let p1 = int_values(&pop.individuals[i1]).clone();
            let p2 = int_values(&pop.individuals[i2]).clone();
            let (mut c1, mut c2) = sbx_round_pair(&p1, &p2, &lo, &hi, self.eta_c, self.p_c, ctx.rng);
            pm_round_mutation(&mut c1, &lo, &hi, self.eta_m, pm, ctx.rng);
            pm_round_mutation(&mut c2, &lo, &hi, self.eta_m, pm, ctx.rng);
            out.push(Genotype { blocks: vec![BlockValues::Int(c1)] });
            if out.len() < pop.len() {
                out.push(Genotype { blocks: vec![BlockValues::Int(c2)] });
            }
        }
        out
    }
    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/ga-int", SupportedBlocks::Only(vec!["int"])).with_min_pop(2)
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_generator("gen/int-sbx", |p| Ok(Box::new(IntSbxGenerator::from_params(p)?)));
    reg.register_generator("gen/int-pm", |p| Ok(Box::new(IntPmGenerator::from_params(p)?)));
    reg.register_generator("gen/ga-int", |p| Ok(Box::new(GaIntGenerator::from_params(p)?)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::problem::{Evaluator, Problem};
    use sezgi_core::space::SearchSpace;
    use sezgi_core::state::Blackboard;

    /// A minimal `Problem` with a single `Int { lo, hi, n }` block, used
    /// only to construct a valid `Evaluator`/`Ctx` for these tests -- mirrors
    /// `bin_ops.rs`'s own `BinProblem` scaffold. Fitness = sum of squared
    /// deviation from `target` (an IntQuadratic-shaped inline toy, per the
    /// task brief -- not a public problem type, Task 5's scope).
    struct IntQuadratic { space: SearchSpace, target: Vec<i64> }
    impl IntQuadratic {
        fn new(lo: i64, hi: i64, target: Vec<i64>) -> Self {
            let n = target.len();
            Self { space: SearchSpace::new(vec![Block::Int { lo, hi, n }]).unwrap(), target }
        }
    }
    impl Problem for IntQuadratic {
        fn space(&self) -> &SearchSpace { &self.space }
        fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<f64> {
            pop.iter().map(|g| {
                int_values(g).iter().zip(&self.target)
                    .map(|(&x, &t)| ((x - t) * (x - t)) as f64)
                    .sum()
            }).collect()
        }
    }

    fn g(xs: Vec<i64>) -> Genotype { Genotype { blocks: vec![BlockValues::Int(xs)] } }

    fn pop_n(n: usize, dim: usize, lo: i64, hi: i64) -> Population {
        let span = (hi - lo + 1).max(1);
        Population {
            individuals: (0..n).map(|i| {
                g((0..dim).map(|j| lo + ((j + i) as i64 % span)).collect())
            }).collect(),
            fitness: (0..n).map(|i| (n - i) as f64).collect(),
        }
    }

    // ==================================================================
    // round_repair (RoundingRepair port)
    // ==================================================================

    #[test]
    fn round_repair_ties_to_even_matches_numpy_around() {
        // np.around(0.5) == 0.0, np.around(1.5) == 2.0, np.around(2.5) == 2.0,
        // np.around(-0.5) == -0.0, np.around(-1.5) == -2.0 -- verified via the
        // scratchpad oracle probe (see task-3-report.md).
        assert_eq!(round_repair(0.5), 0);
        assert_eq!(round_repair(1.5), 2);
        assert_eq!(round_repair(2.5), 2);
        assert_eq!(round_repair(-0.5), 0);
        assert_eq!(round_repair(-1.5), -2);
        assert_eq!(round_repair(3.5), 4);
        assert_eq!(round_repair(3.2), 3);
        assert_eq!(round_repair(3.7), 4);
    }

    // ==================================================================
    // Hand-traced seeded-RNG exactness + pymoo oracle anchor (required
    // tests 1 and 2 -- see task-3-report.md for the full derivation and the
    // pymoo probe script/output this anchors against)
    // ==================================================================

    // Seed 0, path [] -- first four draws (reconnaissance, full f64
    // precision; the SAME seed 0 reconnaissance `bin_ops.rs`'s own
    // hand-trace test anchors, independently re-confirmed here) are:
    //   d0 = 0.32457526803140668292  (pair gate: <= p_c=1.0 -> always passes)
    //   d1 = 0.38223929651167343202  (per-variable exchange gate: <= 0.5 -> proceeds)
    //   d2 = 0.35961720764735527478  (the shared SBX "rand" draw)
    //   d3 = 0.01145550893465363540  (final per-variable swap gate: <= 0.5 -> swap)
    // p1=[2], p2=[9] (y1=2, y2=9 since p1<p2), lo=[0], hi=[15], eta_c=15.0,
    // p_c=1.0. Fed through pymoo 0.6.2's OWN `cross_sbx` (scratchpad oracle
    // probe `oracle_probe.py::sbx_scenario`, calling
    // `pymoo.operators.crossover.sbx.cross_sbx` directly with a scripted
    // random_state returning d2 as its shared "rand" draw, `prob_var=1.0`/
    // `prob_bin=0.0` to force pymoo's own cross-mask/exchange draws to
    // constants that isolate the per-variable kernel -- see this module's
    // PROVENANCE doc), pymoo computes (pre-rounding, pre any pymoo-side
    // exchange):
    //   Q[0,0,0] = 2.0714330794903666  (y1 side, "child1")
    //   Q[1,0,0] = 8.928644413531657   (y2 side, "child2")
    // -- this is the SAME `calc_betaq`/beta/alpha/v1/v2 formula as
    // `sbx_pair` (module doc, PROVENANCE) with the SAME
    // y1/y2/xl/xu/eta_c/rand inputs, so it is an EXACT cross-check of the
    // shared mathematical kernel (NOT of pymoo's own outer exchange-gate
    // structure, which differs from KanGAL's final swap-gate -- see the
    // module doc). `sbx_pair`'s OWN final swap gate (d3=0.0115 <= 0.5)
    // swaps the y1/y2-side values into (c1,c2) = (v2,v1) =
    // (8.928644413531657, 2.0714330794903666) -> rounded (ties-to-even,
    // neither is a tie): (9, 2).
    #[test]
    fn sbx_round_pair_hand_trace_matches_pymoo_oracle_kernel() {
        let rng_before = RngStream::from_master(0, &[]);
        let mut rng = rng_before.clone();
        let (c1, c2) = sbx_round_pair(&[2], &[9], &[0], &[15], 15.0, 1.0, &mut rng);
        // pymoo's pre-rounding kernel values were v1=2.0714330794903666 (y1
        // side) and v2=8.928644413531657 (y2 side); sbx_pair's own swap
        // gate (d3<=0.5) swaps them into c1=round(v2)=9, c2=round(v1)=2.
        assert_eq!(c1, vec![9]);
        assert_eq!(c2, vec![2]);

        let mut twin = rng_before;
        for _ in 0..4 { let _ = twin.next_f64(); }
        assert_eq!(rng.next_f64(), twin.next_f64(),
            "sbx_round_pair (gate pass, 1 var, gate+rand+swap all resolved) must consume exactly 4 draws");
    }

    // Seed 42, path [] -- first two draws are:
    //   d0 = 0.81430514512290985696  (per-variable mutation gate: p_m=1.0 -> always passes)
    //   d1 = 0.31882104006166112065  (the "rnd" value draw, <= 0.5 branch)
    // x=[5], lo=[0], hi=[10], eta_m=20.0, p_m=1.0. Fed through pymoo 0.6.2's
    // OWN `mut_pm` (scratchpad oracle probe `oracle_probe.py::pm_scenario`,
    // calling `pymoo.operators.mutation.pm.mut_pm` directly with a scripted
    // random_state returning d1 as the `rand` value draw, `prob=1.0` to
    // force pymoo's own `mut_binomial` gate draw to a constant that isolates
    // the per-variable kernel -- see this module's PROVENANCE doc), pymoo
    // computes (pre-rounding): y_new = 4.78800419982762 -- an EXACT
    // cross-check of `polynomial_mutation`'s per-variable kernel against
    // pymoo's own `mut_pm` (same delta1/delta2/deltaq formula, same
    // gate/value draw order for n_var=1: see the module doc's PROVENANCE
    // section). Rounded (not a tie): 5.
    #[test]
    fn pm_round_mutation_hand_trace_matches_pymoo_oracle_kernel() {
        let rng_before = RngStream::from_master(42, &[]);
        let mut rng = rng_before.clone();
        let mut x = vec![5i64];
        pm_round_mutation(&mut x, &[0], &[10], 20.0, 1.0, &mut rng);
        // pymoo's pre-rounding kernel value was y_new=4.78800419982762 ->
        // round_ties_even -> 5.
        assert_eq!(x, vec![5]);

        let mut twin = rng_before;
        for _ in 0..2 { let _ = twin.next_f64(); }
        assert_eq!(rng.next_f64(), twin.next_f64(),
            "pm_round_mutation (gate pass, 1 var) must consume exactly 2 draws (gate + value)");
    }

    #[test]
    fn sbx_round_pair_gate_fail_is_verbatim_parents() {
        let mut rng = RngStream::from_master(55, &[]);
        let (c1, c2) = sbx_round_pair(&[1, 2, 3], &[9, 8, 7], &[0, 0, 0], &[10, 10, 10], 15.0, 0.0, &mut rng);
        assert_eq!(c1, vec![1, 2, 3], "p_c=0 must leave the block unchanged (gate always fails)");
        assert_eq!(c2, vec![9, 8, 7], "p_c=0 must leave the block unchanged (gate always fails)");
    }

    #[test]
    fn pm_round_mutation_zero_p_m_is_identity() {
        let x0 = vec![1i64, 5, 9, 0, 10];
        let mut x = x0.clone();
        let mut rng = RngStream::from_master(2024, &[]);
        pm_round_mutation(&mut x, &[0; 5], &[10; 5], 20.0, 0.0, &mut rng);
        assert_eq!(x, x0, "p_m=0 must leave every variable unchanged");
    }

    // ==================================================================
    // Statistical oracle anchor (required test 2, statistical half): where
    // pymoo's own outer exchange-gate structure (`prob_bin`-driven XOR swap,
    // `sbx.py`) and `sbx_pair`'s KanGAL final swap gate (hardcoded `<=0.5`)
    // are STRUCTURALLY different mechanisms -- not the same formula fed the
    // same draw, so no exact per-draw comparison is possible -- this anchors
    // `sbx_round_pair`'s/`pm_round_mutation`'s empirical output DISTRIBUTION
    // against pymoo 0.6.2's own empirical distribution, both run with their
    // OWN independent RNG (numpy's `default_rng(12345)` for pymoo, this
    // crate's `RngStream` for Rust -- no shared draws, unlike the hand-trace
    // tests above), over N=20000 trials each (scratchpad
    // `oracle_probe_stats.py`, checked into the task-3 report). Anchor
    // numbers (pymoo, N=20000):
    //   PM:  x=50, lo=0, hi=100, eta_m=20, p_m=1.0 (always mutate) -- delta
    //        (rounded_new - 50): mean=0.041800, std=6.345176, frac_zero=0.099200
    //   SBX: p1=20, p2=80, lo=0, hi=100, eta_c=15, prob_var=0.5 (matches
    //        sbx_pair's own hardcoded per-variable exchange rate),
    //        prob_bin=0.5 -- child1: mean=34.912850, std=26.069307
    // Tolerance bands below are deliberately generous (independent RNG,
    // finite-N sampling noise on BOTH sides, plus the documented structural
    // exchange-gate delta) -- they anchor "same distribution family, same
    // scale", not bit-for-bit agreement.
    #[test]
    fn pm_round_mutation_distribution_matches_pymoo_oracle_statistically() {
        let mut rng = RngStream::from_master(4242, &[]);
        let trials = 20_000;
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut zero_count = 0usize;
        for _ in 0..trials {
            let mut x = vec![50i64];
            pm_round_mutation(&mut x, &[0], &[100], 20.0, 1.0, &mut rng);
            let delta = (x[0] - 50) as f64;
            sum += delta;
            sum_sq += delta * delta;
            if x[0] == 50 { zero_count += 1; }
        }
        let mean = sum / trials as f64;
        let var = sum_sq / trials as f64 - mean * mean;
        let std = var.sqrt();
        let frac_zero = zero_count as f64 / trials as f64;
        // pymoo anchor: mean=0.0418, std=6.345, frac_zero=0.0992.
        assert!(mean.abs() < 1.0, "PM delta mean {mean} should be near pymoo's oracle anchor 0.0418 (symmetric bounds around x=50)");
        assert!((4.5..8.5).contains(&std), "PM delta std {std} should be near pymoo's oracle anchor 6.345");
        assert!((0.04..0.18).contains(&frac_zero), "PM frac_zero {frac_zero} should be near pymoo's oracle anchor 0.0992");
    }

    #[test]
    fn sbx_round_pair_distribution_matches_pymoo_oracle_statistically() {
        let mut rng = RngStream::from_master(4343, &[]);
        let trials = 20_000;
        let mut sum1 = 0.0f64;
        let mut sum1_sq = 0.0f64;
        let mut max_mid_dev: f64 = 0.0;
        for _ in 0..trials {
            let (c1, c2) = sbx_round_pair(&[20], &[80], &[0], &[100], 15.0, 1.0, &mut rng);
            sum1 += c1[0] as f64;
            sum1_sq += (c1[0] as f64) * (c1[0] as f64);
            // Symmetric bounds/parents (y1-yl == yu-y2 == 20): the same
            // algebraic identity as nsga2.rs's own
            // `sbx_pair_distribution_sanity_symmetric_pair` test holds
            // EXACTLY here too (pre-rounding); post-rounding it can be off
            // by at most 0.5 per side.
            let mid_dev = ((c1[0] + c2[0]) as f64 / 2.0 - 50.0).abs();
            if mid_dev > max_mid_dev { max_mid_dev = mid_dev; }
        }
        let mean1 = sum1 / trials as f64;
        let var1 = sum1_sq / trials as f64 - mean1 * mean1;
        let std1 = var1.sqrt();
        // pymoo anchor: child1 mean=34.9129, std=26.0693.
        assert!((25.0..45.0).contains(&mean1), "SBX child1 mean {mean1} should be near pymoo's oracle anchor 34.9129");
        assert!((18.0..34.0).contains(&std1), "SBX child1 std {std1} should be near pymoo's oracle anchor 26.0693");
        assert!(max_mid_dev <= 1.0, "symmetric-pair rounded midpoint must stay within 1.0 of the exact pre-rounding midpoint (max dev {max_mid_dev})");
    }

    #[test]
    fn sbx_round_pair_equal_parents_are_eps_guard_verbatim() {
        // Integer parents equal on a variable is common (unlike Float): the
        // reused sbx_pair's EPS guard (|p1-p2|<=1e-14) catches this and
        // copies verbatim, avoiding a div-by-zero in beta's (y2-y1)
        // denominator -- verified here as a boundary-safety regression.
        let mut rng = RngStream::from_master(9, &[]);
        let (c1, c2) = sbx_round_pair(&[5, 5], &[5, 9], &[0, 0], &[10, 10], 15.0, 1.0, &mut rng);
        assert_eq!(c1[0], 5);
        assert_eq!(c2[0], 5);
    }

    // ==================================================================
    // Boundary integrity property test (required test 4): offspring never
    // outside [lo,hi], across many seeds, for gen/int-sbx, gen/int-pm and
    // gen/ga-int.
    // ==================================================================

    #[test]
    fn boundary_integrity_offspring_never_out_of_bounds_across_seeds() {
        let lo = -7i64;
        let hi = 12i64;
        let n = 6usize;
        let p = IntQuadratic::new(lo, hi, vec![0; n]);
        let space = p.space();
        let sbx = IntSbxGenerator::from_params(&serde_json::json!({"eta_c": 3.0})).unwrap();
        let pm = IntPmGenerator::from_params(&serde_json::json!({"eta_m": 3.0, "p_m": 0.9})).unwrap();
        let ga = GaIntGenerator::from_params(&serde_json::json!({"eta_c": 3.0, "eta_m": 3.0, "p_m": 0.9})).unwrap();
        let mut eval = Evaluator::new(&p, 100_000_000);
        let mut bb = Blackboard::new();
        for seed in 0..300u64 {
            let pop = pop_n(9, n, lo, hi);
            let mut rng = RngStream::from_master(seed, &[0]);
            {
                let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
                for ind in sbx.generate(&pop, &mut ctx) {
                    for &x in int_values(&ind) { assert!((lo..=hi).contains(&x), "gen/int-sbx out of bounds: {x}"); }
                }
            }
            {
                let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
                for ind in pm.generate(&pop, &mut ctx) {
                    for &x in int_values(&ind) { assert!((lo..=hi).contains(&x), "gen/int-pm out of bounds: {x}"); }
                }
            }
            {
                let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
                for ind in ga.generate(&pop, &mut ctx) {
                    for &x in int_values(&ind) { assert!((lo..=hi).contains(&x), "gen/ga-int out of bounds: {x}"); }
                }
            }
        }
    }

    // ==================================================================
    // gen/int-sbx (IntSbxGenerator)
    // ==================================================================

    #[test]
    fn int_sbx_generator_validity_property_seeded_batch() {
        let n = 5usize;
        let p = IntQuadratic::new(-10, 10, vec![0; n]);
        let space = p.space();
        let gen = IntSbxGenerator::from_params(&serde_json::json!({})).unwrap();
        let mut eval = Evaluator::new(&p, 1_000_000);
        let mut rng = RngStream::from_master(3, &[0]);
        let mut bb = Blackboard::new();
        for _ in 0..100 {
            let pop = pop_n(9, n, -10, 10);
            let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
            let off = gen.generate(&pop, &mut ctx);
            assert_eq!(off.len(), pop.len());
            for ind in &off { assert_eq!(int_values(ind).len(), n); }
        }
    }

    #[test]
    fn int_sbx_generator_deterministic_same_seed() {
        let n = 4usize;
        let p = IntQuadratic::new(0, 20, vec![0; n]);
        let space = p.space();
        let gen = IntSbxGenerator::from_params(&serde_json::json!({})).unwrap();
        let pop = pop_n(6, n, 0, 20);
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
    fn int_sbx_generator_min_pop_below_2_panics() {
        let n = 3usize;
        let p = IntQuadratic::new(0, 5, vec![0; n]);
        let space = p.space();
        let gen = IntSbxGenerator::from_params(&serde_json::json!({})).unwrap();
        let pop = pop_n(1, n, 0, 5);
        let mut eval = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gen.generate(&pop, &mut ctx)
        }));
        assert!(result.is_err(), "gen/int-sbx must reject pop_size < 2 at runtime as a backstop");
    }

    #[test]
    fn int_sbx_generator_meta_min_pop_2_and_block_restricted() {
        let m = IntSbxGenerator::from_params(&serde_json::json!({})).unwrap().meta();
        assert_eq!(m.min_pop, 2);
        assert!(m.supports_block("int"));
        assert!(!m.supports_block("float"));
        assert_eq!(m.kind, "gen/int-sbx");
    }

    #[test]
    fn int_sbx_params_validate() {
        assert!(IntSbxGenerator::from_params(&serde_json::json!({"p_c": 1.5})).is_err());
        assert!(IntSbxGenerator::from_params(&serde_json::json!({"tournament_k": 0})).is_err());
        assert!(IntSbxGenerator::from_params(&serde_json::json!({"eta_c": 0.0})).is_err());
        assert!(IntSbxGenerator::from_params(&serde_json::json!({"eta_c": -1.0})).is_err());
        let g = IntSbxGenerator::from_params(&serde_json::json!({})).unwrap();
        assert_eq!(g.tournament_k, 2);
        assert_eq!(g.p_c, 0.9);
        assert_eq!(g.eta_c, 15.0);
    }

    /// `gen/int-sbx` is an M3-8 typed family (historically parsed only
    /// `p_c`) -- it now ALSO accepts legacy `pc`, resolving identically to
    /// the equivalent `p_c` spec (M3-8 deferral cleanup, `params.rs`).
    #[test]
    fn int_sbx_pc_alias_resolves_identically_to_canonical_p_c() {
        let legacy = IntSbxGenerator::from_params(&serde_json::json!({"pc": 0.42})).unwrap();
        let canonical = IntSbxGenerator::from_params(&serde_json::json!({"p_c": 0.42})).unwrap();
        assert_eq!(legacy.p_c, canonical.p_c);
        assert_eq!(legacy.p_c, 0.42);
    }

    /// When both spellings are present on the same block, canonical `p_c`
    /// wins outright -- `pc`'s value is never used.
    #[test]
    fn int_sbx_p_c_wins_over_pc_when_both_present() {
        let g = IntSbxGenerator::from_params(&serde_json::json!({"p_c": 0.42, "pc": 0.99})).unwrap();
        assert_eq!(g.p_c, 0.42);
    }

    // ==================================================================
    // gen/int-pm (IntPmGenerator)
    // ==================================================================

    #[test]
    fn int_pm_generator_validity_property_seeded_batch() {
        let n = 5usize;
        let p = IntQuadratic::new(-10, 10, vec![0; n]);
        let space = p.space();
        let gen = IntPmGenerator::from_params(&serde_json::json!({"p_m": 0.9})).unwrap();
        let mut eval = Evaluator::new(&p, 1_000_000);
        let mut rng = RngStream::from_master(6, &[0]);
        let mut bb = Blackboard::new();
        for _ in 0..200 {
            let pop = pop_n(5, n, -10, 10);
            let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
            let off = gen.generate(&pop, &mut ctx);
            assert_eq!(off.len(), pop.len());
            for ind in &off { assert_eq!(int_values(ind).len(), n); }
        }
    }

    #[test]
    fn int_pm_generator_deterministic_same_seed() {
        let n = 4usize;
        let p = IntQuadratic::new(0, 20, vec![0; n]);
        let space = p.space();
        let gen = IntPmGenerator::from_params(&serde_json::json!({"p_m": 0.5})).unwrap();
        let pop = pop_n(5, n, 0, 20);
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
    fn int_pm_generator_zero_p_m_leaves_population_unchanged() {
        let n = 4usize;
        let p = IntQuadratic::new(0, 20, vec![0; n]);
        let space = p.space();
        let gen = IntPmGenerator::from_params(&serde_json::json!({"p_m": 0.0})).unwrap();
        let pop = pop_n(4, n, 0, 20);
        let mut eval = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(4, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
        let off = gen.generate(&pop, &mut ctx);
        for (o, ind) in off.iter().zip(&pop.individuals) {
            assert_eq!(int_values(o), int_values(ind));
        }
    }

    #[test]
    fn int_pm_generator_meta_default_min_pop_and_block_restricted() {
        let m = IntPmGenerator::from_params(&serde_json::json!({})).unwrap().meta();
        assert_eq!(m.min_pop, 1);
        assert!(m.supports_block("int"));
        assert!(!m.supports_block("permutation"));
        assert_eq!(m.kind, "gen/int-pm");
    }

    #[test]
    fn int_pm_default_p_m_is_one_over_n() {
        let n = 6usize;
        let p = IntQuadratic::new(0, 20, vec![0; n]);
        let space = p.space();
        let gen = IntPmGenerator::from_params(&serde_json::json!({})).unwrap();
        assert_eq!(gen.p_m, None); // resolved lazily against n inside generate()
        let pop = Population { individuals: vec![g(vec![10; n])], fitness: vec![0.0] };
        let mut eval = Evaluator::new(&p, 100);
        // Find a seed whose first next_f64() draw is >= 1/6 (var0's gate
        // fails under the 1/6 default) so var0 must stay unchanged.
        let seed = (0..2000u64).find(|&s| {
            let mut r = RngStream::from_master(s, &[0]);
            r.next_f64() >= 1.0 / 6.0
        }).unwrap();
        let mut rng = RngStream::from_master(seed, &[0]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
        let off = gen.generate(&pop, &mut ctx);
        assert_eq!(int_values(&off[0])[0], 10, "var0's gate must fail (unchanged) when the draw is >= 1/n");
    }

    #[test]
    fn int_pm_params_validate() {
        assert!(IntPmGenerator::from_params(&serde_json::json!({"p_m": 1.5})).is_err());
        assert!(IntPmGenerator::from_params(&serde_json::json!({"p_m": -0.1})).is_err());
        assert!(IntPmGenerator::from_params(&serde_json::json!({"eta_m": 0.0})).is_err());
        assert!(IntPmGenerator::from_params(&serde_json::json!({"p_m": 0.3})).is_ok());
        let g = IntPmGenerator::from_params(&serde_json::json!({})).unwrap();
        assert_eq!(g.eta_m, 20.0);
    }

    // ==================================================================
    // gen/ga-int (GaIntGenerator)
    // ==================================================================

    #[test]
    fn ga_int_generator_meta_min_pop_2_and_block_restricted() {
        let m = GaIntGenerator::from_params(&serde_json::json!({})).unwrap().meta();
        assert_eq!(m.min_pop, 2);
        assert!(m.supports_block("int"));
        assert!(!m.supports_block("binary"));
        assert_eq!(m.kind, "gen/ga-int");
    }

    #[test]
    fn ga_int_params_validate() {
        assert!(GaIntGenerator::from_params(&serde_json::json!({"p_c": 1.5})).is_err());
        assert!(GaIntGenerator::from_params(&serde_json::json!({"tournament_k": 0})).is_err());
        assert!(GaIntGenerator::from_params(&serde_json::json!({"p_m": -0.1})).is_err());
        assert!(GaIntGenerator::from_params(&serde_json::json!({"p_m": 1.1})).is_err());
        assert!(GaIntGenerator::from_params(&serde_json::json!({"eta_c": 0.0})).is_err());
        assert!(GaIntGenerator::from_params(&serde_json::json!({"eta_m": -5.0})).is_err());
        let g = GaIntGenerator::from_params(&serde_json::json!({})).unwrap();
        assert_eq!(g.tournament_k, 2);
        assert_eq!(g.p_c, 0.9);
        assert_eq!(g.eta_c, 15.0);
        assert_eq!(g.eta_m, 20.0);
        assert_eq!(g.p_m, None); // resolved lazily against n inside generate()
    }

    /// `gen/ga-int` is an M3-8 typed family (historically parsed only
    /// `p_c`) -- it now ALSO accepts legacy `pc`, resolving identically to
    /// the equivalent `p_c` spec (M3-8 deferral cleanup, `params.rs`).
    #[test]
    fn ga_int_pc_alias_resolves_identically_to_canonical_p_c() {
        let legacy = GaIntGenerator::from_params(&serde_json::json!({"pc": 0.42})).unwrap();
        let canonical = GaIntGenerator::from_params(&serde_json::json!({"p_c": 0.42})).unwrap();
        assert_eq!(legacy.p_c, canonical.p_c);
        assert_eq!(legacy.p_c, 0.42);
    }

    /// When both spellings are present on the same block, canonical `p_c`
    /// wins outright -- `pc`'s value is never used.
    #[test]
    fn ga_int_p_c_wins_over_pc_when_both_present() {
        let g = GaIntGenerator::from_params(&serde_json::json!({"p_c": 0.42, "pc": 0.99})).unwrap();
        assert_eq!(g.p_c, 0.42);
    }

    #[test]
    fn ga_int_generator_validity_property_seeded_batch() {
        let n = 5usize;
        let p = IntQuadratic::new(-10, 10, vec![0; n]);
        let space = p.space();
        let gen = GaIntGenerator::from_params(&serde_json::json!({})).unwrap();
        let mut eval = Evaluator::new(&p, 1_000_000);
        let mut rng = RngStream::from_master(31, &[0]);
        let mut bb = Blackboard::new();
        for _ in 0..100 {
            let pop = pop_n(9, n, -10, 10);
            let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
            let off = gen.generate(&pop, &mut ctx);
            assert_eq!(off.len(), pop.len());
            for ind in &off { assert_eq!(int_values(ind).len(), n); }
        }
    }

    #[test]
    fn ga_int_generator_deterministic_same_seed() {
        let n = 4usize;
        let p = IntQuadratic::new(0, 20, vec![0; n]);
        let space = p.space();
        let gen = GaIntGenerator::from_params(&serde_json::json!({})).unwrap();
        let pop = pop_n(6, n, 0, 20);
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
    fn ga_int_generator_min_pop_below_2_panics() {
        let n = 3usize;
        let p = IntQuadratic::new(0, 5, vec![0; n]);
        let space = p.space();
        let gen = GaIntGenerator::from_params(&serde_json::json!({})).unwrap();
        let pop = pop_n(1, n, 0, 5);
        let mut eval = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut eval, iteration: 0 };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gen.generate(&pop, &mut ctx)
        }));
        assert!(result.is_err(), "gen/ga-int must reject pop_size < 2 at runtime as a backstop");
    }

    // ==================================================================
    // registry / spec-validation (required test 5)
    // ==================================================================

    #[test]
    fn registry_resolves_all_three_kinds() {
        let mut reg = Registry::new();
        register(&mut reg);
        assert!(reg.build_generator("gen/int-sbx", &serde_json::json!({})).is_ok());
        assert!(reg.build_generator("gen/int-pm", &serde_json::json!({})).is_ok());
        assert!(reg.build_generator("gen/ga-int", &serde_json::json!({})).is_ok());
    }

    #[test]
    fn spec_validation_rejects_gen_ga_int_on_non_int_space() {
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
                generator: ComponentSpec { kind: "gen/ga-int".into(), params: serde_json::json!({}) },
                replacer: ComponentSpec { kind: "replace/mu-plus-lambda".into(), params: serde_json::json!({}) },
                adapter: None,
            }],
            termination: TerminationSpec { budget: 100, target: None },
            restart: None,
        };
        let e = spec.validate(&reg, &space);
        match e {
            Err(SpecError::UnsupportedBlock { kind, block }) => {
                assert_eq!(kind, "gen/ga-int");
                assert_eq!(block, "float");
            }
            other => panic!("expected SpecError::UnsupportedBlock, got {other:?}"),
        }
    }

    #[test]
    fn spec_validation_accepts_gen_ga_int_on_int_space() {
        use sezgi_core::spec::{AlgorithmSpec, ComponentSpec, StageSpec, TerminationSpec};
        let mut reg = Registry::new();
        register(&mut reg);
        crate::init::register(&mut reg);
        crate::boundary::register(&mut reg);
        crate::replace::register(&mut reg);
        let space = SearchSpace::new(vec![Block::Int { lo: 0, hi: 100, n: 4 }]).unwrap();
        let spec = AlgorithmSpec {
            name: "t".into(), pop_size: 10,
            init: ComponentSpec { kind: "init/uniform".into(), params: serde_json::json!({}) },
            boundary: ComponentSpec { kind: "boundary/clamp".into(), params: serde_json::json!({}) },
            stages: vec![StageSpec {
                generator: ComponentSpec { kind: "gen/ga-int".into(), params: serde_json::json!({}) },
                replacer: ComponentSpec { kind: "replace/mu-plus-lambda".into(), params: serde_json::json!({}) },
                adapter: None,
            }],
            termination: TerminationSpec { budget: 100, target: None },
            restart: None,
        };
        assert!(spec.validate(&reg, &space).is_ok());
    }

    // ==================================================================
    // IntQuadratic-shaped toy convergence via the spec path AND via
    // presets::ga_int (required test 3)
    // ==================================================================

    #[test]
    fn ga_int_converges_on_int_quadratic_via_engine_spec_path() {
        use sezgi_core::spec::{AlgorithmSpec, ComponentSpec, StageSpec, TerminationSpec};
        use sezgi_core::engine::{Engine, RunConfig};

        let target = vec![7i64, -3, 0, 15, -20];
        let p = IntQuadratic::new(-20, 20, target.clone());
        let mut reg = Registry::new();
        register(&mut reg);
        crate::init::register(&mut reg);
        crate::boundary::register(&mut reg);
        crate::replace::register(&mut reg);

        let spec = AlgorithmSpec {
            name: "ga-int-toy".into(), pop_size: 40,
            init: ComponentSpec { kind: "init/uniform".into(), params: serde_json::json!({}) },
            boundary: ComponentSpec { kind: "boundary/clamp".into(), params: serde_json::json!({}) },
            stages: vec![StageSpec {
                generator: ComponentSpec { kind: "gen/ga-int".into(), params: serde_json::json!({}) },
                replacer: ComponentSpec { kind: "replace/mu-plus-lambda".into(), params: serde_json::json!({}) },
                adapter: None,
            }],
            termination: TerminationSpec { budget: 40_000, target: None },
            restart: None,
        };
        let e = Engine::from_spec(&spec, &reg, p.space()).unwrap();
        let r = e.run(&p, RunConfig { master_seed: 2026, run_id: 0 }, None).unwrap();
        assert!(r.best_f < 5.0,
            "gen/ga-int must drive an IntQuadratic toy near its integer optimum within budget, got {}", r.best_f);
    }

    #[test]
    fn ga_int_converges_on_int_quadratic_via_presets_ga_int() {
        use sezgi_core::engine::{Engine, RunConfig};

        let target = vec![10i64, 10, -10, -10];
        let p = IntQuadratic::new(-15, 15, target.clone());
        let mut reg = Registry::new();
        crate::init::register(&mut reg);
        crate::boundary::register(&mut reg);
        crate::replace::register(&mut reg);
        register(&mut reg);

        let spec = crate::presets::ga_int(40, 40_000);
        let e = Engine::from_spec(&spec, &reg, p.space()).unwrap();
        let r = e.run(&p, RunConfig { master_seed: 77, run_id: 0 }, None).unwrap();
        assert!(r.best_f < 5.0,
            "presets::ga_int must drive an IntQuadratic toy near its integer optimum within budget, got {}", r.best_f);
    }
}
