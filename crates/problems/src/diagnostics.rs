//! Diagnostic problem trio (M3-8 Task 5): [`OneMax`], [`IntQuadratic`],
//! [`CatMatch`] -- single-objective, single-block problems whose sole
//! purpose is to give the Binary/Int/Categorical typed operator families
//! (`sezgi_components::bin_ops`/`int_ops`/`cat_ops`, M3-8 Tasks 2-4) a
//! REACHABLE target: before this module, those families were fully
//! implemented and unit-tested against inline toy `Problem`s local to their
//! own test modules, but the public API had no way to actually RUN
//! `ga_bin`/`ga_int`/`ga_cat` (or `gen/compound`, [`crate`]'s sibling
//! `sezgi_components::compound`) end to end and watch them converge.
//!
//! ## Honest scope (Task 5 brief, scope ruling 3 -- quoted from the
//! milestone plan's approved ruling)
//!
//! This trio is a **diagnostic, not a benchmark suite**. Each problem is a
//! minimal, hand-verifiable, single-global-optimum landscape (closed-form
//! fitness, no local optima, no published benchmark literature behind it
//! the way `crate::bbob`/`crate::cec2014`/`crate::wfg`/`crate::tsp` carry) --
//! it exists ONLY to make the typed presets reachable and testable, not to
//! serve as a research-grade evaluation target. Do not cite these as
//! benchmark results; they are smoke tests with known answers.
//!
//! ## Design spec §3 (quoted, `docs/superpowers/specs/2026-08-27-sezgi-design.md`)
//!
//! "`SearchSpace = [ Float(lo,hi)×n | Int(lo,hi)×n | Categorical(k)×n |
//! Permutation(n) | Binary(n) ]`. `Genotype` consists of separate typed
//! slices corresponding to these blocks -- the 'everything embedded as
//! float' (mealpy) approach was rejected... Operators declare the block
//! types they support as metadata... In mixed spaces, operators are applied
//! per block (compound operator)." [`OneMax`] targets `Block::Binary`,
//! [`IntQuadratic`] targets `Block::Int`, [`CatMatch`] targets
//! `Block::Categorical` -- one problem per non-Float, non-Permutation block
//! type (Float already has `sezgi_core::problem::SphereShifted`/the BBOB
//! suite; Permutation already has [`crate::tsp::Tsp`]).
//!
//! ## Conventions (mirrors `crate::tsp::Tsp`'s and
//! `sezgi_core::problem::SphereShifted`'s own single-objective shape)
//!
//! A lightweight struct holding a precomputed target/space, a `new`
//! constructor (the brief's PINNED signatures -- consumed verbatim by M3-8
//! Tasks 9/10's Python/R bindings), `Problem::optimum()` returning the known
//! value, and **minimization throughout** -- this crate's universal
//! convention (lower fitness is better: `sezgi_core::problem::Population::
//! best_index`, every typed generator's own tournament selection compares
//! `pop.fitness[c] < pop.fitness[best]`). [`OneMax`] is conventionally
//! stated as a MAXIMIZATION problem (Goldberg 1989, maximize the count of
//! `1`-bits); here it is re-expressed as "minimize the zero-bit count" --
//! the same optimum location (all-`true`), just re-signed to fit this
//! crate's convention, exactly as `crate::tsp::Tsp` and every other
//! `Problem` impl in this workspace already does.
//!
//! Every genotype passed to `evaluate_batch` that does not match the
//! expected single block (wrong `BlockValues` variant, or right variant but
//! wrong length) evaluates to `f64::INFINITY` rather than panicking --
//! the same defensive sentinel convention `crate::tsp::Tsp::evaluate_batch`/
//! `sezgi_core::problem::SphereShifted` already use for a malformed
//! genotype (never expected via the normal engine path, since
//! `SearchSpace::validate` and `boundary/clamp` keep every charged
//! evaluation well-formed; this is a defensive backstop, not a runtime
//! check this module relies on).

use sezgi_core::problem::Problem;
use sezgi_core::rng::RngStream;
use sezgi_core::space::{Block, BlockValues, Genotype, SearchSpace};

/// OneMax (Goldberg 1989): maximize the count of `true` bits in an
/// `n_bits`-length [`Block::Binary`] genotype -- the textbook GA diagnostic.
/// This crate minimizes throughout (module doc), so [`OneMax::evaluate_batch`]
/// returns the ZERO-bit count (`n_bits - popcount`), not the one-bit count
/// directly: "maximize ones" re-expressed as "minimize zeros", same optimum
/// location (all-`true`) either way. Known optimum: `0.0`, at the
/// all-`true` genotype.
pub struct OneMax {
    n_bits: usize,
    space: SearchSpace,
}

impl OneMax {
    /// `n_bits`: the length of the single `Block::Binary` this problem's
    /// space consists of (PINNED constructor shape, Task 5 brief -- T9/T10's
    /// Python/R bindings consume this signature verbatim).
    pub fn new(n_bits: usize) -> Self {
        let space = SearchSpace::new(vec![Block::Binary { n: n_bits }])
            .expect("Block::Binary has no bounds for SearchSpace::new to reject");
        Self { n_bits, space }
    }

    pub fn n_bits(&self) -> usize {
        self.n_bits
    }
}

impl Problem for OneMax {
    fn space(&self) -> &SearchSpace {
        &self.space
    }

    fn optimum(&self) -> Option<f64> {
        Some(0.0)
    }

    fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<f64> {
        pop.iter()
            .map(|g| match g.blocks.first() {
                Some(BlockValues::Bin(bits)) if bits.len() == self.n_bits => {
                    bits.iter().filter(|&&b| !b).count() as f64
                }
                _ => f64::INFINITY,
            })
            .collect()
    }
}

/// Fixed internal seed for [`IntQuadratic`]'s derived target (module doc:
/// deterministic and NOT caller-supplied, unlike [`CatMatch::new`]'s
/// explicit `seed` parameter -- the brief pins `IntQuadratic::new(lo, hi,
/// n)` with no `seed` argument, so every call with the same bounds must
/// reproduce the SAME target without the caller remembering a seed).
/// Arbitrary but fixed -- chosen once, never re-derived; changing it would
/// change every `IntQuadratic` instance's optimum location and is exactly
/// the kind of accidental-break a golden-value regression test (below)
/// exists to catch.
const INT_QUADRATIC_SEED: u64 = 20_260_901;

/// IntQuadratic: sum-of-squared-distance-to-a-fixed-target quadratic on a
/// [`Block::Int`] genotype -- the integer counterpart of
/// `sezgi_core::problem::SphereShifted`'s float shift. The target
/// ([`IntQuadratic::optimum_x`]) is drawn once at construction, one
/// per-dimension uniform draw in `[lo, hi]` each (via
/// [`RngStream::next_below`], no modulo bias), from
/// `RngStream::from_master(INT_QUADRATIC_SEED, &[lo as u64, hi as u64, n as
/// u64])` -- so distinct `(lo, hi, n)` triples get distinct, but each
/// individually reproducible, targets (same seed => same call => same
/// target, always). `f(x) = sum((x_i - target_i)^2)`, minimized (module
/// doc); known optimum `0.0` at `x == target`.
pub struct IntQuadratic {
    target: Vec<i64>,
    space: SearchSpace,
}

impl IntQuadratic {
    /// `lo`/`hi`: inclusive bounds of the single `Block::Int`. `n`: its
    /// length (PINNED constructor shape, Task 5 brief). Panics (via
    /// `SearchSpace::new`) if `lo >= hi`, the same bound-validation
    /// `sezgi_core::problem::SphereShifted::new` relies on for its own
    /// `Block::Float`.
    pub fn new(lo: i64, hi: i64, n: usize) -> Self {
        let space = SearchSpace::new(vec![Block::Int { lo, hi, n }])
            .expect("Block::Int bounds validated by SearchSpace::new (lo must be < hi)");
        let mut rng = RngStream::from_master(INT_QUADRATIC_SEED, &[lo as u64, hi as u64, n as u64]);
        let width = (hi - lo + 1) as u64;
        let target: Vec<i64> = (0..n).map(|_| lo + rng.next_below(width) as i64).collect();
        Self { target, space }
    }

    /// The fixed target [`IntQuadratic::evaluate_batch`] measures distance
    /// to (module doc) -- exposed so callers/tests can construct pinned
    /// genotypes at (or near) the known optimum without re-deriving the RNG
    /// draw themselves.
    pub fn optimum_x(&self) -> &[i64] {
        &self.target
    }
}

impl Problem for IntQuadratic {
    fn space(&self) -> &SearchSpace {
        &self.space
    }

    fn optimum(&self) -> Option<f64> {
        Some(0.0)
    }

    fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<f64> {
        pop.iter()
            .map(|g| match g.blocks.first() {
                Some(BlockValues::Int(xs)) if xs.len() == self.target.len() => xs
                    .iter()
                    .zip(&self.target)
                    .map(|(x, t)| ((x - t) * (x - t)) as f64)
                    .sum(),
                _ => f64::INFINITY,
            })
            .collect()
    }
}

/// CatMatch: Hamming-distance-to-a-target matching problem on a
/// [`Block::Categorical`] genotype (`k` categories per gene, `n` genes) --
/// the categorical counterpart of [`IntQuadratic`]/[`OneMax`]. Unlike
/// [`IntQuadratic::new`]'s fixed internal seed, `CatMatch::new`'s `seed` is
/// an explicit caller-supplied parameter (PINNED constructor shape, Task 5
/// brief: `CatMatch::new(k, n, seed)`) -- the target category vector
/// ([`CatMatch::optimum_x`]) is drawn once at construction, one
/// per-dimension uniform draw in `[0, k)` each, from
/// `RngStream::from_master(seed, &[k as u64, n as u64])`. `f(x)` = the
/// number of genes where `x_i != target_i` (mismatch count), minimized
/// (module doc); known optimum `0.0` at `x == target`.
pub struct CatMatch {
    k: u32,
    target: Vec<u32>,
    space: SearchSpace,
}

impl CatMatch {
    /// `k`: category count per gene. `n`: gene count (the single
    /// `Block::Categorical`'s length). `seed`: the master seed the target
    /// is drawn from (PINNED constructor shape, Task 5 brief).
    pub fn new(k: u32, n: usize, seed: u64) -> Self {
        let space = SearchSpace::new(vec![Block::Categorical { k, n }])
            .expect("Block::Categorical has no bounds for SearchSpace::new to reject");
        let mut rng = RngStream::from_master(seed, &[k as u64, n as u64]);
        let target: Vec<u32> = (0..n).map(|_| rng.next_below(k as u64) as u32).collect();
        Self { k, target, space }
    }

    pub fn k(&self) -> u32 {
        self.k
    }

    /// The fixed target [`CatMatch::evaluate_batch`] counts mismatches
    /// against (module doc) -- exposed for the same reason
    /// [`IntQuadratic::optimum_x`] is.
    pub fn optimum_x(&self) -> &[u32] {
        &self.target
    }
}

impl Problem for CatMatch {
    fn space(&self) -> &SearchSpace {
        &self.space
    }

    fn optimum(&self) -> Option<f64> {
        Some(0.0)
    }

    fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<f64> {
        pop.iter()
            .map(|g| match g.blocks.first() {
                Some(BlockValues::Cat(xs)) if xs.len() == self.target.len() => {
                    xs.iter().zip(&self.target).filter(|(x, t)| x != t).count() as f64
                }
                _ => f64::INFINITY,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::problem::Evaluator;

    fn bin(xs: &[bool]) -> Genotype {
        Genotype { blocks: vec![BlockValues::Bin(xs.to_vec())] }
    }
    fn int(xs: &[i64]) -> Genotype {
        Genotype { blocks: vec![BlockValues::Int(xs.to_vec())] }
    }
    fn cat(xs: &[u32]) -> Genotype {
        Genotype { blocks: vec![BlockValues::Cat(xs.to_vec())] }
    }

    // ==================================================================
    // OneMax -- hand fixtures (task-5 required test 1)
    // ==================================================================

    #[test]
    fn one_max_space_and_optimum() {
        let p = OneMax::new(6);
        assert_eq!(p.n_bits(), 6);
        assert_eq!(p.space().blocks(), &[Block::Binary { n: 6 }]);
        assert_eq!(Problem::optimum(&p), Some(0.0));
    }

    #[test]
    fn one_max_all_true_is_zero() {
        let p = OneMax::new(5);
        assert_eq!(p.evaluate_batch(&[bin(&[true; 5])]), vec![0.0]);
    }

    #[test]
    fn one_max_all_false_is_n_bits() {
        let p = OneMax::new(5);
        assert_eq!(p.evaluate_batch(&[bin(&[false; 5])]), vec![5.0]);
    }

    // Hand fixture: bits = [T,F,T,F,F] -> zero-count = 3 (indices 1,3,4 are
    // false). f = n_bits - popcount = 5 - 2 = 3, equivalently the direct
    // zero-count -- both formulations agree by construction.
    #[test]
    fn one_max_mixed_pattern_hand_fixture() {
        let p = OneMax::new(5);
        let x = bin(&[true, false, true, false, false]);
        assert_eq!(p.evaluate_batch(&[x]), vec![3.0]);
    }

    #[test]
    fn one_max_malformed_genotype_is_infinity() {
        let p = OneMax::new(5);
        assert_eq!(p.evaluate_batch(&[bin(&[true; 4])]), vec![f64::INFINITY]);
        let wrong_type = Genotype { blocks: vec![BlockValues::Int(vec![0; 5])] };
        assert_eq!(p.evaluate_batch(&[wrong_type]), vec![f64::INFINITY]);
    }

    #[test]
    fn one_max_evaluator_integration() {
        let p = OneMax::new(4);
        let mut ev = Evaluator::new(&p, 10);
        ev.evaluate(&[bin(&[true, true, true, true]), bin(&[false, false, false, false])]).unwrap();
        assert_eq!(ev.used(), 2);
        assert_eq!(ev.best_so_far(), Some(0.0));
    }

    // ==================================================================
    // IntQuadratic -- golden target + hand-derived fixtures (task-5
    // required test 1). The target below is a GOLDEN VALUE (rng.rs's own
    // convention: "this test breaks if the implementation changes
    // accidentally") -- obtained by running IntQuadratic::new(-5, 5,
    // 4).optimum_x() once and recording the output, not by hand-simulating
    // xoshiro256++. Every other assertion in this block is then derived
    // algebraically FROM that recorded target, by hand.
    // ==================================================================

    #[test]
    fn int_quadratic_target_is_deterministic_and_in_bounds() {
        let a = IntQuadratic::new(-5, 5, 4);
        let b = IntQuadratic::new(-5, 5, 4);
        assert_eq!(a.optimum_x(), b.optimum_x(), "same (lo,hi,n) must reproduce the same target");
        for &t in a.optimum_x() {
            assert!((-5..=5).contains(&t));
        }
        assert_eq!(a.optimum_x().len(), 4);
    }

    #[test]
    fn int_quadratic_golden_target_for_pinned_bounds() {
        // Golden (recorded once via IntQuadratic::new(-5, 5, 4).optimum_x(),
        // INT_QUADRATIC_SEED fixed): [0, -2, 2, 0].
        let p = IntQuadratic::new(-5, 5, 4);
        assert_eq!(p.optimum_x(), &[0, -2, 2, 0]);
    }

    #[test]
    fn int_quadratic_at_target_is_zero() {
        let p = IntQuadratic::new(-5, 5, 4);
        let target = p.optimum_x().to_vec();
        assert_eq!(p.evaluate_batch(&[int(&target)]), vec![0.0]);
        assert_eq!(Problem::optimum(&p), Some(0.0));
    }

    // Hand derivation from the golden target [0, -2, 2, 0]: probe x = [0,
    // 0, 0, 0]. f = (0-0)^2 + (0-(-2))^2 + (0-2)^2 + (0-0)^2
    //   = 0 + 4 + 4 + 0 = 8.
    #[test]
    fn int_quadratic_zero_vector_hand_fixture() {
        let p = IntQuadratic::new(-5, 5, 4);
        assert_eq!(p.evaluate_batch(&[int(&[0, 0, 0, 0])]), vec![8.0]);
    }

    // Hand derivation from the golden target [0, -2, 2, 0]: probe
    // x = [-5, 5, -5, 5] (the bound corners). f = (-5-0)^2 + (5-(-2))^2 +
    // (-5-2)^2 + (5-0)^2 = 25 + 49 + 49 + 25 = 148.
    #[test]
    fn int_quadratic_corner_vector_hand_fixture() {
        let p = IntQuadratic::new(-5, 5, 4);
        assert_eq!(p.evaluate_batch(&[int(&[-5, 5, -5, 5])]), vec![148.0]);
    }

    #[test]
    fn int_quadratic_different_bounds_stay_in_their_own_range() {
        // A different (lo, hi, n) triple is a different RngStream path
        // (module doc: path includes lo/hi/n) -- exercised here for a
        // wider, positive-only range to confirm the derivation isn't an
        // artifact of the pinned (-5, 5, 4) golden's symmetric bounds.
        let p = IntQuadratic::new(0, 20, 3);
        assert_eq!(p.optimum_x().len(), 3);
        for &t in p.optimum_x() {
            assert!((0..=20).contains(&t));
        }
    }

    #[test]
    fn int_quadratic_malformed_genotype_is_infinity() {
        let p = IntQuadratic::new(-5, 5, 4);
        assert_eq!(p.evaluate_batch(&[int(&[0, 0, 0])]), vec![f64::INFINITY]);
        let wrong_type = Genotype { blocks: vec![BlockValues::Float(vec![0.0; 4])] };
        assert_eq!(p.evaluate_batch(&[wrong_type]), vec![f64::INFINITY]);
    }

    #[test]
    #[should_panic]
    fn int_quadratic_invalid_bounds_panics() {
        IntQuadratic::new(5, 5, 3);
    }

    // ==================================================================
    // CatMatch -- golden target + hand-derived fixtures (task-5 required
    // test 1). Same golden-value convention as IntQuadratic above: the
    // target for (k=4, n=5, seed=777) was recorded once via
    // CatMatch::new(4, 5, 777).optimum_x(), then every fixture below is
    // derived by hand from that recorded value.
    // ==================================================================

    #[test]
    fn cat_match_target_is_deterministic_and_in_bounds() {
        let a = CatMatch::new(4, 5, 777);
        let b = CatMatch::new(4, 5, 777);
        assert_eq!(a.optimum_x(), b.optimum_x(), "same (k,n,seed) must reproduce the same target");
        assert_eq!(a.k(), 4);
        for &t in a.optimum_x() {
            assert!(t < 4);
        }
        assert_eq!(a.optimum_x().len(), 5);
    }

    #[test]
    fn cat_match_different_seed_diverges() {
        let a = CatMatch::new(4, 5, 777);
        let b = CatMatch::new(4, 5, 778);
        assert_ne!(a.optimum_x(), b.optimum_x());
    }

    #[test]
    fn cat_match_golden_target_for_pinned_seed() {
        // Golden (recorded once via CatMatch::new(4, 5, 777).optimum_x()):
        // [2, 3, 1, 0, 2].
        let p = CatMatch::new(4, 5, 777);
        assert_eq!(p.optimum_x(), &[2, 3, 1, 0, 2]);
    }

    #[test]
    fn cat_match_at_target_is_zero() {
        let p = CatMatch::new(4, 5, 777);
        let target = p.optimum_x().to_vec();
        assert_eq!(p.evaluate_batch(&[cat(&target)]), vec![0.0]);
        assert_eq!(Problem::optimum(&p), Some(0.0));
    }

    // Hand derivation from the golden target [2, 3, 1, 0, 2]: probe
    // x = [0, 0, 0, 0, 0]. Mismatches at indices 0 (2!=0), 1 (3!=0), 2
    // (1!=0), 4 (2!=0); index 3 matches (0==0). f = 4.
    #[test]
    fn cat_match_all_zero_hand_fixture() {
        let p = CatMatch::new(4, 5, 777);
        assert_eq!(p.evaluate_batch(&[cat(&[0, 0, 0, 0, 0])]), vec![4.0]);
    }

    // Hand derivation from the golden target [2, 3, 1, 0, 2]: probe
    // x = [2, 1, 1, 0, 0] -- matches at indices 0 (2==2), 2 (1==1), 3
    // (0==0); mismatches at indices 1 (1!=3), 4 (0!=2). f = 2.
    #[test]
    fn cat_match_partial_overlap_hand_fixture() {
        let p = CatMatch::new(4, 5, 777);
        assert_eq!(p.evaluate_batch(&[cat(&[2, 1, 1, 0, 0])]), vec![2.0]);
    }

    #[test]
    fn cat_match_malformed_genotype_is_infinity() {
        let p = CatMatch::new(4, 5, 777);
        assert_eq!(p.evaluate_batch(&[cat(&[0, 0, 0])]), vec![f64::INFINITY]);
        let wrong_type = Genotype { blocks: vec![BlockValues::Int(vec![0; 5])] };
        assert_eq!(p.evaluate_batch(&[wrong_type]), vec![f64::INFINITY]);
    }
}
