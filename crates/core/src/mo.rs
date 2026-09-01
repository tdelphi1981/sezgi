//! Multi-objective (MO) core surface.
//!
//! This module is a parallel, purely additive surface alongside the scalar
//! [`crate::problem`] module: it does not modify [`crate::problem::Problem`],
//! [`crate::problem::Evaluator`], [`crate::engine::Engine`], `Ctx`, or
//! [`crate::problem::Population`] in any way. [`MoProblem`]/[`MoEvaluator`]
//! mirror the scalar `Problem`/`Evaluator` conventions (batch evaluation, the
//! all-or-nothing evaluation budget, wrong-length panics) but evaluate to a
//! vector of objectives per individual instead of a single scalar fitness.
//! NSGA-II and the ZDT/ZDT5/DTLZ/WFG benchmark suites build on this
//! surface. There is no scalar "best" tracked here (multi-objective
//! optimization has no single best point). MO run logging, once a recorded
//! deferral, is closed as of M3-7: NSGA-II exposes a batch observer hook and
//! `sezgi-bench`'s `mo_archive` writes the archive-first "sezgi-moa v1"
//! format (see docs/DECISIONS.md's M3-7 record).

use crate::problem::BudgetExhausted;
use crate::space::{Genotype, SearchSpace};

/// A multi-objective optimization problem. All objectives are minimized.
///
/// **PINNED once merged**: this trait's method signatures are a contract
/// that NSGA-II, the ZDT/DTLZ suites, and IGD-based tests build on.
pub trait MoProblem: Send + Sync {
    fn space(&self) -> &SearchSpace;
    fn n_objectives(&self) -> usize;
    /// One inner `Vec` per individual, each of length [`Self::n_objectives`].
    /// All objectives are minimized.
    fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<Vec<f64>>;
    /// A deterministic n-point sample of the analytic Pareto front in
    /// OBJECTIVE space, if known. Used by IGD and tests.
    fn pareto_front(&self, n: usize) -> Option<Vec<Vec<f64>>> { let _ = n; None }

    /// Optional per-individual inequality-constraint values: one inner
    /// `Vec` per individual in `pop` (rows, same order/length as `pop`),
    /// each of length equal to the problem's own constraint count `ncon`
    /// (cols = `g_1..g_ncon`). **Convention: `g_j >= 0` means constraint
    /// `j` is SATISFIED** -- the Deb/Thiele/Laumanns/Zitzler 2005 DTLZ book
    /// chapter's own convention (`docs/DECISIONS.md`'s M3-7 record), also
    /// the convention `sezgi_components::nsga2`'s constrained-domination
    /// support is pinned against (see that module's doc for the KanGAL
    /// `nsga2r.c` provenance: `constr[j] < 0.0` is the C's own violated-row
    /// test, i.e. the SAME sign convention, `g_j >= 0` feasible).
    ///
    /// Default `None`: an unconstrained problem (the overwhelming
    /// majority -- every M3-2 `MoProblem` impl, `Zdt`/`Dtlz` 1-7, is
    /// unconstrained) never overrides this method, and every constrained
    /// consumer (`nsga2_run`) takes the byte-identical unconstrained code
    /// path whenever it returns `None`.
    fn evaluate_constraints_batch(&self, pop: &[Genotype]) -> Option<Vec<Vec<f64>>> {
        let _ = pop;
        None
    }
}

/// A population paired with its multi-objective evaluations.
///
/// **PINNED once merged**: `individuals` and `objectives` are always the
/// same length; each inner `objectives` vector has length equal to the
/// producing [`MoProblem`]'s [`MoProblem::n_objectives`].
///
/// `constraints`/`violations` mirror how `objectives` is stored, added for
/// the constraint channel (see [`MoProblem::evaluate_constraints_batch`]):
/// both are `Some` together and `None` together, never mixed -- `Some` iff
/// the producing problem's `evaluate_constraints_batch` returned `Some`.
/// When `Some`, `constraints` is one row per individual (same convention as
/// `evaluate_constraints_batch`: `g_j >= 0` feasible) and `violations` is
/// one scalar per individual, `<= 0.0` (`0.0` = fully feasible) -- see
/// `sezgi_components::nsga2`'s module doc for the exact accumulation
/// formula this crate's consumers pin (KanGAL `eval.c`).
#[derive(Debug, Clone, Default)]
pub struct MoPopulation {
    pub individuals: Vec<Genotype>,
    pub objectives: Vec<Vec<f64>>,
    pub constraints: Option<Vec<Vec<f64>>>,
    pub violations: Option<Vec<f64>>,
}

impl MoPopulation {
    pub fn len(&self) -> usize { self.individuals.len() }
    pub fn is_empty(&self) -> bool { self.individuals.is_empty() }
}

/// The sole gateway every charged multi-objective evaluation passes through,
/// mirroring [`crate::problem::Evaluator`]. There is no scalar best tracked
/// here (there is no scalar best in MO), and no observer hook in this first
/// version.
///
/// **PINNED once merged.**
pub struct MoEvaluator<'a> {
    problem: &'a dyn MoProblem,
    budget: u64,
    used: u64, // NOT pub: cannot be incremented from outside
}

impl<'a> MoEvaluator<'a> {
    pub fn new(problem: &'a dyn MoProblem, budget: u64) -> Self {
        Self { problem, budget, used: 0 }
    }
    pub fn used(&self) -> u64 { self.used }
    pub fn budget(&self) -> u64 { self.budget }
    pub fn problem(&self) -> &dyn MoProblem { self.problem }

    /// All-or-nothing budget rule, IDENTICAL to the scalar `Evaluator`: a
    /// batch that would exceed the budget is refused entirely
    /// (`Err(BudgetExhausted)`), and `used` does not move.
    pub fn evaluate(&mut self, pop: &[Genotype]) -> Result<Vec<Vec<f64>>, BudgetExhausted> {
        let req = pop.len() as u64;
        if self.used + req > self.budget {
            return Err(BudgetExhausted { used: self.used, budget: self.budget, requested: req });
        }
        let fs = self.problem.evaluate_batch(pop);
        assert_eq!(fs.len(), pop.len(),
            "MoProblem::evaluate_batch returned wrong length: {} != {}", fs.len(), pop.len());
        let n_obj = self.problem.n_objectives();
        for row in &fs {
            assert_eq!(row.len(), n_obj,
                "MoProblem::evaluate_batch returned a row of wrong length: {} != {}", row.len(), n_obj);
        }
        self.used += req;
        Ok(fs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::{Block, BlockValues};

    fn g(xs: &[f64]) -> Genotype {
        Genotype { blocks: vec![BlockValues::Float(xs.to_vec())] }
    }

    /// f1 = x0^2, f2 = (x0-2)^2 on [-5, 5]: a tiny 2-objective test problem.
    struct TwoObj { space: SearchSpace }

    impl TwoObj {
        fn new() -> Self {
            let space = SearchSpace::new(vec![Block::Float { lo: -5.0, hi: 5.0, n: 1 }]).unwrap();
            Self { space }
        }
    }

    impl MoProblem for TwoObj {
        fn space(&self) -> &SearchSpace { &self.space }
        fn n_objectives(&self) -> usize { 2 }
        fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<Vec<f64>> {
            pop.iter().map(|g| {
                let BlockValues::Float(xs) = &g.blocks[0] else { return vec![f64::INFINITY; 2] };
                let x0 = xs[0];
                vec![x0.powi(2), (x0 - 2.0).powi(2)]
            }).collect()
        }
    }

    #[test]
    fn two_obj_evaluates_both_objectives() {
        let p = TwoObj::new();
        assert_eq!(p.evaluate_batch(&[g(&[0.0])]), vec![vec![0.0, 4.0]]);
        assert_eq!(p.evaluate_batch(&[g(&[2.0])]), vec![vec![4.0, 0.0]]);
        assert_eq!(p.n_objectives(), 2);
    }

    // ---- evaluate_constraints_batch: default None -----------------------

    #[test]
    fn evaluate_constraints_batch_default_is_none() {
        // TwoObj never overrides evaluate_constraints_batch -- must inherit
        // the trait default (None), the unconstrained-problem byte-identical
        // path every M3-2 MoProblem impl (Zdt, Dtlz) relies on.
        let p = TwoObj::new();
        assert_eq!(p.evaluate_constraints_batch(&[g(&[0.0]), g(&[1.0])]), None);
    }

    // ---- evaluate_constraints_batch: an overriding constrained problem ---

    /// f1 = x0, f2 = -x0 (both minimized) on [-5, 5], ONE constraint
    /// g0 = x0 - 1 (feasible region x0 >= 1, per the g_j >= 0 convention).
    struct OneConstraint { space: SearchSpace }
    impl OneConstraint {
        fn new() -> Self {
            let space = SearchSpace::new(vec![Block::Float { lo: -5.0, hi: 5.0, n: 1 }]).unwrap();
            Self { space }
        }
    }
    impl MoProblem for OneConstraint {
        fn space(&self) -> &SearchSpace { &self.space }
        fn n_objectives(&self) -> usize { 2 }
        fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<Vec<f64>> {
            pop.iter().map(|g| {
                let BlockValues::Float(xs) = &g.blocks[0] else { return vec![f64::INFINITY; 2] };
                vec![xs[0], -xs[0]]
            }).collect()
        }
        fn evaluate_constraints_batch(&self, pop: &[Genotype]) -> Option<Vec<Vec<f64>>> {
            Some(pop.iter().map(|g| {
                let BlockValues::Float(xs) = &g.blocks[0] else { return vec![f64::NEG_INFINITY] };
                vec![xs[0] - 1.0]
            }).collect())
        }
    }

    #[test]
    fn evaluate_constraints_batch_override_returns_rows_matching_pop() {
        let p = OneConstraint::new();
        let rows = p.evaluate_constraints_batch(&[g(&[0.0]), g(&[1.0]), g(&[3.0])]).unwrap();
        assert_eq!(rows, vec![vec![-1.0], vec![0.0], vec![2.0]]);
    }

    // ---- MoPopulation: constraints/violations fields ---------------------

    #[test]
    fn mo_population_default_has_no_constraints_or_violations() {
        let pop = MoPopulation::default();
        assert_eq!(pop.constraints, None);
        assert_eq!(pop.violations, None);
        assert!(pop.is_empty());
    }

    #[test]
    fn mo_population_carries_constraints_and_violations_alongside_objectives() {
        let pop = MoPopulation {
            individuals: vec![g(&[0.0]), g(&[1.0])],
            objectives: vec![vec![0.0, 0.0], vec![1.0, -1.0]],
            constraints: Some(vec![vec![-1.0], vec![0.0]]),
            violations: Some(vec![-1.0, 0.0]),
        };
        assert_eq!(pop.len(), 2);
        assert_eq!(pop.constraints, Some(vec![vec![-1.0], vec![0.0]]));
        assert_eq!(pop.violations, Some(vec![-1.0, 0.0]));
    }

    #[test]
    fn evaluator_counts_batch_size() {
        let p = TwoObj::new();
        let mut ev = MoEvaluator::new(&p, 100);
        let out = ev.evaluate(&[g(&[0.0]), g(&[1.0]), g(&[2.0])]).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(ev.used(), 3);
        assert_eq!(ev.budget(), 100);
    }

    #[test]
    fn budget_exhaustion_is_error() {
        let p = TwoObj::new();
        let mut ev = MoEvaluator::new(&p, 1);
        assert!(ev.evaluate(&[g(&[0.0]), g(&[0.0])]).is_err());
        assert_eq!(ev.used(), 0, "a batch request exceeding the budget is not evaluated at all");
    }

    struct MisbehavingOuter { space: SearchSpace }
    impl MoProblem for MisbehavingOuter {
        fn space(&self) -> &SearchSpace { &self.space }
        fn n_objectives(&self) -> usize { 2 }
        fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<Vec<f64>> {
            // Deliberately returns one fewer row than individuals.
            pop.iter().skip(1).map(|_| vec![0.0, 0.0]).collect()
        }
    }

    #[test]
    #[should_panic(expected = "wrong length")]
    fn evaluate_panics_on_wrong_outer_length() {
        let space = SearchSpace::new(vec![Block::Float { lo: -1.0, hi: 1.0, n: 1 }]).unwrap();
        let p = MisbehavingOuter { space };
        let mut ev = MoEvaluator::new(&p, 10);
        let _ = ev.evaluate(&[g(&[0.0]), g(&[0.0])]);
    }

    struct MisbehavingInner { space: SearchSpace }
    impl MoProblem for MisbehavingInner {
        fn space(&self) -> &SearchSpace { &self.space }
        fn n_objectives(&self) -> usize { 2 }
        fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<Vec<f64>> {
            // Deliberately returns rows of the wrong inner length (1 instead of 2).
            pop.iter().map(|_| vec![0.0]).collect()
        }
    }

    #[test]
    #[should_panic(expected = "wrong length")]
    fn evaluate_panics_on_wrong_inner_length() {
        let space = SearchSpace::new(vec![Block::Float { lo: -1.0, hi: 1.0, n: 1 }]).unwrap();
        let p = MisbehavingInner { space };
        let mut ev = MoEvaluator::new(&p, 10);
        let _ = ev.evaluate(&[g(&[0.0])]);
    }
}
