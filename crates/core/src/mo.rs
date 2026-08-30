//! Multi-objective (MO) core surface.
//!
//! This module is a parallel, purely additive surface alongside the scalar
//! [`crate::problem`] module: it does not modify [`crate::problem::Problem`],
//! [`crate::problem::Evaluator`], [`crate::engine::Engine`], `Ctx`, or
//! [`crate::problem::Population`] in any way. [`MoProblem`]/[`MoEvaluator`]
//! mirror the scalar `Problem`/`Evaluator` conventions (batch evaluation, the
//! all-or-nothing evaluation budget, wrong-length panics) but evaluate to a
//! vector of objectives per individual instead of a single scalar fitness.
//! NSGA-II and the ZDT/DTLZ benchmark suites are intended to build on this
//! surface. There is no scalar "best" tracked here (multi-objective
//! optimization has no single best point), and there is no `EvalObserver`
//! analogue in this first version -- MO-specific logging formats (e.g.
//! IOH/COCO-biobj) are a recorded deferral.

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
}

/// A population paired with its multi-objective evaluations.
///
/// **PINNED once merged**: `individuals` and `objectives` are always the
/// same length; each inner `objectives` vector has length equal to the
/// producing [`MoProblem`]'s [`MoProblem::n_objectives`].
#[derive(Debug, Clone, Default)]
pub struct MoPopulation {
    pub individuals: Vec<Genotype>,
    pub objectives: Vec<Vec<f64>>,
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
