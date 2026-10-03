use crate::space::{Block, BlockValues, Genotype, SearchSpace};

pub trait Problem: Send + Sync {
    fn space(&self) -> &SearchSpace;
    fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<f64>;
    fn optimum(&self) -> Option<f64> { None }
}

#[derive(Debug, Clone, Default)]
pub struct Population {
    pub individuals: Vec<Genotype>,
    pub fitness: Vec<f64>,
}

impl Population {
    pub fn best_index(&self) -> Option<usize> {
        self.fitness.iter().enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
    }
    pub fn len(&self) -> usize { self.individuals.len() }
    pub fn is_empty(&self) -> bool { self.individuals.is_empty() }
}

#[derive(Debug, thiserror::Error)]
#[error("budget exhausted: {used}/{budget}, requested {requested} more")]
pub struct BudgetExhausted { pub used: u64, pub budget: u64, pub requested: u64 }

pub trait EvalObserver: Send {
    fn on_eval(&mut self, eval_index: u64, f: f64, best_so_far: f64);
}

/// `Evaluator` is the sole gateway every charged evaluation in this crate
/// passes through -- the engine's own per-stage/setup/restart calls AND any
/// `Generator`/`Adapter`'s own internal `ctx.eval.evaluate(..)` calls (e.g.
/// `gen/hho`'s dive-trial evaluations) all funnel through this one method.
/// So tracking the best-so-far GENOTYPE here, alongside the scalar fitness,
/// makes `Evaluator` the single source of truth for "the best point observed
/// over every charged evaluation, from anywhere" -- rather than requiring
/// every caller to separately reconstruct that view from whichever subset of
/// evaluations happens to flow through it (see `Engine::run`'s former
/// `global_best`, which only ever saw the engine's own call sites and so
/// missed a generator's internal evaluate calls whose winning point was
/// never itself returned to the engine -- `gen/hho`'s raw dive candidates
/// being the one case in this crate where the returned/re-evaluated point
/// differs from the internally-evaluated one, via boundary repair; M3-1
/// Task 1, unifying `RunResult::best_f`/`best_x` with this tracking).
///
/// **Best-comparison rule (defines `best_f`/`best_x` tie/NaN semantics):**
/// [`Self::evaluate`] updates `best`/`best_x` under strict improvement,
/// `!(b <= f)` where `b` is the current best -- i.e. a candidate replaces the
/// incumbent unless the incumbent already compares `<=` it. Two notable
/// consequences for a user-supplied `Problem` (e.g. `from_callable`) that
/// can return NaN or signed zero -- irrelevant for this project's own
/// BBOB/f0 problems, which never produce NaN (goldens confirm bit-identical
/// output): a NaN fitness compares `false` to everything (`b <= NaN` is
/// always `false`), so `!(b <= f)` is `true` and a NaN fitness is treated as
/// an improvement the first time it is seen -- it can become (and, if
/// nothing subsequently improves on it, stay) the reported `best_f`; and
/// `-0.0`/`+0.0` are `<=`-equal, so an EARLIER zero is always kept over a
/// later one of the other sign (no `total_cmp`-style sign-breaking, unlike
/// the removed engine-side `global_best`'s old comparator).
pub struct Evaluator<'a> {
    problem: &'a dyn Problem,
    budget: u64,
    used: u64,                    // NOT pub: cannot be incremented from outside
    best: Option<f64>,
    best_x: Option<Genotype>,
    observer: Option<Box<dyn EvalObserver>>,
}

impl<'a> Evaluator<'a> {
    pub fn new(problem: &'a dyn Problem, budget: u64) -> Self {
        Self { problem, budget, used: 0, best: None, best_x: None, observer: None }
    }
    pub fn set_observer(&mut self, obs: Box<dyn EvalObserver>) { self.observer = Some(obs); }
    pub fn used(&self) -> u64 { self.used }
    pub fn budget(&self) -> u64 { self.budget }
    pub fn best_so_far(&self) -> Option<f64> { self.best }
    /// The genotype paired with [`Self::best_so_far`]'s fitness value --
    /// updated by the SAME strict-improvement comparison, in the same pass,
    /// over every charged evaluation this `Evaluator` has ever performed.
    pub fn best_x_so_far(&self) -> Option<&Genotype> { self.best_x.as_ref() }
    pub fn problem(&self) -> &dyn Problem { self.problem }

    pub fn evaluate(&mut self, pop: &[Genotype]) -> Result<Vec<f64>, BudgetExhausted> {
        let req = pop.len() as u64;
        if self.used + req > self.budget {
            return Err(BudgetExhausted { used: self.used, budget: self.budget, requested: req });
        }
        // O(blocks) shape guard: block count and per-block kind only (not
        // value length, which would be O(genome)).
        let space = self.problem.space();
        for g in pop {
            if let Err(detail) = shape_check(space, g) {
                panic!("genotype shape mismatch: {detail}");
            }
        }
        let fs = self.problem.evaluate_batch(pop);
        assert_eq!(fs.len(), pop.len(),
            "Problem::evaluate_batch returned wrong length: {} != {}", fs.len(), pop.len());
        for (g, &f) in pop.iter().zip(&fs) {
            self.used += 1;
            let improved = !matches!(self.best, Some(b) if b <= f);
            if improved {
                self.best = Some(f);
                self.best_x = Some(g.clone());
            }
            let best = self.best.expect("just set above, on this or an earlier call");
            if let Some(o) = self.observer.as_mut() { o.on_eval(self.used, f, best); }
        }
        Ok(fs)
    }
}

fn block_kind(b: &Block) -> &'static str {
    match b {
        Block::Float { .. } => "Float",
        Block::Int { .. } => "Int",
        Block::Categorical { .. } => "Categorical",
        Block::Permutation { .. } => "Permutation",
        Block::Binary { .. } => "Binary",
    }
}

fn values_kind(v: &BlockValues) -> &'static str {
    match v {
        BlockValues::Float(_) => "Float",
        BlockValues::Int(_) => "Int",
        BlockValues::Cat(_) => "Categorical",
        BlockValues::Perm(_) => "Permutation",
        BlockValues::Bin(_) => "Binary",
    }
}

/// Block count and per-block kind check, naming expected vs got.
fn shape_check(space: &SearchSpace, g: &Genotype) -> Result<(), String> {
    if g.blocks.len() != space.blocks().len() {
        return Err(format!("expected {} block(s), got {}",
                           space.blocks().len(), g.blocks.len()));
    }
    for (i, (b, v)) in space.blocks().iter().zip(&g.blocks).enumerate() {
        let (want, got) = (block_kind(b), values_kind(v));
        if want != got {
            return Err(format!("block {i}: expected kind {want}, got {got}"));
        }
    }
    Ok(())
}

/// Test/diagnostic problem: the optimum is at the `shift` point (not the center).
pub struct SphereShifted { shift: Vec<f64>, space: SearchSpace }

impl SphereShifted {
    pub fn new(shift: Vec<f64>, lo: f64, hi: f64) -> Self {
        let space = SearchSpace::new(vec![Block::Float { lo, hi, n: shift.len() }]).unwrap();
        Self { shift, space }
    }
}

impl Problem for SphereShifted {
    fn space(&self) -> &SearchSpace { &self.space }
    fn optimum(&self) -> Option<f64> { Some(0.0) }
    fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<f64> {
        pop.iter().map(|g| {
            let BlockValues::Float(xs) = &g.blocks[0] else { return f64::INFINITY };
            xs.iter().zip(&self.shift).map(|(x, s)| (x - s).powi(2)).sum()
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::{BlockValues, Genotype};

    fn g(xs: &[f64]) -> Genotype {
        Genotype { blocks: vec![BlockValues::Float(xs.to_vec())] }
    }

    fn sphere() -> SphereShifted {
        SphereShifted::new(vec![1.0, -2.0], -5.0, 5.0)
    }

    #[test]
    fn sphere_optimum_at_shift() {
        let p = sphere();
        assert_eq!(p.evaluate_batch(&[g(&[1.0, -2.0])]), vec![0.0]);
        assert_eq!(p.optimum(), Some(0.0));
    }

    #[test]
    fn evaluator_counts_and_tracks_best() {
        let p = sphere();
        let mut ev = Evaluator::new(&p, 100);
        ev.evaluate(&[g(&[0.0, 0.0]), g(&[1.0, -2.0])]).unwrap();
        assert_eq!(ev.used(), 2);
        assert_eq!(ev.best_so_far(), Some(0.0));
    }

    #[test]
    #[should_panic(expected = "genotype shape mismatch: expected 1 block(s), got 2")]
    fn evaluator_rejects_block_count_mismatch() {
        let p = sphere();
        let mut ev = Evaluator::new(&p, 10);
        let bad = Genotype { blocks: vec![
            BlockValues::Float(vec![0.0, 0.0]), BlockValues::Bin(vec![true])] };
        let _ = ev.evaluate(&[bad]);
    }

    #[test]
    #[should_panic(expected = "genotype shape mismatch: block 0: expected kind Float, got Binary")]
    fn evaluator_rejects_block_kind_mismatch() {
        let p = sphere();
        let mut ev = Evaluator::new(&p, 10);
        let bad = Genotype { blocks: vec![BlockValues::Bin(vec![true, false])] };
        let _ = ev.evaluate(&[bad]);
    }

    #[test]
    fn budget_exhaustion_is_error() {
        let p = sphere();
        let mut ev = Evaluator::new(&p, 1);
        assert!(ev.evaluate(&[g(&[0.0, 0.0]), g(&[0.0, 0.0])]).is_err());
        assert_eq!(ev.used(), 0, "a batch request exceeding the budget is not evaluated at all");
    }

    #[test]
    fn observer_sees_every_eval() {
        struct Probe(std::sync::Arc<std::sync::Mutex<Vec<u64>>>);
        impl EvalObserver for Probe {
            fn on_eval(&mut self, i: u64, _f: f64, _b: f64) { self.0.lock().unwrap().push(i); }
        }
        let seen = std::sync::Arc::new(std::sync::Mutex::new(vec![]));
        let p = sphere();
        let mut ev = Evaluator::new(&p, 10);
        ev.set_observer(Box::new(Probe(seen.clone())));
        ev.evaluate(&[g(&[0.0, 0.0]), g(&[1.0, 1.0]), g(&[2.0, 2.0])]).unwrap();
        assert_eq!(*seen.lock().unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn population_best_index_is_min() {
        let pop = Population {
            individuals: vec![g(&[0.0, 0.0]), g(&[1.0, -2.0])],
            fitness: vec![5.0, 0.0],
        };
        assert_eq!(pop.best_index(), Some(1));
    }

    struct MisbehavingProblem { space: SearchSpace }
    impl Problem for MisbehavingProblem {
        fn space(&self) -> &SearchSpace { &self.space }
        fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<f64> {
            // Deliberately returns a result of the wrong length (one short).
            pop.iter().skip(1).map(|_| 0.0).collect()
        }
    }

    #[test]
    #[should_panic(expected = "wrong length")]
    fn evaluate_panics_on_wrong_batch_length() {
        let space = SearchSpace::new(vec![Block::Float { lo: -1.0, hi: 1.0, n: 1 }]).unwrap();
        let p = MisbehavingProblem { space };
        let mut ev = Evaluator::new(&p, 10);
        let _ = ev.evaluate(&[g(&[0.0, 0.0]), g(&[0.0, 0.0])]);
    }
}
