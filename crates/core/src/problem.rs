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
#[error("bütçe tükendi: {used}/{budget}, istenen ek {requested}")]
pub struct BudgetExhausted { pub used: u64, pub budget: u64, pub requested: u64 }

pub trait EvalObserver: Send {
    fn on_eval(&mut self, eval_index: u64, f: f64, best_so_far: f64);
}

pub struct Evaluator<'a> {
    problem: &'a dyn Problem,
    budget: u64,
    used: u64,                    // pub DEĞİL: dışarıdan artırılamaz
    best: Option<f64>,
    observer: Option<Box<dyn EvalObserver>>,
}

impl<'a> Evaluator<'a> {
    pub fn new(problem: &'a dyn Problem, budget: u64) -> Self {
        Self { problem, budget, used: 0, best: None, observer: None }
    }
    pub fn set_observer(&mut self, obs: Box<dyn EvalObserver>) { self.observer = Some(obs); }
    pub fn used(&self) -> u64 { self.used }
    pub fn budget(&self) -> u64 { self.budget }
    pub fn best_so_far(&self) -> Option<f64> { self.best }
    pub fn problem(&self) -> &dyn Problem { self.problem }

    pub fn evaluate(&mut self, pop: &[Genotype]) -> Result<Vec<f64>, BudgetExhausted> {
        let req = pop.len() as u64;
        if self.used + req > self.budget {
            return Err(BudgetExhausted { used: self.used, budget: self.budget, requested: req });
        }
        let fs = self.problem.evaluate_batch(pop);
        assert_eq!(fs.len(), pop.len(),
            "Problem::evaluate_batch yanlış uzunluk döndürdü: {} != {}", fs.len(), pop.len());
        for &f in &fs {
            self.used += 1;
            let best = match self.best {
                Some(b) if b <= f => b,
                _ => f,
            };
            self.best = Some(best);
            if let Some(o) = self.observer.as_mut() { o.on_eval(self.used, f, best); }
        }
        Ok(fs)
    }
}

/// Test/tanı problemi: optimum `shift` noktasında (merkezde değil).
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
    fn budget_exhaustion_is_error() {
        let p = sphere();
        let mut ev = Evaluator::new(&p, 1);
        assert!(ev.evaluate(&[g(&[0.0, 0.0]), g(&[0.0, 0.0])]).is_err());
        assert_eq!(ev.used(), 0, "bütçeyi aşan toplu istek hiç değerlendirilmez");
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
            // Kasıtlı olarak yanlış uzunlukta sonuç döndürür (bir eksik).
            pop.iter().skip(1).map(|_| 0.0).collect()
        }
    }

    #[test]
    #[should_panic(expected = "yanlış uzunluk")]
    fn evaluate_panics_on_wrong_batch_length() {
        let space = SearchSpace::new(vec![Block::Float { lo: -1.0, hi: 1.0, n: 1 }]).unwrap();
        let p = MisbehavingProblem { space };
        let mut ev = Evaluator::new(&p, 10);
        let _ = ev.evaluate(&[g(&[0.0, 0.0]), g(&[0.0, 0.0])]);
    }
}
