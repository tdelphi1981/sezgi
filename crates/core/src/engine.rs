use crate::component::{BoundaryHandler, Ctx, Generator, Initializer, Registry, Replacer};
use crate::problem::{EvalObserver, Evaluator, Population, Problem};
use crate::rng::RngStream;
use crate::space::{Genotype, SearchSpace};
use crate::spec::{AlgorithmSpec, SpecError};
use crate::state::Blackboard;

pub struct Engine {
    init: Box<dyn Initializer>,
    boundary: Box<dyn BoundaryHandler>,
    stages: Vec<(Box<dyn Generator>, Box<dyn Replacer>)>,
    pop_size: usize,
    budget: u64,
    target: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct RunConfig { pub master_seed: u64, pub run_id: u64 }

#[derive(Debug, Clone)]
pub struct RunResult {
    pub best_f: f64,
    pub best_x: Genotype,
    pub evals_used: u64,
    pub iterations: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error(transparent)]
    Spec(#[from] SpecError),
    #[error("başlatıcı boş popülasyon üretti")]
    EmptyPopulation,
    #[error("bütçe ({budget}) popülasyondan ({pop_size}) küçük")]
    BudgetSmallerThanPopulation { budget: u64, pop_size: usize },
}

impl Engine {
    pub fn from_spec(spec: &AlgorithmSpec, reg: &Registry, space: &SearchSpace)
        -> Result<Self, SpecError> {
        spec.validate(reg, space)?;
        Ok(Self {
            init: reg.build_initializer(&spec.init.kind, &spec.init.params)?,
            boundary: reg.build_boundary(&spec.boundary.kind, &spec.boundary.params)?,
            stages: spec.stages.iter().map(|st| Ok((
                reg.build_generator(&st.generator.kind, &st.generator.params)?,
                reg.build_replacer(&st.replacer.kind, &st.replacer.params)?,
            ))).collect::<Result<_, SpecError>>()?,
            pop_size: spec.pop_size,
            budget: spec.termination.budget,
            target: spec.termination.target,
        })
    }

    pub fn run(&self, problem: &dyn Problem, cfg: RunConfig,
               observer: Option<Box<dyn EvalObserver>>) -> Result<RunResult, EngineError> {
        let space = problem.space();
        let mut eval = Evaluator::new(problem, self.budget);
        if let Some(obs) = observer { eval.set_observer(obs); }
        let mut bb = Blackboard::new();
        let mut iterations = 0u64;

        // RNG akış yolu sözleşmesi — manifest bu yolları kaydeder (Task 16)
        let mut init_rng = RngStream::from_master(cfg.master_seed, &[cfg.run_id, 0]);
        // NOT: 500+ stage'de replacer yoluyla çakışır (2+2*499=1000); motor yolu ileride ayrılacaksa ofseti büyüt.
        let mut boundary_rng = RngStream::from_master(cfg.master_seed, &[cfg.run_id, 1000]);
        let mut stage_rngs: Vec<(RngStream, RngStream)> = (0..self.stages.len()).map(|i| (
            RngStream::from_master(cfg.master_seed, &[cfg.run_id, 1 + 2 * i as u64]),
            RngStream::from_master(cfg.master_seed, &[cfg.run_id, 2 + 2 * i as u64]),
        )).collect();

        // İlklendirme
        let individuals = {
            let mut ctx = Ctx { space, rng: &mut init_rng, bb: &mut bb,
                                eval: &mut eval, iteration: 0 };
            self.init.initialize(self.pop_size, &mut ctx)
        };
        if individuals.is_empty() { return Err(EngineError::EmptyPopulation); }
        let fitness = match eval.evaluate(&individuals) {
            Ok(f) => f,
            Err(_) => return Err(EngineError::BudgetSmallerThanPopulation {
                budget: self.budget,
                pop_size: self.pop_size,
            }),
        };
        let mut pop = Population { individuals, fitness };

        // Küresel en iyi: popülasyon değiştirici elitist olmasa bile
        // şimdiye kadar değerlendirilen en iyi (fitness, genotip) çiftini izler.
        let mut global_best: Option<(f64, Genotype)> = None;
        let update_global_best = |gb: &mut Option<(f64, Genotype)>, xs: &[Genotype], fs: &[f64]| {
            for (g, &f) in xs.iter().zip(fs) {
                let better = match gb {
                    Some((bf, _)) => f.total_cmp(bf) == std::cmp::Ordering::Less,
                    None => true,
                };
                if better { *gb = Some((f, g.clone())); }
            }
        };
        update_global_best(&mut global_best, &pop.individuals, &pop.fitness);

        let reached = |eval: &Evaluator| -> bool {
            matches!((self.target, eval.best_so_far()),
                     (Some(t), Some(b)) if b <= t)
        };

        'outer: while !reached(&eval) {
            for (si, (gen, rep)) in self.stages.iter().enumerate() {
                let mut offspring = {
                    let (gr, _) = &mut stage_rngs[si];
                    let mut ctx = Ctx { space, rng: gr, bb: &mut bb,
                                        eval: &mut eval, iteration: iterations };
                    gen.generate(&pop, &mut ctx)
                };
                for g in &mut offspring {
                    let mut ctx = Ctx { space, rng: &mut boundary_rng, bb: &mut bb,
                                        eval: &mut eval, iteration: iterations };
                    self.boundary.repair(g, space, &mut ctx);
                }
                let off_fit = match eval.evaluate(&offspring) {
                    Ok(f) => f,
                    Err(_) => break 'outer, // bütçe doldu: temiz çıkış
                };
                update_global_best(&mut global_best, &offspring, &off_fit);
                {
                    let (_, rr) = &mut stage_rngs[si];
                    let mut ctx = Ctx { space, rng: rr, bb: &mut bb,
                                        eval: &mut eval, iteration: iterations };
                    rep.replace(&mut pop, offspring, off_fit, &mut ctx);
                }
                if reached(&eval) { break 'outer; }
            }
            iterations += 1;
        }

        let (best_f, best_x) = global_best.expect("en az ilklendirme değerlendirilmiş olmalı");
        Ok(RunResult {
            best_f,
            best_x,
            evals_used: eval.used(),
            iterations,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::*;
    use crate::problem::{EvalObserver, Population, SphereShifted};
    use crate::space::{Block, BlockValues, Genotype, SearchSpace};
    use crate::spec::*;

    fn uniform_sample(space: &SearchSpace, rng: &mut crate::rng::RngStream) -> Genotype {
        let blocks = space.blocks().iter().map(|b| match *b {
            Block::Float { lo, hi, n } =>
                BlockValues::Float((0..n).map(|_| lo + (hi - lo) * rng.next_f64()).collect()),
            _ => unreachable!("bu test yalnız float kullanır"),
        }).collect();
        Genotype { blocks }
    }

    struct UInit;
    impl Initializer for UInit {
        fn initialize(&self, n: usize, ctx: &mut Ctx) -> Vec<Genotype> {
            (0..n).map(|_| uniform_sample(ctx.space, ctx.rng)).collect()
        }
        fn meta(&self) -> ComponentMeta {
            ComponentMeta { kind: "u-init", supported_blocks: SupportedBlocks::All,
                requires: vec![], provides: vec![] }
        }
    }
    struct Resample;
    impl Generator for Resample {
        fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
            (0..pop.len()).map(|_| uniform_sample(ctx.space, ctx.rng)).collect()
        }
        fn meta(&self) -> ComponentMeta {
            ComponentMeta { kind: "resample", supported_blocks: SupportedBlocks::All,
                requires: vec![], provides: vec![] }
        }
    }
    struct Greedy;
    impl Replacer for Greedy {
        fn replace(&self, pop: &mut Population, off_i: Vec<Genotype>,
                   off_f: Vec<f64>, _c: &mut Ctx) {
            for (i, (gi, fi)) in off_i.into_iter().zip(off_f).enumerate() {
                if fi < pop.fitness[i] { pop.individuals[i] = gi; pop.fitness[i] = fi; }
            }
        }
        fn meta(&self) -> ComponentMeta {
            ComponentMeta { kind: "greedy", supported_blocks: SupportedBlocks::All,
                requires: vec![], provides: vec![] }
        }
    }
    struct NoB;
    impl BoundaryHandler for NoB {
        fn repair(&self, _g: &mut Genotype, _s: &SearchSpace, _c: &mut Ctx) {}
        fn meta(&self) -> ComponentMeta {
            ComponentMeta { kind: "no-b", supported_blocks: SupportedBlocks::All,
                requires: vec![], provides: vec![] }
        }
    }

    fn setup() -> (Registry, AlgorithmSpec, SphereShifted) {
        let mut reg = Registry::new();
        reg.register_initializer("u-init", |_| Ok(Box::new(UInit)));
        reg.register_generator("resample", |_| Ok(Box::new(Resample)));
        reg.register_replacer("greedy", |_| Ok(Box::new(Greedy)));
        reg.register_boundary("no-b", |_| Ok(Box::new(NoB)));
        let spec = AlgorithmSpec {
            name: "rastgele-arama".into(), pop_size: 10,
            init: ComponentSpec { kind: "u-init".into(), params: serde_json::json!({}) },
            boundary: ComponentSpec { kind: "no-b".into(), params: serde_json::json!({}) },
            stages: vec![StageSpec {
                generator: ComponentSpec { kind: "resample".into(), params: serde_json::json!({}) },
                replacer: ComponentSpec { kind: "greedy".into(), params: serde_json::json!({}) },
            }],
            termination: TerminationSpec { budget: 500, target: None },
        };
        (reg, spec, SphereShifted::new(vec![1.0, -2.0], -5.0, 5.0))
    }

    #[test]
    fn respects_budget_exactly() {
        let (reg, spec, p) = setup();
        let e = Engine::from_spec(&spec, &reg, p.space()).unwrap();
        let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
        assert!(r.evals_used <= 500);
        assert!(r.evals_used >= 500 - spec.pop_size as u64, "bütçenin tamamına yakınını kullanmalı");
    }

    #[test]
    fn deterministic_given_seed() {
        let (reg, spec, p) = setup();
        let e = Engine::from_spec(&spec, &reg, p.space()).unwrap();
        let r1 = e.run(&p, RunConfig { master_seed: 7, run_id: 3 }, None).unwrap();
        let r2 = e.run(&p, RunConfig { master_seed: 7, run_id: 3 }, None).unwrap();
        assert_eq!(r1.best_f, r2.best_f);
        assert_eq!(r1.best_x, r2.best_x);
    }

    #[test]
    fn different_run_ids_differ() {
        let (reg, spec, p) = setup();
        let e = Engine::from_spec(&spec, &reg, p.space()).unwrap();
        let r1 = e.run(&p, RunConfig { master_seed: 7, run_id: 0 }, None).unwrap();
        let r2 = e.run(&p, RunConfig { master_seed: 7, run_id: 1 }, None).unwrap();
        assert_ne!(r1.best_f, r2.best_f);
    }

    #[test]
    fn target_stops_early() {
        let (reg, mut spec, p) = setup();
        spec.termination.target = Some(1e9); // her değer hedefi sağlar
        let e = Engine::from_spec(&spec, &reg, p.space()).unwrap();
        let r = e.run(&p, RunConfig { master_seed: 7, run_id: 0 }, None).unwrap();
        assert_eq!(r.evals_used, spec.pop_size as u64, "init sonrası durmalı");
    }

    #[test]
    fn random_search_improves_over_init() {
        let (reg, spec, p) = setup();
        let e = Engine::from_spec(&spec, &reg, p.space()).unwrap();
        let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
        assert!(r.best_f < 5.0, "500 değerlendirmede sphere'de makul ilerleme: {}", r.best_f);
    }

    /// Elitist OLMAYAN bir replacer: gelen yavruları koşulsuz kabul eder
    /// (popülasyon geriye gidebilir — PSO gibi). Motor yine de şimdiye kadar
    /// görülen en iyiyi (best_so_far) doğru şekilde döndürmeli.
    struct Unconditional;
    impl Replacer for Unconditional {
        fn replace(&self, pop: &mut Population, off_i: Vec<Genotype>,
                   off_fit: Vec<f64>, _c: &mut Ctx) {
            pop.individuals = off_i;
            pop.fitness = off_fit;
        }
        fn meta(&self) -> ComponentMeta {
            ComponentMeta { kind: "unconditional", supported_blocks: SupportedBlocks::All,
                requires: vec![], provides: vec![] }
        }
    }

    #[test]
    fn non_elitist_replacer_keeps_global_best() {
        let mut reg = Registry::new();
        reg.register_initializer("u-init", |_| Ok(Box::new(UInit)));
        reg.register_generator("resample", |_| Ok(Box::new(Resample)));
        reg.register_replacer("unconditional", |_| Ok(Box::new(Unconditional)));
        reg.register_boundary("no-b", |_| Ok(Box::new(NoB)));
        let spec = AlgorithmSpec {
            name: "koşulsuz-arama".into(), pop_size: 10,
            init: ComponentSpec { kind: "u-init".into(), params: serde_json::json!({}) },
            boundary: ComponentSpec { kind: "no-b".into(), params: serde_json::json!({}) },
            stages: vec![StageSpec {
                generator: ComponentSpec { kind: "resample".into(), params: serde_json::json!({}) },
                replacer: ComponentSpec { kind: "unconditional".into(), params: serde_json::json!({}) },
            }],
            termination: TerminationSpec { budget: 500, target: None },
        };
        let p = SphereShifted::new(vec![1.0, -2.0], -5.0, 5.0);
        let e = Engine::from_spec(&spec, &reg, p.space()).unwrap();

        struct MinObserver(std::sync::Arc<std::sync::Mutex<f64>>);
        impl EvalObserver for MinObserver {
            fn on_eval(&mut self, _eval_index: u64, _f: f64, best_so_far: f64) {
                let mut m = self.0.lock().unwrap();
                if best_so_far < *m { *m = best_so_far; }
            }
        }
        let observed_min = std::sync::Arc::new(std::sync::Mutex::new(f64::INFINITY));
        let obs = MinObserver(observed_min.clone());

        let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, Some(Box::new(obs))).unwrap();
        let observed_min = *observed_min.lock().unwrap();
        assert_eq!(r.best_f, observed_min,
            "koşulsuz (elitist olmayan) replacer altında motor küresel en iyiyi kaybetmemeli");
    }

    #[test]
    fn budget_smaller_than_pop_is_clear_error() {
        let (reg, mut spec, p) = setup();
        spec.termination.budget = 5; // pop_size=10'dan küçük
        let e = Engine::from_spec(&spec, &reg, p.space()).unwrap();
        let err = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap_err();
        assert!(matches!(err, EngineError::BudgetSmallerThanPopulation { budget: 5, pop_size: 10 }),
            "bütçe popülasyondan küçük olduğunda ayrı hata döndermeli, aldı: {:?}", err);
    }
}
